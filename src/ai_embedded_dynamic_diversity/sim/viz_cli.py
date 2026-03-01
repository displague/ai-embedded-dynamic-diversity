from __future__ import annotations

import random
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import typer

from ai_embedded_dynamic_diversity.config import model_config_for_profile
from ai_embedded_dynamic_diversity.models import ModelCore
from ai_embedded_dynamic_diversity.sim.embodiments import device_map_for_embodiment, get_embodiment
from ai_embedded_dynamic_diversity.sim.world import DynamicDiversityWorld, EnvironmentControls, WorldState

app = typer.Typer(add_completion=False)


@dataclass
class VizParams:
    steps: int
    remap_every: int
    force_mode: str
    force_start: int
    force_duration: int
    force_sustain: float
    force_x: float
    force_y: float
    force_z: float
    wind_x: float
    wind_y: float
    wind_z: float
    wind_variation: float
    light_x: float
    light_y: float
    light_z: float
    light_intensity: float
    light_drift_x: float
    light_drift_y: float
    light_drift_z: float


def _checkpoint_label(path: str) -> str:
    return Path(path).stem


def _resolve_storyboard_embodiments(embodiments_csv: str, fallback: list[str]) -> list[str]:
    if not embodiments_csv.strip():
        return list(fallback)
    names: list[str] = []
    for token in embodiments_csv.replace(";", ",").split(","):
        name = token.strip().lower()
        if not name:
            continue
        if name not in names:
            names.append(name)
    return names


def _resolve_storyboard_scenarios(scenario_profile: str, scenarios_csv: str) -> list[str]:
    if scenarios_csv.strip():
        return [x.strip().lower() for x in scenarios_csv.replace(";", ",").split(",") if x.strip()]
    normalized = scenario_profile.strip().lower()
    profiles = {
        "standard": ["gust", "force"],
        "hardy": ["storm", "crosswind", "blackout"],
        "extreme": ["storm", "crosswind", "blackout"],
        "calibrated_large_v1": ["storm", "crosswind", "blackout", "latency-storm", "friction-shift", "persistent-gust"],
    }
    if normalized not in profiles:
        allowed = ", ".join(sorted(profiles))
        raise ValueError(f"Unknown scenario profile '{scenario_profile}'. Allowed: {allowed}")
    return profiles[normalized]


def _select_storyboard_checkpoints(ranked: list[dict], top_k: int) -> list[str]:
    if top_k <= 0:
        raise ValueError("top_k must be > 0")
    if not ranked:
        return []
    return [str(item["checkpoint"]) for item in ranked[:top_k] if "checkpoint" in item]


def _load_model(weights: str | None, profile: str, device: torch.device) -> tuple[ModelCore, object]:
    if weights:
        ckpt = torch.load(weights, map_location=device)
        cfg = model_config_for_profile(ckpt.get("profile", profile))
        cfg = type(cfg)(**ckpt["model_config"])
        model = ModelCore(**cfg.__dict__).to(device)
        model.load_state_dict(ckpt["model"], strict=False)
        model.eval()
        return model, cfg
    cfg = model_config_for_profile(profile)
    model = ModelCore(**cfg.__dict__).to(device)
    model.eval()
    return model, cfg


def _force_scale(mode: str, step: int, start: int, duration: int, sustain: float) -> tuple[float, bool, bool]:
    rel = step - start
    if mode == "none":
        return 0.0, False, False
    if mode == "poke":
        return (1.0, True, False) if rel == 0 else (0.0, False, False)
    if mode == "press":
        if 0 <= rel < duration:
            return max(0.0, sustain), True, False
        return 0.0, False, False
    if mode == "push":
        if 0 <= rel < duration:
            ramp = min(1.0, rel / max(1, duration // 3))
            decay = 1.0 - max(0.0, rel - duration // 2) / max(1, duration // 2)
            return max(0.0, ramp * decay * sustain), True, False
        return 0.0, False, False
    if mode == "continuous-blow":
        if rel >= 0:
            return max(0.0, sustain * (0.8 + 0.2 * torch.sin(torch.tensor(rel / 6.0)).item())), True, False
        return 0.0, False, False
    if mode == "thrust":
        if 0 <= rel < duration:
            return max(0.0, sustain * (0.92 ** rel)), True, False
        return 0.0, False, False
    if mode == "move":
        if 0 <= rel < duration:
            return max(0.0, sustain), False, True
        return 0.0, False, False
    raise ValueError(f"Unsupported force mode: {mode}")


def _controls_for_step(world: DynamicDiversityWorld, params: VizParams, step: int, device: torch.device) -> EnvironmentControls:
    controls = world.default_controls(1)

    wind = torch.tensor([[params.wind_x, params.wind_y, params.wind_z]], device=device)
    if params.wind_variation > 0.0:
        phase = step / 7.0
        wind = wind + params.wind_variation * torch.tensor([[torch.sin(torch.tensor(phase)).item(), torch.cos(torch.tensor(phase / 2.0)).item(), torch.sin(torch.tensor(phase / 3.0)).item()]], device=device)
    controls.wind = wind

    controls.light_position = torch.tensor(
        [[
            params.light_x + params.light_drift_x * step,
            params.light_y + params.light_drift_y * step,
            params.light_z + params.light_drift_z * step,
        ]],
        device=device,
    ).clamp(-1.0, 1.0)
    controls.light_intensity = torch.tensor([[params.light_intensity]], device=device)

    scale, force_active, move_active = _force_scale(params.force_mode, step, params.force_start, params.force_duration, params.force_sustain)
    force_vec = torch.tensor([[params.force_x, params.force_y, params.force_z]], device=device)
    controls.force_vector = force_vec
    controls.force_strength = torch.tensor([[scale]], device=device)
    controls.force_active = torch.tensor([[1.0 if force_active else 0.0]], device=device)
    controls.force_position = torch.zeros(1, 3, device=device)
    if move_active:
        controls.move_object_delta = 0.03 * scale * force_vec
    return controls


def _clone_state(state: WorldState) -> WorldState:
    return WorldState(
        life=state.life.clone(),
        resources=state.resources.clone(),
        stress=state.stress.clone(),
        object_pos=state.object_pos.clone(),
        object_vel=state.object_vel.clone(),
    )


def _simulate(
    model: ModelCore,
    cfg,
    world: DynamicDiversityWorld,
    initial_state: WorldState,
    params: VizParams,
    embodiment_name: str,
    projection: torch.Tensor,
    device: torch.device,
    seed_offset: int,
) -> dict[str, list[float] | list]:
    model.eval()
    state = _clone_state(initial_state)
    memory = model.init_memory(1, cfg.memory_slots, cfg.memory_dim, device)

    embodiment = get_embodiment(embodiment_name)
    mapping = device_map_for_embodiment(cfg.io_channels, embodiment, device=device, permutation_seed=seed_offset)

    life_frames = []
    mismatch_values = []
    vitality_values = []
    wind_values = []
    force_values = []
    remap_steps = []
    object_trajectories = []

    for step in range(params.steps):
        remap_code = torch.zeros(1, cfg.max_remap_groups, device=device)
        if step > 0 and step % params.remap_every == 0:
            remap_steps.append(step)
            mapping = device_map_for_embodiment(cfg.io_channels, embodiment, device=device, permutation_seed=seed_offset + step)
            remap_code[:, step % cfg.max_remap_groups] = 1.0

        controls = _controls_for_step(world, params, step, device)
        with torch.no_grad():
            obs = world.encode_observation(state, signal_dim=cfg.signal_dim)
            out = model(obs, memory, remap_code)
            memory = out["memory"]

            desired = torch.tanh(obs @ projection)
            applied = out["io"] @ mapping
            mismatch_values.append(float(torch.mean((applied - desired) ** 2).item()))

            action = applied.mean(dim=1, keepdim=True).repeat(1, world.x * world.y * world.z)
            state = world.step(state, action, controls=controls)
            vitality_values.append(float(state.life.mean().item()))
            wind_values.append(float(torch.norm(controls.wind, dim=1).mean().item()))
            force_values.append(float((controls.force_strength * controls.force_active).mean().item()))
            object_trajectories.append(state.object_pos.detach().cpu().numpy()[0].tolist())

            frame = state.life[0, 0, world.z // 2].detach().cpu().numpy()
            life_frames.append(frame)

    return {
        "life_frames": life_frames,
        "mismatch_values": mismatch_values,
        "vitality_values": vitality_values,
        "wind_values": wind_values,
        "force_values": force_values,
        "remap_steps": remap_steps,
        "object_pos": object_trajectories,
    }


def _classify_adaptation_signature(result: dict) -> dict[str, object]:
    mismatch = [float(x) for x in result.get("mismatch_values", [])]
    vitality = [float(x) for x in result.get("vitality_values", [])]
    remap_events = int(len(result.get("remap_steps", [])))
    if not mismatch or not vitality:
        return {
            "label": "insufficient-data",
            "severity": 0.0,
            "vitality_collapse": False,
            "mismatch_elevated": False,
            "recovery_failure": False,
            "remap_events": remap_events,
            "details": {},
        }

    n = len(mismatch)
    start_n = max(1, n // 4)
    end_start = max(0, n - start_n)

    start_mismatch = float(sum(mismatch[:start_n]) / start_n)
    end_mismatch = float(sum(mismatch[end_start:]) / max(1, n - end_start))
    end_vitality = float(sum(vitality[end_start:]) / max(1, n - end_start))
    final_mismatch = float(mismatch[-1])
    final_vitality = float(vitality[-1])

    vitality_collapse = final_vitality < 0.05 and end_vitality < 0.08
    mismatch_elevated = final_mismatch > 0.30 and end_mismatch > 0.28
    recovery_failure = (end_mismatch - start_mismatch) > 0.08

    severity = (
        (0.45 if vitality_collapse else 0.0)
        + (0.35 if mismatch_elevated else 0.0)
        + (0.20 if recovery_failure else 0.0)
    )

    if severity >= 0.7:
        label = "failure"
    elif severity >= 0.35:
        label = "degraded"
    else:
        label = "stable"

    return {
        "label": label,
        "severity": float(min(1.0, severity)),
        "vitality_collapse": bool(vitality_collapse),
        "mismatch_elevated": bool(mismatch_elevated),
        "recovery_failure": bool(recovery_failure),
        "remap_events": remap_events,
        "details": {
            "start_mismatch": start_mismatch,
            "end_mismatch": end_mismatch,
            "final_mismatch": final_mismatch,
            "end_vitality": end_vitality,
            "final_vitality": final_vitality,
        },
    }


def _save_metrics(output: str, result: dict) -> None:
    """Saves simulation metrics to JSON or CSV."""
    import csv
    
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    steps = len(result["mismatch_values"])
    rows = []
    for i in range(steps):
        row = {
            "step": i,
            "mismatch": result["mismatch_values"][i],
            "vitality": result["vitality_values"][i],
            "wind": result["wind_values"][i],
            "force": result["force_values"][i],
        }
        if "object_pos" in result and result["object_pos"]:
            pos = result["object_pos"][i]
            row.update({"obj_x": pos[0], "obj_y": pos[1], "obj_z": pos[2]})
        rows.append(row)

    if path.suffix.lower() == ".json":
        path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    else:
        fieldnames = ["step", "mismatch", "vitality", "wind", "force"]
        if "object_pos" in result and result["object_pos"]:
            fieldnames += ["obj_x", "obj_y", "obj_z"]
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    print({"metrics_saved": str(path)})


def _save_single(output: str, result: dict, title: str) -> None:
    import matplotlib.pyplot as plt
    from matplotlib import animation

    Path(output).parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    ax_life = axes[0, 0]
    ax_adapt = axes[0, 1]
    ax_env = axes[1, 0]
    ax_obj = axes[1, 1]

    heat = ax_life.imshow(result["life_frames"][0], cmap="viridis", vmin=0.0, vmax=1.0)
    ax_life.set_title("Life Slice")

    ax_adapt.set_title("Adaptation")
    ax_adapt.set_xlim(0, max(1, len(result["mismatch_values"]) - 1))
    ymax = max(max(result["mismatch_values"]), max(result["vitality_values"]), 1e-3)
    ax_adapt.set_ylim(0, ymax * 1.15)
    line_mismatch, = ax_adapt.plot([], [], color="tab:red", label="control mismatch")
    line_vitality, = ax_adapt.plot([], [], color="tab:green", label="vitality")
    for rs in result["remap_steps"]:
        ax_adapt.axvline(rs, color="tab:blue", alpha=0.25)
    ax_adapt.legend(loc="upper right")

    ax_env.set_title("Environment Controls")
    ax_env.set_xlim(0, max(1, len(result["wind_values"]) - 1))
    env_max = max(max(result["wind_values"]), max(result["force_values"]), 1e-3)
    ax_env.set_ylim(0, env_max * 1.2)
    line_wind, = ax_env.plot([], [], color="tab:cyan", label="wind magnitude")
    line_force, = ax_env.plot([], [], color="tab:orange", label="force strength")
    ax_env.legend(loc="upper right")

    # Task 3: Trajectory Overlay
    ax_obj.set_title("Object Trajectory (X-Y)")
    ax_obj.set_xlim(-1.1, 1.1)
    ax_obj.set_ylim(-1.1, 1.1)
    ax_obj.grid(True, alpha=0.3)
    line_traj, = ax_obj.plot([], [], color="magenta", lw=1.5, label="path")
    point_obj, = ax_obj.plot([], [], "o", color="red")

    fig.suptitle(title)

    def _init():
        heat.set_data(result["life_frames"][0])
        line_mismatch.set_data([], [])
        line_vitality.set_data([], [])
        line_wind.set_data([], [])
        line_force.set_data([], [])
        line_traj.set_data([], [])
        point_obj.set_data([], [])
        return (heat, line_mismatch, line_vitality, line_wind, line_force, line_traj, point_obj)

    def _update(i: int):
        xs = list(range(i + 1))
        heat.set_data(result["life_frames"][i])
        line_mismatch.set_data(xs, result["mismatch_values"][: i + 1])
        line_vitality.set_data(xs, result["vitality_values"][: i + 1])
        line_wind.set_data(xs, result["wind_values"][: i + 1])
        line_force.set_data(xs, result["force_values"][: i + 1])
        
        if result["object_pos"]:
            path_xy = result["object_pos"][: i + 1]
            px = [p[0] for p in path_xy]
            py = [p[1] for p in path_xy]
            line_traj.set_data(px, py)
            point_obj.set_data([px[-1]], [py[-1]])
            
        return (heat, line_mismatch, line_vitality, line_wind, line_force, line_traj, point_obj)

    anim = animation.FuncAnimation(fig, _update, init_func=_init, frames=len(result["life_frames"]), interval=70, blit=True)

    ext = Path(output).suffix.lower()
    if ext == ".gif":
        anim.save(output, writer=animation.PillowWriter(fps=14))
    else:
        anim.save(output, writer="ffmpeg", fps=20)
    plt.close(fig)


def _save_compare(output: str, left: dict, right: dict, left_name: str, right_name: str) -> None:
    import matplotlib.pyplot as plt
    from matplotlib import animation

    Path(output).parent.mkdir(parents=True, exist_ok=True)

    frames = min(len(left["life_frames"]), len(right["life_frames"]))
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))

    ax_left = axes[0, 0]
    ax_right = axes[0, 1]
    ax_lines = axes[1, 0]
    ax_traj = axes[1, 1]

    heat_left = ax_left.imshow(left["life_frames"][0], cmap="viridis", vmin=0.0, vmax=1.0)
    heat_right = ax_right.imshow(right["life_frames"][0], cmap="viridis", vmin=0.0, vmax=1.0)
    ax_left.set_title(left_name)
    ax_right.set_title(right_name)

    ax_lines.set_xlim(0, max(1, frames - 1))
    ymax = max(max(left["mismatch_values"]), max(right["mismatch_values"]), max(left["vitality_values"]), max(right["vitality_values"]), 1e-3)
    ax_lines.set_ylim(0, ymax * 1.15)
    ll_m, = ax_lines.plot([], [], color="tab:red", label=f"{left_name} mismatch")
    rr_m, = ax_lines.plot([], [], color="tab:orange", label=f"{right_name} mismatch")
    ll_v, = ax_lines.plot([], [], color="tab:green", label=f"{left_name} vitality")
    rr_v, = ax_lines.plot([], [], color="tab:blue", label=f"{right_name} vitality")
    ax_lines.legend(loc="upper right", fontsize=8)

    ax_traj.set_title("Trajectories (X-Y)")
    ax_traj.set_xlim(-1.1, 1.1)
    ax_traj.set_ylim(-1.1, 1.1)
    l_path, = ax_traj.plot([], [], color="tab:green", alpha=0.6, label=left_name)
    r_path, = ax_traj.plot([], [], color="tab:blue", alpha=0.6, label=right_name)
    ax_traj.legend(loc="upper right", fontsize=8)

    def _init():
        ll_m.set_data([], [])
        rr_m.set_data([], [])
        ll_v.set_data([], [])
        rr_v.set_data([], [])
        l_path.set_data([], [])
        r_path.set_data([], [])
        return (heat_left, heat_right, ll_m, rr_m, ll_v, rr_v, l_path, r_path)

    def _update(i: int):
        xs = list(range(i + 1))
        heat_left.set_data(left["life_frames"][i])
        heat_right.set_data(right["life_frames"][i])
        ll_m.set_data(xs, left["mismatch_values"][: i + 1])
        rr_m.set_data(xs, right["mismatch_values"][: i + 1])
        ll_v.set_data(xs, left["vitality_values"][: i + 1])
        rr_v.set_data(xs, right["vitality_values"][: i + 1])
        
        if left["object_pos"]:
            lpx = [p[0] for p in left["object_pos"][: i + 1]]
            lpy = [p[1] for p in left["object_pos"][: i + 1]]
            l_path.set_data(lpx, lpy)
        if right["object_pos"]:
            rpx = [p[0] for p in right["object_pos"][: i + 1]]
            rpy = [p[1] for p in right["object_pos"][: i + 1]]
            r_path.set_data(rpx, rpy)
            
        return (heat_left, heat_right, ll_m, rr_m, ll_v, rr_v, l_path, r_path)

    anim = animation.FuncAnimation(fig, _update, init_func=_init, frames=frames, interval=70, blit=True)
    ext = Path(output).suffix.lower()
    if ext == ".gif":
        anim.save(output, writer=animation.PillowWriter(fps=14))
    else:
        anim.save(output, writer="ffmpeg", fps=20)
    plt.close(fig)


def _scenario_viz_overrides(name: str) -> dict:
    scenario = name.strip().lower()
    if scenario == "gust":
        return {
            "force_mode": "press",
            "force_start": 18,
            "force_duration": 16,
            "force_sustain": 0.7,
            "force_x": 0.6,
            "force_y": 0.0,
            "force_z": 0.0,
            "wind_x": 0.45,
            "wind_y": 0.1,
            "wind_z": 0.0,
            "wind_variation": 0.25,
            "light_x": -0.3,
            "light_y": 0.0,
            "light_z": 0.3,
            "light_intensity": 0.8,
            "light_drift_x": 0.002,
            "light_drift_y": 0.0,
            "light_drift_z": 0.0,
        }
    if scenario == "force":
        return {
            "force_mode": "thrust",
            "force_start": 20,
            "force_duration": 26,
            "force_sustain": 1.0,
            "force_x": 0.85,
            "force_y": 0.0,
            "force_z": 0.0,
            "wind_x": 0.2,
            "wind_y": 0.0,
            "wind_z": 0.0,
            "wind_variation": 0.1,
            "light_x": -0.15,
            "light_y": 0.0,
            "light_z": 0.25,
            "light_intensity": 0.7,
            "light_drift_x": 0.0,
            "light_drift_y": 0.0,
            "light_drift_z": 0.0,
        }
    if scenario == "storm":
        return {
            "force_mode": "continuous-blow",
            "force_start": 10,
            "force_duration": 36,
            "force_sustain": 1.1,
            "force_x": 1.1,
            "force_y": 0.2,
            "force_z": 0.0,
            "wind_x": 0.8,
            "wind_y": 0.35,
            "wind_z": 0.0,
            "wind_variation": 0.45,
            "light_x": -0.4,
            "light_y": 0.0,
            "light_z": 0.25,
            "light_intensity": 0.55,
            "light_drift_x": 0.004,
            "light_drift_y": 0.0,
            "light_drift_z": 0.0,
        }
    if scenario == "blackout":
        return {
            "force_mode": "press",
            "force_start": 16,
            "force_duration": 26,
            "force_sustain": 0.9,
            "force_x": 0.75,
            "force_y": 0.0,
            "force_z": 0.0,
            "wind_x": 0.25,
            "wind_y": 0.0,
            "wind_z": 0.0,
            "wind_variation": 0.15,
            "light_x": 0.25,
            "light_y": 0.0,
            "light_z": 0.1,
            "light_intensity": 0.15,
            "light_drift_x": -0.004,
            "light_drift_y": 0.0,
            "light_drift_z": 0.0,
        }
    if scenario == "crosswind":
        return {
            "force_mode": "thrust",
            "force_start": 14,
            "force_duration": 34,
            "force_sustain": 0.95,
            "force_x": 0.55,
            "force_y": 0.85,
            "force_z": 0.0,
            "wind_x": 0.15,
            "wind_y": 0.85,
            "wind_z": 0.0,
            "wind_variation": 0.3,
            "light_x": -0.2,
            "light_y": 0.15,
            "light_z": 0.2,
            "light_intensity": 0.65,
            "light_drift_x": 0.0,
            "light_drift_y": -0.002,
            "light_drift_z": 0.0,
        }
    if scenario == "latency-storm":
        return {
            "force_mode": "continuous-blow",
            "force_start": 8,
            "force_duration": 44,
            "force_sustain": 1.2,
            "force_x": 1.2,
            "force_y": 0.35,
            "force_z": 0.0,
            "wind_x": 0.9,
            "wind_y": 0.4,
            "wind_z": 0.1,
            "wind_variation": 0.55,
            "light_x": -0.45,
            "light_y": 0.0,
            "light_z": 0.2,
            "light_intensity": 0.45,
            "light_drift_x": 0.004,
            "light_drift_y": 0.001,
            "light_drift_z": 0.0,
        }
    if scenario == "friction-shift":
        return {
            "force_mode": "thrust",
            "force_start": 12,
            "force_duration": 36,
            "force_sustain": 1.0,
            "force_x": 0.65,
            "force_y": 0.95,
            "force_z": 0.0,
            "wind_x": 0.25,
            "wind_y": 0.7,
            "wind_z": 0.0,
            "wind_variation": 0.35,
            "light_x": -0.25,
            "light_y": 0.2,
            "light_z": 0.15,
            "light_intensity": 0.55,
            "light_drift_x": 0.0,
            "light_drift_y": -0.002,
            "light_drift_z": 0.0,
        }
    if scenario == "persistent-gust":
        return {
            "force_mode": "continuous-blow",
            "force_start": 6,
            "force_duration": 52,
            "force_sustain": 1.05,
            "force_x": 0.95,
            "force_y": 0.25,
            "force_z": 0.0,
            "wind_x": 0.75,
            "wind_y": 0.25,
            "wind_z": 0.0,
            "wind_variation": 0.4,
            "light_x": -0.3,
            "light_y": -0.1,
            "light_z": 0.2,
            "light_intensity": 0.6,
            "light_drift_x": 0.002,
            "light_drift_y": 0.001,
            "light_drift_z": 0.0,
        }
    raise ValueError(f"Unsupported storyboard scenario: {name}")


def _params_for_scenario(base: VizParams, scenario_name: str) -> VizParams:
    merged = {**asdict(base), **_scenario_viz_overrides(scenario_name)}
    return VizParams(**merged)


def _save_storyboard_montage(output: str, panels: dict[str, dict], title: str) -> None:
    import matplotlib.pyplot as plt
    from matplotlib import animation

    if not panels:
        return
    Path(output).parent.mkdir(parents=True, exist_ok=True)

    names = list(panels.keys())
    frames = min(len(panels[name]["life_frames"]) for name in names)
    cols = 2
    rows = (len(names) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows))
    if not hasattr(axes, "__iter__"):
        axes = [axes]
    axes_list = list(axes.flatten()) if hasattr(axes, "flatten") else list(axes)

    heatmaps = {}
    texts = {}
    for idx, name in enumerate(names):
        ax = axes_list[idx]
        result = panels[name]
        heat = ax.imshow(result["life_frames"][0], cmap="viridis", vmin=0.0, vmax=1.0)
        ax.set_title(name)
        txt = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top", color="white", fontsize=9, family="monospace")
        heatmaps[name] = heat
        texts[name] = txt

    for ax in axes_list[len(names):]:
        ax.axis("off")

    fig.suptitle(title)

    def _init():
        artists = []
        for name in names:
            heatmaps[name].set_data(panels[name]["life_frames"][0])
            texts[name].set_text("")
            artists.extend([heatmaps[name], texts[name]])
        return tuple(artists)

    def _update(i: int):
        artists = []
        for name in names:
            result = panels[name]
            heatmaps[name].set_data(result["life_frames"][i])
            texts[name].set_text(
                f"mismatch={result['mismatch_values'][i]:.4f}\n"
                f"vitality={result['vitality_values'][i]:.4f}\n"
                f"force={result['force_values'][i]:.3f}"
            )
            artists.extend([heatmaps[name], texts[name]])
        return tuple(artists)

    anim = animation.FuncAnimation(fig, _update, init_func=_init, frames=frames, interval=80, blit=True)
    ext = Path(output).suffix.lower()
    if ext == ".gif":
        anim.save(output, writer=animation.PillowWriter(fps=12))
    else:
        anim.save(output, writer="ffmpeg", fps=18)
    plt.close(fig)


@app.command()
def run(
    weights: str = "",
    profile: str = "pi5",
    embodiment: str = "hexapod",
    steps: int = 160,
    remap_every: int = 20,
    world_x: int = 20,
    world_y: int = 20,
    world_z: int = 10,
    resource_channels: int = 5,
    output: str = "artifacts/adaptation.gif",
    device: str = "cpu",
    seed: int = 7,
    force_mode: str = "push",
    force_start: int = 24,
    force_duration: int = 28,
    force_sustain: float = 0.8,
    force_x: float = 0.8,
    force_y: float = 0.0,
    force_z: float = 0.0,
    wind_x: float = 0.4,
    wind_y: float = 0.0,
    wind_z: float = 0.0,
    wind_variation: float = 0.2,
    light_x: float = -0.3,
    light_y: float = 0.0,
    light_z: float = 0.2,
    light_intensity: float = 0.75,
    light_drift_x: float = 0.002,
    light_drift_y: float = 0.0,
    light_drift_z: float = 0.0,
    metrics_out: str = "",
) -> None:
    try:
        import matplotlib.pyplot as _  # noqa: F401
    except ImportError as exc:
        raise typer.BadParameter("matplotlib is required for visualization. Install dependencies with uv sync.") from exc

    random.seed(seed)
    torch.manual_seed(seed)
    dev = torch.device(device)

    world = DynamicDiversityWorld(world_x, world_y, world_z, resource_channels, decay=0.03, device=str(dev))
    model, cfg = _load_model(weights if weights else None, profile, dev)
    params = VizParams(
        steps=steps,
        remap_every=remap_every,
        force_mode=force_mode,
        force_start=force_start,
        force_duration=force_duration,
        force_sustain=force_sustain,
        force_x=force_x,
        force_y=force_y,
        force_z=force_z,
        wind_x=wind_x,
        wind_y=wind_y,
        wind_z=wind_z,
        wind_variation=wind_variation,
        light_x=light_x,
        light_y=light_y,
        light_z=light_z,
        light_intensity=light_intensity,
        light_drift_x=light_drift_x,
        light_drift_y=light_drift_y,
        light_drift_z=light_drift_z,
    )

    initial_state = world.init(batch_size=1)
    control_dim = len(get_embodiment(embodiment).controls)
    projection = torch.randn(cfg.signal_dim, control_dim, device=dev) * 0.3
    result = _simulate(model, cfg, world, initial_state, params, embodiment, projection, dev, seed_offset=seed)
    _save_single(output, result, f"Embodiment={embodiment} force={force_mode}")
    signature = _classify_adaptation_signature(result)

    if metrics_out:
        _save_metrics(metrics_out, result)

    print(
        {
            "visualization": output,
            "embodiment": embodiment,
            "steps": steps,
            "remap_events": len(result["remap_steps"]),
            "final_mismatch": result["mismatch_values"][-1],
            "final_vitality": result["vitality_values"][-1],
            "adaptation_signature": signature,
        }
    )


@app.command()
def compare(
    left_weights: str,
    right_weights: str,
    profile: str = "pi5",
    embodiment: str = "hexapod",
    steps: int = 160,
    remap_every: int = 20,
    output: str = "artifacts/adaptation-compare.gif",
    device: str = "cpu",
    seed: int = 7,
    force_mode: str = "push",
    force_start: int = 24,
    force_duration: int = 28,
    force_sustain: float = 0.8,
    force_x: float = 0.8,
    force_y: float = 0.0,
    force_z: float = 0.0,
    wind_x: float = 0.4,
    wind_y: float = 0.0,
    wind_z: float = 0.0,
    wind_variation: float = 0.2,
    light_x: float = -0.3,
    light_y: float = 0.0,
    light_z: float = 0.2,
    light_intensity: float = 0.75,
    light_drift_x: float = 0.002,
    light_drift_y: float = 0.0,
    light_drift_z: float = 0.0,
    metrics_left: str = "",
    metrics_right: str = "",
) -> None:
    try:
        import matplotlib.pyplot as _  # noqa: F401
    except ImportError as exc:
        raise typer.BadParameter("matplotlib is required for visualization. Install dependencies with uv sync.") from exc

    random.seed(seed)
    torch.manual_seed(seed)
    dev = torch.device(device)

    world = DynamicDiversityWorld(20, 20, 10, 5, decay=0.03, device=str(dev))
    left_model, left_cfg = _load_model(left_weights, profile, dev)
    right_model, right_cfg = _load_model(right_weights, profile, dev)

    if left_cfg.signal_dim != right_cfg.signal_dim or left_cfg.io_channels != right_cfg.io_channels:
        raise typer.BadParameter("Models are not comparable: signal/io dimensions differ.")

    params = VizParams(
        steps=steps,
        remap_every=remap_every,
        force_mode=force_mode,
        force_start=force_start,
        force_duration=force_duration,
        force_sustain=force_sustain,
        force_x=force_x,
        force_y=force_y,
        force_z=force_z,
        wind_x=wind_x,
        wind_y=wind_y,
        wind_z=wind_z,
        wind_variation=wind_variation,
        light_x=light_x,
        light_y=light_y,
        light_z=light_z,
        light_intensity=light_intensity,
        light_drift_x=light_drift_x,
        light_drift_y=light_drift_y,
        light_drift_z=light_drift_z,
    )

    initial_state = world.init(batch_size=1)
    control_dim = len(get_embodiment(embodiment).controls)
    projection = torch.randn(left_cfg.signal_dim, control_dim, device=dev) * 0.3

    left_result = _simulate(left_model, left_cfg, world, initial_state, params, embodiment, projection, dev, seed_offset=seed)
    right_result = _simulate(right_model, right_cfg, world, initial_state, params, embodiment, projection, dev, seed_offset=seed)
    left_signature = _classify_adaptation_signature(left_result)
    right_signature = _classify_adaptation_signature(right_result)

    _save_compare(output, left_result, right_result, left_name="left", right_name="right")
    
    if metrics_left:
        _save_metrics(metrics_left, left_result)
    if metrics_right:
        _save_metrics(metrics_right, right_result)

    print(
        {
            "visualization": output,
            "embodiment": embodiment,
            "left_final_mismatch": left_result["mismatch_values"][-1],
            "right_final_mismatch": right_result["mismatch_values"][-1],
            "left_final_vitality": left_result["vitality_values"][-1],
            "right_final_vitality": right_result["vitality_values"][-1],
            "left_signature": left_signature,
            "right_signature": right_signature,
        }
    )


@app.command()
def batch_force(
    weights: str = "artifacts/model-core-champion-v09.pt",
    profile: str = "pi5",
    embodiment: str = "car",
    output_dir: str = "artifacts/viz-batch-force",
    steps: int = 160,
    device: str = "cpu",
    seed: int = 7,
) -> None:
    """Generates visualizations for all supported force modes."""
    modes = ["poke", "press", "push", "continuous-blow", "thrust", "move"]
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    dev = torch.device(device)
    model, cfg = _load_model(weights, profile, dev)
    
    world = DynamicDiversityWorld(20, 20, 10, 5, decay=0.03, device=str(dev))
    initial_state = world.init(batch_size=1)
    control_dim = len(get_embodiment(embodiment).controls)
    projection = torch.randn(cfg.signal_dim, control_dim, device=dev) * 0.3

    print({"info": f"Starting batch force visualization for {embodiment} in {output_dir}"})
    summaries: list[dict[str, object]] = []

    for mode in modes:
        params = VizParams(
            steps=steps,
            remap_every=20,
            force_mode=mode,
            force_start=20,
            force_duration=30,
            force_sustain=0.8,
            force_x=0.8,
            force_y=0.0,
            force_z=0.0,
            wind_x=0.2,
            wind_y=0.0,
            wind_z=0.0,
            wind_variation=0.1,
            light_x=-0.3,
            light_y=0.0,
            light_z=0.2,
            light_intensity=0.75,
            light_drift_x=0.002,
            light_drift_y=0.0,
            light_drift_z=0.0,
        )
        
        result = _simulate(model, cfg, world, initial_state, params, embodiment, projection, dev, seed_offset=seed)
        
        gif_name = f"{embodiment}-force-{mode}.gif"
        metrics_name = f"{embodiment}-force-{mode}-metrics.json"
        
        _save_single(str(out_path / gif_name), result, title=f"Force Mode: {mode}")
        _save_metrics(str(out_path / metrics_name), result)
        signature = _classify_adaptation_signature(result)
        summaries.append(
            {
                "mode": mode,
                "gif": str(out_path / gif_name),
                "metrics": str(out_path / metrics_name),
                "final_mismatch": float(result["mismatch_values"][-1]),
                "final_vitality": float(result["vitality_values"][-1]),
                "adaptation_signature": signature,
            }
        )

    summary_path = out_path / f"{embodiment}-batch-summary.json"
    summary_path.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print({"batch_complete": str(out_path), "modes": len(modes), "summary": str(summary_path)})


@app.command()
def storyboard(
    cross_eval_json: str = "artifacts/cross-eval-summary.json",
    top_k: int = 2,
    embodiments: str = "",
    scenario_profile: str = "hardy",
    scenarios: str = "",
    profile: str = "",
    steps: int = 170,
    remap_every: int = 18,
    world_x: int = 20,
    world_y: int = 20,
    world_z: int = 10,
    resource_channels: int = 5,
    output_dir: str = "artifacts/viz-storyboard",
    manifest_output: str = "artifacts/viz-storyboard/manifest.json",
    include_montage: bool = True,
    device: str = "cpu",
    seed: int = 11,
) -> None:
    try:
        import matplotlib.pyplot as _  # noqa: F401
    except ImportError as exc:
        raise typer.BadParameter("matplotlib is required for visualization. Install dependencies with uv sync.") from exc

    payload = json.loads(Path(cross_eval_json).read_text(encoding="utf-8"))
    ranked = payload.get("ranked", [])
    if not isinstance(ranked, list) or not ranked:
        raise typer.BadParameter(f"No ranked checkpoints found in {cross_eval_json}")

    selected_checkpoints = _select_storyboard_checkpoints(ranked, top_k=top_k)
    if not selected_checkpoints:
        raise typer.BadParameter("No checkpoints available for storyboard generation")

    cfg = payload.get("config", {})
    fallback_embodiments = [str(x).strip().lower() for x in cfg.get("embodiments", []) if str(x).strip()]
    if not fallback_embodiments:
        fallback_embodiments = ["hexapod", "car", "drone", "polymorph120"]
    embodiment_list = _resolve_storyboard_embodiments(embodiments, fallback=fallback_embodiments)
    scenario_names = _resolve_storyboard_scenarios(scenario_profile=scenario_profile, scenarios_csv=scenarios)
    if not scenario_names:
        raise typer.BadParameter("No storyboard scenarios were selected")

    resolved_profile = profile.strip() or str(cfg.get("profile", "pi5"))
    dev = torch.device(device)
    random.seed(seed)
    torch.manual_seed(seed)

    base_params = VizParams(
        steps=steps,
        remap_every=remap_every,
        force_mode="push",
        force_start=24,
        force_duration=28,
        force_sustain=0.8,
        force_x=0.8,
        force_y=0.0,
        force_z=0.0,
        wind_x=0.4,
        wind_y=0.0,
        wind_z=0.0,
        wind_variation=0.2,
        light_x=-0.3,
        light_y=0.0,
        light_z=0.2,
        light_intensity=0.75,
        light_drift_x=0.002,
        light_drift_y=0.0,
        light_drift_z=0.0,
    )

    world = DynamicDiversityWorld(world_x, world_y, world_z, resource_channels, decay=0.03, device=str(dev))
    base_state = world.init(batch_size=1)
    model_cache: dict[str, tuple[ModelCore, object]] = {}
    results_for_montage: dict[str, dict] = {}
    generated_artifacts: list[dict] = []

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for scenario_idx, scenario_name in enumerate(scenario_names):
        scenario_params = _params_for_scenario(base_params, scenario_name)
        for emb_idx, embodiment_name in enumerate(embodiment_list):
            get_embodiment(embodiment_name)
            run_results: list[tuple[str, dict, object]] = []
            for rank_idx, ckpt_path in enumerate(selected_checkpoints):
                if ckpt_path not in model_cache:
                    model_cache[ckpt_path] = _load_model(ckpt_path, resolved_profile, dev)
                model, cfg_obj = model_cache[ckpt_path]
                control_dim = len(get_embodiment(embodiment_name).controls)
                local_seed = seed + emb_idx * 100 + scenario_idx * 1000
                torch.manual_seed(local_seed)
                projection = torch.randn(cfg_obj.signal_dim, control_dim, device=dev) * 0.3
                result = _simulate(
                    model=model,
                    cfg=cfg_obj,
                    world=world,
                    initial_state=base_state,
                    params=scenario_params,
                    embodiment_name=embodiment_name,
                    projection=projection,
                    device=dev,
                    seed_offset=local_seed + rank_idx * 17,
                )
                run_results.append((ckpt_path, result, cfg_obj))

            primary_ckpt, primary_result, _ = run_results[0]
            evo_name = f"{scenario_name}-{embodiment_name}-{_checkpoint_label(primary_ckpt)}-evolution.gif"
            evo_path = str(out_dir / evo_name)
            _save_single(
                evo_path,
                primary_result,
                title=f"{embodiment_name} | {scenario_name} | {_checkpoint_label(primary_ckpt)}",
            )
            
            # Save individual metrics for primary result
            metrics_evo_name = f"{scenario_name}-{embodiment_name}-{_checkpoint_label(primary_ckpt)}-metrics.json"
            _save_metrics(str(out_dir / metrics_evo_name), primary_result)

            item = {
                "scenario": scenario_name,
                "embodiment": embodiment_name,
                "primary_checkpoint": primary_ckpt,
                "evolution_gif": evo_path,
                "evolution_metrics": str(out_dir / metrics_evo_name),
                "final_mismatch": float(primary_result["mismatch_values"][-1]),
                "final_vitality": float(primary_result["vitality_values"][-1]),
                "adaptation_signature": _classify_adaptation_signature(primary_result),
            }
            if scenario_idx == 0:
                results_for_montage[embodiment_name] = primary_result

            if len(run_results) >= 2:
                left_ckpt, left_result, _ = run_results[0]
                right_ckpt, right_result, _ = run_results[1]
                compare_name = (
                    f"{scenario_name}-{embodiment_name}-{_checkpoint_label(left_ckpt)}-vs-{_checkpoint_label(right_ckpt)}.gif"
                )
                compare_path = str(out_dir / compare_name)
                _save_compare(
                    compare_path,
                    left_result,
                    right_result,
                    left_name=_checkpoint_label(left_ckpt),
                    right_name=_checkpoint_label(right_ckpt),
                )
                
                # Save comparison metrics
                metrics_left_name = f"{scenario_name}-{embodiment_name}-{_checkpoint_label(left_ckpt)}-compare-left-metrics.json"
                metrics_right_name = f"{scenario_name}-{embodiment_name}-{_checkpoint_label(right_ckpt)}-compare-right-metrics.json"
                _save_metrics(str(out_dir / metrics_left_name), left_result)
                _save_metrics(str(out_dir / metrics_right_name), right_result)
                
                item["compare_gif"] = compare_path
                item["compare_left_checkpoint"] = left_ckpt
                item["compare_right_checkpoint"] = right_ckpt
                item["compare_left_metrics"] = str(out_dir / metrics_left_name)
                item["compare_right_metrics"] = str(out_dir / metrics_right_name)
                item["compare_left_final_mismatch"] = float(left_result["mismatch_values"][-1])
                item["compare_right_final_mismatch"] = float(right_result["mismatch_values"][-1])
                item["compare_left_signature"] = _classify_adaptation_signature(left_result)
                item["compare_right_signature"] = _classify_adaptation_signature(right_result)
            generated_artifacts.append(item)

    montage_path = ""
    if include_montage and results_for_montage:
        montage_path = str(out_dir / "convergence-storyboard.gif")
        _save_storyboard_montage(
            output=montage_path,
            panels=results_for_montage,
            title=f"Embodied Convergence Storyboard ({scenario_names[0]})",
        )

    manifest = {
        "cross_eval_json": cross_eval_json,
        "profile": resolved_profile,
        "selected_checkpoints": selected_checkpoints,
        "embodiments": embodiment_list,
        "scenarios": scenario_names,
        "steps": steps,
        "remap_every": remap_every,
        "output_dir": output_dir,
        "montage_gif": montage_path,
        "artifacts": generated_artifacts,
    }
    Path(manifest_output).parent.mkdir(parents=True, exist_ok=True)
    Path(manifest_output).write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(
        {
            "manifest": manifest_output,
            "output_dir": output_dir,
            "embodiments": len(embodiment_list),
            "scenarios": len(scenario_names),
            "checkpoints": len(selected_checkpoints),
            "generated_artifacts": len(generated_artifacts),
            "montage": montage_path,
        }
    )


if __name__ == "__main__":
    app()
