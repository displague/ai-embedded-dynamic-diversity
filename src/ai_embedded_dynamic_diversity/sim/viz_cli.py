from __future__ import annotations

import random
from dataclasses import dataclass
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


def _load_model(weights: str | None, profile: str, device: torch.device) -> tuple[ModelCore, object]:
    if weights:
        ckpt = torch.load(weights, map_location=device)
        cfg = model_config_for_profile(ckpt.get("profile", profile))
        cfg = type(cfg)(**ckpt["model_config"])
        model = ModelCore(**cfg.__dict__).to(device)
        model.load_state_dict(ckpt["model"])
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

            frame = state.life[0, 0, world.z // 2].detach().cpu().numpy()
            life_frames.append(frame)

    return {
        "life_frames": life_frames,
        "mismatch_values": mismatch_values,
        "vitality_values": vitality_values,
        "wind_values": wind_values,
        "force_values": force_values,
        "remap_steps": remap_steps,
    }


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

    ax_obj.axis("off")
    text = ax_obj.text(0.02, 0.98, "", va="top", family="monospace", fontsize=10)

    fig.suptitle(title)

    def _init():
        heat.set_data(result["life_frames"][0])
        line_mismatch.set_data([], [])
        line_vitality.set_data([], [])
        line_wind.set_data([], [])
        line_force.set_data([], [])
        text.set_text("")
        return (heat, line_mismatch, line_vitality, line_wind, line_force, text)

    def _update(i: int):
        xs = list(range(i + 1))
        heat.set_data(result["life_frames"][i])
        line_mismatch.set_data(xs, result["mismatch_values"][: i + 1])
        line_vitality.set_data(xs, result["vitality_values"][: i + 1])
        line_wind.set_data(xs, result["wind_values"][: i + 1])
        line_force.set_data(xs, result["force_values"][: i + 1])
        text.set_text(
            f"step={i}\n"
            f"mismatch={result['mismatch_values'][i]:.4f}\n"
            f"vitality={result['vitality_values'][i]:.4f}\n"
            f"wind={result['wind_values'][i]:.3f}\n"
            f"force={result['force_values'][i]:.3f}"
        )
        return (heat, line_mismatch, line_vitality, line_wind, line_force, text)

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
    ax_text = axes[1, 1]

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

    ax_text.axis("off")
    txt = ax_text.text(0.02, 0.98, "", va="top", family="monospace", fontsize=10)

    def _init():
        ll_m.set_data([], [])
        rr_m.set_data([], [])
        ll_v.set_data([], [])
        rr_v.set_data([], [])
        txt.set_text("")
        return (heat_left, heat_right, ll_m, rr_m, ll_v, rr_v, txt)

    def _update(i: int):
        xs = list(range(i + 1))
        heat_left.set_data(left["life_frames"][i])
        heat_right.set_data(right["life_frames"][i])
        ll_m.set_data(xs, left["mismatch_values"][: i + 1])
        rr_m.set_data(xs, right["mismatch_values"][: i + 1])
        ll_v.set_data(xs, left["vitality_values"][: i + 1])
        rr_v.set_data(xs, right["vitality_values"][: i + 1])
        txt.set_text(
            f"step={i}\n"
            f"{left_name}: mismatch={left['mismatch_values'][i]:.4f} vitality={left['vitality_values'][i]:.4f}\n"
            f"{right_name}: mismatch={right['mismatch_values'][i]:.4f} vitality={right['vitality_values'][i]:.4f}"
        )
        return (heat_left, heat_right, ll_m, rr_m, ll_v, rr_v, txt)

    anim = animation.FuncAnimation(fig, _update, init_func=_init, frames=frames, interval=70, blit=True)
    ext = Path(output).suffix.lower()
    if ext == ".gif":
        anim.save(output, writer=animation.PillowWriter(fps=14))
    else:
        anim.save(output, writer="ffmpeg", fps=20)
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

    print(
        {
            "visualization": output,
            "embodiment": embodiment,
            "steps": steps,
            "remap_events": len(result["remap_steps"]),
            "final_mismatch": result["mismatch_values"][-1],
            "final_vitality": result["vitality_values"][-1],
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

    _save_compare(output, left_result, right_result, left_name="left", right_name="right")
    print(
        {
            "visualization": output,
            "embodiment": embodiment,
            "left_final_mismatch": left_result["mismatch_values"][-1],
            "right_final_mismatch": right_result["mismatch_values"][-1],
            "left_final_vitality": left_result["vitality_values"][-1],
            "right_final_vitality": right_result["vitality_values"][-1],
        }
    )


if __name__ == "__main__":
    app()
