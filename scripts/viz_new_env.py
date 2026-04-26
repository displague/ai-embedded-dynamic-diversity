"""
Visualize champion models in the new_env_v1 world (40x40x20 toroidal, occlusion,
physics objects, hazard zones).

Produces:
  artifacts/new-env-evolution/
    viz-new-env-champion-storm.gif          new champion, storm scenario
    viz-new-env-champion-hazard-sweep.gif   champion across all 3 hazard types
    viz-new-env-compare-storm.gif           champion-new-env-v1 vs champion-v08
    viz-new-env-compare-blackout.gif        same, blackout scenario
    viz-new-env-phys-obj-traj.gif           physics object trajectory panel
    viz-training-progress.gif               fitness + GDI over 40 generations
"""
from __future__ import annotations

import json
import math
import sys
from dataclasses import asdict
from pathlib import Path

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import animation
from matplotlib.gridspec import GridSpec

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ai_embedded_dynamic_diversity.config import world_config_for_profile, WorldConfig
from ai_embedded_dynamic_diversity.models import ModelCore
from ai_embedded_dynamic_diversity.config import ModelConfig
from ai_embedded_dynamic_diversity.sim.signaling import SignalingWorld
from ai_embedded_dynamic_diversity.sim.world import WorldState, EnvironmentControls
from ai_embedded_dynamic_diversity.sim.embodiments import device_map_for_embodiment, get_embodiment
from ai_embedded_dynamic_diversity.sim.viz_cli import (
    VizParams, _controls_for_step, _simulate as _sim_legacy,
    _clone_state, _classify_adaptation_signature,
)

OUT = Path("artifacts/new-env-evolution")
OUT.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
STEPS = 120
REMAP_EVERY = 18
SEED = 42

# ---------------------------------------------------------------------------
# Load helpers
# ---------------------------------------------------------------------------

def load_model(path: str) -> tuple[ModelCore, ModelConfig]:
    ck = torch.load(path, map_location=DEVICE, weights_only=False)
    cfg = ModelConfig(**ck["model_config"])
    m = ModelCore(**asdict(cfg)).to(DEVICE)
    m.load_state_dict(ck["model"], strict=False)
    m.eval()
    return m, cfg


def make_world(wcfg: WorldConfig) -> SignalingWorld:
    return SignalingWorld(
        wcfg.x, wcfg.y, wcfg.z, wcfg.resource_channels, wcfg.decay,
        device=str(DEVICE),
        actuation_delay_steps=wcfg.actuation_delay_steps,
        actuation_noise_std=wcfg.actuation_noise_std,
        sensor_latency_steps=wcfg.sensor_latency_steps,
        sensor_dropout_burst_prob=wcfg.sensor_dropout_burst_prob,
        surface_friction_scale=wcfg.surface_friction_scale,
        disturbance_correlation_horizon=wcfg.disturbance_correlation_horizon,
        num_occlusion_objects=wcfg.num_occlusion_objects,
        occlusion_seed=wcfg.occlusion_seed,
        num_physics_objects=wcfg.num_physics_objects,
        phys_mass=wcfg.phys_mass,
        phys_friction=wcfg.phys_friction,
        hazard_zones=wcfg.hazard_zones,
    )


# ---------------------------------------------------------------------------
# Simulation helper for the rich world (returns extra channels)
# ---------------------------------------------------------------------------

def simulate_rich(
    model: ModelCore,
    cfg: ModelConfig,
    world: SignalingWorld,
    wcfg: WorldConfig,
    params: VizParams,
    embodiment_name: str,
    seed_offset: int,
) -> dict:
    torch.manual_seed(seed_offset)
    model.eval()
    state = world.init(batch_size=1)
    memory = model.init_memory(1, cfg.memory_slots, cfg.memory_dim, DEVICE)
    embodiment = get_embodiment(embodiment_name)
    mapping = device_map_for_embodiment(cfg.io_channels, embodiment, device=DEVICE, permutation_seed=seed_offset)
    proj = torch.randn(cfg.signal_dim, len(embodiment.controls), device=DEVICE) * 0.3

    life_frames, stress_frames, occ_frames = [], [], []
    mismatch_v, vitality_v, stress_v, wind_v, force_v = [], [], [], [], []
    hazard_v = []
    phys_traj = [[] for _ in range(wcfg.num_physics_objects)]
    remap_steps = []

    for step in range(params.steps):
        remap_code = torch.zeros(1, cfg.max_remap_groups, device=DEVICE)
        if step > 0 and step % params.remap_every == 0:
            remap_steps.append(step)
            mapping = device_map_for_embodiment(cfg.io_channels, embodiment, device=DEVICE, permutation_seed=seed_offset + step)
            remap_code[:, step % cfg.max_remap_groups] = 1.0

        controls = _controls_for_step(world, params, step, DEVICE)
        with torch.no_grad():
            obs = world.encode_observation(state, signal_dim=cfg.signal_dim)
            out = model(obs, memory, remap_code)
            memory = out["memory"]
            desired = torch.tanh(obs @ proj)
            applied = out["io"] @ mapping
            mismatch_v.append(float(torch.mean((applied - desired) ** 2).item()))
            action = applied.mean(dim=1, keepdim=True).repeat(1, world.x * world.y * world.z)
            state = world.step(state, action, controls=controls)

        vitality_v.append(float(state.life.mean().item()))
        stress_v.append(float(state.stress.mean().item()))
        wind_v.append(float(torch.norm(controls.wind, dim=1).mean().item()))
        force_v.append(float((controls.force_strength * controls.force_active).mean().item()))
        hazard_v.append(float(state.stress.max().item()))

        # Mid-Z slice for life and stress
        mid_z = world.z // 2
        life_frames.append(state.life[0, 0, mid_z].detach().cpu().numpy().copy())
        stress_frames.append(state.stress[0, 0, mid_z].detach().cpu().numpy().copy())
        occ_frames.append(state.occlusion_mask[0, 0, mid_z].detach().cpu().numpy().copy())

        for i in range(wcfg.num_physics_objects):
            phys_traj[i].append(state.phys_pos[0, i].detach().cpu().tolist())

    return {
        "life_frames": life_frames,
        "stress_frames": stress_frames,
        "occ_frames": occ_frames,
        "mismatch_values": mismatch_v,
        "vitality_values": vitality_v,
        "stress_values": stress_v,
        "wind_values": wind_v,
        "force_values": force_v,
        "hazard_values": hazard_v,
        "phys_traj": phys_traj,
        "remap_steps": remap_steps,
        "object_pos": [],
    }


# ---------------------------------------------------------------------------
# Scenario param helpers
# ---------------------------------------------------------------------------

def _base_params(force_mode: str = "continuous-blow", volatility_wind: float = 0.6) -> VizParams:
    return VizParams(
        steps=STEPS, remap_every=REMAP_EVERY,
        force_mode=force_mode, force_start=12, force_duration=80, force_sustain=0.95,
        force_x=0.9, force_y=0.2, force_z=0.0,
        wind_x=volatility_wind, wind_y=0.3, wind_z=0.0, wind_variation=0.35,
        light_x=-0.35, light_y=0.0, light_z=0.25, light_intensity=0.65,
        light_drift_x=0.005, light_drift_y=0.001, light_drift_z=0.0,
    )


def _blackout_params() -> VizParams:
    return VizParams(
        steps=STEPS, remap_every=REMAP_EVERY,
        force_mode="press", force_start=20, force_duration=50, force_sustain=0.85,
        force_x=0.7, force_y=0.0, force_z=0.0,
        wind_x=0.25, wind_y=0.0, wind_z=0.0, wind_variation=0.15,
        light_x=0.3, light_y=0.0, light_z=0.1, light_intensity=0.12,
        light_drift_x=-0.004, light_drift_y=0.0, light_drift_z=0.0,
    )


# ---------------------------------------------------------------------------
# GIF 1: New champion in storm scenario — 4-panel (life, stress, metrics, phys)
# ---------------------------------------------------------------------------

def viz_champion_storm(champ_model, champ_cfg, wcfg_new, world_new):
    print("  [1/5] champion storm scenario...")
    params = _base_params("continuous-blow", volatility_wind=0.75)
    r = simulate_rich(champ_model, champ_cfg, world_new, wcfg_new, params, "car", SEED)

    frames = len(r["life_frames"])
    fig = plt.figure(figsize=(14, 8))
    gs = GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)
    ax_life   = fig.add_subplot(gs[0, 0])
    ax_stress = fig.add_subplot(gs[0, 1])
    ax_adapt  = fig.add_subplot(gs[0, 2])
    ax_env    = fig.add_subplot(gs[1, 0])
    ax_phys   = fig.add_subplot(gs[1, 1])
    ax_haz    = fig.add_subplot(gs[1, 2])

    # Static occlusion overlay
    occ = r["occ_frames"][0]
    h_life   = ax_life.imshow(r["life_frames"][0], cmap="viridis", vmin=0, vmax=1)
    ax_life.imshow(occ, cmap="Reds", alpha=0.45, vmin=0, vmax=1)
    ax_life.set_title("Life (mid-Z) + Occlusion")

    h_stress = ax_stress.imshow(r["stress_frames"][0], cmap="hot", vmin=0, vmax=1)
    ax_stress.set_title("Stress field")

    # Adaptation lines
    ax_adapt.set_xlim(0, frames - 1); ax_adapt.set_ylim(0, 1.15)
    l_mis, = ax_adapt.plot([], [], color="crimson",    label="mismatch")
    l_vit, = ax_adapt.plot([], [], color="limegreen",  label="vitality")
    for rs in r["remap_steps"]:
        ax_adapt.axvline(rs, color="dodgerblue", alpha=0.3, lw=1)
    ax_adapt.legend(fontsize=7); ax_adapt.set_title("Adaptation")

    # Env controls
    ax_env.set_xlim(0, frames - 1); ax_env.set_ylim(0, max(max(r["wind_values"]), max(r["force_values"]), 0.01) * 1.2)
    l_wind,  = ax_env.plot([], [], color="cyan",   label="wind")
    l_force, = ax_env.plot([], [], color="orange", label="force")
    ax_env.legend(fontsize=7); ax_env.set_title("Environment")

    # Physics object trajectories
    colors_phys = ["magenta", "cyan", "yellow"]
    phys_lines = []
    phys_pts   = []
    ax_phys.set_xlim(-1.1, 1.1); ax_phys.set_ylim(-1.1, 1.1)
    ax_phys.grid(True, alpha=0.2); ax_phys.set_title("Physics obj. (X-Y)")
    for i, traj in enumerate(r["phys_traj"]):
        ln, = ax_phys.plot([], [], color=colors_phys[i % len(colors_phys)], lw=1.2, alpha=0.7, label=f"obj {i}")
        pt, = ax_phys.plot([], [], "o", color=colors_phys[i % len(colors_phys)], ms=5)
        phys_lines.append(ln); phys_pts.append(pt)
    ax_phys.legend(fontsize=7)

    # Hazard (max stress proxy)
    ax_haz.set_xlim(0, frames - 1); ax_haz.set_ylim(0, 1.05)
    l_haz, = ax_haz.plot([], [], color="red", lw=1.5, label="peak stress")
    ax_haz.axhline(0.5, color="orange", ls="--", lw=0.8, alpha=0.6)
    ax_haz.legend(fontsize=7); ax_haz.set_title("Hazard pressure")

    fig.suptitle("New Champion — new_env_v1 — Storm (car)", fontsize=12)

    def _init():
        h_life.set_data(r["life_frames"][0])
        h_stress.set_data(r["stress_frames"][0])
        l_mis.set_data([], []); l_vit.set_data([], [])
        l_wind.set_data([], []); l_force.set_data([], [])
        l_haz.set_data([], [])
        for ln, pt in zip(phys_lines, phys_pts):
            ln.set_data([], []); pt.set_data([], [])
        return [h_life, h_stress, l_mis, l_vit, l_wind, l_force, l_haz] + phys_lines + phys_pts

    def _update(i):
        xs = list(range(i + 1))
        h_life.set_data(r["life_frames"][i])
        h_stress.set_data(r["stress_frames"][i])
        l_mis.set_data(xs, r["mismatch_values"][:i+1])
        l_vit.set_data(xs, r["vitality_values"][:i+1])
        l_wind.set_data(xs, r["wind_values"][:i+1])
        l_force.set_data(xs, r["force_values"][:i+1])
        l_haz.set_data(xs, r["hazard_values"][:i+1])
        for k, (ln, pt) in enumerate(zip(phys_lines, phys_pts)):
            if k < len(r["phys_traj"]) and r["phys_traj"][k]:
                px = [p[0] for p in r["phys_traj"][k][:i+1]]
                py = [p[1] for p in r["phys_traj"][k][:i+1]]
                ln.set_data(px, py)
                if px: pt.set_data([px[-1]], [py[-1]])
        return [h_life, h_stress, l_mis, l_vit, l_wind, l_force, l_haz] + phys_lines + phys_pts

    anim = animation.FuncAnimation(fig, _update, init_func=_init, frames=frames, interval=70, blit=True)
    out = str(OUT / "viz-new-env-champion-storm.gif")
    anim.save(out, writer=animation.PillowWriter(fps=14))
    plt.close(fig)
    print(f"    -> {out}")
    return out


# ---------------------------------------------------------------------------
# GIF 2: Champion vs Legacy (v08) comparison — storm + blackout side-by-side
# ---------------------------------------------------------------------------

def viz_compare(champ_model, champ_cfg, legacy_model, legacy_cfg, wcfg_new, world_new):
    print("  [2/5] champion vs legacy comparison (storm + blackout)...")
    scenarios = [("storm", _base_params("continuous-blow", 0.75)), ("blackout", _blackout_params())]
    outputs = []

    for scenario_name, params in scenarios:
        r_new = simulate_rich(champ_model, champ_cfg, world_new, wcfg_new, params, "hexapod", SEED + 1)
        r_leg = simulate_rich(legacy_model, legacy_cfg, world_new, wcfg_new, params, "hexapod", SEED + 2)

        frames = min(len(r_new["life_frames"]), len(r_leg["life_frames"]))
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        fig.suptitle(f"new-env champion vs champion-v08  |  {scenario_name}  |  hexapod", fontsize=11)

        ax_nl, ax_ll = axes[0, 0], axes[0, 1]
        ax_lines     = axes[0, 2]
        ax_ns, ax_ls = axes[1, 0], axes[1, 1]
        ax_phys      = axes[1, 2]

        h_nl = ax_nl.imshow(r_new["life_frames"][0], cmap="viridis", vmin=0, vmax=1)
        ax_nl.imshow(r_new["occ_frames"][0], cmap="Reds", alpha=0.4, vmin=0, vmax=1)
        ax_nl.set_title("New champion — Life")

        h_ll = ax_ll.imshow(r_leg["life_frames"][0], cmap="viridis", vmin=0, vmax=1)
        ax_ll.imshow(r_leg["occ_frames"][0], cmap="Reds", alpha=0.4, vmin=0, vmax=1)
        ax_ll.set_title("champion-v08 — Life")

        ax_lines.set_xlim(0, frames - 1)
        ymax = max(max(r_new["mismatch_values"]), max(r_leg["mismatch_values"]),
                   max(r_new["vitality_values"]), max(r_leg["vitality_values"]), 0.01) * 1.15
        ax_lines.set_ylim(0, ymax)
        lnm, = ax_lines.plot([], [], color="tab:blue",   label="new mismatch")
        llm, = ax_lines.plot([], [], color="tab:orange", label="v08 mismatch")
        lnv, = ax_lines.plot([], [], color="tab:cyan",   label="new vitality",  ls="--")
        llv, = ax_lines.plot([], [], color="tab:red",    label="v08 vitality",  ls="--")
        ax_lines.legend(fontsize=7); ax_lines.set_title("Mismatch / Vitality")
        for rs in r_new["remap_steps"]:
            ax_lines.axvline(rs, color="gray", alpha=0.25, lw=0.8)

        h_ns = ax_ns.imshow(r_new["stress_frames"][0], cmap="hot", vmin=0, vmax=1)
        ax_ns.set_title("New — Stress")
        h_ls = ax_ls.imshow(r_leg["stress_frames"][0], cmap="hot", vmin=0, vmax=1)
        ax_ls.set_title("v08 — Stress")

        ax_phys.set_xlim(-1.1, 1.1); ax_phys.set_ylim(-1.1, 1.1)
        ax_phys.grid(True, alpha=0.2); ax_phys.set_title("Phys obj trajectories (new champ)")
        colors_p = ["magenta", "cyan", "yellow"]
        pl, pp = [], []
        for i in range(len(r_new["phys_traj"])):
            ln, = ax_phys.plot([], [], color=colors_p[i % 3], lw=1.2, alpha=0.7)
            pt, = ax_phys.plot([], [], "o", color=colors_p[i % 3], ms=5)
            pl.append(ln); pp.append(pt)

        def _init():
            h_nl.set_data(r_new["life_frames"][0]); h_ll.set_data(r_leg["life_frames"][0])
            h_ns.set_data(r_new["stress_frames"][0]); h_ls.set_data(r_leg["stress_frames"][0])
            lnm.set_data([], []); llm.set_data([], [])
            lnv.set_data([], []); llv.set_data([], [])
            for ln, pt in zip(pl, pp): ln.set_data([], []); pt.set_data([], [])
            return [h_nl, h_ll, h_ns, h_ls, lnm, llm, lnv, llv] + pl + pp

        def _update(i):
            xs = list(range(i + 1))
            h_nl.set_data(r_new["life_frames"][i]); h_ll.set_data(r_leg["life_frames"][i])
            h_ns.set_data(r_new["stress_frames"][i]); h_ls.set_data(r_leg["stress_frames"][i])
            lnm.set_data(xs, r_new["mismatch_values"][:i+1])
            llm.set_data(xs, r_leg["mismatch_values"][:i+1])
            lnv.set_data(xs, r_new["vitality_values"][:i+1])
            llv.set_data(xs, r_leg["vitality_values"][:i+1])
            for k, (ln, pt) in enumerate(zip(pl, pp)):
                if k < len(r_new["phys_traj"]) and r_new["phys_traj"][k]:
                    px = [p[0] for p in r_new["phys_traj"][k][:i+1]]
                    py = [p[1] for p in r_new["phys_traj"][k][:i+1]]
                    ln.set_data(px, py)
                    if px: pt.set_data([px[-1]], [py[-1]])
            return [h_nl, h_ll, h_ns, h_ls, lnm, llm, lnv, llv] + pl + pp

        anim = animation.FuncAnimation(fig, _update, init_func=_init, frames=frames, interval=70, blit=True)
        out = str(OUT / f"viz-new-env-compare-{scenario_name}.gif")
        anim.save(out, writer=animation.PillowWriter(fps=14))
        plt.close(fig)
        print(f"    -> {out}")
        outputs.append(out)
    return outputs


# ---------------------------------------------------------------------------
# GIF 3: Hazard-sweep panel — champion in each of the 3 hazard kinds
# ---------------------------------------------------------------------------

def viz_hazard_sweep(champ_model, champ_cfg, wcfg_new, world_new):
    print("  [3/5] hazard sweep (3 hazard kinds)...")
    # Run under 3 different lighting / wind configs to activate each hazard type

    configs = [
        ("light_triggered", VizParams(
            steps=STEPS, remap_every=REMAP_EVERY,
            force_mode="press", force_start=20, force_duration=50, force_sustain=0.7,
            force_x=0.6, force_y=0.0, force_z=0.0,
            wind_x=0.1, wind_y=0.0, wind_z=0.0, wind_variation=0.05,
            light_x=0.75, light_y=0.25, light_z=0.0, light_intensity=0.9,  # bright — triggers light hazard
            light_drift_x=0.0, light_drift_y=0.002, light_drift_z=0.0,
        )),
        ("airflow", VizParams(
            steps=STEPS, remap_every=REMAP_EVERY,
            force_mode="continuous-blow", force_start=8, force_duration=80, force_sustain=1.1,
            force_x=1.1, force_y=0.4, force_z=0.0,
            wind_x=0.85, wind_y=0.5, wind_z=0.0, wind_variation=0.45,  # high wind — triggers airflow hazard
            light_x=-0.3, light_y=0.0, light_z=0.2, light_intensity=0.3,
            light_drift_x=0.003, light_drift_y=0.0, light_drift_z=0.0,
        )),
        ("periodic", VizParams(
            steps=STEPS, remap_every=REMAP_EVERY,
            force_mode="press", force_start=14, force_duration=40, force_sustain=0.8,
            force_x=0.7, force_y=0.1, force_z=0.0,
            wind_x=0.2, wind_y=0.1, wind_z=0.0, wind_variation=0.1,
            light_x=0.0, light_y=0.0, light_z=0.5, light_intensity=0.55,
            light_drift_x=0.0, light_drift_y=0.0, light_drift_z=0.0,
        )),
    ]

    results = {}
    for label, params in configs:
        results[label] = simulate_rich(champ_model, champ_cfg, world_new, wcfg_new, params, "drone", SEED + 10)

    frames = min(len(v["life_frames"]) for v in results.values())
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle("new-env champion — Hazard sweep (drone) | light / airflow / periodic", fontsize=11)

    row0_ax = [axes[0, 0], axes[0, 1], axes[0, 2]]
    row1_ax = [axes[1, 0], axes[1, 1], axes[1, 2]]
    labels = list(results.keys())

    heats, stresses, mis_lines, vit_lines = {}, {}, {}, {}
    for idx, lbl in enumerate(labels):
        r = results[lbl]
        h = row0_ax[idx].imshow(r["life_frames"][0], cmap="viridis", vmin=0, vmax=1)
        row0_ax[idx].imshow(r["occ_frames"][0], cmap="Reds", alpha=0.4, vmin=0, vmax=1)
        row0_ax[idx].set_title(f"{lbl} — Life")
        heats[lbl] = h

        s = row1_ax[idx].imshow(r["stress_frames"][0], cmap="hot", vmin=0, vmax=1)
        row1_ax[idx].set_title(f"{lbl} — Stress")
        stresses[lbl] = s

    fig2, ax2 = plt.subplots(1, 1, figsize=(7, 3))
    colors_h = {"light_triggered": "gold", "airflow": "cyan", "periodic": "magenta"}
    for lbl in labels:
        r = results[lbl]
        ax2.plot(r["hazard_values"], color=colors_h[lbl], label=lbl, lw=1.5)
    ax2.set_title("Peak stress by hazard type"); ax2.legend(fontsize=8)
    ax2.set_xlabel("step"); ax2.set_ylabel("peak stress")
    fig2.tight_layout()
    static_out = str(OUT / "viz-hazard-sweep-static.png")
    fig2.savefig(static_out, dpi=120)
    plt.close(fig2)
    print(f"    -> {static_out} (static hazard comparison)")

    def _init():
        arts = []
        for lbl in labels:
            heats[lbl].set_data(results[lbl]["life_frames"][0])
            stresses[lbl].set_data(results[lbl]["stress_frames"][0])
            arts += [heats[lbl], stresses[lbl]]
        return tuple(arts)

    def _update(i):
        arts = []
        for lbl in labels:
            heats[lbl].set_data(results[lbl]["life_frames"][i])
            stresses[lbl].set_data(results[lbl]["stress_frames"][i])
            arts += [heats[lbl], stresses[lbl]]
        return tuple(arts)

    anim = animation.FuncAnimation(fig, _update, init_func=_init, frames=frames, interval=75, blit=True)
    out = str(OUT / "viz-new-env-champion-hazard-sweep.gif")
    anim.save(out, writer=animation.PillowWriter(fps=13))
    plt.close(fig)
    print(f"    -> {out}")
    return out


# ---------------------------------------------------------------------------
# GIF 4: Training progress — fitness + GDI over 40 generations
# ---------------------------------------------------------------------------

def viz_training_progress():
    print("  [4/5] training progress chart...")
    metrics_path = OUT / "champion-new-env-v1.metrics.json"
    if not metrics_path.exists():
        print("    SKIP — metrics file not found")
        return None

    records = json.loads(metrics_path.read_text())["records"]
    gens      = [r["generation"] for r in records]
    best_fit  = [r["best_fitness"] for r in records]
    mean_fit  = [r["mean_fitness"] for r in records]
    gdi       = [r.get("gdi", 0) for r in records]
    w_div     = [r.get("weight_div", 0) for r in records]
    b_div     = [r.get("behavior_div", 0) for r in records]
    lin_ent   = [r.get("lineage_entropy", 0) for r in records]
    wpred     = [r.get("mean_world_pred_loss", 0) for r in records]
    sig_rel   = [r.get("mean_signal_reliability", 0) for r in records]

    fig, axes = plt.subplots(2, 3, figsize=(15, 7))
    fig.suptitle("Evolution in new_env_v1 — 40 generations (pi5 × pop-6)", fontsize=12)

    axes[0,0].plot(gens, best_fit,  color="gold",      lw=2, label="best fitness")
    axes[0,0].plot(gens, mean_fit,  color="lightblue", lw=1, ls="--", label="mean fitness")
    axes[0,0].fill_between(gens, mean_fit, best_fit, alpha=0.15, color="gold")
    axes[0,0].set_title("Fitness"); axes[0,0].legend(fontsize=8)
    axes[0,0].axhline(0, color="white", lw=0.5, alpha=0.4)

    axes[0,1].plot(gens, gdi,     color="violet",    lw=2, label="GDI composite")
    axes[0,1].plot(gens, b_div,   color="cyan",      lw=1, label="behavior_div")
    axes[0,1].plot(gens, lin_ent, color="lime",      lw=1, ls="--", label="lineage entropy")
    axes[0,1].set_title("Genetic Diversity Index"); axes[0,1].legend(fontsize=8)

    axes[0,2].plot(gens, w_div, color="orange", lw=1.5, label="weight_div")
    axes[0,2].set_title("Weight Diversity"); axes[0,2].legend(fontsize=8)

    axes[1,0].plot(gens, wpred, color="tomato", lw=1.5, label="world pred loss")
    axes[1,0].set_title("JEPA World Predictor Loss"); axes[1,0].legend(fontsize=8)

    axes[1,1].plot(gens, sig_rel, color="deepskyblue", lw=1.5, label="signal reliability")
    axes[1,1].set_title("Signal Reliability"); axes[1,1].legend(fontsize=8)
    axes[1,1].set_ylim(0, 1.05)

    env_vol = [r.get("env_volatility", 0) for r in records]
    noise   = [r.get("noise_strength", 0) for r in records]
    axes[1,2].plot(gens, env_vol, color="salmon",      lw=1.5, label="env volatility")
    axes[1,2].plot(gens, noise,   color="plum",        lw=1.5, ls="--", label="noise strength")
    axes[1,2].set_title("Curriculum Schedule"); axes[1,2].legend(fontsize=8)

    for ax in axes.flat:
        ax.set_xlabel("generation", fontsize=8)
        ax.grid(True, alpha=0.15)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = str(OUT / "viz-training-progress.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"    -> {out}")
    return out


# ---------------------------------------------------------------------------
# GIF 5: Phase-1 evaluation comparison bar chart
# ---------------------------------------------------------------------------

def viz_phase1_eval():
    print("  [5/5] phase-1 evaluation bar chart...")
    eval_path = OUT / "phase1-eval.json"
    if not eval_path.exists():
        print("    SKIP — eval file not found")
        return None

    records = json.loads(eval_path.read_text())
    names   = [Path(r["checkpoint"]).stem for r in records]
    # Shorten names
    names   = [n.replace("model-core-", "").replace("coevo-guardrail-", "")
                .replace("shared-extreme-", "se-").replace("champion-", "v")
                .replace("transfercal-", "tc-") for n in names]

    worlds  = ["legacy_20x20x10", "base_40x40x20", "new_env_v1"]
    colors  = ["#4a90d9", "#7bc87a", "#e67e22"]
    labels  = ["Legacy 20×20×10", "Base 40×40×20", "new_env_v1"]

    x = range(len(names))
    w = 0.25
    fig, ax = plt.subplots(figsize=(12, 5))
    for i, (world, col, lbl) in enumerate(zip(worlds, colors, labels)):
        vals = [r.get(world, 0) for r in records]
        offset = (i - 1) * w
        bars = ax.bar([xi + offset for xi in x], vals, width=w, color=col, label=lbl, alpha=0.85)

    ax.set_xticks(list(x))
    ax.set_xticklabels(names, rotation=25, ha="right", fontsize=9)
    ax.axhline(0, color="white", lw=0.8, alpha=0.5)
    ax.set_ylabel("Fitness")
    ax.set_title("Phase-1 Cross-World Evaluation — Top 5 Champions", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.15, axis="y")
    fig.tight_layout()
    out = str(OUT / "viz-phase1-eval.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"    -> {out}")
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(SEED)

    print("Loading models...")
    champ_model,  champ_cfg  = load_model("artifacts/new-env-evolution/champion-new-env-v1.pt")
    legacy_model, legacy_cfg = load_model("artifacts/model-core-champion-v08.pt")

    wcfg_new = world_config_for_profile("new_env_v1")
    world_new = make_world(wcfg_new)

    print("\nGenerating visualizations...")
    viz_champion_storm(champ_model, champ_cfg, wcfg_new, world_new)
    viz_compare(champ_model, champ_cfg, legacy_model, legacy_cfg, wcfg_new, world_new)
    viz_hazard_sweep(champ_model, champ_cfg, wcfg_new, world_new)
    viz_training_progress()
    viz_phase1_eval()

    print("\nAll visualizations saved to", OUT)
