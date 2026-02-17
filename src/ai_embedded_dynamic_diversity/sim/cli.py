from __future__ import annotations

import json
import time
from pathlib import Path

import torch
import typer

from ai_embedded_dynamic_diversity.config import ModelConfig, WorldConfig, model_config_for_profile
from ai_embedded_dynamic_diversity.models import ModelCore
from ai_embedded_dynamic_diversity.sim.embodiments import embodiment_dof_table, device_map_for_embodiment, get_embodiment
from ai_embedded_dynamic_diversity.sim.world import DynamicDiversityWorld

app = typer.Typer(add_completion=False)


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(v) for v in values)
    rank = pct * (len(ordered) - 1)
    lo = int(rank)
    hi = min(len(ordered) - 1, lo + 1)
    alpha = rank - lo
    return (1.0 - alpha) * ordered[lo] + alpha * ordered[hi]


def _load_model(weights: str, profile: str, device: torch.device) -> tuple[ModelCore, ModelConfig, dict]:
    ckpt_meta: dict = {}
    if weights:
        raw = torch.load(weights, map_location=device)
        if "model_config" in raw:
            cfg = ModelConfig(**raw["model_config"])
        else:
            cfg = model_config_for_profile(raw.get("profile", profile))
        model = ModelCore(**cfg.__dict__).to(device)
        model.load_state_dict(raw["model"])
        ckpt_meta = {
            "weights": weights,
            "profile": raw.get("profile", profile),
            "run_flags": raw.get("run_flags", {}),
        }
    else:
        cfg = model_config_for_profile(profile)
        model = ModelCore(**cfg.__dict__).to(device)
        ckpt_meta = {
            "weights": "",
            "profile": profile,
            "run_flags": {},
        }
    model.eval()
    return model, cfg, ckpt_meta


def profile_embodiment_metrics(
    embodiment: str,
    profile: str,
    weights: str,
    steps: int,
    batch_size: int,
    remap_every: int,
    world_x: int,
    world_y: int,
    world_z: int,
    resource_channels: int,
    env_volatility: float,
    firing_threshold: float,
    readiness_active_threshold: float,
    device: str,
    seed: int,
) -> dict:
    if steps <= 0:
        raise ValueError("steps must be > 0")
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    if remap_every < 0:
        raise ValueError("remap_every must be >= 0")
    if env_volatility < 0.0:
        raise ValueError("env_volatility must be >= 0.0")

    torch.manual_seed(seed)
    dev = torch.device(device)
    emb = get_embodiment(embodiment)
    model, cfg, ckpt_meta = _load_model(weights, profile, dev)

    world = DynamicDiversityWorld(world_x, world_y, world_z, resource_channels, decay=0.03, device=str(dev))
    state = world.init(batch_size=batch_size)
    memory = model.init_memory(batch_size, cfg.memory_slots, cfg.memory_dim, dev)

    control_dim = len(emb.controls)
    projection = torch.randn(cfg.signal_dim, control_dim, device=dev) * 0.3

    mapping_seed = seed + 37
    mapping = device_map_for_embodiment(cfg.io_channels, emb, device=dev, permutation_seed=mapping_seed)

    step_latency_ms: list[float] = []
    mismatch_values: list[float] = []
    vitality_values: list[float] = []
    stress_values: list[float] = []
    energy_values: list[float] = []
    readiness_sparsity_values: list[float] = []
    readiness_sat_low_values: list[float] = []
    readiness_sat_high_values: list[float] = []
    channel_firing_values: list[float] = []
    memory_entropy_values: list[float] = []
    remap_events = 0
    channel_usage_acc = torch.zeros(control_dim, device=dev)

    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)

    for step_index in range(steps):
        t0 = time.perf_counter()
        remap_code = torch.zeros(batch_size, cfg.max_remap_groups, device=dev)
        if remap_every > 0 and step_index > 0 and step_index % remap_every == 0:
            remap_events += 1
            mapping_seed += 97
            mapping = device_map_for_embodiment(cfg.io_channels, emb, device=dev, permutation_seed=mapping_seed)
            remap_code[:, step_index % cfg.max_remap_groups] = 1.0

        controls = world.random_controls(batch_size, env_volatility, step_index=step_index)
        with torch.no_grad():
            obs = world.encode_observation(state, signal_dim=cfg.signal_dim)
            out = model(obs, memory, remap_code)
            memory = out["memory"]

            desired = torch.tanh(obs @ projection)
            applied = out["io"] @ mapping
            mismatch = torch.mean((applied - desired) ** 2)
            mismatch_values.append(float(mismatch.item()))

            action_field = applied.mean(dim=1, keepdim=True).repeat(1, world_x * world_y * world_z)
            state = world.step(state, action_field, controls=controls)
            vitality_values.append(float(state.life.mean().item()))
            stress_values.append(float(state.stress.mean().item()))
            energy_values.append(float(out["energy"].mean().item()))

            readiness = out["readiness"]
            readiness_sparsity_values.append(float((readiness < readiness_active_threshold).float().mean().item()))
            readiness_sat_low_values.append(float((readiness < 0.05).float().mean().item()))
            readiness_sat_high_values.append(float((readiness > 0.95).float().mean().item()))
            channel_firing_values.append(float((torch.abs(applied) > firing_threshold).float().mean().item()))

            weights_tensor = out["memory_weights"].clamp_min(1e-8)
            normalizer = torch.log(torch.tensor(float(weights_tensor.shape[1]), device=dev)).clamp_min(1e-8)
            entropy = -(weights_tensor * weights_tensor.log()).sum(dim=1) / normalizer
            memory_entropy_values.append(float(entropy.mean().item()))

            channel_usage_acc += torch.abs(applied).mean(dim=0)

        if dev.type == "cuda":
            torch.cuda.synchronize(dev)
        step_latency_ms.append((time.perf_counter() - t0) * 1000.0)

    channel_usage = (channel_usage_acc / max(1, steps)).detach().cpu().tolist()
    low_count = min(8, len(channel_usage))
    low_usage_indices = sorted(range(len(channel_usage)), key=lambda i: channel_usage[i])[:low_count]
    high_usage_indices = sorted(range(len(channel_usage)), key=lambda i: channel_usage[i], reverse=True)[:low_count]

    mapping_coverage = float((mapping.sum(dim=0) > 0).float().mean().item())
    model_param_count = sum(int(p.numel()) for p in model.parameters())
    model_param_bytes = sum(int(p.numel()) * int(p.element_size()) for p in model.parameters())
    state_bytes = (
        state.life.numel() * state.life.element_size()
        + state.resources.numel() * state.resources.element_size()
        + state.stress.numel() * state.stress.element_size()
        + state.object_pos.numel() * state.object_pos.element_size()
        + state.object_vel.numel() * state.object_vel.element_size()
    )
    memory_bytes = memory.numel() * memory.element_size()
    peak_gpu_bytes = int(torch.cuda.max_memory_allocated(dev)) if dev.type == "cuda" else 0

    return {
        "embodiment": {
            "name": emb.name,
            "control_dof": len(emb.controls),
            "sensor_channels": len(emb.sensors),
        },
        "model": {
            "profile": ckpt_meta.get("profile", profile),
            "weights": ckpt_meta.get("weights", ""),
            "config": cfg.__dict__,
            "run_flags": ckpt_meta.get("run_flags", {}),
            "params_million": model_param_count / 1_000_000.0,
            "size_mb": model_param_bytes / (1024.0 * 1024.0),
        },
        "runtime": {
            "device": str(dev),
            "batch_size": batch_size,
            "steps": steps,
            "remap_events": remap_events,
            "step_latency_ms_p50": _percentile(step_latency_ms, 0.50),
            "step_latency_ms_p95": _percentile(step_latency_ms, 0.95),
            "state_mb": state_bytes / (1024.0 * 1024.0),
            "memory_tensor_mb": memory_bytes / (1024.0 * 1024.0),
            "peak_gpu_alloc_mb": peak_gpu_bytes / (1024.0 * 1024.0),
        },
        "metrics": {
            "mean_mismatch": sum(mismatch_values) / max(1, len(mismatch_values)),
            "mean_vitality": sum(vitality_values) / max(1, len(vitality_values)),
            "mean_stress": sum(stress_values) / max(1, len(stress_values)),
            "mean_energy": sum(energy_values) / max(1, len(energy_values)),
            "channel_firing_fraction": sum(channel_firing_values) / max(1, len(channel_firing_values)),
            "readiness_sparsity": sum(readiness_sparsity_values) / max(1, len(readiness_sparsity_values)),
            "readiness_saturation_low": sum(readiness_sat_low_values) / max(1, len(readiness_sat_low_values)),
            "readiness_saturation_high": sum(readiness_sat_high_values) / max(1, len(readiness_sat_high_values)),
            "memory_weight_entropy": sum(memory_entropy_values) / max(1, len(memory_entropy_values)),
            "mapping_coverage": mapping_coverage,
        },
        "io_profile": {
            "low_usage_channels": [{"index": int(i), "usage": float(channel_usage[i])} for i in low_usage_indices],
            "high_usage_channels": [{"index": int(i), "usage": float(channel_usage[i])} for i in high_usage_indices],
        },
    }


@app.command()
def rollout(steps: int = 10, batch_size: int = 2, device: str = "cpu") -> None:
    cfg = WorldConfig()
    world = DynamicDiversityWorld(cfg.x, cfg.y, cfg.z, cfg.resource_channels, cfg.decay, device=device)
    state = world.init(batch_size)
    action = torch.zeros(batch_size, cfg.x * cfg.y * cfg.z, device=device)
    for _ in range(steps):
        state = world.step(state, action)
    obs = world.encode_observation(state, signal_dim=32)
    print({"obs_shape": tuple(obs.shape), "life_mean": state.life.mean().item(), "stress_mean": state.stress.mean().item()})


@app.command()
def embodiments() -> None:
    rows = embodiment_dof_table()
    print({"embodiments": rows})


@app.command()
def profiler(
    embodiment: str = "polymorph120",
    profile: str = "pi5",
    weights: str = "",
    steps: int = 120,
    batch_size: int = 4,
    remap_every: int = 15,
    world_x: int = 20,
    world_y: int = 20,
    world_z: int = 10,
    resource_channels: int = 5,
    env_volatility: float = 0.30,
    firing_threshold: float = 0.20,
    readiness_active_threshold: float = 0.20,
    device: str = "cpu",
    seed: int = 13,
    output: str = "artifacts/embodiment-profile.json",
) -> None:
    try:
        payload = profile_embodiment_metrics(
            embodiment=embodiment,
            profile=profile,
            weights=weights,
            steps=steps,
            batch_size=batch_size,
            remap_every=remap_every,
            world_x=world_x,
            world_y=world_y,
            world_z=world_z,
            resource_channels=resource_channels,
            env_volatility=env_volatility,
            firing_threshold=firing_threshold,
            readiness_active_threshold=readiness_active_threshold,
            device=device,
            seed=seed,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        {
            "output": output,
            "embodiment": payload["embodiment"]["name"],
            "mean_mismatch": payload["metrics"]["mean_mismatch"],
            "latency_p50_ms": payload["runtime"]["step_latency_ms_p50"],
            "latency_p95_ms": payload["runtime"]["step_latency_ms_p95"],
        }
    )


if __name__ == "__main__":
    app()
