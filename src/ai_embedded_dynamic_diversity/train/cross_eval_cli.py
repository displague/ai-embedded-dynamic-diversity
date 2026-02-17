from __future__ import annotations

import glob
import json
import math
from dataclasses import dataclass
from pathlib import Path

import torch
import typer

from ai_embedded_dynamic_diversity.config import ModelConfig, model_config_for_profile
from ai_embedded_dynamic_diversity.models import ModelCore
from ai_embedded_dynamic_diversity.sim.embodiments import device_map_for_embodiment, get_embodiment
from ai_embedded_dynamic_diversity.sim.world import DynamicDiversityWorld

app = typer.Typer(add_completion=False)


@dataclass(frozen=True)
class ScenarioSpec:
    name: str
    wind: tuple[float, float, float]
    wind_variation: float
    light_pos: tuple[float, float, float]
    light_drift: tuple[float, float, float]
    light_intensity: float
    force_vector: tuple[float, float, float]
    force_strength: float
    force_start: int
    force_duration: int
    force_pattern: str


@dataclass(frozen=True)
class CapabilityProxySignals:
    bio_chemical: float
    bio_phototaxis: float
    bio_mechano: float
    tech_rf_noise: float
    tech_telemetry: float
    tech_bus_load: float
    target_signal: float
    threat_active: bool


def _scenario_catalog() -> dict[str, ScenarioSpec]:
    return {
        "mild": ScenarioSpec(
            name="mild",
            wind=(0.15, 0.0, 0.0),
            wind_variation=0.05,
            light_pos=(-0.2, 0.0, 0.2),
            light_drift=(0.001, 0.0, 0.0),
            light_intensity=0.65,
            force_vector=(0.0, 0.0, 0.0),
            force_strength=0.0,
            force_start=0,
            force_duration=0,
            force_pattern="none",
        ),
        "gust": ScenarioSpec(
            name="gust",
            wind=(0.45, 0.1, 0.0),
            wind_variation=0.25,
            light_pos=(-0.3, 0.0, 0.3),
            light_drift=(0.002, 0.0, 0.0),
            light_intensity=0.8,
            force_vector=(0.6, 0.0, 0.0),
            force_strength=0.7,
            force_start=18,
            force_duration=16,
            force_pattern="pulse",
        ),
        "force": ScenarioSpec(
            name="force",
            wind=(0.2, 0.0, 0.0),
            wind_variation=0.1,
            light_pos=(-0.15, 0.0, 0.25),
            light_drift=(0.0, 0.0, 0.0),
            light_intensity=0.7,
            force_vector=(0.85, 0.0, 0.0),
            force_strength=1.0,
            force_start=20,
            force_duration=26,
            force_pattern="decay",
        ),
        "storm": ScenarioSpec(
            name="storm",
            wind=(0.8, 0.35, 0.0),
            wind_variation=0.45,
            light_pos=(-0.4, 0.0, 0.25),
            light_drift=(0.004, 0.0, 0.0),
            light_intensity=0.55,
            force_vector=(1.1, 0.2, 0.0),
            force_strength=1.1,
            force_start=10,
            force_duration=36,
            force_pattern="sine",
        ),
        "blackout": ScenarioSpec(
            name="blackout",
            wind=(0.25, 0.0, 0.0),
            wind_variation=0.15,
            light_pos=(0.25, 0.0, 0.1),
            light_drift=(-0.004, 0.0, 0.0),
            light_intensity=0.15,
            force_vector=(0.75, 0.0, 0.0),
            force_strength=0.9,
            force_start=16,
            force_duration=26,
            force_pattern="pulse",
        ),
        "crosswind": ScenarioSpec(
            name="crosswind",
            wind=(0.15, 0.85, 0.0),
            wind_variation=0.3,
            light_pos=(-0.2, 0.15, 0.2),
            light_drift=(0.0, -0.002, 0.0),
            light_intensity=0.65,
            force_vector=(0.55, 0.85, 0.0),
            force_strength=0.95,
            force_start=14,
            force_duration=34,
            force_pattern="decay",
        ),
    }


def _resolve_scenarios(scenarios_csv: str) -> list[ScenarioSpec]:
    catalog = _scenario_catalog()
    names = [x.strip().lower() for x in scenarios_csv.split(",") if x.strip()]
    if not names:
        raise ValueError("No scenario names were provided")
    resolved: list[ScenarioSpec] = []
    for name in names:
        if name not in catalog:
            allowed = ", ".join(sorted(catalog))
            raise ValueError(f"Unknown scenario '{name}'. Allowed: {allowed}")
        resolved.append(catalog[name])
    return resolved


def _resolve_scenario_profile(profile: str) -> list[ScenarioSpec]:
    normalized = profile.strip().lower()
    profiles = {
        "standard": "mild,gust,force",
        "hardy": "gust,force,storm,blackout,crosswind",
        "extreme": "force,storm,blackout,crosswind",
    }
    if normalized not in profiles:
        allowed = ", ".join(sorted(profiles))
        raise ValueError(f"Unknown scenario profile '{profile}'. Allowed: {allowed}")
    return _resolve_scenarios(profiles[normalized])


def _parse_embodiment_weights(weights_csv: str, embodiments: list[str]) -> dict[str, float]:
    weights = {emb: 1.0 for emb in embodiments}
    normalized = weights_csv.replace(";", ",").strip()
    if not normalized:
        return weights

    for raw in normalized.split(","):
        token = raw.strip()
        if not token:
            continue
        if "=" not in token:
            raise ValueError(f"Invalid embodiment weight token '{token}'. Expected format '<embodiment>=<weight>'.")
        name, value_raw = [x.strip().lower() for x in token.split("=", maxsplit=1)]
        if name not in weights:
            allowed = ", ".join(sorted(weights))
            raise ValueError(f"Unknown embodiment '{name}' in embodiment weights. Allowed: {allowed}")
        try:
            value = float(value_raw)
        except ValueError as exc:
            raise ValueError(f"Invalid weight for embodiment '{name}': '{value_raw}'") from exc
        if value <= 0.0:
            raise ValueError(f"Weight for embodiment '{name}' must be > 0.0")
        weights[name] = value
    return weights


def _weighted_transfer_score(
    by_embodiment: dict[str, dict[str, float]],
    embodiments: list[str],
    embodiment_weights: dict[str, float],
) -> float:
    denom = sum(float(embodiment_weights.get(emb, 1.0)) for emb in embodiments)
    if denom <= 0.0:
        return 0.0
    numer = 0.0
    for emb in embodiments:
        score = float(by_embodiment.get(emb, {}).get("transfer_score", 0.0))
        weight = float(embodiment_weights.get(emb, 1.0))
        numer += score * weight
    return numer / denom


def _resolve_capability_profile(profile: str) -> str:
    normalized = profile.strip().lower()
    if not normalized:
        return "none"
    allowed = {"none", "bio-tech-v1"}
    if normalized not in allowed:
        raise ValueError(f"Unknown capability profile '{profile}'. Allowed: {', '.join(sorted(allowed))}")
    return normalized


def _safe_corr(a: list[float], b: list[float]) -> float:
    if len(a) != len(b) or len(a) < 2:
        return 0.0
    mean_a = sum(a) / len(a)
    mean_b = sum(b) / len(b)
    da = [x - mean_a for x in a]
    db = [x - mean_b for x in b]
    var_a = sum(x * x for x in da)
    var_b = sum(x * x for x in db)
    if var_a <= 1e-10 or var_b <= 1e-10:
        return 0.0
    cov = sum(x * y for x, y in zip(da, db)) / len(a)
    corr = cov / math.sqrt((var_a / len(a)) * (var_b / len(b)))
    return max(-1.0, min(1.0, corr))


def _binary_auc(scores: list[float], labels: list[int]) -> float:
    if len(scores) != len(labels) or len(scores) < 2:
        return 0.5
    pos = [float(s) for s, y in zip(scores, labels) if int(y) == 1]
    neg = [float(s) for s, y in zip(scores, labels) if int(y) == 0]
    if not pos or not neg:
        return 0.5
    wins = 0.0
    ties = 0.0
    for p in pos:
        for n in neg:
            if p > n:
                wins += 1.0
            elif p == n:
                ties += 1.0
    return (wins + 0.5 * ties) / (len(pos) * len(neg))


def _capability_proxy_signals(
    state,
    controls,
    scenario: ScenarioSpec,
    step: int,
    remap_happened: bool,
    seed: int,
) -> CapabilityProxySignals:
    life_mean = float(state.life.mean().item())
    stress_mean = float(state.stress.mean().item())
    resource_mean = float(state.resources[:, :1].mean().item())

    light_intensity = float(controls.light_intensity.mean().item())
    light_pos = controls.light_position[0]
    object_pos = state.object_pos[0]
    light_dist = float(torch.norm(light_pos - object_pos).item())
    wind_mag = float(torch.norm(controls.wind[0]).item())
    vel_mag = float(torch.norm(state.object_vel[0]).item())
    force_mag = float(torch.norm(controls.force_vector[0]).item())
    force_strength = float(controls.force_strength[0, 0].item())
    force_active = float(controls.force_active[0, 0].item())

    # Biological-style proxies: chemotaxis, phototaxis, mechanosensation.
    bio_chemical = resource_mean - stress_mean
    bio_phototaxis = light_intensity * max(0.0, 1.0 - light_dist / 2.5)
    bio_mechano = vel_mag + force_mag * force_strength * force_active

    # Technological-style proxies: RF/interference, telemetry heartbeat, bus/remap load.
    tech_rf_noise = wind_mag + 0.5 * force_strength * force_active
    phase = (step + (seed % 17)) / 5.0
    tech_telemetry = 0.5 + 0.5 * math.sin(phase)
    tech_bus_load = 0.35 * stress_mean + (0.65 if remap_happened else 0.0)

    # Anonymous target signal: mixed latent of biological + technological factors.
    target_signal = math.tanh(
        0.24 * bio_chemical
        + 0.20 * bio_phototaxis
        + 0.18 * bio_mechano
        + 0.14 * tech_rf_noise
        + 0.12 * tech_telemetry
        + 0.12 * tech_bus_load
    )
    threat_active = (
        force_active > 0.5
        or tech_rf_noise > 0.95
        or stress_mean > 0.60
        or (scenario.name in {"storm", "crosswind"} and wind_mag > 0.75)
    )
    return CapabilityProxySignals(
        bio_chemical=bio_chemical,
        bio_phototaxis=bio_phototaxis,
        bio_mechano=bio_mechano,
        tech_rf_noise=tech_rf_noise,
        tech_telemetry=tech_telemetry,
        tech_bus_load=tech_bus_load,
        target_signal=target_signal,
        threat_active=threat_active,
    )


def _load_model(weights: str, profile: str, device: torch.device) -> tuple[ModelCore, ModelConfig, dict]:
    ckpt = torch.load(weights, map_location=device)
    if "model_config" in ckpt:
        cfg = ModelConfig(**ckpt["model_config"])
    else:
        cfg = model_config_for_profile(profile)
    model = ModelCore(**cfg.__dict__).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, cfg, ckpt


def compute_recovery_score(mismatch: list[float], remap_steps: list[int], window: int = 5) -> float:
    if not mismatch or not remap_steps:
        return 0.0
    scores = []
    n = len(mismatch)
    for step in remap_steps:
        post_start = step
        post_end = min(step + window, n)
        late_start = min(step + window, n)
        late_end = min(step + 2 * window, n)
        if post_start >= post_end or late_start >= late_end:
            continue
        post = sum(mismatch[post_start:post_end]) / max(1, post_end - post_start)
        late = sum(mismatch[late_start:late_end]) / max(1, late_end - late_start)
        if post <= 1e-8:
            continue
        score = max(0.0, min(1.0, (post - late) / post))
        scores.append(score)
    if not scores:
        return 0.0
    return sum(scores) / len(scores)


def _force_scale(spec: ScenarioSpec, step: int) -> tuple[float, bool]:
    if spec.force_pattern == "none":
        return 0.0, False
    rel = step - spec.force_start
    if rel < 0:
        return 0.0, False
    if spec.force_pattern == "pulse":
        return (spec.force_strength, True) if rel < max(1, spec.force_duration) else (0.0, False)
    if spec.force_pattern == "decay":
        if rel < max(1, spec.force_duration):
            return spec.force_strength * (0.93 ** rel), True
        return 0.0, False
    if spec.force_pattern == "sine":
        return spec.force_strength * (0.5 + 0.5 * math.sin(rel / 4.0)), True
    return 0.0, False


def rollout_metrics(
    model: ModelCore,
    cfg: ModelConfig,
    embodiment_name: str,
    scenario: ScenarioSpec,
    steps: int,
    remap_every: int,
    seed: int,
    world_dims: tuple[int, int, int, int],
    device: torch.device,
    capability_profile: str = "none",
) -> dict[str, float | int]:
    wx, wy, wz, resource_channels = world_dims
    torch.manual_seed(seed)

    world = DynamicDiversityWorld(wx, wy, wz, resource_channels, decay=0.03, device=str(device))
    state = world.init(batch_size=1)
    memory = model.init_memory(1, cfg.memory_slots, cfg.memory_dim, device)

    emb = get_embodiment(embodiment_name)
    control_dim = len(emb.controls)
    projection = torch.randn(cfg.signal_dim, control_dim, device=device) * 0.3

    mapping = device_map_for_embodiment(cfg.io_channels, emb, device=device, permutation_seed=seed)

    mismatch_values: list[float] = []
    vitality_values: list[float] = []
    stress_values: list[float] = []
    remap_steps: list[int] = []
    signal_target_values: list[float] = []
    signal_emit_values: list[float] = []
    detection_score_values: list[float] = []
    threat_labels: list[int] = []
    threat_steps = 0
    evasion_success_steps = 0

    for step in range(steps):
        remap_code = torch.zeros(1, cfg.max_remap_groups, device=device)
        remap_happened = False
        if step > 0 and step % remap_every == 0:
            remap_steps.append(step)
            mapping = device_map_for_embodiment(cfg.io_channels, emb, device=device, permutation_seed=seed + step * 17)
            remap_code[:, step % cfg.max_remap_groups] = 1.0
            remap_happened = True

        controls = world.default_controls(1)
        phase = step / 7.0
        controls.wind = torch.tensor(
            [[
                scenario.wind[0] + scenario.wind_variation * math.sin(phase),
                scenario.wind[1] + scenario.wind_variation * math.cos(phase / 2.0),
                scenario.wind[2] + scenario.wind_variation * math.sin(phase / 3.0),
            ]],
            device=device,
        )
        controls.light_position = torch.tensor(
            [[
                scenario.light_pos[0] + scenario.light_drift[0] * step,
                scenario.light_pos[1] + scenario.light_drift[1] * step,
                scenario.light_pos[2] + scenario.light_drift[2] * step,
            ]],
            device=device,
        ).clamp(-1.0, 1.0)
        controls.light_intensity = torch.tensor([[scenario.light_intensity]], device=device)

        force_scale, force_active = _force_scale(scenario, step)
        controls.force_vector = torch.tensor([[scenario.force_vector[0], scenario.force_vector[1], scenario.force_vector[2]]], device=device)
        controls.force_strength = torch.tensor([[force_scale]], device=device)
        controls.force_active = torch.tensor([[1.0 if force_active else 0.0]], device=device)
        controls.force_position = torch.zeros(1, 3, device=device)

        with torch.no_grad():
            obs = world.encode_observation(state, signal_dim=cfg.signal_dim)
            out = model(obs, memory, remap_code)
            memory = out["memory"]

            desired = torch.tanh(obs @ projection)
            applied = out["io"] @ mapping
            mismatch = float(torch.mean((applied - desired) ** 2).item())
            mismatch_values.append(mismatch)
            signal_emit = float(torch.tanh(out["io"].mean()).item())
            detection_score = float(
                torch.sigmoid(1.2 * torch.mean(torch.abs(out["readiness"])) + 0.8 * torch.mean(out["energy"])).item()
            )

            action = applied.mean(dim=1, keepdim=True).repeat(1, wx * wy * wz)
            state = world.step(state, action, controls=controls)
            vitality_values.append(float(state.life.mean().item()))
            stress_values.append(float(state.stress.mean().item()))

            if capability_profile != "none":
                proxy = _capability_proxy_signals(
                    state=state,
                    controls=controls,
                    scenario=scenario,
                    step=step,
                    remap_happened=remap_happened,
                    seed=seed,
                )
                threat = 1 if proxy.threat_active else 0
                threat_labels.append(threat)
                signal_target_values.append(proxy.target_signal)
                signal_emit_values.append(signal_emit)
                detection_score_values.append(detection_score)
                if threat:
                    threat_steps += 1
                    step_stress = stress_values[-1]
                    step_vitality = vitality_values[-1]
                    # Evasion proxy: remain viable and controlled while threat is active.
                    if mismatch < 0.90 and step_stress < 0.78 and step_vitality > 0.012:
                        evasion_success_steps += 1

    mean_mismatch = sum(mismatch_values) / max(1, len(mismatch_values))
    mean_vitality = sum(vitality_values) / max(1, len(vitality_values))
    recovery = compute_recovery_score(mismatch_values, remap_steps)

    transfer_score = (
        0.55 * (1.0 / (1.0 + mean_mismatch))
        + 0.30 * mean_vitality
        + 0.15 * recovery
    )

    metrics = {
        "mean_mismatch": mean_mismatch,
        "final_mismatch": mismatch_values[-1],
        "mean_vitality": mean_vitality,
        "final_vitality": vitality_values[-1],
        "recovery": recovery,
        "transfer_score": transfer_score,
        "remap_events": len(remap_steps),
    }
    if capability_profile != "none":
        signal_corr_raw = _safe_corr(signal_emit_values, signal_target_values)
        signal_reliability = abs(signal_corr_raw)
        signal_detection_auc_raw = _binary_auc(detection_score_values, threat_labels)
        signal_detection_auc = max(signal_detection_auc_raw, 1.0 - signal_detection_auc_raw)
        evasion_success = float(evasion_success_steps / max(1, threat_steps))
        capability_score = 0.40 * signal_reliability + 0.30 * signal_detection_auc + 0.30 * evasion_success
        metrics.update(
            {
                "signal_corr_raw": signal_corr_raw,
                "signal_reliability": signal_reliability,
                "signal_detection_auc_raw": signal_detection_auc_raw,
                "signal_detection_auc": signal_detection_auc,
                "evasion_success": evasion_success,
                "capability_score": capability_score,
                "threat_steps": threat_steps,
            }
        )
    return metrics


@app.command()
def run(
    checkpoints_glob: str = "artifacts/parallel-long/variant-*.pt",
    checkpoints_dir: str = "",
    checkpoints_list: str = "",
    profile: str = "pi5",
    embodiments: str = "hexapod,car,drone",
    embodiment_weights: str = "",
    scenarios: str = "mild,gust,force",
    scenario_profile: str = "",
    runs_per_combo: int = 2,
    steps: int = 90,
    remap_every: int = 15,
    capability_profile: str = "none",
    capability_score_weight: float = 0.0,
    world_x: int = 20,
    world_y: int = 20,
    world_z: int = 10,
    resource_channels: int = 5,
    device: str = "cpu",
    seed: int = 31,
    output: str = "artifacts/cross-eval-summary.json",
) -> None:
    if checkpoints_list:
        raw = checkpoints_list.replace(";", ",")
        ckpts = [x.strip() for x in raw.split(",") if x.strip()]
    elif checkpoints_dir:
        ckpts = sorted(str(p) for p in Path(checkpoints_dir).glob("variant-*.pt"))
    else:
        ckpts = sorted(glob.glob(checkpoints_glob))
    if not ckpts:
        if checkpoints_dir:
            raise typer.BadParameter(f"No checkpoints matched in directory: {checkpoints_dir}")
        raise typer.BadParameter(f"No checkpoints matched: {checkpoints_glob}")

    if scenario_profile:
        scenario_specs = _resolve_scenario_profile(scenario_profile)
    else:
        scenario_specs = _resolve_scenarios(scenarios)
    embodiment_list = [x.strip().lower() for x in embodiments.split(",") if x.strip()]
    if not embodiment_list:
        raise typer.BadParameter("No embodiments were provided")
    try:
        embodiment_weight_map = _parse_embodiment_weights(embodiment_weights, embodiment_list)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    try:
        capability_profile_resolved = _resolve_capability_profile(capability_profile)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    if capability_score_weight < 0.0:
        raise typer.BadParameter("capability_score_weight must be >= 0.0")

    dev = torch.device(device)
    world_dims = (world_x, world_y, world_z, resource_channels)

    ranked: list[dict] = []
    for ckpt_path in ckpts:
        model, cfg, raw_ckpt = _load_model(ckpt_path, profile, dev)

        run_rows: list[dict] = []
        for emb_idx, emb_name in enumerate(embodiment_list):
            get_embodiment(emb_name)
            for scen_idx, scen in enumerate(scenario_specs):
                for rep in range(runs_per_combo):
                    combo_seed = seed + emb_idx * 1000 + scen_idx * 100 + rep
                    metrics = rollout_metrics(
                        model=model,
                        cfg=cfg,
                        embodiment_name=emb_name,
                        scenario=scen,
                        steps=steps,
                        remap_every=remap_every,
                        seed=combo_seed,
                        world_dims=world_dims,
                        device=dev,
                        capability_profile=capability_profile_resolved,
                    )
                    run_rows.append({
                        "embodiment": emb_name,
                        "scenario": scen.name,
                        "rep": rep,
                        **metrics,
                    })

        def _avg(key: str) -> float:
            return float(sum(float(row[key]) for row in run_rows) / max(1, len(run_rows)))

        by_embodiment = {}
        for emb_name in embodiment_list:
            subset = [row for row in run_rows if row["embodiment"] == emb_name]
            by_embodiment[emb_name] = {
                "transfer_score": float(sum(float(x["transfer_score"]) for x in subset) / max(1, len(subset))),
                "mean_mismatch": float(sum(float(x["mean_mismatch"]) for x in subset) / max(1, len(subset))),
                "mean_vitality": float(sum(float(x["mean_vitality"]) for x in subset) / max(1, len(subset))),
                "recovery": float(sum(float(x["recovery"]) for x in subset) / max(1, len(subset))),
            }
            if capability_profile_resolved != "none":
                by_embodiment[emb_name].update(
                    {
                        "signal_reliability": float(sum(float(x["signal_reliability"]) for x in subset) / max(1, len(subset))),
                        "signal_detection_auc": float(sum(float(x["signal_detection_auc"]) for x in subset) / max(1, len(subset))),
                        "evasion_success": float(sum(float(x["evasion_success"]) for x in subset) / max(1, len(subset))),
                        "capability_score": float(sum(float(x["capability_score"]) for x in subset) / max(1, len(subset))),
                    }
                )

        base_transfer_weighted = _weighted_transfer_score(
            by_embodiment=by_embodiment,
            embodiments=embodiment_list,
            embodiment_weights=embodiment_weight_map,
        )
        overall_capability_score = 0.0
        if capability_profile_resolved != "none":
            overall_capability_score = float(
                sum(float(row["capability_score"]) for row in run_rows) / max(1, len(run_rows))
            )
        ranking_score = base_transfer_weighted + capability_score_weight * overall_capability_score

        ranked.append(
            {
                "checkpoint": ckpt_path,
                "flags": raw_ckpt.get("run_flags", {}),
                "overall_transfer_score": ranking_score,
                "overall_transfer_score_unweighted": _avg("transfer_score"),
                "overall_transfer_score_weighted": base_transfer_weighted,
                "overall_mean_mismatch": _avg("mean_mismatch"),
                "overall_mean_vitality": _avg("mean_vitality"),
                "overall_recovery": _avg("recovery"),
                "overall_capability_score": overall_capability_score,
                "by_embodiment": by_embodiment,
                "runs": run_rows,
            }
        )

    ranked.sort(key=lambda x: x["overall_transfer_score"], reverse=True)
    payload = {
        "config": {
            "checkpoints_glob": checkpoints_glob,
            "checkpoints_dir": checkpoints_dir,
            "checkpoints_list": checkpoints_list,
            "profile": profile,
            "embodiments": embodiment_list,
            "embodiment_weights": embodiment_weight_map,
            "scenarios": [s.name for s in scenario_specs],
            "scenario_profile": scenario_profile,
            "runs_per_combo": runs_per_combo,
            "steps": steps,
            "remap_every": remap_every,
            "capability_profile": capability_profile_resolved,
            "capability_score_weight": capability_score_weight,
            "world": {"x": world_x, "y": world_y, "z": world_z, "resource_channels": resource_channels},
        },
        "ranked": ranked,
    }

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print({
        "output": output,
        "checkpoints": len(ranked),
        "best_checkpoint": ranked[0]["checkpoint"],
        "best_score": ranked[0]["overall_transfer_score"],
    })


if __name__ == "__main__":
    app()
