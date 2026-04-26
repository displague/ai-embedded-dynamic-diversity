"""
Evaluate and evolve top champions under the new 40x40x20 toroidal environment.

Phase 1: Benchmark top candidates on three worlds:
  - legacy: 20x20x10 zero-padded (original training world)
  - base_new: 40x40x20 toroidal (bare)
  - new_env_v1: 40x40x20 toroidal + occlusion + physics + hazards

Phase 2: Evolve the top seeds in the new_env_v1 world using the strongest
         training config (curriculum, transfer, autopoietic, guardrail, noise,
         force curriculum, world predictor, GDI logging).
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

import torch

# Allow running from project root: python scripts/eval_evolve_new_env.py
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ai_embedded_dynamic_diversity.config import (
    ModelConfig,
    WorldConfig,
    world_config_for_profile,
)
from ai_embedded_dynamic_diversity.models import ModelCore
from ai_embedded_dynamic_diversity.sim.signaling import SignalingWorld
from ai_embedded_dynamic_diversity.train.cli import evaluate_fitness, _build_transfer_states, run
from ai_embedded_dynamic_diversity.train.device import choose_device

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEVICE = "cuda"
EVAL_STEPS = 16
EVAL_BATCH = 8
EVAL_ENV_VOLATILITY = 0.45

CANDIDATES = [
    "artifacts/model-core-champion-v08.pt",
    "artifacts/model-core-coevo-guardrail-transfercal-v06.pt",
    "artifacts/model-core-coevo-guardrail-shared-extreme-v04.pt",
    "artifacts/model-core-coevo-guardrail-shared-extreme-v05.pt",
    "artifacts/model-core-champion-v09.pt",
]

OUT_DIR = Path("artifacts/new-env-evolution")
EVAL_JSON = OUT_DIR / "phase1-eval.json"


def _make_world(wcfg: WorldConfig, dev: torch.device) -> SignalingWorld:
    return SignalingWorld(
        wcfg.x, wcfg.y, wcfg.z, wcfg.resource_channels, wcfg.decay,
        device=str(dev),
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
# Phase 1: Evaluation
# ---------------------------------------------------------------------------

def phase1_eval(dev: torch.device) -> list[dict]:
    worlds = {
        "legacy_20x20x10":  _make_world(WorldConfig(x=20, y=20, z=10), dev),
        "base_40x40x20":    _make_world(WorldConfig(), dev),
        "new_env_v1":       _make_world(world_config_for_profile("new_env_v1"), dev),
    }
    wcfgs = {
        "legacy_20x20x10":  WorldConfig(x=20, y=20, z=10),
        "base_40x40x20":    WorldConfig(),
        "new_env_v1":       world_config_for_profile("new_env_v1"),
    }

    results = []
    for ckpt_path in CANDIDATES:
        if not os.path.exists(ckpt_path):
            print(f"  SKIP (missing): {ckpt_path}")
            continue
        ckpt = torch.load(ckpt_path, map_location=dev, weights_only=False)
        mcfg_dict = ckpt.get("model_config", {})
        mcfg = ModelConfig(**mcfg_dict)
        model = ModelCore(**asdict(mcfg)).to(dev)
        model.load_state_dict(ckpt["model"])

        row = {"checkpoint": ckpt_path, "profile": ckpt.get("profile"), "model_config": mcfg_dict}
        print(f"\n  {Path(ckpt_path).name}  (profile={row['profile']})")

        transfer_states = _build_transfer_states(
            ["hexapod", "car", "drone", "polymorph120"],
            mcfg, dev, seed=9001
        )

        for world_name, world in worlds.items():
            wcfg = wcfgs[world_name]
            fitness = evaluate_fitness(
                model, world, mcfg, wcfg, dev,
                steps=EVAL_STEPS,
                batch=EVAL_BATCH,
                env_volatility=EVAL_ENV_VOLATILITY,
                transfer_states=transfer_states,
                transfer_fitness_weight=0.10,
                transfer_samples_per_step=2,
                noise_profile="dropout-quant-v2",
                noise_strength=0.8,
                noise_seed=12345,
                autopoietic_fitness_weight=0.15,
                genetic_memory_persistence_weight=0.05,
                force_curriculum_mode="continuous-blow",
                force_curriculum_strength=0.6,
            )
            row[world_name] = round(fitness, 5)
            print(f"    {world_name:22s}: {fitness:+.5f}")

        results.append(row)

    return results


# ---------------------------------------------------------------------------
# Phase 2: Evolution
# ---------------------------------------------------------------------------

def phase2_evolve(seeds: list[str], dev: torch.device) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    save_path = str(OUT_DIR / "champion-new-env-v1.pt")
    metrics_path = str(OUT_DIR / "champion-new-env-v1.metrics.json")

    print("\n=== Phase 2: Evolution in new_env_v1 world ===")
    print(f"  Seeds: {seeds}")
    print(f"  Save:  {save_path}")

    # Build the init_weights_cycle CSV from seed paths
    cycle_csv = ",".join(seeds)

    from typer.testing import CliRunner
    import typer
    mini_app = typer.Typer()
    mini_app.command()(run)
    runner = CliRunner()

    args = [
        # Epochs and batch
        "--epochs=40",
        "--batch-size=12",
        "--unroll-steps=16",
        "--lr=1.5e-4",
        "--device=cuda",
        # Architecture — keep pi5 profile that champions use
        "--profile=pi5",
        "--world-profile=new_env_v1",
        # Coevolution
        "--coevolution",
        "--population-size=6",
        "--elite-fraction=0.4",
        "--mutation-std=0.008",
        # Curriculum
        "--enable-curriculum",
        "--curriculum-power=1.2",
        "--remap-probability-start=0.12",
        "--remap-probability-end=0.45",
        "--env-volatility-start=0.08",
        "--env-volatility-end=0.65",
        # Transfer
        "--enable-embodiment-transfer-loss",
        "--embodiments=hexapod,car,drone,polymorph120",
        "--transfer-loss-weight=0.40",
        "--transfer-fitness-weight=0.12",
        "--transfer-samples-per-step=3",
        # Autopoietic
        "--enable-autopoietic-objective",
        "--autopoietic-loss-weight=0.14",
        "--autopoietic-fitness-gain=0.16",
        # Capability guardrail
        "--enable-capability-guardrail",
        "--signal-reliability-floor=0.50",
        "--conjoining-gain-floor=0.25",
        "--capability-guardrail-penalty-weight=0.20",
        # Noise curriculum
        "--noise-profile=dropout-quant-v2",
        "--enable-noise-curriculum",
        "--noise-strength-start=0.2",
        "--noise-strength-end=0.9",
        # Force curriculum
        "--force-curriculum-mode=continuous-blow",
        "--force-curriculum-strength-start=0.1",
        "--force-curriculum-strength-end=0.8",
        # World predictor (JEPA)
        "--enable-world-predictor",
        "--world-pred-loss-weight=0.04",
        # GDI diversity bonus
        "--diversity-selection-bonus=0.05",
        # Architecture extras matching champions
        "--enable-multi-scale-gating",
        # AMP
        "--use-amp",
        "--allow-tf32",
        # Seeds and output
        "--init-weights-cycle=" + cycle_csv,
        "--seed=314159",
        "--save-path=" + save_path,
        "--metrics-path=" + metrics_path,
    ]

    result = runner.invoke(mini_app, args)
    # Write output to log file
    log_path = OUT_DIR / "train.log"
    log_path.write_text(result.output, encoding="utf-8")
    print(f"  Training output written to {log_path}")

    if result.exception:
        import traceback
        traceback.print_exception(type(result.exception), result.exception, result.exception.__traceback__)
        raise result.exception

    print(f"  Evolution complete. exit_code={result.exit_code}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dev = choose_device(DEVICE, strict=True)
    print(f"Device: {dev}  ({torch.cuda.get_device_name(0) if dev.type == 'cuda' else 'cpu'})")

    # Phase 1
    print("\n=== Phase 1: Evaluating top candidates on three worlds ===")
    eval_results = phase1_eval(dev)
    EVAL_JSON.write_text(json.dumps(eval_results, indent=2), encoding="utf-8")
    print(f"\n  Results saved to {EVAL_JSON}")

    # Rank by new_env_v1 fitness and pick top 3 as seeds
    ranked = sorted(
        [r for r in eval_results if "new_env_v1" in r],
        key=lambda r: r["new_env_v1"],
        reverse=True,
    )
    print("\n  Ranking by new_env_v1 fitness:")
    for i, r in enumerate(ranked):
        name = Path(r["checkpoint"]).name
        print(f"    #{i+1}  {name:52s}  new_env={r['new_env_v1']:+.5f}  legacy={r.get('legacy_20x20x10',0):+.5f}")

    seeds = [r["checkpoint"] for r in ranked[:3]]

    # Phase 2
    phase2_evolve(seeds, dev)
