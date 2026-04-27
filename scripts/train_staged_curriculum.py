"""
Staged curriculum training for the new 40x40x20 toroidal environment.

Solves the problem of all environment complexity being active from gen 1
(which caused fitness to peak at generation 7 then decline in eval_evolve_new_env.py).

Three sequential phases, each seeding from the best checkpoint of the prior:

  Phase 1 (30 gens, pop=6): occlusion-only world (no physics objects, no hazard zones)
    -> artifacts/staged/phase1-best.pt

  Phase 2 (30 gens, pop=6): occlusion + physics objects
    -> artifacts/staged/phase2-best.pt

  Phase 3 (40 gens, pop=6): full new_env_v1 (occlusion + physics + all 3 hazard zones)
    -> artifacts/staged/phase3-best.pt

Diversity bonus raised to 0.15 (from 0.05 in eval_evolve_new_env.py) to
push species_count above 1 throughout evolution. Closes #6, Closes #7.
"""
from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

import torch
import typer
from typer.testing import CliRunner

# Allow running from project root: python scripts/train_staged_curriculum.py
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ai_embedded_dynamic_diversity.config import (
    ModelConfig,
    WorldConfig,
    world_config_for_profile,
)
from ai_embedded_dynamic_diversity.models import ModelCore
from ai_embedded_dynamic_diversity.sim.signaling import SignalingWorld
from ai_embedded_dynamic_diversity.train.cli import (
    _build_transfer_states,
    evaluate_fitness,
    run,
)
from ai_embedded_dynamic_diversity.train.device import choose_device

# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------

DEVICE = __import__("os").environ.get("TRAIN_DEVICE", "cuda")
EVAL_STEPS = 16
EVAL_BATCH = 8
EVAL_ENV_VOLATILITY = 0.45

OUT_DIR = Path("artifacts/staged")

# Diversity bonus — raised from 0.05 to push species_count > 1 (issue #7)
DIVERSITY_BONUS = 0.15

# Seed checkpoints to start Phase 1 from (same candidates as eval_evolve_new_env.py)
SEED_CHECKPOINTS = [
    "artifacts/model-core-champion-v08.pt",
    "artifacts/model-core-coevo-guardrail-transfercal-v06.pt",
    "artifacts/model-core-coevo-guardrail-shared-extreme-v04.pt",
]


# ---------------------------------------------------------------------------
# World config helpers
# ---------------------------------------------------------------------------

def _phase1_world_config() -> WorldConfig:
    """Phase 1: new_env_v1 base (40x40x20 + occlusion) with physics & hazards removed."""
    base = world_config_for_profile("new_env_v1")
    return WorldConfig(
        x=base.x,
        y=base.y,
        z=base.z,
        resource_channels=base.resource_channels,
        decay=base.decay,
        actuation_delay_steps=base.actuation_delay_steps,
        actuation_noise_std=base.actuation_noise_std,
        sensor_latency_steps=base.sensor_latency_steps,
        sensor_dropout_burst_prob=base.sensor_dropout_burst_prob,
        surface_friction_scale=base.surface_friction_scale,
        disturbance_correlation_horizon=base.disturbance_correlation_horizon,
        num_occlusion_objects=base.num_occlusion_objects,
        occlusion_seed=base.occlusion_seed,
        # Removed complexity:
        num_physics_objects=0,
        phys_mass=base.phys_mass,
        phys_friction=base.phys_friction,
        hazard_zones=[],
    )


def _phase2_world_config() -> WorldConfig:
    """Phase 2: phase 1 + physics objects (still no hazard zones)."""
    base = world_config_for_profile("new_env_v1")
    return WorldConfig(
        x=base.x,
        y=base.y,
        z=base.z,
        resource_channels=base.resource_channels,
        decay=base.decay,
        actuation_delay_steps=base.actuation_delay_steps,
        actuation_noise_std=base.actuation_noise_std,
        sensor_latency_steps=base.sensor_latency_steps,
        sensor_dropout_burst_prob=base.sensor_dropout_burst_prob,
        surface_friction_scale=base.surface_friction_scale,
        disturbance_correlation_horizon=base.disturbance_correlation_horizon,
        num_occlusion_objects=base.num_occlusion_objects,
        occlusion_seed=base.occlusion_seed,
        # Added:
        num_physics_objects=base.num_physics_objects,
        phys_mass=base.phys_mass,
        phys_friction=base.phys_friction,
        # Still removed:
        hazard_zones=[],
    )


def _phase3_world_config() -> WorldConfig:
    """Phase 3: full new_env_v1 — occlusion + physics + all 3 hazard zones."""
    return world_config_for_profile("new_env_v1")


# ---------------------------------------------------------------------------
# World factory (mirrors eval_evolve_new_env._make_world)
# ---------------------------------------------------------------------------

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
# Shared training-args builder
# ---------------------------------------------------------------------------

def _strong_training_args(
    epochs: int,
    save_path: str,
    metrics_path: str,
    seed_csv: str,
) -> list[str]:
    """Return base CLI args for one training phase.

    The world-profile sentinel (e.g. ``_staged_phase1``) is appended by
    ``_run_phase()``, which also monkey-patches ``world_config_for_profile``
    in the CLI module so the sentinel resolves to the exact per-phase
    ``WorldConfig``.  This avoids leaking internal WorldConfig fields through
    a string-based CLI boundary.
    """
    return [
        # Epochs and batch
        f"--epochs={epochs}",
        "--batch-size=12",
        "--unroll-steps=16",
        "--lr=1.5e-4",
        f"--device={DEVICE}",
        # Architecture — pi5 profile matching existing champions
        "--profile=pi5",
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
        # GDI diversity bonus (raised to 0.15 — issue #7)
        f"--diversity-selection-bonus={DIVERSITY_BONUS}",
        # Architecture extras matching champions
        "--enable-multi-scale-gating",
        # AMP
        "--use-amp",
        "--allow-tf32",
        # Seeds and output
        f"--init-weights-cycle={seed_csv}",
        "--seed=314159",
        f"--save-path={save_path}",
        f"--metrics-path={metrics_path}",
    ]


# ---------------------------------------------------------------------------
# Phase runner
# ---------------------------------------------------------------------------

def _run_phase(
    phase_num: int,
    epochs: int,
    wcfg: WorldConfig,
    seed_csv: str,
    save_path: Path,
    metrics_path: Path,
    log_path: Path,
) -> None:
    """
    Run one training phase via typer CliRunner, patching world_config_for_profile
    so that the sentinel profile '_staged_phaseN' resolves to our WorldConfig.
    """
    import ai_embedded_dynamic_diversity.train.cli as _cli_module
    from ai_embedded_dynamic_diversity.config import world_config_for_profile as _orig_wcfp

    sentinel = f"_staged_phase{phase_num}"

    def _patched_wcfp(profile: str) -> WorldConfig:
        if profile == sentinel:
            return wcfg
        return _orig_wcfp(profile)

    # Patch for the duration of this phase
    _cli_module.world_config_for_profile = _patched_wcfp

    mini_app = typer.Typer()
    mini_app.command()(run)
    runner = CliRunner()

    args = _strong_training_args(
        epochs=epochs,
        save_path=str(save_path),
        metrics_path=str(metrics_path),
        seed_csv=seed_csv,
    ) + [f"--world-profile={sentinel}"]

    result = runner.invoke(mini_app, args)

    # Restore original
    _cli_module.world_config_for_profile = _orig_wcfp

    log_path.write_text(result.output, encoding="utf-8")
    print(f"  Training output written to {log_path}")

    if result.exception:
        import traceback
        traceback.print_exception(type(result.exception), result.exception, result.exception.__traceback__)
        raise result.exception

    print(f"  Phase {phase_num} complete. exit_code={result.exit_code}")


# ---------------------------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------------------------

def _eval_checkpoint(ckpt_path: Path, wcfg: WorldConfig, dev: torch.device) -> float:
    """Evaluate a checkpoint on a given world config and return fitness."""
    ckpt = torch.load(str(ckpt_path), map_location=dev, weights_only=False)
    mcfg_dict = ckpt.get("model_config", {})
    mcfg = ModelConfig(**mcfg_dict)
    model = ModelCore(**asdict(mcfg)).to(dev)
    model.load_state_dict(ckpt["model"])

    world = _make_world(wcfg, dev)
    transfer_states = _build_transfer_states(
        ["hexapod", "car", "drone", "polymorph120"],
        mcfg, dev, seed=9001
    )
    return evaluate_fitness(
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dev = choose_device(DEVICE, strict=True)
    print(f"Device: {dev}  ({torch.cuda.get_device_name(0) if dev.type == 'cuda' else 'cpu'})")

    # Filter seed checkpoints to those that exist
    available_seeds = [p for p in SEED_CHECKPOINTS if Path(p).exists()]
    if not available_seeds:
        # Graceful fallback: start from scratch if no seeds are found
        print("  WARNING: no seed checkpoints found — starting phase 1 from random init")
        available_seeds = []

    # -----------------------------------------------------------------------
    # Phase 1: occlusion-only world
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("PHASE 1 / 3 — Occlusion-only world (no physics, no hazard zones)")
    print(f"  epochs=30  pop=6  diversity_bonus={DIVERSITY_BONUS}")
    print("=" * 70)

    phase1_save = OUT_DIR / "phase1-best.pt"
    phase1_metrics = OUT_DIR / "phase1-best.metrics.json"
    phase1_log = OUT_DIR / "phase1-train.log"
    phase1_wcfg = _phase1_world_config()

    print(f"  World: {phase1_wcfg.x}x{phase1_wcfg.y}x{phase1_wcfg.z}"
          f"  occlusion={phase1_wcfg.num_occlusion_objects}"
          f"  physics={phase1_wcfg.num_physics_objects}"
          f"  hazards={len(phase1_wcfg.hazard_zones)}")

    seed_csv_p1 = ",".join(available_seeds) if available_seeds else ""
    # Empty seed_csv is fine — the CLI starts from random init when no cycle given.

    _run_phase(
        phase_num=1,
        epochs=30,
        wcfg=phase1_wcfg,
        seed_csv=seed_csv_p1,
        save_path=phase1_save,
        metrics_path=phase1_metrics,
        log_path=phase1_log,
    )

    # -----------------------------------------------------------------------
    # Phase 2: occlusion + physics (seed from phase 1 best)
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("PHASE 2 / 3 — Occlusion + physics objects (no hazard zones)")
    print(f"  epochs=30  pop=6  diversity_bonus={DIVERSITY_BONUS}")
    print(f"  Seeding from: {phase1_save}")
    print("=" * 70)

    phase2_save = OUT_DIR / "phase2-best.pt"
    phase2_metrics = OUT_DIR / "phase2-best.metrics.json"
    phase2_log = OUT_DIR / "phase2-train.log"
    phase2_wcfg = _phase2_world_config()

    print(f"  World: {phase2_wcfg.x}x{phase2_wcfg.y}x{phase2_wcfg.z}"
          f"  occlusion={phase2_wcfg.num_occlusion_objects}"
          f"  physics={phase2_wcfg.num_physics_objects}"
          f"  hazards={len(phase2_wcfg.hazard_zones)}")

    _run_phase(
        phase_num=2,
        epochs=30,
        wcfg=phase2_wcfg,
        seed_csv=str(phase1_save),
        save_path=phase2_save,
        metrics_path=phase2_metrics,
        log_path=phase2_log,
    )

    # -----------------------------------------------------------------------
    # Phase 3: full new_env_v1 (seed from phase 2 best)
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("PHASE 3 / 3 — Full new_env_v1 (occlusion + physics + 3 hazard zones)")
    print(f"  epochs=40  pop=6  diversity_bonus={DIVERSITY_BONUS}")
    print(f"  Seeding from: {phase2_save}")
    print("=" * 70)

    phase3_save = OUT_DIR / "phase3-best.pt"
    phase3_metrics = OUT_DIR / "phase3-best.metrics.json"
    phase3_log = OUT_DIR / "phase3-train.log"
    phase3_wcfg = _phase3_world_config()

    print(f"  World: {phase3_wcfg.x}x{phase3_wcfg.y}x{phase3_wcfg.z}"
          f"  occlusion={phase3_wcfg.num_occlusion_objects}"
          f"  physics={phase3_wcfg.num_physics_objects}"
          f"  hazards={len(phase3_wcfg.hazard_zones)}")

    _run_phase(
        phase_num=3,
        epochs=40,
        wcfg=phase3_wcfg,
        seed_csv=str(phase2_save),
        save_path=phase3_save,
        metrics_path=phase3_metrics,
        log_path=phase3_log,
    )

    # -----------------------------------------------------------------------
    # Cross-phase evaluation: compare phase 1 seed vs phase 3 best on full env
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("CROSS-PHASE EVALUATION — phase1-best vs phase3-best on full new_env_v1")
    print("=" * 70)

    eval_results = {}
    full_wcfg = _phase3_world_config()

    for label, ckpt_path in [("phase1_best", phase1_save), ("phase3_best", phase3_save)]:
        if not ckpt_path.exists():
            print(f"  SKIP (missing): {ckpt_path}")
            continue
        fitness = _eval_checkpoint(ckpt_path, full_wcfg, dev)
        eval_results[label] = fitness  # store raw float; round only for display/JSON
        print(f"  {label:15s}: {fitness:+.5f}")

    eval_json_path = OUT_DIR / "cross-phase-eval.json"
    eval_json_path.write_text(
        json.dumps({k: round(v, 5) for k, v in eval_results.items()}, indent=2),
        encoding="utf-8",
    )
    print(f"\n  Cross-phase eval saved to {eval_json_path}")

    if "phase1_best" in eval_results and "phase3_best" in eval_results:
        delta = eval_results["phase3_best"] - eval_results["phase1_best"]
        print(f"\n  Fitness improvement (phase3 - phase1): {delta:+.5f}")
        if delta > 0:
            print("  => Staged curriculum improved fitness over direct phase-1 seed.")
        else:
            print("  => WARNING: phase3 did not outperform phase1 seed on full env.")

    print("\nDone. Artifacts saved to:", OUT_DIR)
