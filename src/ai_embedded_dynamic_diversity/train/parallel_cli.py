from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path

import torch
import typer
from rich import print

from ai_embedded_dynamic_diversity.config import model_config_for_profile
from ai_embedded_dynamic_diversity.train.device import choose_device, device_runtime_snapshot

app = typer.Typer(add_completion=False)


def _run_variant(cmd: list[str], log_path: Path):
    with open(log_path, "w") as f:
        # We must inherit the current environment to ensure CUDA/venv visibility
        env = os.environ.copy()
        try:
            return subprocess.run(cmd, check=True, stdout=f, stderr=subprocess.STDOUT, env=env)
        except subprocess.CalledProcessError as e:
            print(f"[bold red]Variant failed: {' '.join(cmd)}[/bold red]")
            print(f"[red]Exit code: {e.returncode}[/red]")
            return e


@app.command()
def run(
    variants: int = 4,
    epochs: int = 20,
    batch_size: int = 16,
    unroll_steps: int = 12,
    lr: float = 2e-4,
    profile: str = "base",
    world_profile: str = "",
    device: str = "cuda",
    strict_device: bool = True,
    coevolution: bool = True,
    population_size: int = 4,
    embodiments: str = "hexapod,car,drone,polymorph120",
    enable_embodiment_transfer_loss: bool = True,
    transfer_loss_weight: float = 0.35,
    transfer_fitness_weight: float = 0.08,
    transfer_samples_per_step: int = 3,
    enable_autopoietic_objective: bool = False,
    autopoietic_loss_weight: float = 0.10,
    autopoietic_fitness_gain: float = 0.15,
    autopoietic_self_repair_weight: float = 0.35,
    autopoietic_closure_weight: float = 0.45,
    autopoietic_resource_cycle_weight: float = 0.20,
    autopoietic_loss_weight_cycle: str = "",
    remap_loss_weight: float = 0.1,
    detection_loss_weight: float = 0.1,
    emergent_signal_loss_weight: float = 0.05,
    enable_adaptive_loss: bool = False,
    adaptive_loss_alpha: float = 0.1,
    memory_persistence_loss_weight: float = 0.05,
    genetic_memory_persistence_weight: float = 0.05,
    paging_loss_weight: float = 0.01,
    enable_curriculum: bool = False,
    curriculum_power: float = 1.0,
    curriculum_power_cycle: str = "",
    remap_probability_start: float = 0.1,
    remap_probability_end: float = 0.35,
    env_volatility_start: float = 0.05,
    env_volatility_end: float = 0.55,
    noise_profile: str = "none",
    enable_noise_curriculum: bool = False,
    noise_strength_start: float = 0.2,
    noise_strength_end: float = 1.0,
    force_curriculum_mode: str = "none",
    force_curriculum_strength_start: float = 0.0,
    force_curriculum_strength_end: float = 1.0,
    enable_multi_scale_gating: bool = True,
    init_weights: str = "",
    out_dir: str = "artifacts/parallel",
    max_workers: int = 2,
    seed: int = 42,
) -> None:
    output_path = Path(out_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    resolved_device = choose_device(device, strict=strict_device)

    variant_cmds = []
    
    # Cycles: allow rotating parameters across variants
    auto_cycle = [float(x) for x in autopoietic_loss_weight_cycle.split(",") if x.strip()] if autopoietic_loss_weight_cycle else []
    curr_cycle = [float(x) for x in curriculum_power_cycle.split(",") if x.strip()] if curriculum_power_cycle else []

    for i in range(variants):
        variant_seed = seed + i * 1337
        variant_output = output_path / f"variant-{i:02d}.pt"
        variant_metrics = output_path / f"variant-{i:02d}.metrics.json"
        
        # Use venv python or sys.executable
        py_exe = sys.executable
        
        cmd = [
            py_exe, "-m", "ai_embedded_dynamic_diversity.train.cli",
            "--epochs", str(epochs),
            "--batch-size", str(batch_size),
            "--unroll-steps", str(unroll_steps),
            "--lr", str(lr),
            "--profile", profile,
            "--world-profile", world_profile,
            "--device", device,
            "--embodiments", embodiments,
            "--transfer-loss-weight", str(transfer_loss_weight),
            "--transfer-fitness-weight", str(transfer_fitness_weight),
            "--transfer-samples-per-step", str(transfer_samples_per_step),
            "--remap-loss-weight", str(remap_loss_weight),
            "--detection-loss-weight", str(detection_loss_weight),
            "--emergent-signal-loss-weight", str(emergent_signal_loss_weight),
            "--memory-persistence-loss-weight", str(memory_persistence_loss_weight),
            "--genetic-memory-persistence-weight", str(genetic_memory_persistence_weight),
            "--paging-loss-weight", str(paging_loss_weight),
            "--remap-probability-start", str(remap_probability_start),
            "--remap-probability-end", str(remap_probability_end),
            "--env-volatility-start", str(env_volatility_start),
            "--env-volatility-end", str(env_volatility_end),
            "--noise-profile", noise_profile,
            "--noise-strength-start", str(noise_strength_start),
            "--noise-strength-end", str(noise_strength_end),
            "--force-curriculum-mode", force_curriculum_mode,
            "--force-curriculum-strength-start", str(force_curriculum_strength_start),
            "--force-curriculum-strength-end", str(force_curriculum_strength_end),
            "--seed", str(variant_seed),
            "--save-path", str(variant_output),
            "--metrics-path", str(variant_metrics),
        ]
        cmd.append("--strict-device" if strict_device else "--no-strict-device")

        if coevolution:
            cmd.append("--coevolution")
            cmd.extend(["--population-size", str(population_size)])
        if enable_embodiment_transfer_loss:
            cmd.append("--enable-embodiment-transfer-loss")
        if enable_autopoietic_objective:
            cmd.append("--enable-autopoietic-objective")
            aw = auto_cycle[i % len(auto_cycle)] if auto_cycle else autopoietic_loss_weight
            cmd.extend(["--autopoietic-loss-weight", str(aw)])
            cmd.extend(["--autopoietic-fitness-gain", str(autopoietic_fitness_gain)])
            cmd.extend(["--autopoietic-self-repair-weight", str(autopoietic_self_repair_weight)])
            cmd.extend(["--autopoietic-closure-weight", str(autopoietic_closure_weight)])
            cmd.extend(["--autopoietic-resource-cycle-weight", str(autopoietic_resource_cycle_weight)])
        
        if enable_adaptive_loss:
            cmd.append("--enable-adaptive-loss")
            cmd.extend(["--adaptive-loss-alpha", str(adaptive_loss_alpha)])

        if enable_curriculum:
            cmd.append("--enable-curriculum")
            cp = curr_cycle[i % len(curr_cycle)] if curr_cycle else curriculum_power
            cmd.extend(["--curriculum-power", str(cp)])
        if enable_noise_curriculum:
            cmd.append("--enable-noise-curriculum")
        if enable_multi_scale_gating:
            cmd.append("--enable-multi-scale-gating")
        if init_weights:
            cmd.extend(["--init-weights", init_weights])

        variant_cmds.append((cmd, output_path / f"variant-{i:02d}.log"))

    print(
        {
            "launch": "parallel-variants",
            "variants": variants,
            "max_workers": max_workers,
            "requested_device": device,
            "resolved_device": str(resolved_device),
            "strict_device": strict_device,
            "runtime": device_runtime_snapshot(),
        }
    )
    start_time = time.perf_counter()

    if max_workers <= 1:
        # Avoid multiprocessing setup overhead/permission issues in constrained hosts.
        results = [_run_variant(cmd, log) for cmd, log in variant_cmds]
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_run_variant, cmd, log) for cmd, log in variant_cmds]
            results = [f.result() for f in futures]

    duration = time.perf_counter() - start_time
    print(f"[bold green]Parallel sweep complete in {duration:.2f}s[/bold green]")

    # Aggregate summary
    summary = {
        "variants": variants,
        "duration_s": duration,
        "results": []
    }
    for i in range(variants):
        m_path = output_path / f"variant-{i:02d}.metrics.json"
        if m_path.exists():
            data = json.loads(m_path.read_text())
            last_record = data["records"][-1] if data["records"] else {}
            summary["results"].append({
                "variant": i,
                "best_fitness": last_record.get("best_fitness") or last_record.get("fitness"),
                "mean_transfer_mismatch": last_record.get("mean_transfer_mismatch"),
                "metrics_file": str(m_path)
            })

    summary["results"].sort(key=lambda x: x["best_fitness"] if x["best_fitness"] is not None else -1e9, reverse=True)
    (output_path / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[bold yellow]Summary saved to {output_path / 'summary.json'}[/bold yellow]")


if __name__ == "__main__":
    app()
