from __future__ import annotations

import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import typer

app = typer.Typer(add_completion=False)


def _run_variant(cmd: list[str], env: dict[str, str]) -> tuple[int, str]:
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    output = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
    return proc.returncode, output


@app.command()
def run(
    variants: int = 4,
    profile: str = "pi5",
    epochs: int = 8,
    batch_size: int = 10,
    unroll_steps: int = 10,
    device: str = "cuda",
    device_pool: str = "",
    coevolution: bool = True,
    population_size: int = 4,
    embodiments: str = "hexapod,car,drone,polymorph120",
    enable_embodiment_transfer_loss: bool = True,
    transfer_loss_weight: float = 0.35,
    transfer_fitness_weight: float = 0.08,
    transfer_samples_per_step: int = 3,
    use_amp: bool = True,
    allow_tf32: bool = True,
    compile_model: bool = False,
    init_weights: str = "",
    max_workers: int = 2,
    out_dir: str = "artifacts/parallel",
) -> None:
    """Launch multiple diverse training variants in parallel and keep all checkpoints."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["PYTHONPATH"] = "src"
    device_list = [x.strip() for x in device_pool.replace(";", ",").split(",") if x.strip()]

    gating_modes = ["sigmoid", "symplectic"]
    topk_values = [0, 4]
    dmd_flags = [False, True]
    phase_flags = [False, True]

    jobs: list[tuple[int, list[str], str, str]] = []
    for i in range(variants):
        gating_mode = gating_modes[i % len(gating_modes)]
        topk_gating = topk_values[i % len(topk_values)]
        enable_dmd = dmd_flags[i % len(dmd_flags)]
        enable_phase = phase_flags[(i // 2) % len(phase_flags)]
        save_path = str(Path(out_dir) / f"variant-{i:02d}.pt")
        metrics_path = str(Path(out_dir) / f"variant-{i:02d}.metrics.json")
        enable_curriculum = i % 2 == 0
        enable_genetic_memory = i % 3 == 0
        assigned_device = device_list[i % len(device_list)] if device_list else device
        variant_transfer_loss_weight = transfer_loss_weight * (0.85 + 0.1 * (i % 3))
        variant_transfer_fitness_weight = transfer_fitness_weight * (0.9 + 0.1 * ((i + 1) % 3))

        cmd = [
            sys.executable,
            "-m",
            "ai_embedded_dynamic_diversity.train.cli",
            "--profile",
            profile,
            "--epochs",
            str(epochs),
            "--batch-size",
            str(batch_size),
            "--unroll-steps",
            str(unroll_steps),
            "--device",
            assigned_device,
            "--gating-mode",
            gating_mode,
            "--topk-gating",
            str(topk_gating),
            "--embodiments",
            embodiments,
            "--transfer-loss-weight",
            str(variant_transfer_loss_weight),
            "--transfer-fitness-weight",
            str(variant_transfer_fitness_weight),
            "--transfer-samples-per-step",
            str(transfer_samples_per_step),
            "--seed",
            str(13 + i),
            "--save-path",
            save_path,
            "--metrics-path",
            metrics_path,
        ]
        cmd += ["--use-amp"] if use_amp else ["--no-use-amp"]
        cmd += ["--allow-tf32"] if allow_tf32 else ["--no-allow-tf32"]
        cmd += ["--compile-model"] if compile_model else ["--no-compile-model"]
        if enable_embodiment_transfer_loss:
            cmd += ["--enable-embodiment-transfer-loss"]
        if init_weights:
            cmd += ["--init-weights", init_weights]
        if coevolution:
            cmd += ["--coevolution", "--population-size", str(population_size)]
        if enable_dmd:
            cmd += ["--enable-dmd-gating"]
        if enable_phase:
            cmd += ["--enable-phase-gating"]
        if enable_curriculum:
            cmd += ["--enable-curriculum"]
        if enable_genetic_memory:
            cmd += ["--enable-genetic-memory"]
        jobs.append((i, cmd, save_path, assigned_device))

    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        future_map = {ex.submit(_run_variant, cmd, env): (idx, save_path, assigned_device) for idx, cmd, save_path, assigned_device in jobs}
        for fut in as_completed(future_map):
            idx, save_path, assigned_device = future_map[fut]
            code, output = fut.result()
            log_path = str(Path(out_dir) / f"variant-{idx:02d}.log")
            Path(log_path).write_text(output, encoding="utf-8")
            metrics_file = str(Path(out_dir) / f"variant-{idx:02d}.metrics.json")
            metrics_payload = {}
            if Path(metrics_file).exists():
                try:
                    metrics_payload = json.loads(Path(metrics_file).read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    metrics_payload = {}
            records = metrics_payload.get("records", [])
            last = records[-1] if records else {}
            results.append(
                {
                    "variant": idx,
                    "code": code,
                    "device": assigned_device,
                    "checkpoint": save_path,
                    "metrics": metrics_file,
                    "log": log_path,
                    "flags": metrics_payload.get("flags", {}),
                    "fitness": last.get("best_fitness", last.get("fitness")),
                    "mean_step_ms": last.get("mean_step_ms"),
                }
            )
            print({"variant": idx, "code": code, "checkpoint": save_path})

    summary_path = str(Path(out_dir) / "summary.json")
    Path(summary_path).write_text(json.dumps(sorted(results, key=lambda x: x["variant"]), indent=2), encoding="utf-8")
    print({"summary": summary_path, "variants": variants})


if __name__ == "__main__":
    app()
