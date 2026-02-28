from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

import torch
import typer

from ai_embedded_dynamic_diversity.config import model_config_for_profile
from ai_embedded_dynamic_diversity.models import ModelCore

app = typer.Typer(add_completion=False)


def run_benchmark(
    weights: str = "",
    profile: str = "pi5",
    steps: int = 300,
    warmup_steps: int = 30,
    batch_size: int = 1,
    device: str = "cpu",
) -> dict[str, object]:
    if steps <= 0:
        raise ValueError("steps must be > 0")
    if warmup_steps < 0:
        raise ValueError("warmup_steps must be >= 0")
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")

    dev = torch.device(device)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    cfg = model_config_for_profile(profile)
    model = ModelCore(**cfg.__dict__).to(dev)

    if weights:
        ckpt = torch.load(weights, map_location=dev)
        model.load_state_dict(ckpt["model"])

    model.eval()
    signal = torch.randn(batch_size, cfg.signal_dim, device=dev)
    memory = torch.zeros(batch_size, cfg.memory_slots, cfg.memory_dim, device=dev)
    remap = torch.zeros(batch_size, cfg.max_remap_groups, device=dev)

    with torch.no_grad():
        for _ in range(warmup_steps):
            out = model(signal, memory, remap)
            memory = out["memory"]

    timings_ms = []
    with torch.no_grad():
        for _ in range(steps):
            start = time.perf_counter()
            out = model(signal, memory, remap)
            if dev.type == "cuda":
                torch.cuda.synchronize()
            end = time.perf_counter()
            timings_ms.append((end - start) * 1000.0)
            memory = out["memory"]

    p50 = statistics.median(timings_ms)
    p95 = sorted(timings_ms)[int(0.95 * (len(timings_ms) - 1))]
    return {
        "device": str(dev),
        "profile": profile,
        "batch_size": batch_size,
        "steps": steps,
        "warmup_steps": warmup_steps,
        "p50_ms": p50,
        "p95_ms": p95,
        "avg_ms": statistics.mean(timings_ms),
    }


def _parse_int_csv(csv_value: str, name: str) -> list[int]:
    values: list[int] = []
    for raw in csv_value.split(","):
        token = raw.strip()
        if not token:
            continue
        try:
            parsed = int(token)
        except ValueError as exc:  # pragma: no cover - defensive
            raise ValueError(f"Invalid integer in {name}: {token}") from exc
        if parsed <= 0:
            raise ValueError(f"All values in {name} must be > 0")
        values.append(parsed)
    if not values:
        raise ValueError(f"{name} must include at least one value")
    return values


def _parse_targets(csv_value: str) -> list[tuple[str, str]]:
    targets: list[tuple[str, str]] = []
    for raw in csv_value.split(","):
        token = raw.strip()
        if not token:
            continue
        if "=" in token:
            label, device = token.split("=", 1)
        else:
            label, device = token, token
        label = label.strip().lower()
        device = device.strip().lower()
        if not label:
            raise ValueError("Target label cannot be empty")
        if not device:
            raise ValueError(f"Target device cannot be empty for label '{label}'")
        targets.append((label, device))
    if not targets:
        raise ValueError("targets must include at least one item")
    return targets


def build_latency_matrix(
    weights: str,
    profile: str,
    targets: str,
    batch_sizes: str,
    steps: int,
    warmup_steps: int,
) -> dict[str, object]:
    parsed_targets = _parse_targets(targets)
    parsed_batch_sizes = _parse_int_csv(batch_sizes, "batch_sizes")
    rows: list[dict[str, object]] = []

    for label, target_device in parsed_targets:
        lower_device = target_device.lower()
        if lower_device in {"external", "pending", "n/a", "na"}:
            rows.append(
                {
                    "target": label,
                    "device": target_device,
                    "status": "hardware_pending",
                    "reason": "Run this target on the corresponding hardware and merge artifacts.",
                    "runs": [],
                }
            )
            continue
        if lower_device == "cuda" and not torch.cuda.is_available():
            rows.append(
                {
                    "target": label,
                    "device": target_device,
                    "status": "skipped",
                    "reason": "CUDA unavailable on this host.",
                    "runs": [],
                }
            )
            continue

        runs: list[dict[str, object]] = []
        for batch_size in parsed_batch_sizes:
            runs.append(
                run_benchmark(
                    weights=weights,
                    profile=profile,
                    steps=steps,
                    warmup_steps=warmup_steps,
                    batch_size=batch_size,
                    device=target_device,
                )
            )
        rows.append({"target": label, "device": target_device, "status": "ok", "runs": runs})

    return {
        "profile": profile,
        "weights": weights if weights else None,
        "steps": steps,
        "warmup_steps": warmup_steps,
        "batch_sizes": parsed_batch_sizes,
        "targets": rows,
    }


@app.command()
def run(
    weights: str = "",
    profile: str = "pi5",
    steps: int = 300,
    warmup_steps: int = 30,
    batch_size: int = 1,
    device: str = "cpu",
) -> None:
    print(run_benchmark(weights=weights, profile=profile, steps=steps, warmup_steps=warmup_steps, batch_size=batch_size, device=device))


@app.command()
def matrix(
    weights: str = "",
    profile: str = "pi5",
    targets: str = "cpu=cpu,cuda=cuda,pi5=external,jetson=external,mobile=external",
    batch_sizes: str = "1,4,8",
    steps: int = 300,
    warmup_steps: int = 30,
    output: str = "artifacts/latency-matrix.json",
) -> None:
    payload = build_latency_matrix(
        weights=weights,
        profile=profile,
        targets=targets,
        batch_sizes=batch_sizes,
        steps=steps,
        warmup_steps=warmup_steps,
    )
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print({"latency_matrix": str(output_path), "targets": len(payload["targets"])})


if __name__ == "__main__":
    app()
