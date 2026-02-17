from __future__ import annotations

import statistics
import time

import torch
import typer

from ai_embedded_dynamic_diversity.config import model_config_for_profile
from ai_embedded_dynamic_diversity.models import ModelCore

app = typer.Typer(add_completion=False)


@app.command()
def run(
    weights: str = "",
    profile: str = "pi5",
    steps: int = 300,
    warmup_steps: int = 30,
    batch_size: int = 1,
    device: str = "cpu",
) -> None:
    dev = torch.device(device)
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
    print({"device": str(dev), "profile": profile, "batch_size": batch_size, "p50_ms": p50, "p95_ms": p95, "avg_ms": statistics.mean(timings_ms)})


if __name__ == "__main__":
    app()
