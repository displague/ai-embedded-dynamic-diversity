from __future__ import annotations

import json
from pathlib import Path

import torch
import typer

from ai_embedded_dynamic_diversity.config import model_config_for_profile
from ai_embedded_dynamic_diversity.models import ModelCore

app = typer.Typer(add_completion=False)


@torch.no_grad()
def _long_horizon_recovery(model: ModelCore, cfg, steps: int = 80, shock_step: int = 28) -> float:
    device = next(model.parameters()).device
    memory = model.init_memory(1, cfg.memory_slots, cfg.memory_dim, device)
    remap = torch.zeros(1, cfg.max_remap_groups, device=device)
    signal = torch.zeros(1, cfg.signal_dim, device=device)

    baseline = None
    post_shock = []
    for t in range(steps):
        if t == shock_step:
            signal = torch.randn_like(signal) * 2.2
        elif t > shock_step:
            signal = signal * 0.88
        out = model(signal, memory, remap)
        memory = out["memory"]
        marker = out["io"].abs().mean().item()
        if t == shock_step - 1:
            baseline = marker
        if t > shock_step:
            post_shock.append(marker)
    if baseline is None or not post_shock:
        return 0.0
    recovery_error = sum(abs(v - baseline) for v in post_shock[-20:]) / min(20, len(post_shock))
    return 1.0 / (1e-6 + recovery_error)


@app.command()
def run(
    profile: str = "pi5",
    device: str = "cpu",
    output: str = "artifacts/gating-bench.json",
) -> None:
    dev = torch.device(device)
    base_cfg = model_config_for_profile(profile)

    variants = [
        {"name": "sigmoid", "gating_mode": "sigmoid", "topk_gating": 0, "enable_dmd_gating": False, "enable_phase_gating": False},
        {"name": "symplectic", "gating_mode": "symplectic", "topk_gating": 0, "enable_dmd_gating": False, "enable_phase_gating": False},
        {"name": "symplectic_topk", "gating_mode": "symplectic", "topk_gating": min(4, base_cfg.memory_slots), "enable_dmd_gating": False, "enable_phase_gating": False},
        {"name": "symplectic_dmd_phase", "gating_mode": "symplectic", "topk_gating": min(4, base_cfg.memory_slots), "enable_dmd_gating": True, "enable_phase_gating": True},
    ]

    results = []
    for variant in variants:
        cfg = model_config_for_profile(profile)
        cfg.gating_mode = variant["gating_mode"]
        cfg.topk_gating = variant["topk_gating"]
        cfg.enable_dmd_gating = variant["enable_dmd_gating"]
        cfg.enable_phase_gating = variant["enable_phase_gating"]

        model = ModelCore(**cfg.__dict__).to(dev).eval()
        score = _long_horizon_recovery(model, cfg)
        results.append({"name": variant["name"], "recovery_score": score})
        print({"variant": variant["name"], "recovery_score": score})

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print({"saved": output})


if __name__ == "__main__":
    app()
