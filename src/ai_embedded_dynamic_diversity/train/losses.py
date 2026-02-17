from __future__ import annotations

import torch
from torch import nn


def loss_fn(
    outputs: dict[str, torch.Tensor],
    target_signal: torch.Tensor,
    entropy_weight: float,
    energy_weight: float,
    memory_consistency_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    io = outputs["io"]
    readiness = outputs["readiness"]
    energy = outputs["energy"].mean()
    memory_weights = outputs["memory_weights"]

    recon = nn.functional.mse_loss(io, target_signal)
    entropy = -torch.mean(torch.sum(memory_weights * torch.log(memory_weights.clamp_min(1e-8)), dim=-1))
    readiness_sparsity = torch.mean(torch.abs(readiness))

    time_consistency = torch.mean(torch.abs(outputs["memory"][:, 1:] - outputs["memory"][:, :-1]))
    total = recon + entropy_weight * entropy + energy_weight * (energy + 0.5 * readiness_sparsity) + memory_consistency_weight * time_consistency
    logs = {
        "loss": total.item(),
        "recon": recon.item(),
        "entropy": entropy.item(),
        "energy": energy.item(),
        "memory_consistency": time_consistency.item(),
    }
    return total, logs
