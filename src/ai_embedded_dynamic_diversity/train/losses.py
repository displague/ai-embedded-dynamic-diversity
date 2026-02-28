from __future__ import annotations

import torch
from torch import nn


def loss_fn(
    outputs: dict[str, torch.Tensor],
    target_signal: torch.Tensor,
    entropy_weight: float,
    energy_weight: float,
    memory_consistency_weight: float,
    remap_loss_weight: float = 0.1,
    target_remap_code: torch.Tensor | None = None,
    detection_loss_weight: float = 0.1,
    target_signal_type: torch.Tensor | None = None,
    emergent_signal_loss_weight: float = 0.05,
) -> tuple[torch.Tensor, dict[str, float]]:
    io = outputs["io"]
    readiness = outputs["readiness"]
    energy = outputs["energy"].mean()
    memory_weights = outputs["memory_weights"]
    predicted_remap = outputs["predicted_remap"]
    predicted_signal_type = outputs["predicted_signal_type"]
    emergent_signal = outputs["emergent_signal"]

    recon = nn.functional.mse_loss(io, target_signal)
    entropy = -torch.mean(torch.sum(memory_weights * torch.log(memory_weights.clamp_min(1e-8)), dim=-1))
    readiness_sparsity = torch.mean(torch.abs(readiness))

    remap_loss = torch.tensor(0.0, device=io.device)
    if target_remap_code is not None:
        remap_loss = nn.functional.mse_loss(predicted_remap, target_remap_code)

    detection_loss = torch.tensor(0.0, device=io.device)
    if target_signal_type is not None:
        detection_loss = nn.functional.cross_entropy(predicted_signal_type, target_signal_type)

    # Emergent signal loss: reward signal variance (avoiding constant/zero signals)
    # We use negative variance as loss to maximize it.
    emergent_signal_loss = -torch.var(emergent_signal, dim=0).mean()

    time_consistency = torch.mean(torch.abs(outputs["memory"][:, 1:] - outputs["memory"][:, :-1]))
    total = (
        recon
        + entropy_weight * entropy
        + energy_weight * (energy + 0.5 * readiness_sparsity)
        + memory_consistency_weight * time_consistency
        + remap_loss_weight * remap_loss
        + detection_loss_weight * detection_loss
        + emergent_signal_loss_weight * emergent_signal_loss
    )
    logs = {
        "loss": total.item(),
        "recon": recon.item(),
        "entropy": entropy.item(),
        "energy": energy.item(),
        "memory_consistency": time_consistency.item(),
        "remap_loss": remap_loss.item(),
        "detection_loss": detection_loss.item(),
        "emergent_signal_loss": emergent_signal_loss.item(),
    }
    return total, logs
