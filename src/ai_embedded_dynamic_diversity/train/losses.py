from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def io_differentiation_loss(io: torch.Tensor, margin: float = 0.15) -> torch.Tensor:
    """Penalise near-uniform IO outputs across channels (channel collapse).

    Computes per-sample std across io_channels and penalises when it falls
    below `margin`. Zero gradient once std exceeds the margin.
    """
    channel_std = io.std(dim=1)                    # [batch]
    return torch.relu(margin - channel_std).mean()


def dof_coverage_loss(applied: torch.Tensor, threshold: float = 0.05) -> torch.Tensor:
    """Penalise any control channel whose mean absolute activation is below threshold.

    Ensures no DOF is permanently silent across the batch.
    """
    mean_activation = applied.abs().mean(dim=0)    # [control_dim]
    return torch.relu(threshold - mean_activation).mean()


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
    memory_persistence_loss_weight: float = 0.05,
    initial_memory: torch.Tensor | None = None,
    paging_loss_weight: float = 0.01,
    io_diff_weight: float = 0.02,
    dof_coverage_weight: float = 0.01,
    applied_signal: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    io = outputs["io"]
    readiness = outputs["readiness"]
    energy = outputs["energy"].mean()
    memory_weights = outputs["memory_weights"]
    predicted_remap = outputs["predicted_remap"]
    predicted_signal_type = outputs["predicted_signal_type"]
    emergent_signal = outputs["emergent_signal"]
    memory = outputs["memory"]

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
    emergent_signal_loss = -torch.var(emergent_signal, dim=0).mean()

    # Memory persistence loss: keep current memory close to initial genetic prior
    memory_persistence_loss = torch.tensor(0.0, device=io.device)
    if initial_memory is not None:
        memory_persistence_loss = nn.functional.mse_loss(memory, initial_memory)

    # Paging loss: encourage using fewer memory slots per sample (L1 sparsity)
    paging_loss = torch.mean(torch.sum(torch.abs(memory_weights), dim=-1))

    # IO channel differentiation: penalise near-uniform outputs across channels
    io_diff = io_differentiation_loss(io) if io_diff_weight > 0.0 else io.new_zeros(())

    # DOF coverage: penalise permanently-silent control channels
    dof_cov = (
        dof_coverage_loss(applied_signal)
        if dof_coverage_weight > 0.0 and applied_signal is not None
        else io.new_zeros(())
    )

    time_consistency = torch.mean(torch.abs(outputs["memory"][:, 1:] - outputs["memory"][:, :-1]))
    total = (
        recon
        + entropy_weight * entropy
        + energy_weight * (energy + 0.5 * readiness_sparsity)
        + memory_consistency_weight * time_consistency
        + remap_loss_weight * remap_loss
        + detection_loss_weight * detection_loss
        + emergent_signal_loss_weight * emergent_signal_loss
        + memory_persistence_loss_weight * memory_persistence_loss
        + paging_loss_weight * paging_loss
        + io_diff_weight * io_diff
        + dof_coverage_weight * dof_cov
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
        "memory_persistence_loss": memory_persistence_loss.item(),
        "paging_loss": paging_loss.item(),
        "io_diff_loss": io_diff.item(),
        "dof_coverage_loss": dof_cov.item(),
    }
    return total, logs


def world_prediction_loss(
    pred_latent: torch.Tensor,
    actual_latent: torch.Tensor,
    sigreg_weight: float = 0.1,
) -> torch.Tensor:
    """
    JEPA-style world-prediction loss (LeWM-inspired).

    MSE between predicted and actual next latent, plus SIGReg variance
    regularisation to prevent representational collapse.

    Args:
        pred_latent:   predicted next-step latent [B, D]
        actual_latent: actual next-step latent     [B, D]  (detached from model graph)
        sigreg_weight: weight for variance regularisation term
    """
    mse = F.mse_loss(pred_latent, actual_latent.detach())
    # SIGReg: penalise low variance across batch to prevent collapse
    sigreg = sigreg_weight * torch.relu(1.0 - pred_latent.var(dim=0).mean())
    return mse + sigreg

