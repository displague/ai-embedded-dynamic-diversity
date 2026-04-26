from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LatentWorldPredictor(nn.Module):
    """
    JEPA-style latent world predictor (LeWM-inspired).

    Learns f(z_t, a_t) → ẑ_{t+1} in latent space via a 2-layer MLP.
    Trained with next-embedding MSE + SIGReg to prevent representational collapse.
    Attached alongside ModelCore — no shared parameters.
    """

    def __init__(self, latent_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + action_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: current latent observation [B, latent_dim]
            a: current action embedding [B, action_dim]
        Returns:
            predicted next latent [B, latent_dim]
        """
        return self.net(torch.cat([z, a], dim=-1))
