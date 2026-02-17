from __future__ import annotations

import torch
from torch import nn


class PassiveTensorField(nn.Module):
    """Continuously interprets anonymous multi-modal signals into edge-node readiness."""

    def __init__(self, signal_dim: int, hidden_dim: int, edge_nodes: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(signal_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.edge_projection = nn.Linear(hidden_dim, edge_nodes)
        self.energy_projection = nn.Linear(hidden_dim, 1)

    def forward(self, signal: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        latent = self.encoder(signal)
        readiness = torch.sigmoid(self.edge_projection(latent))
        energy = torch.relu(self.energy_projection(latent))
        return latent, readiness, energy


class ActiveMemoryTensor(nn.Module):
    """Single-process latent capture into reusable memory slots (latent+genetic memory)."""

    def __init__(
        self,
        hidden_dim: int,
        memory_slots: int,
        memory_dim: int,
        gating_mode: str = "sigmoid",
        topk: int = 0,
        enable_dmd_gating: bool = False,
        enable_phase_gating: bool = False,
    ):
        super().__init__()
        self.memory_slots = memory_slots
        self.memory_dim = memory_dim
        self.gating_mode = gating_mode
        self.topk = topk
        self.enable_dmd_gating = enable_dmd_gating
        self.enable_phase_gating = enable_phase_gating

        self.write_gate = nn.Linear(hidden_dim + memory_dim, 1)
        self.symplectic_gate = nn.Linear(hidden_dim + memory_dim, 2)
        self.symplectic_scale = nn.Parameter(torch.tensor(1.0))
        self.write_value = nn.Linear(hidden_dim, memory_dim)
        self.read_query = nn.Linear(hidden_dim, memory_dim)
        self.phase_proj = nn.Linear(hidden_dim, 1)
        self.dmd_proj = nn.Linear(hidden_dim + memory_dim, 1)
        self.memory_key = nn.Parameter(torch.randn(memory_slots, memory_dim) * 0.02)

    def forward(self, latent: torch.Tensor, memory: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        query = self.read_query(latent)
        logits = torch.einsum("bd,sd->bs", query, self.memory_key)
        weights = torch.softmax(logits, dim=-1)
        if self.topk > 0 and self.topk < self.memory_slots:
            topk_idx = torch.topk(weights, k=self.topk, dim=-1).indices
            mask = torch.zeros_like(weights).scatter(1, topk_idx, 1.0)
            weights = weights * mask
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        read = torch.einsum("bs,bsd->bd", weights, memory)

        gate_input = torch.cat([latent, read], dim=-1)
        if self.gating_mode == "symplectic":
            sym = torch.tanh(self.symplectic_gate(gate_input) * self.symplectic_scale)
            gate = (torch.max(sym[:, :1], sym[:, 1:]) + 1.0) * 0.5
        else:
            gate = torch.sigmoid(self.write_gate(gate_input))
        if self.enable_dmd_gating:
            dmd_signal = torch.tanh(self.dmd_proj(gate_input))
            gate = torch.clamp(gate * (0.75 + 0.5 * (dmd_signal + 1.0) * 0.5), 0.0, 1.0)
        if self.enable_phase_gating:
            phase = self.phase_proj(latent)
            phase_factor = 0.5 * (1.0 + torch.cos(phase))
            gate = gate * phase_factor
        value = self.write_value(latent).unsqueeze(1)
        new_memory = (1.0 - gate.unsqueeze(-1) * weights.unsqueeze(-1)) * memory + gate.unsqueeze(-1) * weights.unsqueeze(-1) * value
        return read, new_memory, weights


class AnonymousEdgeRouter(nn.Module):
    """Maps persistent anonymous channels to edges while supporting dynamic remapping."""

    def __init__(self, edge_nodes: int, io_channels: int, max_remap_groups: int):
        super().__init__()
        self.edge_nodes = edge_nodes
        self.io_channels = io_channels
        self.max_remap_groups = max_remap_groups
        self.base_map = nn.Parameter(torch.randn(io_channels, edge_nodes) * 0.05)
        self.adapt = nn.Sequential(
            nn.Linear(edge_nodes + max_remap_groups, edge_nodes),
            nn.SiLU(),
            nn.Linear(edge_nodes, edge_nodes),
        )

    def forward(self, readiness: torch.Tensor, remap_code: torch.Tensor) -> torch.Tensor:
        map_logits = self.base_map + self.adapt(torch.cat([readiness, remap_code], dim=-1)).unsqueeze(1)
        map_weights = torch.softmax(map_logits, dim=-1)
        outputs = torch.einsum("bce,be->bc", map_weights, readiness)
        return outputs


class ModelCore(nn.Module):
    def __init__(
        self,
        signal_dim: int,
        hidden_dim: int,
        edge_nodes: int,
        memory_slots: int,
        memory_dim: int,
        io_channels: int,
        max_remap_groups: int,
        gating_mode: str = "sigmoid",
        topk_gating: int = 0,
        enable_dmd_gating: bool = False,
        enable_phase_gating: bool = False,
    ):
        super().__init__()
        self.passive = PassiveTensorField(signal_dim, hidden_dim, edge_nodes)
        self.active = ActiveMemoryTensor(
            hidden_dim,
            memory_slots,
            memory_dim,
            gating_mode=gating_mode,
            topk=topk_gating,
            enable_dmd_gating=enable_dmd_gating,
            enable_phase_gating=enable_phase_gating,
        )
        self.memory_to_edge = nn.Linear(memory_dim, edge_nodes)
        self.router = AnonymousEdgeRouter(edge_nodes, io_channels, max_remap_groups)

    def init_memory(self, batch_size: int, memory_slots: int, memory_dim: int, device: str | torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, memory_slots, memory_dim, device=device)

    def forward(self, signal: torch.Tensor, memory: torch.Tensor, remap_code: torch.Tensor) -> dict[str, torch.Tensor]:
        latent, readiness, energy = self.passive(signal)
        read, new_memory, memory_weights = self.active(latent, memory)
        refined_readiness = torch.sigmoid(readiness + self.memory_to_edge(read))
        io = self.router(refined_readiness, remap_code)
        return {
            "latent": latent,
            "readiness": refined_readiness,
            "energy": energy,
            "io": io,
            "memory": new_memory,
            "memory_weights": memory_weights,
        }
