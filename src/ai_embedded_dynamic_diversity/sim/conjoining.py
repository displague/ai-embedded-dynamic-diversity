from __future__ import annotations

import torch
from ai_embedded_dynamic_diversity.sim.world import WorldState, EnvironmentControls
from ai_embedded_dynamic_diversity.sim.signaling import SignalingWorld

class ConjoiningWorld(SignalingWorld):
    """Simulates environment/tool coupling and organism-formation tracks."""
    
    def __init__(self, x: int, y: int, z: int, resource_channels: int, decay: float = 0.03, device: str = "cpu"):
        super().__init__(x, y, z, resource_channels, decay, device)
        self.tool_pos = None
        self.is_attached = None

    def init(self, batch_size: int) -> WorldState:
        state = super().init(batch_size)
        # Random tool positions
        self.tool_pos = torch.randn(batch_size, 3, device=self.device) * 0.5
        self.is_attached = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        return state

    def step(self, state: WorldState, action_field: torch.Tensor, controls: EnvironmentControls) -> WorldState:
        # Before moving, check for new attachments
        dist = torch.norm(state.object_pos - self.tool_pos, dim=1)
        self.is_attached |= (dist < 0.1)
        
        # Capture old object pos
        old_obj_pos = state.object_pos.clone()
        
        # Step the core world
        state = super().step(state, action_field, controls)
        
        # Update tool positions for attached agents
        if self.is_attached is not None:
            # Tool moves with object if attached
            diff = state.object_pos - old_obj_pos
            self.tool_pos[self.is_attached] += diff[self.is_attached]
            
        return state

    def encode_observation_with_conjoining(self, state: WorldState, signal_dim: int, labels: torch.Tensor) -> torch.Tensor:
        """Adds tool/conjoining status to anonymous observation."""
        obs = self.encode_observation_with_signals(state, signal_dim, labels)
        
        # Mix in relative tool position and attachment status
        if self.tool_pos is not None:
            rel_tool = self.tool_pos - state.object_pos
            # Use specific channels for tool info (e.g. 10-13)
            obs[:, 10:13] = obs[:, 10:13] + 0.3 * rel_tool
            obs[:, 13] = obs[:, 13] + 0.5 * self.is_attached.float()
            
        return torch.clamp(obs, 0.0, 1.0)
