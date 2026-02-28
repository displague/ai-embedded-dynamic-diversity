from __future__ import annotations

import torch
import time
from ai_embedded_dynamic_diversity.sim.signaling import SignalingWorld
from ai_embedded_dynamic_diversity.sim.world import WorldState, EnvironmentControls

class MimicryWorld(SignalingWorld):
    """World with peer agents that emit periodic or reactive signals for mimicry tracks."""
    
    def __init__(self, x: int, y: int, z: int, resource_channels: int, decay: float = 0.03, device: str = "cpu"):
        super().__init__(x, y, z, resource_channels, decay, device)
        self.peer_signals = None

    def init(self, batch_size: int) -> WorldState:
        state = super().init(batch_size)
        # Peers emit signals in a hidden dimension
        # (For now, let's just create a dynamic target signal that the model *should* learn to mirror)
        self.peer_signals = torch.zeros(batch_size, 8, device=self.device) # 8 = emergent_signal_dim
        return state

    def step(self, state: WorldState, action_field: torch.Tensor, controls: EnvironmentControls) -> WorldState:
        state = super().step(state, action_field, controls)
        
        # Update peer signals: complex periodic oscillation
        t = time.time()
        B = state.life.size(0)
        # Create a "peer" signal that is a function of world state (e.g. resources + light)
        # This makes the "correct" mimicry signal emergent from the environment.
        res_mean = state.resources[:, :1].mean(dim=(2, 3, 4))
        light_phase = torch.sin(torch.tensor(t * 2.0, device=self.device))
        
        for i in range(8):
            self.peer_signals[:, i] = torch.sin(light_phase + i * 0.5) * 0.5 + 0.5 * res_mean.squeeze()
            
        return state

    def encode_observation_with_mimicry(self, state: WorldState, signal_dim: int, labels: torch.Tensor) -> torch.Tensor:
        """Mixes peer signals into the anonymous observation stream."""
        obs = self.encode_observation_with_signals(state, signal_dim, labels)
        
        if self.peer_signals is not None:
            # Hide peer signals in some anonymous channels (e.g. 20-28)
            # The model must find these and mirror them in its emergent_signal_head
            obs[:, 20:28] = obs[:, 20:28] + 0.4 * self.peer_signals
            
        return torch.clamp(obs, 0.0, 1.0)
