from __future__ import annotations

import torch
from ai_embedded_dynamic_diversity.sim.world import DynamicDiversityWorld, WorldState, EnvironmentControls

class SignalingWorld(DynamicDiversityWorld):
    """Extends world with explicit signal injection for detection tasks."""
    
    def __init__(self, x: int, y: int, z: int, resource_channels: int, decay: float = 0.03, device: str = "cpu"):
        super().__init__(x, y, z, resource_channels, decay, device)
        # Signal types: 0=none, 1=peer, 2=environment, 3=threat
        self.signal_types = 4
        # Threat agent position state (not in core WorldState yet, we'll manage it here for now)
        self.threat_pos = None 

    def init(self, batch_size: int) -> WorldState:
        state = super().init(batch_size)
        # Initialize threat agent positions randomly at the edges
        self.threat_pos = torch.sign(torch.randn(batch_size, 3, device=self.device)) * 0.9
        return state

    def step(self, state: WorldState, action_field: torch.Tensor, controls: EnvironmentControls) -> WorldState:
        # Move threat agent towards the object
        if self.threat_pos is not None:
            direction = state.object_pos - self.threat_pos
            dist = torch.norm(direction, dim=1, keepdim=True).clamp_min(1e-6)
            move = (direction / dist) * 0.05 # Speed
            self.threat_pos = self.threat_pos + move
            
            # If threat agent is very close to object, it increases stress
            proximity = torch.norm(state.object_pos - self.threat_pos, dim=1)
            collision_mask = proximity < 0.15
            if collision_mask.any():
                # Apply local stress at object position
                # For simplicity, we just increase global stress component here
                state.stress.add_(collision_mask.float().view(-1, 1, 1, 1, 1) * 0.1)
        
        return super().step(state, action_field, controls)

    def inject_signals(self, batch_size: int, p_peer: float = 0.1, p_env: float = 0.1, p_threat: float = 0.1) -> torch.Tensor:
        """Generates a batch of signal labels and their anonymous representations."""
        # labels: (B,)
        labels = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        r = torch.rand(batch_size, device=self.device)
        
        peer_mask = r < p_peer
        env_mask = (r >= p_peer) & (r < p_peer + p_env)
        threat_mask = (r >= p_peer + p_env) & (r < p_peer + p_env + p_threat)
        
        labels[peer_mask] = 1
        labels[env_mask] = 2
        labels[threat_mask] = 3
        
        # signals: (B, signal_dim) - we'll embed these into the observation later
        return labels

    def encode_observation_with_signals(self, state: WorldState, signal_dim: int, labels: torch.Tensor) -> torch.Tensor:
        """Encodes observation and mixes in anonymous signals based on labels."""
        obs = self.encode_observation(state, signal_dim)
        
        # Create anonymous signal patterns
        # Peer: harmonic oscillations
        # Env: low-frequency noise
        # Threat: high-frequency erratic bursts
        
        B = obs.size(0)
        signal_mix = torch.zeros_like(obs)
        
        t = torch.linspace(0, 1, signal_dim, device=self.device).unsqueeze(0).repeat(B, 1)
        
        # Peer pattern (label 1)
        peer_pattern = torch.sin(2 * 3.14159 * 5 * t) * 0.5 + 0.5
        
        # Env pattern (label 2)
        env_pattern = torch.sin(2 * 3.14159 * 1 * t) * 0.3 + 0.5
        
        # Threat pattern (label 3)
        threat_pattern = (torch.rand_like(t) > 0.8).float()
        
        peer_idx = (labels == 1).nonzero(as_tuple=True)[0]
        env_idx = (labels == 2).nonzero(as_tuple=True)[0]
        threat_idx = (labels == 3).nonzero(as_tuple=True)[0]
        
        if peer_idx.numel() > 0:
            signal_mix[peer_idx] = peer_pattern[peer_idx]
        if env_idx.numel() > 0:
            signal_mix[env_idx] = env_pattern[env_idx]
        if threat_idx.numel() > 0:
            signal_mix[threat_idx] = threat_pattern[threat_idx]
            
        # If threat agent is active, mix its relative position into the anonymous stream
        if self.threat_pos is not None:
            rel_pos = self.threat_pos - state.object_pos
            # Inject relative position into first 3 channels of signal_mix for all samples
            # (In a real anonymous scenario, this would be just another signal)
            signal_mix[:, :3] = signal_mix[:, :3] + 0.5 * rel_pos
            
        # Mix signals into observation (e.g. addition or replacement of some channels)
        # For "anonymous" streams, we'll just add them with some strength
        strength = 0.2
        return torch.clamp(obs + strength * signal_mix, 0.0, 1.0)
