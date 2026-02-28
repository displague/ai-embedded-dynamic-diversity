from __future__ import annotations

import torch

class IOAdapter:
    """Hardware-agnostic runtime I/O abstraction for ModelCore deployment."""
    
    def __init__(self, signal_dim: int):
        self.signal_dim = signal_dim

    def normalize(self, raw_input: torch.Tensor) -> torch.Tensor:
        """Normalizes arbitrary hardware input to [0, 1] for ModelCore."""
        # Anonymous-channel robust normalization: 
        # use per-sample min/max span or moving window statistics.
        # Here we use a simple min-max per sample for functional baseline.
        if raw_input.dim() == 1:
            raw_input = raw_input.unsqueeze(0)
            
        min_v = raw_input.min(dim=1, keepdim=True).values
        max_v = raw_input.max(dim=1, keepdim=True).values
        span = (max_v - min_v).clamp_min(1e-6)
        
        normalized = (raw_input - min_v) / span
        return normalized

    def denormalize(self, model_output: torch.Tensor, target_range: tuple[float, float] = (0.0, 1.0)) -> torch.Tensor:
        """Maps ModelCore output [0, 1] to specific hardware ranges."""
        low, high = target_range
        mapped = low + model_output * (high - low)
        
        if mapped.dim() > 1 and mapped.size(0) == 1:
            mapped = mapped.squeeze(0)
            
        return mapped
