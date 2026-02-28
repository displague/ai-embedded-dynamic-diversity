from __future__ import annotations

import torch
from ai_embedded_dynamic_diversity.models import ModelCore

class AdaptiveStreamer:
    """Adaptive batch/chunk streaming for reduced peak memory during long rollouts."""
    
    def __init__(self, model: ModelCore, chunk_size: int = 16, device: str = "cpu"):
        self.model = model
        self.chunk_size = chunk_size
        self.device = torch.device(device)
        self.memory = None

    def stream_rollout(self, signals: torch.Tensor):
        """Processes a long sequence of signals in memory-efficient chunks."""
        # signals: (TotalSteps, signal_dim)
        total_steps = signals.size(0)
        
        if self.memory is None:
            # Initialize memory for a single sample (batch=1)
            # (Note: Streamer assumes sequential processing for one deployment)
            self.memory = self.model.init_memory(1, self.model.memory_slots, self.model.memory_dim, self.device)
            
        self.model.eval()
        with torch.no_grad():
            for i in range(0, total_steps, self.chunk_size):
                chunk = signals[i : i + self.chunk_size].to(self.device)
                chunk_outputs = []
                
                for step in range(chunk.size(0)):
                    # Step-by-step unroll to keep memory state correct
                    # (In our current ModelCore, batch dim is 0)
                    obs = chunk[step].unsqueeze(0)
                    out = self.model(obs, self.memory)
                    self.memory = out["memory"]
                    chunk_outputs.append(out["io"])
                
                yield torch.cat(chunk_outputs, dim=0)

    def reset(self):
        """Resets the internal memory state."""
        self.memory = None
