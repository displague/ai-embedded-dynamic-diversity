from __future__ import annotations

import copy
import torch
from torch import nn, optim
from ai_embedded_dynamic_diversity.models import ModelCore
from ai_embedded_dynamic_diversity.sim.embodiments import device_map_for_embodiment, get_embodiment
from ai_embedded_dynamic_diversity.train.losses import loss_fn

class FewShotAdaptor:
    """Protocol for rapid (N < 10 steps) embodiment adaptation."""
    
    def __init__(self, lr: float = 1e-3, steps: int = 5):
        self.lr = lr
        self.steps = steps

    def adapt(
        self, 
        model: ModelCore, 
        embodiment_name: str,
        calibration_data: list[tuple[torch.Tensor, torch.Tensor]], # [(obs, target_signal)]
        device: str | torch.device = "cpu"
    ) -> ModelCore:
        """Fine-tunes a copy of the model on a small calibration set."""
        dev = torch.device(device)
        adapted_model = copy.deepcopy(model).to(dev)
        adapted_model.train()
        
        # We typically only want to adapt the projection/remap heads if possible,
        # but few-shot often benefits from small full-model adjustments.
        optimizer = optim.AdamW(adapted_model.parameters(), lr=self.lr)
        
        emb = get_embodiment(embodiment_name)
        # Create a fixed remap code for this embodiment for calibration
        # In a real scenario, we might use the predicted remap or a known one.
        remap_code = torch.zeros(1, model.router.max_remap_groups, device=dev)
        # For simplicity in few-shot, we assume we know which group we're targeting 
        # or we let the model's internal predictor settle.
        
        for _ in range(self.steps):
            total_loss = 0.0
            for obs, target in calibration_data:
                obs = obs.to(dev)
                target = target.to(dev)
                
                optimizer.zero_grad()
                
                # Initial memory for calibration sample
                memory = adapted_model.init_memory(obs.size(0), 16, 64, dev) # Use default dims for now
                
                out = adapted_model(obs, memory) # Use predicted remap
                
                # We use a simplified loss for calibration: reconstruction only or standard loss
                loss, _ = loss_fn(
                    out, 
                    target, 
                    entropy_weight=0.01, 
                    energy_weight=0.01, 
                    memory_consistency_weight=0.01
                )
                
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                
        adapted_model.eval()
        return adapted_model
