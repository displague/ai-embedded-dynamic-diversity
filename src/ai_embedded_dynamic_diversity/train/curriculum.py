from __future__ import annotations

import torch

class AdaptiveLossController:
    """Dynamically adjusts loss weights to prevent signaling collapse and balance tracks."""
    
    def __init__(
        self, 
        initial_weights: dict[str, float], 
        alpha: float = 0.1, 
        target_ratios: dict[str, float] | None = None
    ):
        self.weights = initial_weights.copy()
        self.alpha = alpha
        # Target ratios of component contribution to total loss
        # Default targets if none provided
        self.target_ratios = target_ratios or {
            "detection_loss": 0.15,
            "remap_loss": 0.15,
            "emergent_signal_loss": 0.10,
            "recon": 0.40,
        }
        self.moving_avgs = {}

    def step(self, logs: dict[str, float]) -> dict[str, float]:
        """Updates weights based on recent logs and returns the new weight set."""
        total_loss = logs.get("loss", 1.0)
        
        for key, value in logs.items():
            if key not in self.moving_avgs:
                self.moving_avgs[key] = value
            else:
                self.moving_avgs[key] = 0.9 * self.moving_avgs[key] + 0.1 * value

        # Adjust weights for specific tracks to maintain "pressure"
        # We only adjust weights that are present in target_ratios and initial_weights
        for component, target_ratio in self.target_ratios.items():
            weight_key = f"{component}_weight"
            if component == "recon":
                continue # We usually keep recon as the anchor
            
            # Check if we have a corresponding weight in our set
            actual_weight_key = component if component.endswith("_weight") else f"{component}_weight"
            if actual_weight_key not in self.weights:
                continue
                
            current_val = self.moving_avgs.get(component, 0.0)
            current_contribution = self.weights[actual_weight_key] * current_val
            current_ratio = current_contribution / max(1e-8, total_loss)
            
            if current_ratio < target_ratio:
                # Under-pressured: increase weight
                self.weights[actual_weight_key] *= (1.0 + self.alpha)
            elif current_ratio > target_ratio * 2.0:
                # Over-pressured: decrease weight slightly to allow other tracks to breathe
                self.weights[actual_weight_key] *= (1.0 - self.alpha * 0.5)
            
            # Clamp weights to reasonable ranges to prevent explosion
            self.weights[actual_weight_key] = max(1e-4, min(self.weights[actual_weight_key], 2.0))

        return self.weights
