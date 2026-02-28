from __future__ import annotations

import torch
from abc import ABC, abstractmethod
from ai_embedded_dynamic_diversity.sim.world import WorldState

class Objective(ABC):
    """Base class for policy-level task objectives."""
    
    @abstractmethod
    def compute_reward(self, state: WorldState, **kwargs) -> torch.Tensor:
        """Compute task-specific reward for current state."""
        pass
    
    @abstractmethod
    def is_completed(self, state: WorldState, **kwargs) -> torch.Tensor:
        """Check if task objective is met."""
        pass

class NavigationObjective(Objective):
    """Drive embodiment/object towards a specific target position."""
    
    def __init__(self, target_pos: torch.Tensor, threshold: float = 0.1):
        self.target_pos = target_pos # Expected shape: (B, 3) or (1, 3)
        self.threshold = threshold

    def compute_reward(self, state: WorldState, **kwargs) -> torch.Tensor:
        # Distance-based reward: closer is better
        dist = torch.norm(state.object_pos - self.target_pos, dim=1)
        # Normalize: -1.0 to 1.0 coords mean max distance is sqrt(3*2^2) = ~3.46
        # Use exponential decay for reward
        reward = torch.exp(-1.5 * dist)
        return reward

    def is_completed(self, state: WorldState, **kwargs) -> torch.Tensor:
        dist = torch.norm(state.object_pos - self.target_pos, dim=1)
        return dist < self.threshold

class StabilityObjective(Objective):
    """Minimize overall world stress perceived by the embodiment."""
    
    def __init__(self, target_stress: float = 0.0, threshold: float = 0.2):
        self.target_stress = target_stress
        self.threshold = threshold

    def compute_reward(self, state: WorldState, **kwargs) -> torch.Tensor:
        # Mean stress across the world: 0.0 is ideal
        mean_stress = state.stress.mean(dim=(1, 2, 3, 4))
        # Closer to target_stress is better
        dist = torch.abs(mean_stress - self.target_stress)
        return torch.exp(-2.0 * dist)

    def is_completed(self, state: WorldState) -> torch.Tensor:
        mean_stress = state.stress.mean(dim=(1, 2, 3, 4))
        return mean_stress < self.threshold

class EvasionObjective(Objective):
    """Maximize distance from a threat agent."""
    
    def __init__(self, safe_distance: float = 0.5):
        self.safe_distance = safe_distance

    def compute_reward(self, state: WorldState, threat_pos: torch.Tensor | None = None) -> torch.Tensor:
        if threat_pos is None:
            return torch.ones(state.life.size(0), device=state.life.device)
        
        dist = torch.norm(state.object_pos - threat_pos, dim=1)
        # Higher reward for larger distance, capped at 1.0
        reward = torch.clamp(dist / self.safe_distance, 0.0, 1.0)
        return reward

    def is_completed(self, state: WorldState, threat_pos: torch.Tensor | None = None) -> torch.Tensor:
        if threat_pos is None:
            return torch.ones(state.life.size(0), dtype=torch.bool, device=state.life.device)
        
        dist = torch.norm(state.object_pos - threat_pos, dim=1)
        return dist >= self.safe_distance

class CombinedObjective(Objective):
    """Blends multiple objectives with specific weights."""
    
    def __init__(self, objectives: list[tuple[Objective, float]]):
        self.objectives = objectives # List of (Objective, Weight)

    def compute_reward(self, state: WorldState, **kwargs) -> torch.Tensor:
        total_reward = torch.zeros(state.life.size(0), device=state.life.device)
        for obj, weight in self.objectives:
            total_reward += weight * obj.compute_reward(state, **kwargs)
        return total_reward

    def is_completed(self, state: WorldState, **kwargs) -> torch.Tensor:
        # Combined objective is "completed" if all sub-objectives are completed
        completed = torch.ones(state.life.size(0), dtype=torch.bool, device=state.life.device)
        for obj, _ in self.objectives:
            completed &= obj.is_completed(state, **kwargs)
        return completed
