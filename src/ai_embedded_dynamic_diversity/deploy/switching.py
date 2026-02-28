from __future__ import annotations

import torch
from ai_embedded_dynamic_diversity.models import ModelCore
from ai_embedded_dynamic_diversity.models.memory_bank import GeneticMemoryBank

class EmbodimentSwitcher:
    """Manages runtime embodiment switching and memory handoff policies."""
    
    def __init__(self, model: ModelCore, memory_bank: GeneticMemoryBank | None = None):
        self.model = model
        self.bank = memory_bank

    def switch(
        self, 
        target_embodiment: str, 
        current_memory: torch.Tensor | None = None, 
        mode: str = "stateless"
    ) -> torch.Tensor:
        """
        Executes an embodiment switch and returns the next memory state.
        
        Modes:
        - stateless: reset memory to zeros.
        - bank: retrieve prior for target embodiment from GeneticMemoryBank.
        - handoff: preserve current memory state during the switch.
        """
        if mode == "stateless":
            # Reset to zeros using model's stored dimensions
            return self.model.init_memory(1, self.model.memory_slots, self.model.memory_dim, "cpu")
            
        if mode == "bank":
            if self.bank:
                retrieved = self.bank.retrieve(target_embodiment)
                if retrieved is not None:
                    return retrieved
            # Fallback to stateless if not found
            return self.model.init_memory(1, self.model.memory_slots, self.model.memory_dim, "cpu")
            
        if mode == "handoff":
            if current_memory is not None:
                return current_memory
            # Fallback if no current memory
            return self.model.init_memory(1, self.model.memory_slots, self.model.memory_dim, "cpu")
            
        raise ValueError(f"Unknown switch mode '{mode}'")
