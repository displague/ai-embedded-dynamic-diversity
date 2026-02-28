from __future__ import annotations

import torch
from pathlib import Path

class GeneticMemoryBank:
    """Stores and retrieves genetic memory tensors across morphology families."""
    
    def __init__(self):
        self.bank: dict[str, torch.Tensor] = {}

    def store(self, key: str, memory: torch.Tensor) -> None:
        """Stores a genetic memory tensor."""
        self.bank[key.lower()] = memory.detach().cpu()

    def retrieve(self, key: str) -> torch.Tensor | None:
        """Retrieves a genetic memory tensor if it exists."""
        return self.bank.get(key.lower())

    def save(self, path: str) -> None:
        """Saves the memory bank to a file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.bank, path)

    def load(self, path: str) -> None:
        """Loads the memory bank from a file."""
        if Path(path).exists():
            self.bank = torch.load(path, map_location="cpu")
