import torch
import pytest
from ai_embedded_dynamic_diversity.models.memory_bank import GeneticMemoryBank

def test_memory_bank_store_and_retrieve():
    bank = GeneticMemoryBank()
    memory = torch.randn(1, 16, 64)
    bank.store("hexapod", memory)
    
    retrieved = bank.retrieve("hexapod")
    assert torch.equal(retrieved, memory)

def test_memory_bank_retrieve_non_existent():
    bank = GeneticMemoryBank()
    # Should return zeros or None? Let's say zeros based on dimensions
    # but we need dimensions. Let's make it return None and handle at caller.
    assert bank.retrieve("unknown") is None

def test_memory_bank_save_load(tmp_path):
    bank = GeneticMemoryBank()
    memory = torch.randn(1, 16, 64)
    bank.store("hexapod", memory)
    
    save_path = tmp_path / "bank.pt"
    bank.save(str(save_path))
    
    new_bank = GeneticMemoryBank()
    new_bank.load(str(save_path))
    
    assert torch.equal(new_bank.retrieve("hexapod"), memory)
