import torch
import pytest
from ai_embedded_dynamic_diversity.models import ModelCore
from ai_embedded_dynamic_diversity.models.memory_bank import GeneticMemoryBank
from ai_embedded_dynamic_diversity.deploy.switching import EmbodimentSwitcher

def test_embodiment_switcher_stateless_reset():
    model = ModelCore(10, 4, 32, 10, 16, 64, 4)
    bank = GeneticMemoryBank()
    switcher = EmbodimentSwitcher(model, bank)
    
    # Run some steps to populate memory
    memory = model.init_memory(1, 16, 64, "cpu")
    out = model(torch.randn(1, 10), memory)
    active_memory = out["memory"]
    
    # Switch with stateless reset (default)
    new_memory = switcher.switch("hexapod", current_memory=active_memory, mode="stateless")
    
    # Should be zeros if bank is empty
    assert torch.all(new_memory == 0.0)

def test_embodiment_switcher_with_bank_retrieval():
    model = ModelCore(10, 4, 32, 10, 16, 64, 4)
    bank = GeneticMemoryBank()
    saved_mem = torch.randn(1, 16, 64)
    bank.store("hexapod", saved_mem)
    
    switcher = EmbodimentSwitcher(model, bank)
    
    # Switch should retrieve from bank
    new_memory = switcher.switch("hexapod", mode="bank")
    assert torch.equal(new_memory, saved_mem)

def test_embodiment_switcher_handoff():
    model = ModelCore(10, 4, 32, 10, 16, 64, 4)
    bank = GeneticMemoryBank()
    switcher = EmbodimentSwitcher(model, bank)
    
    current_mem = torch.randn(1, 16, 64)
    
    # Handoff mode should preserve current memory even if switching embodiment
    new_memory = switcher.switch("car", current_memory=current_mem, mode="handoff")
    assert torch.equal(new_memory, current_mem)
