import torch
import pytest
from ai_embedded_dynamic_diversity.models.constructor_tape import load_constructor_tape, config_from_constructor_tape
from ai_embedded_dynamic_diversity.models.universal_constructor import UniversalConstructor

def test_v7_tape_construction():
    tape_path = "artifacts/model-core-v07.tape.json"
    tape = load_constructor_tape(tape_path)
    
    # Verify new fields are present in tokens
    token_keys = [t.split("=")[0] for t in tape.tokens]
    assert "enable_multi_scale_gating" in token_keys
    assert "emergent_signal_dim" in token_keys
    
    # Build model from tape
    constructor = UniversalConstructor(base_profile="pi5")
    constructed = constructor.build(tape)
    
    assert constructed.model.enable_multi_scale_gating is True
    assert constructed.config.emergent_signal_dim == 8
    
    # Verify forward pass
    signal = torch.randn(1, constructed.config.signal_dim)
    memory = constructed.model.init_memory(1, constructed.config.memory_slots, constructed.config.memory_dim, "cpu")
    out = constructed.model(signal, memory)
    
    assert "emergent_signal" in out
    assert out["emergent_signal"].shape == (1, 8)
