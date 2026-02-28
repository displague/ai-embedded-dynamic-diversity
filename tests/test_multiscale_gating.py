import torch
import pytest
from ai_embedded_dynamic_diversity.models import ModelCore

def test_model_core_with_multi_scale_gating():
    # Setup model config with multi-scale gating enabled
    model = ModelCore(
        signal_dim=16,
        hidden_dim=32,
        edge_nodes=10,
        memory_slots=8,
        memory_dim=16,
        io_channels=4,
        max_remap_groups=4,
        enable_multi_scale_gating=True
    )
    
    assert model.enable_multi_scale_gating is True
    assert hasattr(model, "global_gate_proj")
    assert hasattr(model, "local_gate_proj")
    
    # Run forward pass
    signal = torch.randn(1, 16)
    memory = model.init_memory(1, 8, 16, "cpu")
    out = model(signal, memory)
    
    assert "readiness" in out
    assert out["readiness"].shape == (1, 10)
    # Check that readiness is within [0, 1]
    assert torch.all(out["readiness"] >= 0.0)
    assert torch.all(out["readiness"] <= 1.0)

def test_model_core_without_multi_scale_gating():
    # Setup model config with multi-scale gating disabled
    model = ModelCore(
        signal_dim=16,
        hidden_dim=32,
        edge_nodes=10,
        memory_slots=8,
        memory_dim=16,
        io_channels=4,
        max_remap_groups=4,
        enable_multi_scale_gating=False
    )
    
    assert model.enable_multi_scale_gating is False
    assert not hasattr(model, "global_gate_proj")
    assert not hasattr(model, "local_gate_proj")
