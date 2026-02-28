import torch
import pytest
from torch import nn
from ai_embedded_dynamic_diversity.models import ModelCore
from ai_embedded_dynamic_diversity.train.quantization import prepare_qat_model

def test_qat_preparation_scaffold():
    model = ModelCore(
        signal_dim=16,
        hidden_dim=32,
        edge_nodes=10,
        memory_slots=8,
        memory_dim=16,
        io_channels=4,
        max_remap_groups=4
    )
    
    prepared = prepare_qat_model(model)
    
    # Run forward pass to ensure functional integrity
    signal = torch.randn(1, 16)
    memory = model.init_memory(1, 8, 16, "cpu")
    out = prepared(signal, memory)
    assert "io" in out

def test_dynamic_quantization_functional():
    # Verify that the model remains functional after dynamic quantization attempt
    model = ModelCore(
        signal_dim=16,
        hidden_dim=32,
        edge_nodes=10,
        memory_slots=8,
        memory_dim=16,
        io_channels=4,
        max_remap_groups=4
    )
    
    # This might not change types on all platforms (e.g. windows without backend)
    # but should not crash and should return a functional model.
    quantized = torch.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
    
    signal = torch.randn(1, 16)
    memory = model.init_memory(1, 8, 16, "cpu")
    out = quantized(signal, memory)
    assert "io" in out
