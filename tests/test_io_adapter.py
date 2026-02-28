import torch
import pytest
from ai_embedded_dynamic_diversity.deploy.io_adapter import IOAdapter

def test_io_adapter_normalization():
    adapter = IOAdapter(signal_dim=4)
    # Raw hardware data (e.g. 0-255 or arbitrary ranges)
    raw_data = torch.tensor([128.0, 0.5, -10.0, 1000.0])
    
    # Normalize to [0, 1] range for ModelCore
    normalized = adapter.normalize(raw_data)
    assert normalized.shape == (1, 4)
    assert torch.all(normalized >= 0.0)
    assert torch.all(normalized <= 1.0)

def test_io_adapter_denormalization():
    adapter = IOAdapter(signal_dim=4)
    # Model output in [0, 1]
    model_output = torch.tensor([[0.5, 0.1, 0.9, 0.0]])
    
    # Denormalize to hardware ranges (e.g. PWM 0-100)
    raw_output = adapter.denormalize(model_output, target_range=(0, 100))
    assert raw_output.shape == (4,)
    assert torch.all(raw_output >= 0.0)
    assert torch.all(raw_output <= 100.0)
    assert raw_output[0] == 50.0
