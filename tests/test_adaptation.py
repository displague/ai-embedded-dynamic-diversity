import torch
import pytest
from ai_embedded_dynamic_diversity.models import ModelCore
from ai_embedded_dynamic_diversity.train.adaptation import FewShotAdaptor
from ai_embedded_dynamic_diversity.train.losses import loss_fn

def test_few_shot_adaptation_reduces_loss():
    # Setup model
    model = ModelCore(
        edge_nodes=10, 
        io_channels=4, 
        hidden_dim=32, 
        signal_dim=10, 
        memory_slots=16, 
        memory_dim=64, 
        max_remap_groups=4
    )
    
    # Create dummy calibration data (1 sample)
    obs = torch.randn(1, 10)
    target = torch.randn(1, 4)
    calibration_data = [(obs, target)]
    
    # Initial loss
    memory = model.init_memory(1, 16, 64, "cpu")
    with torch.no_grad():
        out_initial = model(obs, memory)
        loss_initial, _ = loss_fn(out_initial, target, 0.01, 0.01, 0.01)
    
    # Adapt
    adaptor = FewShotAdaptor(lr=1e-2, steps=5)
    adapted_model = adaptor.adapt(model, "car", calibration_data)
    
    # Adapted loss
    with torch.no_grad():
        out_adapted = adapted_model(obs, memory)
        loss_adapted, _ = loss_fn(out_adapted, target, 0.01, 0.01, 0.01)
    
    assert loss_adapted < loss_initial
    print(f"Initial loss: {loss_initial:.6f}, Adapted loss: {loss_adapted:.6f}")
