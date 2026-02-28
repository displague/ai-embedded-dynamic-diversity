import torch
import pytest
from ai_embedded_dynamic_diversity.sim.signaling import SignalingWorld
from ai_embedded_dynamic_diversity.models import ModelCore
from ai_embedded_dynamic_diversity.train.losses import loss_fn

def test_signaling_world_injection():
    world = SignalingWorld(10, 10, 10, 4)
    batch_size = 16
    labels = world.inject_signals(batch_size, p_peer=0.3, p_env=0.3, p_threat=0.3)
    
    assert labels.shape == (batch_size,)
    assert labels.dtype == torch.long
    assert torch.all(labels >= 0)
    assert torch.all(labels <= 3)
    
    state = world.init(batch_size)
    obs = world.encode_observation_with_signals(state, signal_dim=64, labels=labels)
    assert obs.shape == (batch_size, 64)
    assert torch.all(obs >= 0.0)
    assert torch.all(obs <= 1.0)

def test_model_signal_detection_forward():
    model = ModelCore(
        signal_dim=16,
        hidden_dim=32,
        edge_nodes=10,
        memory_slots=8,
        memory_dim=16,
        io_channels=4,
        max_remap_groups=4
    )
    
    signal = torch.randn(1, 16)
    memory = model.init_memory(1, 8, 16, "cpu")
    out = model(signal, memory)
    
    assert "predicted_signal_type" in out
    assert out["predicted_signal_type"].shape == (1, 4)

def test_signal_detection_loss():
    # Outputs with 4 classes
    outputs = {
        "io": torch.randn(2, 4),
        "readiness": torch.rand(2, 10),
        "energy": torch.rand(2, 1),
        "memory_weights": torch.rand(2, 8),
        "memory": torch.randn(2, 8, 16),
        "predicted_remap": torch.rand(2, 4),
        "predicted_signal_type": torch.randn(2, 4)
    }
    target_signal = torch.randn(2, 4)
    target_signal_type = torch.tensor([1, 3], dtype=torch.long)
    
    total, logs = loss_fn(
        outputs,
        target_signal,
        entropy_weight=0.1,
        energy_weight=0.1,
        memory_consistency_weight=0.1,
        detection_loss_weight=1.0,
        target_signal_type=target_signal_type
    )
    
    assert "detection_loss" in logs
    assert logs["detection_loss"] >= 0.0
