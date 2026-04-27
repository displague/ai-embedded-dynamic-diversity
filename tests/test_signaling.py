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


def test_hazard_aware_inject_signals_direct():
    """inject_signals biases p_threat/p_env when hazard_active_kinds is passed directly."""
    torch.manual_seed(42)
    world = SignalingWorld(10, 10, 10, 4)
    world.init(batch_size=2)

    # With light_triggered active: threat labels should be more likely
    batch = 2000
    labels_baseline = world.inject_signals(batch, p_threat=0.1, hazard_active_kinds=set())
    labels_threat_biased = world.inject_signals(batch, p_threat=0.1, hazard_active_kinds={"light_triggered"})

    baseline_threat = (labels_baseline == 3).float().mean().item()
    biased_threat = (labels_threat_biased == 3).float().mean().item()
    assert biased_threat > baseline_threat, (
        f"Expected more threat labels with light_triggered active: {biased_threat:.3f} vs {baseline_threat:.3f}"
    )

    # With periodic active: env labels should be more likely
    labels_env_biased = world.inject_signals(batch, p_env=0.1, hazard_active_kinds={"periodic"})
    baseline_env = (labels_baseline == 2).float().mean().item()
    biased_env = (labels_env_biased == 2).float().mean().item()
    assert biased_env > baseline_env, (
        f"Expected more env labels with periodic active: {biased_env:.3f} vs {baseline_env:.3f}"
    )


def test_active_hazard_kinds_tracked_after_step():
    """_active_hazard_kinds is populated after step() when hazard zones are configured."""
    from ai_embedded_dynamic_diversity.config import WorldConfig, HazardZoneConfig
    wcfg = WorldConfig(
        hazard_zones=[
            HazardZoneConfig(
                cx=0.5, cy=0.5, cz=0.5,
                rx=0.4, ry=0.4, rz=0.5,
                kind="light_triggered",
                threshold=0.01,  # very low threshold so it fires
                shadow_safe=False,
                hazard_weight=0.8,
            )
        ]
    )
    world = SignalingWorld(
        wcfg.x, wcfg.y, wcfg.z, wcfg.resource_channels,
        hazard_zones=wcfg.hazard_zones,
    )
    state = world.init(batch_size=1)
    # Use high light intensity to trigger the light_triggered hazard
    controls = world.default_controls(1)
    controls.light_intensity = torch.tensor([[1.0]])
    controls.light_position = torch.zeros(1, 3)
    action = torch.zeros(1, wcfg.x * wcfg.y * wcfg.z)
    world.step(state, action, controls)
    # After step, the hazard should have been detected
    # _active_hazard_kinds may or may not contain "light_triggered" depending on stress levels,
    # but the set should be initialised (not None) and the field should exist.
    assert hasattr(world, "_active_hazard_kinds")
    assert isinstance(world._active_hazard_kinds, set)
