import torch
import pytest
from ai_embedded_dynamic_diversity.sim.mimicry import MimicryWorld
from ai_embedded_dynamic_diversity.sim.world import EnvironmentControls

def test_mimicry_world_signal_injection():
    world = MimicryWorld(10, 10, 10, 4)
    state = world.init(batch_size=2)
    
    # Run step to generate peer signals
    action_field = torch.zeros(2, 1000)
    controls = EnvironmentControls(
        wind=torch.zeros(2,3), 
        light_position=torch.zeros(2,3), 
        light_intensity=torch.ones(2,1), 
        force_position=torch.zeros(2,3), 
        force_vector=torch.zeros(2,3),
        force_strength=torch.zeros(2,1),
        force_active=torch.zeros(2,1),
        move_object_delta=torch.zeros(2,3)
    )
    state = world.step(state, action_field, controls)
    
    assert world.peer_signals is not None
    assert world.peer_signals.shape == (2, 8)
    
    # Check injection into observation
    labels = torch.zeros(2, dtype=torch.long)
    obs = world.encode_observation_with_mimicry(state, signal_dim=64, labels=labels)
    
    assert obs.shape == (2, 64)
    # Channels 20-28 should contain peer signal info
    # In a zero-signal world (zeros), this would be exactly 0.4 * peer_signals
    # Since we have resources and light, it will be mixed.
    assert not torch.all(obs[:, 20:28] == 0.0)
