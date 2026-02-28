import torch
import pytest
from ai_embedded_dynamic_diversity.sim.conjoining import ConjoiningWorld
from ai_embedded_dynamic_diversity.sim.world import EnvironmentControls

def test_conjoining_world_attachment_logic():
    world = ConjoiningWorld(10, 10, 10, 4)
    state = world.init(batch_size=1)
    
    # Place tool near object
    world.tool_pos = state.object_pos + torch.tensor([[0.05, 0.0, 0.0]])
    
    # Step should trigger attachment
    action_field = torch.zeros(1, 1000)
    controls = EnvironmentControls(
        wind=torch.zeros(1,3), 
        light_position=torch.zeros(1,3), 
        light_intensity=torch.zeros(1,1), 
        force_position=torch.zeros(1,3), 
        force_vector=torch.zeros(1,3),
        force_strength=torch.zeros(1,1),
        force_active=torch.zeros(1,1),
        move_object_delta=torch.zeros(1,3)
    )
    state = world.step(state, action_field, controls)
    
    assert world.is_attached[0] == True

def test_conjoining_world_move_with_attachment():
    world = ConjoiningWorld(10, 10, 10, 4)
    state = world.init(batch_size=1)
    world.is_attached = torch.tensor([True], device=world.device)
    world.tool_pos = state.object_pos.clone()
    
    old_obj_pos = state.object_pos.clone()
    # Apply move action (force simple movement by setting move_object_delta)
    action_field = torch.zeros(1, 1000)
    controls = EnvironmentControls(
        wind=torch.zeros(1,3), 
        light_position=torch.zeros(1,3), 
        light_intensity=torch.zeros(1,1), 
        force_position=torch.zeros(1,3), 
        force_vector=torch.zeros(1,3),
        force_strength=torch.zeros(1,1),
        force_active=torch.zeros(1,1),
        move_object_delta=torch.tensor([[0.1, 0.0, 0.0]])
    )
    state = world.step(state, action_field, controls)
    
    # Object moved
    assert not torch.equal(state.object_pos, old_obj_pos)
    # Tool moved with it
    assert torch.allclose(world.tool_pos, state.object_pos, atol=1e-2)
