import torch
import pytest
from ai_embedded_dynamic_diversity.sim.world import DynamicDiversityWorld, WorldState
from ai_embedded_dynamic_diversity.sim.objectives import NavigationObjective, StabilityObjective, CombinedObjective, EvasionObjective

def test_navigation_objective_reward_increases_as_distance_decreases():
    # Setup world and objective
    world = DynamicDiversityWorld(10, 10, 10, 4)
    objective = NavigationObjective(target_pos=torch.tensor([[0.5, 0.5, 0.5]]))
    
    # State 1: Far from target
    state_far = world.init(batch_size=1)
    state_far.object_pos = torch.tensor([[0.0, 0.0, 0.0]])
    reward_far = objective.compute_reward(state_far)
    
    # State 2: Near target
    state_near = world.init(batch_size=1)
    state_near.object_pos = torch.tensor([[0.4, 0.4, 0.4]])
    reward_near = objective.compute_reward(state_near)
    
    assert reward_near > reward_far
    assert reward_near <= 1.0
    assert reward_far >= 0.0

def test_navigation_objective_completion_threshold():
    objective = NavigationObjective(target_pos=torch.tensor([[0.5, 0.5, 0.5]]), threshold=0.1)
    
    # Not completed
    state_far = WorldState(torch.zeros(1,1,1,1,1), torch.zeros(1,4,1,1,1), torch.zeros(1,1,1,1,1), 
                           torch.tensor([[0.0, 0.0, 0.0]]), torch.zeros(1,3))
    assert not objective.is_completed(state_far)
    
    # Completed
    state_near = WorldState(torch.zeros(1,1,1,1,1), torch.zeros(1,4,1,1,1), torch.zeros(1,1,1,1,1), 
                            torch.tensor([[0.45, 0.45, 0.45]]), torch.zeros(1,3))
    assert objective.is_completed(state_near)

def test_stability_objective_reward_increases_as_stress_decreases():
    world = DynamicDiversityWorld(10, 10, 10, 4)
    objective = StabilityObjective(target_stress=0.0)
    
    # State 1: High stress
    state_stressed = world.init(batch_size=1)
    state_stressed.stress = torch.full((1, 1, 10, 10, 10), 0.8)
    reward_stressed = objective.compute_reward(state_stressed)
    
    # State 2: Low stress
    state_calm = world.init(batch_size=1)
    state_calm.stress = torch.full((1, 1, 10, 10, 10), 0.1)
    reward_calm = objective.compute_reward(state_calm)
    
    assert reward_calm > reward_stressed

def test_combined_objective_blends_correctly():
    nav_obj = NavigationObjective(target_pos=torch.tensor([[0.5, 0.5, 0.5]]))
    stab_obj = StabilityObjective(target_stress=0.0)
    combined = CombinedObjective([(nav_obj, 0.5), (stab_obj, 0.5)])
    
    state = WorldState(torch.zeros(1,1,1,1,1), torch.zeros(1,4,1,1,1), torch.zeros(1,1,1,1,1), 
                       torch.tensor([[0.0, 0.0, 0.0]]), torch.zeros(1,3))
    state.stress = torch.full((1, 1, 1, 1, 1), 0.5)
    
    r_nav = nav_obj.compute_reward(state)
    r_stab = stab_obj.compute_reward(state)
    r_combined = combined.compute_reward(state)
    
    assert torch.allclose(r_combined, 0.5 * r_nav + 0.5 * r_stab)

def test_evasion_objective_reward_increases_with_distance():
    objective = EvasionObjective(safe_distance=1.0)
    
    # State: object at origin
    state = WorldState(torch.zeros(1,1,1,1,1), torch.zeros(1,4,1,1,1), torch.zeros(1,1,1,1,1), 
                       torch.tensor([[0.0, 0.0, 0.0]]), torch.zeros(1,3))
    
    # Threat near
    threat_near = torch.tensor([[0.1, 0.0, 0.0]])
    reward_near = objective.compute_reward(state, threat_pos=threat_near)
    
    # Threat far
    threat_far = torch.tensor([[0.8, 0.0, 0.0]])
    reward_far = objective.compute_reward(state, threat_pos=threat_far)
    
    assert reward_far > reward_near
    assert reward_far <= 1.0
