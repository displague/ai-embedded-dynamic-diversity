from __future__ import annotations

import torch

from ai_embedded_dynamic_diversity.config import world_config_for_profile
from ai_embedded_dynamic_diversity.sim.world import DynamicDiversityWorld


def test_world_config_large_profile_has_realism_knobs() -> None:
    cfg = world_config_for_profile("large_v1")
    assert cfg.x > 20
    assert cfg.y > 20
    assert cfg.actuation_delay_steps >= 1
    assert cfg.sensor_latency_steps >= 1
    assert cfg.surface_friction_scale < 1.0


def test_world_sensor_latency_delays_observation() -> None:
    world = DynamicDiversityWorld(
        8,
        8,
        4,
        3,
        sensor_latency_steps=1,
        device="cpu",
    )
    state = world.init(batch_size=1)
    obs0 = world.encode_observation(state, signal_dim=16)
    obs1 = world.encode_observation(state, signal_dim=16)
    assert torch.allclose(obs0, torch.zeros_like(obs0))
    assert not torch.allclose(obs1, torch.zeros_like(obs1))
