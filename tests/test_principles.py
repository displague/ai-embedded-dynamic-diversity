from __future__ import annotations

import torch

from ai_embedded_dynamic_diversity.models import ModelCore
from ai_embedded_dynamic_diversity.sim.world import DynamicDiversityWorld


def _tiny_model(**overrides) -> ModelCore:
    cfg = {
        "signal_dim": 16,
        "hidden_dim": 24,
        "edge_nodes": 20,
        "memory_slots": 8,
        "memory_dim": 12,
        "io_channels": 6,
        "max_remap_groups": 4,
        "gating_mode": "sigmoid",
        "topk_gating": 0,
        "enable_dmd_gating": False,
        "enable_phase_gating": False,
    }
    cfg.update(overrides)
    torch.manual_seed(11)
    return ModelCore(**cfg)


def test_memory_updates_from_zero_state() -> None:
    model = _tiny_model()
    signal = torch.randn(2, 16)
    memory = torch.zeros(2, 8, 12)
    remap = torch.zeros(2, 4)

    out = model(signal, memory, remap)
    assert out["memory"].shape == (2, 8, 12)
    assert torch.norm(out["memory"]).item() > 0.0


def test_interconnected_memory_influences_io() -> None:
    model = _tiny_model()
    signal = torch.randn(1, 16)
    remap = torch.zeros(1, 4)
    memory0 = torch.zeros(1, 8, 12)

    out1 = model(signal, memory0, remap)
    out2 = model(signal, out1["memory"].detach(), remap)

    # Primitive interconnected behavior: same input, updated memory alters output pathway.
    delta = torch.mean(torch.abs(out2["io"] - out1["io"])).item()
    assert delta > 1e-7


def test_remap_code_changes_output_projection() -> None:
    model = _tiny_model()
    signal = torch.randn(1, 16)
    memory = torch.zeros(1, 8, 12)

    remap_a = torch.zeros(1, 4)
    remap_b = torch.zeros(1, 4)
    remap_b[:, 2] = 1.0

    io_a = model(signal, memory, remap_a)["io"]
    io_b = model(signal, memory, remap_b)["io"]

    shift = torch.mean(torch.abs(io_a - io_b)).item()
    assert shift > 1e-7


def test_topk_gating_enforces_sparse_memory_paging() -> None:
    model = _tiny_model(gating_mode="symplectic", topk_gating=3, enable_dmd_gating=True, enable_phase_gating=True)
    signal = torch.randn(3, 16)
    memory = torch.zeros(3, 8, 12)
    remap = torch.zeros(3, 4)

    weights = model(signal, memory, remap)["memory_weights"]
    nonzero = (weights > 1e-8).sum(dim=-1)
    assert torch.all(nonzero <= 3)


def test_world_controls_affect_stress_and_object_motion() -> None:
    torch.manual_seed(5)
    world = DynamicDiversityWorld(x=8, y=8, z=4, resource_channels=3, device="cpu")
    state = world.init(batch_size=1)
    action = torch.zeros(1, 8 * 8 * 4)

    neutral = world.default_controls(batch_size=1)
    stressed = world.default_controls(batch_size=1)
    stressed.wind[:] = torch.tensor([[1.0, 0.0, 0.0]])
    stressed.force_active[:] = 1.0
    stressed.force_strength[:] = 1.0
    stressed.force_vector[:] = torch.tensor([[0.7, 0.0, 0.0]])

    s_neutral = world.step(state, action, controls=neutral)
    s_stressed = world.step(state, action, controls=stressed)

    assert s_stressed.stress.mean().item() > s_neutral.stress.mean().item()
    assert torch.norm(s_stressed.object_pos - s_neutral.object_pos).item() > 0.0
