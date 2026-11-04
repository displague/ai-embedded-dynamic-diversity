"""The plain-network baseline (issue #51): same interface as the core, about the same size, starts at the standing
pose, and saves and loads through the same policy files as the core."""
from __future__ import annotations

import torch

from ai_embedded_dynamic_diversity.rl.actor import CoreActor, PlainMLP, build_policy_net


def test_size_and_start():
    net = PlainMLP(104, 32)
    n = sum(p.numel() for p in net.parameters())
    assert 50_000 < n < 53_000                  # the grown core has 51,484
    actor = CoreActor(net)
    mem = actor.initial_state(4, "cpu")
    mean, mem2 = actor(torch.randn(4, 104), mem)
    assert mean.shape == (4, 32)
    assert mean.abs().max() < 0.1               # starts near the standing pose
    assert torch.equal(mem, mem2)


def test_saved_roundtrip():
    net = PlainMLP(104, 32)
    saved = dict(arch="mlp", core=net.state_dict(),
                 core_config=dict(signal_dim=104, io_channels=32, hidden=net.hidden))
    net2 = build_policy_net(saved)
    x = torch.randn(3, 104)
    m = torch.zeros(3, 1, 1)
    assert torch.allclose(net(x, m)["io"], net2(x, m)["io"])
