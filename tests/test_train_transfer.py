from __future__ import annotations

import torch

from ai_embedded_dynamic_diversity.config import model_config_for_profile
from ai_embedded_dynamic_diversity.train.cli import (
    _build_transfer_states,
    _resolve_embodiments,
    _transfer_mismatch_loss,
)


def test_resolve_embodiments_deduplicates_and_validates() -> None:
    names = _resolve_embodiments("hexapod, car, polymorph120, car")
    assert names == ["hexapod", "car", "polymorph120"]


def test_build_transfer_states_matches_dims() -> None:
    mcfg = model_config_for_profile("pi5")
    names = ["hexapod", "car", "polymorph120"]
    states = _build_transfer_states(names, mcfg, torch.device("cpu"), seed=17)
    assert set(states.keys()) == set(names)
    assert states["hexapod"].projection.shape == (mcfg.signal_dim, 10)
    assert states["car"].mapping.shape == (mcfg.io_channels, 6)
    assert states["polymorph120"].mapping.shape == (mcfg.io_channels, 120)


def test_transfer_mismatch_loss_returns_scalar() -> None:
    mcfg = model_config_for_profile("pi5")
    states = _build_transfer_states(["hexapod", "car"], mcfg, torch.device("cpu"), seed=23)
    obs = torch.randn(4, mcfg.signal_dim)
    out_io = torch.randn(4, mcfg.io_channels)
    mismatch, mismatch_value, remap_events = _transfer_mismatch_loss(
        out_io=out_io,
        obs=obs,
        transfer_states=states,
        mcfg=mcfg,
        dev=torch.device("cpu"),
        sample_count=2,
        remap_probability=0.2,
    )
    assert mismatch.ndim == 0
    assert mismatch_value >= 0.0
    assert remap_events >= 0
