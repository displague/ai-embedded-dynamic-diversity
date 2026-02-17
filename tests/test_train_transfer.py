from __future__ import annotations

import torch
import typer
import pytest

from ai_embedded_dynamic_diversity.config import model_config_for_profile
from ai_embedded_dynamic_diversity.train.cli import (
    _apply_observation_noise,
    _build_transfer_states,
    _resolve_embodiments,
    _resolve_noise_profile,
    _transfer_mismatch_loss,
    choose_device,
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


def test_choose_device_cuda_strict_raises_when_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(typer.BadParameter):
        choose_device("cuda", strict=True)


def test_choose_device_cuda_non_strict_falls_back_to_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert choose_device("cuda", strict=False).type == "cpu"


def test_resolve_noise_profile_accepts_and_rejects() -> None:
    assert _resolve_noise_profile("none") == "none"
    assert _resolve_noise_profile("dropout-quant-v1") == "dropout-quant-v1"
    assert _resolve_noise_profile("dropout-quant-v2") == "dropout-quant-v2"
    with pytest.raises(ValueError):
        _resolve_noise_profile("bad-noise")


def test_apply_observation_noise_strength_controls_effect() -> None:
    obs = torch.linspace(-1.0, 1.0, 24).view(2, 12)
    out_none = _apply_observation_noise(obs, profile="dropout-quant-v1", seed=9, step=3, strength=0.0)
    out_noise = _apply_observation_noise(obs, profile="dropout-quant-v1", seed=9, step=3, strength=1.0)
    assert torch.allclose(out_none, obs)
    assert not torch.allclose(out_noise, obs)
