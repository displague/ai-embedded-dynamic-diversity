from __future__ import annotations

import pytest

from ai_embedded_dynamic_diversity.sim.viz_cli import (
    _scenario_viz_overrides,
    _resolve_storyboard_embodiments,
    _resolve_storyboard_scenarios,
    _select_storyboard_checkpoints,
)


def test_resolve_storyboard_embodiments_fallback_and_dedup() -> None:
    fallback = ["hexapod", "car", "drone"]
    assert _resolve_storyboard_embodiments("", fallback=fallback) == fallback
    assert _resolve_storyboard_embodiments("car, drone,car", fallback=fallback) == ["car", "drone"]


def test_resolve_storyboard_scenarios_profile_and_override() -> None:
    assert _resolve_storyboard_scenarios("hardy", "") == ["storm", "crosswind", "blackout"]
    assert _resolve_storyboard_scenarios("calibrated_large_v1", "") == [
        "storm",
        "crosswind",
        "blackout",
        "latency-storm",
        "friction-shift",
        "persistent-gust",
    ]
    assert _resolve_storyboard_scenarios("standard", "gust,force") == ["gust", "force"]
    with pytest.raises(ValueError):
        _resolve_storyboard_scenarios("unknown", "")


def test_scenario_viz_overrides_support_calibrated_large_scenarios() -> None:
    latency = _scenario_viz_overrides("latency-storm")
    friction = _scenario_viz_overrides("friction-shift")
    gust = _scenario_viz_overrides("persistent-gust")
    assert latency["wind_variation"] >= 0.5
    assert friction["force_y"] >= 0.9
    assert gust["force_mode"] == "continuous-blow"


def test_select_storyboard_checkpoints_clamps_top_k() -> None:
    ranked = [
        {"checkpoint": "a.pt"},
        {"checkpoint": "b.pt"},
        {"checkpoint": "c.pt"},
    ]
    assert _select_storyboard_checkpoints(ranked, top_k=2) == ["a.pt", "b.pt"]
    assert _select_storyboard_checkpoints(ranked, top_k=5) == ["a.pt", "b.pt", "c.pt"]
    with pytest.raises(ValueError):
        _select_storyboard_checkpoints(ranked, top_k=0)
