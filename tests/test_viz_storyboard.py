from __future__ import annotations

import pytest

from ai_embedded_dynamic_diversity.sim.viz_cli import (
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
    assert _resolve_storyboard_scenarios("standard", "gust,force") == ["gust", "force"]
    with pytest.raises(ValueError):
        _resolve_storyboard_scenarios("unknown", "")


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
