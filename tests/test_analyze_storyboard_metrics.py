from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_analyzer():
    script_path = Path("scripts/analyze_storyboard_metrics.py")
    spec = importlib.util.spec_from_file_location("analyze_storyboard_metrics", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_split_scenario_embodiment_preserves_hyphenated_scenarios() -> None:
    analyzer = _load_analyzer()

    assert analyzer._split_scenario_embodiment("latency-storm-car") == ("latency-storm", "car")
    assert analyzer._split_scenario_embodiment("storm-hexapod") == ("storm", "hexapod")
    assert analyzer._split_scenario_embodiment("blackout") == ("blackout", "unknown")
