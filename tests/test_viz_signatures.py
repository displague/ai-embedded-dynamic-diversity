from __future__ import annotations

from ai_embedded_dynamic_diversity.sim.viz_cli import _classify_adaptation_signature


def test_signature_detects_failure() -> None:
    result = {
        "mismatch_values": [0.10, 0.12, 0.14, 0.32, 0.36, 0.40, 0.39],
        "vitality_values": [0.22, 0.18, 0.12, 0.08, 0.04, 0.02, 0.01],
        "remap_steps": [3, 5],
    }
    sig = _classify_adaptation_signature(result)
    assert sig["label"] == "failure"
    assert bool(sig["vitality_collapse"]) is True
    assert bool(sig["mismatch_elevated"]) is True


def test_signature_detects_stable() -> None:
    result = {
        "mismatch_values": [0.16, 0.14, 0.12, 0.11, 0.10, 0.10, 0.09],
        "vitality_values": [0.18, 0.19, 0.20, 0.20, 0.21, 0.20, 0.21],
        "remap_steps": [3],
    }
    sig = _classify_adaptation_signature(result)
    assert sig["label"] == "stable"
    assert float(sig["severity"]) < 0.35
