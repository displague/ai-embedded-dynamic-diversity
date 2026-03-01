from __future__ import annotations

from ai_embedded_dynamic_diversity.sim.embodiments import get_embodiment
from ai_embedded_dynamic_diversity.sim.humanoid_compliance import (
    evaluate_humanoid_compliance,
    resolve_humanoid_compliance_profile,
)


def test_resolve_humanoid_compliance_profile_known_profiles() -> None:
    strict = resolve_humanoid_compliance_profile("human_rigid_v1")
    relaxed = resolve_humanoid_compliance_profile("human_rigid_relaxed_v1")
    assert strict.required_embodiment == "humanoid120"
    assert strict.min_overall_score > relaxed.min_overall_score


def test_humanoid120_compliance_evaluates_and_passes_for_reasonable_stats() -> None:
    emb = get_embodiment("humanoid120")
    profile = resolve_humanoid_compliance_profile("human_rigid_v1")
    report = evaluate_humanoid_compliance(
        emb,
        profile,
        mean_mismatch=0.32,
        mean_vitality=0.69,
        recovery=0.74,
        autopoiesis_score=0.66,
    )
    assert report["evaluated_embodiment"] == "humanoid120"
    assert float(report["overall_score"]) >= 0.0
    assert isinstance(report["components"], dict)
    assert bool(report["pass"]) is True


def test_non_humanoid_fails_required_embodiment_gate() -> None:
    emb = get_embodiment("polymorph120")
    profile = resolve_humanoid_compliance_profile("human_rigid_v1")
    report = evaluate_humanoid_compliance(
        emb,
        profile,
        mean_mismatch=0.25,
        mean_vitality=0.75,
        recovery=0.80,
        autopoiesis_score=0.70,
    )
    assert bool(report["embodiment_matches_required"]) is False
    assert bool(report["pass"]) is False
