from __future__ import annotations

from dataclasses import dataclass

from ai_embedded_dynamic_diversity.sim.embodiments import Embodiment


@dataclass(frozen=True)
class HumanoidComplianceProfile:
    name: str
    required_embodiment: str
    min_overall_score: float
    expected_mass_distribution: dict[str, float]
    expected_structural_composition: dict[str, float]
    expected_control_bands: dict[str, tuple[int, int]]
    required_sensors: tuple[str, ...]


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def resolve_humanoid_compliance_profile(profile_name: str) -> HumanoidComplianceProfile:
    normalized = profile_name.strip().lower() or "human_rigid_v1"
    if normalized == "human_rigid_v1":
        return HumanoidComplianceProfile(
            name="human_rigid_v1",
            required_embodiment="humanoid120",
            min_overall_score=0.72,
            expected_mass_distribution={
                "lower_body": 0.48,
                "upper_body": 0.42,
                "head_neck": 0.10,
            },
            expected_structural_composition={
                "rigid_frame": 0.58,
                "compliant_tissue": 0.27,
                "actuation_bundle": 0.15,
            },
            expected_control_bands={
                "humanoid_leg_joint": (34, 40),
                "humanoid_arm_joint": (30, 36),
                "humanoid_spine_joint": (14, 18),
                "humanoid_hand_joint": (22, 26),
                "humanoid_neck_head_joint": (10, 14),
            },
            required_sensors=(
                "vision",
                "stereo_audio",
                "imu",
                "pressure",
                "strain",
            ),
        )
    if normalized == "human_rigid_relaxed_v1":
        strict = resolve_humanoid_compliance_profile("human_rigid_v1")
        return HumanoidComplianceProfile(
            name="human_rigid_relaxed_v1",
            required_embodiment=strict.required_embodiment,
            min_overall_score=0.64,
            expected_mass_distribution=strict.expected_mass_distribution,
            expected_structural_composition=strict.expected_structural_composition,
            expected_control_bands={
                "humanoid_leg_joint": (32, 42),
                "humanoid_arm_joint": (28, 38),
                "humanoid_spine_joint": (12, 20),
                "humanoid_hand_joint": (20, 28),
                "humanoid_neck_head_joint": (8, 16),
            },
            required_sensors=strict.required_sensors,
        )
    raise ValueError(
        f"Unknown humanoid compliance profile '{profile_name}'. "
        "Allowed: human_rigid_v1, human_rigid_relaxed_v1"
    )


def _control_band_counts(embodiment: Embodiment) -> dict[str, int]:
    counts = {
        "humanoid_leg_joint": 0,
        "humanoid_arm_joint": 0,
        "humanoid_spine_joint": 0,
        "humanoid_hand_joint": 0,
        "humanoid_neck_head_joint": 0,
    }
    for name in embodiment.controls:
        for prefix in counts:
            if name.startswith(prefix):
                counts[prefix] += 1
                break
    return counts


def _band_compliance_score(bands: dict[str, int], expected: dict[str, tuple[int, int]]) -> float:
    scores: list[float] = []
    for key, (low, high) in expected.items():
        val = int(bands.get(key, 0))
        if low <= val <= high:
            scores.append(1.0)
        elif val < low:
            scores.append(_clamp01(1.0 - ((low - val) / max(1, low))))
        else:
            scores.append(_clamp01(1.0 - ((val - high) / max(1, high))))
    return sum(scores) / max(1, len(scores))


def _sensor_compliance_score(embodiment: Embodiment, required: tuple[str, ...]) -> float:
    present = set(embodiment.sensors)
    hit = sum(1 for s in required if s in present)
    return hit / max(1, len(required))


def _distribution_distance_score(reference: dict[str, float], realized: dict[str, float]) -> float:
    keys = set(reference).union(realized)
    l1 = sum(abs(float(reference.get(k, 0.0)) - float(realized.get(k, 0.0))) for k in keys)
    return _clamp01(1.0 - 0.5 * l1)


def evaluate_humanoid_compliance(
    embodiment: Embodiment,
    profile: HumanoidComplianceProfile,
    *,
    mean_mismatch: float,
    mean_vitality: float,
    recovery: float,
    autopoiesis_score: float,
    mass_distribution: dict[str, float] | None = None,
    structural_composition: dict[str, float] | None = None,
) -> dict[str, float | bool | str | dict[str, float] | dict[str, int]]:
    band_counts = _control_band_counts(embodiment)
    band_score = _band_compliance_score(band_counts, profile.expected_control_bands)
    sensor_score = _sensor_compliance_score(embodiment, profile.required_sensors)

    realized_mass = mass_distribution or profile.expected_mass_distribution
    realized_comp = structural_composition or profile.expected_structural_composition
    mass_score = _distribution_distance_score(profile.expected_mass_distribution, realized_mass)
    composition_score = _distribution_distance_score(profile.expected_structural_composition, realized_comp)

    stability_score = _clamp01(
        0.35 * (1.0 / (1.0 + float(mean_mismatch)))
        + 0.25 * _clamp01(float(mean_vitality))
        + 0.20 * _clamp01(float(recovery))
        + 0.20 * _clamp01(float(autopoiesis_score))
    )
    score = (
        0.24 * band_score
        + 0.18 * sensor_score
        + 0.18 * mass_score
        + 0.18 * composition_score
        + 0.22 * stability_score
    )
    embodiment_matches = embodiment.name.lower() == profile.required_embodiment.lower()
    pass_gate = embodiment_matches and score >= profile.min_overall_score

    return {
        "profile": profile.name,
        "required_embodiment": profile.required_embodiment,
        "evaluated_embodiment": embodiment.name,
        "embodiment_matches_required": embodiment_matches,
        "overall_score": float(score),
        "min_required_score": float(profile.min_overall_score),
        "pass": bool(pass_gate),
        "components": {
            "control_band_score": float(band_score),
            "sensor_score": float(sensor_score),
            "mass_distribution_score": float(mass_score),
            "structural_composition_score": float(composition_score),
            "stability_score": float(stability_score),
        },
        "control_band_counts": band_counts,
    }
