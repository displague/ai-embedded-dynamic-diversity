from __future__ import annotations


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def repair_response_score(mismatch: list[float], remap_steps: list[int], window: int = 5) -> float:
    if not mismatch or not remap_steps:
        return 0.0
    n = len(mismatch)
    scores: list[float] = []
    for step in remap_steps:
        post_start = step
        post_end = min(step + window, n)
        late_start = min(step + window, n)
        late_end = min(step + 2 * window, n)
        if post_start >= post_end or late_start >= late_end:
            continue
        post = sum(mismatch[post_start:post_end]) / max(1, post_end - post_start)
        late = sum(mismatch[late_start:late_end]) / max(1, late_end - late_start)
        if post <= 1e-8:
            continue
        scores.append(_clamp01((post - late) / post))
    if not scores:
        return 0.0
    return sum(scores) / len(scores)


def autopoietic_metrics(
    mismatch_values: list[float],
    vitality_values: list[float],
    stress_values: list[float],
    energy_values: list[float],
    remap_steps: list[int],
    resource_values: list[float] | None = None,
) -> dict[str, float]:
    mean_mismatch = sum(mismatch_values) / max(1, len(mismatch_values))
    mean_vitality = sum(vitality_values) / max(1, len(vitality_values))
    mean_stress = sum(stress_values) / max(1, len(stress_values))
    mean_energy = sum(energy_values) / max(1, len(energy_values))
    mean_resource = (
        sum(float(x) for x in resource_values) / max(1, len(resource_values))
        if resource_values
        else 0.0
    )

    closure_resilience = _clamp01((1.0 / (1.0 + mean_mismatch)) * (0.6 + 0.4 * mean_vitality))
    self_repair_response = repair_response_score(mismatch_values, remap_steps, window=5)

    if len(mismatch_values) >= 2:
        diffs = [abs(mismatch_values[i] - mismatch_values[i - 1]) for i in range(1, len(mismatch_values))]
        smoothness = 1.0 / (1.0 + (sum(diffs) / max(1, len(diffs))))
    else:
        smoothness = 0.0
    organizational_persistence = _clamp01(0.65 * smoothness + 0.35 * (1.0 - min(1.0, mean_stress)))

    resource_cycle_efficiency = _clamp01(mean_vitality / (mean_energy + mean_stress + 1e-6))
    if resource_values:
        resource_cycle_efficiency = _clamp01(0.7 * resource_cycle_efficiency + 0.3 * mean_resource)

    autopoietic_score = _clamp01(
        0.35 * closure_resilience
        + 0.30 * self_repair_response
        + 0.20 * organizational_persistence
        + 0.15 * resource_cycle_efficiency
    )

    return {
        "closure_resilience": closure_resilience,
        "organizational_persistence": organizational_persistence,
        "self_repair_response": self_repair_response,
        "resource_cycle_efficiency": resource_cycle_efficiency,
        "autopoietic_score": autopoietic_score,
    }
