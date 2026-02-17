from __future__ import annotations

import pytest

from ai_embedded_dynamic_diversity.sim.cli import profile_embodiment_metrics


def test_profile_embodiment_metrics_polymorph_has_expected_sections() -> None:
    payload = profile_embodiment_metrics(
        embodiment="polymorph120",
        profile="pi5",
        weights="",
        steps=4,
        batch_size=2,
        remap_every=2,
        world_x=8,
        world_y=8,
        world_z=4,
        resource_channels=3,
        env_volatility=0.2,
        firing_threshold=0.2,
        readiness_active_threshold=0.2,
        device="cpu",
        seed=11,
    )

    assert payload["embodiment"]["name"] == "polymorph120"
    assert payload["embodiment"]["control_dof"] == 120
    assert "model" in payload
    assert "runtime" in payload
    assert "metrics" in payload
    assert "io_profile" in payload

    metrics = payload["metrics"]
    assert 0.0 <= float(metrics["channel_firing_fraction"]) <= 1.0
    assert 0.0 <= float(metrics["readiness_sparsity"]) <= 1.0
    assert 0.0 <= float(metrics["readiness_saturation_low"]) <= 1.0
    assert 0.0 <= float(metrics["readiness_saturation_high"]) <= 1.0
    assert 0.0 <= float(metrics["memory_weight_entropy"]) <= 1.0
    assert float(metrics["mean_mismatch"]) >= 0.0


def test_profile_embodiment_metrics_rejects_invalid_steps() -> None:
    with pytest.raises(ValueError):
        profile_embodiment_metrics(
            embodiment="hexapod",
            profile="pi5",
            weights="",
            steps=0,
            batch_size=2,
            remap_every=2,
            world_x=8,
            world_y=8,
            world_z=4,
            resource_channels=3,
            env_volatility=0.2,
            firing_threshold=0.2,
            readiness_active_threshold=0.2,
            device="cpu",
            seed=11,
        )
