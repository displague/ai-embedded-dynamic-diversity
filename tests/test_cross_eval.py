from __future__ import annotations

import pytest
import torch

from ai_embedded_dynamic_diversity.config import ModelConfig
from ai_embedded_dynamic_diversity.models import ModelCore
from ai_embedded_dynamic_diversity.train.cross_eval_cli import (
    ScenarioSpec,
    _apply_observation_noise,
    _binary_auc,
    _checkmate_metrics,
    _parse_embodiment_weights,
    _resolve_noise_profile,
    _resolve_subset_embodiments,
    _resolve_train_embodiments,
    _resolve_scenario_profile,
    _transfer_ratio_matrix,
    _weighted_transfer_score,
    compute_recovery_score,
    rollout_metrics,
)


def test_compute_recovery_score_detects_post_remap_improvement() -> None:
    mismatch = [0.2, 0.2, 0.2, 0.9, 0.8, 0.6, 0.5, 0.4, 0.35, 0.3]
    score = compute_recovery_score(mismatch, [3], window=3)
    assert 0.0 < score <= 1.0


def test_rollout_metrics_returns_transfer_keys() -> None:
    cfg = ModelConfig(
        signal_dim=16,
        hidden_dim=24,
        edge_nodes=20,
        memory_slots=8,
        memory_dim=12,
        io_channels=6,
        max_remap_groups=4,
    )
    model = ModelCore(**cfg.__dict__)
    scenario = ScenarioSpec(
        name="test",
        wind=(0.2, 0.0, 0.0),
        wind_variation=0.1,
        light_pos=(-0.2, 0.0, 0.2),
        light_drift=(0.0, 0.0, 0.0),
        light_intensity=0.6,
        force_vector=(0.5, 0.0, 0.0),
        force_strength=0.5,
        force_start=4,
        force_duration=6,
        force_pattern="pulse",
    )

    metrics = rollout_metrics(
        model=model,
        cfg=cfg,
        embodiment_name="hexapod",
        scenario=scenario,
        steps=16,
        remap_every=4,
        seed=5,
        world_dims=(8, 8, 4, 3),
        device=model.passive.edge_projection.weight.device,
    )

    assert "transfer_score" in metrics
    assert "mean_mismatch" in metrics
    assert "mean_vitality" in metrics
    assert metrics["remap_events"] >= 1


def test_hardy_profile_resolves_multiple_scenarios() -> None:
    scenarios = _resolve_scenario_profile("hardy")
    names = [s.name for s in scenarios]
    assert "storm" in names
    assert "blackout" in names
    assert len(names) >= 5


def test_parse_embodiment_weights_defaults_and_overrides() -> None:
    names = ["hexapod", "car", "drone"]
    defaults = _parse_embodiment_weights("", names)
    assert defaults == {"hexapod": 1.0, "car": 1.0, "drone": 1.0}

    weighted = _parse_embodiment_weights("car=2.5,drone=1.2", names)
    assert weighted["hexapod"] == 1.0
    assert weighted["car"] == 2.5
    assert weighted["drone"] == 1.2


def test_parse_embodiment_weights_rejects_unknown_name() -> None:
    with pytest.raises(ValueError):
        _parse_embodiment_weights("crawler=2.0", ["hexapod", "car", "drone"])


def test_weighted_transfer_score_prioritizes_target_embodiment() -> None:
    by_embodiment = {
        "hexapod": {"transfer_score": 0.40},
        "car": {"transfer_score": 0.20},
        "drone": {"transfer_score": 0.30},
    }
    score_equal = _weighted_transfer_score(
        by_embodiment=by_embodiment,
        embodiments=["hexapod", "car", "drone"],
        embodiment_weights={"hexapod": 1.0, "car": 1.0, "drone": 1.0},
    )
    score_car_heavy = _weighted_transfer_score(
        by_embodiment=by_embodiment,
        embodiments=["hexapod", "car", "drone"],
        embodiment_weights={"hexapod": 1.0, "car": 3.0, "drone": 1.0},
    )
    assert score_equal == pytest.approx(0.30)
    assert score_car_heavy == pytest.approx((0.40 + 0.20 * 3.0 + 0.30) / 5.0)
    assert score_car_heavy < score_equal


def test_binary_auc_orders_scores() -> None:
    auc = _binary_auc(
        scores=[0.2, 0.3, 0.8, 0.9],
        labels=[0, 0, 1, 1],
    )
    assert auc == pytest.approx(1.0)

    auc_mixed = _binary_auc(
        scores=[0.1, 0.9, 0.8, 0.2],
        labels=[0, 0, 1, 1],
    )
    assert 0.0 <= auc_mixed <= 1.0


def test_resolve_noise_profile_accepts_known_values() -> None:
    assert _resolve_noise_profile("none") == "none"
    assert _resolve_noise_profile("dropout-quant-v1") == "dropout-quant-v1"
    assert _resolve_noise_profile("dropout-quant-v2") == "dropout-quant-v2"
    with pytest.raises(ValueError):
        _resolve_noise_profile("bad-profile")


def test_apply_observation_noise_changes_tensor_for_noise_profile() -> None:
    obs = torch.linspace(-1.0, 1.0, 16).view(1, 16)
    out = _apply_observation_noise(obs, profile="dropout-quant-v1", seed=3, step=2)
    assert out.shape == obs.shape
    assert not torch.allclose(out, obs)


def test_resolve_subset_embodiments_validates_membership() -> None:
    names = _resolve_subset_embodiments("hexapod,car", ["hexapod", "car", "drone"])
    assert names == ["hexapod", "car"]

    with pytest.raises(ValueError):
        _resolve_subset_embodiments("crawler", ["hexapod", "car", "drone"])


def test_resolve_train_embodiments_prefers_flags_when_cli_empty() -> None:
    inferred = _resolve_train_embodiments(
        train_embodiments_csv="",
        all_embodiments=["hexapod", "car", "drone", "polymorph120"],
        checkpoint_flags={"embodiments": ["hexapod", "car"]},
    )
    assert inferred == ["hexapod", "car"]


def test_transfer_ratio_matrix_and_checkmate_metrics() -> None:
    by_embodiment = {
        "hexapod": {"transfer_score": 0.40},
        "car": {"transfer_score": 0.44},
        "drone": {"transfer_score": 0.36},
        "polymorph120": {"transfer_score": 0.34},
    }
    matrix = _transfer_ratio_matrix(
        by_embodiment=by_embodiment,
        source_embodiments=["hexapod", "car"],
        target_embodiments=["hexapod", "car", "drone", "polymorph120"],
    )
    assert matrix["hexapod"]["car"] == pytest.approx(1.1)
    assert matrix["car"]["drone"] == pytest.approx(0.36 / 0.44)

    checkmate = _checkmate_metrics(
        by_embodiment=by_embodiment,
        all_embodiments=["hexapod", "car", "drone", "polymorph120"],
        train_embodiments=["hexapod", "car"],
        threshold=0.85,
    )
    assert checkmate["checkmate_train_embodiments"] == ["hexapod", "car"]
    assert checkmate["checkmate_heldout_embodiments"] == ["drone", "polymorph120"]
    assert checkmate["checkmate_pass_all"] is False
    assert checkmate["checkmate_pass_heldout"] is False


def test_rollout_metrics_capability_profile_returns_proxy_metrics() -> None:
    cfg = ModelConfig(
        signal_dim=16,
        hidden_dim=24,
        edge_nodes=20,
        memory_slots=8,
        memory_dim=12,
        io_channels=6,
        max_remap_groups=4,
    )
    model = ModelCore(**cfg.__dict__)
    scenario = ScenarioSpec(
        name="storm",
        wind=(0.8, 0.35, 0.0),
        wind_variation=0.45,
        light_pos=(-0.4, 0.0, 0.25),
        light_drift=(0.004, 0.0, 0.0),
        light_intensity=0.55,
        force_vector=(1.1, 0.2, 0.0),
        force_strength=1.1,
        force_start=4,
        force_duration=8,
        force_pattern="pulse",
    )

    metrics = rollout_metrics(
        model=model,
        cfg=cfg,
        embodiment_name="polymorph120",
        scenario=scenario,
        steps=20,
        remap_every=5,
        seed=19,
        world_dims=(8, 8, 4, 3),
        device=model.passive.edge_projection.weight.device,
        capability_profile="bio-tech-v1",
    )

    assert "capability_score" in metrics
    assert "signal_reliability" in metrics
    assert "signal_corr_raw" in metrics
    assert "signal_detection_auc" in metrics
    assert "signal_detection_auc_raw" in metrics
    assert "evasion_success" in metrics
    assert "threat_steps" in metrics
    assert 0.0 <= float(metrics["signal_reliability"]) <= 1.0
    assert -1.0 <= float(metrics["signal_corr_raw"]) <= 1.0
    assert 0.0 <= float(metrics["signal_detection_auc_raw"]) <= 1.0
    assert 0.5 <= float(metrics["signal_detection_auc"]) <= 1.0
    assert 0.0 <= float(metrics["evasion_success"]) <= 1.0
    assert int(metrics["threat_steps"]) >= 1
