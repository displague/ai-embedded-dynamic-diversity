from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from ai_embedded_dynamic_diversity.config import ModelConfig
from ai_embedded_dynamic_diversity.models import ModelCore
from ai_embedded_dynamic_diversity.train.cross_eval_cli import (
    ScenarioSpec,
    _apply_observation_noise,
    _binary_auc,
    _checkmate_metrics,
    _cycle_output_path,
    _parse_embodiment_weights,
    _build_convergence_threshold_payload,
    _prelife_metrics_for_checkpoint,
    _resolve_prelife_profile,
    _resolve_prelife_seeds,
    _resolve_noise_profile,
    _resolve_humanoid_embodiment_name,
    _resolve_world_randomization_manifest,
    _sample_world_params_for_run,
    _resolve_subset_embodiments,
    _resolve_train_embodiments,
    _resolve_scenario_profile,
    _transfer_ratio_matrix,
    _weighted_transfer_score,
    _normalize_prelife_score,
    compute_recovery_score,
    rollout_metrics,
)
from ai_embedded_dynamic_diversity.sim.humanoid_compliance import resolve_humanoid_compliance_profile


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
        world_params={},
        device=model.passive.edge_projection.weight.device,
    )

    assert "transfer_score" in metrics
    assert "mean_mismatch" in metrics
    assert "mean_vitality" in metrics
    assert metrics["remap_events"] >= 1
    assert "autopoiesis_score" in metrics
    assert 0.0 <= float(metrics["autopoiesis_score"]) <= 1.0


def test_hardy_profile_resolves_multiple_scenarios() -> None:
    scenarios = _resolve_scenario_profile("hardy")
    names = [s.name for s in scenarios]
    assert "storm" in names
    assert "blackout" in names
    assert len(names) >= 5


def test_calibrated_large_profile_resolves_new_scenarios() -> None:
    scenarios = _resolve_scenario_profile("calibrated_large_v1")
    names = [s.name for s in scenarios]
    assert "latency-storm" in names
    assert "friction-shift" in names
    assert "persistent-gust" in names


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
        world_params={},
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


def test_resolve_prelife_profile_and_seeds() -> None:
    assert _resolve_prelife_profile("none") == "none"
    assert _resolve_prelife_profile("dense-vs-control-v1") == "dense-vs-control-v1"
    with pytest.raises(ValueError):
        _resolve_prelife_profile("bad")

    assert _resolve_prelife_seeds("3", base_seed=100) == [100, 101, 102]
    assert _resolve_prelife_seeds("7,9", base_seed=100) == [7, 9]


def test_normalize_prelife_score_in_unit_range() -> None:
    score = _normalize_prelife_score(
        {
            "dense_replication_rate": 2.0,
            "control_replication_rate": 0.2,
            "dense_self_modification_rate": 0.8,
            "dense_novelty_growth_slope": 0.4,
            "dense_description_copy_fidelity": 0.9,
            "dense_symbiogenesis_event_count": 2.0,
        }
    )
    assert 0.0 <= score <= 1.0


def test_prelife_metrics_for_checkpoint_returns_score() -> None:
    raw, score = _prelife_metrics_for_checkpoint(
        profile="dense-vs-control-v1",
        prelife_steps=40,
        prelife_seeds=[3, 5],
    )
    assert "dense_replication_rate" in raw
    assert "control_replication_rate" in raw
    assert 0.0 <= score <= 1.0


def test_build_convergence_threshold_payload_progression() -> None:
    payload = _build_convergence_threshold_payload(
        path="artifacts/test-thresholds.json",
        symbio_min_threshold=0.45,
        autopoiesis_min_threshold=0.55,
        best_symbio_contrast=0.50,
        best_autopoiesis_score=0.60,
    )
    assert payload["cycle_index"] >= 0
    next_thresholds = payload["next_recommended_thresholds"]
    assert float(next_thresholds["symbio_min_threshold"]) >= 0.45
    assert float(next_thresholds["autopoiesis_min_threshold"]) >= 0.55


def test_build_convergence_threshold_payload_custom_ratchet_steps() -> None:
    payload = _build_convergence_threshold_payload(
        path="artifacts/test-thresholds-custom.json",
        symbio_min_threshold=0.47,
        autopoiesis_min_threshold=0.38,
        best_symbio_contrast=0.60,
        best_autopoiesis_score=0.50,
        symbio_step=0.01,
        autopoiesis_step=0.02,
    )
    next_thresholds = payload["next_recommended_thresholds"]
    assert float(next_thresholds["symbio_min_threshold"]) == pytest.approx(0.48)
    assert float(next_thresholds["autopoiesis_min_threshold"]) == pytest.approx(0.40)


def test_cycle_output_path_suffixes_cycle_index() -> None:
    assert _cycle_output_path("artifacts/cross-eval.json", 3).endswith("cross-eval-cycle3.json")


def test_humanoid_profile_and_name_resolution() -> None:
    assert _resolve_humanoid_embodiment_name("") == "humanoid120"
    assert _resolve_humanoid_embodiment_name("HuManoiD120") == "humanoid120"
    profile = resolve_humanoid_compliance_profile("human_rigid_v1")
    assert profile.required_embodiment == "humanoid120"


def test_resolve_world_randomization_manifest_empty_and_missing() -> None:
    assert _resolve_world_randomization_manifest("") == {}
    with pytest.raises(ValueError):
        _resolve_world_randomization_manifest("does-not-exist.json")


def test_resolve_world_randomization_manifest_loads_json() -> None:
    p = Path("artifacts/test-world-randomization-manifest.json")
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {"global": {"surface_friction_scale": {"min": 0.9, "max": 1.1}}}
    p.write_text(json.dumps(payload), encoding="utf-8")
    out = _resolve_world_randomization_manifest(str(p))
    assert out["global"]["surface_friction_scale"]["min"] == 0.9


def test_sample_world_params_for_run_deterministic_with_seed() -> None:
    base = {
        "decay": 0.03,
        "actuation_delay_steps": 1,
        "actuation_noise_std": 0.0,
        "sensor_latency_steps": 1,
        "sensor_dropout_burst_prob": 0.0,
        "surface_friction_scale": 1.0,
        "disturbance_correlation_horizon": 1,
    }
    manifest = {
        "global": {"surface_friction_scale": {"min": 0.8, "max": 1.2}},
        "scenarios": {"storm": {"actuation_delay_steps": {"min": 2, "max": 4}}},
    }
    a = _sample_world_params_for_run(base, "storm", manifest, seed=123)
    b = _sample_world_params_for_run(base, "storm", manifest, seed=123)
    c = _sample_world_params_for_run(base, "storm", manifest, seed=124)
    assert a == b
    assert a != c
    assert 0.8 <= float(a["surface_friction_scale"]) <= 1.2
    assert 2 <= int(a["actuation_delay_steps"]) <= 4
