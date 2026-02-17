from __future__ import annotations

from ai_embedded_dynamic_diversity.config import ModelConfig
from ai_embedded_dynamic_diversity.models import ModelCore
from ai_embedded_dynamic_diversity.train.cross_eval_cli import ScenarioSpec, compute_recovery_score, rollout_metrics


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
