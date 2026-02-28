from __future__ import annotations

import json
from pathlib import Path

from ai_embedded_dynamic_diversity.sim import prelife_cli


def test_prelife_simulation_emits_required_metrics() -> None:
    payload = prelife_cli.run_prelife_simulation(
        substrate="bytecode_dense",
        steps=80,
        seed=5,
        initial_agents=10,
        max_agents=40,
    )
    metrics = payload["metrics"]
    assert "first_replication_step" in metrics
    assert "replication_rate" in metrics
    assert "self_modification_rate" in metrics
    assert "lineage_depth_p50" in metrics
    assert "lineage_depth_p95" in metrics
    assert "novelty_growth_slope" in metrics
    assert "description_copy_fidelity" in metrics
    assert "symbiogenesis_event_count" in metrics
    assert 0.0 <= float(metrics["description_copy_fidelity"]) <= 1.0


def test_prelife_dense_beats_control_replication_rate() -> None:
    dense = prelife_cli.run_prelife_simulation("bytecode_dense", steps=120, seed=13)
    control = prelife_cli.run_prelife_simulation("sublike_control", steps=120, seed=13)
    assert float(dense["metrics"]["replication_rate"]) >= float(control["metrics"]["replication_rate"])


def test_prelife_detectors() -> None:
    assert prelife_cli.detect_self_modification_events("ABCM", "ABRM") is True
    assert prelife_cli.detect_self_modification_events("ABCM", "ABCM") is False
    assert prelife_cli.detect_symbiogenesis_event((4, 9)) is True
    assert prelife_cli.detect_symbiogenesis_event((7,)) is False


def test_prelife_report_writes_markdown_and_csv() -> None:
    payload = prelife_cli.run_prelife_simulation("bytecode_sparse", steps=60, seed=3)
    input_path = Path("artifacts/test-prelife.json")
    md_path = Path("artifacts/test-prelife.md")
    csv_path = Path("artifacts/test-prelife.csv")
    input_path.parent.mkdir(parents=True, exist_ok=True)
    input_path.write_text(json.dumps(payload), encoding="utf-8")

    prelife_cli.report(input_path=str(input_path), markdown_out=str(md_path), csv_out=str(csv_path))
    assert md_path.exists()
    assert csv_path.exists()
