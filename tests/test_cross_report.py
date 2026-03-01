from __future__ import annotations

import json
import shutil
from pathlib import Path

from ai_embedded_dynamic_diversity.train.cross_report_cli import run


def test_cross_report_includes_mimicry_and_conjoining_columns() -> None:
    out_dir = Path("artifacts/test-cross-report")
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    input_path = out_dir / "cross-eval.json"
    markdown_out = out_dir / "cross-eval.md"
    csv_out = out_dir / "cross-eval.csv"

    payload = {
        "config": {
            "embodiments": ["hexapod", "car"],
            "capability_profile": "bio-tech-v1",
            "capability_score_weight": 0.25,
            "prelife_profile": "none",
            "prelife_score_weight": 0.0,
            "autopoiesis_score_weight": 0.15,
            "enable_humanoid_compliance": True,
            "humanoid_embodiment_name": "humanoid120",
            "humanoid_compliance_profile": "human_rigid_v1",
            "embodiment_weights": {"hexapod": 1.0, "car": 1.0},
            "checkmate_threshold": 0.85,
        },
        "ranked": [
            {
                "checkpoint": "artifacts/model-core-champion-v09.pt",
                "overall_transfer_score": 0.8,
                "overall_transfer_score_weighted": 0.6,
                "overall_transfer_score_unweighted": 0.8,
                "overall_capability_score": 0.5,
                "overall_prelife_score": 0.0,
                "overall_autopoiesis_score": 0.4,
                "overall_mean_mismatch": 0.2,
                "overall_mean_vitality": 0.7,
                "overall_recovery": 0.8,
                "ranking_component_transfer": 0.6,
                "ranking_component_capability": 0.125,
                "ranking_component_prelife": 0.0,
                "ranking_component_autopoiesis": 0.06,
                "humanoid_compliance": {
                    "overall_score": 0.83,
                    "pass": True,
                },
                "checkmate_pass_all": True,
                "checkmate_pass_heldout": True,
                "checkmate_min_effectiveness": 0.9,
                "checkmate_mean_effectiveness": 0.95,
                "checkmate_heldout_effectiveness": 0.0,
                "checkmate_train_embodiments": ["hexapod", "car"],
                "checkmate_heldout_embodiments": [],
                "symbio_gate_pass": True,
                "autopoiesis_gate_pass": True,
                "convergence_gate_pass": True,
                "promotion_eligible": True,
                "flags": {},
                "by_embodiment": {
                    "hexapod": {
                        "transfer_score": 0.81,
                        "capability_score": 0.51,
                        "signal_reliability": 0.91,
                        "signal_detection_auc": 0.88,
                        "evasion_success": 0.80,
                        "mimicry_reliability": 0.67,
                        "conjoining_gain": 0.63,
                    },
                    "car": {
                        "transfer_score": 0.79,
                        "capability_score": 0.49,
                        "signal_reliability": 0.89,
                        "signal_detection_auc": 0.86,
                        "evasion_success": 0.82,
                        "mimicry_reliability": 0.71,
                        "conjoining_gain": 0.69,
                    },
                },
            }
        ],
    }
    input_path.write_text(json.dumps(payload), encoding="utf-8")

    run(input_path=str(input_path), markdown_out=str(markdown_out), csv_out=str(csv_out))

    csv_text = csv_out.read_text(encoding="utf-8")
    assert "overall_mimicry_reliability" in csv_text
    assert "overall_conjoining_gain" in csv_text
    assert "hexapod_mimicry_reliability" in csv_text
    assert "car_conjoining_gain" in csv_text
    assert "humanoid_compliance_score" in csv_text
    assert "humanoid_compliance_pass" in csv_text

    md_text = markdown_out.read_text(encoding="utf-8")
    assert "Mimicry Reliability" in md_text
    assert "Conjoining Gain" in md_text
    assert "Humanoid Compliance" in md_text
