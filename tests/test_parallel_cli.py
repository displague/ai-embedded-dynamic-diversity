from __future__ import annotations

import json
from pathlib import Path

from ai_embedded_dynamic_diversity.train import parallel_cli


def test_parallel_cli_forwards_constructor_tape_cycle(monkeypatch) -> None:
    captured_cmds: list[list[str]] = []

    class _Proc:
        def __init__(self) -> None:
            self.returncode = 0
            self.stdout = ""
            self.stderr = ""

    def fake_run(cmd, capture_output, text, env):
        captured_cmds.append(list(cmd))
        metrics_idx = cmd.index("--metrics-path") + 1
        metrics_path = Path(cmd[metrics_idx])
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps({"flags": {"constructor_tape_path": "artifacts/tape-a.json", "constructor_tape_version": 1}, "records": [{"fitness": 0.1, "mean_step_ms": 1.2}]}), encoding="utf-8")
        return _Proc()

    monkeypatch.setattr(parallel_cli.subprocess, "run", fake_run)

    out_dir = Path("artifacts/test-parallel-out")
    out_dir.mkdir(parents=True, exist_ok=True)
    parallel_cli.run(
        variants=2,
        max_workers=1,
        out_dir=str(out_dir),
        constructor_tape_cycle="artifacts/tape-a.json,artifacts/tape-b.json",
        enable_autopoietic_objective=True,
        autopoietic_loss_weight_cycle="0.1,0.2",
        device="cpu",
    )

    assert any("--constructor-tape-path" in cmd for cmd in captured_cmds)
    assert any("--enable-autopoietic-objective" in cmd for cmd in captured_cmds)
    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert "constructor_tape_path" in summary[0]
    assert "constructor_tape_version" in summary[0]
