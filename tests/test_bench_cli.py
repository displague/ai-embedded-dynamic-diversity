from __future__ import annotations

import pytest

from ai_embedded_dynamic_diversity.deploy import bench_cli


def test_parse_targets_supports_alias_and_explicit_device() -> None:
    parsed = bench_cli._parse_targets("cpu=cpu,cuda=gpu,pi5=external,mobile")
    assert parsed == [("cpu", "cpu"), ("cuda", "gpu"), ("pi5", "external"), ("mobile", "mobile")]


def test_parse_batch_sizes_requires_positive_values() -> None:
    assert bench_cli._parse_int_csv("1,4,8", "batch_sizes") == [1, 4, 8]
    with pytest.raises(ValueError):
        bench_cli._parse_int_csv("1,0", "batch_sizes")


def test_run_benchmark_rejects_invalid_steps() -> None:
    with pytest.raises(ValueError):
        bench_cli.run_benchmark(steps=0)


def test_build_latency_matrix_marks_external_and_unavailable_cuda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(bench_cli.torch.cuda, "is_available", lambda: False)

    payload = bench_cli.build_latency_matrix(
        weights="",
        profile="pi5",
        targets="cpu=cpu,cuda=cuda,pi5=external",
        batch_sizes="1",
        steps=2,
        warmup_steps=1,
    )

    rows = payload["targets"]
    assert rows[0]["target"] == "cpu"
    assert rows[0]["status"] == "ok"
    assert len(rows[0]["runs"]) == 1
    assert rows[1]["target"] == "cuda"
    assert rows[1]["status"] == "skipped"
    assert rows[2]["target"] == "pi5"
    assert rows[2]["status"] == "hardware_pending"
