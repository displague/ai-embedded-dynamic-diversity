from __future__ import annotations

import os
import sys

import torch
import typer


def device_runtime_snapshot() -> dict[str, object]:
    return {
        "sys_executable": sys.executable,
        "torch_version": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        "venv": os.environ.get("VIRTUAL_ENV", ""),
    }


def choose_device(preferred: str, strict: bool = True) -> torch.device:
    normalized = preferred.strip().lower()
    if normalized == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if strict:
            snapshot = device_runtime_snapshot()
            raise typer.BadParameter(
                "CUDA was requested but is unavailable in this runtime. "
                f"Snapshot: {snapshot}. "
                "Use the project venv CUDA torch interpreter or switch to --device cpu."
            )
        return torch.device("cpu")
    if normalized == "mps":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if strict:
            raise typer.BadParameter("MPS was requested but is unavailable in this runtime.")
        return torch.device("cpu")
    if normalized == "cpu":
        return torch.device("cpu")
    if strict:
        raise typer.BadParameter(f"Unknown device '{preferred}'. Expected one of: cpu, cuda, mps.")
    return torch.device("cpu")
