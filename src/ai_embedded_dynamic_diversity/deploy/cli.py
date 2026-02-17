from __future__ import annotations

from pathlib import Path

import torch
import typer
from torch import nn

from ai_embedded_dynamic_diversity.config import ModelConfig
from ai_embedded_dynamic_diversity.models import ModelCore

app = typer.Typer(add_completion=False)


class OnnxIOWrapper(nn.Module):
    def __init__(self, model: ModelCore):
        super().__init__()
        self.model = model

    def forward(self, signal: torch.Tensor, memory: torch.Tensor, remap: torch.Tensor) -> torch.Tensor:
        return self.model(signal, memory, remap)["io"]


class TorchScriptWrapper(nn.Module):
    def __init__(self, model: ModelCore):
        super().__init__()
        self.model = model

    def forward(
        self, signal: torch.Tensor, memory: torch.Tensor, remap: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        out = self.model(signal, memory, remap)
        return out["io"], out["readiness"], out["memory"]


def _load_model(weights: str, profile: str | None = None) -> tuple[ModelCore, ModelConfig]:
    ckpt = torch.load(weights, map_location="cpu")
    cfg = ModelConfig(**ckpt["model_config"])
    model = ModelCore(**cfg.__dict__)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, cfg


@app.command()
def torchscript(weights: str = "artifacts/model-core.pt", output: str = "artifacts/model-core.ts") -> None:
    model, cfg = _load_model(weights)

    signal = torch.randn(1, cfg.signal_dim)
    memory = torch.zeros(1, cfg.memory_slots, cfg.memory_dim)
    remap = torch.zeros(1, cfg.max_remap_groups)

    traced = torch.jit.trace(TorchScriptWrapper(model), (signal, memory, remap))
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    traced.save(output)
    print({"torchscript": output})


@app.command()
def onnx(weights: str = "artifacts/model-core.pt", output: str = "artifacts/model-core.onnx", opset: int = 17) -> None:
    model, cfg = _load_model(weights)

    signal = torch.randn(1, cfg.signal_dim)
    memory = torch.zeros(1, cfg.memory_slots, cfg.memory_dim)
    remap = torch.zeros(1, cfg.max_remap_groups)

    wrapper = OnnxIOWrapper(model)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper,
        (signal, memory, remap),
        output,
        input_names=["signal", "memory", "remap"],
        output_names=["io"],
        opset_version=opset,
        dynamic_axes={"signal": {0: "batch"}, "memory": {0: "batch"}, "remap": {0: "batch"}},
    )
    print({"onnx": output})


@app.command()
def quantized_torchscript(weights: str = "artifacts/model-core.pt", output: str = "artifacts/model-core-int8.ts") -> None:
    model, cfg = _load_model(weights)
    wrapped = TorchScriptWrapper(model)
    quantized = torch.quantization.quantize_dynamic(wrapped, {nn.Linear}, dtype=torch.qint8)

    signal = torch.randn(1, cfg.signal_dim)
    memory = torch.zeros(1, cfg.memory_slots, cfg.memory_dim)
    remap = torch.zeros(1, cfg.max_remap_groups)
    traced = torch.jit.trace(quantized, (signal, memory, remap))

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    traced.save(output)
    print({"quantized_torchscript": output})


if __name__ == "__main__":
    app()
