import torch
from torch import nn
from ai_embedded_dynamic_diversity.config import model_config_for_profile
from ai_embedded_dynamic_diversity.models import ModelCore
from ai_embedded_dynamic_diversity.deploy.cli import TorchScriptWrapper

def export_dynamic_int8(weights_path: str, output_path: str):
    ckpt = torch.load(weights_path, map_location="cpu")
    cfg = model_config_for_profile("pi5")
    cfg = type(cfg)(**ckpt["model_config"])

    model = ModelCore(**cfg.__dict__)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()

    # Apply dynamic quantization to Linear layers
    quantized = torch.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)

    wrapped = TorchScriptWrapper(quantized)

    # Trace for deployment
    signal = torch.randn(1, cfg.signal_dim)
    memory = torch.zeros(1, cfg.memory_slots, cfg.memory_dim)
    remap = torch.zeros(1, cfg.max_remap_groups)

    traced = torch.jit.trace(wrapped, (signal, memory, remap))
    traced.save(output_path)
    print(f"Exported INT8 model to {output_path}")

if __name__ == "__main__":
    export_dynamic_int8("artifacts/model-core-champion-v08.pt", "artifacts/champion-v08-int8.ts")
