from __future__ import annotations

import torch
from torch import nn
from ai_embedded_dynamic_diversity.models import ModelCore

def prepare_qat_model(model: ModelCore, backend: str = "qnnpack") -> nn.Module:
    """Prepares a ModelCore for Quantization-Aware Training."""
    # QNNPACK is good for ARM (Pi 5), fbgemm for x86.
    model.qconfig = torch.ao.quantization.get_default_qat_qconfig(backend)
    # Prepare model for QAT (inserts observers and fake quant modules)
    prepared = torch.ao.quantization.prepare_qat(model, inplace=False)
    return prepared

def convert_qat_model(model: nn.Module) -> nn.Module:
    """Converts a trained QAT model to a quantized model."""
    model.eval()
    quantized = torch.ao.quantization.convert(model, inplace=False)
    return quantized
