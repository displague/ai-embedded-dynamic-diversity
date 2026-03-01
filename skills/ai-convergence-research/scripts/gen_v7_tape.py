from __future__ import annotations

import torch
import json
from ai_embedded_dynamic_diversity.models.constructor_tape import ConstructorTape, save_constructor_tape

def generate_v7_tape(weights_path: str, output_path: str):
    ckpt = torch.load(weights_path, map_location="cpu")
    mcfg_dict = ckpt.get("model_config", {})
    
    tokens = []
    for k, v in mcfg_dict.items():
        if isinstance(v, bool):
            tokens.append(f"{k}={str(v).lower()}")
        else:
            tokens.append(f"{k}={v}")
            
    tape = ConstructorTape(version=7, tokens=sorted(tokens))
    save_constructor_tape(tape, output_path)
    print(f"Generated tape at {output_path}")

if __name__ == "__main__":
    generate_v7_tape("artifacts/model-core-champion-v07.pt", "artifacts/model-core-v07.tape.json")
