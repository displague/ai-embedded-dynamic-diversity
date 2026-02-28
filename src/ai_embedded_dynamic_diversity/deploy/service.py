from __future__ import annotations

import torch
from fastapi import FastAPI, Body
from pydantic import BaseModel
from ai_embedded_dynamic_diversity.models import ModelCore
from ai_embedded_dynamic_diversity.deploy.telemetry import TelemetryCollector

class InferenceRequest(BaseModel):
    signal: list[float]
    remap_code: list[float] | None = None
    metadata: dict | None = None

class InferenceResponse(BaseModel):
    output: list[float]
    readiness: list[float]
    predicted_remap: list[float]
    predicted_signal_type: int

def create_app(model: ModelCore, collector: TelemetryCollector | None = None):
    app = FastAPI(title="AI Embedded Dynamic Diversity Inference Service")
    
    # Initialize memory for batch=1
    memory = model.init_memory(1, model.memory_slots, model.memory_dim, "cpu")
    
    @app.post("/infer", response_model=InferenceResponse)
    async def infer(request: InferenceRequest):
        nonlocal memory
        
        sig_tensor = torch.tensor([request.signal], dtype=torch.float32)
        remap_tensor = None
        if request.remap_code:
            remap_tensor = torch.tensor([request.remap_code], dtype=torch.float32)
            
        model.eval()
        with torch.no_grad():
            out = model(sig_tensor, memory, remap_tensor)
            memory = out["memory"]
            
        # Capture telemetry
        if collector:
            collector.capture(sig_tensor, out["io"], request.metadata)
            
        return {
            "output": out["io"].tolist()[0],
            "readiness": out["readiness"].tolist()[0],
            "predicted_remap": out["predicted_remap"].tolist()[0],
            "predicted_signal_type": int(torch.argmax(out["predicted_signal_type"], dim=1).item())
        }

    @app.get("/health")
    async def health():
        return {"status": "ok", "model": model.__class__.__name__}

    return app
