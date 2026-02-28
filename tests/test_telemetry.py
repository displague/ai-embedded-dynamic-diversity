import torch
import json
import pytest
from ai_embedded_dynamic_diversity.deploy.telemetry import TelemetryCollector

def test_telemetry_capture(tmp_path):
    log_dir = tmp_path / "telemetry"
    collector = TelemetryCollector(log_dir=str(log_dir))
    
    signal = torch.randn(1, 10)
    output = torch.randn(1, 4)
    metadata = {"embodiment": "hexapod", "stress": 0.5}
    
    collector.capture(signal, output, metadata)
    
    # Check if file created
    logs = list(log_dir.glob("*.json"))
    assert len(logs) == 1
    
    with open(logs[0], "r") as f:
        data = json.load(f)
        assert data["metadata"]["embodiment"] == "hexapod"
        assert len(data["signal"]) == 10

def test_telemetry_get_retraining_data(tmp_path):
    log_dir = tmp_path / "telemetry"
    collector = TelemetryCollector(log_dir=str(log_dir))
    
    # Capture multiple steps
    for i in range(3):
        collector.capture(torch.randn(1, 10), torch.randn(1, 4), {"step": i})
        
    retraining_data = collector.get_retraining_data()
    assert len(retraining_data) == 3
    assert "signal" in retraining_data[0]
    assert "output" in retraining_data[0]
