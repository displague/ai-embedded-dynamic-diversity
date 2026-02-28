import pytest
from fastapi.testclient import TestClient
from ai_embedded_dynamic_diversity.models import ModelCore
from ai_embedded_dynamic_diversity.deploy.service import create_app

def test_service_inference():
    model = ModelCore(10, 4, 32, 10, 16, 64, 4)
    app = create_app(model)
    client = TestClient(app)
    
    response = client.post(
        "/infer", 
        json={"signal": [0.1] * 10, "metadata": {"test": True}}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert len(data["output"]) == 4
    assert "readiness" in data
    assert "predicted_remap" in data

def test_service_health():
    model = ModelCore(10, 4, 32, 10, 16, 64, 4)
    app = create_app(model)
    client = TestClient(app)
    
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
