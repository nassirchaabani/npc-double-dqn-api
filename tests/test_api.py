from fastapi.testclient import TestClient

from src.api import app


def test_health_without_model():
    with TestClient(app) as client:
        response = client.get("/health")
    assert response.status_code == 200
    assert "model_loaded" in response.json()


def test_predict_refuses_when_model_missing():
    with TestClient(app) as client:
        response = client.post("/predict", json={"values": [0, 0, 1, 1, 2, 2]})
    assert response.status_code == 503
