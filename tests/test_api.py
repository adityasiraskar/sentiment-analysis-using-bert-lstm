"""API tests using FastAPI TestClient with a mocked predictor.

We avoid loading a real trained model in tests by monkeypatching
`get_predictor` in `api.main` directly.
"""
from __future__ import annotations

from fastapi.testclient import TestClient

from api.main import app


class FakePredictor:
    def predict(self, texts):
        if isinstance(texts, str):
            texts = [texts]
            
        results = []
        for t in texts:
            results.append({
                "text": t,
                "label": "positive",
                "confidence": 0.99,
                "probabilities": {"negative": 0.005, "neutral": 0.005, "positive": 0.99}
            })
        return results


def test_health_when_model_not_loaded(monkeypatch):
    monkeypatch.setattr("api.main.get_predictor", lambda: None)
    monkeypatch.setattr("api.main.get_load_error", lambda: "Not loaded")
    
    with TestClient(app) as client:
        response = client.get("/health")
        
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "degraded"
    assert body["model_loaded"] is False
    assert body["load_error"] == "Not loaded"


def test_health_when_model_loaded(monkeypatch):
    monkeypatch.setattr("api.main.get_predictor", lambda: FakePredictor())
    monkeypatch.setattr("api.main.get_model_type", lambda: "bert_fake")
    
    with TestClient(app) as client:
        response = client.get("/health")
        
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["model_type"] == "bert_fake"


def test_predict_returns_503_when_model_not_loaded(monkeypatch):
    monkeypatch.setattr("api.main.get_predictor", lambda: None)
    
    with TestClient(app) as client:
        response = client.post("/predict", json={"text": "Great product!"})
        
    assert response.status_code == 503


def test_predict_success(monkeypatch):
    monkeypatch.setattr("api.main.get_predictor", lambda: FakePredictor())
    monkeypatch.setattr("api.main.get_model_type", lambda: "bert_fake")
    
    with TestClient(app) as client:
        response = client.post("/predict", json={"text": "Great product!"})
        
    assert response.status_code == 200
    body = response.json()
    assert body["model_used"] == "bert_fake"
    assert body["prediction"]["label"] == "positive"
    assert body["prediction"]["confidence"] == 0.99


def test_predict_rejects_empty_text(monkeypatch):
    monkeypatch.setattr("api.main.get_predictor", lambda: FakePredictor())
    
    with TestClient(app) as client:
        response = client.post("/predict", json={"text": ""})
        
    assert response.status_code == 422


def test_predict_batch_success(monkeypatch):
    monkeypatch.setattr("api.main.get_predictor", lambda: FakePredictor())
    monkeypatch.setattr("api.main.get_model_type", lambda: "bert_fake")
    
    with TestClient(app) as client:
        response = client.post("/predict/batch", json={"texts": ["Great!", "Terrible."]})
        
    assert response.status_code == 200
    body = response.json()
    assert body["model_used"] == "bert_fake"
    assert len(body["predictions"]) == 2
    assert body["predictions"][0]["text"] == "Great!"
    assert body["predictions"][1]["text"] == "Terrible."