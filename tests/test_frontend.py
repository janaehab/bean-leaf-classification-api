"""LeafLens homepage smoke tests; predictor is injected to avoid model downloads."""
from __future__ import annotations

from fastapi.testclient import TestClient
from image_api.app import create_app


class FakePredictor:
    classes = ["angular_leaf_spot", "bean_rust", "healthy"]
    version = "example-model"

    def predict(self, _content):
        return {"label": "healthy", "confidence": 0.8,
                "scores": {"angular_leaf_spot": 0.1, "bean_rust": 0.1, "healthy": 0.8},
                "model_version": self.version}


def test_homepage_is_served_without_breaking_api():
    with TestClient(create_app(predictor=FakePredictor())) as client:
        response = client.get("/")
        assert response.status_code == 200
        assert "LeafLens AI" in response.text
        assert "/static/styles.css" in response.text
        assert "/static/app.js" in response.text
        assert client.get("/ready").json() == {"status": "ready"}
        assert client.get("/model-info").json()["version"] == "example-model"
        assert client.get("/docs").status_code == 200


def test_frontend_assets_load():
    with TestClient(create_app(predictor=FakePredictor())) as client:
        assert client.get("/static/styles.css").status_code == 200
        js = client.get("/static/app.js")
        assert js.status_code == 200
        assert 'fetch("/predict"' in js.text
        assert client.get("/static/favicon.svg").status_code == 200
