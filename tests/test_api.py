"""Actual API tests, including a real randomly initialized checkpoint (not model accuracy)."""
import io

import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image

from image_api.app import create_app, MAX_BYTES
from image_api.model import build_model, DEFAULT_CLASSES


@pytest.fixture
def example_image():
    buf = io.BytesIO()
    Image.new("RGB", (80, 80), color=(25, 180, 85)).save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def untrained_checkpoint(tmp_path):
    path = tmp_path / "fake_untrained_weights.pt"
    torch.save({"model_state_dict": build_model(3, pretrained=False).state_dict(),
                "classes": DEFAULT_CLASSES, "version": "TEST_ONLY_NOT_TRAINED"}, path)
    return path


def test_liveness_and_readiness_without_model(tmp_path):
    app = create_app(model_path=str(tmp_path / "missing.pt"))
    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/ready").status_code == 503
        assert client.get("/model-info").json()["loaded"] is False
        assert client.post("/predict", files={"file": ("a.png", b"1", "image/png")}).status_code == 503


def test_prediction_and_model_info_with_untrained_fixture(untrained_checkpoint, example_image):
    app = create_app(model_path=str(untrained_checkpoint))
    with TestClient(app) as client:
        assert client.get("/ready").status_code == 200
        info = client.get("/model-info").json()
        assert info["classes"] == DEFAULT_CLASSES
        response = client.post("/predict", files={"file": ("leaf.png", example_image, "image/png")})
        assert response.status_code == 200
        answer = response.json()
        assert answer["label"] in DEFAULT_CLASSES
        assert 0 <= answer["confidence"] <= 1
        assert answer["model_version"] == "TEST_ONLY_NOT_TRAINED"


def test_rejects_bad_content_type(untrained_checkpoint):
    with TestClient(create_app(model_path=str(untrained_checkpoint))) as client:
        response = client.post("/predict", files={"file": ("malware.exe", b"x", "application/octet-stream")})
        assert response.status_code == 415


def test_rejects_corrupted_image(untrained_checkpoint):
    with TestClient(create_app(model_path=str(untrained_checkpoint))) as client:
        response = client.post("/predict", files={"file": ("bad.png", b"broken", "image/png")})
        assert response.status_code == 422


def test_rejects_large_upload(untrained_checkpoint):
    with TestClient(create_app(model_path=str(untrained_checkpoint))) as client:
        response = client.post("/predict", files={"file": ("big.png", b"a" * (MAX_BYTES + 1), "image/png")})
        assert response.status_code == 413
