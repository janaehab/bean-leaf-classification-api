"""ONNX-specific API contract tests. Real-model test is skipped if model artifacts absent."""
from __future__ import annotations

import io
from pathlib import Path
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from image_api.app import create_app


def _png():
    buf = io.BytesIO()
    Image.new("RGB", (100, 100), color=(65, 110, 75)).save(buf, format="PNG")
    return buf.getvalue()


def test_missing_onnx_model_is_not_ready(tmp_path):
    app = create_app(backend="onnx", onnx_path=str(tmp_path / "absent.onnx"))
    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/ready").status_code == 503
        assert client.get("/model-info").json()["backend"] == "onnx"
        assert client.post("/predict", files={"file": ("leaf.png", _png(), "image/png")}).status_code == 503


def test_live_onnx_model_if_available():
    pytest.importorskip("onnxruntime")
    graph = Path("artifacts/beans_frozen_seed21.onnx")
    metadata = Path(str(graph) + ".metadata.json")
    data = Path(str(graph) + ".data")
    if not all(p.is_file() for p in (graph, metadata, data)):
        pytest.skip("Real ONNX artifacts not present (run scripts.write_onnx_metadata first)")
    with TestClient(create_app(backend="onnx", onnx_path=str(graph))) as client:
        assert client.get("/ready").status_code == 200
        info = client.get("/model-info").json()
        assert info["backend"] == "onnx"
        assert info["version"] == "beans-mnv3-seed21"
        response = client.post("/predict", files={"file": ("leaf.png", _png(), "image/png")})
        assert response.status_code == 200
        result = response.json()
        assert result["label"] in info["classes"]
        assert result["model_version"] == info["version"]
        assert abs(sum(result["scores"].values()) - 1) < 0.00001


def test_preprocessing_matches_torchvision_if_available():
    pytest.importorskip("onnxruntime")
    torchvision = pytest.importorskip("torchvision")
    from image_api.onnx_model import ONNXPredictor
    from torchvision.models import MobileNet_V3_Small_Weights
    import numpy as np

    # Construct only preprocessing parameters; no ONNX session required.
    predictor = object.__new__(ONNXPredictor)
    predictor.resize_short_side = 256
    predictor.crop_size = 224
    predictor.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3,1,1)
    predictor.std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3,1,1)
    values = np.random.default_rng(123).integers(0, 256, size=(310, 420, 3), dtype=np.uint8)
    im = Image.fromarray(values)
    expected = MobileNet_V3_Small_Weights.IMAGENET1K_V1.transforms()(im).unsqueeze(0).numpy()
    actual = predictor.preprocess(im)
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)
