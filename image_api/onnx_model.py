"""CPU-only ONNX inference, with preprocessing matching MobileNetV3 ImageNet weights.

The runtime deliberately does not import torch or torchvision. Train/export with
PyTorch, but serve with Pillow, NumPy and ONNX Runtime alone.
"""
from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image, UnidentifiedImageError


class ONNXPredictor:
    backend = "onnxruntime"

    def __init__(self, model_path: str | Path, *, threads: int = 2):
        self.path = Path(model_path)
        if not self.path.is_file():
            raise FileNotFoundError(f"Missing ONNX model: {self.path}")
        meta_path = Path(str(self.path) + ".metadata.json")
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        classes = metadata["classes"]
        if not isinstance(classes, list) or not classes or not all(isinstance(c, str) for c in classes):
            raise ValueError("Invalid model class metadata")
        self.classes = classes
        self.version = str(metadata["version"])
        self.metadata = metadata
        preprocessing = metadata["preprocessing"]
        self.resize_short_side = int(preprocessing["resize_short_side"])
        self.crop_size = int(preprocessing["crop_size"])
        self.mean = np.asarray(preprocessing["mean"], dtype=np.float32).reshape(3, 1, 1)
        self.std = np.asarray(preprocessing["std"], dtype=np.float32).reshape(3, 1, 1)
        if self.resize_short_side < self.crop_size or self.crop_size <= 0 or np.any(self.std <= 0):
            raise ValueError("Invalid preprocessing configuration")
        if preprocessing["interpolation"] != "bilinear":
            raise ValueError("Expected bilinear preprocessing")
        if threads < 1:
            raise ValueError("threads must be >= 1")
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        self.session = ort.InferenceSession(str(self.path), sess_options=options, providers=["CPUExecutionProvider"])
        if "CPUExecutionProvider" not in self.session.get_providers():
            raise RuntimeError("ONNX Runtime CPU provider unavailable")
        inputs = self.session.get_inputs()
        if len(inputs) != 1:
            raise ValueError("Expected one model input")
        self.input_name = inputs[0].name
        outputs = self.session.get_outputs()
        if len(outputs) != 1:
            raise ValueError("Expected one model output")
        dims = outputs[0].shape
        if len(dims) == 2 and isinstance(dims[-1], int) and dims[-1] != len(classes):
            raise ValueError("ONNX output class count differs from metadata")

    def preprocess(self, image: Image.Image) -> np.ndarray:
        """Match torchvision ImageNet1K_V1 transforms: Resize(256), CenterCrop(224), ToTensor, Normalize."""
        im = image.convert("RGB")
        w, h = im.size
        if w <= 0 or h <= 0:
            raise ValueError("Invalid image dimensions")
        short = self.resize_short_side
        if w < h:
            new_w, new_h = short, int(short * h / w)
        else:
            new_w, new_h = int(short * w / h), short
        im = im.resize((new_w, new_h), resample=Image.Resampling.BILINEAR)
        left = int(round((new_w - self.crop_size) / 2.0))
        top = int(round((new_h - self.crop_size) / 2.0))
        im = im.crop((left, top, left + self.crop_size, top + self.crop_size))
        tensor = np.asarray(im, dtype=np.float32).transpose(2, 0, 1) / np.float32(255.0)
        tensor = (tensor - self.mean) / self.std
        return np.ascontiguousarray(tensor[None, ...], dtype=np.float32)

    def predict(self, content: bytes) -> dict:
        try:
            with Image.open(io.BytesIO(content)) as im:
                batch = self.preprocess(im)
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            raise ValueError("Invalid image") from exc
        logits = np.asarray(self.session.run(None, {self.input_name: batch})[0], dtype=np.float64)
        if logits.shape != (1, len(self.classes)) or not np.all(np.isfinite(logits)):
            raise RuntimeError("Unexpected ONNX model output")
        shifted = logits[0] - logits[0].max()
        exp = np.exp(shifted)
        probabilities = exp / exp.sum()
        best = int(probabilities.argmax())
        return {
            "label": self.classes[best],
            "confidence": round(float(probabilities[best]), 6),
            "scores": {name: round(float(prob), 6) for name, prob in zip(self.classes, probabilities)},
            "model_version": self.version,
        }
