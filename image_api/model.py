"""Model construction and inference, shared by training and API."""
from __future__ import annotations

import io
from pathlib import Path

import torch
from PIL import Image, UnidentifiedImageError
from torch import nn
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

DEFAULT_CLASSES = ["angular_leaf_spot", "bean_rust", "healthy"]
WEIGHTS = MobileNet_V3_Small_Weights.IMAGENET1K_V1


def build_model(num_classes: int = 3, pretrained: bool = False) -> nn.Module:
    """Use ImageNet weights during training; checkpoints during serving."""
    model = mobilenet_v3_small(weights=WEIGHTS if pretrained else None)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    return model


class ModelPredictor:
    def __init__(self, checkpoint: str | Path):
        self.device = torch.device("cpu")  # predictable inference environment
        payload = torch.load(checkpoint, map_location=self.device, weights_only=True)
        classes = payload["classes"]
        if not isinstance(classes, list) or not classes or not all(isinstance(x, str) for x in classes):
            raise ValueError("Invalid checkpoint classes")
        self.classes = classes
        self.version = str(payload.get("version", "unknown"))
        self.metadata = payload.get("metadata", {})
        self.model = build_model(num_classes=len(classes), pretrained=False)
        self.model.load_state_dict(payload["model_state_dict"], strict=True)
        self.model.to(self.device).eval()
        self.transform = WEIGHTS.transforms()

    def predict(self, content: bytes) -> dict:
        try:
            with Image.open(io.BytesIO(content)) as im:
                image = im.convert("RGB")
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            raise ValueError("Uploaded data is not a valid image") from exc
        batch = self.transform(image).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            logits = self.model(batch)
            scores = torch.softmax(logits, dim=1)[0].tolist()
        best = max(range(len(scores)), key=scores.__getitem__)
        return {
            "label": self.classes[best],
            "confidence": round(float(scores[best]), 6),
            "scores": {name: round(float(score), 6) for name, score in zip(self.classes, scores)},
            "model_version": self.version,
        }
