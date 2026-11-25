"""Create model metadata for the ONNX-only API from the selected PyTorch checkpoint.

Does not re-export the model, retrain, or overwrite the ONNX graph/weights.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="artifacts/beans_frozen_seed21.pt")
    parser.add_argument("--onnx", default="artifacts/beans_frozen_seed21.onnx")
    args = parser.parse_args()
    source = Path(args.checkpoint)
    target_model = Path(args.onnx)
    if not source.is_file() or not target_model.is_file():
        parser.error("Checkpoint and ONNX graph must already exist")
    payload = torch.load(str(source), map_location="cpu", weights_only=True)
    meta = {
        "version": str(payload["version"]),
        "classes": payload["classes"],
        "architecture": "mobilenet_v3_small",
        "source_checkpoint": source.name,
        "preprocessing": {
            "resize_short_side": 256,
            "crop_size": 224,
            "interpolation": "bilinear",
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        },
    }
    out = Path(str(target_model) + ".metadata.json")
    out.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Wrote metadata: {out}")
    print(f"Model version: {meta['version']}")


if __name__ == "__main__":
    main()
