"""Export the SELECTED PyTorch checkpoint to an ONNX inference graph.

Run from the project root:
    python -m scripts.export_onnx --checkpoint artifacts/beans_frozen_seed21.pt
"""
from __future__ import annotations

import argparse
from pathlib import Path

import onnx
import torch

from image_api.model import ModelPredictor


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', default='artifacts/beans_frozen_seed21.pt')
    p.add_argument('--output', default='artifacts/beans_frozen_seed21.onnx')
    args = p.parse_args()

    source = Path(args.checkpoint)
    if not source.is_file():
        p.error(f'Checkpoint does not exist: {source}')
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    predictor = ModelPredictor(source)
    predictor.model.eval()

    # Fixed batch=1, 224x224 RGB; same as the current single-image API.
    sample = torch.randn(1, 3, 224, 224, dtype=torch.float32)
    torch.onnx.export(
        predictor.model,
        (sample,),
        str(target),
        input_names=['image'],
        output_names=['logits'],
        opset_version=18,
        dynamo=True,
    )
    onnx.checker.check_model(str(target))
    print(f'Exported and validated: {target}')
    print(f'Checkpoint model version: {predictor.version}')
    print(f'ONNX size: {target.stat().st_size / (1024 * 1024):.2f} MiB')


if __name__ == '__main__':
    main()
