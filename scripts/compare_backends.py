"""Compare selected checkpoint's PyTorch vs ONNX Runtime inference on CPU.

Uses the identical PREPROCESSED image tensor for both engines. Timings
exclude HTTP, decoding and preprocessing; they are model-only timings.
"""
from __future__ import annotations

import argparse
import io
import json
import statistics
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image
import torch

from image_api.model import ModelPredictor


def measure(fn, warmup: int, runs: int):
    for _ in range(warmup):
        fn()
    values = []
    for _ in range(runs):
        start = time.perf_counter()
        fn()
        values.append((time.perf_counter() - start) * 1000)
    return {
        'mean_ms': round(statistics.mean(values), 3),
        'p50_ms': round(float(np.percentile(values, 50)), 3),
        'p95_ms': round(float(np.percentile(values, 95)), 3),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', default='artifacts/beans_frozen_seed21.pt')
    p.add_argument('--onnx', default='artifacts/beans_frozen_seed21.onnx')
    p.add_argument('--image', default='sample_leaf.png')
    p.add_argument('--warmup', type=int, default=20)
    p.add_argument('--runs', type=int, default=150)
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--output', default='results/backend_benchmark.json')
    args = p.parse_args()
    if args.runs < 1 or args.warmup < 0 or args.threads < 1:
        p.error('runs and threads must be positive; warmup must not be negative')

    torch.set_num_threads(args.threads)
    predictor = ModelPredictor(args.checkpoint)
    predictor.model.eval()
    image_path = Path(args.image)
    with Image.open(image_path) as im:
        rgb = im.convert('RGB')
        batch = predictor.transform(rgb).unsqueeze(0).contiguous()
    arr = batch.numpy().astype(np.float32, copy=False)

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = args.threads
    opts.inter_op_num_threads = 1
    session = ort.InferenceSession(str(Path(args.onnx)), sess_options=opts, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    assert session.get_providers()[0] == 'CPUExecutionProvider'

    def pytorch():
        with torch.inference_mode():
            return predictor.model(batch)

    def onnx():
        return session.run(None, {input_name: arr})[0]

    torch_logits = pytorch().numpy()
    onnx_logits = onnx()
    pt_top1 = int(torch_logits.argmax(axis=1)[0])
    ort_top1 = int(onnx_logits.argmax(axis=1)[0])
    diff = float(np.max(np.abs(torch_logits - onnx_logits)))
    result = {
        'checkpoint': args.checkpoint,
        'model_version': predictor.version,
        'onnx': args.onnx,
        'image': args.image,
        'threads_per_backend': args.threads,
        'warmup': args.warmup,
        'runs': args.runs,
        'scope': 'CPU model-only inference; preprocessing and HTTP excluded',
        'pytorch_predicted_class': predictor.classes[pt_top1],
        'onnx_predicted_class': predictor.classes[ort_top1],
        'same_predicted_class': pt_top1 == ort_top1,
        'max_absolute_logit_difference': diff,
        'logits_close_at_1e_minus_3': bool(np.allclose(torch_logits, onnx_logits, rtol=1e-3, atol=1e-3)),
        'pytorch': measure(pytorch, args.warmup, args.runs),
        'onnxruntime': measure(onnx, args.warmup, args.runs),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))
    print(f'\nSaved: {output}')
    if not result['same_predicted_class'] or not result['logits_close_at_1e_minus_3']:
        print('WARNING: Backend outputs differ on the sample; investigate before serving ONNX.')


if __name__ == '__main__':
    main()
