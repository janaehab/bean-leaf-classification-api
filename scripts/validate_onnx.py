"""Validate ONNX vs PyTorch on the entire Beans validation split.

Copy into scripts/validate_onnx.py and run from the project root:
python -m scripts.validate_onnx --checkpoint artifacts/beans_frozen_seed21.pt --onnx artifacts/beans_frozen_seed21.onnx

This is an equivalence check, not another model-selection experiment. Does not use the test split.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score

from image_api.model import ModelPredictor
from scripts.train import DATASET_ID


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', default='artifacts/beans_frozen_seed21.pt')
    parser.add_argument('--onnx', default='artifacts/beans_frozen_seed21.onnx')
    parser.add_argument('--output', default='results/onnx_validation.json')
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error('--threads must be positive')

    torch.set_num_threads(args.threads)
    predictor = ModelPredictor(args.checkpoint)
    ds = load_dataset(DATASET_ID)
    label_names = list(ds['validation'].features['labels'].names)
    if label_names != predictor.classes:
        raise ValueError(f'Class order mismatch: dataset={label_names}, checkpoint={predictor.classes}')

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = args.threads
    opts.inter_op_num_threads = 1
    session = ort.InferenceSession(args.onnx, sess_options=opts, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    if 'CPUExecutionProvider' not in session.get_providers():
        raise RuntimeError(f'Unexpected execution providers: {session.get_providers()}')

    truths: list[int] = []
    pt_preds: list[int] = []
    ort_preds: list[int] = []
    max_abs = 0.0
    all_close = 0
    mismatches = []
    for idx, record in enumerate(ds['validation']):
        # Apply EXACTLY the same preprocessing once for both backends.
        rgb = record['image'].convert('RGB')
        batch = predictor.transform(rgb).unsqueeze(0).contiguous()
        with torch.inference_mode():
            pt_logits = predictor.model(batch).numpy()
        ort_logits = session.run(None, {input_name: batch.numpy().astype(np.float32, copy=False)})[0]
        if pt_logits.shape != ort_logits.shape:
            raise RuntimeError(f'Output shape mismatch at image {idx}: {pt_logits.shape} vs {ort_logits.shape}')

        difference = float(np.max(np.abs(pt_logits - ort_logits)))
        max_abs = max(max_abs, difference)
        close = bool(np.allclose(pt_logits, ort_logits, rtol=1e-3, atol=1e-3))
        all_close += int(close)
        pt_label = int(np.argmax(pt_logits, axis=1)[0])
        ort_label = int(np.argmax(ort_logits, axis=1)[0])
        if pt_label != ort_label or not close:
            mismatches.append({'index': idx, 'pytorch': predictor.classes[pt_label],
                               'onnx': predictor.classes[ort_label],
                               'max_absolute_logit_difference': difference,
                               'logits_close': close})
        truths.append(int(record['labels']))
        pt_preds.append(pt_label)
        ort_preds.append(ort_label)

    labels = list(range(len(predictor.classes)))
    result = {
        'checkpoint': args.checkpoint,
        'onnx': args.onnx,
        'model_version': predictor.version,
        'dataset': DATASET_ID,
        'split': 'validation',
        'num_examples': len(truths),
        'same_top1_count': int(sum(x == y for x, y in zip(pt_preds, ort_preds))),
        'same_top1_fraction': float(np.mean(np.array(pt_preds) == np.array(ort_preds))),
        'logits_close_count_at_rtol_1e-3_atol_1e-3': all_close,
        'max_absolute_logit_difference': max_abs,
        'pytorch_accuracy': float(accuracy_score(truths, pt_preds)),
        'onnx_accuracy': float(accuracy_score(truths, ort_preds)),
        'pytorch_macro_f1': float(f1_score(truths, pt_preds, labels=labels, average='macro', zero_division=0)),
        'onnx_macro_f1': float(f1_score(truths, ort_preds, labels=labels, average='macro', zero_division=0)),
        'differences': mismatches,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))
    print(f'\nSaved: {output}')
    if mismatches:
        raise SystemExit('CHECK FAILED: Investigate backend mismatches before enabling ONNX serving.')


if __name__ == '__main__':
    main()
