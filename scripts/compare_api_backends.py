"""Verify end-to-end HTTP prediction equivalence on the Beans validation split.

Requires TWO running servers, one PyTorch and one ONNX Runtime. Compares
both preprocessing and model outputs; does not evaluate on the test split.
"""
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import httpx
from datasets import load_dataset

DATASET = "AI-Lab-Makerere/beans"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--torch-url", default="http://127.0.0.1:8000")
    parser.add_argument("--onnx-url", default="http://127.0.0.1:8001")
    parser.add_argument("--limit", type=int, default=0, help="0 means all 133 validation samples")
    parser.add_argument("--output", default="results/http_equivalence.json")
    args = parser.parse_args()

    validation = load_dataset(DATASET, split="validation")
    if args.limit > 0:
        validation = validation.select(range(min(args.limit, len(validation))))
    exact_matches = 0
    largest_score_difference = 0.0
    differences = []
    with httpx.Client(timeout=30.0) as http:
        for index, row in enumerate(validation):
            buf = io.BytesIO()
            row["image"].convert("RGB").save(buf, format="PNG")
            upload = {"file": ("leaf.png", buf.getvalue(), "image/png")}
            pt_response = http.post(args.torch_url.rstrip("/") + "/predict", files=upload)
            ort_response = http.post(args.onnx_url.rstrip("/") + "/predict", files=upload)
            pt_response.raise_for_status()
            ort_response.raise_for_status()
            pt = pt_response.json()
            ort = ort_response.json()
            if pt["label"] == ort["label"]:
                exact_matches += 1
            else:
                differences.append({"index": index, "torch": pt["label"], "onnx": ort["label"]})
            if pt["scores"].keys() != ort["scores"].keys():
                raise ValueError(f"Different output class mapping for sample {index}")
            largest_score_difference = max(
                largest_score_difference,
                *(abs(float(pt["scores"][k]) - float(ort["scores"][k])) for k in pt["scores"])
            )
    report = {
        "dataset": DATASET, "split": "validation", "num_examples": len(validation),
        "same_top1_count": exact_matches,
        "same_top1_fraction": exact_matches / len(validation),
        "max_absolute_probability_difference": round(largest_score_difference, 8),
        "differences": differences,
        "note": "End-to-end HTTP equivalence including decoding/preprocessing, not a latency benchmark",
    }
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Saved: {target}")
    if differences:
        raise SystemExit("Different predictions detected: investigate before deployment")


if __name__ == "__main__":
    main()
