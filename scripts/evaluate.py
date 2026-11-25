"""Final held-out test evaluation. Do not tune hyperparameters on test results."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader

from image_api.model import ModelPredictor
from scripts.train import BeanDataset, DATASET_ID


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="artifacts/beans_mobilenetv3.pt")
    p.add_argument("--batch-size", type=int, default=16)
    args = p.parse_args()
    predictor = ModelPredictor(args.checkpoint)
    classes = predictor.classes
    ds = load_dataset(DATASET_ID)
    test = DataLoader(BeanDataset(ds["test"], predictor.transform), batch_size=args.batch_size, num_workers=0)
    ground_truth, predicted = [], []
    with torch.inference_mode():
        for x, y in test:
            ground_truth.extend(y.tolist())
            predicted.extend(predictor.model(x).argmax(1).tolist())
    indices = list(range(len(classes)))
    report = classification_report(ground_truth, predicted, labels=indices, target_names=classes,
                                   output_dict=True, zero_division=0)
    counts = np.bincount([int(row["labels"]) for row in ds["train"]], minlength=len(classes))
    majority = int(np.argmax(counts))
    result = {
        "dataset": DATASET_ID,
        "split": "test",
        "num_examples": len(ground_truth),
        "model_version": predictor.version,
        "accuracy": float(accuracy_score(ground_truth, predicted)),
        "macro_f1": float(f1_score(ground_truth, predicted, labels=indices, average="macro", zero_division=0)),
        "majority_class_baseline_accuracy": float(accuracy_score(ground_truth, [majority]*len(ground_truth))),
        "majority_class_baseline_macro_f1": float(f1_score(ground_truth, [majority]*len(ground_truth), labels=indices, average="macro", zero_division=0)),
        "classification_report": report,
        "confusion_matrix": confusion_matrix(ground_truth, predicted, labels=indices).tolist(),
    }
    out = Path("results"); out.mkdir(exist_ok=True)
    (out / "test_metrics.json").write_text(json.dumps(result, indent=2))
    with (out / "confusion_matrix.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["true\\pred", *classes])
        for label, cells in zip(classes, result["confusion_matrix"]):
            writer.writerow([label, *cells])
    print(json.dumps({k:v for k,v in result.items() if k not in ("classification_report", "confusion_matrix")}, indent=2))
    print("Saved results/test_metrics.json and results/confusion_matrix.csv")


if __name__ == "__main__":
    main()
