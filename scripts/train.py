"""Fine-tune MobileNetV3-Small on the Beans dataset (no test-set access)."""
from __future__ import annotations

import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from sklearn.metrics import accuracy_score, f1_score

from image_api.model import build_model, WEIGHTS

DATASET_ID = "AI-Lab-Makerere/beans"


class BeanDataset(Dataset):
    def __init__(self, split, transform):
        self.split = split
        self.transform = transform

    def __len__(self):
        return len(self.split)

    def __getitem__(self, idx):
        row = self.split[int(idx)]
        return self.transform(row["image"].convert("RGB")), int(row["labels"])


def evaluate(model, loader, device):
    model.eval()
    truth, preds = [], []
    with torch.inference_mode():
        for images, labels in loader:
            scores = model(images.to(device))
            preds.extend(scores.argmax(dim=1).cpu().tolist())
            truth.extend(labels.tolist())
    return {"accuracy": float(accuracy_score(truth, preds)),
            "macro_f1": float(f1_score(truth, preds, labels=list(range(3)), average="macro", zero_division=0))}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--output", default="artifacts/beans_mobilenetv3.pt")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--unfreeze-last-block", action="store_true", help="Fine-tune last feature block as well")
    args = p.parse_args()
    if args.epochs < 1:
        p.error("--epochs must be positive")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from datasets import load_dataset  # only training requires this dependency
    data = load_dataset(DATASET_ID)
    names = list(data["train"].features["labels"].names)
    if len(names) != 3:
        raise ValueError(f"Expected 3 classes, got {names}")
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.75, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    train = DataLoader(BeanDataset(data["train"], train_transform), batch_size=args.batch_size, shuffle=True, num_workers=0)
    val = DataLoader(BeanDataset(data["validation"], WEIGHTS.transforms()), batch_size=args.batch_size, num_workers=0)
    model = build_model(len(names), pretrained=True).to(device)
    for parameter in model.features.parameters():
        parameter.requires_grad = False
    if args.unfreeze_last_block:
        for parameter in model.features[-1].parameters():
            parameter.requires_grad = True
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    best = -1.0
    log = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        # Frozen feature extractor must keep its batch-norm stats unchanged.
        model.features.eval()  # frozen blocks retain ImageNet BatchNorm statistics
        if args.unfreeze_last_block:
            model.features[-1].train()  # only the unfrozen final block updates its stats
        losses = []
        for x, y in train:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        metrics = evaluate(model, val, device)
        row = {"epoch": epoch, "train_loss": round(float(np.mean(losses)), 5), **metrics}
        log.append(row)
        print(json.dumps(row))
        if metrics["macro_f1"] > best:
            best = metrics["macro_f1"]
            torch.save({
                "model_state_dict": model.state_dict(),
                "classes": names,
                "version": f"beans-mnv3-seed{args.seed}",
                "metadata": {"dataset": DATASET_ID, "seed": args.seed,
                             "epochs_completed": epoch, "validation": metrics,
                             "unfreeze_last_block": args.unfreeze_last_block,
                             "created_at": datetime.now(timezone.utc).isoformat()},
            }, path)
            print(f"Best checkpoint saved: {path}")
    Path("results").mkdir(exist_ok=True)
    Path("results/train_history.json").write_text(json.dumps(log, indent=2))
    print("Validation results only. Run scripts/evaluate.py ONCE for the held-out test set.")


if __name__ == "__main__":
    main()
