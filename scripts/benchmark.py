"""Measure end-to-end sequential HTTP latency (NOT a throughput/load test)."""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import httpx
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True, help="Path to JPEG/PNG/WebP example")
    p.add_argument("--url", default="http://127.0.0.1:8000")
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--runs", type=int, default=100)
    args = p.parse_args()
    if args.runs < 1:
        p.error("--runs must be positive")
    img = Path(args.image)
    data = img.read_bytes()
    mime = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp"}.get(img.suffix.lower())
    if not mime:
        p.error("Provide .jpg, .jpeg, .png, or .webp image")
    timings = []
    with httpx.Client(timeout=30.0) as client:
        for i in range(args.warmup + args.runs):
            start = time.perf_counter()
            response = client.post(args.url.rstrip("/") + "/predict", files={"file": (img.name, data, mime)})
            delta = (time.perf_counter() - start) * 1000
            response.raise_for_status()
            if i >= args.warmup:
                timings.append(delta)
    results = {"url": args.url, "image": str(img), "warmup_requests": args.warmup,
               "measured_requests": args.runs, "concurrency": 1,
               "mean_ms": round(statistics.mean(timings), 3),
               "p50_ms": round(float(np.percentile(timings, 50)), 3),
               "p95_ms": round(float(np.percentile(timings, 95)), 3)}
    Path("results").mkdir(exist_ok=True)
    Path("results/http_latency.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
