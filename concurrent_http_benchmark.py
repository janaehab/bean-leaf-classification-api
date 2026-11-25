"""Benchmark a FastAPI /predict endpoint under controlled client concurrency.

This measures end-to-end HTTP request latency from a single local client.
It does not establish cloud throughput or multi-client performance.

Usage:
  python concurrent_http_benchmark.py --url http://127.0.0.1:8001 \
      --image sample_leaf.png --concurrency 5 --runs 150 \
      --output results/onnx_concurrency5.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import httpx


def percentile(values: list[float], pct: float) -> float:
    """Linear interpolation, matching numpy's usual percentile method."""
    ordered = sorted(values)
    if not ordered:
        raise ValueError("No successful request latencies")
    pos = (len(ordered) - 1) * pct / 100
    low = int(pos)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (pos - low)


async def benchmark(args: argparse.Namespace) -> dict:
    image_path = Path(args.image)
    image_bytes = image_path.read_bytes()
    mime = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp"}.get(image_path.suffix.lower())
    if mime is None:
        raise ValueError("Image must have a JPEG, PNG, or WebP extension")
    if not image_bytes:
        raise ValueError("Image file is empty")
    url = args.url.rstrip("/") + "/predict"
    latencies_ms: list[float] = []
    errors: list[str] = []
    statuses: Counter[str] = Counter()
    semaphore = asyncio.Semaphore(args.concurrency)
    limits = httpx.Limits(max_connections=args.concurrency, max_keepalive_connections=args.concurrency)
    timeout = httpx.Timeout(args.timeout)

    async with httpx.AsyncClient(limits=limits, timeout=timeout) as client:
        async def request(record: bool) -> None:
            async with semaphore:
                start = time.perf_counter()
                try:
                    response = await client.post(url, files={"file": (image_path.name, image_bytes, mime)})
                    elapsed = (time.perf_counter() - start) * 1000
                    if record:
                        statuses[str(response.status_code)] += 1
                        if response.status_code == 200:
                            latencies_ms.append(elapsed)
                        elif len(errors) < 5:
                            errors.append(f"HTTP {response.status_code}: {response.text[:120]}")
                except httpx.HTTPError as exc:
                    if record:
                        statuses["transport_error"] += 1
                        if len(errors) < 5:
                            errors.append(f"{type(exc).__name__}: {str(exc)[:120]}")

        # Warmup in sequential mode, not included in measurements.
        for _ in range(args.warmup):
            await request(record=False)
        wall_start = time.perf_counter()
        await asyncio.gather(*(request(record=True) for _ in range(args.runs)))
        elapsed_seconds = time.perf_counter() - wall_start

    successes = len(latencies_ms)
    result = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "url": args.url,
        "image": str(image_path),
        "image_bytes": len(image_bytes),
        "warmup_requests": args.warmup,
        "measured_requests": args.runs,
        "concurrency": args.concurrency,
        "successful_requests": successes,
        "failed_requests": args.runs - successes,
        "status_counts": dict(statuses),
        "total_wall_seconds": round(elapsed_seconds, 3),
        "throughput_requests_per_second": round(successes / elapsed_seconds, 3),
        "latency_ms_successful_requests": None if not successes else {
            "mean": round(statistics.mean(latencies_ms), 3),
            "p50": round(percentile(latencies_ms, 50), 3),
            "p95": round(percentile(latencies_ms, 95), 3),
            "p99": round(percentile(latencies_ms, 99), 3),
        },
        "sample_errors": errors,
        "scope": "single-client end-to-end HTTP benchmark; image upload, decoding, preprocessing, inference, and response",
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--concurrency", type=int, required=True)
    parser.add_argument("--runs", type=int, default=150)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.concurrency < 1 or args.runs < 1 or args.warmup < 0 or args.timeout <= 0:
        parser.error("Concurrency, runs, and timeout must be positive; warmup may be zero")
    result = asyncio.run(benchmark(args))
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
