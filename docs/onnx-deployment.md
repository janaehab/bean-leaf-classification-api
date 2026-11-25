# ONNX Runtime: serving, validation, and Docker

[← Back to project overview](../README.md)

This guide describes the **implemented ONNX Runtime serving path** for the Bean Leaf Classification API. It is intended for engineers who want to run the container, reproduce the CPU-backend comparison, or understand the ONNX artifact layout. For modeling results, architecture, and limitations, see the [main README](../README.md).

> **Status:** ONNX inference has been validated locally, including a Docker container. This repository does **not** claim public-cloud deployment or production readiness.

## 1. Model artifacts

The ONNX-only runtime uses three tracked files:

| File | Role |
|---|---|
| `artifacts/beans_frozen_seed21.onnx` | Serialized computation graph |
| `artifacts/beans_frozen_seed21.onnx.data` | External model weights — **required** alongside the graph |
| `artifacts/beans_frozen_seed21.onnx.metadata.json` | Labels, model version, and preprocessing settings |

All three files are copied by [`Dockerfile.onnx`](../Dockerfile.onnx). Keep the graph and external weights in the same directory. The `.pt` training checkpoint is intentionally excluded from Git; **it is not needed to run the prepackaged ONNX service**.

The API loads this graph at startup using `ONNX_MODEL_PATH` and creates an ONNX Runtime `CPUExecutionProvider` session. The ONNX serving container does not install PyTorch or torchvision. Image preprocessing is implemented with Pillow and NumPy in [`image_api/onnx_model.py`](../image_api/onnx_model.py).

## 2. Quick start: Docker (recommended)

From the repository root, with Docker Desktop / Docker Engine running:

```powershell
docker build -f Dockerfile.onnx -t bean-leaf-onnx:local .
docker run --rm -p 8002:8000 bean-leaf-onnx:local
```

The container listens on port **8000** internally; port **8002** is the host port used here to avoid conflicts with local development servers. Leave this terminal running while testing the service.

Open `http://127.0.0.1:8002/docs` for Swagger UI, or run in a **second** terminal from the repository root:

```powershell
curl.exe http://127.0.0.1:8002/ready
curl.exe http://127.0.0.1:8002/model-info
curl.exe -X POST "http://127.0.0.1:8002/predict" -F "file=@sample_leaf.png"
```

`/ready` should respond with `{"status":"ready"}`. `/model-info` should identify version `beans-mnv3-seed21` and backend `onnx`. `/predict` returns a label, model softmax confidence, per-class scores, and model version. These confidences are **not calibrated disease probabilities**.

To stop the container, press **Ctrl+C** in its terminal.

## 3. Run ONNX locally without Docker

Use a supported Python environment (the included Dockerfile uses Python 3.12). From the project root, activate your virtual environment and install the serving dependencies:

```powershell
python -m pip install -r requirements-onnx-api.txt
```

In the **same PowerShell terminal**:

```powershell
$env:MODEL_BACKEND = "onnx"
$env:ONNX_MODEL_PATH = "artifacts/beans_frozen_seed21.onnx"
$env:ONNX_THREADS = "2"
python -m uvicorn image_api.app:app --port 8001
```

The corresponding endpoints are available at `http://127.0.0.1:8001/ready`, `/model-info`, `/predict`, and `/docs`.

> These `$env:` commands use **PowerShell**, not Windows CMD. Environment variables are terminal-specific. Avoid `--reload` during latency benchmarks because auto-restarts can distort measurements.

## 4. Compare PyTorch and ONNX end to end

This optional step requires a **local** `artifacts/beans_frozen_seed21.pt` checkpoint, which is **not included in Git**. If you only want to run the published ONNX API, skip this section.

Install training/test dependencies into the development environment:

```powershell
python -m pip install -r requirements-training.txt
python -m pip install -r requirements-onnx-api.txt
```

Start the PyTorch server in a **separate PowerShell terminal**:

```powershell
$env:MODEL_BACKEND = "pytorch"
$env:MODEL_PATH = "artifacts/beans_frozen_seed21.pt"
python -m uvicorn image_api.app:app --port 8000
```

Keep ONNX running on port 8001 (Section 3). In a **third terminal**, compare predictions through the real HTTP endpoints:

```powershell
python -m scripts.compare_api_backends --torch-url http://127.0.0.1:8000 --onnx-url http://127.0.0.1:8001
```

The recorded local validation run produced **133/133 matching top-1 predictions**, with a maximum absolute difference of `0.000003` between reported probabilities. Evidence: [`results/http_equivalence.json`](../results/http_equivalence.json). This is validation-set equivalence, not a guarantee of exact agreement on every possible input.

### Model-only CPU comparison

With the same checkpoint, you can reproduce the inference-only comparison (excludes upload, decoding, preprocessing, and JSON serialization):

```powershell
python -m scripts.compare_backends --checkpoint artifacts/beans_frozen_seed21.pt --onnx artifacts/beans_frozen_seed21.onnx --image sample_leaf.png --runs 150 --threads 2
```

Recorded result: **24.85 ms** mean PyTorch inference versus **4.77 ms** mean ONNX inference, or a **5.21×** mean model-only speedup in this local run. Evidence: [`results/backend_benchmark.json`](../results/backend_benchmark.json).

The end-to-end sequential HTTP experiment separately recorded **76.02 ms** mean for PyTorch versus **39.39 ms** for ONNX. Do **not** treat the model-only figure as an API speedup. Threading, hardware, and system load should be normalized for a more rigorous comparison.

## 5. Load-test the Docker service

The included script measures single-client HTTP throughput and latency over a fixed request count. It is a local development benchmark, not a production-scale test.

With the container from Section 2 running and the project's development dependencies installed:

```powershell
python concurrent_http_benchmark.py --url http://127.0.0.1:8002 --image sample_leaf.png --concurrency 5 --runs 3000 --output results/docker_load_c5.json
```

Previously recorded local Docker Desktop measurements:

| Concurrency | Requests | Throughput | p95 latency | Failed requests |
|---:|---:|---:|---:|---:|
| 1 | 1,500 | 14.13 req/s | 90.74 ms | 0 |
| 5 | 3,000 | 58.85 req/s | 119.71 ms | 0 |
| 10 | 3,000 | 64.19 req/s | 228.43 ms | 0 |

The tests used the **same 558 KB image repeatedly**, on one development computer using Docker Desktop. Larger and more varied workloads, consistent CPU allocations, repeated runs, and cloud-side measurements are needed before claiming sustained production performance. See [`results/`](../results/) for the saved benchmark files.

## 6. Regenerate the ONNX artifacts (optional)

Re-exporting requires the PyTorch checkpoint. To reproduce the selected frozen-feature training configuration, run from the root:

```powershell
python -m pip install -r requirements-training.txt
python -m scripts.train --epochs 8 --batch-size 8 --lr 0.0001 --seed 21 --output artifacts/beans_frozen_seed21.pt
python -m pip install onnx onnxscript onnxruntime
python -m scripts.export_onnx --checkpoint artifacts/beans_frozen_seed21.pt --output artifacts/beans_frozen_seed21.onnx
python -m scripts.write_onnx_metadata --checkpoint artifacts/beans_frozen_seed21.pt --onnx artifacts/beans_frozen_seed21.onnx
```

This can overwrite local outputs. Back up artifacts before regenerating them. Export may write **both** `.onnx` and `.onnx.data`; verify both are present before rebuilding the Docker image. Check prediction equivalence again after re-export.

## 7. Troubleshooting

| Symptom | Check |
|---|---|
| `/ready` returns 503 | Ensure the graph, external weights, and metadata are available at the configured path; review startup logs |
| `FileNotFoundError` loading ONNX | Ensure `.onnx` and `.onnx.data` were not separated or renamed |
| Wrong `backend` in `/model-info` | Set `MODEL_BACKEND=onnx` in the terminal running Uvicorn, then restart it |
| Port 8002 already in use | Stop the prior container or map a different host port, e.g. `-p 8003:8000` |
| `docker` command not found | Install Docker Desktop and reopen the terminal; verify `docker info` shows the server |
| Docker build fails in clean CI | Confirm `artifacts/` contains the graph, `.onnx.data`, and metadata tracked in Git |
| PyTorch comparison fails | Retrain / restore the local `.pt` checkpoint; it is intentionally untracked |

## 8. Operational limitations

- No public-cloud deployment is currently claimed.
- The API has file-type, byte-count, and pixel-count checks, but that alone does not make it suitable for exposure to untrusted public traffic. Add authentication, rate limits, monitoring, dependency vulnerability scanning, and appropriate resource limits before production use.
- The ONNX Dockerfile uses a non-root user and a `/ready` health check. A successful Docker build and health check are not substitutes for security review.
- This is a portfolio classifier evaluated on a small curated dataset, **not** a validated crop-disease diagnostic tool.

For the benchmark tables, model-selection experiments, and project overview, return to the [main README](../README.md).
