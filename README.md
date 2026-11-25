# Bean Leaf Classification API

**CPU-optimized image classification with PyTorch, ONNX Runtime, FastAPI, and Docker.**

An end-to-end ML engineering project that trains a three-class bean-leaf classifier, compares transfer-learning strategies, validates model conversion, and serves predictions through a tested HTTP API. The selected model uses **MobileNetV3-Small** and is packaged for CPU inference with **ONNX Runtime**.

> **Scope:** This is a portfolio/demo classifier evaluated on the Makerere Beans dataset, **not** a validated plant-disease diagnostic product. Cloud hosting has not yet been completed. Reported performance is from local experiments.

## Results at a glance

| Measure | Result | Evaluation context |
|---|---:|---|
| **Test accuracy** | **87.50%** | 112 / 128 held-out test images |
| **Test macro-F1** | **87.66%** | Three classes, equal class weighting |
| Best selected-checkpoint validation macro-F1 | 93.24% | 133 validation images; seed 21 |
| ONNX–PyTorch prediction agreement | **133 / 133** | Validation images sent to *both HTTP APIs* |
| Mean model-only CPU latency | 24.85 → **4.77 ms** | PyTorch → ONNX Runtime; 150 runs, 2 configured threads per backend |
| Mean sequential local HTTP latency | 76.02 → **39.39 ms** | PyTorch → ONNX; 150 requests per backend |
| Containerized throughput (concurrency 5) | **58.85 requests/s** | 3,000 requests, localhost through Docker Desktop |
| Containerized test request failures | **0 / 7,500** | Three local load runs at concurrency 1, 5, and 10 |

**Interpreting the speedup:** ONNX was **5.21× faster for model-only inference** and **1.93× faster for mean sequential localhost HTTP responses** in these particular runs. The comparisons are preliminary, not hardware-normalized cloud benchmarks; PyTorch and ONNX server threading settings may differ, and HTTP benchmarks used a single repeated image. See [Performance and limitations](#performance-and-limitations).

Recorded evidence: [`test_metrics_frozen_seed21.json`](results/test_metrics_frozen_seed21.json), [`onnx_validation.json`](results/onnx_validation.json), [`http_equivalence.json`](results/http_equivalence.json), [`backend_benchmark.json`](results/backend_benchmark.json), and [`docker_load_c5.json`](results/docker_load_c5.json).

## Architecture

```text
Makerere Beans dataset (train / validation / test)
       │
       ├── Augmentation + preprocessing
       ▼
MobileNetV3-Small (ImageNet pretrained)
       │
       ├── Frozen features vs. partial fine-tuning
       ├── 3 random seeds per configuration
       └── Checkpoint selection by validation macro-F1
                    │
                    ▼
        Selected PyTorch checkpoint
                    │
             Export to ONNX
                    │
            ONNX model + weights
                    │
  Upload ──► FastAPI image validation ──► ONNX Runtime CPU inference
                    │
                    ▼
       Class label + scores + model version
                    │
             Docker container
```

The API also supports a **PyTorch** backend for equivalence and performance comparisons. ONNX serving uses NumPy/Pillow for preprocessing and does **not** require PyTorch in the ONNX Docker image.

## Data and modeling

- **Dataset:** [Makerere AI Lab Beans](https://huggingface.co/datasets/AI-Lab-Makerere/beans).
- **Classes:** `angular_leaf_spot`, `bean_rust`, `healthy`.
- **Official splits:** 1,034 train / 133 validation / 128 test images.
- **Model:** ImageNet-pretrained `mobilenet_v3_small`, with a new three-class classification head.
- **Selection criterion:** Highest **validation macro-F1** within each training run. The held-out test set is for final reporting, not hyperparameter selection.

### Controlled training comparison

Frozen-feature training and partial fine-tuning were compared across seeds **42, 7, and 21** (eight epochs, batch size 8, learning rate 0.0001).

| Seed | Frozen features: best val macro-F1 | Partial fine-tuning: best val macro-F1 |
|---|---:|---:|
| 42 | 92.50% | 92.52% |
| 7 | 92.51% | 92.43% |
| 21 | **93.24%** | 93.23% |
| **Mean across seeds** | **92.75%** | **92.73%** |

Partial fine-tuning did **not** show a consistent validation advantage in these experiments, so the frozen-feature **seed-21** checkpoint was selected for serving. Three seeds and a small validation split are insufficient to establish general superiority. See the training histories in [`results/`](results/).

### Held-out test performance and errors

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Angular leaf spot | 83.33% | 81.40% | 82.35% | 43 |
| Bean rust | 80.00% | 83.72% | 81.82% | 43 |
| Healthy | 100.00% | 97.62% | 98.80% | 42 |

Of the **16** mistakes on the 128-image test split, **15** were confusions between angular leaf spot and bean rust. The model performs better on healthy leaves than on distinguishing those two diseases. The test-set result is less favorable than the selected validation result; generalization beyond this small curated dataset is not established.

## Quick start: run the ONNX API in Docker

Requirements: Docker with the engine running. The repository includes the ONNX graph, its **external `.onnx.data` weights**, and the metadata file.

```bash
docker build -f Dockerfile.onnx -t bean-leaf-onnx:local .
docker run --rm -p 8002:8000 bean-leaf-onnx:local
```

Then open [`http://127.0.0.1:8002/docs`](http://127.0.0.1:8002/docs) or check readiness:

```bash
curl http://127.0.0.1:8002/ready
curl http://127.0.0.1:8002/model-info
curl -X POST http://127.0.0.1:8002/predict -F "file=@sample_leaf.png"
```

A successful prediction includes `label`, `confidence`, `scores`, and `model_version`. **The confidence score is a model softmax output, not a calibrated disease probability.**

### API reference

| Endpoint | Behavior |
|---|---|
| `GET /health` | Liveness check |
| `GET /ready` | Model readiness; returns `503` if unavailable |
| `GET /model-info` | Loaded model version, classes, architecture and inference backend |
| `POST /predict` | Multipart `file` upload; classification response |

Supported file types: JPEG, PNG, WebP. The API rejects empty, invalid/corrupt, oversized (over 5 MiB), or oversized-pixel-count (over 20 million pixels) images. The softmax scores are not calibrated.

## Development, training, and evaluation

Use a compatible Python environment and install the training dependencies:

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements-training.txt
python -m pip install -r requirements-onnx-api.txt
python -m pytest -q
```

To reproduce the **selected training configuration** (dataset and ImageNet pretrained weights are downloaded automatically):

```bash
python -m scripts.train --epochs 8 --batch-size 8 --lr 0.0001 --seed 21 --output artifacts/beans_frozen_seed21.pt
```

For the corresponding partial fine-tuning experiment, append `--unfreeze-last-block` and choose a different `--output` path. **Training overwrites `results/train_history.json`**; copy it to a run-specific filename after each experiment.

To evaluate a checkpoint (this overwrites the default `results/test_metrics.json` and `results/confusion_matrix.csv`):

```bash
python -m scripts.evaluate --checkpoint artifacts/beans_frozen_seed21.pt
```

The `.pt` checkpoint is intentionally **not included in Git**. To serve or re-export the PyTorch backend, retrain with the command above. For ordinary API use, run the prepackaged ONNX Docker image instead.

## Verification and benchmarking

- **API tests:** `python -m pytest -q` (8 tests passed locally at the time of initial verification).
- **Backends:** Both APIs matched predicted labels on **133/133 validation images**, with a maximum probability difference of approximately **0.000003** through the HTTP path.
- **Model-only latency:** Compared 150 inferences per backend, excluding image preprocessing and HTTP.
- **HTTP latency:** Compared sequential requests using the same sample image.
- **Load tests:** Containerized API received 1,500 requests at concurrency 1; 3,000 each at concurrency 5 and 10, with zero failures in those runs.

Example local load test against the Docker service:

```bash
python concurrent_http_benchmark.py --url http://127.0.0.1:8002 --image sample_leaf.png --concurrency 5 --runs 3000 --output results/docker_load_c5.json
```

### Performance and limitations

| Docker concurrency | Requests | Throughput | Mean latency | p95 latency | Failures |
|---|---:|---:|---:|---:|---:|
| 1 | 1,500 | 14.13 req/s | 70.66 ms | 90.74 ms | 0 |
| 5 | 3,000 | 58.85 req/s | 84.01 ms | 119.71 ms | 0 |
| 10 | 3,000 | 64.19 req/s | 151.94 ms | 228.43 ms | 0 |

Moving from concurrency **5 to 10** increased throughput by about **9.1%** but increased p95 latency by about **90.8%**: a measurable throughput–latency tradeoff under this workload.

**Benchmark limitations:** Results were measured on a Windows development machine using localhost and, for Docker, Docker Desktop's Linux VM. The exact CPU model, thread allocation across all runs, and system load were not standardized. The same 558 KB image was repeatedly uploaded. These are local performance observations—not service-level objectives, production reliability claims, or cloud performance guarantees.

## CI and deployment status

GitHub Actions is configured to check Python syntax, install testing/ONNX dependencies, run API tests, and build/start the Docker image with a readiness check. **Confirm the workflow status in the repository's Actions tab**; a successful local run is not evidence that CI has passed.

The service has been built and run **locally in Docker**. It has **not yet been deployed to a public cloud environment**. Authentication, rate limiting, systematic security scanning, external monitoring, and more representative real-world testing would be required before public production use.

## Repository guide

```text
image_api/                  FastAPI service and PyTorch/ONNX predictors
scripts/                    Training, evaluation, export and comparison scripts
tests/                      API and backend tests
artifacts/                  Tracked ONNX graph, external weights and metadata
results/                    Selected model evaluation and benchmark outputs
Dockerfile.onnx             CPU-only ONNX serving image
Dockerfile                  Original PyTorch serving image (requires local .pt checkpoint)
concurrent_http_benchmark.py  Concurrent HTTP load testing
.github/workflows/ci.yml    Continuous integration workflow
```

**Technology:** Python · PyTorch / torchvision · ONNX Runtime · FastAPI · NumPy · Pillow · scikit-learn · Docker · pytest · GitHub Actions
