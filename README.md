# Bean Leaf Image Classification API — Engineering Starter

> **Status:** Starter implementation, not a trained or deployed model. There are no real evaluation or latency results yet. Complete all steps and publish your own measurements before claiming them on your CV.

A reproducible end-to-end starter project for **bean-leaf disease image classification**. It trains a transfer-learning classifier, evaluates a held-out dataset, serves predictions via FastAPI, and includes Docker, tests, CI, and HTTP latency measurement.

## Dataset

[Makerere AI Lab Beans dataset](https://huggingface.co/datasets/AI-Lab-Makerere/beans): `angular_leaf_spot`, `bean_rust`, and `healthy`. Source splits: 1,034 train, 133 validation, and 128 test. [License: MIT](https://huggingface.co/datasets/AI-Lab-Makerere/beans/blob/main/README.md). Do not claim performance on real-world plant diagnosis without representative testing; these images come from one curated dataset.

## 1. Install

Use **Python 3.11** in a fresh virtual environment. This project needs internet access *once* to fetch training data and pretrained ImageNet weights.

```bash
python -m venv .venv
# macOS / Linux:
source .venv/bin/activate
# Windows PowerShell:
# .venv\Scripts\Activate.ps1
pip install -r requirements-training.txt
```

If you encounter problems installing PyTorch for your machine (especially a GPU), use the [official PyTorch installer instructions](https://pytorch.org/get-started/locally/) for compatible Torch/TorchVision builds before installing the remaining dependencies.

## 2. Train (feature extraction)

```bash
python -m scripts.train --epochs 3 --batch-size 16
```

This loads pretrained MobileNetV3-Small, freezes its feature extractor, and trains the classification head. It selects the best checkpoint based on **validation macro-F1**, saving `artifacts/beans_mobilenetv3.pt`.

Optional experiment (not required for today):

```bash
python -m scripts.train --epochs 3 --unfreeze-last-block --output artifacts/beans_mnv3_finetuned.pt
```

**Do not touch the test set for model choice or hyperparameter tuning.** The provided test evaluation is a final check.

## 3. Test API

```bash
pytest -q
```

The automated API test creates a **random, untrained model fixture** to verify request/response behavior. Passing these tests does **not** demonstrate classification accuracy. The tests verify readiness, missing model behavior, successful prediction output, and invalid input handling.

## 4. Evaluate on held-out test set

Only once the model/hyperparameters are selected using validation results:

```bash
python -m scripts.evaluate --checkpoint artifacts/beans_mobilenetv3.pt
```

Outputs `results/test_metrics.json` and `results/confusion_matrix.csv`, with model accuracy, macro-F1, per-class results, and a majority-class baseline. Your real results will be generated locally; no scores are supplied in this starter.

## 5. Run local API

```bash
uvicorn image_api.app:app --host 127.0.0.1 --port 8000
```

Then visit `http://127.0.0.1:8000/docs` for the interactive API documentation. Health/readiness endpoint semantics:

- `GET /health`: process is alive.
- `GET /ready`: returns HTTP 503 until a valid trained checkpoint has loaded.
- `GET /model-info`: architecture, classes, model version and load status.
- `POST /predict`: JPEG/PNG/WebP upload (`file` multipart form field); max 5 MiB and 20 million image pixels; returns class, confidence, per-class scores, and model version. **Confidence is not calibrated probability** by default.

Example:

```bash
curl -X POST http://127.0.0.1:8000/predict -F "file=@path/to/leaf.jpg"
```

## 6. Benchmark HTTP inference

With the API running, export a **validation** image for a smoke test (not for model selection):

```bash
python -m scripts.export_sample --output sample_leaf.png
```

Then benchmark the same image repeatedly:

```bash
python -m scripts.benchmark --image sample_leaf.png --runs 100 --warmup 10
```

Outputs `results/http_latency.json` with p50/p95 and mean sequential HTTP latencies. Report your CPU/GPU, OS, model version, payload, and whether the API was local or remotely hosted; this is **not** a concurrent load test and does not establish throughput.

## 7. Docker

After training and saving the checkpoint:

```bash
docker build -t bean-leaf-api .
docker run --rm -p 8000:8000 bean-leaf-api
```

Then check `http://127.0.0.1:8000/ready` and `http://127.0.0.1:8000/docs`. The Docker image includes the checkpoint *only if you built it after training*. This starter is not cloud-deployed. Use secrets and endpoint authentication before making it public; avoid exposing diagnostic data about users or accepting untrusted images without further hardening.

## Architecture

```text
Makerere Beans dataset ─> image transforms ─> pretrained MobileNetV3-Small
                          └─> frozen feature extractor + trained classifier
                                ├─> validation macro-F1 -> best checkpoint
                                ├─> held-out test -> metrics + confusion matrix
                                └─> FastAPI -> image validation -> prediction
                                                └─> Docker -> HTTP benchmark
```

## Engineering scope and limitations

- This starter is **not** a production diagnosis tool.
- All training/validation metrics must come from your own run.
- The default uses frozen ImageNet features for fast training; a second fine-tuned experiment is optional.
- Error handling, health endpoints and tests are provided, but deployment, actual monitoring, CI success, and reproducible benchmark reports still need to be verified on your setup.
- Do not commit downloaded dataset images or model weights without checking licensing/hosting limits.
- For a stronger CV: run real benchmarks, document failures, add cloud deployment only if time permits, and clearly distinguish what you implemented from this starter scaffold.

## Future improvements (not necessary for day one)

- Repeated runs with multiple random seeds, confidence intervals, and documented hyperparameter sweeps.
- Fine-tuning comparison against frozen backbone and majority-class baseline.
- Concurrent load testing with Locust and measured CPU/memory usage.
- ONNX Runtime conversion and fair inference comparison.
- CI deployment job and secure hosted endpoint.
