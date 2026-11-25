# ONNX Runtime API Integration — for AI Image Classification Starter

This add-on was prepared against the **original starter `image_api/app.py`**, and uses
`artifacts/beans_frozen_seed21.pt` and the corresponding `.onnx` / `.onnx.data` pair.
It does **not** train or replace any model checkpoints.

## Install safely

1. Stop Uvicorn (`Ctrl+C`).
2. Make a backup using `Copy-Item image_api\app.py image_api\app_before_onnx.bak`. The `.bak` file will not be imported as Python code.
3. Copy this add-on's `image_api/app.py` to your `image_api/app.py` (replace) and `image_api/onnx_model.py` to the same folder.
4. Copy `scripts/write_onnx_metadata.py` and `tests/test_onnx_api.py` to their folders; copy `requirements-onnx-api.txt` and `Dockerfile.onnx` to project root.
5. Verify both the `.onnx` and `.onnx.data` files are still together in `artifacts/`.

## Generate metadata (requires existing Torch install, no retraining)

```powershell
python -m scripts.write_onnx_metadata --checkpoint artifacts/beans_frozen_seed21.pt --onnx artifacts/beans_frozen_seed21.onnx
```

Install ONNX deps into existing `.venv` if needed (you already did this):

```powershell
python -m pip install -r requirements-onnx-api.txt
```

## Run and verify ONNX API on port 8001

Run these in the same PowerShell terminal:

```powershell
$env:MODEL_BACKEND = "onnx"
$env:ONNX_MODEL_PATH = "artifacts/beans_frozen_seed21.onnx"
$env:ONNX_THREADS = "2"
python -m uvicorn image_api.app:app --port 8001
```

In another terminal, activate your venv, then:

```powershell
Invoke-RestMethod http://127.0.0.1:8001/ready
Invoke-RestMethod http://127.0.0.1:8001/model-info
curl.exe -X POST "http://127.0.0.1:8001/predict" -F "file=@sample_leaf.png"
python -m pytest -q
python -m scripts.benchmark --url http://127.0.0.1:8001 --image sample_leaf.png --runs 100
```

The API should report `backend: "onnx"` and `version: "beans-mnv3-seed21"`.
Do not compare model-only latency with HTTP latency; compare HTTP to HTTP under comparable conditions.

For an apples-to-apples HTTP comparison, keep the original PyTorch server running on port 8000 with:

```powershell
$env:MODEL_BACKEND = "pytorch"
$env:MODEL_PATH = "artifacts/beans_frozen_seed21.pt"
python -m uvicorn image_api.app:app --port 8000
```

Each terminal has independent environment settings. Benchmark both ports using the same command, image, and request count.

## ONNX-only Docker image (does not install PyTorch)

```powershell
docker build -f Dockerfile.onnx -t bean-api-onnx .
docker run --rm -p 8001:8000 bean-api-onnx
```

Do not include personal training data in a public Docker image or public repository.
Your root `.dockerignore` excludes `.venv`, tests, and results; `Dockerfile.onnx` copies the named model artifacts only.

## Caveats

- ONNX preprocessing is a NumPy/Pillow reimplementation of the selected torchvision ImageNet transformations. The additional test compares tensors numerically. Compare predictions between the two APIs as well before deployment.
- Validation agreement on 133 samples is already established for the *same tensors* in the prior backend experiment, not for this new HTTP preprocessing path.
- CPU tests and localhost response times do not predict cloud latency or concurrent throughput. Follow with load testing.
- This add-on assumes an ONNX file with a single image input, a single three-class output, fixed batch size 1, and external `.onnx.data` weights.

## Optional: end-to-end prediction parity on the whole validation split

With both servers running on the two ports, the following verifies the actual
API paths (including two implementations of image preprocessing):

```powershell
python -m scripts.compare_api_backends --torch-url http://127.0.0.1:8000 --onnx-url http://127.0.0.1:8001
```

This needs the training dependencies (`datasets`, `httpx`), which are installed in
your current development `.venv`. The Docker ONNX runtime does not need them.

Expected behavior is 133 matching top-1 predictions; the result must be
**measured**, not assumed. The script saves `results/http_equivalence.json`.
