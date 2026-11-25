"""FastAPI serving for either PyTorch or optimized ONNX Runtime backends."""
from __future__ import annotations

import io
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError
from starlette.concurrency import run_in_threadpool

MAX_BYTES = 5 * 1024 * 1024
MAX_PIXELS = 20_000_000
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}
log = logging.getLogger("image_api")


def create_app(model_path: str | None = None, predictor=None, backend: str | None = None, onnx_path: str | None = None) -> FastAPI:
    """Predictor injection keeps tests independent of weights/model downloads."""
    selected_backend = (backend or os.getenv("MODEL_BACKEND", "pytorch")).lower()
    if selected_backend not in {"pytorch", "onnx"}:
        raise ValueError("MODEL_BACKEND must be 'pytorch' or 'onnx'")
    checkpoint = model_path if model_path is not None else os.getenv("MODEL_PATH", "artifacts/beans_mobilenetv3.pt")
    graph = onnx_path if onnx_path is not None else os.getenv("ONNX_MODEL_PATH", "artifacts/beans_frozen_seed21.onnx")
    active_path = graph if selected_backend == "onnx" else checkpoint

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.predictor = predictor
        app.state.load_error = None
        if app.state.predictor is None:
            try:
                if selected_backend == "onnx":
                    from image_api.onnx_model import ONNXPredictor
                    app.state.predictor = ONNXPredictor(Path(graph), threads=int(os.getenv("ONNX_THREADS", "2")))
                else:
                    from image_api.model import ModelPredictor
                    app.state.predictor = ModelPredictor(Path(checkpoint))
            except Exception as exc:
                # Make /ready return 503; never return dummy predictions.
                log.exception("Could not initialize %s model", selected_backend)
                app.state.load_error = type(exc).__name__
        yield

    app = FastAPI(title="LeafLens AI | Bean Leaf Classification API", version="0.3.0", lifespan=lifespan)

    # Static frontend shares this FastAPI origin with /predict, so no separate
    # frontend hosting or CORS configuration is needed.
    frontend_dir = Path(__file__).resolve().parent / "frontend"
    if frontend_dir.is_dir():
        app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")

        @app.get("/", include_in_schema=False)
        def homepage():
            return FileResponse(frontend_dir / "index.html", media_type="text/html")

    @app.get("/health")
    def health():
        return {"status": "alive"}

    @app.get("/ready")
    def ready():
        if app.state.predictor is None:
            return JSONResponse({"status": "not_ready", "reason": "model_unavailable"}, status_code=503)
        return {"status": "ready"}

    @app.get("/model-info")
    def model_info():
        model = app.state.predictor
        if model is None:
            return {"loaded": False, "model_path": active_path, "backend": selected_backend}
        return {"loaded": True, "classes": model.classes, "version": model.version,
                "architecture": "mobilenet_v3_small", "backend": selected_backend}

    @app.post("/predict")
    async def predict(file: UploadFile = File(...)):
        if app.state.predictor is None:
            raise HTTPException(503, detail="Model unavailable")
        if file.content_type not in ALLOWED_TYPES:
            raise HTTPException(415, detail="Allowed: JPEG, PNG, WebP")
        content = await file.read(MAX_BYTES + 1)
        if not content:
            raise HTTPException(400, detail="Empty file")
        if len(content) > MAX_BYTES:
            raise HTTPException(413, detail="Image exceeds 5 MiB limit")
        try:
            with Image.open(io.BytesIO(content)) as im:
                if im.width * im.height > MAX_PIXELS:
                    raise HTTPException(413, detail="Image pixel count exceeds limit")
                im.verify()
        except HTTPException:
            raise
        except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError):
            raise HTTPException(422, detail="Invalid or corrupted image") from None
        try:
            # CPU inference is blocking; keep event loop responsive under concurrent load.
            return await run_in_threadpool(app.state.predictor.predict, content)
        except ValueError:
            raise HTTPException(422, detail="Invalid or corrupted image") from None

    return app


app = create_app()
