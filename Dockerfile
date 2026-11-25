FROM python:3.11-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 MODEL_PATH=/app/artifacts/beans_mobilenetv3.pt
# CPU-only Torch wheels keep the image smaller than GPU-enabled wheels.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.10.0 torchvision==0.25.0
COPY requirements-api.txt .
RUN grep -vE '^(torch|torchvision)==' requirements-api.txt > /tmp/api-deps.txt && pip install --no-cache-dir -r /tmp/api-deps.txt
COPY image_api/ image_api/
COPY artifacts/ artifacts/
EXPOSE 8000
CMD ["uvicorn", "image_api.app:app", "--host", "0.0.0.0", "--port", "8000"]
