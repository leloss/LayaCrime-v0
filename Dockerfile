FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_OFFLINE=1 \
    LAYA_HOST=0.0.0.0 \
    LAYA_PORT=8000 \
    LAYA_ENABLE_FINE_TUNING=false \
    LAYA_MODEL_BUNDLE=/models/laya \
    LAYA_FINE_TUNED_MODELS_DIR=/models/fine-tuned

WORKDIR /app

RUN apt-get update \
    && apt-get install --yes --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY datasets/README.md ./datasets/README.md
COPY datasets/adverse-media-public-tuning-2000 ./datasets/adverse-media-public-tuning-2000
COPY datasets/adverse-media-public-holdout-1000 ./datasets/adverse-media-public-holdout-1000
RUN python -m pip install . \
    && useradd --create-home --uid 10001 app \
    && mkdir -p /app/artifacts /models \
    && chown -R app:app /app/artifacts /models

USER app
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=5m --retries=3 \
    CMD python -c "import json, urllib.request; data=json.load(urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)); raise SystemExit(0 if data.get('status') == 'ok' else 1)"

CMD ["laya-adverse-media"]
