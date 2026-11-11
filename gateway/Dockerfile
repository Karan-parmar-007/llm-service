# LLM half only (SPEECH_ENABLED=0): Flask, gunicorn, httpx. Speech libraries are never
# installed. Built for linux/arm64 (Oracle Ampere LLM VPS) by CI.
FROM python:3.12-slim AS builder

WORKDIR /build

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

FROM python:3.12-slim

LABEL org.opencontainers.image.source=https://github.com/Karan-parmar-007/llm-service
LABEL org.opencontainers.image.description="Studio gateway: LLM in front of Ollama"

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    LLM_ENABLED=1 \
    SPEECH_ENABLED=0 \
    BIND=0.0.0.0:8011

RUN useradd --create-home --uid 1000 appuser

COPY --from=builder /opt/venv /opt/venv
COPY --chown=appuser:appuser *.py ./

# Cross-encoder reranker (int8 ONNX, ~150 MB) baked into the image so /rerank works offline.
ENV RERANK_MODEL_DIR=/opt/reranker
RUN python fetch_reranker.py /opt/reranker && chown -R appuser:appuser /opt/reranker

USER appuser

EXPOSE 8011

# One worker on purpose: the lanes and queued jobs live in this process's memory.
# Threads, not processes: a caller waiting in line (X-Wait) holds one thread, so there
# must be room for every lane's waiting line (5 lanes x LANE_MAX_WAITING) plus streams.
CMD ["sh", "-c", "exec gunicorn -w 1 --threads ${GUNICORN_THREADS:-64} --timeout 330 -b \"$BIND\" app:app"]
