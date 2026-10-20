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

LABEL org.opencontainers.image.source=https://github.com/Karan-parmar-007/studio-services
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

USER appuser

EXPOSE 8011

# One worker on purpose: the lanes and queued jobs live in this process's memory.
CMD ["sh", "-c", "exec gunicorn -w 1 --threads 8 --timeout 300 -b \"$BIND\" app:app"]
