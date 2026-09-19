# syntax=docker/dockerfile:1

# ---------- Stage 1: build (dependencies installed here) ----------
FROM python:3.12-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build
COPY requirements.txt .
RUN pip install --prefix=/install -r requirements.txt


# ---------- Stage 2: runtime (only what is needed to run) ----------
FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    LOG_LEVEL=INFO

RUN useradd --create-home --uid 1000 app

WORKDIR /app
COPY --from=builder /install /usr/local
COPY ai/ ./ai/
COPY src/ ./src/
COPY data/ ./data/
COPY demo_ai.py ./

RUN mkdir -p /app/digests /app/artefacts \
    && chown -R app:app /app

USER app

# API keys are NOT baked into the image: pass them at run time (--env-file .env)
ENTRYPOINT ["python", "-m", "src.cli"]
CMD ["run-daily", "--user", "khagani"]