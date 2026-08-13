# --- Base Image ---
FROM python:3.10-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/huggingface \
    NLTK_DATA=/app/.cache/nltk_data

WORKDIR /app

# System deps needed to build some Python wheels (kept minimal).
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

# --- Python dependencies ---
COPY requirements-api.txt .
# CPU-only torch keeps the API image smaller and avoids downloading CUDA
# runtime packages during Docker builds.
ARG TORCH_VERSION=2.2.2+cpu
RUN pip install --no-cache-dir --timeout 1000 --retries 10 \
        --trusted-host pypi.org \
        --trusted-host files.pythonhosted.org \
        --trusted-host download.pytorch.org \
        --index-url https://download.pytorch.org/whl/cpu \
        --extra-index-url https://pypi.org/simple \
        "torch==${TORCH_VERSION}" \
    && pip install --no-cache-dir --timeout 1000 --retries 10 \
        --trusted-host pypi.org \
        --trusted-host files.pythonhosted.org \
        -r requirements-api.txt

# Pre-download NLTK corpora used by src/data/preprocessing.py so the
# container works fully offline at runtime.
RUN python -m nltk.downloader -d /app/.cache/nltk_data punkt punkt_tab stopwords wordnet omw-1.4

# --- Application code ---
COPY src/ /app/src/
COPY api/ /app/api/
COPY config.yaml /app/

# Trained model artifacts are expected to be mounted or copied in separately
# (e.g., via docker-compose.yml), since they are large binary files excluded
# from the git repo and the build context via .dockerignore.
RUN mkdir -p /app/models

# Create a non-root user for defense-in-depth
RUN useradd --create-home --uid 1000 appuser \
    && chown -R appuser:appuser /app
USER appuser

ENV MODEL_TYPE=bert
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
