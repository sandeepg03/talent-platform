# ─────────────────────────────────────────────────────────────────
# AI Talent Intelligence Platform — Dockerfile
# ─────────────────────────────────────────────────────────────────
# Two-stage build:
#   Stage 1 (builder): install deps into a venv
#   Stage 2 (runtime): copy venv + src, run as non-root
#
# Build:
#   docker build -t talent-platform:latest .
#
# Run precompute (one-time):
#   docker run --rm -v $(pwd)/data:/app/data -v $(pwd)/artifacts:/app/artifacts \
#     talent-platform:latest python precompute.py
#
# Run rank:
#   docker run --rm -v $(pwd)/data:/app/data -v $(pwd)/artifacts:/app/artifacts \
#     -v $(pwd)/submission.csv:/app/submission.csv \
#     talent-platform:latest python rank.py
#
# Run FastAPI:
#   docker run --rm -p 8000:8000 \
#     -v $(pwd)/data:/app/data -v $(pwd)/artifacts:/app/artifacts \
#     talent-platform:latest
# ─────────────────────────────────────────────────────────────────

FROM python:3.12-slim AS builder

WORKDIR /build

# System deps for faiss-cpu and python-docx
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create venv
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY pyproject.toml ./
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -e ".[all]" 2>/dev/null || \
       pip install --no-cache-dir \
         fastapi \
         uvicorn[standard] \
         streamlit \
         sentence-transformers \
         faiss-cpu \
         python-docx \
         pydantic \
         loguru \
         joblib \
         plotly \
         pandas \
         numpy \
         requests \
         httpx \
         pytest \
         pytest-anyio

# ─────────────────────────────────────────────────────────────────
FROM python:3.12-slim AS runtime

WORKDIR /app

# Runtime system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy venv from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy source
COPY src/ ./src/
COPY precompute.py rank.py ./
COPY pyproject.toml ./

# Create non-root user
RUN useradd -m -u 1000 appuser
RUN mkdir -p data artifacts && chown -R appuser:appuser /app
USER appuser

# Expose FastAPI port
EXPOSE 8000

# Default: run FastAPI server
# Override with `docker run ... python rank.py` for CLI usage
CMD ["uvicorn", "src.api.server:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1"]
