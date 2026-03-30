# =============================================================================
# Dockerfile — A/B Testing & Statistical Inference Dashboard
# =============================================================================
#
# Build:
#   docker build -t ab-dashboard .
#
# Run:
#   docker run -p 8501:8501 ab-dashboard
#
# Run with custom env:
#   docker run -p 8501:8501 --env-file .env ab-dashboard
#
# Run with mounted data directory (persists exports):
#   docker run -p 8501:8501 -v $(pwd)/data:/app/data ab-dashboard
# =============================================================================

# ── Base image ────────────────────────────────────────────────────────────────
# python:3.11-slim — smaller than full python image, no dev tools
FROM python:3.11-slim

# ── Labels ────────────────────────────────────────────────────────────────────
LABEL maintainer="your-email@example.com"
LABEL version="2.0.0"
LABEL description="A/B Testing & Statistical Inference Dashboard"

# ── Environment variables ─────────────────────────────────────────────────────
# Prevents Python from writing .pyc files and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Application environment — override at runtime with --env APP_ENV=production
ENV APP_ENV=production
ENV APP_PORT=8501

# Streamlit — disable telemetry and configure headless mode
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ENABLE_CORS=false

# ── System dependencies ───────────────────────────────────────────────────────
# build-essential: needed to compile scipy/numpy C extensions if no wheel exists
# curl: used by HEALTHCHECK
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ─────────────────────────────────────────────────────────
WORKDIR /app

# ── Install Python dependencies ───────────────────────────────────────────────
# Copy requirements first — Docker layer caching means pip only re-runs
# when requirements.txt changes, not on every code change.
COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ── Copy application source ───────────────────────────────────────────────────
# Copy in order of change frequency (least → most) to maximise cache hits
COPY pyproject.toml .
COPY config/ ./config/
COPY src/     ./src/
COPY tests/   ./tests/
COPY conftest.py .
COPY app.py   .

# ── Create required runtime directories ──────────────────────────────────────
RUN mkdir -p logs data/samples data/exports

# ── Non-root user for security ────────────────────────────────────────────────
RUN useradd --create-home --shell /bin/bash appuser \
    && chown -R appuser:appuser /app
USER appuser

# ── Expose port ───────────────────────────────────────────────────────────────
EXPOSE 8501

# ── Health check ──────────────────────────────────────────────────────────────
# Docker will mark the container as unhealthy if Streamlit stops responding
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# ── Entrypoint ────────────────────────────────────────────────────────────────
ENTRYPOINT ["python", "-m", "streamlit", "run", "app.py"]
CMD ["--server.port=8501", "--server.address=0.0.0.0"]
