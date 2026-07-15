# ============================================================================
# Hybrid RAG Engine — Multi-stage Production Dockerfile
# ============================================================================
# Build:  docker build -t hybrid-rag-engine .
# Run:    docker run -p 5000:5000 --env-file .env hybrid-rag-engine
# ============================================================================

# ── Stage 1: Base with dependencies ─────────────────────────────────
FROM python:3.11-slim AS base

# Security: run as non-root user
RUN groupadd -r raguser && useradd -r -g raguser raguser

WORKDIR /app

# Install system dependencies for unstructured + sentence-transformers
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (cache layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Stage 2: Application ────────────────────────────────────────────
FROM base AS app

WORKDIR /app

# Copy application code
COPY src/ src/
COPY app/ app/
COPY pyproject.toml .

# Create data directory (will be mounted as volume in production)
RUN mkdir -p data/documents data/chroma_db

# Set ownership
RUN chown -R raguser:raguser /app

USER raguser

# Environment defaults
ENV ENVIRONMENT=production
ENV LOG_LEVEL=INFO
ENV APP_HOST=0.0.0.0
ENV APP_PORT=5000

EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:5000/health')"

# Production WSGI server (not Flask dev server)
CMD ["python", "-m", "waitress", "--host=0.0.0.0", "--port=5000", "app.app:app"]
