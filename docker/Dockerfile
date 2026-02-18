# ──────────────────────────────────────────────
# Stage 1: Builder — install dependencies
# ──────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc libpq-dev && \
    rm -rf /var/lib/apt/lists/*

# Copy dependency definition first (cache layer)
COPY pyproject.toml ./

# Install production dependencies only (no dev extras)
RUN pip install --no-cache-dir --prefix=/install .

# ──────────────────────────────────────────────
# Stage 2: Runtime — lean production image
# ──────────────────────────────────────────────
FROM python:3.11-slim AS runtime

# Install runtime-only system deps (libpq for psycopg2)
RUN apt-get update && \
    apt-get install -y --no-install-recommends libpq5 curl && \
    rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from builder
COPY --from=builder /install /usr/local

# Create non-root user (Rule #9: security requirement)
RUN groupadd --gid 1000 appuser && \
    useradd --uid 1000 --gid 1000 --create-home appuser

WORKDIR /app

# Copy application code first
COPY src/ ./src/
COPY configs/ ./configs/
COPY migrations/ ./migrations/
COPY .env.example .env

# Create directories for data/artifacts and set ownership
RUN mkdir -p /app/data/artifacts /app/data/raw /app/data/plots && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose API port (read from env or default)
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:${API_PORT:-8000}/health || exit 1

# Run FastAPI with uvicorn, allowing env override for port
# Note: The app itself should read the host/port from config/env
CMD ["uvicorn", "src.serving.main:app"]
