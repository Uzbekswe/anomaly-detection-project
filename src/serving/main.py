"""FastAPI application factory for the anomaly detection service.

Usage:
    uvicorn src.serving.main:app --reload --port 8000
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.serving.db import close_pool as close_db_pool
from src.serving.middleware import RequestLoggingMiddleware, TimingMiddleware
from src.serving.predictor import AnomalyPredictor, load_serving_config
from src.serving.router import router, set_predictor

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SERVING_CONFIG_PATH = PROJECT_ROOT / "configs" / "serving_config.yaml"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: load model on startup, cleanup on shutdown."""
    # Startup
    logger.info("Starting anomaly detection service...")
    predictor = AnomalyPredictor()
    try:
        predictor.load()
        logger.info("Model loaded successfully: %s", predictor.model_version)
    except Exception as e:
        logger.warning("Model loading failed: %s. Service will start without model.", e)

    set_predictor(predictor)

    yield

    # Shutdown
    close_db_pool()
    logger.info("Shutting down anomaly detection service.")


def create_app(config_path: Path = SERVING_CONFIG_PATH) -> FastAPI:
    """FastAPI application factory.

    Args:
        config_path: Path to serving_config.yaml.

    Returns:
        Configured FastAPI application.
    """
    config = load_serving_config(config_path)

    app = FastAPI(
        title="Manufacturing Anomaly Detection API",
        description="Detect equipment anomalies from IoT sensor time-series data",
        version="0.1.0",
        lifespan=lifespan,
    )

    # CORS middleware
    cors_config = config.get("cors", {})
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_config.get("allow_origins", ["*"]),
        allow_credentials=True,
        allow_methods=cors_config.get("allow_methods", ["GET", "POST"]),
        allow_headers=cors_config.get("allow_headers", ["*"]),
    )

    # Custom middleware
    middleware_config = config.get("middleware", {})
    if middleware_config.get("enable_timing", True):
        app.add_middleware(TimingMiddleware)
    if middleware_config.get("enable_request_logging", True):
        app.add_middleware(RequestLoggingMiddleware)

    # Global exception handler — bad input = 422, not 500
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.error("Unhandled exception on %s: %s", request.url.path, exc, exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error", "error_type": type(exc).__name__},
        )

    # Include routes
    app.include_router(router)

    return app


# Default app instance for uvicorn
app = create_app()


if __name__ == "__main__":
    import uvicorn

    config = load_serving_config()
    api_config = config.get("api", {})

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s — %(name)s — %(levelname)s — %(message)s",
    )

    uvicorn.run(
        "src.serving.main:app",
        host=api_config.get("host", "0.0.0.0"),
        port=int(api_config.get("port", 8000)),
        reload=True,
    )
