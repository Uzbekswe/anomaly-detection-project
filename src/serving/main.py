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
from src.serving.predictor import (
    AnomalyPredictor,
    find_latest_run_id,
    load_model_from_run,
    load_serving_config,
)
from src.serving.router import router, set_predictor

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SERVING_CONFIG_PATH = PROJECT_ROOT / "configs" / "serving_config.yaml"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: load model on startup, cleanup on shutdown."""
    # Startup
    logger.info("Starting anomaly detection service...")
    
    config = load_serving_config(SERVING_CONFIG_PATH)
    model_name = config["model"]["name"]
    model_stage = config["model"]["stage"]

    try:
        run_id = find_latest_run_id(model_name, model_stage)
        model_container = load_model_from_run(run_id)
        predictor = AnomalyPredictor(model_container)
        set_predictor(predictor)
        logger.info("Model loaded successfully: %s", predictor.model_container.model_version)
    except Exception as e:
        logger.error("Model loading failed: %s. Service will start without a model.", e, exc_info=True)
        # Start with an unloaded predictor so /health endpoint works
        set_predictor(AnomalyPredictor(None))

    yield

    # Shutdown
    close_db_pool()
    logger.info("Shutting down anomaly detection service.")


def create_app() -> FastAPI:
    """FastAPI application factory."""
    config = load_serving_config(SERVING_CONFIG_PATH)
    api_config = config.get("api", {})

    app = FastAPI(
        title="Manufacturing Anomaly Detection API",
        description="Detect equipment anomalies from IoT sensor time-series data",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.state.config = config
    
    # Configure logging
    log_level = api_config.get("log_level", "INFO").upper()
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s — %(name)s — %(levelname)s — %(message)s",
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

    # Global exception handler
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
    
    # This block is for local development only
    config = load_serving_config(SERVING_CONFIG_PATH)
    api_config = config.get("api", {})

    uvicorn.run(
        "src.serving.main:app",
        host=api_config.get("host", "0.0.0.0"),
        port=int(api_config.get("port", 8000)),
        reload=True,
        log_level=api_config.get("log_level", "info").lower(),
    )
