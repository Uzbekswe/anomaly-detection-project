"""API route definitions for the anomaly detection service.

Endpoints:
    POST /detect       — Single window anomaly detection
    POST /detect/batch — Batch anomaly detection
    GET  /health       — Health check
"""

from __future__ import annotations

import logging
import time

from fastapi import APIRouter, Depends, HTTPException, Request

from src.serving.db import store_anomaly_event, store_anomaly_events_batch
from src.serving.predictor import AnomalyPredictor
from src.serving.schemas import (
    BatchDetectRequest,
    BatchDetectResponse,
    HealthResponse,
    PredictionResponse,
    SensorWindow,
)

logger = logging.getLogger(__name__)

router = APIRouter()

# Predictor instance — initialized by the app factory in main.py
predictor: AnomalyPredictor | None = None


def set_predictor(p: AnomalyPredictor) -> None:
    """Set the global predictor instance. Called during app startup."""
    global predictor
    predictor = p


def _get_predictor() -> AnomalyPredictor:
    """Get the predictor, raising 503 if not loaded."""
    if predictor is None or not predictor.is_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Service is starting up.",
        )
    return predictor


def get_config(request: Request) -> dict:
    """Get the app config from the request state."""
    return request.app.state.config


# ──────────────────────────────────────────────
# POST /detect
# ──────────────────────────────────────────────


@router.post("/detect", response_model=PredictionResponse)
async def detect(request_body: SensorWindow, config: dict = Depends(get_config)) -> PredictionResponse:
    """Detect anomaly in a single sensor window."""
    p = _get_predictor()

    start = time.perf_counter()
    prediction_result = p.predict(request_body.window)
    elapsed_ms = (time.perf_counter() - start) * 1000

    result = PredictionResponse(
        sensor_id=request_body.sensor_id,
        timestamp=request_body.timestamp,
        processing_time_ms=round(elapsed_ms, 2),
        **prediction_result,
    )

    # Persist to database (graceful — won't fail the request)
    if result.is_anomaly:
        store_anomaly_event(
            config=config,
            sensor_id=result.sensor_id,
            detected_at=result.timestamp,
            anomaly_score=result.anomaly_score,
            is_anomaly=result.is_anomaly,
            confidence=result.confidence,
            model_version=result.model_version,
        )

    return result


# ──────────────────────────────────────────────
# POST /detect/batch
# ──────────────────────────────────────────────


@router.post("/detect/batch", response_model=BatchDetectResponse)
async def detect_batch(request_body: BatchDetectRequest, config: dict = Depends(get_config)) -> BatchDetectResponse:
    """Detect anomalies in a batch of sensor windows."""
    p = _get_predictor()

    # Enforce max batch size from config
    max_batch_size = int(config.get("batch", {}).get("max_batch_size", 500))
    if len(request_body.windows) > max_batch_size:
        raise HTTPException(
            status_code=422,
            detail=f"Batch size {len(request_body.windows)} exceeds maximum of {max_batch_size}",
        )

    results = []
    events_to_store = []
    for sensor_window in request_body.windows:
        start = time.perf_counter()
        prediction_result = p.predict(sensor_window.window)
        elapsed_ms = (time.perf_counter() - start) * 1000

        result = PredictionResponse(
            sensor_id=sensor_window.sensor_id,
            timestamp=sensor_window.timestamp,
            processing_time_ms=round(elapsed_ms, 2),
            **prediction_result,
        )
        results.append(result)

        if result.is_anomaly:
            events_to_store.append({
                "sensor_id": result.sensor_id,
                "detected_at": result.timestamp,
                "anomaly_score": result.anomaly_score,
                "is_anomaly": result.is_anomaly,
                "confidence": result.confidence,
                "model_version": result.model_version,
            })

    # Persist batch to database if there are any anomalies
    if events_to_store:
        store_anomaly_events_batch(config=config, events=events_to_store)

    return BatchDetectResponse(
        results=results,
        total_processed=len(results),
    )


# ──────────────────────────────────────────────
# GET /health
# ──────────────────────────────────────────────


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Health check endpoint."""
    loaded = predictor is not None and predictor.is_loaded
    version = "none"
    if loaded and predictor.model_container is not None:
        version = predictor.model_container.model_version

    return HealthResponse(
        status="healthy" if loaded else "unhealthy",
        model_loaded=loaded,
        model_version=version,
    )
