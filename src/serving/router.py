"""API route definitions for the anomaly detection service.

Endpoints:
    POST /detect       — Single window anomaly detection
    POST /detect/batch — Batch anomaly detection
    GET  /health       — Health check
"""

from __future__ import annotations

import logging
import time

from fastapi import APIRouter, HTTPException

from src.serving.db import store_anomaly_event, store_anomaly_events_batch
from src.serving.predictor import AnomalyPredictor, load_serving_config
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


# ──────────────────────────────────────────────
# POST /detect
# ──────────────────────────────────────────────


@router.post("/detect", response_model=PredictionResponse)
async def detect(request: SensorWindow) -> PredictionResponse:
    """Detect anomaly in a single sensor window."""
    p = _get_predictor()

    start = time.perf_counter()
    anomaly_score, is_anomaly, confidence = p.predict(request.window)
    elapsed_ms = (time.perf_counter() - start) * 1000

    result = PredictionResponse(
        sensor_id=request.sensor_id,
        timestamp=request.timestamp,
        anomaly_score=anomaly_score,
        is_anomaly=is_anomaly,
        confidence=confidence,
        model_version=p.model_version,
        processing_time_ms=round(elapsed_ms, 2),
    )

    # Persist to database (graceful — won't fail the request)
    store_anomaly_event(
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
async def detect_batch(request: BatchDetectRequest) -> BatchDetectResponse:
    """Detect anomalies in a batch of sensor windows."""
    p = _get_predictor()

    # Enforce max batch size from config
    config = load_serving_config()
    max_batch_size = int(config.get("batch", {}).get("max_batch_size", 500))
    if len(request.windows) > max_batch_size:
        raise HTTPException(
            status_code=422,
            detail=f"Batch size {len(request.windows)} exceeds maximum of {max_batch_size}",
        )

    results = []
    for sensor_window in request.windows:
        start = time.perf_counter()
        anomaly_score, is_anomaly, confidence = p.predict(sensor_window.window)
        elapsed_ms = (time.perf_counter() - start) * 1000

        results.append(PredictionResponse(
            sensor_id=sensor_window.sensor_id,
            timestamp=sensor_window.timestamp,
            anomaly_score=anomaly_score,
            is_anomaly=is_anomaly,
            confidence=confidence,
            model_version=p.model_version,
            processing_time_ms=round(elapsed_ms, 2),
        ))

    # Persist batch to database (graceful — won't fail the request)
    store_anomaly_events_batch([
        {
            "sensor_id": r.sensor_id,
            "detected_at": r.timestamp,
            "anomaly_score": r.anomaly_score,
            "is_anomaly": r.is_anomaly,
            "confidence": r.confidence,
            "model_version": r.model_version,
        }
        for r in results
    ])

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
    version = predictor.model_version if loaded else "none"

    return HealthResponse(
        status="healthy" if loaded else "unhealthy",
        model_loaded=loaded,
        model_version=version,
    )
