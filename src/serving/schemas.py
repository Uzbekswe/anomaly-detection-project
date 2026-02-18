"""Pydantic request/response models for the anomaly detection API.

Matches the API contract in CLAUDE.md Section 5 exactly.
Rule: All types must be explicit — no Any types.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field, field_validator

# ──────────────────────────────────────────────
# Request schemas
# ──────────────────────────────────────────────


class SensorWindow(BaseModel):
    """Single anomaly detection request."""

    sensor_id: str = Field(
        min_length=1,
        max_length=64,
        description="Unique sensor/engine identifier",
        examples=["engine_001"],
    )
    window: list[list[float]] = Field(
        description="Sensor readings: shape [window_size, num_sensors]",
    )
    timestamp: datetime = Field(
        description="Timestamp of the reading (ISO 8601 UTC)",
        examples=["2024-01-15T09:23:11Z"],
    )

    @field_validator("window")
    @classmethod
    def validate_window_shape(cls, v: list[list[float]]) -> list[list[float]]:
        if len(v) == 0:
            raise ValueError("Window must not be empty")
        first_len = len(v[0])
        if first_len == 0:
            raise ValueError("Each timestep must have at least one sensor value")
        for i, row in enumerate(v):
            if len(row) != first_len:
                raise ValueError(
                    f"Inconsistent sensor count: row 0 has {first_len}, "
                    f"row {i} has {len(row)}"
                )
        return v


class BatchDetectRequest(BaseModel):
    """Batch anomaly detection request."""

    windows: list[SensorWindow] = Field(
        min_length=1,
        description="Array of individual detection requests",
    )


# ──────────────────────────────────────────────
# Response schemas
# ──────────────────────────────────────────────


class PredictionResponse(BaseModel):
    """Single anomaly detection response."""

    sensor_id: str = Field(description="Echo of the input sensor_id")
    timestamp: datetime = Field(description="Echo of the input timestamp")
    anomaly_score: float = Field(
        ge=0.0,
        description="Anomaly score (higher = more anomalous)",
    )
    is_anomaly: bool = Field(description="Binary anomaly decision")
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Model confidence in the prediction",
    )
    model_version: str = Field(description="Model name and version used")
    processing_time_ms: float = Field(
        ge=0.0,
        description="Inference time in milliseconds",
    )


class BatchDetectResponse(BaseModel):
    """Batch anomaly detection response."""

    results: list[PredictionResponse] = Field(
        description="Array of individual detection responses",
    )
    total_processed: int = Field(
        ge=0,
        description="Total number of windows processed",
    )


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(description="Service status", examples=["healthy"])
    model_loaded: bool = Field(description="Whether the model is loaded and ready")
    model_version: str = Field(description="Currently loaded model version")


# ──────────────────────────────────────────────
# Error schemas
# ──────────────────────────────────────────────


class ErrorResponse(BaseModel):
    """Standard error response."""

    detail: str = Field(description="Error description")
    error_type: str = Field(description="Error category")
