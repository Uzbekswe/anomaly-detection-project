"""Prometheus custom metrics for the anomaly detection service.

Exposes key operational metrics:
  - Anomaly detection rate (percentage of requests flagged as anomalies)
  - Inference latency (histogram)
  - Total predictions served (counter)
  - Model info (gauge with version label)
"""

from __future__ import annotations

import logging

from prometheus_client import Counter, Gauge, Histogram, Info

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Counters
# ──────────────────────────────────────────────

PREDICTIONS_TOTAL = Counter(
    "anomaly_predictions_total",
    "Total number of anomaly predictions served",
    ["model_version"],
)

ANOMALIES_DETECTED = Counter(
    "anomaly_detections_total",
    "Total number of anomalies detected",
    ["model_version", "sensor_id"],
)

# ──────────────────────────────────────────────
# Histograms
# ──────────────────────────────────────────────

INFERENCE_LATENCY = Histogram(
    "anomaly_inference_latency_ms",
    "Inference latency in milliseconds",
    ["model_version"],
    buckets=[1, 2, 5, 10, 20, 50, 100, 200, 500, 1000],
)

ANOMALY_SCORE = Histogram(
    "anomaly_score_distribution",
    "Distribution of anomaly scores",
    ["model_version"],
    buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)

# ──────────────────────────────────────────────
# Gauges
# ──────────────────────────────────────────────

MODEL_LOADED = Gauge(
    "anomaly_model_loaded",
    "Whether the anomaly detection model is loaded (1=yes, 0=no)",
)

ANOMALY_RATE = Gauge(
    "anomaly_rate_percent",
    "Rolling anomaly detection rate (percentage of last N predictions)",
)

# ──────────────────────────────────────────────
# Info
# ──────────────────────────────────────────────

MODEL_INFO = Info(
    "anomaly_model",
    "Information about the loaded anomaly detection model",
)


# ──────────────────────────────────────────────
# Helper functions
# ──────────────────────────────────────────────


def record_prediction(
    model_version: str,
    sensor_id: str,
    anomaly_score: float,
    is_anomaly: bool,
    latency_ms: float,
) -> None:
    """Record a single prediction in all relevant metrics.

    Args:
        model_version: Version string of the model used.
        sensor_id: Sensor that was evaluated.
        anomaly_score: Anomaly score from the model.
        is_anomaly: Whether the prediction was classified as anomalous.
        latency_ms: Inference time in milliseconds.
    """
    PREDICTIONS_TOTAL.labels(model_version=model_version).inc()
    INFERENCE_LATENCY.labels(model_version=model_version).observe(latency_ms)
    ANOMALY_SCORE.labels(model_version=model_version).observe(anomaly_score)

    if is_anomaly:
        ANOMALIES_DETECTED.labels(
            model_version=model_version,
            sensor_id=sensor_id,
        ).inc()


def set_model_info(model_version: str, model_type: str = "lstm_autoencoder") -> None:
    """Update model info metric on load.

    Args:
        model_version: Version string.
        model_type: Type of model (lstm_autoencoder, isolation_forest, patchtst).
    """
    MODEL_LOADED.set(1)
    MODEL_INFO.info({
        "version": model_version,
        "type": model_type,
    })


def set_model_unloaded() -> None:
    """Mark model as unloaded in metrics."""
    MODEL_LOADED.set(0)


def update_anomaly_rate(rate_percent: float) -> None:
    """Update the rolling anomaly rate gauge.

    Args:
        rate_percent: Percentage of recent predictions that are anomalies.
    """
    ANOMALY_RATE.set(rate_percent)
