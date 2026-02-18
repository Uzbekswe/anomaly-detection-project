"""Prometheus custom metrics for the anomaly detection service.

Exposes key operational metrics. This script can be run standalone
to start a Prometheus exporter on port 8008 for testing purposes.
"""

from __future__ import annotations

import logging
import random
import time

from prometheus_client import Counter, Gauge, Histogram, Info, start_http_server

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Metric Definitions
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

MODEL_LOADED = Gauge(
    "anomaly_model_loaded",
    "Whether the anomaly detection model is loaded (1=yes, 0=no)",
)

# ──────────────────────────────────────────────
# Helper Functions
# ──────────────────────────────────────────────


def record_prediction(
    model_version: str,
    sensor_id: str,
    anomaly_score: float,
    is_anomaly: bool,
    latency_ms: float,
) -> None:
    """Record a single prediction in all relevant metrics."""
    PREDICTIONS_TOTAL.labels(model_version=model_version).inc()
    INFERENCE_LATENCY.labels(model_version=model_version).observe(latency_ms)
    ANOMALY_SCORE.labels(model_version=model_version).observe(anomaly_score)

    if is_anomaly:
        ANOMALIES_DETECTED.labels(
            model_version=model_version,
            sensor_id=sensor_id,
        ).inc()


def set_model_info(model_version: str, model_type: str) -> None:
    """Update model info metric on load."""
    MODEL_LOADED.set(1)
    # Use a separate Info metric for static model details
    model_info = Info("anomaly_model_details", "Information about the loaded model")
    model_info.info({
        "version": model_version,
        "type": model_type,
    })

def set_model_unloaded() -> None:
    """Mark model as unloaded in metrics."""
    MODEL_LOADED.set(0)


# ──────────────────────────────────────────────
# Main entry point for standalone testing
# ──────────────────────────────────────────────

def main():
    """Run a standalone Prometheus exporter for metrics testing."""
    logging.basicConfig(level=logging.INFO)
    logger.info("Starting Prometheus metrics exporter on port 8008...")
    start_http_server(8008)

    # Simulate model loading
    model_version = "lstm_ae:abcdef12"
    set_model_info(model_version=model_version, model_type="lstm_ae")
    print(f"Simulating metrics for model: {model_version}")
    print("Exporter running. Access metrics at http://localhost:8008")

    # Simulate some prediction traffic
    try:
        while True:
            sensor = f"sensor_{random.randint(1, 5)}"
            score = random.random()
            latency = random.uniform(5, 50)
            is_anomaly = score > 0.8

            record_prediction(
                model_version=model_version,
                sensor_id=sensor,
                anomaly_score=score,
                is_anomaly=is_anomaly,
                latency_ms=latency,
            )
            time.sleep(2)
    except KeyboardInterrupt:
        print("\nStopping metrics exporter.")


if __name__ == "__main__":
    main()
