"""Integration tests for the FastAPI anomaly detection API.

Tests hit real FastAPI endpoints via httpx async test client.
The predictor is mocked to isolate tests from needing a trained model.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from httpx import ASGITransport, AsyncClient

from src.serving.middleware import RequestLoggingMiddleware, TimingMiddleware
from src.serving.router import router, set_predictor

# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────


def _make_mock_predictor(
    *,
    is_loaded: bool = True,
    anomaly_score: float = 0.72,
    is_anomaly: bool = True,
    confidence: float = 0.88,
    model_version: str = "lstm_autoencoder_test_v1",
) -> MagicMock:
    """Create a mock AnomalyPredictor with configurable return values."""
    mock = MagicMock()
    mock.is_loaded = is_loaded
    mock.model_version = model_version
    mock.predict.return_value = (anomaly_score, is_anomaly, confidence)
    return mock


def _valid_detect_payload() -> dict:
    """Return a valid POST /detect request body."""
    return {
        "sensor_id": "engine_001",
        "window": [[0.5 + i * 0.01, 0.6 + i * 0.01] for i in range(30)],
        "timestamp": "2024-01-15T09:23:11Z",
    }


def _build_test_app(predictor_mock: MagicMock) -> FastAPI:
    """Build a minimal FastAPI app for testing with the given mock predictor."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        set_predictor(predictor_mock)
        yield

    app = FastAPI(title="Test App", lifespan=lifespan)
    app.add_middleware(TimingMiddleware)
    app.add_middleware(RequestLoggingMiddleware)

    @app.exception_handler(Exception)
    async def global_exception_handler(request, exc):
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error", "error_type": type(exc).__name__},
        )

    app.include_router(router)
    return app


# ──────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────


@pytest.fixture
def mock_predictor() -> MagicMock:
    """Return a loaded mock predictor with default values."""
    return _make_mock_predictor()


@pytest.fixture
def unloaded_predictor() -> MagicMock:
    """Return an unloaded mock predictor (simulates startup without model)."""
    return _make_mock_predictor(is_loaded=False)


@pytest.fixture
async def client(mock_predictor: MagicMock) -> AsyncClient:
    """Async httpx client for the app with a loaded model."""
    app = _build_test_app(mock_predictor)
    # Set predictor directly (ASGITransport may not trigger lifespan)
    set_predictor(mock_predictor)
    transport = ASGITransport(app=app)
    with patch("src.serving.router.store_anomaly_event"), \
         patch("src.serving.router.store_anomaly_events_batch"):
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            yield ac


@pytest.fixture
async def client_no_model(unloaded_predictor: MagicMock) -> AsyncClient:
    """Async httpx client for the app without a loaded model."""
    app = _build_test_app(unloaded_predictor)
    set_predictor(unloaded_predictor)
    transport = ASGITransport(app=app)
    with patch("src.serving.router.store_anomaly_event"), \
         patch("src.serving.router.store_anomaly_events_batch"):
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            yield ac


# ──────────────────────────────────────────────
# GET /health
# ──────────────────────────────────────────────


class TestHealthEndpoint:
    @pytest.mark.asyncio
    async def test_health_when_model_loaded(self, client: AsyncClient) -> None:
        resp = await client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "healthy"
        assert body["model_loaded"] is True
        assert body["model_version"] == "lstm_autoencoder_test_v1"

    @pytest.mark.asyncio
    async def test_health_when_model_not_loaded(self, client_no_model: AsyncClient) -> None:
        resp = await client_no_model.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "unhealthy"
        assert body["model_loaded"] is False
        assert body["model_version"] == "none"

    @pytest.mark.asyncio
    async def test_health_has_timing_headers(self, client: AsyncClient) -> None:
        resp = await client.get("/health")
        assert "x-process-time-ms" in resp.headers
        assert "x-request-id" in resp.headers


# ──────────────────────────────────────────────
# POST /detect — successful cases
# ──────────────────────────────────────────────


class TestDetectEndpoint:
    @pytest.mark.asyncio
    async def test_detect_valid_request(
        self, client: AsyncClient, mock_predictor: MagicMock
    ) -> None:
        payload = _valid_detect_payload()
        resp = await client.post("/detect", json=payload)
        assert resp.status_code == 200

        body = resp.json()
        assert body["sensor_id"] == "engine_001"
        assert body["timestamp"] == "2024-01-15T09:23:11Z"
        assert body["anomaly_score"] == 0.72
        assert body["is_anomaly"] is True
        assert body["confidence"] == 0.88
        assert body["model_version"] == "lstm_autoencoder_test_v1"
        assert body["processing_time_ms"] >= 0

        # Verify predictor was called with the window data
        mock_predictor.predict.assert_called_once()

    @pytest.mark.asyncio
    async def test_detect_response_matches_schema(self, client: AsyncClient) -> None:
        """All fields defined in the API contract must be present."""
        resp = await client.post("/detect", json=_valid_detect_payload())
        body = resp.json()

        required_fields = {
            "sensor_id",
            "timestamp",
            "anomaly_score",
            "is_anomaly",
            "confidence",
            "model_version",
            "processing_time_ms",
        }
        assert required_fields.issubset(body.keys())

    @pytest.mark.asyncio
    async def test_detect_echoes_sensor_id(self, client: AsyncClient) -> None:
        payload = _valid_detect_payload()
        payload["sensor_id"] = "turbine_42"
        resp = await client.post("/detect", json=payload)
        assert resp.json()["sensor_id"] == "turbine_42"

    @pytest.mark.asyncio
    async def test_detect_echoes_timestamp(self, client: AsyncClient) -> None:
        payload = _valid_detect_payload()
        payload["timestamp"] = "2025-06-01T12:00:00Z"
        resp = await client.post("/detect", json=payload)
        assert resp.json()["timestamp"] == "2025-06-01T12:00:00Z"


# ──────────────────────────────────────────────
# POST /detect — validation errors (422)
# ──────────────────────────────────────────────


class TestDetectValidation:
    @pytest.mark.asyncio
    async def test_missing_sensor_id(self, client: AsyncClient) -> None:
        payload = _valid_detect_payload()
        del payload["sensor_id"]
        resp = await client.post("/detect", json=payload)
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_empty_sensor_id(self, client: AsyncClient) -> None:
        payload = _valid_detect_payload()
        payload["sensor_id"] = ""
        resp = await client.post("/detect", json=payload)
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_sensor_id_too_long(self, client: AsyncClient) -> None:
        payload = _valid_detect_payload()
        payload["sensor_id"] = "x" * 65
        resp = await client.post("/detect", json=payload)
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_empty_window(self, client: AsyncClient) -> None:
        payload = _valid_detect_payload()
        payload["window"] = []
        resp = await client.post("/detect", json=payload)
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_window_with_empty_row(self, client: AsyncClient) -> None:
        payload = _valid_detect_payload()
        payload["window"] = [[]]
        resp = await client.post("/detect", json=payload)
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_inconsistent_window_rows(self, client: AsyncClient) -> None:
        payload = _valid_detect_payload()
        payload["window"] = [[0.5, 0.6], [0.7]]
        resp = await client.post("/detect", json=payload)
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_missing_window(self, client: AsyncClient) -> None:
        payload = _valid_detect_payload()
        del payload["window"]
        resp = await client.post("/detect", json=payload)
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_missing_timestamp(self, client: AsyncClient) -> None:
        payload = _valid_detect_payload()
        del payload["timestamp"]
        resp = await client.post("/detect", json=payload)
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_invalid_timestamp_format(self, client: AsyncClient) -> None:
        payload = _valid_detect_payload()
        payload["timestamp"] = "not-a-date"
        resp = await client.post("/detect", json=payload)
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_empty_body(self, client: AsyncClient) -> None:
        resp = await client.post("/detect", json={})
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_no_body(self, client: AsyncClient) -> None:
        resp = await client.post("/detect")
        assert resp.status_code == 422


# ──────────────────────────────────────────────
# POST /detect — model not loaded (503)
# ──────────────────────────────────────────────


class TestDetectModelNotLoaded:
    @pytest.mark.asyncio
    async def test_detect_returns_503(self, client_no_model: AsyncClient) -> None:
        resp = await client_no_model.post("/detect", json=_valid_detect_payload())
        assert resp.status_code == 503

    @pytest.mark.asyncio
    async def test_detect_503_has_detail(self, client_no_model: AsyncClient) -> None:
        resp = await client_no_model.post("/detect", json=_valid_detect_payload())
        body = resp.json()
        assert "detail" in body
        assert "not loaded" in body["detail"].lower()


# ──────────────────────────────────────────────
# POST /detect/batch — successful cases
# ──────────────────────────────────────────────


class TestBatchDetectEndpoint:
    @pytest.mark.asyncio
    async def test_batch_single_window(self, client: AsyncClient) -> None:
        payload = {"windows": [_valid_detect_payload()]}
        resp = await client.post("/detect/batch", json=payload)
        assert resp.status_code == 200
        body = resp.json()
        assert body["total_processed"] == 1
        assert len(body["results"]) == 1
        assert body["results"][0]["sensor_id"] == "engine_001"

    @pytest.mark.asyncio
    async def test_batch_multiple_windows(
        self, client: AsyncClient, mock_predictor: MagicMock
    ) -> None:
        windows = []
        for i in range(3):
            w = _valid_detect_payload()
            w["sensor_id"] = f"engine_{i:03d}"
            windows.append(w)

        resp = await client.post("/detect/batch", json={"windows": windows})
        assert resp.status_code == 200
        body = resp.json()
        assert body["total_processed"] == 3
        assert len(body["results"]) == 3

        # Verify sensor_ids are echoed in order
        for i, result in enumerate(body["results"]):
            assert result["sensor_id"] == f"engine_{i:03d}"

    @pytest.mark.asyncio
    async def test_batch_response_matches_schema(self, client: AsyncClient) -> None:
        payload = {"windows": [_valid_detect_payload()]}
        resp = await client.post("/detect/batch", json=payload)
        body = resp.json()
        assert "results" in body
        assert "total_processed" in body

        # Each result must match PredictionResponse schema
        result = body["results"][0]
        required_fields = {
            "sensor_id",
            "timestamp",
            "anomaly_score",
            "is_anomaly",
            "confidence",
            "model_version",
            "processing_time_ms",
        }
        assert required_fields.issubset(result.keys())


# ──────────────────────────────────────────────
# POST /detect/batch — validation errors (422)
# ──────────────────────────────────────────────


class TestBatchDetectValidation:
    @pytest.mark.asyncio
    async def test_empty_batch(self, client: AsyncClient) -> None:
        resp = await client.post("/detect/batch", json={"windows": []})
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_batch_with_invalid_window(self, client: AsyncClient) -> None:
        payload = {
            "windows": [
                {
                    "sensor_id": "",  # invalid: empty
                    "window": [[0.5]],
                    "timestamp": "2024-01-15T09:23:11Z",
                }
            ]
        }
        resp = await client.post("/detect/batch", json=payload)
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_batch_missing_windows_field(self, client: AsyncClient) -> None:
        resp = await client.post("/detect/batch", json={})
        assert resp.status_code == 422


# ──────────────────────────────────────────────
# POST /detect/batch — model not loaded (503)
# ──────────────────────────────────────────────


class TestBatchDetectModelNotLoaded:
    @pytest.mark.asyncio
    async def test_batch_returns_503(self, client_no_model: AsyncClient) -> None:
        payload = {"windows": [_valid_detect_payload()]}
        resp = await client_no_model.post("/detect/batch", json=payload)
        assert resp.status_code == 503


# ──────────────────────────────────────────────
# Middleware — timing & request-id headers
# ──────────────────────────────────────────────


class TestMiddleware:
    @pytest.mark.asyncio
    async def test_timing_header_on_detect(self, client: AsyncClient) -> None:
        resp = await client.post("/detect", json=_valid_detect_payload())
        assert "x-process-time-ms" in resp.headers
        # Value should be a valid float
        float(resp.headers["x-process-time-ms"])

    @pytest.mark.asyncio
    async def test_request_id_header_on_detect(self, client: AsyncClient) -> None:
        resp = await client.post("/detect", json=_valid_detect_payload())
        assert "x-request-id" in resp.headers
        assert len(resp.headers["x-request-id"]) == 8

    @pytest.mark.asyncio
    async def test_timing_header_on_batch(self, client: AsyncClient) -> None:
        payload = {"windows": [_valid_detect_payload()]}
        resp = await client.post("/detect/batch", json=payload)
        assert "x-process-time-ms" in resp.headers

    @pytest.mark.asyncio
    async def test_unique_request_ids(self, client: AsyncClient) -> None:
        resp1 = await client.post("/detect", json=_valid_detect_payload())
        resp2 = await client.post("/detect", json=_valid_detect_payload())
        assert resp1.headers["x-request-id"] != resp2.headers["x-request-id"]


# ──────────────────────────────────────────────
# Content-type & edge cases
# ──────────────────────────────────────────────


class TestContentType:
    @pytest.mark.asyncio
    async def test_health_returns_json(self, client: AsyncClient) -> None:
        resp = await client.get("/health")
        assert resp.headers["content-type"] == "application/json"

    @pytest.mark.asyncio
    async def test_detect_returns_json(self, client: AsyncClient) -> None:
        resp = await client.post("/detect", json=_valid_detect_payload())
        assert resp.headers["content-type"] == "application/json"

    @pytest.mark.asyncio
    async def test_404_for_unknown_endpoint(self, client: AsyncClient) -> None:
        resp = await client.get("/unknown")
        assert resp.status_code == 404
