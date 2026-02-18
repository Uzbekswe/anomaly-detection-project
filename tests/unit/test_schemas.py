"""Unit tests for src/data/schemas.py and src/serving/schemas.py."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from src.data.schemas import DatasetStats, LabeledSensorReading, RULLabel, SensorReading
from src.serving.schemas import (
    BatchDetectRequest,
    ErrorResponse,
    HealthResponse,
    PredictionResponse,
    SensorWindow,
)

# ──────────────────────────────────────────────
# Data schemas: SensorReading
# ──────────────────────────────────────────────


class TestSensorReading:
    def _valid_data(self) -> dict:
        data = {"unit_id": 1, "time_cycles": 10, "op_setting_1": 0.5, "op_setting_2": 0.3, "op_setting_3": 100.0}
        for i in range(1, 22):
            data[f"sensor_{i}"] = 500.0 + i
        return data

    def test_valid_reading(self) -> None:
        reading = SensorReading(**self._valid_data())
        assert reading.unit_id == 1
        assert reading.sensor_1 == 501.0

    def test_unit_id_must_be_positive(self) -> None:
        data = self._valid_data()
        data["unit_id"] = 0
        with pytest.raises(ValidationError):
            SensorReading(**data)

    def test_time_cycles_must_be_positive(self) -> None:
        data = self._valid_data()
        data["time_cycles"] = 0
        with pytest.raises(ValidationError):
            SensorReading(**data)

    def test_missing_sensor_raises(self) -> None:
        data = self._valid_data()
        del data["sensor_21"]
        with pytest.raises(ValidationError):
            SensorReading(**data)


# ──────────────────────────────────────────────
# Data schemas: LabeledSensorReading
# ──────────────────────────────────────────────


class TestLabeledSensorReading:
    def _valid_data(self) -> dict:
        data = {"unit_id": 1, "time_cycles": 10, "op_setting_1": 0.5, "op_setting_2": 0.3, "op_setting_3": 100.0}
        for i in range(1, 22):
            data[f"sensor_{i}"] = 500.0 + i
        data["rul"] = 25
        data["is_anomaly"] = 1
        data["timestamp"] = datetime(2024, 1, 15, 9, 0, 0, tzinfo=timezone.utc)
        return data

    def test_valid_labeled_reading(self) -> None:
        reading = LabeledSensorReading(**self._valid_data())
        assert reading.is_anomaly == 1
        assert reading.rul == 25

    def test_anomaly_must_be_binary(self) -> None:
        data = self._valid_data()
        data["is_anomaly"] = 2
        with pytest.raises(ValidationError):
            LabeledSensorReading(**data)

    def test_rul_cannot_be_negative(self) -> None:
        data = self._valid_data()
        data["rul"] = -1
        with pytest.raises(ValidationError):
            LabeledSensorReading(**data)


# ──────────────────────────────────────────────
# Data schemas: DatasetStats
# ──────────────────────────────────────────────


class TestDatasetStats:
    def test_valid_stats(self) -> None:
        stats = DatasetStats(
            total_rows=20000,
            num_units=100,
            num_columns=26,
            anomaly_count=3000,
            anomaly_rate=0.15,
            min_cycles_per_unit=128,
            max_cycles_per_unit=362,
            avg_cycles_per_unit=200.0,
            null_count=0,
            duplicate_count=0,
        )
        assert stats.anomaly_rate == 0.15

    def test_anomaly_rate_bounded(self) -> None:
        with pytest.raises(ValidationError):
            DatasetStats(
                total_rows=100, num_units=1, num_columns=26,
                anomaly_count=0, anomaly_rate=1.5,
                min_cycles_per_unit=100, max_cycles_per_unit=100,
                avg_cycles_per_unit=100.0, null_count=0, duplicate_count=0,
            )


# ──────────────────────────────────────────────
# Data schemas: RULLabel
# ──────────────────────────────────────────────


class TestRULLabel:
    def test_valid_rul(self) -> None:
        label = RULLabel(rul=112)
        assert label.rul == 112

    def test_negative_rul_raises(self) -> None:
        with pytest.raises(ValidationError):
            RULLabel(rul=-5)


# ──────────────────────────────────────────────
# Serving schemas: SensorWindow
# ──────────────────────────────────────────────


class TestSensorWindow:
    def test_valid_request(self) -> None:
        window = SensorWindow(
            sensor_id="engine_001",
            window=[[0.5, 0.6], [0.7, 0.8]],
            timestamp=datetime(2024, 1, 15, 9, 23, 11, tzinfo=timezone.utc),
        )
        assert window.sensor_id == "engine_001"
        assert len(window.window) == 2

    def test_empty_sensor_id_raises(self) -> None:
        with pytest.raises(ValidationError):
            SensorWindow(
                sensor_id="",
                window=[[0.5]],
                timestamp=datetime.now(timezone.utc),
            )

    def test_empty_window_raises(self) -> None:
        with pytest.raises(ValidationError):
            SensorWindow(
                sensor_id="engine_001",
                window=[],
                timestamp=datetime.now(timezone.utc),
            )

    def test_inconsistent_row_lengths_raises(self) -> None:
        with pytest.raises(ValidationError, match="Inconsistent sensor count"):
            SensorWindow(
                sensor_id="engine_001",
                window=[[0.5, 0.6], [0.7]],
                timestamp=datetime.now(timezone.utc),
            )

    def test_empty_row_raises(self) -> None:
        with pytest.raises(ValidationError, match="at least one sensor"):
            SensorWindow(
                sensor_id="engine_001",
                window=[[]],
                timestamp=datetime.now(timezone.utc),
            )

    def test_sensor_id_max_length(self) -> None:
        with pytest.raises(ValidationError):
            SensorWindow(
                sensor_id="x" * 65,
                window=[[0.5]],
                timestamp=datetime.now(timezone.utc),
            )


# ──────────────────────────────────────────────
# Serving schemas: BatchDetectRequest
# ──────────────────────────────────────────────


class TestBatchDetectRequest:
    def test_valid_batch(self) -> None:
        batch = BatchDetectRequest(
            windows=[
                SensorWindow(
                    sensor_id="engine_001",
                    window=[[0.5, 0.6]],
                    timestamp=datetime.now(timezone.utc),
                ),
            ]
        )
        assert len(batch.windows) == 1

    def test_empty_batch_raises(self) -> None:
        with pytest.raises(ValidationError):
            BatchDetectRequest(windows=[])


# ──────────────────────────────────────────────
# Serving schemas: PredictionResponse
# ──────────────────────────────────────────────


class TestPredictionResponse:
    def test_valid_response(self) -> None:
        resp = PredictionResponse(
            sensor_id="engine_001",
            timestamp=datetime(2024, 1, 15, 9, 23, 11, tzinfo=timezone.utc),
            anomaly_score=0.847,
            is_anomaly=True,
            confidence=0.91,
            model_version="lstm_autoencoder_v2",
            processing_time_ms=12.4,
        )
        assert resp.is_anomaly is True
        assert resp.confidence == 0.91

    def test_negative_score_raises(self) -> None:
        with pytest.raises(ValidationError):
            PredictionResponse(
                sensor_id="e1", timestamp=datetime.now(timezone.utc),
                anomaly_score=-0.1, is_anomaly=False, confidence=0.5,
                model_version="v1", processing_time_ms=1.0,
            )

    def test_confidence_out_of_range_raises(self) -> None:
        with pytest.raises(ValidationError):
            PredictionResponse(
                sensor_id="e1", timestamp=datetime.now(timezone.utc),
                anomaly_score=0.5, is_anomaly=False, confidence=1.5,
                model_version="v1", processing_time_ms=1.0,
            )


# ──────────────────────────────────────────────
# Serving schemas: HealthResponse
# ──────────────────────────────────────────────


class TestHealthResponse:
    def test_valid_health(self) -> None:
        resp = HealthResponse(
            status="healthy",
            model_loaded=True,
            model_version="lstm_autoencoder_v2",
        )
        assert resp.model_loaded is True


# ──────────────────────────────────────────────
# Serving schemas: ErrorResponse
# ──────────────────────────────────────────────


class TestErrorResponse:
    def test_valid_error(self) -> None:
        resp = ErrorResponse(detail="Something went wrong", error_type="ValueError")
        assert resp.error_type == "ValueError"
