"""Pydantic schemas for raw CMAPSS data validation.

These schemas define the expected shape and constraints of the raw data
after parsing, before feature engineering. Used by validate.py to enforce
data quality.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field, field_validator


class SensorReading(BaseModel):
    """A single row from the CMAPSS dataset after ingestion."""

    unit_id: int = Field(ge=1, description="Engine unit identifier")
    time_cycles: int = Field(ge=1, description="Operating cycle number")

    # Operational settings
    op_setting_1: float
    op_setting_2: float
    op_setting_3: float

    # 21 sensor measurements
    sensor_1: float
    sensor_2: float
    sensor_3: float
    sensor_4: float
    sensor_5: float
    sensor_6: float
    sensor_7: float
    sensor_8: float
    sensor_9: float
    sensor_10: float
    sensor_11: float
    sensor_12: float
    sensor_13: float
    sensor_14: float
    sensor_15: float
    sensor_16: float
    sensor_17: float
    sensor_18: float
    sensor_19: float
    sensor_20: float
    sensor_21: float


class LabeledSensorReading(SensorReading):
    """Sensor reading with computed RUL, anomaly label, and simulated timestamp."""

    rul: int = Field(ge=0, description="Remaining Useful Life in cycles")
    is_anomaly: int = Field(ge=0, le=1, description="Binary anomaly label (0 or 1)")
    timestamp: datetime = Field(description="Simulated ingestion timestamp (UTC)")

    @field_validator("is_anomaly")
    @classmethod
    def validate_binary(cls, v: int) -> int:
        if v not in (0, 1):
            raise ValueError(f"is_anomaly must be 0 or 1, got {v}")
        return v


class DatasetStats(BaseModel):
    """Summary statistics for a validated dataset — used for reporting."""

    total_rows: int = Field(ge=0)
    num_units: int = Field(ge=1)
    num_columns: int = Field(ge=26)
    anomaly_count: int = Field(ge=0)
    anomaly_rate: float = Field(ge=0.0, le=1.0)
    min_cycles_per_unit: int = Field(ge=1)
    max_cycles_per_unit: int = Field(ge=1)
    avg_cycles_per_unit: float = Field(ge=1.0)
    null_count: int = Field(ge=0)
    duplicate_count: int = Field(ge=0)


class RULLabel(BaseModel):
    """A single RUL label from RUL_FD001.txt."""

    rul: int = Field(ge=0, description="Remaining Useful Life at last test cycle")
