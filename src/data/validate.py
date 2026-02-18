"""Data quality checks for CMAPSS datasets.

Validates raw and labeled DataFrames against expected schemas,
checks for nulls, duplicates, sensor ranges, and structural integrity.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.data.ingest import CMAPSS_COLUMN_NAMES
from src.data.schemas import DatasetStats, LabeledSensorReading, SensorReading

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Validation result
# ──────────────────────────────────────────────


@dataclass
class ValidationResult:
    """Collects pass/fail results from all validation checks."""

    passed: bool = True
    checks: list[dict[str, str | bool]] = field(default_factory=list)

    def add(self, name: str, passed: bool, detail: str = "") -> None:
        self.checks.append({"name": name, "passed": passed, "detail": detail})
        if not passed:
            self.passed = False
            logger.warning("FAIL — %s: %s", name, detail)
        else:
            logger.info("PASS — %s", name)

    def summary(self) -> str:
        lines = []
        for c in self.checks:
            status = "PASS" if c["passed"] else "FAIL"
            line = f"  [{status}] {c['name']}"
            if c["detail"]:
                line += f" — {c['detail']}"
            lines.append(line)
        header = "VALIDATION PASSED" if self.passed else "VALIDATION FAILED"
        return f"{header}\n" + "\n".join(lines)


# ──────────────────────────────────────────────
# Column validation
# ──────────────────────────────────────────────


def check_columns(df: pd.DataFrame, result: ValidationResult) -> None:
    """Verify expected columns are present."""
    expected = set(CMAPSS_COLUMN_NAMES)
    actual = set(df.columns)
    missing = expected - actual
    extra = actual - expected - {"rul", "is_anomaly", "timestamp"}

    if missing:
        result.add("columns_present", False, f"Missing columns: {missing}")
    elif extra:
        result.add("columns_present", True, f"Extra columns (ok): {extra}")
    else:
        result.add("columns_present", True)


# ──────────────────────────────────────────────
# Null checks
# ──────────────────────────────────────────────


def check_nulls(df: pd.DataFrame, result: ValidationResult) -> None:
    """Check for null/NaN values."""
    null_count = df.isnull().sum().sum()
    if null_count > 0:
        null_cols = df.columns[df.isnull().any()].tolist()
        result.add("no_nulls", False, f"{null_count} nulls in columns: {null_cols}")
    else:
        result.add("no_nulls", True)


# ──────────────────────────────────────────────
# Duplicate checks
# ──────────────────────────────────────────────


def check_duplicates(df: pd.DataFrame, result: ValidationResult) -> None:
    """Check for duplicate (unit_id, time_cycles) pairs."""
    dupes = df.duplicated(subset=["unit_id", "time_cycles"], keep=False).sum()
    if dupes > 0:
        result.add("no_duplicates", False, f"{dupes} duplicate (unit_id, time_cycles) rows")
    else:
        result.add("no_duplicates", True)


# ──────────────────────────────────────────────
# Type checks
# ──────────────────────────────────────────────


def check_types(df: pd.DataFrame, result: ValidationResult) -> None:
    """Check that unit_id and time_cycles are integers, sensors are numeric."""
    type_issues = []

    if not pd.api.types.is_integer_dtype(df["unit_id"]):
        type_issues.append("unit_id is not integer")
    if not pd.api.types.is_integer_dtype(df["time_cycles"]):
        type_issues.append("time_cycles is not integer")

    sensor_cols = [c for c in df.columns if c.startswith("sensor_") or c.startswith("op_setting_")]
    for col in sensor_cols:
        if not pd.api.types.is_numeric_dtype(df[col]):
            type_issues.append(f"{col} is not numeric")

    if type_issues:
        result.add("column_types", False, "; ".join(type_issues))
    else:
        result.add("column_types", True)


# ──────────────────────────────────────────────
# Range checks
# ──────────────────────────────────────────────


def check_ranges(df: pd.DataFrame, result: ValidationResult) -> None:
    """Check that key columns are within expected ranges."""
    issues = []

    if (df["unit_id"] < 1).any():
        issues.append("unit_id contains values < 1")
    if (df["time_cycles"] < 1).any():
        issues.append("time_cycles contains values < 1")

    # Check for infinite values in sensor columns
    sensor_cols = [c for c in df.columns if c.startswith("sensor_")]
    for col in sensor_cols:
        if np.isinf(df[col]).any():
            issues.append(f"{col} contains infinite values")

    if issues:
        result.add("value_ranges", False, "; ".join(issues))
    else:
        result.add("value_ranges", True)


# ──────────────────────────────────────────────
# Temporal continuity
# ──────────────────────────────────────────────


def check_temporal_continuity(df: pd.DataFrame, result: ValidationResult) -> None:
    """Verify that time_cycles are sequential within each unit (no gaps)."""
    gap_units = []

    for unit_id, group in df.groupby("unit_id"):
        cycles = group["time_cycles"].sort_values()
        expected = range(cycles.iloc[0], cycles.iloc[-1] + 1)
        if len(cycles) != len(expected):
            gap_units.append(unit_id)

    if gap_units:
        result.add(
            "temporal_continuity",
            False,
            f"{len(gap_units)} units have cycle gaps: {gap_units[:5]}...",
        )
    else:
        result.add("temporal_continuity", True)


# ──────────────────────────────────────────────
# Label checks (for labeled data)
# ──────────────────────────────────────────────


def check_labels(df: pd.DataFrame, result: ValidationResult) -> None:
    """Validate RUL and is_anomaly columns if present."""
    if "rul" in df.columns:
        if (df["rul"] < 0).any():
            result.add("rul_non_negative", False, "RUL contains negative values")
        else:
            result.add("rul_non_negative", True)

    if "is_anomaly" in df.columns:
        unique_vals = set(df["is_anomaly"].unique())
        if not unique_vals.issubset({0, 1}):
            result.add("anomaly_binary", False, f"is_anomaly values: {unique_vals}")
        else:
            result.add("anomaly_binary", True)


# ──────────────────────────────────────────────
# Schema validation (Pydantic row-level)
# ──────────────────────────────────────────────


def check_schema_sample(
    df: pd.DataFrame,
    result: ValidationResult,
    sample_size: int = 100,
) -> None:
    """Validate a sample of rows against the Pydantic schema.

    Uses LabeledSensorReading if labels are present, else SensorReading.
    """
    is_labeled = "rul" in df.columns and "is_anomaly" in df.columns and "timestamp" in df.columns
    schema_cls = LabeledSensorReading if is_labeled else SensorReading

    sample = df.sample(n=min(sample_size, len(df)), random_state=42)
    errors = []

    for idx, row in sample.iterrows():
        try:
            schema_cls.model_validate(row.to_dict())
        except Exception as e:
            errors.append(f"Row {idx}: {e}")

    if errors:
        result.add(
            "schema_validation",
            False,
            f"{len(errors)}/{len(sample)} sampled rows failed: {errors[0]}",
        )
    else:
        result.add("schema_validation", True, f"{len(sample)} sampled rows passed")


# ──────────────────────────────────────────────
# Dataset statistics
# ──────────────────────────────────────────────


def compute_dataset_stats(df: pd.DataFrame) -> DatasetStats:
    """Compute summary statistics for reporting."""
    cycles_per_unit = df.groupby("unit_id")["time_cycles"].max()

    return DatasetStats(
        total_rows=len(df),
        num_units=df["unit_id"].nunique(),
        num_columns=len(df.columns),
        anomaly_count=int(df["is_anomaly"].sum()) if "is_anomaly" in df.columns else 0,
        anomaly_rate=float(df["is_anomaly"].mean()) if "is_anomaly" in df.columns else 0.0,
        min_cycles_per_unit=int(cycles_per_unit.min()),
        max_cycles_per_unit=int(cycles_per_unit.max()),
        avg_cycles_per_unit=float(cycles_per_unit.mean()),
        null_count=int(df.isnull().sum().sum()),
        duplicate_count=int(df.duplicated(subset=["unit_id", "time_cycles"]).sum()),
    )


# ──────────────────────────────────────────────
# Main validation entry point
# ──────────────────────────────────────────────


def validate_dataframe(df: pd.DataFrame) -> ValidationResult:
    """Run all validation checks on a CMAPSS DataFrame.

    Args:
        df: DataFrame from ingest.py (raw or labeled).

    Returns:
        ValidationResult with pass/fail for each check.
    """
    result = ValidationResult()

    check_columns(df, result)
    check_nulls(df, result)
    check_duplicates(df, result)
    check_types(df, result)
    check_ranges(df, result)
    check_temporal_continuity(df, result)
    check_labels(df, result)
    check_schema_sample(df, result)

    logger.info("\n%s", result.summary())
    return result
