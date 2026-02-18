"""Unit tests for src/features/engineer.py."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.engineer import (
    MinMaxScaler,
    StandardScaler,
    add_rolling_features,
    create_scaler,
    create_sliding_windows,
    drop_low_variance_sensors,
    get_feature_columns,
    fit_scaler,
    transform_features,
)

# ──────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Create a sample DataFrame mimicking CMAPSS data for 2 units."""
    np.random.seed(42)
    rows = []
    for unit_id in [1, 2]:
        for cycle in range(1, 51):
            row = {
                "unit_id": unit_id,
                "time_cycles": cycle,
                "op_setting_1": np.random.normal(0.5, 0.1),
                "op_setting_2": np.random.normal(0.3, 0.05),
                "op_setting_3": 100.0,  # constant — would be dropped
            }
            for i in range(1, 22):
                row[f"sensor_{i}"] = np.random.normal(500 + i * 10, 5)
            row["rul"] = 50 - cycle
            row["is_anomaly"] = 1 if (50 - cycle) < 30 else 0
            rows.append(row)
    return pd.DataFrame(rows)


@pytest.fixture
def sensor_columns() -> list[str]:
    return [f"sensor_{i}" for i in range(1, 22)]


@pytest.fixture
def drop_sensors() -> list[str]:
    return ["sensor_1", "sensor_5", "sensor_6", "sensor_10", "sensor_16", "sensor_18", "sensor_19"]


# ──────────────────────────────────────────────
# MinMaxScaler
# ──────────────────────────────────────────────


class TestMinMaxScaler:
    def test_fit_stores_min_max(self) -> None:
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [10.0, 20.0, 30.0]})
        scaler = MinMaxScaler()
        scaler.fit(df)
        assert scaler.min_vals == {"a": 1.0, "b": 10.0}
        assert scaler.max_vals == {"a": 3.0, "b": 30.0}

    def test_transform_scales_to_0_1(self) -> None:
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [10.0, 20.0, 30.0]})
        scaler = MinMaxScaler().fit(df)
        result = scaler.transform(df)
        assert result["a"].min() == pytest.approx(0.0)
        assert result["a"].max() == pytest.approx(1.0)
        assert result["b"].min() == pytest.approx(0.0)
        assert result["b"].max() == pytest.approx(1.0)

    def test_transform_zero_range(self) -> None:
        df = pd.DataFrame({"a": [5.0, 5.0, 5.0]})
        scaler = MinMaxScaler().fit(df)
        result = scaler.transform(df)
        assert (result["a"] == 0.0).all()

    def test_serialization_roundtrip(self) -> None:
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [10.0, 20.0, 30.0]})
        scaler = MinMaxScaler()
        scaler.fit(df)
        data = scaler.to_dict()
        restored = MinMaxScaler.from_dict(data)
        assert restored.min_vals == scaler.min_vals
        assert restored.max_vals == scaler.max_vals

    def test_transform_unseen_data(self) -> None:
        train = pd.DataFrame({"a": [0.0, 10.0]})
        test = pd.DataFrame({"a": [5.0]})
        scaler = MinMaxScaler()
        scaler.fit(train)
        result = scaler.transform(test)
        assert result["a"].iloc[0] == pytest.approx(0.5)


# ──────────────────────────────────────────────
# StandardScaler
# ──────────────────────────────────────────────


class TestStandardScaler:
    def test_fit_stores_mean_std(self) -> None:
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
        scaler = StandardScaler()
        scaler.fit(df)
        assert scaler.mean_vals["a"] == pytest.approx(2.0)
        assert scaler.std_vals["a"] == pytest.approx(1.0)

    def test_transform_zero_mean(self) -> None:
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
        scaler = StandardScaler().fit(df)
        result = scaler.transform(df)
        assert result["a"].mean() == pytest.approx(0.0, abs=1e-10)

    def test_serialization_roundtrip(self) -> None:
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
        scaler = StandardScaler()
        scaler.fit(df)
        data = scaler.to_dict()
        restored = StandardScaler.from_dict(data)
        assert restored.mean_vals == scaler.mean_vals
        assert restored.std_vals == scaler.std_vals


# ──────────────────────────────────────────────
# create_scaler
# ──────────────────────────────────────────────


class TestCreateScaler:
    def test_min_max(self) -> None:
        scaler = create_scaler("min_max")
        assert isinstance(scaler, MinMaxScaler)

    def test_standard(self) -> None:
        scaler = create_scaler("standard")
        assert isinstance(scaler, StandardScaler)

    def test_unknown_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown normalization"):
            create_scaler("invalid")


# ──────────────────────────────────────────────
# drop_low_variance_sensors
# ──────────────────────────────────────────────


class TestDropLowVarianceSensors:
    def test_drops_specified_sensors(self, sample_df: pd.DataFrame, drop_sensors: list[str]) -> None:
        result = drop_low_variance_sensors(sample_df, drop_sensors)
        for sensor in drop_sensors:
            assert sensor not in result.columns

    def test_keeps_other_columns(self, sample_df: pd.DataFrame, drop_sensors: list[str]) -> None:
        result = drop_low_variance_sensors(sample_df, drop_sensors)
        assert "unit_id" in result.columns
        assert "sensor_2" in result.columns
        assert "sensor_3" in result.columns

    def test_row_count_unchanged(self, sample_df: pd.DataFrame, drop_sensors: list[str]) -> None:
        result = drop_low_variance_sensors(sample_df, drop_sensors)
        assert len(result) == len(sample_df)


# ──────────────────────────────────────────────
# add_rolling_features
# ──────────────────────────────────────────────


class TestAddRollingFeatures:
    def test_adds_correct_number_of_columns(self, sample_df: pd.DataFrame) -> None:
        sensors = ["sensor_2", "sensor_3"]
        windows = [5, 10]
        stats = ["mean", "std"]
        result = add_rolling_features(sample_df, sensors, windows, stats)
        expected_new = len(sensors) * len(windows) * len(stats)
        assert len(result.columns) == len(sample_df.columns) + expected_new

    def test_no_nans_in_result(self, sample_df: pd.DataFrame) -> None:
        sensors = ["sensor_2"]
        result = add_rolling_features(sample_df, sensors, [5], ["mean", "std"])
        assert result.isnull().sum().sum() == 0

    def test_column_naming_convention(self, sample_df: pd.DataFrame) -> None:
        sensors = ["sensor_2"]
        result = add_rolling_features(sample_df, sensors, [5], ["mean"])
        assert "sensor_2_rolling_mean_5" in result.columns

    def test_invalid_stat_raises(self, sample_df: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="Unsupported rolling statistic"):
            add_rolling_features(sample_df, ["sensor_2"], [5], ["median"])


# ──────────────────────────────────────────────
# get_feature_columns
# ──────────────────────────────────────────────


class TestGetFeatureColumns:
    def test_excludes_metadata(self, sample_df: pd.DataFrame) -> None:
        cols = get_feature_columns(sample_df)
        assert "unit_id" not in cols
        assert "time_cycles" not in cols
        assert "rul" not in cols
        assert "is_anomaly" not in cols

    def test_includes_sensors(self, sample_df: pd.DataFrame) -> None:
        cols = get_feature_columns(sample_df)
        assert "sensor_2" in cols
        assert "op_setting_1" in cols


# ──────────────────────────────────────────────
# Normalization
# ──────────────────────────────────────────────


class TestNormalization:
    def test_fit_scaler_returns_fitted_scaler(self, sample_df: pd.DataFrame) -> None:
        scaler = fit_scaler(sample_df, method="min_max")
        assert isinstance(scaler, MinMaxScaler)
        assert len(scaler.min_vals) > 0

    def test_transform_features_uses_fitted_scaler(self, sample_df: pd.DataFrame) -> None:
        scaler = fit_scaler(sample_df, method="min_max")
        result_df = transform_features(sample_df, scaler)
        feature_cols = get_feature_columns(result_df)
        # All values should be in [0, 1] since we use same data
        for col in feature_cols:
            assert result_df[col].min() >= -0.01
            assert result_df[col].max() <= 1.01


# ──────────────────────────────────────────────
# create_sliding_windows
# ──────────────────────────────────────────────


class TestCreateSlidingWindows:
    def test_output_shape(self, sample_df: pd.DataFrame) -> None:
        feature_cols = get_feature_columns(sample_df)
        windows, labels, metadata = create_sliding_windows(
            sample_df, window_size=10, feature_columns=feature_cols
        )
        assert windows.ndim == 3
        assert windows.shape[1] == 10
        assert windows.shape[2] == len(feature_cols)

    def test_labels_match_last_timestep(self, sample_df: pd.DataFrame) -> None:
        feature_cols = get_feature_columns(sample_df)
        windows, labels, metadata = create_sliding_windows(
            sample_df, window_size=10, feature_columns=feature_cols
        )
        assert labels.dtype == np.int64
        assert set(np.unique(labels)).issubset({0, 1})

    def test_metadata_shape(self, sample_df: pd.DataFrame) -> None:
        feature_cols = get_feature_columns(sample_df)
        windows, labels, metadata = create_sliding_windows(
            sample_df, window_size=10, feature_columns=feature_cols
        )
        assert metadata.shape[1] == 2  # unit_id, time_cycles

    def test_skips_short_units(self) -> None:
        """Units with fewer rows than window_size should be skipped."""
        df = pd.DataFrame({
            "unit_id": [1] * 5,
            "time_cycles": list(range(1, 6)),
            "sensor_2": np.random.randn(5),
            "is_anomaly": [0] * 5,
        })
        windows, labels, metadata = create_sliding_windows(
            df, window_size=10, feature_columns=["sensor_2"]
        )
        assert len(windows) == 0

    def test_window_count(self, sample_df: pd.DataFrame) -> None:
        """Each unit of length L produces L - window_size + 1 windows."""
        feature_cols = get_feature_columns(sample_df)
        window_size = 10
        windows, _, _ = create_sliding_windows(
            sample_df, window_size=window_size, feature_columns=feature_cols
        )
        # 2 units x 50 cycles each → 2 * (50 - 10 + 1) = 82 windows
        assert len(windows) == 2 * (50 - window_size + 1)


