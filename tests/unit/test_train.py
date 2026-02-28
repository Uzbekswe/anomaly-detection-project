"""Unit tests for src/models/train.py utility functions.

Tests the pure/utility functions that don't require MLflow or GPU access.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from src.models.train import (
    find_optimal_threshold,
    load_training_config,
    resolve_device,
    set_seed,
    split_data_unit_aware,
)

# ──────────────────────────────────────────────
# TestSetSeed
# ──────────────────────────────────────────────


class TestSetSeed:
    def test_numpy_reproducibility(self) -> None:
        set_seed(42)
        a = np.random.rand(5)
        set_seed(42)
        b = np.random.rand(5)
        np.testing.assert_array_equal(a, b)

    def test_torch_reproducibility(self) -> None:
        set_seed(42)
        a = torch.rand(5)
        set_seed(42)
        b = torch.rand(5)
        assert torch.equal(a, b)

    def test_pythonhashseed_set(self) -> None:
        import os

        set_seed(123)
        assert os.environ["PYTHONHASHSEED"] == "123"


# ──────────────────────────────────────────────
# TestResolveDevice
# ──────────────────────────────────────────────


class TestResolveDevice:
    def test_cpu_explicit(self) -> None:
        device = resolve_device("cpu")
        assert device == torch.device("cpu")

    def test_auto_fallback(self) -> None:
        device = resolve_device("auto")
        # On CI/test machines without GPU, should get cpu or mps
        assert device.type in ("cpu", "cuda", "mps")

    def test_cuda_explicit(self) -> None:
        device = resolve_device("cuda")
        assert device == torch.device("cuda")


# ──────────────────────────────────────────────
# TestLoadTrainingConfig
# ──────────────────────────────────────────────


class TestLoadTrainingConfig:
    def test_loads_yaml(self, tmp_path: Path) -> None:
        config_file = tmp_path / "config.yaml"
        config_file.write_text("epochs: 10\nbatch_size: 32\n")
        config = load_training_config(config_file)
        assert config["epochs"] == 10
        assert config["batch_size"] == 32

    def test_missing_file_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_training_config(Path("/nonexistent/config.yaml"))


# ──────────────────────────────────────────────
# TestSplitDataUnitAware
# ──────────────────────────────────────────────


class TestSplitDataUnitAware:
    @pytest.fixture
    def sample_df(self) -> pd.DataFrame:
        import pandas as pd

        # 10 units, 20 rows each
        rows = []
        for unit_id in range(1, 11):
            for cycle in range(1, 21):
                rows.append({"unit_id": unit_id, "cycle": cycle, "sensor_1": np.random.rand()})
        return pd.DataFrame(rows)

    def test_units_never_split_across_sets(self, sample_df: pd.DataFrame) -> None:
        splits = split_data_unit_aware(
            sample_df, test_size=0.2, validation_size=0.1, random_state=42
        )
        train_units = set(splits["train"]["unit_id"].unique())
        val_units = set(splits["val"]["unit_id"].unique())
        test_units = set(splits["test"]["unit_id"].unique())

        # No unit appears in more than one split
        assert train_units.isdisjoint(val_units)
        assert train_units.isdisjoint(test_units)
        assert val_units.isdisjoint(test_units)

    def test_no_rows_lost(self, sample_df: pd.DataFrame) -> None:
        splits = split_data_unit_aware(
            sample_df, test_size=0.2, validation_size=0.1, random_state=42
        )
        total = len(splits["train"]) + len(splits["val"]) + len(splits["test"])
        assert total == len(sample_df)

    def test_reproducibility(self, sample_df: pd.DataFrame) -> None:
        splits1 = split_data_unit_aware(
            sample_df, test_size=0.2, validation_size=0.1, random_state=42
        )
        splits2 = split_data_unit_aware(
            sample_df, test_size=0.2, validation_size=0.1, random_state=42
        )
        assert set(splits1["train"]["unit_id"].unique()) == set(
            splits2["train"]["unit_id"].unique()
        )
        assert set(splits1["test"]["unit_id"].unique()) == set(splits2["test"]["unit_id"].unique())

    def test_different_seeds_differ(self, sample_df: pd.DataFrame) -> None:
        splits1 = split_data_unit_aware(
            sample_df, test_size=0.2, validation_size=0.1, random_state=42
        )
        splits2 = split_data_unit_aware(
            sample_df, test_size=0.2, validation_size=0.1, random_state=99
        )
        # With different seeds, at least one split should have different units
        train1 = set(splits1["train"]["unit_id"].unique())
        train2 = set(splits2["train"]["unit_id"].unique())
        assert train1 != train2


# ──────────────────────────────────────────────
# TestFindOptimalThreshold
# ──────────────────────────────────────────────


class TestFindOptimalThreshold:
    def test_returns_threshold_in_range(self) -> None:
        scores = np.array([0.1, 0.2, 0.3, 0.8, 0.9])
        labels = np.array([0, 0, 0, 1, 1])
        threshold, metrics = find_optimal_threshold(scores, labels, 0.0, 1.0, 100)
        assert 0.0 <= threshold <= 1.0

    def test_returns_metrics_dict(self) -> None:
        scores = np.array([0.1, 0.2, 0.3, 0.8, 0.9])
        labels = np.array([0, 0, 0, 1, 1])
        _, metrics = find_optimal_threshold(scores, labels, 0.0, 1.0, 100)
        assert "f1" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "auc_roc" in metrics

    def test_perfect_separation(self) -> None:
        scores = np.array([0.0, 0.0, 0.0, 1.0, 1.0])
        labels = np.array([0, 0, 0, 1, 1])
        _, metrics = find_optimal_threshold(scores, labels, 0.0, 1.0, 100)
        assert metrics["f1"] == pytest.approx(1.0)

    def test_single_class_labels(self) -> None:
        scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        labels = np.array([0, 0, 0, 0, 0])
        threshold, metrics = find_optimal_threshold(scores, labels, 0.0, 1.0, 100)
        # Should not crash; auc_roc should be absent with single class
        assert "auc_roc" not in metrics
