"""Feature drift detection using Evidently.

Monitors whether incoming sensor data distributions have shifted
compared to the training reference data, which could degrade model performance.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


class DriftDetector:
    """Detects feature drift between reference (training) and current data.

    Uses statistical tests to compare distributions column-by-column.
    """

    def __init__(self, reference_data: pd.DataFrame | None = None) -> None:
        """Initialize with optional reference data.

        Args:
            reference_data: Training data used as the baseline distribution.
        """
        self.reference_data = reference_data
        self._is_fitted = False

    def fit(self, reference_data: pd.DataFrame) -> None:
        """Set the reference (training) data for comparison.

        Args:
            reference_data: DataFrame of features from training set.
        """
        self.reference_data = reference_data
        self._is_fitted = True
        logger.info(
            "DriftDetector fitted with reference data: %d samples, %d features",
            len(reference_data),
            len(reference_data.columns),
        )

    def detect(self, current_data: pd.DataFrame) -> dict:
        """Check for drift between reference and current data.

        Args:
            current_data: DataFrame of recent feature values.

        Returns:
            Dict with keys:
                - is_drifted: bool indicating if significant drift detected
                - drifted_features: list of feature names with drift
                - drift_scores: dict mapping feature name to drift p-value
                - num_features_drifted: count of drifted features
        """
        if not self._is_fitted or self.reference_data is None:
            raise RuntimeError("DriftDetector not fitted. Call fit() first.")

        try:
            from evidently.metric_preset import DataDriftPreset
            from evidently.report import Report

            report = Report(metrics=[DataDriftPreset()])
            report.run(
                reference_data=self.reference_data,
                current_data=current_data,
            )

            result = report.as_dict()
            drift_info = result["metrics"][0]["result"]

            drifted_features = [
                col_name
                for col_name, col_data in drift_info.get("drift_by_columns", {}).items()
                if col_data.get("drift_detected", False)
            ]

            drift_scores = {
                col_name: col_data.get("drift_score", 1.0)
                for col_name, col_data in drift_info.get("drift_by_columns", {}).items()
            }

            return {
                "is_drifted": drift_info.get("dataset_drift", False),
                "drifted_features": drifted_features,
                "drift_scores": drift_scores,
                "num_features_drifted": len(drifted_features),
            }

        except ImportError:
            logger.warning("Evidently not installed. Using fallback KS-test drift detection.")
            return self._fallback_detect(current_data)

    def _fallback_detect(self, current_data: pd.DataFrame) -> dict:
        """Fallback drift detection using scipy KS-test when Evidently is unavailable."""
        from scipy import stats

        drifted_features = []
        drift_scores: dict[str, float] = {}
        p_value_threshold = 0.05

        common_cols = [c for c in self.reference_data.columns if c in current_data.columns]

        for col in common_cols:
            ref_values = self.reference_data[col].dropna().values
            cur_values = current_data[col].dropna().values

            if len(ref_values) == 0 or len(cur_values) == 0:
                continue

            statistic, p_value = stats.ks_2samp(ref_values, cur_values)
            drift_scores[col] = float(p_value)

            if p_value < p_value_threshold:
                drifted_features.append(col)

        return {
            "is_drifted": len(drifted_features) > len(common_cols) * 0.5,
            "drifted_features": drifted_features,
            "drift_scores": drift_scores,
            "num_features_drifted": len(drifted_features),
        }

    def save_reference(self, path: Path) -> None:
        """Save reference data to disk for later use."""
        if self.reference_data is not None:
            self.reference_data.to_parquet(path, index=False)
            logger.info("Reference data saved to %s", path)

    def load_reference(self, path: Path) -> None:
        """Load reference data from disk."""
        self.reference_data = pd.read_parquet(path)
        self._is_fitted = True
        logger.info("Reference data loaded from %s: %d samples", path, len(self.reference_data))
