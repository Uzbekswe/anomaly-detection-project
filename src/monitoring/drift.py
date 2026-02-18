"""Feature drift detection using Evidently.

Monitors whether incoming sensor data distributions have shifted
compared to the training reference data, which could degrade model performance.

This script can be run to compare two datasets and generate a drift report.

Usage:
    python src/monitoring/drift.py \\
        --reference data/processed/train.parquet \\
        --current data/processed/recent.parquet \\
        --output_report reports/drift_report.html
"""

from __future__ import annotations

import argparse
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
        self._is_fitted = self.reference_data is not None

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

    def detect(self, current_data: pd.DataFrame, report_path: Path | None = None) -> dict:
        """Check for drift between reference and current data.

        Args:
            current_data: DataFrame of recent feature values.
            report_path: If provided, save the Evidently HTML report here.

        Returns:
            Dict with drift analysis results.
        """
        if not self._is_fitted or self.reference_data is None:
            raise RuntimeError("DriftDetector not fitted. Call fit() or initialize with reference_data.")

        try:
            from evidently.metric_preset import DataDriftPreset
            from evidently.report import Report

            report = Report(metrics=[DataDriftPreset()])
            report.run(
                reference_data=self.reference_data,
                current_data=current_data,
            )

            if report_path:
                report.save_html(str(report_path))
                logger.info("Drift report saved to %s", report_path)

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
                "num_features": drift_info.get("number_of_columns", 0),
                "num_features_drifted": drift_info.get("number_of_drifted_columns", 0),
                "drifted_features": drifted_features,
                "drift_scores": drift_scores,
            }

        except ImportError:
            logger.warning("Evidently not installed. Using fallback KS-test drift detection.")
            return self._fallback_detect(current_data)

    def _fallback_detect(self, current_data: pd.DataFrame) -> dict:
        """Fallback drift detection using scipy KS-test."""
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

            _, p_value = stats.ks_2samp(ref_values, cur_values)
            drift_scores[col] = float(p_value)

            if p_value < p_value_threshold:
                drifted_features.append(col)

        is_drifted = len(drifted_features) > len(common_cols) * 0.1 # Drift if >10% of features drift
        return {
            "is_drifted": is_drifted,
            "num_features": len(common_cols),
            "num_features_drifted": len(drifted_features),
            "drifted_features": drifted_features,
            "drift_scores": drift_scores,
        }

def main():
    """Command-line interface for drift detection."""
    parser = argparse.ArgumentParser(description="Detect data drift between two datasets.")
    parser.add_argument(
        "--reference",
        type=Path,
        required=True,
        help="Path to the reference dataset (e.g., train.parquet).",
    )
    parser.add_argument(
        "--current",
        type=Path,
        required=True,
        help="Path to the current dataset to check for drift.",
    )
    parser.add_argument(
        "--output_report",
        type=Path,
        default=None,
        help="Optional path to save the HTML drift report.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(name)s — %(levelname)s — %(message)s")

    # Load data
    logger.info("Loading reference data from %s", args.reference)
    if not args.reference.exists():
        raise FileNotFoundError(f"Reference data not found at {args.reference}")
    reference_df = pd.read_parquet(args.reference)

    logger.info("Loading current data from %s", args.current)
    if not args.current.exists():
        raise FileNotFoundError(f"Current data not found at {args.current}")
    current_df = pd.read_parquet(args.current)
    
    # Ensure columns match for comparison
    common_columns = list(set(reference_df.columns) & set(current_df.columns))
    reference_df = reference_df[common_columns]
    current_df = current_df[common_columns]

    # Run detection
    detector = DriftDetector(reference_df)
    drift_results = detector.detect(current_df, report_path=args.output_report)

    # Print summary
    print("\n" + "="*50)
    print("              Data Drift Report Summary")
    print("="*50)
    print(f"  Overall drift detected: {drift_results['is_drifted']}")
    print(f"  Drifted features: {drift_results['num_features_drifted']} / {drift_results['num_features']}")
    print("-"*50)
    if drift_results["num_features_drifted"] > 0:
        print("  Drifted feature details (p-value):")
        for feature in drift_results["drifted_features"]:
            p_value = drift_results["drift_scores"].get(feature, "N/A")
            print(f"    - {feature:<20} (p-value: {p_value:.4f})")
    else:
        print("  No significant feature drift detected.")
    print("="*50)

    if args.output_report:
        print(f"\nFull HTML report saved to: {args.output_report.resolve()}")


if __name__ == "__main__":
    main()
