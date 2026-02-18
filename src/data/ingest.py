"""CMAPSS FD001 data ingestion — download, parse, label, and simulate streaming."""

from __future__ import annotations

import argparse
import io
import logging
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import yaml

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
MODEL_CONFIG_PATH = PROJECT_ROOT / "configs" / "model_config.yaml"

CMAPSS_COLUMN_NAMES = (
    ["unit_id", "time_cycles"]
    + [f"op_setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)

EXPECTED_FILES = ["train_FD001.txt", "test_FD001.txt", "RUL_FD001.txt"]

# Known download URLs (tried in order)
DOWNLOAD_URLS = [
    # NASA data portal — direct zip link (may change)
    "https://data.nasa.gov/download/brfb-kqar/application%2Fx-zip-compressed",
    # PHM Society S3 mirror
    "https://phm-datasets.s3.amazonaws.com/NASA/17.+Turbofan+Engine+Degradation+Simulation+Data+Set+2.zip",
]


# ──────────────────────────────────────────────
# Config loader
# ──────────────────────────────────────────────


def load_data_config(config_path: Path = MODEL_CONFIG_PATH) -> dict:
    """Load the data section from model_config.yaml."""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config["data"]


# ──────────────────────────────────────────────
# Download
# ──────────────────────────────────────────────


def download_cmapss(output_dir: Path = RAW_DATA_DIR) -> Path:
    """Download CMAPSS dataset zip and extract FD001 files.

    Tries multiple URLs in order. Falls back with a clear error
    message if none succeed.

    Returns:
        Path to the output directory containing the extracted files.
    """
    import requests

    output_dir.mkdir(parents=True, exist_ok=True)

    # Skip download if files already exist
    if all((output_dir / f).exists() for f in EXPECTED_FILES):
        logger.info("CMAPSS FD001 files already present in %s — skipping download.", output_dir)
        return output_dir

    for url in DOWNLOAD_URLS:
        logger.info("Attempting download from: %s", url)
        try:
            resp = requests.get(url, timeout=120, stream=True)
            resp.raise_for_status()

            zip_bytes = io.BytesIO(resp.content)
            with zipfile.ZipFile(zip_bytes) as zf:
                _extract_fd001_files(zf, output_dir)

            logger.info("Successfully downloaded and extracted to %s", output_dir)
            return output_dir

        except (requests.RequestException, zipfile.BadZipFile, KeyError) as exc:
            logger.warning("Download from %s failed: %s", url, exc)
            continue

    raise RuntimeError(
        "Could not download CMAPSS dataset from any known URL.\n"
        "Please download manually and place train_FD001.txt, test_FD001.txt, "
        f"and RUL_FD001.txt into: {output_dir}\n"
        "Kaggle: https://www.kaggle.com/datasets/behrad3d/nasa-cmaps"
    )


def _extract_fd001_files(zf: zipfile.ZipFile, output_dir: Path) -> None:
    """Extract only the FD001 files from the zip archive.

    Handles nested directory structures inside the zip.
    """
    extracted = set()

    for member in zf.namelist():
        basename = Path(member).name
        if basename in EXPECTED_FILES:
            data = zf.read(member)
            dest = output_dir / basename
            dest.write_bytes(data)
            extracted.add(basename)
            logger.info("Extracted: %s", basename)

    missing = set(EXPECTED_FILES) - extracted
    if missing:
        raise KeyError(f"Zip archive missing required FD001 files: {missing}")


# ──────────────────────────────────────────────
# Parsing
# ──────────────────────────────────────────────


def load_train_data(data_dir: Path = RAW_DATA_DIR) -> pd.DataFrame:
    """Load and parse train_FD001.txt into a DataFrame with proper column names."""
    filepath = data_dir / "train_FD001.txt"
    logger.info("Loading training data from %s", filepath)

    df = pd.read_csv(
        filepath,
        sep=r"\s+",
        header=None,
        names=CMAPSS_COLUMN_NAMES,
    )
    logger.info("Loaded training data: %d rows, %d columns", len(df), len(df.columns))
    return df


def load_test_data(data_dir: Path = RAW_DATA_DIR) -> tuple[pd.DataFrame, pd.Series]:
    """Load test_FD001.txt and RUL_FD001.txt.

    Returns:
        Tuple of (test_df, rul_series).
    """
    test_path = data_dir / "test_FD001.txt"
    rul_path = data_dir / "RUL_FD001.txt"

    logger.info("Loading test data from %s", test_path)
    test_df = pd.read_csv(
        test_path,
        sep=r"\s+",
        header=None,
        names=CMAPSS_COLUMN_NAMES,
    )

    logger.info("Loading RUL labels from %s", rul_path)
    rul_series = pd.read_csv(rul_path, header=None, names=["rul"]).squeeze("columns")

    logger.info("Loaded test data: %d rows, %d units", len(test_df), test_df["unit_id"].nunique())
    return test_df, rul_series


# ──────────────────────────────────────────────
# Labeling
# ──────────────────────────────────────────────


def compute_rul(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Remaining Useful Life (RUL) for each row.

    RUL = max_cycle_for_unit - current_cycle.
    """
    df = df.copy()
    max_cycles = df.groupby("unit_id")["time_cycles"].max().rename("max_cycle")
    df = df.merge(max_cycles, on="unit_id")
    df["rul"] = df["max_cycle"] - df["time_cycles"]
    df = df.drop(columns=["max_cycle"])
    return df


def label_anomalies(df: pd.DataFrame, rul_threshold: int | None = None) -> pd.DataFrame:
    """Add binary anomaly labels based on RUL threshold.

    Rows with RUL < rul_threshold are labeled as anomalies (1).
    Threshold is read from model_config.yaml if not provided.
    """
    if rul_threshold is None:
        config = load_data_config()
        rul_threshold = config["rul_anomaly_threshold"]

    df = df.copy()

    # Ensure RUL column exists
    if "rul" not in df.columns:
        df = compute_rul(df)

    df["is_anomaly"] = (df["rul"] < rul_threshold).astype(int)
    n_anomaly = df["is_anomaly"].sum()
    logger.info(
        "Labeled anomalies: %d / %d rows (%.1f%%) with RUL < %d",
        n_anomaly,
        len(df),
        100 * n_anomaly / len(df),
        rul_threshold,
    )
    return df


# ──────────────────────────────────────────────
# Streaming simulation
# ──────────────────────────────────────────────


def add_simulated_timestamps(
    df: pd.DataFrame,
    start_time: datetime | None = None,
    interval_seconds: int = 1,
) -> pd.DataFrame:
    """Add synthetic timestamps to simulate real-time sensor ingestion.

    Each row within a unit gets a timestamp spaced by `interval_seconds`.
    Different units start at staggered times to simulate parallel engines.

    Args:
        df: DataFrame with unit_id and time_cycles columns.
        start_time: Base timestamp. Defaults to 2024-01-01T00:00:00Z.
        interval_seconds: Seconds between consecutive readings.

    Returns:
        DataFrame with a new 'timestamp' column.
    """
    if start_time is None:
        start_time = datetime(2024, 1, 1, tzinfo=timezone.utc)

    df = df.copy()
    timestamps = []

    for unit_id in df["unit_id"].unique():
        unit_mask = df["unit_id"] == unit_id
        n_rows = unit_mask.sum()

        # Stagger each unit's start by its ID (in hours) for realism
        unit_start = start_time + timedelta(hours=int(unit_id) - 1)
        unit_timestamps = [
            unit_start + timedelta(seconds=i * interval_seconds) for i in range(n_rows)
        ]
        timestamps.extend(unit_timestamps)

    df["timestamp"] = timestamps
    logger.info("Added simulated timestamps starting from %s", start_time.isoformat())
    return df


# ──────────────────────────────────────────────
# Full pipeline
# ──────────────────────────────────────────────


def ingest_train_data(
    data_dir: Path = RAW_DATA_DIR,
    download: bool = False,
) -> pd.DataFrame:
    """Full ingestion pipeline: download (optional) → parse → label → timestamp.

    Returns:
        DataFrame with columns: unit_id, time_cycles, op_settings, sensors,
        rul, is_anomaly, timestamp.
    """
    if download:
        download_cmapss(data_dir)

    df = load_train_data(data_dir)
    df = compute_rul(df)
    df = label_anomalies(df)
    df = add_simulated_timestamps(df)

    logger.info(
        "Ingestion complete: %d rows, %d units, columns: %s",
        len(df),
        df["unit_id"].nunique(),
        list(df.columns),
    )
    return df


def ingest_test_data(
    data_dir: Path = RAW_DATA_DIR,
    download: bool = False,
) -> tuple[pd.DataFrame, pd.Series]:
    """Ingest test data with timestamps.

    Returns:
        Tuple of (test_df with timestamps, rul_series).
    """
    if download:
        download_cmapss(data_dir)

    test_df, rul_series = load_test_data(data_dir)
    test_df = add_simulated_timestamps(test_df)

    return test_df, rul_series


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="CMAPSS FD001 Data Ingestion")
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download CMAPSS dataset before ingestion",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=RAW_DATA_DIR,
        help=f"Directory for raw data files (default: {RAW_DATA_DIR})",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(name)s — %(message)s")

    df = ingest_train_data(data_dir=args.data_dir, download=args.download)

    print(f"\n{'='*60}")
    print(f"Ingested training data: {len(df)} rows, {df['unit_id'].nunique()} units")
    print(f"Anomaly rate: {df['is_anomaly'].mean():.1%}")
    print(f"Columns: {list(df.columns)}")
    print(f"{'='*60}")
    print(df.head())


if __name__ == "__main__":
    main()
