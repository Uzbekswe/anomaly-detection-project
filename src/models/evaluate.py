"""Model evaluation — threshold curves, metrics comparison, and visualization.

Loads trained models and MLflow runs, produces:
    - Threshold vs F1/Precision/Recall curves per model
    - Confusion matrices per model
    - Side-by-side model comparison table
    - Plots saved as artifacts to MLflow

Usage:
    python src/models/evaluate.py --experiment anomaly_detection_cmapss
    python src/models/evaluate.py --experiment anomaly_detection_cmapss --data-dir data/raw
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend for server/CI

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import yaml
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PLOTS_DIR = PROJECT_ROOT / "data" / "plots"


# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────


def load_training_config(
    config_path: Path = PROJECT_ROOT / "configs" / "training_config.yaml",
) -> dict:
    """Load training configuration."""
    with open(config_path) as f:
        return yaml.safe_load(f)


# ──────────────────────────────────────────────
# Threshold curve
# ──────────────────────────────────────────────


def plot_threshold_curve(
    scores: np.ndarray,
    labels: np.ndarray,
    model_name: str,
    search_min: float = 0.01,
    search_max: float = 1.0,
    search_steps: int = 100,
    save_path: Path | None = None,
) -> Path:
    """Plot F1, Precision, Recall vs threshold.

    Args:
        scores: Anomaly scores (higher = more anomalous).
        labels: Ground truth binary labels.
        model_name: Name for the plot title.
        search_min: Minimum threshold.
        search_max: Maximum threshold.
        search_steps: Number of thresholds to evaluate.
        save_path: Path to save the plot. Auto-generated if None.

    Returns:
        Path to saved plot.
    """
    thresholds = np.linspace(search_min, search_max, search_steps)
    f1_scores = []
    precisions = []
    recalls = []

    for t in thresholds:
        preds = (scores > t).astype(int)
        f1_scores.append(f1_score(labels, preds, zero_division=0))
        precisions.append(precision_score(labels, preds, zero_division=0))
        recalls.append(recall_score(labels, preds, zero_division=0))

    # Find best threshold
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(thresholds, f1_scores, label="F1", linewidth=2)
    ax.plot(thresholds, precisions, label="Precision", linewidth=2)
    ax.plot(thresholds, recalls, label="Recall", linewidth=2)
    ax.axvline(x=best_threshold, color="red", linestyle="--", alpha=0.7,
               label=f"Best threshold={best_threshold:.4f}")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Score")
    ax.set_title(f"{model_name} — Threshold vs Metrics")
    ax.legend()
    ax.grid(True, alpha=0.3)

    if save_path is None:
        save_path = PLOTS_DIR / f"{model_name}_threshold_curve.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info("Saved threshold curve: %s", save_path)
    return save_path


# ──────────────────────────────────────────────
# Confusion matrix
# ──────────────────────────────────────────────


def plot_confusion_matrix(
    labels: np.ndarray,
    predictions: np.ndarray,
    model_name: str,
    save_path: Path | None = None,
) -> Path:
    """Plot and save confusion matrix.

    Returns:
        Path to saved plot.
    """
    cm = confusion_matrix(labels, predictions)

    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(cm, display_labels=["Normal", "Anomaly"])
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    ax.set_title(f"{model_name} — Confusion Matrix")

    if save_path is None:
        save_path = PLOTS_DIR / f"{model_name}_confusion_matrix.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info("Saved confusion matrix: %s", save_path)
    return save_path


# ──────────────────────────────────────────────
# ROC curve
# ──────────────────────────────────────────────


def plot_roc_curve(
    scores: np.ndarray,
    labels: np.ndarray,
    model_name: str,
    save_path: Path | None = None,
) -> Path:
    """Plot ROC curve with AUC.

    Returns:
        Path to saved plot.
    """
    fpr, tpr, _ = roc_curve(labels, scores)
    auc = roc_auc_score(labels, scores)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, linewidth=2, label=f"AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"{model_name} — ROC Curve")
    ax.legend()
    ax.grid(True, alpha=0.3)

    if save_path is None:
        save_path = PLOTS_DIR / f"{model_name}_roc_curve.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info("Saved ROC curve: %s", save_path)
    return save_path


# ──────────────────────────────────────────────
# Score distribution
# ──────────────────────────────────────────────


def plot_score_distribution(
    scores: np.ndarray,
    labels: np.ndarray,
    model_name: str,
    threshold: float,
    save_path: Path | None = None,
) -> Path:
    """Plot anomaly score distribution for normal vs anomaly samples.

    Returns:
        Path to saved plot.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    normal_scores = scores[labels == 0]
    anomaly_scores = scores[labels == 1]

    ax.hist(normal_scores, bins=50, alpha=0.6, label="Normal", color="blue", density=True)
    ax.hist(anomaly_scores, bins=50, alpha=0.6, label="Anomaly", color="red", density=True)
    ax.axvline(x=threshold, color="black", linestyle="--", linewidth=2,
               label=f"Threshold={threshold:.4f}")
    ax.set_xlabel("Anomaly Score")
    ax.set_ylabel("Density")
    ax.set_title(f"{model_name} — Score Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)

    if save_path is None:
        save_path = PLOTS_DIR / f"{model_name}_score_distribution.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info("Saved score distribution: %s", save_path)
    return save_path


# ──────────────────────────────────────────────
# Model comparison
# ──────────────────────────────────────────────


def plot_model_comparison(
    results: dict[str, dict[str, float]],
    save_path: Path | None = None,
) -> Path:
    """Plot side-by-side bar chart comparing all models.

    Args:
        results: {model_name: {metric_name: value}}

    Returns:
        Path to saved plot.
    """
    metrics = ["f1", "precision", "recall", "auc_roc"]
    model_names = list(results.keys())
    x = np.arange(len(metrics))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, model_name in enumerate(model_names):
        values = [results[model_name].get(m, 0) for m in metrics]
        ax.bar(x + i * width, values, width, label=model_name, alpha=0.85)

        # Add value labels on bars
        for j, v in enumerate(values):
            ax.text(x[j] + i * width, v + 0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xlabel("Metric")
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison — Test Set Metrics")
    ax.set_xticks(x + width * (len(model_names) - 1) / 2)
    ax.set_xticklabels([m.upper().replace("_", " ") for m in metrics])
    ax.set_ylim(0, 1.15)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    if save_path is None:
        save_path = PLOTS_DIR / "model_comparison.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info("Saved model comparison: %s", save_path)
    return save_path


# ──────────────────────────────────────────────
# Evaluate from MLflow experiment
# ──────────────────────────────────────────────


def get_experiment_results(experiment_name: str) -> dict[str, dict]:
    """Load all run results from an MLflow experiment.

    Returns:
        {model_type: {params: {...}, metrics: {...}}}
    """
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"Experiment '{experiment_name}' not found in MLflow")

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        output_format="list",
    )

    results = {}
    for run in runs:
        model_type = run.data.tags.get("model_type", "unknown")
        results[model_type] = {
            "run_id": run.info.run_id,
            "params": run.data.params,
            "metrics": run.data.metrics,
            "status": run.info.status,
        }
        logger.info(
            "Loaded run: %s (%s) — test_f1=%.4f",
            model_type,
            run.info.run_id[:8],
            run.data.metrics.get("test_f1", 0),
        )

    return results


def print_comparison_table(results: dict[str, dict]) -> None:
    """Print a formatted comparison table to stdout."""
    metrics_to_show = [
        "test_f1", "test_precision", "test_recall", "test_auc_roc",
        "threshold", "training_time_sec", "model_size_mb",
    ]

    header = f"{'Metric':<25}" + "".join(f"{m:<20}" for m in results)
    print("\n" + "=" * len(header))
    print("MODEL COMPARISON")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for metric in metrics_to_show:
        row = f"{metric:<25}"
        for model_data in results.values():
            val = model_data["metrics"].get(metric, "N/A")
            if isinstance(val, float):
                row += f"{val:<20.4f}"
            else:
                row += f"{str(val):<20}"
        print(row)

    print("=" * len(header))


# ──────────────────────────────────────────────
# Full evaluation pipeline
# ──────────────────────────────────────────────


def evaluate_all_models(
    experiment_name: str,
    data_dir: Path | None = None,
) -> dict[str, dict[str, float]]:
    """Run full evaluation: load models, compute metrics, generate all plots.

    Args:
        experiment_name: MLflow experiment name.
        data_dir: Path to raw data directory (for re-running inference).

    Returns:
        {model_name: {metric: value}} for all models.
    """
    # Load experiment results from MLflow
    results = get_experiment_results(experiment_name)
    print_comparison_table(results)

    # Prepare comparison data for plotting
    comparison = {}
    for model_type, data in results.items():
        metrics = data["metrics"]
        comparison[model_type] = {
            "f1": metrics.get("test_f1", 0),
            "precision": metrics.get("test_precision", 0),
            "recall": metrics.get("test_recall", 0),
            "auc_roc": metrics.get("test_auc_roc", 0),
        }

    # Generate comparison plot
    if comparison:
        plot_path = plot_model_comparison(comparison)
        logger.info("Model comparison plot saved: %s", plot_path)

    return comparison


# ──────────────────────────────────────────────
# Evaluate a single model with full plots
# ──────────────────────────────────────────────


def evaluate_single_model(
    model_name: str,
    scores: np.ndarray,
    labels: np.ndarray,
    threshold: float,
    search_min: float = 0.01,
    search_max: float = 1.0,
    search_steps: int = 100,
) -> dict[str, float]:
    """Generate all evaluation plots and metrics for a single model.

    Called during training to produce artifacts for MLflow logging.

    Args:
        model_name: Display name for plots.
        scores: Anomaly scores on test set.
        labels: Ground truth labels on test set.
        threshold: Optimal threshold from validation.
        search_min: Threshold search range min.
        search_max: Threshold search range max.
        search_steps: Number of threshold steps.

    Returns:
        Dict of test metrics.
    """
    predictions = (scores > threshold).astype(int)

    # Compute metrics
    metrics = {
        "f1": f1_score(labels, predictions, zero_division=0),
        "precision": precision_score(labels, predictions, zero_division=0),
        "recall": recall_score(labels, predictions, zero_division=0),
    }
    if len(np.unique(labels)) > 1:
        metrics["auc_roc"] = roc_auc_score(labels, scores)

    # Generate plots
    plots = []
    plots.append(plot_threshold_curve(
        scores, labels, model_name, search_min, search_max, search_steps,
    ))
    plots.append(plot_confusion_matrix(labels, predictions, model_name))
    plots.append(plot_score_distribution(scores, labels, model_name, threshold))

    if len(np.unique(labels)) > 1:
        plots.append(plot_roc_curve(scores, labels, model_name))

    # Log plots to active MLflow run if one exists
    if mlflow.active_run():
        for plot_path in plots:
            mlflow.log_artifact(str(plot_path))

    logger.info(
        "%s evaluation: F1=%.4f, P=%.4f, R=%.4f, AUC=%.4f",
        model_name,
        metrics["f1"],
        metrics["precision"],
        metrics["recall"],
        metrics.get("auc_roc", 0),
    )
    return metrics


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Model Evaluation & Comparison")
    parser.add_argument(
        "--experiment",
        type=str,
        default="anomaly_detection_cmapss",
        help="MLflow experiment name",
    )
    parser.add_argument(
        "--tracking-uri",
        type=str,
        default=None,
        help="MLflow tracking URI (default: from env or http://localhost:5000)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Path to raw data directory for re-running inference",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s — %(name)s — %(levelname)s — %(message)s",
    )

    # Setup MLflow
    tracking_uri = args.tracking_uri or os.environ.get(
        "MLFLOW_TRACKING_URI", "http://localhost:5000"
    )
    mlflow.set_tracking_uri(tracking_uri)

    # Run evaluation
    evaluate_all_models(
        experiment_name=args.experiment,
        data_dir=args.data_dir,
    )

    print("\nEvaluation complete. Plots saved to:", PLOTS_DIR)


if __name__ == "__main__":
    main()
