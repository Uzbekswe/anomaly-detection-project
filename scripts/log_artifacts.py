"""Script to log locally saved artifacts to MLflow.

This script connects to the MLflow Tracking Server, finds the latest runs for
each model type, and logs the corresponding locally saved artifacts.
"""

import argparse
import logging
from pathlib import Path

import mlflow
import yaml
from mlflow.tracking import MlflowClient

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_ARTIFACT_PATH = PROJECT_ROOT / "data" / "artifacts"


def main() -> None:
    parser = argparse.ArgumentParser(description="Log artifacts to MLflow")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "training_config.yaml",
        help="Path to training_config.yaml",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s — %(name)s — %(levelname)s — %(message)s",
    )

    with open(args.config) as f:
        training_config = yaml.safe_load(f)

    mlflow_config = training_config["mlflow"]
    mlflow.set_tracking_uri(mlflow_config["tracking_uri"])
    client = MlflowClient()

    experiment = client.get_experiment_by_name(mlflow_config["experiment_name"])
    if not experiment:
        logger.error("Experiment '%s' not found.", mlflow_config["experiment_name"])
        return

    runs = client.search_runs(experiment.experiment_id, order_by=["start_time DESC"])

    model_types = ["lstm_ae", "isolation_forest", "patchtst"]
    for model_type in model_types:
        latest_run = next((r for r in runs if r.data.tags.get("model_type") == model_type), None)
        if not latest_run:
            logger.warning("No run found for model_type: %s", model_type)
            continue

        run_id = latest_run.info.run_id
        logger.info("Found latest run for %s: %s", model_type, run_id)

        if model_type == "lstm_ae":
            artifact_path = LOCAL_ARTIFACT_PATH / "lstm_autoencoder.pt"
            if artifact_path.exists():
                client.log_artifact(run_id, str(artifact_path), artifact_path="model")
                logger.info("Logged %s to run %s", artifact_path.name, run_id)

        elif model_type == "isolation_forest":
            artifact_path = LOCAL_ARTIFACT_PATH / "isolation_forest.joblib"
            if artifact_path.exists():
                client.log_artifact(run_id, str(artifact_path), artifact_path="model")
                logger.info("Logged %s to run %s", artifact_path.name, run_id)

        elif model_type == "patchtst":
            artifact_path = LOCAL_ARTIFACT_PATH / "patchtst.pt"
            if artifact_path.exists():
                client.log_artifact(run_id, str(artifact_path), artifact_path="model")
                logger.info("Logged %s to run %s", artifact_path.name, run_id)

    # Log the scaler to the "feature_engineering" run
    feature_eng_run = next((r for r in runs if r.data.tags.get("mlflow.runName") == "feature_engineering"), None)
    if feature_eng_run:
        run_id = feature_eng_run.info.run_id
        scaler_path = LOCAL_ARTIFACT_PATH / "scaler.json"
        if scaler_path.exists():
            client.log_artifact(run_id, str(scaler_path))
            logger.info("Logged scaler.json to feature_engineering run %s", run_id)


if __name__ == "__main__":
    main()
