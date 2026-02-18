#!/usr/bin/env python
"""Quick training script that wraps src/models/train.py main()."""
import os
import sys
import logging

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Use local MLflow tracking
os.environ.setdefault("MLFLOW_TRACKING_URI", "./mlruns")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(name)s — %(levelname)s — %(message)s",
)

if __name__ == "__main__":
    from src.models.train import main
    main()
