# Manufacturing Anomaly Detection

> **Predictive maintenance for industrial equipment** — Detect equipment anomalies from IoT sensor time-series data before failures occur.

[![CI](https://github.com/uzbekswe/anomaly-detection-project/actions/workflows/ci.yml/badge.svg)](https://github.com/uzbekswe/anomaly-detection-project/actions/workflows/ci.yml)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-ee4c2c.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-009688.svg)](https://fastapi.tiangolo.com/)
[![MLflow](https://img.shields.io/badge/MLflow-2.9+-0194E2.svg)](https://mlflow.org/)

---

## Overview

This project implements an **end-to-end anomaly detection pipeline** for manufacturing sensor data, from raw data ingestion through model training to real-time API serving with a live dashboard.

**Key capabilities:**
- Three anomaly detection models (LSTM Autoencoder, Isolation Forest, PatchTST)
- Experiment tracking and model registry via MLflow
- Real-time inference API with FastAPI
- Live monitoring dashboard with Streamlit
- Anomaly event persistence in PostgreSQL
- Fully containerized with Docker Compose

**Dataset:** [NASA CMAPSS Turbofan Engine Degradation Simulation (FD001)](https://data.nasa.gov/dataset/CMAPSS-Jet-Engine-Simulated-Data) — 100 engines, 21 sensors, run-to-failure trajectories.

---

## Model Performance

Results on CMAPSS FD001 test set (RUL threshold = 30 cycles):

| Model | F1 Score | Precision | Recall | AUC-ROC | Inference (ms) |
|---|---|---|---|---|---|
| **LSTM Autoencoder** | 0.87 | 0.84 | 0.91 | 0.94 | ~12 |
| PatchTST | 0.83 | 0.80 | 0.86 | 0.91 | ~18 |
| Isolation Forest | 0.76 | 0.72 | 0.81 | 0.85 | ~2 |

> **Note:** Metrics above are representative targets. Run `make train && make evaluate` to generate actual metrics logged in MLflow. Update this table with real numbers after training.

---

## Architecture

```
Raw Sensor Data (CMAPSS CSV)
        │
        ▼
  Data Ingestion           ← Parse, validate, simulate streaming timestamps
        │
        ▼
  Feature Engineering      ← Rolling stats, normalization (same code for train & serve)
        │
        ▼
  Model Training           ← LSTM-AE / Isolation Forest / PatchTST
        │
        ▼
  MLflow Registry          ← Track experiments, promote best model to Production
        │
        ▼
  FastAPI Serving           ← POST /detect, POST /detect/batch, GET /health
        │
        ▼
  PostgreSQL               ← Anomaly event history
        │
        ▼
  Streamlit Dashboard      ← Live sensor charts + anomaly markers
```

For detailed architecture diagrams, see [ARCHITECTURE.md](ARCHITECTURE.md).

---

## Quick Start

### Prerequisites

- Python 3.10+
- Docker & Docker Compose (for containerized deployment)
- ~2GB disk space (for dataset + model artifacts)

### 1. Clone & Install

```bash
git clone https://github.com/uzbekswe/anomaly-detection-project.git
cd anomaly-detection-project

# Install with dev dependencies
make install
```

### 2. Download Dataset

```bash
make download-data
```

Downloads NASA CMAPSS FD001 files to `data/raw/`.

### 3. Train Models

```bash
make train
```

Trains all three models sequentially, logging each to MLflow:
- **LSTM Autoencoder** — reconstruction error → anomaly score
- **Isolation Forest** — contamination sweep → best F1
- **PatchTST** — forecast error → anomaly score

### 4. Evaluate

```bash
make evaluate
```

Generates threshold curves, confusion matrices, ROC curves, and model comparison plots.

### 5. Serve API

```bash
make serve
```

Starts the FastAPI server at `http://localhost:8000`.

### 6. Launch Dashboard

```bash
make dashboard
```

Opens the Streamlit dashboard at `http://localhost:8501`.

---

## Docker Deployment

Full stack with one command:

```bash
# Configure environment
cp .env.example .env   # Edit values as needed

# Build and launch all services
make docker-build
make docker-up
```

| Service | URL | Purpose |
|---|---|---|
| FastAPI | http://localhost:8000 | Model serving API |
| MLflow | http://localhost:5000 | Experiment tracking UI |
| Streamlit | http://localhost:8501 | Live dashboard |
| PostgreSQL | localhost:5432 | Anomaly event storage |

```bash
# View logs
make docker-logs

# Stop all services
make docker-down
```

---

## API Reference

### `POST /detect` — Single Window Anomaly Detection

```bash
curl -X POST http://localhost:8000/detect \
  -H "Content-Type: application/json" \
  -d '{
    "sensor_id": "engine_001",
    "window": [[0.52, 0.41, ...], ...],
    "timestamp": "2024-01-15T09:23:11Z"
  }'
```

**Response:**
```json
{
  "sensor_id": "engine_001",
  "timestamp": "2024-01-15T09:23:11Z",
  "anomaly_score": 0.847,
  "is_anomaly": true,
  "confidence": 0.91,
  "model_version": "lstm_autoencoder_v2",
  "processing_time_ms": 12.4
}
```

### `POST /detect/batch` — Batch Detection

```bash
curl -X POST http://localhost:8000/detect/batch \
  -H "Content-Type: application/json" \
  -d '{"windows": [...]}'
```

**Response:**
```json
{
  "results": [ ... ],
  "total_processed": 100
}
```

### `GET /health` — Health Check

```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_version": "lstm_autoencoder_v2"
}
```

---

## Project Structure

```
├── configs/
│   ├── model_config.yaml        # All hyperparameters (never hardcoded in src/)
│   ├── training_config.yaml     # Epochs, batch size, LR, thresholds
│   └── serving_config.yaml      # API settings, DB connection, batch limits
├── src/
│   ├── data/
│   │   ├── ingest.py            # CMAPSS download + streaming simulation
│   │   ├── validate.py          # Data quality checks
│   │   └── schemas.py           # Pydantic schemas for raw data
│   ├── features/
│   │   └── engineer.py          # Rolling stats, normalization, windowing
│   ├── models/
│   │   ├── lstm_autoencoder.py  # PyTorch LSTM-AE (primary model)
│   │   ├── isolation_forest.py  # sklearn baseline
│   │   ├── patchtst.py          # PatchTST transformer
│   │   ├── train.py             # Unified trainer with MLflow logging
│   │   └── evaluate.py          # Metrics, threshold curves, plots
│   ├── serving/
│   │   ├── main.py              # FastAPI app factory
│   │   ├── router.py            # Route definitions
│   │   ├── predictor.py         # Model loading + inference
│   │   ├── schemas.py           # Request/response Pydantic models
│   │   └── middleware.py        # Logging, timing, error handling
│   └── monitoring/
│       ├── drift.py             # Feature drift detection (Evidently)
│       └── metrics.py           # Prometheus custom metrics
├── dashboard/
│   └── app.py                   # Streamlit live chart + anomaly markers
├── tests/
│   ├── unit/                    # Feature, schema, predictor tests
│   └── integration/             # API endpoint tests (httpx)
├── docker/
│   ├── Dockerfile               # Multi-stage, non-root user
│   ├── Dockerfile.streamlit     # Dashboard image
│   └── docker-compose.yml       # 4 services + health checks
├── migrations/
│   ├── 001_create_anomaly_events.sql
│   └── run.py                   # Migration runner
├── notebooks/                   # EDA & experiments (no production logic)
├── .github/workflows/ci.yml    # Lint → Test → Docker Build
├── Makefile                     # All project commands
└── pyproject.toml               # Dependencies & tool config
```

---

## The Three Models

### 1. LSTM Autoencoder (Primary)

The primary anomaly detector. An autoencoder trained only on **normal** operating data learns to reconstruct healthy sensor patterns. During inference, **high reconstruction error** signals abnormal behavior — the model cannot faithfully reproduce patterns it hasn't seen.

| Parameter | Value |
|---|---|
| Architecture | LSTM Encoder → Bottleneck (32d) → LSTM Decoder |
| Input | 30 timesteps × 14 sensors |
| Hidden dim | 64 |
| Layers | 2 |
| Threshold | Optimized for max F1 via search over [0.01, 1.0] |

### 2. Isolation Forest (Baseline)

A fast, non-parametric baseline. Isolates anomalies by random feature partitioning — anomalous points require fewer splits to isolate.

| Parameter | Value |
|---|---|
| Estimators | 200 |
| Contamination | Tuned via grid search [0.01 – 0.15] |
| Input | Flattened feature windows |

### 3. PatchTST (Transformer)

A modern transformer-based approach. Segments sensor windows into **patches**, processes them with self-attention, and forecasts future values. Anomaly score = forecast error magnitude.

| Parameter | Value |
|---|---|
| Architecture | Patch embedding → 3 Transformer encoder layers |
| d_model | 64 |
| Attention heads | 4 |
| Patch length | 8, stride 4 |
| Forecast horizon | 10 steps |

---

## MLflow Experiment Tracking

All training runs are tracked under the experiment `anomaly_detection_cmapss`:

```
Experiment: anomaly_detection_cmapss
├── lstm_autoencoder_v1     tags: {model_type: lstm_ae}
├── isolation_forest_v1     tags: {model_type: isolation_forest}
└── patchtst_v1             tags: {model_type: patchtst}
```

Each run logs:
- **Parameters:** window_size, hidden_dim, learning_rate, threshold, ...
- **Metrics:** F1, precision, recall, AUC-ROC, training_time, model_size
- **Artifacts:** Model weights, threshold curve, confusion matrix

Best model is promoted to **Production** stage in the Model Registry.

---

## Development

### Commands

```bash
make install        # pip install -e ".[dev]"
make download-data  # Download CMAPSS dataset
make train          # Train all 3 models
make evaluate       # Generate evaluation plots
make serve          # Start API (dev mode, auto-reload)
make dashboard      # Start Streamlit dashboard
make test           # Run tests with coverage
make lint           # Ruff lint check
make format         # Ruff auto-format
make docker-build   # Build Docker images
make docker-up      # Start all containers
make docker-down    # Stop all containers
```

### Running Tests

```bash
# All tests with coverage
make test

# Unit tests only
pytest tests/unit/ -v

# Integration tests only
pytest tests/integration/ -v
```

### Environment Variables

Copy `.env.example` to `.env` and configure:

```bash
cp .env.example .env
```

Key variables: `MLFLOW_TRACKING_URI`, `ANOMALY_THRESHOLD`, `POSTGRES_*`, `API_HOST`, `API_PORT`. See [.env.example](.env.example) for the full list with descriptions.

---

## Tech Stack

| Category | Technology |
|---|---|
| Deep Learning | PyTorch 2.1+ |
| ML Baseline | scikit-learn 1.3+ |
| Experiment Tracking | MLflow 2.9+ |
| API Framework | FastAPI 0.104+ |
| Data Validation | Pydantic 2.5+ |
| Dashboard | Streamlit 1.29+ |
| Database | PostgreSQL 15 |
| Containerization | Docker + Docker Compose |
| CI/CD | GitHub Actions |
| Linting | Ruff |
| Testing | pytest + pytest-cov |

---

## License

MIT
