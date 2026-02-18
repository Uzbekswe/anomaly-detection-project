# CLAUDE.md — Manufacturing Anomaly Detection (IoT Time Series)

> This file is the persistent memory for Claude Code across all sessions on this project.
> Read this fully before doing anything. Never hardcode values that belong in configs/.

---

## 1. Project Identity

| Field | Value |
|---|---|
| Project Name | Manufacturing Anomaly Detection |
| Business Goal | Detect equipment anomalies from sensor data before failures (predictive maintenance) |
| Target Market | Korean smart factory / manufacturing AI roles (Samsung, Hyundai, POSCO ecosystem) |
| Target Salary Band | 60M KRW (Junior–Middle ML Engineer) |
| Dataset | NASA CMAPSS Turbofan Engine Degradation (FD001 subset to start) |
| Status | In active development |

---

## 2. Architecture Overview

```
Raw Sensor Data (CMAPSS CSV)
        │
        ▼
  src/data/ingest.py          ← Parse, validate, simulate streaming timestamps
        │
        ▼
  src/features/engineer.py    ← Rolling stats, normalization (IDENTICAL logic used in serving)
        │
        ▼
  src/models/
    ├── lstm_autoencoder.py   ← PyTorch LSTM-AE, reconstruction error → anomaly score
    ├── isolation_forest.py   ← sklearn baseline, contamination tuned via MLflow
    └── patchtst.py           ← PatchTST forecasting-based, anomaly = forecast error > threshold
        │
        ▼
  MLflow Tracking Server      ← All 3 models logged as runs under one experiment
  (Model Registry)            ← Best model promoted to "Production" stage
        │
        ▼
  src/serving/main.py         ← FastAPI: POST /detect, POST /detect/batch, GET /health
        │
        ▼
  PostgreSQL                  ← Stores anomaly event history
        │
        ▼
  Streamlit Dashboard         ← Live sensor chart, anomaly markers, reconstruction error plot
```

---

## 3. Repository Structure (Do Not Deviate)

```
project-root/
├── CLAUDE.md                  ← This file
├── README.md                  ← Public-facing doc (HR/TL reads this first)
├── ARCHITECTURE.md            ← Detailed Mermaid diagrams + design decisions
├── Makefile                   ← make train | make serve | make test | make docker-build
├── pyproject.toml             ← Dependency management (not raw requirements.txt)
├── .env.example               ← Documented env vars, NO actual secrets
├── .gitignore
├── configs/
│   ├── model_config.yaml      ← All hyperparameters live here, never hardcoded
│   ├── training_config.yaml   ← Epochs, batch size, learning rate, thresholds
│   └── serving_config.yaml    ← API settings, timeout, max batch size
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── ingest.py          ← Download/load CMAPSS, simulate stream with timestamps
│   │   ├── validate.py        ← Data quality checks (Great Expectations or manual)
│   │   └── schemas.py         ← Pydantic schemas for raw data shape
│   ├── features/
│   │   ├── __init__.py
│   │   └── engineer.py        ← Rolling mean/std, min-max normalization, window creation
│   │                            CRITICAL: This code runs in BOTH training and serving
│   ├── models/
│   │   ├── __init__.py
│   │   ├── lstm_autoencoder.py
│   │   ├── isolation_forest.py
│   │   ├── patchtst.py
│   │   ├── train.py           ← Unified training entry point, logs all 3 to MLflow
│   │   └── evaluate.py        ← F1/Precision/Recall at multiple thresholds, plots
│   ├── serving/
│   │   ├── __init__.py
│   │   ├── main.py            ← FastAPI app factory
│   │   ├── router.py          ← Route definitions
│   │   ├── schemas.py         ← Pydantic request/response models (SensorWindow, PredictionResponse)
│   │   ├── predictor.py       ← Loads model from MLflow registry, runs inference
│   │   └── middleware.py      ← Logging, timing, error handling middleware
│   └── monitoring/
│       ├── __init__.py
│       ├── drift.py           ← Evidently-based feature drift detection
│       └── metrics.py         ← Prometheus custom metrics (anomaly rate, latency)
├── dashboard/
│   └── app.py                 ← Streamlit dashboard (separate from src/ — not production code)
├── tests/
│   ├── unit/
│   │   ├── test_features.py
│   │   ├── test_schemas.py
│   │   └── test_predictor.py
│   └── integration/
│       └── test_api.py        ← Hit real FastAPI endpoints with httpx
├── notebooks/
│   ├── 01_eda.ipynb           ← EDA only, no production logic
│   ├── 02_model_experiments.ipynb
│   └── 03_threshold_analysis.ipynb
├── docker/
│   ├── Dockerfile             ← Multi-stage build (builder → runtime, non-root user)
│   ├── Dockerfile.streamlit
│   └── docker-compose.yml     ← All 4 services: fastapi, mlflow, streamlit, postgres
├── .github/
│   └── workflows/
│       └── ci.yml             ← lint → test → build → (future) push ECR
└── data/
    └── raw/                   ← CMAPSS files land here (gitignored)
```

---

## 4. The Three Models — Key Decisions

### Model 1: LSTM Autoencoder (Primary / Most Impressive)
- **Framework:** PyTorch
- **Input:** Sliding window of 30 timesteps × N sensor channels
- **Architecture:** LSTM encoder → bottleneck → LSTM decoder → reconstruct input
- **Anomaly logic:** Reconstruction error (MSE) > learned threshold = anomaly
- **Threshold tuning:** Logged in MLflow across range, pick threshold maximizing F1
- **Why this model:** Shows PyTorch custom training loop skill, most visually explainable

### Model 2: Isolation Forest (Baseline)
- **Framework:** sklearn
- **Input:** Flattened feature window or summary stats per window
- **Key param:** `contamination` — tune this, log results to MLflow
- **Why this model:** Fast baseline to beat, shows you know to start simple

### Model 3: PatchTST (Forecasting-Based)
- **Framework:** PyTorch (HuggingFace transformers or manual impl)
- **Input:** Sensor window → forecast next N steps → anomaly score = forecast error
- **Why this model:** Shows awareness of modern time-series transformers, differentiator

### MLflow Experiment Structure
```
Experiment: "anomaly_detection_cmapss"
├── Run: lstm_autoencoder_v1        tags: {model_type: lstm_ae}
├── Run: isolation_forest_v1        tags: {model_type: isolation_forest}
└── Run: patchtst_v1                tags: {model_type: patchtst}

Logged per run:
  Params: window_size, hidden_dim, num_layers, threshold, contamination, ...
  Metrics: f1, precision, recall, auc_roc, training_time_sec, model_size_mb
  Artifacts: model weights, threshold curve plot, confusion matrix
```

---

## 5. API Contract (Do Not Change Without Updating Tests)

### POST /detect
```json
Request:
{
  "sensor_id": "engine_001",
  "window": [[0.52, 0.41, ...], ...],   // shape: [30, num_sensors]
  "timestamp": "2024-01-15T09:23:11Z"
}

Response:
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

### POST /detect/batch
```json
Request:  { "windows": [ ...array of single /detect request bodies... ] }
Response: { "results": [ ...array of /detect responses... ], "total_processed": 100 }
```

### GET /health
```json
{ "status": "healthy", "model_loaded": true, "model_version": "lstm_autoencoder_v2" }
```

---

## 6. Database Schema (PostgreSQL)

```sql
-- Table: anomaly_events
CREATE TABLE anomaly_events (
    id              SERIAL PRIMARY KEY,
    sensor_id       VARCHAR(64) NOT NULL,
    detected_at     TIMESTAMPTZ NOT NULL,
    anomaly_score   FLOAT NOT NULL,
    is_anomaly      BOOLEAN NOT NULL,
    confidence      FLOAT NOT NULL,
    model_version   VARCHAR(128) NOT NULL,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_anomaly_events_sensor_id ON anomaly_events(sensor_id);
CREATE INDEX idx_anomaly_events_detected_at ON anomaly_events(detected_at);
```

---

## 7. Environment Variables (All Must Be in .env.example)

```bash
# MLflow
MLFLOW_TRACKING_URI=http://mlflow:5000
MLFLOW_EXPERIMENT_NAME=anomaly_detection_cmapss

# Model
MODEL_NAME=lstm_autoencoder
MODEL_STAGE=Production
ANOMALY_THRESHOLD=0.65         # overridden by MLflow logged threshold if loaded from registry

# Database
POSTGRES_HOST=postgres
POSTGRES_PORT=5432
POSTGRES_DB=anomaly_db
POSTGRES_USER=anomaly_user
POSTGRES_PASSWORD=changeme_in_prod

# API
API_HOST=0.0.0.0
API_PORT=8000
LOG_LEVEL=INFO
MAX_BATCH_SIZE=500

# Streamlit
STREAMLIT_API_URL=http://fastapi:8000
STREAMLIT_REFRESH_INTERVAL_SEC=5
```

---

## 8. Docker Services (docker-compose.yml)

| Service | Image | Port | Purpose |
|---|---|---|---|
| `fastapi` | Custom (docker/Dockerfile) | 8000 | Model serving API |
| `mlflow` | ghcr.io/mlflow/mlflow | 5000 | Experiment tracking UI |
| `streamlit` | Custom (docker/Dockerfile.streamlit) | 8501 | Live dashboard |
| `postgres` | postgres:15-alpine | 5432 | Anomaly event history |

All services communicate on a shared Docker network: `anomaly-net`
Health checks required on fastapi and postgres before dependent services start.

---

## 9. Key Commands (Makefile targets)

```bash
make install        # pip install -e ".[dev]"
make download-data  # python src/data/ingest.py --download
make train          # python src/models/train.py --config configs/training_config.yaml
make evaluate       # python src/models/evaluate.py --experiment anomaly_detection_cmapss
make serve          # uvicorn src.serving.main:app --reload --port 8000
make dashboard      # streamlit run dashboard/app.py
make test           # pytest tests/ --cov=src --cov-report=term-missing
make lint           # ruff check src/ tests/
make format         # ruff format src/ tests/
make docker-build   # docker compose build
make docker-up      # docker compose up -d
make docker-down    # docker compose down
make docker-logs    # docker compose logs -f fastapi
```

---

## 10. Strict Rules — Never Violate These

1. **No hardcoded thresholds or hyperparameters in src/.** All values come from `configs/*.yaml`.
2. **Feature engineering code in `src/features/engineer.py` must be identical for training and serving.** No copy-paste divergence. Import the same function in both places.
3. **No production logic in `notebooks/`.** If notebook code is useful, refactor it into `src/` first.
4. **No secrets in the repo.** Only `.env.example` with placeholder values. Real `.env` is gitignored.
5. **All Pydantic schemas must have explicit types.** No `Any` types in request/response schemas.
6. **Every new function in `src/` needs a corresponding unit test.** Target: ≥80% coverage.
7. **MLflow must log: params, metrics, and model artifact for every training run.** Never run training without tracking.
8. **FastAPI endpoints must validate inputs strictly.** Bad input = 422, not 500.
9. **Docker containers run as non-root users.** Security requirement, not optional.
10. **README.md must show real metric numbers.** Update it after each meaningful training run.

---

## 11. Current Progress Tracker

Update this section as work completes. Do not delete completed items.

- [x] Project scaffold (folders, pyproject.toml, Makefile, .gitignore)
- [x] configs/ YAML files written
- [x] .env.example written
- [x] src/data/ingest.py — CMAPSS download + streaming simulation
- [x] src/features/engineer.py — rolling features + normalization + windowing
- [x] src/models/lstm_autoencoder.py — PyTorch model definition
- [x] src/models/isolation_forest.py — sklearn baseline
- [x] src/models/patchtst.py — PatchTST implementation
- [x] src/models/train.py — unified trainer, MLflow logging
- [x] src/models/evaluate.py — threshold curve, F1/P/R metrics
- [x] src/serving/schemas.py — Pydantic request/response models
- [x] src/serving/predictor.py — model loading from MLflow registry
- [x] src/serving/router.py — /detect, /detect/batch, /health
- [x] src/serving/main.py — FastAPI app factory + middleware
- [x] PostgreSQL schema migration
- [x] dashboard/app.py — Streamlit live chart + anomaly markers
- [x] docker/Dockerfile — multi-stage, non-root
- [x] docker/Dockerfile.streamlit
- [x] docker/docker-compose.yml — all 4 services + health checks
- [x] tests/unit/ — feature, schema, predictor tests
- [x] tests/integration/ — API endpoint tests
- [x] .github/workflows/ci.yml — lint + test + build
- [x] README.md — final with real metrics
- [x] ARCHITECTURE.md — Mermaid diagrams

---

## 12. Dataset Notes (NASA CMAPSS FD001)

- **Download:** https://data.nasa.gov/dataset/CMAPSS-Jet-Engine-Simulated-Data (or Kaggle mirror)
- **File used:** `train_FD001.txt`, `test_FD001.txt`, `RUL_FD001.txt`
- **Columns:** unit_id, cycle, op_setting_1-3, sensor_1-21 (26 total)
- **Anomaly label strategy:** cycles near end-of-life (RUL < threshold) = anomaly. This converts regression labels to binary anomaly classification.
- **Simulated streaming:** Add artificial timestamps to each row (1 row = 1 second) to simulate real-time sensor ingestion.
- **Sensors to drop:** Sensors with near-zero variance across the dataset (identified in EDA notebook): typically sensors 1, 5, 6, 10, 16, 18, 19.

---

## 13. Learning Checkpoints (For Your Reference — Not Seen by Claude)

After each component, before moving on, make sure you can answer:

- **After Docker:** How do services find each other by hostname? What does a health check do? What's a multi-stage build for?
- **After MLflow:** What is the difference between a run, an experiment, and the model registry? What does "Production" stage mean?
- **After FastAPI:** Why use Pydantic? What happens when validation fails? What is an async endpoint?
- **After LSTM-AE:** What is reconstruction error? Why does a higher error signal an anomaly? What is the bottleneck doing?
- **After threshold tuning:** Why not just pick threshold = 0.5? What is the precision-recall tradeoff here?
