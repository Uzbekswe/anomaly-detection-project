.PHONY: install download-data train evaluate serve dashboard test lint format \
       docker-build docker-up docker-down docker-logs migrate clean

# ──────────────────────────────────────────────
# Development
# ──────────────────────────────────────────────

install:
	pip install -e ".[dev]"

download-data:
	python src/data/ingest.py --download

# ──────────────────────────────────────────────
# Training & Evaluation
# ──────────────────────────────────────────────

train:
	python src/models/train.py --config configs/training_config.yaml

evaluate:
	python src/models/evaluate.py --experiment anomaly_detection_cmapss

# ──────────────────────────────────────────────
# Serving
# ──────────────────────────────────────────────

serve:
	uvicorn src.serving.main:app --reload --port 8000

dashboard:
	streamlit run dashboard/app.py

# ──────────────────────────────────────────────
# Quality
# ──────────────────────────────────────────────

test:
	pytest tests/ --cov=src --cov-report=term-missing

lint:
	ruff check src/ tests/

format:
	ruff format src/ tests/

# ──────────────────────────────────────────────
# Docker
# ──────────────────────────────────────────────

docker-build:
	docker compose -f docker/docker-compose.yml build

docker-up:
	docker compose -f docker/docker-compose.yml up -d

docker-down:
	docker compose -f docker/docker-compose.yml down

docker-logs:
	docker compose -f docker/docker-compose.yml logs -f fastapi

# ──────────────────────────────────────────────
# Database
# ──────────────────────────────────────────────

migrate:
	python migrations/run.py

# ──────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov dist *.egg-info
