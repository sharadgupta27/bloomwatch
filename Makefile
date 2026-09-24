# Container-first entrypoints. Every target runs in Docker so the host needs nothing
# but Docker itself. `make local-*` variants exist for the conda fallback documented
# in docs/SETUP.md.
.PHONY: help build ingest train evaluate results advisory api mlflow cases test lint typecheck fmt clean local-test local-ingest local-cases local-advisory

COMPOSE := docker compose
PY      ?= python

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-14s\033[0m %s\n",$$1,$$2}'

build:      ## Build all container images
	$(COMPOSE) build

mlflow:     ## Start the MLflow tracking server on :5000
	$(COMPOSE) up -d mlflow

ingest:     ## Build the Zarr datacubes from Sentinel-3 OLCI
	$(COMPOSE) run --rm ingest

train:      ## Train (GPU; needs the NVIDIA Container Toolkit)
	$(COMPOSE) run --rm train

evaluate:   ## Score models/best.ckpt on both splits (leave-lake-out and temporal), with TTA
	$(COMPOSE) run --rm tools python pipelines/evaluate.py --checkpoint /models/best.ckpt --out artifacts/eval --tta --cubes /app/data/processed/cubes
	$(COMPOSE) run --rm tools python pipelines/evaluate.py --checkpoint /models/best-temporal.ckpt --out artifacts/eval_temporal --split-mode temporal --tta --cubes /app/data/processed/cubes

results:    ## Regenerate the results tables and the model card's quantitative section
	$(COMPOSE) run --rm tools python scripts/make_results_table.py --scores artifacts/eval/scores.json --temporal artifacts/eval_temporal/scores.json --model-card docs/MODEL_CARD.md

advisory:   ## Today's advisory feed (GeoJSON + RSS) into artifacts/advisory -- the scheduled job
	$(COMPOSE) run --rm advisory

api:        ## Serve the app, API and dashboard on :8000  (/app is the interactive map)
	$(COMPOSE) up api

cases:      ## Render the case-study imagery the dashboard shows (CPU, a few minutes)
	$(COMPOSE) run --rm tools python scripts/make_case_studies.py --cases-per-lake 2 --device cpu --cubes /app/data/processed/cubes

test:       ## Run the test suite inside the image
	$(COMPOSE) run --rm tools pytest -m "not network" -q

lint:       ## Ruff check + format check
	$(COMPOSE) run --rm tools sh -c "ruff check src tests pipelines scripts serving && ruff format --check src tests pipelines"

typecheck:  ## mypy over the whole codebase (targets are set in pyproject.toml)
	$(COMPOSE) run --rm tools mypy

fmt:        ## Apply ruff formatting
	ruff format src tests pipelines && ruff check --fix src tests pipelines scripts serving

clean:      ## Remove build/run artefacts (leaves data volumes intact)
	rm -rf outputs multirun lightning_logs .pytest_cache .ruff_cache .mypy_cache htmlcov .coverage

# ---- conda fallback (see docs/SETUP.md) -------------------------------------------
local-test:
	PYTHONPATH=src $(PY) -m pytest -m "not network" -q

local-ingest:
	PYTHONPATH=src $(PY) pipelines/ingest.py --start 2018-01-01 --end 2024-12-31 --workers 12

local-cases:
	PYTHONPATH=src $(PY) scripts/make_case_studies.py --cases-per-lake 2 --device cpu
	PYTHONPATH=src $(PY) scripts/make_dashboard.py

local-advisory:
	PYTHONPATH=src $(PY) pipelines/advisory.py --out artifacts/advisory
