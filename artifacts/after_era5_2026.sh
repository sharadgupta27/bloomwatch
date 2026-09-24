#!/usr/bin/env bash
# Re-score the weather ensemble on 2026 once ERA5 2026 is in the met stores.
cd "$(dirname "$0")/.."
until grep -q "built .* met stores" artifacts/ingest_era5_2026.out; do sleep 300; done
export MLFLOW_DISABLE_AGENT_HINT=1
.venv/Scripts/python.exe pipelines/evaluate.py --checkpoint models/ens_met --cubes data/processed/cubes_live --split-mode temporal --val-years 2025 --test-years 2026 --splits test --tta --out artifacts/eval_nrt2026_ens_met_weather > artifacts/eval_nrt2026_ens_met_weather.out 2>&1
echo "$(date -Is) done exit=$?"
