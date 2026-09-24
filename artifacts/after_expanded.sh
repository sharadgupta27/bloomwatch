#!/usr/bin/env bash
cd "$(dirname "$0")/.."
until grep -q "queue finished" artifacts/run_queue_expanded.log; do sleep 120; done
export MLFLOW_DISABLE_AGENT_HINT=1
mkdir -p models/ens_expanded_met
for s in 1337 1 2; do cp artifacts/expanded_met_s$s/checkpoints/best-*.ckpt models/ens_expanded_met/met_s$s.ckpt; done
.venv/Scripts/python.exe pipelines/evaluate.py --checkpoint models/ens_expanded_met --aoi data/aoi/lakes_expanded.geojson --cubes data/processed/cubes_expanded --out artifacts/eval_ens_expanded_met_raw --tta --no-calibration > artifacts/eval_ens_expanded_met_raw.out 2>&1
echo "$(date -Is) eval done exit=$?"
