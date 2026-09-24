#!/usr/bin/env bash
# Expanded-registry queue (2026-09-23): 15 training lakes, same val/test lakes plus new ones.
set -u
cd "$(dirname "$0")/.."
export MLFLOW_DISABLE_AGENT_HINT=1
PY=.venv/Scripts/python.exe
MET='+data.met=[wind_speed,wind_u,wind_v,t2m,ssrd,tp]'
REG="paths.aoi=\${paths.repo_root}/data/aoi/lakes_expanded.geojson"
CUBES="paths.cubes=\${paths.repo_root}/data/processed/cubes_expanded"
for s in 1337 1 2; do
  name=expanded_met_s$s
  echo "$(date -Is) start $name"
  $PY pipelines/train.py model=anomaly_exceed "$MET" "$REG" "$CUBES" seed=$s run_name=$name hydra.run.dir=artifacts/$name > artifacts/$name.out 2>&1
  echo "$(date -Is) done $name exit=$?"
done
echo "$(date -Is) queue finished"
