#!/usr/bin/env bash
# GPU queue for the ensemble comparison (2026-09-23). Sequential: one 4 GB card.
set -u
cd "$(dirname "$0")/.."
export MLFLOW_DISABLE_AGENT_HINT=1
PY=.venv/Scripts/python.exe
MET='+data.met=[wind_speed,wind_u,wind_v,t2m,ssrd,tp]'
run() {  # name, overrides...
  local name=$1; shift
  echo "$(date -Is) start $name"
  $PY pipelines/train.py "$@" run_name=$name hydra.run.dir=artifacts/$name > artifacts/$name.out 2>&1
  echo "$(date -Is) done $name exit=$?"
}
run exceed_met_s1337 model=anomaly_exceed "$MET" seed=1337
run exceed_met_s1    model=anomaly_exceed "$MET" seed=1
run exceed_met_s2    model=anomaly_exceed "$MET" seed=2
run clim_s1          model=anomaly_clim seed=1
run clim_s2          model=anomaly_clim seed=2
echo "$(date -Is) queue finished"
