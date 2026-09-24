#!/usr/bin/env bash
cd "$(dirname "$0")/.."
export MLFLOW_DISABLE_AGENT_HINT=1
PY=.venv/Scripts/python.exe
ev() { echo "$(date -Is) eval $2"; $PY pipelines/evaluate.py --checkpoint "$1" --out artifacts/$2 --tta > artifacts/$2.out 2>&1; echo "$(date -Is) done $2 exit=$?"; }
ev models/ens_clim eval_ens_clim
ev models/ens_met eval_ens_met
ev models/ens_clim/clim_s1.ckpt eval_clim_s1
ev models/ens_clim/clim_s2.ckpt eval_clim_s2
ev models/ens_met/met_s1.ckpt eval_exceed_met_s1
ev models/ens_met/met_s2.ckpt eval_exceed_met_s2
echo "$(date -Is) evals finished"
