#!/usr/bin/env bash
# Global multilevel pipeline (2026-09-23). Waits for the four ingest jobs, then:
#   merge cubes -> thermal classes -> 3 GPU seeds of anomaly_multilevel -> raw eval ->
#   per-trophic-class calibration -> calibrated eval -> incumbents on the same data -> gate.
# Nothing here touches models/best.ckpt or models/ens_met; promotion is left to a person.
set -u
cd "$(dirname "$0")/.."
export MLFLOW_DISABLE_AGENT_HINT=1 OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4
PY=.venv/Scripts/python.exe
AOI=data/aoi/lakes_global.geojson
CUBES=data/processed/cubes_global
MET=data/processed/met_global
log() { echo "$(date -Is) $*"; }

# `run_global.sh train` resumes at step 3 over the already-merged cubes. Re-merging would
# rewrite every cube, change its digest and force a fresh 8.5 GB decode into data/cache/mmap.
if [ "${1:-}" != "train" ]; then

for job in ingest_global_A ingest_global_B ingest_global_s3a ingest_global_s3a_old; do
  until grep -q "done $job " artifacts/ingest_global.log; do sleep 300; done
done
# A killed duplicate of job C wrote a false "done ingest_global_s3a" line (2026-09-23 22:04),
# so the log alone is not proof: also wait until no ingest job is alive and C and D have
# written their end-of-run reports (a killed run never writes one).
while ps -ef | grep -v grep | grep -q "ingest_global.sh"; do sleep 300; done
for job in ingest_global_A ingest_global_B ingest_global_s3a ingest_global_s3a_old; do
  [ -s artifacts/$job.json ] || { log "$job ended without a report -- not merging"; exit 1; }
done
until grep -q "built .* met stores" artifacts/ingest_met_global.out; do sleep 60; done
log "ingests finished"

# 1. merge: existing cubes first, so no day already trained on changes value
$PY scripts/merge_cubes.py --aoi $AOI --out $CUBES \
  --parts data/processed/cubes_expanded data/processed/cubes_global_backfill \
          data/processed/cubes_global_new data/processed/cubes_global_s3a \
  > artifacts/merge_global.out 2>&1 || { log "merge failed"; exit 1; }
log "merged"

# 2. thermal class from each lake's own weather record
$PY scripts/annotate_groups.py --aoi $AOI --met $MET > artifacts/annotate_groups.out 2>&1
$PY scripts/compare_met_sources.py > artifacts/compare_met_sources.out 2>&1
log "annotated"
fi

# 3. three seeds on the GPU. Six loader workers: with num_workers=0 the GPU sat at ~1% and an
#    epoch took an hour (2026-09-25); 6 workers read ~4x faster and still leave ~1 GB RAM free
#    (8 left 0.3 GB). Val gets 2: six more idle val workers took commit to 55.8/58.9 GB.
#    The batch size stays 8 so these runs remain comparable with ens_met.
MET_KEYS='+data.met=[wind_speed,wind_u,wind_v,t2m,ssrd,tp]'
OVR=("paths.aoi=\${paths.repo_root}/$AOI" "paths.cubes=\${paths.repo_root}/$CUBES"
     "paths.met=\${paths.repo_root}/$MET" "+paths.mmap=\${paths.repo_root}/data/cache/mmap"
     data.num_workers=6 data.val_num_workers=2)
for s in 1337 1 2; do
  name=global_ml_s$s
  # A seed that already finished is kept, not retrained: the log must say it exited 0 and
  # its best checkpoint must exist (a stopped run leaves neither).
  if grep -q "done $name exit=0" artifacts/run_global.log && ls artifacts/$name/checkpoints/best-*.ckpt >/dev/null 2>&1; then
    log "skip $name (already finished)"
    continue
  fi
  log "start $name"
  $PY pipelines/train.py model=anomaly_multilevel "$MET_KEYS" "${OVR[@]}" seed=$s \
      run_name=$name hydra.run.dir=artifacts/$name > artifacts/$name.out 2>&1
  log "done $name exit=$?"
done
mkdir -p models/ens_global_ml
for s in 1337 1 2; do
  cp artifacts/global_ml_s$s/checkpoints/best-*.ckpt models/ens_global_ml/ml_s$s.ckpt
done

# 4-6. raw eval, per-trophic-class calibration fitted on val, calibrated eval
EV=(--aoi $AOI --cubes $CUBES --met $MET --tta)
$PY pipelines/evaluate.py --checkpoint models/ens_global_ml "${EV[@]}" --no-calibration \
    --out artifacts/eval_global_ml_raw > artifacts/eval_global_ml_raw.out 2>&1
$PY scripts/calibrate_exceedance.py --scores artifacts/eval_global_ml_raw/scores.json \
    --checkpoint models/ens_global_ml --by trophic --aoi $AOI > artifacts/calibrate_global_ml.out 2>&1
$PY pipelines/evaluate.py --checkpoint models/ens_global_ml "${EV[@]}" \
    --out artifacts/eval_global_ml > artifacts/eval_global_ml.out 2>&1
log "candidate evaluated"

# 7. incumbents on exactly the same lakes, days and forcing
$PY pipelines/evaluate.py --checkpoint models/ens_met "${EV[@]}" \
    --out artifacts/eval_global_incumbent_ens_met > artifacts/eval_global_incumbent_ens_met.out 2>&1
$PY pipelines/evaluate.py --checkpoint models/ens_expanded_met "${EV[@]}" \
    --out artifacts/eval_global_incumbent_ens_expanded > artifacts/eval_global_incumbent_ens_expanded.out 2>&1
log "incumbents evaluated"

# 8. gate: with the small-lake rule (the default since 2026-09-24, the one that decides) and
#    strict (every held-out lake counts, for reference)
for inc in ens_met ens_expanded; do
  $PY scripts/promotion_gate.py --candidate artifacts/eval_global_ml/scores.json \
      --incumbent artifacts/eval_global_incumbent_$inc/scores.json --min-samples 0 \
      --report artifacts/gate_global_vs_${inc}_strict.json > artifacts/gate_global_vs_${inc}_strict.out 2>&1
  $PY scripts/promotion_gate.py --candidate artifacts/eval_global_ml/scores.json \
      --incumbent artifacts/eval_global_incumbent_$inc/scores.json --min-samples 100 \
      --report artifacts/gate_global_vs_${inc}_min100.json > artifacts/gate_global_vs_${inc}_min100.out 2>&1
done
log "pipeline finished"
