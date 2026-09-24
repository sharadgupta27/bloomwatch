#!/usr/bin/env bash
# Global-registry ingest (2026-09-23). All-year, 2016-05 .. 2025-12 (2026 stays the NRT
# holdout). Each job writes append-only into its own root; scripts/merge_cubes.py builds
# data/processed/cubes_global from them once all three finish.
#   A  Planetary Computer 2017-11..2025-12, the 21 existing lakes, skipping days already
#      in cubes_expanded (off-season months and 2017-11..2018-03)  -> cubes_global_backfill
#   B  Planetary Computer 2017-11..2025-12, the 25 new lakes        -> cubes_global_new
#   C  EUMETSAT BC003 reprocessing 2016-05..2017-10, all 46 lakes   -> cubes_global_s3a
# (Planetary Computer's mirror starts 2017-11; EUMETSAT is the only full-res source before.)
set -u
cd "$(dirname "$0")/.."
# One BLAS thread per worker: with the default (one per core) every one of ~26 worker
# processes reserved per-thread buffers and the first attempt died of commit exhaustion.
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=.venv/Scripts/python.exe
AOI=data/aoi/lakes_global.geojson
OLD="erie_west okeechobee taihu balaton victoria_ne winnipeg st_clair saginaw green_bay utah_lake pontchartrain ijsselmeer curonian neagh chaohu poyang kasumigaura victoria_mwanza hartbeespoort alexandrina chapala"
NEW="dianchi geneva hongze erhai biwa laguna_de_bay kinneret peipsi vortsjarv malaren szczecin_lagoon ladoga_south garda lake_george_ug tana upper_klamath winnebago lake_of_the_woods tahoe valencia_ve maracaibo titicaca_puno guaiba taupo te_waihora"
run() { # name, workers, source, start, end, root, extra, lakes...
  local name=$1 w=$2 src=$3 s=$4 e=$5 root=$6 extra=$7; shift 7
  echo "$(date -Is) start $name"
  $PY pipelines/ingest.py --aoi $AOI --cube-root $root --source $src --all-year \
      --start $s --end $e --workers $w $extra --lakes "$@" --report artifacts/$name.json \
      > artifacts/$name.out 2>&1
  echo "$(date -Is) done $name exit=$?"
}
case "${1:-}" in
  A) run ingest_global_A 7 planetary-computer 2017-11-01 2025-12-31 data/processed/cubes_global_backfill "--skip-existing-in data/processed/cubes_expanded" $OLD ;;
  B) run ingest_global_B 7 planetary-computer 2017-11-01 2025-12-31 data/processed/cubes_global_new "" $NEW ;;
  C) run ingest_global_s3a 6 eumetsat-reprocessed 2016-05-01 2017-10-31 data/processed/cubes_global_s3a "" $NEW ;;
  D) run ingest_global_s3a_old 5 eumetsat-reprocessed 2016-05-01 2017-10-31 data/processed/cubes_global_s3a "" $OLD ;;
  *) echo "usage: $0 A|B|C|D"; exit 2 ;;
esac
