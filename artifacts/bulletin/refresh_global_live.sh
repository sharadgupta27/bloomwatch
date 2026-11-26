#!/usr/bin/env bash
# Build the operational global live inputs for a bulletin: a copy of the 46-lake archive
# cubes refreshed with the last 14 days from the EUMETSAT Data Store, and an Open-Meteo
# met store running up to its ~5-day lag. The training roots (cubes_global, met_global)
# are never written.
#   bash artifacts/bulletin/refresh_global_live.sh 2026-09-27 [shards]
# Lakes are dealt round-robin to `shards` concurrent processes (default 3): disjoint lake
# sets, so no two processes write one cube. ~4 ingest jobs is this machine's RAM ceiling.
set -u
cd "$(dirname "$0")/../.."
ANCHOR=${1:-$(date -u +%F)}
SHARDS=${2:-3}
export MLFLOW_DISABLE_AGENT_HINT=1 OPENBLAS_NUM_THREADS=1
PY=.venv/Scripts/python.exe
AOI=data/aoi/lakes_global.geojson
LIVE=data/processed/cubes_global_live
OUT=artifacts/bulletin/$ANCHOR
mkdir -p "$OUT/refresh"

if [ ! -d "$LIVE" ]; then
  echo "$(date) copying cubes_global -> $LIVE"
  cp -r data/processed/cubes_global "$LIVE"
fi

echo "$(date) met_live (Open-Meteo, all lakes)"
$PY pipelines/ingest_era5.py --source open-meteo --aoi "$AOI" \
  --start 2016 --end 2026 --out data/processed/met_live --workers 1
echo "$(date) met_live exit=$?"

echo "$(date) EUMETSAT refresh, anchor $ANCHOR, $SHARDS shards"
for k in $(seq 0 $((SHARDS - 1))); do
  LAKES=$($PY -c "
import json
ids = [f['properties']['lake_id'] for f in json.load(open('$AOI'))['features']]
print(' '.join(ids[$k::$SHARDS]))")
  echo "$(date) shard $k: $LAKES"
  # --no-drift: this pass only refreshes cubes; the bulletin makes the forecasts.
  $PY pipelines/advisory.py --anchor "$ANCHOR" --aoi "$AOI" --cubes "$LIVE" \
    --met-root data/processed/met_live --refresh-days 14 --refresh-workers 4 --no-drift \
    --lakes $LAKES --out "$OUT/refresh/shard_$k" > "$OUT/refresh/shard_$k.out" 2>&1 &
done
wait
echo "$(date) refresh done"
grep -h "refresh failed" "$OUT"/refresh/shard_*.out || echo "no refresh failures"
