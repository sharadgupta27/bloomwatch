"""Build training cubes from NOAA CoastWatch ERDDAP, at any resolution in the ladder.

Why a second ingest path
------------------------
`pipelines/ingest.py` reads Sentinel-3 OLCI L2 WFR granules from Planetary Computer:
native 300 m, full WQSF quality control, and the path the current model was trained
through. It is also slow -- each granule is a few hundred megabytes of swath that has to
be downloaded and resampled to keep a lake-sized crop of it -- and its mirror stops at
2026-02.

That cost is why the training set is three lakes. And three lakes is the measured
bottleneck: the evaluation showed capacity *hurting* on this dataset, with the
generalisation gap dominated by how few water bodies the model had ever seen.

ERDDAP subsets server-side. Asking for one lake for one year is a bounding-box request
returning a few megabytes, so the marginal cost of the twentieth lake is minutes rather
than hours. That makes a genuinely larger training set reachable, which is the change
most likely to move accuracy.

What is given up
----------------
* **A different sensor, at 750 m and 4 km.** VIIRS is not OLCI. Its retrieval differs,
  its band set differs, and it was designed for the open ocean. A model trained on it is
  not a model trained on OLCI, and cross-source evaluation numbers are not comparable.
* **No WQSF flags.** These are finished Level-3 composites; the project's own QC
  (`data.qc`) and the flag-frequency drift telemetry do not apply. A pixel is present or
  it is NaN, and `n_obs` is therefore 1 or 0 rather than a scene count.
* **No water mask from the data.** The archive path knows a pixel is water because
  INLAND_WATER is set even under cloud. Here water can only be inferred from where
  retrievals succeeded, so the mask is built from the whole ingested record rather than
  from a fortnight -- which, over years, is a good deal of evidence.

Usage
-----
    python pipelines/ingest_erddap.py --product viirs-750m --start 2016-01-01 \\
        --end 2025-08-01 --lakes erie_west taihu okeechobee
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from habforecast.aoi import AOIRegistry
from habforecast.data.coastwatch import PRODUCTS, CoastWatchReader
from habforecast.data.store import DailyFrame, LakeCube

logger = logging.getLogger("ingest_erddap")

DEFAULT_AOI = Path("data/aoi/lakes.geojson")
# Longest window asked for in one request. Also the cap on the size-derived chunk below.
CHUNK_DAYS = 180
# ERDDAP refuses an over-large subset outright, with 413, and answers 500 when it is
# loaded. Both arrive per *request*, so the window has to shrink with the grid: measured,
# 180 days of Lake Winnipeg at 750 m (165x165) is refused every time while the same
# request for a 69x69 lake succeeds. This budget keeps a request near a few million cells.
CELL_BUDGET = 1_500_000
MIN_CHUNK_DAYS = 12
# How many times to halve a window that still fails before giving up on it.
MAX_SPLITS = 3
# Politeness gap between requests to a free public endpoint, enforced across *all*
# workers rather than per worker: the point is to bound the rate this client asks NOAA to
# serve, and per-thread pauses would scale that rate with the worker count.
REQUEST_PAUSE_S = 1.0
_rate_lock = threading.Lock()
_last_request = [0.0]


def _throttle() -> None:
    with _rate_lock:
        wait = REQUEST_PAUSE_S - (time.monotonic() - _last_request[0])
        if wait > 0:
            time.sleep(wait)
        _last_request[0] = time.monotonic()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--product", default="viirs-750m", choices=sorted(PRODUCTS))
    p.add_argument("--aoi", type=Path, default=DEFAULT_AOI)
    p.add_argument(
        "--cube-root", type=Path, default=None, help="default: data/processed/cubes_<product>"
    )
    p.add_argument("--lakes", nargs="*", default=None)
    p.add_argument("--start", default="2016-01-01")
    p.add_argument("--end", default=None, help="default: the newest day the product holds")
    p.add_argument(
        "--radius-km",
        type=float,
        default=None,
        help="override the registry AOI with a box of this half-width",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=4,
        help="lakes fetched concurrently; the request rate is capped globally",
    )
    p.add_argument(
        "--all-year",
        action="store_true",
        help="fetch every month, not just the lake's bloom season",
    )
    p.add_argument("--report", type=Path, default=None)
    p.add_argument("--list-products", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def season_windows(aoi, start: date, end: date, bloom_only: bool) -> list[tuple[date, date]]:
    """The date ranges worth fetching for this lake.

    Bloom season only, by default, for two reasons. It matches what the existing 300 m
    cubes contain -- `pipelines/ingest.py` filters the same way -- so cubes built by the
    two paths mean the same thing. And off-season frames over a mid-latitude lake are ice,
    low sun and near-zero chlorophyll: they cost the same to fetch as a July bloom and
    teach the model almost nothing about one.

    A window that wraps the new year (southern-hemisphere lakes: November to April) is
    handled by walking each calendar year and clipping, rather than by arithmetic on month
    numbers that silently produces an empty range.
    """
    if not bloom_only:
        return [(start, end)]
    first, last = aoi.bloom_season
    out: list[tuple[date, date]] = []
    for year in range(start.year, end.year + 1):
        if first <= last:
            spans = [(date(year, first, 1), _month_end(year, last))]
        else:  # wraps: e.g. November -> April
            spans = [
                (date(year, 1, 1), _month_end(year, last)),
                (date(year, first, 1), date(year, 12, 31)),
            ]
        for lo, hi in spans:
            lo, hi = max(lo, start), min(hi, end)
            if lo <= hi:
                out.append((lo, hi))
    return sorted(out)


def fetch_window(reader, grid, start: date, end: date, depth: int = 0):
    """Fetch one window, halving it and retrying when the service refuses the size.

    A 413 means the subset is too large and a 500 usually means the service is busy; both
    are answered the same way, because a smaller request is the fix for either. Returning
    an empty list on exhaustion rather than raising keeps one stubborn fortnight from
    losing the other nine years of a lake.
    """
    _throttle()
    try:
        # No budget: an ingest window is orders of magnitude larger than a live lake
        # request, and legitimately takes minutes. The live cap exists to fail a user's
        # click fast, which is the opposite of what a batch job wants.
        return reader.frames(grid, start, end, budget_s=None)
    except Exception as exc:
        span = (end - start).days + 1
        if depth >= MAX_SPLITS or span <= MIN_CHUNK_DAYS:
            logger.warning("  %s..%s giving up after %d splits: %s", start, end, depth, exc)
            return []
        mid = start + timedelta(days=span // 2)
        logger.info("  %s..%s failed (%s); splitting", start, end, exc)
        left = fetch_window(reader, grid, start, mid - timedelta(days=1), depth + 1)
        right = fetch_window(reader, grid, mid, end, depth + 1)
        return left + right


def _month_end(year: int, month: int) -> date:
    return date(year + (month == 12), (month % 12) + 1, 1) - timedelta(days=1)


def ingest_lake(
    aoi,
    reader: CoastWatchReader,
    cube_root: Path,
    start: date,
    end: date,
    bloom_only: bool = True,
) -> dict:
    """Fetch one lake over a date range and append it to that lake's cube."""
    grid = aoi.grid(resolution=reader.spec.resolution_m)
    bbox = grid.bounds_lonlat
    lo = reader.earliest_available(bbox)
    hi = reader.available_through(bbox)
    if hi is None:
        return {"lake_id": aoi.lake_id, "status": "no coverage", "days": 0}
    start = max(start, lo or start)
    end = min(end, hi)
    if start > end:
        return {"lake_id": aoi.lake_id, "status": "record does not reach that range", "days": 0}

    # Chunk length falls out of the grid size, not a constant: the same 180-day request
    # that works for a small lake is refused for a large one.
    height, width = grid.shape
    chunk_days = int(min(CHUNK_DAYS, max(MIN_CHUNK_DAYS, CELL_BUDGET // max(height * width, 1))))

    cube = LakeCube(cube_root / f"{aoi.lake_id}.zarr", grid)
    have = cube.existing_days()  # empty when the cube does not exist yet
    written = 0
    requested = 0
    t0 = time.monotonic()

    for win_start, win_end in season_windows(aoi, start, end, bloom_only):
        cursor = win_start
        while cursor <= win_end:
            stop = min(cursor + timedelta(days=chunk_days - 1), win_end)
            # Skip a chunk already fully ingested: the store is idempotent, but the network
            # round trip is the expensive part and there is no reason to pay it twice.
            span_days = {cursor + timedelta(days=i) for i in range((stop - cursor).days + 1)}
            if span_days <= have:
                logger.info("  %s..%s already ingested", cursor, stop)
                cursor = stop + timedelta(days=1)
                continue
            frames = fetch_window(reader, grid, cursor, stop)
            requested += 1

            daily = [
                DailyFrame(
                    day=f.day,
                    log_chl=f.log_chl,
                    # These are finished composites: one value per pixel per day, with no
                    # scene count to report. 1 means "present", which is what `valid` reads.
                    n_obs=f.valid.astype(np.uint8),
                    scene_ids=(reader.spec.key,),
                    mean_valid_fraction=float(f.valid.mean()),
                    flag_stats={},
                )
                for f in frames
                if f.day not in have and f.valid.any()
            ]
            if daily:
                written += cube.write(daily)
                have.update(f.day for f in daily)
            logger.info("  %s..%s  %d frames, %d written", cursor, stop, len(frames), len(daily))
            cursor = stop + timedelta(days=1)

    return {
        "lake_id": aoi.lake_id,
        "status": "ok",
        "grid": list(grid.shape),
        "resolution_m": reader.spec.resolution_m,
        "days": written,
        "record": [start.isoformat(), end.isoformat()],
        "requests": requested,
        "chunk_days": chunk_days,
        "seconds": round(time.monotonic() - t0, 1),
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    if args.list_products:
        for key, spec in PRODUCTS.items():
            print(f"{key:14} {spec.resolution_m:6.0f} m  {spec.record:24} {spec.label}")
            print(f"{'':14} {spec.note}\n")
        return 0

    reader = CoastWatchReader(product=args.product)
    # Resolve the sector index once, here, before any worker starts. Letting each thread
    # discover it independently means six simultaneous identical requests to the same
    # endpoint and six writers to the same cache file; measured, three of twenty-one lakes
    # came back with zero days from that race and no error to show for it.
    _ = reader.sectors
    cube_root = args.cube_root or Path(f"data/processed/cubes_{args.product}")
    cube_root.mkdir(parents=True, exist_ok=True)

    registry = AOIRegistry.from_geojson(args.aoi)
    wanted = args.lakes or registry.lake_ids
    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end) if args.end else datetime.now().date()

    print(f"product {args.product} ({reader.spec.resolution_m:.0f} m) -> {cube_root}")
    print(f"{len(wanted)} lakes, {args.workers} workers, {start} to {end}", flush=True)
    results = []
    # Each worker gets its own reader: the sector index is cached per instance and the
    # underlying HTTP calls are independent, so nothing is shared but the rate limiter.
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {
            pool.submit(
                ingest_lake,
                registry[lake_id],
                # One reader shared by every worker: after the index is loaded it holds no
                # mutable state, and `frames` only reads it.
                reader,
                cube_root,
                start,
                end,
                not args.all_year,
            ): lake_id
            for lake_id in wanted
        }
        for fut in as_completed(futures):
            lake_id = futures[fut]
            try:
                res = fut.result()
            except Exception as exc:
                logger.exception("%s failed", lake_id)
                res = {"lake_id": lake_id, "status": f"failed: {exc}", "days": 0}
            results.append(res)
            parts = [res["status"], f"{res['days']} days"]
            if "grid" in res:
                parts.append(f"grid {res['grid'][0]}x{res['grid'][1]}")
            if res.get("seconds"):
                parts.append(f"{res['seconds']:.0f}s")
            done = len(results)
            print(f"  [{done}/{len(wanted)}] {lake_id}: " + ", ".join(parts), flush=True)

    results.sort(key=lambda r: r["lake_id"])

    report = {
        "product": args.product,
        "resolution_m": reader.spec.resolution_m,
        "cube_root": str(cube_root),
        "generated": datetime.now().isoformat(timespec="seconds"),
        "lakes": results,
        "total_days": sum(r["days"] for r in results),
    }
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n{report['total_days']} lake-days written to {cube_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
