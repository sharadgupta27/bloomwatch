"""Bulk ingest driver: build the per-lake Zarr datacubes from Sentinel-3 OLCI.

Deliberately argparse rather than Hydra. Data ingestion has a handful of operational
knobs and no hyperparameter sweep, so a plain CLI keeps it runnable from cron, CI, or a
container entrypoint without a config tree. Hydra is used where it earns its keep --
`pipelines/train.py`, where sweeping is the whole point.

The run is idempotent: days already present in a cube are skipped, so an interrupted
multi-hour ingest resumes by re-running the identical command.

Examples
--------
    python pipelines/ingest.py --start 2017-01-01 --end 2024-12-31
    python pipelines/ingest.py --lakes erie_west taihu --workers 12
    python pipelines/ingest.py --all-year --min-valid 0.05
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from habforecast.aoi import AOIRegistry  # noqa: E402
from habforecast.data.aux_layers import AUX_LAYERS, parse_aux  # noqa: E402
from habforecast.data.ingest import ingest_lake  # noqa: E402
from habforecast.data.olci import make_reader  # noqa: E402

DEFAULT_AOI = REPO_ROOT / "data" / "aoi" / "lakes.geojson"
DEFAULT_CUBE_ROOT = REPO_ROOT / "data" / "processed" / "cubes"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--aoi", type=Path, default=DEFAULT_AOI, help="lake registry GeoJSON")
    p.add_argument("--cube-root", type=Path, default=DEFAULT_CUBE_ROOT, help="output Zarr root")
    p.add_argument("--lakes", nargs="*", default=None, help="lake_ids to ingest (default: all)")
    p.add_argument("--start", default="2017-01-01", help="ISO start date")
    p.add_argument("--end", default="2024-12-31", help="ISO end date")
    p.add_argument("--workers", type=int, default=10, help="concurrent scene reads")
    p.add_argument(
        "--min-valid",
        type=float,
        default=0.02,
        help="reject scenes with less than this fraction of QC-passing grid coverage",
    )
    p.add_argument(
        "--all-year",
        action="store_true",
        help="ingest every month rather than only each lake's configured bloom season",
    )
    p.add_argument(
        "--aux",
        default="",
        help=(
            "comma-separated auxiliary layers read from the same granules: "
            + ", ".join(f"{k} ({v.why})" for k, v in AUX_LAYERS.items())
        ),
    )
    p.add_argument(
        "--source",
        choices=["planetary-computer", "eumetsat", "eumetsat-reprocessed"],
        default="planetary-computer",
        help="planetary-computer: the training archive (lags ~6-7 months). eumetsat: the "
        "EUMETSAT Data Store, hours behind real time, same product and QC; needs "
        "~/.eumdac/credentials (see src/habforecast/data/eumetsat.py). "
        "eumetsat-reprocessed: the Data Store's BC003 reprocessing, the only full-resolution "
        "source for 2016 to October 2017",
    )
    p.add_argument(
        "--skip-existing-in",
        type=Path,
        default=None,
        help="also skip days already held by <DIR>/<lake_id>.zarr -- for a backfill written "
        "to its own root and merged afterwards with scripts/merge_cubes.py",
    )
    p.add_argument("--report", type=Path, default=None, help="write a JSON ingest report here")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )
    # These libraries are chatty at INFO and drown the ingest log.
    for noisy in ("botocore", "urllib3", "fsspec", "aiohttp", "asyncio", "numexpr"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    registry = AOIRegistry.from_geojson(args.aoi)
    lakes = args.lakes or registry.lake_ids
    unknown = set(lakes) - set(registry.lake_ids)
    if unknown:
        raise SystemExit(f"unknown lake_id(s): {sorted(unknown)}; known: {registry.lake_ids}")

    args.cube_root.mkdir(parents=True, exist_ok=True)
    reports = {}
    t_all = time.time()

    for lake_id in lakes:
        aoi = registry[lake_id]
        logging.info("=" * 72)
        logging.info(
            "%s -- %s (split=%s, season=%s)", lake_id, aoi.name, aoi.split, aoi.bloom_season
        )
        t0 = time.time()
        skip = None
        if args.skip_existing_in is not None:
            from habforecast.data.store import LakeCube

            skip = LakeCube(args.skip_existing_in / f"{lake_id}.zarr", aoi.grid()).existing_days()
            logging.info("%s: skipping %d days held in %s", lake_id, len(skip), args.skip_existing_in)
        rep = ingest_lake(
            aoi,
            args.start,
            args.end,
            str(args.cube_root),
            aoi_path=str(args.aoi),
            min_valid_fraction=args.min_valid,
            aux=parse_aux(args.aux),
            bloom_season_only=not args.all_year,
            max_workers=args.workers,
            reader=make_reader(args.source),
            skip_days=skip,
        )
        elapsed = time.time() - t0
        logging.info("%s finished in %.1f min", lake_id, elapsed / 60)
        reports[lake_id] = {
            "scenes_found": rep.scenes_found,
            "scenes_read": rep.scenes_read,
            "scenes_rejected": rep.scenes_rejected,
            "scenes_failed": rep.scenes_failed,
            "days_written": rep.days_written,
            "read_success_rate": round(rep.read_success_rate, 4),
            "elapsed_s": round(elapsed, 1),
            "failures": rep.failures[:50],
        }

    total = time.time() - t_all
    logging.info("=" * 72)
    logging.info("ALL DONE in %.1f min", total / 60)
    for lake_id, r in reports.items():
        logging.info(
            "  %-14s %4d days  |  %4d/%4d scenes read  |  %3d failed",
            lake_id,
            r["days_written"],
            r["scenes_read"],
            r["scenes_found"],
            r["scenes_failed"],
        )

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(
                {
                    "args": vars(args)
                    | {
                        "aoi": str(args.aoi),
                        "cube_root": str(args.cube_root),
                        "report": str(args.report),
                    },
                    "total_elapsed_s": round(total, 1),
                    "lakes": reports,
                },
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )
        logging.info("report -> %s", args.report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
