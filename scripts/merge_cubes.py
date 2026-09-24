"""Merge per-lake cubes from several ingest roots into one, once.

    python scripts/merge_cubes.py --aoi data/aoi/lakes_global.geojson \
        --parts data/processed/cubes_expanded data/processed/cubes_global_backfill \
                data/processed/cubes_global_new data/processed/cubes_global_s3a \
        --out data/processed/cubes_global

Why this exists: extending a cube backwards in time, or into months between those it
holds, is an out-of-order write, and `LakeCube.write` handles one by loading the whole
cube and rewriting it. A backfill of every off-season month does that ~50 times per lake,
each holding two copies of the cube in RAM -- which is what ran the machine out of
memory on 2026-09-23. So each source ingests append-only into its own root (skipping days
another root already has, `pipelines/ingest.py --skip-existing-in`), and this writes each
lake once.

Where two parts hold the same day, the *first* part listed wins: list the cubes existing
results were computed from first, so no day already trained on changes value.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import xarray as xr

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from habforecast.aoi import AOIRegistry  # noqa: E402
from habforecast.data.store import LakeCube  # noqa: E402

logger = logging.getLogger(__name__)


def merge_lake(lake_id: str, parts: list[Path], out: Path, registry: AOIRegistry) -> dict:
    grid = registry[lake_id].grid()
    pieces, flags, used = [], {}, []
    for root in parts:
        path = root / f"{lake_id}.zarr"
        cube = LakeCube(path, grid)
        if not cube.exists:
            continue
        with xr.open_zarr(path, consolidated=True) as ds:
            if (ds.sizes["y"], ds.sizes["x"]) != grid.shape:
                raise ValueError(
                    f"{lake_id}: {path} grid {(ds.sizes['y'], ds.sizes['x'])} != registry "
                    f"{grid.shape}; refusing to merge cubes on different grids"
                )
            pieces.append(ds.load())
        used.append(str(root))
        # Earlier parts win for flag stats too, matching the frames kept.
        for day, stats in cube.read_flag_stats().items():
            flags.setdefault(day, stats)
    if not pieces:
        return {"lake_id": lake_id, "days": 0, "parts": []}

    merged = xr.concat(pieces, dim="time", join="outer")
    # np.unique keeps the first occurrence in concatenation order, i.e. the earliest part.
    _, first = np.unique(merged["time"].values, return_index=True)
    merged = merged.isel(time=np.sort(first)).sortby("time")

    target = LakeCube(out / f"{lake_id}.zarr", grid)
    chunks = {"time": min(32, merged.sizes["time"]), "y": -1, "x": -1}
    target._write_fresh(merged.chunk(chunks))
    target.flag_stats_path.write_text(
        json.dumps(dict(sorted(flags.items())), indent=1), encoding="utf-8"
    )
    days = merged["time"].values
    return {
        "lake_id": lake_id,
        "days": int(merged.sizes["time"]),
        "first": str(days[0])[:10],
        "last": str(days[-1])[:10],
        "parts": used,
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--aoi", type=Path, required=True)
    p.add_argument("--parts", type=Path, nargs="+", required=True, help="earliest wins")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--lakes", nargs="*", default=None)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")

    if args.out.resolve() in {r.resolve() for r in args.parts}:
        raise SystemExit("--out must not be one of --parts")
    registry = AOIRegistry.from_geojson(args.aoi)
    args.out.mkdir(parents=True, exist_ok=True)
    report = []
    for lake_id in args.lakes or registry.lake_ids:
        r = merge_lake(lake_id, args.parts, args.out, registry)
        logger.info("%s: %s", lake_id, r)
        report.append(r)
    (args.out / "merge_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    missing = [r["lake_id"] for r in report if not r["days"]]
    if missing:
        logger.warning("no cube in any part for: %s", ", ".join(missing))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
