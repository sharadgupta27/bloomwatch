"""Restore georeferencing metadata on cubes written before the append-attrs fix.

`to_zarr(mode="a")` overwrites a group's attributes with those of the appended dataset.
Because the ingest appends month by month with blocks that carry no attrs, every cube
built before that fix lost its CRS, resolution and grid definition. The array data and
the x/y coordinates are unaffected, so the metadata can simply be written back from the
AOI registry -- no re-ingest needed.

    python scripts/repair_cube_attrs.py            # report only
    python scripts/repair_cube_attrs.py --apply
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import zarr  # noqa: E402

from habforecast.aoi import AOIRegistry  # noqa: E402

REQUIRED = ("lake_id", "crs_epsg", "resolution_m", "grid")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--aoi", type=Path, default=REPO_ROOT / "data" / "aoi" / "lakes.geojson")
    p.add_argument("--cubes", type=Path, default=REPO_ROOT / "data" / "processed" / "cubes")
    p.add_argument("--apply", action="store_true", help="write the attrs (default: report only)")
    args = p.parse_args(argv)

    registry = AOIRegistry.from_geojson(args.aoi)
    repaired = intact = missing = 0

    for aoi in registry:
        path = args.cubes / f"{aoi.lake_id}.zarr"
        if not path.exists():
            print(f"  {aoi.lake_id:14s} no cube")
            missing += 1
            continue

        root = zarr.open_group(str(path), mode="a" if args.apply else "r")
        absent = [k for k in REQUIRED if root.attrs.get(k) is None]
        if not absent:
            print(f"  {aoi.lake_id:14s} OK")
            intact += 1
            continue

        print(f"  {aoi.lake_id:14s} missing {absent}" + ("" if args.apply else "  (dry run)"))
        if args.apply:
            grid = aoi.grid()
            root.attrs.update(
                {
                    "lake_id": grid.lake_id,
                    "crs_epsg": grid.epsg,
                    "resolution_m": grid.resolution,
                    "grid": json.dumps(asdict(grid)),
                    "title": f"habforecast daily chlorophyll cube -- {grid.lake_id}",
                    "conventions": "CF-1.10",
                    "repaired_utc": datetime.now(UTC).isoformat(timespec="seconds"),
                }
            )
            zarr.consolidate_metadata(str(path))
            repaired += 1

    print(f"\n  intact={intact} repaired={repaired} missing={missing}")
    if not args.apply and repaired == 0 and intact < len(registry):
        print("  re-run with --apply to write the metadata back")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
