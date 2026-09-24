"""Build the ERA5 meteorological store for every registered lake.

    python pipelines/ingest_era5.py                          # all lakes, 2018-2025
    python pipelines/ingest_era5.py --lakes erie_west taihu
    python pipelines/ingest_era5.py --start 2018 --end 2025 --force
    python pipelines/ingest_era5.py --source open-meteo --aoi data/aoi/lakes_global.geojson         --out data/processed/met_global --start 2016 --end 2026

Two sources of the same ERA5 reanalysis (see `habforecast.data.era5`):

* `cds` -- the Copernicus CDS, the lake bbox at 0.25 degrees, 3-hourly. Queued: hours per
  lake, and an account with several requests in flight is refused.
* `open-meteo` -- Open-Meteo's historical API, hourly, the ERA5 cell at the lake centre.
  Seconds per lake-decade; metered at ~10,000 calls a day (a lake-decade is ~260), and
  backs off on HTTP 429. No account needed.

Requires a Copernicus CDS account and `~/.cdsapirc`. The first run on a new account fails
with *"required licences not accepted"* until the ERA5 licence is accepted once, in a
browser, at the URL the error prints -- this script repeats that URL rather than letting
a 403 look like a credential problem.

Downloads are cached per lake-year under `data/cache/era5`, so a re-run costs nothing and
an interrupted run resumes. One CDS request per lake-year; they queue, so the first full
build takes a while and every one after it is instant.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

logger = logging.getLogger(__name__)

LICENCE_URL = (
    "https://cds.climate.copernicus.eu/datasets/"
    "reanalysis-era5-single-levels?tab=download#manage-licences"
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--aoi", type=Path, default=REPO_ROOT / "data" / "aoi" / "lakes.geojson")
    p.add_argument("--cache", type=Path, default=REPO_ROOT / "data" / "cache" / "era5")
    p.add_argument("--out", type=Path, default=REPO_ROOT / "data" / "processed" / "met")
    p.add_argument("--lakes", nargs="*", default=None, help="subset of lake_ids")
    p.add_argument("--start", type=int, default=2018, help="first year (OLCI archive starts 2018)")
    p.add_argument("--end", type=int, default=2025)
    p.add_argument("--force", action="store_true", help="re-download cached chunks")
    p.add_argument(
        "--cubes",
        type=Path,
        default=REPO_ROOT / "data" / "processed" / "cubes",
        help="chlorophyll cubes, read to work out which months each lake needs",
    )
    p.add_argument("--workers", type=int, default=3, help="concurrent CDS requests")
    p.add_argument("--source", choices=["cds", "open-meteo"], default="cds")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")

    from habforecast.aoi import AOIRegistry
    from habforecast.data import era5

    registry = AOIRegistry.from_geojson(args.aoi)
    lake_ids = args.lakes or registry.lake_ids
    years = list(range(args.start, args.end + 1))

    client = None
    if args.source == "cds":
        import cdsapi

        client = cdsapi.Client()

    failures = []
    for lake_id in lake_ids:
        aoi = registry[lake_id]
        try:
            if args.source == "open-meteo":
                from datetime import date

                era5.build_store_open_meteo(
                    aoi,
                    date(args.start, 1, 1),
                    date(args.end, 12, 31),
                    args.cache / "open_meteo",
                    args.out,
                )
                continue
            era5.build_store(
                aoi,
                years,
                args.cache,
                args.out,
                client=client,
                cube_path=args.cubes / f"{lake_id}.zarr",
                workers=args.workers,
            )
        except Exception as exc:  # one lake failing must not lose the others
            if "licence" in str(exc).lower():
                logger.error(
                    "ERA5 licence not accepted for this account. Accept it once at:\n  %s",
                    LICENCE_URL,
                )
                return 2
            logger.error("%s: %s: %s", lake_id, type(exc).__name__, exc)
            failures.append(lake_id)

    done = [lid for lid in lake_ids if lid not in failures]
    logger.info("built %d/%d met stores in %s", len(done), len(lake_ids), args.out)
    if failures:
        logger.warning("failed: %s", ", ".join(failures))
    return 1 if not done else 0


if __name__ == "__main__":
    raise SystemExit(main())
