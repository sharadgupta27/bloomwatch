"""Scheduled advisory job: forecast every registry lake and publish the feed as files.

This is the "daily forecast" the catalogue asks for, at the scale this project runs at.
No orchestrator is introduced for it -- a container that runs once and exits, started by
cron, a GitHub Actions schedule or `docker compose run`, is the whole of the operational
need, and it is one more thing that cannot silently stop working. The API serves the
same feed live at `/advisory` and `/advisory/feed.xml`; this writes it to disk so it can
be published from static hosting, attached to an email, or diffed against yesterday's.

    python pipelines/advisory.py                       # today, into artifacts/advisory/
    python pipelines/advisory.py --anchor 2024-08-15   # a specific issue date
    python pipelines/advisory.py --allow-live          # fetch STAC for dates past the archive

Outputs: `advisory.geojson`, `feed.xml`, `drift.json`, and a dated copy of each so the
history is kept. Exit status is non-zero when *no* lake could be forecast (1), or -- only
with `--fail-on-drift` -- when an input drift alert fired (3); a single cloudy lake is
reported inside the feed, not as a failure.

**Keeping the inputs fresh.** `--refresh-days N` first appends the last N days of
Sentinel-3 OLCI from the EUMETSAT Data Store to the lake cubes under `--cubes` (the same
product and QC as the training archive, hours behind real time; `data/eumetsat.py`), then
forecasts from them -- so "today's" advisory is built from this week's imagery rather than
from the Planetary Computer mirror's six-month-old newest day. The ingest is idempotent:
days already in a cube are skipped, so a daily run fetches roughly one new day per lake.
Point `--cubes` at an operational copy (`data/processed/cubes_live`), never at the
training cubes a published number depends on. Without EUMETSAT credentials the refresh
is skipped with a warning and the job runs on whatever the cubes hold.

**Drift alerting.** Every forecast's input window is checked against the same lake's own
archive at the same time of year (monitoring.drift.window_drift). The result rides on
each feature as `properties.drift`, so a subscriber sees it beside the level; the run
writes `drift.json`; and when any lake drifts the job logs a warning, posts to
`--alert-webhook` (or `$HAB_ALERT_WEBHOOK`; Slack-compatible `{"text": ...}`) and, with
`--fail-on-drift`, exits 3 so a scheduler that alerts on failure alerts on this too.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))  # monitoring/
sys.path.insert(0, str(REPO_ROOT / "src"))

# Same-season reference for the drift check: +/- this many days of the anchor's day of
# year, from the archive *before* the input window (never the window itself).
DRIFT_SEASON_HALFWIDTH = 30
# Spatial stride when sampling the reference: every other pixel in each direction is
# still ~30k pixels per day on the smallest lake, and a quarter of the read.
DRIFT_STRIDE = 2

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=REPO_ROOT / "models" / "best.ckpt",
        help="the model that draws the chlorophyll map",
    )
    p.add_argument(
        "--prob-checkpoint",
        default=os.getenv("HAB_PROB_CHECKPOINT", str(REPO_ROOT / "models" / "ens_met")),
        help="the model the advisory probabilities come from (an ensemble directory, .ckpt "
        "or ONNX export); 'none' uses --checkpoint for both",
    )
    p.add_argument("--aoi", type=Path, default=REPO_ROOT / "data" / "aoi" / "lakes.geojson")
    p.add_argument("--cubes", type=Path, default=REPO_ROOT / "data" / "processed" / "cubes")
    p.add_argument("--out", type=Path, default=REPO_ROOT / "artifacts" / "advisory")
    p.add_argument("--anchor", type=str, default=None, help="ISO issue date; default today")
    p.add_argument("--lakes", nargs="*", default=None, help="subset of lake_ids")
    p.add_argument("--allow-live", action="store_true")
    p.add_argument("--min-fraction", type=float, default=0.10)
    p.add_argument("--base-url", default="", help="public URL prefix for links in the RSS")
    p.add_argument("--device", default=None)
    p.add_argument("--no-tta", action="store_true")
    p.add_argument(
        "--refresh-days",
        type=int,
        default=0,
        help="append the last N days from the EUMETSAT Data Store to --cubes before forecasting",
    )
    p.add_argument("--refresh-workers", type=int, default=8)
    p.add_argument("--no-drift", action="store_true", help="skip the input drift check")
    p.add_argument(
        "--alert-webhook",
        default=os.getenv("HAB_ALERT_WEBHOOK"),
        help="POST a Slack-compatible JSON message here when a drift alert fires",
    )
    p.add_argument(
        "--fail-on-drift", action="store_true", help="exit 3 when any lake's inputs drifted"
    )
    return p


def seasonal_reference(cube_path: Path, anchor: date, input_days: int, water: np.ndarray):
    """Daily statistics of the lake's archive near this day of year, before the window.

    None without a cube or without any archived day in the season.
    """
    import pandas as pd
    import xarray as xr
    from monitoring.drift import daily_stats

    if not cube_path.exists():
        return None
    ds = xr.open_zarr(cube_path, consolidated=True)
    times = pd.to_datetime(ds.time.values)
    doy = times.dayofyear.to_numpy()
    gap = np.abs(doy - anchor.timetuple().tm_yday)
    gap = np.minimum(gap, 365 - gap)
    cutoff = pd.Timestamp(anchor - timedelta(days=input_days))
    idx = np.flatnonzero((gap <= DRIFT_SEASON_HALFWIDTH) & (times < cutoff))
    if idx.size == 0:
        return None
    block = ds["log_chl"].isel(time=idx.tolist()).values[:, ::DRIFT_STRIDE, ::DRIFT_STRIDE]
    w = water[::DRIFT_STRIDE, ::DRIFT_STRIDE].astype(bool)
    return daily_stats(block[:, w])


def refresh_cubes(args, anchor: date) -> dict:
    """Append the last `--refresh-days` of Data Store imagery to each lake's cube."""
    from habforecast.aoi import AOIRegistry
    from habforecast.data import eumetsat
    from habforecast.data.ingest import ingest_lake
    from habforecast.data.olci import make_reader

    if not eumetsat.available():
        logger.warning(
            "--refresh-days set but no EUMETSAT credentials (%s); forecasting from the cubes "
            "as they are",
            eumetsat.credentials_path(),
        )
        return {}
    registry = AOIRegistry.from_geojson(args.aoi)
    start = (anchor - timedelta(days=args.refresh_days)).isoformat()
    written = {}
    for lake_id in args.lakes or registry.lake_ids:
        try:
            rep = ingest_lake(
                registry[lake_id],
                start,
                anchor.isoformat(),
                str(args.cubes),
                aoi_path=str(args.aoi),
                max_workers=args.refresh_workers,
                reader=make_reader("eumetsat"),
            )
            written[lake_id] = rep.days_written
        except Exception as exc:  # a failed refresh degrades freshness, not the feed
            logger.error("%s: refresh failed (%s: %s)", lake_id, type(exc).__name__, exc)
            written[lake_id] = None
    logger.info("refreshed from the EUMETSAT Data Store: %s", written)
    return written


def check_drift(fc: dict, results: dict, cube_root: Path, input_days: int) -> dict:
    """Attach a drift verdict to every forecast feature; return the run's drift summary."""
    from monitoring.drift import daily_stats, window_drift

    reports = {}
    for feat in fc["features"]:
        props = feat["properties"]
        result = results.get(props["lake_id"])
        if props.get("status") != "ok" or result is None or result.input_log_chl is None:
            continue
        ref = seasonal_reference(
            cube_root / f"{result.lake_id}.zarr", result.anchor_date, input_days, result.water
        )
        if ref is None:
            props["drift"] = {"checked": False, "reason": "no same-season archive"}
            continue
        # The same stride as the reference, so valid fractions are like for like.
        window = result.input_log_chl[:, ::DRIFT_STRIDE, ::DRIFT_STRIDE]
        water = result.water[::DRIFT_STRIDE, ::DRIFT_STRIDE]
        report = window_drift(
            result.lake_id, ref, daily_stats(window[:, water]), source=result.source
        )
        props["drift"] = {
            "checked": True,
            "any_drift": report.any_drift,
            "drifted_metrics": report.drifted_metrics,
        }
        reports[result.lake_id] = report.to_dict()
    alerts = sorted(k for k, r in reports.items() if r["any_drift"])
    return {"alerts": alerts, "reports": reports}


def post_alert(url: str, anchor: date, drift: dict) -> None:
    """Best effort: an unreachable webhook must not take the feed down with it."""
    import urllib.request

    lines = [f"habforecast {anchor}: input drift on {', '.join(drift['alerts'])}"]
    for lake in drift["alerts"]:
        metrics = drift["reports"][lake]["drifted_metrics"]
        lines.append(f"- {lake}: {', '.join(metrics)}")
    body = json.dumps({"text": "\n".join(lines), "drift": drift}).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            logger.info("drift alert posted (%s)", resp.status)
    except Exception as exc:
        logger.error("drift alert could not be posted: %s: %s", type(exc).__name__, exc)


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")

    from habforecast import advisory
    from habforecast.inference.predictor import load_predictor

    anchor = date.fromisoformat(args.anchor) if args.anchor else datetime.now(UTC).date()
    if args.refresh_days > 0:
        refresh_cubes(args, anchor)
    predictor = load_predictor(
        args.checkpoint, args.aoi, args.cubes, device=args.device, tta=not args.no_tta
    )
    prob_predictor = None
    if args.prob_checkpoint and args.prob_checkpoint.lower() != "none":
        prob_path = Path(args.prob_checkpoint)
        if not prob_path.exists():
            raise SystemExit(
                f"--prob-checkpoint {prob_path} does not exist; pass 'none' to take "
                "probabilities from --checkpoint"
            )
        prob_predictor = load_predictor(
            prob_path, args.aoi, args.cubes, device=args.device, tta=not args.no_tta
        )
    results: dict = {}
    fc = advisory.build(
        predictor,
        anchor=anchor,
        lake_ids=args.lakes,
        allow_live=args.allow_live,
        min_fraction=args.min_fraction,
        results=results,
        prob_predictor=prob_predictor,
    )
    drift: dict = {"alerts": [], "reports": {}}
    if not args.no_drift:
        drift = check_drift(fc, results, args.cubes, predictor.input_days)

    args.out.mkdir(parents=True, exist_ok=True)
    geojson = json.dumps(fc, indent=2)
    xml = advisory.rss(fc, base_url=args.base_url)
    drift_json = json.dumps({"anchor_date": anchor.isoformat(), **drift}, indent=2)
    outputs = [("advisory.geojson", geojson), ("feed.xml", xml)]
    if not args.no_drift:
        outputs.append(("drift.json", drift_json))
    for name, content in outputs:
        (args.out / name).write_text(content, encoding="utf-8")
        stem, ext = name.rsplit(".", 1)
        (args.out / f"{stem}-{anchor.isoformat()}.{ext}").write_text(content, encoding="utf-8")

    ok = [f for f in fc["features"] if f["properties"]["status"] == "ok"]
    for f in fc["features"]:
        p = f["properties"]
        logger.info(
            "%-14s %-12s %s",
            p["lake_id"],
            p["level"],
            p.get("level_date") or p.get("reason", ""),
        )
    logger.info("wrote %s (%d/%d lakes forecast)", args.out, len(ok), len(fc["features"]))
    if drift["alerts"]:
        logger.warning(
            "INPUT DRIFT on %s -- see %s", ", ".join(drift["alerts"]), args.out / "drift.json"
        )
        if args.alert_webhook:
            post_alert(args.alert_webhook, anchor, drift)
    if not ok:
        return 1
    return 3 if (drift["alerts"] and args.fail_on_drift) else 0


if __name__ == "__main__":
    raise SystemExit(main())
