"""HAB bulletin: NCCOS-style forecast products for any registry lake, as one PDF.

NOAA NCCOS publishes, for Lake Erie only, four products: the latest satellite image of
cyanobacterial density, a bloom position forecast, a wind-induced mixing forecast and a
scum-risk curve. This builds the same set for every lake in a registry (default: the
46-lake global one), from this project's own inputs:

- **Satellite imagery** -- the most recent Sentinel-3 OLCI 300 m scene in the lake's cube
  and a 7-day maximum composite (the recent bloom extent under intermittent cloud).
- **Bloom forecast** -- the served map model's chlorophyll-a at every lead, and the
  probability of exceeding 20 mg m^-3 from the probability model, with the advisory level
  computed by `habforecast.advisory` exactly as the daily feed computes it.
- **Wind & mixing** -- hourly 10 m wind at the lake centre from Open-Meteo (5 days past, 4
  ahead, local time), each arrow classed by NCCOS's mixing rule of thumb (blooms mix
  through the water column above ~15 kn), and a 6-hourly surface-scum risk.
- A cover table of every lake's level, and a methods page with the 2026 near-real-time
  verification this model has.

    python scripts/make_bulletin.py --anchor 2026-09-27                   # all 46 lakes
    python scripts/make_bulletin.py --anchor 2026-09-27 --lakes erie_west taihu

**What differs from NCCOS.** Their bloom position forecast advects the observed bloom with
a 3-D hydrodynamic model (LEOFS); ours is a statistical forecast of chlorophyll-a from the
lake's recent imagery, its own climatology and weather. Their density is the
Cyanobacteria Index; ours is chlorophyll-a, which does not separate cyanobacteria from
other algae. Their mixing classes and scum risk come from a calibrated Erie model; ours are
a transparent heuristic on wind speed (constants below), gated on whether a bloom is
forecast at all. Every page says so.

Run `artifacts/bulletin/refresh_global_live.sh <anchor>` first so the cubes hold this
fortnight's imagery; without it a lake whose cube ends months ago is reported unavailable.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
from datetime import date, timedelta
from pathlib import Path

# torch before anything that builds pandas/pyarrow string indexes (see tests/conftest.py).
import torch  # noqa: F401

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from habforecast.bulletin import (  # noqa: E402
    LEVEL_COLOURS,
    LEVEL_RANK,
    Geo,
    banded,
    bloom_share,
    fetch_wind,
    footer,
    forecast_pages,
    header,
    map_share,
    new_page,
    recent_imagery,
    write_pdf,
)

logger = logging.getLogger(__name__)


def open_cube(path: Path):
    """The lake's cube, or None when it has never been ingested."""
    import xarray as xr

    return xr.open_zarr(path, consolidated=True) if path.exists() else None


def cube_frames(ds, anchor: date, window: int) -> tuple[list[date], list]:
    """Observed days and their log10-chlorophyll frames in the `window` days to `anchor`."""
    import numpy as np

    times = ds["time"].values.astype("datetime64[D]")
    start = np.datetime64(anchor - timedelta(days=window - 1))
    idx = np.flatnonzero((times >= start) & (times <= np.datetime64(anchor)))
    if idx.size == 0:
        return [], []
    block = ds["log_chl"].isel(time=idx.tolist()).values
    return [times[i].astype(object) for i in idx], list(block)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--aoi", type=Path, default=REPO_ROOT / "data/aoi/lakes_global.geojson")
    p.add_argument("--lakes", nargs="*", default=None, help="subset of lake_ids; default all")
    p.add_argument("--anchor", default=None, help="ISO issue date; default today")
    p.add_argument("--cubes", type=Path, default=REPO_ROOT / "data/processed/cubes_global_live")
    p.add_argument("--met-root", type=Path, default=REPO_ROOT / "data/processed/met_live")
    p.add_argument("--checkpoint", type=Path, default=REPO_ROOT / "models/best.ckpt")
    p.add_argument(
        "--prob-checkpoint",
        default=os.getenv("HAB_PROB_CHECKPOINT", str(REPO_ROOT / "models/ens_met")),
        help="probability model; 'none' takes probabilities from --checkpoint",
    )
    p.add_argument("--out", type=Path, default=None, help="default artifacts/bulletin/<anchor>")
    p.add_argument("--min-fraction", type=float, default=0.10)
    p.add_argument("--composite-days", type=int, default=7)
    p.add_argument("--device", default=None)
    p.add_argument("--no-tta", action="store_true")
    p.add_argument("--no-wind", action="store_true", help="skip the Open-Meteo wind pages")
    p.add_argument("--no-png", action="store_true", help="PDF only")
    return p


def cover_pages(rows: list[dict], anchor: date, model: str, per_page: int = 24):
    rows = sorted(rows, key=lambda r: (LEVEL_RANK.get(r["level"], 9), -r["peak20"], r["name"]))
    counts = {}
    for r in rows:
        counts[r["level"]] = counts.get(r["level"], 0) + 1
    figs = []
    for start in range(0, len(rows), per_page):
        chunk = rows[start : start + per_page]
        fig = new_page()
        header(
            fig,
            "Bloom Watch — Harmful Algal Bloom Bulletin",
            f"Issued {anchor.isoformat()}  ·  {len(rows)} lakes  ·  "
            + ", ".join(
                f"{k}: {v}"
                for k, v in sorted(counts.items(), key=lambda kv: LEVEL_RANK.get(kv[0], 9))
            )
            + f"  ·  model {model}",
        )
        ax = fig.add_axes([0.03, 0.06, 0.94, 0.82])
        ax.axis("off")
        cols = [
            "Lake",
            "Country",
            "Level",
            "From",
            "Confidence",
            "Map % >20",
            "Latest scene",
            "Obs. days",
            "Note",
        ]
        cells, colours = [], []
        for r in chunk:
            cells.append(
                [
                    r["name"][:38],
                    r["country"][:12],
                    r["level"].upper(),
                    r["level_date"] or "",
                    r["confidence"],
                    r["peak20_txt"],
                    r["latest_scene"] or "none",
                    r["obs_days"],
                    r["note"][:48],
                ]
            )
            colours.append(
                ["white", "white", LEVEL_COLOURS.get(r["level"], "white")] + ["white"] * 6
            )
        tab = ax.table(
            cellText=cells,
            colLabels=cols,
            cellColours=colours,
            loc="upper center",
            cellLoc="left",
            colWidths=[0.2, 0.07, 0.08, 0.08, 0.08, 0.07, 0.08, 0.06, 0.28],
        )
        tab.auto_set_font_size(False)
        tab.set_fontsize(7)
        tab.scale(1, 1.25)
        footer(
            fig,
            "Levels (habforecast.advisory): WATCH/WARNING/ALERT when >= 10 % of the "
            "lake's water is forecast above 10/20/50 mg m^-3 chlorophyll-a within 7 days, held "
            "between the level with p >= 0.7 (likely) and p >= 0.3 (possible);\nwhere the "
            "daily feed's expected-fraction rule gives a different level, the Note says so. "
            "Advisory only -- satellite chlorophyll-a is not a toxin measurement.",
        )
        figs.append(fig)
    return figs


def methods_page(anchor: date, model: str, foot: str):
    fig = new_page()
    header(fig, "Methods & Verification")
    text = (
        "Inputs. Sentinel-3 OLCI 300 m chlorophyll-a (Planetary Computer archive 2017-2026, EUMETSAT "
        "reprocessed 2016-17, EUMETSAT Data Store for the last 14 days), gridded per lake at 300 m; "
        "daily weather from the Open-Meteo ERA5 archive (lags ~5 days: the newest input days carry "
        "no weather and are masked, as the model supports); hourly wind forecast from Open-Meteo.\n\n"
        f"Models. Map: {model.split(' + ')[0]}. Probabilities: {model.split(' + ')[-1]}. "
        "Forecast leads 1, 3, 5 and 7 days from a 14-day input window, anchored on the lake's own "
        "leave-one-year-out seasonal climatology. Lakes the models were not trained on are forecast "
        "with the same weights; their skill is not verified here.\n\n"
        "What this is not. Not a hydrodynamic bloom-transport model (NCCOS uses LEOFS for Lake Erie); "
        "not a Cyanobacteria Index or toxin product. Mixing classes and scum risk are a documented "
        "wind heuristic, not NCCOS's calibrated scheme.\n\n"
        "Verification: 2026 near-real-time holdout (never trained on), per lake and lead, for "
        "the lakes it has been run on -- every other lake in this bulletin is unverified in "
        "2026. Skill = 1 - RMSE/RMSE(persistence) in log10 chl-a; PR-AUC for exceeding "
        "20 mg m^-3 vs persistence."
    )
    fig.text(0.05, 0.88, text, fontsize=8.5, va="top", wrap=True)
    table = nrt_table()
    if table:
        ax = fig.add_axes([0.05, 0.1, 0.9, 0.5])
        ax.axis("off")
        cols = [
            "Lake",
            "Lead",
            "Skill vs persistence",
            "R²",
            "PR-AUC@20 (p model)",
            "PR-AUC@20 persistence",
        ]
        tab = ax.table(cellText=table, colLabels=cols, loc="upper center")
        tab.auto_set_font_size(False)
        tab.set_fontsize(7)
        tab.scale(1, 1.0)
    footer(fig, foot)
    return fig


def nrt_table() -> list[list[str]]:
    """Per-lake, per-lead 2026 NRT verification from the evaluation artifacts, if present."""
    best = REPO_ROOT / "artifacts/eval_nrt2026_best/skill_by_lead.csv"
    ens = REPO_ROOT / "artifacts/eval_nrt2026_ens_met/skill_by_lead.csv"
    if not best.exists():
        return []
    prauc = {}
    if ens.exists():
        with ens.open(newline="") as f:
            for r in csv.DictReader(f):
                prauc[(r["lake_id"], r["lead_days"])] = r["thr20_pr_auc"]
    out = []
    with best.open(newline="") as f:
        for r in csv.DictReader(f):
            key = (r["lake_id"], r["lead_days"])
            out.append(
                [
                    r["lake_id"],
                    f"+{r['lead_days']} d",
                    f"{float(r['skill_vs_persistence']):+.2f}",
                    f"{float(r['r2']):.2f}",
                    f"{float(prauc[key]):.2f}" if key in prauc else "",
                    f"{float(r['persistence_thr20_pr_auc']):.2f}",
                ]
            )
    return out


# ----------------------------------------------------------------------------- main --
def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")

    from habforecast import advisory
    from habforecast.inference.predictor import load_predictor

    anchor = date.fromisoformat(args.anchor) if args.anchor else date.today()
    out = args.out or REPO_ROOT / "artifacts" / "bulletin" / anchor.isoformat()
    (out / "png").mkdir(parents=True, exist_ok=True)
    (out / "wind").mkdir(parents=True, exist_ok=True)

    predictor = load_predictor(
        args.checkpoint,
        args.aoi,
        args.cubes,
        device=args.device,
        tta=not args.no_tta,
        met_root=args.met_root,
    )
    prob_predictor = None
    if args.prob_checkpoint.lower() != "none":
        prob_predictor = load_predictor(
            args.prob_checkpoint,
            args.aoi,
            args.cubes,
            device=args.device,
            tta=not args.no_tta,
            met_root=args.met_root,
        )
    lake_ids = args.lakes or predictor.registry.lake_ids
    results: dict = {}
    fc = advisory.build(
        predictor,
        anchor=anchor,
        lake_ids=lake_ids,
        allow_live=False,
        min_fraction=args.min_fraction,
        results=results,
        prob_predictor=prob_predictor,
    )
    (out / "advisory.geojson").write_text(json.dumps(fc, indent=2), encoding="utf-8")

    model = next(
        (
            f["properties"]["model_version"]
            for f in fc["features"]
            if f["properties"].get("status") == "ok"
        ),
        str(args.checkpoint.name),
    )
    foot = f"Bloom Watch · issued {anchor.isoformat()} · {model} · " + advisory.DISCLAIMER
    rows, lake_pages, summary = [], [], []
    from habforecast.data.era5 import lake_point

    for feat in fc["features"]:
        props = feat["properties"]
        lake_id = props["lake_id"]
        aoi = predictor.registry[lake_id]
        row = {
            "lake_id": lake_id,
            "name": props["name"],
            "country": props.get("country", ""),
            "level": props["level"],
            "level_date": props.get("level_date"),
            "confidence": props.get("confidence", "") or props.get("basis", ""),
            "peak20": 0.0,
            "peak20_txt": "",
            "latest_scene": None,
            "obs_days": "",
            "note": "",
        }
        cube = open_cube(args.cubes / f"{lake_id}.zarr")
        if props["status"] != "ok" or lake_id not in results:
            reason = props.get("reason", "no forecast")
            if cube is None:
                reason = "no OLCI cube (no usable retrievals)"
            elif not aoi.in_bloom_season(anchor.month):
                # The ingest fetches imagery only in the configured bloom months.
                m0, m1 = aoi.bloom_season
                reason = f"off-season (months {m0}-{m1})"
            elif "no archived observations" in reason:
                reason = "no clear scene in 14 days"
            row["note"] = reason
            if cube is not None:
                t = cube["time"].values
                row["latest_scene"] = str(t[-1])[:10] if t.size else None
            rows.append(row)
            summary.append({**row, "status": "unavailable"})
            logger.info("%-20s unavailable: %s", lake_id, reason)
            continue
        result = results[lake_id]
        props = banded(props)
        row.update(level=props["level"], level_date=props.get("level_date"))
        # From the chlorophyll map, not the expected fraction: the latter carries the
        # probability floor `banded` corrects for.
        peak20 = map_share(props)
        geo = Geo.from_lonlat(cube["lon"].values, cube["lat"].values)
        img = recent_imagery(
            *cube_frames(cube, anchor, args.composite_days),
            result.water,
            anchor,
            args.composite_days,
        )
        row.update(
            peak20=peak20,
            peak20_txt=f"{peak20:.1%}",
            latest_scene=img["latest_date"],
            obs_days=f"{props['n_observed_days']}/14",
        )
        if props.get("feed_level"):
            row["note"] = f"feed rule: {props['feed_level']} (diffuse low p)"
        elif props.get("confidence") == "uncertain":
            if props.get("possible_level") != props["level"]:
                row["note"] = f"could reach {props.get('possible_level')}"
            else:
                row["note"] = f"top of band (likely: {props.get('likely_level')})"

        wind = None
        if not args.no_wind:
            wind = fetch_wind(*lake_point(aoi), out / "wind" / f"{lake_id}.json")
        figs = forecast_pages(props, geo, result, img, wind, foot)
        for name, fig in figs:
            if not args.no_png:
                fig.savefig(out / "png" / f"{lake_id}_{name}.png", dpi=110)
            lake_pages.append((LEVEL_RANK.get(props["level"], 9), -peak20, lake_id, fig))
        rows.append(row)
        summary.append(
            {
                **row,
                "status": "ok",
                "scum_gate_share20": bloom_share(result),
                "latest_cover": img["latest_cover"],
                "wind": "ok" if wind is not None else "unavailable",
                "forecasts": props["forecasts"],
            }
        )
        logger.info(
            "%-20s %-8s peak>20 %.1f%%  latest scene %s",
            lake_id,
            props["level"],
            100 * peak20,
            img["latest_date"],
        )

    # Lakes in severity order after the cover, so the pages that matter come first.
    lake_pages.sort(key=lambda p: p[:3])
    pdf_path = out / f"bloom_watch_bulletin_{anchor.isoformat()}.pdf"
    figures = [
        *cover_pages(rows, anchor, model),
        *(fig for *_, fig in lake_pages),
        methods_page(anchor, model, foot),
    ]
    write_pdf(figures, pdf_path, f"Bloom Watch HAB Bulletin {anchor.isoformat()}")

    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    with (out / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        keys = [
            "lake_id",
            "name",
            "country",
            "level",
            "level_date",
            "confidence",
            "peak20",
            "latest_scene",
            "obs_days",
            "note",
        ]
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    ok = sum(1 for s in summary if s["status"] == "ok")
    logger.info("wrote %s (%d/%d lakes forecast)", pdf_path, ok, len(summary))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
