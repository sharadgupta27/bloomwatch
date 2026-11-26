"""Validate Bloom Watch against public field samples (GLORIA + US EPA Water Quality Portal).

Two questions, kept apart because they fail for different reasons:

1. **Is the satellite right?** Same-day matchups of the archived OLCI chlorophyll (the
   model's input *and* its training target) against laboratory chlorophyll-a. This is
   the ceiling: a forecast cannot be closer to the water than the retrieval it learned.
2. **Is the forecast right?** For every sample on day t and every lead L, the forecast
   issued on t-L from the archive is compared with the bottle -- beside persistence
   ("the last thing the satellite saw") and the seasonal climatology, on exactly the same
   samples, so the comparison is paired.

Nothing here was used in training: the field samples are an independent measurement.
The satellite archive the forecasts are issued from is the one the model was trained on,
so on training lakes the *inputs* are familiar; the split of each lake is reported so a
reader can look at held-out lakes alone.

    python scripts/validate_insitu.py                                  # all lakes
    python scripts/validate_insitu.py --lakes erie_west saginaw --max-anchors 50

Writes artifacts/insitu_validation/{matchups.csv.gz, summary.json}. The API serves the
summary at /api/validation-summary and the app shows it in "How good is it?".
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import sys
import time
from collections import defaultdict
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402,F401  (before pandas string indexes -- see tests/conftest.py)
from pyproj import Transformer  # noqa: E402

from habforecast.eval.matchup import (  # noqa: E402
    THRESHOLDS_MG_M3,
    box_median,
    categorical,
    cluster_bootstrap_mae_gain,
    continuous,
)
from habforecast.inference.predictor import load_predictor  # noqa: E402

log = logging.getLogger("validate_insitu")


def load_samples(path: Path, start: date, end: date) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[(df["variable"] == "chlorophyll_a") & (df["value"] > 0)].copy()
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df = df[(df["date"] >= start) & (df["date"] <= end)]
    return df.reset_index(drop=True)


def place_on_grid(df: pd.DataFrame, grid) -> pd.DataFrame:
    tf = Transformer.from_crs("EPSG:4326", f"EPSG:{grid.epsg}", always_xy=True)
    x, y = tf.transform(df["lon"].to_numpy(), df["lat"].to_numpy())
    col = np.floor((np.asarray(x) - grid.x_min) / grid.resolution).astype(int)
    row = np.floor((grid.y_max - np.asarray(y)) / grid.resolution).astype(int)
    out = df.assign(row=row, col=col)
    inside = (row >= 0) & (row < grid.height) & (col >= 0) & (col < grid.width)
    return out[inside]


def near_water(water: np.ndarray, row: int, col: int) -> bool:
    return bool(water[max(0, row - 1) : row + 2, max(0, col - 1) : col + 2].any())


def satellite_box(ds, days: dict, d: date, row: int, col: int) -> tuple[float, int | None]:
    """Same-day box median, else the nearer of the day before/after. (value, |dt|)."""
    for dt in (0, -1, 1):
        i = days.get(d + timedelta(days=dt))
        if i is None:
            continue
        r0, c0 = max(0, row - 1), max(0, col - 1)
        block = ds["log_chl"].isel(time=i, y=slice(r0, row + 2), x=slice(c0, col + 2)).values
        v = box_median(np.asarray(block, dtype=float), row - r0, col - c0)
        if np.isfinite(v):
            return v, abs(dt)
    return float("nan"), None


def persistence_box(window: np.ndarray | None, row: int, col: int) -> float:
    """Newest finite box median in the input window (T, H, W), oldest first."""
    if window is None:
        return float("nan")
    for t in range(window.shape[0] - 1, -1, -1):
        v = box_median(window[t], row, col)
        if np.isfinite(v):
            return v
    return float("nan")


def run_lake(predictor, aoi, samples: pd.DataFrame, max_anchors: int | None) -> list[dict]:
    lake_id = aoi.lake_id
    ds, days = predictor._open_cube(lake_id)
    if ds is None:
        log.info("%s: no cube", lake_id)
        return []
    grid = aoi.grid()
    water = predictor._water_mask(lake_id, ds) > 0
    w, s, e, n = aoi.bbox
    sub = samples[samples.lon.between(w, e) & samples.lat.between(s, n)]
    sub = place_on_grid(sub, grid)
    if sub.empty:
        return []
    sub = sub[[near_water(water, r, c) for r, c in zip(sub.row, sub.col, strict=True)]]
    if sub.empty:
        return []
    # Replicates (depths, duplicate submissions) at one pixel on one day are one matchup.
    agg = (
        sub.groupby(["date", "row", "col"])
        .agg(
            insitu=("value", "median"),
            n_raw=("value", "size"),
            source=("source", "first"),
            site=("site", "first"),
            lat=("lat", "mean"),
            lon=("lon", "mean"),
        )
        .reset_index()
    )
    rows: list[dict] = []
    for rec in agg.itertuples(index=False):
        sat, sat_dt = satellite_box(ds, days, rec.date, rec.row, rec.col)
        rows.append(
            {
                "lake_id": lake_id,
                "split": aoi.split,
                "date": rec.date.isoformat(),
                "row": rec.row,
                "col": rec.col,
                "lat": rec.lat,
                "lon": rec.lon,
                "source": rec.source,
                "site": rec.site,
                "n_raw": rec.n_raw,
                "insitu_mg_m3": rec.insitu,
                "kind": "satellite",
                "lead": 0,
                "model_log10": sat,
                "sat_dt_days": sat_dt,
                "persistence_log10": np.nan,
                "climatology_log10": np.nan,
            }
        )

    by_anchor: dict[date, list] = defaultdict(list)
    for rec in agg.itertuples(index=False):
        for li, lead in enumerate(predictor.lead_times):
            by_anchor[rec.date - timedelta(days=lead)].append((li, lead, rec))
    anchors = sorted(by_anchor)
    if max_anchors and len(anchors) > max_anchors:
        pick = np.linspace(0, len(anchors) - 1, max_anchors).round().astype(int)
        anchors = [anchors[i] for i in sorted(set(pick))]
    t0 = time.time()
    skipped = 0
    for k, anchor in enumerate(anchors):
        try:
            res = predictor.predict(lake_id, anchor, allow_live=False)
        except ValueError:
            skipped += 1
            continue
        # The baseline is the archive's seasonal mean whether or not the model reads it.
        clim = predictor._climatology(lake_id, anchor)
        clim_log = None
        if clim is not None:
            cf, cm = clim
            clim_log = np.where(cm > 0, predictor.norm.invert(cf), np.nan)
        for li, lead, rec in by_anchor[anchor]:
            rows.append(
                {
                    "lake_id": lake_id,
                    "split": aoi.split,
                    "date": rec.date.isoformat(),
                    "row": rec.row,
                    "col": rec.col,
                    "lat": rec.lat,
                    "lon": rec.lon,
                    "source": rec.source,
                    "site": rec.site,
                    "n_raw": rec.n_raw,
                    "insitu_mg_m3": rec.insitu,
                    "kind": "forecast",
                    "lead": lead,
                    "model_log10": box_median(res.log_chl[li], rec.row, rec.col),
                    "sat_dt_days": None,
                    "persistence_log10": persistence_box(res.input_log_chl, rec.row, rec.col),
                    "climatology_log10": (
                        box_median(clim_log[li], rec.row, rec.col)
                        if clim_log is not None
                        else np.nan
                    ),
                }
            )
        if (k + 1) % 50 == 0:
            log.info("%s: %d/%d anchors, %.0f s", lake_id, k + 1, len(anchors), time.time() - t0)
    log.info(
        "%s: %d samples, %d anchors (%d without imagery) in %.0f s",
        lake_id,
        len(agg),
        len(anchors),
        skipped,
        time.time() - t0,
    )
    return rows


def _r(x, nd=3):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return None
    return round(float(x), nd)


def _clean(obj):
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean(v) for v in obj]
    if isinstance(obj, float):
        return _r(obj)
    return obj


def lake_sentence(name: str, sat: dict) -> str:
    # Fewer than ten same-day pairs is noise, not a property of the lake.
    if not sat or sat.get("n", 0) < 10:
        return f"Too few same-day field samples at {name} to say how the satellite compares."
    within = round(100 * sat["within_2x"])
    b = sat["bias_factor"]
    lean = (
        f"tends to read {b:.1f}x high"
        if b >= 1.25
        else f"tends to read {1 / b:.1f}x low"
        if b <= 0.8
        else "shows no strong high or low lean"
    )
    return (
        f"At {name}, the satellite was within a factor of 2 of the field sample "
        f"{within}% of the time (n={sat['n']}) and {lean}."
    )


def summarise(df: pd.DataFrame, names: dict[str, str], meta: dict) -> dict:
    obs_log = np.log10(df["insitu_mg_m3"].to_numpy())
    df = df.assign(obs_log=obs_log, cluster=df["lake_id"] + "|" + df["date"])

    sat = df[(df.kind == "satellite") & df.model_log10.notna()]
    sat0 = sat[sat.sat_dt_days == 0]
    satellite = {
        "same_day": continuous(sat0.model_log10, sat0.obs_log),
        "within_1_day": continuous(sat.model_log10, sat.obs_log),
        "detection_same_day": [
            categorical(10**sat0.model_log10, sat0.insitu_mg_m3, t) for t in THRESHOLDS_MG_M3
        ],
    }

    forecast = []
    fc = df[df.kind == "forecast"]
    for lead, g in fc.groupby("lead"):
        paired = g[g.model_log10.notna() & g.persistence_log10.notna()]
        entry = {
            "lead_days": int(lead),
            "model": continuous(paired.model_log10, paired.obs_log),
            "persistence": continuous(paired.persistence_log10, paired.obs_log),
            "skill_vs_persistence": cluster_bootstrap_mae_gain(
                paired.model_log10, paired.persistence_log10, paired.obs_log, paired.cluster
            ),
            "detection": [
                {
                    "model": categorical(10**paired.model_log10, paired.insitu_mg_m3, t),
                    "persistence": categorical(
                        10**paired.persistence_log10, paired.insitu_mg_m3, t
                    ),
                }
                for t in THRESHOLDS_MG_M3
            ],
        }
        pc = g[g.model_log10.notna() & g.climatology_log10.notna()]
        if len(pc) >= 5:
            entry["climatology"] = continuous(pc.climatology_log10, pc.obs_log)
            entry["skill_vs_climatology"] = cluster_bootstrap_mae_gain(
                pc.model_log10, pc.climatology_log10, pc.obs_log, pc.cluster
            )
        held = paired[paired.split != "train"]
        if len(held) >= 5:
            entry["held_out_lakes"] = {
                "model": continuous(held.model_log10, held.obs_log),
                "persistence": continuous(held.persistence_log10, held.obs_log),
            }
        forecast.append(entry)

    by_lake = []
    for lake_id, g in df.groupby("lake_id"):
        s = g[(g.kind == "satellite") & (g.sat_dt_days == 0)]
        s_cont = continuous(s.model_log10, s.obs_log)
        f1 = g[(g.kind == "forecast") & (g.lead == g[g.kind == "forecast"].lead.min())]
        f1 = f1[f1.model_log10.notna()]
        by_lake.append(
            {
                "lake_id": lake_id,
                "name": names.get(lake_id, lake_id),
                "split": g.split.iloc[0],
                "sources": sorted(g.source.unique().tolist()),
                "samples": int((g.kind == "satellite").sum()),
                "first": g.date.min(),
                "last": g.date.max(),
                "satellite_same_day": s_cont,
                "forecast_shortest_lead": continuous(f1.model_log10, f1.obs_log),
                "sentence": lake_sentence(names.get(lake_id, lake_id), s_cont),
            }
        )
    by_lake.sort(key=lambda r: -r["samples"])

    correction = loo_correction(df)
    return _clean(
        {
            "meta": meta,
            "headline": headlines(satellite, forecast, sat0.lake_id.nunique(), correction),
            "satellite": satellite,
            "forecast": forecast,
            "bias_correction_experiment": correction,
            "by_lake": by_lake,
        }
    )


def _verdict(sk: dict, better: str, worse: str) -> str:
    ci = sk.get("ci95")
    if ci and ci[0] > 0:
        return better
    if ci and ci[1] < 0:
        return worse
    return "no better or worse than"


def headlines(satellite: dict, forecast: list[dict], n_lakes: int, correction: dict) -> list[str]:
    """Plain statements a member of the public can read, failures included."""
    out = []
    sd = satellite["same_day"]
    if sd.get("n"):
        lean = sd["bias_factor"]
        how = (
            f"about {1 / lean:.1f}x lower than"
            if lean < 0.8
            else (f"about {lean:.1f}x higher than" if lean > 1.25 else "close to")
        )
        out.append(
            f"On {sd['n']} same-day matchups across {n_lakes} lakes, the satellite ranked "
            f"waters well (rank correlation {sd.get('spearman_rho', float('nan')):.2f}) but read "
            f"{how} the laboratory value on average; it was within a factor of 2 "
            f"{round(100 * sd['within_2x'])}% of the time."
        )
    if forecast and forecast[0]["model"].get("n"):
        f = forecast[0]
        out.append(
            f"The {f['lead_days']}-day forecast was "
            + _verdict(
                f["skill_vs_persistence"],
                "closer to the field samples than",
                "further from them than",
            )
            + " repeating the last satellite image, but "
            + _verdict(
                f.get("skill_vs_climatology", {}), "also closer than", "further from them than"
            )
            + " the lake's usual value for the time of year."
        )
        det20 = next((d for d in f["detection"] if d["model"]["threshold_mg_m3"] == 20.0), None)
        if det20 and det20["model"]["observed_events"]:
            fix = correction.get("forecast_lead_1", {}).get("corrected", {}).get("pod_20")
            out.append(
                f"Because of that low reading, the forecast flagged only "
                f"{round(100 * (det20['model']['pod'] or 0))}% of field samples above "
                f"20 ug/L. A single correction learned from the other lakes raises that to "
                f"{round(100 * fix)}%, at the cost of more false alarms — not yet applied."
                if fix is not None
                else ""
            )
    return [h for h in out if h]


def loo_correction(df: pd.DataFrame) -> dict:
    """What a one-number multiplicative calibration to field samples would change.

    The factor applied to each lake is the median log ratio (bottle / prediction) on all
    *other* lakes, so no lake is scored with a correction fitted on its own samples. This
    is an experiment recorded for a decision, not something the service applies.
    """
    out = {}
    for name, kind, lead in (
        ("satellite_same_day", "satellite", 0),
        ("forecast_lead_1", "forecast", 1),
    ):
        g = df[(df.kind == kind) & (df.lead == lead) & df.model_log10.notna()]
        if kind == "satellite":
            g = g[g.sat_dt_days == 0]
        if g.lake_id.nunique() < 3:
            continue
        shift = pd.Series(np.nan, index=g.index)
        for lake in g.lake_id.unique():
            other = g[g.lake_id != lake]
            shift[g.lake_id == lake] = float(np.median(other.obs_log - other.model_log10))
        corrected = g.model_log10 + shift
        block = {"median_factor": float(10 ** shift.median())}
        for label, col in (("raw", g.model_log10), ("corrected", corrected)):
            c = continuous(col, g.obs_log)
            row = {"within_2x": c.get("within_2x"), "bias_factor": c.get("bias_factor")}
            for t in THRESHOLDS_MG_M3:
                d = categorical(10**col, g.insitu_mg_m3, t)
                row[f"pod_{t:g}"] = d["pod"]
                row[f"far_{t:g}"] = d["far"]
            block[label] = row
        out[name] = block
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--checkpoint", default="models/best.ckpt")
    ap.add_argument("--cubes", default="data/processed/cubes_global_live")
    ap.add_argument("--aoi", default="data/aoi/lakes_global.geojson")
    ap.add_argument("--insitu", default="data/validation/insitu.csv.gz")
    ap.add_argument("--out", default="artifacts/insitu_validation")
    ap.add_argument("--lakes", nargs="*", help="lake_ids (default: every lake with samples)")
    ap.add_argument("--max-anchors", type=int, default=None, help="cap forecasts per lake")
    ap.add_argument("--no-tta", action="store_true")
    ap.add_argument(
        "--from-matchups",
        action="store_true",
        help="rebuild summary.json from an existing matchups.csv.gz without re-forecasting",
    )
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    out = Path(args.out)
    t0 = time.time()
    if args.from_matchups:
        from habforecast.aoi import AOIRegistry

        reg = AOIRegistry.from_geojson(args.aoi)
        df = pd.read_csv(out / "matchups.csv.gz", dtype={"date": str})
        previous = json.loads((out / "summary.json").read_text(encoding="utf-8")).get("meta", {})
    else:
        predictor = load_predictor(args.checkpoint, args.aoi, args.cubes, tta=not args.no_tta)
        samples = load_samples(Path(args.insitu), date(2016, 4, 1), datetime.now(UTC).date())
        reg = predictor.registry
        lakes = args.lakes or list(reg.lake_ids)
        rows: list[dict] = []
        for lake_id in lakes:
            rows += run_lake(predictor, reg[lake_id], samples, args.max_anchors)
        if not rows:
            raise SystemExit("no matchups found")
        df = pd.DataFrame(rows)
        previous = {}
        out.mkdir(parents=True, exist_ok=True)
        with gzip.open(out / "matchups.csv.gz", "wt", encoding="utf-8", newline="") as fh:
            df.to_csv(fh, index=False)
    meta = {
        "generated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "model": Path(args.checkpoint).stem,
        "tta": not args.no_tta,
        "cubes": args.cubes,
        "insitu_sources": {
            "GLORIA": "Lehmann et al. 2023, PANGAEA, CC-BY 4.0",
            "EPA WQP": "US EPA / USGS Water Quality Portal",
        },
        "method": (
            "3x3-pixel median at 300 m vs laboratory chlorophyll-a (replicates at one pixel "
            "and day merged); log10 errors; 95% intervals by bootstrap over lake-days. "
            "Satellite = archived OLCI retrieval on the sample day (or +/-1 day). Forecast "
            "= issued L days before the sample from the archive; persistence = the newest "
            "satellite value in the forecast's input window; climatology = +/-15-day "
            "seasonal mean from other years."
        ),
        "caveats": [
            "Field samples are point measurements, often near shore or at depth; a 300 m "
            "pixel averages the surface. Some disagreement is representativeness, not error.",
            "On training lakes the satellite inputs were seen during training (the field "
            "samples were not); held-out lakes are reported separately.",
            "Most samples are from US lakes (EPA WQP) and from 2016-2022 (GLORIA ends 2022).",
        ],
        "runtime_s": previous.get("runtime_s", round(time.time() - t0, 1)),
    }
    names = {lid: reg[lid].name for lid in reg.lake_ids}
    summary = summarise(df, names, meta)
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\n".join(summary["headline"]))
    print(f"wrote {out / 'summary.json'} ({len(df)} rows, {meta['runtime_s']} s)")


if __name__ == "__main__":
    main()
