"""Render real bloom episodes -- observation, forecast and error -- as embeddable PNGs.

The results dashboard was, until this script existed, entirely abstract: skill curves and
RMSE tables. Those answer "is the model any good?" but not "what does it actually see?",
and for a geospatial model the second question is the one a reviewer asks first.

This walks each lake's cube, finds episodes where the *target* days were genuinely
observed (so a forecast can be scored against something real rather than against a gap),
runs every available checkpoint over them, and writes ``artifacts/case_studies.json``:
base64 PNGs plus per-frame statistics, ready to inline into a self-contained page.

    python scripts/make_case_studies.py
    python scripts/make_case_studies.py --cases-per-lake 2 --lakes balaton victoria_ne

Rendering decisions that carry meaning:

* One fixed colour scale (1-100 mg/m^3, log) across every panel and every lake. An
  auto-scaled image makes a calm lake look identical to a blooming one, which is the most
  misleading thing a bloom map can do.
* Water that was *not observed* renders as a distinct flat grey, never as low
  chlorophyll. Roughly two thirds of pixels are cloud-gapped on a typical day; painting
  those as "clean water" would invent the very thing the model exists to predict.
* Error maps are diverging with a neutral midpoint, so over- and under-prediction are
  told apart by hue and zero error reads as absent.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import logging
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

logger = logging.getLogger("case_studies")

# Fixed display range, shared by every panel: 1 -> 100 mg/m^3 in log10.
VMIN, VMAX = 0.0, 2.0
# Diverging error range, log10 units. +-0.6 is a factor of ~4 either way, which spans
# essentially every error the model makes without letting one outlier flatten the map.
EMAX = 0.6

# Sequential bloom ramp, matching the map overlay the API already serves.
CHL_STOPS = np.array(
    [[38, 70, 160], [40, 160, 140], [180, 210, 60], [240, 180, 40], [200, 40, 40]],
    dtype=np.float64,
)
# Diverging: cool (under-predicted) -> neutral -> warm (over-predicted).
ERR_STOPS = np.array(
    [[33, 102, 172], [103, 169, 207], [222, 226, 228], [239, 138, 98], [178, 24, 43]],
    dtype=np.float64,
)
# Water with no retrieval. Deliberately desaturated: it must not read as a data value.
NODATA_RGB = (124, 136, 140)


def _ramp(norm: np.ndarray, stops: np.ndarray) -> np.ndarray:
    pos = np.linspace(0.0, 1.0, len(stops))
    flat = np.nan_to_num(norm, nan=0.0).ravel()
    rgb = np.stack([np.interp(flat, pos, stops[:, c]) for c in range(3)], axis=-1)
    return rgb.reshape((*norm.shape, 3)).astype(np.uint8)


def _encode(rgb: np.ndarray, alpha: np.ndarray) -> str:
    """RGBA array -> base64 data URI, quantised so the page stays small."""
    from PIL import Image

    rgba = np.dstack([rgb, alpha]).astype(np.uint8)
    img = Image.fromarray(rgba, mode="RGBA")
    # These are ramp images drawn from a single continuous scale, so a 256-entry palette
    # is visually lossless and roughly a third the bytes of full RGBA -- which matters
    # when a hundred of them are inlined into one page. Dithering must be off: on a
    # smooth predicted field Floyd-Steinberg lays down a diagonal weave that reads as
    # spatial structure the model never produced.
    img = img.quantize(colors=256, method=Image.Quantize.FASTOCTREE, dither=Image.Dither.NONE)
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def render_field(log_chl: np.ndarray, water: np.ndarray) -> str:
    """A chlorophyll field. Water without a retrieval is grey, land is transparent."""
    valid = water & np.isfinite(log_chl)
    norm = np.clip((log_chl - VMIN) / (VMAX - VMIN), 0.0, 1.0)
    rgb = _ramp(norm, CHL_STOPS)
    rgb[water & ~valid] = NODATA_RGB
    alpha = np.where(water, 255, 0).astype(np.uint8)
    return _encode(rgb, alpha)


def render_error(pred: np.ndarray, truth: np.ndarray, water: np.ndarray) -> str:
    """Signed prediction error, drawn only where a truth value exists."""
    scorable = water & np.isfinite(pred) & np.isfinite(truth)
    diff = np.where(scorable, pred - truth, np.nan)
    norm = np.clip((diff + EMAX) / (2 * EMAX), 0.0, 1.0)
    rgb = _ramp(norm, ERR_STOPS)
    rgb[water & ~scorable] = NODATA_RGB
    alpha = np.where(water, 255, 0).astype(np.uint8)
    return _encode(rgb, alpha)


def roughness(log_chl: np.ndarray) -> float | None:
    """Pixel-scale texture: neighbour difference relative to the field's own spread.

    Reported because it is the one defect an image shows and a scalar metric hides. RMSE
    barely penalises zero-mean pixel speckle, so a model can score well while producing a
    field that is visibly grainier than anything the sensor ever measured -- which matters
    the moment a product thresholds area rather than averaging.
    """
    b = np.where(np.isfinite(log_chl), log_chl, np.nan)
    if np.isfinite(b).sum() < 100:
        return None
    with np.errstate(invalid="ignore"):
        d = float(np.nanstd(b[1:, 1:] - b[:-1, :-1]))
        s = float(np.nanstd(b))
    return d / s if s > 0 else None


def frame_stats(log_chl: np.ndarray, water: np.ndarray, cell_km2: float) -> dict:
    masked = np.where(water, log_chl, np.nan)
    v = masked[np.isfinite(masked)]
    if v.size == 0:
        return {"coverage": 0.0, "median": None, "p90": None, "bloom_km2": None, "roughness": None}
    return {
        "coverage": float(v.size / max(int(water.sum()), 1)),
        "median": float(10 ** np.median(v)),
        "p90": float(10 ** np.percentile(v, 90)),
        "bloom_km2": float((v >= np.log10(20)).sum() * cell_km2),
        "roughness": roughness(masked),
    }


def pair_stats(pred: np.ndarray, truth: np.ndarray, water: np.ndarray) -> dict:
    m = water & np.isfinite(pred) & np.isfinite(truth)
    if not m.any():
        return {"rmse": None, "bias": None, "n": 0}
    d = pred[m] - truth[m]
    return {"rmse": float(np.sqrt(np.mean(d**2))), "bias": float(np.mean(d)), "n": int(m.sum())}


# ------------------------------------------------------------------- case selection --
def _daily_coverage(ds: xr.Dataset, water: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-day water coverage and 90th-percentile intensity, read block by block."""
    n_water = max(int(water.sum()), 1)
    n_time = ds.sizes["time"]
    cover = np.zeros(n_time)
    intensity = np.full(n_time, -9.0)
    da = ds["log_chl"]
    for start in range(0, n_time, 32):
        block = da.isel(time=slice(start, start + 32)).values
        for k in range(block.shape[0]):
            f = block[k]
            v = f[water & np.isfinite(f)]
            cover[start + k] = v.size / n_water
            if v.size:
                intensity[start + k] = float(np.percentile(v, 90))
    return cover, intensity


def find_cases(
    ds: xr.Dataset,
    water: np.ndarray,
    leads: tuple[int, ...],
    n_cases: int,
    input_days: int,
    min_target_coverage: float = 0.25,
) -> list[date]:
    """Anchor dates whose *target* frames are well observed and worth looking at.

    A case is only useful if the days being forecast were actually seen, otherwise the
    truth panel is empty grey and there is nothing to compare against. Among the
    qualifying anchors this prefers high chlorophyll: a calm lake is a dull picture and,
    more to the point, the model's behaviour during a bloom is the behaviour that matters.
    """
    times = [pd.Timestamp(t).date() for t in ds.time.values]
    index = {d: i for i, d in enumerate(times)}
    cover, intensity = _daily_coverage(ds, water)

    scored: list[tuple[float, date]] = []
    for anchor in times:
        tgt = [index.get(anchor + timedelta(days=ld)) for ld in leads]
        if any(j is None for j in tgt):
            continue
        tgt_cover = float(np.mean([cover[j] for j in tgt]))
        if tgt_cover < min_target_coverage:
            continue
        # The input window must contain something too, or the model is forecasting from
        # nothing and the panel says more about cloud cover than about the model.
        window = [index.get(anchor - timedelta(days=k)) for k in range(input_days)]
        seen = [cover[j] for j in window if j is not None]
        if not seen or float(np.mean(seen)) < 0.1:
            continue
        peak = float(max(intensity[j] for j in tgt))
        scored.append((tgt_cover + 0.6 * peak, anchor))

    scored.sort(key=lambda t: (-t[0], t[1]))
    # Spread the picks: three views of the same fortnight is one episode, not three.
    picked: list[date] = []
    for _, anchor in scored:
        if any(abs((anchor - p).days) < 60 for p in picked):
            continue
        picked.append(anchor)
        if len(picked) >= n_cases:
            break
    return picked


def truth_frame(ds: xr.Dataset, times: dict[date, int], day: date) -> np.ndarray | None:
    i = times.get(day)
    if i is None:
        return None
    return ds["log_chl"].isel(time=i).values.astype(np.float32)


def persistence_field(
    ds: xr.Dataset, times: dict[date, int], anchor: date, input_days: int, shape: tuple[int, int]
) -> np.ndarray:
    """The baseline every model is scored against: most recent observation per pixel.

    Worth rendering rather than only scoring, because it explains something the metrics
    hide. This field is a mosaic -- one pixel's value can come from yesterday and its
    neighbour's from nine days ago -- so it carries pixel-scale seams that no single day's
    composite has. Measured, it is 2.2x rougher than the observed target field *before any
    network is involved*, which is most of the graininess visible in every forecast built
    on top of it.
    """
    out = np.full(shape, np.nan, dtype=np.float32)
    for k in range(input_days):
        f = truth_frame(ds, times, anchor - timedelta(days=k))
        if f is None:
            continue
        fill = np.isnan(out) & np.isfinite(f)
        out[fill] = f[fill]
    return out


def build(
    checkpoints: dict[str, Path],
    aoi_path: Path,
    cube_root: Path,
    lakes: list[str] | None,
    n_cases: int,
    device: str | None,
) -> dict:
    from habforecast.inference.predictor import Predictor

    predictors: dict[str, Predictor] = {}
    for name, ckpt in checkpoints.items():
        logger.info("loading %s from %s", name, ckpt)
        predictors[name] = Predictor(ckpt, aoi_path, cube_root, device=device).load()

    lead = next(iter(predictors.values()))
    registry = lead.registry
    leads = lead.lead_times
    out_cases = []

    for aoi in registry:
        if lakes and aoi.lake_id not in lakes:
            continue
        cube = cube_root / f"{aoi.lake_id}.zarr"
        if not cube.exists():
            logger.info("%s: no cube, skipping", aoi.lake_id)
            continue
        grid = aoi.grid()
        cell_km2 = (grid.resolution / 1000.0) ** 2

        with xr.open_zarr(cube, consolidated=True) as ds:
            water = lead._water_mask(aoi.lake_id, ds).astype(bool)
            times = {pd.Timestamp(t).date(): i for i, t in enumerate(ds.time.values)}
            anchors = find_cases(ds, water, leads, n_cases, lead.input_days)
            logger.info("%s: %d case(s) -- %s", aoi.lake_id, len(anchors), anchors)
            blank = np.full(water.shape, np.nan, dtype=np.float32)

            for anchor in anchors:
                # The observation window exactly as the model receives it: gaps and all.
                window = []
                for k in range(lead.input_days - 1, -1, -1):
                    d = anchor - timedelta(days=k)
                    f = truth_frame(ds, times, d)
                    field = blank if f is None else f
                    window.append(
                        {
                            "date": d.isoformat(),
                            "observed": bool(np.isfinite(field[water]).any()),
                            "image": render_field(field, water),
                            **frame_stats(field, water, cell_km2),
                        }
                    )

                truths = {}
                for ld in leads:
                    f = truth_frame(ds, times, anchor + timedelta(days=ld))
                    if f is not None:
                        truths[ld] = f

                # Persistence is rendered as a model in its own right, so the gallery can
                # answer "did the network beat doing nothing?" by eye and not only by a
                # skill score four sections further down the page.
                pers = persistence_field(ds, times, anchor, lead.input_days, water.shape)
                models = {
                    "Persistence": {
                        "model_version": "baseline",
                        "leads": [
                            {
                                "lead_days": ld,
                                "valid_date": (anchor + timedelta(days=ld)).isoformat(),
                                "prediction": render_field(pers, water),
                                "pred_stats": frame_stats(pers, water, cell_km2),
                                **(
                                    {
                                        "error": render_error(pers, truths[ld], water),
                                        "error_stats": pair_stats(pers, truths[ld], water),
                                    }
                                    if ld in truths
                                    else {}
                                ),
                            }
                            for ld in leads
                        ],
                    }
                }
                for name, pr in predictors.items():
                    result = pr.predict(aoi.lake_id, anchor, allow_live=False)
                    per_lead = []
                    for i, ld in enumerate(result.lead_times):
                        pred = result.log_chl[i]
                        entry = {
                            "lead_days": ld,
                            "valid_date": (anchor + timedelta(days=ld)).isoformat(),
                            "prediction": render_field(pred, water),
                            "pred_stats": frame_stats(pred, water, cell_km2),
                        }
                        t = truths.get(ld)
                        if t is not None:
                            entry["error"] = render_error(pred, t, water)
                            entry["error_stats"] = pair_stats(pred, t, water)
                        per_lead.append(entry)
                    models[name] = {"model_version": result.model_version, "leads": per_lead}

                out_cases.append(
                    {
                        "lake_id": aoi.lake_id,
                        "lake_name": aoi.name,
                        "split": aoi.split,
                        "anchor": anchor.isoformat(),
                        "shape": list(water.shape),
                        "water_km2": float(water.sum() * cell_km2),
                        "observed_days": sum(1 for w in window if w["observed"]),
                        "input_days": lead.input_days,
                        "window": window,
                        "truth": {
                            str(ld): {
                                "date": (anchor + timedelta(days=ld)).isoformat(),
                                "image": render_field(t, water),
                                **frame_stats(t, water, cell_km2),
                            }
                            for ld, t in truths.items()
                        },
                        "models": models,
                    }
                )

    out_cases.sort(key=lambda c: (c["split"] != "test", c["lake_id"], c["anchor"]))
    return {
        "leads": list(leads),
        "scale": {"vmin_mg_m3": 10**VMIN, "vmax_mg_m3": 10**VMAX, "err_max_log10": EMAX},
        "cases": out_cases,
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--aoi", type=Path, default=REPO_ROOT / "data/aoi/lakes.geojson")
    p.add_argument("--cubes", type=Path, default=REPO_ROOT / "data/processed/cubes")
    p.add_argument("--out", type=Path, default=REPO_ROOT / "artifacts/case_studies.json")
    p.add_argument("--lakes", nargs="*", default=None)
    p.add_argument("--cases-per-lake", type=int, default=1)
    p.add_argument("--device", default=None)
    p.add_argument(
        "--checkpoint",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="repeatable; defaults to the U-Net and ConvLSTM run checkpoints",
    )
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    if args.checkpoint:
        cks = {}
        for spec in args.checkpoint:
            name, _, path = spec.partition("=")
            cks[name] = Path(path)
    else:
        cks = {}
        for name, run in (("Anomaly", "anomaly"), ("ConvLSTM", "convlstm"), ("U-Net", "unet")):
            found = sorted((REPO_ROOT / "artifacts" / run / "checkpoints").glob("best-*.ckpt"))
            if found:
                cks[name] = found[-1]
    if not cks:
        raise SystemExit("no checkpoints found -- pass --checkpoint NAME=PATH")

    payload = build(cks, args.aoi, args.cubes, args.lakes, args.cases_per_lake, args.device)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload), encoding="utf-8")
    mb = args.out.stat().st_size / 1e6
    print(f"wrote {args.out}  ({len(payload['cases'])} case(s), {mb:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
