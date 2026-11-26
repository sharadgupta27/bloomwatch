"""Render the README's figures from the project's own outputs.

    python scripts/make_readme_figures.py

Reads only artifacts this repository produces -- the leave-lake-out evaluation
(`artifacts/eval_anomaly_clim_tta/scores.json`), the in-situ matchups
(`artifacts/insitu_validation/`, from `scripts/validate_insitu.py`), one lake's Zarr cube,
and two pages of the sample bulletin -- and writes PNGs to `docs/img/`. Re-run it after
any of those change; nothing in a figure is typed in by hand.

Charts follow one style: white surface, recessive grid, one y-axis per panel, at most two
series (blue = the model or the raw value, orange = the comparison), each labelled
directly so colour is never the only key.
"""

# ruff: noqa: RUF001  (multiplication and minus signs in chart labels are intended)
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

BLUE, ORANGE = "#257de0", "#f47230"  # validated pair (adjacent CVD dE 26.6, normal 35.2)
INK, INK_2, GRID, BAND = "#191919", "#666666", "#e6e6e6", "#eeeeee"
DPI = 160


def _style(ax, ylabel: str | None = None) -> None:
    ax.set_facecolor("white")
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(INK_2)
    ax.tick_params(colors=INK_2, labelsize=9, length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK_2, fontsize=9)


def _title(fig, title: str, sub: str) -> None:
    fig.text(0.012, 0.975, title, ha="left", va="top", fontsize=13, weight="bold", color=INK)
    fig.text(0.012, 0.915, sub, ha="left", va="top", fontsize=9.5, color=INK_2)


def _save(fig, out: Path) -> None:
    fig.savefig(out, dpi=DPI, facecolor="white")
    plt.close(fig)
    print("wrote", out.relative_to(REPO))


# ------------------------------------------------------------------------------------
def skill_by_lead(out_dir: Path) -> None:
    rows = json.loads((REPO / "artifacts/eval_anomaly_clim_tta/scores.json").read_text())
    df = pd.DataFrame(rows)
    names = {"victoria_ne": "Victoria (test, never seen)", "balaton": "Balaton (validation)"}
    colours = {"victoria_ne": BLUE, "balaton": ORANGE}
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.9), sharey=True)
    fig.subplots_adjust(left=0.07, right=0.80, top=0.74, bottom=0.14, wspace=0.12)
    for ax, col, label in (
        (axes[0], "skill_vs_persistence", 'vs persistence ("tomorrow = today")'),
        (axes[1], "skill_vs_climatology", "vs seasonal climatology"),
    ):
        _style(ax, "skill (1 − RMSE / baseline RMSE)" if ax is axes[0] else None)
        ax.axhline(0, color=INK_2, linewidth=1)
        for lake, g in df.groupby("lake_id"):
            g = g.sort_values("lead_days")
            ax.plot(
                g.lead_days,
                g[col],
                color=colours[lake],
                linewidth=2,
                marker="o",
                markersize=6,
                markeredgecolor="white",
                markeredgewidth=1.5,
            )
            last = g.iloc[-1]
            if ax is axes[1]:
                ax.annotate(
                    names[lake],
                    (last.lead_days, last[col]),
                    xytext=(10, 0),
                    textcoords="offset points",
                    va="center",
                    fontsize=9,
                    color=INK,
                )
            else:
                at = g.iloc[1]  # the 3-day point: clear of the y-axis and of the other line
                below = lake == "victoria_ne"
                ax.annotate(
                    names[lake].split(" (")[0],
                    (at.lead_days, at[col]),
                    xytext=(0, -15 if below else 9),
                    textcoords="offset points",
                    ha="center",
                    fontsize=9,
                    color=INK,
                )
        ax.set_title(label, fontsize=10, color=INK, loc="left")
        ax.set_xticks([1, 3, 5, 7], ["1 d", "3 d", "5 d", "7 d"])
        ax.set_xlabel("forecast lead", color=INK_2, fontsize=9)
    axes[0].text(7.2, 0.012, "better than baseline ↑", fontsize=8, color=INK_2, ha="right")
    _title(
        fig,
        "Forecast skill on lakes held out of training",
        "Served model (anomaly + climatology input, flip TTA). Above 0 = beats the baseline.",
    )
    _save(fig, out_dir / "skill_by_lead.png")


# ------------------------------------------------------------------------------------
def insitu_scatter(out_dir: Path) -> None:
    df = pd.read_csv(REPO / "artifacts/insitu_validation/matchups.csv.gz")
    s = df[(df.kind == "satellite") & (df.sat_dt_days == 0) & df.model_log10.notna()]
    lab = s.insitu_mg_m3.to_numpy()
    sat = 10 ** s.model_log10.to_numpy()
    ratio = float(10 ** np.median(np.log10(sat / lab)))
    within = float(np.mean(np.abs(np.log10(sat / lab)) <= np.log10(2)))

    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    fig.subplots_adjust(left=0.13, right=0.96, top=0.82, bottom=0.11)
    _style(ax, "satellite chlorophyll-a, same day (µg/L)")
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    lo, hi = 0.1, 400
    x = np.array([lo, hi])
    ax.fill_between(x, x / 2, x * 2, color=BAND, zorder=0)
    ax.plot(x, x, color=INK_2, linewidth=1, zorder=1)
    ax.plot(x, x * ratio, color=ORANGE, linewidth=2, linestyle=(0, (5, 3)), zorder=3)
    ax.scatter(lab, sat, s=22, color=BLUE, alpha=0.6, edgecolors="white", linewidths=0.6, zorder=2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("laboratory chlorophyll-a (µg/L)", color=INK_2, fontsize=9)
    ax.text(150, 150 * 1.25, "1 : 1", fontsize=8.5, color=INK_2, ha="right", va="bottom")
    ax.text(0.13, 0.13 * 2.9, "within ×2", fontsize=8.5, color=INK_2)
    ax.text(
        350,
        0.14,
        f"dashed: median fit,\nsatellite = {ratio:.2f} × lab",
        fontsize=9,
        color=INK,
        ha="right",
        va="bottom",
    )
    for t in (10, 20, 50):
        ax.axvline(t, color=GRID, linewidth=0.8, linestyle=":")
    _title(
        fig,
        "Satellite vs laboratory chlorophyll-a",
        f"{len(s)} same-day matchups, {s.lake_id.nunique()} lakes (GLORIA + US EPA WQP). "
        f"Within ×2: {within:.0%}.",
    )
    _save(fig, out_dir / "insitu_scatter.png")


# ------------------------------------------------------------------------------------
def insitu_detection(out_dir: Path) -> None:
    summary = json.loads((REPO / "artifacts/insitu_validation/summary.json").read_text())
    exp = summary["bias_correction_experiment"]["forecast_lead_1"]
    thresholds = ["10", "20", "50"]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.9), sharey=True)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.72, bottom=0.14, wspace=0.12)
    width = 0.36
    xs = np.arange(len(thresholds))
    for ax, key, label in (
        (axes[0], "pod", "Hit rate — blooms caught"),
        (axes[1], "far", "False-alarm ratio — flags that were wrong"),
    ):
        _style(ax, "share" if ax is axes[0] else None)
        for off, (name, colour) in zip(
            (-width / 2, width / 2),
            (("raw", BLUE), ("corrected", ORANGE)),
            strict=True,
        ):
            vals = [exp[name].get(f"{key}_{t}") for t in thresholds]
            heights = [v if v is not None else 0 for v in vals]
            ax.bar(xs + off, heights, width, color=colour, edgecolor="white", linewidth=2)
            for xi, v in zip(xs + off, vals, strict=True):
                ax.text(
                    xi,
                    (v or 0) + 0.02,
                    "n/a" if v is None else f"{v:.0%}",
                    ha="center",
                    va="bottom",
                    fontsize=8.5,
                    color=INK,
                )
        ax.set_xticks(xs, [f"≥ {t} µg/L" for t in thresholds])
        ax.set_ylim(0, 1)
        ax.set_title(label, fontsize=10, color=INK, loc="left")
    # legend: a key is required for two series; direct labels sit under the first group
    from matplotlib.patches import Patch

    axes[1].legend(
        handles=[
            Patch(color=BLUE, label="forecast as served"),
            Patch(
                color=ORANGE, label=f"with a ×{exp['median_factor']:.1f} correction (not applied)"
            ),
        ],
        frameon=False,
        fontsize=8.5,
        loc="upper left",
        labelcolor=INK,
    )
    _title(
        fig,
        "1-day forecast vs field samples: detecting a bloom",
        "Correction fitted leave-one-lake-out. It catches far more blooms and raises false "
        "alarms — a public-health trade-off, so it is not applied.",
    )
    _save(fig, out_dir / "insitu_detection.png")


# ------------------------------------------------------------------------------------
def missing_data(out_dir: Path, lake: str = "erie_west", year: int = 2019) -> None:
    import xarray as xr

    ds = xr.open_zarr(REPO / f"data/processed/cubes_global_live/{lake}.zarr", consolidated=True)
    masks = sorted((REPO / "data/cache/statics").glob(f"water_{lake}_*.npy"))
    sel = ds["log_chl"].sel(time=slice(f"{year}-05-01", f"{year}-10-31"))
    vals = sel.values
    # the predictor's cached mask when present; otherwise the same 5% rule on this season
    water = np.load(masks[-1]) > 0 if masks else np.isfinite(vals).mean(axis=0) >= 0.05
    frac = pd.Series(
        np.isfinite(vals[:, water]).mean(axis=1), index=pd.to_datetime(sel.time.values)
    )
    days = pd.date_range(f"{year}-05-01", f"{year}-10-31", freq="D")
    frac = frac.reindex(days, fill_value=0.0)

    fig, ax = plt.subplots(figsize=(9.6, 3.3))
    fig.subplots_adjust(left=0.07, right=0.98, top=0.72, bottom=0.14)
    _style(ax, "share of lake observed")
    ax.bar(frac.index, frac.values, width=0.9, color=BLUE)
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 0.5, 1], ["0%", "50%", "100%"])
    ax.set_xlim(days[0] - pd.Timedelta(days=1), days[-1] + pd.Timedelta(days=1))
    import matplotlib.dates as mdates

    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    empty = float((frac == 0).mean())
    _title(
        fig,
        f"Most of the data is missing: Lake Erie, bloom season {year}",
        f"Daily share of the lake's water pixels with a usable retrieval. "
        f"{empty:.0%} of days have none at all (cloud, glint, no overpass).",
    )
    _save(fig, out_dir / "missing_data.png")


# ------------------------------------------------------------------------------------
def bulletin_pages(out_dir: Path, anchor: str = "2026-09-27") -> None:
    from PIL import Image, ImageChops

    src = REPO / "artifacts/bulletin" / anchor / "png"
    for name, out in (
        ("taihu_forecast.png", "bulletin_forecast.png"),
        ("erie_west_imagery.png", "bulletin_imagery.png"),
    ):
        img = Image.open(src / name).convert("RGB")
        bg = Image.new("RGB", img.size, (255, 255, 255))
        box = ImageChops.difference(img, bg).getbbox()
        if box:
            pad = 16
            box = (
                max(0, box[0] - pad),
                max(0, box[1] - pad),
                min(img.width, box[2] + pad),
                min(img.height, box[3] + pad),
            )
            img = img.crop(box)
        img.save(out_dir / out, optimize=True)
        print("wrote", (out_dir / out).relative_to(REPO))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(REPO / "docs/img"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    skill_by_lead(out)
    insitu_scatter(out)
    insitu_detection(out)
    missing_data(out)
    bulletin_pages(out)


if __name__ == "__main__":
    main()
