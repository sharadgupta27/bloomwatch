"""HAB bulletin pages: NCCOS-style forecast products for any forecast, as PDF pages.

NOAA NCCOS publishes, for Lake Erie only, the latest satellite image of bloom density, a
bloom forecast, a wind-induced mixing forecast and a surface-scum risk curve. This module
draws the same set from any `ForecastResult` and the imagery it was made from, so it
serves both callers that need it:

- `scripts/make_bulletin.py` -- the daily bulletin for every registry lake, one PDF;
- the app's `/api/forecast/jobs/{id}/bulletin.pdf` -- a bulletin for whatever area a user
  drew on the map (`area_bulletin_pdf`).

Figures are built with the object-oriented `matplotlib.figure.Figure` API and never touch
`pyplot`: the API renders in a thread pool, and pyplot's global figure registry is not
thread-safe.

**What differs from NCCOS**, and is said on every page it applies to: their bloom forecast
advects the observed bloom with a 3-D hydrodynamic model (LEOFS), ours is a statistical
forecast of chlorophyll-a; their density is the Cyanobacteria Index, ours chlorophyll-a,
which does not separate cyanobacteria from other algae; their mixing and scum-risk schemes
are calibrated for Erie, ours are the transparent wind heuristic in the constants below.
"""

from __future__ import annotations

import io
import json
import logging
import math
import time
import urllib.parse
import urllib.request
import warnings
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------- styling --
PAGE = (11.69, 8.27)  # A4 landscape, inches
LAND = np.array([0.85, 0.85, 0.85, 1.0])
NO_DATA = np.array([0.55, 0.55, 0.55, 1.0])  # water under cloud / not observed
ABSENT = np.array([0.0, 0.0, 0.0, 1.0])  # water observed, below the lowest threshold
# Chlorophyll-a colour scale: log10 mg m^-3 from the "watch" threshold to 200.
CHL_MIN, CHL_MAX = 10.0, 200.0
CHL_TICKS = (10, 20, 50, 100, 200)
CHL_CMAP_NAME = "turbo"
PROB_CMAP_NAME = "magma_r"
LEVEL_COLOURS = {
    "none": "#4d9221",
    "watch": "#e6c229",
    "warning": "#f17105",
    "alert": "#b10026",
    "unavailable": "#8c8c8c",
}
LEVEL_RANK = {"alert": 0, "warning": 1, "watch": 2, "none": 3, "unavailable": 4}
OLCI_SOURCE = (
    "Sentinel-3 OLCI 300 m chlorophyll-a (Copernicus / EUMETSAT, processed by Bloom Watch)"
)

# Mixing classes on 10 m wind speed (knots). NCCOS: "Blooms mix through the water column at
# wind speeds > 15 knots"; the lower breaks are this project's reading of their legend.
MIXING = (
    ("Weak mixing", 0.0, "#5b7f00"),
    ("Slight mixing", 5.0, "#000000"),
    ("Mixing possible", 10.0, "#ff6600"),
    ("Mixing likely", 15.0, "#8b0000"),
)
# Surface-scum risk from the 6-hourly mean wind (knots): calm water lets buoyant
# cyanobacteria accumulate at the surface. Upper bounds of High / Medium / Low.
SCUM_WIND_BREAKS = (4.0, 7.0, 10.0)
SCUM_LEVELS = ("V.Low", "Low", "Medium", "High")
# ...gated on the bloom: the forecast share of water above 20 mg m^-3 below which the risk
# is capped at V.Low, and below which it is capped at Low.
SCUM_GATE_NONE, SCUM_GATE_LOW = 0.005, 0.02
# The wind page shows 5 days past and 4 ahead of *now*; for an older issue date it would
# describe different weather from the forecast's, so it is left out beyond this age.
WIND_MAX_ANCHOR_AGE_DAYS = 3

OPEN_METEO_FORECAST = "https://api.open-meteo.com/v1/forecast"


def _cmap(name: str):
    import matplotlib

    return matplotlib.colormaps[name]


def new_page():
    from matplotlib.figure import Figure

    return Figure(figsize=PAGE)


# ----------------------------------------------------------------------- geography --
@dataclass(frozen=True)
class Geo:
    """Where a grid sits: the lon/lat extent imshow needs and the latitude for aspect."""

    extent: tuple[float, float, float, float]  # west, east, south, north
    lat0: float

    @classmethod
    def from_lonlat(cls, lon: np.ndarray, lat: np.ndarray) -> Geo:
        return cls(
            (
                float(np.nanmin(lon)),
                float(np.nanmax(lon)),
                float(np.nanmin(lat)),
                float(np.nanmax(lat)),
            ),
            float(np.nanmean(lat)),
        )

    @property
    def centre(self) -> tuple[float, float]:
        west, east, south, north = self.extent
        return (south + north) / 2.0, (west + east) / 2.0


# ------------------------------------------------------------------------ imagery --
def recent_imagery(
    days: list[date], frames: list[np.ndarray], water: np.ndarray, anchor: date, window: int = 7
) -> dict:
    """Most recent frame with >= 5 % of the water seen, and the max composite of `window`.

    `days` / `frames` are the observed daily log10-chlorophyll fields in any order (NaN
    where not retrieved); frames after `anchor` are ignored.
    """
    start = anchor - timedelta(days=window - 1)
    picked = sorted(
        ((d, f) for d, f in zip(days, frames, strict=True) if start <= d <= anchor),
        key=lambda df: df[0],
    )
    out = {
        "latest": None,
        "latest_date": None,
        "latest_cover": 0.0,
        "composite": None,
        "composite_days": [],
        "window_days": window,
    }
    if not picked:
        return out
    n_water = max(int(water.sum()), 1)
    cover = [float((np.isfinite(f) & water).sum()) / n_water for _, f in picked]
    for (d, f), c in zip(reversed(picked), reversed(cover), strict=True):
        if c >= 0.05:
            out.update(latest=f, latest_date=d.isoformat(), latest_cover=c)
            break
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN pixels stay NaN
        out["composite"] = np.nanmax(np.stack([f for _, f in picked]), axis=0)
    out["composite_days"] = [
        d.isoformat() for (d, _), c in zip(picked, cover, strict=True) if c > 0
    ]
    return out


# --------------------------------------------------------------------------- wind --
def fetch_wind(lat: float, lon: float, cache: Path | None = None, attempts: int = 4) -> dict | None:
    """Hourly 10 m wind (kn), 5 days past and 5 ahead of now, in local time; None on failure.

    Open-Meteo's best-match models; the "past" part is model analysis, not a station.
    """
    if cache is not None and cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    q = urllib.parse.urlencode(
        {
            "latitude": f"{lat:.4f}",
            "longitude": f"{lon:.4f}",
            "hourly": "wind_speed_10m,wind_direction_10m",
            "wind_speed_unit": "kn",
            "past_days": 5,
            "forecast_days": 5,
            "timezone": "auto",
        }
    )
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(f"{OPEN_METEO_FORECAST}?{q}", timeout=30) as r:
                payload = json.loads(r.read())
            if cache is not None:
                cache.write_text(json.dumps(payload), encoding="utf-8")
            return payload
        except Exception as exc:  # the wind page is optional; the bulletin is not
            logger.warning(
                "Open-Meteo wind (%.2f, %.2f) attempt %d: %s", lat, lon, attempt + 1, exc
            )
            if attempt + 1 < attempts:
                time.sleep(5 * (attempt + 1))
    return None


def mixing_class(speed_kn: float) -> int:
    k = 0
    for i, (_, lo, _) in enumerate(MIXING):
        if speed_kn >= lo:
            k = i
    return k


def scum_risk(speed_kn: float, bloom_share: float) -> int:
    """0 (V.Low) .. 3 (High) from the wind, capped when little or no bloom is forecast."""
    lo, mid, hi = SCUM_WIND_BREAKS
    risk = 3 if speed_kn < lo else 2 if speed_kn < mid else 1 if speed_kn < hi else 0
    if bloom_share < SCUM_GATE_NONE:
        return 0
    if bloom_share < SCUM_GATE_LOW:
        return min(risk, 1)
    return risk


def bloom_share(result) -> float:
    """Share of water expected above 20 mg m^-3 on the first lead; gates the scum risk."""
    p20 = result.exceedance(20.0)
    if p20 is not None:
        return float(np.nanmean(p20[0][result.water]))
    return result.bloom_area_km2(20.0, 1.0)[0] / max(int(result.water.sum()), 1)


# ------------------------------------------------------------------------- levels --
def banded(props: dict) -> dict:
    """The advisory properties with the level held inside its likely..possible band.

    `habforecast.advisory` takes the level from the *expected* fraction of water above each
    threshold (the lake mean of p). A probability model that puts a low floor -- 0.1-0.2 --
    on every pixel of a lake unlike its training lakes reaches a 10 % expected fraction with
    no pixel even possibly (p >= 0.3) above the threshold, so the level lands above its own
    `possible_level`. The advisory module documents the band as bracketing the answer; the
    bulletin enforces that and keeps the feed's level as `feed_level` so nothing is hidden.
    The feed itself is unchanged.
    """
    likely, possible = props.get("likely_level"), props.get("possible_level")
    if not possible:
        return props
    from habforecast.advisory import LEVELS

    order = [name for name, _ in LEVELS]
    level = props["level"]
    clamped = level
    if order.index(level) > order.index(possible):
        clamped = possible
    elif likely and order.index(level) < order.index(likely):
        clamped = likely
    if clamped == level:
        return props
    # The feed's level date belongs to the feed's level; the band carries no lead.
    return {**props, "level": clamped, "level_date": None, "feed_level": level}


def headline(props: dict) -> str:
    level = props["level"]
    text = f"Advisory level: {level.upper()}"
    if props.get("level_date"):
        text += f" from {props['level_date']}"
    elif level != "none":
        text += " within 7 days"
    possible = props.get("possible_level")
    if props.get("feed_level"):
        text += f"  (feed rule: {props['feed_level'].upper()} from diffuse low probabilities)"
    elif props.get("confidence") == "uncertain" and possible and possible != level:
        text += f"  (uncertain; could reach {possible.upper()})"
    elif props.get("confidence"):
        text += f"  ({props['confidence']})"
    return text


# ---------------------------------------------------------------------- page parts --
def chl_rgba(log_chl: np.ndarray, water: np.ndarray) -> np.ndarray:
    """RGBA image: land, water without data, water below 10 mg m^-3, and the colour scale."""
    img = np.broadcast_to(LAND, (*water.shape, 4)).copy()
    img[water] = NO_DATA
    seen = water & np.isfinite(log_chl)
    img[seen] = ABSENT
    lo, hi = math.log10(CHL_MIN), math.log10(CHL_MAX)
    bloom = seen & (log_chl >= lo)
    img[bloom] = _cmap(CHL_CMAP_NAME)(np.clip((log_chl[bloom] - lo) / (hi - lo), 0, 1))
    return img


def prob_rgba(prob: np.ndarray, water: np.ndarray) -> np.ndarray:
    img = np.broadcast_to(LAND, (*water.shape, 4)).copy()
    ok = water & np.isfinite(prob)
    img[water] = NO_DATA
    img[ok] = _cmap(PROB_CMAP_NAME)(np.clip(prob[ok], 0, 1))
    return img


def show_map(ax, img: np.ndarray, geo: Geo, title: str, small: bool = False) -> None:
    ax.imshow(img, extent=geo.extent, origin="upper", interpolation="nearest")
    ax.set_aspect(1.0 / math.cos(math.radians(geo.lat0)))
    ax.set_title(title, fontsize=8 if small else 10)
    ax.tick_params(labelsize=6 if small else 7)
    if small:
        ax.set_xticks([])
        ax.set_yticks([])


def _scalar_mappable(vmin: float, vmax: float, cmap_name: str):
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    return ScalarMappable(norm=Normalize(vmin, vmax), cmap=_cmap(cmap_name))


def chl_colorbar(fig, cax, label: str = "Chlorophyll-a (mg m$^{-3}$)") -> None:
    sm = _scalar_mappable(math.log10(CHL_MIN), math.log10(CHL_MAX), CHL_CMAP_NAME)
    cb = fig.colorbar(sm, cax=cax)
    cb.set_ticks([math.log10(t) for t in CHL_TICKS], labels=[str(t) for t in CHL_TICKS])
    cb.ax.tick_params(labelsize=7)
    cb.set_label(label, fontsize=8)
    cax.text(
        1.0,
        -0.04,
        "black: < 10\ngrey: no data",
        transform=cax.transAxes,
        fontsize=6,
        va="top",
        ha="right",
    )


def footer(fig, text: str) -> None:
    fig.text(0.01, 0.008, text, fontsize=6, color="#555555", ha="left", va="bottom")


def header(fig, title: str, subtitle: str = "") -> None:
    fig.text(0.5, 0.965, title, fontsize=15, ha="center", va="top", weight="bold")
    if subtitle:
        fig.text(0.5, 0.925, subtitle, fontsize=8.5, ha="center", va="top", wrap=True)


def level_box(fig, props: dict, y: float = 0.885, fontsize: float = 11) -> None:
    fig.text(
        0.5,
        y,
        headline(props),
        ha="center",
        fontsize=fontsize,
        weight="bold",
        color="white",
        bbox={
            "boxstyle": "round,pad=0.35",
            "fc": LEVEL_COLOURS.get(props["level"], "#555"),
            "ec": "none",
        },
    )


# ---------------------------------------------------------------------------- pages --
def page_imagery(
    props: dict, geo: Geo, water: np.ndarray, img: dict, foot: str, source: str = OLCI_SOURCE
):
    fig = new_page()
    header(
        fig,
        f"{props['name']} — Satellite Imagery",
        f"{source}. Chlorophyll-a is a proxy for bloom biomass; it does not separate "
        "cyanobacteria from other algae.",
    )
    ax1 = fig.add_axes([0.04, 0.1, 0.42, 0.76])
    ax2 = fig.add_axes([0.5, 0.1, 0.42, 0.76])
    cax = fig.add_axes([0.94, 0.2, 0.015, 0.56])
    empty = np.full(water.shape, np.nan, dtype=np.float32)
    if img["latest"] is not None:
        t1 = f"Most recent scene {img['latest_date']}  ({img['latest_cover']:.0%} of water seen)"
        show_map(ax1, chl_rgba(img["latest"], water), geo, t1)
    else:
        show_map(
            ax1,
            chl_rgba(empty, water),
            geo,
            f"No usable scene in the last {img['window_days']} days (cloud or no data)",
        )
    comp = img["composite"] if img["composite"] is not None else empty
    show_map(
        ax2,
        chl_rgba(comp, water),
        geo,
        f"{img['window_days']}-day maximum composite to {props['anchor_date']} "
        f"({len(img['composite_days'])} scenes)",
    )
    chl_colorbar(fig, cax)
    footer(fig, foot)
    return fig


def page_forecast(props: dict, geo: Geo, result, foot: str):
    fig = new_page()
    observed = props.get("input_observed_days") or []
    header(
        fig,
        f"{props['name']} — Bloom Forecast",
        f"Issued {props['anchor_date']} from {len(observed)} observed day(s) in the 14-day "
        f"window{f' (last {observed[-1]})' if observed else ''}. Statistical forecast of "
        "chlorophyll-a (no hydrodynamic transport).",
    )
    level_box(fig, props)
    leads = result.lead_times
    w = 0.84 / len(leads)
    prob20 = result.exceedance(20.0)
    for i, lead in enumerate(leads):
        valid = (result.anchor_date + timedelta(days=lead)).isoformat()
        ax = fig.add_axes([0.04 + i * w, 0.47, w - 0.01, 0.36])
        show_map(
            ax,
            chl_rgba(result.log_chl[i], result.water),
            geo,
            f"Day +{lead}  ({valid})",
            small=True,
        )
        if prob20 is not None:
            ax = fig.add_axes([0.04 + i * w, 0.1, w - 0.01, 0.36])
            show_map(
                ax, prob_rgba(prob20[i], result.water), geo, f"P(chl > 20)  day +{lead}", small=True
            )
    chl_colorbar(fig, fig.add_axes([0.9, 0.5, 0.012, 0.3]), "Forecast chl-a (mg m$^{-3}$)")
    if prob20 is not None:
        cb = fig.colorbar(
            _scalar_mappable(0, 1, PROB_CMAP_NAME), cax=fig.add_axes([0.9, 0.13, 0.012, 0.3])
        )
        cb.set_label("P(chl-a > 20 mg m$^{-3}$)", fontsize=8)
        cb.ax.tick_params(labelsize=7)
    else:
        fig.text(
            0.46,
            0.28,
            "This model has no exceedance-probability head:\n"
            "only the chlorophyll forecast is available.",
            ha="center",
            fontsize=9,
            color="#555555",
        )
    rows = []
    for fc in props["forecasts"]:
        a = fc["bloom_area_km2"]
        e = fc.get("exceed_probability", {}).get("20", {})
        rows.append(
            f"+{fc['lead_days']}d: >10 {a['10']:.0f} km², >20 {a['20']:.0f} km², "
            f">50 {a['50']:.0f} km²"
            + (f", expected >20 {e['expected_area_km2']:.0f} km²" if e else "")
        )
    rows.append(f"water area {props['water_area_km2']:.0f} km²")
    fig.text(
        0.04,
        0.075,
        "Forecast bloom area  " + "   |   ".join(rows[:2]) + "\n" + "   |   ".join(rows[2:]),
        fontsize=7,
        va="top",
        linespacing=1.6,
    )
    footer(fig, foot)
    return fig


def page_wind(
    props: dict,
    wind: dict,
    anchor: date,
    image_date: str | None,
    share: float,
    foot: str,
    where: str = "lake centre",
):
    import matplotlib.dates as mdates

    h = wind["hourly"]
    t = np.array([datetime.fromisoformat(s) for s in h["time"]])
    spd = np.array([np.nan if v is None else v for v in h["wind_speed_10m"]], dtype=float)
    wdir = np.array([np.nan if v is None else v for v in h["wind_direction_10m"]], dtype=float)
    now = datetime.now(UTC).replace(tzinfo=None) + timedelta(
        seconds=wind.get("utc_offset_seconds", 0)
    )
    past = t <= now
    tz = wind.get("timezone_abbreviation", "local")

    fig = new_page()
    header(
        fig,
        f"{props['name']} — Wind & Mixing Forecast",
        f"Hourly 10 m wind at the {where} ({wind['latitude']:.2f}, {wind['longitude']:.2f}) "
        "from Open-Meteo (past = model analysis, not a station record). Arrows point downwind.\n"
        "Blooms mix through the water column at wind speeds > 15 knots.",
    )
    ax = fig.add_axes([0.06, 0.14, 0.62, 0.7])
    ax.plot(t[past], spd[past], "-", color="#333333", lw=2.2, label="Past (analysis)")
    ax.plot(t[~past], spd[~past], "--", color="#111111", lw=2.2, label="Forecast")
    sel = np.arange(0, len(t), 3)
    sel = sel[np.isfinite(spd[sel]) & np.isfinite(wdir[sel])]
    toward = np.radians(wdir[sel] + 180.0)
    cls = np.array([mixing_class(s) for s in spd[sel]])
    for k, (name, _, colour) in enumerate(MIXING):
        m = cls == k
        if m.any():
            ax.quiver(
                mdates.date2num(t[sel][m]),
                spd[sel][m],
                np.sin(toward[m]),
                np.cos(toward[m]),
                color=colour,
                angles="uv",
                scale=28,
                width=0.004,
                pivot="tail",
                zorder=3,
            )
        ax.plot([], [], marker=r"$\rightarrow$", ms=14, ls="", color=colour, label=name)
    if image_date:
        it = datetime.fromisoformat(image_date) + timedelta(hours=11)
        ax.axvline(it, color="blue", ls="--", lw=2)
        ax.text(it, 0.5, " Image time", rotation=90, color="blue", va="bottom", fontsize=9)
    ax.axvline(now, color="#777777", ls=":", lw=1.2)
    top = max(20.0, float(np.nanmax(spd)) + 5 if np.isfinite(spd).any() else 20.0)
    ax.set_ylim(0, top)
    ax.axhline(15, color="#8b0000", lw=0.8, ls=":")
    ax.set_ylabel("Speed in knots", fontsize=11, weight="bold")
    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.grid(True, ls="--", lw=0.5)
    ax.legend(loc="upper left", fontsize=8, ncol=2, framealpha=0.9)
    ax.set_xlabel(f"Time ({tz})")

    # Scum risk: 6-hourly mean speed from the issue date forward.
    ax2 = fig.add_axes([0.75, 0.14, 0.22, 0.7])
    start = datetime.combine(anchor, datetime.min.time())
    pts_t, pts_r = [], []
    for k in range(4 * 5):
        a, b = start + timedelta(hours=6 * k), start + timedelta(hours=6 * (k + 1))
        m = (t >= a) & (t < b) & np.isfinite(spd)
        if m.any():
            pts_t.append(a + timedelta(hours=3))
            pts_r.append(scum_risk(float(spd[m].mean()), share))
    for i, c in enumerate(["#d9efe9", "#8fd4bd", "#5aab73", "#44775a"]):
        ax2.axhspan(i - 0.5, i + 0.5, color=c)
    ax2.plot(pts_t, pts_r, "-o", color="black", ms=3, lw=1.2)
    ax2.set_ylim(-0.5, 3.5)
    ax2.set_yticks(range(4), SCUM_LEVELS)
    ax2.xaxis.set_major_locator(mdates.DayLocator())
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%d\n%b"))
    ax2.tick_params(labelsize=8)
    ax2.set_title("Forecast Surface-Scum Risk", fontsize=10)
    ax2.set_xlabel(f"Time ({tz})", fontsize=8)
    gate = (
        "no bloom forecast -> capped at V.Low"
        if share < SCUM_GATE_NONE
        else "small bloom -> capped at Low"
        if share < SCUM_GATE_LOW
        else "bloom present"
    )
    fig.text(
        0.75,
        0.075,
        "Calm-wind heuristic on the 6-h mean wind:\n"
        f"<{SCUM_WIND_BREAKS[0]:g} kn High, <{SCUM_WIND_BREAKS[1]:g} Medium, "
        f"<{SCUM_WIND_BREAKS[2]:g} Low, else V.Low.\n"
        f"Gated on the bloom: {share:.1%} of water\n"
        f"expected > 20 mg m$^{{-3}}$ at day +1:\n{gate}.",
        fontsize=6.5,
        va="top",
    )
    footer(fig, foot)
    return fig


def forecast_pages(
    props: dict,
    geo: Geo,
    result,
    img: dict,
    wind: dict | None,
    foot: str,
    source: str = OLCI_SOURCE,
    where: str = "lake centre",
) -> list:
    """The three product pages for one area: imagery, forecast, wind (when there is wind)."""
    pages = [
        ("imagery", page_imagery(props, geo, result.water, img, foot, source)),
        ("forecast", page_forecast(props, geo, result, foot)),
    ]
    if wind is not None:
        pages.append(
            (
                "wind",
                page_wind(
                    props,
                    wind,
                    result.anchor_date,
                    img["latest_date"],
                    bloom_share(result),
                    foot,
                    where,
                ),
            )
        )
    return pages


def write_pdf(figures: list, target, title: str) -> None:
    """Save figures, in order, as one PDF to a path or a binary file object."""
    from matplotlib.backends.backend_pdf import PdfPages

    with PdfPages(target) as pdf:
        for fig in figures:
            pdf.savefig(fig)
        info = pdf.infodict()
        info["Title"] = title
        info["Author"] = "Bloom Watch"


# --------------------------------------------------------------- drawn-area bulletin --
def area_summary_page(props: dict, facts: list[tuple[str, str]], caveats: list[str], foot: str):
    """The first page of a drawn-area bulletin: the level, the provenance, the caveats."""
    fig = new_page()
    header(
        fig,
        f"Bloom Watch — Area Bulletin: {props['name']}",
        f"Issued {datetime.now(UTC).date().isoformat()} for a user-drawn area, forecast anchored "
        f"on {props['anchor_date']}.",
    )
    level_box(fig, props, y=0.85, fontsize=12)
    ax = fig.add_axes([0.06, 0.42, 0.88, 0.36])
    ax.axis("off")
    tab = ax.table(
        cellText=[[k, v] for k, v in facts],
        loc="upper center",
        cellLoc="left",
        colWidths=[0.25, 0.75],
    )
    tab.auto_set_font_size(False)
    tab.set_fontsize(8)
    tab.scale(1, 1.35)
    notes = caveats or ["None reported."]
    fig.text(0.06, 0.38, "Caveats for this forecast", fontsize=10, weight="bold", va="top")
    fig.text(
        0.06,
        0.345,
        "\n".join(f"•  {c}" for c in notes),
        fontsize=7.5,
        va="top",
        wrap=True,
        linespacing=1.5,
    )
    fig.text(
        0.06,
        0.1,
        "Levels: WATCH/WARNING/ALERT when >= 10 % of the area's water is forecast above "
        "10/20/50 mg m^-3 chlorophyll-a within 7 days. Statistical forecast of chlorophyll-a, "
        "not a hydrodynamic bloom-transport model,\nnot a cyanobacteria index or toxin product. "
        "Mixing classes and scum risk are a documented wind heuristic, not NCCOS's calibrated "
        "scheme.",
        fontsize=7,
        va="top",
        color="#333333",
    )
    footer(fig, foot)
    return fig


def area_bulletin_pdf(res, wind: dict | None = None, fetch_wind_live: bool = True) -> bytes:
    """A bulletin PDF for an ad-hoc forecast (`habforecast.inference.adhoc.AdhocResult`).

    Same pages as the registry bulletin, preceded by a summary page that carries the
    route and every caveat the app shows -- a PDF travels without the app around it, so
    it must not drop the warnings. The wind page is fetched live unless `wind` is given,
    and is left out when the forecast's anchor is older than a few days (it would describe
    different weather) or the fetch fails; the summary page says which.
    """
    from habforecast import advisory

    result = res.forecast
    resolution = res.route.resolution_m if res.route else None
    grid = res.aoi.grid(resolution) if resolution else res.aoi.grid()
    lon, lat = grid.lonlat
    geo = Geo.from_lonlat(lon, lat)

    adv = advisory.advise(result, res.cell_area_km2)
    props = banded(advisory.feature(adv, res.aoi)["properties"])
    days = sorted(res.history)
    img = recent_imagery(days, [res.history[d] for d in days], result.water, result.anchor_date)

    wind_note = "shown"
    age = (datetime.now(UTC).date() - result.anchor_date).days
    if wind is None and age > WIND_MAX_ANCHOR_AGE_DAYS:
        wind_note = (
            f"omitted: the forecast is anchored {age} days ago and the wind record covers "
            "only the last 5 days"
        )
    elif wind is None and fetch_wind_live:
        wind = fetch_wind(*geo.centre, attempts=2)
        if wind is None:
            wind_note = "omitted: the Open-Meteo wind service did not answer"
    elif wind is None:
        wind_note = "omitted"

    source = getattr(res, "source_label", "") or OLCI_SOURCE
    west, south, east, north = res.aoi.bbox
    facts = [
        ("Area (W, S, E, N)", f"{west:.3f}, {south:.3f}, {east:.3f}, {north:.3f}"),
        (
            "Water mapped",
            f"{res.water_area_km2:.0f} km² ({res.water_pixels} cells, "
            f"{res.water_confidence:.0%} mean coverage in the window)",
        ),
        ("Imagery source", source),
        ("Observed days", ", ".join(d.isoformat() for d in res.observed_days) or "none"),
        ("Scenes", f"{res.scenes_used} used of {res.scenes_found} found"),
        ("Model", result.model_version),
        *_route_facts(res.route),
        ("Map % > 20 mg m^-3", f"{map_share(props):.1%} (largest across leads)"),
        ("Wind & mixing page", wind_note),
    ]
    foot = f"Bloom Watch · area bulletin · {result.model_version} · " + advisory.DISCLAIMER
    figures = [area_summary_page(props, facts, res.caveats(), foot)]
    figures += [
        f for _, f in forecast_pages(props, geo, result, img, wind, foot, source, "area centre")
    ]
    buf = io.BytesIO()
    write_pdf(figures, buf, f"Bloom Watch Area Bulletin {props['name']} {props['anchor_date']}")
    return buf.getvalue()


def _route_facts(route) -> list[tuple[str, str]]:
    """What `habforecast.routing` decided for this area, for the summary table."""
    if route is None:
        return [("Route", "not recorded")]
    water = route.water
    kind = water.setting + (f", {water.optical_class}" if water.optical_class != "unknown" else "")
    step = route.sources[0]
    model = (
        f"{route.model.label} ({route.model.train_bodies} training water bodies)"
        if route.model
        else f"no learned model defensible here -- baseline: {route.baseline}"
    )
    return [
        ("Water body", kind),
        (
            "Product",
            step.label
            if f"{step.resolution_m:g} m" in step.label
            else f"{step.label} at {step.resolution_m:g} m",
        ),
        ("Model choice", model),
    ]


def map_share(props: dict) -> float:
    """Largest share of water above 20 mg m^-3 on the chlorophyll map, across leads."""
    return max(f["bloom_area_km2"]["20"] for f in props["forecasts"]) / max(
        props["water_area_km2"], 1e-9
    )
