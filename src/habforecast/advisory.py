"""The advisory feed: forecasts turned into something a water manager can subscribe to.

A forecast field is the model's output; an advisory is the product. This module turns one
`ForecastResult` per lake into two delivery surfaces the catalogue asks for:

- a **GeoJSON FeatureCollection** -- one feature per lake, the lake's footprint as the
  geometry, and the forecast's operational content as properties: the advisory level, the
  first lead at which each threshold is crossed, the predicted bloom area per lead, and
  the provenance (which input days were observed, which source served the window, which
  checkpoint ran). A GIS or a dashboard can consume it directly.
- an **RSS 2.0 feed** -- one item per lake, so the same content reaches an inbox, a Slack
  channel or a phone through any feed reader, with no client code at all.

Both are pure functions of the forecast results, so the API endpoint and the scheduled
job (`pipelines/advisory.py`) render exactly the same thing.

The advisory level is a deliberately simple rule on the *fraction of the lake* above each
chlorophyll threshold, because a fraction is what an operator can act on and what a
threshold crossing at a single 300 m pixel is not. The thresholds are the project's
standard 10 / 20 / 50 mg/m^3 (see eval.metrics); the fraction is configurable. Every
property the level was computed from is carried alongside it, so a consumer with a
different appetite for false alarms can apply their own rule to the same numbers.

**With an exceedance head** (a checkpoint trained with `exceed_thresholds`, or an ensemble
of them) the fractions come from the probabilities instead of from thresholding the map,
and the advisory carries its own uncertainty:

- the *expected* fraction of water above a threshold is the lake mean of p(exceed) -- a
  calibrated quantity when p is (the evaluation reports the reliability that says so) --
  and it sets the level, exactly as the thresholded fraction does without a head;
- the *likely* fraction counts pixels with p >= 0.7 and the *possible* fraction pixels with
  p >= 0.3. The level each of those would give brackets the answer. When they agree the
  advisory is `confident`; when they do not it is `uncertain`, and it names the level the
  lake could reach (`possible_level`) instead of pretending to a hard one.

A feed that says "97% of the lake above 20 mg/m^3" now says how sure it is of that.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import UTC, date, datetime, timedelta
from email.utils import format_datetime
from xml.sax.saxutils import escape

import numpy as np

from habforecast.aoi import LakeAOI
from habforecast.eval.metrics import BLOOM_THRESHOLDS_MG_M3
from habforecast.inference.predictor import ForecastResult

# Fraction of the lake's water that must exceed a threshold for that threshold to count.
DEFAULT_MIN_FRACTION = 0.10
# Probability cut-offs for the likely / possible band around the expected fraction.
LIKELY_PROB = 0.7
POSSIBLE_PROB = 0.3

# Advisory levels, in ascending severity, and the threshold that triggers each.
LEVELS: tuple[tuple[str, float | None], ...] = (
    ("none", None),
    ("watch", 10.0),
    ("warning", 20.0),
    ("alert", 50.0),
)

DISCLAIMER = (
    "Advisory only. Satellite chlorophyll-a is not a direct cyanotoxin measurement; confirm "
    "with in-situ sampling before any public health action."
)


@dataclass(frozen=True)
class Advisory:
    """One lake's advisory, derived from its forecast."""

    lake_id: str
    lake_name: str
    anchor_date: date
    level: str
    # Per threshold (as "10", "20", "50"): the first lead in days at which the fraction
    # of the lake above it reaches `min_fraction`, or None if it never does.
    first_exceedance_lead: dict[str, int | None]
    # Per threshold: the largest fraction of the lake above it across all leads.
    peak_fraction: dict[str, float]
    # Lead at which the level was reached.
    level_lead_days: int | None
    summary: dict
    min_fraction: float
    # Probabilistic advisories only (a forecast with exceedance probabilities); None
    # otherwise, and the feed then carries the deterministic rule unchanged.
    confidence: str | None = None  # "confident" | "uncertain"
    likely_level: str | None = None
    possible_level: str | None = None
    # Per threshold: the largest likely / possible fraction across leads.
    fraction_band: dict[str, dict[str, float]] | None = None

    @property
    def level_date(self) -> date | None:
        if self.level_lead_days is None:
            return None
        return self.anchor_date + timedelta(days=self.level_lead_days)


def _fractions(result: ForecastResult, cell_area_km2: float) -> dict[str, list[float]]:
    """Fraction of water above each threshold, per lead."""
    water_km2 = float(result.water.sum() * cell_area_km2)
    out: dict[str, list[float]] = {}
    for thr in BLOOM_THRESHOLDS_MG_M3:
        areas = result.bloom_area_km2(thr, cell_area_km2)
        out[f"{thr:g}"] = [a / water_km2 if water_km2 > 0 else 0.0 for a in areas]
    return out


def with_probabilities(map_result: ForecastResult, prob_result: ForecastResult) -> ForecastResult:
    """The map from one model, the exceedance probabilities from another.

    The served split since 2026-09-23: `best.ckpt` draws the chlorophyll map (its
    regression is the stronger one) and `models/ens_met` -- ERA5 forcing plus the
    exceedance head, three seeds, recalibrated -- supplies p(exceed), which is the better
    ranking of where a threshold will be crossed (docs/MODEL_CARD.md). Both are run on the
    same lake grid and anchor, so the fields line up pixel for pixel; a mismatch is refused
    rather than silently misregistered.
    """
    if prob_result.exceed_prob is None:
        return map_result
    if prob_result.water.shape != map_result.water.shape or prob_result.anchor_date != (
        map_result.anchor_date
    ):
        raise ValueError(
            f"{map_result.lake_id}: probability forecast does not match the map "
            f"({prob_result.water.shape} @ {prob_result.anchor_date} vs "
            f"{map_result.water.shape} @ {map_result.anchor_date})"
        )
    prob = np.where(map_result.water[None, None], prob_result.exceed_prob, np.nan)
    return replace(
        map_result,
        exceed_prob=prob.astype(np.float32),
        exceed_thresholds=prob_result.exceed_thresholds,
        spread=prob_result.spread,
        ensemble_size=prob_result.ensemble_size,
        model_version=f"{map_result.model_version} (map) + {prob_result.model_version} (p)",
    )


def _prob_fractions(result: ForecastResult) -> dict[str, dict[str, list[float]]] | None:
    """Expected / likely / possible fraction of water above each threshold, per lead."""
    if result.exceed_prob is None:
        return None
    water = result.water
    n = float(water.sum())
    out: dict[str, dict[str, list[float]]] = {"expected": {}, "likely": {}, "possible": {}}
    for thr in BLOOM_THRESHOLDS_MG_M3:
        prob = result.exceedance(thr)
        key = f"{thr:g}"
        if prob is None or n == 0:
            for kind in out:
                out[kind][key] = [0.0] * len(result.lead_times)
            continue
        p = np.nan_to_num(prob[:, water], nan=0.0)  # (L, n_water)
        out["expected"][key] = [float(v) for v in p.mean(axis=1)]
        out["likely"][key] = [float(v) for v in (p >= LIKELY_PROB).mean(axis=1)]
        out["possible"][key] = [float(v) for v in (p >= POSSIBLE_PROB).mean(axis=1)]
    return out


def _level(
    fractions: dict[str, list[float]], lead_times, min_fraction: float
) -> tuple[str, int | None, dict[str, int | None]]:
    """Highest level whose threshold is reached, the lead it is reached at, and the first
    lead each threshold is reached at."""
    first: dict[str, int | None] = {}
    for key, per_lead in fractions.items():
        hit = [lead for lead, f in zip(lead_times, per_lead, strict=True) if f >= min_fraction]
        first[key] = min(hit) if hit else None
    level, level_lead = "none", None
    for name, thr in LEVELS:
        if thr is None:
            continue
        lead = first.get(f"{thr:g}")
        if lead is not None:
            level, level_lead = name, lead
    return level, level_lead, first


def advise(
    result: ForecastResult,
    cell_area_km2: float,
    min_fraction: float = DEFAULT_MIN_FRACTION,
) -> Advisory:
    """Derive the advisory for one lake from its forecast."""
    prob = _prob_fractions(result)
    fractions = prob["expected"] if prob is not None else _fractions(result, cell_area_km2)
    peak = {key: (max(v) if v else 0.0) for key, v in fractions.items()}
    level, level_lead, first = _level(fractions, result.lead_times, min_fraction)

    extra: dict = {}
    if prob is not None:
        likely, _, _ = _level(prob["likely"], result.lead_times, min_fraction)
        possible, _, _ = _level(prob["possible"], result.lead_times, min_fraction)
        extra = {
            "confidence": "confident" if likely == possible else "uncertain",
            "likely_level": likely,
            "possible_level": possible,
            "fraction_band": {
                key: {
                    "likely": max(prob["likely"][key]) if prob["likely"][key] else 0.0,
                    "possible": max(prob["possible"][key]) if prob["possible"][key] else 0.0,
                }
                for key in fractions
            },
        }

    return Advisory(
        lake_id=result.lake_id,
        lake_name=result.lake_name,
        anchor_date=result.anchor_date,
        level=level,
        first_exceedance_lead=first,
        peak_fraction=peak,
        level_lead_days=level_lead,
        summary=result.summary(cell_area_km2),
        min_fraction=min_fraction,
        **extra,
    )


def _bbox_polygon(aoi: LakeAOI) -> dict:
    west, south, east, north = aoi.bbox
    return {
        "type": "Polygon",
        "coordinates": [
            [[west, south], [east, south], [east, north], [west, north], [west, south]]
        ],
    }


def feature(advisory: Advisory, aoi: LakeAOI) -> dict:
    """A GeoJSON Feature for one lake's advisory."""
    props = {
        "lake_id": advisory.lake_id,
        "name": advisory.lake_name,
        "country": aoi.country,
        "status": "ok",
        "level": advisory.level,
        "level_lead_days": advisory.level_lead_days,
        "level_date": advisory.level_date.isoformat() if advisory.level_date else None,
        "min_fraction": advisory.min_fraction,
        "first_exceedance_lead_days": advisory.first_exceedance_lead,
        "peak_fraction_above": advisory.peak_fraction,
        "anchor_date": advisory.anchor_date.isoformat(),
        "input_observed_days": advisory.summary["input_observed_days"],
        "n_observed_days": len(advisory.summary["input_observed_days"]),
        "source": advisory.summary["source"],
        "model_version": advisory.summary["model_version"],
        "water_area_km2": advisory.summary["water_area_km2"],
        "forecasts": advisory.summary["forecasts"],
        "ensemble_size": advisory.summary.get("ensemble_size", 1),
        # "probabilistic" when the level came from exceedance probabilities; the band and
        # confidence fields below are present only then.
        "basis": "probabilistic" if advisory.confidence else "deterministic",
        "disclaimer": DISCLAIMER,
    }
    if advisory.confidence:
        props.update(
            {
                "confidence": advisory.confidence,
                "likely_level": advisory.likely_level,
                "possible_level": advisory.possible_level,
                "fraction_band": advisory.fraction_band,
                "prob_cutoffs": {"likely": LIKELY_PROB, "possible": POSSIBLE_PROB},
            }
        )
    return {
        "type": "Feature",
        "id": advisory.lake_id,
        "geometry": _bbox_polygon(aoi),
        "properties": props,
    }


def unavailable_feature(aoi: LakeAOI, anchor: date, reason: str) -> dict:
    """A lake the forecast could not be made for -- persistent cloud, usually.

    The feed carries it rather than dropping it: "no forecast for Lake Erie today" is
    information, and a subscriber who stops seeing a lake would otherwise not know why.
    """
    return {
        "type": "Feature",
        "id": aoi.lake_id,
        "geometry": _bbox_polygon(aoi),
        "properties": {
            "lake_id": aoi.lake_id,
            "name": aoi.name,
            "country": aoi.country,
            "status": "unavailable",
            "level": "unavailable",
            "reason": reason,
            "anchor_date": anchor.isoformat(),
            "disclaimer": DISCLAIMER,
        },
    }


def collection(features: list[dict], generated: datetime | None = None) -> dict:
    generated = generated or datetime.now(UTC)
    return {
        "type": "FeatureCollection",
        "generated_utc": generated.isoformat(timespec="seconds"),
        "levels": [name for name, _ in LEVELS],
        "features": features,
    }


def _headline(props: dict) -> str:
    name = props["name"]
    if props.get("status") != "ok":
        return f"{name}: no forecast ({props.get('reason', 'unavailable')})"
    level = props["level"]
    if level == "none" and props.get("confidence") == "uncertain":
        return (
            f"{name}: no advisory — UNCERTAIN, could reach "
            f"{props['possible_level'].upper()} within 7 d"
        )
    if level == "none":
        return f"{name}: no advisory (below {props['min_fraction']:.0%} of lake at 10 mg/m³ through 7 d)"
    thr = next(t for n, t in LEVELS if n == level)
    frac = props["peak_fraction_above"][f"{thr:g}"]
    when = props["level_date"]
    if props.get("basis") == "probabilistic":
        band = props["fraction_band"][f"{thr:g}"]
        head = (
            f"{name}: {level.upper()} — expected {frac:.0%} of lake above {thr:g} mg/m³ by "
            f"{when} (+{props['level_lead_days']} d; likely {band['likely']:.0%}, "
            f"possibly {band['possible']:.0%})"
        )
        if props["confidence"] == "uncertain" and props["possible_level"] != level:
            head += f" — UNCERTAIN, could reach {props['possible_level'].upper()}"
        elif props["confidence"] == "uncertain":
            head += " — UNCERTAIN"
        return head
    return (
        f"{name}: {level.upper()} — {frac:.0%} of lake above {thr:g} mg/m³ by {when} "
        f"(+{props['level_lead_days']} d)"
    )


def _describe(props: dict) -> str:
    if props.get("status") != "ok":
        return f"{props.get('reason', 'unavailable')}. {DISCLAIMER}"
    lines = []
    for f in props["forecasts"]:
        areas = f["bloom_area_km2"]
        med = f["median_chl_mg_m3"]
        lines.append(
            f"+{f['lead_days']} d ({f['valid_date']}): median "
            f"{med:.1f} mg/m³, above 20 mg/m³ over {areas['20']:.0f} km², "
            f"above 50 mg/m³ over {areas['50']:.0f} km²"
            if med is not None
            else f"+{f['lead_days']} d ({f['valid_date']}): no prediction"
        )
    drift = props.get("drift") or {}
    if drift.get("any_drift"):
        lines.append(
            "DATA DRIFT: this lake's inputs are unlike its archive for the season ("
            + ", ".join(drift.get("drifted_metrics", []))
            + ") -- review before acting."
        )
    obs = props["n_observed_days"]
    lines.append(
        f"Built from {obs} observed day{'s' if obs != 1 else ''} in the input window "
        f"({props['source']}), checkpoint {props['model_version']}."
    )
    lines.append(DISCLAIMER)
    return " ".join(lines)


def rss(fc: dict, base_url: str = "", title: str = "habforecast bloom advisories") -> str:
    """Render a FeatureCollection as an RSS 2.0 feed, one item per lake."""
    generated = datetime.fromisoformat(fc["generated_utc"])
    if generated.tzinfo is None:
        generated = generated.replace(tzinfo=UTC)
    link = f"{base_url}/app" if base_url else "/app"
    items = []
    for feat in fc["features"]:
        p = feat["properties"]
        guid = f"{p['lake_id']}:{p['anchor_date']}:{p['level']}"
        lake_link = (
            f"{base_url}/app?lake={p['lake_id']}" if base_url else f"/app?lake={p['lake_id']}"
        )
        items.append(
            "    <item>\n"
            f"      <title>{escape(_headline(p))}</title>\n"
            f"      <link>{escape(lake_link)}</link>\n"
            f'      <guid isPermaLink="false">{escape(guid)}</guid>\n'
            f"      <pubDate>{format_datetime(generated)}</pubDate>\n"
            f"      <category>{escape(p['level'])}</category>\n"
            f"      <description>{escape(_describe(p))}</description>\n"
            "    </item>"
        )
    body = "\n".join(items)
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<rss version="2.0">\n'
        "  <channel>\n"
        f"    <title>{escape(title)}</title>\n"
        f"    <link>{escape(link)}</link>\n"
        "    <description>Short-range cyanobacterial bloom advisories from Sentinel-3 OLCI. "
        f"{escape(DISCLAIMER)}</description>\n"
        f"    <lastBuildDate>{format_datetime(generated)}</lastBuildDate>\n"
        f"{body}\n"
        "  </channel>\n"
        "</rss>\n"
    )


def build(
    predictor,
    anchor: date | None = None,
    lake_ids: list[str] | None = None,
    allow_live: bool = False,
    min_fraction: float = DEFAULT_MIN_FRACTION,
    results: dict | None = None,
    prob_predictor=None,
) -> dict:
    """Run every registry lake through the predictor and assemble the collection.

    A lake that cannot be forecast (no usable observations in its window) becomes an
    `unavailable` feature rather than an exception: the feed for four lakes must not fail
    because the fifth was under cloud all fortnight.
    """
    anchor = anchor or datetime.now(UTC).date()
    ids = lake_ids or predictor.registry.lake_ids
    feats = []
    for lake_id in ids:
        aoi = predictor.registry[lake_id]
        try:
            result = predictor.predict(lake_id, anchor, allow_live=allow_live)
            if prob_predictor is not None:
                result = with_probabilities(
                    result, prob_predictor.predict(lake_id, anchor, allow_live=allow_live)
                )
        except ValueError as exc:
            feats.append(unavailable_feature(aoi, anchor, str(exc)))
            continue
        if results is not None:
            # The raw forecasts, for a caller that checks more than the feed carries
            # (the daily job's input drift check).
            results[lake_id] = result
        cell_km2 = (aoi.grid().resolution / 1000.0) ** 2
        feats.append(feature(advise(result, cell_km2, min_fraction), aoi))
    return collection(feats)
