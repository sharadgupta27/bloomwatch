"""The public outlook: one plain-language risk statement per monitored lake.

The advisory feed (`habforecast.advisory`) is written for agencies -- thresholds, lead
times, fractions of water above a chlorophyll level. Someone deciding whether to take the
dog to the lake on Saturday needs three things from the same numbers: how worried to be,
what to do about it, and how old the information is. This module is only that
translation. It adds no new thresholds and makes no new prediction: the level is the
advisory's own, held inside its likely..possible band exactly as the bulletin does
(`bulletin.banded`), so the map, the PDF and the feed can never disagree.

Two honesty rules are enforced here rather than left to the page:

* **Age is part of the answer.** Each lake is forecast from the newest imagery its
  archive holds, which can be weeks old if the live refresh has not run. A level from
  stale imagery is still returned (it is the best information there is) but flagged
  `stale`, and the page says so before it says anything else.
* **Chlorophyll is not toxin.** Every entry carries the same caveat, and the advice is
  the general "when in doubt, stay out" guidance public-health agencies give for any
  visible bloom -- never "safe to swim", because a satellite cannot say that.
"""

from __future__ import annotations

from datetime import date

# Imagery older than this is reported as out of date. A bloom can form or disperse within
# days; the model's own longest lead is 7, so a forecast issued from a fortnight-old
# window is describing a lake that has moved on.
STALE_AFTER_DAYS = 10

CAVEAT = (
    "Estimated from satellite images of chlorophyll (algae), not a toxin test. Blooms can "
    "form or move within hours. Local health advisories and posted signs always take "
    "precedence."
)

GENERAL_ADVICE = (
    "Whatever the level: if water looks green, blue-green, brown or red, has scum, foam or "
    "a paint-like film, or smells bad, stay out and keep pets out.",
    "Seek medical (or veterinary) advice if someone becomes unwell after contact, and "
    "mention the bloom.",
)

# Ascending severity, keyed by advisory.LEVELS names. `threshold` is the advisory's own
# chlorophyll trigger (mg/m3 = ug/L), stated so the page can show where the words come from.
PUBLIC_LEVELS: dict[str, dict] = {
    "none": {
        "rank": 0,
        "label": "Low",
        "threshold": None,
        "headline": "No widespread algal bloom is expected.",
        "advice": [
            "Usual water-safety precautions apply.",
            "Small, local blooms near shore can still occur — check before you go in.",
        ],
    },
    "watch": {
        "rank": 1,
        "label": "Elevated",
        "threshold": 10.0,
        "headline": "Algae levels are elevated in parts of the lake.",
        "advice": [
            "Avoid swallowing lake water.",
            "Rinse off with clean water after swimming or paddling.",
            "Don't let dogs swim in or drink from discoloured or scummy water.",
        ],
    },
    "warning": {
        "rank": 2,
        "label": "Bloom likely",
        "threshold": 20.0,
        "headline": "A bloom is likely over a substantial part of the lake.",
        "advice": [
            "Avoid swimming and water contact where the water is green or scummy.",
            "Keep children and pets out of the water and away from the shoreline scum.",
            "Don't drink, cook with or give animals untreated lake water.",
        ],
    },
    "alert": {
        "rank": 3,
        "label": "Severe bloom",
        "threshold": 50.0,
        "headline": "A dense bloom is likely. Toxins are more likely at this level.",
        "advice": [
            "Stay out of the water near visible scum or discoloured water.",
            "Keep pets and livestock away — dogs can be poisoned by licking their fur.",
            "Do not drink, cook with or irrigate with untreated lake water.",
            "Check your local health authority for closures and drinking-water notices.",
        ],
    },
    "unavailable": {
        "rank": -1,
        "label": "No data",
        "threshold": None,
        "headline": "There is not enough recent cloud-free imagery to forecast this lake.",
        "advice": ["Check local advisories, and stay out of water that looks or smells bad."],
    },
}


def latest_anchor(days, today: date) -> date | None:
    """The newest archived day at or before `today`, from a cube's calendar index."""
    usable = [d for d in days if d <= today]
    return max(usable) if usable else None


def held_in_band(props: dict) -> dict:
    """`bulletin.banded` without importing the bulletin (which needs matplotlib)."""
    likely, possible = props.get("likely_level"), props.get("possible_level")
    order = [
        k for k, v in sorted(PUBLIC_LEVELS.items(), key=lambda kv: kv[1]["rank"]) if v["rank"] >= 0
    ]
    level = props.get("level")
    if not possible or level not in order or possible not in order:
        return props
    clamped = level
    if order.index(level) > order.index(possible):
        clamped = possible
    elif likely in order and order.index(level) < order.index(likely):
        clamped = likely
    if clamped == level:
        return props
    return {**props, "level": clamped, "feed_level": level}


def entry(
    props: dict, centre: tuple[float, float], today: date, validation: dict | None = None
) -> dict:
    """One lake's public outlook from its advisory feature properties.

    `centre` is (lat, lon) for the map marker. `validation`, when given, is this lake's
    row from the in-situ validation summary, passed through so the page can say how the
    satellite compared with field samples *here*.
    """
    props = held_in_band(props)
    level = props.get("level", "unavailable")
    if level not in PUBLIC_LEVELS:
        level = "unavailable"
    spec = PUBLIC_LEVELS[level]
    anchor = props.get("anchor_date")
    age = (today - date.fromisoformat(anchor)).days if anchor and level != "unavailable" else None
    out = {
        "lake_id": props.get("lake_id"),
        "name": props.get("name"),
        "country": props.get("country", ""),
        "lat": round(centre[0], 4),
        "lon": round(centre[1], 4),
        "level": level,
        "rank": spec["rank"],
        "label": spec["label"],
        "headline": spec["headline"],
        "advice": list(spec["advice"]),
        "threshold_mg_m3": spec["threshold"],
        "imagery_date": anchor if level != "unavailable" else None,
        "imagery_age_days": age,
        "stale": bool(age is not None and age > STALE_AFTER_DAYS),
        "level_date": props.get("level_date"),
        "confidence": props.get("confidence"),
        "reason": props.get("reason"),
    }
    if "feed_level" in props:
        out["feed_level"] = props["feed_level"]
    peak = props.get("peak_fraction_above") or {}
    if spec["threshold"] is not None:
        out["share_of_lake"] = round(float(peak.get(f"{spec['threshold']:g}", 0.0)), 3)
    if validation:
        out["validation"] = validation
    return out


def summary(entries: list[dict], today: date) -> dict:
    """The whole outlook, most severe first, with the shared caveat and advice."""
    ordered = sorted(entries, key=lambda e: (-e["rank"], e["name"] or ""))
    return {
        "generated": today.isoformat(),
        "stale_after_days": STALE_AFTER_DAYS,
        "levels": {
            k: {"label": v["label"], "rank": v["rank"], "threshold_mg_m3": v["threshold"]}
            for k, v in PUBLIC_LEVELS.items()
        },
        "caveat": CAVEAT,
        "general_advice": list(GENERAL_ADVICE),
        "lakes": ordered,
    }
