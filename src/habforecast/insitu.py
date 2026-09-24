"""In-situ measurements near a point, so a forecast can be read beside a bottle sample.

Everything else this project reports is satellite-against-satellite: the forecast is
scored against persistence and climatology, all three derived from the same retrieval. A
user clicking a pixel and seeing 34 mg/m3 has no way to tell whether that number has ever
been checked against water. This module answers "has anyone measured this lake, and what
did they find", from the public archives surveyed in `docs/VALIDATION_DATA.md`.

The index is a flat table built offline by `scripts/build_insitu_index.py` -- tens of
thousands of rows, small enough to hold in a few numpy arrays and scan per query, and
therefore with no service to be down and no key to leak. `HAB_INSITU` points at it.

Two decisions worth stating, because both could be made dishonestly:

* **Chlorophyll and microcystin are kept apart.** They are different quantities in
  different units, and the reason for carrying the toxin at all is that chlorophyll is
  only a proxy for it. Nothing here averages one into the other.
* **A same-week matchup is rare, and pretending otherwise would be the whole bug.**
  GLORIA ends in 2022 and most monitoring is sparse, so a forecast anchored on today will
  usually have nothing inside its own fortnight. `context()` therefore returns in-window
  points and same-season-other-year points in *separate* lists, each labelled with the
  date it actually came from.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import os
from dataclasses import dataclass
from datetime import date, timedelta
from functools import lru_cache
from pathlib import Path

import numpy as np

DEFAULT_INDEX = Path(os.getenv("HAB_INSITU", "data/validation/insitu.csv.gz"))

CHLOROPHYLL = "chlorophyll_a"
MICROCYSTIN = "microcystin"
UNITS = {CHLOROPHYLL: "mg/m3", MICROCYSTIN: "ug/L"}

# WHO recreational guidance value for microcystin-LR. Above it a water body is a public
# health question rather than a limnological one.
WHO_MICROCYSTIN_UGL = 8.0

COLUMNS = ["source", "site", "variable", "date", "lat", "lon", "value", "flag"]

# Most other-year samples a single chart carries before the marks stop being readable.
SEASONAL_LIMIT = 60

EARTH_RADIUS_KM = 6371.0088


@dataclass(frozen=True)
class Observation:
    """One measurement someone made in the water, with its distance from the query."""

    source: str
    site: str
    variable: str
    day: date
    lat: float
    lon: float
    value: float
    flag: str
    distance_km: float

    @property
    def unit(self) -> str:
        return UNITS.get(self.variable, "")

    @property
    def non_detect(self) -> bool:
        """A reported absence. Real information: it says the toxin was not there."""
        return self.flag == "nd"

    @property
    def exceeds_who(self) -> bool:
        return self.variable == MICROCYSTIN and self.value >= WHO_MICROCYSTIN_UGL

    def as_dict(self) -> dict:
        out = {
            "source": self.source,
            "site": self.site,
            "variable": self.variable,
            "unit": self.unit,
            "date": self.day.isoformat(),
            "lat": round(self.lat, 5),
            "lon": round(self.lon, 5),
            "value": round(self.value, 3),
            "distance_km": round(self.distance_km, 2),
        }
        if self.flag:
            out["flag"] = self.flag
        if self.exceeds_who:
            out["exceeds_who"] = True
        return out


def haversine_km(lat: float, lon: float, lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    """Great-circle distance from one point to many, in kilometres."""
    p = np.pi / 180.0
    dlat = (lats - lat) * p
    dlon = (lons - lon) * p
    a = np.sin(dlat / 2) ** 2 + np.cos(lat * p) * np.cos(lats * p) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


class InSituIndex:
    """A flat table of georeferenced measurements, scanned by proximity."""

    def __init__(self, rows: list[dict]):
        self.source = np.array([r["source"] for r in rows], dtype=object)
        self.site = np.array([r["site"] for r in rows], dtype=object)
        self.variable = np.array([r["variable"] for r in rows], dtype=object)
        self.flag = np.array([r.get("flag") or "" for r in rows], dtype=object)
        self.lat = np.array([float(r["lat"]) for r in rows], dtype=np.float64)
        self.lon = np.array([float(r["lon"]) for r in rows], dtype=np.float64)
        self.value = np.array([float(r["value"]) for r in rows], dtype=np.float64)
        self.day = np.array([date.fromisoformat(r["date"]) for r in rows], dtype=object)

    def __len__(self) -> int:
        return int(self.lat.size)

    @classmethod
    def from_csv(cls, path: Path) -> InSituIndex:
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt", encoding="utf-8", newline="") as fh:
            return cls(list(csv.DictReader(fh)))

    def near(
        self,
        lat: float,
        lon: float,
        radius_km: float,
        variable: str | None = None,
    ) -> list[Observation]:
        """Every measurement within `radius_km`, nearest first.

        A degree-box prefilter comes first because this runs on a click: it drops the
        great-circle arithmetic to the handful of rows that could possibly qualify. The
        box is deliberately generous -- it only has to be a superset -- and the exact
        distance test behind it is what decides membership.
        """
        if not len(self):
            return []
        dlat = radius_km / 111.19
        cos = max(float(np.cos(np.radians(lat))), 1e-6)
        dlon = min(180.0, radius_km / (111.19 * cos))
        box = (
            (self.lat >= lat - dlat)
            & (self.lat <= lat + dlat)
            # Wrapped, so a query beside the antimeridian still sees the other side of it.
            & (np.abs(((self.lon - lon + 180.0) % 360.0) - 180.0) <= dlon)
        )
        if variable is not None:
            box &= self.variable == variable
        idx = np.flatnonzero(box)
        if not idx.size:
            return []
        dist = haversine_km(lat, lon, self.lat[idx], self.lon[idx])
        keep = dist <= radius_km
        idx, dist = idx[keep], dist[keep]
        order = np.argsort(dist, kind="stable")
        return [
            Observation(
                source=str(self.source[i]),
                site=str(self.site[i]),
                variable=str(self.variable[i]),
                day=self.day[i],
                lat=float(self.lat[i]),
                lon=float(self.lon[i]),
                value=float(self.value[i]),
                flag=str(self.flag[i]),
                distance_km=float(d),
            )
            for i, d in zip(idx[order], dist[order], strict=True)
        ]


@lru_cache(maxsize=4)
def load_index(path: Path | None = None) -> InSituIndex | None:
    """The index, or None when it has not been built. Never raises on a missing file.

    A missing index is the normal state of a fresh checkout -- the underlying archives are
    ~100 MB of third-party data that is not ours to redistribute -- so the app degrades to
    "no validation data here" rather than to a 500.
    """
    p = Path(path) if path else DEFAULT_INDEX
    if not p.exists():
        return None
    return InSituIndex.from_csv(p)


def _doy_offset(day: date, reference: date) -> int:
    """Signed day-of-year distance from `reference`, wrapped across the new year.

    A sample on 2 January is nine days from 24 December, not three hundred and fifty-six.
    """
    raw = (day.timetuple().tm_yday - reference.timetuple().tm_yday) % 365
    return raw - 365 if raw > 182 else raw


def context(
    lat: float,
    lon: float,
    start: date,
    end: date,
    radius_km: float = 10.0,
    season_days: int = 10,
    index: InSituIndex | None = None,
    path: Path | None = None,
) -> dict:
    """What in-situ data exists near a clicked pixel, split by how comparable it is.

    `window` is the only list that is a true matchup: measurements taken near this point
    while the plotted window was running. `seasonal` is the same place at the same time of
    year in some other year -- context for what this water normally does in this week, and
    not verification of anything. They are returned separately, and a caller that merges
    them is making a claim the data does not support.
    """
    idx = index if index is not None else load_index(path)
    if idx is None:
        return {
            "available": False,
            "radius_km": radius_km,
            "note": (
                "No in-situ index is installed. Build one with python scripts/build_insitu_index.py"
            ),
        }

    hits = idx.near(lat, lon, radius_km)
    span = (end - start).days
    window, seasonal = [], []
    for obs in hits:
        if start <= obs.day <= end:
            window.append(obs.as_dict())
            continue
        off = _doy_offset(obs.day, start)
        if -season_days <= off <= span + season_days:
            row = obs.as_dict()
            row["day_offset"] = off
            row["plotted_date"] = (start + timedelta(days=off)).isoformat()
            seasonal.append(row)

    by_variable: dict[str, dict] = {}
    for obs in hits:
        slot = by_variable.setdefault(
            obs.variable,
            {"variable": obs.variable, "unit": obs.unit, "n": 0, "first": None, "last": None},
        )
        slot["n"] += 1
        iso = obs.day.isoformat()
        slot["first"] = iso if slot["first"] is None else min(slot["first"], iso)
        slot["last"] = iso if slot["last"] is None else max(slot["last"], iso)

    return {
        "available": True,
        "radius_km": radius_km,
        "total": len(hits),
        "nearest_km": round(hits[0].distance_km, 2) if hits else None,
        "sources": sorted({o.source for o in hits}),
        "variables": sorted(by_variable.values(), key=lambda v: -v["n"]),
        "window": window,
        # Seasonal context is unbounded in principle -- a well-sampled station can carry a
        # decade of Augusts -- and a 300 px chart cannot separate more marks than this.
        # `near()` returns nearest first, so the cut keeps the samples closest to the
        # clicked pixel, which are also the ones that say most about it.
        "seasonal": seasonal[:SEASONAL_LIMIT],
        "seasonal_total": len(seasonal),
        "note": _note(hits, window, seasonal, radius_km),
    }


def _plural(n: int, one: str, many: str) -> str:
    """These strings are read by a user in a panel, not by a developer in a log."""
    return f"{n} {one}" if n == 1 else f"{n} {many}"


def _note(hits: list, window: list, seasonal: list, radius_km: float) -> str:
    if not hits:
        return f"No public in-situ measurement within {radius_km:g} km of this point."
    if window:
        n = _plural(len(window), "in-situ measurement falls", "in-situ measurements fall")
        return f"{n} inside this window."
    if seasonal:
        n = _plural(len(seasonal), "sample was", "samples were")
        return (
            f"No in-situ measurement inside this window. {n} taken near this point at the "
            "same time of year in other years, drawn hollow for context — they do not "
            "verify this forecast."
        )
    n = _plural(len(hits), "in-situ measurement", "in-situ measurements")
    return f"{n} within {radius_km:g} km, but none in this week of the year."


# --------------------------------------------------------------- validation sites ---
# A site earns a place in the catalogue by having enough measurements to say something.
# Below this a "site" is one boat trip, and offering it as somewhere to go and compare
# oversells what is there.
MIN_SITE_SAMPLES = 8
# A name that spans more than this is not one water body: "Lake Michigan" is legitimately
# 500 km long, but a station name reused by two states is not a place. Sites wider than
# this are kept and flagged rather than dropped, because the big ones are real.
WIDE_SITE_KM = 400.0
# Two same-named groups closer than this are one water body split by a cell boundary.
MERGE_RADIUS_KM = 60.0
# Names that identify nothing. Kept out rather than shown as places to go and compare.
PLACEHOLDER_NAMES = frozenset({"unnamed", "unknown", "na", "n/a", "none", "null", "site", "lake"})
# The app refuses an area of interest wider than this (habforecast.aoi.MAX_ADHOC_SIDE_KM),
# so a sea gets a workable box at its centre rather than a request that cannot be served.
MAX_AOI_SIDE_KM = 80.0
# Built a little inside the limit rather than exactly on it. A box computed to be 80.0 km
# across arrives at the validator as 80.0000001 and is refused, which presents as the
# forecast button staying dead after a perfectly good click.
AOI_SAFETY = 0.95


def _span_km(west: float, south: float, east: float, north: float) -> float:
    mid = (south + north) / 2.0
    return max(
        haversine_km(mid, west, np.array([mid]), np.array([east]))[0],
        haversine_km(south, (west + east) / 2, np.array([north]), np.array([(west + east) / 2]))[0],
    )


def _forecast_box(west, south, east, north) -> tuple[list[float], bool]:
    """The bbox to forecast on, shrunk to the app's limit if the water body is bigger.

    Returned with a flag rather than silently: a user who clicks "Baltic Sea" and gets an
    80 km square should be told that is what happened.
    """
    if _span_km(west, south, east, north) <= MAX_AOI_SIDE_KM:
        return [west, south, east, north], False
    clat, clon = (south + north) / 2.0, (west + east) / 2.0
    half = MAX_AOI_SIDE_KM * AOI_SAFETY / 2.0
    dlat = half / 111.19
    dlon = half / (111.19 * max(np.cos(np.radians(clat)), 1e-6))
    return [clon - dlon, clat - dlat, clon + dlon, clat + dlat], True


def _summarise(rows_idx: np.ndarray, index: InSituIndex) -> dict:
    by_var: dict[str, int] = {}
    for v in index.variable[rows_idx]:
        by_var[str(v)] = by_var.get(str(v), 0) + 1
    days = [d.isoformat() for d in index.day[rows_idx]]
    return {
        "n": int(rows_idx.size),
        "chlorophyll_a": by_var.get(CHLOROPHYLL, 0),
        "microcystin": by_var.get(MICROCYSTIN, 0),
        "sources": sorted({str(s) for s in index.source[rows_idx]}),
        "first": min(days),
        "last": max(days),
    }


def _registry_sites(aoi_path: Path, index: InSituIndex) -> list[dict]:
    """The lakes this model is actually built around, with whatever has been measured in them."""
    try:
        doc = json.loads(Path(aoi_path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    out = []
    for f in doc.get("features", []):
        ring = f["geometry"]["coordinates"][0]
        lons = [p[0] for p in ring]
        lats = [p[1] for p in ring]
        west, south, east, north = min(lons), min(lats), max(lons), max(lats)
        inside = np.flatnonzero(
            (index.lat >= south) & (index.lat <= north) & (index.lon >= west) & (index.lon <= east)
        )
        props = f.get("properties", {})
        box, clipped = _forecast_box(west, south, east, north)
        site = {
            "id": props.get("lake_id", ""),
            "name": props.get("name", props.get("lake_id", "")),
            "region": props.get("country", ""),
            "kind": "modelled",
            "split": props.get("split", ""),
            "bbox": [west, south, east, north],
            "forecast_bbox": box,
            "clipped_to_limit": clipped,
            "centre": [round((south + north) / 2, 5), round((west + east) / 2, 5)],
            "forecast_span_km": round(_span_km(*box), 1),
            "span_km": round(_span_km(west, south, east, north), 1),
        }
        site.update(
            _summarise(inside, index)
            if inside.size
            else {
                "n": 0,
                "chlorophyll_a": 0,
                "microcystin": 0,
                "sources": [],
                "first": None,
                "last": None,
            }
        )
        out.append(site)
    return out


def _named_sites(index: InSituIndex, taken: list[dict]) -> list[dict]:
    """Water bodies the measurements themselves name, outside the registry.

    GLORIA carries a `Site_name` per sample -- "Lake Peipsi", "Baltic Sea", "Curonian
    Lagoon" -- which is how seas and coastal waters enter a catalogue that would otherwise
    be twenty-one lakes.

    Grouping by that name alone does not work, and the failure is not subtle. The United
    States has a Silver Lake, a Round Lake and a Cedar Lake in most states, so one name
    collects points 400 km apart into a "site" that is nowhere; the same grouping turns
    every Atlantic sample into one entry spanning an ocean. So points are binned to a
    degree cell *and* a name, neighbouring cells sharing a name are merged, and any name
    that survives in more than one place is qualified by its coordinates -- which is how
    two Silver Lakes stay two lakes and a stretch of the Atlantic says which stretch.
    """
    boxes = [s["bbox"] for s in taken]
    groups: dict[tuple, list[int]] = {}
    for i, raw in enumerate(index.site):
        label = str(raw).strip()
        if len(label) < 3 or label.strip("()-_ ").lower() in PLACEHOLDER_NAMES:
            continue
        key = (label.casefold(), round(float(index.lat[i])), round(float(index.lon[i])))
        groups.setdefault(key, []).append(i)

    # A lake on a cell boundary is split across two keys; put it back together, but only
    # where the pieces are close enough to be the same water.
    merged: list[tuple[str, list[int]]] = []
    for (label, _la, _lo), rows in sorted(groups.items()):
        for i, (name, have) in enumerate(merged):
            if name.casefold() != label:
                continue
            if _haversine_centres(index, have, rows) <= MERGE_RADIUS_KM:
                merged[i] = (name, have + rows)
                break
        else:
            merged.append((str(index.site[rows[0]]).strip(), rows))

    seen_names: dict[str, int] = {}
    for name, _rows in merged:
        seen_names[name] = seen_names.get(name, 0) + 1

    out = []
    for name, rows in merged:
        idx = np.array(rows)
        # Chlorophyll, specifically -- not measurements in general. This list exists so a
        # user can put a forecast beside a measurement of the same quantity, and the
        # forecast is chlorophyll. Dropping toxin-only entries removes several hundred
        # state beach-monitoring stations that would otherwise dominate it: they are the
        # best-sampled sites in the archive and they are ponds a few hundred metres
        # across, where a 300 m forecast has nothing to say. Their microcystin still
        # reaches the user through the pixel probe, which reports it without plotting it.
        if int((index.variable[idx] == CHLOROPHYLL).sum()) < MIN_SITE_SAMPLES:
            continue
        west, east = float(index.lon[idx].min()), float(index.lon[idx].max())
        south, north = float(index.lat[idx].min()), float(index.lat[idx].max())
        if _span_km(west, south, east, north) > WIDE_SITE_KM:
            continue  # still not one place; better absent than wrong
        clat, clon = (south + north) / 2.0, (west + east) / 2.0
        # Already offered as a registry lake: the same water under two names is one entry.
        if any(w <= clon <= e and s <= clat <= n for w, s, e, n in boxes):
            continue
        if _span_km(west, south, east, north) < 2.0:  # a single station has no extent
            d = 0.02
            west, south, east, north = west - d, south - d, east + d, north + d
        box, clipped = _forecast_box(west, south, east, north)
        label = name if seen_names.get(name, 0) < 2 else f"{name} ({_coord_label(clat, clon)})"
        site = {
            "id": "site-" + hashlib.sha1(f"{name}{clat:.2f}{clon:.2f}".encode()).hexdigest()[:8],
            "name": label,
            "region": "",
            "kind": "measured",
            "split": "",
            "bbox": [west, south, east, north],
            "forecast_bbox": box,
            "clipped_to_limit": clipped,
            "centre": [round(clat, 5), round(clon, 5)],
            "forecast_span_km": round(_span_km(*box), 1),
            "span_km": round(_span_km(west, south, east, north), 1),
        }
        site.update(_summarise(idx, index))
        out.append(site)
    return out


def _haversine_centres(index: InSituIndex, a: list[int], b: list[int]) -> float:
    ai, bi = np.array(a), np.array(b)
    return float(
        haversine_km(
            float(index.lat[ai].mean()),
            float(index.lon[ai].mean()),
            np.array([float(index.lat[bi].mean())]),
            np.array([float(index.lon[bi].mean())]),
        )[0]
    )


def _coord_label(lat: float, lon: float) -> str:
    return f"{abs(lat):.1f}°{'N' if lat >= 0 else 'S'} {abs(lon):.1f}°{'E' if lon >= 0 else 'W'}"


@lru_cache(maxsize=4)
def validation_sites(aoi_path: Path | None = None, index_path: Path | None = None) -> list[dict]:
    """Every water body a user could go and compare a forecast against, with what is there.

    Two kinds, kept distinct because they promise different things. **Modelled** lakes are
    the registry the model was trained and evaluated on -- a forecast there is a forecast
    on familiar water. **Measured** water bodies are places the in-situ archives cover and
    this project has never seen; a forecast there is the model extrapolating, and the
    field data is the only thing that would catch it being wrong.

    Sorted by how much has been measured, because that is what the list is for.
    """
    index = load_index(index_path)
    if index is None or not len(index):
        return []
    registry = _registry_sites(
        Path(aoi_path) if aoi_path else Path("data/aoi/lakes_expanded.geojson"), index
    )
    sites = registry + _named_sites(index, registry)
    sites.sort(key=lambda s: (s["kind"] != "modelled", -s["chlorophyll_a"], s["name"]))
    return sites
