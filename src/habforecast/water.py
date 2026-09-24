"""What kind of water is this? -- the question every other choice depends on.

A forecast over Lake Erie and a forecast over the Gulf of Mexico are not the same job.
They differ in the retrieval algorithm that is valid, in the resolution the feature needs,
in which products even cover the area, and -- most importantly here -- in whether this
project has ever measured a model on water like it. So the area a user draws is classified
first, and everything downstream is chosen from the answer (`habforecast.routing`).

Two axes, because they are genuinely independent and each decides something different.

**Setting** is hydrology, and decides *which products exist*:

* `inland` -- a lake, reservoir or river. Not connected to the ocean.
* `coastal` -- marine water on the shelf, an estuary, a bay, a lagoon.
* `open-ocean` -- marine water beyond the shelf.

**Optical class** decides *which retrieval is trustworthy*, and it does not follow setting
exactly:

* `case-2` -- optically complex water, where suspended sediment and dissolved organic
  matter drive reflectance independently of chlorophyll. All inland water, and marine
  water near the coast. Band-ratio retrievals (VIIRS/MODIS OCx) are not designed for it
  and mask it heavily -- measured in this project at **2.6% valid pixels on Taihu**. OLCI's
  CHL_NN neural retrieval is designed for exactly this water.
* `case-1` -- open ocean, where chlorophyll dominates the optics and the standard
  band-ratio retrievals are the validated ones.

The Curonian Lagoon is why the two axes are separate: it is marine by hydrology and
Case-2 by optics, and getting either half wrong picks the wrong product.

The classification is offline, deterministic and costs a few milliseconds. It reads one
packed Natural Earth ocean polygon (`data/reference/ne10_ocean.npz`, built by
`scripts/build_coast_reference.py`); if that file is absent, classification degrades to
`unknown` and the router says so rather than guessing.
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

INLAND = "inland"
COASTAL = "coastal"
OPEN_OCEAN = "open-ocean"
UNKNOWN = "unknown"

CASE_1 = "case-1"
CASE_2 = "case-2"

# Within this of the shore, marine water is optically complex often enough that assuming
# otherwise is the more expensive mistake: river plumes, resuspended sediment and
# estuarine gradients all live here. A working threshold, not a physical boundary -- a
# plume can run far past it, and a steep rocky coast can be Case-1 well inside it.
NEARSHORE_KM = 40.0
# Roughly the shelf-break scale. Past this, features are large, the water is Case-1, and a
# 4 km product resolves what there is to resolve.
SHELF_KM = 200.0

# Natural Earth includes these in its ocean polygon. They are endorheic -- no outflow to
# the sea, no tidal exchange, their own salinity and their own optics -- so for every
# decision this module feeds they behave as very large lakes, and both have documented
# cyanobacterial blooms. Listed explicitly rather than inferred, because "enclosed" is not
# a property the polygon carries.
ENCLOSED_SEAS: tuple[tuple[str, float, float, float, float], ...] = (
    ("Caspian Sea", 46.0, 36.0, 55.5, 47.5),
    ("Aral Sea", 57.5, 43.5, 62.0, 47.0),
)

REFERENCE = Path(os.getenv("HAB_COAST_REFERENCE", "data/reference/ne10_ocean.npz"))

EARTH_RADIUS_KM = 6371.0088

_ref_lock = threading.Lock()
_ref: dict | None = None
_ref_missing = False


def _reference_path() -> Path:
    """Where the packed polygon lives, whether the process runs from the repo or not."""
    if REFERENCE.is_absolute():
        return REFERENCE
    here = Path.cwd() / REFERENCE
    if here.exists():
        return here
    return Path(__file__).resolve().parents[2] / REFERENCE


def _load() -> dict | None:
    """Load and index the ocean polygon once. Returns None when it is not installed."""
    global _ref, _ref_missing
    if _ref is not None or _ref_missing:
        return _ref
    with _ref_lock:
        if _ref is not None or _ref_missing:
            return _ref
        path = _reference_path()
        if not path.exists():
            logger.warning(
                "no coastline reference at %s -- water bodies cannot be classified. "
                "Build it with scripts/build_coast_reference.py",
                path,
            )
            _ref_missing = True
            return None
        data = np.load(path)
        xy = data["xy"].astype(np.float64) / float(data["scale"])
        offsets = data["offsets"]
        rings = [xy[offsets[i] : offsets[i + 1]] for i in range(len(offsets) - 1)]
        # Each ring's edges precomputed as (x, y) -> (x2, y2). One ring here is the world
        # ocean's own outline at 200 k vertices, and rolling it inside the point test cost
        # more than the test: 96 ms per classification became 12 ms by hoisting this.
        edges = [(r[:, 0], r[:, 1], np.roll(r[:, 0], -1), np.roll(r[:, 1], -1)) for r in rings]
        bounds = np.array(
            [[r[:, 0].min(), r[:, 1].min(), r[:, 0].max(), r[:, 1].max()] for r in rings]
        )
        # The polygon's boundary is the coastline everywhere except where it runs along
        # the edges of the map, which are an artefact of drawing a sphere as a rectangle
        # and are not shore. Dropping them keeps a mid-Pacific point from being told it is
        # a few kilometres from land.
        edge = (np.abs(xy[:, 0]) >= 179.99) | (np.abs(xy[:, 1]) >= 89.99)
        coast = xy[~edge]
        lon, lat = np.radians(coast[:, 0]), np.radians(coast[:, 1])
        _ref = {
            "edges": edges,
            "bounds": bounds,
            "coast_xyz": np.stack(
                [np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], axis=1
            ).astype(np.float32),
        }
        logger.info("coastline reference: %d rings, %d coast vertices", len(rings), len(coast))
        return _ref


def _in_ocean(lon: float, lat: float, ref: dict) -> bool:
    """Even-odd ray cast against every ring whose bounding box contains the point.

    Holes (islands) and outer rings go through the same parity rule, which is why the
    rings are kept flat rather than grouped into polygons: a point on an island inside the
    ocean crosses one extra boundary and comes out not-ocean, which is correct.
    """
    b = ref["bounds"]
    cand = np.flatnonzero((b[:, 0] <= lon) & (b[:, 2] >= lon) & (b[:, 1] <= lat) & (b[:, 3] >= lat))
    inside = False
    for i in cand:
        x, y, x2, y2 = ref["edges"][i]
        crosses = (y > lat) != (y2 > lat)
        if not crosses.any():
            continue
        xi, yi, xj, yj = x[crosses], y[crosses], x2[crosses], y2[crosses]
        xint = xi + (lat - yi) * (xj - xi) / (yj - yi)
        if int(np.count_nonzero(xint > lon)) % 2:
            inside = not inside
    return inside


def _coast_km(points, ref: dict) -> np.ndarray:
    """Great-circle distance from each (lat, lon) to the nearest coastline vertex, in km.

    Every point in one matrix product against 400 k coast vertices: the whole AOI costs
    one BLAS call rather than twenty-five.
    """
    pts = np.radians(np.asarray(points, dtype=np.float64))
    a, b = pts[:, 0], pts[:, 1]
    p = np.stack([np.cos(a) * np.cos(b), np.cos(a) * np.sin(b), np.sin(a)], axis=0)
    cos_max = (ref["coast_xyz"] @ p.astype(np.float32)).max(axis=0)
    return EARTH_RADIUS_KM * np.arccos(np.clip(cos_max, -1.0, 1.0))


def _enclosed_sea(lon: float, lat: float) -> str | None:
    for name, west, south, east, north in ENCLOSED_SEAS:
        if west <= lon <= east and south <= lat <= north:
            return name
    return None


@dataclass(frozen=True)
class WaterBody:
    """What the drawn area is, with the evidence that says so."""

    setting: str  # inland | coastal | open-ocean | unknown
    optical_class: str  # case-1 | case-2 | unknown
    coast_distance_km: float | None  # nearest approach of the AOI to the coast
    marine_fraction: float  # 0 = wholly continental, 1 = wholly marine
    side_km: float
    centre: tuple[float, float]  # (lat, lon)
    enclosed_sea: str | None = None
    evidence: list[str] = field(default_factory=list)

    @property
    def known(self) -> bool:
        return self.setting != UNKNOWN

    @property
    def mixed(self) -> bool:
        """The AOI straddles the shoreline -- a river mouth, an estuary, a barrier island."""
        return 0.0 < self.marine_fraction < 1.0

    def as_dict(self) -> dict:
        return {
            "setting": self.setting,
            "optical_class": self.optical_class,
            "coast_distance_km": (
                None if self.coast_distance_km is None else round(self.coast_distance_km, 1)
            ),
            "marine_fraction": round(self.marine_fraction, 2),
            "mixed": self.mixed,
            "side_km": round(self.side_km, 1),
            "enclosed_sea": self.enclosed_sea,
            "evidence": list(self.evidence),
        }


def _samples(west: float, south: float, east: float, north: float, n: int = 5):
    """A small grid over the AOI, so a shoreline crossing the area is actually noticed."""
    lons = np.linspace(west, east, n)
    lats = np.linspace(south, north, n)
    return [(float(la), float(lo)) for la in lats for lo in lons]


def bbox_side_km(west: float, south: float, east: float, north: float) -> float:
    """The larger side of the box in kilometres -- the number that decides resolution."""
    mid = np.radians((south + north) / 2.0)
    return float(max((east - west) * 111.32 * np.cos(mid), (north - south) * 110.57))


def classify(bbox: tuple[float, float, float, float]) -> WaterBody:
    """Classify the area `bbox` = (west, south, east, north) in degrees.

    Never raises. Without the reference file the result is `unknown`, which the router
    treats as "say so, and take the route that assumes the least" rather than as a
    default.
    """
    west, south, east, north = bbox
    centre = ((south + north) / 2.0, (west + east) / 2.0)
    side = bbox_side_km(west, south, east, north)

    sea = _enclosed_sea(centre[1], centre[0])
    if sea is not None:
        return WaterBody(
            setting=INLAND,
            optical_class=CASE_2,
            coast_distance_km=None,
            marine_fraction=0.0,
            side_km=side,
            centre=centre,
            enclosed_sea=sea,
            evidence=[
                f"{sea} is endorheic -- no outflow and no tidal exchange with the ocean, "
                "so it is treated as inland water despite being mapped as sea."
            ],
        )

    ref = _load()
    if ref is None:
        return WaterBody(
            setting=UNKNOWN,
            optical_class=UNKNOWN,
            coast_distance_km=None,
            marine_fraction=0.0,
            side_km=side,
            centre=centre,
            evidence=["The coastline reference is not installed, so this area was not classified."],
        )

    points = _samples(west, south, east, north)
    marine = [_in_ocean(lo, la, ref) for la, lo in points]
    fraction = float(np.mean(marine))
    # The nearest approach, not the centre's distance: an AOI whose corner touches the
    # shore is a coastal problem even if its middle sits 60 km out.
    coast = float(_coast_km(points, ref).min())

    evidence = [f"nearest coast {coast:.0f} km", f"{fraction:.0%} of the area is marine"]
    if fraction == 0.0:
        setting = INLAND
        evidence.append("no part of the area falls inside the ocean polygon")
    elif fraction < 1.0:
        setting = COASTAL
        evidence.append("the area straddles the shoreline")
    elif coast > SHELF_KM:
        setting = OPEN_OCEAN
        evidence.append(f"more than {SHELF_KM:.0f} km from any shore")
    else:
        setting = COASTAL

    optical = CASE_2 if setting == INLAND or coast <= NEARSHORE_KM else CASE_1
    return WaterBody(
        setting=setting,
        optical_class=optical,
        coast_distance_km=coast,
        marine_fraction=fraction,
        side_km=side,
        centre=centre,
        evidence=evidence,
    )
