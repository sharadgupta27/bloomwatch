"""Area-of-interest registry and the fixed analysis grid for each lake.

Every lake gets a *deterministic* target grid: a regular raster in the lake's local
UTM zone at a fixed ground spacing. Determinism matters more than it might seem -
the grid definition is what makes an ingest run reproducible and what lets a Zarr
store be appended to across months of incremental ingests without silent
misalignment. The grid is derived purely from the AOI polygon and the configured
resolution, so recomputing it always yields the same array shape and transform.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np
from pyproj import CRS, Transformer

# Sentinel-3 OLCI full-resolution ground sampling distance at nadir.
OLCI_FR_RESOLUTION_M = 300.0

# Largest ad-hoc selection accepted, per side. A user dragging a box across an ocean would
# otherwise ask for a grid of tens of millions of cells and a hundred granule reads: the
# request would not fail, it would just consume the container until something killed it.
# Refusing with a number the user can act on is the better behaviour.
MAX_ADHOC_SIDE_KM = 80.0


@dataclass(frozen=True)
class LakeGrid:
    """A fixed, regular analysis grid for one lake, in projected (UTM) coordinates."""

    lake_id: str
    epsg: int
    x_min: float
    y_min: float
    width: int
    height: int
    resolution: float

    @property
    def crs(self) -> CRS:
        return CRS.from_epsg(self.epsg)

    @property
    def shape(self) -> tuple[int, int]:
        """(height, width) - numpy row/col order."""
        return (self.height, self.width)

    @property
    def transform(self) -> tuple[float, float, float, float, float, float]:
        """GDAL-style affine transform (a, b, c, d, e, f) with north-up orientation."""
        return (self.resolution, 0.0, self.x_min, 0.0, -self.resolution, self.y_max)

    @property
    def y_max(self) -> float:
        return self.y_min + self.height * self.resolution

    @property
    def x_max(self) -> float:
        return self.x_min + self.width * self.resolution

    @cached_property
    def x_coords(self) -> np.ndarray:
        """Cell-centre easting for each column."""
        return self.x_min + (np.arange(self.width) + 0.5) * self.resolution

    @cached_property
    def y_coords(self) -> np.ndarray:
        """Cell-centre northing for each row, descending (north-up)."""
        return self.y_max - (np.arange(self.height) + 0.5) * self.resolution

    @cached_property
    def lonlat(self) -> tuple[np.ndarray, np.ndarray]:
        """Per-cell (lon, lat) in EPSG:4326, shape (height, width) each."""
        xx, yy = np.meshgrid(self.x_coords, self.y_coords)
        tf = Transformer.from_crs(self.crs, CRS.from_epsg(4326), always_xy=True)
        lon, lat = tf.transform(xx, yy)
        return lon, lat

    @cached_property
    def bounds_lonlat(self) -> tuple[float, float, float, float]:
        """(west, south, east, north) in EPSG:4326, safe for a STAC bbox query."""
        lon, lat = self.lonlat
        return (float(lon.min()), float(lat.min()), float(lon.max()), float(lat.max()))


@dataclass(frozen=True)
class LakeAOI:
    """One lake: its polygon, metadata, and dataset split assignment."""

    lake_id: str
    name: str
    country: str
    split: str
    trophic_state: str
    bloom_season: tuple[int, int]
    utm_epsg: int
    bbox: tuple[float, float, float, float]
    notes: str = ""
    # Levels of the multilevel model and its calibration (see habforecast.groups). A
    # registry written before they existed leaves them unknown, never guessed.
    continent: str = "unknown"
    thermal_class: str = "unknown"

    @property
    def groups(self):
        from habforecast.groups import LakeGroups

        return LakeGroups(
            trophic=self.trophic_state.lower(),
            thermal=self.thermal_class.lower(),
            continent=self.continent.lower(),
        )

    def in_bloom_season(self, month: int) -> bool:
        """Whether `month` falls in this lake's bloom window (inclusive, may wrap)."""
        start, end = self.bloom_season
        if start <= end:
            return start <= month <= end
        return month >= start or month <= end

    def grid(self, resolution: float = OLCI_FR_RESOLUTION_M) -> LakeGrid:
        """Build the deterministic analysis grid for this AOI.

        The AOI bbox is projected to UTM and then *snapped outward* to whole
        multiples of `resolution`. Snapping is what guarantees that two runs -
        or two lakes sharing a UTM zone - produce grids on a common lattice.
        """
        tf = Transformer.from_crs(CRS.from_epsg(4326), CRS.from_epsg(self.utm_epsg), always_xy=True)
        west, south, east, north = self.bbox
        # Project all four corners; a projected bbox is not the projection of the bbox.
        lons = [west, east, east, west]
        lats = [south, south, north, north]
        xs, ys = tf.transform(lons, lats)

        x_min = np.floor(min(xs) / resolution) * resolution
        x_max = np.ceil(max(xs) / resolution) * resolution
        y_min = np.floor(min(ys) / resolution) * resolution
        y_max = np.ceil(max(ys) / resolution) * resolution

        return LakeGrid(
            lake_id=self.lake_id,
            epsg=self.utm_epsg,
            x_min=float(x_min),
            y_min=float(y_min),
            width=round((x_max - x_min) / resolution),
            height=round((y_max - y_min) / resolution),
            resolution=float(resolution),
        )


def utm_epsg_for(lat: float, lon: float) -> int:
    """EPSG code of the UTM zone containing a point.

    Chosen over a single global projection because UTM keeps 300 m cells very close to
    square anywhere they are used, which is what makes an area threshold in km^2 mean the
    same thing at Lake Victoria as at Lake Erie. Above 84 deg or below -80 deg UTM is not
    defined and the caller gets an error rather than a silently distorted grid.
    """
    if not -80.0 <= lat <= 84.0:
        raise ValueError(
            f"latitude {lat:.3f} is outside the UTM domain (-80 to 84); "
            "polar waters are not supported"
        )
    if not -180.0 <= lon <= 180.0:
        raise ValueError(f"longitude {lon:.3f} is out of range")
    zone = int((lon + 180.0) // 6.0) + 1
    zone = min(max(zone, 1), 60)
    return (32600 if lat >= 0 else 32700) + zone


def adhoc_aoi(
    lat: float,
    lon: float,
    radius_km: float = 12.0,
    name: str = "Selected area",
    lake_id: str = "adhoc",
) -> LakeAOI:
    """An AOI centred on an arbitrary point, for forecasting somewhere with no cube.

    This is what lets the app answer "what about *my* lake?". The registry's five lakes
    have years of ingested history; an arbitrary point has none, so everything downstream
    has to come from a live STAC fetch over the request window.

    The bbox is built in metres and converted back to degrees rather than by adding a
    fixed number of degrees to the longitude: a degree of longitude is 111 km at the
    equator and 30 km at 74 deg N, so the naive version would hand a user near the poles
    an area several times larger than they asked for.
    """
    if radius_km <= 0:
        raise ValueError("radius_km must be positive")
    epsg = utm_epsg_for(lat, lon)
    fwd = Transformer.from_crs(CRS.from_epsg(4326), CRS.from_epsg(epsg), always_xy=True)
    inv = Transformer.from_crs(CRS.from_epsg(epsg), CRS.from_epsg(4326), always_xy=True)
    cx, cy = fwd.transform(lon, lat)
    r = radius_km * 1000.0
    lons, lats = inv.transform([cx - r, cx + r, cx + r, cx - r], [cy - r, cy - r, cy + r, cy + r])
    return LakeAOI(
        lake_id=lake_id,
        name=name,
        country="",
        split="adhoc",
        trophic_state="unknown",
        # No prior on when this water body blooms, so every month is in season. Saying
        # "not in season" for a lake we know nothing about would be a fabricated claim.
        bloom_season=(1, 12),
        utm_epsg=epsg,
        bbox=(min(lons), min(lats), max(lons), max(lats)),
        notes=f"ad-hoc AOI, {radius_km:g} km radius about {lat:.4f}, {lon:.4f}",
    )


def adhoc_aoi_from_bbox(
    west: float,
    south: float,
    east: float,
    north: float,
    name: str = "Selected area",
    lake_id: str = "adhoc",
) -> LakeAOI:
    """An AOI from a box the user drew on the map.

    The UTM zone is chosen from the box's centre, so a selection straddling a zone
    boundary is still projected consistently -- at these extents the distortion at the far
    edge is far below one 300 m cell.
    """
    if east <= west or north <= south:
        raise ValueError("bbox must be [west, south, east, north] with east>west, north>south")
    lat_c, lon_c = (south + north) / 2.0, (west + east) / 2.0
    epsg = utm_epsg_for(lat_c, lon_c)
    tf = Transformer.from_crs(CRS.from_epsg(4326), CRS.from_epsg(epsg), always_xy=True)
    xs, ys = tf.transform([west, east, east, west], [south, south, north, north])
    side_km = max(max(xs) - min(xs), max(ys) - min(ys)) / 1000.0
    if side_km > MAX_ADHOC_SIDE_KM:
        raise ValueError(
            f"that area is about {side_km:.0f} km across; the limit is "
            f"{MAX_ADHOC_SIDE_KM:.0f} km. Zoom in and select a smaller area -- a single "
            "lake or bay is what this model forecasts."
        )
    return LakeAOI(
        lake_id=lake_id,
        name=name,
        country="",
        split="adhoc",
        trophic_state="unknown",
        bloom_season=(1, 12),
        utm_epsg=epsg,
        bbox=(west, south, east, north),
        notes=f"ad-hoc AOI from a drawn box, {side_km:.1f} km across",
    )


class AOIRegistry:
    """Loads the lake registry from GeoJSON and exposes lookup / split filtering."""

    def __init__(self, aois: list[LakeAOI]):
        self._aois = {a.lake_id: a for a in aois}
        if len(self._aois) != len(aois):
            raise ValueError("duplicate lake_id in AOI registry")

    @classmethod
    def from_geojson(cls, path: str | Path) -> AOIRegistry:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        aois = []
        for feat in raw["features"]:
            props = feat["properties"]
            ring = feat["geometry"]["coordinates"][0]
            lons = [p[0] for p in ring]
            lats = [p[1] for p in ring]
            aois.append(
                LakeAOI(
                    lake_id=props["lake_id"],
                    name=props["name"],
                    country=props["country"],
                    split=props["split"],
                    trophic_state=props.get("trophic_state", "unknown"),
                    bloom_season=tuple(props.get("bloom_season", [1, 12])),
                    utm_epsg=int(props["utm_epsg"]),
                    bbox=(min(lons), min(lats), max(lons), max(lats)),
                    notes=props.get("notes", ""),
                    continent=props.get("continent", "unknown"),
                    thermal_class=props.get("thermal_class", "unknown"),
                )
            )
        return cls(aois)

    def __getitem__(self, lake_id: str) -> LakeAOI:
        return self._aois[lake_id]

    def __iter__(self):
        return iter(self._aois.values())

    def __len__(self) -> int:
        return len(self._aois)

    @property
    def lake_ids(self) -> list[str]:
        return list(self._aois)

    def split(self, split: str) -> list[LakeAOI]:
        """All AOIs assigned to a given split ('train' | 'val' | 'test')."""
        return [a for a in self._aois.values() if a.split == split]
