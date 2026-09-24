"""Resample OLCI's curvilinear swath geolocation onto a lake's regular analysis grid.

OLCI L2 products are delivered on the instrument's native swath: a (rows, columns)
array with a per-pixel latitude/longitude grid that is *not* a regular lattice. To
build a time series we must place every scene on a common, fixed grid.

We use nearest-neighbour resampling with a distance cutoff. Nearest-neighbour is the
right choice here rather than bilinear or averaging: CHL_NN is a strongly non-linear
retrieval and blooms have sharp fronts, so interpolating across a bloom edge invents
values that the sensor never observed. The distance cutoff prevents a valid pixel far
outside the swath from being smeared across a gap.
"""

from __future__ import annotations

import logging

import numpy as np
from pyproj import CRS, Transformer
from scipy.spatial import cKDTree

from habforecast.aoi import LakeGrid

logger = logging.getLogger(__name__)


class SwathResampler:
    """Nearest-neighbour swath-to-grid resampler for one fixed target grid.

    The target grid's cell centres are projected once and cached; each scene then only
    pays for building a KD-tree over its own (small, windowed) swath points.
    """

    def __init__(self, grid: LakeGrid, max_distance_m: float | None = None):
        self.grid = grid
        # A cutoff of ~0.8 * cell size keeps each target cell tied to a genuinely
        # co-located observation instead of borrowing from a neighbouring cell.
        self.max_distance_m = (
            max_distance_m if max_distance_m is not None else 0.8 * grid.resolution
        )
        xx, yy = np.meshgrid(grid.x_coords, grid.y_coords)
        self._target_xy = np.column_stack([xx.ravel(), yy.ravel()])
        self._to_grid = Transformer.from_crs(CRS.from_epsg(4326), grid.crs, always_xy=True)

    def __call__(
        self,
        lon: np.ndarray,
        lat: np.ndarray,
        values: dict[str, np.ndarray],
        source_valid: np.ndarray | None = None,
    ) -> tuple[dict[str, np.ndarray], np.ndarray]:
        """Resample `values` from swath geometry onto the target grid.

        Args:
            lon, lat: per-pixel geolocation of the swath window, shape (R, C).
            values: named swath arrays, each shape (R, C), to resample.
            source_valid: optional boolean (R, C); False pixels are excluded from the
                KD-tree entirely so an invalid observation can never win a match.

        Returns:
            (resampled, covered) where `resampled` maps each name to a (H, W) float32
            array (NaN where uncovered) and `covered` is the (H, W) boolean mask of
            target cells that found a source pixel within the distance cutoff.
        """
        lon = np.asarray(lon, dtype=np.float64)
        lat = np.asarray(lat, dtype=np.float64)
        if lon.shape != lat.shape:
            raise ValueError(f"lon/lat shape mismatch: {lon.shape} vs {lat.shape}")

        finite = np.isfinite(lon) & np.isfinite(lat)
        if source_valid is not None:
            finite &= np.asarray(source_valid, dtype=bool)

        h, w = self.grid.shape
        empty = {k: np.full((h, w), np.nan, dtype=np.float32) for k in values}
        if not finite.any():
            return empty, np.zeros((h, w), dtype=bool)

        src_x, src_y = self._to_grid.transform(lon[finite], lat[finite])

        # pyproj returns inf for points it cannot project into the target UTM zone, which
        # happens on the far edge of a 1270 km swath. cKDTree rejects those outright with
        # "ValueError: data must be finite, check for nan or inf values", killing the whole
        # scene read. Dropping them costs nothing -- they are, by construction, nowhere
        # near the lake.
        projectable = np.isfinite(src_x) & np.isfinite(src_y)
        if not projectable.all():
            n_bad = int((~projectable).sum())
            logger.debug("%s: dropped %d unprojectable source pixels", self.grid.lake_id, n_bad)
            src_x, src_y = src_x[projectable], src_y[projectable]
            # Keep the value gather aligned with the surviving coordinates.
            finite = finite.copy()
            finite[finite] = projectable

        if src_x.size == 0:
            return empty, np.zeros((h, w), dtype=bool)

        src_xy = np.column_stack([src_x, src_y])
        tree = cKDTree(src_xy)
        dist, idx = tree.query(
            self._target_xy, k=1, distance_upper_bound=self.max_distance_m, workers=-1
        )
        # cKDTree marks misses with index == n and distance == inf.
        covered_flat = np.isfinite(dist) & (idx < len(src_xy))
        idx_safe = np.where(covered_flat, idx, 0)

        out = {}
        for name, arr in values.items():
            flat_src = np.asarray(arr, dtype=np.float32)[finite]
            picked = flat_src[idx_safe]
            out[name] = np.where(covered_flat, picked, np.nan).reshape(h, w).astype(np.float32)

        return out, covered_flat.reshape(h, w)
