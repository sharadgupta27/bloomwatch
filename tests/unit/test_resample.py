"""Tests for swath-to-grid resampling.

The unprojectable-pixel test is a regression guard. OLCI swaths are ~1270 km wide, so
the far edge of a granule can fall outside the target UTM zone entirely; pyproj returns
inf there, and `scipy.spatial.cKDTree` rejects non-finite input with
"ValueError: data must be finite" -- which killed an entire scene read during the real
ingest. Those pixels are by construction nowhere near the lake, so dropping them is free.
"""

from __future__ import annotations

import numpy as np
import pytest

from habforecast.aoi import LakeGrid
from habforecast.data.resample import SwathResampler


@pytest.fixture
def grid() -> LakeGrid:
    # Small grid in UTM 17N, roughly over Lake Erie's western basin.
    return LakeGrid(
        lake_id="test",
        epsg=32617,
        x_min=300000.0,
        y_min=4600000.0,
        width=12,
        height=10,
        resolution=300.0,
    )


def swath_over(grid: LakeGrid, n=40):
    """A lon/lat swath that covers the grid."""
    lon, lat = grid.lonlat
    lo_lon, hi_lon = lon.min() - 0.02, lon.max() + 0.02
    lo_lat, hi_lat = lat.min() - 0.02, lat.max() + 0.02
    la, lo = np.meshgrid(
        np.linspace(hi_lat, lo_lat, n), np.linspace(lo_lon, hi_lon, n), indexing="ij"
    )
    return lo, la


class TestResampling:
    def test_covers_the_grid(self, grid):
        lon, lat = swath_over(grid)
        vals = {"log_chl": np.full(lon.shape, 1.5, dtype=np.float32)}
        out, covered = SwathResampler(grid, max_distance_m=500.0)(lon, lat, vals)
        assert out["log_chl"].shape == grid.shape
        assert covered.mean() > 0.8
        assert np.allclose(out["log_chl"][covered], 1.5)

    def test_uncovered_cells_are_nan(self, grid):
        lon, lat = swath_over(grid)
        vals = {"log_chl": np.full(lon.shape, 1.0, dtype=np.float32)}
        # A tiny cutoff means almost nothing matches.
        out, covered = SwathResampler(grid, max_distance_m=1.0)(lon, lat, vals)
        assert np.isnan(out["log_chl"][~covered]).all()

    def test_source_valid_mask_excludes_pixels(self, grid):
        lon, lat = swath_over(grid)
        vals = {"log_chl": np.full(lon.shape, 2.0, dtype=np.float32)}
        none_valid = np.zeros(lon.shape, dtype=bool)
        out, covered = SwathResampler(grid)(lon, lat, vals, source_valid=none_valid)
        assert not covered.any()
        assert np.isnan(out["log_chl"]).all()

    def test_nan_geolocation_is_ignored(self, grid):
        lon, lat = swath_over(grid)
        lon = lon.copy()
        lon[0, :] = np.nan
        vals = {"log_chl": np.full(lon.shape, 1.0, dtype=np.float32)}
        out, covered = SwathResampler(grid, max_distance_m=500.0)(lon, lat, vals)
        assert np.isfinite(out["log_chl"][covered]).all()

    def test_unprojectable_pixels_do_not_crash(self, grid):
        """Antipodal coordinates project to inf in UTM; cKDTree must never see them."""
        lon, lat = swath_over(grid)
        lon, lat = lon.copy(), lat.copy()
        # Half the swath placed on the other side of the planet.
        lon[: lon.shape[0] // 2, :] = 150.0
        lat[: lat.shape[0] // 2, :] = -80.0
        vals = {"log_chl": np.full(lon.shape, 1.25, dtype=np.float32)}

        out, covered = SwathResampler(grid, max_distance_m=500.0)(lon, lat, vals)
        assert out["log_chl"].shape == grid.shape
        assert covered.any(), "the valid half of the swath should still resample"
        assert np.allclose(out["log_chl"][covered], 1.25)

    def test_all_unprojectable_returns_empty(self, grid):
        lon = np.full((8, 8), 150.0)
        lat = np.full((8, 8), -80.0)
        vals = {"log_chl": np.ones((8, 8), dtype=np.float32)}
        out, covered = SwathResampler(grid, max_distance_m=500.0)(lon, lat, vals)
        assert not covered.any()
        assert np.isnan(out["log_chl"]).all()

    def test_empty_input_returns_empty(self, grid):
        lon = np.full((4, 4), np.nan)
        lat = np.full((4, 4), np.nan)
        _out, covered = SwathResampler(grid)(lon, lat, {"log_chl": np.ones((4, 4), np.float32)})
        assert not covered.any()

    def test_shape_mismatch_raises(self, grid):
        with pytest.raises(ValueError, match="shape mismatch"):
            SwathResampler(grid)(np.zeros((4, 4)), np.zeros((5, 5)), {"v": np.zeros((4, 4))})

    def test_nearest_neighbour_does_not_interpolate(self, grid):
        """Values must be carried, never blended -- bloom fronts are sharp."""
        lon, lat = swath_over(grid)
        vals = np.zeros(lon.shape, dtype=np.float32)
        vals[:, : lon.shape[1] // 2] = 3.0
        out, covered = SwathResampler(grid, max_distance_m=500.0)(lon, lat, {"v": vals})
        present = np.unique(out["v"][covered])
        assert set(np.round(present, 6)) <= {0.0, 3.0}, f"interpolated values appeared: {present}"
