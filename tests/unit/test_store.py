"""Tests for the Zarr datacube: compositing, idempotency, and out-of-order writes.

The out-of-order case is a regression test for a real failure. The store originally only
supported strict append, on the reasoning that ingest walks forward through time. Then a
backfill run tried to write 2018-07 into a cube that already held 2018-08 (from an
earlier partial test run) and the whole multi-lake ingest aborted.

Strict append was not merely inconvenient -- silently allowing it would have been worse.
Zarr cannot insert into the middle of a dimension, so an unsorted time axis makes every
subsequent `.sel()` slice quietly wrong. Hence: fast append when ordered, merge-and-
rewrite when not, and a test for each.
"""

from __future__ import annotations

from datetime import UTC, date, datetime

import numpy as np
import pytest
import xarray as xr

from habforecast.aoi import LakeGrid
from habforecast.data.olci import SceneObservation
from habforecast.data.store import DailyFrame, LakeCube, composite_day, group_by_day

H, W = 8, 10


@pytest.fixture
def grid() -> LakeGrid:
    return LakeGrid(
        lake_id="testlake",
        epsg=32617,
        x_min=300000.0,
        y_min=4600000.0,
        width=W,
        height=H,
        resolution=300.0,
    )


def make_obs(day: date, value: float, valid_frac: float = 1.0, seed: int = 0) -> SceneObservation:
    rng = np.random.default_rng(seed)
    log_chl = np.full((H, W), value, dtype=np.float32)
    valid = rng.random((H, W)) < valid_frac
    log_chl[~valid] = np.nan
    return SceneObservation(
        lake_id="testlake",
        item_id=f"S3A_{day:%Y%m%d}_{seed}",
        time=datetime(day.year, day.month, day.day, 15, 30, tzinfo=UTC),
        platform="Sentinel-3A",
        log_chl=log_chl,
        valid=valid,
    )


def make_frame(day: date, value: float) -> DailyFrame:
    return DailyFrame(
        day=day,
        log_chl=np.full((H, W), value, dtype=np.float32),
        n_obs=np.ones((H, W), dtype=np.uint8),
        scene_ids=(f"scene_{day}",),
        mean_valid_fraction=1.0,
    )


class TestCompositing:
    def test_averages_in_log_space(self):
        """Geometric mean: chlorophyll is log-normal, so the mean of logs is correct."""
        day = date(2023, 8, 1)
        frame = composite_day(day, [make_obs(day, 1.0), make_obs(day, 3.0)])
        assert frame is not None
        assert np.allclose(frame.log_chl, 2.0)
        assert frame.n_obs.max() == 2

    def test_counts_only_valid_contributions(self):
        day = date(2023, 8, 1)
        a = make_obs(day, 1.0)
        b = make_obs(day, 3.0)
        b.valid[:] = False
        b.log_chl[:] = np.nan
        frame = composite_day(day, [a, b])
        assert np.allclose(frame.log_chl, 1.0)
        assert frame.n_obs.max() == 1

    def test_all_invalid_returns_none(self):
        day = date(2023, 8, 1)
        obs = make_obs(day, 1.0)
        obs.valid[:] = False
        obs.log_chl[:] = np.nan
        assert composite_day(day, [obs]) is None

    def test_empty_list_returns_none(self):
        assert composite_day(date(2023, 8, 1), []) is None

    def test_shape_mismatch_raises(self):
        day = date(2023, 8, 1)
        a = make_obs(day, 1.0)
        b = make_obs(day, 1.0)
        b.log_chl = np.zeros((H + 1, W), dtype=np.float32)
        with pytest.raises(ValueError, match="shape"):
            composite_day(day, [a, b])

    def test_group_by_day_buckets_utc_dates(self):
        d1, d2 = date(2023, 8, 1), date(2023, 8, 2)
        grouped = group_by_day(
            [make_obs(d1, 1.0, seed=1), make_obs(d1, 2.0, seed=2), make_obs(d2, 3.0, seed=3)]
        )
        assert set(grouped) == {d1, d2}
        assert len(grouped[d1]) == 2


class TestCubeWriting:
    def test_creates_cube_with_expected_structure(self, tmp_path, grid):
        cube = LakeCube(tmp_path / "testlake.zarr", grid)
        assert cube.write([make_frame(date(2023, 8, 1), 1.0)]) == 1
        assert cube.exists

        ds = xr.open_zarr(cube.path, consolidated=True)
        assert dict(ds.sizes) == {"time": 1, "y": H, "x": W}
        assert set(ds.data_vars) == {"log_chl", "n_obs"}
        assert {"lat", "lon", "time", "x", "y"} <= set(ds.coords)
        assert ds.attrs["lake_id"] == "testlake"
        assert ds.attrs["crs_epsg"] == 32617

    def test_append_is_ordered(self, tmp_path, grid):
        cube = LakeCube(tmp_path / "c.zarr", grid)
        cube.write([make_frame(date(2023, 8, 1), 1.0)])
        cube.write([make_frame(date(2023, 8, 5), 2.0)])
        ds = xr.open_zarr(cube.path, consolidated=True)
        times = [np.datetime64(t, "D") for t in ds.time.values]
        assert times == sorted(times)
        assert ds.sizes["time"] == 2

    def test_is_idempotent(self, tmp_path, grid):
        """Re-writing an already-stored day must be a no-op, not a duplicate."""
        cube = LakeCube(tmp_path / "c.zarr", grid)
        frames = [make_frame(date(2023, 8, 1), 1.0)]
        assert cube.write(frames) == 1
        assert cube.write(frames) == 0
        assert xr.open_zarr(cube.path, consolidated=True).sizes["time"] == 1

    def test_out_of_order_write_merges_and_stays_sorted(self, tmp_path, grid):
        """Backfilling an earlier month must not corrupt the time axis."""
        cube = LakeCube(tmp_path / "c.zarr", grid)
        cube.write([make_frame(date(2023, 8, 1), 8.0)])
        written = cube.write([make_frame(date(2023, 7, 1), 7.0)])  # earlier than stored
        assert written == 1

        ds = xr.open_zarr(cube.path, consolidated=True)
        times = [np.datetime64(t, "D") for t in ds.time.values]
        assert times == sorted(times), "time axis unsorted after backfill"
        assert len(times) == len(set(times)), "duplicate timestamps after merge"
        assert ds.sizes["time"] == 2
        # Values must follow their own timestamps through the merge.
        assert float(ds.log_chl.isel(time=0).mean()) == pytest.approx(7.0)
        assert float(ds.log_chl.isel(time=1).mean()) == pytest.approx(8.0)

    def test_interleaved_backfill_preserves_all_days(self, tmp_path, grid):
        cube = LakeCube(tmp_path / "c.zarr", grid)
        for d in (date(2023, 8, 10), date(2023, 8, 1), date(2023, 8, 20), date(2023, 8, 5)):
            cube.write([make_frame(d, float(d.day))])
        ds = xr.open_zarr(cube.path, consolidated=True)
        days = [
            int(np.datetime64(t, "D").astype("datetime64[D]").item().day) for t in ds.time.values
        ]
        assert days == [1, 5, 10, 20]

    def test_existing_days_reports_stored_dates(self, tmp_path, grid):
        cube = LakeCube(tmp_path / "c.zarr", grid)
        assert cube.existing_days() == set()
        cube.write([make_frame(date(2023, 8, 1), 1.0), make_frame(date(2023, 8, 2), 2.0)])
        assert cube.existing_days() == {date(2023, 8, 1), date(2023, 8, 2)}

    def test_empty_write_is_noop(self, tmp_path, grid):
        cube = LakeCube(tmp_path / "c.zarr", grid)
        assert cube.write([]) == 0
        assert not cube.exists

    def test_open_missing_cube_raises(self, tmp_path, grid):
        with pytest.raises(FileNotFoundError):
            LakeCube(tmp_path / "nope.zarr", grid).open()

    def test_nan_survives_roundtrip(self, tmp_path, grid):
        """Unobserved pixels must come back as NaN, not as a fill sentinel."""
        cube = LakeCube(tmp_path / "c.zarr", grid)
        frame = make_frame(date(2023, 8, 1), 1.0)
        frame.log_chl[0, 0] = np.nan
        frame.n_obs[0, 0] = 0
        cube.write([frame])
        ds = xr.open_zarr(cube.path, consolidated=True)
        assert np.isnan(float(ds.log_chl.isel(time=0, y=0, x=0)))


class TestAttrsSurviveEveryWritePath:
    """Georeferencing metadata must survive fresh, append AND merge writes.

    Regression test for a data-integrity bug: `to_zarr(mode="a")` overwrites the group
    attributes with those of the dataset being appended. Blocks built by `_empty_like`
    carry no attrs, so every incremental append silently erased the CRS, resolution and
    grid definition -- which had already happened to all five production cubes. The data
    stayed correct but the cubes stopped being self-describing.
    """

    REQUIRED = ("lake_id", "crs_epsg", "resolution_m", "grid")

    def _attrs(self, cube):
        return xr.open_zarr(cube.path, consolidated=True).attrs

    def test_fresh_write_sets_attrs(self, tmp_path, grid):
        cube = LakeCube(tmp_path / "c.zarr", grid)
        cube.write([make_frame(date(2023, 8, 10), 1.0)])
        attrs = self._attrs(cube)
        for key in self.REQUIRED:
            assert attrs.get(key) is not None, f"fresh write lost '{key}'"

    def test_append_preserves_attrs(self, tmp_path, grid):
        cube = LakeCube(tmp_path / "c.zarr", grid)
        cube.write([make_frame(date(2023, 8, 10), 1.0)])
        cube.write([make_frame(date(2023, 8, 20), 3.0)])  # append path
        attrs = self._attrs(cube)
        for key in self.REQUIRED:
            assert attrs.get(key) is not None, f"append erased '{key}'"
        assert attrs["crs_epsg"] == grid.epsg
        assert attrs["resolution_m"] == grid.resolution

    def test_merge_preserves_attrs(self, tmp_path, grid):
        cube = LakeCube(tmp_path / "c.zarr", grid)
        cube.write([make_frame(date(2023, 8, 10), 1.0)])
        cube.write([make_frame(date(2023, 8, 1), 2.0)])  # merge path
        attrs = self._attrs(cube)
        for key in self.REQUIRED:
            assert attrs.get(key) is not None, f"merge erased '{key}'"

    def test_attrs_survive_a_long_append_chain(self, tmp_path, grid):
        """The real ingest appends month after month; attrs must survive all of them."""
        cube = LakeCube(tmp_path / "c.zarr", grid)
        for day in range(1, 15):
            cube.write([make_frame(date(2023, 8, day), float(day))])
        attrs = self._attrs(cube)
        assert attrs.get("crs_epsg") == grid.epsg
        assert attrs.get("lake_id") == grid.lake_id

    def test_crs_is_recoverable_from_the_cube_alone(self, tmp_path, grid):
        """A consumer with only the .zarr must be able to georeference it."""
        import json

        cube = LakeCube(tmp_path / "c.zarr", grid)
        cube.write([make_frame(date(2023, 8, 10), 1.0)])
        cube.write([make_frame(date(2023, 8, 20), 2.0)])
        attrs = self._attrs(cube)
        stored = json.loads(attrs["grid"])
        assert stored["epsg"] == grid.epsg
        assert stored["resolution"] == grid.resolution
        assert stored["width"] == grid.width and stored["height"] == grid.height


class TestFlagStatsPersistence:
    """WQSF flag frequencies must actually reach disk.

    They were computed per scene at real cost and then dropped on the floor, while both
    the qc docstring and `monitoring.drift.compare_flag_frequencies` assumed they were
    stored -- making the drift check dead code that could never fire.
    """

    def test_composite_averages_flag_stats_across_scenes(self):
        day = date(2023, 8, 1)
        a, b = make_obs(day, 1.0, seed=1), make_obs(day, 2.0, seed=2)
        a.flag_stats = {"CLOUD": 0.2, "INLAND_WATER": 0.6}
        b.flag_stats = {"CLOUD": 0.4, "INLAND_WATER": 0.5}
        frame = composite_day(day, [a, b])
        assert frame.flag_stats["CLOUD"] == pytest.approx(0.3)
        assert frame.flag_stats["INLAND_WATER"] == pytest.approx(0.55)

    def test_missing_flag_counts_as_zero(self):
        day = date(2023, 8, 1)
        a, b = make_obs(day, 1.0, seed=1), make_obs(day, 2.0, seed=2)
        a.flag_stats = {"CLOUD": 0.4}
        b.flag_stats = {}
        assert composite_day(day, [a, b]).flag_stats["CLOUD"] == pytest.approx(0.2)

    def test_written_to_sidecar(self, tmp_path, grid):
        cube = LakeCube(tmp_path / "c.zarr", grid)
        f = make_frame(date(2023, 8, 1), 1.0)
        object.__setattr__(f, "flag_stats", {"CLOUD": 0.25, "AC_FAIL": 0.01})
        cube.write([f])
        assert cube.flag_stats_path.exists()
        assert cube.read_flag_stats()["2023-08-01"]["CLOUD"] == pytest.approx(0.25)

    def test_sidecar_accumulates_across_writes(self, tmp_path, grid):
        cube = LakeCube(tmp_path / "c.zarr", grid)
        for day, val in ((date(2023, 8, 1), 0.1), (date(2023, 8, 2), 0.2)):
            f = make_frame(day, 1.0)
            object.__setattr__(f, "flag_stats", {"CLOUD": val})
            cube.write([f])
        stats = cube.read_flag_stats()
        assert sorted(stats) == ["2023-08-01", "2023-08-02"]
        assert stats["2023-08-02"]["CLOUD"] == pytest.approx(0.2)

    def test_absent_sidecar_returns_empty_not_error(self, tmp_path, grid):
        """Cubes ingested before the sidecar existed must degrade quietly."""
        cube = LakeCube(tmp_path / "c.zarr", grid)
        cube.write([make_frame(date(2023, 8, 1), 1.0)])  # no flag stats
        assert cube.read_flag_stats() == {}

    def test_corrupt_sidecar_does_not_crash(self, tmp_path, grid):
        cube = LakeCube(tmp_path / "c.zarr", grid)
        cube.write([make_frame(date(2023, 8, 1), 1.0)])
        cube.flag_stats_path.write_text("{not json", encoding="utf-8")
        assert cube.read_flag_stats() == {}

    def test_stats_are_consumable_by_the_drift_monitor(self, tmp_path, grid):
        """Closes the loop the qc docstring promises."""
        from monitoring.drift import compare_flag_frequencies

        cube = LakeCube(tmp_path / "c.zarr", grid)
        ref = make_frame(date(2023, 8, 1), 1.0)
        object.__setattr__(ref, "flag_stats", {"CLOUD_MARGIN": 0.20, "INLAND_WATER": 0.55})
        cur = make_frame(date(2023, 8, 2), 1.0)
        object.__setattr__(cur, "flag_stats", {"CLOUD_MARGIN": 0.60, "INLAND_WATER": 0.54})
        cube.write([ref])
        cube.write([cur])

        stats = cube.read_flag_stats()
        drifted = [
            m
            for m in compare_flag_frequencies(stats["2023-08-01"], stats["2023-08-02"])
            if m.drifted
        ]
        assert [m.name for m in drifted] == ["flag_CLOUD_MARGIN"]
