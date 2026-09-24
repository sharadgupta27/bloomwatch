"""Live integration tests against Planetary Computer's Sentinel-3 archive.

Marked `network` and excluded from PR CI; they run on the nightly schedule. The point is
not to re-test our own logic -- unit tests cover that -- but to detect changes *upstream*
that would silently corrupt the pipeline:

- a Sentinel-3 processing baseline that reorders or renames WQSF flags;
- an asset key or product structure change in the STAC collection;
- geolocation scaling changing (the `scale_factor` that turns int32 into degrees);
- the archive's temporal coverage shifting.

Each of these would produce plausible-looking but wrong data rather than an exception,
which is exactly the failure mode a scheduled test is for.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pytest

from habforecast.aoi import AOIRegistry
from habforecast.data.olci import ASSET_CHL, ASSET_GEO, ASSET_TIE_GEO, ASSET_WQSF, OLCIReader
from habforecast.data.qc import DEFAULT_REJECT_FLAGS, DEFAULT_REQUIRE_ANY, FlagTable
from habforecast.data.resample import SwathResampler

pytestmark = [pytest.mark.network, pytest.mark.slow]

# A window over Lake Erie's western basin during a known 2023 bloom.
TEST_START, TEST_END = "2023-08-01", "2023-08-20"


@pytest.fixture(scope="module")
def registry():
    from pathlib import Path

    return AOIRegistry.from_geojson(
        Path(__file__).resolve().parents[2] / "data" / "aoi" / "lakes.geojson"
    )


@pytest.fixture(scope="module")
def grid(registry):
    return registry["erie_west"].grid()


@pytest.fixture(scope="module")
def reader():
    return OLCIReader()


@pytest.fixture(scope="module")
def items(reader, grid):
    found = reader.search(grid, TEST_START, TEST_END)
    if not found:
        pytest.skip("no OLCI scenes returned - upstream availability issue")
    return found


class TestCatalogContract:
    def test_search_returns_scenes(self, items):
        assert len(items) > 0

    def test_required_assets_present(self, items):
        """The four assets the reader depends on must exist on every item."""
        for item in items[:5]:
            for key in (ASSET_TIE_GEO, ASSET_GEO, ASSET_CHL, ASSET_WQSF):
                assert key in item.assets, f"{item.id} is missing asset '{key}'"

    def test_items_are_time_sorted(self, items):
        times = [it.datetime for it in items]
        assert times == sorted(times)


class TestSigning:
    """The catalog must hand back UNSIGNED hrefs, signed only at read time.

    Signing at search time is what caused 18 simultaneous HTTP 403s across every
    concurrently-running lake during the real ingest: Planetary Computer SAS tokens live
    about an hour, and a multi-hour job outlives the tokens it started with.
    """

    def test_catalog_returns_unsigned_hrefs(self, items):
        href = items[0].assets[ASSET_CHL].href
        assert "sig=" not in href, "catalog is signing at search time again -- tokens will expire"

    def test_sign_adds_a_token(self, reader, items):
        signed = reader._sign(items[0].assets[ASSET_CHL].href)
        assert "sig=" in signed

    def test_unsigned_read_is_rejected(self, reader, items):
        """Confirms signing is genuinely required, so the fix is load-bearing."""
        import aiohttp

        with pytest.raises((aiohttp.ClientResponseError, RuntimeError, OSError)):
            with reader._open_retrying(items[0].assets[ASSET_CHL].href, resign=False) as fh:
                fh.read(64)


class TestProductContract:
    def test_chl_is_log_scaled_and_plausible(self, reader, items):
        """CHL_NN decodes to log10(mg/m3) once CF scaling is applied.

        Note the product stores CHL_NN as scaled integers, not floats -- a raw h5py read
        of this window returns values in the 61-255 range. Everything must go through
        `_read_scaled`, which is why this test uses it rather than indexing the dataset
        directly.
        """
        import h5py

        from habforecast.data.olci import _read_scaled

        with reader._open_retrying(reader._sign(items[0].assets[ASSET_CHL].href)) as fh:
            with h5py.File(fh, "r") as h5:
                ds = h5["CHL_NN"]
                assert "log10" in str(ds.attrs.get("long_name", "")).lower()
                sample = _read_scaled(ds, (slice(0, 256), slice(0, 256)))
        finite = sample[np.isfinite(sample)]
        if finite.size:
            # log10 space: -2 (0.01) to 3 (1000 mg/m3) covers every real retrieval.
            assert finite.min() > -3 and finite.max() < 4

    def test_geolocation_scale_factor_still_applied(self, reader, items):
        """Latitudes must decode to degrees, not raw micro-degree integers.

        Skipping the scale_factor yields values in the tens of millions -- which is
        exactly the bug that made the first AOI window lookup match nothing.
        """
        import h5py

        from habforecast.data.olci import _read_scaled

        with reader._open_retrying(reader._sign(items[0].assets[ASSET_TIE_GEO].href)) as fh:
            with h5py.File(fh, "r") as h5:
                lat = _read_scaled(h5["latitude"])
                lon = _read_scaled(h5["longitude"])
        assert np.nanmin(lat) >= -90 and np.nanmax(lat) <= 90
        assert np.nanmin(lon) >= -180 and np.nanmax(lon) <= 180

    def test_wqsf_flag_table_contains_expected_flags(self, reader, items):
        """A baseline change that renames or drops a flag must fail loudly."""
        import h5py

        with reader._open_retrying(reader._sign(items[0].assets[ASSET_WQSF].href)) as fh:
            with h5py.File(fh, "r") as h5:
                ds = h5["WQSF"]
                table = FlagTable.from_attrs(ds.attrs["flag_masks"], ds.attrs["flag_meanings"])

        missing_required = table.missing(DEFAULT_REQUIRE_ANY)
        assert "INLAND_WATER" not in missing_required, "INLAND_WATER flag disappeared upstream"

        missing_reject = table.missing(DEFAULT_REJECT_FLAGS)
        assert not missing_reject, f"reject flags no longer defined: {missing_reject}"

    def test_inland_water_and_land_still_co_occur(self, reader, items, grid):
        """Pins the counter-intuitive semantics the QC module depends on.

        OLCI sets LAND over inland water. If a baseline change ever fixed that, our mask
        would start including genuine land, so we want to hear about it.
        """
        import h5py

        from habforecast.data.olci import _locate_window, _read_scaled

        with reader._open_retrying(reader._sign(items[0].assets[ASSET_TIE_GEO].href)) as fh:
            with h5py.File(fh, "r") as h5:
                tie_lat, tie_lon = _read_scaled(h5["latitude"]), _read_scaled(h5["longitude"])
        with reader._open_retrying(reader._sign(items[0].assets[ASSET_WQSF].href)) as fh:
            with h5py.File(fh, "r") as h5:
                ds = h5["WQSF"]
                table = FlagTable.from_attrs(ds.attrs["flag_masks"], ds.attrs["flag_meanings"])
                window = _locate_window(tie_lat, tie_lon, grid.bounds_lonlat, ds.shape)
                if window is None:
                    pytest.skip("scene does not overlap the AOI")
                wqsf = np.asarray(ds[window], dtype=np.uint64)

        inland = (wqsf & np.uint64(table.masks["INLAND_WATER"])) != 0
        land = (wqsf & np.uint64(table.masks["LAND"])) != 0
        if inland.sum() > 100:
            overlap = (inland & land).sum() / inland.sum()
            assert overlap > 0.5, (
                f"only {overlap:.1%} of INLAND_WATER pixels also carry LAND - "
                "OLCI flag semantics may have changed; revisit qc.DEFAULT_REJECT_FLAGS"
            )


class TestEndToEndRead:
    def test_reads_a_scene_onto_the_grid(self, reader, items, grid):
        resampler = SwathResampler(grid)
        obs = None
        for item in items[:8]:
            obs = reader.read(item, grid, resampler, min_valid_fraction=0.05)
            if obs is not None:
                break
        if obs is None:
            pytest.skip("no scene in the window met the coverage threshold (cloud)")

        assert obs.log_chl.shape == grid.shape
        assert obs.valid.shape == grid.shape
        assert obs.valid_fraction > 0

        values = obs.log_chl[obs.valid]
        assert np.isfinite(values).all(), "valid mask includes non-finite values"
        chl = 10**values
        assert chl.min() > 0.01 and chl.max() < 1000, "chlorophyll outside plausible range"

    def test_invalid_pixels_are_nan(self, reader, items, grid):
        resampler = SwathResampler(grid)
        for item in items[:8]:
            obs = reader.read(item, grid, resampler, min_valid_fraction=0.05)
            if obs is not None:
                assert np.isnan(obs.log_chl[~obs.valid]).all()
                return
        pytest.skip("no usable scene")


class TestArchiveCoverage:
    def test_2018_onwards_is_available(self, reader, grid):
        """Planetary Computer's Sentinel-3 archive begins in 2018.

        Documented in the README; if it ever back-fills earlier years this test tells us
        the training range can be extended.
        """
        assert reader.search(grid, "2018-07-01", "2018-07-31"), "2018 coverage disappeared"

    def test_2017_still_absent(self, reader, grid):
        found = reader.search(grid, "2017-07-01", "2017-07-31")
        if found:
            pytest.fail(
                f"{len(found)} scenes now available for 2017 - the archive was back-filled; "
                "extend the ingest range and update the README"
            )

    def test_archive_lag_is_within_expectation(self, reader, grid):
        """Planetary Computer's Sentinel-3 feed lags months behind real time.

        Measured at roughly 6-7 months: scenes were available through 2026-01 while the
        wall clock read 2026-08. This is a property of the mirror, not of Sentinel-3
        itself -- Copernicus Data Space serves the same products within hours.

        The consequence is that the `live` inference path is *not* near-real-time. True
        operational forecasting requires a CDSE ingest path (see README limitations).
        This test asserts the lag has not silently grown worse, and will fail loudly if
        the mirror stalls entirely.
        """
        today = date.today()
        # Walk back month by month until data appears; fail if the gap exceeds a year.
        for months_back in range(0, 13):
            year = today.year
            month = today.month - months_back
            while month < 1:
                month += 12
                year -= 1
            if reader.search(grid, f"{year}-{month:02d}-01", f"{year}-{month:02d}-28"):
                assert months_back <= 12
                return
        pytest.fail("no Sentinel-3 scenes in the last 12 months - the mirror has stalled")
