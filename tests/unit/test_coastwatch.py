"""The ERDDAP client decides which pixels reach the model, so its addressing must be exact.

Every case here is a mistake that was actually made against the live service and cost a
silent empty result rather than an error:

* a time bound one second past the axis maximum -> 404 for the whole request;
* the two satellites holding different newest days -> a window built from the later one
  killed the request to the earlier one, which was the satellite with the data;
* a bare ``.squeeze()`` collapsing a one-cell-wide subset to 1-D;
* a sector family that mixes 300 m tiles with coarse continental overviews under one
  naming scheme, so the wrong pick silently halves the resolution.

No network. The griddap URL and the regridding are pure functions of the inputs.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pytest

from habforecast.data.coastwatch import (
    PRODUCTS,
    Sector,
    _griddap_url,
    _nearest_index,
    sectors_for,
)


def sector(
    ds_id="noaacwS3AOLCIchlaSectorKFDaily",
    west=20.0,
    south=-15.0,
    east=40.0,
    north=0.0,
    min_time="2026-05-28T01:55:34Z",
    max_time="2026-08-26T01:24:51Z",
    res=0.0025,
):
    return Sector(ds_id, west, south, east, north, min_time, max_time, res)


BBOX = (34.0, -0.5, 34.4, -0.1)


class TestProductLadder:
    def test_every_product_is_addressable(self):
        for key, spec in PRODUCTS.items():
            assert spec.resolution_m > 0
            assert spec.sectored or spec.datasets, f"{key} has no way to be addressed"
            assert spec.record, f"{key} does not say what period it covers"

    def test_resolutions_are_distinct_and_ordered(self):
        by_res = sorted({p.resolution_m for p in PRODUCTS.values()})
        assert by_res == [300.0, 750.0, 2000.0, 4000.0]


class TestGriddapURL:
    def test_includes_the_altitude_axis(self):
        """The grids are (time, altitude, lat, lon); omitting altitude is a 404."""
        url = _griddap_url(sector(), BBOX, date(2026, 8, 1), date(2026, 8, 14))
        assert "[0]" in url, url

    def test_latitude_is_ordered_north_to_south(self):
        url = _griddap_url(sector(), BBOX, date(2026, 8, 1), date(2026, 8, 14))
        lat_part = url.split("[0]")[1]
        north, south = lat_part.split("][")[0].strip("[]()").split("):(")
        assert float(north) > float(south), "latitude axis is stored descending"

    def test_a_request_past_the_axis_end_uses_the_axis_bound_verbatim(self):
        """ERDDAP compares the timestamp, not the day: 08-26T23:59 is out of range for an
        axis ending 08-26T01:24, and it refuses the whole request rather than clamping."""
        s = sector(max_time="2026-08-26T01:24:51Z")
        url = _griddap_url(s, BBOX, date(2026, 8, 1), date(2026, 8, 26))
        assert "2026-08-26T01%3A24%3A51Z" in url or "2026-08-26T01:24:51Z" in url
        assert "23%3A59%3A59" not in url and "23:59:59" not in url

    def test_a_request_before_the_axis_start_uses_the_axis_bound(self):
        s = sector(min_time="2026-05-28T01:55:34Z")
        url = _griddap_url(s, BBOX, date(2026, 1, 1), date(2026, 8, 1))
        assert "2026-05-28T01" in url.replace("%3A", ":")

    def test_a_window_entirely_outside_the_record_is_not_requested(self):
        s = sector(min_time="2026-05-28T00:00:00Z", max_time="2026-08-26T00:00:00Z")
        assert _griddap_url(s, BBOX, date(2020, 1, 1), date(2020, 2, 1)) is None
        assert _griddap_url(s, BBOX, date(2027, 1, 1), date(2027, 2, 1)) is None

    def test_the_bbox_is_clipped_to_the_sector(self):
        s = sector(west=20.0, east=40.0, south=-15.0, north=0.0)
        url = _griddap_url(s, (10.0, -20.0, 50.0, 10.0), date(2026, 8, 1), date(2026, 8, 2))
        decoded = url.replace("%3A", ":")
        assert "(20.0)" in decoded or "(20)" in decoded
        assert "(50" not in decoded, "an unclipped east edge would 404"

    def test_an_aoi_beside_the_sector_yields_no_request(self):
        s = sector(west=20.0, east=40.0, south=-15.0, north=0.0)
        assert (
            _griddap_url(s, (60.0, -10.0, 70.0, -5.0), date(2026, 8, 1), date(2026, 8, 2)) is None
        )


class TestSectorSelection:
    def test_one_sector_per_satellite_is_chosen(self):
        pool = [
            sector("noaacwS3AOLCIchlaSectorKFDaily"),
            sector("noaacwS3BOLCIchlaSectorKFDaily"),
        ]
        got = sectors_for(BBOX, pool, "olci-300m")
        assert {s.satellite for s in got} == {"S3A", "S3B"}

    def test_the_coarse_overview_tile_is_rejected(self):
        """The sector family mixes 300 m tiles with continental overviews. Picking the
        overview silently returns a fifteen-times coarser field under the same name."""
        fine = sector("noaacwS3AOLCIchlaSectorKFDaily", res=0.0025)
        coarse = sector(
            "noaacwS3AOLCIchlaSectorXXDaily", west=-20, east=60, south=-40, north=40, res=0.0375
        )
        got = sectors_for(BBOX, [fine, coarse], "olci-300m")
        assert [s.dataset_id for s in got] == ["noaacwS3AOLCIchlaSectorKFDaily"]

    def test_a_global_product_is_not_filtered_by_tile_size(self):
        """Global grids are one coarse dataset by design; the tile-size filter must not
        throw them away."""
        glob = sector(
            "noaacwS3AOLCIchlaDaily", west=-180, east=180, south=-90, north=90, res=0.0375
        )
        assert sectors_for(BBOX, [glob], "olci-4km")

    def test_no_coverage_returns_nothing_rather_than_a_wrong_tile(self):
        far = sector(west=100.0, east=120.0, south=20.0, north=40.0)
        assert sectors_for(BBOX, [far], "olci-300m") == []


class TestNearestIndex:
    def test_maps_each_target_to_its_closest_source_cell(self):
        src = np.array([10.0, 11.0, 12.0, 13.0])
        idx = _nearest_index(np.array([10.4, 11.6, 12.9]), src)
        assert list(idx) == [0, 2, 3]

    def test_handles_a_descending_source_axis(self):
        """Latitude comes back north-to-south; a sorted-ascending assumption inverts the map."""
        asc = np.array([10.0, 11.0, 12.0, 13.0])
        desc = asc[::-1].copy()
        targets = np.array([10.4, 12.9])
        got_asc = asc[_nearest_index(targets, asc)]
        got_desc = desc[_nearest_index(targets, desc)]
        assert np.allclose(got_asc, got_desc)

    def test_targets_beyond_the_ends_clamp_to_the_edge_cell(self):
        src = np.array([10.0, 11.0, 12.0])
        idx = _nearest_index(np.array([-5.0, 99.0]), src)
        assert list(idx) == [0, 2]

    def test_a_single_cell_axis_does_not_raise(self):
        idx = _nearest_index(np.array([5.0, 6.0]), np.array([5.5]))
        assert list(idx) == [0, 0]


class TestSectorMetadata:
    @pytest.mark.parametrize(
        ("ds_id", "expected"),
        [
            ("noaacwS3AOLCIchlaSectorKFDaily", "S3A"),
            ("noaacwS3BOLCIchlaSectorKFDaily", "S3B"),
            ("noaacwNPPVIIRSSCIchlaSectorVYDaily", "NPP"),
        ],
    )
    def test_satellite_is_read_from_the_dataset_id(self, ds_id, expected):
        assert sector(ds_id).satellite == expected

    def test_an_unparseable_timestamp_is_none_not_a_crash(self):
        assert sector(max_time="").newest is None
        assert sector(min_time="not a date").oldest is None

    def test_covers_is_an_overlap_test_not_a_containment_test(self):
        """An AOI straddling a tile edge must still select that tile."""
        s = sector(west=20.0, east=40.0, south=-15.0, north=0.0)
        assert s.covers(39.0, -1.0, 41.0, 1.0)
        assert not s.covers(41.0, -1.0, 45.0, 1.0)


def test_withdrawn_product_is_source_unavailable_not_a_raw_404(monkeypatch, tmp_path):
    """A 404 on the dataset catalogue means the product is gone (olci-300m, 2026-09-23),
    which callers must see as an unavailable source they can fall back from."""
    import urllib.error

    from habforecast.data import coastwatch

    def gone(*_a, **_k):
        raise urllib.error.HTTPError("u", 404, "Not Found", {}, None)

    monkeypatch.setattr(coastwatch, "_get", gone)
    monkeypatch.setattr(coastwatch, "_withdrawn", {})
    with pytest.raises(coastwatch.SourceUnavailableError, match="withdrawn"):
        coastwatch.sector_index(tmp_path, refresh=True, product="olci-300m")


def test_withdrawn_product_is_not_asked_about_again(monkeypatch, tmp_path):
    """Re-asking cost 33 s then 2.5 s per request, all under "locating imagery"."""
    import urllib.error

    from habforecast.data import coastwatch

    calls = []

    def gone(*_a, **_k):
        calls.append(1)
        raise urllib.error.HTTPError("u", 404, "Not Found", {}, None)

    monkeypatch.setattr(coastwatch, "_get", gone)
    monkeypatch.setattr(coastwatch, "_withdrawn", {})
    for _ in range(3):
        with pytest.raises(coastwatch.SourceUnavailableError):
            coastwatch.sector_index(tmp_path, product="olci-300m")
    assert len(calls) == 1
