"""Location resolution and ad-hoc AOI construction -- the front door of the app.

Everything here runs offline. The live fetch that sits behind these pieces is exercised
by `tests/integration/test_adhoc_live.py`, which is network-marked.
"""

from __future__ import annotations

import math

import pytest

from habforecast.aoi import adhoc_aoi, utm_epsg_for
from habforecast.places import GAZETTEER, parse_coordinates, search


class TestParseCoordinates:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("41.75, -83.3", (41.75, -83.3)),
            ("41.75,-83.3", (41.75, -83.3)),
            ("41.75 -83.3", (41.75, -83.3)),
            ("41.75N 83.3W", (41.75, -83.3)),
            ("41.75 N, 83.3 W", (41.75, -83.3)),
            ("-0.3 34.2", (-0.3, 34.2)),
            ("0.3S 34.2E", (-0.3, 34.2)),
            ("0.3 S; 34.2 E", (-0.3, 34.2)),
        ],
    )
    def test_accepts_the_forms_people_actually_paste(self, text, expected):
        got = parse_coordinates(text)
        assert got is not None
        assert got[0] == pytest.approx(expected[0])
        assert got[1] == pytest.approx(expected[1])

    @pytest.mark.parametrize(
        "text", ["lake erie", "", "200, 400", "91, 0", "0, 181", "41.75", "not a place"]
    )
    def test_rejects_non_coordinates_so_the_caller_can_fall_back_to_search(self, text):
        assert parse_coordinates(text) is None

    def test_a_hemisphere_letter_overrides_a_stray_sign(self):
        """ "-41.75N" is contradictory; the letter is the more deliberate signal."""
        assert parse_coordinates("-41.75N, 83.3E")[0] == pytest.approx(41.75)


class TestGazetteerSearch:
    def test_word_prefix_matches_rank_above_buried_matches(self):
        names = [p.name for p in search("erie")]
        assert names and all("Erie" in n for n in names)

    def test_is_accent_and_case_insensitive(self):
        assert search("VANERN") and search("vanern")

    def test_unknown_text_returns_nothing_rather_than_a_wrong_guess(self):
        assert search("qwertyuiop") == []

    def test_every_entry_is_a_plausible_point_on_earth(self):
        for p in GAZETTEER:
            assert -90 <= p.lat <= 90, p.name
            assert -180 <= p.lon <= 180, p.name
            # Everything in the list is a water body Sentinel-3 can see, so it must be
            # inside the UTM domain the ad-hoc grid builder supports.
            assert -80 <= p.lat <= 84, p.name

    def test_names_are_unique(self):
        names = [p.name for p in GAZETTEER]
        assert len(names) == len(set(names))


class TestUtmZone:
    @pytest.mark.parametrize(
        ("lat", "lon", "epsg"),
        [
            (41.75, -83.30, 32617),  # Lake Erie
            (-0.30, 34.20, 32736),  # Lake Victoria, southern hemisphere
            (46.85, 17.75, 32633),  # Lake Balaton
            (0.0, 0.0, 32631),  # null island, zone 31 north
        ],
    )
    def test_picks_the_right_zone(self, lat, lon, epsg):
        assert utm_epsg_for(lat, lon) == epsg

    @pytest.mark.parametrize("lat", [85.0, -81.0, 90.0])
    def test_refuses_polar_latitudes_instead_of_distorting_silently(self, lat):
        with pytest.raises(ValueError, match="UTM domain"):
            utm_epsg_for(lat, 0.0)

    def test_longitude_180_does_not_overflow_to_zone_61(self):
        assert utm_epsg_for(0.0, 180.0) == 32660


class TestAdhocAoi:
    def test_grid_covers_the_requested_radius(self):
        aoi = adhoc_aoi(41.75, -83.30, radius_km=12)
        grid = aoi.grid()
        # 12 km radius -> 24 km across -> 80 cells at 300 m. The grid comes out a little
        # larger: the AOI is a square in UTM, its lat/lon bounding box is the envelope of
        # that square's corners, and re-projecting that envelope adds a few cells. Erring
        # outward is the right direction -- the requested area is always covered.
        assert 80 <= grid.width <= 90
        assert 80 <= grid.height <= 90

    @pytest.mark.parametrize("lat", [0.0, 45.0, 60.0, 70.0])
    def test_the_area_is_the_same_at_every_latitude(self, lat):
        """Built in metres, not degrees.

        A degree of longitude is 111 km at the equator and 38 km at 70 deg N. Adding a
        fixed number of degrees would hand a user near the pole an area several times
        larger than the one they asked for.
        """
        grid = adhoc_aoi(lat, 10.0, radius_km=12).grid()
        span_km = grid.width * grid.resolution / 1000.0
        assert span_km == pytest.approx(24.0, abs=1.5)

    def test_the_bbox_contains_the_requested_centre(self):
        for lat, lon in [(41.75, -83.3), (-0.3, 34.2), (-33.9, 151.2), (60.2, 24.9)]:
            west, south, east, north = adhoc_aoi(lat, lon, 15).bbox
            assert west < lon < east
            assert south < lat < north

    def test_a_wider_radius_makes_a_bigger_grid(self):
        small = adhoc_aoi(41.75, -83.3, 5).grid()
        large = adhoc_aoi(41.75, -83.3, 25).grid()
        assert large.width > small.width * 3

    def test_rejects_a_nonsense_radius(self):
        with pytest.raises(ValueError, match="radius_km"):
            adhoc_aoi(41.75, -83.3, 0)

    def test_claims_no_bloom_season_it_does_not_know(self):
        """A registry lake has a researched bloom window. An arbitrary point does not,
        and inventing one would be a fabricated claim about a real place."""
        aoi = adhoc_aoi(41.75, -83.3, 12)
        assert aoi.bloom_season == (1, 12)
        assert all(aoi.in_bloom_season(m) for m in range(1, 13))

    def test_grid_is_square_within_a_cell(self):
        grid = adhoc_aoi(-0.3, 34.2, 12).grid()
        assert abs(grid.width - grid.height) <= 2

    def test_lonlat_of_the_grid_centre_is_close_to_the_request(self):
        lat, lon = 46.85, 17.75
        grid = adhoc_aoi(lat, lon, 12).grid()
        lons, lats = grid.lonlat
        cy, cx = grid.height // 2, grid.width // 2
        assert math.isclose(float(lats[cy, cx]), lat, abs_tol=0.05)
        assert math.isclose(float(lons[cy, cx]), lon, abs_tol=0.05)


class TestAdhocAoiFromBbox:
    """The path a drawn rectangle takes into the model."""

    def test_builds_a_grid_covering_the_drawn_box(self):
        from habforecast.aoi import adhoc_aoi_from_bbox

        aoi = adhoc_aoi_from_bbox(-83.5, 41.6, -83.1, 41.9, "drawn")
        grid = aoi.grid()
        assert aoi.bbox == (-83.5, 41.6, -83.1, 41.9)
        assert grid.width > 100 and grid.height > 100
        assert aoi.utm_epsg == 32617

    def test_refuses_an_area_too_large_to_serve(self):
        """A box dragged across an ocean is tens of millions of cells and a hundred
        granule reads. The request would not fail, it would consume the container until
        something killed it -- so it is refused with a number the user can act on."""
        from habforecast.aoi import MAX_ADHOC_SIDE_KM, adhoc_aoi_from_bbox

        with pytest.raises(ValueError, match=r"limit is \d+ km"):
            adhoc_aoi_from_bbox(-84.0, 41.0, -82.0, 43.0)
        # And one just inside the limit is accepted.
        ok = adhoc_aoi_from_bbox(-83.5, 41.6, -83.1, 41.9)
        assert ok.grid().width * ok.grid().resolution / 1000 < MAX_ADHOC_SIDE_KM

    @pytest.mark.parametrize(
        "bad",
        [(-83.1, 41.6, -83.5, 41.9), (-83.5, 41.9, -83.1, 41.6), (-83.5, 41.6, -83.5, 41.9)],
    )
    def test_rejects_a_degenerate_or_inverted_box(self, bad):
        from habforecast.aoi import adhoc_aoi_from_bbox

        with pytest.raises(ValueError, match="west, south, east, north"):
            adhoc_aoi_from_bbox(*bad)

    def test_picks_the_utm_zone_from_the_centre_not_a_corner(self):
        """A box straddling a zone boundary must still project consistently."""
        from habforecast.aoi import adhoc_aoi_from_bbox, utm_epsg_for

        west, east = -84.2, -83.8  # zone 16/17 boundary sits at -84
        aoi = adhoc_aoi_from_bbox(west, 41.6, east, 41.9)
        assert aoi.utm_epsg == utm_epsg_for(41.75, (west + east) / 2)

    def test_claims_no_bloom_season_for_a_drawn_area(self):
        from habforecast.aoi import adhoc_aoi_from_bbox

        assert adhoc_aoi_from_bbox(-83.5, 41.6, -83.1, 41.9).bloom_season == (1, 12)


class TestAnchorSearch:
    """Finding the newest imagery date: small windows first, never overlapping."""

    def test_spans_start_small_and_cover_the_lookback_exactly(self):
        from habforecast.inference.adhoc import _search_spans

        spans = list(_search_spans(400))
        assert spans[:3] == [3, 7, 20]
        assert max(spans) == 30
        assert sum(spans) == 400
        assert list(_search_spans(5)) == [3, 2]

    def test_recent_imagery_is_found_in_the_first_small_window(self, monkeypatch):
        from datetime import date, datetime

        from habforecast.data import olci
        from habforecast.inference import adhoc

        asked = []

        class Item:
            def __init__(self, d):
                self.datetime = datetime(d.year, d.month, d.day, 17)

        class Reader:
            def search(self, grid, start, end):
                asked.append((start, end))
                return [Item(date(2026, 9, 22))] if end >= "2026-09-22" else []

        monkeypatch.setattr(olci, "default_reader", Reader)
        monkeypatch.setattr(adhoc, "_anchor_cache", {})
        aoi = adhoc.build_aoi(bbox=(-83.5, 41.6, -83.1, 41.9))
        assert adhoc.resolve_anchor(None, aoi, date(2026, 9, 23)) == date(2026, 9, 22)
        assert asked == [("2026-09-20", "2026-09-23")]

    def test_a_months_old_archive_is_still_reached_without_overlap(self, monkeypatch):
        import itertools
        from datetime import date, datetime, timedelta

        from habforecast.data import olci
        from habforecast.inference import adhoc

        asked = []
        newest = date(2026, 2, 25)

        taken = datetime(2026, 2, 25, 17)

        class Item:
            datetime = taken

        class Reader:
            def search(self, grid, start, end):
                asked.append((date.fromisoformat(start), date.fromisoformat(end)))
                lo, hi = asked[-1]
                return [Item()] if lo <= newest <= hi else []

        monkeypatch.setattr(olci, "default_reader", Reader)
        monkeypatch.setattr(adhoc, "_anchor_cache", {})
        aoi = adhoc.build_aoi(bbox=(-83.5, 41.6, -83.1, 41.9))
        assert adhoc.resolve_anchor(None, aoi, date(2026, 9, 23)) == newest
        for (lo, _), (_, hi) in itertools.pairwise(asked):
            assert hi == lo, "consecutive windows must meet, not overlap or leave gaps"
        assert all(hi - lo <= timedelta(days=30) for lo, hi in asked)
