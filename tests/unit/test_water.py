"""Classifying the water body: the calls that decide which product gets fetched.

These are checked against real coordinates rather than synthetic geometry, because the
failure mode that matters is not "the ray cast is wrong" -- it is "Lake Erie was called
ocean", and only a real lake can catch that. The set below is chosen to cover the cases
where a naive land/sea test gives the wrong answer:

* a Great Lake, which is huge, far inland, and reads as sea to anything that only knows
  "is there water here";
* the Caspian, which Natural Earth maps *as* ocean and which is hydrologically a lake;
* the Curonian Lagoon, which is marine by hydrology and optically complex by physics --
  the case that forces setting and optical class to be two separate answers;
* mid-ocean near the antimeridian, where the polygon's artificial map edge would look
  like a coastline if it were not filtered out.

No network. Skipped in full when the reference file has not been built.
"""

from __future__ import annotations

import numpy as np
import pytest

from habforecast import water

pytestmark = pytest.mark.skipif(
    not water._reference_path().exists(),
    reason="coastline reference not built (scripts/build_coast_reference.py)",
)

# west, south, east, north
AREAS = {
    "erie_west": (-83.6, 41.55, -82.9, 41.95),
    "taihu": (119.9, 30.9, 120.6, 31.5),
    "okeechobee": (-81.1, 26.7, -80.6, 27.2),
    "victoria_ne": (33.9, -0.6, 34.6, 0.1),
    "balaton": (17.4, 46.7, 18.1, 47.0),
    "chesapeake": (-76.5, 37.8, -76.0, 38.3),
    "curonian": (20.9, 55.2, 21.4, 55.6),
    "florida_shelf": (-83.2, 26.6, -82.6, 27.2),
    "gulf_open": (-90.3, 25.7, -89.7, 26.3),
    "atlantic_open": (-45.3, 39.7, -44.7, 40.3),
    "pacific_dateline": (-179.3, -0.3, -178.7, 0.3),
    "caspian": (50.7, 40.7, 51.3, 41.3),
}


class TestSetting:
    @pytest.mark.parametrize("area", ["erie_west", "taihu", "okeechobee", "victoria_ne", "balaton"])
    def test_a_lake_is_inland_however_large(self, area):
        assert water.classify(AREAS[area]).setting == water.INLAND

    @pytest.mark.parametrize("area", ["chesapeake", "curonian", "florida_shelf"])
    def test_a_bay_lagoon_or_shelf_is_coastal(self, area):
        assert water.classify(AREAS[area]).setting == water.COASTAL

    @pytest.mark.parametrize("area", ["gulf_open", "atlantic_open"])
    def test_water_beyond_the_shelf_is_open_ocean(self, area):
        assert water.classify(AREAS[area]).setting == water.OPEN_OCEAN

    def test_the_caspian_is_inland_although_the_map_calls_it_sea(self):
        """Endorheic: no outflow, no tidal exchange, and its own optics."""
        got = water.classify(AREAS["caspian"])
        assert got.setting == water.INLAND
        assert got.enclosed_sea == "Caspian Sea"
        assert "endorheic" in got.evidence[0]


class TestOpticalClass:
    def test_a_lagoon_is_marine_and_still_optically_complex(self):
        """The case that makes setting and optical class two answers instead of one."""
        got = water.classify(AREAS["curonian"])
        assert got.setting == water.COASTAL
        assert got.optical_class == water.CASE_2

    def test_open_ocean_is_case_1(self):
        assert water.classify(AREAS["gulf_open"]).optical_class == water.CASE_1

    def test_every_lake_is_case_2(self):
        for area in ("erie_west", "taihu", "victoria_ne"):
            assert water.classify(AREAS[area]).optical_class == water.CASE_2


class TestCoastDistance:
    def test_a_far_inland_lake_is_hundreds_of_kilometres_from_the_sea(self):
        got = water.classify(AREAS["erie_west"])
        assert 400 < got.coast_distance_km < 800

    def test_a_bay_touches_the_shore(self):
        assert water.classify(AREAS["chesapeake"]).coast_distance_km < 5

    def test_the_map_edge_is_not_mistaken_for_a_coastline(self):
        """The ocean polygon closes along +/-180 degrees; that seam is not shore."""
        got = water.classify(AREAS["pacific_dateline"])
        assert got.setting == water.OPEN_OCEAN
        assert got.coast_distance_km > 200

    def test_the_distance_is_the_nearest_approach_not_the_centre(self):
        """An area whose corner reaches the shore is a coastal problem all the same."""
        west, south, east, north = -76.6, 37.6, -75.4, 38.4  # Chesapeake mouth outward
        body = water.classify((west, south, east, north))
        centre_only = water._coast_km([((south + north) / 2, (west + east) / 2)], water._load())[0]
        assert body.coast_distance_km <= centre_only


class TestMixedAreas:
    def test_an_area_across_the_shoreline_reports_itself_as_mixed(self):
        got = water.classify(AREAS["chesapeake"])
        assert got.mixed
        assert 0 < got.marine_fraction < 1

    def test_a_lake_is_not_mixed(self):
        assert not water.classify(AREAS["erie_west"]).mixed


class TestDegradation:
    def test_a_missing_reference_gives_unknown_rather_than_a_guess(self, monkeypatch, tmp_path):
        monkeypatch.setattr(water, "REFERENCE", tmp_path / "absent.npz")
        monkeypatch.setattr(water, "_ref", None)
        monkeypatch.setattr(water, "_ref_missing", False)
        got = water.classify(AREAS["erie_west"])
        assert got.setting == water.UNKNOWN
        assert got.optical_class == water.UNKNOWN
        assert "not installed" in got.evidence[0]

    def test_an_unclassified_area_still_reports_its_size(self, monkeypatch, tmp_path):
        monkeypatch.setattr(water, "REFERENCE", tmp_path / "absent.npz")
        monkeypatch.setattr(water, "_ref", None)
        monkeypatch.setattr(water, "_ref_missing", False)
        assert water.classify(AREAS["erie_west"]).side_km > 10


class TestGeometry:
    def test_side_length_is_the_longer_edge_in_kilometres(self):
        # One degree of latitude is about 110.6 km everywhere.
        assert water.bbox_side_km(0.0, 0.0, 0.1, 1.0) == pytest.approx(110.6, abs=1.0)

    def test_longitude_shrinks_towards_the_pole(self):
        equator = water.bbox_side_km(0.0, 0.0, 1.0, 0.01)
        north = water.bbox_side_km(0.0, 60.0, 1.0, 60.01)
        assert north == pytest.approx(equator * np.cos(np.radians(60.0)), rel=0.01)

    def test_the_serialised_form_carries_the_evidence(self):
        got = water.classify(AREAS["chesapeake"]).as_dict()
        assert got["setting"] == water.COASTAL
        assert got["mixed"] is True
        assert got["evidence"], "a classification with no evidence cannot be argued with"
