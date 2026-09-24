"""The in-situ lookup decides what a user is told about verification, so it must not lie.

Two failures would be worse than having no feature at all, and most of these cases exist
to prevent one of them:

* **presenting a 2015 sample as if it verified a 2026 forecast** -- the window and the
  same-season-other-year lists are built separately and must stay that way;
* **mixing chlorophyll with microcystin** -- different quantities, different units, and
  the whole reason for carrying the toxin is that chlorophyll is only its proxy.

The rest is proximity arithmetic, which is exact and therefore worth pinning: a radius is
a circle, not a bounding box, and it has to survive the antimeridian and the poles.
"""

from __future__ import annotations

import csv
import gzip
import json
from datetime import date
from pathlib import Path

import pytest

from habforecast.insitu import (
    CHLOROPHYLL,
    COLUMNS,
    MICROCYSTIN,
    WHO_MICROCYSTIN_UGL,
    InSituIndex,
    _doy_offset,
    context,
    load_index,
    validation_sites,
)

ANCHOR = date(2026, 8, 20)
START, END = date(2026, 8, 7), date(2026, 8, 27)


def row(lat, lon, day, value=10.0, variable=CHLOROPHYLL, source="GLORIA", site="X", flag=""):
    return {
        "source": source,
        "site": site,
        "variable": variable,
        "date": day,
        "lat": lat,
        "lon": lon,
        "value": value,
        "flag": flag,
    }


@pytest.fixture
def index():
    return InSituIndex(
        [
            row(41.700, -83.200, "2026-08-12", 44.0),  # inside the window, 0 km
            row(41.750, -83.200, "2015-08-14", 61.0),  # same season, another year
            row(41.700, -83.260, "2019-02-03", 3.0),  # nearby, wrong season
            row(41.705, -83.205, "2026-08-13", 12.5, MICROCYSTIN, "EPA WQP", "Beach"),
            row(42.400, -83.200, "2026-08-12", 99.0),  # ~78 km away: never in range
        ]
    )


class TestProximity:
    def test_results_come_back_nearest_first(self, index):
        got = index.near(41.700, -83.200, 10.0)
        assert [o.distance_km for o in got] == sorted(o.distance_km for o in got)

    def test_the_radius_is_a_circle_not_the_prefilter_box(self):
        """The box prefilter is a superset; a corner of it is 12 km from the centre."""
        corner = InSituIndex([row(41.780, -83.100, "2026-08-12")])
        assert corner.near(41.700, -83.200, 10.0) == []
        assert len(corner.near(41.700, -83.200, 13.0)) == 1

    def test_far_points_are_excluded(self, index):
        assert all(o.lat < 42.0 for o in index.near(41.700, -83.200, 10.0))
        assert any(o.lat > 42.0 for o in index.near(41.700, -83.200, 100.0))

    def test_variable_filter_never_mixes_quantities(self, index):
        chl = index.near(41.700, -83.200, 10.0, variable=CHLOROPHYLL)
        tox = index.near(41.700, -83.200, 10.0, variable=MICROCYSTIN)
        assert {o.variable for o in chl} == {CHLOROPHYLL}
        assert {o.variable for o in tox} == {MICROCYSTIN}
        assert chl and tox

    def test_the_antimeridian_does_not_split_a_neighbourhood(self):
        """179.99E and 179.99W are 2 km apart; a plain longitude difference calls it 40000."""
        idx = InSituIndex([row(0.0, 179.99, "2026-08-12")])
        assert len(idx.near(0.0, -179.99, 10.0)) == 1

    def test_a_pole_query_does_not_divide_by_zero(self):
        """cos(90 deg) is zero, and the longitude half-width is radius/(111*cos)."""
        idx = InSituIndex([row(89.999, 10.0, "2026-08-12")])
        got = idx.near(90.0, 0.0, 5.0)
        assert len(got) == 1 and got[0].distance_km < 5.0

    def test_an_empty_index_answers_rather_than_crashes(self):
        assert InSituIndex([]).near(0.0, 0.0, 10.0) == []


class TestSeasonalOffset:
    def test_january_is_near_december_not_a_year_from_it(self):
        assert _doy_offset(date(2020, 1, 2), date(2019, 12, 24)) == 9

    def test_the_sign_says_which_side(self):
        assert _doy_offset(date(2019, 12, 20), date(2019, 12, 24)) == -4


class TestContext:
    def test_a_true_matchup_lands_in_the_window(self, index):
        got = context(41.700, -83.200, START, END, radius_km=10.0, index=index)
        dates = {o["date"] for o in got["window"]}
        assert dates == {"2026-08-12", "2026-08-13"}
        assert got["note"] == "2 in-situ measurements fall inside this window."

    def test_other_years_never_leak_into_the_window(self, index):
        got = context(41.700, -83.200, START, END, radius_km=10.0, index=index)
        assert all(o["date"].startswith("2026") for o in got["window"])
        assert {o["date"] for o in got["seasonal"]} == {"2015-08-14"}

    def test_seasonal_points_carry_the_date_they_are_drawn_at_and_the_one_they_are_from(
        self, index
    ):
        """Plotting a 2015 sample on a 2026 axis is only honest if both dates travel with it."""
        got = context(41.700, -83.200, START, END, radius_km=10.0, index=index)
        point = got["seasonal"][0]
        assert point["date"] == "2015-08-14"
        assert point["plotted_date"].startswith("2026-")
        assert point["day_offset"] == 7

    def test_the_wrong_season_is_counted_but_not_plotted(self, index):
        got = context(41.700, -83.200, START, END, radius_km=10.0, index=index)
        assert got["total"] == 4
        assert "2019-02-03" not in {o["date"] for o in got["window"] + got["seasonal"]}

    def test_the_note_is_written_for_a_reader_not_a_log(self, index):
        """It is rendered verbatim in a panel, so "1 sample(s) were" is a visible defect."""
        one = context(41.700, -83.200, date(2026, 8, 12), date(2026, 8, 12), 1.0, index=index)
        assert one["note"] == "1 in-situ measurement falls inside this window."

    def test_nothing_nearby_says_so_plainly(self, index):
        got = context(0.0, 0.0, START, END, radius_km=10.0, index=index)
        assert got["total"] == 0
        assert got["window"] == [] and got["seasonal"] == []
        assert "No public in-situ measurement" in got["note"]

    def test_the_summary_reports_each_quantity_separately(self, index):
        got = context(41.700, -83.200, START, END, radius_km=10.0, index=index)
        by = {v["variable"]: v for v in got["variables"]}
        assert by[CHLOROPHYLL]["unit"] == "mg/m3"
        assert by[MICROCYSTIN]["unit"] == "ug/L"
        assert by[CHLOROPHYLL]["first"] == "2015-08-14"

    def test_a_who_exceedance_is_flagged(self, index):
        got = context(41.700, -83.200, START, END, radius_km=10.0, index=index)
        tox = next(o for o in got["window"] if o["variable"] == MICROCYSTIN)
        assert tox["value"] >= WHO_MICROCYSTIN_UGL
        assert tox["exceeds_who"] is True

    def test_a_missing_index_degrades_instead_of_raising(self, tmp_path):
        got = context(41.7, -83.2, START, END, path=tmp_path / "absent.csv.gz")
        assert got["available"] is False
        assert "build_insitu_index" in got["note"]


class TestIndexFile:
    def test_a_gzipped_index_round_trips(self, tmp_path):
        path = tmp_path / "insitu.csv.gz"
        with gzip.open(path, "wt", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=COLUMNS)
            w.writeheader()
            w.writerow(row(41.7, -83.2, "2026-08-12", 44.0))
        load_index.cache_clear()
        idx = load_index(path)
        assert len(idx) == 1
        assert idx.near(41.7, -83.2, 1.0)[0].value == 44.0

    def test_an_absent_index_is_none_not_an_error(self, tmp_path: Path):
        load_index.cache_clear()
        assert load_index(tmp_path / "nope.csv.gz") is None


class TestValidationSites:
    """The catalogue behind the sidebar: where a forecast can be checked against a bottle.

    Grouping measurements into places is the part that goes wrong quietly. Naming alone
    put every Silver Lake in the United States into one 400 km "site" and every Atlantic
    sample into one entry spanning an ocean, which is a list that looks authoritative and
    sends people nowhere.
    """

    @pytest.fixture
    def aoi_file(self, tmp_path):
        doc = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "properties": {"lake_id": "erie_west", "name": "Erie West", "split": "train"},
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [
                            [
                                [-83.5, 41.5],
                                [-83.0, 41.5],
                                [-83.0, 42.0],
                                [-83.5, 42.0],
                                [-83.5, 41.5],
                            ]
                        ],
                    },
                }
            ],
        }
        p = tmp_path / "lakes.geojson"
        p.write_text(json.dumps(doc), encoding="utf-8")
        return p

    def write_index(self, tmp_path, rows):
        path = tmp_path / "insitu.csv.gz"
        with gzip.open(path, "wt", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=COLUMNS)
            w.writeheader()
            for r in rows:
                w.writerow(r)
        load_index.cache_clear()
        validation_sites.cache_clear()
        return path

    def test_a_registry_lake_is_listed_with_what_was_measured_in_it(self, tmp_path, aoi_file):
        rows = [row(41.7, -83.2, f"2020-08-0{i % 9 + 1}", 20.0, site="Erie") for i in range(12)]
        idx = self.write_index(tmp_path, rows)
        sites = validation_sites(aoi_path=aoi_file, index_path=idx)
        erie = next(s for s in sites if s["id"] == "erie_west")
        assert erie["kind"] == "modelled"
        assert erie["chlorophyll_a"] == 12
        assert erie["split"] == "train"

    def test_a_registry_lake_with_no_field_data_is_still_offered(self, tmp_path, aoi_file):
        """It is where the model was evaluated; its absence from the list would mislead."""
        idx = self.write_index(tmp_path, [row(0.0, 0.0, "2020-08-01", 5.0, site="Nowhere")])
        sites = validation_sites(aoi_path=aoi_file, index_path=idx)
        erie = next(s for s in sites if s["id"] == "erie_west")
        assert erie["n"] == 0 and erie["kind"] == "modelled"

    def test_one_name_in_two_places_becomes_two_sites(self, tmp_path, aoi_file):
        """Iowa's Silver Lake and Minnesota's are not one water body 400 km across."""
        rows = [row(43.0, -95.0, "2020-08-01", 12.0, site="Silver Lake") for _ in range(10)]
        rows += [row(46.5, -94.0, "2020-08-02", 9.0, site="Silver Lake") for _ in range(10)]
        idx = self.write_index(tmp_path, rows)
        silver = [
            s
            for s in validation_sites(aoi_path=aoi_file, index_path=idx)
            if s["name"].startswith("Silver Lake")
        ]
        assert len(silver) == 2
        assert all(s["span_km"] < 50 for s in silver), "a site must be one place"
        # Ambiguous names are qualified, so the list never shows the same label twice.
        assert len({s["name"] for s in silver}) == 2
        assert all("°" in s["name"] for s in silver)

    def test_a_lake_split_across_a_cell_boundary_is_put_back_together(self, tmp_path, aoi_file):
        rows = [row(44.98, -93.0, "2020-08-01", 12.0, site="Border Lake") for _ in range(6)]
        rows += [row(45.02, -93.0, "2020-08-02", 12.0, site="Border Lake") for _ in range(6)]
        idx = self.write_index(tmp_path, rows)
        got = [
            s
            for s in validation_sites(aoi_path=aoi_file, index_path=idx)
            if s["name"].startswith("Border Lake")
        ]
        assert len(got) == 1 and got[0]["n"] == 12

    def test_toxin_only_sites_are_left_out(self, tmp_path, aoi_file):
        """A beach station samples a pond the 300 m grid cannot resolve, and carries no
        chlorophyll to compare a chlorophyll forecast against."""
        rows = [
            row(
                40.0,
                -96.5,
                f"2020-07-0{i % 9 + 1}",
                3.0,
                MICROCYSTIN,
                "EPA WQP",
                "Wagon Train Beach",
            )
            for i in range(30)
        ]
        idx = self.write_index(tmp_path, rows)
        names = {s["name"] for s in validation_sites(aoi_path=aoi_file, index_path=idx)}
        assert "Wagon Train Beach" not in names

    def test_a_placeholder_name_is_not_a_place(self, tmp_path, aoi_file):
        rows = [row(30.0, 20.0, "2020-08-01", 8.0, site="UNNAMED") for _ in range(15)]
        idx = self.write_index(tmp_path, rows)
        names = {s["name"].lower() for s in validation_sites(aoi_path=aoi_file, index_path=idx)}
        assert "unnamed" not in names

    def test_a_water_body_larger_than_the_limit_gets_a_forecastable_box(self, tmp_path, aoi_file):
        """A sea cannot be one 300 m forecast; the substitute must be inside the app's cap."""
        # Spread along one latitude so the pieces stay close enough to be recognised as
        # one water body: the grouping deliberately refuses to merge distant samples, so
        # a sparsely sampled sea is several sites rather than one wrong one.
        rows = [row(38.5, -77.0 + 0.085 * i, "2020-08-01", 8.0, site="Big Bay") for i in range(14)]
        idx = self.write_index(tmp_path, rows)
        bay = next(
            s
            for s in validation_sites(aoi_path=aoi_file, index_path=idx)
            if s["name"].startswith("Big Bay")
        )
        assert bay["clipped_to_limit"] is True
        assert bay["forecast_span_km"] < 80.0, "the box the app is handed must not be refused"
        assert bay["span_km"] > bay["forecast_span_km"]

    def test_the_modelled_lakes_come_first(self, tmp_path, aoi_file):
        rows = [row(41.7, -83.2, "2020-08-01", 20.0, site="Erie") for _ in range(3)]
        rows += [row(46.4, 6.5, "2020-08-01", 15.0, site="Lake Geneva") for _ in range(40)]
        idx = self.write_index(tmp_path, rows)
        sites = validation_sites(aoi_path=aoi_file, index_path=idx)
        assert sites[0]["kind"] == "modelled", "the reference set leads regardless of counts"

    def test_no_index_is_an_empty_list_not_an_error(self, tmp_path, aoi_file):
        load_index.cache_clear()
        validation_sites.cache_clear()
        assert validation_sites(aoi_path=aoi_file, index_path=tmp_path / "absent.csv.gz") == []
