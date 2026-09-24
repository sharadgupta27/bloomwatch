"""The product-and-model policy: the decisions, and the ones it must refuse to make.

Routing is where a wrong answer is quietest. Nothing raises when a lake is forecast with a
checkpoint trained at the wrong scale, or when an open-ocean request is served by a
checkpoint that measured negative skill -- a plausible picture comes back either way. So
what these tests hold down is not the plumbing but the policy:

* a checkpoint never leaves the product it was trained on, because a convolutional
  receptive field is a number of pixels and moving it changes what it means;
* the 4 km checkpoint is never served, because it was measured and it has no skill;
* marine water always carries the disclosure that no model here was trained on it;
* the skill figures quoted in the app match the evaluation artifacts on disk.

No network, no model loaded.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from habforecast import routing, water

REPO = Path(__file__).resolve().parents[2]

pytestmark = pytest.mark.skipif(
    not water._reference_path().exists(),
    reason="coastline reference not built (scripts/build_coast_reference.py)",
)

LAKE = (-83.6, 41.55, -82.9, 41.95)  # Lake Erie, western basin
BAY = (-76.5, 37.8, -76.0, 38.3)  # Chesapeake
OCEAN = (-90.3, 25.7, -89.7, 26.3)  # Gulf of Mexico, 290 km out
POND = (-81.05, 26.95, -80.97, 27.02)  # 8 km across


class TestInlandWater:
    def test_a_lake_is_fetched_at_300_m_from_the_neural_retrieval(self):
        got = routing.plan(LAKE)
        assert got.product == "olci-300m"
        assert got.resolution_m == 300.0

    def test_a_lake_runs_the_checkpoint_trained_at_that_resolution(self):
        got = routing.plan(LAKE)
        assert got.model is not None
        assert got.model.grid_m == got.resolution_m

    def test_the_archive_is_the_fallback_and_is_the_same_resolution(self):
        """A coarser product is a different route, not a fallback -- the model is pinned."""
        got = routing.plan(LAKE)
        assert [s.role for s in got.sources] == ["primary", "fallback"]
        assert {s.resolution_m for s in got.sources} == {300.0}

    def test_the_reason_names_the_retrieval_argument(self):
        joined = " ".join(routing.plan(LAKE).reasons)
        assert "Case-2" in joined
        assert "Taihu" in joined, "the coverage measurement is the whole argument"


class TestOpenOcean:
    def test_open_ocean_moves_to_the_global_grids(self):
        got = routing.plan(OCEAN)
        assert got.product in ("viirs-4km", "olci-4km")
        assert got.resolution_m == 4000.0

    def test_no_learned_model_is_served_where_none_measured_any_skill(self):
        got = routing.plan(OCEAN)
        assert got.model is None
        assert got.baseline == routing.PERSISTENCE
        assert got.forecasts is False

    def test_the_persistence_fallback_is_disclosed_not_disguised(self):
        joined = " ".join(routing.plan(OCEAN).caveats)
        assert "Persistence" in joined

    def test_the_unservable_checkpoint_is_still_listed_with_its_result(self):
        """A catalogue that hides its failures is a brochure."""
        alt = routing.plan(OCEAN).alternatives
        assert [a["key"] for a in alt] == ["viirs-4km"]
        assert alt[0]["servable"] is False


class TestMarineDisclosure:
    @pytest.mark.parametrize("bbox", [BAY, OCEAN])
    def test_every_marine_route_says_no_model_was_trained_on_marine_water(self, bbox):
        joined = " ".join(routing.plan(bbox).caveats)
        assert "marine water" in joined
        assert "every training body is a lake" in joined

    def test_an_inland_route_does_not_carry_that_warning(self):
        joined = " ".join(routing.plan(LAKE).caveats)
        assert "every training body is a lake" not in joined

    def test_an_area_across_the_shoreline_is_flagged_as_mixed(self):
        joined = " ".join(routing.plan(BAY).caveats)
        assert "straddles the shoreline" in joined


class TestResolutionFloor:
    def test_a_small_water_body_is_not_offered_a_coarse_alternative(self):
        """Eight kilometres is eleven pixels at 750 m, below what the network needs."""
        got = routing.plan(POND)
        assert got.alternatives == []
        assert any("only rung that resolves it" in c for c in got.caveats)

    def test_a_large_lake_is_offered_the_better_trained_checkpoint(self):
        alt = routing.plan(LAKE).alternatives
        assert [a["key"] for a in alt] == ["viirs-750m"]
        assert alt[0]["beats_climatology"] is True

    def test_the_floor_is_the_pixel_count_the_network_needs(self):
        assert routing.coarsest_useful_m(16.0) == pytest.approx(1000.0)
        assert routing.coarsest_useful_m(4.8) == pytest.approx(300.0)


class TestStaleProducts:
    def test_choosing_the_750_m_route_today_warns_that_it_stopped_publishing(self):
        """Found by running it: asking for today returned imagery from the previous August."""
        why = routing.plan(LAKE, date(2026, 8, 30)).alternatives[0]["why"]
        assert "stopped publishing" in why

    def test_a_date_inside_its_record_carries_no_such_warning(self):
        why = routing.plan(LAKE, date(2024, 7, 1)).alternatives[0]["why"]
        assert "stopped publishing" not in why


class TestOverride:
    def test_an_override_moves_the_product_with_the_model(self):
        forced = routing.override_model(routing.plan(LAKE), "viirs-750m")
        assert forced.model.key == "viirs-750m"
        assert forced.product == "viirs-750m"
        assert forced.resolution_m == 750.0

    def test_the_unservable_checkpoint_cannot_be_forced_into_service(self):
        with pytest.raises(ValueError, match="not served"):
            routing.override_model(routing.plan(OCEAN), "viirs-4km")

    def test_an_unknown_model_is_refused_by_name(self):
        with pytest.raises(KeyError, match="nonesuch"):
            routing.override_model(routing.plan(LAKE), "nonesuch")

    def test_an_override_records_that_a_person_made_the_choice(self):
        forced = routing.override_model(routing.plan(LAKE), "viirs-750m")
        assert any("overriding the router" in r for r in forced.reasons)


class TestCatalogueMatchesTheEvaluation:
    """The skill figures are quoted to users, so they must match the run that produced them."""

    def test_the_quoted_skill_is_the_measured_skill(self):
        summary = json.loads(
            (REPO / "artifacts" / "multires" / "summary.json").read_text(encoding="utf-8")
        )
        arms = {
            "olci-300m": "olci-300m-3lakes",
            "viirs-750m": "viirs-750m-15lakes",
            "viirs-4km": "viirs-4km-15lakes",
        }
        for key, arm in arms.items():
            for lead, quoted in routing.MODELS[key].vs_persistence.items():
                measured = summary[arm][f"test_d{lead}"]["skill_vs_persistence"]
                assert quoted == pytest.approx(measured, abs=5e-4), f"{key} d{lead}"

    def test_only_checkpoints_that_beat_persistence_are_servable(self):
        for m in routing.MODELS.values():
            if m.servable:
                assert all(v > 0 for v in m.vs_persistence.values()), m.key

    def test_every_model_is_pinned_to_a_real_product(self):
        from habforecast.data.coastwatch import PRODUCTS

        for m in routing.MODELS.values():
            assert m.product in PRODUCTS
            assert PRODUCTS[m.product].resolution_m == m.grid_m


class TestSerialisation:
    def test_the_plan_survives_the_round_trip_the_app_makes(self):
        payload = json.loads(json.dumps(routing.plan(LAKE).as_dict()))
        assert payload["model"]["key"] == "olci-300m"
        assert payload["sources"][0]["product"] == "olci-300m"
        assert payload["water"]["setting"] == "inland"

    def test_an_unclassified_area_still_produces_a_usable_plan(self, monkeypatch, tmp_path):
        monkeypatch.setattr(water, "REFERENCE", tmp_path / "absent.npz")
        monkeypatch.setattr(water, "_ref", None)
        monkeypatch.setattr(water, "_ref_missing", False)
        got = routing.plan(LAKE)
        assert got.model is not None, "an unknown water body must still be forecastable"
        assert got.product == "olci-300m", "the conservative route is the 300 m one"
        assert any("could not be classified" in c for c in got.caveats)
