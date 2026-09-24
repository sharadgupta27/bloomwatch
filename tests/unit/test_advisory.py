"""The advisory feed: the rule, the GeoJSON, the RSS, and the two API routes.

The forecast is faked with a synthetic `ForecastResult`, so these pin the product logic
-- which threshold at which lead becomes which level, and what a subscriber sees when a
lake could not be forecast -- independently of any checkpoint.
"""

from __future__ import annotations

import importlib
import json
import xml.etree.ElementTree as ET
from datetime import date
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from habforecast import advisory
from habforecast.aoi import AOIRegistry
from habforecast.inference.predictor import ForecastResult

REPO_ROOT = Path(__file__).resolve().parents[2]
LEADS = (1, 3, 5, 7)
H, W = 20, 30


def _result(lake_id: str, name: str, fractions_above_20: list[float], scum: float = 0.0):
    """A forecast where a given fraction of the lake sits above 20 mg/m^3 at each lead,
    and `scum` of it above 50 mg/m^3."""
    water = np.ones((H, W), dtype=bool)
    water[:2] = False
    n_water = int(water.sum())
    log_chl = np.full((len(LEADS), H, W), np.log10(5.0), dtype=np.float32)
    flat_idx = np.flatnonzero(water.ravel())
    for li, frac in enumerate(fractions_above_20):
        frame = log_chl[li].ravel()
        k = round(frac * n_water)
        frame[flat_idx[:k]] = np.log10(25.0)
        ks = round(scum * n_water)
        frame[flat_idx[:ks]] = np.log10(80.0)
    log_chl[:, ~water] = np.nan
    return ForecastResult(
        lake_id=lake_id,
        lake_name=name,
        anchor_date=date(2024, 8, 15),
        lead_times=LEADS,
        log_chl=log_chl,
        water=water,
        input_observed_days=[date(2024, 8, 14), date(2024, 8, 12)],
        source="cube",
        model_version="test-ckpt",
    )


@pytest.fixture
def registry():
    return AOIRegistry.from_geojson(REPO_ROOT / "data" / "aoi" / "lakes.geojson")


class TestRule:
    def test_quiet_lake_has_no_advisory(self):
        adv = advisory.advise(_result("erie_west", "Erie", [0.0, 0.0, 0.02, 0.05]), 0.09)
        assert adv.level == "none"
        assert adv.level_lead_days is None
        assert adv.first_exceedance_lead["20"] is None

    def test_level_is_the_highest_threshold_crossed_and_its_first_lead(self):
        adv = advisory.advise(_result("erie_west", "Erie", [0.0, 0.05, 0.30, 0.40]), 0.09)
        assert adv.level == "warning"
        assert adv.level_lead_days == 5
        assert adv.level_date == date(2024, 8, 20)
        # 10 mg/m^3 is crossed wherever 20 is, so its first lead is the same here.
        assert adv.first_exceedance_lead == {"10": 5, "20": 5, "50": None}
        assert adv.peak_fraction["20"] == pytest.approx(0.40, abs=0.01)

    def test_scum_raises_an_alert(self):
        adv = advisory.advise(_result("taihu", "Taihu", [0.5] * 4, scum=0.15), 0.09)
        assert adv.level == "alert"
        assert adv.level_lead_days == 1

    def test_min_fraction_is_honoured(self):
        r = _result("erie_west", "Erie", [0.12, 0.12, 0.12, 0.12])
        assert advisory.advise(r, 0.09, min_fraction=0.10).level == "warning"
        assert advisory.advise(r, 0.09, min_fraction=0.20).level == "none"


class TestGeoJSON:
    def test_feature_carries_geometry_provenance_and_disclaimer(self, registry):
        aoi = registry["erie_west"]
        adv = advisory.advise(_result("erie_west", aoi.name, [0.0, 0.3, 0.3, 0.3]), 0.09)
        feat = advisory.feature(adv, aoi)
        assert feat["type"] == "Feature" and feat["id"] == "erie_west"
        ring = feat["geometry"]["coordinates"][0]
        assert ring[0] == ring[-1] and len(ring) == 5
        p = feat["properties"]
        assert p["status"] == "ok" and p["level"] == "warning" and p["level_lead_days"] == 3
        assert p["n_observed_days"] == 2 and p["source"] == "cube"
        assert p["model_version"] == "test-ckpt"
        assert "toxin" in p["disclaimer"]
        assert [f["lead_days"] for f in p["forecasts"]] == list(LEADS)
        json.dumps(feat)  # serialisable

    def test_unavailable_lake_stays_in_the_feed(self, registry):
        aoi = registry["balaton"]
        feat = advisory.unavailable_feature(aoi, date(2024, 8, 15), "persistent cloud cover")
        assert feat["properties"]["status"] == "unavailable"
        assert feat["properties"]["level"] == "unavailable"
        assert "cloud" in feat["properties"]["reason"]

    def test_collection_lists_the_level_vocabulary(self):
        fc = advisory.collection([])
        assert fc["type"] == "FeatureCollection"
        assert fc["levels"] == ["none", "watch", "warning", "alert"]
        assert fc["generated_utc"].endswith("+00:00")


class TestRSS:
    def test_feed_is_well_formed_and_one_item_per_lake(self, registry):
        feats = [
            advisory.feature(
                advisory.advise(_result("erie_west", "Erie", [0.0, 0.3, 0.3, 0.3]), 0.09),
                registry["erie_west"],
            ),
            advisory.unavailable_feature(registry["balaton"], date(2024, 8, 15), "cloud"),
        ]
        xml = advisory.rss(advisory.collection(feats), base_url="https://example.org")
        root = ET.fromstring(xml)
        items = root.findall("./channel/item")
        assert len(items) == 2
        titles = [i.findtext("title") for i in items]
        assert titles[0].startswith("Erie: WARNING")
        assert "2024-08-18" in titles[0] and "+3 d" in titles[0]
        assert "no forecast" in titles[1]
        assert items[0].findtext("link") == "https://example.org/app?lake=erie_west"
        assert items[0].findtext("category") == "warning"
        assert "toxin" in items[0].findtext("description")
        guids = [i.findtext("guid") for i in items]
        assert len(set(guids)) == 2


class _FakePredictor:
    """Serves synthetic forecasts for the registry; one lake is always under cloud."""

    model_version = "fake"

    def __init__(self, registry):
        self.registry = registry

    def predict(self, lake_id, anchor=None, allow_live=True):
        if lake_id == "balaton":
            raise ValueError("no usable observations for balaton -- persistent cloud cover")
        frac = {"erie_west": 0.0, "okeechobee": 0.5, "taihu": 0.15, "victoria_ne": 0.0}[lake_id]
        return _result(lake_id, self.registry[lake_id].name, [frac] * 4)


class TestBuild:
    def test_build_runs_every_lake_and_keeps_failures(self, registry):
        fc = advisory.build(_FakePredictor(registry), anchor=date(2024, 8, 15))
        by_id = {f["id"]: f["properties"] for f in fc["features"]}
        assert set(by_id) == set(registry.lake_ids)
        assert by_id["balaton"]["status"] == "unavailable"
        assert by_id["okeechobee"]["level"] == "warning"
        assert by_id["erie_west"]["level"] == "none"


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setenv("HAB_AOI", str(REPO_ROOT / "data" / "aoi" / "lakes.geojson"))
    monkeypatch.setenv("HAB_CUBES", str(tmp_path / "cubes"))
    monkeypatch.setenv("HAB_CHECKPOINT", str(tmp_path / "absent.ckpt"))
    monkeypatch.setenv("HAB_PROB_CHECKPOINT", str(tmp_path / "absent_prob"))
    import serving.api.main as api

    importlib.reload(api)
    return TestClient(api.app), api


class TestRoutes:
    def test_without_a_model_the_feed_is_503(self, client):
        c, _api = client
        assert c.get("/advisory").status_code == 503
        assert c.get("/advisory/feed.xml").status_code == 503

    def test_geojson_and_rss_agree(self, client, registry):
        c, api = client
        api._predictor = _FakePredictor(registry)
        r = c.get("/advisory", params={"anchor": "2024-08-15"})
        assert r.status_code == 200
        fc = r.json()
        assert fc["type"] == "FeatureCollection" and len(fc["features"]) == 5
        levels = {f["id"]: f["properties"]["level"] for f in fc["features"]}
        assert levels["okeechobee"] == "warning" and levels["balaton"] == "unavailable"

        x = c.get("/advisory/feed.xml", params={"anchor": "2024-08-15"})
        assert x.status_code == 200
        assert x.headers["content-type"].startswith("application/rss+xml")
        root = ET.fromstring(x.text)
        assert len(root.findall("./channel/item")) == 5
        assert root.find("./channel/item/link").text.startswith("http://testserver/app")

    def test_bad_inputs_are_422(self, client, registry):
        c, api = client
        api._predictor = _FakePredictor(registry)
        assert c.get("/advisory", params={"anchor": "not-a-date"}).status_code == 422
        assert c.get("/advisory", params={"min_fraction": 0}).status_code == 422


class _FakeProbPredictor:
    """Probabilities only: every water pixel has p(>=20) = 0.9 and p(>=50) = 0.05."""

    def __init__(self, registry):
        self.registry = registry

    def predict(self, lake_id, anchor=None, allow_live=True):
        r = _result(lake_id, self.registry[lake_id].name, [0.0] * 4)
        prob = np.zeros((len(LEADS), 3, H, W), dtype=np.float32)
        prob[:, 0], prob[:, 1], prob[:, 2] = 0.95, 0.9, 0.05
        r.exceed_prob, r.exceed_thresholds = prob, (10.0, 20.0, 50.0)
        r.ensemble_size, r.model_version = 3, "ens_met-x3"
        return r


class TestMapPlusProbabilities:
    def test_map_kept_probabilities_taken(self, registry):
        map_r = _result("erie_west", "Erie", [0.0, 0.0, 0.0, 0.0])
        merged = advisory.with_probabilities(
            map_r, _FakeProbPredictor(registry).predict("erie_west")
        )
        np.testing.assert_array_equal(merged.log_chl, map_r.log_chl)
        assert merged.ensemble_size == 3
        assert "(map)" in merged.model_version and "(p)" in merged.model_version
        # Land stays NaN even though the probability model filled it.
        assert np.isnan(merged.exceed_prob[0, 1][~map_r.water]).all()
        assert advisory.advise(merged, 0.09).level == "warning"

    def test_misregistered_grids_are_refused(self, registry):
        map_r = _result("erie_west", "Erie", [0.0] * 4)
        other = _FakeProbPredictor(registry).predict("erie_west")
        other.water = other.water[:-1]
        with pytest.raises(ValueError, match="does not match"):
            advisory.with_probabilities(map_r, other)

    def test_api_advisory_reads_the_probability_model(self, client, registry):
        c, api = client
        api._predictor = _FakePredictor(registry)
        api._prob_predictor = _FakeProbPredictor(registry)
        fc = c.get("/advisory", params={"anchor": "2024-08-15"}).json()
        props = {f["id"]: f["properties"] for f in fc["features"]}
        assert props["erie_west"]["level"] == "warning"  # the map alone says "none"
        assert props["erie_west"]["basis"] == "probabilistic"
        assert props["erie_west"]["ensemble_size"] == 3
