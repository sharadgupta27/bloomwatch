"""API contract tests.

The behaviour that matters most here is what happens with **no model checkpoint**. The
container must still start, pass its health check, and serve the catalog -- otherwise a
deployment where the model volume has not mounted yet fails its readiness probe and
restart-loops instead of reporting a clear reason. The CI docker job asserts exactly this
against a real container; these tests assert it at the application level.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def client(monkeypatch, tmp_path):
    """A client with a deliberately absent checkpoint."""
    monkeypatch.setenv("HAB_AOI", str(REPO_ROOT / "data" / "aoi" / "lakes.geojson"))
    monkeypatch.setenv("HAB_CUBES", str(tmp_path / "cubes"))
    monkeypatch.setenv("HAB_CHECKPOINT", str(tmp_path / "absent.ckpt"))
    monkeypatch.setenv("HAB_PROB_CHECKPOINT", str(tmp_path / "absent_prob"))

    import serving.api.main as api

    importlib.reload(api)  # re-read the env vars at module scope
    return TestClient(api.app)


class TestHealth:
    def test_healthy_without_a_model(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert body["model_loaded"] is False
        assert body["checkpoint_present"] is False

    def test_reports_the_registry_size(self, client):
        assert client.get("/health").json()["lakes"] == 5


class TestCatalog:
    def test_lists_every_lake(self, client):
        r = client.get("/lakes")
        assert r.status_code == 200
        lakes = r.json()
        assert len(lakes) == 5
        assert {lk["lake_id"] for lk in lakes} == {
            "erie_west",
            "okeechobee",
            "taihu",
            "balaton",
            "victoria_ne",
        }

    def test_exposes_split_assignment(self, client):
        """Consumers must be able to tell which lakes the model actually trained on."""
        lakes = {lk["lake_id"]: lk for lk in client.get("/lakes").json()}
        assert lakes["victoria_ne"]["split"] == "test"
        assert lakes["balaton"]["split"] == "val"
        assert lakes["erie_west"]["split"] == "train"

    def test_bbox_and_grid_shape_are_sane(self, client):
        for lk in client.get("/lakes").json():
            west, south, east, north = lk["bbox"]
            assert west < east and south < north
            h, w = lk["grid_shape"]
            assert h > 0 and w > 0

    def test_archived_days_is_null_without_cubes(self, client):
        assert all(lk["archived_days"] is None for lk in client.get("/lakes").json())


class TestForecastWithoutModel:
    def test_returns_503_not_500(self, client):
        """A missing model is unavailability, not a server error."""
        r = client.get("/forecast", params={"lake_id": "erie_west"})
        assert r.status_code == 503
        assert "model unavailable" in r.json()["detail"]

    def test_image_endpoint_also_503(self, client):
        r = client.get("/forecast/image", params={"lake_id": "erie_west", "lead_days": 7})
        assert r.status_code == 503

    def test_failure_is_cached_and_reported_in_health(self, client):
        client.get("/forecast", params={"lake_id": "erie_west"})
        body = client.get("/health").json()
        assert body["detail"] is not None
        assert "FileNotFoundError" in body["detail"]


class TestValidation:
    def test_missing_lake_id_is_422(self, client):
        assert client.get("/forecast").status_code == 422


class TestUI:
    def test_index_serves_the_map(self, client):
        r = client.get("/")
        assert r.status_code == 200
        assert "Bloom Watch" in r.text
        assert "/forecast" in r.text

    def test_openapi_schema_is_valid(self, client):
        schema = client.get("/openapi.json").json()
        for path in ("/health", "/lakes", "/forecast", "/forecast/image"):
            assert path in schema["paths"]

    def test_metrics_endpoint_responds(self, client):
        assert client.get("/metrics").status_code == 200


class TestProbeValueEncoding:
    """The app's inspect tool reports a number, so the number has to survive the wire.

    The overlay PNG cannot answer this: it is 8-bit through a five-stop ramp and is not
    invertible. These tests pin the separate quantised grid that carries the values.
    """

    @staticmethod
    def _decode(b64: str, shape):
        import base64

        import numpy as np

        return np.frombuffer(base64.b64decode(b64), dtype="<u2").reshape(shape)

    @staticmethod
    def _to_mg(codes, lo, hi):
        import numpy as np

        return 10.0 ** (lo + (codes.astype(np.float64) - 1) / 65534.0 * (hi - lo))

    def test_round_trips_within_sensor_precision(self, client):
        import numpy as np
        import serving.api.main as api

        truth = np.array([[0.0, 1.0, 2.0], [-0.5, 0.3, 2.7]], dtype=np.float32)
        water = np.ones_like(truth, dtype=bool)
        codes = self._decode(api._encode_values(truth, water), truth.shape)
        back = np.log10(self._to_mg(codes, api.VALUE_LO, api.VALUE_HI))
        # 65534 steps over four decades: the error is ~1e-4 dex, four orders of magnitude
        # below the retrieval's own uncertainty.
        assert np.abs(back - truth).max() < 1e-3

    def test_land_and_cloud_get_the_reserved_no_value_code(self, client):
        import numpy as np
        import serving.api.main as api

        truth = np.array([[1.0, 1.0], [1.0, np.nan]], dtype=np.float32)
        water = np.array([[True, False], [True, True]])
        codes = self._decode(api._encode_values(truth, water), truth.shape)
        assert codes[0, 1] == api.NO_VALUE, "dry land must not carry a value"
        assert codes[1, 1] == api.NO_VALUE, "an unpredicted pixel must not carry a value"
        assert (codes[[0, 1], [0, 0]] != api.NO_VALUE).all()

    def test_values_outside_the_scale_clamp_rather_than_wrap(self, client):
        """uint16 arithmetic wraps silently; a 500 mg/m3 scum must not read as clear water."""
        import numpy as np
        import serving.api.main as api

        truth = np.array([[-9.0, 9.0]], dtype=np.float32)
        codes = self._decode(api._encode_values(truth, np.ones_like(truth, bool)), truth.shape)
        assert codes[0, 0] == 1
        assert codes[0, 1] == 65535


class TestInSituEndpoint:
    """The endpoint a click on the forecast calls to ask "has anyone measured this water".

    It is the only endpoint whose *absence of data* is the common case, so the shape of an
    empty answer matters as much as a full one: the app has to be able to say "nothing
    within 10 km" without an error path.
    """

    @pytest.fixture
    def stocked(self, client, tmp_path, monkeypatch):
        import csv
        import gzip

        import serving.api.main as api

        from habforecast.insitu import COLUMNS, load_index

        path = tmp_path / "insitu.csv.gz"
        with gzip.open(path, "wt", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=COLUMNS)
            w.writeheader()
            w.writerow(
                {
                    "source": "GLORIA",
                    "site": "Lake Erie",
                    "variable": "chlorophyll_a",
                    "date": "2026-08-12",
                    "lat": 41.7,
                    "lon": -83.2,
                    "value": 44.0,
                    "flag": "",
                }
            )
        load_index.cache_clear()
        monkeypatch.setattr(api, "INSITU", path)
        return client

    def test_a_nearby_sample_comes_back_in_the_window(self, stocked):
        r = stocked.get(
            "/api/insitu",
            params={"lat": 41.7, "lon": -83.2, "start": "2026-08-07", "end": "2026-08-27"},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["available"] is True
        assert [p["value"] for p in body["window"]] == [44.0]

    def test_an_empty_neighbourhood_is_a_200_not_a_404(self, stocked):
        r = stocked.get(
            "/api/insitu",
            params={"lat": -20.0, "lon": 30.0, "start": "2026-08-07", "end": "2026-08-27"},
        )
        assert r.status_code == 200
        assert r.json()["total"] == 0

    def test_a_missing_index_is_reported_rather_than_raised(self, client, tmp_path, monkeypatch):
        """A fresh checkout has no index; the app must degrade, not 500."""
        import serving.api.main as api

        from habforecast.insitu import load_index

        load_index.cache_clear()
        monkeypatch.setattr(api, "INSITU", tmp_path / "absent.csv.gz")
        r = client.get(
            "/api/insitu",
            params={"lat": 41.7, "lon": -83.2, "start": "2026-08-07", "end": "2026-08-27"},
        )
        assert r.status_code == 200
        assert r.json()["available"] is False

    def test_a_backwards_window_is_refused(self, stocked):
        r = stocked.get(
            "/api/insitu",
            params={"lat": 41.7, "lon": -83.2, "start": "2026-08-27", "end": "2026-08-07"},
        )
        assert r.status_code == 422

    def test_the_radius_is_bounded(self, stocked):
        r = stocked.get(
            "/api/insitu",
            params={
                "lat": 41.7,
                "lon": -83.2,
                "start": "2026-08-07",
                "end": "2026-08-27",
                "radius_km": 5000,
            },
        )
        assert r.status_code == 422


class TestValidationSitesEndpoint:
    def test_it_answers_even_with_no_index(self, client, tmp_path, monkeypatch):
        import serving.api.main as api

        from habforecast.insitu import load_index, validation_sites

        load_index.cache_clear()
        validation_sites.cache_clear()
        monkeypatch.setattr(api, "INSITU", tmp_path / "absent.csv.gz")
        r = client.get("/api/validation-sites")
        assert r.status_code == 200
        assert r.json()["sites"] == []

    def test_it_names_where_the_field_data_came_from(self, client, tmp_path, monkeypatch):
        """The list is a claim about other people's measurements; it has to credit them."""
        import serving.api.main as api

        monkeypatch.setattr(api, "INSITU", tmp_path / "absent.csv.gz")
        body = client.get("/api/validation-sites").json()
        assert "GLORIA" in body["attribution"] and "EPA" in body["attribution"]


class TestRoutePreview:
    """The decision has to be inspectable *before* a request spends thirty seconds on it.

    And it has to be the same decision: this endpoint calls the same `plan` the forecast
    calls, so a preview that disagreed with what ran would be a bug in one shared path
    rather than a drift between two.
    """

    def test_a_lake_is_routed_to_the_300_m_neural_retrieval(self, client):
        r = client.get(
            "/api/route", params={"west": -83.6, "south": 41.55, "east": -82.9, "north": 41.95}
        )
        assert r.status_code == 200
        body = r.json()
        assert body["water"]["setting"] == "inland"
        assert body["sources"][0]["product"] == "olci-300m"
        assert body["model"]["key"] == "olci-300m"

    def test_open_ocean_is_routed_to_a_baseline_and_says_so(self, client):
        r = client.get(
            "/api/route", params={"west": -90.3, "south": 25.7, "east": -89.7, "north": 26.3}
        )
        body = r.json()
        assert body["water"]["setting"] == "open-ocean"
        assert body["model"] is None
        assert body["baseline"] == "persistence"
        assert any("Persistence" in c for c in body["caveats"])

    def test_the_preview_costs_no_imagery_and_no_model(self, client):
        client.get(
            "/api/route", params={"west": -83.6, "south": 41.55, "east": -82.9, "north": 41.95}
        )
        assert client.get("/health").json()["model_loaded"] is False

    def test_an_inverted_box_is_refused(self, client):
        r = client.get(
            "/api/route", params={"west": -82.9, "south": 41.55, "east": -83.6, "north": 41.95}
        )
        assert r.status_code == 422


class TestModelCatalogue:
    def test_it_lists_the_checkpoint_that_is_not_served(self, client):
        models = {m["key"]: m for m in client.get("/api/models").json()["models"]}
        assert models["viirs-4km"]["servable"] is False
        assert models["viirs-750m"]["servable"] is True

    def test_each_entry_carries_the_skill_it_measured(self, client):
        for m in client.get("/api/models").json()["models"]:
            assert m["skill_vs_persistence"], m["key"]
            assert m["train_bodies"] > 0


class TestModelOverride:
    def test_an_unservable_model_is_refused_with_a_reason(self, client):
        r = client.post(
            "/api/forecast/jobs",
            json={"lat": 41.75, "lon": -83.3, "radius_km": 12, "model": "viirs-4km"},
        )
        assert r.status_code == 422
        assert "not served" in r.json()["detail"]

    def test_an_unknown_model_is_refused(self, client):
        r = client.post(
            "/api/forecast/jobs",
            json={"lat": 41.75, "lon": -83.3, "radius_km": 12, "model": "nonesuch"},
        )
        assert r.status_code == 422
