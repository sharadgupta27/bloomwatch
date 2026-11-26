"""Public outlook, in-situ matchup scoring, and the serving speed-ups of 2026-11-26.

The speed-ups change *how* numbers are computed (a disk-cached water mask, a climatology
read once for all leads, a cube-backed path for monitored lakes), so the tests here pin
that the numbers themselves did not change.
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from habforecast import outlook
from habforecast.eval import matchup

# ------------------------------------------------------------------------- outlook ---


def _props(level="watch", likely=None, possible=None, anchor="2026-09-26", **kw):
    p = {
        "lake_id": "x",
        "name": "Lake X",
        "country": "XX",
        "level": level,
        "anchor_date": anchor,
        "peak_fraction_above": {"10": 0.4, "20": 0.1, "50": 0.0},
        **kw,
    }
    if possible:
        p.update(likely_level=likely, possible_level=possible)
    return p


class TestOutlook:
    def test_level_above_its_possible_band_is_held_to_it(self):
        out = outlook.held_in_band(_props("alert", likely="none", possible="watch"))
        assert out["level"] == "watch"
        assert out["feed_level"] == "alert"  # the feed's own answer is kept, not hidden

    def test_level_below_its_likely_band_is_raised(self):
        out = outlook.held_in_band(_props("none", likely="warning", possible="alert"))
        assert out["level"] == "warning"

    def test_level_inside_band_is_untouched(self):
        p = _props("watch", likely="none", possible="warning")
        assert outlook.held_in_band(p) is p

    def test_deterministic_advisory_passes_through(self):
        p = _props("alert")
        assert outlook.held_in_band(p) is p

    def test_age_and_staleness(self):
        today = date(2026, 9, 30)
        fresh = outlook.entry(_props(anchor="2026-09-26"), (1.0, 2.0), today)
        assert fresh["imagery_age_days"] == 4 and not fresh["stale"]
        old = outlook.entry(_props(anchor="2026-09-01"), (1.0, 2.0), today)
        assert old["stale"] and old["imagery_age_days"] == 29

    def test_share_of_lake_uses_the_levels_own_threshold(self):
        e = outlook.entry(_props("watch"), (0, 0), date(2026, 9, 27))
        assert e["threshold_mg_m3"] == 10.0 and e["share_of_lake"] == 0.4

    def test_unavailable_has_no_age_and_advice_never_says_safe(self):
        e = outlook.entry(
            {
                "lake_id": "t",
                "name": "T",
                "status": "unavailable",
                "level": "unavailable",
                "anchor_date": "2026-09-01",
                "reason": "cloud",
            },
            (0, 0),
            date(2026, 9, 2),
        )
        assert e["imagery_age_days"] is None and not e["stale"] and e["reason"] == "cloud"
        for spec in outlook.PUBLIC_LEVELS.values():
            text = " ".join([spec["headline"], *spec["advice"]]).lower()
            assert "safe to" not in text and "is safe" not in text

    def test_summary_is_most_severe_first(self):
        today = date(2026, 9, 27)
        es = [
            outlook.entry({**_props(lvl), "name": n}, (0, 0), today)
            for lvl, n in (("none", "A"), ("alert", "B"), ("watch", "C"))
        ]
        assert [e["level"] for e in outlook.summary(es, today)["lakes"]] == [
            "alert",
            "watch",
            "none",
        ]

    def test_latest_anchor_ignores_the_future(self):
        days = {date(2026, 9, 1): 0, date(2026, 9, 5): 1, date(2026, 12, 1): 2}
        assert outlook.latest_anchor(days, date(2026, 11, 26)) == date(2026, 9, 5)
        assert outlook.latest_anchor({}, date(2026, 11, 26)) is None


# ------------------------------------------------------------------------- matchup ---


class TestMatchup:
    def test_box_median_ignores_nan_and_clips_at_edges(self):
        f = np.full((4, 4), np.nan)
        f[0, 0], f[0, 1], f[1, 0] = 1.0, 2.0, 3.0
        assert matchup.box_median(f, 0, 0) == 2.0
        assert np.isnan(matchup.box_median(f, 3, 3))

    def test_continuous_in_log_space(self):
        obs = np.log10([10.0, 10.0, 10.0, 10.0])
        pred = np.log10([19.0, 19.0, 6.0, 40.0])
        c = matchup.continuous(pred, obs)
        assert c["n"] == 4
        assert c["within_2x"] == pytest.approx(0.75)
        assert c["bias_factor"] == pytest.approx(10 ** np.mean(np.log10([1.9, 1.9, 0.6, 4])))

    def test_categorical_counts(self):
        obs = np.array([5, 15, 25, 60, 8.0])
        pred = np.array([12, 9, 30, 55, 7.0])
        c = matchup.categorical(pred, obs, 10.0)
        assert (c["hits"], c["misses"], c["false_alarms"], c["correct_negatives"]) == (2, 1, 1, 1)
        assert c["pod"] == pytest.approx(2 / 3) and c["far"] == pytest.approx(1 / 3)

    def test_undefined_scores_are_none_not_zero(self):
        c = matchup.categorical(np.array([1.0, 2.0]), np.array([1.0, 2.0]), 50.0)
        assert c["pod"] is None and c["far"] is None and c["csi"] is None

    def test_bootstrap_sign_follows_the_better_predictor(self):
        rng = np.random.default_rng(1)
        obs = rng.normal(1.0, 0.4, 200)
        good = obs + rng.normal(0, 0.05, 200)
        bad = obs + rng.normal(0, 0.4, 200)
        clusters = np.repeat(np.arange(50), 4)
        out = matchup.cluster_bootstrap_mae_gain(good, bad, obs, clusters, n_boot=300)
        assert out["skill"] > 0.5 and out["ci95"][0] > 0
        assert out["clusters"] == 50


# ------------------------------------------- predictor speed-ups and the fast path ---


@pytest.fixture
def lake_setup(tmp_path, monkeypatch):
    """A one-lake registry, a 60-day cube on that lake's grid, and a tiny checkpoint."""
    torch = pytest.importorskip("torch")
    xr = pytest.importorskip("xarray")
    from habforecast.aoi import AOIRegistry
    from habforecast.models.anomaly import AnomalyForecaster
    from habforecast.models.lit import HABForecastModule

    reg_path = tmp_path / "lakes.geojson"
    reg_path.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "properties": {
                            "lake_id": "tiny",
                            "name": "Tiny Lake",
                            "split": "test",
                            "country": "XX",
                            "trophic_state": "eutrophic",
                            "bloom_season": [1, 12],
                            "utm_epsg": 32633,
                        },
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [
                                [
                                    [17.0, 46.0],
                                    [17.06, 46.0],
                                    [17.06, 46.05],
                                    [17.0, 46.05],
                                    [17.0, 46.0],
                                ]
                            ],
                        },
                    }
                ],
            }
        )
    )
    grid = AOIRegistry.from_geojson(reg_path)["tiny"].grid()
    rng = np.random.default_rng(0)
    days = pd.date_range("2024-06-01", periods=400, freq="D")
    vals = rng.normal(1.0, 0.3, (len(days), *grid.shape)).astype(np.float32)
    vals[:, :2, :] = np.nan  # a strip of land
    vals[rng.random(len(days)) < 0.4] = np.nan  # cloudy days
    cubes = tmp_path / "cubes"
    xr.Dataset({"log_chl": (("time", "y", "x"), vals)}, coords={"time": days.values}).to_zarr(
        cubes / "tiny.zarr", mode="w", consolidated=True
    )

    torch.manual_seed(0)
    kwargs = {"base": 8, "use_clim": True}
    net = AnomalyForecaster(input_days=14, n_leads=4, **kwargs)
    module = HABForecastModule(
        net,
        lead_times=(1, 3, 5, 7),
        norm_mean=1.0,
        norm_std=0.45,
        model_config={
            "name": "anomaly",
            "input_days": 14,
            "aux_keys": [],
            "met_keys": [],
            **kwargs,
        },
    )
    ckpt = tmp_path / "m.ckpt"
    torch.save({"state_dict": module.state_dict(), "hyper_parameters": dict(module.hparams)}, ckpt)
    statics = tmp_path / "statics"
    monkeypatch.setenv("HAB_STATIC_CACHE", str(statics))
    return reg_path, cubes, ckpt, statics, days


def _predictor(lake_setup):
    from habforecast.inference.predictor import Predictor

    reg_path, cubes, ckpt, *_ = lake_setup
    return Predictor(ckpt, reg_path, cubes, device="cpu").load()


class TestServingSpeedups:
    def test_serving_path_does_not_import_lightning(self):
        import subprocess
        import sys

        code = (
            "import sys, torch; from habforecast.inference.predictor import ensemble_forecast;"
            "import habforecast.inference.adhoc; print('lightning' in sys.modules)"
        )
        out = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            check=True,
            cwd=Path(__file__).resolve().parents[2],
            env={
                **__import__("os").environ,
                "PYTHONPATH": str(Path(__file__).resolve().parents[2] / "src"),
            },
        )
        assert out.stdout.strip().endswith("False")

    def test_water_mask_is_written_once_and_reused(self, lake_setup):
        statics = lake_setup[3]
        p1 = _predictor(lake_setup)
        ds, _ = p1._open_cube("tiny")
        w1 = p1._water_mask("tiny", ds)
        files = list(statics.glob("water_tiny_*.npy"))
        assert len(files) == 1
        p2 = _predictor(lake_setup)
        ds2, _ = p2._open_cube("tiny")
        # Prove the second process reads the file: tamper with it and see the tamper.
        np.save(files[0], np.zeros_like(w1))
        assert not p2._water_mask("tiny", ds2).any()
        assert w1[:2].sum() == 0 and w1[2:].all()

    def test_one_pass_climatology_matches_per_lead_reference(self, lake_setup):
        from habforecast.inference.predictor import CLIM_HALFWIDTH_DAYS

        p = _predictor(lake_setup)
        anchor = date(2025, 6, 20)
        clim, mask = p._climatology("tiny", anchor)
        ds, _ = p._open_cube("tiny")
        ts = pd.to_datetime(ds.time.values)
        doys, years = np.asarray(ts.dayofyear), np.asarray(ts.year)
        for li, lead in enumerate(p.lead_times):
            t = (anchor + timedelta(days=lead)).timetuple().tm_yday
            d = np.abs(doys - t)
            d = np.minimum(d, 366 - d)
            rows = np.flatnonzero((d <= CLIM_HALFWIDTH_DAYS) & (years != anchor.year))
            block = ds["log_chl"].isel(time=rows).values
            with np.errstate(invalid="ignore"):
                ref = np.nanmean(block, axis=0)
            has = np.isfinite(ref)
            np.testing.assert_array_equal(mask[li] > 0, has)
            np.testing.assert_allclose(clim[li][has], p.norm.apply(ref[has]), atol=1e-5)
        again, _ = p._climatology("tiny", anchor)  # memoised path
        np.testing.assert_array_equal(again, clim)

    def test_archive_fast_path_matches_predict(self, lake_setup):
        from habforecast.inference.adhoc import ARCHIVE_CUBE_SOURCE, forecast_from_archive

        days = lake_setup[4]
        p = _predictor(lake_setup)
        res = forecast_from_archive(p, "tiny")
        assert res is not None and res.source == ARCHIVE_CUBE_SOURCE
        last = max(d.date() for d in days)
        assert res.forecast.anchor_date == last
        direct = p.predict("tiny", last, allow_live=False)
        np.testing.assert_allclose(res.forecast.log_chl, direct.log_chl, equal_nan=True)
        assert set(res.history) == set(res.observed_days)
        for frame in res.history.values():
            assert np.isnan(frame[~res.forecast.water.astype(bool)]).all()
        assert res.water_confidence == 1.0 and res.scenes_found == res.scenes_used
        assert not any("rejected by quality control" in c for c in res.caveats())

    def test_archive_fast_path_declines_what_it_cannot_serve(self, lake_setup):
        from habforecast.inference.adhoc import forecast_from_archive

        p = _predictor(lake_setup)
        assert forecast_from_archive(p, "no_such_lake") is None
        assert forecast_from_archive(p, "tiny", anchor=date(2030, 1, 1)) is None
        assert forecast_from_archive(p, "tiny", anchor=date(2020, 1, 1)) is None  # no imagery


# ---------------------------------------------------------------------------- API ---


def test_validation_summary_404_until_run(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    monkeypatch.setenv("HAB_ARTIFACTS", str(tmp_path))
    import importlib

    import serving.api.main as api

    importlib.reload(api)
    client = TestClient(api.app)
    assert client.get("/api/validation-summary").status_code == 404
    (tmp_path / "insitu_validation").mkdir()
    (tmp_path / "insitu_validation" / "summary.json").write_text(
        json.dumps({"headline": ["x"], "by_lake": []})
    )
    assert client.get("/api/validation-summary").json()["headline"] == ["x"]


def test_registry_route_uses_the_archive_model_and_drops_the_drawn_area_caveat():
    from habforecast.routing import plan, registry_route

    drawn = plan((-83.5, 41.4, -82.7, 41.95))
    assert drawn.model.key != "olci-300m-clim"
    lake = registry_route(drawn, "Lake Erie")
    assert lake.model.key == "olci-300m-clim"
    assert lake.sources[0].product == "bloom-watch-archive"
    assert not any("has no multi-year archive" in c for c in lake.caveats)
    assert len(lake.caveats) == len(drawn.caveats) - sum(
        "has no multi-year archive" in c for c in drawn.caveats
    )


def test_oversized_drawn_area_is_a_422_not_a_500(monkeypatch, tmp_path):
    import serving.api.main as api
    from fastapi.testclient import TestClient

    client = TestClient(api.app)
    r = client.post(
        "/api/forecast/jobs",
        json={"lat": 41.6, "lon": -83.0, "bbox": [-84.0, 41.0, -82.0, 42.0], "name": "big"},
    )
    assert r.status_code == 422
    assert "limit" in r.json()["detail"]


class TestStoredDataClipping:
    """An area inside a monitored lake, answered from its archive, holds only that area."""

    def test_box_inside_the_lake_is_clipped_to_the_box(self, lake_setup):
        from habforecast.inference.adhoc import forecast_from_archive, shape_clip

        p = _predictor(lake_setup)
        whole = forecast_from_archive(p, "tiny")
        box = (17.0, 46.0, 17.03, 46.05)  # the western half of the lake
        part = forecast_from_archive(p, "tiny", clip=shape_clip(bbox=box))
        assert part.clipped and not whole.clipped
        assert 0 < part.water_pixels < whole.water_pixels
        lon, _lat = p.registry["tiny"].grid().lonlat
        outside = lon > 17.03
        assert not part.forecast.water[outside].any()
        assert np.isnan(part.forecast.log_chl[:, outside]).all()
        inside = part.forecast.water.astype(bool)
        # Inside the box the numbers are the whole-lake forecast's, unchanged.
        np.testing.assert_allclose(
            part.forecast.log_chl[:, inside], whole.forecast.log_chl[:, inside]
        )
        for frame in part.history.values():
            assert np.isnan(frame[outside]).all()

    def test_circle_and_polygon_clip_too(self, lake_setup):
        from habforecast.inference.adhoc import forecast_from_archive, shape_clip

        p = _predictor(lake_setup)
        circle = forecast_from_archive(p, "tiny", clip=shape_clip(circle=(46.03, 17.03, 1.5)))
        ring = [[(17.0, 46.02), (17.06, 46.02), (17.06, 46.05), (17.0, 46.05), (17.0, 46.02)]]
        poly = forecast_from_archive(p, "tiny", clip=shape_clip(rings=ring))
        assert circle.water_pixels > 0 and poly.water_pixels > 0

    def test_shape_with_no_lake_water_says_so(self, lake_setup):
        from habforecast.inference.adhoc import forecast_from_archive, shape_clip

        p = _predictor(lake_setup)
        with pytest.raises(ValueError, match="contains none of its water"):
            forecast_from_archive(p, "tiny", clip=shape_clip(bbox=(10.0, 10.0, 10.1, 10.1)))


def test_route_preview_reports_stored_data(monkeypatch):
    import serving.api.main as api
    from fastapi.testclient import TestClient

    from habforecast.stored import StoredMatch

    m = StoredMatch(
        "erie_west",
        "Lake Erie (Western Basin)",
        1.0,
        date(2026, 11, 25),
        True,
        date(2026, 11, 25),
        "This area lies inside Lake Erie, so it is forecast "
        "from the stored archive — nothing is downloaded.",
    )
    monkeypatch.setattr(api, "_stored_match", lambda bbox, anchor: m)
    r = TestClient(api.app).get(
        "/api/route", params={"west": -83.4, "south": 41.5, "east": -82.7, "north": 42.0}
    )
    body = r.json()
    assert r.status_code == 200 and body["stored"]["usable"]
    assert body["sources"][0]["product"] == "bloom-watch-archive"
    assert body["model"]["key"] == "olci-300m-clim"


def test_oversized_area_inside_a_stored_lake_is_accepted(monkeypatch):
    import serving.api.main as api
    from fastapi.testclient import TestClient

    from habforecast.stored import StoredMatch

    m = StoredMatch(
        "erie_west",
        "Lake Erie (Western Basin)",
        1.0,
        date(2026, 11, 25),
        True,
        date(2026, 11, 25),
        "inside",
    )
    monkeypatch.setattr(api, "_stored_match", lambda bbox, anchor: m)
    r = TestClient(api.app).post(
        "/api/forecast/jobs",
        json={"lat": 41.6, "lon": -83.0, "bbox": [-84.0, 41.0, -82.0, 42.0], "name": "big"},
    )
    assert r.status_code == 200  # was 422: the size limit is for downloaded areas
