"""The bulletin: pages drawn for a user-drawn area, the level band, and the PDF route.

The forecast is a synthetic `AdhocResult` on a real ad-hoc grid, so these pin what the
bulletin shows -- which pages, which caveats, when the wind page is left out -- without a
checkpoint, a satellite fetch or the wind service.
"""

from __future__ import annotations

import importlib
import re
import time
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("matplotlib")

from habforecast import bulletin
from habforecast.aoi import adhoc_aoi
from habforecast.inference.adhoc import AdhocResult
from habforecast.inference.predictor import ForecastResult

REPO_ROOT = Path(__file__).resolve().parents[2]
LEADS = (1, 3, 5, 7)


def _pages(pdf: bytes) -> int:
    return len(re.findall(rb"/Type\s*/Page\b(?!s)", pdf))


def _adhoc(anchor: date, with_prob: bool = True, caveat_days: int = 5) -> AdhocResult:
    aoi = adhoc_aoi(41.75, -83.3, radius_km=6.0, name="Maumee Bay test box")
    h, w = aoi.grid().shape
    yy, xx = np.mgrid[:h, :w]
    water = ((yy - h / 2) / (h / 2.2)) ** 2 + ((xx - w / 2) / (w / 2.2)) ** 2 < 1
    bloom = water & (xx < w / 3)  # a bloom along one shore
    log_chl = np.where(water, 0.8, np.nan).astype(np.float32)
    log_chl[bloom] = 1.6  # ~40 mg/m^3
    fc = np.stack([log_chl] * len(LEADS))
    prob = None
    if with_prob:
        p = np.where(water, 0.05, np.nan).astype(np.float32)
        p = np.stack([np.stack([np.where(bloom, 0.9, p)] * 3)] * len(LEADS))
        prob = p
    days = [anchor - timedelta(days=d) for d in range(caveat_days, 0, -1)]
    result = ForecastResult(
        lake_id="adhoc",
        lake_name=aoi.name,
        anchor_date=anchor,
        lead_times=LEADS,
        log_chl=fc,
        water=water,
        input_observed_days=days,
        source="live",
        model_version="test-model",
        exceed_prob=prob,
        exceed_thresholds=(10.0, 20.0, 50.0) if with_prob else (),
    )
    return AdhocResult(
        forecast=result,
        aoi=aoi,
        scenes_found=len(days) + 2,
        scenes_used=len(days),
        observed_days=days,
        water_pixels=int(water.sum()),
        water_confidence=0.6,
        cell_area_km2=0.09,
        history={d: log_chl.copy() for d in days},
    )


def _wind(anchor: date) -> dict:
    start = datetime.combine(anchor - timedelta(days=5), datetime.min.time())
    hours = [start + timedelta(hours=i) for i in range(24 * 10)]
    return {
        "latitude": 41.75,
        "longitude": -83.3,
        "utc_offset_seconds": -4 * 3600,
        "timezone_abbreviation": "GMT-4",
        "hourly": {
            "time": [t.isoformat(timespec="minutes") for t in hours],
            "wind_speed_10m": [3.0 + 12.0 * abs(np.sin(i / 20.0)) for i in range(len(hours))],
            "wind_direction_10m": [(i * 7) % 360 for i in range(len(hours))],
        },
    }


class TestAreaBulletin:
    def test_summary_imagery_forecast_and_wind_pages(self):
        today = datetime.now(UTC).date()
        pdf = bulletin.area_bulletin_pdf(_adhoc(today), wind=_wind(today))
        assert pdf.startswith(b"%PDF")
        assert _pages(pdf) == 4

    def test_wind_page_left_out_without_wind(self):
        today = datetime.now(UTC).date()
        pdf = bulletin.area_bulletin_pdf(_adhoc(today), fetch_wind_live=False)
        assert _pages(pdf) == 3

    def test_old_anchor_never_fetches_wind(self, monkeypatch):
        """Wind for the last five days says nothing about a forecast from last spring."""
        called = []
        monkeypatch.setattr(bulletin, "fetch_wind", lambda *a, **k: called.append(a))
        old = datetime.now(UTC).date() - timedelta(days=200)
        pdf = bulletin.area_bulletin_pdf(_adhoc(old))
        assert called == []
        assert _pages(pdf) == 3

    def test_a_model_without_a_probability_head_still_draws(self):
        today = datetime.now(UTC).date()
        pdf = bulletin.area_bulletin_pdf(_adhoc(today, with_prob=False), fetch_wind_live=False)
        assert _pages(pdf) == 3


class TestImagery:
    def test_latest_is_the_newest_frame_that_saw_the_water(self):
        water = np.ones((4, 4), bool)
        seen = np.full((4, 4), 1.0)
        cloudy = np.full((4, 4), np.nan)
        days = [date(2026, 9, 20), date(2026, 9, 22), date(2026, 9, 24)]
        img = bulletin.recent_imagery(days, [seen, seen * 1.5, cloudy], water, date(2026, 9, 25))
        assert img["latest_date"] == "2026-09-22"
        assert np.allclose(img["composite"], 1.5)
        assert img["composite_days"] == ["2026-09-20", "2026-09-22"]

    def test_frames_after_the_anchor_are_ignored(self):
        water = np.ones((2, 2), bool)
        img = bulletin.recent_imagery(
            [date(2026, 9, 30)], [np.ones((2, 2))], water, date(2026, 9, 25)
        )
        assert img["latest"] is None and img["composite"] is None


class TestLevelBand:
    def test_level_above_its_possible_band_is_held_to_it(self):
        props = {"level": "alert", "likely_level": "none", "possible_level": "watch",
                 "level_date": "2026-09-28"}  # fmt: skip
        out = bulletin.banded(props)
        assert out["level"] == "watch"
        assert out["feed_level"] == "alert"
        assert out["level_date"] is None

    def test_level_inside_the_band_is_untouched(self):
        props = {"level": "watch", "likely_level": "none", "possible_level": "warning"}
        assert bulletin.banded(props) is props

    def test_deterministic_levels_have_no_band(self):
        props = {"level": "alert"}
        assert bulletin.banded(props) is props


class TestScumRisk:
    def test_calm_water_with_a_bloom_is_high(self):
        assert bulletin.scum_risk(2.0, 0.2) == 3

    def test_no_bloom_caps_it_at_very_low(self):
        assert bulletin.scum_risk(2.0, 0.0) == 0

    def test_a_small_bloom_caps_it_at_low(self):
        assert bulletin.scum_risk(2.0, 0.01) == 1

    def test_strong_wind_mixes_it_away(self):
        assert bulletin.scum_risk(20.0, 0.5) == 0


class TestBulletinRoute:
    @pytest.fixture
    def api(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HAB_AOI", str(REPO_ROOT / "data" / "aoi" / "lakes.geojson"))
        monkeypatch.setenv("HAB_CUBES", str(tmp_path / "cubes"))
        monkeypatch.setenv("HAB_CHECKPOINT", str(tmp_path / "absent.ckpt"))
        monkeypatch.setenv("HAB_PROB_CHECKPOINT", str(tmp_path / "absent_prob"))
        import serving.api.main as api

        importlib.reload(api)
        return api

    def _done_job(self, api) -> str:
        job = api._jobs.submit("adhoc-forecast", {}, lambda job: {"ready": True})
        for _ in range(100):
            if job.status == "done":
                break
            time.sleep(0.02)
        assert job.status == "done"
        return job.id

    def test_unknown_job_is_404(self, api):
        from fastapi.testclient import TestClient

        assert TestClient(api.app).get("/api/forecast/jobs/nope/bulletin.pdf").status_code == 404

    def test_a_job_whose_inputs_were_evicted_says_so(self, api):
        from fastapi.testclient import TestClient

        jid = self._done_job(api)
        r = TestClient(api.app).get(f"/api/forecast/jobs/{jid}/bulletin.pdf")
        assert r.status_code == 410
        assert "Run the forecast again" in r.json()["detail"]

    def test_serves_the_pdf_and_renders_it_once(self, api, monkeypatch):
        from fastapi.testclient import TestClient

        jid = self._done_job(api)
        old = datetime.now(UTC).date() - timedelta(days=30)  # no wind fetch
        api._job_keys[jid] = "k"
        api._store_bulletin_input("k", _adhoc(old))
        client = TestClient(api.app)
        r = client.get(f"/api/forecast/jobs/{jid}/bulletin.pdf")
        assert r.status_code == 200
        assert r.headers["content-type"] == "application/pdf"
        assert "Maumee_Bay_test_box" in r.headers["content-disposition"]
        assert _pages(r.content) == 3

        monkeypatch.setattr(
            "habforecast.bulletin.area_bulletin_pdf",
            lambda *a, **k: pytest.fail("rendered twice"),
        )
        assert client.get(f"/api/forecast/jobs/{jid}/bulletin.pdf").content == r.content

    def test_inputs_are_bounded(self, api):
        for i in range(api.MAX_BULLETIN_INPUTS + 3):
            api._store_bulletin_input(f"k{i}", object())
        assert len(api._bulletin_inputs) == api.MAX_BULLETIN_INPUTS
