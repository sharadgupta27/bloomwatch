"""How the live source behaves when the server misbehaves, which is most of the cost.

An ad-hoc forecast is almost entirely network, so its worst case is set here rather than
in the model. Every case below is a measured failure mode from 2026-08-29, when NOAA's
ERDDAP proxy spent the afternoon answering 502 after sixty seconds:

* an unconditional retry made that *worse* -- three attempts turned one 60 s failure into
  186 s, so the loop is bounded by a total budget and not by an attempt count;
* the anchor ladder then asked the same dead server about six earlier fortnights, which
  is how a single outage became a six-minute wait;
* and every later request re-discovered the outage from scratch.

The distinction the whole design turns on: **404 means the selection is empty and is an
answer; 502 means nobody answered.** Confusing the two is what makes a bad afternoon look
like a broken lake.

No network -- `urlopen` is replaced throughout.
"""

from __future__ import annotations

import email.message
import time
import urllib.error
from datetime import date

import numpy as np
import pytest

from habforecast.data import coastwatch as cw
from habforecast.data.coastwatch import SourceUnavailableError


class FakeResponse:
    def __init__(self, payload: bytes):
        self._payload = payload

    def read(self):
        return self._payload

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def http_error(code: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError("http://x", code, "boom", email.message.Message(), None)


@pytest.fixture(autouse=True)
def _clean_breaker():
    cw.reset_breaker()
    yield
    cw.reset_breaker()


@pytest.fixture
def no_sleep(monkeypatch):
    """Backoff pauses are real seconds; the behaviour under test is not about them."""
    monkeypatch.setattr(cw.time, "sleep", lambda _s: None)


class TestRetryPolicy:
    def test_an_empty_selection_is_not_retried(self, monkeypatch, no_sleep):
        """404 is ERDDAP saying "nothing there", the commonest answer of all."""
        calls = []

        def fake(req, timeout=None):
            calls.append(1)
            raise http_error(404)

        monkeypatch.setattr(cw.urllib.request, "urlopen", fake)
        with pytest.raises(urllib.error.HTTPError):
            cw._get("http://x")
        assert len(calls) == 1

    def test_a_transient_failure_is_retried_and_can_succeed(self, monkeypatch, no_sleep):
        seen = []

        def fake(req, timeout=None):
            seen.append(1)
            if len(seen) < 2:
                raise http_error(502)
            return FakeResponse(b"ok")

        monkeypatch.setattr(cw.urllib.request, "urlopen", fake)
        assert cw._get("http://x") == b"ok"
        assert len(seen) == 2

    def test_the_budget_caps_the_whole_call_not_each_attempt(self, monkeypatch):
        """The regression this exists for: retrying a slow failure made it three times slower."""
        attempts = []

        def slow_failure(req, timeout=None):
            attempts.append(1)
            time.sleep(0.15)
            raise http_error(502)

        monkeypatch.setattr(cw.urllib.request, "urlopen", slow_failure)
        started = time.monotonic()
        with pytest.raises(urllib.error.HTTPError):
            cw._get("http://x", attempts=5, budget_s=0.25)
        elapsed = time.monotonic() - started
        assert elapsed < 1.0, f"a bounded call took {elapsed:.2f}s"
        assert len(attempts) < 5, "the budget must stop the loop before the attempts do"

    def test_a_client_error_is_raised_rather_than_retried(self, monkeypatch, no_sleep):
        monkeypatch.setattr(
            cw.urllib.request,
            "urlopen",
            lambda req, timeout=None: (_ for _ in ()).throw(http_error(403)),
        )
        with pytest.raises(urllib.error.HTTPError):
            cw._get("http://x")


class TestResponseCache:
    def test_a_stored_response_is_served_without_the_network(self, monkeypatch, tmp_path):
        calls = []

        def once(req, timeout=None):
            calls.append(1)
            return FakeResponse(b"payload")

        monkeypatch.setattr(cw.urllib.request, "urlopen", once)
        for _ in range(3):
            got = cw._get("http://x", cache_ttl_h=1.0, cache_root=tmp_path)
            assert got == b"payload"
        assert len(calls) == 1, "the cache did not spare the repeat requests"

    def test_an_expired_entry_is_refetched(self, monkeypatch, tmp_path):
        monkeypatch.setattr(
            cw.urllib.request, "urlopen", lambda req, timeout=None: FakeResponse(b"v")
        )
        cw._get("http://x", cache_ttl_h=1.0, cache_root=tmp_path)
        assert cw._cached("http://x", ttl_h=0.0, root=tmp_path) is None

    def test_caching_is_off_by_default(self, monkeypatch, tmp_path):
        monkeypatch.setattr(
            cw.urllib.request, "urlopen", lambda req, timeout=None: FakeResponse(b"v")
        )
        cw._get("http://x", cache_root=tmp_path)
        assert not list(tmp_path.glob("*.bin"))


class TestCircuitBreaker:
    def test_it_opens_after_repeated_failures_and_then_fails_instantly(self):
        for _ in range(cw.BREAKER_FAILURES):
            cw._breaker_record(ok=False)
        assert cw.breaker_state()["open"] is True
        assert cw._breaker_seconds_left() > 0

    def test_one_success_closes_it(self):
        for _ in range(cw.BREAKER_FAILURES):
            cw._breaker_record(ok=False)
        cw._breaker_record(ok=True)
        assert cw.breaker_state()["open"] is False

    def test_a_single_failure_does_not_cut_the_source_off(self):
        cw._breaker_record(ok=False)
        assert cw.breaker_state()["open"] is False

    def test_it_reopens_the_door_after_the_cooldown(self, monkeypatch):
        for _ in range(cw.BREAKER_FAILURES):
            cw._breaker_record(ok=False)
        base = time.monotonic()
        monkeypatch.setattr(cw.time, "monotonic", lambda: base + cw.BREAKER_COOLDOWN_S + 1)
        assert cw._breaker_seconds_left() == 0.0


class FakeGrid:
    """The minimum `frames` needs: a footprint and a lon/lat mesh."""

    shape = (4, 4)
    bounds_lonlat = (-83.4, 41.6, -83.0, 41.9)

    @property
    def lonlat(self):
        lon, lat = np.meshgrid(np.linspace(-83.4, -83.0, 4), np.linspace(41.9, 41.6, 4))
        return lon, lat


def sector(ds_id="noaacwS3AOLCIchlaSectorEIDaily"):
    return cw.Sector(
        ds_id, -84.0, 41.0, -82.0, 42.5, "2026-05-28T00:00:00Z", "2026-08-26T00:00:00Z", 0.0025
    )


class TestFramesFailureSemantics:
    def test_no_coverage_is_an_empty_answer_not_an_outage(self, monkeypatch):
        """Every sector 404s: the fortnight is empty, and an earlier one may not be."""
        reader = cw.CoastWatchReader.__new__(cw.CoastWatchReader)
        reader.product = "olci-300m"
        reader._sectors = [
            sector("noaacwS3AOLCIchlaSectorEIDaily"),
            sector("noaacwS3BOLCIchlaSectorEIDaily"),
        ]
        reader._cache_dir = None
        monkeypatch.setattr(cw.CoastWatchReader, "_read_sector", lambda *a, **k: {})
        got = reader.frames(FakeGrid(), date(2026, 8, 13), date(2026, 8, 26))
        assert got == []
        assert cw.breaker_state()["open"] is False

    def test_every_sector_failing_in_transport_is_an_outage(self, monkeypatch):
        reader = cw.CoastWatchReader.__new__(cw.CoastWatchReader)
        reader.product = "olci-300m"
        reader._sectors = [
            sector("noaacwS3AOLCIchlaSectorEIDaily"),
            sector("noaacwS3BOLCIchlaSectorEIDaily"),
        ]
        reader._cache_dir = None

        def boom(*a, **k):
            raise http_error(502)

        monkeypatch.setattr(cw.CoastWatchReader, "_read_sector", boom)
        with pytest.raises(SourceUnavailableError):
            reader.frames(FakeGrid(), date(2026, 8, 13), date(2026, 8, 26))
        # Two sectors refused, so one request is enough evidence: discovering an outage
        # costs a full budget, and paying that twice is the thing being avoided.
        assert cw.breaker_state()["consecutive_failures"] == 2
        assert cw.breaker_state()["open"] is True

    def test_one_satellite_surviving_is_still_a_result(self, monkeypatch):
        """S3B answering while S3A 502s must produce a forecast, not an outage."""
        reader = cw.CoastWatchReader.__new__(cw.CoastWatchReader)
        reader.product = "olci-300m"
        reader._sectors = [
            sector("noaacwS3AOLCIchlaSectorEIDaily"),
            sector("noaacwS3BOLCIchlaSectorEIDaily"),
        ]
        reader._cache_dir = None
        field = np.full((4, 4), 1.2, dtype=np.float32)

        def mixed(self, sec, *a, **k):
            if sec.satellite == "S3A":
                raise http_error(502)
            return {date(2026, 8, 20): field}

        monkeypatch.setattr(cw.CoastWatchReader, "_read_sector", mixed)
        got = reader.frames(FakeGrid(), date(2026, 8, 13), date(2026, 8, 26))
        assert len(got) == 1 and got[0].day == date(2026, 8, 20)
        assert cw.breaker_state()["open"] is False

    def test_an_open_breaker_short_circuits_before_any_request(self, monkeypatch):
        reader = cw.CoastWatchReader.__new__(cw.CoastWatchReader)
        reader.product = "olci-300m"
        reader._sectors = [sector()]
        reader._cache_dir = None
        called = []
        monkeypatch.setattr(
            cw.CoastWatchReader, "_read_sector", lambda *a, **k: called.append(1) or {}
        )
        for _ in range(cw.BREAKER_FAILURES):
            cw._breaker_record(ok=False)
        with pytest.raises(SourceUnavailableError):
            reader.frames(FakeGrid(), date(2026, 8, 13), date(2026, 8, 26))
        assert called == [], "the breaker must answer without touching the network"
