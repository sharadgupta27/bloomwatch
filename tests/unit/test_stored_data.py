"""Answering from stored data: which lake can answer an area, and the scene cache."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime

import numpy as np
import pytest

from habforecast import stored
from habforecast.data import scene_cache

ERIE = ("erie_west", "Lake Erie (Western Basin)", (-83.6, 41.3, -82.4, 42.1))
VICTORIA = ("victoria_ne", "Lake Victoria (NE)", (32.5, -1.5, 35.0, 0.5))
TODAY = date(2026, 11, 27)


def _newest(days_ago: int | None):
    return lambda _lake: (
        None if days_ago is None else date.fromordinal(TODAY.toordinal() - days_ago)
    )


class TestMatch:
    def test_area_inside_a_lake_with_a_fresh_archive_is_answered_from_storage(self):
        m = stored.match((-83.4, 41.5, -82.7, 42.0), [ERIE, VICTORIA], _newest(2), None, TODAY)
        assert m.usable and m.lake_id == "erie_west"
        assert m.anchor == date(2026, 11, 25)  # the archive's newest day
        assert "nothing is downloaded" in m.reason

    def test_newest_requested_from_a_stale_archive_goes_live_and_says_how_to_avoid_it(self):
        m = stored.match((-83.4, 41.5, -82.7, 42.0), [ERIE], _newest(60), None, TODAY)
        assert m is not None and not m.usable and m.anchor is None
        assert "60 days ago" in m.reason and "Pick a date on or before" in m.reason

    def test_a_date_inside_a_stale_archive_is_still_answered_from_storage(self):
        asked = date(2026, 9, 1)
        m = stored.match((-83.4, 41.5, -82.7, 42.0), [ERIE], _newest(60), asked, TODAY)
        assert m.usable and m.anchor == asked

    def test_a_date_after_the_archive_ends_goes_live(self):
        m = stored.match((-83.4, 41.5, -82.7, 42.0), [ERIE], _newest(60), TODAY, TODAY)
        assert not m.usable and "before the date asked for" in m.reason

    def test_area_mostly_outside_any_lake_is_not_matched(self):
        # Half the box is east of the lake's extent: the archive never recorded it.
        assert stored.match((-83.0, 41.5, -81.6, 42.0), [ERIE], _newest(1), None, TODAY) is None

    def test_lake_without_an_archive_is_named_but_not_used(self):
        m = stored.match((-83.4, 41.5, -82.7, 42.0), [ERIE], _newest(None), None, TODAY)
        assert not m.usable and "no archive" in m.reason

    def test_tightest_covering_lake_wins_and_generators_are_fine(self):
        big = ("big", "Big", (-90.0, 30.0, -70.0, 50.0))
        m = stored.match(
            (-83.4, 41.5, -82.7, 42.0), (x for x in [big, ERIE]), _newest(1), None, TODAY
        )
        assert m.lake_id == "erie_west"

    def test_coverage_is_area_weighted(self):
        assert stored.coverage((0, 0, 1, 1), (0, 0, 1, 1)) == pytest.approx(1.0)
        assert stored.coverage((0, 0, 2, 1), (0, 0, 1, 1)) == pytest.approx(0.5, rel=1e-3)
        assert stored.coverage((5, 5, 6, 6), (0, 0, 1, 1)) == 0.0


# ------------------------------------------------------------------- scene cache ---
@dataclass(frozen=True)
class _Grid:
    lake_id: str = "adhoc"
    epsg: int = 32617
    x_min: float = 1000.0
    y_min: float = 2000.0
    width: int = 4
    height: int = 3
    resolution: float = 300.0


def _obs(valid_fraction: float = 0.5):
    from habforecast.data.olci import SceneObservation

    valid = np.zeros((3, 4), dtype=bool)
    valid.flat[: round(valid_fraction * 12)] = True
    chl = np.where(valid, 1.2, np.nan).astype(np.float32)
    return SceneObservation(
        lake_id="adhoc",
        item_id="S3A_X",
        time=datetime(2026, 11, 23, 15, 31, tzinfo=UTC),
        platform="sentinel-3a",
        log_chl=chl,
        valid=valid,
        flag_stats={"LAND": 0.2},
        aux={"mci": np.ones((3, 4), dtype=np.float32)},
    )


@pytest.fixture
def cache_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("HAB_SCENE_CACHE", str(tmp_path / "scenes"))
    return tmp_path / "scenes"


class TestSceneCache:
    def test_round_trip(self, cache_dir):
        k = scene_cache.key("R", "S3A_X", _Grid(), 240.0)
        assert scene_cache.load(k, 0.01) is None
        scene_cache.save(k, _obs(), 0.01)
        kind, obs = scene_cache.load(k, 0.01)
        assert kind == "hit"
        assert obs.item_id == "S3A_X" and obs.time.tzinfo is not None
        np.testing.assert_array_equal(obs.valid, _obs().valid)
        np.testing.assert_array_equal(obs.aux["mci"], _obs().aux["mci"])

    def test_key_separates_grids_and_sources(self):
        a = scene_cache.key("R", "S", _Grid(), 240.0)
        assert a != scene_cache.key("R", "S", _Grid(width=5), 240.0)
        assert a != scene_cache.key("Other", "S", _Grid(), 240.0)
        assert a == scene_cache.key("R", "S", _Grid(), 240.0)

    def test_stored_scene_is_filtered_by_the_threshold_asked(self, cache_dir):
        k = scene_cache.key("R", "S", _Grid(), 240.0)
        scene_cache.save(k, _obs(0.25), 0.01)
        assert scene_cache.load(k, 0.5) == ("rejected", None)

    def test_rejection_is_reused_only_for_an_equally_strict_request(self, cache_dir):
        k = scene_cache.key("R", "S", _Grid(), 240.0)
        scene_cache.save(k, None, 0.2)
        assert scene_cache.load(k, 0.2) == ("rejected", None)
        assert scene_cache.load(k, 0.5) == ("rejected", None)
        assert scene_cache.load(k, 0.01) is None  # looser: read it again

    def test_disabled_by_an_empty_setting(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HAB_SCENE_CACHE", "")
        k = scene_cache.key("R", "S", _Grid(), 240.0)
        scene_cache.save(k, _obs(), 0.01)
        assert scene_cache.load(k, 0.01) is None

    def test_prune_removes_oldest_first(self, cache_dir):
        import os

        for i in range(4):
            k = f"{i:032d}"
            scene_cache.save(k, _obs(), 0.01)
            os.utime(cache_dir / f"{k}.npz", (1000 + i, 1000 + i))
        one = (cache_dir / f"{0:032d}.npz").stat().st_size
        removed = scene_cache.prune(limit_mb=(2.5 * one) / 2**20)
        assert removed == 2
        assert not (cache_dir / f"{0:032d}.npz").exists()
        assert (cache_dir / f"{3:032d}.npz").exists()


def test_read_many_downloads_each_scene_once(cache_dir):
    from habforecast.data.olci import read_many

    calls = []

    class Reader:
        def read(self, item, grid, resampler, min_valid_fraction):
            calls.append(item.id)
            if item.id == "cloudy":
                return None
            o = _obs()
            return o

    @dataclass
    class Item:
        id: str

    @dataclass
    class Resampler:
        max_distance_m: float = 240.0

    items = [Item("a"), Item("b"), Item("cloudy")]
    first = read_many(Reader(), items, _Grid(), Resampler(), min_valid_fraction=0.01)
    second = read_many(Reader(), items, _Grid(), Resampler(), min_valid_fraction=0.01)
    assert len(first) == len(second) == 2
    assert sorted(calls) == ["a", "b", "cloudy"]  # nothing read twice, rejection included


def test_read_many_does_not_store_a_failed_read(cache_dir):
    from habforecast.data.olci import read_many

    attempts = []

    class Flaky:
        def read(self, item, grid, resampler, min_valid_fraction):
            attempts.append(1)
            if len(attempts) == 1:
                raise TimeoutError("stalled")
            return _obs()

    @dataclass
    class Item:
        id: str = "x"

    @dataclass
    class Resampler:
        max_distance_m: float = 240.0

    assert read_many(Flaky(), [Item()], _Grid(), Resampler()) == []
    assert len(read_many(Flaky(), [Item()], _Grid(), Resampler())) == 1
