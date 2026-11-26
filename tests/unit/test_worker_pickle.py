"""DataLoader workers get mapped arrays by path, not by copy, and distinct random streams."""

from __future__ import annotations

import pickle

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from habforecast.features import dataset as dataset_mod
from habforecast.features.datamodule import _init_worker
from habforecast.features.dataset import LakeSeries


def _series(tmp_path, n=40, h=16, w=16) -> LakeSeries:
    rng = np.random.default_rng(0)
    values = rng.normal(size=(n, h, w)).astype(np.float16)
    values[rng.random(values.shape) < 0.3] = np.nan
    path = tmp_path / "lake.values.abc.npy"
    if not path.exists():  # a second series maps the same file, as a re-run would
        np.save(path, values)
    return LakeSeries(
        lake_id="lake",
        split="train",
        days=np.datetime64("2020-06-01") + np.arange(n),
        values=np.load(path, mmap_mode="r"),
        water=np.ones((h, w), bool),
        shape=(h, w),
    )


def test_mapped_series_pickles_by_path(tmp_path):
    sr = _series(tmp_path)
    sr.prepare_climatology()
    blob = pickle.dumps(sr)
    assert len(blob) < sr.values.nbytes // 4  # the frames are not in the pickle
    back = pickle.loads(blob)
    for k in ("values", "_zero_filled", "_observed"):
        arr = getattr(back, k)
        assert isinstance(arr, np.memmap)
        np.testing.assert_array_equal(arr, getattr(sr, k))
    assert back.frame(back.days[3].astype(object)) is not None
    np.testing.assert_array_equal(
        back.climatology(doy=160, exclude_year=1999), sr.climatology(doy=160, exclude_year=1999)
    )


def test_prepare_climatology_reuses_finished_files(tmp_path):
    sr = _series(tmp_path)
    sr.prepare_climatology()
    zero = tmp_path / "lake.values.abc.zero.npy"
    stamp = zero.stat().st_mtime_ns
    again = _series(tmp_path)
    again.prepare_climatology()
    assert zero.stat().st_mtime_ns == stamp
    assert not list(tmp_path.glob("*.part"))


def test_in_ram_series_still_pickles(tmp_path):
    sr = _series(tmp_path)
    object.__setattr__(sr, "values", np.array(sr.values))
    back = pickle.loads(pickle.dumps(sr))
    np.testing.assert_array_equal(back.values, sr.values)


class _Draws(Dataset):
    def __init__(self):
        self.rng = np.random.default_rng(0)

    def __len__(self):
        return 8

    def __getitem__(self, i):
        return torch.tensor(self.rng.random())


def test_workers_draw_distinct_streams():
    budget = dataset_mod.CLIM_CACHE_MAX_BYTES
    loader = DataLoader(_Draws(), batch_size=1, num_workers=2, worker_init_fn=_init_worker)
    draws = [float(x) for x in loader]
    assert len(set(np.round(draws, 12))) == len(draws)
    assert budget == dataset_mod.CLIM_CACHE_MAX_BYTES  # only the workers' copies shrink
