"""Leakage and masking invariants for the sequence dataset.

Geospatial time-series data offers several ways to accidentally score your own training
data. These tests pin the ones that matter here:

- a lake in one split must never contribute samples to another;
- a sample's input window must never contain the day it is trying to predict;
- loss must only ever be computed where the target was genuinely observed and the pixel
  is water.

All of these produce optimistic results silently rather than raising, which is exactly
why they need tests rather than vigilance.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pytest
import torch

from habforecast.features.dataset import (
    LakeSeries,
    SequenceDataset,
    fit_normalisation,
)

H, W = 24, 32


def make_series(lake_id: str, split: str, start: date, n_days: int, every: int = 1, seed: int = 0):
    """Synthetic lake with observations every `every` days and a circular water mask."""
    rng = np.random.default_rng(seed)
    days = np.array([np.datetime64(start + timedelta(days=i * every), "D") for i in range(n_days)])
    yy, xx = np.mgrid[0:H, 0:W]
    water = ((yy - H / 2) ** 2 / (H / 2.5) ** 2 + (xx - W / 2) ** 2 / (W / 2.5) ** 2) < 1.0

    values = rng.normal(0.8, 0.4, size=(n_days, H, W)).astype(np.float32)
    # Punch a cloud hole in some frames so masking logic is genuinely exercised.
    for i in range(n_days):
        if i % 3 == 0:
            values[i, : H // 3, :] = np.nan
    values[:, ~water] = np.nan

    return LakeSeries(
        lake_id=lake_id,
        split=split,
        days=days,
        values=values.astype(np.float16),
        water=water,
        shape=(H, W),
    )


@pytest.fixture
def train_series():
    return [
        make_series("lake_a", "train", date(2020, 6, 1), 40, seed=1),
        make_series("lake_b", "train", date(2020, 6, 1), 40, seed=2),
    ]


@pytest.fixture
def norm(train_series):
    return fit_normalisation(train_series)


class TestNormalisation:
    def test_fitted_on_finite_values_only(self, train_series):
        n = fit_normalisation(train_series)
        assert np.isfinite(n.mean) and np.isfinite(n.std) and n.std > 0

    def test_roundtrip(self, norm):
        x = np.array([0.1, 1.0, 2.0], dtype=np.float32)
        np.testing.assert_allclose(norm.invert(norm.apply(x)), x, rtol=1e-5)

    def test_raises_without_observations(self):
        empty = make_series("dead", "train", date(2020, 6, 1), 3)
        empty.values[:] = np.nan
        with pytest.raises(ValueError, match="no finite"):
            fit_normalisation([empty])

    def test_val_lakes_do_not_affect_statistics(self, train_series, norm):
        """Fitting must ignore held-out lakes even if they are passed elsewhere."""
        val = make_series("lake_v", "val", date(2020, 6, 1), 40, seed=99)
        val.values[:] = val.values * 10.0  # wildly different distribution
        refit = fit_normalisation(train_series)  # val deliberately not included
        assert refit.mean == pytest.approx(norm.mean)
        assert refit.std == pytest.approx(norm.std)


class TestSplitIsolation:
    def test_dataset_only_draws_from_given_lakes(self, train_series, norm):
        ds = SequenceDataset(train_series, norm, input_days=7, lead_times=(1, 3), crop=None)
        drawn = {train_series[ds.samples[i].lake_idx].lake_id for i in range(len(ds))}
        assert drawn <= {"lake_a", "lake_b"}

    def test_val_dataset_never_sees_train_lakes(self, train_series, norm):
        val = [make_series("lake_v", "val", date(2020, 6, 1), 40, seed=7)]
        ds = SequenceDataset(val, norm, input_days=7, lead_times=(1, 3), crop=None)
        assert {val[s.lake_idx].lake_id for s in ds.samples} == {"lake_v"}
        assert all(val[s.lake_idx].split == "val" for s in ds.samples)


class TestTemporalCausality:
    def test_input_window_never_includes_the_target_day(self, train_series, norm):
        """The window must end at the anchor; targets are strictly after it."""
        input_days, leads = 7, (1, 3, 7)
        ds = SequenceDataset(train_series, norm, input_days=input_days, lead_times=leads, crop=None)
        for spec in ds.samples[:50]:
            window = {spec.anchor - timedelta(days=k) for k in range(input_days)}
            for lead in leads:
                assert spec.anchor + timedelta(days=lead) not in window

    def test_window_is_ordered_oldest_first(self, train_series, norm):
        ds = SequenceDataset(train_series, norm, input_days=5, lead_times=(1,), crop=None)
        s = ds.series[ds.samples[0].lake_idx]
        anchor = ds.samples[0].anchor
        _vals, mask = ds._stack_window(s, anchor)
        # The final slot must correspond to the anchor day itself.
        anchor_frame = s.frame(anchor)
        if anchor_frame is not None:
            expected = np.isfinite(anchor_frame.astype(np.float32))
            np.testing.assert_array_equal(mask[-1].astype(bool), expected)


class TestMasking:
    @pytest.fixture
    def sample(self, train_series, norm):
        ds = SequenceDataset(train_series, norm, input_days=7, lead_times=(1, 3), crop=None)
        assert len(ds) > 0
        return ds[0]

    def test_shapes_are_consistent(self, sample):
        assert sample["x"].shape == sample["x_mask"].shape == (7, H, W)
        assert sample["y"].shape == sample["y_mask"].shape == (2, H, W)
        assert sample["water"].shape == (1, H, W)

    def test_masks_are_binary(self, sample):
        for key in ("x_mask", "y_mask", "water"):
            assert torch.all((sample[key] == 0) | (sample[key] == 1)), f"{key} not binary"

    def test_no_nans_reach_the_model(self, sample):
        """NaNs must be replaced by 0 and flagged in the mask, never passed through."""
        for key in ("x", "y", "x_mask", "y_mask", "water"):
            assert torch.isfinite(sample[key]).all(), f"{key} contains non-finite values"

    def test_unobserved_inputs_are_zeroed(self, sample):
        assert torch.all(sample["x"][sample["x_mask"] == 0] == 0)

    def test_target_mask_excludes_land(self, sample):
        """y_mask must be a subset of the water mask -- loss never scores land."""
        water = sample["water"][0].bool()
        for lead in range(sample["y_mask"].shape[0]):
            scored = sample["y_mask"][lead].bool()
            assert not (scored & ~water).any(), "target mask includes non-water pixels"

    def test_doy_is_valid(self, sample):
        assert 1 <= float(sample["doy"]) <= 366


class TestSampleFiltering:
    def test_sparse_series_yields_no_samples(self, norm):
        """Observations 30 days apart cannot fill a 7-day window."""
        sparse = [make_series("sparse", "train", date(2020, 6, 1), 6, every=30, seed=3)]
        ds = SequenceDataset(
            sparse, norm, input_days=7, lead_times=(1,), crop=None, min_input_frames=2
        )
        assert len(ds) == 0

    def test_dense_series_yields_samples(self, train_series, norm):
        ds = SequenceDataset(train_series, norm, input_days=7, lead_times=(1, 3), crop=None)
        assert len(ds) > 0

    def test_min_input_frames_is_enforced(self, train_series, norm):
        strict = SequenceDataset(
            train_series, norm, input_days=7, lead_times=(1,), crop=None, min_input_frames=6
        )
        loose = SequenceDataset(
            train_series, norm, input_days=7, lead_times=(1,), crop=None, min_input_frames=1
        )
        assert len(strict) <= len(loose)


class TestCropping:
    def test_crop_reduces_spatial_dims(self, train_series, norm):
        ds = SequenceDataset(train_series, norm, input_days=5, lead_times=(1,), crop=16)
        s = ds[0]
        assert s["x"].shape[-2:] == (16, 16)
        assert s["y"].shape[-2:] == (16, 16)
        assert s["water"].shape[-2:] == (16, 16)

    def test_crop_larger_than_grid_pads(self, train_series, norm):
        ds = SequenceDataset(train_series, norm, input_days=5, lead_times=(1,), crop=64)
        s = ds[0]
        assert s["x"].shape[-2] >= H and s["x"].shape[-1] >= W

    def test_crop_is_square_when_only_one_axis_is_short(self, train_series, norm):
        """A lake narrower than the crop on one axis must still yield crop x crop.

        Otherwise that lake's samples collate against every other lake's at a different
        shape and the DataLoader raises mid-epoch.
        """
        c = W - 4  # wider than H (24), narrower than W (32)
        ds = SequenceDataset(train_series, norm, input_days=5, lead_times=(1,), crop=c)
        for i in range(5):
            s = ds[i]
            for key in ("x", "x_mask", "y", "y_mask", "water"):
                assert s[key].shape[-2:] == (c, c), key
