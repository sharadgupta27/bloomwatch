"""Auxiliary input layers: the arithmetic, the plumbing, and the ways they could lie.

Adding channels to a forecaster is easy to do wrongly in ways that never raise. The three
that would matter here, and that these tests exist to prevent:

* **A spectral index that is not the index.** MCI is a three-band baseline subtraction;
  get the interpolation weight wrong and it still produces plausible numbers, still
  trains, and measures something else.
* **A channel mis-registered against the chlorophyll it explains.** Crops and flips are
  applied to the value stack; an auxiliary layer left out of one of them is offset by a
  few hundred metres from the pixel it is supposed to describe.
* **A gap presented as a measurement.** An unretrieved layer must arrive as zero *with a
  zero mask*, which the model reads as "never observed" -- not as a real value of zero,
  which for MCI is the middle of its range.

No network.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pytest

from habforecast.data.aux_layers import (
    AUX_LAYERS,
    OA10_NM,
    OA11_NM,
    OA12_NM,
    clean,
    parse_aux,
)
from habforecast.features.dataset import (
    LakeSeries,
    Normalisation,
    SequenceDataset,
    fit_aux_normalisation,
)


class TestSpectralIndex:
    def test_a_flat_spectrum_has_no_peak(self):
        """MCI measures height above a baseline; a flat spectrum is its own baseline."""
        flat = {k: np.full((4, 4), 0.03, dtype=np.float32) for k in ("oa10", "oa11", "oa12")}
        assert np.allclose(AUX_LAYERS["mci"].compute(flat), 0.0, atol=1e-7)

    def test_a_linear_ramp_also_has_no_peak(self):
        """The baseline is interpolated between 681 and 754 nm, not held flat."""
        r681, r754 = 0.02, 0.06
        frac = (OA11_NM - OA10_NM) / (OA12_NM - OA10_NM)
        bands = {
            "oa10": np.full((3, 3), r681, dtype=np.float32),
            "oa11": np.full((3, 3), r681 + frac * (r754 - r681), dtype=np.float32),
            "oa12": np.full((3, 3), r754, dtype=np.float32),
        }
        assert np.allclose(AUX_LAYERS["mci"].compute(bands), 0.0, atol=1e-7)

    def test_a_709_nm_peak_is_positive(self):
        """The signature of surface cyanobacteria: reflectance where water should absorb."""
        bands = {
            "oa10": np.full((3, 3), 0.02, dtype=np.float32),
            "oa11": np.full((3, 3), 0.05, dtype=np.float32),
            "oa12": np.full((3, 3), 0.02, dtype=np.float32),
        }
        assert (AUX_LAYERS["mci"].compute(bands) > 0.02).all()

    def test_the_index_is_not_a_restatement_of_brightness(self):
        """Doubling every band leaves MCI unchanged only if the baseline is subtracted."""
        bands = {
            "oa10": np.full((2, 2), 0.02, dtype=np.float32),
            "oa11": np.full((2, 2), 0.05, dtype=np.float32),
            "oa12": np.full((2, 2), 0.03, dtype=np.float32),
        }
        doubled = {k: v * 2 for k, v in bands.items()}
        assert np.allclose(
            AUX_LAYERS["mci"].compute(doubled), 2 * AUX_LAYERS["mci"].compute(bands), atol=1e-7
        )


class TestRangeGuard:
    def test_an_impossible_value_becomes_a_gap_not_a_clamp(self):
        """A reflectance of 40 is a decoding fault; clamping would feed the model a lie."""
        got = clean("mci", np.array([0.01, 40.0, -9999.0, np.nan], dtype=np.float32))
        assert got[0] == pytest.approx(0.01)
        assert np.isnan(got[1]) and np.isnan(got[2]) and np.isnan(got[3])

    def test_every_layer_declares_a_range_and_a_reason(self):
        for key, layer in AUX_LAYERS.items():
            lo, hi = layer.valid_range
            assert lo < hi, key
            assert layer.why and layer.units, f"{key} does not say what it contributes"
            assert layer.sources, f"{key} has no source asset"


class TestParsing:
    def test_an_unknown_layer_is_refused_by_name(self):
        with pytest.raises(ValueError, match="chlorophyll_squared"):
            parse_aux("mci,chlorophyll_squared")

    def test_empty_means_none(self):
        assert parse_aux("") == () and parse_aux(None) == ()

    def test_whitespace_is_tolerated(self):
        assert parse_aux(" mci , tsm ") == ("mci", "tsm")


def series(lake_id="erie", split="train", n=30, size=24, with_aux=("mci",), seed=0):
    rng = np.random.default_rng(seed)
    days = np.array([np.datetime64(date(2024, 6, 1) + timedelta(days=i), "D") for i in range(n)])
    values = rng.normal(1.0, 0.3, (n, size, size)).astype(np.float32)
    values[rng.random((n, size, size)) < 0.3] = np.nan  # cloud
    aux = {}
    for k in with_aux:
        layer = rng.normal(0.01, 0.005, (n, size, size)).astype(np.float32)
        layer[rng.random((n, size, size)) < 0.5] = np.nan  # sparser than chlorophyll
        aux[k] = layer.astype(np.float16)
    return LakeSeries(
        lake_id=lake_id,
        split=split,
        days=days,
        values=values.astype(np.float16),
        water=np.ones((size, size), dtype=bool),
        shape=(size, size),
        aux=aux,
    )


class TestDatasetPlumbing:
    def make(self, s, keys=("mci",), crop=None):
        return SequenceDataset(
            [s],
            norm=Normalisation(mean=1.0, std=0.4),
            input_days=8,
            lead_times=(1, 3),
            crop=crop,
            min_input_frames=1,
            min_target_coverage=0.0,
            aux_keys=keys,
            aux_stats={"mci": (0.01, 0.005)},
        )

    def test_the_batch_carries_one_channel_group_per_layer(self):
        ds = self.make(series())
        item = ds[0]
        assert item["aux"].shape == (1, 8, 24, 24)
        assert item["aux_mask"].shape == item["aux"].shape

    def test_an_unobserved_layer_pixel_is_zero_with_a_zero_mask(self):
        """The model reads that as "never seen", which is a state it already handles."""
        ds = self.make(series())
        item = ds[0]
        gap = item["aux_mask"] == 0
        assert gap.any(), "the fixture should contain gaps"
        assert (item["aux"][gap] == 0).all()

    def test_a_cube_without_the_layer_contributes_nothing_rather_than_a_guess(self):
        ds = self.make(series(with_aux=()), keys=("mci",))
        item = ds[0]
        assert item["aux"].shape[0] == 1
        assert item["aux_mask"].sum() == 0, "a missing layer must never look observed"
        assert item["aux"].abs().sum() == 0

    def test_no_aux_keys_yields_an_empty_channel_axis(self):
        ds = self.make(series(), keys=())
        assert ds[0]["aux"].shape[0] == 0

    def test_a_crop_moves_the_layer_with_the_chlorophyll(self):
        """Mis-registration is the failure that trains happily and means nothing."""
        s = series(size=32)
        # A layer that is an exact copy of chlorophyll must survive cropping as a copy.
        s.aux["mci"] = s.values.copy()
        ds = SequenceDataset(
            [s],
            norm=Normalisation(mean=0.0, std=1.0),
            input_days=8,
            lead_times=(1,),
            crop=16,
            min_input_frames=1,
            min_target_coverage=0.0,
            aux_keys=("mci",),
            aux_stats={"mci": (0.0, 1.0)},
        )
        item = ds[0]
        assert item["aux"].shape[-2:] == (16, 16)
        observed = item["x_mask"] > 0
        assert observed.any()
        assert np.allclose(
            item["aux"][0][observed].numpy(), item["x"][observed].numpy(), atol=1e-3
        ), "the auxiliary crop is offset from the chlorophyll crop"


class TestAuxNormalisation:
    def test_statistics_come_from_the_given_lakes_only(self):
        train = series(lake_id="a", seed=1)
        stats = fit_aux_normalisation([train], ("mci",))
        assert set(stats) == {"mci"}
        mean, std = stats["mci"]
        finite = train.aux["mci"][np.isfinite(train.aux["mci"])].astype(np.float32)
        assert mean == pytest.approx(float(finite.mean()), rel=1e-3)
        assert std > 0

    def test_a_layer_nobody_has_is_simply_absent(self):
        assert fit_aux_normalisation([series(with_aux=())], ("mci",)) == {}

    def test_a_constant_layer_does_not_divide_by_zero(self):
        s = series()
        s.aux["mci"] = np.full_like(s.aux["mci"], 0.02)
        assert fit_aux_normalisation([s], ("mci",))["mci"][1] == 1.0
