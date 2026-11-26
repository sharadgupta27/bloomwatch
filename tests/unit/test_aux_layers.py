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
    OA06_NM,
    OA07_NM,
    OA08_NM,
    OA10_NM,
    OA11_NM,
    OA12_NM,
    asset_reads,
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


def flat(value=0.03, size=3):
    """One reflectance value in every band an index might ask for."""
    names = ("oa06", "oa07", "oa08", "oa10", "oa11", "oa12")
    return {k: np.full((size, size), value, dtype=np.float32) for k in names}


class TestCyanobacteriaIndex:
    """CI is the only layer claiming to tell cyanobacteria from other algae. The sign is
    the whole claim, so these tests pin the sign rather than the magnitude."""

    def test_a_flat_spectrum_has_no_shape(self):
        assert np.allclose(AUX_LAYERS["ci"].compute(flat()), 0.0, atol=1e-7)

    def test_a_linear_ramp_has_no_shape(self):
        """The baseline is interpolated 665 -> 709, so a straight spectrum reads zero."""
        r665, r709 = 0.02, 0.05
        frac = (OA10_NM - OA08_NM) / (OA11_NM - OA08_NM)
        bands = {
            "oa08": np.full((3, 3), r665, dtype=np.float32),
            "oa10": np.full((3, 3), r665 + frac * (r709 - r665), dtype=np.float32),
            "oa11": np.full((3, 3), r709, dtype=np.float32),
        }
        assert np.allclose(AUX_LAYERS["ci"].compute(bands), 0.0, atol=1e-7)

    def test_a_fluorescence_peak_reads_negative(self):
        """Eukaryotic algae fluoresce at 681 nm. That must not be called cyanobacteria."""
        bands = {
            "oa08": np.full((3, 3), 0.02, dtype=np.float32),
            "oa10": np.full((3, 3), 0.05, dtype=np.float32),
            "oa11": np.full((3, 3), 0.02, dtype=np.float32),
        }
        assert (AUX_LAYERS["ci"].compute(bands) < 0).all()

    def test_a_681_nm_trough_reads_positive(self):
        """Cyanobacteria lack that fluorescence, so the curvature reverses."""
        bands = {
            "oa08": np.full((3, 3), 0.05, dtype=np.float32),
            "oa10": np.full((3, 3), 0.02, dtype=np.float32),
            "oa11": np.full((3, 3), 0.05, dtype=np.float32),
        }
        assert (AUX_LAYERS["ci"].compute(bands) > 0).all()

    def test_a_dense_eukaryotic_bloom_separates_the_two_indices(self):
        """The case the pair exists for, and the only one where they must disagree.

        A heavy diatom or green bloom shows *both* a near-infrared peak at 709 nm and a
        chlorophyll fluorescence peak at 681 nm. MCI sees the first and calls it dense --
        correctly. CI sees the second and says it is not cyanobacterial. If CI merely
        echoed MCI's sign here it would be an expensive restatement of it.
        """
        bands = {
            "oa08": np.full((3, 3), 0.02, dtype=np.float32),  # 665 baseline
            "oa10": np.full((3, 3), 0.06, dtype=np.float32),  # 681 fluorescence peak
            "oa11": np.full((3, 3), 0.07, dtype=np.float32),  # 709 NIR peak
            "oa12": np.full((3, 3), 0.02, dtype=np.float32),  # 754 baseline
        }
        assert (AUX_LAYERS["mci"].compute(bands) > 0).all(), "dense surface bloom"
        assert (AUX_LAYERS["ci"].compute(bands) < 0).all(), "but not a cyanobacterial one"

    def test_the_two_indices_agree_when_fluorescence_is_absent(self):
        """A near-infrared peak with no 681 nm fluorescence is cyanobacteria, and both
        indices should say so -- they are not meant to disagree in general."""
        bands = flat()
        bands["oa11"] = np.full((3, 3), 0.09, dtype=np.float32)
        assert (AUX_LAYERS["mci"].compute(bands) > 0).all()
        assert (AUX_LAYERS["ci"].compute(bands) > 0).all()


class TestPhycocyanin:
    def test_a_linear_ramp_has_no_trough(self):
        r560, r665 = 0.02, 0.06
        frac = (OA07_NM - OA06_NM) / (OA08_NM - OA06_NM)
        bands = {
            "oa06": np.full((3, 3), r560, dtype=np.float32),
            "oa07": np.full((3, 3), r560 + frac * (r665 - r560), dtype=np.float32),
            "oa08": np.full((3, 3), r665, dtype=np.float32),
        }
        assert np.allclose(AUX_LAYERS["pc620"].compute(bands), 0.0, atol=1e-7)

    def test_absorption_at_620_nm_is_positive(self):
        """Larger means more bloom, as for every other layer here."""
        bands = flat(0.04)
        bands["oa07"] = np.full((3, 3), 0.02, dtype=np.float32)
        assert (AUX_LAYERS["pc620"].compute(bands) > 0).all()

    def test_it_measures_the_dip_not_the_brightness(self):
        """Raw Oa07 would track how bright the water is, which the model already sees."""
        bands = flat(0.04)
        bands["oa07"] = np.full((3, 3), 0.02, dtype=np.float32)
        brighter = {k: v * 3 for k, v in bands.items()}
        assert np.allclose(
            AUX_LAYERS["pc620"].compute(brighter),
            3 * AUX_LAYERS["pc620"].compute(bands),
            atol=1e-7,
        )


class TestReadCost:
    def test_shared_bands_are_counted_once(self):
        """Summing ASSET_COST would overstate the spectral set; the reader caches bands."""
        spectral = ("mci", "ci", "pc620")
        assert sum(len(AUX_LAYERS[k].sources) for k in spectral) == 9
        assert asset_reads(spectral) == 6

    def test_a_single_layer_costs_its_own_assets(self):
        assert asset_reads(("mci",)) == 3
        assert asset_reads(("tsm",)) == 1
        assert asset_reads(()) == 0


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
