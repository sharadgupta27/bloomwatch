"""Tests for drift detection.

The three drift families exist because they demand different responses, and conflating
them is the usual failure of generic drift tooling:

- observation drift  -> investigate the upstream data, do not retrain
- distribution drift -> check against in-situ, possibly retrain
- geographic drift   -> refuse the prediction or flag it out-of-distribution

A geographic-novelty check that silently returned "no drift" for a tropical lake when
the model had only seen temperate ones would let the system produce confident nonsense,
so its sensitivity is tested explicitly.
"""

from __future__ import annotations

import numpy as np
import pytest
from monitoring.drift import (
    PSI_MODERATE,
    PSI_NO_DRIFT,
    build_report,
    compare_distributions,
    compare_flag_frequencies,
    compare_observation_density,
    geographic_novelty,
    population_stability_index,
)


class TestPSI:
    def test_identical_samples_score_zero(self):
        rng = np.random.default_rng(0)
        x = rng.normal(0.8, 0.4, 20000)
        assert population_stability_index(x, x) == pytest.approx(0.0, abs=1e-9)

    def test_same_distribution_stays_below_no_drift_band(self):
        rng = np.random.default_rng(1)
        a = rng.normal(0.8, 0.4, 20000)
        b = rng.normal(0.8, 0.4, 20000)
        assert population_stability_index(a, b) < PSI_NO_DRIFT

    def test_shifted_distribution_exceeds_moderate_band(self):
        rng = np.random.default_rng(2)
        a = rng.normal(0.8, 0.4, 20000)
        b = rng.normal(1.8, 0.4, 20000)  # a full log10 higher: 6x the chlorophyll
        assert population_stability_index(a, b) > PSI_MODERATE

    def test_is_monotone_in_shift_magnitude(self):
        rng = np.random.default_rng(3)
        ref = rng.normal(0.8, 0.4, 20000)
        psis = [
            population_stability_index(ref, rng.normal(0.8 + d, 0.4, 20000))
            for d in (0.1, 0.5, 1.5)
        ]
        assert psis == sorted(psis)

    def test_handles_degenerate_inputs(self):
        assert np.isnan(population_stability_index(np.array([1.0, 2.0]), np.array([1.0])))
        assert np.isnan(population_stability_index(np.arange(50.0), np.array([])))

    def test_ignores_nans(self):
        rng = np.random.default_rng(4)
        a = rng.normal(0, 1, 5000)
        b = np.concatenate([a.copy(), np.full(500, np.nan)])
        assert population_stability_index(a, b) == pytest.approx(0.0, abs=1e-6)


class TestDistributionComparison:
    def test_no_drift_on_identical_data(self):
        rng = np.random.default_rng(5)
        x = rng.normal(0.8, 0.4, 10000)
        metrics = compare_distributions(x, x.copy())
        assert not any(m.drifted for m in metrics)

    def test_detects_mean_shift(self):
        rng = np.random.default_rng(6)
        a = rng.normal(0.8, 0.4, 10000)
        b = rng.normal(2.0, 0.4, 10000)
        metrics = compare_distributions(a, b)
        assert any(m.drifted for m in metrics)
        names = {m.name for m in metrics if m.drifted}
        assert "log_chl_psi" in names

    def test_detects_variance_change(self):
        rng = np.random.default_rng(7)
        a = rng.normal(0.8, 0.2, 10000)
        b = rng.normal(0.8, 1.2, 10000)
        metrics = compare_distributions(a, b)
        sd = next(m for m in metrics if m.name.endswith("sd_ratio"))
        assert sd.drifted


class TestObservationDensity:
    def test_stable_coverage_is_not_drift(self):
        m = compare_observation_density(0.30, 0.28)
        assert not m.drifted

    def test_halved_coverage_is_drift(self):
        """A coverage collapse usually means an upstream processing change."""
        m = compare_observation_density(0.30, 0.10)
        assert m.drifted
        assert m.value == pytest.approx(2 / 3, rel=1e-6)

    def test_zero_reference_does_not_divide_by_zero(self):
        m = compare_observation_density(0.0, 0.2)
        assert np.isnan(m.value)
        assert not m.drifted


class TestFlagFrequencies:
    def test_stable_flags_pass(self):
        ref = {"CLOUD": 0.15, "CLOUD_MARGIN": 0.20, "INLAND_WATER": 0.55}
        cur = {"CLOUD": 0.17, "CLOUD_MARGIN": 0.19, "INLAND_WATER": 0.54}
        assert not any(m.drifted for m in compare_flag_frequencies(ref, cur))

    def test_flag_jump_is_flagged(self):
        ref = {"CLOUD_MARGIN": 0.20, "INLAND_WATER": 0.55}
        cur = {"CLOUD_MARGIN": 0.55, "INLAND_WATER": 0.54}
        drifted = [m for m in compare_flag_frequencies(ref, cur) if m.drifted]
        assert [m.name for m in drifted] == ["flag_CLOUD_MARGIN"]

    def test_missing_flags_are_skipped_not_fatal(self):
        assert compare_flag_frequencies({"CLOUD": 0.1}, {"OTHER": 0.9}) == []


class TestGeographicNovelty:
    REF = {
        "median_chl": 0.8,
        "median_chl_sd": 0.2,
        "chl_spread": 0.4,
        "chl_spread_sd": 0.1,
        "seasonal_amplitude": 0.6,
        "seasonal_amplitude_sd": 0.15,
    }

    def test_similar_lake_is_not_novel(self):
        cur = {"median_chl": 0.85, "chl_spread": 0.42, "seasonal_amplitude": 0.58}
        assert not geographic_novelty(self.REF, cur).drifted

    def test_tropical_lake_is_flagged_as_novel(self):
        """The case the metric exists for: no seasonal cycle, different chlorophyll regime.

        A model trained only on temperate lakes will still emit a confident forecast here,
        so this must fire rather than pass quietly.
        """
        cur = {"median_chl": 1.6, "chl_spread": 0.9, "seasonal_amplitude": 0.05}
        m = geographic_novelty(self.REF, cur)
        assert m.drifted, f"tropical lake scored only {m.value:.2f} z-distance"

    def test_scales_with_distance(self):
        near = geographic_novelty(
            self.REF, {"median_chl": 0.9, "chl_spread": 0.45, "seasonal_amplitude": 0.6}
        ).value
        far = geographic_novelty(
            self.REF, {"median_chl": 2.0, "chl_spread": 1.2, "seasonal_amplitude": 0.0}
        ).value
        assert far > near

    def test_no_shared_descriptors_returns_nan(self):
        m = geographic_novelty({"a": 1.0}, {"b": 2.0})
        assert np.isnan(m.value)
        assert not m.drifted


class TestReport:
    def test_aggregates_and_serialises(self, tmp_path):
        rng = np.random.default_rng(8)
        metrics = compare_distributions(rng.normal(0.8, 0.4, 5000), rng.normal(2.5, 0.4, 5000))
        report = build_report("train_lakes", "victoria_ne", metrics)
        assert report.any_drift
        assert report.drifted_metrics

        path = report.save(tmp_path / "drift.json")
        assert path.exists()

        import json

        loaded = json.loads(path.read_text(encoding="utf-8"))
        assert loaded["current"] == "victoria_ne"
        assert loaded["any_drift"] is True

        rendered = report.render()
        assert "DRIFT DETECTED" in rendered
        assert "victoria_ne" in rendered

    def test_clean_report_says_no_drift(self):
        rng = np.random.default_rng(9)
        x = rng.normal(0.8, 0.4, 5000)
        report = build_report("train_lakes", "balaton", compare_distributions(x, x.copy()))
        assert not report.any_drift
        assert "no significant drift" in report.render()
