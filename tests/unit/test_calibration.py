"""Exceedance-probability recalibration: identity without a fit, monotone with one, and a
fit that recovers a known miscalibration from a binned reliability table."""

from __future__ import annotations

import json

import numpy as np
import pytest

from habforecast import calibration
from habforecast.eval.metrics import StreamingReliability

LEADS, THRS = (1, 3), (10.0, 20.0)


def _prob(seed=0):
    return np.random.default_rng(seed).uniform(0.01, 0.99, size=(2, 2, 2, 5, 5)).astype(np.float32)


def test_no_calibration_is_identity():
    p = _prob()
    assert calibration.apply(p, None, LEADS, THRS) is p


def test_missing_key_is_identity_and_map_is_monotone():
    p = _prob()
    calib = {"params": {"1:20": [1.5, -1.0]}}
    q = calibration.apply(p, calib, LEADS, THRS)
    np.testing.assert_array_equal(q[:, 1], p[:, 1])  # lead 3 untouched
    np.testing.assert_array_equal(q[:, 0, 0], p[:, 0, 0])  # threshold 10 untouched
    order_p = np.argsort(p[:, 0, 1].ravel())
    assert np.all(np.diff(q[:, 0, 1].ravel()[order_p]) >= 0)


def test_fit_recovers_a_known_distortion():
    rng = np.random.default_rng(1)
    true_p = rng.uniform(0.001, 0.999, 400_000)
    exceed = rng.uniform(size=true_p.size) < true_p
    # An overconfident forecaster: logit inflated by 2 and shifted by +1.
    z = np.log(true_p / (1 - true_p))
    shown = 1 / (1 + np.exp(-(2.0 * z + 1.0)))
    rel = StreamingReliability(20.0, n_bins=20)
    rel.update_masked(shown, np.where(exceed, np.log10(30.0), np.log10(5.0)))
    a, b = calibration.fit_from_reliability(rel.result()["thr20_reliability"])
    assert a == pytest.approx(0.5, abs=0.08)
    assert b == pytest.approx(-0.5, abs=0.12)


def test_too_few_exceedances_refuses_to_fit():
    bins = [{"n": 10_000, "mean_prob": 0.05, "obs_freq": 0.001}]
    assert calibration.fit_from_reliability(bins) is None


def test_paths_and_loading(tmp_path):
    ckpt = tmp_path / "m.ckpt"
    assert calibration.calibration_path(ckpt).name == "m.calibration.json"
    ens = tmp_path / "ens"
    ens.mkdir()
    (ens / "calibration.json").write_text(json.dumps({"params": {}}))
    assert calibration.load(ens) == {"params": {}}
    assert calibration.load(ckpt) is None
