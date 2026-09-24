"""Exceedance head, probabilistic scoring, ensembles and the probabilistic advisory.

What these pin:

- the head starts life ranking pixels exactly as the regression does, so it has to earn
  every departure (the same "start as the thing it extends" rule as the rest of the model);
- the focal loss ignores unobserved pixels and reduces to cross-entropy at gamma = 0;
- test-time augmentation keeps the probability map registered with the forecast;
- the reliability accumulator reports a perfectly calibrated forecast as calibrated;
- an ensemble's spread is zero for identical members and its mean equals one member;
- the advisory says "uncertain" exactly when the likely and possible levels disagree.
"""

from __future__ import annotations

import math
from datetime import date

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from habforecast import advisory
from habforecast.eval.metrics import StreamingEvaluation, StreamingReliability
from habforecast.inference.predictor import BaselinePredictor, ForecastResult, ensemble_forecast
from habforecast.models.anomaly import AnomalyForecaster
from habforecast.models.baselines import PersistenceBaseline
from habforecast.models.lit import HABForecastModule, masked_focal, model_inputs, predict_tta

THRESHOLDS = (10.0, 20.0, 50.0)
NORM = (1.0, 0.45)


def _model(seed: int = 0, **kw) -> AnomalyForecaster:
    torch.manual_seed(seed)
    return AnomalyForecaster(
        use_clim=True,
        exceed_thresholds=THRESHOLDS,
        norm_mean=NORM[0],
        norm_std=NORM[1],
        **kw,
    ).eval()


def _batch(b: int = 2, h: int = 48, w: int = 40, seed: int = 0) -> dict:
    g = torch.Generator().manual_seed(seed)
    x_mask = (torch.rand(b, 14, h, w, generator=g) > 0.5).float()
    return {
        "x": torch.randn(b, 14, h, w, generator=g) * x_mask,
        "x_mask": x_mask,
        "water": torch.ones(b, 1, h, w),
        "doy": torch.tensor([150.0, 220.0][:b]),
        "clim": torch.randn(b, 4, h, w, generator=g),
        "clim_mask": torch.ones(b, 4, h, w),
        "y": torch.randn(b, 4, h, w, generator=g),
        "y_mask": (torch.rand(b, 4, h, w, generator=g) > 0.4).float(),
    }


class TestExceedanceHead:
    def test_untrained_head_is_a_monotone_function_of_the_regression(self):
        m = _model()
        batch = _batch()
        with torch.no_grad():
            pred, logits = m(**model_inputs(m, batch), return_exceed=True)
        assert logits.shape == (2, 4, 3, 48, 40)
        for k, thr in enumerate(THRESHOLDS):
            thr_std = (math.log10(thr) - NORM[0]) / NORM[1]
            torch.testing.assert_close(logits[:, :, k], 3.0 * (pred - thr_std), atol=1e-4, rtol=0)

    def test_regression_output_unchanged_by_asking_for_the_head(self):
        m = _model()
        batch = _batch()
        with torch.no_grad():
            plain = m(**model_inputs(m, batch))
            pred, _ = m(**model_inputs(m, batch), return_exceed=True)
        torch.testing.assert_close(plain, pred)

    def test_head_cannot_move_the_regression(self):
        """The margin is detached: the focal loss reaches the trunk, never the map's anchor
        arithmetic through the margin."""
        m = _model()
        m.train()
        batch = _batch()
        _, logits = m(**model_inputs(m, batch), return_exceed=True)
        logits.sum().backward()
        # The delta heads only affect the regression; their gradient must come from the
        # trunk features the head shares, which the zero-initialised heads do not see.
        assert m.head_coarse.weight.grad is None or torch.all(m.head_coarse.weight.grad == 0)

    def test_model_without_thresholds_refuses_return_exceed(self):
        m = AnomalyForecaster(use_clim=True)
        with pytest.raises(ValueError):
            m(**model_inputs(m, _batch()), return_exceed=True)

    def test_training_step_adds_focal_loss(self):
        m = _model()
        module = HABForecastModule(m, lead_times=(1, 3, 5, 7))
        module.log = lambda *a, **k: None  # no trainer attached
        loss = module._step(_batch(), "train")
        assert torch.isfinite(loss)
        loss.backward()
        assert m.exceed_head.out_weight.grad is not None


class TestFocalLoss:
    def test_gamma_zero_is_masked_cross_entropy(self):
        logits = torch.randn(2, 4, 3, 5, 5)
        target = (torch.rand_like(logits) > 0.7).float()
        mask = (torch.rand_like(logits) > 0.3).float()
        got = masked_focal(logits, target, mask, gamma=0.0)
        ce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        torch.testing.assert_close(got, (ce * mask).sum() / mask.sum())

    def test_unobserved_pixels_contribute_nothing(self):
        logits = torch.zeros(1, 1, 1, 2, 2)
        target = torch.tensor([[[[[1.0, 0.0], [1.0, 0.0]]]]])
        mask = torch.tensor([[[[[1.0, 1.0], [0.0, 0.0]]]]])
        wild = logits.clone()
        wild[..., 1, :] = 50.0  # nonsense where unobserved
        assert masked_focal(logits, target, mask) == pytest.approx(
            float(masked_focal(wild, target, mask))
        )

    def test_focusing_down_weights_easy_pixels(self):
        target = torch.ones(1, 1, 1, 1, 2)
        mask = torch.ones_like(target)
        easy = torch.tensor([[[[[6.0, 6.0]]]]])
        assert masked_focal(easy, target, mask, gamma=2.0) < masked_focal(
            easy, target, mask, gamma=0.0
        )


class TestTTA:
    def test_probability_is_registered_with_the_forecast(self):
        m = _model()
        batch = _batch()
        inputs = model_inputs(m, batch)
        with torch.no_grad():
            pred, prob = predict_tta(m, inputs, flips=True, with_exceed=True)
            pred_plain = predict_tta(m, inputs, flips=True)
        torch.testing.assert_close(pred, pred_plain)
        assert prob.shape == (2, 4, 3, 48, 40)
        assert torch.all((prob >= 0) & (prob <= 1))
        # Untrained: p is a monotone function of each orientation's forecast, so the
        # pixel ranking of p follows the TTA-mean forecast closely.
        flat_p = prob[:, 0, 1].flatten()
        flat_f = pred[:, 0].flatten()
        corr = torch.corrcoef(torch.stack([flat_p, flat_f]))[0, 1]
        assert corr > 0.8


class TestEnsemble:
    def test_identical_members_have_zero_spread(self):
        m = _model()
        batch = _batch()
        mean, prob, spread = ensemble_forecast([m, m], batch, tta=False)
        with torch.no_grad():
            single = m(**model_inputs(m, batch))
        torch.testing.assert_close(mean, single)
        assert prob is not None
        assert float(spread.abs().max()) < 1e-5

    def test_different_seeds_spread_and_single_member_has_none(self):
        batch = _batch()
        a, b = _model(0), _model(1)
        # Untrained anomaly models are persistence; perturb the heads so they differ.
        with torch.no_grad():
            for i, m in enumerate((a, b)):
                m.head_mid.bias.fill_(0.1 * (i + 1))
        _, _, spread = ensemble_forecast([a, b], batch, tta=False)
        assert float(spread.max()) > 0
        _, _, none = ensemble_forecast([a], batch, tta=False)
        assert none is None

    def test_mixed_ensemble_returns_no_probability(self):
        batch = _batch()
        plain = AnomalyForecaster(use_clim=True).eval()
        _, prob, _ = ensemble_forecast([_model(), plain], batch, tta=False)
        assert prob is None


class TestReliability:
    def test_calibrated_forecast(self):
        rng = np.random.default_rng(0)
        p = rng.uniform(0, 1, 200_000)
        exceed = rng.uniform(0, 1, p.size) < p
        truth = np.where(exceed, np.log10(30.0), np.log10(5.0))
        rel = StreamingReliability(20.0)
        rel.update_masked(p, truth)
        out = rel.result()
        assert out["thr20_ece"] < 0.01
        # Brier of a calibrated uniform forecast is E[p(1-p)] = 1/6; base rate 0.5 -> 0.25.
        assert out["thr20_brier"] == pytest.approx(1 / 6, abs=0.005)
        assert out["thr20_bss"] == pytest.approx(1 - (1 / 6) / 0.25, abs=0.02)

    def test_overconfident_forecast_has_large_ece(self):
        truth = np.array([np.log10(30.0)] * 50 + [np.log10(5.0)] * 50)
        rel = StreamingReliability(20.0)
        rel.update_masked(np.full(100, 0.99), truth)
        assert rel.result()["thr20_ece"] > 0.4

    def test_head_ranking_reported_only_with_probabilities(self):
        ev = StreamingEvaluation("lake", 1)
        truth = np.log10(np.array([[5.0, 30.0], [60.0, 8.0]]))
        mask = np.ones_like(truth)
        ev.update(truth, truth, mask)
        assert ev.probabilistic() == {}
        prob = np.stack([(truth >= np.log10(t)).astype(float) for t in THRESHOLDS])
        ev.update(truth, truth, mask, prob=prob, spread=np.full_like(truth, 0.1))
        out = ev.probabilistic()
        assert out["head_thr20_pr_auc"] == pytest.approx(1.0)
        assert out["ensemble_spread"] == pytest.approx(0.1)


class TestBaselinePredictor:
    def test_numpy_persistence_matches_torch(self):
        batch = _batch()
        ref = PersistenceBaseline(n_leads=4)(batch["x"], batch["x_mask"], batch["water"])
        got, prob, spread = BaselinePredictor().forward_numpy(
            batch["x"].numpy(), batch["x_mask"].numpy(), batch["water"].numpy(), None
        )
        np.testing.assert_allclose(got, ref.numpy(), atol=1e-6)
        assert prob is None and spread is None


def _prob_result(p20: float, spread_band: float) -> ForecastResult:
    """A lake where every water pixel has p(>20) = p20 +- spread_band (half each way)."""
    h, w = 10, 10
    water = np.ones((h, w), dtype=bool)
    prob = np.zeros((4, 3, h, w), dtype=np.float32)
    field = np.full((h, w), p20, dtype=np.float32)
    field[: h // 2] += spread_band
    field[h // 2 :] -= spread_band
    prob[:, 1] = np.clip(field, 0, 1)
    prob[:, 0] = np.clip(field + 0.2, 0, 1)
    return ForecastResult(
        lake_id="erie_west",
        lake_name="Erie",
        anchor_date=date(2024, 8, 15),
        lead_times=(1, 3, 5, 7),
        log_chl=np.full((4, h, w), np.log10(15.0), dtype=np.float32),
        water=water,
        input_observed_days=[date(2024, 8, 14)],
        exceed_prob=prob,
        exceed_thresholds=THRESHOLDS,
    )


class TestProbabilisticAdvisory:
    def test_confident_warning(self):
        adv = advisory.advise(_prob_result(0.9, 0.05), 0.09)
        assert adv.level == "warning"
        assert adv.confidence == "confident"
        assert adv.peak_fraction["20"] == pytest.approx(0.9, abs=1e-6)

    def test_uncertain_when_band_straddles_a_level(self):
        # Expected fraction above 20 is 0.5 of the lake, but no pixel is "likely".
        adv = advisory.advise(_prob_result(0.5, 0.1), 0.09)
        assert adv.level == "warning"
        assert adv.likely_level == "watch"
        assert adv.possible_level == "warning"
        assert adv.confidence == "uncertain"

    def test_feed_carries_band_and_headline_says_uncertain(self, tmp_path):
        from tests.unit.test_advisory import REPO_ROOT

        from habforecast.aoi import AOIRegistry

        registry = AOIRegistry.from_geojson(REPO_ROOT / "data" / "aoi" / "lakes.geojson")
        adv = advisory.advise(_prob_result(0.5, 0.1), 0.09)
        feat = advisory.feature(adv, registry["erie_west"])
        props = feat["properties"]
        assert props["basis"] == "probabilistic"
        assert props["fraction_band"]["20"]["likely"] == 0.0
        assert "UNCERTAIN" in advisory._headline(props)
        assert "exceed_probability" in props["forecasts"][0]

    def test_deterministic_forecast_is_unchanged(self):
        r = _prob_result(0.9, 0.0)
        r.exceed_prob = None
        adv = advisory.advise(r, 0.09)
        assert adv.confidence is None
