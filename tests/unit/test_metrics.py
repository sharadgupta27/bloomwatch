"""Tests for evaluation metrics.

The event scores and the skill score are what the README and model card quote, so their
edge cases matter more than usual. A silently wrong CSI or an inverted skill sign would
turn the project's headline claim into a false one.
"""

from __future__ import annotations

import numpy as np
import pytest

from habforecast.eval.metrics import (
    BLOOM_THRESHOLDS_MG_M3,
    evaluate_forecast,
    event_scores,
    regression_scores,
    skill_score,
)


class TestRegressionScores:
    def test_perfect_prediction(self):
        x = np.array([[0.5, 1.0], [1.5, 2.0]])
        s = regression_scores(x, x, np.ones_like(x))
        assert s.rmse == pytest.approx(0.0)
        assert s.mae == pytest.approx(0.0)
        assert s.bias == pytest.approx(0.0)
        assert s.n == 4

    def test_constant_offset_is_pure_bias(self):
        true = np.array([[1.0, 1.0], [1.0, 1.0]])
        pred = true + 0.25
        s = regression_scores(pred, true, np.ones_like(true))
        assert s.bias == pytest.approx(0.25)
        assert s.mae == pytest.approx(0.25)
        assert s.rmse == pytest.approx(0.25)

    def test_mask_excludes_pixels_from_every_statistic(self):
        true = np.zeros((2, 2))
        pred = np.zeros((2, 2))
        pred[0, 0] = 1000.0
        mask = np.ones((2, 2))
        mask[0, 0] = 0
        s = regression_scores(pred, true, mask)
        assert s.n == 3
        assert s.rmse == pytest.approx(0.0)

    def test_nan_inputs_are_dropped(self):
        true = np.array([1.0, 2.0, np.nan])
        pred = np.array([1.0, 2.0, 5.0])
        s = regression_scores(pred, true, np.ones(3))
        assert s.n == 2
        assert s.rmse == pytest.approx(0.0)

    def test_empty_mask_returns_nan_not_crash(self):
        s = regression_scores(np.ones((2, 2)), np.zeros((2, 2)), np.zeros((2, 2)))
        assert s.n == 0
        assert np.isnan(s.rmse)


class TestEventScores:
    def test_contingency_table_counts(self):
        # threshold 20 mg/m3 -> log10 = 1.301
        true = np.log10(np.array([50.0, 50.0, 5.0, 5.0]))
        pred = np.log10(np.array([50.0, 5.0, 50.0, 5.0]))
        ev = event_scores(pred, true, np.ones(4), threshold_mg_m3=20.0)
        assert (ev.hits, ev.misses, ev.false_alarms, ev.correct_negatives) == (1, 1, 1, 1)

    def test_derived_rates(self):
        true = np.log10(np.array([50.0, 50.0, 5.0, 5.0]))
        pred = np.log10(np.array([50.0, 5.0, 50.0, 5.0]))
        ev = event_scores(pred, true, np.ones(4), 20.0)
        assert ev.pod == pytest.approx(0.5)
        assert ev.far == pytest.approx(0.5)
        assert ev.csi == pytest.approx(1 / 3)
        assert ev.f1 == pytest.approx(0.5)

    def test_csi_ignores_correct_negatives(self):
        """CSI must not be inflated by the huge clear-water class.

        This is why CSI is the operational standard rather than accuracy: 99% of pixels
        are below threshold, so accuracy would read ~0.99 for a model that never
        predicts a bloom at all.
        """
        true = np.full(1000, np.log10(1.0))
        pred = np.full(1000, np.log10(1.0))
        true[:2] = np.log10(50.0)  # two real bloom pixels, both missed
        ev = event_scores(pred, true, np.ones(1000), 20.0)
        assert ev.correct_negatives == 998
        assert ev.csi == pytest.approx(0.0)
        assert ev.pod == pytest.approx(0.0)

    def test_no_events_gives_nan_rates(self):
        flat = np.full(4, np.log10(1.0))
        ev = event_scores(flat, flat, np.ones(4), 20.0)
        assert np.isnan(ev.pod)
        assert np.isnan(ev.csi)

    @pytest.mark.parametrize("threshold", BLOOM_THRESHOLDS_MG_M3)
    def test_all_configured_thresholds_work(self, threshold):
        true = np.log10(np.array([1.0, 15.0, 30.0, 80.0]))
        ev = event_scores(true, true, np.ones(4), threshold)
        assert ev.misses == 0 and ev.false_alarms == 0

    def test_threshold_ordering_is_monotone(self):
        """A higher threshold can never flag more pixels than a lower one."""
        rng = np.random.default_rng(0)
        pred = np.log10(rng.uniform(1, 200, 500))
        true = np.log10(rng.uniform(1, 200, 500))
        counts = [
            event_scores(pred, true, np.ones(500), t).hits
            + event_scores(pred, true, np.ones(500), t).false_alarms
            for t in (10.0, 20.0, 50.0)
        ]
        assert counts == sorted(counts, reverse=True)


class TestSkillScore:
    def test_perfect_model_scores_one(self):
        assert skill_score(0.0, 1.0) == pytest.approx(1.0)

    def test_matching_baseline_scores_zero(self):
        assert skill_score(0.5, 0.5) == pytest.approx(0.0)

    def test_worse_than_baseline_is_negative(self):
        assert skill_score(1.0, 0.5) < 0

    def test_zero_baseline_returns_nan(self):
        assert np.isnan(skill_score(0.5, 0.0))


class TestEvaluateForecast:
    def test_assembles_full_result(self):
        rng = np.random.default_rng(1)
        true = np.log10(rng.uniform(1, 60, (4, 16, 16)))
        pred = true + rng.normal(0, 0.1, true.shape)
        pers = true + rng.normal(0, 0.3, true.shape)
        mask = (rng.random(true.shape) < 0.5).astype(float)

        ev = evaluate_forecast(pred, true, mask, "erie_west", 7, persistence_log=pers)
        assert ev.lake_id == "erie_west"
        assert ev.lead_days == 7
        assert len(ev.events) == len(BLOOM_THRESHOLDS_MG_M3)
        # pred is closer to truth than persistence, so skill must be positive.
        assert ev.skill_vs_persistence > 0

    def test_to_dict_is_flat_and_serialisable(self):
        import json

        true = np.log10(np.full((2, 4, 4), 30.0))
        ev = evaluate_forecast(true, true, np.ones_like(true), "taihu", 3, persistence_log=true)
        d = ev.to_dict()
        assert d["lake_id"] == "taihu"
        assert "thr20_csi" in d
        assert all(not isinstance(v, dict) for v in d.values())
        json.dumps(d)  # must not raise


class TestStreamingEquivalence:
    """Streaming accumulators must reproduce the batch metrics exactly.

    They exist because collecting arrays first needed 9.6 GB for one split and ran the
    machine out of memory. That is only a safe substitution if the numbers are identical,
    so the equivalence is pinned rather than assumed.
    """

    @staticmethod
    def _chunks(seed=0, n=25, shape=(12, 14)):
        rng = np.random.default_rng(seed)
        for _ in range(n):
            true = np.log10(rng.uniform(0.5, 120.0, shape))
            pred = true + rng.normal(0, 0.25, shape)
            mask = (rng.random(shape) < 0.6).astype(float)
            yield pred, true, mask

    def test_regression_matches_batch(self):
        from habforecast.eval.metrics import StreamingRegression

        chunks = list(self._chunks(seed=1))
        acc = StreamingRegression()
        for p, t, m in chunks:
            acc.update(p, t, m)
        streamed = acc.result()

        batch = regression_scores(
            np.stack([c[0] for c in chunks]),
            np.stack([c[1] for c in chunks]),
            np.stack([c[2] for c in chunks]),
        )
        assert streamed.n == batch.n
        assert streamed.rmse == pytest.approx(batch.rmse, rel=1e-9)
        assert streamed.mae == pytest.approx(batch.mae, rel=1e-9)
        assert streamed.bias == pytest.approx(batch.bias, abs=1e-9)
        assert streamed.r2 == pytest.approx(batch.r2, rel=1e-6)

    @pytest.mark.parametrize("threshold", BLOOM_THRESHOLDS_MG_M3)
    def test_events_match_batch(self, threshold):
        from habforecast.eval.metrics import StreamingEvents

        chunks = list(self._chunks(seed=2))
        acc = StreamingEvents(threshold)
        for p, t, m in chunks:
            acc.update(p, t, m)
        streamed = acc.result()

        batch = event_scores(
            np.stack([c[0] for c in chunks]),
            np.stack([c[1] for c in chunks]),
            np.stack([c[2] for c in chunks]),
            threshold,
        )
        assert (streamed.hits, streamed.misses, streamed.false_alarms) == (
            batch.hits,
            batch.misses,
            batch.false_alarms,
        )
        assert streamed.csi == pytest.approx(batch.csi, rel=1e-12)

    def test_full_evaluation_matches_batch(self):
        from habforecast.eval.metrics import StreamingEvaluation

        chunks = list(self._chunks(seed=3))
        pers = [c[1] + 0.4 for c in chunks]  # a deliberately worse baseline

        acc = StreamingEvaluation("erie_west", 7)
        for (p, t, m), pr in zip(chunks, pers, strict=True):
            acc.update(p, t, m, persistence=pr)
        streamed = acc.result()

        batch = evaluate_forecast(
            np.stack([c[0] for c in chunks]),
            np.stack([c[1] for c in chunks]),
            np.stack([c[2] for c in chunks]),
            "erie_west",
            7,
            persistence_log=np.stack(pers),
        )
        assert streamed.regression.rmse == pytest.approx(batch.regression.rmse, rel=1e-9)
        assert streamed.persistence_rmse == pytest.approx(batch.persistence_rmse, rel=1e-9)
        assert streamed.skill_vs_persistence == pytest.approx(batch.skill_vs_persistence, rel=1e-9)
        assert streamed.skill_vs_persistence > 0

    def test_empty_stream_is_nan_not_crash(self):
        from habforecast.eval.metrics import StreamingEvaluation

        acc = StreamingEvaluation("x", 1)
        res = acc.result()
        assert res.regression.n == 0
        assert np.isnan(res.regression.rmse)

    def test_fully_masked_updates_contribute_nothing(self):
        from habforecast.eval.metrics import StreamingRegression

        acc = StreamingRegression()
        acc.update(np.ones((4, 4)), np.zeros((4, 4)), np.zeros((4, 4)))
        assert acc.result().n == 0

    def test_memory_is_constant_in_stream_length(self):
        """The accumulator must not grow with the number of samples."""
        from habforecast.eval.metrics import StreamingRegression

        acc = StreamingRegression()
        assert not hasattr(acc, "__dict__"), "__slots__ dropped; accumulator may grow"
        for p, t, m in self._chunks(seed=4, n=200, shape=(32, 32)):
            acc.update(p, t, m)
        assert acc.result().n > 0

    def test_r2_survives_a_large_offset(self):
        """Regression guard for catastrophic cancellation in the variance.

        With `sum(t^2) - n*mean^2`, truth values around 1e6 with a 1e-3 spread produce a
        sum of squares 400x too large (8.0 against a true 0.02), silently wrecking R^2.
        Welford's parallel form is exact here.
        """
        from habforecast.eval.metrics import StreamingRegression

        rng = np.random.default_rng(0)
        offset = 1e6
        chunks = []
        for _ in range(50):
            t = offset + rng.normal(0, 1e-3, (20, 20))
            p = t + rng.normal(0, 1e-3, (20, 20))
            chunks.append((p, t, np.ones((20, 20))))

        acc = StreamingRegression()
        for p, t, m in chunks:
            acc.update(p, t, m)

        batch = regression_scores(
            np.stack([c[0] for c in chunks]),
            np.stack([c[1] for c in chunks]),
            np.stack([c[2] for c in chunks]),
        )
        assert acc.result().r2 == pytest.approx(batch.r2, abs=1e-6)


class TestStreamingRanking:
    """ROC-AUC / PR-AUC accumulated as score histograms must match an exact ranking."""

    @staticmethod
    def _reference(p: np.ndarray, y: np.ndarray) -> tuple[float, float]:
        order = np.argsort(-p, kind="stable")
        ys = y[order]
        tp, fp = np.cumsum(ys), np.cumsum(~ys)
        tpr = np.concatenate([[0.0], tp / ys.sum()])
        fpr = np.concatenate([[0.0], fp / (~ys).sum()])
        trapezoid = getattr(np, "trapezoid", None) or np.trapz  # type: ignore[attr-defined]
        roc = float(trapezoid(tpr, fpr))
        precision, recall = tp / (tp + fp), tp / ys.sum()
        ap = float(np.sum(np.diff(np.concatenate([[0.0], recall])) * precision))
        return roc, ap

    def test_matches_exact_ranking_within_bin_resolution(self):
        from habforecast.eval.metrics import StreamingRanking

        rng = np.random.default_rng(7)
        truth = rng.normal(1.0, 0.5, 50_000)
        pred = truth + rng.normal(0.0, 0.4, truth.size)
        rk = StreamingRanking(20.0)
        for chunk in np.array_split(np.arange(truth.size), 5):
            rk.update_masked(pred[chunk], truth[chunk])
        got = rk.result()
        roc, ap = self._reference(pred, truth >= np.log10(20.0))
        assert got.roc_auc == pytest.approx(roc, abs=3e-3)
        assert got.pr_auc == pytest.approx(ap, abs=3e-3)
        assert got.prevalence == pytest.approx(float((truth >= np.log10(20.0)).mean()))

    def test_perfect_ranking_is_one(self):
        from habforecast.eval.metrics import StreamingRanking

        truth = np.linspace(0.0, 2.0, 1000)
        rk = StreamingRanking(20.0)
        rk.update_masked(truth, truth)
        got = rk.result()
        assert got.roc_auc == pytest.approx(1.0, abs=1e-3)
        assert got.pr_auc == pytest.approx(1.0, abs=1e-3)

    def test_constant_score_is_chance(self):
        """A forecast that says the same thing everywhere cannot rank: ROC-AUC 0.5,
        PR-AUC equal to the prevalence."""
        from habforecast.eval.metrics import StreamingRanking

        rng = np.random.default_rng(1)
        truth = rng.normal(1.2, 0.4, 20_000)
        rk = StreamingRanking(20.0)
        rk.update_masked(np.full_like(truth, 1.0), truth)
        got = rk.result()
        assert got.roc_auc == pytest.approx(0.5, abs=1e-9)
        assert got.pr_auc == pytest.approx(got.prevalence, abs=1e-9)

    def test_single_class_is_nan_not_crash(self):
        from habforecast.eval.metrics import StreamingRanking

        rk = StreamingRanking(20.0)
        rk.update_masked(np.array([0.1, 0.2]), np.array([0.1, 0.2]))  # no positives
        got = rk.result()
        assert np.isnan(got.roc_auc) and np.isnan(got.pr_auc)

    def test_out_of_range_scores_are_clipped_not_dropped(self):
        from habforecast.eval.metrics import RANK_RANGE, StreamingRanking

        rk = StreamingRanking(20.0)
        rk.update_masked(np.array([RANK_RANGE[0] - 5, RANK_RANGE[1] + 5]), np.array([0.0, 2.0]))
        assert rk.result().n == 2

    def test_evaluation_reports_model_and_baseline_aucs(self):
        from habforecast.eval.metrics import StreamingEvaluation

        rng = np.random.default_rng(3)
        acc = StreamingEvaluation("lake", 3)
        for _ in range(4):
            truth = rng.normal(1.2, 0.4, (32, 32))
            acc.update(
                truth + rng.normal(0, 0.2, truth.shape),
                truth,
                np.ones_like(truth),
                persistence=truth + rng.normal(0, 0.5, truth.shape),
                climatology=np.full_like(truth, 1.2),
            )
        row = acc.result().to_dict()
        base = acc.baseline_rankings()
        for thr in BLOOM_THRESHOLDS_MG_M3:
            assert f"thr{thr:g}_roc_auc" in row and f"thr{thr:g}_pr_auc" in row
            assert f"persistence_thr{thr:g}_roc_auc" in base
            assert f"climatology_thr{thr:g}_pr_auc" in base
        # The sharper forecast must rank better than the noisier baseline.
        assert row["thr20_roc_auc"] > base["persistence_thr20_roc_auc"]
        # A constant climatology cannot rank at all.
        assert base["climatology_thr20_roc_auc"] == pytest.approx(0.5, abs=1e-9)
