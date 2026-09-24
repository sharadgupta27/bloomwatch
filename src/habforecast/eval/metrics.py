"""Evaluation metrics for masked bloom forecasts.

Two families, because they answer different questions:

*Regression* metrics (RMSE, MAE, bias) say how close the predicted chlorophyll field is.
They are computed in log10 space, which is where the error is roughly homoscedastic --
reporting RMSE in raw mg/m^3 would be dominated entirely by a handful of scum pixels.

*Event* metrics (POD, FAR, CSI, F1) say whether the forecast would have triggered the
right advisory. This is what actually matters operationally: a water manager does not
act on a chlorophyll value, they act on a threshold crossing. A model can have excellent
RMSE and still be useless if it never crosses the advisory threshold.

Every metric is masked. Pixels that were cloudy in the target frame contribute nothing,
and land contributes nothing. Unmasked metrics on this data are meaningless -- they are
dominated by the ~60-75% of pixels that were never observed.

Skill is always reported *relative to persistence*. For short-range forecasting of a
slowly-evolving field, persistence is a genuinely strong baseline, and a model that
cannot beat it has not earned its complexity.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Chlorophyll-a thresholds in mg/m^3 (== ug/L), converted to log10 internally.
# 20 is a widely used bloom threshold for lakes; 50 aligns with WHO's moderate
# recreational health risk tier for cyanobacterial dominance.
BLOOM_THRESHOLDS_MG_M3: tuple[float, ...] = (10.0, 20.0, 50.0)


def _masked(pred: np.ndarray, true: np.ndarray, mask: np.ndarray):
    m = mask.astype(bool) & np.isfinite(pred) & np.isfinite(true)
    return pred[m], true[m], int(m.sum())


@dataclass
class RegressionScores:
    n: int
    rmse: float
    mae: float
    bias: float
    r2: float

    def to_dict(self, prefix: str = "") -> dict[str, float]:
        return {
            f"{prefix}n": float(self.n),
            f"{prefix}rmse": self.rmse,
            f"{prefix}mae": self.mae,
            f"{prefix}bias": self.bias,
            f"{prefix}r2": self.r2,
        }


def regression_scores(pred: np.ndarray, true: np.ndarray, mask: np.ndarray) -> RegressionScores:
    """RMSE / MAE / bias / R^2 in log10 space over masked pixels."""
    p, t, n = _masked(pred, true, mask)
    if n == 0:
        nan = float("nan")
        return RegressionScores(0, nan, nan, nan, nan)
    err = p - t
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((t - t.mean()) ** 2))
    return RegressionScores(
        n=n,
        rmse=float(np.sqrt(np.mean(err**2))),
        mae=float(np.mean(np.abs(err))),
        bias=float(np.mean(err)),
        r2=float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
    )


@dataclass
class EventScores:
    threshold_mg_m3: float
    n: int
    hits: int
    misses: int
    false_alarms: int
    correct_negatives: int

    @property
    def pod(self) -> float:
        """Probability of detection (recall) -- fraction of real bloom pixels caught."""
        d = self.hits + self.misses
        return self.hits / d if d else float("nan")

    @property
    def far(self) -> float:
        """False alarm ratio -- fraction of bloom warnings that were wrong."""
        d = self.hits + self.false_alarms
        return self.false_alarms / d if d else float("nan")

    @property
    def csi(self) -> float:
        """Critical success index (threat score) -- the standard skill score in
        operational forecasting; ignores the huge correct-negative class."""
        d = self.hits + self.misses + self.false_alarms
        return self.hits / d if d else float("nan")

    @property
    def f1(self) -> float:
        d = 2 * self.hits + self.misses + self.false_alarms
        return 2 * self.hits / d if d else float("nan")

    def to_dict(self, prefix: str = "") -> dict[str, float]:
        tag = f"{prefix}thr{self.threshold_mg_m3:g}_"
        return {
            f"{tag}pod": self.pod,
            f"{tag}far": self.far,
            f"{tag}csi": self.csi,
            f"{tag}f1": self.f1,
            f"{tag}hits": float(self.hits),
            f"{tag}misses": float(self.misses),
            f"{tag}false_alarms": float(self.false_alarms),
        }


def event_scores(
    pred_log: np.ndarray,
    true_log: np.ndarray,
    mask: np.ndarray,
    threshold_mg_m3: float,
) -> EventScores:
    """Contingency-table scores for crossing a chlorophyll advisory threshold."""
    p, t, n = _masked(pred_log, true_log, mask)
    thr = np.log10(threshold_mg_m3)
    if n == 0:
        return EventScores(threshold_mg_m3, 0, 0, 0, 0, 0)
    pe, te = p >= thr, t >= thr
    return EventScores(
        threshold_mg_m3=threshold_mg_m3,
        n=n,
        hits=int(np.sum(pe & te)),
        misses=int(np.sum(~pe & te)),
        false_alarms=int(np.sum(pe & ~te)),
        correct_negatives=int(np.sum(~pe & ~te)),
    )


def skill_score(model_rmse: float, baseline_rmse: float) -> float:
    """Murphy skill score: 1 = perfect, 0 = no better than baseline, <0 = worse."""
    if not np.isfinite(baseline_rmse) or baseline_rmse <= 0:
        return float("nan")
    return 1.0 - (model_rmse**2) / (baseline_rmse**2)


@dataclass
class ForecastEvaluation:
    """Full evaluation for one (lake, lead time) cell of the results table."""

    lake_id: str
    lead_days: int
    regression: RegressionScores
    events: list[EventScores] = field(default_factory=list)
    persistence_rmse: float = float("nan")
    rankings: list[RankingScores] = field(default_factory=list)

    @property
    def skill_vs_persistence(self) -> float:
        return skill_score(self.regression.rmse, self.persistence_rmse)

    def to_dict(self) -> dict[str, float | str | int]:
        out: dict[str, float | str | int] = {
            "lake_id": self.lake_id,
            "lead_days": self.lead_days,
            "persistence_rmse": self.persistence_rmse,
            "skill_vs_persistence": self.skill_vs_persistence,
        }
        out.update(self.regression.to_dict())
        for ev in self.events:
            out.update(ev.to_dict())
        for rk in self.rankings:
            out.update(rk.to_dict())
        return out


def evaluate_forecast(
    pred_log: np.ndarray,
    true_log: np.ndarray,
    mask: np.ndarray,
    lake_id: str,
    lead_days: int,
    persistence_log: np.ndarray | None = None,
    thresholds: tuple[float, ...] = BLOOM_THRESHOLDS_MG_M3,
) -> ForecastEvaluation:
    """Score one lake / lead-time combination against observed truth."""
    reg = regression_scores(pred_log, true_log, mask)
    events = [event_scores(pred_log, true_log, mask, thr) for thr in thresholds]
    pers = (
        regression_scores(persistence_log, true_log, mask).rmse
        if persistence_log is not None
        else float("nan")
    )
    return ForecastEvaluation(
        lake_id=lake_id,
        lead_days=lead_days,
        regression=reg,
        events=events,
        persistence_rmse=pers,
    )


# ---------------------------------------------------------------------------------
# Streaming accumulators
#
# The batch functions above take whole arrays, which is fine for a single frame and
# catastrophic for a whole split: scoring Lake Victoria's 916 test samples by collecting
# arrays first needs 9.6 GB of RAM (916 samples x 4 leads x 5 arrays x 522 KB) before any
# metric is computed, and duly ran the machine out of memory.
#
# Every quantity reported is expressible from running sums, so the accumulators below
# produce bit-comparable results in O(1) memory. `tests/unit/test_metrics.py` pins that
# equivalence against the batch implementations.
# ---------------------------------------------------------------------------------


class StreamingRegression:
    """Running RMSE / MAE / bias / R^2 over masked pixels.

    The variance of the truth (needed for R^2) is accumulated with the parallel form of
    Welford's algorithm rather than as ``sum(t^2) - n * mean^2``. The naive form suffers
    catastrophic cancellation when the mean is large relative to the spread, and can even
    return a negative sum of squares. It happens to be safe for log10 chlorophyll
    (mean ~1, sd ~0.4) but is a trap waiting for any variable with a large offset.
    """

    __slots__ = ("_m2_true", "_mean_true", "n", "sum_abs_err", "sum_err", "sum_sq_err")

    def __init__(self):
        self.n = 0
        self.sum_err = 0.0
        self.sum_sq_err = 0.0
        self.sum_abs_err = 0.0
        self._mean_true = 0.0  # running mean of truth (Welford)
        self._m2_true = 0.0  # running sum of squared deviations (Welford)

    def update(self, pred: np.ndarray, true: np.ndarray, mask: np.ndarray) -> None:
        p, t, _n = _masked(pred, true, mask)
        self.update_masked(p, t)

    def update_masked(self, p: np.ndarray, t: np.ndarray) -> None:
        """Accumulate already-masked, finite, flat prediction / truth vectors."""
        n = int(p.size)
        if n == 0:
            return
        err = p - t
        self.sum_err += float(err.sum())
        self.sum_sq_err += float(np.sum(err**2))
        self.sum_abs_err += float(np.sum(np.abs(err)))

        # Chan et al. parallel variant of Welford: merge this chunk's (n, mean, M2) into
        # the running estimate, which is stable regardless of the offset.
        t = t.astype(np.float64, copy=False)
        chunk_mean = float(t.mean())
        chunk_m2 = float(np.sum((t - chunk_mean) ** 2))
        if self.n == 0:
            self._mean_true, self._m2_true = chunk_mean, chunk_m2
        else:
            delta = chunk_mean - self._mean_true
            total = self.n + n
            self._m2_true += chunk_m2 + delta * delta * self.n * n / total
            self._mean_true += delta * n / total
        self.n += n

    def result(self) -> RegressionScores:
        if self.n == 0:
            nan = float("nan")
            return RegressionScores(0, nan, nan, nan, nan)
        ss_tot = self._m2_true
        return RegressionScores(
            n=self.n,
            rmse=float(np.sqrt(self.sum_sq_err / self.n)),
            mae=float(self.sum_abs_err / self.n),
            bias=float(self.sum_err / self.n),
            r2=float(1.0 - self.sum_sq_err / ss_tot) if ss_tot > 0 else float("nan"),
        )


class StreamingEvents:
    """Running contingency table for one advisory threshold."""

    __slots__ = ("_thr", "correct_negatives", "false_alarms", "hits", "misses", "threshold")

    def __init__(self, threshold_mg_m3: float):
        self.threshold = threshold_mg_m3
        self._thr = float(np.log10(threshold_mg_m3))
        self.hits = 0
        self.misses = 0
        self.false_alarms = 0
        self.correct_negatives = 0

    def update(self, pred_log: np.ndarray, true_log: np.ndarray, mask: np.ndarray) -> None:
        p, t, _n = _masked(pred_log, true_log, mask)
        self.update_masked(p, t)

    def update_masked(self, p: np.ndarray, t: np.ndarray) -> None:
        """Accumulate already-masked, finite, flat prediction / truth vectors."""
        if p.size == 0:
            return
        pe, te = p >= self._thr, t >= self._thr
        self.hits += int(np.sum(pe & te))
        self.misses += int(np.sum(~pe & te))
        self.false_alarms += int(np.sum(pe & ~te))
        self.correct_negatives += int(np.sum(~pe & ~te))

    def result(self) -> EventScores:
        return EventScores(
            threshold_mg_m3=self.threshold,
            n=self.hits + self.misses + self.false_alarms + self.correct_negatives,
            hits=self.hits,
            misses=self.misses,
            false_alarms=self.false_alarms,
            correct_negatives=self.correct_negatives,
        )


# Score histogram for the threshold-free event metrics. log10 chlorophyll spans roughly
# -1 to 2.5 on every lake here; 0.01 log10 per bin (a 2.3% change in concentration) is far
# below the retrieval's own uncertainty, so binning the scores costs nothing measurable
# and keeps the accumulator at two small integer arrays instead of every pixel ever seen.
RANK_BINS = 512
RANK_RANGE = (-1.5, 3.62)


@dataclass
class RankingScores:
    """Threshold-free event skill for one advisory level."""

    threshold_mg_m3: float
    n: int
    positives: int
    roc_auc: float
    pr_auc: float

    @property
    def prevalence(self) -> float:
        return self.positives / self.n if self.n else float("nan")

    def to_dict(self, prefix: str = "") -> dict[str, float]:
        tag = f"{prefix}thr{self.threshold_mg_m3:g}_"
        return {
            f"{tag}roc_auc": self.roc_auc,
            f"{tag}pr_auc": self.pr_auc,
            f"{tag}prevalence": self.prevalence,
        }


class StreamingRanking:
    """Running ROC-AUC and PR-AUC for "chlorophyll exceeds the threshold".

    POD / FAR / CSI score the forecast at one operating point -- the threshold itself --
    and a model that is systematically biased low can look useless there while ranking
    pixels almost perfectly. The catalogue this project follows asks for ROC-AUC and,
    because blooms are rare, PR-AUC: the area under precision-recall does not reward the
    correct negatives that make up most of every lake. Both are accumulated as score
    histograms of the positive and negative classes, from which every operating point is
    recoverable in O(bins), so the whole split still streams in constant memory.
    """

    __slots__ = ("_range", "_thr", "neg", "pos", "threshold")

    def __init__(self, threshold_mg_m3: float, score_range: tuple[float, float] = RANK_RANGE):
        self.threshold = threshold_mg_m3
        self._thr = float(np.log10(threshold_mg_m3))
        # The score being ranked need not be chlorophyll: the exceedance head's logits are
        # ranked by the same machinery over LOGIT_RANGE. Truth is always log10 chlorophyll.
        self._range = score_range
        self.pos = np.zeros(RANK_BINS, dtype=np.int64)
        self.neg = np.zeros(RANK_BINS, dtype=np.int64)

    def update_masked(self, p: np.ndarray, t: np.ndarray) -> None:
        """Accumulate already-masked, finite, flat score / truth vectors."""
        if p.size == 0:
            return
        lo, hi = self._range
        idx = ((p - lo) / (hi - lo) * RANK_BINS).astype(np.int64)
        np.clip(idx, 0, RANK_BINS - 1, out=idx)
        te = t >= self._thr
        self.pos += np.bincount(idx[te], minlength=RANK_BINS)
        self.neg += np.bincount(idx[~te], minlength=RANK_BINS)

    def result(self) -> RankingScores:
        n_pos, n_neg = int(self.pos.sum()), int(self.neg.sum())
        n = n_pos + n_neg
        if n_pos == 0 or n_neg == 0:
            nan = float("nan")
            return RankingScores(self.threshold, n, n_pos, nan, nan)
        # Sweep the decision threshold from the highest bin downwards: each step admits
        # one more bin as "bloom", so cumulative sums give TP and FP at every operating
        # point, and recall rises monotonically along the sweep.
        tp = np.cumsum(self.pos[::-1]).astype(np.float64)
        fp = np.cumsum(self.neg[::-1]).astype(np.float64)
        tpr = np.concatenate([[0.0], tp / n_pos])
        fpr = np.concatenate([[0.0], fp / n_neg])
        # np.trapz is the NumPy < 2 name; NumPy 2 renamed it and later removed the alias.
        trapezoid = getattr(np, "trapezoid", None) or np.trapz  # type: ignore[attr-defined]
        roc = float(trapezoid(tpr, fpr))
        # Average precision: precision at each recall step, weighted by the recall gained
        # there. Ties inside a bin are admitted together, which is the standard step
        # interpolation and the convention sklearn's average_precision_score uses.
        precision = tp / np.maximum(tp + fp, 1.0)
        recall = tp / n_pos
        prev_recall = np.concatenate([[0.0], recall[:-1]])
        ap = float(np.sum((recall - prev_recall) * precision))
        return RankingScores(self.threshold, n, n_pos, roc, ap)


# Exceedance probabilities are ranked as logits: near 0 and 1 a probability histogram
# would put most pixels in its first bin. +-14 is p in [8e-7, 1 - 8e-7], and 512 bins over
# it are 0.055 logit wide -- finer than anything a probability is quoted to.
LOGIT_RANGE = (-14.0, 14.0)
_P_EPS = 1e-6


def prob_to_logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, _P_EPS, 1.0 - _P_EPS)
    return np.log(p) - np.log1p(-p)


class StreamingReliability:
    """Calibration of a probability of exceedance: Brier score and a reliability table.

    A probability is only worth quoting on an advisory if, of all the pixels given 30%,
    about 30% turn out to exceed the threshold. The table records, per probability bin,
    how many pixels fell there, their mean forecast probability and their observed
    frequency -- the reliability diagram -- and the Brier score is referenced to the base
    rate so it reads as skill (BSS > 0 beats forecasting the prevalence everywhere).
    """

    __slots__ = ("_thr", "brier_sum", "count", "n_bins", "sum_obs", "sum_prob", "threshold")

    def __init__(self, threshold_mg_m3: float, n_bins: int = 10):
        self.threshold = threshold_mg_m3
        self._thr = float(np.log10(threshold_mg_m3))
        self.n_bins = n_bins
        self.count = np.zeros(n_bins, dtype=np.int64)
        self.sum_prob = np.zeros(n_bins, dtype=np.float64)
        self.sum_obs = np.zeros(n_bins, dtype=np.float64)
        self.brier_sum = 0.0

    def update_masked(self, prob: np.ndarray, t: np.ndarray) -> None:
        """Already-masked flat probability / log10-truth vectors."""
        if prob.size == 0:
            return
        obs = (t >= self._thr).astype(np.float64)
        prob = prob.astype(np.float64)
        idx = np.clip((prob * self.n_bins).astype(np.int64), 0, self.n_bins - 1)
        self.count += np.bincount(idx, minlength=self.n_bins)
        self.sum_prob += np.bincount(idx, weights=prob, minlength=self.n_bins)
        self.sum_obs += np.bincount(idx, weights=obs, minlength=self.n_bins)
        self.brier_sum += float(((prob - obs) ** 2).sum())

    def result(self, prefix: str = "") -> dict:
        tag = f"{prefix}thr{self.threshold:g}_"
        n = int(self.count.sum())
        if n == 0:
            nan = float("nan")
            return {f"{tag}brier": nan, f"{tag}bss": nan, f"{tag}ece": nan}
        base = float(self.sum_obs.sum() / n)
        brier = self.brier_sum / n
        ref = base * (1.0 - base)
        occupied = self.count > 0
        safe = np.maximum(self.count, 1)
        mean_p = np.where(occupied, self.sum_prob / safe, np.nan)
        freq = np.where(occupied, self.sum_obs / safe, np.nan)
        ece = float(np.nansum(self.count / n * np.abs(mean_p - freq)))
        return {
            f"{tag}brier": brier,
            f"{tag}bss": 1.0 - brier / ref if ref > 0 else float("nan"),
            f"{tag}ece": ece,
            f"{tag}reliability": [
                {"n": int(c), "mean_prob": float(mp), "obs_freq": float(f)}
                for c, mp, f in zip(self.count, mean_p, freq, strict=True)
                if c > 0
            ],
        }


class StreamingEvaluation:
    """All metrics for one (lake, lead) cell, accumulated sample by sample."""

    def __init__(
        self,
        lake_id: str,
        lead_days: int,
        thresholds: tuple[float, ...] = BLOOM_THRESHOLDS_MG_M3,
    ):
        self.lake_id = lake_id
        self.lead_days = lead_days
        self.n_samples = 0
        self.model = StreamingRegression()
        self.persistence = StreamingRegression()
        self.climatology = StreamingRegression()
        self.events = [StreamingEvents(t) for t in thresholds]
        # Threshold-free event skill, for the model and for both baselines: a ranking
        # metric is only informative relative to what persistence already achieves.
        self.rankings = [StreamingRanking(t) for t in thresholds]
        self.persistence_rankings = [StreamingRanking(t) for t in thresholds]
        self.climatology_rankings = [StreamingRanking(t) for t in thresholds]
        # Filled only when a forecast carries exceedance probabilities (a model with an
        # exceedance head, or an ensemble of them).
        self.thresholds = tuple(thresholds)
        self.head_rankings = [StreamingRanking(t, LOGIT_RANGE) for t in thresholds]
        self.reliability = [StreamingReliability(t) for t in thresholds]
        self.has_prob = False
        # Ensemble spread against ensemble-mean error: a calibrated ensemble has the two
        # equal on average (spread / RMSE ~ 1); below 1 it is overconfident.
        self.spread_sq_sum = 0.0
        self.spread_n = 0

    def update(
        self, pred, true, mask, persistence=None, climatology=None, prob=None, spread=None
    ) -> None:
        """One sample at this lead. `prob` is (K, H, W) p(exceed) for each threshold,
        `spread` the (H, W) ensemble standard deviation in log10."""
        self.n_samples += 1
        # The mask is shared by every accumulator, so it is built once. Each forecast is
        # still screened for its own non-finite pixels, exactly as the batch functions
        # do, which is what keeps the streaming numbers bit-comparable to them.
        base = np.asarray(mask).astype(bool) & np.isfinite(true)
        m = base & np.isfinite(pred)
        p, t = pred[m], true[m]
        self.model.update_masked(p, t)
        for ev in self.events:
            ev.update_masked(p, t)
        for rk in self.rankings:
            rk.update_masked(p, t)
        if prob is not None:
            self.has_prob = True
            for k in range(len(self.thresholds)):
                pk = prob[k][m]
                self.head_rankings[k].update_masked(prob_to_logit(pk), t)
                self.reliability[k].update_masked(pk, t)
        if spread is not None:
            sp = spread[m].astype(np.float64)
            self.spread_sq_sum += float((sp**2).sum())
            self.spread_n += int(sp.size)
        if persistence is not None:
            m = base & np.isfinite(persistence)
            pm, tm = persistence[m], true[m]
            self.persistence.update_masked(pm, tm)
            for rk in self.persistence_rankings:
                rk.update_masked(pm, tm)
        if climatology is not None:
            m = base & np.isfinite(climatology)
            cm, tm = climatology[m], true[m]
            self.climatology.update_masked(cm, tm)
            for rk in self.climatology_rankings:
                rk.update_masked(cm, tm)

    def result(self) -> ForecastEvaluation:
        return ForecastEvaluation(
            lake_id=self.lake_id,
            lead_days=self.lead_days,
            regression=self.model.result(),
            events=[e.result() for e in self.events],
            persistence_rmse=self.persistence.result().rmse,
            rankings=[r.result() for r in self.rankings],
        )

    def probabilistic(self) -> dict:
        """Head ranking and calibration, keyed `head_thr20_pr_auc`, `thr20_brier`, ...

        Empty for a forecast with no probabilities, so older score files keep their shape.
        """
        out: dict = {}
        if self.has_prob:
            for rk in self.head_rankings:
                out.update(rk.result().to_dict(prefix="head_"))
            for rel in self.reliability:
                out.update(rel.result())
        if self.spread_n:
            out["ensemble_spread"] = float(np.sqrt(self.spread_sq_sum / self.spread_n))
        return out

    def baseline_rankings(self) -> dict[str, float]:
        """ROC / PR AUC of both baselines, keyed like the model's with a prefix."""
        out: dict[str, float] = {}
        for rk in self.persistence_rankings:
            out.update(rk.result().to_dict(prefix="persistence_"))
        for rk in self.climatology_rankings:
            out.update(rk.result().to_dict(prefix="climatology_"))
        return out
