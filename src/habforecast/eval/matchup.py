"""Scoring satellite and forecast chlorophyll against field samples.

Everything else in `habforecast.eval` scores the model against the satellite -- the same
retrieval it was trained to reproduce. That measures forecasting skill, but not whether
the numbers mean anything in the water. This module scores against laboratory
chlorophyll-a from bottle samples (`habforecast.insitu`), which no part of training ever
saw.

Conventions, chosen to match the ocean-colour matchup literature so the numbers are
comparable with published OLCI validations:

* **Log space.** Chlorophyll spans four orders of magnitude and its errors are
  multiplicative, so differences are taken in log10. `bias` and `mae` are reported both
  as log10 values and as factors (10**x): "the satellite reads 1.6x high on average".
* **A 3x3 pixel box, median of what is valid.** One 300 m pixel against one bottle is
  dominated by patchiness and geolocation; the box median is the standard compromise.
* **Clustered bootstrap.** Several stations sampled on the same day of the same lake
  share the same satellite scene and the same weather, so they are not independent.
  Intervals resample (lake, date) clusters, not rows.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

LOG2 = math.log10(2.0)
# The advisory's own chlorophyll triggers (habforecast.advisory.LEVELS), so detection
# skill is scored at the thresholds the public outlook actually uses.
THRESHOLDS_MG_M3 = (10.0, 20.0, 50.0)


def box_median(field: np.ndarray, row: int, col: int, half: int = 1) -> float:
    """Median of the finite values in a (2*half+1)^2 box, NaN if none."""
    h, w = field.shape[-2:]
    r0, r1 = max(0, row - half), min(h, row + half + 1)
    c0, c1 = max(0, col - half), min(w, col + half + 1)
    vals = field[..., r0:r1, c0:c1]
    vals = vals[np.isfinite(vals)]
    return float(np.median(vals)) if vals.size else float("nan")


def continuous(pred_log: np.ndarray, obs_log: np.ndarray) -> dict:
    """Agreement in log10 space between paired predictions and observations."""
    pred_log = np.asarray(pred_log, dtype=float)
    obs_log = np.asarray(obs_log, dtype=float)
    ok = np.isfinite(pred_log) & np.isfinite(obs_log)
    p, o = pred_log[ok], obs_log[ok]
    n = int(ok.sum())
    if n == 0:
        return {"n": 0}
    d = p - o
    out = {
        "n": n,
        "bias_log10": float(d.mean()),
        "bias_factor": float(10 ** d.mean()),
        "mae_log10": float(np.abs(d).mean()),
        "mae_factor": float(10 ** np.abs(d).mean()),
        "rmse_log10": float(np.sqrt((d**2).mean())),
        "within_2x": float((np.abs(d) <= LOG2).mean()),
    }
    if n >= 3 and p.std() > 0 and o.std() > 0:
        out["pearson_r_log"] = float(np.corrcoef(p, o)[0, 1])
        out["spearman_rho"] = float(_spearman(p, o))
    return out


def _rank(a: np.ndarray) -> np.ndarray:
    order = a.argsort(kind="mergesort")
    ranks = np.empty(len(a), dtype=float)
    ranks[order] = np.arange(len(a), dtype=float)
    # average ties
    _, inv, counts = np.unique(a, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, weights=ranks)
    return (sums / counts)[inv]


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.corrcoef(_rank(a), _rank(b))[0, 1])


def categorical(pred_mg: np.ndarray, obs_mg: np.ndarray, threshold: float) -> dict:
    """Detection of `obs >= threshold` by `pred >= threshold`.

    POD (hit rate), FAR (share of predicted events that did not happen) and CSI (threat
    score) are undefined when their denominator is empty, and returned as None then --
    a zero would read as a measured failure.
    """
    pred_mg = np.asarray(pred_mg, dtype=float)
    obs_mg = np.asarray(obs_mg, dtype=float)
    ok = np.isfinite(pred_mg) & np.isfinite(obs_mg)
    p, o = pred_mg[ok] >= threshold, obs_mg[ok] >= threshold
    hits = int((p & o).sum())
    misses = int((~p & o).sum())
    false_alarms = int((p & ~o).sum())
    correct_neg = int((~p & ~o).sum())

    def ratio(a: int, b: int) -> float | None:
        return a / b if b else None

    return {
        "threshold_mg_m3": threshold,
        "n": int(ok.sum()),
        "observed_events": hits + misses,
        "hits": hits,
        "misses": misses,
        "false_alarms": false_alarms,
        "correct_negatives": correct_neg,
        "pod": ratio(hits, hits + misses),
        "far": ratio(false_alarms, hits + false_alarms),
        "csi": ratio(hits, hits + misses + false_alarms),
        "accuracy": ratio(hits + correct_neg, int(ok.sum())),
    }


def cluster_bootstrap_mae_gain(
    model_log: np.ndarray,
    base_log: np.ndarray,
    obs_log: np.ndarray,
    clusters: Sequence,
    n_boot: int = 2000,
    seed: int = 0,
) -> dict:
    """Skill = 1 - MAE(model)/MAE(baseline) on paired samples, with a 95% interval
    from resampling whole clusters. Positive = the model is closer to the bottles."""
    model_log = np.asarray(model_log, dtype=float)
    base_log = np.asarray(base_log, dtype=float)
    obs_log = np.asarray(obs_log, dtype=float)
    ok = np.isfinite(model_log) & np.isfinite(base_log) & np.isfinite(obs_log)
    if ok.sum() < 5:
        return {"n": int(ok.sum())}
    em = np.abs(model_log[ok] - obs_log[ok])
    eb = np.abs(base_log[ok] - obs_log[ok])
    keys = np.asarray([str(c) for c in np.asarray(clusters, dtype=object)[ok]])
    uniq, inv = np.unique(keys, return_inverse=True)
    sm = np.bincount(inv, weights=em)
    sb = np.bincount(inv, weights=eb)
    cnt = np.bincount(inv).astype(float)
    point = 1.0 - em.mean() / eb.mean() if eb.mean() > 0 else float("nan")
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(uniq), size=(n_boot, len(uniq)))
    m = sm[draws].sum(1) / cnt[draws].sum(1)
    b = sb[draws].sum(1) / cnt[draws].sum(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        gains = 1.0 - m / b
    gains = gains[np.isfinite(gains)]
    return {
        "n": int(ok.sum()),
        "clusters": len(uniq),
        "skill": float(point),
        "ci95": [float(np.percentile(gains, 2.5)), float(np.percentile(gains, 97.5))],
        "model_mae_log10": float(em.mean()),
        "baseline_mae_log10": float(eb.mean()),
    }
