"""Recalibration of exceedance probabilities: a per-(lead, threshold) logistic map.

The exceedance head's raw probabilities rank well and are miscalibrated in a lake-
dependent way. Measured on the three-seed ERA5 ensemble (2026-09-23): the validation lake
(Balaton, 2% of pixels above 20 mg/m^3) scored a Brier skill of -0.5 to -0.9 against its
base rate -- confident false alarms -- while the held-out lake (Victoria, 8%) scored
+0.23 to +0.30 with an S-shaped reliability curve.

Two parameters per (lead, threshold), `p' = sigmoid(a * logit(p) + b)` (Platt scaling),
fitted on the *validation* lake only. Fitted there, it moved Victoria -- which it never
saw -- from +0.23..+0.30 to +0.29..+0.35: it transfers, which is the only reason it is
applied. Anything with more freedom (isotonic, per-bin) would be fitting one lake's base
rate, and the next lake has a different one.

The map is monotone in p, so it changes calibration and never ranking: PR-AUC and ROC-AUC
are identical before and after.

**Per trophic class, pooled toward the global map (2026-09-23).** The single global map
did not transfer across base rates: fitted where exceedances are rare it pulled every
probability down, which hurt Chapala (46% of pixels above 20 mg/m^3). Trophic class is
the registry's statement of a lake's base rate, so the calibration now has one level per
class -- `groups["trophic:<class>"]` -- each fitted on that class's validation lakes with
a quadratic penalty toward the global (a, b). The penalty is worth `prior_positives`
exceedances: a class whose validation lakes saw a few hundred exceedances stays close to
the global map; one with tens of thousands gets its own. A class with no validation lake
at all -- and any lake with no registry entry -- is served the global map. Still two
parameters per (class, lead, threshold), still monotone, still rank-preserving.

Stored as `calibration.json` beside the checkpoint (or inside an ensemble / ONNX
directory) and applied by `inference.predictor` for every backend, and by
`pipelines/evaluate.py`, so the probability evaluated is the probability served.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

CALIBRATION_FILE = "calibration.json"
_EPS = 1e-6


def calibration_path(checkpoint: str | Path) -> Path:
    """Where the calibration for `checkpoint` lives: inside a directory, beside a file."""
    path = Path(checkpoint)
    return path / CALIBRATION_FILE if path.is_dir() else path.with_suffix(".calibration.json")


def load(checkpoint: str | Path) -> dict | None:
    path = calibration_path(checkpoint)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def params_for(calib: dict, groups=None) -> dict:
    """The (lead:thr) -> [a, b] table for a lake: its trophic class's where the
    calibration has one, the global table otherwise, key by key."""
    params = dict(calib.get("params", {}))
    if groups is not None and calib.get("groups"):
        factor = calib.get("group_factor", "trophic")
        params.update(calib["groups"].get(groups.key(factor), {}))
    return params


def apply(
    prob: np.ndarray | None, calib: dict | None, lead_times, thresholds, groups=None
) -> np.ndarray | None:
    """Recalibrate a (..., L, K, H, W) probability array. Identity where no fit exists.

    `groups` is the lake's `habforecast.groups.LakeGroups`; see `params_for`.
    """
    if calib is None or prob is None:
        return prob
    out = np.array(prob, dtype=np.float32, copy=True)
    params = params_for(calib, groups)
    for li, lead in enumerate(lead_times):
        for k, thr in enumerate(thresholds):
            ab = params.get(f"{int(lead)}:{thr:g}")
            if ab is None:
                continue
            p = np.clip(out[..., li, k, :, :], _EPS, 1.0 - _EPS)
            z = ab[0] * (np.log(p) - np.log1p(-p)) + ab[1]
            out[..., li, k, :, :] = 1.0 / (1.0 + np.exp(-z))
    return out


def fit_from_reliability(
    bins: list[dict],
    min_positives: float = 200.0,
    prior: list[float] | None = None,
    prior_positives: float = 0.0,
) -> list[float] | None:
    """Fit (a, b) from a binned reliability table (`StreamingReliability` output).

    Within a bin every pixel is represented by the bin's mean probability, which with
    ten bins and millions of pixels costs little. Returns None when the table holds too
    few exceedances to fit two parameters honestly -- the 50 mg/m^3 level on a mesotrophic
    validation lake -- and the probability is then served uncalibrated.

    With a `prior` (the global fit), the objective adds
    `prior_positives * ((a - a0)^2 + (b - b0)^2)`: partial pooling toward it, with a
    weight equivalent to that many exceedances. `min_positives` then no longer applies --
    a thin class is exactly the case the prior is for -- but a class with no exceedance at
    all returns the prior unchanged.
    """
    from scipy.optimize import minimize

    n = np.array([b["n"] for b in bins], dtype=float)
    p = np.clip(np.array([b["mean_prob"] for b in bins]), _EPS, 1 - _EPS)
    f = np.array([b["obs_freq"] for b in bins])
    positives = float((n * f).sum())
    if prior is not None:
        if positives <= 0:
            return [float(prior[0]), float(prior[1])]
    elif positives < min_positives:
        return None
    x = np.log(p) - np.log1p(-p)
    a0, b0 = (float(v) for v in (prior if prior is not None else (1.0, 0.0)))
    lam = float(prior_positives) if prior is not None else 0.0

    def nll(ab):
        q = np.clip(1 / (1 + np.exp(-(ab[0] * x + ab[1]))), 1e-9, 1 - 1e-9)
        data = -float(np.sum(n * (f * np.log(q) + (1 - f) * np.log(1 - q))))
        return data + lam * ((ab[0] - a0) ** 2 + (ab[1] - b0) ** 2)

    res = minimize(nll, [a0, b0], method="Nelder-Mead")
    a, b = (float(v) for v in res.x)
    # A non-positive slope would invert the ranking; refuse rather than serve that.
    return [a, b] if a > 0 else None
