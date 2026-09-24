"""Drift detection for satellite bloom forecasting.

Standard tabular drift tooling asks "has the feature distribution moved?". For this
system that question is necessary but not sufficient, because there are three distinct
things that drift and they demand different responses:

1. **Observation drift** -- the sensor's usable coverage changes. A drop in valid-pixel
   fraction or a jump in CLOUD_MARGIN frequency usually signals an upstream processing
   baseline change, not a change in the lakes. Response: investigate the data, do not
   retrain.

2. **Distribution drift** -- chlorophyll values themselves shift. Could be a real
   eutrophication trend, could be a retrieval change. Response: check against in-situ.

3. **Geographic drift** -- the model is being asked about a lake unlike anything it
   trained on. This is the one that matters most here and the one generic tooling
   misses entirely: the model will happily produce confident nonsense for a tropical
   lake when it only ever saw temperate ones. Response: refuse, or flag the output as
   out-of-distribution.

Population Stability Index is used rather than a KS test because PSI is stable under the
large sample sizes here (millions of pixels), where KS reports significance for
differences far too small to matter operationally.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# Conventional PSI bands from credit-risk practice, which is where the metric comes from.
PSI_NO_DRIFT = 0.10
PSI_MODERATE = 0.25


@dataclass
class DriftMetric:
    name: str
    value: float
    threshold: float
    drifted: bool
    note: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class DriftReport:
    reference: str
    current: str
    generated_utc: str
    metrics: list[DriftMetric] = field(default_factory=list)

    @property
    def any_drift(self) -> bool:
        return any(m.drifted for m in self.metrics)

    @property
    def drifted_metrics(self) -> list[str]:
        return [m.name for m in self.metrics if m.drifted]

    def to_dict(self) -> dict:
        return {
            "reference": self.reference,
            "current": self.current,
            "generated_utc": self.generated_utc,
            "any_drift": self.any_drift,
            "drifted_metrics": self.drifted_metrics,
            "metrics": [m.to_dict() for m in self.metrics],
        }

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path

    def render(self) -> str:
        lines = [
            f"Drift report: {self.current} vs reference {self.reference}",
            f"  generated {self.generated_utc}",
            f"  verdict: {'DRIFT DETECTED' if self.any_drift else 'no significant drift'}",
            "",
        ]
        for m in self.metrics:
            flag = "DRIFT" if m.drifted else "ok   "
            lines.append(f"  [{flag}] {m.name:28s} {m.value:9.4f} (threshold {m.threshold:.3f})")
            if m.note:
                lines.append(f"          {m.note}")
        return "\n".join(lines)


def population_stability_index(
    reference: np.ndarray, current: np.ndarray, n_bins: int = 10, eps: float = 1e-6
) -> float:
    """PSI between two samples, using reference quantiles as bin edges.

    Quantile bins (not equal-width) keep the metric meaningful for the skewed,
    log-normal-ish chlorophyll distribution.
    """
    ref = np.asarray(reference, dtype=np.float64)
    cur = np.asarray(current, dtype=np.float64)
    ref, cur = ref[np.isfinite(ref)], cur[np.isfinite(cur)]
    if ref.size < n_bins or cur.size == 0:
        return float("nan")

    edges = np.unique(np.quantile(ref, np.linspace(0, 1, n_bins + 1)))
    if edges.size < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf

    ref_frac = np.histogram(ref, bins=edges)[0] / ref.size
    cur_frac = np.histogram(cur, bins=edges)[0] / cur.size
    ref_frac = np.clip(ref_frac, eps, None)
    cur_frac = np.clip(cur_frac, eps, None)
    return float(np.sum((cur_frac - ref_frac) * np.log(cur_frac / ref_frac)))


def compare_distributions(
    reference: np.ndarray,
    current: np.ndarray,
    name: str = "log_chl",
    threshold: float = PSI_MODERATE,
) -> list[DriftMetric]:
    """PSI plus first- and second-moment shifts for a value distribution."""
    ref = np.asarray(reference, dtype=np.float64)
    cur = np.asarray(current, dtype=np.float64)
    ref, cur = ref[np.isfinite(ref)], cur[np.isfinite(cur)]

    psi = population_stability_index(ref, cur)
    band = (
        "stable"
        if psi < PSI_NO_DRIFT
        else "moderate shift"
        if psi < PSI_MODERATE
        else "major shift"
    )
    metrics = [
        DriftMetric(f"{name}_psi", psi, threshold, bool(psi >= threshold), band),
    ]
    if ref.size and cur.size:
        ref_sd = float(ref.std()) or 1.0
        mean_shift = abs(float(cur.mean()) - float(ref.mean())) / ref_sd
        sd_ratio = float(cur.std()) / ref_sd
        metrics.append(
            DriftMetric(
                f"{name}_mean_shift_sd",
                mean_shift,
                0.5,
                mean_shift >= 0.5,
                f"reference mean {ref.mean():.3f}, current {cur.mean():.3f} (log10 mg/m3)",
            )
        )
        metrics.append(
            DriftMetric(
                f"{name}_sd_ratio",
                sd_ratio,
                1.5,
                not (0.67 <= sd_ratio <= 1.5),
                "spread ratio current/reference",
            )
        )
    return metrics


def compare_observation_density(
    reference_valid_fraction: float,
    current_valid_fraction: float,
    threshold: float = 0.5,
) -> DriftMetric:
    """Relative change in usable-pixel fraction.

    A large drop rarely means the lakes changed; it usually means cloud masking or the
    upstream processor did. Worth an alert that says "check the data", not "retrain".
    """
    if reference_valid_fraction <= 0:
        return DriftMetric(
            "observation_density", float("nan"), threshold, False, "no reference coverage"
        )
    rel = abs(current_valid_fraction - reference_valid_fraction) / reference_valid_fraction
    return DriftMetric(
        "observation_density_rel_change",
        rel,
        threshold,
        rel >= threshold,
        f"reference {reference_valid_fraction:.1%} valid, current {current_valid_fraction:.1%}",
    )


def compare_flag_frequencies(
    reference: dict[str, float],
    current: dict[str, float],
    threshold: float = 0.15,
    watch: tuple[str, ...] = ("CLOUD", "CLOUD_MARGIN", "AC_FAIL", "OCNN_FAIL", "INLAND_WATER"),
) -> list[DriftMetric]:
    """Absolute change in the frequency of QC flags that indicate processing changes."""
    out = []
    for flag in watch:
        if flag not in reference or flag not in current:
            continue
        delta = abs(current[flag] - reference[flag])
        out.append(
            DriftMetric(
                f"flag_{flag}",
                delta,
                threshold,
                delta >= threshold,
                f"reference {reference[flag]:.1%}, current {current[flag]:.1%}",
            )
        )
    return out


def geographic_novelty(
    reference_stats: dict[str, float],
    current_stats: dict[str, float],
    threshold: float = 3.0,
) -> DriftMetric:
    """How far a new lake sits from the training lakes, in reference standard deviations.

    Deliberately simple and interpretable: a Mahalanobis-style distance over a handful of
    lake-level descriptors (median chlorophyll, spread, seasonal amplitude, observation
    density). A learned embedding distance would be more sensitive but far harder for an
    operator to act on, and the point of this metric is to justify refusing a prediction.

    KNOWN LIMITATION -- the scale is only as good as the reference sample. Measured on
    this project's three training lakes, tropical Lake Victoria scores 0.89 and temperate
    Lake Balaton scores 0.86: the metric cannot tell them apart, because the spread
    *across* three quite different training lakes is wide enough to absorb the difference.
    The seasonal-amplitude descriptor does move in the right direction (0.080 for
    Victoria against a reference of 0.125 +- 0.063), just not by 3 sigma.

    In other words, with n=3 reference lakes this is a smoke alarm, not a smoke detector.
    Treat a high score as informative and a low score as uninformative. Making it
    genuinely discriminative needs either many more reference lakes, or a per-descriptor
    threshold rather than a pooled RMS z-distance -- see `seasonal_amplitude` alone,
    which separates the two lakes far better than the aggregate does.
    """
    keys = sorted(set(reference_stats) & set(current_stats))
    if not keys:
        return DriftMetric("geographic_novelty", float("nan"), threshold, False, "no shared stats")

    dists = []
    for k in keys:
        sd = reference_stats.get(f"{k}_sd", 0.0) or 1.0
        dists.append(abs(current_stats[k] - reference_stats[k]) / sd)
    dist = float(np.sqrt(np.mean(np.square(dists))))
    return DriftMetric(
        "geographic_novelty",
        dist,
        threshold,
        dist >= threshold,
        f"RMS z-distance over {len(keys)} lake descriptors: {', '.join(keys)}",
    )


def build_report(reference_name: str, current_name: str, metrics: list[DriftMetric]) -> DriftReport:
    return DriftReport(
        reference=reference_name,
        current=current_name,
        generated_utc=datetime.now(UTC).isoformat(timespec="seconds"),
        metrics=metrics,
    )


# Minimum water pixels a day needs before its statistics count, in either sample.
MIN_DAY_PIXELS = 50
# Tail probability for the daily input check: a window statistic below the 1st or above
# the 99th percentile of the lake's same-season archive fires.
WINDOW_TAIL = 0.01


def daily_stats(block: np.ndarray, min_pixels: int = MIN_DAY_PIXELS) -> dict[str, np.ndarray]:
    """Per-day statistics of a (days, water_pixels) log10 block, NaN where unobserved.

    Days with fewer than `min_pixels` valid pixels are dropped: their median is noise.
    """
    finite = np.isfinite(block)
    keep = finite.sum(axis=1) >= min_pixels
    block, finite = block[keep], finite[keep]
    if not len(block):
        return {k: np.array([]) for k in ("median", "p90", "spatial_sd", "valid_fraction")}
    with np.errstate(all="ignore"):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return {
                "median": np.nanmedian(block, axis=1),
                "p90": np.nanpercentile(block, 90, axis=1),
                "spatial_sd": np.nanstd(block, axis=1),
                "valid_fraction": finite.mean(axis=1),
            }


def window_drift(
    lake_id: str,
    reference: dict[str, np.ndarray],
    window: dict[str, np.ndarray],
    source: str = "cube",
    tail: float = WINDOW_TAIL,
) -> DriftReport:
    """Are one forecast's inputs unlike anything this lake has shown at this time of year?

    The daily advisory job's check (pipelines/advisory.py). Both arguments are
    `daily_stats` outputs: `reference` from the lake's own archive within a few weeks of
    the same day of year in earlier seasons, `window` from the days observed in the input
    window. Each statistic's window median is placed within the reference distribution of
    the same statistic and fires below the `tail` or above the `1 - tail` quantile.

    Why not PSI, which `monitor.py` uses: a 14-day window of one lake is a few highly
    autocorrelated frames of the same field, so its pixel distribution is always narrower
    than eight seasons of archive, and PSI against it measured 0.3-4.5 on every lake on
    every date tried -- an alarm that always rings. The question a daily alert should ask
    is narrower: is today's level, spread or coverage *outside this lake's experience*?
    The usual causes of a yes are all worth a person looking before the advisory goes
    out -- an upstream processing change, a different source than the archive (the live
    NRT path), or conditions the forecast is extrapolating into.

    The reference being the same lake, not the pooled training lakes, is deliberate: a
    held-out lake differs from those by design, and `monitor.py` answers that once, before
    the lake is trusted.
    """
    metrics = []
    for key, ref in reference.items():
        cur = window.get(key, np.array([]))
        ref = ref[np.isfinite(ref)]
        cur = cur[np.isfinite(cur)]
        if ref.size < 20 or cur.size == 0:
            continue
        value = float(np.median(cur))
        pct = float((ref < value).mean() + 0.5 * (ref == value).mean())
        lo, hi = np.quantile(ref, [tail, 1.0 - tail])
        metrics.append(
            DriftMetric(
                f"window_{key}_percentile",
                pct,
                tail,
                bool(value < lo or value > hi),
                f"window {value:.3f} vs same-season archive [{lo:.3f}, {hi:.3f}] "
                f"({tail:.0%}-{1 - tail:.0%}, {ref.size} days)",
            )
        )
    return build_report(
        f"{lake_id} archive, same season", f"{lake_id} input window ({source})", metrics
    )
