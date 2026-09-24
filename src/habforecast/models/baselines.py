"""Non-learned baselines. These are the bar the model has to clear.

Short-range forecasting of a geophysical field is a domain where naive baselines are
embarrassingly strong. Chlorophyll fields are autocorrelated over days, so "tomorrow
looks like today" captures most of the variance at a 1-day lead. Any project that
reports only RMSE, without a baseline comparison, is hiding this.

Two baselines:

- **Persistence** -- carry the most recent observed value forward. Very strong at short
  lead, degrades with horizon. This is the reference for the headline skill score.
- **Climatology** -- the per-pixel seasonal mean for that day of year. Weak at 1 day but
  it *stops degrading*, so it eventually overtakes persistence. Where those two curves
  cross is the interesting part of the problem.

**Climatology is computed leave-one-year-out.** The first version of this class fitted
the seasonal mean on the target lake's entire record, including the exact days it was
then scored against. That is leakage, and it showed: the leaky climatology beat the
model at every lead time on the validation lake. Excluding the evaluated sample's own
year makes it an honest baseline -- still strong, because a seasonal lookup genuinely is
strong on a lake with a stable annual cycle, but no longer clairvoyant.

Both expose the same forward signature as the neural models so the evaluation loop does
not need to special-case them.
"""

from __future__ import annotations

from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn

from habforecast.features.dataset import LakeSeries, Normalisation
from habforecast.models.convlstm import _last_observed


class PersistenceBaseline(nn.Module):
    """Carry the most recent observed value forward to every lead time."""

    def __init__(self, n_leads: int = 3):
        super().__init__()
        self.n_leads = n_leads

    def forward(self, x, x_mask, water, doy=None):
        last, has = _last_observed(x, x_mask)
        out = (last * has).unsqueeze(1).repeat(1, self.n_leads, 1, 1)
        return out * water


class ClimatologyBaseline(nn.Module):
    """Per-pixel seasonal mean chlorophyll, computed leave-one-year-out.

    For a sample anchored on day-of-year `d` in year `y`, the climatology is the mean of
    all observations within +/- `doy_halfwidth` days of `d` drawn from *every year except
    `y`*. Computed on demand and cached by (lake, doy, year): the per-lake record is only
    a few hundred frames, so a masked nanmean over the matching subset costs about a
    millisecond and avoids materialising a (year, doy, H, W) table that would run to
    hundreds of megabytes.
    """

    def __init__(self, n_leads: int = 3, doy_halfwidth: int = 15, cache_size: int = 64):
        super().__init__()
        self.n_leads = n_leads
        self.doy_halfwidth = doy_halfwidth
        # Bounded LRU. Each cached frame is H*W float32 -- 0.5 MB for the largest lake --
        # and the key space is (doy x year), so an unbounded dict grows to hundreds of MB
        # over a full evaluation. Evaluation walks samples in time order, so consecutive
        # lookups share keys and a small cache still hits most of the time.
        self.cache_size = cache_size
        self._series: list[LakeSeries] = []
        self._norm: Normalisation | None = None
        self._fallback: dict[int, float] = {}
        self._cache: OrderedDict[tuple[int, int, int], np.ndarray] = OrderedDict()

    def fit(self, series: list[LakeSeries], norm: Normalisation) -> ClimatologyBaseline:
        """Record the lakes and their fallback means. No averaging happens here.

        The seasonal mean itself comes from `LakeSeries.climatology`, which is also what
        feeds the `use_clim` model -- so the baseline and the model's prior are one
        computation, and a change to the window cannot move one without the other.
        """
        self._series = series
        self._norm = norm
        for li, s in enumerate(series):
            src = s.clim_source if s.clim_source is not None else s
            finite = src.values[np.isfinite(src.values)].astype(np.float32)
            self._fallback[li] = float(norm.apply(finite.mean())) if finite.size else 0.0
        return self

    def _climatology(self, lake_idx: int, doy: int, exclude_year: int) -> np.ndarray:
        """Per-pixel mean over a circular day-of-year window, excluding one year."""
        key = (lake_idx, int(doy), int(exclude_year))
        cached = self._cache.get(key)
        if cached is not None:
            # Refresh recency for the LRU eviction below.
            self._cache.move_to_end(key)
            return cached

        if self._norm is None:
            raise RuntimeError("call fit() before predicting")
        s = self._series[lake_idx]
        mean = s.climatology(doy, exclude_year, halfwidth=self.doy_halfwidth)
        if mean is None:
            # No other year covers this part of the calendar; fall back to the lake mean
            # rather than silently returning zeros, which would read as "no bloom".
            out = np.full(s.shape, self._fallback[lake_idx], dtype=np.float32)
        else:
            out = self._norm.apply(mean).astype(np.float32)
            out = np.nan_to_num(out, nan=self._fallback[lake_idx])

        self._cache[key] = out
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return out

    def forward(self, x, x_mask, water, doy=None, lake_idx=None, year=None):
        b, _, h, w = x.shape
        out = torch.zeros(b, self.n_leads, h, w, device=x.device, dtype=x.dtype)
        if doy is None or lake_idx is None:
            return out

        for i in range(b):
            li = int(lake_idx[i])
            if li >= len(self._series):
                continue
            # exclude_year = -1 when no year is supplied, which excludes nothing and
            # reproduces the old (leaky) behaviour -- only used by callers that have no
            # year information, never by the evaluation pipeline.
            exclude = int(year[i]) if year is not None else -1
            frame = self._climatology(li, int(doy[i]), exclude)[:h, :w]
            out[i] = torch.from_numpy(frame).to(x.device, x.dtype).unsqueeze(0)
        return out * water
