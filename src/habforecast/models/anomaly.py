"""Level-invariant anomaly forecaster -- the architecture built to fix the generalisation gap.

Why a third architecture exists
-------------------------------
The ConvLSTM and the U-Net both score well on the validation lake and collapse on the
held-out one (ConvLSTM +0.39 val, -1.23 test at 5 days). Measuring the five cubes shows
why, and it is not a capacity problem:

    lake          mean log10 chl     relative level
    erie_west          0.562              x0.37
    taihu              1.116              x1.33
    okeechobee         1.296              x2.02      <- training lakes span 5.5x
    balaton            0.753   (val)      x0.58
    victoria_ne        0.694   (test)     x0.50

Chlorophyll is standardised by ONE global mean and standard deviation fitted on the three
training lakes. Their pooled mean is 0.99, but Erie sits at 0.56 and Okeechobee at 1.30 --
a 5.5x spread in mg/m^3 between lakes the model is asked to treat with one set of
constants. Lake Victoria then arrives 0.8 standard deviations below that pooled mean, in a
regime the network barely saw, and the correction it has learned to add is calibrated for
somewhere else. That is visible in the error maps as a wall of over-prediction: +0.291
log10 mean bias at 7 days on Victoria, roughly a doubling of concentration everywhere.

So the fix is not more capacity. It is three structural changes:

1. **Reversible instance normalisation.** Each sample is standardised by the statistics of
   its OWN observed window, and the prediction is mapped back through the same constants.
   The network never sees a lake's absolute trophic level, so a lake being twice as green
   as anything in training is no longer out of distribution -- only its *dynamics* are.
   This is the change that targets the failure above directly.

2. **A bounded delta on top of persistence.** The previous models added an unbounded
   residual to the last observation. Out of domain that residual is free to be arbitrarily
   wrong, which is exactly how a model ends up *worse* than doing nothing. Here the
   departure is `scale * tanh(...)` with `scale` learned per lead and initialised small, so
   the model starts life as persistence and has a hard ceiling on how far it can talk
   itself away from it.

3. **A smooth, multi-scale delta.** Measured against the observed composites, both previous
   models emit fields 1.8-2.2x too rough at the pixel scale: they are painting speckle that
   the sensor never resolves and that no forecast can know. The correction here is
   predicted at 1/8 and 1/4 resolution and bilinearly upsampled, so pixel-scale noise is
   not representable in the first place rather than being penalised after the fact.

A fifth change, added after the first three were measured (`use_clim`): the seasonal
climatology as an input, and as a second anchor. With the changes above the model beat
persistence on the held-out lake at every lead (+0.04 to +0.06) and still lost to a
per-pixel seasonal climatology at every lead (-0.01 at 1 day to -0.15 at 7). Climatology's
error is flat with lead time and persistence's grows, so the two cross, and a model that
can only see a fortnight has no way to know where the season is heading. Here the
leave-one-year-out seasonal mean for each target date is fed as a channel, and the anchor
becomes a learned, spatially varying blend `(1 - b) * persistence + b * climatology` with
`b` starting near zero -- so the network can regress toward the seasonal expectation where
the last observation is old or the season is turning, and stay on persistence where the
field is fresh. A lake with no history (a first-time ad-hoc area) simply has the channel
masked out, which training with climatology dropout teaches the model to expect.

A fourth, quieter change: the 14-step sequence is collapsed into ~10 mask-aware summary
channels (recency-weighted means at several decay rates, last value, age, coverage, trend)
before any convolution runs. With three training lakes, a recurrent encoder over 14 steps
has far more capacity than the data can constrain -- the +0.39/-1.23 split is what
overfitting looks like. Fewer parameters and stronger priors is the correct direction here,
not more layers.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from habforecast.models.convlstm import _last_observed_index

# Guards the per-sample standard deviation. A window can be nearly uniform -- a small lake
# under uniform bloom -- and dividing by its own tiny spread would amplify retrieval noise
# into the dominant signal.
MIN_SIGMA = 0.15
# Below this many observed water pixels the window statistics are not worth trusting, and
# the sample falls back to the global standardisation it arrived in.
MIN_OBS_PIXELS = 64


def masked_moments(
    x: torch.Tensor, mask: torch.Tensor, water: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-sample mean and std over observed water pixels of the whole window.

    Returns (mean, std, valid) shaped (B,1,1,1) so they broadcast over (B,T,H,W).
    `valid` is 0 for samples with too few observations to estimate anything.
    """
    m = mask * water  # (B, T, H, W); water broadcasts over T
    n = m.sum(dim=(1, 2, 3), keepdim=True)
    enough = (n >= MIN_OBS_PIXELS).to(x.dtype)
    denom = n.clamp(min=1.0)
    mean = (x * m).sum(dim=(1, 2, 3), keepdim=True) / denom
    var = ((x - mean) ** 2 * m).sum(dim=(1, 2, 3), keepdim=True) / denom
    std = var.clamp(min=0.0).sqrt().clamp(min=MIN_SIGMA)
    # Neutral constants where the estimate is untrustworthy: mean 0, std 1 leaves the
    # incoming global standardisation untouched for that sample.
    mean = mean * enough
    std = std * enough + (1.0 - enough)
    return mean, std, enough


def last_observed_with_age(
    x: torch.Tensor, mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Most recent observed value per pixel, how old it is, and whether one exists.

    Age is in steps back from the anchor, normalised to [0, 1]. It matters: a persistence
    forecast anchored on a 12-day-old observation deserves less confidence than one
    anchored on yesterday, and the network can only know that if it is told.
    """
    t = x.shape[1]
    idx, has = _last_observed_index(mask)
    last = x.gather(1, idx.unsqueeze(1)).squeeze(1) * has
    age = (float(t - 1) - idx.to(x.dtype)) * has / max(t - 1, 1)
    return last, age, has


class WindowSummary(nn.Module):
    """Collapse a masked (B,T,H,W) window into a small stack of per-pixel channels.

    The decay rates are learned but few: three timescales spanning "yesterday" to "the
    whole fortnight" is enough to express both a current state and a trend, and every
    additional one is capacity that three lakes cannot pay for.
    """

    def __init__(self, n_scales: int = 3):
        super().__init__()
        # Initialised to roughly 1.5, 4 and 12 days. softplus keeps them positive.
        init = torch.tensor([1.5, 4.0, 12.0])[:n_scales]
        self.raw_tau = nn.Parameter(torch.log(torch.expm1(init)))
        self.n_scales = n_scales

    @property
    def out_channels(self) -> int:
        # per-scale mean + per-scale coverage, then last, age, has_last, coverage, trend
        return 2 * self.n_scales + 5

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        _b, t, _h, _w = x.shape
        tau = F.softplus(self.raw_tau).clamp(min=0.3)  # (S,)
        # Steps back from the anchor: 0 is the anchor day itself.
        back = torch.arange(t - 1, -1, -1, device=x.device, dtype=x.dtype)
        # (S, T) recency weights
        wts = torch.exp(-back.unsqueeze(0) / tau.unsqueeze(1))

        # Every scale at once: contracting the time axis against the (S, T) weight table
        # is one batched matmul, and it never materialises a (B, S, T, H, W) product.
        #
        # Kept out of autocast on purpose. Under AMP the matmul runs in float16, and the
        # backward of `num / wsum` for a pixel seen only at the far end of the window
        # (wsum near the 1e-4 floor) is num / wsum^2 -- comfortably past float16's 65504.
        # The gradient scaler then skips the step and halves its scale, every few steps,
        # until the scale underflows and the next update writes NaN into the parameters.
        # Measured: training died at epoch 9 with the loss finite at every step. The
        # contraction is a few megabytes of arithmetic; float32 costs nothing here.
        b, h, w = x.shape[0], x.shape[2], x.shape[3]
        with torch.autocast(device_type=x.device.type, enabled=False):
            wts32 = wts.float()
            flat_x = (x * mask).reshape(b, t, h * w).float()
            flat_m = mask.reshape(b, t, h * w).float()
            wsum = torch.matmul(wts32, flat_m).view(b, self.n_scales, h, w)  # (B, S, H, W)
            means = torch.matmul(wts32, flat_x).view(b, self.n_scales, h, w) / wsum.clamp(min=1e-4)
            # Coverage under each weighting: how much of the timescale was actually seen.
            covs = wsum / wts32.sum(dim=1).clamp(min=1e-4).view(1, -1, 1, 1)
        # Interleaved (mean_0, cov_0, mean_1, cov_1, ...) -- the channel order the
        # trained checkpoints expect.
        per_scale = torch.stack([means, covs], dim=2).reshape(b, 2 * self.n_scales, h, w)

        last, age, has_last = last_observed_with_age(x, mask)
        coverage = mask.mean(dim=1)
        # Short timescale minus long timescale: a signed rate of change, which is the one
        # thing persistence cannot express and the only reason to run a model at all.
        trend = means[:, 0] - means[:, -1]
        rest = torch.stack([last, age, has_last, coverage, trend], dim=1)
        return torch.cat([per_scale, rest], dim=1)


class MetEncoder(nn.Module):
    """Daily meteorological forcing over the window, reduced to a few global channels.

    The weather arrives as lake-mean daily scalars (see data.era5), not as a field: at
    0.25 degrees the reanalysis has no structure inside a lake to give, and pretending
    otherwise by interpolating it to 300 m would invent gradients the model would happily
    learn. So each variable is collapsed to four numbers over the window -- its mean, its
    mean over the last few days, the difference between those two, and how much of the
    window was covered -- and a small MLP turns the stack into `out_channels` values that
    are broadcast across the grid, exactly as the day-of-year encoding already is.

    Four summaries rather than the raw 14-day series on purpose. Three training lakes and
    ~1,500 samples cannot constrain an encoder that reads every day separately, and the
    mechanism does not need one: what matters is whether the last few days were calm and
    warm relative to the fortnight, which is what these four numbers say.
    """

    RECENT_DAYS = 3

    def __init__(self, n_vars: int, out_channels: int = 8, n_future: int = 0):
        super().__init__()
        self.n_vars = n_vars
        self.n_future = n_future
        self.out_channels = out_channels
        in_dim = n_vars * 4 + n_future
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 2 * out_channels),
            nn.SiLU(inplace=True),
            nn.Linear(2 * out_channels, out_channels),
        )
        # Start at exactly zero: the forcing begins as no contribution at all, so an
        # untrained model is the one it extends and the channels have to earn their effect.
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def summarise(self, met: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """(B, V, T) -> (B, V*4): window mean, recent mean, trend, coverage."""
        denom = mask.sum(dim=2).clamp(min=1e-4)
        window = (met * mask).sum(dim=2) / denom
        recent_slice = slice(-self.RECENT_DAYS, None)
        r_mask = mask[:, :, recent_slice]
        r_denom = r_mask.sum(dim=2).clamp(min=1e-4)
        recent = (met[:, :, recent_slice] * r_mask).sum(dim=2) / r_denom
        coverage = mask.mean(dim=2)
        # Where nothing was observed both means are zero and the trend is zero, which
        # under a zero coverage channel reads as "no information" rather than as calm.
        return torch.cat([window, recent, recent - window, coverage], dim=1)

    def forward(
        self,
        met: torch.Tensor,
        met_mask: torch.Tensor,
        future: torch.Tensor | None = None,
        future_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        feats = self.summarise(met, met_mask)
        if self.n_future:
            if future is None or future_mask is None:
                feats = torch.cat([feats, feats.new_zeros((feats.shape[0], self.n_future))], dim=1)
            else:
                # Mean over the forecast horizon, masked. Deliberately not per lead: this
                # path exists to measure whether knowing the coming week's weather helps
                # at all, and a per-lead answer costs a per-lead trunk.
                denom = future_mask.sum(dim=2).clamp(min=1e-4)
                feats = torch.cat([feats, (future * future_mask).sum(dim=2) / denom], dim=1)
        return self.mlp(feats)


class ExceedanceHead(nn.Module):
    """Per-pixel probability that chlorophyll exceeds each advisory threshold, per lead.

    Why a separate head rather than thresholding the regression
    -----------------------------------------------------------
    The regression is trained to be close in log10 on average. Its ranking of the
    bloomiest pixels -- PR-AUC for "exceeds 20 mg/m^3" -- came out level with persistence
    (0.616 vs 0.611 on the held-out lake at 1 day): a symmetric Huber loss spends its
    effort on the 92% of pixels that are nowhere near the threshold. An operator acts on
    the crossing, so this head is trained on the crossing, with focal loss, and the
    advisory levels are read from it. The regression stays as the map.

    How it is built
    ---------------
    The logit is a learned per-(lead, threshold) scale times the regression's margin over
    the threshold, plus a pixel-wise correction:

        logit = s_lk * (pred - thr_k) + MLP([margin, margin / window_sigma,
                                              window summary, trunk context at 1/4])

    The correction's output layer starts at zero, so an untrained head ranks pixels
    exactly as the regression does and has to earn every departure from it -- the same
    "start as the thing it extends" rule as the delta and the climatology gate. The margin
    is *detached*: the head reads the regression but cannot pull the map toward whatever
    makes a threshold easier to classify. The trunk context is not detached, which is the
    "same trunk" -- the shared features are shaped by both objectives.

    The window summary is at full resolution, unlike the delta: whether a pixel is over
    the line depends on that pixel's last observation and its age, which is pixel-scale
    information the smooth delta deliberately cannot represent.
    """

    def __init__(
        self,
        n_leads: int,
        thresholds_std: Sequence[float],
        summary_channels: int,
        trunk_channels: int,
        hidden: int = 16,
        init_scale: float = 3.0,
    ):
        super().__init__()
        self.n_leads = n_leads
        self.n_thr = len(thresholds_std)
        lk = n_leads * self.n_thr
        self.register_buffer("thr", torch.tensor(list(thresholds_std), dtype=torch.float32))
        self.ctx_pixel = nn.Conv2d(summary_channels + 1, hidden, 1)
        self.ctx_trunk = nn.Conv2d(trunk_channels, hidden, 1)
        self.embed = nn.Parameter(torch.zeros(lk, hidden))
        self.w_margin = nn.Parameter(torch.randn(hidden) * 0.1)
        self.w_margin_sigma = nn.Parameter(torch.randn(hidden) * 0.1)
        self.out_weight = nn.Parameter(torch.zeros(lk, hidden))
        self.out_bias = nn.Parameter(torch.zeros(lk))
        self.log_scale = nn.Parameter(torch.full((lk,), math.log(init_scale)))

    def forward(
        self,
        pred: torch.Tensor,  # (B, L, H, W) regression, globally standardised
        summary: torch.Tensor,  # (B, S, H, W) window summary
        water: torch.Tensor,  # (B, 1, H, W)
        trunk: torch.Tensor,  # (B, C, h, w) trunk features at 1/4
        sigma: torch.Tensor,  # (B, 1, 1, 1) window std, global units
    ) -> torch.Tensor:
        b, n_l, h, w = pred.shape
        lk = n_l * self.n_thr
        thr = self.thr.to(pred.dtype).view(1, 1, -1, 1, 1)
        margin = (pred.detach().unsqueeze(2) - thr).reshape(b, lk, h, w)
        margin_sigma = margin / sigma

        ctx = self.ctx_pixel(torch.cat([summary, water], dim=1))
        ctx = ctx + F.interpolate(
            self.ctx_trunk(trunk), size=(h, w), mode="bilinear", align_corners=False
        )
        hidden = (
            ctx.unsqueeze(1)
            + self.embed.to(ctx.dtype).view(1, lk, -1, 1, 1)
            + margin.unsqueeze(2) * self.w_margin.to(ctx.dtype).view(1, 1, -1, 1, 1)
            + margin_sigma.unsqueeze(2) * self.w_margin_sigma.to(ctx.dtype).view(1, 1, -1, 1, 1)
        )
        hidden = F.silu(hidden)
        corr = (hidden * self.out_weight.to(ctx.dtype).view(1, lk, -1, 1, 1)).sum(dim=2)
        corr = corr + self.out_bias.to(ctx.dtype).view(1, lk, 1, 1)
        logits = torch.exp(self.log_scale).to(ctx.dtype).view(1, lk, 1, 1) * margin + corr
        return logits.view(b, n_l, self.n_thr, h, w)


class FiLM(nn.Module):
    """Per-lead feature modulation, so all lead times share one trunk.

    The previous models emitted one output channel per lead from a shared head, which lets
    a 7-day forecast and a 1-day forecast diverge only in the final layer. Modulating the
    trunk instead means the lead time conditions the whole decode, while the parameters
    stay shared -- the right trade when there are four leads and very little data.
    """

    def __init__(self, n_leads: int, channels: int):
        super().__init__()
        self.embed = nn.Embedding(n_leads, 2 * channels)
        nn.init.zeros_(self.embed.weight)
        self.channels = channels

    def forward(self, feat: torch.Tensor, lead_idx: torch.Tensor) -> torch.Tensor:
        gb = self.embed(lead_idx)  # (B, 2C)
        gamma, beta = gb.chunk(2, dim=1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        return feat * (1.0 + gamma) + beta

    def head_all_leads(self, feat: torch.Tensor, head: nn.Conv2d) -> torch.Tensor:
        """`head(self(feat, lead))` for every lead at once, as one (B, L, h, w) map.

        A 1x1 single-channel head applied to a FiLM-modulated map is itself linear in the
        features: W . (f (1 + gamma) + beta) + b = (W (1 + gamma)) . f + (W . beta + b).
        Folding the modulation into the head's weights therefore turns L forward passes
        through the trunk's output into one 1x1 convolution with L output channels. The
        arithmetic is identical; only the number of kernel launches changes.
        """
        gamma, beta = self.embed.weight.chunk(2, dim=1)  # (L, C) each
        w = head.weight.reshape(1, -1)  # (1, C)
        weight = (w * (1.0 + gamma)).unsqueeze(-1).unsqueeze(-1)  # (L, C, 1, 1)
        bias = (beta * w).sum(dim=1) + head.bias  # (L,)
        return F.conv2d(feat, weight.to(feat.dtype), bias.to(feat.dtype))


class GroupFiLM(nn.Module):
    """Lake-group conditioning: the group levels of the multilevel model.

    Each factor in `habforecast.groups` (trophic class, thermal regime, continent) has an
    embedding table; a lake's embeddings are summed -- additive group effects, as in a
    multilevel regression with crossed random intercepts -- and projected to a per-channel
    scale and shift of a trunk feature map, one projection per modulated stage.

    Two properties make this partial pooling rather than a set of separate models:

    * **Index 0 is "unknown" and is pinned to zero** (`padding_idx`): it receives no
      gradient, so an unknown factor contributes exactly nothing. With every factor
      unknown the network *is* the shared population model -- which is what an ad-hoc
      area with no registry entry is served, and what `group_dropout` in training keeps
      the network good at.
    * **The projections start at zero**, so an untrained model is the architecture it
      extends and each group has to earn its departure from the population, the same rule
      the forcing encoder and the delta heads follow.
    """

    def __init__(self, vocab_sizes: Sequence[int], dim: int, channels: Sequence[int]):
        super().__init__()
        self.n_factors = len(vocab_sizes)
        self.embeds = nn.ModuleList(nn.Embedding(n, dim, padding_idx=0) for n in vocab_sizes)
        for emb in self.embeds:
            nn.init.normal_(emb.weight, std=0.1)
            with torch.no_grad():
                emb.weight[0].zero_()
        # No bias: with one, a trained model would add a learned offset even when every
        # factor is unknown, and "unknown" would stop being the population model.
        self.proj = nn.ModuleList(nn.Linear(dim, 2 * c, bias=False) for c in channels)
        for lin in self.proj:
            nn.init.zeros_(lin.weight)

    def embedding(self, groups: torch.Tensor) -> torch.Tensor:
        """(B, F) long -> (B, dim), the summed group effect."""
        return sum(emb(groups[:, i]) for i, emb in enumerate(self.embeds))

    def modulate(self, feat: torch.Tensor, e: torch.Tensor, stage: int) -> torch.Tensor:
        gamma, beta = self.proj[stage](e).to(feat.dtype).chunk(2, dim=1)
        return feat * (1.0 + gamma[..., None, None]) + beta[..., None, None]


def _conv_block(cin: int, cout: int, groups: int = 8) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1),
        nn.GroupNorm(min(groups, cout), cout),
        nn.SiLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1),
        nn.GroupNorm(min(groups, cout), cout),
        nn.SiLU(inplace=True),
    )


def _pool_if_possible(t: torch.Tensor) -> torch.Tensor:
    """Halve a feature map, unless it is already too small to halve.

    Returned unchanged at 1x1: the alternative is a 0x0 tensor, which is not a degraded
    answer but a crash.
    """
    return F.avg_pool2d(t, 2) if min(t.shape[-2:]) >= 2 else t


class AnomalyForecaster(nn.Module):
    """Level-invariant, persistence-anchored, smooth-delta forecaster.

    Args:
        input_days: length of the past window (used only for shape bookkeeping).
        n_leads: number of forecast horizons emitted.
        base: trunk width. 24-32 is the useful range here; more overfits.
        use_doy: append day-of-year sin/cos channels.
        delta_scale: initial cap on the departure from persistence, in units of the
            window's own standard deviation. 0.5 lets the model move a pixel by about a
            factor of two at most before training widens it.
        n_scales: number of recency timescales in the window summary.
        n_aux: number of auxiliary physical layers (see data.aux_layers). Each adds a
            recency-weighted mean and a coverage channel, summarised by the same
            machinery as chlorophyll so the two are directly comparable to the trunk.
            Zero -- the default -- reproduces the original architecture exactly, which
            is what keeps existing checkpoints loadable.
        use_clim: feed the per-lead seasonal climatology (and its availability mask) as
            input channels, and blend the anchor between persistence and climatology
            with a learned, spatially varying weight. Off by default for the same
            reason as `n_aux`: the served checkpoints predate it.
        clim_gate_init: initial logit of the climatology weight. -2 starts the anchor at
            88% persistence / 12% climatology where a climatology exists, and at exact
            persistence where it does not.
        n_met: number of ERA5 forcing variables (see data.era5). Zero -- the default --
            reproduces the architecture exactly, which keeps existing checkpoints
            loadable.
        met_channels: width of the broadcast forcing embedding.
        met_future: include the forcing at the target dates. **Perfect prognosis**: this
            is future weather and measures a ceiling, not a deployable system.
        exceed_thresholds: advisory levels in mg/m^3 for the exceedance head (see
            `ExceedanceHead`). Empty -- the default -- builds no head, which keeps every
            existing checkpoint loadable.
        group_dim: width of the lake-group embeddings (see `GroupFiLM` and
            `habforecast.groups`). Zero -- the default -- builds no group conditioning,
            which keeps every existing checkpoint loadable.
        norm_mean, norm_std: the global log10 standardisation the model's inputs arrive
            in. Only the exceedance head needs them, to place an absolute threshold in the
            model's own units; they are recorded in the checkpoint's model_config.
    """

    def __init__(
        self,
        input_days: int = 14,
        n_leads: int = 4,
        base: int = 24,
        use_doy: bool = True,
        delta_scale: float = 0.5,
        n_scales: int = 3,
        dropout: float = 0.1,
        n_aux: int = 0,
        use_clim: bool = False,
        clim_gate_init: float = -2.0,
        n_met: int = 0,
        met_channels: int = 8,
        met_future: bool = False,
        exceed_thresholds: Sequence[float] = (),
        exceed_hidden: int = 16,
        norm_mean: float = 0.0,
        norm_std: float = 1.0,
        group_dim: int = 0,
    ):
        super().__init__()
        self.input_days = input_days
        self.n_leads = n_leads
        self.use_doy = use_doy
        self.n_aux = n_aux
        self.use_clim = use_clim
        self.n_met = n_met
        # Perfect prognosis. Recorded on the module so a checkpoint carries the fact that
        # its numbers used future weather -- see data/era5.py.
        self.met_future = bool(met_future) and bool(n_met)

        self.summary = WindowSummary(n_scales=n_scales)
        # Auxiliary layers get one timescale, not `n_scales`. They are slower-moving and
        # far sparser than chlorophyll -- reflectance-derived layers only exist on clear
        # days -- so several recency weightings of the same thin signal would spend
        # parameters on noise. One masked mean plus its coverage is the honest summary.
        self.aux_summary = WindowSummary(n_scales=1) if n_aux else None
        aux_ch = n_aux * (self.aux_summary.out_channels if self.aux_summary else 0)
        # Climatology arrives per lead: one value channel and one mask channel each.
        clim_ch = 2 * n_leads if use_clim else 0
        self.met_encoder = (
            MetEncoder(n_met, met_channels, n_future=n_met if met_future else 0) if n_met else None
        )
        met_ch = met_channels if n_met else 0
        in_ch = self.summary.out_channels + 1 + (2 if use_doy else 0) + aux_ch + clim_ch + met_ch

        self.stem = _conv_block(in_ch, base)
        self.down1 = _conv_block(base, base * 2)  # 1/2
        self.down2 = _conv_block(base * 2, base * 4)  # 1/4  <- mid head
        self.down3 = _conv_block(base * 4, base * 4)  # 1/8  <- coarse head
        self.drop = nn.Dropout2d(dropout)

        self.film_coarse = FiLM(n_leads, base * 4)
        self.film_mid = FiLM(n_leads, base * 4)

        # Two delta heads at different resolutions, both upsampled to full size. Neither
        # can represent pixel-scale structure, which is the point.
        self.head_coarse = nn.Conv2d(base * 4, 1, 1)
        self.head_mid = nn.Conv2d(base * 4, 1, 1)
        for head in (self.head_coarse, self.head_mid):
            assert head.bias is not None
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

        # One learned cap per lead, in window-sigma units. Longer leads normally learn a
        # wider one; starting them equal lets the data decide.
        self.log_scale = nn.Parameter(
            torch.full((n_leads,), float(torch.log(torch.tensor(delta_scale))))
        )

        # The climatology gate: how far the anchor moves from the last observation toward
        # the seasonal mean, per pixel and per lead. Predicted at 1/4 resolution like the
        # delta, so it is smooth by construction, and gated by the climatology mask so a
        # lake with no history is anchored on persistence exactly as before.
        if use_clim:
            self.film_gate = FiLM(n_leads, base * 4)
            self.head_gate = nn.Conv2d(base * 4, 1, 1)
            assert self.head_gate.bias is not None
            nn.init.zeros_(self.head_gate.weight)
            nn.init.constant_(self.head_gate.bias, float(clim_gate_init))

        # Multilevel conditioning on the lake's group levels, modulating the two stages the
        # delta, gate and exceedance heads read from. Built only when asked for.
        self.group_film = None
        self.n_group_factors = 0
        if group_dim:
            from habforecast.groups import FACTORS

            self.group_film = GroupFiLM(
                [len(v) for v in FACTORS.values()], int(group_dim), [base * 4, base * 4]
            )
            self.n_group_factors = self.group_film.n_factors

        self.exceed_thresholds = tuple(float(t) for t in exceed_thresholds)
        self.exceed_head = None
        if self.exceed_thresholds:
            thr_std = [(math.log10(t) - norm_mean) / norm_std for t in self.exceed_thresholds]
            self.exceed_head = ExceedanceHead(
                n_leads,
                thr_std,
                summary_channels=self.summary.out_channels,
                trunk_channels=base * 4,
                hidden=exceed_hidden,
                # A regression error of ~0.5 global sigma, read as a logistic, is a slope
                # of ~3 per sigma: the calibrated starting point if the head learned nothing.
                init_scale=3.0,
            )

    @staticmethod
    def _doy_channels(doy: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        ang = 2.0 * torch.pi * doy / 365.25
        pair = torch.stack([torch.sin(ang), torch.cos(ang)], dim=1)
        return pair.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, *size)

    def forward(
        self,
        x: torch.Tensor,  # (B, T, H, W) globally standardised, 0 where unobserved
        x_mask: torch.Tensor,  # (B, T, H, W) 1 where observed
        water: torch.Tensor,  # (B, 1, H, W)
        doy: torch.Tensor | None = None,  # (B,)
        aux: torch.Tensor | None = None,  # (B, A, T, H, W) standardised, 0 where absent
        aux_mask: torch.Tensor | None = None,  # (B, A, T, H, W) 1 where observed
        clim: torch.Tensor | None = None,  # (B, L, H, W) standardised, 0 where absent
        clim_mask: torch.Tensor | None = None,  # (B, L, H, W) 1 where a climatology exists
        met: torch.Tensor | None = None,  # (B, V, T) standardised daily forcing
        met_mask: torch.Tensor | None = None,  # (B, V, T) 1 where the store covered the day
        met_future: torch.Tensor | None = None,  # (B, V, L) perfect prognosis only
        met_future_mask: torch.Tensor | None = None,
        groups: torch.Tensor | None = None,  # (B, F) long, habforecast.groups indices
        return_exceed: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """The regression forecast (B, L, H, W); with `return_exceed`, also the
        exceedance logits (B, L, K, H, W) for the K thresholds in `exceed_thresholds`."""
        _b, _t, h, w = x.shape

        # --- 1. reversible instance normalisation -----------------------------------
        mu, sigma, _ = masked_moments(x, x_mask, water)
        z = (x - mu) / sigma * x_mask  # unobserved stays exactly 0, not -mu/sigma

        # --- 2. window summary + spatial trunk --------------------------------------
        feats = self.summary(z, x_mask)
        parts = [feats, water]
        if self.use_doy and doy is not None:
            parts.append(self._doy_channels(doy.to(x.dtype), (h, w)))
        if self.n_aux and (aux is None or aux_mask is None or not aux.shape[1]):
            # A caller with no auxiliary layers at all -- an older cube, or the ad-hoc
            # path before those layers are fetched live. All-zero values with an all-zero
            # mask is precisely "never observed", which is a state every one of these
            # channels already sees on a cloudy fortnight, so the model degrades to the
            # chlorophyll-only case instead of refusing to run.
            zeros = x.new_zeros((x.shape[0], self.n_aux, *x.shape[1:]))
            aux, aux_mask = zeros, zeros
        if self.aux_summary is not None and aux is not None and aux_mask is not None:
            # Each layer is summarised on its own: they are different quantities, and a
            # shared summary would let a well-observed layer stand in for a missing one.
            for a in range(min(self.n_aux, aux.shape[1])):
                parts.append(self.aux_summary(aux[:, a], aux_mask[:, a]))
        if self.met_encoder is not None:
            if met is None or met_mask is None:
                # A lake with no met store, or a serving path that has not fetched one.
                # An all-zero window under an all-zero mask is the "never observed" state
                # the encoder's coverage channel already expresses.
                met = x.new_zeros((x.shape[0], self.n_met, self.input_days))
                met_mask = torch.zeros_like(met)
            embedding = self.met_encoder(met, met_mask, met_future, met_future_mask)
            parts.append(embedding.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, h, w))
        if self.use_clim:
            if clim is None or clim_mask is None:
                # No history for this water: an all-zero channel under an all-zero mask
                # is the state climatology dropout trained the model to handle.
                clim = x.new_zeros((x.shape[0], self.n_leads, h, w))
                clim_mask = torch.zeros_like(clim)
            clim_mask = clim_mask * water  # no anchor over land either way
            clim = clim * clim_mask
            # Into the window's own units, like everything else the trunk sees, so "the
            # season expects more than the lake shows now" reads the same on every lake.
            parts += [(clim - mu) / sigma * clim_mask, clim_mask]
        f0 = self.stem(torch.cat(parts, dim=1))
        f1 = self.down1(_pool_if_possible(f0))  # 1/2
        f2 = self.down2(_pool_if_possible(f1))  # 1/4
        g_emb = None
        if self.group_film is not None and groups is not None:
            # No groups at all (an ad-hoc area, an older caller) skips this entirely,
            # which is numerically the same as every factor unknown.
            g_emb = self.group_film.embedding(groups.long())
            f2 = self.group_film.modulate(f2, g_emb, 0)
        # The coarse stage stops downsampling once there is nothing left to downsample.
        # A 12 km lake on a 4 km grid is 7 pixels across, and three halvings of that is a
        # 0x0 feature map -- avg_pool2d raises rather than returning an empty tensor, so
        # without this the model does not run at all at coarse resolutions. Above ~16 px
        # every branch here takes the pooling path, so nothing changes for the grids the
        # model was trained on.
        f3 = self.drop(self.down3(_pool_if_possible(f2)))  # 1/8, or 1/4 when tiny
        if g_emb is not None:
            f3 = self.group_film.modulate(f3, g_emb, 1)

        # --- 3. bounded, smooth delta per lead --------------------------------------
        scale = torch.exp(self.log_scale).view(1, -1, 1, 1)  # (1, L, 1, 1)
        c = self.film_coarse.head_all_leads(f3, self.head_coarse)  # (B, L, H/8, W/8)
        m = self.film_mid.head_all_leads(f2, self.head_mid)  # (B, L, H/4, W/4)
        delta = F.interpolate(c, size=(h, w), mode="bilinear", align_corners=False)
        delta = delta + F.interpolate(m, size=(h, w), mode="bilinear", align_corners=False)
        delta = scale * torch.tanh(delta)  # (B, L, H, W), in window-sigma units

        # --- 4. anchor, in the ORIGINAL units ---------------------------------------
        # The anchor is taken from x directly rather than reconstructed from z. Round-
        # tripping it through (x - mu)/sigma * sigma + mu costs ~1e-3 of float32 error,
        # which is small but lands on the one quantity that must be exact: at
        # initialisation this model *is* persistence, and a test asserting that should not
        # have to tolerate drift the architecture never needed to introduce.
        last_x, _, has_last = last_observed_with_age(x, x_mask)
        anchor = (has_last * last_x + (1.0 - has_last) * mu.squeeze(1)).unsqueeze(1)
        if self.use_clim:
            # Blend toward the seasonal mean by a learned, smooth, per-lead weight. Where
            # the climatology mask is zero the weight is zero and the anchor is exactly
            # the persistence anchor above.
            g = self.film_gate.head_all_leads(f2, self.head_gate)  # (B, L, H/4, W/4)
            gate = torch.sigmoid(
                F.interpolate(g, size=(h, w), mode="bilinear", align_corners=False)
            )
            gate = gate * clim_mask
            anchor = (1.0 - gate) * anchor + gate * clim
        pred = anchor + delta * sigma
        if not return_exceed:
            return pred * water
        if self.exceed_head is None:
            raise ValueError("return_exceed=True on a model built without exceed_thresholds")
        logits = self.exceed_head(pred, feats, water, f2, sigma)
        return pred * water, logits
