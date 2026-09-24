"""A deeper, wider anomaly forecaster: residual stages plus global attention.

Kept in its own module rather than added to `anomaly.py` because `AnomalyForecaster` is
frozen. It produced the published leave-lake-out numbers and the checkpoint the dashboard
renders, and changing its module structure would silently break loading that checkpoint.
Two classes can be compared; one mutated class cannot.

Every structural guarantee of the shallow model is preserved, and for the same reasons:

* reversible instance normalisation, so the network never sees a lake's absolute
  trophic level (the training lakes span 5.5x in mg/m^3);
* an exact persistence anchor read straight from the input;
* a bounded `scale * tanh(...)` delta, so an unfamiliar lake cannot produce an unbounded
  correction -- the failure that made the ConvLSTM *worse* than doing nothing;
* a delta decoded only at coarse resolutions, so it cannot be pixel speckle.

What depth adds: a receptive field that covers a whole lake rather than a neighbourhood,
capacity to represent more than one bloom regime, and -- via bottleneck attention -- the
ability to condition one end of a lake on what the far end is doing.

A caution that belongs in the code and not only in a report: on this dataset capacity has
so far been *inversely* related to held-out skill.

    model            params    held-out skill @ 5 days
    U-Net             851 K            -0.355
    ConvLSTM          356 K            -1.225
    AnomalyForecaster 333 K            +0.042

Three training lakes is very little to constrain a deep network, and what produces the
skill in the shallow model is its priors, not its size. This class exists so that claim
can be tested rather than asserted. Compare on the held-out lake before preferring it.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from habforecast.models.anomaly import (
    FiLM,
    WindowSummary,
    _pool_if_possible,
    last_observed_with_age,
    masked_moments,
)


class ResBlock(nn.Module):
    """Pre-activation residual block.

    GroupNorm rather than BatchNorm: batches here are 8 crops, and BatchNorm statistics
    over 8 samples are noisy enough to hurt. The second convolution is zero-initialised
    so the block starts as an identity map -- a deep stack of identities trains from the
    same place a shallow network does, which is what makes adding depth safe rather than
    a fresh optimisation problem.
    """

    def __init__(self, channels: int, groups: int = 8, dropout: float = 0.0):
        super().__init__()
        g = min(groups, channels)
        self.norm1 = nn.GroupNorm(g, channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(g, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        assert self.conv2.bias is not None
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(self.drop(F.silu(self.norm2(h))))
        return x + h


class BottleneckAttention(nn.Module):
    """Self-attention over the coarsest feature map.

    Convolutions give a local receptive field; a bloom is a lake-wide phenomenon, and
    whether the far end of the lake is blooming is genuinely informative about this end.
    At 1/16 resolution a 128x128 crop is 8x8 = 64 tokens, so full attention costs almost
    nothing and buys a global receptive field that stacking 3x3 convolutions cannot.

    Zero-initialised output projection, so the module begins as an identity.
    """

    def __init__(self, channels: int, groups: int = 8):
        super().__init__()
        self.norm = nn.GroupNorm(min(groups, channels), channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
        assert self.proj.bias is not None
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        q, k, v = self.qkv(self.norm(x)).chunk(3, dim=1)
        q = q.reshape(b, c, h * w).transpose(1, 2)
        k = k.reshape(b, c, h * w).transpose(1, 2)
        v = v.reshape(b, c, h * w).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).reshape(b, c, h, w)
        return x + self.proj(out)


class DeepAnomalyForecaster(nn.Module):
    """Level-invariant, persistence-anchored forecaster with a deep residual trunk.

    Args:
        input_days: length of the past window (shape bookkeeping only).
        n_leads: number of forecast horizons emitted.
        base: width of the first stage; channels double per stage, capped at 8x base.
        depth: number of downsampling stages. The delta is decoded at the two coarsest.
        blocks: residual blocks per stage.
        use_attention: self-attention at the bottleneck.
        delta_scale: initial cap on the departure from persistence, in window sigmas.
    """

    def __init__(
        self,
        input_days: int = 14,
        n_leads: int = 4,
        base: int = 32,
        depth: int = 4,
        blocks: int = 2,
        use_doy: bool = True,
        use_attention: bool = True,
        delta_scale: float = 0.5,
        n_scales: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        if depth < 2:
            raise ValueError("depth must be at least 2: the delta needs two coarse heads")
        self.input_days = input_days
        self.n_leads = n_leads
        self.use_doy = use_doy
        self.depth = depth

        self.summary = WindowSummary(n_scales=n_scales)
        in_ch = self.summary.out_channels + 1 + (2 if use_doy else 0)

        # Widths double per stage but are capped: at 1/16 resolution the map is tiny and
        # extra channels there buy parameters, not receptive field.
        widths = [min(base * 2**i, base * 8) for i in range(depth + 1)]
        self.widths = widths

        self.stem = nn.Conv2d(in_ch, widths[0], 3, padding=1)
        self.projections = nn.ModuleList()
        self.stages = nn.ModuleList()
        for i in range(depth):
            self.projections.append(nn.Conv2d(widths[i], widths[i + 1], 1))
            self.stages.append(
                nn.Sequential(*[ResBlock(widths[i + 1], dropout=dropout) for _ in range(blocks)])
            )
        self.attn = BottleneckAttention(widths[depth]) if use_attention else nn.Identity()

        c_coarse, c_mid = widths[depth], widths[depth - 1]
        self.film_coarse = FiLM(n_leads, c_coarse)
        self.film_mid = FiLM(n_leads, c_mid + c_coarse)
        self.head_coarse = nn.Conv2d(c_coarse, 1, 1)
        self.head_mid = nn.Conv2d(c_mid + c_coarse, 1, 1)
        for head in (self.head_coarse, self.head_mid):
            assert head.bias is not None
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

        self.log_scale = nn.Parameter(
            torch.full((n_leads,), float(torch.log(torch.tensor(delta_scale))))
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
    ) -> torch.Tensor:
        _b, _t, h, w = x.shape

        mu, sigma, _ = masked_moments(x, x_mask, water)
        z = (x - mu) / sigma * x_mask

        parts = [self.summary(z, x_mask), water]
        if self.use_doy and doy is not None:
            parts.append(self._doy_channels(doy.to(x.dtype), (h, w)))
        f = self.stem(torch.cat(parts, dim=1))

        feats = []
        for proj, stage in zip(self.projections, self.stages, strict=True):
            # Stop halving once there is nothing left to halve: a 12 km lake on a
            # 4 km grid is 7 px, and `avg_pool2d` raises on a 1x1 input rather than
            # returning an empty tensor. See `_pool_if_possible` in anomaly.py.
            f = stage(proj(_pool_if_possible(f)))
            feats.append(f)
        feats[-1] = self.attn(feats[-1])

        coarse_feat = feats[-1]  # 1/2^depth
        mid_feat = feats[-2]  # 1/2^(depth-1)
        # The mid head sees the bottleneck's global summary alongside its own detail.
        mid_feat = torch.cat(
            [
                mid_feat,
                F.interpolate(
                    coarse_feat, size=mid_feat.shape[-2:], mode="bilinear", align_corners=False
                ),
            ],
            dim=1,
        )

        scale = torch.exp(self.log_scale).view(1, -1, 1, 1)
        c = self.film_coarse.head_all_leads(coarse_feat, self.head_coarse)
        m = self.film_mid.head_all_leads(mid_feat, self.head_mid)
        delta = F.interpolate(c, size=(h, w), mode="bilinear", align_corners=False)
        delta = delta + F.interpolate(m, size=(h, w), mode="bilinear", align_corners=False)
        delta = scale * torch.tanh(delta)

        last_x, _, has_last = last_observed_with_age(x, x_mask)
        anchor = has_last * last_x + (1.0 - has_last) * mu.squeeze(1)
        return (anchor.unsqueeze(1) + delta * sigma) * water
