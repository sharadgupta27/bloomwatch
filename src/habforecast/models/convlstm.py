"""ConvLSTM encoder-forecaster for gap-riddled satellite image sequences.

A ConvLSTM keeps a spatial hidden state and updates it with convolutions, so it can
represent advection -- a bloom patch being pushed across the lake by wind -- in a way
that a per-pixel recurrent model cannot. That matters here: Lake Erie blooms visibly
translate between overpasses.

The adaptation for this problem is that **each input step carries its own observation
mask**, and the cell's input gate sees it. Unobserved pixels therefore arrive as an
explicit "no information" signal rather than as a zero the network must learn to
distinguish from a genuine zero-anomaly reading. With ~25-40% temporal coverage this is
not a refinement; without it the model spends its capacity learning cloud patterns.

Sized for 4 GB of VRAM: 32-48 hidden channels at 128x128 with AMP trains at batch 8
using well under 3 GB.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvLSTMCell(nn.Module):
    """A single ConvLSTM cell with grouped-norm-stabilised gates."""

    def __init__(self, in_channels: int, hidden_channels: int, kernel_size: int = 3):
        super().__init__()
        self.hidden_channels = hidden_channels
        padding = kernel_size // 2
        # One convolution produces all four gates; cheaper than four separate convs.
        self.conv = nn.Conv2d(
            in_channels + hidden_channels,
            4 * hidden_channels,
            kernel_size,
            padding=padding,
            bias=True,
        )
        # Forget-gate bias at 1.0 is the standard LSTM trick: it starts the cell in a
        # remembering regime, which matters when many timesteps carry no observation.
        bias = self.conv.bias
        assert bias is not None  # built with bias=True above
        with torch.no_grad():
            bias.zero_()
            bias[hidden_channels : 2 * hidden_channels].fill_(1.0)

    def forward(
        self, x: torch.Tensor, state: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h, c = state
        gates = self.conv(torch.cat([x, h], dim=1))
        i, f, o, g = gates.chunk(4, dim=1)
        i, f, o = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o)
        g = torch.tanh(g)
        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

    def init_state(self, batch: int, size: tuple[int, int], device, dtype):
        h = torch.zeros(batch, self.hidden_channels, *size, device=device, dtype=dtype)
        return h, torch.zeros_like(h)


class ConvLSTMForecaster(nn.Module):
    """Encode a masked observation sequence, then emit one map per lead time.

    Args:
        input_days: number of past daily frames.
        n_leads: number of forecast lead times to emit.
        hidden: hidden channels per ConvLSTM layer.
        layers: stacked ConvLSTM layers.
        use_doy: append a two-channel sin/cos day-of-year encoding. Bloom phenology is
            strongly seasonal, and this is far cheaper than letting the network infer
            season from chlorophyll magnitude alone.
    """

    def __init__(
        self,
        input_days: int = 14,
        n_leads: int = 3,
        hidden: int = 48,
        layers: int = 2,
        kernel_size: int = 3,
        use_doy: bool = True,
        dropout: float = 0.1,
        gradient_checkpointing: bool = True,
    ):
        super().__init__()
        self.input_days = input_days
        self.n_leads = n_leads
        self.use_doy = use_doy
        # A ConvLSTM keeps every timestep's gate activations alive for the backward pass:
        # T x layers x 4 x hidden x H x W. At T=14, batch 8, 128x128 that measured 4.83 GB
        # -- over this project's 4 GB budget. Checkpointing recomputes each step during
        # backward instead of storing it, trading roughly 30% more compute for ~O(1)
        # timestep memory. Training-only; inference under no_grad stores nothing anyway.
        self.gradient_checkpointing = gradient_checkpointing

        # per-step channels: value, observation mask, static water mask, (+2 doy)
        in_ch = 3 + (2 if use_doy else 0)
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 3, padding=1),
            nn.GroupNorm(8, hidden),
            nn.SiLU(),
        )
        self.cells = nn.ModuleList(
            [ConvLSTMCell(hidden, hidden, kernel_size) for _ in range(layers)]
        )
        self.norms = nn.ModuleList([nn.GroupNorm(8, hidden) for _ in range(layers)])
        self.dropout = nn.Dropout2d(dropout)

        self.head = nn.Sequential(
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.GroupNorm(8, hidden),
            nn.SiLU(),
            nn.Conv2d(hidden, n_leads, 1),
        )

    def _cell_stack(self, inp_cat: torch.Tensor, *flat_state: torch.Tensor):
        """One timestep through the stem and every ConvLSTM layer.

        Takes and returns a flat tensor tuple so it can be wrapped in
        `torch.utils.checkpoint`, which does not accept nested containers.
        """
        inp = self.stem(inp_cat)
        new_state: list[torch.Tensor] = []
        for li, cell in enumerate(self.cells):
            h, c = flat_state[2 * li], flat_state[2 * li + 1]
            h, c = cell(inp, (h, c))
            new_state += [h, c]
            inp = self.dropout(self.norms[li](h))
        return tuple(new_state)

    @staticmethod
    def _doy_channels(doy: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        angle = 2 * torch.pi * doy / 365.25
        sin = torch.sin(angle).view(-1, 1, 1, 1).expand(-1, 1, *size)
        cos = torch.cos(angle).view(-1, 1, 1, 1).expand(-1, 1, *size)
        return torch.cat([sin, cos], dim=1)

    def forward(
        self,
        x: torch.Tensor,  # (B, T, H, W) standardised values, 0 where unobserved
        x_mask: torch.Tensor,  # (B, T, H, W) 1 where observed
        water: torch.Tensor,  # (B, 1, H, W)
        doy: torch.Tensor | None = None,  # (B,)
    ) -> torch.Tensor:
        b, t, h, w = x.shape
        size = (h, w)
        device, dtype = x.device, x.dtype

        # Flat [h0, c0, h1, c1, ...] so the whole state can cross a checkpoint boundary,
        # which only accepts and returns tensors.
        flat_state: list[torch.Tensor] = []
        for cell in self.cells:
            h0, c0 = cell.init_state(b, size, device, dtype)
            flat_state += [h0, c0]

        doy_ch = (
            self._doy_channels(doy.to(dtype), size) if (self.use_doy and doy is not None) else None
        )

        use_ckpt = self.gradient_checkpointing and self.training and torch.is_grad_enabled()
        for step in range(t):
            parts = [x[:, step : step + 1], x_mask[:, step : step + 1], water]
            if doy_ch is not None:
                parts.append(doy_ch)
            inp_cat = torch.cat(parts, dim=1)

            if use_ckpt:
                flat_state = list(
                    torch.utils.checkpoint.checkpoint(
                        self._cell_stack, inp_cat, *flat_state, use_reentrant=False
                    )
                )
            else:
                flat_state = list(self._cell_stack(inp_cat, *flat_state))

        out = self.head(flat_state[-2])  # final layer's hidden state

        # Anchor the forecast on the most recent observation: the network predicts a
        # *change* from the last seen state rather than the absolute field. This is the
        # single most effective inductive bias here -- it makes persistence the model's
        # default behaviour and forces it to earn any departure from it.
        last_obs, has_last = _last_observed(x, x_mask)
        out = out + last_obs.unsqueeze(1) * has_last.unsqueeze(1)
        return out * water  # never predict over land


def _last_observed_index(mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Index of the newest observed step per pixel, and whether any step was observed.

    Shapes (B, H, W). Where nothing was observed the index is 0 and `has` is 0; callers
    multiply by `has` so that slot contributes nothing. One `max` over the time axis
    replaces a Python loop of T dependent kernel launches, which for a 14-day window is
    the single most repeated piece of arithmetic in both training and serving.
    """
    t = mask.shape[1]
    steps = torch.arange(1, t + 1, device=mask.device, dtype=mask.dtype).view(1, t, 1, 1)
    # A mask is 0 or 1, so the weighted max lands on the newest observed step.
    newest, idx = (mask * steps).max(dim=1)
    has = (newest > 0).to(mask.dtype)
    return idx, has


def _last_observed(x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Most recent observed value per pixel, and whether any observation exists.

    Takes the newest observed frame at each pixel, which is what "persistence" means when
    the latest frame is partly cloudy.
    """
    idx, has = _last_observed_index(mask)
    last = x.gather(1, idx.unsqueeze(1)).squeeze(1) * has
    return last, has


class UNetForecaster(nn.Module):
    """A compact U-Net treating the time window as input channels.

    Included as a deliberate control. With only a few thousand training samples the
    recurrent model's extra capacity may not pay for itself, and a feed-forward net that
    sees the whole window at once is a genuinely competitive -- and much faster --
    alternative. Reporting both is the honest thing to do.
    """

    def __init__(
        self,
        input_days: int = 14,
        n_leads: int = 3,
        base: int = 32,
        use_doy: bool = True,
    ):
        super().__init__()
        self.use_doy = use_doy
        in_ch = 2 * input_days + 1 + (2 if use_doy else 0)

        def block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1),
                nn.GroupNorm(8, cout),
                nn.SiLU(),
                nn.Conv2d(cout, cout, 3, padding=1),
                nn.GroupNorm(8, cout),
                nn.SiLU(),
            )

        self.enc1 = block(in_ch, base)
        self.enc2 = block(base, base * 2)
        self.enc3 = block(base * 2, base * 4)
        self.bottleneck = block(base * 4, base * 4)
        self.dec3 = block(base * 8, base * 2)
        self.dec2 = block(base * 4, base)
        self.dec1 = block(base * 2, base)
        self.out = nn.Conv2d(base, n_leads, 1)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x, x_mask, water, doy=None):
        parts = [x, x_mask, water]
        if self.use_doy and doy is not None:
            parts.append(ConvLSTMForecaster._doy_channels(doy.to(x.dtype), x.shape[-2:]))
        z = torch.cat(parts, dim=1)

        e1 = self.enc1(z)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))

        d3 = self.dec3(torch.cat([F.interpolate(b, size=e3.shape[-2:], mode="nearest"), e3], 1))
        d2 = self.dec2(torch.cat([F.interpolate(d3, size=e2.shape[-2:], mode="nearest"), e2], 1))
        d1 = self.dec1(torch.cat([F.interpolate(d2, size=e1.shape[-2:], mode="nearest"), e1], 1))
        out = self.out(d1)

        last_obs, has_last = _last_observed(x, x_mask)
        out = out + last_obs.unsqueeze(1) * has_last.unsqueeze(1)
        return out * water


def build_model(name: str, **kwargs) -> nn.Module:
    """Factory used by the Hydra training config."""
    from habforecast.models.anomaly import AnomalyForecaster
    from habforecast.models.anomaly_deep import DeepAnomalyForecaster

    models = {
        "convlstm": ConvLSTMForecaster,
        "unet": UNetForecaster,
        "anomaly": AnomalyForecaster,
        "anomaly_deep": DeepAnomalyForecaster,
    }
    if name not in models:
        raise ValueError(f"unknown model '{name}'; available: {sorted(models)}")
    return models[name](**kwargs)
