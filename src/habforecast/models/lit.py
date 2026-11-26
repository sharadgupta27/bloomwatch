"""Lightning module: masked training objective, per-lead metrics, persistence skill.

The loss is *masked and normalised by the mask*, not by the tensor size. That distinction
is the whole ballgame on this dataset. With 60-75% of target pixels unobserved, dividing
by `numel()` would make the loss depend mostly on how cloudy a sample happened to be,
and the optimiser would learn to shrink predictions toward zero because that is the
cheapest way to reduce a loss dominated by masked-out positions.

Huber rather than MSE: chlorophyll retrievals have a heavy right tail (surface scum can
exceed 300 mg/m^3) and a handful of such pixels under MSE would dominate every gradient.

Persistence skill is logged every validation epoch rather than computed once at the end,
because it is the number that decides whether the model is worth anything -- it belongs
in the training curve, not in a footnote.

UNITS: every RMSE logged here is in *standardised* units (the model's own space), not
log10 mg/m^3. Comparing a training curve directly against the numbers in
`artifacts/eval/scores.json` will therefore mislead -- those are converted back to log10
by `pipelines/evaluate.py`. The skill scores are unaffected, because a ratio of two RMSEs
in the same units is scale-invariant, which is the main reason skill rather than raw RMSE
is the quantity tracked during training.
"""

from __future__ import annotations

import logging
import math
from typing import cast

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning.fabric.utilities.data import AttributeDict

from habforecast.models.convlstm import _last_observed

logger = logging.getLogger(__name__)


def masked_huber(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    delta: float = 1.0,
    lead_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Huber loss averaged over observed pixels only.

    Args:
        pred, target, mask: (B, L, H, W). `mask` is 1 where the target was observed.
        lead_weights: optional (L,) weighting. Longer leads are harder; up-weighting them
            stops the model from optimising almost entirely for the easy 1-day case.
    """
    elementwise = F.huber_loss(pred, target, reduction="none", delta=delta)
    m = mask
    if lead_weights is not None:
        w = lead_weights.view(1, -1, 1, 1).to(pred.device, pred.dtype)
        elementwise = elementwise * w
        m = m * w
    denom = m.sum().clamp_min(1.0)
    return (elementwise * mask).sum() / denom


def masked_focal(
    logits: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    gamma: float = 2.0,
    lead_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Sigmoid focal loss (Lin et al. 2017) averaged over observed pixels only.

    Blooms are rare -- 2% of observed Balaton pixels exceed 20 mg/m^3, 8% of Victoria's --
    and plain cross-entropy spends almost all its gradient confirming that clear water is
    clear. The (1 - p_t)^gamma factor down-weights the easy pixels so the head learns the
    boundary. No class-balancing alpha: it shifts the probabilities away from the base
    rate, and the point of this head is a probability an advisory can quote.

    Args:
        logits, target, mask: (B, L, K, H, W). Computed in float32 whatever the autocast.
        lead_weights: optional (L,), as in `masked_huber`.
    """
    logits = logits.float()
    target = target.float()
    m = mask.float()
    ce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    p = torch.sigmoid(logits)
    p_t = p * target + (1.0 - p) * (1.0 - target)
    loss = ce * (1.0 - p_t).pow(gamma) if gamma else ce
    if lead_weights is not None:
        m = m * lead_weights.view(1, -1, 1, 1, 1).to(m.device, m.dtype)
    return (loss * m).sum() / m.sum().clamp_min(1.0)


def masked_rmse(pred, target, mask) -> torch.Tensor:
    denom = mask.sum().clamp_min(1.0)
    return torch.sqrt(((pred - target) ** 2 * mask).sum() / denom)


# Re-exported: these live in models.forward so serving can use them without Lightning.
from habforecast.models.forward import (  # noqa: E402,F401
    _FLIPS,
    _NON_SPATIAL,
    _flip_inputs,
    _predict_tta_exceed,
    has_exceed_head,
    model_inputs,
    predict_tta,
)


class HABForecastModule(L.LightningModule):
    """Trains a forecaster and reports skill against persistence at every lead time."""

    def __init__(
        self,
        model: nn.Module,
        lead_times: tuple[int, ...] = (1, 3, 7),
        lr: float = 3e-4,
        weight_decay: float = 1e-4,
        huber_delta: float = 1.0,
        lead_weighting: str = "linear",
        warmup_epochs: int = 2,
        max_epochs: int = 60,
        norm_mean: float = 0.0,
        norm_std: float = 1.0,
        model_config: dict | None = None,
        exceed_weight: float = 1.0,
        focal_gamma: float = 2.0,
    ):
        super().__init__()
        # `model` is a module, not a hyperparameter -- storing it in the checkpoint hparams
        # would serialise the weights twice. `model_config` records how to rebuild it.
        #
        # This matters: architecture cannot reliably be recovered from a state_dict alone.
        # A U-Net with use_doy=True and input_days=T has exactly the same input-channel
        # count (2T+3) as one with use_doy=False and input_days=T+1 (2(T+1)+1), so shape
        # inference silently picks the wrong one and `load_state_dict` succeeds anyway.
        self.save_hyperparameters(ignore=["model"])
        self.model = model
        self.lead_times = tuple(lead_times)
        self.register_buffer("lead_weights", self._make_lead_weights(lead_weighting))

    @property
    def hparams(self) -> AttributeDict:
        """`save_hyperparameters` always stores an AttributeDict; Lightning's annotation
        is the wider MutableMapping, which hides every attribute read from type checkers."""
        return cast(AttributeDict, super().hparams)

    def _make_lead_weights(self, scheme: str) -> torch.Tensor:
        n = len(self.lead_times)
        if scheme == "uniform":
            w = torch.ones(n)
        elif scheme == "linear":
            w = torch.tensor([float(t) for t in self.lead_times])
        elif scheme == "sqrt":
            w = torch.tensor([float(t) ** 0.5 for t in self.lead_times])
        else:
            raise ValueError(f"unknown lead_weighting '{scheme}'")
        return w / w.mean()

    def forward(self, batch) -> torch.Tensor:
        return self.model(**model_inputs(self.model, batch))

    # ----------------------------------------------------------------- shared steps --
    def _step(self, batch, stage: str) -> torch.Tensor:
        y, y_mask = batch["y"], batch["y_mask"]
        bs = y.shape[0]
        logits = None
        if has_exceed_head(self.model):
            pred, logits = self.model(**model_inputs(self.model, batch), return_exceed=True)
        else:
            pred = self(batch)

        loss = masked_huber(
            pred, y, y_mask, delta=self.hparams.huber_delta, lead_weights=self.lead_weights
        )
        if logits is not None:
            thr = self.model.exceed_head.thr.view(1, 1, -1, 1, 1)
            target = (y.unsqueeze(2) >= thr).to(logits.dtype)
            mask = y_mask.unsqueeze(2).expand_as(target)
            exceed = masked_focal(
                logits, target, mask, self.hparams.focal_gamma, lead_weights=self.lead_weights
            )
            self.log(f"{stage}/regression_loss", loss, batch_size=bs, sync_dist=True)
            self.log(f"{stage}/exceed_loss", exceed, batch_size=bs, sync_dist=True)
            loss = loss + self.hparams.exceed_weight * exceed
            if stage != "train":
                with torch.no_grad():
                    self._log_exceed_metrics(logits, target, mask, stage, bs)
        self.log(f"{stage}/loss", loss, prog_bar=(stage == "train"), batch_size=bs, sync_dist=True)

        with torch.no_grad():
            self._log_lead_metrics(pred, batch, stage, bs)
        return loss

    def _log_lead_metrics(self, pred, batch, stage: str, bs: int) -> None:
        y, y_mask = batch["y"], batch["y_mask"]
        last, has = _last_observed(batch["x"], batch["x_mask"])
        persistence = (last * has).unsqueeze(1) * batch["water"]  # broadcasts over leads

        # Every lead's sums in three reductions, and a single device->host transfer for
        # the per-lead counts. The previous per-lead loop synchronised the GPU on every
        # `m.sum() < 1` test -- four stalls per step for four leads -- for arithmetic that
        # is the same whichever order the leads are visited in.
        m = y_mask.to(pred.dtype)
        counts = m.sum(dim=(0, 2, 3))  # (L,)
        model_sse = ((pred - y) ** 2 * m).sum(dim=(0, 2, 3))
        pers_sse = ((persistence - y) ** 2 * m).sum(dim=(0, 2, 3))
        denom = counts.clamp_min(1.0)
        rmse = torch.sqrt(model_sse / denom)
        rmse_p = torch.sqrt(pers_sse / denom)
        # Murphy skill score: >0 means the model beats persistence at this lead.
        skill = 1.0 - (rmse**2) / (rmse_p**2).clamp_min(1e-8)

        present = [n >= 1 for n in counts.tolist()]
        for i, lead in enumerate(self.lead_times):
            if not present[i]:
                continue
            self.log(f"{stage}/rmse_d{lead}", rmse[i], batch_size=bs, sync_dist=True)
            self.log(f"{stage}/rmse_persist_d{lead}", rmse_p[i], batch_size=bs, sync_dist=True)
            self.log(
                f"{stage}/skill_d{lead}",
                skill[i],
                prog_bar=(stage == "val"),
                batch_size=bs,
                sync_dist=True,
            )

        # A lead with no observed target contributes zero to every sum, so pooling over
        # all leads equals pooling over the present ones.
        if any(present):
            overall = 1.0 - (model_sse.sum() / pers_sse.sum().clamp_min(1e-8))
            self.log(
                f"{stage}/skill_overall",
                overall,
                prog_bar=(stage == "val"),
                batch_size=bs,
                sync_dist=True,
            )
            self.log(
                f"{stage}/rmse",
                torch.sqrt(model_sse.sum() / counts.sum()),
                prog_bar=True,
                batch_size=bs,
                sync_dist=True,
            )

    def _log_exceed_metrics(self, logits, target, mask, stage: str, bs: int) -> None:
        """Brier score per threshold, pooled over leads -- the calibration curve's number.

        The ranking metrics need the whole split (see eval.metrics.StreamingRanking), so
        they are left to `pipelines/evaluate.py`; Brier is a per-pixel mean and streams.
        """
        prob = torch.sigmoid(logits.float())
        m = mask.float()
        se = ((prob - target.float()) ** 2 * m).sum(dim=(0, 1, 3, 4))
        n = m.sum(dim=(0, 1, 3, 4)).clamp_min(1.0)
        for k, thr in enumerate(self.model.exceed_thresholds):
            self.log(f"{stage}/brier_thr{thr:g}", se[k] / n[k], batch_size=bs, sync_dist=True)

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._step(batch, "test")

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        pred = self(batch)  # regression only; the exceedance head is read via predict_tta
        # Return in physical units: log10 mg/m^3.
        return pred * self.hparams.norm_std + self.hparams.norm_mean

    # ---------------------------------------------------------------------- optimiser --
    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        warmup = max(1, int(self.hparams.warmup_epochs))
        total = max(warmup + 1, int(self.hparams.max_epochs))

        def lr_lambda(epoch: int) -> float:
            if epoch < warmup:
                return (epoch + 1) / warmup
            progress = (epoch - warmup) / max(1, total - warmup)
            return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "epoch"}}
