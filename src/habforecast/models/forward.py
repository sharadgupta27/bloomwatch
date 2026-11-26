"""How a forecaster is called: which inputs it reads, and test-time flip averaging.

Split out of `models.lit` so the serving path can run a network without importing
Lightning. `import lightning` pulls in torch._dynamo and the tensorboard logger and cost
~15 s on the first forecast after an API restart (profiled 2026-11-26) -- for helpers
that only ever touch torch. `models.lit` re-exports everything here, so training, the
evaluation pipeline and the tests call exactly the same functions.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def model_inputs(model: nn.Module, batch: dict) -> dict[str, torch.Tensor | None]:
    """The keyword arguments a forecaster takes, drawn from a dataset batch.

    One place decides which optional channels a model reads -- auxiliary layers for a
    checkpoint trained with them, the seasonal climatology for `use_clim` -- so training,
    evaluation and serving cannot drift into calling the same network three different
    ways. A model that predates a channel is never handed it, which is what keeps every
    older checkpoint loadable and its numbers unchanged.
    """
    kwargs: dict[str, torch.Tensor | None] = {
        "x": batch["x"],
        "x_mask": batch["x_mask"],
        "water": batch["water"],
        "doy": batch.get("doy"),
    }
    if getattr(model, "n_aux", 0):
        kwargs["aux"] = batch.get("aux")
        kwargs["aux_mask"] = batch.get("aux_mask")
    if getattr(model, "use_clim", False):
        kwargs["clim"] = batch.get("clim")
        kwargs["clim_mask"] = batch.get("clim_mask")
    if getattr(model, "n_met", 0):
        kwargs["met"] = batch.get("met")
        kwargs["met_mask"] = batch.get("met_mask")
        if getattr(model, "met_future", False):
            kwargs["met_future"] = batch.get("met_future")
            kwargs["met_future_mask"] = batch.get("met_future_mask")
    if getattr(model, "n_group_factors", 0):
        kwargs["groups"] = batch.get("groups")
    return kwargs


# Spatial flips share the last two axes on every tensor a forecaster reads, so one table
# of axis pairs covers (B,T,H,W), (B,A,T,H,W) and (B,L,H,W) alike -- but only for the
# tensors whose last two axes really are y and x. The meteorological forcing is (B,V,T):
# it has rank 3 like a spatial stack and its last axis is *time*, so a rank test alone
# would quietly reverse the weather series, and flipping its other axis would permute the
# variables. Those keys are excluded by name.
_FLIPS: tuple[tuple[int, ...], ...] = ((), (-1,), (-2,), (-1, -2))

_NON_SPATIAL: frozenset[str] = frozenset(
    {"doy", "met", "met_mask", "met_future", "met_future_mask", "lake_idx", "year", "groups"}
)


def predict_tta(
    model: nn.Module, inputs: dict, flips: bool = True, with_exceed: bool = False
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Forecast averaged over the dihedral flips the model was trained under.

    Every training sample is randomly flipped horizontally and vertically, so the four
    orientations are equally in-distribution and the network's answers on them are four
    draws of the same estimate. Averaging them is a free variance reduction with no new
    parameters, the same trick as in every segmentation benchmark. It is applied only at
    evaluation and serving; the flipped inputs are built from the same dict the plain
    call uses, so a channel cannot be flipped in one tensor and not another.

    With `with_exceed` the model's exceedance head is read too, and the return value is
    `(forecast, probability)`: the probability is averaged in probability space, not as
    logits, so the TTA mean is the same kind of quantity an ensemble mean is.
    """
    if with_exceed:
        return _predict_tta_exceed(model, inputs, flips)
    if not flips:
        return model(**inputs)
    total: torch.Tensor | None = None
    for dims in _FLIPS:
        out = model(**_flip_inputs(inputs, dims))
        out = torch.flip(out, dims) if dims else out
        total = out if total is None else total + out
    assert total is not None
    return total / len(_FLIPS)


def _flip_inputs(inputs: dict, dims: tuple[int, ...]) -> dict:
    return {
        k: (
            torch.flip(v, dims)
            if (dims and torch.is_tensor(v) and v.dim() >= 3 and k not in _NON_SPATIAL)
            else v
        )
        for k, v in inputs.items()
    }


def _predict_tta_exceed(
    model: nn.Module, inputs: dict, flips: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    total_pred: torch.Tensor | None = None
    total_prob: torch.Tensor | None = None
    orientations = _FLIPS if flips else ((),)
    for dims in orientations:
        pred, logits = model(**_flip_inputs(inputs, dims), return_exceed=True)
        prob = torch.sigmoid(logits.float())
        if dims:
            pred, prob = torch.flip(pred, dims), torch.flip(prob, dims)
        total_pred = pred if total_pred is None else total_pred + pred
        total_prob = prob if total_prob is None else total_prob + prob
    assert total_pred is not None and total_prob is not None  # never zero orientations
    n = len(orientations)
    return total_pred / n, total_prob / n


def has_exceed_head(model: nn.Module) -> bool:
    return bool(getattr(model, "exceed_thresholds", ()))
