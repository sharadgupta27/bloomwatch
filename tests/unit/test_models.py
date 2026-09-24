"""Model contract tests: shapes, masking behaviour, and the persistence prior.

The most important test here is `test_predicts_persistence_at_init`. Both forecasters add
their output to the last observed value, so an untrained network starts as a persistence
forecaster. That is the inductive bias the whole design rests on -- it means the model
begins at the baseline instead of at noise, and every training step has to justify a
departure from it. If a refactor ever breaks that anchoring, the skill score would
degrade silently rather than error, so it gets an explicit test.
"""

from __future__ import annotations

import pytest
import torch

from habforecast.models.baselines import PersistenceBaseline
from habforecast.models.convlstm import (
    ConvLSTMCell,
    _last_observed,
    build_model,
)
from habforecast.models.lit import masked_huber, masked_rmse

B, T, H, W, L = 2, 6, 32, 32, 3


def make_batch(observed_fraction: float = 0.4, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(B, T, H, W, generator=g)
    x_mask = (torch.rand(B, T, H, W, generator=g) < observed_fraction).float()
    water = (torch.rand(B, 1, H, W, generator=g) < 0.7).float()
    doy = torch.tensor([180.0, 200.0])
    return x * x_mask, x_mask, water, doy


class TestBuildModel:
    @pytest.mark.parametrize("name", ["convlstm", "unet"])
    def test_builds_and_runs(self, name):
        model = build_model(name, input_days=T, n_leads=L)
        x, xm, water, doy = make_batch()
        out = model(x, xm, water, doy)
        assert out.shape == (B, L, H, W)
        assert torch.isfinite(out).all()

    def test_unknown_name_raises(self):
        with pytest.raises(ValueError, match="unknown model"):
            build_model("transformer_xl")

    @pytest.mark.parametrize("name", ["convlstm", "unet"])
    def test_parameter_budget(self, name):
        """Models must stay small enough to train on 4 GB."""
        model = build_model(name, input_days=14, n_leads=4)
        n = sum(p.numel() for p in model.parameters())
        assert n < 5_000_000, f"{name} has {n / 1e6:.2f}M params - too large for the VRAM budget"


class TestLastObserved:
    def test_takes_most_recent_observation(self):
        x = torch.zeros(1, 3, 1, 1)
        mask = torch.zeros(1, 3, 1, 1)
        x[0, 0, 0, 0], mask[0, 0, 0, 0] = 5.0, 1.0
        x[0, 2, 0, 0], mask[0, 2, 0, 0] = 9.0, 1.0
        last, filled = _last_observed(x, mask)
        assert last.item() == pytest.approx(9.0)
        assert filled.item() == pytest.approx(1.0)

    def test_falls_back_through_gaps(self):
        """When the newest frame is cloudy, walk back to the last clear one."""
        x = torch.zeros(1, 3, 1, 1)
        mask = torch.zeros(1, 3, 1, 1)
        x[0, 0, 0, 0], mask[0, 0, 0, 0] = 7.0, 1.0  # only the oldest is observed
        last, filled = _last_observed(x, mask)
        assert last.item() == pytest.approx(7.0)
        assert filled.item() == pytest.approx(1.0)

    def test_reports_unfilled_when_nothing_observed(self):
        last, filled = _last_observed(torch.zeros(1, 3, 1, 1), torch.zeros(1, 3, 1, 1))
        assert filled.item() == pytest.approx(0.0)
        assert last.item() == pytest.approx(0.0)

    def test_matches_the_backward_walk_on_random_windows(self):
        """The vectorised gather must equal the step-by-step walk it replaced."""
        x, mask, _, _ = make_batch(observed_fraction=0.3, seed=3)
        last, filled = _last_observed(x, mask)
        ref_last = torch.zeros(B, H, W)
        ref_filled = torch.zeros(B, H, W)
        for step in range(T - 1, -1, -1):
            take = mask[:, step] * (1.0 - ref_filled)
            ref_last = ref_last + take * x[:, step]
            ref_filled = torch.clamp(ref_filled + take, max=1.0)
        torch.testing.assert_close(last, ref_last)
        torch.testing.assert_close(filled, ref_filled)


class TestPersistencePrior:
    @pytest.mark.parametrize("name", ["convlstm", "unet"])
    def test_predicts_persistence_at_init(self, name):
        """An untrained model must reproduce persistence, not noise.

        Both heads end in a conv whose output is *added* to the last observed value. With
        the head zeroed, the model is exactly persistence -- the baseline it must beat.
        """
        model = build_model(name, input_days=T, n_leads=L).eval()
        head = model.head[-1] if name == "convlstm" else model.out
        with torch.no_grad():
            head.weight.zero_()
            head.bias.zero_()

        x, xm, water, doy = make_batch()
        with torch.no_grad():
            out = model(x, xm, water, doy)
            expected = PersistenceBaseline(n_leads=L)(x, xm, water)
        torch.testing.assert_close(out, expected, rtol=1e-4, atol=1e-5)


class TestLandMasking:
    @pytest.mark.parametrize("name", ["convlstm", "unet"])
    def test_never_predicts_over_land(self, name):
        model = build_model(name, input_days=T, n_leads=L).eval()
        x, xm, _, doy = make_batch()
        water = torch.zeros(B, 1, H, W)
        water[:, :, :16, :16] = 1.0
        with torch.no_grad():
            out = model(x, xm, water, doy)
        land = water.expand(-1, L, -1, -1) == 0
        assert torch.all(out[land] == 0), "model produced non-zero output over land"


class TestConvLSTMCell:
    def test_state_shapes(self):
        cell = ConvLSTMCell(4, 8)
        h, c = cell.init_state(2, (16, 16), torch.device("cpu"), torch.float32)
        assert h.shape == c.shape == (2, 8, 16, 16)

    def test_forget_gate_bias_initialised_to_one(self):
        """Standard LSTM trick: start in a remembering regime.

        Matters more than usual here, where many timesteps carry no observation at all.
        """
        hidden = 8
        cell = ConvLSTMCell(4, hidden)
        forget_bias = cell.conv.bias[hidden : 2 * hidden]
        assert torch.allclose(forget_bias, torch.ones(hidden))

    def test_step_changes_state(self):
        cell = ConvLSTMCell(3, 6)
        state = cell.init_state(1, (8, 8), torch.device("cpu"), torch.float32)
        h, _c = cell(torch.randn(1, 3, 8, 8), state)
        assert h.shape == (1, 6, 8, 8)
        assert not torch.allclose(h, state[0])


class TestMaskedLosses:
    def test_ignores_masked_pixels_entirely(self):
        pred = torch.zeros(1, 1, 4, 4)
        target = torch.zeros(1, 1, 4, 4)
        target[0, 0, 0, 0] = 100.0  # huge error, but masked out
        mask = torch.ones(1, 1, 4, 4)
        mask[0, 0, 0, 0] = 0.0
        assert masked_huber(pred, target, mask).item() == pytest.approx(0.0)
        assert masked_rmse(pred, target, mask).item() == pytest.approx(0.0)

    def test_normalises_by_mask_not_numel(self):
        """Two samples with identical errors but different cloud cover must score alike.

        Dividing by numel() instead would make the loss a function of cloudiness, and the
        optimiser would learn to shrink predictions toward zero.
        """
        pred = torch.full((1, 1, 8, 8), 1.0)
        target = torch.zeros(1, 1, 8, 8)
        sparse = torch.zeros(1, 1, 8, 8)
        sparse[0, 0, :2, :2] = 1.0
        dense = torch.ones(1, 1, 8, 8)
        assert masked_huber(pred, target, sparse).item() == pytest.approx(
            masked_huber(pred, target, dense).item(), rel=1e-5
        )

    def test_empty_mask_does_not_divide_by_zero(self):
        loss = masked_huber(
            torch.ones(1, 1, 4, 4), torch.zeros(1, 1, 4, 4), torch.zeros(1, 1, 4, 4)
        )
        assert torch.isfinite(loss)

    def test_lead_weighting_upweights_longer_horizons(self):
        pred = torch.zeros(1, 2, 4, 4)
        target = torch.zeros(1, 2, 4, 4)
        target[0, 1] = 1.0  # error only at the longer lead
        mask = torch.ones(1, 2, 4, 4)
        uniform = masked_huber(pred, target, mask, lead_weights=torch.tensor([1.0, 1.0]))
        weighted = masked_huber(pred, target, mask, lead_weights=torch.tensor([0.5, 1.5]))
        assert weighted > uniform


class TestGradients:
    @pytest.mark.parametrize("name", ["convlstm", "unet"])
    def test_backward_produces_finite_grads(self, name):
        model = build_model(name, input_days=T, n_leads=L)
        x, xm, water, doy = make_batch()
        out = model(x, xm, water, doy)
        target = torch.randn_like(out)
        mask = (torch.rand_like(out) < 0.3).float()
        masked_huber(out, target, mask).backward()

        touched = [p for p in model.parameters() if p.grad is not None]
        assert touched, "no parameter received a gradient"
        for p in touched:
            assert torch.isfinite(p.grad).all()


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
class TestVRAMBudget:
    @pytest.mark.parametrize("name", ["convlstm", "unet"])
    def test_fits_in_four_gigabytes(self, name):
        """Training-step memory must stay inside the 4 GB card this was built for."""
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        dev = torch.device("cuda")

        model = build_model(name, input_days=14, n_leads=4).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
        bs, crop = 8, 128
        x = torch.randn(bs, 14, crop, crop, device=dev)
        xm = (torch.rand(bs, 14, crop, crop, device=dev) < 0.4).float()
        water = torch.ones(bs, 1, crop, crop, device=dev)
        doy = torch.full((bs,), 200.0, device=dev)
        y = torch.randn(bs, 4, crop, crop, device=dev)
        ym = torch.ones_like(y)

        with torch.autocast("cuda", dtype=torch.float16):
            loss = masked_huber(model(x, xm, water, doy), y, ym)
        loss.backward()
        opt.step()

        peak_gb = torch.cuda.max_memory_allocated() / 1e9
        torch.cuda.empty_cache()
        assert peak_gb < 3.2, f"{name} peaked at {peak_gb:.2f} GB - too close to the 4 GB limit"
