"""Contract tests for the level-invariant anomaly forecaster.

These are not "does it run" tests. Each one pins a structural guarantee that the
architecture exists in order to provide, and that a refactor could silently remove while
leaving the loss curve looking fine:

* level invariance -- the reason it should generalise to an unseen lake at all;
* persistence at initialisation -- the reason training starts from a sane baseline;
* a hard cap on the departure from persistence -- the reason it cannot be catastrophically
  worse than doing nothing, which is exactly what the previous two models were;
* smoothness -- the reason it does not paint pixel-scale speckle the sensor never resolved.
"""

from __future__ import annotations

import pytest
import torch

from habforecast.models.anomaly import (
    MIN_SIGMA,
    AnomalyForecaster,
    last_observed_with_age,
    masked_moments,
)
from habforecast.models.anomaly_deep import DeepAnomalyForecaster

# Both architectures make the same four promises. Parametrising the contract here is what
# stops the deep variant quietly dropping one of them while still training to a nice loss.
ARCHITECTURES = [
    pytest.param(lambda **kw: AnomalyForecaster(base=16, **kw), id="shallow"),
    pytest.param(lambda **kw: DeepAnomalyForecaster(base=16, depth=3, blocks=1, **kw), id="deep"),
]


@pytest.fixture
def sample():
    torch.manual_seed(0)
    x = torch.randn(2, 14, 48, 48)
    mask = (torch.rand(2, 14, 48, 48) > 0.5).float()
    water = torch.ones(2, 1, 48, 48)
    doy = torch.tensor([200.0, 30.0])
    return x * mask, mask, water, doy


@pytest.fixture(params=ARCHITECTURES)
def build(request):
    """Factory for one architecture; every contract test runs against all of them."""
    return request.param


@pytest.fixture
def model(build):
    return build(input_days=14, n_leads=4).eval()


def persistence(x, mask):
    last, _, has = last_observed_with_age(x, mask)
    return (last * has).unsqueeze(1)


class TestFoldedLeadHeads:
    def test_all_leads_at_once_equals_one_lead_at_a_time(self, model, sample):
        """The folded head is the per-lead FiLM + 1x1 conv, not an approximation of it."""
        torch.manual_seed(1)
        with torch.no_grad():
            for module in (model.film_coarse, model.head_coarse):
                for p in module.parameters():
                    p.normal_(std=0.3)
        feat = torch.randn(2, model.head_coarse.in_channels, 6, 6)
        folded = model.film_coarse.head_all_leads(feat, model.head_coarse)
        for li in range(model.n_leads):
            idx = torch.full((2,), li, dtype=torch.long)
            one = model.head_coarse(model.film_coarse(feat, idx)).squeeze(1)
            torch.testing.assert_close(folded[:, li], one, rtol=1e-4, atol=1e-5)

    def test_age_and_anchor_match_a_backward_walk(self, sample):
        x, mask, _, _ = sample
        last, age, has = last_observed_with_age(x, mask)
        t = x.shape[1]
        ref_last = torch.zeros_like(last)
        ref_age = torch.zeros_like(age)
        ref_has = torch.zeros_like(has)
        for step in range(t - 1, -1, -1):
            take = mask[:, step] * (1.0 - ref_has)
            ref_last = ref_last + take * x[:, step]
            ref_age = ref_age + take * float(t - 1 - step)
            ref_has = torch.clamp(ref_has + take, max=1.0)
        torch.testing.assert_close(last, ref_last)
        torch.testing.assert_close(age, ref_age / (t - 1))
        torch.testing.assert_close(has, ref_has)


class TestLevelInvariance:
    """The property the whole architecture is built around."""

    def test_shifting_the_lake_shifts_the_forecast_by_the_same_amount(self, model, sample):
        x, mask, water, doy = sample
        with torch.no_grad():
            base = model(x, mask, water, doy)
            # The same lake, three orders of magnitude greener.
            shifted = model((x / mask.clamp(min=1) + 3.0) * mask, mask, water, doy)
        assert torch.allclose(shifted - 3.0, base, atol=1e-5)

    def test_a_lake_outside_the_training_range_is_not_out_of_distribution(self, model, sample):
        """A +5 sigma level shift must not change the forecast's *shape* at all."""
        x, mask, water, doy = sample
        with torch.no_grad():
            base = model(x, mask, water, doy)
            far = model((x / mask.clamp(min=1) + 5.0) * mask, mask, water, doy)
        centred_base = base - base.mean()
        centred_far = far - far.mean()
        assert torch.allclose(centred_base, centred_far, atol=1e-4)


class TestPersistenceAnchor:
    def test_initialises_to_exact_persistence_where_a_pixel_was_ever_seen(self, model, sample):
        """Bit-for-bit, not approximately.

        The anchor is read straight out of the input rather than reconstructed from the
        instance-normalised copy, precisely so this holds exactly. A round trip through
        (x - mu) / sigma * sigma + mu would leave ~1e-3 of float32 drift on the one
        quantity the architecture promises to reproduce.
        """
        x, mask, water, doy = sample
        _, _, has = last_observed_with_age(x, mask)
        with torch.no_grad():
            out = model(x, mask, water, doy)
        seen = has.unsqueeze(1).expand_as(out) > 0
        assert torch.equal(out[seen], persistence(x, mask).expand_as(out)[seen])

    def test_a_never_observed_pixel_falls_back_to_the_window_mean(self, model):
        """Not to zero: zero is a real chlorophyll value (1 mg/m^3), not "no idea"."""
        x = torch.randn(1, 14, 32, 32) + 4.0
        mask = torch.ones(1, 14, 32, 32)
        mask[..., :4, :4] = 0.0  # a corner nothing ever saw
        water = torch.ones(1, 1, 32, 32)
        with torch.no_grad():
            out = model(x * mask, mask, water, torch.tensor([100.0]))
            mu, _, _ = masked_moments(x * mask, mask, water)
        assert torch.allclose(out[0, 0, :4, :4], mu.reshape(()).expand(4, 4), atol=1e-5)

    def test_all_leads_agree_at_initialisation(self, model, sample):
        x, mask, water, doy = sample
        with torch.no_grad():
            out = model(x, mask, water, doy)
        assert torch.allclose(out[:, 0], out[:, 3], atol=1e-6)

    def test_departure_from_persistence_is_hard_capped(self, sample, build):
        """Even a pathological head cannot walk further than `delta_scale` sigmas away.

        This is what stops the model being *worse* than doing nothing on a lake it has
        never seen -- the failure mode of both earlier architectures.
        """
        x, mask, water, doy = sample
        m = build(input_days=14, n_leads=4, delta_scale=0.5).eval()
        with torch.no_grad():
            for p in m.head_coarse.parameters():
                p.fill_(50.0)
            for p in m.head_mid.parameters():
                p.fill_(-50.0)
            out = m(x, mask, water, doy)
            _, sigma, _ = masked_moments(x, mask, water)
        departure = (out - persistence(x, mask)).abs().amax() / sigma.amax()
        assert departure <= 0.5 + 1e-4


class TestSmoothness:
    def test_the_predicted_departure_carries_no_pixel_scale_structure(self, sample, build):
        """The delta is decoded at 1/8 and 1/4 resolution, so it cannot be speckle.

        Measured on the trained models, both predecessors emit fields 1.8-2.2x rougher
        than the observed composites. Here roughness is bounded by construction.
        """
        x, mask, water, doy = sample
        torch.manual_seed(1)
        m = build(input_days=14, n_leads=4)
        with torch.no_grad():
            # Randomise the heads so the delta is non-trivial.
            for p in list(m.head_coarse.parameters()) + list(m.head_mid.parameters()):
                p.normal_(0, 1.0)
            out = m(x, mask, water, doy)
        delta = (out - persistence(x, mask))[0, 0]
        # Neighbour difference relative to the field's own spread; a bilinearly upsampled
        # field is far below the ~1.0 that pure pixel noise produces.
        rough = (delta[1:, 1:] - delta[:-1, :-1]).std() / delta.std().clamp(min=1e-6)
        noise = torch.randn(48, 48)
        noise_rough = (noise[1:, 1:] - noise[:-1, :-1]).std() / noise.std()
        # Measured: ~0.50 for the architecture against ~1.43 for full-resolution noise.
        # The bar is set against that noise floor rather than at an absolute number, so
        # the test states the guarantee ("cannot be speckle") rather than a magic value.
        assert rough < 0.5 * noise_rough


class TestMaskedMoments:
    def test_ignores_unobserved_pixels(self):
        x = torch.zeros(1, 4, 8, 8)
        mask = torch.zeros(1, 4, 8, 8)
        x[0, 0, :4, :4] = 5.0
        mask[0, 0, :4, :4] = 1.0
        water = torch.ones(1, 1, 8, 8)
        mean, std, valid = masked_moments(x, mask, water)
        assert valid.item() == 0.0, "16 pixels is below the trust threshold"
        assert mean.item() == 0.0 and std.item() == 1.0, "falls back to a no-op"

    def test_uses_window_statistics_when_there_is_enough_data(self):
        torch.manual_seed(0)
        x = torch.randn(1, 4, 32, 32) * 2.0 + 7.0
        mask = torch.ones(1, 4, 32, 32)
        water = torch.ones(1, 1, 32, 32)
        mean, std, valid = masked_moments(x, mask, water)
        assert valid.item() == 1.0
        assert mean.item() == pytest.approx(7.0, abs=0.1)
        assert std.item() == pytest.approx(2.0, abs=0.1)

    def test_a_flat_window_cannot_divide_by_almost_zero(self):
        x = torch.full((1, 4, 32, 32), 3.0)
        mask = torch.ones(1, 4, 32, 32)
        water = torch.ones(1, 1, 32, 32)
        _, std, _ = masked_moments(x, mask, water)
        assert std.item() == pytest.approx(MIN_SIGMA)


class TestLastObserved:
    def test_walks_back_to_the_most_recent_observation_and_reports_its_age(self):
        x = torch.zeros(1, 5, 2, 2)
        mask = torch.zeros(1, 5, 2, 2)
        x[0, 2] = 4.0
        mask[0, 2] = 1.0  # observed two steps before the anchor
        last, age, has = last_observed_with_age(x, mask)
        assert torch.allclose(last, torch.full((1, 2, 2), 4.0))
        assert torch.allclose(age, torch.full((1, 2, 2), 2.0 / 4.0))
        assert torch.allclose(has, torch.ones(1, 2, 2))

    def test_reports_no_anchor_when_a_pixel_was_never_seen(self):
        x = torch.zeros(1, 5, 2, 2)
        mask = torch.zeros(1, 5, 2, 2)
        _, _, has = last_observed_with_age(x, mask)
        assert torch.allclose(has, torch.zeros(1, 2, 2))


@pytest.mark.parametrize("build", ARCHITECTURES)
def test_never_predicts_over_land(build):
    torch.manual_seed(0)
    x = torch.randn(1, 14, 32, 32)
    mask = torch.ones(1, 14, 32, 32)
    water = torch.zeros(1, 1, 32, 32)
    water[..., :16, :16] = 1.0
    m = build(input_days=14, n_leads=4).eval()
    with torch.no_grad():
        out = m(x, mask, water, torch.tensor([100.0]))
    assert out[..., 16:, :].abs().max() == 0.0
    assert out[..., :, 16:].abs().max() == 0.0


def test_is_smaller_than_both_predecessors():
    """Three training lakes cannot constrain a large model; capacity is the enemy here."""
    from habforecast.models.convlstm import build_model

    n = {
        name: sum(p.numel() for p in build_model(name, input_days=14, n_leads=4, **kw).parameters())
        for name, kw in (
            ("anomaly", {"base": 24}),
            ("convlstm", {"hidden": 48, "layers": 2}),
            ("unet", {"base": 32}),
        )
    }
    assert n["anomaly"] < n["convlstm"] < n["unet"]


class TestCoarseGrids:
    """A lake on a coarse grid is a few pixels across, and the trunk must survive that.

    At 4 km a 12 km reservoir is 7 px. Three halvings of 7 is a 0x0 feature map, and
    `avg_pool2d` raises rather than returning an empty tensor -- so before the guard the
    model could not be evaluated at all on the coarse rung of the resolution ladder, which
    is exactly where the multi-resolution comparison needed it.
    """

    @pytest.mark.parametrize("size", [34, 24, 13, 7, 3, 1])
    @pytest.mark.parametrize("arch", ARCHITECTURES)
    def test_runs_on_a_grid_too_small_to_downsample_three_times(self, arch, size):
        model = arch().eval()
        x = torch.rand(2, 14, size, size)
        mask = torch.ones_like(x)
        water = torch.ones(2, 1, size, size)
        with torch.no_grad():
            out = model(x, mask, water, torch.tensor([180.0, 40.0]))
        assert out.shape == (2, 4, size, size)
        assert torch.isfinite(out).all()

    @pytest.mark.parametrize("arch", ARCHITECTURES)
    def test_a_large_grid_is_unaffected_by_the_guard(self, arch):
        """The guard must be inert above ~16 px, or it would silently change the
        architecture the existing checkpoints were trained with."""
        model = arch().eval()
        x = torch.rand(1, 14, 64, 64)
        mask = torch.ones_like(x)
        water = torch.ones(1, 1, 64, 64)
        with torch.no_grad():
            out = model(x, mask, water, torch.tensor([180.0]))
        assert out.shape == (1, 4, 64, 64)
        assert torch.isfinite(out).all()
