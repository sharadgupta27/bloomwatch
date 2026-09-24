"""Tests for the persistence and climatology baselines.

The leave-one-year-out test is a regression guard for a real methodological bug. The
first version of ClimatologyBaseline fitted the seasonal mean on the target lake's whole
record, including the exact days it was then scored against. The leaky baseline beat the
model at every lead time on the validation lake, which would have been reported as a
finding about the model rather than a bug in the harness. Leakage of this kind never
raises -- it just makes a baseline look good -- so it needs a test.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pytest
import torch

from habforecast.features.dataset import LakeSeries, Normalisation
from habforecast.models.baselines import ClimatologyBaseline, PersistenceBaseline

H, W = 8, 10


def make_series(years, doy=200, value_by_year=None, lake_id="lake"):
    """One frame per year at the same day-of-year, each with a distinct value."""
    days, vals = [], []
    for y in years:
        d = date(y, 1, 1) + timedelta(days=doy - 1)
        days.append(np.datetime64(d, "D"))
        v = value_by_year[y] if value_by_year else 1.0
        vals.append(np.full((H, W), v, dtype=np.float32))
    return LakeSeries(
        lake_id=lake_id,
        split="val",
        days=np.array(days),
        values=np.stack(vals).astype(np.float16),
        water=np.ones((H, W), dtype=bool),
        shape=(H, W),
    )


IDENTITY = Normalisation(mean=0.0, std=1.0)


class TestPersistence:
    def test_repeats_last_observation_across_leads(self):
        x = torch.zeros(1, 3, 4, 4)
        m = torch.zeros(1, 3, 4, 4)
        x[0, 2] = 2.5
        m[0, 2] = 1.0
        out = PersistenceBaseline(n_leads=3)(x, m, torch.ones(1, 1, 4, 4))
        assert out.shape == (1, 3, 4, 4)
        for lead in range(3):
            assert torch.allclose(out[0, lead], torch.full((4, 4), 2.5))

    def test_respects_the_water_mask(self):
        x = torch.ones(1, 2, 4, 4)
        m = torch.ones(1, 2, 4, 4)
        water = torch.zeros(1, 1, 4, 4)
        water[0, 0, :2, :2] = 1.0
        out = PersistenceBaseline(n_leads=2)(x, m, water)
        assert out[0, 0, 3, 3] == 0.0
        assert out[0, 0, 0, 0] == 1.0


class TestClimatologyLeakage:
    """The evaluated year must never contribute to its own climatology."""

    def test_excludes_the_evaluated_year(self):
        years = [2019, 2020, 2021]
        # 2020 is a wild outlier; excluding it must change the answer.
        series = make_series(years, value_by_year={2019: 1.0, 2020: 100.0, 2021: 1.0})
        clim = ClimatologyBaseline(n_leads=1).fit([series], IDENTITY)

        x = torch.zeros(1, 2, H, W)
        m = torch.zeros(1, 2, H, W)
        water = torch.ones(1, 1, H, W)
        doy = torch.tensor([200.0])
        lake = torch.tensor([0])

        excl = clim(x, m, water, doy, lake, torch.tensor([2020]))
        # Only 2019 and 2021 remain, both 1.0.
        assert float(excl[0, 0].mean()) == pytest.approx(1.0, abs=1e-4)

    def test_leaky_variant_differs(self):
        """Sanity check that the exclusion is actually doing something."""
        series = make_series([2019, 2020, 2021], value_by_year={2019: 1.0, 2020: 100.0, 2021: 1.0})
        clim = ClimatologyBaseline(n_leads=1).fit([series], IDENTITY)
        args = (
            torch.zeros(1, 2, H, W),
            torch.zeros(1, 2, H, W),
            torch.ones(1, 1, H, W),
            torch.tensor([200.0]),
            torch.tensor([0]),
        )

        honest = float(clim(*args, torch.tensor([2020]))[0, 0].mean())
        leaky = float(clim(*args, None)[0, 0].mean())  # year=None excludes nothing
        assert leaky > honest
        assert leaky == pytest.approx(34.0, abs=1e-2)  # (1 + 100 + 1) / 3

    def test_falls_back_when_no_other_year_covers_the_window(self):
        """A single-year record has no leave-one-out climatology; must not return zeros.

        Returning zeros would read as 'no bloom predicted', which is the dangerous
        direction to fail in.
        """
        series = make_series([2020], value_by_year={2020: 5.0})
        clim = ClimatologyBaseline(n_leads=1).fit([series], IDENTITY)
        out = clim(
            torch.zeros(1, 2, H, W),
            torch.zeros(1, 2, H, W),
            torch.ones(1, 1, H, W),
            torch.tensor([200.0]),
            torch.tensor([0]),
            torch.tensor([2020]),
        )
        assert float(out[0, 0].mean()) == pytest.approx(5.0, abs=1e-2)

    def test_output_is_finite_and_shaped(self):
        series = make_series([2019, 2020, 2021])
        clim = ClimatologyBaseline(n_leads=3).fit([series], IDENTITY)
        out = clim(
            torch.zeros(2, 2, H, W),
            torch.zeros(2, 2, H, W),
            torch.ones(2, 1, H, W),
            torch.tensor([200.0, 200.0]),
            torch.tensor([0, 0]),
            torch.tensor([2019, 2021]),
        )
        assert out.shape == (2, 3, H, W)
        assert torch.isfinite(out).all()

    def test_respects_water_mask(self):
        series = make_series([2019, 2020, 2021])
        clim = ClimatologyBaseline(n_leads=1).fit([series], IDENTITY)
        water = torch.zeros(1, 1, H, W)
        water[0, 0, :3, :3] = 1.0
        out = clim(
            torch.zeros(1, 2, H, W),
            torch.zeros(1, 2, H, W),
            water,
            torch.tensor([200.0]),
            torch.tensor([0]),
            torch.tensor([2020]),
        )
        assert float(out[0, 0, H - 1, W - 1]) == 0.0

    def test_doy_window_is_circular(self):
        """Day 5 and day 362 are 8 days apart, not 357."""
        series = make_series([2019, 2020], doy=362, value_by_year={2019: 7.0, 2020: 7.0})
        clim = ClimatologyBaseline(n_leads=1, doy_halfwidth=15).fit([series], IDENTITY)
        out = clim(
            torch.zeros(1, 2, H, W),
            torch.zeros(1, 2, H, W),
            torch.ones(1, 1, H, W),
            torch.tensor([5.0]),
            torch.tensor([0]),
            torch.tensor([2020]),
        )
        assert float(out[0, 0].mean()) == pytest.approx(7.0, abs=1e-2)
