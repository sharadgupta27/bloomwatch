"""ERA5 meteorological forcing: the reduction, the request planner, and the leakage guard.

The contract that matters most is the last one. Reanalysis makes the weather at the
*target* date trivially available, and a forecast cannot have it. Every score in this
repository would improve if that leaked, so the default path is tested to never read a
day later than the anchor, and the perfect-prognosis path is tested to be the only way to
opt into it.
"""

from __future__ import annotations

import io
import zipfile
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest
import torch
import xarray as xr

from habforecast.data import era5
from habforecast.features.dataset import LakeSeries, Normalisation, SequenceDataset
from habforecast.models.anomaly import AnomalyForecaster, MetEncoder
from habforecast.models.lit import model_inputs, predict_tta

H, W = 16, 20


def _hourly(days: int = 2, step_hours: int = 1, u=None) -> xr.Dataset:
    n = days * (24 // step_hours)
    t = pd.date_range("2024-08-15", periods=n, freq=f"{step_hours}h")
    shape = (n, 2, 2)
    u = np.zeros(shape, np.float32) if u is None else u
    return xr.Dataset(
        {
            "u10": (("valid_time", "latitude", "longitude"), u),
            "v10": (("valid_time", "latitude", "longitude"), np.zeros(shape, np.float32)),
            "t2m": (("valid_time", "latitude", "longitude"), np.full(shape, 293.15, np.float32)),
            "ssrd": (("valid_time", "latitude", "longitude"), np.full(shape, 1e6, np.float32)),
            "tp": (("valid_time", "latitude", "longitude"), np.full(shape, 0.001, np.float32)),
        },
        coords={"valid_time": t, "latitude": [41.5, 41.75], "longitude": [-83.0, -82.75]},
    )


def _flat(da) -> np.ndarray:
    return da.mean(dim=[d for d in da.dims if d != "time"]).values


class TestDailyReduction:
    def test_wind_speed_is_the_mean_of_speeds_not_the_speed_of_the_mean(self):
        """A day that blows hard east then hard west averages to a near-zero *vector* and
        is a gale. Recording it as calm would invert the one signal this forcing is for:
        calm water is when a bloom floats up and the satellite sees it."""
        u = np.zeros((48, 2, 2), np.float32)
        u[:12] = 10.0
        u[12:24] = -10.0
        u[24:] = 1.0
        d = era5.to_daily(_hourly(2, 1, u))
        assert _flat(d["wind_speed"]) == pytest.approx([10.0, 1.0], abs=1e-4)
        vector_mean = float(np.hypot(_flat(d["wind_u"])[0], _flat(d["wind_v"])[0]))
        assert vector_mean == pytest.approx(0.0, abs=1e-4)  # the trap

    def test_units_are_converted(self):
        d = era5.to_daily(_hourly(2))
        # float32 holds 293.15 to ~6 significant figures, so the subtraction lands a
        # hair off 20; the tolerance is precision, not a unit question.
        assert _flat(d["t2m"]) == pytest.approx([20.0, 20.0], abs=1e-3)  # K -> degC
        assert _flat(d["ssrd"]) == pytest.approx([24.0, 24.0])  # 24 x 1 MJ
        assert _flat(d["tp"]) == pytest.approx([24.0, 24.0])  # 24 x 1 mm

    def test_accumulations_are_rescaled_by_the_sampling_step(self):
        """A 3-hourly series holds 8 hourly accumulations, not 24. Without the rescale a
        subsampled day would report a third of its real radiation."""
        d = era5.to_daily(_hourly(2, step_hours=3))
        assert _flat(d["ssrd"]) == pytest.approx([24.0, 24.0], rel=1e-3)
        assert _flat(d["tp"]) == pytest.approx([24.0, 24.0], rel=1e-3)

    def test_missing_variables_warn_rather_than_vanish(self, caplog):
        ds = _hourly(1).drop_vars(["ssrd", "tp"])
        with caplog.at_level("WARNING"):
            d = era5.to_daily(ds)
        assert "ssrd" in caplog.text and "tp" in caplog.text
        assert "wind_speed" in d.data_vars


class TestOpenRaw:
    def test_reads_a_zip_of_netcdfs_and_merges_them(self, tmp_path):
        """A multi-variable CDS request returns a ZIP split by stepType even when
        `download_format: unarchived` is asked for, and it still has a .nc name."""
        ds = _hourly(1)
        path = tmp_path / "delivery.nc"
        with zipfile.ZipFile(path, "w") as zf:
            for name, keep in (
                ("data_stream-oper_stepType-instant.nc", ["u10", "v10", "t2m"]),
                ("data_stream-oper_stepType-accum.nc", ["ssrd", "tp"]),
            ):
                buf = io.BytesIO()
                ds[keep].to_netcdf(buf, engine="h5netcdf")
                zf.writestr(name, buf.getvalue())
        merged = era5.open_raw(path)
        assert set(merged.data_vars) == {"u10", "v10", "t2m", "ssrd", "tp"}

    def test_reads_a_plain_netcdf(self, tmp_path):
        path = tmp_path / "plain.nc"
        _hourly(1).to_netcdf(path, engine="h5netcdf")
        assert "u10" in era5.open_raw(path).data_vars


class TestRequestPlanner:
    def _aoi(self, months=(6, 7, 8)):
        from habforecast.aoi import LakeAOI

        return LakeAOI(
            lake_id="x",
            name="X",
            country="",
            split="train",
            trophic_state="eutrophic",
            bloom_season=(min(months), max(months)),
            utm_epsg=32617,
            bbox=(-83.5, 41.3, -82.5, 42.0),
        )

    def test_falls_back_to_the_bloom_season_with_a_lookback_month(self):
        got = era5.needed_months(self._aoi((6, 7, 8)), None)
        assert got == {5, 6, 7, 8}

    def test_wraps_the_lookback_at_the_year_boundary(self):
        aoi = self._aoi()
        object.__setattr__(aoi, "bloom_season", (1, 2))
        assert era5.needed_months(aoi, None) == {12, 1, 2}

    def test_chunks_stay_inside_the_cost_cap(self):
        months = list(range(1, 13))
        chunks = era5.chunk_months(months, n_vars=5, n_times=8)
        assert [m for c in chunks for m in c] == months  # nothing dropped or reordered
        import calendar

        for c in chunks:
            days = sum(calendar.monthrange(2024, m)[1] for m in c)
            assert 5 * days * 8 <= era5.MAX_FIELDS_PER_REQUEST

    def test_a_full_year_hourly_would_not_fit(self):
        """The measurement behind SAMPLE_HOURS: hourly cannot be requested a year at a
        time, which is what the live ingest discovered as a 403."""
        assert era5.MAX_FIELDS_PER_REQUEST < 5 * 365 * 24
        assert 5 * 365 * len(range(0, 24, era5.SAMPLE_HOURS)) > era5.MAX_FIELDS_PER_REQUEST
        assert 5 * 184 * len(range(0, 24, era5.SAMPLE_HOURS)) <= era5.MAX_FIELDS_PER_REQUEST


class TestSeriesMatrix:
    def _frame(self):
        days = pd.date_range("2024-08-01", periods=20).date
        return pd.DataFrame(
            {"wind_speed": np.arange(20, dtype=np.float32), "t2m": np.full(20, 20.0, np.float32)},
            index=days,
        )

    def test_standardises_and_masks_days_outside_the_store(self):
        frame = self._frame()
        stats = {"wind_speed": (10.0, 2.0), "t2m": (20.0, 1.0)}
        days = [date(2024, 7, 30), date(2024, 8, 1), date(2024, 8, 2)]
        vals, mask = era5.series_matrix(frame, days, stats, ("wind_speed", "t2m"))
        assert mask[0, 0] == 0.0 and vals[0, 0] == 0.0  # before the store starts
        assert mask[0, 1] == 1.0
        assert vals[0, 1] == pytest.approx((0.0 - 10.0) / 2.0)
        assert vals[0, 2] == pytest.approx((1.0 - 10.0) / 2.0)

    def test_normalisation_is_fitted_per_variable(self):
        stats = era5.fit_normalisation([self._frame()])
        assert stats["wind_speed"][0] == pytest.approx(9.5)
        assert stats["t2m"][1] == 1.0  # zero spread is guarded, never a divide by zero


class TestMetEncoder:
    def test_starts_as_no_contribution(self):
        enc = MetEncoder(n_vars=3, out_channels=8)
        out = enc(torch.randn(4, 3, 14), torch.ones(4, 3, 14))
        assert torch.allclose(out, torch.zeros_like(out))

    def test_summary_ignores_masked_days_and_reports_coverage(self):
        enc = MetEncoder(n_vars=1, out_channels=4)
        met = torch.tensor([[[1.0, 2.0, 99.0, 3.0]]])
        mask = torch.tensor([[[1.0, 1.0, 0.0, 1.0]]])
        window, recent, trend, coverage = enc.summarise(met, mask)[0].tolist()
        assert window == pytest.approx(2.0)  # 99 excluded
        assert coverage == pytest.approx(0.75)
        assert recent == pytest.approx(2.5)  # last 3 slots, one masked: (2+3)/2
        assert trend == pytest.approx(recent - window)

    def test_an_empty_window_is_no_information_not_calm(self):
        enc = MetEncoder(n_vars=2, out_channels=4)
        feats = enc.summarise(torch.zeros(1, 2, 14), torch.zeros(1, 2, 14))[0]
        assert torch.allclose(feats, torch.zeros_like(feats))


class TestModelIntegration:
    def _batch(self, n_met=3, leads=2):
        g = torch.Generator().manual_seed(0)
        return {
            "x": torch.randn(2, 6, H, W, generator=g),
            "x_mask": (torch.rand(2, 6, H, W, generator=g) < 0.5).float(),
            "water": torch.ones(2, 1, H, W),
            "doy": torch.tensor([150.0, 200.0]),
            "met": torch.randn(2, n_met, 6, generator=g),
            "met_mask": torch.ones(2, n_met, 6),
            "met_future": torch.randn(2, n_met, leads, generator=g),
            "met_future_mask": torch.ones(2, n_met, leads),
        }

    def test_forcing_starts_as_exactly_the_model_it_extends(self):
        torch.manual_seed(0)
        plain = AnomalyForecaster(input_days=6, n_leads=2, base=8).eval()
        torch.manual_seed(0)
        met = AnomalyForecaster(input_days=6, n_leads=2, base=8, n_met=3).eval()
        with torch.no_grad():
            n = plain.stem[0].weight.shape[1]
            met.stem[0].weight.zero_()
            met.stem[0].weight[:, :n] = plain.stem[0].weight
            met.stem[0].bias.copy_(plain.stem[0].bias)
        b = self._batch()
        with torch.no_grad():
            a = plain(**model_inputs(plain, b))
            c = met(**model_inputs(met, b))
        torch.testing.assert_close(a, c, atol=1e-5, rtol=1e-5)

    def test_absent_forcing_is_handled(self):
        model = AnomalyForecaster(input_days=6, n_leads=2, base=8, n_met=3).eval()
        b = self._batch()
        b["met"], b["met_mask"] = None, None
        with torch.no_grad():
            out = model(**model_inputs(model, b))
        assert torch.isfinite(out).all()

    def test_model_inputs_withholds_future_unless_asked(self):
        past_only = AnomalyForecaster(input_days=6, n_leads=2, base=8, n_met=3)
        assert "met_future" not in model_inputs(past_only, self._batch())
        pp = AnomalyForecaster(input_days=6, n_leads=2, base=8, n_met=3, met_future=True)
        assert "met_future" in model_inputs(pp, self._batch())
        assert pp.met_future is True

    def test_tta_does_not_reverse_the_weather_series(self):
        """The met tensors are (B, V, T): rank 3, like a spatial stack, with time last.
        A flip driven by rank alone would run the fortnight backwards."""
        model = AnomalyForecaster(input_days=6, n_leads=2, base=8, n_met=3).eval()
        b = self._batch()
        seen = []
        real = model.met_encoder.forward

        def spy(met, met_mask, future=None, future_mask=None):
            seen.append(met.clone())
            return real(met, met_mask, future, future_mask)

        model.met_encoder.forward = spy
        with torch.no_grad():
            predict_tta(model, model_inputs(model, b))
        assert len(seen) == 4
        for got in seen[1:]:
            torch.testing.assert_close(got, seen[0])


class TestNoTemporalLeakage:
    def _series(self):
        days = np.array(
            [np.datetime64(date(2024, 6, 1) + timedelta(days=i), "D") for i in range(60)]
        )
        rng = np.random.default_rng(0)
        values = rng.normal(1.0, 0.2, (60, H, W)).astype(np.float16)
        met_days = pd.date_range("2024-05-01", periods=120).date
        met = pd.DataFrame(
            # A ramp, so any day leaking in is identifiable by its value alone.
            {"wind_speed": np.arange(120, dtype=np.float32)},
            index=met_days,
        )
        return LakeSeries(
            lake_id="lake",
            split="train",
            days=days,
            values=values,
            water=np.ones((H, W), bool),
            shape=(H, W),
            met=met,
            met_stats={"wind_speed": (0.0, 1.0)},
        )

    def _dataset(self, **kw):
        return SequenceDataset(
            [self._series()],
            Normalisation(1.0, 0.2),
            input_days=6,
            lead_times=(1, 3),
            crop=None,
            met_keys=("wind_speed",),
            **kw,
        )

    def test_the_window_stops_at_the_anchor(self):
        ds = self._dataset()
        for idx in range(0, len(ds), 7):
            spec = ds.samples[idx]
            item = ds[idx]
            anchor_value = (spec.anchor - date(2024, 5, 1)).days
            got = item["met"][0].numpy()
            assert got.max() == pytest.approx(anchor_value), "window reached past the anchor"
            assert got[-1] == pytest.approx(anchor_value)

    def test_future_forcing_is_absent_by_default(self):
        item = self._dataset()[0]
        assert float(item["met_future_mask"].sum()) == 0.0
        assert float(item["met_future"].abs().sum()) == 0.0

    def test_perfect_prognosis_is_the_only_way_in_and_is_the_future(self):
        ds = self._dataset(met_future=True)
        spec = ds.samples[0]
        item = ds[0]
        anchor_value = (spec.anchor - date(2024, 5, 1)).days
        future = item["met_future"][0].numpy()
        assert float(item["met_future_mask"].sum()) == 2.0
        assert future.tolist() == [anchor_value + 1, anchor_value + 3]
        assert future.min() > item["met"][0].numpy().max()
