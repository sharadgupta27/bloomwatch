"""The seasonal climatology as a model input, and the temporal split that changes what it
means.

Three contracts are pinned here, because each is a way the accuracy gain could be a leak:

1. The climatology a sample sees never includes the sample's own year -- the window around
   a target date otherwise overlaps the days being forecast.
2. Under the temporal split, a held-out year's climatology and normalisation come from the
   training years only, and its input windows cannot reach into them.
3. With no climatology available the `use_clim` model is exactly the persistence-anchored
   model it extends, so a lake with no history is forecast the way it always was.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pytest
import torch

from habforecast.features.dataset import LakeSeries, Normalisation, SequenceDataset
from habforecast.models.anomaly import AnomalyForecaster
from habforecast.models.baselines import ClimatologyBaseline
from habforecast.models.lit import model_inputs, predict_tta

H, W = 24, 32


def _reference_climatology(s: LakeSeries, doy: int, exclude_year: int, halfwidth=15):
    """The obvious, slow nanmean over the window -- what the fast path must reproduce."""
    ts = np.array([str(d) for d in s.days])
    doys = np.array([date.fromisoformat(t).timetuple().tm_yday for t in ts])
    years = np.array([int(t[:4]) for t in ts])
    delta = np.abs(doys - doy)
    delta = np.minimum(delta, 366 - delta)
    sel = (delta <= halfwidth) & (years != exclude_year)
    with np.errstate(all="ignore"):
        return np.nanmean(s.values[sel].astype(np.float32), axis=0)


def _series(lake_id="lake", years=(2019, 2020, 2021), seed=0, level=1.0) -> LakeSeries:
    """A synthetic lake with a smooth seasonal cycle plus a per-year offset."""
    rng = np.random.default_rng(seed)
    days, frames = [], []
    for y in years:
        offset = 0.2 * (y - years[0])  # each year runs a little greener
        for d in range(0, 365, 3):  # an observation every third day
            day = date(y, 1, 1) + timedelta(days=d)
            season = 0.5 * np.sin(2 * np.pi * d / 365.0)
            frame = level + offset + season + rng.normal(0, 0.05, (H, W))
            frame[rng.random((H, W)) < 0.3] = np.nan  # cloud
            days.append(np.datetime64(day, "D"))
            frames.append(frame.astype(np.float16))
    water = np.ones((H, W), dtype=bool)
    water[:4, :] = False  # a strip of land
    values = np.stack(frames)
    values[:, ~water] = np.nan
    return LakeSeries(lake_id, "train", np.array(days), values, water, (H, W))


class TestLakeSeriesClimatology:
    def test_excludes_the_named_year(self):
        s = _series()
        # Years 2019/2020/2021 sit at levels 1.0/1.2/1.4, so leaving out the greenest
        # year pulls the mean down by ~0.2 relative to leaving out the palest.
        c_excl_2021 = s.climatology(doy=10, exclude_year=2021)
        c_excl_2019 = s.climatology(doy=10, exclude_year=2019)
        assert c_excl_2021 is not None and c_excl_2019 is not None
        gap = np.nanmean(c_excl_2019[s.water]) - np.nanmean(c_excl_2021[s.water])
        assert gap == pytest.approx(0.2, abs=0.03)
        for year, frame in ((2021, c_excl_2021), (2019, c_excl_2019)):
            ref = _reference_climatology(s, 10, year)
            ok = np.isfinite(ref)
            np.testing.assert_allclose(frame[ok], ref[ok], atol=2e-3)

    def test_nan_over_land_and_none_when_no_other_year_covers_the_window(self):
        s = _series(years=(2020,))
        assert s.climatology(doy=100, exclude_year=2020) is None
        c = s.climatology(doy=100, exclude_year=1999)
        assert c is not None
        assert np.isnan(c[~s.water]).all()

    def test_crop_box_matches_the_full_grid(self):
        s = _series()
        full = s.climatology(doy=200, exclude_year=2020)
        box = (slice(5, 15), slice(8, 20))
        crop = s.climatology(doy=200, exclude_year=2020, box=box)
        np.testing.assert_allclose(crop, full[box], equal_nan=True)

    def test_clim_source_redirects_to_another_series(self):
        full = _series(years=(2019, 2020, 2021))
        train = full.subset_years([2019, 2020], "train")
        held = full.subset_years([2021], "test")
        held.clim_source = train
        c = held.climatology(doy=10, exclude_year=2021)
        # Drawn from 2019+2020, not from 2021's own (greener) frames.
        ref = _reference_climatology(train, 10, 2021)
        ok = np.isfinite(ref)
        np.testing.assert_allclose(c[ok], ref[ok], atol=2e-3)
        own = _reference_climatology(held, 10, 1999)
        assert np.nanmean(c[held.water]) < np.nanmean(own[held.water]) - 0.1

    def test_baseline_uses_the_same_computation(self):
        s = _series()
        norm = Normalisation(0.0, 1.0)
        base = ClimatologyBaseline(n_leads=1).fit([s], norm)
        frame = base._climatology(0, 200, 2020)
        ref = s.climatology(doy=200, exclude_year=2020)
        ok = np.isfinite(ref)
        np.testing.assert_allclose(frame[ok], ref[ok], atol=1e-6)


class TestDatasetClimatologyChannel:
    def _dataset(self, series, **kw):
        return SequenceDataset(
            series,
            Normalisation(1.0, 0.5),
            input_days=6,
            lead_times=(1, 3),
            crop=None,
            climatology=True,
            **kw,
        )

    def test_channel_shapes_and_units(self):
        s = _series()
        ds = self._dataset([s])
        item = ds[0]
        assert tuple(item["clim"].shape) == (2, H, W)
        assert tuple(item["clim_mask"].shape) == (2, H, W)
        # Masked out over land, present over water where other years observed it.
        assert float(item["clim_mask"][:, :4, :].sum()) == 0.0
        assert float(item["clim_mask"][:, 4:, :].mean()) > 0.9
        # Standardised with the dataset's constants: (raw - 1.0) / 0.5.
        raw = s.climatology(
            (ds.samples[0].anchor + timedelta(days=1)).timetuple().tm_yday,
            ds.samples[0].anchor.year,
        )
        ok = np.isfinite(raw)
        np.testing.assert_allclose(item["clim"][0].numpy()[ok], (raw[ok] - 1.0) / 0.5, atol=1e-5)

    def test_own_year_is_never_in_the_climatology(self):
        """A lake with a single year has no climatology at all -- if it did, the only
        place it could have come from is the year being forecast."""
        s = _series(years=(2020,))
        item = self._dataset([s])[0]
        assert float(item["clim_mask"].sum()) == 0.0
        assert float(item["clim"].abs().sum()) == 0.0

    def test_dropout_withholds_the_whole_channel_only_when_augmenting(self):
        s = _series()
        always = self._dataset([s], augment=True, clim_dropout=1.0)
        assert float(always[0]["clim_mask"].sum()) == 0.0
        never = self._dataset([s], augment=False, clim_dropout=1.0)
        assert float(never[0]["clim_mask"].sum()) > 0.0

    def test_off_by_default_costs_nothing(self):
        s = _series()
        ds = SequenceDataset(s and [s], Normalisation(1.0, 0.5), input_days=6, lead_times=(1, 3))
        item = ds[0]
        assert float(item["clim_mask"].sum()) == 0.0


class TestTemporalSplit:
    def test_subset_years_keeps_the_full_water_mask(self):
        s = _series()
        sub = s.subset_years([2021], "test")
        assert sub.n_days == sum(
            1 for d in s.days if d.astype("datetime64[Y]").astype(int) + 1970 == 2021
        )
        np.testing.assert_array_equal(sub.water, s.water)
        assert sub.shape == s.shape

    def test_held_out_windows_cannot_reach_training_years(self):
        """Every input frame and every target of a held-out sample lies in its own years:
        the slice simply does not contain any other day."""
        s = _series()
        held = s.subset_years([2021], "test")
        ds = SequenceDataset(
            [held], Normalisation(1.0, 0.5), input_days=14, lead_times=(1, 3, 7), crop=None
        )
        for spec in ds.samples:
            for row in ds._window_rows(held, spec.anchor) + ds._target_rows(held, spec.anchor):
                if row is not None:
                    assert int(str(held.days[row])[:4]) == 2021

    def test_datamodule_temporal_split_is_disjoint_and_forward(self, tmp_path):
        """End to end through the DataModule with a tiny on-disk registry and cubes."""
        import json

        import xarray as xr

        from habforecast.features.datamodule import HABDataModule

        cubes = tmp_path / "cubes"
        cubes.mkdir()
        registry = {"type": "FeatureCollection", "features": []}
        for lake_id, split in (("a", "train"), ("b", "test")):
            s = _series(lake_id, years=(2019, 2020, 2021, 2022), seed=hash(lake_id) % 100)
            ds = xr.Dataset(
                {"log_chl": (("time", "y", "x"), s.values.astype(np.float32))},
                coords={"time": s.days.astype("datetime64[ns]")},
            )
            ds.to_zarr(cubes / f"{lake_id}.zarr", mode="w", consolidated=True)
            registry["features"].append(
                {
                    "type": "Feature",
                    "properties": {
                        "lake_id": lake_id,
                        "name": lake_id,
                        "split": split,
                        "country": "XX",
                        "trophic_state": "eutrophic",
                        "bloom_season": [1, 12],
                        "utm_epsg": 32633,
                    },
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[[17, 46], [18, 46], [18, 47], [17, 47], [17, 46]]],
                    },
                }
            )
        aoi = tmp_path / "lakes.geojson"
        aoi.write_text(json.dumps(registry))

        dm = HABDataModule(
            aoi,
            cubes,
            input_days=6,
            lead_times=(1, 3),
            crop=None,
            climatology=True,
            split_mode="temporal",
            val_years=(2021,),
            test_years=(2022,),
        )
        dm.setup()
        summary = dm.summary()
        assert summary["split_mode"] == "temporal"
        # Every lake is on every side; the registry's lake split is ignored.
        assert summary["train"]["lakes"] == ["a", "b"]
        assert summary["test"]["lakes"] == ["a", "b"]
        years = {
            split: {int(str(d)[:4]) for s in dm._series[split] for d in s.days}
            for split in ("train", "val", "test")
        }
        assert years["train"] == {2019, 2020}
        assert years["val"] == {2021}
        assert years["test"] == {2022}
        # Held-out climatology comes from the training slice of the same lake.
        for split in ("val", "test"):
            for s in dm._series[split]:
                assert s.clim_source is not None and s.clim_source.split == "train"
                assert s.clim_source.lake_id == s.lake_id
        # Normalisation was fitted before the cut: it cannot know the later years ran
        # greener, so their standardised mean sits above zero.
        item = dm._datasets["test"][0]
        assert float(item["y"][item["y_mask"] > 0].mean()) > 0.0

    def test_rejects_overlapping_years(self, tmp_path):
        from habforecast.features.datamodule import HABDataModule

        with pytest.raises(ValueError, match="overlap"):
            HABDataModule(
                "data/aoi/lakes.geojson",
                tmp_path,
                split_mode="temporal",
                val_years=(2023,),
                test_years=(2023, 2024),
            )


class TestClimatologyModel:
    @pytest.fixture
    def batch(self):
        g = torch.Generator().manual_seed(0)
        b, t, n_leads = 2, 6, 3
        x = torch.randn(b, t, H, W, generator=g)
        x_mask = (torch.rand(b, t, H, W, generator=g) < 0.5).float()
        x = x * x_mask
        water = torch.ones(b, 1, H, W)
        water[:, :, :4, :] = 0
        clim = torch.randn(b, n_leads, H, W, generator=g)
        clim_mask = torch.ones(b, n_leads, H, W)
        return {
            "x": x,
            "x_mask": x_mask,
            "water": water,
            "doy": torch.tensor([150.0, 220.0]),
            "clim": clim,
            "clim_mask": clim_mask,
        }

    def test_without_a_climatology_it_is_the_plain_model(self, batch):
        torch.manual_seed(0)
        plain = AnomalyForecaster(input_days=6, n_leads=3, base=8)
        torch.manual_seed(0)
        clim = AnomalyForecaster(input_days=6, n_leads=3, base=8, use_clim=True)
        # Share every parameter the two have in common; the extra ones stay at init. The
        # stem is wider by the climatology channels, so its shared columns are copied and
        # the new ones zeroed.
        with torch.no_grad():
            src = plain.state_dict()
            for name, param in clim.state_dict().items():
                if name == "stem.0.weight":
                    n = src[name].shape[1]
                    param.zero_()
                    param[:, :n] = src[name]
                elif name in src:
                    param.copy_(src[name])
        inputs = model_inputs(clim, {**batch, "clim": None, "clim_mask": None})
        out_clim = clim(**inputs)
        out_plain = plain(**model_inputs(plain, batch))
        torch.testing.assert_close(out_clim, out_plain, atol=1e-5, rtol=1e-5)

    def test_masked_climatology_leaves_persistence_untouched(self, batch):
        model = AnomalyForecaster(input_days=6, n_leads=3, base=8, use_clim=True).eval()
        batch["clim_mask"] = torch.zeros_like(batch["clim_mask"])
        with torch.no_grad():
            out = model(**model_inputs(model, batch))
        from habforecast.models.convlstm import _last_observed

        last, has = _last_observed(batch["x"], batch["x_mask"])
        seen = (has > 0) & (batch["water"][:, 0] > 0)
        for lead in range(3):
            torch.testing.assert_close(out[:, lead][seen], last[seen], atol=1e-5, rtol=1e-5)

    def test_gate_starts_near_persistence_and_moves_toward_climatology(self, batch):
        model = AnomalyForecaster(input_days=6, n_leads=3, base=8, use_clim=True).eval()
        from habforecast.models.convlstm import _last_observed

        last, has = _last_observed(batch["x"], batch["x_mask"])
        seen = (has > 0) & (batch["water"][:, 0] > 0)
        with torch.no_grad():
            out = model(**model_inputs(model, batch))
        # sigmoid(-2) = 0.119: the anchor is 88% last observation, 12% climatology.
        g = float(torch.sigmoid(torch.tensor(-2.0)))
        expect = (1.0 - g) * last + g * batch["clim"][:, 0]
        torch.testing.assert_close(out[:, 0][seen], expect[seen], atol=1e-4, rtol=1e-4)
        # Opening the gate fully makes the anchor the climatology itself.
        with torch.no_grad():
            model.head_gate.bias.fill_(30.0)
            out = model(**model_inputs(model, batch))
        torch.testing.assert_close(out[:, 0][seen], batch["clim"][:, 0][seen], atol=1e-4, rtol=1e-4)

    def test_gate_never_reaches_over_land(self, batch):
        model = AnomalyForecaster(input_days=6, n_leads=3, base=8, use_clim=True).eval()
        with torch.no_grad():
            model.head_gate.bias.fill_(30.0)
            out = model(**model_inputs(model, batch))
        assert float(out[:, :, :4, :].abs().sum()) == 0.0

    def test_checkpoint_config_round_trip(self):
        from habforecast.models.convlstm import build_model

        net = build_model("anomaly", input_days=6, n_leads=3, base=8, use_clim=True)
        assert net.use_clim
        state = net.state_dict()
        again = build_model("anomaly", input_days=6, n_leads=3, base=8, use_clim=True)
        again.load_state_dict(state)


class TestTTA:
    def test_flip_averaging_is_exact_for_an_equivariant_model(self):
        """Persistence is flip-equivariant, so TTA on the untrained model changes
        nothing -- and a bug that flipped one input but not another would show here."""
        model = AnomalyForecaster(input_days=6, n_leads=2, base=8, use_clim=True).eval()
        g = torch.Generator().manual_seed(1)
        b = {
            "x": torch.randn(1, 6, H, W, generator=g),
            "x_mask": (torch.rand(1, 6, H, W, generator=g) < 0.5).float(),
            "water": torch.ones(1, 1, H, W),
            "doy": torch.tensor([100.0]),
            "clim": torch.randn(1, 2, H, W, generator=g),
            "clim_mask": torch.ones(1, 2, H, W),
        }
        with torch.no_grad():
            plain = model(**model_inputs(model, b))
            tta = predict_tta(model, model_inputs(model, b))
        torch.testing.assert_close(plain, tta, atol=1e-5, rtol=1e-5)

    def test_disabled_is_a_single_call(self):
        model = AnomalyForecaster(input_days=6, n_leads=2, base=8).eval()
        calls = []
        real = model.forward
        model.forward = lambda *a, **k: (calls.append(1), real(*a, **k))[1]
        b = {
            "x": torch.zeros(1, 6, H, W),
            "x_mask": torch.ones(1, 6, H, W),
            "water": torch.ones(1, 1, H, W),
            "doy": torch.tensor([1.0]),
        }
        with torch.no_grad():
            predict_tta(model, model_inputs(model, b), flips=False)
        assert len(calls) == 1
