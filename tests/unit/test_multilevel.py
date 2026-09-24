"""The global-registry additions: lake groups, the multilevel forecaster, per-trophic-class
calibration, the Open-Meteo forcing reduction, the gate's small-lake rule and the
memory-mapped loader."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from habforecast import calibration
from habforecast.groups import (
    CONTINENTS,
    FACTOR_NAMES,
    THERMAL_CLASSES,
    TROPHIC_CLASSES,
    LakeGroups,
    thermal_class,
)
from habforecast.models.anomaly import AnomalyForecaster

LEADS, THRS = (1, 3), (10.0, 20.0)


# --------------------------------------------------------------------------- groups --
def test_vocabularies_start_with_unknown_and_are_append_only():
    # Embeddings are stored by index: these positions are a checkpoint contract.
    assert TROPHIC_CLASSES[:5] == (
        "unknown",
        "oligotrophic",
        "mesotrophic",
        "eutrophic",
        "hypereutrophic",
    )
    assert THERMAL_CLASSES[:4] == ("unknown", "cold", "temperate", "warm")
    assert CONTINENTS[0] == "unknown"
    assert FACTOR_NAMES == ("trophic", "thermal", "continent")


def test_unknown_or_misspelt_levels_map_to_zero_never_a_guess():
    assert LakeGroups().ids() == [0, 0, 0]
    assert LakeGroups(trophic="Eutrophic", thermal="warm", continent="asia").ids() == [
        TROPHIC_CLASSES.index("eutrophic"),
        THERMAL_CLASSES.index("warm"),
        CONTINENTS.index("asia"),
    ]
    assert LakeGroups(trophic="eutropic").ids()[0] == 0


@pytest.mark.parametrize(
    ("t", "expected"),
    [(None, "unknown"), (float("nan"), "unknown"), (4.0, "cold"), (15.0, "temperate"), (25.0, "warm")],
)
def test_thermal_class_edges(t, expected):
    assert thermal_class(t) == expected


def test_global_registry_carries_groups_for_every_lake():
    from habforecast.aoi import AOIRegistry

    path = Path(__file__).resolve().parents[2] / "data" / "aoi" / "lakes_global.geojson"
    if not path.exists():
        pytest.skip("global registry not built")
    reg = AOIRegistry.from_geojson(path)
    for aoi in reg:
        ids = aoi.groups.ids()
        assert ids[0] > 0, f"{aoi.lake_id}: trophic state missing"
        assert ids[2] > 0, f"{aoi.lake_id}: continent missing"
    # Every trophic class the per-class calibration fits must have a validation lake.
    val_classes = {a.trophic_state for a in reg.split("val")}
    assert {"oligotrophic", "mesotrophic", "eutrophic", "hypereutrophic"} <= val_classes


# ----------------------------------------------------------------- multilevel model --
def _batch(b=3, t=14, h=24, w=24, seed=0):
    g = torch.Generator().manual_seed(seed)
    mask = (torch.rand(b, t, h, w, generator=g) > 0.5).float()
    return {
        "x": torch.randn(b, t, h, w, generator=g) * mask,
        "x_mask": mask,
        "water": torch.ones(b, 1, h, w),
        "doy": torch.full((b,), 200.0),
    }


def _groups(b=3):
    return torch.tensor([[3, 3, 2], [1, 1, 3], [4, 2, 5]][:b], dtype=torch.long)


def test_group_conditioning_is_identity_until_trained():
    torch.manual_seed(0)
    plain = AnomalyForecaster(base=16, n_leads=2).eval()
    torch.manual_seed(0)
    multi = AnomalyForecaster(base=16, n_leads=2, group_dim=8).eval()
    # Same trunk weights; the group module is extra and starts as a no-op.
    multi.load_state_dict(plain.state_dict(), strict=False)
    batch = _batch()
    with torch.no_grad():
        ref = plain(**batch)
        np.testing.assert_allclose(multi(**batch, groups=_groups()), ref, atol=1e-6)
        np.testing.assert_allclose(multi(**batch), ref, atol=1e-6)


def test_unknown_groups_are_the_population_model_after_training():
    torch.manual_seed(0)
    multi = AnomalyForecaster(base=16, n_leads=2, group_dim=8).eval()
    assert multi.group_film is not None
    with torch.no_grad():  # as if trained: projections and delta heads no longer zero
        for lin in multi.group_film.proj:
            lin.weight.normal_(0, 0.5)
        for head in (multi.head_coarse, multi.head_mid):
            head.weight.normal_(0, 0.5)
    batch = _batch()
    with torch.no_grad():
        none = multi(**batch)
        zeros = multi(**batch, groups=torch.zeros(3, 3, dtype=torch.long))
        grouped = multi(**batch, groups=_groups())
    np.testing.assert_allclose(zeros, none, atol=1e-5)
    assert not torch.allclose(grouped, none, atol=1e-4)


def test_unknown_embedding_receives_no_gradient():
    multi = AnomalyForecaster(base=16, n_leads=2, group_dim=8)
    assert multi.group_film is not None
    for lin in multi.group_film.proj:
        torch.nn.init.normal_(lin.weight, std=0.1)
    out = multi(**_batch(), groups=torch.tensor([[0, 2, 0], [3, 0, 4], [0, 0, 0]]))
    out.sum().backward()
    for emb in multi.group_film.embeds:
        assert emb.weight.grad is not None
        assert torch.all(emb.weight.grad[0] == 0)
        assert torch.all(emb.weight[0] == 0)


def test_model_inputs_hands_groups_only_to_a_multilevel_model():
    from habforecast.models.lit import model_inputs

    batch = {**_batch(), "groups": _groups()}
    assert "groups" in model_inputs(AnomalyForecaster(base=16, n_leads=2, group_dim=8), batch)
    assert "groups" not in model_inputs(AnomalyForecaster(base=16, n_leads=2), batch)


# ------------------------------------------------------------- pooled calibration --
def _prob(seed=0):
    return np.random.default_rng(seed).uniform(0.01, 0.99, size=(2, 2, 2, 5, 5)).astype(np.float32)


def test_group_map_applies_to_its_class_and_global_elsewhere():
    calib = {
        "params": {"1:20": [1.0, 0.0]},
        "group_factor": "trophic",
        "groups": {"trophic:hypereutrophic": {"1:20": [1.0, 2.0]}},
    }
    p = _prob()
    hyper = calibration.apply(p, calib, LEADS, THRS, groups=LakeGroups(trophic="hypereutrophic"))
    meso = calibration.apply(p, calib, LEADS, THRS, groups=LakeGroups(trophic="mesotrophic"))
    none = calibration.apply(p, calib, LEADS, THRS)
    np.testing.assert_allclose(meso, none)
    np.testing.assert_allclose(none[:, 0, 1], p[:, 0, 1], atol=1e-6)  # global is identity
    assert np.all(hyper[:, 0, 1] > p[:, 0, 1])  # +2 logit shift for the hypereutrophic class
    # Ranking is never changed.
    order = np.argsort(p[:, 0, 1].ravel())
    assert np.all(np.diff(hyper[:, 0, 1].ravel()[order]) >= 0)


def _bins(scale, shift, n=200_000, seed=3):
    from habforecast.eval.metrics import StreamingReliability

    rng = np.random.default_rng(seed)
    true_p = rng.uniform(0.001, 0.999, n)
    exceed = rng.uniform(size=n) < true_p
    logit = np.log(true_p) - np.log1p(-true_p)
    shown = 1 / (1 + np.exp(-(scale * logit + shift)))
    rel = StreamingReliability(20.0, n_bins=20)
    rel.update_masked(shown, np.where(exceed, np.log10(30.0), np.log10(5.0)))
    return rel.result()["thr20_reliability"]


def test_prior_pulls_a_thin_class_and_lets_a_rich_one_go():
    prior = [1.0, 0.0]
    rich = calibration.fit_from_reliability(_bins(2.0, 1.0), prior=prior, prior_positives=500)
    thin = calibration.fit_from_reliability(
        _bins(2.0, 1.0, n=300), prior=prior, prior_positives=500
    )
    free = calibration.fit_from_reliability(_bins(2.0, 1.0))
    assert rich is not None and thin is not None and free is not None
    # The well-sampled class recovers the inverse distortion (a ~ 0.5) like a free fit.
    assert abs(rich[0] - free[0]) < 0.05
    # The thin one stays much nearer the prior than the free fit does.
    assert abs(thin[0] - prior[0]) < abs(free[0] - prior[0]) / 2


def test_class_with_no_exceedances_is_served_the_prior():
    bins = [{"n": 1000, "mean_prob": 0.05, "obs_freq": 0.0}]
    assert calibration.fit_from_reliability(bins, prior=[0.8, -0.3], prior_positives=500) == [
        0.8,
        -0.3,
    ]


# ------------------------------------------------------------------- Open-Meteo --
def test_open_meteo_reduction_matches_the_cds_conventions():
    from habforecast.data.era5 import MET_VARIABLES, open_meteo_to_daily

    hours = [f"2020-07-01T{h:02d}:00" for h in range(24)] + [
        f"2020-07-02T{h:02d}:00" for h in range(24)
    ]
    n = len(hours)
    # Day 1: 4 m/s from the west all day (blowing east: u = +4, v = 0).
    # Day 2: 12 h from the east then 12 h from the west -- mean vector ~0, mean speed 4.
    direction = [270.0] * 24 + [90.0] * 12 + [270.0] * 12
    payload = {
        "latitude": 41.75,
        "longitude": -83.25,
        "hourly": {
            "time": hours,
            "temperature_2m": [20.0] * n,
            "wind_speed_10m": [4.0] * n,
            "wind_direction_10m": direction,
            "shortwave_radiation": [100.0] * n,  # W m-2 every hour -> 8.64 MJ m-2 d-1
            "precipitation": [0.5] * 23 + [None] + [0.0] * 24,
        },
    }
    ds = open_meteo_to_daily(payload)
    assert set(ds.data_vars) == set(MET_VARIABLES)
    np.testing.assert_allclose(ds["wind_u"].values[:, 0, 0], [4.0, 0.0], atol=1e-5)
    np.testing.assert_allclose(ds["wind_speed"].values[:, 0, 0], [4.0, 4.0], atol=1e-5)
    np.testing.assert_allclose(ds["ssrd"].values[:, 0, 0], [8.64, 8.64], rtol=1e-5)
    np.testing.assert_allclose(ds["t2m"].values[:, 0, 0], [20.0, 20.0])
    # A day with a missing hour is missing, not a short total.
    assert np.isnan(ds["tp"].values[0, 0, 0])
    assert ds["tp"].values[1, 0, 0] == 0.0


def test_open_meteo_backoff_reads_the_quota_it_hit():
    from habforecast.data.era5 import _open_meteo_backoff

    assert _open_meteo_backoff('{"reason":"Minutely API request limit exceeded"}') < 120
    assert 600 <= _open_meteo_backoff('{"reason":"Hourly API request limit exceeded"}') <= 3600
    assert _open_meteo_backoff('{"reason":"Daily API request limit exceeded"}') >= 3600


# ------------------------------------------------------------------ promotion gate --
def _gate_module():
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root / "scripts"))
    import promotion_gate

    return promotion_gate


def _row(lake, lead, skill, n):
    return {
        "lake_id": lake,
        "lead_days": lead,
        "skill_vs_persistence": skill,
        "r2": 0.5,
        "thr20_pr_auc": 0.5,
        "n_samples": n,
        "split": "test",
    }


def test_small_lake_rule_keeps_a_28_sample_lake_from_vetoing():
    gate = _gate_module()
    inc = [_row("victoria", 1, 0.20, 900), _row("tiny", 1, 0.30, 28)]
    cand = [_row("victoria", 1, 0.25, 900), _row("tiny", 1, 0.10, 28)]
    res = gate.gate(cand, inc)  # default: lakes under 100 samples cannot veto
    assert res["passed"] and res["excluded"] == ["tiny"] and res["min_samples"] == 100
    assert not gate.gate(cand, inc, min_samples=0)["passed"]  # strict: every lake counts
    assert gate.gate(cand, inc, min_samples=0, weighting="samples")["passed"]


def test_small_lake_rule_still_fails_a_real_regression_on_a_large_lake():
    gate = _gate_module()
    inc = [_row("victoria", 1, 0.25, 900), _row("tiny", 1, 0.10, 28)]
    cand = [_row("victoria", 1, 0.20, 900), _row("tiny", 1, 0.30, 28)]
    assert not gate.gate(cand, inc)["passed"]  # a small lake cannot rescue one either


def test_rows_without_a_sample_count_are_not_excluded():
    gate = _gate_module()
    rows = [{k: v for k, v in _row("victoria", 1, 0.2, 0).items() if k != "n_samples"}]
    res = gate.gate(rows, rows)
    assert res["passed"] and res["excluded"] == []


def test_gate_fails_when_every_lake_is_too_small():
    gate = _gate_module()
    rows = [_row("tiny", 1, 0.3, 28)]
    assert not gate.gate(rows, rows)["passed"]  # no evidence is not a pass


# ------------------------------------------------------------------ memory mapping --
def test_memory_mapped_series_matches_the_in_ram_one(tmp_path):
    import pandas as pd
    import xarray as xr

    from habforecast.features.dataset import load_lake_series

    rng = np.random.default_rng(0)
    vals = rng.normal(size=(20, 8, 9)).astype(np.float32)
    vals[rng.uniform(size=vals.shape) < 0.3] = np.nan
    ds = xr.Dataset(
        {"log_chl": (("time", "y", "x"), vals)},
        coords={"time": pd.date_range("2020-06-01", periods=20).values},
    )
    cube = tmp_path / "lake.zarr"
    ds.to_zarr(cube, consolidated=True)
    ram = load_lake_series(cube, "lake", "train")
    mm = load_lake_series(cube, "lake", "train", mmap_dir=tmp_path / "mm")
    assert isinstance(mm.values, np.memmap)
    np.testing.assert_array_equal(np.asarray(mm.values), ram.values)
    a = ram.climatology(170, 2019, halfwidth=15)
    b = mm.climatology(170, 2019, halfwidth=15)
    np.testing.assert_array_equal(a, b)
