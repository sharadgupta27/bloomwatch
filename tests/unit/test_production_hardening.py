"""ONNX serving, the promotion gate, and the daily input-drift check.

- An ONNX export must forecast what the checkpoint forecasts (including TTA and the
  exceedance head), and the serving path must not need torch to do it.
- The promotion gate must fail a candidate that regresses at *any* lead, and refuse to
  compare evaluations that are not like for like.
- The drift check must stay quiet on an ordinary window and fire on an unprecedented one.
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import promotion_gate  # noqa: E402
from monitoring.drift import daily_stats, window_drift  # noqa: E402

onnx = pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")


def _checkpoint(tmp_path: Path, seed: int, exceed: bool = True) -> Path:
    """A real Lightning-format checkpoint of a small, perturbed anomaly model."""
    from habforecast.models.anomaly import AnomalyForecaster
    from habforecast.models.lit import HABForecastModule

    torch.manual_seed(seed)
    kwargs: dict[str, Any] = {"base": 8, "use_clim": True}
    if exceed:
        kwargs.update(exceed_thresholds=[10.0, 20.0, 50.0], norm_mean=1.0, norm_std=0.45)
    net = AnomalyForecaster(input_days=14, n_leads=4, **kwargs)
    with torch.no_grad():  # move off the persistence initialisation
        for p in net.parameters():
            p.add_(0.05 * torch.randn_like(p))
    module = HABForecastModule(
        net,
        lead_times=(1, 3, 5, 7),
        norm_mean=1.0,
        norm_std=0.45,
        model_config={
            "name": "anomaly",
            "input_days": 14,
            "aux_keys": [],
            "met_keys": [],
            **kwargs,
        },
    )
    path = tmp_path / f"m{seed}.ckpt"
    torch.save({"state_dict": module.state_dict(), "hyper_parameters": dict(module.hparams)}, path)
    return path


class TestOnnx:
    @pytest.fixture
    def ensemble(self, tmp_path):
        d = tmp_path / "ens"
        d.mkdir()
        for seed in (0, 1):
            _checkpoint(d, seed).rename(d / f"member{seed}.ckpt")
        return d

    def test_export_matches_torch_with_tta_and_ensemble(self, ensemble, tmp_path):
        from export_onnx import main as export_main

        from habforecast.inference.onnx_backend import OnnxPredictor
        from habforecast.inference.predictor import Predictor

        out = tmp_path / "onnx"
        assert export_main(["--checkpoint", str(ensemble), "--out", str(out)]) == 0
        manifest = json.loads((out / "manifest.json").read_text())
        assert manifest["members"] == ["member_00.onnx", "member_01.onnx"]
        assert manifest["parity_max_abs_err"] < 1e-3

        aoi = REPO_ROOT / "data" / "aoi" / "lakes.geojson"
        cubes = tmp_path / "cubes"
        ref = Predictor(ensemble, aoi, cubes, device="cpu", tta=True).load()
        got = OnnxPredictor(out, aoi, cubes, tta=True).load()
        g = np.random.default_rng(3)
        m = (g.random((1, 14, 37, 53)) > 0.5).astype(np.float32)
        args = (
            (g.normal(size=m.shape) * m).astype(np.float32),
            m,
            np.ones((1, 1, 37, 53), np.float32),
            np.array([200.0], np.float32),
        )
        for a, b in zip(ref.forward_numpy(*args), got.forward_numpy(*args), strict=True):
            np.testing.assert_allclose(a, b, atol=1e-4)

    def test_onnx_backend_imports_without_torch(self):
        code = (
            "import sys; sys.modules['torch'] = None; sys.path.insert(0, 'src');"
            "import habforecast.inference.onnx_backend, habforecast.inference.adhoc,"
            " habforecast.advisory; print('ok')"
        )
        out = subprocess.run(
            [sys.executable, "-c", code], cwd=REPO_ROOT, capture_output=True, text=True
        )
        assert out.returncode == 0, out.stderr
        assert out.stdout.strip() == "ok"

    def test_serving_requirements_exclude_torch(self):
        from serve_onnx_requirements import requirements

        reqs = requirements(REPO_ROOT / "pyproject.toml")
        names = {r.split(">")[0].split("[")[0].split("=")[0] for r in reqs}
        assert "torch" not in names and "lightning" not in names
        assert "onnxruntime" in names and "fastapi" in names


def _rows(skill: dict[int, float], prauc: dict[int, float], r2: dict[int, float], **extra):
    return [
        {
            "lake_id": "victoria_ne",
            "split": "test",
            "lead_days": lead,
            "skill_vs_persistence": skill[lead],
            "thr20_pr_auc": prauc[lead],
            "r2": r2[lead],
            "split_mode": "lake",
            "tta": True,
            **extra,
        }
        for lead in skill
    ]


BASE = {1: 0.25, 7: 0.30}


class TestPromotionGate:
    def test_equal_passes(self):
        rows = _rows(BASE, BASE, BASE)
        assert promotion_gate.gate(rows, rows)["passed"]

    def test_regression_at_one_lead_fails_even_if_average_improves(self):
        inc = _rows(BASE, BASE, BASE)
        cand = _rows({1: 0.40, 7: 0.28}, BASE, BASE)  # +0.15 at 1 d, -0.02 at 7 d
        result = promotion_gate.gate(cand, inc)
        assert not result["passed"]
        failed = [c for c in result["checks"] if not c["ok"]]
        assert [(c["lead_days"], c["metric"]) for c in failed] == [(7, "skill_vs_persistence")]

    def test_head_prauc_is_preferred_when_present(self):
        inc = _rows(BASE, BASE, BASE)
        cand = _rows(BASE, {1: 0.10, 7: 0.10}, BASE, head_thr20_pr_auc=0.9)
        assert promotion_gate.gate(cand, inc)["passed"]

    def test_r2_regression_fails(self):
        inc = _rows(BASE, BASE, {1: 0.68, 7: 0.65})
        cand = _rows(BASE, BASE, {1: 0.68, 7: 0.60})
        assert not promotion_gate.gate(cand, inc)["passed"]

    def test_not_comparable_across_split_modes(self):
        inc = _rows(BASE, BASE, BASE)
        cand = _rows(BASE, BASE, BASE)
        for r in cand:
            r["split_mode"] = "temporal"
        assert "split_mode" in promotion_gate.comparable(cand, inc)

    def test_promote_keeps_the_incumbent_and_records_scores(self, tmp_path):
        rows = _rows(BASE, BASE, BASE)
        (tmp_path / "cand.json").write_text(json.dumps(rows))
        (tmp_path / "inc.json").write_text(json.dumps(rows))
        (tmp_path / "new.ckpt").write_bytes(b"new")
        target = tmp_path / "models" / "best.ckpt"
        target.parent.mkdir()
        target.write_bytes(b"old")
        code = promotion_gate.main(
            [
                "--candidate", str(tmp_path / "cand.json"),
                "--incumbent", str(tmp_path / "inc.json"),
                "--candidate-checkpoint", str(tmp_path / "new.ckpt"),
                "--promote-to", str(target),
            ]
        )  # fmt: skip
        assert code == 0
        assert target.read_bytes() == b"new"
        assert (target.parent / "best.previous.ckpt").read_bytes() == b"old"
        assert (target.parent / "best.scores.json").exists()
        assert (target.parent / "PROMOTION.json").exists()


class TestWindowDrift:
    @staticmethod
    def _archive(seed=0, days=300, pixels=400, level=1.0):
        g = np.random.default_rng(seed)
        day_level = level + 0.2 * g.standard_normal((days, 1))
        block = day_level + 0.15 * g.standard_normal((days, pixels))
        block[g.random(block.shape) < 0.4] = np.nan
        return block

    def test_ordinary_window_is_quiet(self):
        ref = daily_stats(self._archive(0))
        window = daily_stats(self._archive(1, days=4))
        assert not window_drift("lake", ref, window).any_drift

    def test_unprecedented_level_fires(self):
        ref = daily_stats(self._archive(0))
        window = daily_stats(self._archive(1, days=4, level=2.5))
        report = window_drift("lake", ref, window)
        assert report.any_drift
        assert "window_median_percentile" in report.drifted_metrics

    def test_days_with_too_few_pixels_are_ignored(self):
        block = np.full((3, 400), np.nan)
        block[0, :10] = 1.0
        assert daily_stats(block)["median"].size == 0

    def test_sparse_reference_reports_nothing(self):
        ref = daily_stats(self._archive(0, days=5))
        window = daily_stats(self._archive(1, days=4, level=5.0))
        assert window_drift("lake", ref, window).metrics == []


def test_forecast_result_keeps_its_inputs_out_of_the_summary():
    from habforecast.inference.predictor import ForecastResult

    r = ForecastResult(
        lake_id="x",
        lake_name="x",
        anchor_date=date(2024, 1, 1),
        lead_times=(1,),
        log_chl=np.zeros((1, 2, 2), np.float32),
        water=np.ones((2, 2), bool),
        input_log_chl=np.zeros((14, 2, 2), np.float32),
    )
    assert "input_log_chl" not in json.dumps(r.summary(0.09))
