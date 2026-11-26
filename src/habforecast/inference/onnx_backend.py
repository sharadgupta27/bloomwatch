"""Serve an ONNX export of the forecaster with onnxruntime -- no torch in the process.

The CPU serving image was 2.01 GB, and 750 MB of it was torch, there only to run a 340 k
parameter network. `scripts/export_onnx.py` writes each ensemble member as an ONNX graph
beside a `manifest.json` carrying everything the checkpoint's hyperparameters used to
(normalisation, leads, window, channels, forcing statistics, exceedance thresholds); this
module loads that directory and exposes the same `forward_numpy` the torch `Predictor`
does, so `predict`, the ad-hoc path and the advisory feed run unchanged on top of it.

What it does *not* do is decide correctness on its own: the export script checks every
graph against the checkpoint it came from on inputs of more than one size and records the
largest difference in the manifest, and refuses to write an export that disagrees.
"""

from __future__ import annotations

import json
import logging
from types import SimpleNamespace

import numpy as np

from habforecast.features.normalisation import Normalisation
from habforecast.inference.predictor import Predictor

logger = logging.getLogger(__name__)

MANIFEST_FORMAT = 1
# Spatial inputs are flipped for test-time augmentation; the rest are per-sample scalars
# or daily series and are left alone (see models.forward._NON_SPATIAL for the same rule).
SPATIAL_INPUTS = frozenset({"x", "x_mask", "water", "clim", "clim_mask"})
_FLIPS: tuple[tuple[int, ...], ...] = ((), (-1,), (-2,), (-1, -2))


class OnnxPredictor(Predictor):
    """`Predictor` with onnxruntime sessions in place of torch modules."""

    @staticmethod
    def _resolve_device(device: str | None):
        # onnxruntime's CPU provider; the image this exists for has no GPU.
        return "cpu"

    def load(self) -> OnnxPredictor:
        import onnxruntime as ort

        manifest_path = self.checkpoint_path / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("format") != MANIFEST_FORMAT:
            raise ValueError(
                f"{manifest_path}: unsupported manifest format {manifest.get('format')}"
            )
        self.manifest = manifest
        self.norm = Normalisation(**manifest["norm"])
        self.lead_times = tuple(manifest["lead_times"])
        self.input_days = int(manifest["input_days"])
        self.aux_keys = tuple(manifest.get("aux_keys", ()))
        self.use_clim = bool(manifest.get("use_clim", False))
        self.met_keys = tuple(manifest.get("met_keys", ()))
        self.met_stats = {
            k: (float(v[0]), float(v[1])) for k, v in manifest.get("met_stats", {}).items()
        }
        self._exceed = tuple(float(t) for t in manifest.get("exceed_thresholds", ()))
        self.model_version = manifest.get("model_version", self.checkpoint_path.name)
        from habforecast import calibration

        self.calibration = calibration.load(self.checkpoint_path)

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.members = [
            ort.InferenceSession(
                str(self.checkpoint_path / name),
                sess_options=opts,
                providers=["CPUExecutionProvider"],
            )
            for name in manifest["members"]
        ]
        self._input_names = [i.name for i in self.members[0].get_inputs()]
        # Consumers read `.module.model` only to test that something is loaded.
        self.module = SimpleNamespace(model=self.members[0])
        logger.info(
            "loaded ONNX %s: %d member(s), inputs=%s, exceed=%s, tta=%s",
            self.checkpoint_path.name,
            len(self.members),
            self._input_names,
            self._exceed,
            self.tta,
        )
        return self

    @property
    def exceed_thresholds(self) -> tuple[float, ...]:
        return getattr(self, "_exceed", ())

    def run(self, *args, **kwargs):
        raise NotImplementedError("the ONNX backend takes numpy: use forward_numpy")

    run_full = run

    def lightning_module(self):
        raise NotImplementedError("an ONNX export has no training wrapper; load the .ckpt")

    def _feeds(self, arrays: dict[str, np.ndarray | None], batch: int, h: int, w: int) -> dict:
        """Every input the graph declares, zero-filled (under a zero mask) where absent --
        the same "never observed" state the torch model substitutes for a missing channel."""
        n_leads = len(self.lead_times)
        n_met = len(self.met_keys)
        defaults = {
            "clim": (batch, n_leads, h, w),
            "clim_mask": (batch, n_leads, h, w),
            "met": (batch, n_met, self.input_days),
            "met_mask": (batch, n_met, self.input_days),
        }
        feeds = {}
        for name in self._input_names:
            value = arrays.get(name)
            if name == "groups":
                # Lake-group indices of a multilevel export; all zero is "unknown" on
                # every factor, the population level (see habforecast.groups).
                from habforecast.groups import FACTOR_NAMES

                if value is None:
                    value = np.zeros((batch, len(FACTOR_NAMES)), dtype=np.int64)
                feeds[name] = np.ascontiguousarray(value, dtype=np.int64)
                continue
            if value is None:
                value = np.zeros(defaults[name], dtype=np.float32)
            feeds[name] = np.ascontiguousarray(value, dtype=np.float32)
        return feeds

    def forward_numpy(
        self,
        x: np.ndarray,
        x_mask: np.ndarray,
        water: np.ndarray,
        doy: np.ndarray,
        clim: np.ndarray | None = None,
        clim_mask: np.ndarray | None = None,
        met: np.ndarray | None = None,
        met_mask: np.ndarray | None = None,
        groups=None,
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
        b, _t, h, w = x.shape
        arrays = {
            "x": x,
            "x_mask": x_mask,
            "water": water,
            "doy": doy,
            "clim": clim,
            "clim_mask": clim_mask,
            "met": met,
            "met_mask": met_mask,
            "groups": None if groups is None else np.asarray([groups.ids()] * b, dtype=np.int64),
        }
        base = self._feeds(arrays, b, h, w)
        orientations = _FLIPS if self.tta else ((),)
        preds, probs = [], []
        for session in self.members:
            p_sum = q_sum = None
            for dims in orientations:
                feeds = {
                    k: (np.flip(v, dims).copy() if dims and k in SPATIAL_INPUTS else v)
                    for k, v in base.items()
                }
                out = session.run(None, feeds)
                pred = np.flip(out[0], dims) if dims else out[0]
                p_sum = pred if p_sum is None else p_sum + pred
                if len(out) > 1:
                    prob = np.flip(out[1], dims) if dims else out[1]
                    q_sum = prob if q_sum is None else q_sum + prob
            assert p_sum is not None  # `orientations` is never empty
            n = len(orientations)
            preds.append(p_sum / n)
            if q_sum is not None:
                probs.append(q_sum / n)
        stack = np.stack(preds).astype(np.float32)
        mean = stack.mean(axis=0)
        # ddof=1 matches torch's unbiased std in inference.predictor.ensemble_forecast.
        spread = stack.std(axis=0, ddof=1) if len(preds) > 1 else None
        prob = np.stack(probs).mean(axis=0) if probs and len(probs) == len(preds) else None
        return mean, self.calibrate(prob, groups), spread
