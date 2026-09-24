"""Export a checkpoint (or an ensemble directory of them) to ONNX for torch-free serving.

    python scripts/export_onnx.py --checkpoint models/best.ckpt --out models/onnx/best
    python scripts/export_onnx.py --checkpoint models/ensemble --out models/onnx/ensemble

Writes one `member_NN.onnx` per checkpoint and a `manifest.json` holding what the
checkpoint's hyperparameters held -- normalisation, leads, window, channels, forcing
statistics, exceedance thresholds -- which is everything `inference.onnx_backend` needs
and all it reads. Point `HAB_CHECKPOINT` at the output directory and the API serves it.

**Every graph is checked before it is written.** Each member is run through torch and
onnxruntime on random inputs at two sizes (one of them odd, so a shape baked into the
trace would show) and the export is refused if they differ by more than `--tolerance` in
the model's standardised units. The largest difference is recorded in the manifest, so a
deployed export carries its own parity evidence.

Test-time augmentation is not baked in: the flips are applied by the backend around the
graph, as the torch path applies them around the module, so one graph serves both.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from habforecast.inference.onnx_backend import MANIFEST_FORMAT  # noqa: E402
from habforecast.inference.predictor import Predictor  # noqa: E402
from habforecast.models.lit import has_exceed_head  # noqa: E402

logger = logging.getLogger(__name__)
OPSET = 17


class _Exportable(nn.Module):
    """Positional-argument wrapper: ONNX graphs take a flat input list, the forecaster
    takes keywords, and the exceedance head returns logits where serving wants p."""

    def __init__(self, net: nn.Module, names: list[str]):
        super().__init__()
        self.net = net
        self.names = names
        self.exceed = has_exceed_head(net)

    def forward(self, *args):
        kwargs = dict(zip(self.names, args, strict=True))
        if self.exceed:
            pred, logits = self.net(**kwargs, return_exceed=True)
            return pred, torch.sigmoid(logits)
        return self.net(**kwargs)


def input_names(predictor: Predictor, net: nn.Module) -> list[str]:
    names = ["x", "x_mask", "water", "doy"]
    if predictor.use_clim:
        names += ["clim", "clim_mask"]
    if getattr(net, "n_met", 0):
        names += ["met", "met_mask"]
    if getattr(net, "n_group_factors", 0):
        names += ["groups"]
    return names


def sample_inputs(predictor: Predictor, names: list[str], b: int, h: int, w: int, seed: int):
    g = np.random.default_rng(seed)
    t, n_leads, n_met = predictor.input_days, len(predictor.lead_times), len(predictor.met_keys)
    x_mask = (g.random((b, t, h, w)) > 0.6).astype(np.float32)
    water = np.ones((b, 1, h, w), dtype=np.float32)
    water[:, :, : h // 6, : w // 5] = 0.0  # a land corner, so the mask is exercised
    arrays = {
        "x": (g.normal(size=(b, t, h, w)) * x_mask).astype(np.float32),
        "x_mask": x_mask,
        "water": water,
        "doy": g.uniform(1, 365, size=(b,)).astype(np.float32),
        "clim": g.normal(size=(b, n_leads, h, w)).astype(np.float32),
        "clim_mask": (g.random((b, n_leads, h, w)) > 0.2).astype(np.float32),
        "met": g.normal(size=(b, n_met, t)).astype(np.float32),
        "met_mask": np.ones((b, n_met, t), dtype=np.float32),
    }
    if "groups" in names:
        from habforecast.groups import FACTORS

        arrays["groups"] = np.stack(
            [g.integers(0, len(v), size=b) for v in FACTORS.values()], axis=1
        ).astype(np.int64)
    return [arrays[n] for n in names]


def export_member(net, predictor, out_path: Path, tolerance: float) -> float:
    import onnxruntime as ort

    net = net.float().cpu().eval()
    names = input_names(predictor, net)
    wrapper = _Exportable(net, names).eval()
    example = [torch.from_numpy(a) for a in sample_inputs(predictor, names, 1, 96, 96, 0)]
    outputs = ["forecast", "exceed_prob"] if wrapper.exceed else ["forecast"]
    dynamic = {n: {0: "batch"} for n in names}
    for n in ("x", "x_mask", "water", "clim", "clim_mask"):
        if n in names:
            dynamic[n] = {0: "batch", 2: "height", 3: "width"}
    dynamic["forecast"] = {0: "batch", 2: "height", 3: "width"}
    if wrapper.exceed:
        dynamic["exceed_prob"] = {0: "batch", 3: "height", 4: "width"}
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            tuple(example),
            str(out_path),
            input_names=names,
            output_names=outputs,
            dynamic_axes=dynamic,
            opset_version=OPSET,
            dynamo=False,
        )

    session = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
    worst = 0.0
    for b, h, w, seed in ((1, 96, 96, 1), (2, 131, 157, 2)):
        arrays = sample_inputs(predictor, names, b, h, w, seed)
        with torch.no_grad():
            ref = wrapper(*[torch.from_numpy(a) for a in arrays])
        ref = ref if isinstance(ref, tuple) else (ref,)
        got = session.run(None, dict(zip(names, arrays, strict=True)))
        for r, o in zip(ref, got, strict=True):
            worst = max(worst, float(np.abs(r.numpy() - o).max()))
    if worst > tolerance:
        out_path.unlink(missing_ok=True)
        raise SystemExit(
            f"{out_path.name}: onnxruntime differs from torch by {worst:.2e} (> {tolerance:.0e}); "
            "export refused"
        )
    logger.info("%s: parity with torch %.2e (tolerance %.0e)", out_path.name, worst, tolerance)
    return worst


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--checkpoint", type=Path, required=True, help=".ckpt or ensemble directory")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--aoi", type=Path, default=REPO_ROOT / "data" / "aoi" / "lakes.geojson")
    p.add_argument("--tolerance", type=float, default=1e-3)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")

    predictor = Predictor(
        args.checkpoint, args.aoi, REPO_ROOT / "data" / "processed" / "cubes", device="cpu"
    ).load()
    args.out.mkdir(parents=True, exist_ok=True)
    members, parity = [], []
    for i, net in enumerate(predictor.members):
        name = f"member_{i:02d}.onnx"
        parity.append(export_member(net, predictor, args.out / name, args.tolerance))
        members.append(name)

    sources = (
        sorted(p.name for p in args.checkpoint.glob("*.ckpt"))
        if args.checkpoint.is_dir()
        else [args.checkpoint.name]
    )
    manifest = {
        "format": MANIFEST_FORMAT,
        "model_version": predictor.model_version,
        "members": members,
        "source_checkpoints": sources,
        "norm": {"mean": predictor.norm.mean, "std": predictor.norm.std},
        "lead_times": list(predictor.lead_times),
        "input_days": predictor.input_days,
        "aux_keys": list(predictor.aux_keys),
        "use_clim": predictor.use_clim,
        "met_keys": list(predictor.met_keys),
        "met_stats": {k: list(v) for k, v in predictor.met_stats.items()},
        "exceed_thresholds": list(predictor.exceed_thresholds),
        "opset": OPSET,
        "parity_max_abs_err": max(parity),
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    # The probability recalibration travels with the export, so ONNX serves the same p.
    from habforecast import calibration

    source = calibration.calibration_path(args.checkpoint)
    if source.exists():
        shutil.copy2(source, args.out / calibration.CALIBRATION_FILE)
        logger.info("copied calibration %s", source.name)
    size = sum(f.stat().st_size for f in args.out.glob("*.onnx")) / 1e6
    logger.info("wrote %s: %d member(s), %.1f MB", args.out, len(members), size)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
