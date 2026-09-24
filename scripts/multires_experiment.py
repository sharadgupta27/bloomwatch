"""Train and evaluate the forecaster at each rung of the resolution ladder.

The question
------------
The 300 m evaluation left one clear finding: **capacity was not the constraint**. A
5.8 M-parameter network was worse on the held-out lake than a 333 K one, and the
generalisation gap tracked how few water bodies the model had seen -- three. Every
architectural change after that bought less than the data would.

So the experiment is not "which network", it is "which *data*":

* **more lakes** -- 15 training water bodies instead of 3, reachable because ERDDAP
  subsets server-side and a lake-year costs seconds rather than hours;
* **more years** -- the VIIRS science-quality record starts in 2012, against 2018 for the
  OLCI archive;
* **which resolution** -- the same sensor at 750 m and at 4 km, which is a controlled
  comparison: identical retrieval, identical period, identical lakes, only the pixel
  changes.

What can and cannot be concluded
--------------------------------
A model trained on VIIRS is not comparable *in absolute terms* to one trained on OLCI:
different instrument, different retrieval, different quality masking. What is comparable
is **skill against the same baselines on lakes the model has never seen**, because skill
is measured within each configuration against its own persistence and climatology. That
is the number this script reports, and it is the number the earlier 300 m runs reported.

Every run is leave-lake-out by construction: the split is by water body, assigned
geographically in `scripts/expand_registry.py`, never randomly.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# Crop must fit inside the smallest training grid at that resolution, and should cover a
# comparable *ground area* across rungs so the model sees a similar spatial context.
#   300 m x 128 = 38 km      750 m x 48 = 36 km      4 km x 12 = 48 km
ARMS: dict[str, dict] = {
    "olci-300m": {
        "cubes": "data/processed/cubes",
        "aoi": "data/aoi/lakes.geojson",
        "crop": 128,
        "label": "OLCI 300 m, 3 training lakes (the original)",
    },
    "viirs-750m": {
        "cubes": "data/processed/cubes_viirs-750m",
        "aoi": "data/aoi/lakes_expanded.geojson",
        "crop": 48,
        "label": "VIIRS 750 m, 15 training lakes",
    },
    "viirs-4km": {
        "cubes": "data/processed/cubes_viirs-4km",
        "aoi": "data/aoi/lakes_expanded.geojson",
        "crop": 12,
        "label": "VIIRS 4 km, 15 training lakes",
    },
    "dineof-2km": {
        "cubes": "data/processed/cubes_dineof-2km",
        "aoi": "data/aoi/lakes_expanded.geojson",
        "crop": 24,
        "label": "VIIRS+OLCI gap-filled 2 km, 15 training lakes",
    },
}


def run(cmd: list[str], log: Path) -> int:
    print(f"    $ {' '.join(cmd[:6])} ...", flush=True)
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as fh:
        proc = subprocess.run(cmd, cwd=REPO, stdout=fh, stderr=subprocess.STDOUT, text=True)
    return proc.returncode


def newest_checkpoint(run_root: Path) -> Path | None:
    ckpts = sorted(run_root.rglob("checkpoints/*.ckpt"), key=lambda p: p.stat().st_mtime)
    return ckpts[-1] if ckpts else None


def train_arm(name: str, arm: dict, epochs: int, model: str) -> Path | None:
    cubes = REPO / arm["cubes"]
    if not cubes.exists() or not any(cubes.glob("*.zarr")):
        print(f"  {name}: no cubes at {arm['cubes']} -- skipped", flush=True)
        return None
    before = set(REPO.joinpath("artifacts").rglob("checkpoints/*.ckpt"))
    t0 = time.time()
    code = run(
        [
            sys.executable,
            "pipelines/train.py",
            f"model={model}",
            f"paths.aoi={REPO / arm['aoi']}",
            f"paths.cubes={cubes}",
            f"data.crop={arm['crop']}",
            f"train.max_epochs={epochs}",
            f"run_name=multires-{name}",
            "experiment_name=hab-multires",
        ],
        REPO / "artifacts" / "multires" / f"{name}.train.log",
    )
    after = set(REPO.joinpath("artifacts").rglob("checkpoints/*.ckpt"))
    fresh = sorted(after - before, key=lambda p: p.stat().st_mtime)
    ckpt = fresh[-1] if fresh else newest_checkpoint(REPO / "artifacts")
    print(f"  {name}: train exit {code} in {time.time() - t0:.0f}s -> {ckpt}", flush=True)
    return ckpt if code == 0 else None


def eval_arm(name: str, arm: dict, ckpt: Path) -> dict | None:
    out = REPO / "artifacts" / f"eval_multires_{name}"
    code = run(
        [
            sys.executable,
            "pipelines/evaluate.py",
            "--checkpoint",
            str(ckpt),
            "--aoi",
            str(REPO / arm["aoi"]),
            "--cubes",
            str(REPO / arm["cubes"]),
            "--out",
            str(out),
        ],
        REPO / "artifacts" / "multires" / f"{name}.eval.log",
    )
    scores = out / "scores.json"
    if code != 0 or not scores.exists():
        print(f"  {name}: eval exit {code}, no scores", flush=True)
        return None
    return json.loads(scores.read_text(encoding="utf-8"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arms", nargs="*", default=["viirs-750m"], choices=sorted(ARMS))
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--model", default="anomaly")
    ap.add_argument("--skip-train", action="store_true", help="evaluate existing checkpoints only")
    ap.add_argument("--out", type=Path, default=REPO / "artifacts" / "multires" / "summary.json")
    args = ap.parse_args()

    summary = {"model": args.model, "epochs": args.epochs, "arms": {}}
    for name in args.arms:
        arm = ARMS[name]
        print(f"\n=== {name}: {arm['label']}", flush=True)
        ckpt = None
        if not args.skip_train:
            ckpt = train_arm(name, arm, args.epochs, args.model)
        else:
            ckpt = newest_checkpoint(REPO / "artifacts")
        if ckpt is None:
            summary["arms"][name] = {"status": "no checkpoint"}
            continue
        scores = eval_arm(name, arm, ckpt)
        summary["arms"][name] = {
            "status": "ok" if scores else "eval failed",
            "label": arm["label"],
            "checkpoint": str(ckpt),
            "scores": scores,
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nsummary -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
