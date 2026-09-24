"""Does giving the forecaster more than chlorophyll actually help?

The question
------------
Every model this project has trained sees one physical variable: fourteen daily frames of
log10 chlorophyll, plus masks and the day of year. That is a strong constraint. A bloom is
driven by things the retrieval does not carry -- how turbid the water is, how deep the
light reaches, whether what is floating on the surface is cyanobacteria or sediment -- and
the model can only infer them from where chlorophyll has been.

`habforecast.data.aux_layers` adds three of them, taken from assets already inside the
same OLCI granule, at the same 300 m, with no extra credential:

* **mci** -- the 709 nm peak height, which rises on surface scum that CHL_NN under-reads;
* **tsm** -- suspended sediment, the commonest false positive in a river plume;
* **kd490** -- light attenuation, the depth of the layer a bloom can grow in.

Why this comparison is controlled
---------------------------------
Both arms read **the same cubes**, the same lakes, the same period, the same architecture
and the same leave-lake-out split. The only difference is how many channels reach the
trunk. That is a cleaner comparison than the resolution experiment could be, because
there is no change of sensor, retrieval or masking to confound it.

What would count as a result
----------------------------
Skill against persistence and climatology on the held-out lakes, which is what every
earlier run in this project reported. A gain there is real. A gain on the validation lake
alone is the signature of extra channels being used to memorise, and is the outcome to
expect if three sparse layers are simply more parameters to overfit three lakes with.

    python scripts/aux_experiment.py --epochs 40
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

CUBES = "data/processed/cubes_aux"
AOI = "data/aoi/lakes.geojson"

ARMS: dict[str, dict] = {
    "chl-only": {
        "aux": [],
        "label": "chlorophyll only (the original input)",
    },
    "chl-mci": {
        "aux": ["mci"],
        "label": "chlorophyll + MCI (spectral index alone)",
    },
    "chl-aux3": {
        "aux": ["mci", "tsm", "kd490"],
        "label": "chlorophyll + MCI + suspended matter + light attenuation",
    },
}


def run(cmd: list[str], log: Path) -> int:
    print(f"    $ {' '.join(cmd[:5])} ...", flush=True)
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as fh:
        proc = subprocess.run(cmd, cwd=REPO, stdout=fh, stderr=subprocess.STDOUT, text=True)
    return proc.returncode


def train_arm(name: str, arm: dict, epochs: int, crop: int) -> Path | None:
    before = set(REPO.joinpath("artifacts").rglob("checkpoints/*.ckpt"))
    t0 = time.time()
    aux = "[" + ",".join(arm["aux"]) + "]"
    code = run(
        [
            sys.executable,
            "pipelines/train.py",
            "model=anomaly",
            f"paths.aoi={REPO / AOI}",
            f"paths.cubes={REPO / CUBES}",
            f"data.crop={crop}",
            f"data.aux={aux}",
            f"train.max_epochs={epochs}",
            f"run_name=aux-{name}",
            "experiment_name=hab-aux",
        ],
        REPO / "artifacts" / "aux" / f"{name}.train.log",
    )
    after = set(REPO.joinpath("artifacts").rglob("checkpoints/*.ckpt"))
    fresh = sorted(after - before, key=lambda p: p.stat().st_mtime)
    ckpt = fresh[-1] if fresh else None
    print(f"  {name}: train exit {code} in {time.time() - t0:.0f}s -> {ckpt}", flush=True)
    return ckpt if code == 0 else None


def eval_arm(name: str, ckpt: Path) -> dict | None:
    out = REPO / "artifacts" / f"eval_aux_{name}"
    code = run(
        [
            sys.executable,
            "pipelines/evaluate.py",
            "--checkpoint",
            str(ckpt),
            "--aoi",
            str(REPO / AOI),
            "--cubes",
            str(REPO / CUBES),
            "--out",
            str(out),
        ],
        REPO / "artifacts" / "aux" / f"{name}.eval.log",
    )
    scores = out / "scores.json"
    if code != 0 or not scores.exists():
        print(f"  {name}: eval exit {code}, no scores", flush=True)
        return None
    return json.loads(scores.read_text(encoding="utf-8"))


def skill_row(scores: dict, split: str, baseline: str) -> dict:
    """Mean Murphy skill per lead across the lakes in one split."""
    per_lead: dict[str, list[float]] = {}
    for lake in scores.get("lakes", []):
        if lake.get("split") != split:
            continue
        for lead, entry in (lake.get("leads") or {}).items():
            value = entry.get(f"skill_vs_{baseline}")
            if value is not None:
                per_lead.setdefault(str(lead), []).append(float(value))
    return {k: sum(v) / len(v) for k, v in sorted(per_lead.items(), key=lambda kv: int(kv[0])) if v}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arms", nargs="*", default=sorted(ARMS), choices=sorted(ARMS))
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--crop", type=int, default=128)
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--out", type=Path, default=REPO / "artifacts" / "aux" / "summary.json")
    args = ap.parse_args()

    cubes = REPO / CUBES
    if not cubes.exists() or not any(cubes.glob("*.zarr")):
        print(f"No cubes at {CUBES}. Ingest them first:")
        print(f"  python pipelines/ingest.py --aux mci,tsm,kd490 --cube-root {CUBES}")
        return 1

    summary: dict = {"cubes": CUBES, "epochs": args.epochs, "arms": {}}
    for name in args.arms:
        arm = ARMS[name]
        print(f"\n=== {name}: {arm['label']} ===", flush=True)
        ckpt = None
        if not args.skip_train:
            ckpt = train_arm(name, arm, args.epochs, args.crop)
        if ckpt is None:
            existing = sorted(
                (REPO / "artifacts").rglob(f"*aux-{name}*/checkpoints/*.ckpt"),
                key=lambda p: p.stat().st_mtime,
            )
            ckpt = existing[-1] if existing else None
        if ckpt is None:
            print(f"  {name}: no checkpoint, skipping evaluation", flush=True)
            continue
        scores = eval_arm(name, ckpt)
        if scores is None:
            continue
        summary["arms"][name] = {
            "label": arm["label"],
            "aux": arm["aux"],
            "checkpoint": str(ckpt),
            "test_vs_persistence": skill_row(scores, "test", "persistence"),
            "test_vs_climatology": skill_row(scores, "test", "climatology"),
            "val_vs_persistence": skill_row(scores, "val", "persistence"),
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n--- skill vs persistence, held-out lakes ---")
    leads = sorted({int(k) for a in summary["arms"].values() for k in a["test_vs_persistence"]})
    print(f"{'arm':10}" + "".join(f"{f'd{lead}':>9}" for lead in leads))
    for name, a in summary["arms"].items():
        row = a["test_vs_persistence"]
        print(f"{name:10}" + "".join(f"{row.get(str(lead), float('nan')):+9.3f}" for lead in leads))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
