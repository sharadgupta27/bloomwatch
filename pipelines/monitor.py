"""Drift monitoring: compare a target lake (or a time period) against the training baseline.

Run it before trusting a forecast for a lake the model has not trained on, and on a
schedule against the training lakes to catch upstream processing changes.

    python pipelines/monitor.py --target victoria_ne
    python pipelines/monitor.py --target erie_west --since 2025-01-01
    python pipelines/monitor.py --target balaton --out artifacts/drift/balaton.json

The reference is always the pooled *training* lakes, because that is what the model's
notion of "normal" was actually built from.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from monitoring.drift import (  # noqa: E402
    build_report,
    compare_distributions,
    compare_observation_density,
    geographic_novelty,
)

from habforecast.aoi import AOIRegistry  # noqa: E402

logger = logging.getLogger(__name__)


def lake_descriptors(values: np.ndarray, times: np.ndarray) -> dict[str, float]:
    """Lake-level summary used for the geographic-novelty distance.

    Four interpretable numbers rather than a learned embedding: an operator has to be
    able to look at the output and understand *why* a lake was flagged as unlike the
    training set.
    """
    obs = np.isfinite(values)
    water = obs.mean(axis=0) >= 0.05
    v = values[:, water]
    finite = v[np.isfinite(v)]
    if finite.size == 0:
        return {}

    # Seasonal amplitude: spread of monthly medians. Near zero for tropical lakes with
    # no seasonal cycle -- the single most discriminative descriptor for Lake Victoria.
    months = pd.to_datetime(times).month
    monthly = [
        np.nanmedian(v[months == m])
        for m in sorted(set(months))
        if np.isfinite(v[months == m]).any()
    ]
    amplitude = float(np.std(monthly)) if len(monthly) > 1 else 0.0

    return {
        "median_chl": float(np.median(finite)),
        "chl_spread": float(np.std(finite)),
        "seasonal_amplitude": amplitude,
        "obs_density": float(obs[:, water].mean()),
    }


def load_cube(cube_root: Path, lake_id: str, since: str | None):
    ds = xr.open_zarr(cube_root / f"{lake_id}.zarr", consolidated=True)
    if since:
        ds = ds.sel(time=slice(since, None))
    return ds["log_chl"].values, ds["time"].values


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--target", required=True, help="lake_id to check")
    p.add_argument("--aoi", type=Path, default=REPO_ROOT / "data" / "aoi" / "lakes.geojson")
    p.add_argument("--cubes", type=Path, default=REPO_ROOT / "data" / "processed" / "cubes")
    p.add_argument("--since", default=None, help="restrict the target to dates >= this")
    p.add_argument("--out", type=Path, default=None, help="write the JSON report here")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
    registry = AOIRegistry.from_geojson(args.aoi)

    # --- reference: pooled training lakes -----------------------------------------
    ref_values, ref_desc_parts, ref_density = [], [], []
    for aoi in registry.split("train"):
        path = args.cubes / f"{aoi.lake_id}.zarr"
        if not path.exists():
            logger.warning("%s: no cube, skipping from reference", aoi.lake_id)
            continue
        v, t = load_cube(args.cubes, aoi.lake_id, None)
        obs = np.isfinite(v)
        water = obs.mean(axis=0) >= 0.05
        ref_values.append(v[:, water][np.isfinite(v[:, water])])
        ref_density.append(float(obs[:, water].mean()))
        ref_desc_parts.append(lake_descriptors(v, t))

    if not ref_values:
        raise SystemExit("no training cubes found - cannot build a reference")

    reference = np.concatenate(ref_values)
    ref_desc: dict[str, float] = {}
    keys = ref_desc_parts[0].keys()
    for k in keys:
        vals = [d[k] for d in ref_desc_parts if k in d]
        ref_desc[k] = float(np.mean(vals))
        # Spread *across training lakes* is the natural scale for the z-distance: it says
        # how much variation the model has already seen and coped with.
        ref_desc[f"{k}_sd"] = float(np.std(vals)) or 1e-3

    # --- target --------------------------------------------------------------------
    v, t = load_cube(args.cubes, args.target, args.since)
    obs = np.isfinite(v)
    water = obs.mean(axis=0) >= 0.05
    target = v[:, water][np.isfinite(v[:, water])]
    tgt_density = float(obs[:, water].mean())
    tgt_desc = lake_descriptors(v, t)

    metrics = compare_distributions(reference, target)
    metrics.append(compare_observation_density(float(np.mean(ref_density)), tgt_density))
    metrics.append(geographic_novelty(ref_desc, tgt_desc))

    ref_name = "+".join(a.lake_id for a in registry.split("train"))
    report = build_report(ref_name, args.target, metrics)
    print(report.render())

    print("\n  descriptors (reference mean +- sd across training lakes vs target):")
    for k in sorted(tgt_desc):
        print(
            f"    {k:22s} {ref_desc.get(k, float('nan')):8.3f} "
            f"+-{ref_desc.get(k + '_sd', float('nan')):6.3f}   ->  {tgt_desc[k]:8.3f}"
        )

    if args.out:
        report.save(args.out)
        print(f"\n  report -> {args.out}")

    split = registry[args.target].split
    if report.any_drift and split == "test":
        print(
            "\n  NOTE: drift on the held-out test lake is expected and is the point of the\n"
            "  split. It quantifies how far the model is being asked to extrapolate."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
