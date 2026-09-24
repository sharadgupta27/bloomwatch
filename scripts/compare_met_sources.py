"""Compare the Open-Meteo met stores against the CDS ones, on the lakes that have both.

    python scripts/compare_met_sources.py --cds data/processed/met \
        --open-meteo data/processed/met_global --out artifacts/met_source_comparison.json

Both are ERA5; they differ in footprint (CDS: the mean of the cells of a bbox padded by
0.5 degrees, 3-hourly; Open-Meteo: the one cell at the lake centre, hourly). The model
reads the lake-mean daily value of each variable, so that is what is compared: per lake
and variable, the correlation and mean absolute difference of the daily series, and the
difference in their means -- which after standardisation is what the model would see as a
shift.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from habforecast.data.era5 import load_daily_means  # noqa: E402


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--cds", type=Path, default=REPO_ROOT / "data/processed/met")
    p.add_argument("--open-meteo", type=Path, default=REPO_ROOT / "data/processed/met_global")
    p.add_argument("--out", type=Path, default=REPO_ROOT / "artifacts/met_source_comparison.json")
    args = p.parse_args(argv)

    rows = []
    for store in sorted(args.cds.glob("*.zarr")):
        other = args.open_meteo / store.name
        if not other.exists():
            continue
        a, b = load_daily_means(store), load_daily_means(other)
        common = sorted(set(a.index) & set(b.index))
        for var in a.columns:
            if var not in b.columns:
                continue
            x = a.loc[common, var].to_numpy(np.float64)
            y = b.loc[common, var].to_numpy(np.float64)
            ok = np.isfinite(x) & np.isfinite(y)
            x, y = x[ok], y[ok]
            if x.size < 30:
                continue
            rows.append(
                {
                    "lake_id": store.stem,
                    "variable": var,
                    "days": int(x.size),
                    "corr": round(float(np.corrcoef(x, y)[0, 1]), 4),
                    "mae": round(float(np.abs(x - y).mean()), 4),
                    "mean_cds": round(float(x.mean()), 4),
                    "mean_open_meteo": round(float(y.mean()), 4),
                    # The shift in units of the CDS series' own spread: what a model
                    # standardised on one source would see when fed the other.
                    "bias_in_sd": round(float((y.mean() - x.mean()) / (x.std() or 1.0)), 4),
                }
            )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    by_var: dict[str, list[dict]] = {}
    for r in rows:
        by_var.setdefault(r["variable"], []).append(r)
    print(f"{'variable':11s} {'lakes':>5s} {'corr min/med':>14s} {'|bias| sd max':>14s}")
    for var, rs in by_var.items():
        corr = sorted(r["corr"] for r in rs)
        print(
            f"{var:11s} {len(rs):5d} {corr[0]:6.3f}/{corr[len(corr) // 2]:6.3f} "
            f"{max(abs(r['bias_in_sd']) for r in rs):14.3f}"
        )
    print(f"wrote {args.out} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
