"""Write each lake's thermal class into a registry, from its own ERA5 record.

    python scripts/annotate_groups.py --aoi data/aoi/lakes_global.geojson \
        --met data/processed/met_global

The thermal factor of the multilevel model (`habforecast.groups`) is derived from the
lake's annual-mean 2 m air temperature over the whole met store, not from latitude:
Dianchi (1,890 m) and Titicaca (3,810 m) are tropical by latitude and temperate by
temperature, and it is the temperature the growth physics answers to. A lake without a met
store is left unknown rather than guessed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from habforecast.groups import thermal_class  # noqa: E402


def annual_mean_t2m(store: Path) -> float | None:
    """Mean of the daily lake-mean t2m over whole years only, so a partial current year
    cannot tilt the mean toward the season it happens to end in."""
    import numpy as np
    import xarray as xr

    with xr.open_zarr(store, consolidated=True) as ds:
        if "t2m" not in ds:
            return None
        t2m = ds["t2m"].mean(dim=[d for d in ds["t2m"].dims if d != "time"]).load()
    years = t2m["time"].dt.year.values
    counts = {y: int((years == y).sum()) for y in np.unique(years)}
    full = [y for y, n in counts.items() if n >= 365]
    sel = t2m.values[np.isin(years, full)] if full else t2m.values
    sel = sel[np.isfinite(sel)]
    return float(sel.mean()) if sel.size else None


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--aoi", type=Path, required=True)
    p.add_argument("--met", type=Path, required=True)
    args = p.parse_args(argv)

    doc = json.loads(args.aoi.read_text(encoding="utf-8"))
    for feat in doc["features"]:
        props = feat["properties"]
        store = args.met / f"{props['lake_id']}.zarr"
        mean = annual_mean_t2m(store) if store.exists() else None
        props["thermal_class"] = thermal_class(mean)
        if mean is not None:
            props["mean_t2m_c"] = round(mean, 2)
        print(f"  {props['lake_id']:18s} {mean if mean is not None else float('nan'):6.1f} degC  "
              f"{props['thermal_class']}")
    args.aoi.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    counts: dict[str, int] = {}
    for f in doc["features"]:
        counts[f["properties"]["thermal_class"]] = counts.get(f["properties"]["thermal_class"], 0) + 1
    print(f"wrote {args.aoi}: {counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
