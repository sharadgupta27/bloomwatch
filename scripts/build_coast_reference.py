"""Build the offline land/sea reference the router classifies water bodies with.

Why an offline reference at all
-------------------------------
The router has to answer "is this inland water, a coastal bay, or open ocean?" *before*
it fetches anything, because the answer decides which product to fetch. So the test has
to be local, deterministic and free. A reverse-geocoding call would be none of those, and
would also send every AOI a user draws to a third party.

What is stored, and why it is this and not something else
---------------------------------------------------------
One thing: Natural Earth's 1:10 m **ocean polygon**, quantised to 0.01 degrees (~1.1 km)
and packed as int16. Two questions come out of that single geometry:

* **inland or marine** -- a point-in-polygon test. Inland water is water that is not in
  the ocean polygon: a lake is a point on a continent, and the imagery decides whether
  there is water there. This is a hydrological question, not an optical one, so a coarse
  outline answers it exactly.
* **how far from the coast** -- nearest distance to the polygon's boundary vertices, with
  the artificial map edges at +/-180 and +/-90 dropped. Measured against Natural Earth's
  separate 1:10 m coastline layer at ten test points, this agrees to within 0.5 km, so
  the second file is not worth its 10 MB.

Quantising to 0.01 degrees costs about a kilometre of shoreline precision on a decision
whose thresholds are 40 km and 200 km, and takes the artifact from 3.7 MB to 1.3 MB.

    python scripts/build_coast_reference.py

Source: Natural Earth 1:10 m physical vectors, public domain (naturalearthdata.com).
"""

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "data" / "reference" / "ne10_ocean.npz"
URL = (
    "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/"
    "geojson/ne_10m_ocean.geojson"
)
# 0.01 degrees, which is also what makes int16 sufficient: +/-180 degrees is +/-18000.
SCALE = 100.0


def rings_of(geometry: dict) -> list[np.ndarray]:
    polys = (
        geometry["coordinates"] if geometry["type"] == "MultiPolygon" else [geometry["coordinates"]]
    )
    return [np.asarray(r, dtype=np.float64) for poly in polys for r in poly]


def quantise(ring: np.ndarray) -> np.ndarray | None:
    """Snap to the 0.01 degree lattice and drop the duplicate vertices that creates."""
    q = np.round(ring * SCALE).astype(np.int16)
    keep = np.any(np.diff(q, axis=0) != 0, axis=1)
    q = np.vstack([q[0], q[1:][keep]])
    return q if len(q) >= 4 else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--source", default=URL)
    args = ap.parse_args()

    print(f"fetching {args.source}")
    with urllib.request.urlopen(args.source, timeout=180) as fh:
        doc = json.loads(fh.read().decode("utf-8"))

    rings = [q for r in rings_of(doc["features"][0]["geometry"]) if (q := quantise(r)) is not None]
    xy = np.concatenate(rings)
    offsets = np.cumsum([0] + [len(r) for r in rings]).astype(np.int32)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, xy=xy, offsets=offsets, scale=np.float64(SCALE))
    print(
        f"wrote {args.out} -- {len(rings)} rings, {len(xy)} vertices, "
        f"{args.out.stat().st_size / 1e6:.2f} MB"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
