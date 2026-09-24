"""PACE OCI: what it offers this project, and why it is not the default.

PACE (Plankton, Aerosol, Cloud, ocean Ecosystem) flies OCI, a hyperspectral ocean-colour
imager, and it is the most capable instrument of its kind in orbit. It is the obvious
thing to reach for when asked to improve a bloom forecaster. Two measurements, taken
2026-08-30, decide how it is used here rather than assumed:

* **It needs credentials.** A Level-3 granule URL from NASA's own catalogue redirects to
  `urs.earthdata.nasa.gov` and answers **HTTP 401**. The container this project ships
  runs with no account, which is a deliberate property -- every other source it reads is
  open -- so PACE is opt-in and stays out of the default path.
* **Its mapped products are coarse.** NASA publishes each Level-3 day at 0.1 degrees
  (~11 km) *and* at 4 km; 4 km is the finest mapped rung. This project has already
  measured what 4 km buys on lakes, in a controlled comparison against the same sensor at
  750 m: essentially no skill against persistence, and a collapse on the held-out lakes
  that is what overfitting to a handful of pixels looks like (`docs/DATA_SOURCES.md`).
  PACE Level-2 is ~1.2 km and could help on large water bodies; the mapped Level-3
  cannot, at either rung.

So the honest role for PACE here is **not** higher resolution. It is the one thing no
other open source provides: hyperspectral apparent optical properties, from which
phytoplankton *community composition* can be derived -- the difference between a diatom
bloom and a cyanobacterial one, which is exactly the distinction chlorophyll cannot make
and which decides whether a bloom is a public-health problem. That is a research
direction with its own validation burden, not a channel to bolt on.

With `EARTHDATA_TOKEN` set this module lists and fetches granules; without it, the search
still works (CMR is open) so the catalogue can be inspected and the gate demonstrated.

    python -m habforecast.data.pace --check
"""

from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import date

logger = logging.getLogger(__name__)

CMR = "https://cmr.earthdata.nasa.gov/search"
USER_AGENT = "habforecast/0.1 (bloom forecasting; contact via repository)"

# Verified present in CMR on 2026-08-30. NRT variants trail the science ones by hours.
COLLECTIONS = {
    "bgc": "PACE_OCI_L3M_BGC_NRT",  # chlorophyll-a, phytoplankton carbon
    "aop": "PACE_OCI_L3M_AOP_NRT",  # hyperspectral remote-sensing reflectance
    "iop": "PACE_OCI_L3M_IOP_NRT",  # absorption and backscatter
    "kd": "PACE_OCI_L3M_KD",  # diffuse attenuation
}

# The mapped grids, read off the granule names NASA publishes (`...0p1deg...`, `...4km...`).
L3M_RESOLUTIONS_KM = {"0p1deg": 11.1, "4km": 4.0}
# The finest mapped rung, which is the one worth arguing about.
L3M_KM = min(L3M_RESOLUTIONS_KM.values())
# Level-2 swath, for the record: the only PACE product fine enough to matter for a lake.
L2_KM = 1.2
# Below this many pixels across, the forecaster's third downsample has nothing left.
MIN_USEFUL_PIXELS = 16


@dataclass(frozen=True)
class Granule:
    name: str
    day: date | None
    url: str | None


def _get(url: str, timeout: float = 60.0, token: str | None = None) -> bytes:
    headers = {"User-Agent": USER_AGENT}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def token() -> str | None:
    """The Earthdata bearer token, if the operator supplied one.

    Read from the environment only. This project never puts a credential in a file it
    writes, and never sends one anywhere but the host it was issued for.
    """
    return os.getenv("EARTHDATA_TOKEN") or None


def search(product: str = "bgc", limit: int = 5) -> list[Granule]:
    """List recent granules. CMR is public, so this works without a token."""
    short = COLLECTIONS.get(product, product)
    url = (
        f"{CMR}/granules.umm_json?short_name={urllib.parse.quote(short)}"
        f"&page_size={limit}&sort_key=-start_date"
    )
    doc = json.loads(_get(url))
    out = []
    for item in doc.get("items", []):
        umm = item["umm"]
        href = None
        for link in umm.get("RelatedUrls", []):
            if link.get("Type") == "GET DATA" and link["URL"].endswith(".nc"):
                href = link["URL"]
                break
        start = (umm.get("TemporalExtent", {}).get("RangeDateTime", {}) or {}).get(
            "BeginningDateTime"
        )
        out.append(
            Granule(
                name=umm.get("GranuleUR", ""),
                day=date.fromisoformat(start[:10]) if start else None,
                url=href,
            )
        )
    return out


def fetch(granule: Granule) -> bytes:
    """Download one granule. Requires `EARTHDATA_TOKEN`.

    Raises a message written for whoever has to set the credential up, because the
    unauthenticated failure is a redirect to a login page and reads as a network fault.
    """
    if granule.url is None:
        raise ValueError(f"{granule.name}: catalogue entry carries no data URL")
    tok = token()
    if not tok:
        raise PermissionError(
            "PACE downloads need an Earthdata Login. Create one at "
            "https://urs.earthdata.nasa.gov, issue a bearer token, and set "
            "EARTHDATA_TOKEN. Without it NASA answers 401."
        )
    try:
        return _get(granule.url, timeout=300.0, token=tok)
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403):
            raise PermissionError(
                f"Earthdata rejected the token ({exc.code}). Tokens expire; reissue it."
            ) from exc
        raise


def usable_for(side_km: float, res_km: float = L3M_KM) -> tuple[bool, str]:
    """Whether a product at `res_km` can resolve a water body of this size, and why not.

    Defaults to PACE's finest *mapped* rung. "Resolvable" is a weaker claim than
    "useful": the 4 km arm of this project's own resolution experiment ran fine and had
    no skill, so clearing this bar is necessary and nowhere near sufficient.
    """
    px = side_km / res_km
    if px >= MIN_USEFUL_PIXELS:
        return True, f"{px:.0f} px across at {res_km:.0f} km"
    return False, (
        f"{side_km:.0f} km is {px:.1f} pixels at {res_km:.0f} km; the forecaster needs "
        f"about {MIN_USEFUL_PIXELS} across. Use the 300 m OLCI path for water this size."
    )


def _main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true", help="report catalogue and credentials")
    ap.add_argument("--product", default="bgc", choices=sorted(COLLECTIONS))
    args = ap.parse_args()

    print(f"PACE OCI {args.product} -> {COLLECTIONS[args.product]}")
    try:
        granules = search(args.product, limit=3)
    except Exception as exc:
        print(f"  catalogue unreachable: {type(exc).__name__}: {exc}")
        return 1
    for g in granules:
        print(f"  {g.day}  {g.name}")
    print(f"  credential present: {'yes' if token() else 'no (EARTHDATA_TOKEN unset)'}")
    if granules and args.check:
        try:
            data = fetch(granules[0])
            print(f"  fetched {len(data) / 1e6:.1f} MB")
        except PermissionError as exc:
            print(f"  gated: {exc}")
    for label, res in sorted(L3M_RESOLUTIONS_KM.items(), key=lambda kv: kv[1]):
        print(f"  mapped rung {label} = {res:.0f} km")
        for km in (12, 60, 400):
            ok, why = usable_for(km, res)
            print(f"    {km:4d} km water body: {'resolvable' if ok else 'too small'} -- {why}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
