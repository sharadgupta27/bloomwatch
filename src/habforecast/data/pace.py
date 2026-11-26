"""PACE OCI: what it offers this project, and why it is not the default.

PACE (Plankton, Aerosol, Cloud, ocean Ecosystem) flies OCI, a hyperspectral ocean-colour
imager, and it is the most capable instrument of its kind in orbit. It is the obvious
thing to reach for when asked to improve a bloom forecaster. Two measurements, taken
2026-08-30, decide how it is used here rather than assumed:

* **It needs credentials.** A granule URL from NASA's own catalogue redirects to
  `urs.earthdata.nasa.gov`, which answers **HTTP 401** to an anonymous client. The
  container this project ships runs with no account, which is a deliberate property --
  every other source it reads is open -- so PACE is opt-in and stays out of the default
  path even where a credential is configured. Verified working 2026-09-25 against
  `~/.netrc`: a Level-2 BGC granule from the previous day returned 200 and 9.9 MB of
  netCDF.
* **Its mapped products are coarse.** NASA publishes each Level-3 day at 0.1 degrees
  (~11 km) *and* at 4 km; 4 km is the finest mapped rung. This project has already
  measured what 4 km buys on lakes, in a controlled comparison against the same sensor at
  750 m: essentially no skill against persistence, and a collapse on the held-out lakes
  that is what overfitting to a handful of pixels looks like (`docs/DATA_SOURCES.md`).
  PACE Level-2 is ~1.2 km and could help on large water bodies; the mapped Level-3
  cannot, at either rung.

So the honest role for PACE here is **not** higher resolution, and a credential does not
change that: at 1.2 km a 12 km lake is 10 pixels across, below what the forecaster's
third downsample can use, and only a handful of the 46 registry lakes clear the bar.

It is the one thing no other open source provides: hyperspectral apparent optical
properties, from which phytoplankton *community composition* can be derived -- the
difference between a diatom bloom and a cyanobacterial one, which is exactly the
distinction chlorophyll cannot make and which decides whether a bloom is a public-health
problem. The use that follows from that is **reference data, not an input channel**: the
300 m OLCI indices (`ci`, `pc620` in `habforecast.data.aux_layers`) are cheap proxies for
that distinction, and on the large lakes PACE resolves, its L2 IOP and AOP products can
say whether those proxies are measuring what they claim. That is a validation exercise
with its own burden, not a channel to bolt on.

Credentials, in order of preference: `EARTHDATA_TOKEN`, else a `machine
urs.earthdata.nasa.gov` entry in `~/.netrc`. Without either, search still works (CMR is
open) so the catalogue can be inspected and the gate demonstrated.

    python -m habforecast.data.pace --check --product l2-aop
"""

from __future__ import annotations

import http.cookiejar
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
URS_HOST = "urs.earthdata.nasa.gov"
USER_AGENT = "habforecast/0.1 (bloom forecasting; contact via repository)"

# Verified present in CMR on 2026-08-30. NRT variants trail the science ones by hours.
COLLECTIONS = {
    "bgc": "PACE_OCI_L3M_BGC_NRT",  # chlorophyll-a, phytoplankton carbon
    "aop": "PACE_OCI_L3M_AOP_NRT",  # hyperspectral remote-sensing reflectance
    "iop": "PACE_OCI_L3M_IOP_NRT",  # absorption and backscatter
    "kd": "PACE_OCI_L3M_KD",  # diffuse attenuation
    # Level-2 swath, ~1.2 km: the only rung fine enough to put pixels on a large lake,
    # and the reason a credential is worth having. Verified present in CMR 2026-09-25.
    "l2-aop": "PACE_OCI_L2_AOP",  # hyperspectral Rrs, ~172 bands 340-890 nm
    "l2-iop": "PACE_OCI_L2_IOP",  # absorption/backscatter, incl. phytoplankton absorption
    "l2-bgc": "PACE_OCI_L2_BGC",  # chlorophyll, POC, and the PFT community split
    "l2-aop-nrt": "PACE_OCI_L2_AOP_NRT",
    "l2-bgc-nrt": "PACE_OCI_L2_BGC_NRT",
}

# Which collections are swath Level-2 rather than mapped Level-3.
L2_PRODUCTS = frozenset(k for k in COLLECTIONS if k.startswith("l2-"))

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


def netrc_login() -> tuple[str, str] | None:
    """Earthdata Login credentials from `~/.netrc`, if that is how the host is set up.

    The bearer token is the simpler credential, but `~/.netrc` with a
    `machine urs.earthdata.nasa.gov` entry is the form NASA's own tooling documents, so a
    machine that has used any Earthdata client already has one. Supporting both means the
    operator does not have to issue a second credential to use this path.

    Returns the login pair, never logs it, and is read fresh on each call so a rotated
    file takes effect without a restart.
    """
    try:
        import netrc as _netrc

        auth = _netrc.netrc().authenticators(URS_HOST)
    except (OSError, ImportError):
        return None
    except Exception:  # netrc.NetrcParseError and friends: a malformed file is not fatal
        logger.warning("~/.netrc could not be parsed; ignoring it for Earthdata auth")
        return None
    if not auth:
        return None
    login, _, password = auth
    return (login, password) if login and password else None


def have_credential() -> bool:
    """Whether a PACE download can be attempted at all, by either supported form."""
    return bool(token() or netrc_login())


def _netrc_opener(login: str, password: str) -> urllib.request.OpenerDirector:
    """An opener that can complete the Earthdata Login redirect dance.

    A data URL answers 302 to `urs.earthdata.nasa.gov`, which wants Basic auth and then
    redirects back to the DAAC carrying a session cookie. That needs both a cookie jar
    and a password manager scoped to the URS host -- urllib strips an `Authorization`
    header across a host change, which is exactly why a bare bearer header is not enough
    for every DAAC and why this path exists.
    """
    mgr = urllib.request.HTTPPasswordMgrWithDefaultRealm()
    mgr.add_password(None, f"https://{URS_HOST}", login, password)
    return urllib.request.build_opener(
        urllib.request.HTTPBasicAuthHandler(mgr),
        urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()),
    )


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
    """Download one granule. Needs `EARTHDATA_TOKEN` or a `~/.netrc` Earthdata entry.

    Raises a message written for whoever has to set the credential up, because the
    unauthenticated failure is a redirect to a login page and reads as a network fault.
    """
    if granule.url is None:
        raise ValueError(f"{granule.name}: catalogue entry carries no data URL")
    tok = token()
    login = None if tok else netrc_login()
    if not tok and not login:
        raise PermissionError(
            "PACE downloads need an Earthdata Login. Create one at "
            "https://urs.earthdata.nasa.gov, then either issue a bearer token and set "
            f"EARTHDATA_TOKEN, or add a `machine {URS_HOST}` entry to ~/.netrc. "
            "Without one NASA answers 401."
        )
    try:
        if tok:
            return _get(granule.url, timeout=300.0, token=tok)
        req = urllib.request.Request(granule.url, headers={"User-Agent": USER_AGENT})
        with _netrc_opener(*login).open(req, timeout=300.0) as r:
            return r.read()
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403):
            how = "token" if tok else "~/.netrc credentials"
            raise PermissionError(
                f"Earthdata rejected the {how} ({exc.code}). Tokens expire and are "
                "reissued at https://urs.earthdata.nasa.gov/profile; a netrc login also "
                "fails this way until the PACE data's EULA is accepted once in a browser."
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
    if token():
        print("  credential present: yes (EARTHDATA_TOKEN)")
    elif netrc_login():
        print(f"  credential present: yes (~/.netrc, machine {URS_HOST})")
    else:
        print("  credential present: no (set EARTHDATA_TOKEN or add a ~/.netrc entry)")
    if granules and args.check:
        try:
            data = fetch(granules[0])
            print(f"  fetched {len(data) / 1e6:.1f} MB")
        except PermissionError as exc:
            print(f"  gated: {exc}")
    for label, res in sorted(
        {**L3M_RESOLUTIONS_KM, "L2 swath": L2_KM}.items(), key=lambda kv: kv[1]
    ):
        print(f"  rung {label} = {res:.1f} km")
        for km in (12, 60, 400):
            ok, why = usable_for(km, res)
            print(f"    {km:4d} km water body: {'resolvable' if ok else 'too small'} -- {why}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
