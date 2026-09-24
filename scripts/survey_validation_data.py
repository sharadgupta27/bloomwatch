"""Find public in-situ HAB measurements, and check whether they can validate this model.

Everything this project has measured so far is self-referential: the forecast is scored
against persistence and against climatology, both derived from the same satellite
retrieval. Neither says whether a predicted bloom was a real bloom. That needs in-situ
data -- someone in a boat with a bottle -- and the useful question is not "does such data
exist" (it does) but "does any of it fall inside the water bodies and dates this model
covers".

The four sources queried here, and why each one:

* **GLORIA** (PANGAEA, CC-BY 4.0) -- 7,572 curated in-situ samples from inland and
  coastal waters in 30 countries, every one georeferenced, most carrying a laboratory
  chlorophyll-a measurement. The closest thing to a global validation set for exactly the
  quantity this model predicts.
* **EPA Water Quality Portal** -- US only, but it carries **microcystin**, which is the
  toxin that chlorophyll is merely a proxy for. This is the only source here that can
  answer "did the bloom the model predicted actually make the water dangerous".
* **HAEDAT** (IOC-UNESCO) -- 14,118 expert-reviewed harmful-algae *events* worldwide
  since the 1980s. Rich on species, syndrome and toxin; located by free text and grid
  code rather than coordinates, and dominated by marine shellfish-toxin closures.
* **NOAA GLERL** -- publishes an operational Lake Erie HAB bulletin, i.e. a forecast made
  by somebody else that this one could be compared against directly.

Run with `--matchup` to intersect the downloaded points against the lake registry.
Downloads are cached; nothing is written into the repo but the report.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import urllib.request
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

USER_AGENT = "habforecast/0.1 (bloom forecasting research; contact via repository)"

GLORIA_ZIP = "https://download.pangaea.de/dataset/948492/files/GLORIA-2022.zip"
WQP = "https://www.waterqualitydata.us/data"
HAEDAT = "https://haedat.iode.org/api"

# WHO recreational guidance value for microcystin-LR. Above this a water body is a public
# health question, not a limnological one.
WHO_MICROCYSTIN_UGL = 8.0


def fetch(url: str, dest: Path, timeout: float = 600.0) -> Path:
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as r, dest.open("wb") as fh:
        fh.write(r.read())
    return dest


def _num(value) -> float | None:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(v) else v


# ------------------------------------------------------------------------- sources ---
def gloria(cache: Path) -> list[dict]:
    """Global in-situ chlorophyll with coordinates and dates."""
    path = fetch(GLORIA_ZIP, cache / "GLORIA-2022.zip")
    with zipfile.ZipFile(path) as z:
        name = next(n for n in z.namelist() if n.endswith("meta_and_lab.csv"))
        rows = list(
            csv.DictReader(io.TextIOWrapper(z.open(name), encoding="utf-8", errors="replace"))
        )
    out = []
    for r in rows:
        lat, lon = _num(r.get("Latitude")), _num(r.get("Longitude"))
        chl = _num(r.get("Chla"))
        if chl is None:
            chl = _num(r.get("Chla_plus_phaeo"))
        if lat is None or lon is None:
            continue
        out.append(
            {
                "lat": lat,
                "lon": lon,
                "date": (r.get("Date_Time_UTC") or "")[:10],
                "chl": chl,
                "country": r.get("Country", ""),
                "site": r.get("Site_name", ""),
                "source": "GLORIA",
            }
        )
    return out


def wqp_microcystin(cache: Path, since: str = "01-01-2015") -> tuple[list[dict], list[dict]]:
    """US toxin measurements, plus the stations that carry their coordinates."""
    res = fetch(
        f"{WQP}/Result/search?characteristicName=Microcystin&mimeType=csv&zip=no"
        f"&dataProfile=resultPhysChem&startDateLo={since}",
        cache / "wqp_microcystin_results.csv",
    )
    sta = fetch(
        f"{WQP}/Station/search?characteristicName=Microcystin&mimeType=csv&zip=no"
        f"&startDateLo={since}",
        cache / "wqp_microcystin_stations.csv",
    )
    read = lambda p: list(csv.DictReader(p.open(encoding="utf-8", errors="replace")))  # noqa: E731
    return read(res), read(sta)


def haedat_summary() -> dict:
    """Counts only: the events have no coordinates, so they cannot be matched up here."""
    req = urllib.request.Request(f"{HAEDAT}/events/?limit=1", headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=120) as r:
        return {"events": json.load(r).get("count")}


# ------------------------------------------------------------------------ matchups ---
def load_aois(path: Path) -> list[tuple]:
    out = []
    for f in json.loads(path.read_text(encoding="utf-8"))["features"]:
        ring = f["geometry"]["coordinates"][0]
        lons = [p[0] for p in ring]
        lats = [p[1] for p in ring]
        out.append(
            (
                f["properties"]["lake_id"],
                f["properties"]["split"],
                min(lons),
                min(lats),
                max(lons),
                max(lats),
            )
        )
    return out


def locate(aois, lat: float, lon: float):
    for lake_id, split, w, s, e, n in aois:
        if w <= lon <= e and s <= lat <= n:
            return lake_id, split
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", type=Path, default=Path("data/cache/validation"))
    ap.add_argument("--aoi", type=Path, default=Path("data/aoi/lakes_expanded.geojson"))
    ap.add_argument("--since-year", type=int, default=2016, help="ingested cube period")
    ap.add_argument("--report", type=Path, default=None)
    args = ap.parse_args()

    report: dict = {}

    print("HAEDAT (IOC-UNESCO global harmful algae events)")
    try:
        h = haedat_summary()
        print(f"  {h['events']:,} events; located by text and grid code, not coordinates")
        report["haedat"] = h
    except Exception as exc:
        print(f"  unavailable: {exc}")

    print("\nGLORIA (global in-situ reflectance and water quality)")
    g = gloria(args.cache)
    with_chl = [r for r in g if r["chl"] is not None]
    blooms = [r for r in with_chl if r["chl"] >= 20]
    print(f"  {len(g):,} georeferenced samples, {len(with_chl):,} with chlorophyll-a")
    print(
        f"  {len(blooms):,} at >=20 mg/m3 (bloom conditions), {len({r['country'] for r in g})} countries"
    )
    report["gloria"] = {"samples": len(g), "with_chl": len(with_chl), "blooms": len(blooms)}

    print("\nEPA Water Quality Portal (microcystin)")
    results, stations = wqp_microcystin(args.cache)
    st = {r["MonitoringLocationIdentifier"]: r for r in stations}
    exceed = sum(
        1 for r in results if (_num(r.get("ResultMeasureValue")) or 0) >= WHO_MICROCYSTIN_UGL
    )
    print(f"  {len(results):,} measurements at {len(st):,} stations")
    print(f"  {exceed:,} above the WHO {WHO_MICROCYSTIN_UGL:g} ug/L recreational guideline")
    report["wqp"] = {"results": len(results), "stations": len(st), "exceedances": exceed}

    # --- can any of it validate THIS model? ------------------------------------------
    aois = load_aois(args.aoi)
    print(f"\nMatchups against the {len(aois)} lakes in {args.aoi.name}:")

    hits: dict = defaultdict(list)
    for r in g:
        m = locate(aois, r["lat"], r["lon"])
        if m:
            hits[m].append(r)
    in_window = sum(
        1
        for v in hits.values()
        for r in v
        if r["date"][:4].isdigit() and int(r["date"][:4]) >= args.since_year
    )
    total = sum(len(v) for v in hits.values())
    print(
        f"  GLORIA: {total:,} points inside a modelled lake, {in_window:,} since {args.since_year}"
    )
    for (lake_id, split), v in sorted(hits.items(), key=lambda kv: -len(kv[1]))[:6]:
        print(f"    {lake_id:16} ({split}) {len(v):4d}")

    inside = {
        k
        for k, r in st.items()
        if _num(r.get("LatitudeMeasure")) is not None
        and locate(aois, float(r["LatitudeMeasure"]), float(r["LongitudeMeasure"]))
    }
    tox = [
        r
        for r in results
        if r["MonitoringLocationIdentifier"] in inside
        and r["ActivityStartDate"][:4] >= str(args.since_year)
    ]
    print(
        f"  microcystin: {len(inside)} stations inside modelled lakes, {len(tox)} measurements since {args.since_year}"
    )
    report["matchups"] = {
        "gloria_total": total,
        "gloria_in_window": in_window,
        "microcystin_stations": len(inside),
        "microcystin_results": len(tox),
    }

    # Why the toxin overlap is so small: the monitoring network samples water bodies the
    # sensor cannot resolve.
    types = Counter(
        r.get("MonitoringLocationTypeName", "?")
        for r in stations
        if r.get("MonitoringLocationTypeName")
    )
    print("\n  toxin stations by water-body type:")
    for k, v in types.most_common(5):
        print(f"    {v:6,}  {k}")

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nwrote {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
