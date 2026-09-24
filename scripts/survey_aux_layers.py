"""What else can be fed to the forecaster, measured against the live services.

The model is univariate: fourteen daily frames of log10 chlorophyll, an observation mask,
a water mask and the day of year. Every other physical driver of a cyanobacterial bloom --
how warm the water is, how clear it is, what the spectrum actually looks like -- is
absent, and the model can only infer them from where chlorophyll has been.

This asks what could be added *without credentials*, since a container that needs an
Earthdata login is a container that does not run. Each candidate is probed with a real
request over western Lake Erie; whatever refuses is recorded as refusing.

    python scripts/survey_aux_layers.py
    python scripts/survey_aux_layers.py --search sst temperature
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

ERDDAP = "https://coastwatch.noaa.gov/erddap"
USER_AGENT = "habforecast/0.1 (bloom forecasting; contact via repository)"

# Western Lake Erie: the best-observed bloom in the registry, so an empty answer here
# means the product does not cover inland water rather than that the lake was quiet.
BBOX = (-83.36, 41.60, -83.05, 41.84)
WINDOW = ("2026-08-10", "2026-08-24")

# What each candidate would contribute that chlorophyll alone cannot.
WHY = {
    "sst": "water temperature -- the strongest physical control on cyanobacterial growth",
    "rrs": "remote-sensing reflectance -- the spectrum the retrieval was derived from",
    "pace": "hyperspectral ocean colour, and phytoplankton community composition",
    "kd": "light attenuation -- turbidity and mixing depth",
    "par": "photosynthetically available radiation -- the energy driving growth",
    "wind": "wind stress -- what mixes a surface scum down or piles it on a shore",
}


def get(url: str, timeout: float = 90.0) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def search(term: str, limit: int = 400) -> list[dict]:
    """ERDDAP's own dataset search, as CSV so no extra dependency is needed."""
    url = (
        f"{ERDDAP}/search/advanced.csv?searchFor={urllib.parse.quote(term)}"
        "&page=1&itemsPerPage=1000&protocol=griddap"
    )
    try:
        raw = get(url).decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return []  # ERDDAP answers "nothing matched" with 404
        raise
    rows = list(csv.DictReader(io.StringIO(raw)))
    return rows[:limit]


def probe(dataset_id: str, variable: str) -> dict:
    """Ask for a fortnight over the lake and report what came back."""
    url = (
        f"{ERDDAP}/griddap/{dataset_id}.json?{variable}"
        f"[({WINDOW[0]}T00:00:00Z):({WINDOW[1]}T00:00:00Z)]"
        f"[({BBOX[3]}):({BBOX[1]})][({BBOX[0]}):({BBOX[2]})]"
    )
    started = time.perf_counter()
    try:
        raw = get(url)
    except urllib.error.HTTPError as exc:
        return {
            "ok": False,
            "detail": f"HTTP {exc.code}",
            "seconds": round(time.perf_counter() - started, 1),
        }
    except Exception as exc:
        return {
            "ok": False,
            "detail": type(exc).__name__,
            "seconds": round(time.perf_counter() - started, 1),
        }
    try:
        doc = json.loads(raw)
        rows = doc["table"]["rows"]
        col = doc["table"]["columnNames"].index(variable)
        vals = [r[col] for r in rows if r[col] is not None]
        return {
            "ok": True,
            "cells": len(rows),
            "valid": len(vals),
            "min": round(min(vals), 3) if vals else None,
            "max": round(max(vals), 3) if vals else None,
            "seconds": round(time.perf_counter() - started, 1),
        }
    except Exception as exc:
        return {
            "ok": False,
            "detail": f"unparseable: {exc}",
            "seconds": round(time.perf_counter() - started, 1),
        }


def axes(dataset_id: str) -> dict:
    """Resolution and record length, from the dataset's own metadata."""
    try:
        doc = json.loads(get(f"{ERDDAP}/info/{dataset_id}/index.json"))
    except Exception as exc:
        return {"error": type(exc).__name__ + f": {exc}"}
    out: dict = {}
    for row in doc["table"]["rows"]:
        kind, name, attr, _t, value = row[0], row[1], row[2], row[3], row[4]
        if kind == "attribute" and name == "NC_GLOBAL":
            if attr in ("time_coverage_start", "time_coverage_end", "title"):
                out[attr] = value
            if attr in ("geospatial_lat_resolution", "geospatial_lon_resolution"):
                out[attr] = value
        if kind == "variable" and name not in ("time", "latitude", "longitude", "altitude"):
            out.setdefault("variables", []).append(name)
    return out


CANDIDATES = [
    # (key, dataset id, variable). Chosen from the searches above; the variable names are
    # read back from each dataset's own metadata when a probe fails, so a wrong guess
    # here shows up as a wrong guess rather than as "no data".
    ("sst", "noaacwLEOACSPOSSTL3SCDaily", "sea_surface_temperature"),
    ("sst", "noaacwBLENDEDsstDNDaily", "analysed_sst"),
    ("sst", "noaacrwsstDaily", "analysed_sst"),
    ("kd", "noaacwNPPVIIRSSQkd490Daily", "kd_490"),
    ("kd", "noaacwNPPN20VIIRSkd490Daily", "kd_490"),
    ("kd", "noaacwNPPN20VIIRSkdparDaily", "kd_par"),
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--search", nargs="*", default=["sst", "PACE", "Rrs", "kd490", "par"])
    ap.add_argument("--probe", action="store_true", help="also fetch each candidate")
    ap.add_argument("--report", type=Path, default=None)
    args = ap.parse_args()

    report: dict = {"searches": {}, "probes": {}}

    for term in args.search:
        hits = search(term)
        ids = [h.get("Dataset ID", "") for h in hits]
        print(f"\n{term!r}: {len(hits)} griddap datasets")
        for h in hits[:12]:
            print(f"  {h.get('Dataset ID', ''):34} {h.get('Title', '')[:74]}")
        report["searches"][term] = ids

    if args.probe:
        print("\n--- probing candidates over western Lake Erie ---")
        for key, ds, var in CANDIDATES:
            meta = axes(ds)
            got = probe(ds, var)
            report["probes"][ds] = {"key": key, "variable": var, "meta": meta, "probe": got}
            note = (
                f"{got['valid']}/{got['cells']} cells, {got['min']}..{got['max']}"
                if got["ok"]
                else got["detail"]
            )
            print(f"  {ds:32} {var:22} {note}  ({got['seconds']}s)")
            if "time_coverage_end" in meta:
                print(
                    f"      record {meta.get('time_coverage_start', '?')[:10]} .. {meta['time_coverage_end'][:10]}"
                )

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nwrote {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
