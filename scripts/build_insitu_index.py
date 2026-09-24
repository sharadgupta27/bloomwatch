"""Distil the public in-situ archives into one small table the app can query on a click.

`scripts/survey_validation_data.py` answers "what exists". This answers "what can be
served": it turns ~100 MB of third-party downloads into a single gzipped CSV of about a
megabyte, holding only what a chart needs -- where, when, what, how much.

Three sources, chosen in `docs/VALIDATION_DATA.md`:

* **GLORIA** (PANGAEA, CC-BY 4.0) -- global laboratory chlorophyll-a, 1990-2022. The only
  source here with coverage outside the United States.
* **EPA Water Quality Portal, microcystin** -- the toxin itself, US only, 2015 onward.
* **EPA Water Quality Portal, chlorophyll-a** -- fetched per water body rather than
  nationally, because the national extract is millions of rows and a lake-sized bounding
  box is a hundred. This is what makes a US lake produce a dense time series instead of a
  scatter of points.

The registry's own lakes are fetched by default; `--bbox` adds any other area, which is
how the index grows to cover somewhere a user actually looks.

    python scripts/build_insitu_index.py
    python scripts/build_insitu_index.py --bbox 12.6,45.2,13.1,45.6 --name "Venice lagoon"

Downloads are cached under `data/cache/validation/`; only the index is written into the
repo tree, and it is gitignored for the same reason the caches are.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import sys
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from survey_validation_data import WQP, _num, fetch, gloria, wqp_microcystin  # noqa: E402

from habforecast.insitu import CHLOROPHYLL, COLUMNS, MICROCYSTIN  # noqa: E402

# Reported units, and what to multiply by to reach ug/L (identical to mg/m3 in water).
# Anything not on this list is dropped and counted rather than guessed at: a chlorophyll
# value in the wrong unit by a factor of a thousand would look exactly like a bloom.
UNIT_FACTOR = {
    "ug/l": 1.0,
    "ug/l ": 1.0,
    "ppb": 1.0,
    "mg/m3": 1.0,
    "mg/m^3": 1.0,
    "ug/m3": 0.001,
    "ng/ml": 1.0,
    "mg/l": 1000.0,
    "ppm": 1000.0,
}

# Chlorophyll above this is not a lake measurement, it is a transcription error or an
# extracted-pigment assay reported per gram of filter. GLORIA's own maximum is ~13,000.
MAX_PLAUSIBLE_CHL = 20_000.0


def _unit_factor(raw: str) -> float | None:
    return UNIT_FACTOR.get((raw or "").strip().lower())


# ---------------------------------------------------------------------------- sources --
def gloria_rows(cache: Path) -> list[dict]:
    """Global chlorophyll. GLORIA reports mg/m3 throughout, so no unit conversion."""
    out = []
    for r in gloria(cache):
        chl = r["chl"]
        if chl is None or not (0.0 <= chl <= MAX_PLAUSIBLE_CHL):
            continue
        day = r["date"][:10]
        if len(day) != 10:
            continue
        out.append(
            {
                "source": "GLORIA",
                "site": (r.get("site") or r.get("country") or "")[:60],
                "variable": CHLOROPHYLL,
                "date": day,
                "lat": r["lat"],
                "lon": r["lon"],
                "value": chl,
                "flag": "",
            }
        )
    return out


def _wqp_rows(results: list[dict], stations: dict, source: str, variable: str) -> list[dict]:
    """Turn Water Quality Portal results into index rows, dropping what cannot be placed.

    A result carries its own coordinates only sometimes; the station file carries them
    always, so the join is the reliable path and the activity location is the fallback.
    """
    out, no_place, bad_unit = [], 0, 0
    for r in results:
        st = stations.get(r.get("MonitoringLocationIdentifier", ""))
        lat = _num(r.get("ActivityLocation/LatitudeMeasure"))
        lon = _num(r.get("ActivityLocation/LongitudeMeasure"))
        if lat is None or lon is None:
            lat = _num(st.get("LatitudeMeasure")) if st else None
            lon = _num(st.get("LongitudeMeasure")) if st else None
        day = (r.get("ActivityStartDate") or "")[:10]
        if lat is None or lon is None or len(day) != 10:
            no_place += 1
            continue

        value = _num(r.get("ResultMeasureValue"))
        flag = ""
        if value is None:
            # "Not Detected" is a measurement, not a gap: it says the analyte was looked
            # for and was not there. Recorded as zero and flagged, so a chart can draw it
            # as an absence rather than dropping the only evidence of a clean week.
            if not (r.get("ResultDetectionConditionText") or "").strip():
                continue
            value, flag = 0.0, "nd"
        else:
            factor = _unit_factor(r.get("ResultMeasure/MeasureUnitCode", ""))
            if factor is None:
                bad_unit += 1
                continue
            value *= factor
        if value < 0 or (variable == CHLOROPHYLL and value > MAX_PLAUSIBLE_CHL):
            continue

        name = (st or {}).get("MonitoringLocationName") or r.get("MonitoringLocationName") or ""
        out.append(
            {
                "source": source,
                "site": name[:60],
                "variable": variable,
                "date": day,
                "lat": lat,
                "lon": lon,
                "value": value,
                "flag": flag,
            }
        )
    if no_place or bad_unit:
        print(f"    dropped {no_place:,} unplaceable, {bad_unit:,} in an unrecognised unit")
    return out


def microcystin_rows(cache: Path) -> list[dict]:
    results, stations = wqp_microcystin(cache)
    st = {s["MonitoringLocationIdentifier"]: s for s in stations}
    return _wqp_rows(results, st, "EPA WQP", MICROCYSTIN)


def _wqp_bbox(cache: Path, key: str, bbox: tuple, characteristic: str, since: str) -> list[dict]:
    """One water body's worth of a characteristic, results joined to their stations."""
    box = ",".join(f"{v:.4f}" for v in bbox)
    slug = key.replace("/", "_")[:40]
    res = fetch(
        f"{WQP}/Result/search?characteristicName={characteristic}&bBox={box}"
        f"&mimeType=csv&zip=no&dataProfile=resultPhysChem&startDateLo={since}",
        cache / f"wqp_chl_{slug}_results.csv",
    )
    sta = fetch(
        f"{WQP}/Station/search?characteristicName={characteristic}&bBox={box}"
        f"&mimeType=csv&zip=no&startDateLo={since}",
        cache / f"wqp_chl_{slug}_stations.csv",
    )
    read = lambda p: list(csv.DictReader(p.open(encoding="utf-8", errors="replace")))  # noqa: E731
    stations = {s["MonitoringLocationIdentifier"]: s for s in read(sta)}
    return _wqp_rows(read(res), stations, "EPA WQP", CHLOROPHYLL)


def registry_boxes(path: Path) -> list[tuple[str, tuple]]:
    """Each registered lake as (id, bbox), padded so a shoreline station is not missed."""
    out = []
    for f in json.loads(path.read_text(encoding="utf-8"))["features"]:
        ring = f["geometry"]["coordinates"][0]
        lons = [p[0] for p in ring]
        lats = [p[1] for p in ring]
        pad = 0.05
        out.append(
            (
                f["properties"]["lake_id"],
                (min(lons) - pad, min(lats) - pad, max(lons) + pad, max(lats) + pad),
            )
        )
    return out


# ------------------------------------------------------------------------------ main --
def dedupe(rows: list[dict]) -> list[dict]:
    """Drop exact repeats. Field replicates share a station, a day and a value."""
    seen, out = set(), []
    for r in rows:
        key = (
            r["source"],
            r["variable"],
            r["date"],
            round(r["lat"], 5),
            round(r["lon"], 5),
            round(r["value"], 4),
            r["flag"],
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


def write_index(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS, extrasaction="ignore")
        w.writeheader()
        for r in sorted(rows, key=lambda r: (r["lat"], r["lon"], r["date"])):
            w.writerow(
                {
                    **r,
                    "lat": round(r["lat"], 5),
                    "lon": round(r["lon"], 5),
                    "value": round(r["value"], 3),
                }
            )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", type=Path, default=Path("data/cache/validation"))
    ap.add_argument("--aoi", type=Path, default=Path("data/aoi/lakes_expanded.geojson"))
    ap.add_argument("--out", type=Path, default=Path("data/validation/insitu.csv.gz"))
    ap.add_argument("--since", default="01-01-2015", help="WQP start date, MM-DD-YYYY")
    ap.add_argument(
        "--bbox",
        action="append",
        default=[],
        metavar="W,S,E,N",
        help="extra area to fetch US chlorophyll for; repeatable",
    )
    ap.add_argument("--name", action="append", default=[], help="label for each --bbox")
    ap.add_argument("--skip-chl", action="store_true", help="GLORIA and microcystin only")
    args = ap.parse_args()

    rows: list[dict] = []

    print("GLORIA (global laboratory chlorophyll-a)")
    g = gloria_rows(args.cache)
    print(f"  {len(g):,} rows")
    rows += g

    print("EPA Water Quality Portal (microcystin, national)")
    m = microcystin_rows(args.cache)
    print(f"  {len(m):,} rows")
    rows += m

    if not args.skip_chl:
        boxes = registry_boxes(args.aoi)
        for i, raw in enumerate(args.bbox):
            parts = tuple(float(v) for v in raw.split(","))
            if len(parts) != 4:
                raise SystemExit(f"--bbox wants W,S,E,N; got {raw!r}")
            boxes.append((args.name[i] if i < len(args.name) else f"bbox{i}", parts))

        print(f"EPA Water Quality Portal (chlorophyll-a) over {len(boxes)} areas")
        for key, bbox in boxes:
            try:
                got = _wqp_bbox(args.cache, key, bbox, "Chlorophyll%20a", args.since)
            except Exception as exc:  # one area failing must not lose the other twenty
                print(f"  {key:18} failed: {type(exc).__name__}: {exc}")
                continue
            if got:
                print(f"  {key:18} {len(got):5,} rows")
            rows += got

    rows = dedupe(rows)
    write_index(rows, args.out)

    by_var: dict[str, int] = {}
    for r in rows:
        by_var[r["variable"]] = by_var.get(r["variable"], 0) + 1
    days = [r["date"] for r in rows]
    size_kb = args.out.stat().st_size / 1024
    print(f"\nwrote {args.out}  ({len(rows):,} rows, {size_kb:,.0f} kB gzipped)")
    for k, v in sorted(by_var.items(), key=lambda kv: -kv[1]):
        print(f"  {v:7,}  {k}")
    if days:
        print(f"  {min(days)} to {max(days)}")
    if any(date.fromisoformat(d) > date.today() for d in days):
        print("  note: some sample dates are in the future; the portal carries them as given")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
