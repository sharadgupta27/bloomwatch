"""Build an expanded lake registry for multi-resolution training.

The evaluation of the 300 m model found the ceiling was not the network: capacity was
monotonically unhelpful, and the generalisation gap tracked how few water bodies the
model had ever seen. Three training lakes cannot teach a model what a lake looks like in
general. This writes a registry with enough of them to test that claim.

Selection rules, in order:

1. **Documented bloom history.** Every entry is a water body with a published record of
   cyanobacterial or harmful algal blooms, or a large lake routinely monitored for them.
   A model trained on clear oligotrophic water learns the wrong prior.
2. **Big enough to resolve.** At 750 m a 20 km lake is 27 pixels across, which is the
   floor for a network that downsamples three times. Small reservoirs are excluded even
   where their bloom records are excellent.
3. **Spread across climate, latitude and trophic state.** Leave-lake-out evaluation only
   measures generalisation if the held-out lake is genuinely unlike the training ones.
4. **The original five keep their identity and their split.** Every number previously
   reported was measured with erie_west/okeechobee/taihu training, balaton validating and
   victoria_ne held out, and changing that silently would make the old and new results
   incomparable.

Splits are assigned geographically, never randomly: two lakes on the same shore share
weather, and a random split would let the model see tomorrow's synoptic conditions in the
training set and call the result skill.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# lake_id, name, country, split, trophic, bloom months, centre lat/lon, half-width km, notes
LAKES: list[tuple] = [
    # ---------------------------------------------------------------- original five --
    (
        "erie_west",
        "Lake Erie (Western Basin)",
        "US/CA",
        "train",
        "eutrophic",
        (6, 10),
        41.70,
        -83.02,
        45,
        "Microcystis blooms; 2014 Toledo drinking-water shutdown.",
    ),
    (
        "okeechobee",
        "Lake Okeechobee",
        "US",
        "train",
        "hypereutrophic",
        (5, 10),
        26.95,
        -80.80,
        30,
        "Recurrent Microcystis; discharges seed coastal blooms.",
    ),
    (
        "taihu",
        "Lake Taihu",
        "CN",
        "train",
        "hypereutrophic",
        (5, 11),
        31.20,
        120.20,
        35,
        "2007 Wuxi supply crisis; among the most studied bloom lakes.",
    ),
    (
        "balaton",
        "Lake Balaton",
        "HU",
        "val",
        "mesotrophic",
        (6, 9),
        46.85,
        17.75,
        35,
        "Shallow, well monitored; weaker blooms than the training set.",
    ),
    (
        "victoria_ne",
        "Lake Victoria (NE / Winam Gulf)",
        "KE",
        "test",
        "eutrophic",
        (1, 12),
        -0.30,
        34.20,
        30,
        "Equatorial, year-round; the original held-out lake.",
    ),
    # ------------------------------------------------------- North America (train) --
    (
        "winnipeg",
        "Lake Winnipeg",
        "CA",
        "train",
        "eutrophic",
        (6, 10),
        52.10,
        -97.30,
        60,
        "Large summer cyanobacteria blooms; nutrient-loaded basin.",
    ),
    (
        "st_clair",
        "Lake St. Clair",
        "US/CA",
        "train",
        "eutrophic",
        (6, 10),
        42.42,
        -82.68,
        25,
        "Shallow connecting channel above the Erie western basin.",
    ),
    (
        "saginaw",
        "Lake Huron (Saginaw Bay)",
        "US",
        "train",
        "eutrophic",
        (6, 10),
        43.85,
        -83.65,
        30,
        "Shallow embayment with a long HAB record.",
    ),
    (
        "green_bay",
        "Lake Michigan (Green Bay)",
        "US",
        "train",
        "eutrophic",
        (6, 10),
        44.75,
        -87.90,
        30,
        "Stratified eutrophic bay; recurrent blooms.",
    ),
    (
        "utah_lake",
        "Utah Lake",
        "US",
        "train",
        "hypereutrophic",
        (6, 10),
        40.22,
        -111.80,
        20,
        "Shallow, warm, chronic cyanobacteria advisories.",
    ),
    (
        "pontchartrain",
        "Lake Pontchartrain",
        "US",
        "train",
        "eutrophic",
        (5, 10),
        30.20,
        -90.10,
        30,
        "Brackish estuarine lake; blooms after river diversions.",
    ),
    # -------------------------------------------------------------- Europe (train) --
    (
        "ijsselmeer",
        "IJsselmeer",
        "NL",
        "train",
        "eutrophic",
        (5, 10),
        52.75,
        5.35,
        35,
        "Large shallow freshwater lake, heavily managed.",
    ),
    (
        "curonian",
        "Curonian Lagoon",
        "LT/RU",
        "train",
        "hypereutrophic",
        (5, 10),
        55.30,
        21.05,
        30,
        "Among the most bloom-prone lagoons in Europe.",
    ),
    (
        "neagh",
        "Lough Neagh",
        "GB",
        "train",
        "hypereutrophic",
        (5, 10),
        54.62,
        -6.42,
        20,
        "Severe blooms since 2023; largest lake in the British Isles.",
    ),
    # ----------------------------------------------------------------- Asia (train) --
    (
        "chaohu",
        "Lake Chaohu",
        "CN",
        "train",
        "hypereutrophic",
        (5, 11),
        31.55,
        117.55,
        25,
        "Shallow; blooms comparable to Taihu.",
    ),
    (
        "poyang",
        "Lake Poyang",
        "CN",
        "train",
        "eutrophic",
        (4, 10),
        29.10,
        116.25,
        45,
        "Largest Chinese freshwater lake; extreme seasonal area change.",
    ),
    (
        "kasumigaura",
        "Lake Kasumigaura",
        "JP",
        "train",
        "hypereutrophic",
        (5, 10),
        36.03,
        140.40,
        20,
        "Shallow and turbid; long monitoring record.",
    ),
    # ------------------------------------------------------------ Africa (validate) --
    (
        "victoria_mwanza",
        "Lake Victoria (Mwanza Gulf)",
        "TZ",
        "val",
        "eutrophic",
        (1, 12),
        -2.55,
        32.90,
        30,
        "Same lake, opposite shore -- a strict test of local overfitting.",
    ),
    # ------------------------------------------------------------------ held out ----
    # Deliberately unlike anything in training: a saline terminal lake, a subtropical
    # reservoir on another continent, and a temperate lake in the southern hemisphere.
    (
        "hartbeespoort",
        "Hartbeespoort Dam",
        "ZA",
        "test",
        "hypereutrophic",
        (1, 12),
        -25.74,
        27.85,
        12,
        "Chronic hyperscum; southern hemisphere, small.",
    ),
    (
        "alexandrina",
        "Lake Alexandrina",
        "AU",
        "test",
        "eutrophic",
        (11, 4),
        -35.45,
        139.20,
        25,
        "Terminal lake of the Murray; southern-hemisphere seasonality.",
    ),
    (
        "chapala",
        "Lake Chapala",
        "MX",
        "test",
        "eutrophic",
        (5, 11),
        20.25,
        -103.05,
        30,
        "Largest Mexican lake; subtropical highland.",
    ),
]


# The global registry (2026-09-23): everything above, unchanged in identity and split, plus
# lakes chosen to fill the gaps the multilevel model and the per-trophic-class calibration
# need filled. Two new validation lakes give the calibration a hypereutrophic (Dianchi) and
# an oligotrophic (Geneva) level to fit on; Balaton and Victoria-Mwanza already cover
# mesotrophic and eutrophic. The held-out test set is deliberately left as it was, so the
# promotion gate still compares like with like. Oligotrophic lakes are included on purpose
# now: a model that is told a lake's trophic class has to have seen the clear end too.
GLOBAL_LAKES: list[tuple] = [
    # ------------------------------------------------------------------- Asia ------
    ("dianchi", "Lake Dianchi", "CN", "val", "hypereutrophic", (5, 11), 24.83, 102.70, 20,
     "Plateau lake, 1,890 m; decades of Microcystis blooms. Hypereutrophic validation lake."),
    ("hongze", "Hongze Lake", "CN", "train", "eutrophic", (5, 10), 33.30, 118.70, 30,
     "Large shallow Huai River lake; recurring summer blooms."),
    ("erhai", "Erhai Lake", "CN", "train", "mesotrophic", (6, 10), 25.80, 100.18, 22,
     "Plateau lake at the meso/eutrophic transition; blooms since the 1990s."),
    ("biwa", "Lake Biwa", "JP", "train", "mesotrophic", (6, 10), 35.25, 136.08, 33,
     "Southern basin blooms; long monitoring record."),
    ("laguna_de_bay", "Laguna de Bay", "PH", "train", "eutrophic", (1, 12), 14.38, 121.25, 22,
     "Tropical shallow lake; Microcystis fish kills."),
    ("kinneret", "Lake Kinneret", "IL", "train", "mesotrophic", (1, 12), 32.82, 35.59, 12,
     "Sea of Galilee; spring dinoflagellate and summer cyanobacteria blooms."),
    # ----------------------------------------------------------------- Europe ------
    ("peipsi", "Lake Peipsi", "EE/RU", "train", "eutrophic", (6, 10), 58.40, 27.45, 45,
     "Large shallow transboundary lake; Gloeotrichia and Microcystis blooms."),
    ("vortsjarv", "Lake Vortsjarv", "EE", "train", "eutrophic", (6, 10), 58.28, 26.02, 18,
     "Shallow, turbid, Limnothrix-dominated."),
    ("malaren", "Lake Malaren", "SE", "train", "mesotrophic", (6, 9), 59.45, 17.20, 45,
     "Fragmented basin; eutrophic bays."),
    ("szczecin_lagoon", "Szczecin Lagoon", "PL/DE", "train", "hypereutrophic", (6, 9), 53.77,
     14.20, 25, "Oder estuary lagoon; dense summer cyanobacteria."),
    ("ladoga_south", "Lake Ladoga (Volkhov Bay)", "RU", "train", "mesotrophic", (6, 9), 60.15,
     32.40, 30, "Southern shallows of the largest lake in Europe; river-fed blooms."),
    ("geneva", "Lake Geneva", "CH/FR", "val", "oligotrophic", (6, 10), 46.43, 6.55, 38,
     "Re-oligotrophicated; Planktothrix episodes. Oligotrophic validation lake."),
    ("garda", "Lake Garda", "IT", "train", "oligotrophic", (6, 10), 45.63, 10.67, 27,
     "Deep, clear; occasional Planktothrix/Dolichospermum surface blooms."),
    # ----------------------------------------------------------------- Africa ------
    ("lake_george_ug", "Lake George", "UG", "train", "hypereutrophic", (1, 12), 0.03, 30.20, 10,
     "Equatorial, permanently cyanobacteria-dominated."),
    ("tana", "Lake Tana", "ET", "train", "mesotrophic", (6, 11), 12.00, 37.30, 40,
     "Highland tropical lake, source of the Blue Nile."),
    # ---------------------------------------------------------- North America ------
    ("upper_klamath", "Upper Klamath Lake", "US", "train", "hypereutrophic", (6, 10), 42.40,
     -121.88, 18, "Aphanizomenon blooms every summer."),
    ("winnebago", "Lake Winnebago", "US", "train", "eutrophic", (6, 10), 44.00, -88.43, 25,
     "Shallow, agricultural catchment; recurrent blooms."),
    ("lake_of_the_woods", "Lake of the Woods", "CA/US", "train", "eutrophic", (7, 10), 49.00,
     -94.80, 35, "Southern basin (Big Traverse Bay) blooms."),
    ("tahoe", "Lake Tahoe", "US", "train", "oligotrophic", (6, 10), 39.09, -120.04, 18,
     "Ultra-oligotrophic: the clear-water end of the training set."),
    ("valencia_ve", "Lake Valencia", "VE", "train", "hypereutrophic", (1, 12), 10.18, -67.73, 16,
     "Tropical endorheic lake, persistent cyanobacteria."),
    # ---------------------------------------------------------- South America ------
    ("maracaibo", "Lake Maracaibo", "VE", "train", "eutrophic", (1, 12), 9.80, -71.55, 40,
     "Tropical brackish lake; 2004 and later basin-wide cyanobacteria blooms."),
    ("titicaca_puno", "Lake Titicaca (Puno Bay)", "PE", "train", "mesotrophic", (1, 12), -15.82,
     -69.95, 20, "3,810 m; eutrophied inner bay against an oligotrophic main lake."),
    ("guaiba", "Lake Guaiba", "BR", "train", "eutrophic", (11, 4), -30.20, -51.23, 25,
     "Porto Alegre supply; summer cyanobacteria advisories."),
    # ---------------------------------------------------------------- Oceania ------
    ("taupo", "Lake Taupo", "NZ", "train", "oligotrophic", (11, 4), -38.80, 175.90, 18,
     "Large clear caldera lake; southern-hemisphere oligotrophic reference."),
    ("te_waihora", "Te Waihora / Lake Ellesmere", "NZ", "train", "hypereutrophic", (11, 4),
     -43.78, 172.33, 14, "Shallow coastal lake; hypertrophic, frequent blooms."),
]

# Registry country code -> continent (habforecast.groups.CONTINENTS). A shared border
# code ("US/CA") takes its first country.
CONTINENT_BY_COUNTRY = {
    "US": "north_america", "CA": "north_america", "MX": "north_america",
    "VE": "south_america", "PE": "south_america", "BR": "south_america",
    "CN": "asia", "JP": "asia", "PH": "asia", "IL": "asia",
    "HU": "europe", "NL": "europe", "LT": "europe", "GB": "europe", "EE": "europe",
    "SE": "europe", "PL": "europe", "RU": "europe", "CH": "europe", "IT": "europe",
    "KE": "africa", "TZ": "africa", "ZA": "africa", "UG": "africa", "ET": "africa",
    "AU": "oceania", "NZ": "oceania",
}


def continent_of(country: str) -> str:
    return CONTINENT_BY_COUNTRY.get(country.split("/")[0], "unknown")


def half_width_deg(km: float, lat: float) -> tuple[float, float]:
    import math

    return km / (111.320 * math.cos(math.radians(lat))), km / 110.574


def utm_epsg(lat: float, lon: float) -> int:
    zone = int((lon + 180.0) // 6.0) + 1
    return (32600 if lat >= 0 else 32700) + zone


def build(lakes: list[tuple] | None = None) -> dict:
    features = []
    for lake_id, name, country, split, trophic, season, lat, lon, km, notes in lakes or LAKES:
        dlon, dlat = half_width_deg(km, lat)
        w, e, s, n = lon - dlon, lon + dlon, lat - dlat, lat + dlat
        features.append(
            {
                "type": "Feature",
                "properties": {
                    "lake_id": lake_id,
                    "name": name,
                    "country": country,
                    "split": split,
                    "trophic_state": trophic,
                    "continent": continent_of(country),
                    "bloom_season": list(season),
                    "utm_epsg": utm_epsg(lat, lon),
                    "notes": notes,
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[w, s], [e, s], [e, n], [w, n], [w, s]]],
                },
            }
        )
    return {"type": "FeatureCollection", "features": features}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument(
        "--set",
        choices=["expanded", "global"],
        default="expanded",
        help="expanded: the 21-lake registry; global: those plus GLOBAL_LAKES",
    )
    args = ap.parse_args()
    lakes = LAKES + GLOBAL_LAKES if args.set == "global" else LAKES
    args.out = args.out or Path(f"data/aoi/lakes_{args.set}.geojson")
    # Thermal class comes from each lake's weather (scripts/annotate_groups.py), which
    # does not exist until the met stores are built; carry forward any already set.
    previous = {}
    if args.out.exists():
        previous = {
            f["properties"]["lake_id"]: f["properties"].get("thermal_class")
            for f in json.loads(args.out.read_text(encoding="utf-8"))["features"]
        }
    doc = build(lakes)
    if args.set == "global":
        # The 21 existing lakes are taken verbatim from the expanded registry, not rebuilt:
        # the original five were drawn by hand before this script existed, and a polygon
        # that moves by one pixel gives a grid that no longer matches the lake's cube.
        base = json.loads(Path("data/aoi/lakes_expanded.geojson").read_text(encoding="utf-8"))
        # And the original five come from lakes.geojson, whose polygons their cubes were
        # ingested on. lakes_expanded.geojson rebuilt them from centre and half-width, which
        # gives different grids (erie_west 309x309 against its 267x298 cube); training never
        # noticed because it reads cubes by lake_id, but an ingest appending to them would.
        orig = json.loads(Path("data/aoi/lakes.geojson").read_text(encoding="utf-8"))
        orig_by_id = {f["properties"]["lake_id"]: f for f in orig["features"]}
        base["features"] = [
            {**f, "geometry": orig_by_id[f["properties"]["lake_id"]]["geometry"]}
            if f["properties"]["lake_id"] in orig_by_id
            else f
            for f in base["features"]
        ]
        kept = {f["properties"]["lake_id"] for f in base["features"]}
        for f in base["features"]:
            f["properties"]["continent"] = continent_of(f["properties"]["country"])
        doc["features"] = base["features"] + [
            f for f in doc["features"] if f["properties"]["lake_id"] not in kept
        ]
    for f in doc["features"]:
        if previous.get(f["properties"]["lake_id"]):
            f["properties"]["thermal_class"] = previous[f["properties"]["lake_id"]]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    counts: dict[str, int] = {}
    for f in doc["features"]:
        counts[f["properties"]["split"]] = counts.get(f["properties"]["split"], 0) + 1
    print(f"{len(doc['features'])} lakes -> {args.out}")
    for k in ("train", "val", "test"):
        print(f"  {k:6} {counts.get(k, 0)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
