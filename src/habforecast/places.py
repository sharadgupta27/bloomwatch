"""Turning what a user types into a point on the water.

The app asks for "a location". People type three different things, and all three have to
work without shipping a dependency on an external geocoding service:

1. coordinates -- "41.75, -83.3", "41.75N 83.3W", "-0.3 34.2";
2. the name of a well-known water body -- "erie", "taihu", "okeechobee";
3. the name of anywhere else -- a city, a river, a bay: "Toledo Ohio", "Bengaluru".

The first two are answered locally. The third needs a real geocoder, so `geocode` falls
through to OpenStreetMap's Nominatim, and `find` is the entry point that tries all three
in order. The gazetteer still goes first: its entries are bloom-prone waters with centres
chosen to sit on open water, while a geocoder's label anchor for a lake can land on the
shoreline.

That fallback sends the typed query -- and only the typed query -- to a third party.
Setting HAB_GEOCODER=off disables it and restores the self-contained behaviour.

The coordinates are approximate centres, good to a few kilometres, which is the right
precision for a 300 m grid with a 10-25 km radius around it. They are a starting point the
user can drag, not a survey.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Place:
    name: str
    lat: float
    lon: float
    country: str = ""
    note: str = ""
    # Present when the source knows the feature's extent -- a lake or a city has one, a
    # plain coordinate does not. The app frames the map on it instead of guessing a radius.
    bbox: tuple[float, float, float, float] | None = None
    source: str = "gazetteer"

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "lat": self.lat,
            "lon": self.lon,
            "country": self.country,
            "note": self.note,
            "bbox": list(self.bbox) if self.bbox else None,
            "source": self.source,
        }


# Water bodies with a documented history of cyanobacterial or harmful algal blooms, plus
# the large lakes people are most likely to ask about. Not exhaustive, and not meant to
# be -- it is a shortcut for common cases, not a world gazetteer.
GAZETTEER: tuple[Place, ...] = (
    # --- North America -------------------------------------------------------------
    Place("Lake Erie (western basin)", 41.75, -83.30, "US/CA", "annual Microcystis blooms"),
    Place("Lake Erie (central basin)", 41.90, -81.60, "US/CA"),
    Place("Lake Okeechobee", 26.95, -80.80, "US", "recurring cyanobacteria blooms"),
    Place("Lake Michigan (Green Bay)", 44.75, -87.90, "US"),
    Place("Lake Ontario (Bay of Quinte)", 44.10, -77.10, "CA"),
    Place("Lake Huron (Saginaw Bay)", 43.85, -83.65, "US"),
    Place("Lake Superior", 47.70, -87.50, "US/CA"),
    Place("Lake Winnipeg", 52.10, -97.30, "CA", "large summer cyanobacteria blooms"),
    Place("Lake Champlain", 44.55, -73.30, "US"),
    Place("Utah Lake", 40.22, -111.80, "US"),
    Place("Great Salt Lake", 41.10, -112.50, "US"),
    Place("Clear Lake, California", 39.02, -122.78, "US"),
    Place("Salton Sea", 33.30, -115.85, "US"),
    Place("Chesapeake Bay", 38.30, -76.30, "US"),
    Place("Lake Pontchartrain", 30.20, -90.10, "US"),
    Place("Lake Apopka", 28.62, -81.62, "US"),
    Place("Lake Tahoe", 39.10, -120.03, "US"),
    Place("Lake Nicaragua", 11.60, -85.40, "NI"),
    Place("Lake Chapala", 20.25, -103.05, "MX"),
    # --- South America -------------------------------------------------------------
    Place("Lake Titicaca", -15.80, -69.35, "PE/BO"),
    Place("Laguna Mar Chiquita", -30.70, -62.60, "AR"),
    Place("Rio de la Plata", -34.80, -57.50, "AR/UY"),
    Place("Lake Maracaibo", 9.80, -71.55, "VE"),
    # --- Europe ---------------------------------------------------------------------
    Place("Lake Balaton", 46.85, 17.75, "HU"),
    Place("Lake Vanern", 58.90, 13.30, "SE"),
    Place("Lake Vattern", 58.30, 14.55, "SE"),
    Place("Lake Peipus", 58.70, 27.50, "EE/RU"),
    Place("Lake Geneva", 46.45, 6.50, "CH/FR"),
    Place("Lake Constance", 47.60, 9.35, "DE/AT/CH"),
    Place("IJsselmeer", 52.75, 5.35, "NL"),
    Place("Curonian Lagoon", 55.30, 21.05, "LT/RU", "frequent cyanobacteria blooms"),
    Place("Gulf of Finland", 59.90, 25.00, "FI/EE"),
    Place("Baltic Sea (Gotland basin)", 57.30, 19.50, "", "annual cyanobacteria blooms"),
    Place("Lake Neusiedl", 47.80, 16.75, "AT/HU"),
    Place("Lough Neagh", 54.62, -6.42, "GB", "severe blooms since 2023"),
    Place("Lake Trasimeno", 43.13, 12.10, "IT"),
    Place("Lake Garda", 45.65, 10.68, "IT"),
    Place("Albufera de Valencia", 39.33, -0.35, "ES"),
    # --- Africa ----------------------------------------------------------------------
    Place("Lake Victoria (Winam Gulf)", -0.30, 34.20, "KE", "persistent blooms"),
    Place("Lake Victoria (Mwanza Gulf)", -2.55, 32.90, "TZ"),
    Place("Lake Victoria (northern)", -0.10, 32.90, "UG"),
    Place("Lake Tanganyika", -6.00, 29.60, "TZ/CD/BI/ZM"),
    Place("Lake Malawi", -12.20, 34.55, "MW/MZ/TZ"),
    Place("Lake Chad", 13.20, 14.20, "TD/NG/NE/CM"),
    Place("Lake Nasser", 22.80, 32.20, "EG"),
    Place("Lake Turkana", 3.50, 36.10, "KE"),
    Place("Hartbeespoort Dam", -25.74, 27.85, "ZA", "hypertrophic, chronic blooms"),
    Place("Lake Naivasha", -0.77, 36.35, "KE"),
    # --- Asia -------------------------------------------------------------------------
    Place("Lake Taihu", 31.20, 120.20, "CN", "major 2007 bloom crisis"),
    Place("Lake Chaohu", 31.55, 117.55, "CN"),
    Place("Lake Dianchi", 24.83, 102.70, "CN", "hypertrophic"),
    Place("Lake Poyang", 29.10, 116.25, "CN"),
    Place("Lake Dongting", 29.35, 112.90, "CN"),
    Place("Lake Biwa", 35.30, 136.15, "JP"),
    Place("Lake Kasumigaura", 36.03, 140.40, "JP"),
    Place("Lake Baikal", 53.50, 108.00, "RU"),
    Place("Aral Sea (North)", 46.30, 60.60, "KZ"),
    Place("Lake Balkhash", 46.30, 74.50, "KZ"),
    Place("Lake Van", 38.65, 42.95, "TR"),
    Place("Lake Urmia", 37.60, 45.50, "IR"),
    Place("Sea of Galilee", 32.80, 35.59, "IL"),
    Place("Chilika Lake", 19.72, 85.32, "IN"),
    Place("Vembanad Lake", 9.75, 76.40, "IN"),
    Place("Hussain Sagar", 17.42, 78.47, "IN", "hypertrophic urban lake"),
    Place("Lake Tonle Sap", 12.90, 104.10, "KH"),
    Place("Laguna de Bay", 14.38, 121.25, "PH"),
    # --- Oceania -----------------------------------------------------------------------
    Place("Lake Taupo", -38.80, 175.90, "NZ"),
    Place("Lake Rotorua", -38.08, 176.27, "NZ"),
    Place("Lake Alexandrina", -35.45, 139.20, "AU"),
    Place("Gippsland Lakes", -38.00, 147.60, "AU", "recurrent Nodularia blooms"),
    Place("Moreton Bay", -27.35, 153.30, "AU"),
)


def parse_coordinates(text: str) -> tuple[float, float] | None:
    """Parse "41.75, -83.3" / "41.75N 83.3W" / "-0.3 34.2" into (lat, lon).

    Returns None when the text is not a coordinate pair, so the caller can fall back to a
    name search rather than guessing.
    """
    t = text.strip().replace("°", " ")
    # The two numbers must be separated by a comma/semicolon or by whitespace. Making the
    # separator optional looks harmless and is not: "41.75" then parses as latitude 4,
    # longitude 1.75 -- a valid-looking point in the Gulf of Guinea, silently returned
    # instead of falling through to a name search.
    m = re.match(
        r"^([-+]?\d{1,3}(?:\.\d+)?)\s*([NnSs])?"
        r"(?:\s*[,;]\s*|\s+)"
        r"([-+]?\d{1,3}(?:\.\d+)?)\s*([EeWw])?$",
        t,
    )
    if not m:
        return None
    lat = float(m.group(1))
    lon = float(m.group(3))
    if (ns := m.group(2)) and ns.lower() == "s":
        lat = -abs(lat)
    elif ns:
        lat = abs(lat)
    if (ew := m.group(4)) and ew.lower() == "w":
        lon = -abs(lon)
    elif ew:
        lon = abs(lon)
    if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
        return None
    return lat, lon


def _fold(s: str) -> str:
    """Casefold and strip accents, so "Vanern" matches "Vänern"."""
    nfkd = unicodedata.normalize("NFKD", s)
    return "".join(c for c in nfkd if not unicodedata.combining(c)).casefold()


def search(query: str, limit: int = 8) -> list[Place]:
    """Rank gazetteer entries against a free-text query.

    Prefix matches on a word beat matches buried mid-string, so typing "erie" surfaces
    Lake Erie rather than something that merely contains those letters.
    """
    q = _fold(query).strip()
    if not q:
        return []
    scored: list[tuple[int, int, Place]] = []
    for p in GAZETTEER:
        hay = _fold(f"{p.name} {p.country} {p.note}")
        idx = hay.find(q)
        if idx < 0:
            continue
        words = hay.replace("(", " ").replace(")", " ").split()
        starts_word = any(w.startswith(q) for w in words)
        scored.append((0 if starts_word else 1, idx, p))
    scored.sort(key=lambda t: (t[0], t[1], t[2].name))
    return [p for _, _, p in scored[:limit]]


# ------------------------------------------------------------------------ geocoding ---
# The gazetteer above covers the water bodies this model is about. It does not know
# cities, and "search for a place by name" is a reasonable thing to expect of a map, so
# unknown queries fall through to OpenStreetMap's Nominatim.
#
# What that costs, stated plainly because it is a change in the application's behaviour:
# the text typed into the search box leaves this container and goes to a third party.
# Nothing else does -- no coordinates the user clicked, no forecast, no identity. The
# geocoder is also optional: set HAB_GEOCODER=off and the app falls back to the gazetteer
# and coordinate parsing alone, which is how it worked before.
#
# Nominatim's usage policy requires an identifying User-Agent and at most one request per
# second, and asks that results be cached. All three are honoured below.
NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
GEOCODER_USER_AGENT = "habforecast/0.1 (cyanobacteria bloom forecasting; self-hosted)"
GEOCODER_TIMEOUT_S = 6.0
GEOCODER_MIN_INTERVAL_S = 1.0
GEOCODER_CACHE_SIZE = 512

# Feature classes worth offering. A bloom forecast needs water or somewhere near water; a
# shop or a bus stop is noise, and returning it would push a real lake off the list.
_USEFUL_CLASSES = {"place", "boundary", "natural", "water", "waterway", "landuse"}

_geocode_cache: dict[str, list[Place]] = {}
_last_call = [0.0]


def geocoder_enabled() -> bool:
    import os

    return os.getenv("HAB_GEOCODER", "on").strip().lower() not in ("0", "off", "false", "no")


def geocode(query: str, limit: int = 5) -> list[Place]:
    """Look a free-text place up with Nominatim. Never raises: search must not break."""
    import json
    import os
    import time
    import urllib.error
    import urllib.parse
    import urllib.request

    q = query.strip()
    if not q or not geocoder_enabled():
        return []
    key = f"{q.casefold()}|{limit}"
    if key in _geocode_cache:
        return _geocode_cache[key]

    # One request per second, enforced here rather than trusted to the caller.
    wait = GEOCODER_MIN_INTERVAL_S - (time.monotonic() - _last_call[0])
    if wait > 0:
        time.sleep(min(wait, GEOCODER_MIN_INTERVAL_S))
    _last_call[0] = time.monotonic()

    params = urllib.parse.urlencode(
        {"q": q, "format": "jsonv2", "limit": str(limit), "addressdetails": "0"}
    )
    url = f"{os.getenv('HAB_GEOCODER_URL', NOMINATIM_URL)}?{params}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": GEOCODER_USER_AGENT})
        with urllib.request.urlopen(req, timeout=GEOCODER_TIMEOUT_S) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
        logger.info("geocoder unavailable for %r: %s", q, exc)
        return []

    out: list[Place] = []
    for row in payload if isinstance(payload, list) else []:
        try:
            lat, lon = float(row["lat"]), float(row["lon"])
        except (KeyError, TypeError, ValueError):
            continue
        display = str(row.get("display_name", ""))
        head, _, tail = display.partition(", ")
        country = tail.rsplit(", ", 1)[-1] if tail else ""
        out.append(
            Place(
                name=head or display[:60],
                lat=lat,
                lon=lon,
                country=country,
                note=str(row.get("type", "")).replace("_", " "),
                bbox=_nominatim_bbox(row.get("boundingbox")),
                source="openstreetmap",
            )
        )
    ranked = sorted(out, key=lambda p: 0 if _is_useful(p) else 1)
    if len(_geocode_cache) >= GEOCODER_CACHE_SIZE:
        _geocode_cache.clear()
    _geocode_cache[key] = ranked
    return ranked


def _is_useful(place: Place) -> bool:
    return place.note.split()[0] in _USEFUL_CLASSES if place.note else False


def _nominatim_bbox(raw) -> tuple[float, float, float, float] | None:
    """Nominatim orders its box [south, north, west, east]; ours is west/south/east/north."""
    try:
        south, north, west, east = (float(v) for v in raw)
    except (TypeError, ValueError):
        return None
    if east <= west or north <= south:
        return None
    return (west, south, east, north)


def find(query: str, limit: int = 8) -> list[Place]:
    """The one entry point the API uses: coordinates, then gazetteer, then geocoder.

    The gazetteer goes first even though the geocoder is more general. Its entries are
    curated bloom-prone waters with centres chosen to sit on open water, whereas a
    geocoder's "Lake Erie" is a label anchor that can land on a shoreline -- and for the
    lakes this model was evaluated on, the curated point is the better answer.
    """
    coords = parse_coordinates(query)
    if coords:
        lat, lon = coords
        return [Place(name=f"{lat:.4f}, {lon:.4f}", lat=lat, lon=lon, note="coordinates")]
    local = search(query, limit=limit)
    if len(local) >= limit:
        return local
    seen = {(round(p.lat, 2), round(p.lon, 2)) for p in local}
    for p in geocode(query, limit=limit - len(local)):
        if (round(p.lat, 2), round(p.lon, 2)) not in seen:
            local.append(p)
    return local[:limit]
