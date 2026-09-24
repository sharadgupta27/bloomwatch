"""Near-real-time OLCI chlorophyll from NOAA CoastWatch ERDDAP.

Why this exists
---------------
The Planetary Computer mirror this project ingests from is an *archive*: measured
2026-08-28, its newest Sentinel-3 OLCI L2 WFR granule was 2026-02-25, six months old.
That is a property of the mirror, not of the mission -- OLCI is operational near-real
time, with EUMETSAT publishing NRT products within about three hours of acquisition.
Asking the app to forecast from today and being handed February is the visible symptom.

NOAA CoastWatch republishes Sentinel-3 OLCI chlorophyll as daily Level-3 composites on a
0.0025 degree (~300 m) grid, split into lettered sectors, for **both S3A and S3B**, over
ERDDAP. Three properties make it the right NRT source here:

* **No credentials.** Copernicus Data Space carries the same products within hours, but
  downloading needs a registered account (OAuth2); its catalogue is public but its bytes
  are not. CoastWatch is open, so the container works out of the box.
* **Server-side subsetting.** ERDDAP griddap takes a bounding box and a time range and
  returns only those cells. A 12 km lake for a fortnight is a few hundred kB, against a
  full OLCI granule at ~700 MB, of which we keep a thousandth.
* **Already quality-masked and gridded**, so a live request does no swath resampling.

What it costs
-------------
* **A 90-day rolling window only.** This source can serve a live forecast; it cannot
  serve training or historical evaluation. The Planetary Computer path stays for those,
  and `adhoc` picks between them by the date being asked for.
* **No WQSF flags.** The composites arrive pre-masked, so the project's own QC
  (`data.qc`) and its flag-frequency drift telemetry do not apply to this path. A pixel is
  either present or NaN.
* **The retrieval algorithm is not documented in the product metadata.** The model was
  trained on CHL_NN, the neural-net product for optically complex water. Measured over
  Taihu on 2026-08-26 this source reports a median of 18.6 mg/m3, which sits squarely in
  the distribution the model was trained on for that lake -- but "consistent" is not
  "identical", and it is the reason `AdhocResult` reports which source it used. The
  forecaster normalises each input window to its own mean and standard deviation, so a
  systematic offset or scale difference between algorithms largely cancels; a difference
  in *spatial texture* would not.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

ERDDAP = "https://coastwatch.noaa.gov/erddap"
# ERDDAP asks for an identifying agent, and CoastWatch's instance *refuses* several --
# measured: an agent naming python is answered 403 while the identical request with any
# other agent returns 200. So this names the project and its purpose without the token
# that gets filtered. It carries no user identity and nothing from the request.
USER_AGENT = "habforecast/0.1 (bloom forecasting; contact via repository)"

# The sector datasets come in two flavours: ~300 m tiles (0.0025 deg) and coarser
# continental overviews that share the naming scheme. Anything coarser than this is the
# wrong product for a 300 m forecast grid.
MAX_SECTOR_RESOLUTION_DEG = 0.005

SECTOR_CACHE_TTL_H = 24.0
# A product missing from the catalogue is remembered too, not only one that is present.
# Measured 2026-09-23 over Lake Superior: with olci-300m withdrawn, asking again cost 33 s
# on the first request (a 503 retry before the 404) and 2.5 s on every request after --
# all of it spent inside "locating the most recent imagery", to learn the same answer.
# An hour is short enough that a product NOAA restores is picked up the same afternoon.
WITHDRAWN_TTL_S = 3600.0
_withdrawn: dict[str, float] = {}
_withdrawn_lock = threading.Lock()
HTTP_TIMEOUT_S = 120.0
# Imagery for a past fortnight is immutable, so a stored copy stays correct; the ceiling
# is about the newest day appearing, not about old days changing.
GRID_CACHE_TTL_H = 12.0


# ---------------------------------------------------------------------- products ----
# The resolution ladder this project can actually train on, all of it public and none of
# it needing an account. Two things make these usable where a granule archive is not:
# ERDDAP subsets server-side, so ingesting one lake is a bounding-box request rather than
# a 700 MB download, and every product is already gridded and quality-masked.
#
# Measured 2026-08-28. `sector_pattern` selects per-tile datasets whose footprint must be
# matched against the AOI; `datasets` names a single global grid.


@dataclass(frozen=True)
class Product:
    """One ERDDAP chlorophyll product: a resolution, a record, and how to address it."""

    key: str
    label: str
    resolution_m: float
    sector_pattern: str | None = None
    datasets: tuple[str, ...] = ()
    record: str = ""
    note: str = ""
    # The last day this product published, or None while it is still being updated.
    # Measured against the live sector index over Lake Erie on 2026-08-30; a stopped
    # product still serves history perfectly well, and the router needs to know which is
    # which so it does not offer year-old imagery as a live forecast.
    record_end: date | None = None

    @property
    def sectored(self) -> bool:
        return self.sector_pattern is not None


PRODUCTS: dict[str, Product] = {
    "olci-300m": Product(
        key="olci-300m",
        label="Sentinel-3 OLCI, 300 m",
        resolution_m=300.0,
        sector_pattern="noaacwS3.OLCIchlaSector.*",
        record="rolling 90 days",
        note=(
            "Native OLCI resolution and the only near-real-time option here, but the "
            "rolling window makes it unusable for training. S3A and S3B both."
        ),
    ),
    "viirs-750m": Product(
        key="viirs-750m",
        label="VIIRS science quality, 750 m",
        resolution_m=750.0,
        sector_pattern="noaacwNPPVIIRSSCIchlaSector.*",
        record="2012-01 to 2025-08",
        record_end=date(2025, 8, 19),
        note=(
            "The deepest fine-resolution record available without an account: thirteen "
            "years, against eight for the OLCI archive. A different sensor and a "
            "different retrieval, so a model trained here is not a model trained on "
            "OLCI -- but at 750 m a 20 km lake is still 27 pixels across."
        ),
    ),
    "dineof-2km": Product(
        key="dineof-2km",
        label="VIIRS + OLCI gap-filled (DINEOF), 2 km",
        resolution_m=2000.0,
        datasets=("noaacwNPPN20S3ASCIDINEOF2kmDaily",),
        record="2018-01 to 2026-08",
        note=(
            "Cloud gaps are interpolated away by DINEOF. That is a different learning "
            "problem -- the model would stop having to reason about missingness, which "
            "is most of what this one does -- so it is offered for comparison, not as a "
            "drop-in replacement."
        ),
    ),
    "olci-4km": Product(
        key="olci-4km",
        label="Sentinel-3 OLCI, global 4 km",
        resolution_m=4000.0,
        datasets=("noaacwS3AOLCIchlaDaily", "noaacwS3BOLCIchlaDaily"),
        record="2019-06 to 2026-08",
        note=(
            "Same sensor and family as the 300 m training data, seven years deep, and "
            "cheap to ingest anywhere on Earth. At 4 km most lakes are a handful of "
            "pixels, so it buys breadth at the cost of everything spatial."
        ),
    ),
    "viirs-4km": Product(
        key="viirs-4km",
        label="VIIRS science quality, global 4 km",
        resolution_m=4000.0,
        datasets=("noaacwNPPVIIRSSQchlaDaily",),
        record="2012-01 to 2026-08",
        note="Fourteen years global. The longest record reachable here.",
    ),
}

DEFAULT_PRODUCT = "olci-300m"


@dataclass(frozen=True)
class Sector:
    """One ERDDAP griddap dataset covering a lat/lon tile for one satellite."""

    dataset_id: str
    west: float
    south: float
    east: float
    north: float
    min_time: str
    max_time: str
    resolution_deg: float

    @property
    def satellite(self) -> str:
        # noaacwS3AOLCIchlaSectorKFDaily -> S3A
        return self.dataset_id[6:9]

    @property
    def newest(self) -> date | None:
        return _iso_date(self.max_time)

    @property
    def oldest(self) -> date | None:
        return _iso_date(self.min_time)

    def covers(self, west: float, south: float, east: float, north: float) -> bool:
        return not (
            east < self.west or west > self.east or north < self.south or south > self.north
        )


class SourceUnavailableError(RuntimeError):
    """Every read of a window failed in transport, so nothing is known about it.

    Distinct from "this window holds no imagery", and the distinction is worth an
    exception class because the two demand opposite responses. No imagery means step back
    a fortnight and ask again. A server refusing to answer means the next fortnight will
    be refused too -- and each refusal, measured against CoastWatch's proxy during an
    outage on 2026-08-29, costs a full minute before it fails. Walking the anchor ladder
    through that turned a request into a six-minute wait for an answer that was never
    going to come.
    """


# A 502 from CoastWatch's proxy is a bad minute, not a verdict: the same URL that took
# 60 s to fail during the 2026-08-29 outage answered in 1.5 s once it had passed.
#
# But a retry is only cheap when the failure is cheap, and that is the trap here. The
# first version of this retried three times unconditionally, which turned a single 60 s
# proxy timeout into a 186 s one -- measured, and worse than having no retry at all. So
# the loop is bounded by a *total* budget rather than an attempt count: a fast failure is
# retried, a slow one is reported, and the caller is never made to wait longer for the
# bad news than it would have waited without any of this.
RETRY_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})
RETRY_BACKOFF_S = (1.0, 3.0)
# What one *live* griddap subset may cost in total, retries included. Healthy fetches for
# a lake-sized box over a fortnight measured 1.5-10 s, so this is generous; past it the
# honest answer is "this source is not responding", which the archive can act on. Bulk
# ingest asks for far larger windows and passes its own budget.
GRID_BUDGET_S = 30.0


# --- circuit breaker ----------------------------------------------------------------
# Finding out that a server is down costs a full budget. Finding out twice costs two, and
# a user clicking down a list of lakes during a CoastWatch outage would pay it on every
# one of them. So the outage is remembered: after two consecutive transport failures the
# source is treated as down and the next request fails instantly instead of waiting to be
# told the same thing again.
#
# The cooldown is short on purpose. These outages are minutes, not hours -- the URL that
# took 60 s to 502 on 2026-08-29 answered in 1.5 s once the proxy recovered -- and the
# cost of probing again too early is one slow request, while the cost of staying open too
# long is serving the archive when live imagery is available.
BREAKER_FAILURES = 2
BREAKER_COOLDOWN_S = 180.0

_breaker = {"failures": 0, "opened_at": 0.0}
_breaker_lock = threading.Lock()


def breaker_state() -> dict:
    """For /health and for tests: is the live source currently cut off, and for how long."""
    with _breaker_lock:
        left = 0.0
        if _breaker["opened_at"]:
            left = max(0.0, BREAKER_COOLDOWN_S - (time.monotonic() - _breaker["opened_at"]))
        return {
            "open": left > 0,
            "seconds_remaining": round(left, 1),
            "consecutive_failures": int(_breaker["failures"]),
        }


def reset_breaker() -> None:
    with _breaker_lock:
        _breaker["failures"] = 0
        _breaker["opened_at"] = 0.0


def _breaker_seconds_left() -> float:
    with _breaker_lock:
        if not _breaker["opened_at"]:
            return 0.0
        left = BREAKER_COOLDOWN_S - (time.monotonic() - _breaker["opened_at"])
        if left <= 0:
            # Half-open: let exactly one request through to find out whether it is back.
            _breaker["opened_at"] = 0.0
            _breaker["failures"] = 0
            return 0.0
        return left


def _breaker_record(ok: bool, count: int = 1) -> None:
    """Record evidence. `count` is per failed sector, not per request.

    Two satellites refused in the same second are two independent observations that the
    server is down, not one -- and since each observation costs a full budget to make,
    treating a request as a single data point means paying the discovery twice.
    """
    with _breaker_lock:
        if ok:
            _breaker["failures"] = 0
            _breaker["opened_at"] = 0.0
            return
        _breaker["failures"] = int(_breaker["failures"]) + count
        if _breaker["failures"] >= BREAKER_FAILURES:
            _breaker["opened_at"] = time.monotonic()


def _cache_path(url: str, root: Path | None) -> Path:
    base = root or Path(os.getenv("HAB_HTTP_CACHE", "data/cache/erddap"))
    return base / f"{hashlib.sha1(url.encode('utf-8')).hexdigest()}.bin"


def _cached(url: str, ttl_h: float, root: Path | None) -> bytes | None:
    p = _cache_path(url, root)
    try:
        if not p.exists():
            return None
        if ttl_h <= 0:
            return None
        age_h = (datetime.now(UTC).timestamp() - p.stat().st_mtime) / 3600.0
        # `>=` rather than `>`: on a coarse filesystem clock a file written this instant
        # can report an age of exactly zero, or even slightly negative.
        if age_h >= ttl_h:
            return None
        return p.read_bytes()
    except OSError:
        return None


def _store(url: str, payload: bytes, root: Path | None) -> None:
    p = _cache_path(url, root)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        # Written beside and renamed, so a process killed mid-write cannot leave a
        # truncated NetCDF that every later run would happily read as the real thing.
        tmp = p.with_suffix(".part")
        tmp.write_bytes(payload)
        tmp.replace(p)
    except OSError as exc:
        logger.debug("could not cache %s: %s", url, exc)


def _get(
    url: str,
    timeout: float = HTTP_TIMEOUT_S,
    attempts: int = 3,
    budget_s: float | None = None,
    cache_ttl_h: float = 0.0,
    cache_root: Path | None = None,
) -> bytes:
    """Fetch a URL, retrying what is worth retrying and caching what is worth caching.

    `cache_ttl_h` above zero serves a stored copy when one is fresh enough. Chlorophyll
    for a past fortnight does not change, and the same lake is asked for again the moment
    a second person looks at it, so the cache turns the repeat case from tens of seconds
    into a disk read.

    `budget_s` caps the whole call, retries included. Without it a retry loop punishes
    exactly the case it was meant to help -- see the note on `GRID_BUDGET_S`.
    """
    if cache_ttl_h > 0:
        hit = _cached(url, cache_ttl_h, cache_root)
        if hit is not None:
            logger.debug("cache hit %s", url)
            return hit

    deadline = time.monotonic() + budget_s if budget_s else None
    last: Exception | None = None
    for attempt in range(attempts):
        left = timeout if deadline is None else min(timeout, deadline - time.monotonic())
        if left <= 0:
            break
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=left) as resp:
                payload = resp.read()
            if cache_ttl_h > 0:
                _store(url, payload, cache_root)
            return payload
        except urllib.error.HTTPError as exc:
            # 404 is ERDDAP's way of saying "your selection is empty", which is an answer
            # and not a fault. Retrying it would triple the cost of the commonest case.
            if exc.code not in RETRY_STATUS:
                raise
            last = exc
        except (TimeoutError, urllib.error.URLError, OSError) as exc:
            last = exc
        if attempt < attempts - 1:
            pause = RETRY_BACKOFF_S[min(attempt, len(RETRY_BACKOFF_S) - 1)]
            if deadline is not None and time.monotonic() + pause >= deadline:
                break
            time.sleep(pause)
            logger.info("retrying %s after %s", url.split("?")[0], last)
    raise last  # type: ignore[misc]


def _iso_date(value: str) -> date | None:
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).date()
    except (ValueError, AttributeError):
        return None


def _index_url(pattern: str) -> str:
    query = urllib.parse.quote(
        "datasetID,minLongitude,maxLongitude,minLatitude,maxLatitude,minTime,maxTime", safe=","
    )
    constraint = urllib.parse.quote(f'datasetID=~"{pattern}"', safe="=~")
    return f"{ERDDAP}/tabledap/allDatasets.json?{query}&{constraint}"


def sector_index(
    cache_dir: Path | None = None,
    refresh: bool = False,
    product: str = DEFAULT_PRODUCT,
) -> list[Sector]:
    """Every OLCI chlorophyll sector dataset, with its footprint and newest day.

    Discovered from ERDDAP rather than hard-coded: NOAA adds, retires and renames sectors,
    and a stale table baked into the image would silently start returning "no coverage"
    for places that are in fact covered. Cached for a day because the listing is ~60 kB
    and changes on the scale of months.
    """
    spec = PRODUCTS[product]
    pattern = spec.sector_pattern or f"^({'|'.join(spec.datasets)})$"
    cache = (cache_dir or Path("data/cache")) / f"coastwatch_{product}.json"
    if not refresh and cache.exists():
        age_h = (datetime.now(UTC).timestamp() - cache.stat().st_mtime) / 3600.0
        if age_h < SECTOR_CACHE_TTL_H:
            try:
                return [Sector(**s) for s in json.loads(cache.read_text(encoding="utf-8"))]
            except (json.JSONDecodeError, TypeError) as exc:
                logger.warning("sector cache unreadable (%s); refetching", exc)

    with _withdrawn_lock:
        seen = _withdrawn.get(product)
    if not refresh and seen is not None and time.monotonic() - seen < WITHDRAWN_TTL_S:
        raise SourceUnavailableError(
            f"CoastWatch listed no {product} datasets {time.monotonic() - seen:.0f}s ago; "
            "not asking again yet"
        )

    try:
        payload = json.loads(_get(_index_url(pattern)).decode("utf-8"))
    except urllib.error.HTTPError as exc:
        if exc.code != 404:
            raise
        with _withdrawn_lock:
            _withdrawn[product] = time.monotonic()
        # A 404 on the *catalogue* is not an empty window: it means no dataset on the
        # server matches this product any more. Measured 2026-09-23 for olci-300m: every
        # `noaacwS3?OLCIchlaSector*` dataset had left the catalogue, and the bare
        # HTTPError this used to raise surfaced as a 500 on every live request instead of
        # the "source unavailable" the API is built to report.
        raise SourceUnavailableError(
            f"CoastWatch lists no {product} datasets (pattern {pattern!r}); the product "
            "appears to have been withdrawn or renamed"
        ) from exc
    rows = payload["table"]["rows"]
    sectors: list[Sector] = []
    for ds_id, lon0, lon1, lat0, lat1, min_time, max_time in rows:
        if None in (lon0, lon1, lat0, lat1):
            continue
        # allDatasets reports the footprint but not the cell size; infer it from the tile
        # extent, which is what separates the 300 m tiles from the coarse overviews.
        # Degrees per cell, inferred from the tile extent: a sectored product's tiles are
        # small, a global grid spans the planet at a coarse step.
        span = max(float(lon1) - float(lon0), float(lat1) - float(lat0))
        resolution = spec.resolution_m / 111_320.0 if span <= 100.0 else 0.0375
        sectors.append(
            Sector(
                dataset_id=ds_id,
                west=float(lon0),
                south=float(lat0),
                east=float(lon1),
                north=float(lat1),
                min_time=min_time or "",
                max_time=max_time or "",
                resolution_deg=resolution,
            )
        )
    try:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps([s.__dict__ for s in sectors]), encoding="utf-8")
    except OSError as exc:  # a read-only volume must not break the request
        logger.info("could not cache the sector index: %s", exc)
    return sectors


def sectors_for(
    bbox: tuple[float, float, float, float],
    sectors: list[Sector] | None = None,
    product: str = DEFAULT_PRODUCT,
) -> list[Sector]:
    """The finest-resolution sectors covering `bbox`, one per satellite.

    Both S3A and S3B are returned when both cover the area: the two satellites are on
    offset orbits, so taking both roughly halves the revisit gap, and cloud on one pass is
    often absent on the other.
    """
    spec = PRODUCTS[product]
    pool = sectors if sectors is not None else sector_index(product=product)
    if spec.sectored:
        # Sector families mix fine tiles with coarse continental overviews under one
        # naming scheme; taking the coarse one silently halves the resolution.
        limit = (spec.resolution_m / 111_320.0) * 2.0
        pool = [s for s in pool if s.resolution_deg <= limit]
    hits = [s for s in pool if s.covers(*bbox)]
    best: dict[str, Sector] = {}
    for s in hits:
        current = best.get(s.satellite)
        if current is None or s.resolution_deg < current.resolution_deg:
            best[s.satellite] = s
    return [best[k] for k in sorted(best)]


def latest_available(
    bbox: tuple[float, float, float, float], product: str = DEFAULT_PRODUCT
) -> date | None:
    """Newest day this source holds for an area, or None if it does not cover it."""
    days = [s.newest for s in sectors_for(bbox, product=product) if s.newest]
    return max(days) if days else None


def _griddap_url(sector: Sector, bbox, start: date, end: date) -> str | None:
    """A griddap subset request, or None when the sector cannot answer this window.

    ERDDAP does not clamp an out-of-range constraint, it refuses the whole request with a
    404. Every axis therefore has to be clipped to what this particular dataset holds --
    and the two satellites do not hold the same days: measured 2026-08-28, S3B's Taihu
    sector ran to 08-27 and S3A's only to 08-26, so a window built from the later of the
    two killed the S3A request, which was the one with the data in it.
    """
    west, south, east, north = bbox
    west, east = max(west, sector.west), min(east, sector.east)
    south, north = max(south, sector.south), min(north, sector.north)
    if east <= west or north <= south:
        return None

    lo, hi = sector.oldest, sector.newest
    if (lo and end < lo) or (hi and start > hi):
        return None

    # The comparison ERDDAP makes is against the axis *timestamp*, not the day. A stop of
    # 2026-08-26T23:59:59Z is out of range for an axis whose last value is
    # 2026-08-26T01:24:51Z, even though the day is in range -- so the sector's own bound
    # is used verbatim whenever the request would reach past it.
    first = f"{start.isoformat()}T00:00:00Z"
    last = f"{end.isoformat()}T23:59:59Z"
    if lo and start <= lo:
        first = sector.min_time
    if hi and end >= hi:
        last = sector.max_time

    # Latitude is stored descending, and there is a singleton altitude axis between time
    # and latitude -- omitting it is a 404, which is how this was found.
    sel = f"chlor_a[({first}):({last})][0][({north}):({south})][({west}):({east})]"
    return f"{ERDDAP}/griddap/{sector.dataset_id}.nc?{urllib.parse.quote(sel, safe='[]():,')}"


@dataclass
class DayFrame:
    """One day's chlorophyll field on a target grid."""

    day: date
    log_chl: np.ndarray  # (H, W) float32, NaN where not retrieved
    valid: np.ndarray  # (H, W) bool


def _nearest_index(target: np.ndarray, source: np.ndarray) -> np.ndarray:
    """Index of the nearest `source` coordinate for each `target` value.

    The source axis is regular but may run in either direction, so it is sorted first and
    the answer mapped back. Nearest-neighbour is the honest choice: the source cells are
    ~300 m and the target cells are 300 m, so interpolating would invent a smoothness the
    sensor did not measure.
    """
    order = np.argsort(source)
    ordered = source[order]
    pos = np.clip(np.searchsorted(ordered, target), 1, len(ordered) - 1)
    left, right = ordered[pos - 1], ordered[pos]
    pick = np.where(np.abs(target - left) <= np.abs(right - target), pos - 1, pos)
    return order[pick]


class CoastWatchReader:
    """Daily NRT chlorophyll frames for an arbitrary grid."""

    source_name = "noaa-coastwatch-nrt"

    def __init__(self, cache_dir: Path | None = None, product: str = DEFAULT_PRODUCT):
        if product not in PRODUCTS:
            raise ValueError(f"unknown product {product!r}; try one of {sorted(PRODUCTS)}")
        self.product = product
        self.spec = PRODUCTS[product]
        self._sectors: list[Sector] | None = None
        self._cache_dir = cache_dir

    @property
    def sectors(self) -> list[Sector]:
        if self._sectors is None:
            self._sectors = sector_index(self._cache_dir, product=self.product)
        return self._sectors

    def available_through(self, bbox) -> date | None:
        days = [s.newest for s in sectors_for(bbox, self.sectors, self.product) if s.newest]
        return max(days) if days else None

    def earliest_available(self, bbox) -> date | None:
        days = [s.oldest for s in sectors_for(bbox, self.sectors, self.product) if s.oldest]
        return min(days) if days else None

    def frames(
        self, grid, start: date, end: date, progress=None, budget_s: float | None = GRID_BUDGET_S
    ) -> list[DayFrame]:
        """Fetch `start`..`end` for `grid`'s footprint and regrid onto it.

        Days present in both satellites are averaged in log space, which is where
        chlorophyll is roughly normal and where the model works.
        """
        bbox = grid.bounds_lonlat
        chosen = sectors_for(bbox, self.sectors, self.product)
        if not chosen:
            return []

        left = _breaker_seconds_left()
        if left:
            raise SourceUnavailableError(
                f"CoastWatch was unreachable moments ago; not retrying for {left:.0f}s."
            )

        lon, lat = grid.lonlat
        if progress:
            names = "+".join(s.satellite for s in chosen)
            progress(f"reading {names} near-real-time imagery", 0.25)

        # Concurrently, because these are two independent satellites answered by one
        # server and the request is entirely latency: serially, S3A and S3B cost the sum
        # of two round trips for no reason. The pool is sized to the work, which is two.
        def read(sector):
            return self._read_sector(sector, bbox, start, end, lon, lat, budget_s)

        per_day: dict[date, list[np.ndarray]] = {}
        failures: list[str] = []
        with ThreadPoolExecutor(max_workers=min(4, len(chosen))) as pool:
            for future in [pool.submit(read, s) for s in chosen]:
                try:
                    stack = future.result()
                except Exception as exc:  # one satellite failing must not sink the request
                    logger.warning("sector read failed: %s", exc)
                    failures.append(str(exc))
                    continue
                for day, field in stack.items():
                    per_day.setdefault(day, []).append(field)

        # Nothing came back and every attempt threw: this is a server that would not
        # answer, not a fortnight with no imagery in it. Saying so lets the caller stop
        # rather than ask the same server the same question about an earlier fortnight.
        if not per_day and failures and len(failures) == len(chosen):
            _breaker_record(ok=False, count=len(failures))
            raise SourceUnavailableError(
                f"CoastWatch did not answer for any of {len(chosen)} sector(s): {failures[0]}"
            )
        _breaker_record(ok=True)

        out: list[DayFrame] = []
        for day in sorted(per_day):
            fields = np.stack(per_day[day])
            # A pixel missing from every satellite that day is the normal case, not a
            # numerical accident: nanmean's "empty slice" warning would fire on most of
            # the grid on most days and drown the log.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                merged = np.nanmean(fields, axis=0).astype(np.float32)
            valid = np.isfinite(merged)
            if valid.any():
                out.append(DayFrame(day=day, log_chl=merged, valid=valid))
        return out

    def _read_sector(
        self, sector, bbox, start, end, lon, lat, budget_s: float | None = GRID_BUDGET_S
    ) -> dict[date, np.ndarray]:
        import urllib.error

        import xarray as xr

        url = _griddap_url(sector, bbox, start, end)
        if url is None:
            return {}
        logger.info("coastwatch fetch %s", url)
        try:
            raw = _get(url, budget_s=budget_s, cache_ttl_h=GRID_CACHE_TTL_H)
        except urllib.error.HTTPError as exc:
            # ERDDAP answers an empty selection with 404, not an empty file. A satellite
            # that simply did not pass over this lake in this fortnight is a normal
            # outcome, not a failure to report.
            if exc.code == 404:
                logger.info("no %s coverage for this window", sector.dataset_id)
                return {}
            raise
        ds = xr.open_dataset(raw if isinstance(raw, str) else _as_file(raw))
        try:
            chl = ds["chlor_a"]
            src_lat = np.asarray(ds["latitude"].values, dtype=np.float64)
            src_lon = np.asarray(ds["longitude"].values, dtype=np.float64)
            rows = _nearest_index(lat.ravel(), src_lat).reshape(lat.shape)
            cols = _nearest_index(lon.ravel(), src_lon).reshape(lon.shape)
            # A target cell outside the sector's returned window must be NaN, not the
            # nearest edge pixel smeared across the margin.
            inside = (
                (lat >= src_lat.min())
                & (lat <= src_lat.max())
                & (lon >= src_lon.min())
                & (lon <= src_lon.max())
            )
            frames: dict[date, np.ndarray] = {}
            for t in range(chl.sizes["time"]):
                day = _as_date(ds["time"].values[t])
                # Squeeze the singleton altitude axis by name. A bare .squeeze() also
                # drops latitude or longitude whenever the subset happens to be one cell
                # wide -- which turns the plane 1-D and makes the gather below raise
                # "too many indices". Small AOIs on coarse grids hit that routinely.
                plane = np.asarray(
                    chl.isel(time=t, altitude=0, missing_dims="ignore").values, dtype=np.float32
                )
                if plane.ndim == 1:  # a genuinely one-row or one-column subset
                    plane = plane.reshape(len(src_lat), len(src_lon))
                sampled = plane[rows, cols]
                sampled = np.where(inside & (sampled > 0), sampled, np.nan)
                with np.errstate(divide="ignore", invalid="ignore"):
                    frames[day] = np.log10(sampled).astype(np.float32)
            return frames
        finally:
            ds.close()


def _as_file(raw: bytes):
    import io

    return io.BytesIO(raw)


def _as_date(value) -> date:
    ts = np.datetime64(value, "s").astype("datetime64[s]").astype(object)
    return ts.date() if hasattr(ts, "date") else date.fromisoformat(str(ts)[:10])


def probe_latency(bbox: tuple[float, float, float, float] = (-83.5, 41.6, -83.0, 41.9)) -> dict:
    """How far behind real time each source is, for the ops page and the model card."""
    out: dict[str, object] = {"checked": datetime.now(UTC).date().isoformat()}
    try:
        newest = latest_available(bbox)
        out["coastwatch_newest"] = newest.isoformat() if newest else None
        out["coastwatch_lag_days"] = (datetime.now(UTC).date() - newest).days if newest else None
    except Exception as exc:
        out["coastwatch_error"] = str(exc)
    return out
