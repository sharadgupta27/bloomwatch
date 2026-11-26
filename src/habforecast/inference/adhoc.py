"""Forecast anywhere: build an AOI, fetch Sentinel-3 live, and run the model.

The registry's five lakes have years of ingested history sitting in Zarr cubes, which is
what makes `Predictor.predict` fast. An arbitrary point a user types into the app has
none of that, so everything has to be assembled on demand: search STAC over the request
window, read and reproject each scene onto a freshly-built grid, composite to daily
frames, and derive a water mask from the scenes themselves.

That is slow -- tens of seconds to a few minutes, dominated by network reads -- so this
module reports progress through a callback rather than blocking silently, and the API
runs it as a background job.

Two honest limitations, surfaced in the result rather than hidden:

* **The water mask comes from the request window only.** For a registry lake, "water" is
  a pixel that produced a valid retrieval on at least 5% of *years* of days. Here the
  only evidence available is the fortnight being fetched, so a pixel counts as water if
  any scene in the window flagged it INLAND_WATER or WATER and passed QC. Under a
  persistently cloudy fortnight this understates the water body, and `water_confidence`
  reports how much evidence there was so a caller can say so.
* **Two sources, because one of them is six months old.** Sentinel-3 OLCI is an
  operational near-real-time mission -- EUMETSAT publishes NRT products within ~3 hours
  of acquisition. The Planetary Computer mirror the training pipeline ingests from is an
  archive: measured 2026-08-28 its newest granule was 2026-02-25. Asking this module for
  today and being handed February was the visible symptom, so a second source was added.

  - `_NRTSource` reads NOAA CoastWatch daily Level-3 composites (300 m, S3A and S3B,
    rolling 90 days, no credentials, server-side bbox subsetting). Measured 2026-08-28 it
    was one day behind.
  - `_ArchiveSource` is the original L2 granule path: full WQSF quality control, a
    flag-derived water mask, and the full record back to 2018.

  `_pick_source` chooses by the date requested. They differ in more than latency -- the
  archive path applies this project's own QC and knows where water is from the
  INLAND_WATER flag, while the NRT path receives pre-masked composites and can only infer
  water from where a retrieval succeeded -- so the result reports which one served it and
  `caveats()` says what that costs.

*Which* near-real-time product is fetched, and which checkpoint runs on it, is not decided
here: `habforecast.routing` classifies the water body first and hands down a `Route`. A
Great Lakes bay and a point 300 km into the Gulf of Mexico get different products, a
different grid resolution and -- for the Gulf -- no learned model at all, because none has
been validated there. This module executes that plan and reports it back on the result;
what it still owns is the near-real-time-versus-archive choice, which depends on what a
server actually holds today rather than on geography.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from datetime import UTC, date, datetime, timedelta

import numpy as np

from habforecast.aoi import LakeAOI, LakeGrid, adhoc_aoi, adhoc_aoi_from_bbox
from habforecast.data.coastwatch import SourceUnavailableError
from habforecast.geometry import rings_bounds, rings_mask
from habforecast.inference.predictor import ForecastResult, Predictor
from habforecast.routing import Route, plan

logger = logging.getLogger(__name__)

# A pixel is called water if at least this fraction of the scenes covering it flagged it
# as water and passed QC. One clean look is enough -- INLAND_WATER is not set over land --
# but requiring a nonzero fraction keeps a single mis-flagged pixel from creating an
# island of "lake" in the middle of a field.
WATER_MIN_SCENE_FRACTION = 0.10
# Below this many confirmed water pixels there is no lake in the AOI worth forecasting.
MIN_WATER_PIXELS = 25
# How many successive 14-day windows to try before giving up. Six covers a full season,
# which is what it takes to step past a frozen or persistently clouded winter.
MAX_ANCHOR_ATTEMPTS = 6

ProgressFn = Callable[[str, float], None]

# Finding the archive's newest granule for an area means walking back a month at a time
# until a STAC search returns something. With the mirror six months behind, that is about
# seven round trips -- repeated identically on every request for the same lake, and every
# lake in one basin shares the answer. It changes when new imagery is published, so an
# hour is a safe lifetime and saves the walk.
ANCHOR_CACHE_TTL_S = 3600.0
# Bounded: the key carries the requested date, so a long-running server otherwise gains
# one entry per area per day for as long as it runs.
ANCHOR_CACHE_MAX = 256
_anchor_cache: dict[tuple, tuple[float, date | None]] = {}
_anchor_lock = threading.Lock()


def _remember_anchor(key: tuple, latest: date | None) -> None:
    with _anchor_lock:
        _anchor_cache[key] = (time.monotonic(), latest)
        while len(_anchor_cache) > ANCHOR_CACHE_MAX:
            oldest = min(_anchor_cache, key=lambda k: _anchor_cache[k][0])
            _anchor_cache.pop(oldest, None)


@dataclass
class AdhocResult:
    """A forecast for a location with no ingested history."""

    forecast: ForecastResult
    aoi: LakeAOI
    scenes_found: int
    scenes_used: int
    observed_days: list[date]
    water_pixels: int
    water_confidence: float  # mean per-water-pixel scene coverage in the window
    cell_area_km2: float
    source: str = "planetary-computer-archive"
    source_label: str = "Planetary Computer archive"
    source_latest: date | None = None  # newest day this source holds for the area
    # The input window as it was actually retrieved: day -> log10 chlorophyll over water,
    # NaN elsewhere. The model consumes a normalised copy of this; keeping the physical
    # one lets a click plot what the satellite saw before the forecast, on the same axis
    # as what the model says comes next. Only observed days appear.
    history: dict[date, np.ndarray] = field(default_factory=dict)

    @property
    def water_area_km2(self) -> float:
        return self.water_pixels * self.cell_area_km2

    @property
    def anchor_age_days(self) -> int | None:
        """How stale the forecast's anchor is, in days. The number the user actually asked."""
        if not self.observed_days:
            return None
        return (datetime.now(UTC).date() - max(self.observed_days)).days

    # Auxiliary layers this forecast was able to supply. The live path fetches
    # chlorophyll only, so a checkpoint trained with extra layers runs here with those
    # channels empty -- which the model reads as "never observed" and handles, but which
    # is a weaker input than it was trained on and is therefore reported.
    aux_supplied: tuple[str, ...] = ()
    aux_expected: tuple[str, ...] = ()
    # The product-and-model plan this forecast was produced under. Carried through so the
    # app can show what was chosen and why, rather than presenting a number with no
    # provenance to someone who has to act on it.
    route: Route | None = None
    # True when the result was cut to a requested shape smaller than its grid (a polygon,
    # or an area answered from a monitored lake's archive).
    clipped: bool = False

    def caveats(self) -> list[str]:
        """Plain-language warnings the UI must show. Never silently degrade."""
        out = list(self.route.caveats) if self.route else []
        missing = [k for k in self.aux_expected if k not in self.aux_supplied]
        if missing:
            out.append(
                "This model was trained with additional input layers ("
                + ", ".join(missing)
                + ") that the live path does not fetch, so it is running on chlorophyll "
                "alone. The forecast is weaker than the evaluation it was scored by."
            )
        if self.source == NRT_SOURCE:
            out.append(
                "Near-real-time source: NOAA CoastWatch daily Level-3 composites. These "
                "arrive already quality-masked, so this path does not apply the project's "
                "own WQSF flag screening, and the retrieval algorithm is not documented in "
                "the product metadata -- it may not be the CHL_NN product the model was "
                "trained on."
            )
        age = self.anchor_age_days
        if age is not None and age > 60:
            out.append(
                f"The most recent usable imagery here is {age} days old. This is an "
                "archive query, not a live forecast."
            )
        if self.water_confidence < 0.3:
            out.append(
                "The water mask was derived from a cloudy fortnight, so the mapped water "
                "body is probably smaller than the real one."
            )
        if len(self.observed_days) < 3:
            out.append(
                f"Only {len(self.observed_days)} day(s) in the 14-day window carried a "
                "usable retrieval. The forecast is close to persistence."
            )
        if self.scenes_used < self.scenes_found:
            out.append(
                f"{self.scenes_found - self.scenes_used} of {self.scenes_found} scenes "
                "were rejected by quality control (cloud, glint or failed retrieval)."
            )
        return out


def _noop(_stage: str, _frac: float) -> None:
    pass


def _source_label(source, key: str) -> str:
    """Name the source that served the window, including which product it fetched."""
    label = SOURCE_LABELS.get(key, key)
    if key == ARCHIVE_SOURCE:
        # The full-QC granule path reads the EUMETSAT Data Store whenever credentials are
        # set (data.olci.default_reader), and the label must say which one actually served.
        from habforecast.data import eumetsat

        if eumetsat.available():
            label = "EUMETSAT Data Store (Sentinel-3 OLCI L2, full QC)"
    product = getattr(source, "label", None)
    return f"{label} ({product})" if product and key == NRT_SOURCE else label


# ------------------------------------------------------------------------- sources ---
ARCHIVE_SOURCE = "planetary-computer-archive"
NRT_SOURCE = "noaa-coastwatch-nrt"
SOURCE_LABELS = {
    ARCHIVE_SOURCE: "Planetary Computer archive",
    NRT_SOURCE: "NOAA CoastWatch near-real-time",
}
# What the near-real-time reader fetches when nothing has routed the request. The router
# overrides it per area -- 300 m OLCI sectors do not reach open ocean, and a coarse global
# grid is the wrong product for a lake -- so this is the fallback, not the policy.
DEFAULT_NRT_PRODUCT = "olci-300m"


@dataclass
class Window:
    """A fortnight of daily fields on the analysis grid, from whichever source served it."""

    frames: dict[date, np.ndarray]  # day -> log10 chlorophyll, NaN where not retrieved
    water_fraction: np.ndarray  # per-pixel evidence of water, 0..1
    scenes_found: int
    scenes_used: int
    source: str


class _ArchiveSource:
    """Sentinel-3 OLCI L2 WFR granules from Planetary Computer, resampled per scene.

    This is the path the model was trained through: full quality control from the WQSF
    flag field, and a water mask built from the INLAND_WATER flag itself rather than
    inferred from where retrievals happened to succeed.
    """

    key = ARCHIVE_SOURCE

    def latest(self, predictor, aoi, requested, progress) -> date | None:
        return resolve_anchor(predictor, aoi, requested, progress=progress)

    def window(self, grid, start, end, min_valid_fraction, progress) -> Window:
        from habforecast.data.olci import default_reader, read_many
        from habforecast.data.resample import SwathResampler
        from habforecast.data.store import composite_day, group_by_day

        reader = default_reader()
        resampler = SwathResampler(grid)
        progress("searching the satellite archive", 0.18)
        items = reader.search(grid, start.isoformat(), end.isoformat())
        if not items:
            raise ValueError(
                f"No scenes cover this location between {start} and {end}, which should "
                "not happen after a successful search -- try a nearby date."
            )

        # Threaded: from the Data Store a scene is ~27 range requests of ~1 s latency, which
        # releases the GIL while it waits. (Against Planetary Computer, where decompression
        # dominates, threads buy little -- see `data/parallel.py` -- but cost nothing.)
        observations = read_many(
            reader,
            items,
            grid,
            resampler,
            min_valid_fraction=min_valid_fraction,
            progress=lambda n, total: progress(
                f"reading scene {n} of {total}", 0.2 + 0.6 * n / total
            ),
        )
        if not observations:
            raise ValueError(
                "Scenes were found but none contained usable water pixels here. The area "
                "may be land, too small, or under cloud for the whole two weeks."
            )

        # `valid` already means "flagged INLAND_WATER or WATER and passed QC", so this is
        # a direct measurement of where the water is, not an inference from where
        # retrievals happened to succeed.
        hits = np.zeros(grid.shape, dtype=np.int32)
        for obs in observations:
            hits += obs.valid.astype(np.int32)
        # A day whose scenes passed QC but retrieved no finite chlorophyll composites to
        # None; it is simply an unobserved day, as it is in `predictor.py`.
        composites = {
            day: composite_day(day, obs_list)
            for day, obs_list in group_by_day(observations).items()
        }
        frames = {day: f.log_chl for day, f in composites.items() if f is not None}
        return Window(
            frames=frames,
            water_fraction=hits / max(len(observations), 1),
            scenes_found=len(items),
            scenes_used=len(observations),
            source=self.key,
        )


class _NRTSource:
    """Daily Level-3 composites from NOAA CoastWatch, hours-to-a-day behind real time.

    The water mask is necessarily weaker here. The archive path knows a pixel is water
    because the WQSF flag says so, even under cloud; these composites carry no flags, so
    the only evidence available is "a retrieval succeeded here at least once". A fortnight
    of solid cloud therefore maps less water than really exists -- which is what
    `water_confidence` is for, and why it is reported rather than smoothed over.
    """

    key = NRT_SOURCE

    def __init__(self, product: str = DEFAULT_NRT_PRODUCT):
        from habforecast.data.coastwatch import PRODUCTS, CoastWatchReader

        self.product = product
        self.label = PRODUCTS[product].label
        self.reader = CoastWatchReader(product=product)

    def latest(self, predictor, aoi, requested, progress) -> date | None:
        newest = self.reader.available_through(aoi.grid().bounds_lonlat)
        if newest is None:
            return None
        return min(newest, requested) if requested else newest

    def window(self, grid, start, end, min_valid_fraction, progress) -> Window:
        progress("reading near-real-time imagery", 0.25)
        days = self.reader.frames(grid, start, end, progress=progress)
        if not days:
            raise ValueError(
                "No near-real-time imagery covers this area in that fortnight. NOAA "
                "CoastWatch keeps a rolling 90-day window; older dates come from the "
                "archive instead."
            )
        hits = np.zeros(grid.shape, dtype=np.int32)
        for f in days:
            hits += f.valid.astype(np.int32)
        return Window(
            frames={f.day: f.log_chl for f in days},
            water_fraction=hits / max(len(days), 1),
            scenes_found=len(days),
            scenes_used=len(days),
            source=self.key,
        )


# Route order is a preference, not an instruction: a product that stopped publishing a
# year ago is still catalogued and still "covers" the area, and taking it because it came
# first would hand back last summer's imagery for a request about this week. Measured on
# Lake Erie: the 750 m sectors ended 2025-08-19 while the 300 m ones were a day behind. So
# a step is skipped when a later one is fresher by more than one input window -- long
# enough that ordinary revisit differences never reorder anything.
STALENESS_TOLERANCE_DAYS = 14


def _freshest_enough(found: list) -> tuple:
    """The preferred source among those that are not badly out of date, and its newest day."""
    if not found:
        return None, None
    best = max(day for _, day in found)
    for source, day in found:  # route order
        if (best - day).days <= STALENESS_TOLERANCE_DAYS:
            return source, day
    return max(found, key=lambda pair: pair[1])


def _pick_source(aoi: LakeAOI, requested: date | None, preference: str, route: Route):
    """Choose between near-real-time and archive, and say which was chosen.

    "auto" prefers near-real-time whenever the date being asked for is inside the rolling
    window that source keeps, because that is the only way a request for today returns
    imagery from this week rather than from six months ago.

    Which near-real-time product is asked is the router's decision, not this function's:
    every step in `route.sources` is at the route's own resolution, so trying them in order
    changes where the imagery comes from without changing what the model was trained for.
    """
    if preference == "archive":
        return _ArchiveSource(), None
    live = [s.product for s in route.sources if s.product != ARCHIVE_SOURCE]
    try:
        bounds = aoi.grid(route.resolution_m).bounds_lonlat
        found: list[tuple[_NRTSource, date]] = []
        for product in live:
            candidate = _NRTSource(product)
            covered = candidate.reader.available_through(bounds)
            if covered is None:
                logger.info("%s does not cover this area", product)
                continue
            found.append((candidate, covered))
        nrt, newest = _freshest_enough(found)
        if nrt is None:
            nrt = _NRTSource(live[0] if live else DEFAULT_NRT_PRODUCT)
    except Exception as exc:
        logger.warning("near-real-time source unavailable: %s", exc)
        if preference == "nrt":
            raise ValueError(
                "The near-real-time source could not be reached. Try again, or pick the "
                "archive source."
            ) from exc
        return _ArchiveSource(), None
    if newest is None:
        if preference == "nrt":
            raise ValueError(
                "The near-real-time source does not cover this area. It is gridded by "
                "sector and does not reach every inland water body."
            )
        return _ArchiveSource(), None
    if preference == "nrt":
        return nrt, newest
    # The rolling window is ~90 days; asking for something older is an archive question.
    wanted = requested or datetime.now(UTC).date()
    horizon = newest - timedelta(days=70)
    return (nrt, newest) if wanted >= horizon else (_ArchiveSource(), newest)


def resolve_anchor(
    predictor: Predictor,
    aoi: LakeAOI,
    requested: date | None = None,
    max_lookback_days: int = 400,
    progress: ProgressFn = _noop,
) -> date | None:
    """Most recent date at or before `requested` that STAC actually has scenes for.

    The Planetary Computer mirror this build reads currently lags real time by several
    months, though Sentinel-3 itself is a near-real-time mission. Asking for today and
    getting an empty result would look like a broken app; walking back to the newest
    available date and *saying so* is the honest behaviour.
    """
    from habforecast.data.olci import default_reader

    grid = aoi.grid()
    reader = default_reader()
    end = requested or datetime.now(UTC).date()

    # Rounded to a tenth of a degree: neighbouring requests over one lake are the same
    # question, and answering it once is the point.
    west, south, east, north = grid.bounds_lonlat
    key = (round(west, 1), round(south, 1), round(east, 1), round(north, 1), end.isoformat())
    with _anchor_lock:
        hit = _anchor_cache.get(key)
        if hit is not None and time.monotonic() - hit[0] < ANCHOR_CACHE_TTL_S:
            return hit[1]

    offset = 0
    for span in _search_spans(max_lookback_days):
        hi = end - timedelta(days=offset)
        lo = hi - timedelta(days=span)
        progress(f"searching {lo}-{hi}", min(0.15, 0.02 + offset / max_lookback_days * 0.13))
        items = reader.search(grid, lo.isoformat(), hi.isoformat())
        if items:
            latest = min(max(i.datetime.date() for i in items), end)
            _remember_anchor(key, latest)
            return latest
        offset += span
    _remember_anchor(key, None)
    return None


# The first window is small because the answer is almost always in it: the Data Store is
# hours behind real time, and a search lists *every* granule in its window just to learn
# the newest date. Measured 2026-09-23 over Lake Superior: 3 days -> 5 overpasses, 3.6 s;
# 30 days -> 92 granules, 8.6 s. Windows then widen and never overlap, so a source that
# really is months behind (the Planetary Computer mirror, where an empty search costs
# ~0.1 s) still gets there in a handful of calls rather than one per few days.
SEARCH_SPANS_DAYS = (3, 7, 20)
SEARCH_SPAN_MAX_DAYS = 30


def _search_spans(max_lookback_days: int):
    """Consecutive window lengths, newest first, together covering `max_lookback_days`."""
    covered = 0
    for span in (*SEARCH_SPANS_DAYS, *[SEARCH_SPAN_MAX_DAYS] * max_lookback_days):
        if covered >= max_lookback_days:
            return
        span = min(span, max_lookback_days - covered)
        yield span
        covered += span


def build_aoi(
    lat: float | None = None,
    lon: float | None = None,
    radius_km: float = 12.0,
    bbox: tuple[float, float, float, float] | None = None,
    rings: list[list[tuple[float, float]]] | None = None,
    name: str = "Selected area",
) -> LakeAOI:
    """The area a request describes, whichever of the three shapes it used.

    Separate from `forecast_location` because the caller needs the extent *before* the
    forecast runs: the router classifies the water from it, and the checkpoint the router
    names has to be loaded before anything is fetched.
    """
    if rings:
        west, south, east, north = rings_bounds(rings)
        return adhoc_aoi_from_bbox(west, south, east, north, name=name)
    if bbox is not None:
        return adhoc_aoi_from_bbox(*bbox, name=name)
    if lat is not None and lon is not None:
        return adhoc_aoi(lat, lon, radius_km, name=name)
    raise ValueError("Give a centre point, a bounding box, or a polygon to forecast.")


def forecast_location(
    predictor: Predictor,
    lat: float | None = None,
    lon: float | None = None,
    radius_km: float = 12.0,
    anchor: date | None = None,
    name: str = "Selected area",
    min_valid_fraction: float = 0.005,
    progress: ProgressFn = _noop,
    bbox: tuple[float, float, float, float] | None = None,
    rings: list[list[tuple[float, float]]] | None = None,
    source: str = "auto",
    route: Route | None = None,
) -> AdhocResult:
    """Fetch, assemble and forecast for an arbitrary area.

    Give a centre and a radius, or `bbox` as [west, south, east, north], or `rings` as
    lon/lat polygon rings -- the app sends whichever shape the user drew or uploaded.
    When rings are given the forecast is *clipped to the polygon*: pixels outside it are
    never predicted and never returned, which is both what the user drew and a smaller
    payload.

    `source` is "auto" (near-real-time when the requested date is recent enough),
    "nrt", or "archive".

    `route` is the product-and-model plan from `habforecast.routing`. The caller passes it
    because the caller is what loads the matching checkpoint -- a route names a model, and
    `predictor` has to already be that model. When it is omitted the plan is derived here,
    which keeps this function usable on its own and matches whatever `predictor` is.

    Raises ValueError with a message written for a user, not a developer, when there is
    no usable data -- the app shows these directly.
    """
    if predictor.module is None:
        raise RuntimeError("predictor.load() must be called first")

    aoi = build_aoi(lat, lon, radius_km, bbox, rings, name)
    if route is None:
        route = plan(aoi.bbox, anchor)
    # The analysis grid is built at the routed product's resolution, not always at OLCI's
    # 300 m. Resampling a 4 km composite onto a 300 m grid would make a blocky picture out
    # of thirteen real numbers and charge the user the payload for it.
    grid = aoi.grid(route.resolution_m)
    cell_area_km2 = (grid.resolution / 1000.0) ** 2

    clip = None
    if rings:
        lon_g, lat_g = grid.lonlat
        clip = rings_mask(rings, lon_g, lat_g)
        if not clip.any():
            raise ValueError(
                "The uploaded shape does not enclose any 300 m grid cell. It may be "
                "smaller than a single pixel, or its coordinates may not be longitude "
                "and latitude."
            )

    progress("locating the most recent imagery", 0.02)
    picked, nrt_newest = _pick_source(aoi, anchor, source, route)
    first = picked.latest(predictor, aoi, anchor, progress)
    if first is None and picked.key == NRT_SOURCE:
        logger.info("near-real-time source had nothing; falling back to the archive")
        picked = _ArchiveSource()
        first = picked.latest(predictor, aoi, anchor, progress)
    if first is None:
        raise ValueError(
            "No Sentinel-3 ocean-colour scenes cover this location in the last year. "
            "Open water away from the poles is required."
        )

    # Scenes existing is not the same as scenes being usable. A mid-latitude lake in
    # February is frozen and every granule over it is rejected by QC, which would end the
    # request with "no water here" for a lake that is plainly there in July. So the newest
    # window is only the first candidate: on failure, step back a fortnight and retry, up
    # to a season's worth.
    #
    # Stepping back is the right answer to "this fortnight held nothing" and the wrong
    # answer to "the server would not talk to me" -- the earlier fortnight lives on the
    # same server. `SourceUnavailableError` separates the two, and switches source instead of
    # asking a broken server six more questions at a minute apiece.
    last_error: ValueError | None = None
    step = 0
    while step < MAX_ANCHOR_ATTEMPTS:
        candidate = first - timedelta(days=step * predictor.input_days)
        if step:
            progress(f"that window was unusable - trying {candidate}", 0.2)
        try:
            return _attempt(
                predictor,
                aoi,
                grid,
                cell_area_km2,
                candidate,
                radius_km,
                min_valid_fraction,
                progress,
                picked,
                nrt_newest,
                clip,
                route,
            )
        except SourceUnavailableError as exc:
            logger.info("%s unavailable: %s", picked.key, exc)
            if picked.key != NRT_SOURCE:
                raise ValueError(
                    "The imagery service did not respond. This is usually temporary - "
                    "try again in a minute."
                ) from exc
            progress("the live imagery service is not responding - using the archive", 0.2)
            picked = _ArchiveSource()
            archive_first = picked.latest(predictor, aoi, anchor, progress)
            if archive_first is None:
                break
            first, step = archive_first, 0  # a new source deserves its own newest window
            continue
        except ValueError as exc:
            last_error = exc
            logger.info("anchor %s unusable: %s", candidate, exc)
            # Stepping back through a rolling 90-day window eventually walks off its
            # start; the archive holds those dates, so switch rather than keep failing.
            if picked.key == NRT_SOURCE and step >= 2:
                logger.info("giving up on the near-real-time window; trying the archive")
                picked = _ArchiveSource()
                archive_first = picked.latest(predictor, aoi, anchor, progress)
                if archive_first is None:
                    break
                first, step = archive_first, 0
                continue
        step += 1
    raise last_error or ValueError("No usable imagery was found for this area from either source.")


def _attempt(
    predictor: Predictor,
    aoi: LakeAOI,
    grid,
    cell_area_km2: float,
    anchor: date,
    radius_km: float,
    min_valid_fraction: float,
    progress: ProgressFn,
    source,
    source_latest: date | None,
    clip: np.ndarray | None,
    route: Route,
) -> AdhocResult:
    """One anchor date's worth of fetch-and-forecast. Raises ValueError if unusable."""
    start = anchor - timedelta(days=predictor.input_days - 1)
    window = source.window(grid, start, anchor, min_valid_fraction, progress)

    progress("building the water mask", 0.84)
    water = window.water_fraction >= WATER_MIN_SCENE_FRACTION
    if clip is not None:
        # Clipping *before* the model runs, not after, is the point: the network never
        # sees pixels outside the drawn shape, they never enter the returned arrays, and
        # they never reach the browser.
        water &= clip
    n_water = int(water.sum())
    if n_water < MIN_WATER_PIXELS:
        where = (
            "in the selected area" if clip is not None else f"within {radius_km:g} km of this point"
        )
        raise ValueError(
            f"No water body was detected {where}. Sentinel-3 resolves water at 300 m, so "
            "small ponds and narrow rivers are invisible to it; try a larger area or a "
            "point nearer the middle of a lake, reservoir or coastal bay."
        )
    water_confidence = float(window.water_fraction[water].mean())

    progress("compositing daily frames", 0.9)
    h, w = grid.shape
    vals = np.zeros((predictor.input_days, h, w), dtype=np.float32)
    mask = np.zeros((predictor.input_days, h, w), dtype=np.float32)
    observed: list[date] = []
    history: dict[date, np.ndarray] = {}
    for slot in range(predictor.input_days):
        day = start + timedelta(days=slot)
        frame = window.frames.get(day)
        if frame is None:
            continue
        ok = np.isfinite(frame) & water
        if not ok.any():
            continue
        vals[slot][ok] = predictor.norm.apply(frame[ok])
        mask[slot] = ok.astype(np.float32)
        observed.append(day)
        history[day] = np.where(ok, frame, np.nan).astype(np.float32)

    if not observed:
        raise ValueError(
            "Water was detected but no day in the two-week window produced a usable "
            "chlorophyll retrieval over it -- persistent cloud. Try an earlier date."
        )

    progress("running the model", 0.94)
    forecast = _run(predictor, aoi, vals, mask, water, anchor, observed)
    progress("done", 1.0)

    return AdhocResult(
        forecast=forecast,
        aoi=aoi,
        scenes_found=window.scenes_found,
        scenes_used=window.scenes_used,
        observed_days=observed,
        water_pixels=n_water,
        water_confidence=water_confidence,
        cell_area_km2=cell_area_km2,
        source=window.source,
        source_label=_source_label(source, window.source),
        route=route,
        source_latest=source_latest,
        history=history,
        aux_expected=tuple(getattr(predictor, "aux_keys", ()) or ()),
    )


ARCHIVE_CUBE_SOURCE = "bloom-watch-archive"


def forecast_from_archive(
    predictor: Predictor,
    lake_id: str,
    anchor: date | None = None,
    route: Route | None = None,
    progress: ProgressFn = _noop,
    clip: Callable[[LakeGrid], np.ndarray] | None = None,
) -> AdhocResult | None:
    """The same result `forecast_location` builds, from a registry lake's ingested cube.

    `clip`, given the lake's grid, returns the (H, W) mask of the area actually requested
    (see `shape_clip`); without it the whole lake is returned.

    For the lakes the service already monitors there is nothing to download: the
    fortnight of imagery, quality-controlled through the training pipeline, is on disk,
    and the water mask is the one built from years of retrievals rather than inferred
    from two weeks of them. That turns a 15-60 s network job into about a second.

    Returns None -- and the caller falls back to the live path -- when the lake is not
    in this predictor's registry, has no cube, or the cube has no usable imagery in the
    window before `anchor`. `anchor` None means the newest day the cube holds.
    """
    if lake_id not in predictor.registry.lake_ids:
        return None
    ds, days = predictor._open_cube(lake_id)
    if ds is None or not days:
        return None
    today = datetime.now(UTC).date()
    if anchor is None:
        usable = [d for d in days if d <= today]
        if not usable:
            return None
        anchor = max(usable)
    elif anchor > max(days):
        return None  # newer than the archive: that is the live path's job
    progress("reading the archived fortnight", 0.3)
    try:
        forecast = predictor.predict(lake_id, anchor, allow_live=False)
    except ValueError:
        return None
    progress("done", 1.0)
    aoi = predictor.registry[lake_id]
    grid = aoi.grid()
    if clip is not None:
        # Only the user's shape: the lake's other pixels never enter the result, its
        # statistics or the payload, exactly as the live path clips a drawn polygon.
        mask = clip(grid)
        if not (mask & forecast.water.astype(bool)).any():
            raise ValueError(
                f"The selected area lies inside {aoi.name}'s extent but contains none of "
                "its water. Draw over the lake itself."
            )
        forecast = _clip_forecast(forecast, mask)
    water = forecast.water.astype(bool)
    start = anchor - timedelta(days=predictor.input_days - 1)
    history: dict[date, np.ndarray] = {}
    if forecast.input_log_chl is not None:
        for slot in range(forecast.input_log_chl.shape[0]):
            day = start + timedelta(days=slot)
            if day in forecast.input_observed_days:
                frame = forecast.input_log_chl[slot]
                history[day] = np.where(water & np.isfinite(frame), frame, np.nan).astype(
                    np.float32
                )
    observed = list(forecast.input_observed_days)
    return AdhocResult(
        forecast=forecast,
        aoi=aoi,
        # Days, not granules: the cube stores one composite per day. Equal counts keep
        # the "scenes rejected by QC" caveat from firing on a number it cannot know.
        scenes_found=len(observed),
        scenes_used=len(observed),
        observed_days=observed,
        water_pixels=int(water.sum()),
        # Years of retrievals, not a fortnight: the mask is as good as it gets.
        water_confidence=1.0,
        cell_area_km2=(grid.resolution / 1000.0) ** 2,
        source=ARCHIVE_CUBE_SOURCE,
        source_label="Bloom Watch archive (Sentinel-3 OLCI L2, full quality control)",
        source_latest=max(days),
        history=history,
        route=route,
        aux_expected=tuple(getattr(predictor, "aux_keys", ()) or ()),
        aux_supplied=tuple(getattr(predictor, "aux_keys", ()) or ()),
        clipped=clip is not None,
    )


def shape_clip(
    rings: list[list[tuple[float, float]]] | None = None,
    bbox: tuple[float, float, float, float] | None = None,
    circle: tuple[float, float, float] | None = None,
) -> Callable[[LakeGrid], np.ndarray]:
    """A mask builder for the shape a request described: polygon, box, or (lat, lon, km)."""

    def build(grid: LakeGrid) -> np.ndarray:
        lon, lat = grid.lonlat
        if rings:
            return rings_mask(rings, lon, lat)
        if bbox is not None:
            w, s, e, n = bbox
            return (lon >= w) & (lon <= e) & (lat >= s) & (lat <= n)
        if circle is not None:
            clat, clon, km = circle
            p1, p2 = np.radians(clat), np.radians(lat)
            dlat, dlon = p2 - p1, np.radians(lon - clon)
            a = np.sin(dlat / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlon / 2) ** 2
            return 2 * 6371.0088 * np.arcsin(np.sqrt(a)) <= km
        return np.ones(grid.shape, dtype=bool)

    return build


def _clip_forecast(fc: ForecastResult, mask: np.ndarray) -> ForecastResult:
    """`fc` with everything outside `mask` removed (water False, values NaN)."""
    water = fc.water.astype(bool) & mask

    def cut(a, lead_axes: int):
        if a is None:
            return None
        m = water.reshape((1,) * lead_axes + water.shape)
        return np.where(m, a, np.nan).astype(a.dtype)

    return replace(
        fc,
        water=water.astype(fc.water.dtype),
        log_chl=cut(fc.log_chl, 1),
        exceed_prob=cut(fc.exceed_prob, 2),
        spread=cut(fc.spread, 1),
        input_log_chl=cut(fc.input_log_chl, 1),
    )


def _run(
    predictor: Predictor,
    aoi: LakeAOI,
    vals: np.ndarray,
    mask: np.ndarray,
    water: np.ndarray,
    anchor: date,
    observed: list[date],
) -> ForecastResult:
    doy = np.array([anchor.timetuple().tm_yday], dtype=np.float32)
    # No climatology for an ad-hoc area: there is no archive to draw one from, and a
    # `use_clim` model handles the absent channel by design (it was trained to). The
    # numpy entry point is shared by every backend -- torch, ONNX and the labelled
    # persistence baseline -- so this path imports none of them.
    pred_b, prob_b, _spread = predictor.forward_numpy(
        vals[None], mask[None], water.astype(np.float32)[None, None], doy
    )
    pred = pred_b[0]

    log_chl = predictor.norm.invert(pred)
    log_chl = np.where(water[None], log_chl, np.nan).astype(np.float32)
    return ForecastResult(
        lake_id=aoi.lake_id,
        lake_name=aoi.name,
        anchor_date=anchor,
        lead_times=predictor.lead_times,
        log_chl=log_chl,
        water=water,
        input_observed_days=observed,
        source="live",
        model_version=predictor.model_version,
        exceed_prob=None
        if prob_b is None
        else np.where(water[None, None], prob_b[0], np.nan).astype(np.float32),
        exceed_thresholds=tuple(getattr(predictor, "exceed_thresholds", ()) or ())
        if prob_b is not None
        else (),
        ensemble_size=max(1, len(getattr(predictor, "members", []) or [])),
    )
