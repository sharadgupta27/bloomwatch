"""Forecast API.

Endpoints are deliberately honest about uncertainty. `/forecast` always returns the
observation mask and the list of days that were actually observed in the input window,
because a bloom forecast built from two cloudy frames and one built from twelve clear
ones look identical otherwise -- and only one of them is worth acting on.

The model is loaded lazily on first use so the container starts (and passes its health
check) even when no checkpoint has been mounted yet. `/health` reports model readiness
separately from process liveness for exactly that reason.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import logging
import os
import threading
import time
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import HTMLResponse, PlainTextResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from serving.api.jobs import JobStore

# The model is loaded lazily on the first request; its *module* is imported here, on the
# main thread, on purpose. Request handlers run in worker threads, and on Windows the
# first import of the predictor's native stack (torch, pyproj and pandas/xarray together)
# from a non-main thread corrupts the process: the request succeeds, then the event
# loop's teardown dies with an access violation and no Python traceback. The import
# costs about a second at startup; every package in it is a hard dependency of the
# serving image, so nothing is lost by paying that up front.
import habforecast.inference.predictor  # noqa: F401

# The predictor module no longer imports torch itself -- the ONNX serving image has none
# (see inference.onnx_backend) -- so the main-thread import above no longer covers it.
# Where torch is installed it is imported here, for the reason above; where it is not,
# the ONNX backend is the only one that can load and nothing is lost.
with contextlib.suppress(ImportError):
    import torch  # noqa: F401

logger = logging.getLogger(__name__)

AOI_PATH = Path(os.getenv("HAB_AOI", "data/aoi/lakes.geojson"))
ARTIFACTS = Path(os.getenv("HAB_ARTIFACTS", "artifacts"))
CUBE_ROOT = Path(os.getenv("HAB_CUBES", "data/processed/cubes"))
CHECKPOINT = Path(os.getenv("HAB_CHECKPOINT", "models/best.ckpt"))
# Where /advisory takes its exceedance probabilities from (see advisory.with_probabilities).
# Unset or missing: the advisory falls back to thresholding the map, as it always did.
PROB_CHECKPOINT = Path(os.getenv("HAB_PROB_CHECKPOINT", "models/ens_met"))
# Flip test-time augmentation for every served forecast (four passes instead of one).
TTA = os.getenv("HAB_TTA", "1").lower() not in ("0", "false", "no", "off")
# Public in-situ measurements, distilled by scripts/build_insitu_index.py. Optional: a
# fresh checkout has none, and the app must say "no validation data" rather than fail.
INSITU = Path(os.getenv("HAB_INSITU", "data/validation/insitu.csv.gz"))

app = FastAPI(
    title="Bloom Watch",
    version="0.1.0",
    description=(
        "Short-range cyanobacterial bloom forecasting for inland waters, "
        "from Sentinel-3 OLCI. Forecasts are advisory only - see /model-card."
    ),
)

# An ad-hoc result carries one quantised value grid per lead so the app can report the
# concentration under a click without a round trip per pixel. Those grids are mostly the
# reserved no-data code and compress by an order of magnitude; the PNGs beside them do
# not, which is why the threshold is low rather than absent.
app.add_middleware(GZipMiddleware, minimum_size=2048)

WEB_ROOT = Path(__file__).resolve().parents[1] / "web"
# The app's stylesheet and script are served from here rather than inlined. They are
# large enough that inlining hurts readability and re-download on every navigation, and
# they are local files in the image -- so this adds no external dependency, which was the
# actual reason the small map page is inline.
if WEB_ROOT.exists():
    app.mount("/static", StaticFiles(directory=str(WEB_ROOT)), name="static")

_predictor = None
_load_error: str | None = None

# The registry is a small GeoJSON, but /health is polled and /lakes is the first thing
# the map asks for; parsing it on every call is waste. Re-read only when the file
# changes on disk.
_registry_cache: tuple[tuple[int, int], object] | None = None
_registry_lock = threading.Lock()


def registry():
    """The AOI registry, parsed once per on-disk version of the file."""
    global _registry_cache
    from habforecast.aoi import AOIRegistry

    st = AOI_PATH.stat()
    sig = (st.st_size, st.st_mtime_ns)
    with _registry_lock:
        if _registry_cache is not None and _registry_cache[0] == sig:
            return _registry_cache[1]
        reg = AOIRegistry.from_geojson(AOI_PATH)
        _registry_cache = (sig, reg)
        return reg


# Ad-hoc forecasts run in the background: they take minutes, almost all of it network.
_jobs = JobStore()
# Results are kept out of the poll payload (they contain PNGs) and fetched separately.
_results: dict[str, dict] = {}
# The bulletin PDF is drawn from the forecast *object* (full-resolution fields, the input
# history, the route), which the JSON result cache does not hold. Kept per result-cache key
# so a cached re-run of the same area can still produce one, and rendered at most once.
# Few entries: an object holds every input frame for the area -- tens of MB for a large
# drawn box -- where a JSON result is a few hundred kB.
MAX_BULLETIN_INPUTS = 8
_bulletin_inputs: dict[str, list] = {}  # cache key -> [stored_at, AdhocResult, pdf | None]
_bulletin_lock = threading.Lock()
_job_keys: dict[str, str] = {}  # job id -> result-cache key


def get_predictor():
    """Load the default model on first request; cache the failure so we do not retry per call.

    This is the checkpoint the registry-lake endpoints use. Ad-hoc requests go through
    `predictor_for`, which loads whichever checkpoint the router paired with the water
    body -- they are usually the same file, and deliberately not always.
    """
    global _predictor, _load_error
    if _predictor is not None:
        return _predictor
    if _load_error is not None:
        raise HTTPException(status_code=503, detail=f"model unavailable: {_load_error}")
    try:
        from habforecast.inference.predictor import load_predictor

        if not CHECKPOINT.exists():
            raise FileNotFoundError(f"no checkpoint at {CHECKPOINT}")
        # A .ckpt, an ensemble directory of them, or an ONNX export directory.
        _predictor = load_predictor(CHECKPOINT, AOI_PATH, CUBE_ROOT, tta=TTA)
        return _predictor
    except Exception as exc:
        _load_error = f"{type(exc).__name__}: {exc}"
        logger.error("model load failed: %s", _load_error)
        raise HTTPException(status_code=503, detail=f"model unavailable: {_load_error}") from exc


# One process now serves more than one checkpoint, because the router pairs a model with
# the water body rather than with the deployment. They are small (4 MB) and loading one
# costs about a second, so they are cached for the life of the process and shared across
# requests; the lock keeps two simultaneous first-requests from loading the same file twice.
_predictors: dict[str, object] = {}
_predictor_lock = threading.Lock()


def predictor_for(route):
    """The forecaster this route calls for -- a checkpoint, or a labelled baseline.

    A route with no model is not a failure. It is the open-ocean case, where the only
    products that reach the area are 4 km grids and the 4 km checkpoint measured negative
    skill against persistence; `BaselinePredictor` is what honesty looks like there.
    """
    from habforecast.inference.predictor import BaselinePredictor, load_predictor
    from habforecast.routing import resolve_checkpoint

    key = route.model.key if route.model else f"baseline:{route.baseline}"
    with _predictor_lock:
        hit = _predictors.get(key)
        if hit is not None:
            return hit
        if route.model is None:
            loaded = BaselinePredictor(route.baseline or "persistence").load()
        else:
            path = resolve_checkpoint(route.model)
            if not path.exists():
                raise HTTPException(
                    status_code=503,
                    detail=(
                        f"the router chose {route.model.key} for this area but its "
                        f"checkpoint is not installed at {path}"
                    ),
                )
            loaded = load_predictor(path, AOI_PATH, CUBE_ROOT, tta=TTA)
        _predictors[key] = loaded
        return loaded


def route_for(req, rings):
    """The product-and-model plan for a request, honouring an explicit model override."""
    from habforecast.inference.adhoc import build_aoi
    from habforecast.routing import override_model, plan

    aoi = build_aoi(
        req.lat,
        req.lon,
        req.radius_km,
        tuple(req.bbox) if req.bbox else None,
        rings,
        req.name,
    )
    route = plan(aoi.bbox, req.anchor)
    chosen = getattr(req, "model", "auto")
    if chosen and chosen != "auto":
        try:
            route = override_model(route, chosen)
        except (KeyError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
    return aoi, route


# --------------------------------------------------------------------------- schema --
class LakeInfo(BaseModel):
    lake_id: str
    name: str
    country: str
    split: str
    trophic_state: str
    bloom_season: list[int]
    bbox: list[float] = Field(description="[west, south, east, north] in EPSG:4326")
    grid_shape: list[int]
    archived_days: int | None = None


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    checkpoint: str
    checkpoint_present: bool
    cubes_present: bool
    lakes: int
    detail: str | None = None
    live_source: dict | None = None


# ------------------------------------------------------------------------ endpoints --
@app.get("/health", response_model=HealthResponse, tags=["ops"])
def health() -> HealthResponse:
    """Liveness plus model readiness. The process can be healthy with no model loaded."""
    try:
        n_lakes = len(registry())
    except Exception:
        n_lakes = 0
    # Whether the live imagery source is currently cut off is an operational fact a
    # dashboard should be able to see without reading logs: it is the difference between
    # "the app is slow" and "NOAA is down and we are serving the archive".
    try:
        from habforecast.data.coastwatch import breaker_state

        live = breaker_state()
    except Exception:
        live = None

    return HealthResponse(
        status="ok",
        live_source=live,
        model_loaded=_predictor is not None,
        checkpoint=str(CHECKPOINT),
        checkpoint_present=CHECKPOINT.exists(),
        cubes_present=CUBE_ROOT.exists(),
        lakes=n_lakes,
        detail=_load_error,
    )


@app.get("/lakes", response_model=list[LakeInfo], tags=["catalog"])
def list_lakes() -> list[LakeInfo]:
    """Registered lakes, their split assignment, and archive depth."""
    out = []
    for aoi in registry():
        grid = aoi.grid()
        archived = None
        cube = CUBE_ROOT / f"{aoi.lake_id}.zarr"
        if cube.exists():
            try:
                archived = _archived_days(cube)
            except Exception:
                archived = None
        out.append(
            LakeInfo(
                lake_id=aoi.lake_id,
                name=aoi.name,
                country=aoi.country,
                split=aoi.split,
                trophic_state=aoi.trophic_state,
                bloom_season=list(aoi.bloom_season),
                bbox=list(aoi.bbox),
                grid_shape=list(grid.shape),
                archived_days=archived,
            )
        )
    return out


def _archived_days(cube: Path) -> int:
    """Length of a cube's time axis, read from the consolidated metadata alone.

    Opening through xarray costs ~30 ms per lake for a single integer; the array's
    shape is in the metadata zarr reads first anyway.
    """
    import zarr

    try:
        group = zarr.open_group(cube, mode="r", use_consolidated=True)
    except TypeError:  # zarr 2 has no use_consolidated keyword
        group = zarr.open_consolidated(cube, mode="r")
    times = group["time"]
    assert isinstance(times, zarr.Array)
    return int(times.shape[0])


# --- registry-lake forecast cache ---------------------------------------------------
# The map page asks for /forecast and then /forecast/image once per lead: five identical
# predictions for one view, each reading the cube and running the model. The answer only
# changes when the cube does (a request pinned to a date) or when today's imagery lands
# (a request for "now"), so a short-lived cache turns four of the five into a dict read.
FORECAST_TTL_ANCHORED_S = 3600.0
FORECAST_TTL_TODAY_S = 300.0
MAX_CACHED_FORECASTS = 32

_forecast_cache: dict[tuple, tuple[float, Any, dict]] = {}
_forecast_lock = threading.Lock()


def _predict_cached(predictor, lake_id: str, anchor_date: date, allow_live: bool, pinned: bool):
    """`predictor.predict`, memoised per (lake, anchor) for a bounded time.

    A result computed with live fetching allowed is served to a caller that forbids it
    only when it actually came from the archive -- which is what that caller asked for.
    """
    key = (predictor.model_version, lake_id, anchor_date)
    ttl = FORECAST_TTL_ANCHORED_S if pinned else FORECAST_TTL_TODAY_S
    now = time.time()
    with _forecast_lock:
        hit = _forecast_cache.get(key)
        if hit is not None and now - hit[0] <= ttl and (allow_live or hit[1].source != "live"):
            return hit[1], hit[2]
    result = predictor.predict(lake_id, anchor_date, allow_live=allow_live)
    with _forecast_lock:
        _forecast_cache[key] = (time.time(), result, {})
        while len(_forecast_cache) > MAX_CACHED_FORECASTS:
            oldest = min(_forecast_cache, key=lambda k: _forecast_cache[k][0])
            _forecast_cache.pop(oldest, None)
        return result, _forecast_cache[key][2]


@app.get("/forecast", tags=["forecast"])
def forecast(
    lake_id: str = Query(..., description="lake_id from /lakes"),
    anchor: str | None = Query(None, description="ISO date; defaults to today (UTC)"),
    allow_live: bool = Query(True, description="fetch from STAC if outside the archive"),
) -> dict:
    """Bloom forecast for one lake at all trained lead times.

    Returns predicted chlorophyll statistics and bloom extent per lead time, plus the
    provenance needed to judge the forecast: which input days were actually observed,
    and whether the window came from the archive or a live fetch.
    """
    predictor = get_predictor()
    try:
        anchor_date = date.fromisoformat(anchor) if anchor else datetime.now(UTC).date()
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=f"bad anchor date: {anchor}") from exc

    try:
        result, _pngs = _predict_cached(
            predictor, lake_id, anchor_date, allow_live, pinned=anchor is not None
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"unknown lake_id: {lake_id}") from exc
    except ValueError as exc:
        # No usable observations is a legitimate outcome, not a server fault.
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    grid = predictor.registry[lake_id].grid()
    cell_km2 = (grid.resolution / 1000.0) ** 2
    payload = result.summary(cell_km2)
    payload["disclaimer"] = (
        "Advisory only. Satellite chlorophyll is not a direct cyanotoxin measurement; "
        "confirm with in-situ sampling before any public health action."
    )
    return payload


@app.get("/forecast/image", tags=["forecast"], response_class=Response)
def forecast_image(
    lake_id: str = Query(...),
    lead_days: int = Query(7),
    anchor: str | None = Query(None),
) -> Response:
    """PNG rendering of the predicted bloom field, for the map overlay."""
    predictor = get_predictor()
    try:
        anchor_date = date.fromisoformat(anchor) if anchor else datetime.now(UTC).date()
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=f"bad anchor date: {anchor}") from exc
    try:
        result, pngs = _predict_cached(
            predictor, lake_id, anchor_date, True, pinned=anchor is not None
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"unknown lake_id: {lake_id}") from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    if lead_days not in result.lead_times:
        raise HTTPException(
            status_code=422,
            detail=f"lead_days must be one of {list(result.lead_times)}",
        )
    idx = result.lead_times.index(lead_days)
    # Rendered overlays ride along with the cached forecast they were drawn from.
    png = pngs.get(lead_days)
    if png is None:
        png = pngs[lead_days] = _render_png(result.log_chl[idx], result.water)
    return Response(content=png, media_type="image/png")


_prob_predictor = None
_prob_load_failed = False


def _get_prob_predictor():
    """The advisory's probability model, or None to threshold the map instead.

    A missing or unloadable probability model degrades the advisory to the deterministic
    rule -- it does not take /advisory down, because the map model alone still produces a
    valid (if less sharp) feed. The failure is logged once, not per request.
    """
    global _prob_predictor, _prob_load_failed
    if _prob_predictor is not None or _prob_load_failed:
        return _prob_predictor
    if not PROB_CHECKPOINT.exists():
        _prob_load_failed = True
        logger.info("no probability model at %s; advisory thresholds the map", PROB_CHECKPOINT)
        return None
    try:
        from habforecast.inference.predictor import load_predictor

        _prob_predictor = load_predictor(PROB_CHECKPOINT, AOI_PATH, CUBE_ROOT, tta=TTA)
    except Exception as exc:
        _prob_load_failed = True
        logger.error("probability model failed to load (%s: %s)", type(exc).__name__, exc)
    return _prob_predictor


def _advisory_collection(anchor: str | None, allow_live: bool, min_fraction: float) -> dict:
    """Every registry lake's advisory, built from the same cached forecasts /forecast serves."""
    from habforecast import advisory

    predictor = get_predictor()
    prob_predictor = _get_prob_predictor()
    try:
        anchor_date = date.fromisoformat(anchor) if anchor else datetime.now(UTC).date()
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=f"bad anchor date: {anchor}") from exc
    if not 0.0 < min_fraction <= 1.0:
        raise HTTPException(status_code=422, detail="min_fraction must be in (0, 1]")

    feats = []
    for lake_id in predictor.registry.lake_ids:
        aoi = predictor.registry[lake_id]
        try:
            result, _pngs = _predict_cached(
                predictor, lake_id, anchor_date, allow_live, pinned=anchor is not None
            )
            if prob_predictor is not None:
                result = advisory.with_probabilities(
                    result, prob_predictor.predict(lake_id, anchor_date, allow_live=allow_live)
                )
        except ValueError as exc:
            feats.append(advisory.unavailable_feature(aoi, anchor_date, str(exc)))
            continue
        cell_km2 = (aoi.grid().resolution / 1000.0) ** 2
        feats.append(advisory.feature(advisory.advise(result, cell_km2, min_fraction), aoi))
    return advisory.collection(feats)


@app.get("/advisory", tags=["forecast"])
def advisory_geojson(
    anchor: str | None = Query(None, description="ISO date; defaults to today (UTC)"),
    allow_live: bool = Query(False, description="fetch from STAC if outside the archive"),
    min_fraction: float = Query(
        0.10, description="fraction of the lake above a threshold that triggers that level"
    ),
) -> dict:
    """The advisory feed as GeoJSON: one feature per registry lake.

    Properties carry the advisory level, the first lead at which each chlorophyll
    threshold is crossed over at least `min_fraction` of the lake, the predicted bloom
    area per lead, and the provenance of the forecast. Lakes with no usable observations
    appear with `status: unavailable` rather than being dropped.
    """
    return _advisory_collection(anchor, allow_live, min_fraction)


@app.get("/advisory/feed.xml", tags=["forecast"], response_class=Response)
def advisory_rss(
    request: Request,
    anchor: str | None = Query(None),
    allow_live: bool = Query(False),
    min_fraction: float = Query(0.10),
) -> Response:
    """The same advisories as an RSS 2.0 feed, for any feed reader or automation."""
    from habforecast import advisory

    fc = _advisory_collection(anchor, allow_live, min_fraction)
    base = str(request.base_url).rstrip("/")
    return Response(content=advisory.rss(fc, base_url=base), media_type="application/rss+xml")


class PlaceHit(BaseModel):
    name: str
    lat: float
    lon: float
    country: str = ""
    note: str = ""
    bbox: list[float] | None = None
    source: str = "gazetteer"


@app.get("/api/places", response_model=list[PlaceHit], tags=["app"])
def places(q: str = Query(..., min_length=1, max_length=120)) -> list[PlaceHit]:
    """Resolve free text to candidate points.

    Coordinates, then the curated water-body gazetteer, then a general geocoder. The
    ordering matters: someone who typed numbers means those numbers, and for the lakes
    this model was evaluated on the curated centre sits on open water where a geocoder's
    label anchor may not.
    """
    from habforecast.places import find

    return [PlaceHit(**p.as_dict()) for p in find(q)]


@app.get("/api/insitu", tags=["app"])
def insitu(
    lat: float = Query(..., ge=-90, le=90),
    lon: float = Query(..., ge=-180, le=180),
    start: date = Query(..., description="first day of the plotted window"),
    end: date = Query(..., description="last day of the plotted window"),
    radius_km: float = Query(10.0, ge=0.5, le=100.0),
    season_days: int = Query(10, ge=0, le=60),
) -> dict:
    """Public in-situ measurements near a clicked pixel, for plotting beside the forecast.

    Answers a question the rest of the API cannot: has anyone actually put a bottle in
    this water, and what did the laboratory say. The response separates measurements
    taken *inside* the plotted window -- the only ones that verify anything -- from ones
    taken near this point in the same week of other years, which are context.

    Nothing is fetched from a remote service here. The index is a local file, so a click
    costs a millisecond and leaks nothing about where the user is looking.
    """
    from habforecast.insitu import context

    if end < start:
        raise HTTPException(status_code=422, detail="end must not precede start")
    if (end - start).days > 400:
        raise HTTPException(status_code=422, detail="window must be under 400 days")
    return context(
        lat,
        lon,
        start,
        end,
        radius_km=radius_km,
        season_days=season_days,
        path=INSITU,
    )


@app.get("/api/validation-sites", tags=["app"])
def validation_sites_endpoint() -> dict:
    """Water bodies where a forecast can be put beside a field measurement.

    Two kinds, and the distinction is the point. **Modelled** water bodies are the lakes
    this model was trained and evaluated on. **Measured** ones are places the public
    in-situ archives cover and this project has never seen -- a forecast there is the
    model extrapolating to unfamiliar water, which is exactly where field data earns its
    keep.

    Derived from the same local index the pixel probe reads, so it costs no network and
    is empty rather than broken when no index is installed.
    """
    from habforecast.insitu import validation_sites

    sites = validation_sites(index_path=INSITU)
    return {
        "sites": sites,
        "modelled": sum(1 for s in sites if s["kind"] == "modelled"),
        "measured": sum(1 for s in sites if s["kind"] == "measured"),
        "attribution": ("GLORIA (PANGAEA, CC-BY 4.0) and the US EPA Water Quality Portal"),
    }


class ForecastRequest(BaseModel):
    """A centre plus a radius, a drawn box, or a polygon.

    Precedence is polygon, then box, then centre: the most specific shape the user gave
    is the one they meant, and a polygon additionally *clips* the result to itself.
    """

    lat: float = Field(..., ge=-90, le=90)
    lon: float = Field(..., ge=-180, le=180)
    radius_km: float = Field(12.0, ge=2.0, le=40.0)
    name: str = Field("Selected area", max_length=120)
    bbox: list[float] | None = Field(
        None, description="[west, south, east, north] in EPSG:4326, from a drawn shape"
    )
    rings: list[list[list[float]]] | None = Field(
        None,
        description="Polygon rings as [[[lon, lat], ...], ...]. The forecast is clipped to them.",
    )
    anchor: date | None = Field(
        None, description="Latest day to forecast from; defaults to the newest imagery."
    )
    source: str = Field(
        "auto",
        pattern="^(auto|nrt|archive)$",
        description="auto picks near-real-time for recent dates and the archive for older ones",
    )
    model: str = Field(
        "auto",
        max_length=32,
        description=(
            "auto lets the router pair a checkpoint with the water body (see /api/route); "
            "a key from that route's alternatives overrides it."
        ),
    )


class GeometryResponse(BaseModel):
    """A shape read out of an uploaded file, ready to draw and to forecast on."""

    rings: list[list[list[float]]]
    bbox: list[float]
    ring_count: int
    vertex_count: int
    side_km: float
    too_large: bool
    name: str


@app.post("/api/geometry", response_model=GeometryResponse, tags=["app"])
async def upload_geometry(file: UploadFile = File(...)) -> GeometryResponse:
    """Read a GeoJSON, KML/KMZ or zipped shapefile into lon/lat rings.

    Parsed on the server because two of the three formats are binary containers. The
    response carries the shape's size so the app can refuse an oversized area before
    starting a job that would be refused anyway.
    """
    from habforecast.aoi import MAX_ADHOC_SIDE_KM
    from habforecast.geometry import (
        MAX_UPLOAD_BYTES,
        GeometryError,
        parse_geometry,
        rings_bounds,
    )

    raw = await file.read(MAX_UPLOAD_BYTES + 1)
    try:
        rings = parse_geometry(file.filename or "", raw)
        west, south, east, north = rings_bounds(rings)
    except GeometryError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    side_km = max(
        _haversine_km((south + north) / 2, west, (south + north) / 2, east),
        _haversine_km(south, (west + east) / 2, north, (west + east) / 2),
    )
    stem = Path(file.filename or "shape").stem[:60] or "Uploaded shape"
    return GeometryResponse(
        rings=[[[round(x, 6), round(y, 6)] for x, y in r] for r in rings],
        bbox=[west, south, east, north],
        ring_count=len(rings),
        vertex_count=sum(len(r) for r in rings),
        side_km=round(side_km, 1),
        too_large=side_km > MAX_ADHOC_SIDE_KM,
        name=stem,
    )


def _clean_rings(raw) -> list[list[tuple[float, float]]] | None:
    """Validate polygon rings arriving over the wire.

    The upload endpoint already validated what it parsed, but this request field is
    reachable directly, so the bounds are checked again here rather than trusting a
    caller to have gone through the browser.
    """
    from habforecast.geometry import MAX_RINGS, MAX_VERTICES, MIN_RING_VERTICES

    if not raw:
        return None
    if len(raw) > MAX_RINGS:
        raise HTTPException(status_code=422, detail=f"at most {MAX_RINGS} rings")
    out: list[list[tuple[float, float]]] = []
    total = 0
    for ring in raw:
        pts = [(float(p[0]), float(p[1])) for p in ring if len(p) >= 2]
        if len(pts) < MIN_RING_VERTICES:
            continue
        for lon, lat in pts:
            if not (-180.0 <= lon <= 180.0 and -90.0 <= lat <= 90.0):
                raise HTTPException(status_code=422, detail="ring coordinates must be lon/lat")
        total += len(pts)
        if total > MAX_VERTICES:
            raise HTTPException(status_code=422, detail=f"at most {MAX_VERTICES} vertices")
        out.append(pts)
    if not out:
        raise HTTPException(status_code=422, detail="no ring had three or more points")
    return out


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0088
    p = np.pi / 180.0
    dlat, dlon = (lat2 - lat1) * p, (lon2 - lon1) * p
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1 * p) * np.cos(lat2 * p) * np.sin(dlon / 2) ** 2
    return float(2 * r * np.arcsin(np.sqrt(a)))


# --- finished-forecast cache -------------------------------------------------------
# A forecast is expensive (tens of seconds of network) and perfectly reproducible: the
# same area, anchor and source give the same answer until new imagery lands. The
# validation sidebar makes repeats the normal case -- people click down a list of lakes
# and back again -- so without this every revisit pays full price for a byte-identical
# result.
#
# Two lifetimes, because the two questions differ. A request pinned to a past anchor can
# only change when a reprocessing lands, and an hour is conservative. A request for "the
# newest imagery" must expire quickly enough to pick up today's pass.
RESULT_TTL_LIVE_S = 900.0
RESULT_TTL_ANCHORED_S = 3600.0
MAX_CACHED_RESULTS = 64

_result_cache: dict[str, tuple[float, dict]] = {}
_result_cache_lock = threading.Lock()


def _cache_key(req: ForecastRequest, rings) -> str:
    """Everything that changes the answer, and nothing that does not.

    The name is excluded deliberately: two people who draw the same box and label it
    differently are asking one question, and should not each wait for it.
    """
    shape: object
    if rings:
        shape = [[(round(x, 5), round(y, 5)) for x, y in r] for r in rings]
    elif req.bbox:
        shape = [round(v, 5) for v in req.bbox]
    else:
        shape = [round(req.lat, 5), round(req.lon, 5), round(req.radius_km, 3)]
    payload = json.dumps(
        [
            shape,
            req.anchor.isoformat() if req.anchor else None,
            req.source,
            getattr(req, "model", "auto"),
        ],
        sort_keys=True,
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def _cached_result(key: str, ttl_s: float) -> dict | None:
    with _result_cache_lock:
        hit = _result_cache.get(key)
        if hit is None:
            return None
        made, payload = hit
        if time.time() - made > ttl_s:
            _result_cache.pop(key, None)
            return None
        return payload


def _store_result(key: str, payload: dict) -> None:
    with _result_cache_lock:
        _result_cache[key] = (time.time(), payload)
        while len(_result_cache) > MAX_CACHED_RESULTS:
            oldest = min(_result_cache, key=lambda k: _result_cache[k][0])
            _result_cache.pop(oldest, None)


@app.get("/api/route", tags=["app"])
def route_preview(
    west: float = Query(..., ge=-180, le=180),
    south: float = Query(..., ge=-90, le=90),
    east: float = Query(..., ge=-180, le=180),
    north: float = Query(..., ge=-90, le=90),
) -> dict:
    """What kind of water this is, and what would be used to forecast it.

    Offline and cheap -- no imagery, no model, a couple of milliseconds -- so the app can
    show the decision the moment a shape is drawn, before anyone spends thirty seconds
    finding out. Same function the forecast itself routes through, so the preview cannot
    drift from what actually runs.
    """
    from habforecast.routing import plan

    if east <= west or north <= south:
        raise HTTPException(status_code=422, detail="bbox must be west<east and south<north")
    return plan((west, south, east, north)).as_dict()


@app.get("/api/models", tags=["app"])
def models_catalogue() -> dict:
    """Every checkpoint this build knows about, with the skill it measured.

    Including the one that is not served. A catalogue that hides its failures is a
    brochure, and the 4 km result is the most useful negative finding this project has.
    """
    from habforecast.routing import available_models

    return {"models": available_models()}


@app.post("/api/forecast/jobs", tags=["app"])
def submit_forecast(req: ForecastRequest) -> dict:
    """Start an ad-hoc forecast anywhere on the water. Returns a job to poll."""
    if req.bbox is not None and len(req.bbox) != 4:
        raise HTTPException(status_code=422, detail="bbox must be [west, south, east, north]")
    rings = _clean_rings(req.rings)
    # The route is settled before anything is fetched, because it decides both which
    # checkpoint to load and which product to ask for -- and the two cannot be chosen
    # independently.
    _aoi, route = route_for(req, rings)
    predictor = predictor_for(route)
    key = _cache_key(req, rings)
    ttl = RESULT_TTL_ANCHORED_S if req.anchor else RESULT_TTL_LIVE_S

    def work(job) -> dict:
        from habforecast.inference.adhoc import forecast_location

        _job_keys[job.id] = key
        hit = _cached_result(key, ttl)
        if hit is not None:
            job.stage = "reusing the forecast already computed for this area"
            job.progress = 1.0
            _results[job.id] = {**hit, "cached": True}
            return {"ready": True, "cached": True}

        def progress(stage: str, frac: float) -> None:
            job.stage = stage
            job.progress = frac

        res = forecast_location(
            predictor,
            req.lat,
            req.lon,
            radius_km=req.radius_km,
            anchor=req.anchor,
            name=req.name,
            progress=progress,
            bbox=tuple(req.bbox) if req.bbox else None,
            rings=rings,
            source=req.source,
            route=route,
        )
        payload = res.forecast.summary(res.cell_area_km2)
        west, south, east, north = res.aoi.bbox
        payload.update(
            {
                "bbox": [west, south, east, north],
                "centre": [req.lat, req.lon],
                "radius_km": req.radius_km,
                "grid": list(res.forecast.water.shape),
                "source": res.source,
                "source_label": res.source_label,
                "source_latest": res.source_latest.isoformat() if res.source_latest else None,
                "anchor_age_days": res.anchor_age_days,
                "clipped": bool(rings),
                "scenes_found": res.scenes_found,
                "scenes_used": res.scenes_used,
                "observed_days": [d.isoformat() for d in res.observed_days],
                "water_area_km2": res.water_area_km2,
                "water_confidence": res.water_confidence,
                "caveats": res.caveats(),
                # Why this answer came from this data and this model. Shown in the app:
                # a forecast a city acts on has to be able to explain its own provenance.
                "route": route.as_dict(),
                "model_version": res.forecast.model_version,
                "images": {
                    str(lead): _data_uri(_render_png(res.forecast.log_chl[i], res.forecast.water))
                    for i, lead in enumerate(res.forecast.lead_times)
                },
                # The picture is for looking at; these are for reading. The app's inspect
                # tool decodes them client-side so a click reports a number immediately
                # and keeps working after the job has expired from the store.
                "values": {
                    str(lead): _encode_values(res.forecast.log_chl[i], res.forecast.water)
                    for i, lead in enumerate(res.forecast.lead_times)
                },
                # The input window in the same encoding, so a click can plot what the
                # satellite saw before the anchor on one axis with what the model says
                # comes after it. Only observed days are present -- a clouded day has no
                # frame at all, and inventing one would draw a line through a gap.
                # Measured on an 85x85 grid with seven observed days: 48 kB gzipped on
                # top of a 99 kB payload. Paid once per forecast rather than once per
                # click, which is the trade this encoding exists to make.
                "history": {
                    day.isoformat(): _encode_values(field, res.forecast.water)
                    for day, field in sorted(res.history.items())
                },
                "value_scale": [VALUE_LO, VALUE_HI],
            }
        )
        payload["cached"] = False
        _store_result(key, payload)
        _store_bulletin_input(key, res)
        _results[job.id] = payload
        return {"ready": True}

    job = _jobs.submit("adhoc-forecast", req.model_dump(mode="json"), work)
    # Results live beside the job store rather than in it (they hold PNGs and are only
    # fetched once), so they are pruned here, on the store's own eviction schedule.
    # Without this the dict grew by one full payload per forecast for the life of the
    # process.
    live = _jobs.ids()
    for stale in [jid for jid in _results if jid not in live]:
        _results.pop(stale, None)
    for stale in [jid for jid in _job_keys if jid not in live]:
        _job_keys.pop(stale, None)
    return job.as_dict()


def _store_bulletin_input(key: str, res) -> None:
    with _bulletin_lock:
        _bulletin_inputs[key] = [time.time(), res, None]
        while len(_bulletin_inputs) > MAX_BULLETIN_INPUTS:
            oldest = min(_bulletin_inputs, key=lambda k: _bulletin_inputs[k][0])
            _bulletin_inputs.pop(oldest, None)


@app.get(
    "/api/forecast/jobs/{job_id}/bulletin.pdf",
    tags=["app"],
    response_class=Response,
    responses={200: {"content": {"application/pdf": {}}}},
)
def forecast_bulletin(job_id: str) -> Response:
    """An NCCOS-style bulletin (PDF) for the area a finished job forecast.

    A summary page with the level, the route and every caveat, then the satellite imagery,
    the bloom forecast and -- for a current forecast -- the wind and mixing outlook
    (`habforecast.bulletin`). Rendered on first request and kept with the forecast.
    """
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="unknown or expired job")
    if job.status != "done":
        raise HTTPException(
            status_code=409, detail="The forecast is not finished yet; try again when it is."
        )
    with _bulletin_lock:
        entry = _bulletin_inputs.get(_job_keys.get(job_id, ""))
    if entry is None:
        raise HTTPException(
            status_code=410,
            detail="This forecast's inputs are no longer held in memory, so no bulletin can "
            "be drawn from it. Run the forecast again, then download the bulletin.",
        )
    _stored, res, pdf = entry
    if pdf is None:
        try:
            import matplotlib  # noqa: F401
        except ImportError:
            # An image built before matplotlib joined the `serve` extra: say so, rather
            # than a bare 500 the app can only report as "Internal Server Error".
            raise HTTPException(
                status_code=503,
                detail="This server was built without matplotlib, which draws the bulletin. "
                "Rebuild the API image (docker compose up -d --build api).",
            ) from None
        from habforecast.bulletin import area_bulletin_pdf

        pdf = area_bulletin_pdf(res)
        entry[2] = pdf
    slug = "".join(c if c.isalnum() else "_" for c in res.aoi.name).strip("_") or "area"
    name = f"bloom_watch_{slug}_{res.forecast.anchor_date.isoformat()}.pdf"
    return Response(
        content=pdf,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{name}"'},
    )


@app.get("/api/forecast/jobs/{job_id}", tags=["app"])
def poll_forecast(job_id: str) -> dict:
    """Poll a job. The full result, images included, arrives once status is `done`."""
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="unknown or expired job")
    payload = job.as_dict()
    if job.status == "done":
        payload["result"] = _results.get(job_id)
    return payload


@app.get("/api/jobs/stats", tags=["ops"])
def job_stats() -> dict:
    return _jobs.stats()


@app.get("/app", response_class=HTMLResponse, tags=["ui"])
def app_page() -> str:
    """The interactive forecast application."""
    path = Path(__file__).resolve().parents[1] / "web" / "app.html"
    if not path.exists():
        raise HTTPException(status_code=500, detail="app.html missing from the image")
    return path.read_text(encoding="utf-8")


@app.get("/metrics", response_class=PlainTextResponse, tags=["ops"])
def metrics() -> str:
    """Prometheus exposition. Scraped by the monitoring stack."""
    try:
        from prometheus_client import CONTENT_TYPE_LATEST, generate_latest  # noqa: F401

        return generate_latest().decode("utf-8")
    except ImportError:
        return "# prometheus_client not installed\n"


@app.get("/dashboard", response_class=HTMLResponse, tags=["ui"])
def dashboard() -> str:
    """Evaluation results: skill against both baselines, plus the forecast imagery.

    Rendered on request from `artifacts/eval_*/scores.json` and `case_studies.json` rather
    than served as a pre-built file, so it always reflects the most recent evaluation
    instead of whatever was last committed.

    The case-study imagery is optional: it is expensive to produce (every checkpoint is
    run over every episode) and is therefore generated offline by
    `scripts/make_case_studies.py`. Without it the page still renders, minus the pictures.
    """
    import sys

    scripts = str(Path(__file__).resolve().parents[2] / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    try:
        from make_dashboard import build, load_cases, load_runs
    except ImportError as exc:
        raise HTTPException(
            status_code=500, detail=f"dashboard renderer unavailable: {exc}"
        ) from exc

    runs = load_runs(ARTIFACTS)
    if not runs:
        return (
            "<h1>No evaluation results yet</h1>"
            f"<p>Nothing matching <code>eval_*/scores.json</code> under <code>{ARTIFACTS}</code>."
            " Run <code>python pipelines/evaluate.py --checkpoint &lt;ckpt&gt;</code> first.</p>"
            "<p><a href='/'>&larr; forecast map</a></p>"
        )
    return build(runs, "habforecast - results", load_cases(ARTIFACTS))


@app.get("/", response_class=HTMLResponse, tags=["ui"])
def index() -> str:
    """Minimal map UI, served inline so the image has no static-asset dependency."""
    html_path = Path(__file__).resolve().parents[1] / "web" / "index.html"
    if html_path.exists():
        return html_path.read_text(encoding="utf-8")
    return "<h1>habforecast</h1><p>See <a href='/docs'>/docs</a>.</p>"


def _data_uri(png: bytes) -> str:
    import base64

    return "data:image/png;base64," + base64.b64encode(png).decode("ascii")


# Range of log10(chl) the 16-bit probe codes span. Wider than the colour ramp's 0..2 on
# purpose: the ramp saturates at its ends by design, but a value read from a click must
# report what the model actually said, including 0.3 mg/m3 clear water and a 300 mg/m3
# scum. 65534 steps over 4 decades is a quantisation error of 0.014% -- four orders of
# magnitude below the retrieval's own uncertainty, so it is not a lossy answer.
VALUE_LO, VALUE_HI = -1.0, 3.0
NO_VALUE = 0  # reserved code: land, cloud, or a pixel the model declined to predict


def _encode_values(log_chl: np.ndarray, water: np.ndarray) -> str:
    """Quantise a log-chlorophyll field to base64 little-endian uint16 for the client.

    Sending the field itself, rather than having the browser read colours back out of the
    overlay PNG, is the difference between reporting a measurement and reverse-engineering
    a picture: the PNG is quantised to 8 bits per channel through a five-stop ramp, so two
    concentrations a factor of two apart can share a pixel value near the ends of it.
    """
    import base64

    valid = water & np.isfinite(log_chl)
    norm = (np.nan_to_num(log_chl, nan=0.0) - VALUE_LO) / (VALUE_HI - VALUE_LO)
    codes = np.rint(np.clip(norm, 0.0, 1.0) * 65534.0).astype(np.uint16) + 1
    codes[~valid] = NO_VALUE
    return base64.b64encode(codes.astype("<u2").tobytes()).decode("ascii")


def _render_png(log_chl: np.ndarray, water: np.ndarray) -> bytes:
    """Colour-map a log-chlorophyll field to RGBA PNG bytes.

    Non-water and unpredicted pixels are fully transparent so the overlay does not
    paint a rectangle over the basemap.
    """
    from PIL import Image

    # Fixed scale: 1 to 100 mg/m^3 in log space. A fixed scale matters -- an
    # auto-scaled image makes a calm lake look identical to a blooming one.
    lo, hi = 0.0, 2.0
    norm = np.clip((log_chl - lo) / (hi - lo), 0.0, 1.0)
    valid = water & np.isfinite(log_chl)

    # Blue -> green -> yellow -> red, matching how bloom severity is conventionally shown.
    stops = np.array(
        [[38, 70, 160], [40, 160, 140], [180, 210, 60], [240, 180, 40], [200, 40, 40]],
        dtype=np.float64,
    )
    pos = np.linspace(0.0, 1.0, len(stops))
    flat = np.nan_to_num(norm, nan=0.0).ravel()
    rgb = np.stack([np.interp(flat, pos, stops[:, c]) for c in range(3)], axis=-1)
    rgb = rgb.reshape((*norm.shape, 3)).astype(np.uint8)

    alpha = np.where(valid, 220, 0).astype(np.uint8)
    rgba = np.dstack([rgb, alpha])

    buf = io.BytesIO()
    # zlib's default level. `optimize=True` measured twice the encode time for no
    # smaller a file on these overlays, which are mostly transparent.
    Image.fromarray(rgba, mode="RGBA").save(buf, format="PNG", compress_level=6)
    return buf.getvalue()
