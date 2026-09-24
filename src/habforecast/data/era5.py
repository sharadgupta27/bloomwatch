"""ERA5 meteorological forcing: the driver the forecaster has never been allowed to see.

Everything this model knows about why a bloom grows or disperses, it has had to infer
from where chlorophyll has been. The physics says otherwise. Cyanobacteria regulate their
buoyancy and float to the surface in calm water, which is when a satellite sees a scum;
a day of wind mixes them down through the column and the same bloom largely disappears
from the reflectance, having gone nowhere. Temperature sets the growth rate, light sets
the photosynthesis, and rain sets the nutrient pulse a few days later. A fortnight of
chlorophyll images contains none of this.

**ERA5, not ERA5-Land.** ERA5-Land is the obvious pick -- 9 km against 31 km -- and it is
the wrong one here: it is masked to *land*, so over the middle of Lake Victoria or Erie,
which is the only water this project cares about, it is empty. ERA5 single-levels is
global and continuous over water. The cost is resolution, and at 0.25 degrees a lake is
a handful of cells; that is acceptable because synoptic weather is smooth at this scale,
and it is why the store below keeps ERA5's native cells rather than pretending to 300 m.

**The temporal-leakage trap, which is the whole difficulty.** Weather at the *target*
date is the future. A forecast issued on day T for T+7 cannot know it. Reanalysis makes
it trivially available, and using it would inflate every score in this repository while
describing a system nobody can run. So the two cases are kept apart and named:

* *past forcing* -- weather over the input window, up to and including the anchor. Always
  legitimate, always available operationally, and this is the default.
* *perfect prognosis* -- weather at the target dates as well. This is a measurement of
  the ceiling: "what would this be worth if the weather forecast were perfect?" It is a
  real and standard experiment, and it is not a deployable number. It is off by default,
  recorded in the checkpoint config, and stamped on every evaluation row it touches.

Operationally the gap between the two is the value of wiring in an actual NWP forecast
(ECMWF open data, GFS), which is a different ingest and not built here.

**Why a separate store.** These variables do not come from the OLCI granule, they exist
on days the satellite saw nothing, and the existing cubes are what the current
checkpoints were trained from. Writing them into `data/processed/cubes` would entangle
two providers in one artefact and rewrite files whose contents a published number depends
on. They live in `data/processed/met/<lake_id>.zarr` and are joined at load time.

That density is the point, incidentally. Chlorophyll is missing on 55-70% of days; ERA5
is missing on none. On a fortnight where the satellite saw two frames, the weather still
describes all fourteen, and the gaps are exactly where this model has been weakest.
"""

from __future__ import annotations

import calendar
import io
import logging
import time
import zipfile
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from habforecast.aoi import LakeAOI

logger = logging.getLogger(__name__)

CDS_DATASET = "reanalysis-era5-single-levels"

# Hours sampled per day. **Measured, not assumed.** CDS prices a request by the number of
# fields (variables x days x times) and refuses one past its cap: a full year of five
# hourly variables is 43,800 fields and comes back "cost limits exceeded", while three
# months (11,160) is accepted. Subsampling buys back that headroom, and against a full
# hourly month over Lake Erie it costs:
#
#     sampling     daily-mean wind MAE      daily radiation error
#     3-hourly       0.084 m/s (2.2%)             1.6%
#     6-hourly       0.188 m/s (max 1.03)         9.4%
#     12-hourly      0.381 m/s (max 1.94)        66.6%
#
# 3-hourly is indistinguishable from hourly for the calm-versus-mixed signal this forcing
# exists to carry; 6-hourly starts to lose the radiation total, which is an accumulation
# with a strong diurnal shape. The daily reduction below rescales accumulated variables by
# whatever step it finds in the data, so switching this back to 1 needs no other change.
SAMPLE_HOURS = 3

# Field budget per request, under the cap that rejected a full year and above the three
# months that were accepted. Chunks are packed up to this.
MAX_FIELDS_PER_REQUEST = 11_000

# ERA5 is 0.25 degrees. A bbox is padded by this much so that a lake smaller than one
# cell still lands on at least a 2x2 neighbourhood and bilinear interpolation has
# something to work with.
BBOX_PAD_DEG = 0.5


@dataclass(frozen=True)
class MetVariable:
    """One forcing variable: how CDS names it, how a day is reduced, and why it is here."""

    key: str
    cds_name: str
    # "mean" for a state (temperature, wind), "sum" for a flux accumulated over the hour
    # (radiation, precipitation). Getting this backwards silently scales a variable by 24.
    agg: str
    units: str
    why: str


# Kept deliberately short. Each variable has a mechanism behind it; a channel that merely
# correlates with season is already covered by the day-of-year encoding and the
# climatology input, and would cost training data for nothing.
MET_VARIABLES: dict[str, MetVariable] = {
    "wind_speed": MetVariable(
        "wind_speed",
        "",  # derived from the two components below, never requested directly
        "mean",
        "m s-1",
        "The dominant control on whether a bloom is visible at all: cyanobacteria float "
        "up in calm water and are mixed down by wind, so surface chlorophyll can halve "
        "overnight with no change in biomass.",
    ),
    "wind_u": MetVariable(
        "wind_u",
        "10m_u_component_of_wind",
        "mean",
        "m s-1",
        "Eastward wind. With the northward component it gives the direction a surface "
        "scum is pushed, which is what moves a bloom across a basin between overpasses.",
    ),
    "wind_v": MetVariable(
        "wind_v", "10m_v_component_of_wind", "mean", "m s-1", "Northward wind; see wind_u."
    ),
    "t2m": MetVariable(
        "t2m",
        "2m_temperature",
        "mean",
        "degC",
        "Air temperature, the closest freely available proxy for lake surface temperature "
        "over inland water, which sets cyanobacterial growth rate and their advantage "
        "over other algae above roughly 20 degC.",
    ),
    "ssrd": MetVariable(
        "ssrd",
        "surface_solar_radiation_downwards",
        "sum",
        "MJ m-2 d-1",
        "Light available for photosynthesis, and a cloud proxy: a dull day both slows "
        "growth and is the reason the satellite saw nothing.",
    ),
    "tp": MetVariable(
        "tp",
        "total_precipitation",
        "sum",
        "mm d-1",
        "Rain drives the nutrient runoff that feeds a bloom days later, and stratifies "
        "the surface layer afterwards.",
    ),
}

# What is actually requested from CDS: the derived wind speed has no CDS name.
REQUESTED_VARIABLES: tuple[str, ...] = tuple(
    v.cds_name for v in MET_VARIABLES.values() if v.cds_name
)


def bbox_for(aoi: LakeAOI, pad: float = BBOX_PAD_DEG) -> list[float]:
    """CDS `area` for a lake: [north, west, south, east], padded and clamped."""
    west, south, east, north = aoi.bbox
    return [
        min(90.0, north + pad),
        max(-180.0, west - pad),
        max(-90.0, south - pad),
        min(180.0, east + pad),
    ]


# ERA5 (via its ERA5T preliminary stream) is published about five days behind real time.
# Days after `today - ERA5_LAG_DAYS` are not requested: CDS has nothing for them yet.
ERA5_LAG_DAYS = 6


def availability_cutoff(today: date | None = None) -> date:
    """The last day ERA5 can be assumed to hold."""
    return (today or date.today()) - timedelta(days=ERA5_LAG_DAYS)


def raw_path(
    root: str | Path, lake_id: str, year: int, months: Sequence[int], until: date | None = None
) -> Path:
    """Cache file for one lake-year-chunk. A chunk cut short at `until` (the current year)
    carries the cut-off in its name, so it is refetched once more days exist instead of
    being mistaken forever for the complete chunk."""
    tag = f"{min(months):02d}-{max(months):02d}"
    if until is not None:
        tag += f"_to{until:%Y%m%d}"
    return Path(root) / f"{lake_id}_{year}_{tag}.nc"


def needed_months(aoi: LakeAOI, cube_path: str | Path | None, input_days: int = 14) -> set[int]:
    """Which months of the year this lake actually needs weather for.

    Taken from the cube's own time axis where there is one: the model only ever asks about
    days the satellite has an entry for, so downloading January for a lake whose cube
    starts in June is queue time spent on data nothing reads. The month before each
    present month is included because a sample anchored on the 3rd of a month has an input
    window reaching back into the one before it.

    Falls back to the registry's bloom season, padded the same way, when no cube exists.
    """
    months: set[int] = set()
    if cube_path is not None and Path(cube_path).exists():
        try:
            with xr.open_zarr(cube_path, consolidated=True) as ds:
                months = {int(m) for m in pd.to_datetime(ds["time"].values).month}
        except Exception as exc:  # a missing or half-written cube must not stop the fetch
            logger.warning(
                "%s: could not read %s (%s); using the bloom season", aoi.lake_id, cube_path, exc
            )
    if not months:
        start, end = aoi.bloom_season
        months = (
            set(range(start, end + 1))
            if start <= end
            else set(range(start, 13)) | set(range(1, end + 1))
        )
    # One month of lead-in for the input window, wrapping at the year boundary.
    padded = set(months)
    for m in months:
        padded.add(12 if m == 1 else m - 1)
    return padded


def chunk_months(months: Sequence[int], n_vars: int, n_times: int) -> list[list[int]]:
    """Pack consecutive months into the largest requests the cost cap allows."""
    out: list[list[int]] = []
    current: list[int] = []
    for month in sorted(months):
        trial = [*current, month]
        days = sum(calendar.monthrange(2024, m)[1] for m in trial)  # leap year: upper bound
        if current and n_vars * days * n_times > MAX_FIELDS_PER_REQUEST:
            out.append(current)
            current = [month]
        else:
            current = trial
    if current:
        out.append(current)
    return out


def fetch_chunk(
    aoi: LakeAOI,
    year: int,
    months: Sequence[int],
    cache_root: str | Path,
    client=None,
    force: bool = False,
    until: date | None = None,
) -> Path:
    """Download one lake-year-chunk of ERA5 into the cache, or reuse what is there.

    `until` stops the request at that day (inclusive); used for the current year, whose
    later days do not exist yet.

    The file is written to a temporary name and moved into place, so an interrupted
    download can never be mistaken on the next run for a complete one.
    """
    out = raw_path(cache_root, aoi.lake_id, year, months, until)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and not force:
        logger.info("%s %d %s: cached", aoi.lake_id, year, out.stem.split("_")[-1])
        return out

    if client is None:
        import cdsapi

        client = cdsapi.Client()

    times = [f"{h:02d}:00" for h in range(0, 24, SAMPLE_HOURS)]
    request = {
        "product_type": ["reanalysis"],
        "variable": list(REQUESTED_VARIABLES),
        "year": [str(year)],
        "month": [f"{m:02d}" for m in sorted(months)],
        # Past the cut-off only in the chunk's final month; earlier months are complete.
        "day": [
            f"{d:02d}"
            for d in range(
                1,
                32 if until is None or max(months) < until.month else until.day + 1,
            )
        ],
        "time": times,
        "area": bbox_for(aoi),
        "data_format": "netcdf",
        "download_format": "unarchived",
    }
    tmp = out.with_suffix(".nc.part")
    t0 = time.time()
    logger.info(
        "%s %d months %s: requesting (%d fields)",
        aoi.lake_id,
        year,
        "+".join(f"{m:02d}" for m in sorted(months)),
        len(REQUESTED_VARIABLES)
        * sum(calendar.monthrange(year, m)[1] for m in months)
        * len(times),
    )
    client.retrieve(CDS_DATASET, request, str(tmp))
    tmp.replace(out)
    logger.info(
        "%s %d: wrote %s (%.1f MB) in %.0fs",
        aoi.lake_id,
        year,
        out.name,
        out.stat().st_size / 1e6,
        time.time() - t0,
    )
    return out


def open_raw(path: str | Path) -> xr.Dataset:
    """Open a CDS delivery, whatever shape it arrived in.

    Two things about the current CDS make this more than `xr.open_dataset`:

    1. **A multi-variable request comes back as a ZIP, and `download_format:
       "unarchived"` does not prevent it.** The archive holds one NetCDF per
       `stepType` -- `...stepType-instant.nc` with the state variables (wind components,
       temperature) and `...stepType-accum.nc` with the fluxes accumulated over the hour
       (radiation, precipitation). The file still has a `.nc` name, so the failure is an
       HDF5 "file signature not found" several frames below the cause. A single-variable
       request really does return a plain NetCDF, which is why a probe can pass and the
       real ingest fail.
    2. **xarray cannot guess the engine here.** `netcdf4` is not installed in this
       environment and the error it raises lists both candidate backends without saying
       either is missing, so the engine is named explicitly.

    The members are merged and loaded eagerly: they share coordinates and differ only in
    their variables, and a lake-year of 0.25 degree weather is a few megabytes.
    """
    path = Path(path)
    if not zipfile.is_zipfile(path):
        return xr.open_dataset(path, engine="h5netcdf")
    with zipfile.ZipFile(path) as zf:
        members = [n for n in zf.namelist() if n.endswith(".nc")]
        if not members:
            raise ValueError(f"{path} is a zip with no NetCDF members: {zf.namelist()}")
        parts = [xr.open_dataset(io.BytesIO(zf.read(n)), engine="h5netcdf").load() for n in members]
    # compat is explicit: the members carry disjoint variables on identical coordinates,
    # so there is nothing to reconcile, and xarray warns about the default changing.
    return xr.merge(parts, compat="override", combine_attrs="override")


def _time_dim(ds: xr.Dataset) -> str:
    """The name of the time coordinate.

    The current CDS delivers `valid_time`; the retired one delivered `time`, and cached
    files from either are still readable. Picking the wrong one raises a KeyError deep
    inside the aggregation, well away from the cause.
    """
    for name in ("valid_time", "time"):
        if name in ds.dims:
            return name
    raise KeyError(f"no recognised time dimension in {list(ds.dims)}")


def to_daily(ds: xr.Dataset) -> xr.Dataset:
    """Reduce an hourly ERA5 dataset to daily fields, in usable units.

    Wind speed is computed **per hour and then averaged**. Averaging the components first
    and taking the magnitude of the mean vector is a different and wrong quantity: a day
    that blows hard east all morning and hard west all afternoon averages to a near-zero
    vector, and would be recorded as the calm, scum-forming day it was the opposite of.
    """
    tname = _time_dim(ds)
    ds = ds.rename({tname: "time"})
    # Accumulated variables are per-hour totals, so a subsampled series sums to a
    # fraction of the day. Scaling by the step recovers the daily total (measured at
    # 1.6% error for 3-hourly); inferring the step rather than taking it on trust means a
    # cache holding a mix of hourly and 3-hourly files still reduces correctly.
    step = 1.0
    if ds.sizes.get("time", 0) > 1:
        deltas = np.diff(ds["time"].values).astype("timedelta64[m]").astype(float) / 60.0
        step = float(np.median(deltas)) or 1.0
    # Some deliveries carry a singleton expver/number dimension; squeeze it away so the
    # result is always (time, lat, lon).
    for extra in ("expver", "number"):
        if extra in ds.dims:
            ds = ds.isel({extra: 0}, drop=True)

    out = {}
    day = ds["time"].dt.floor("D")

    if "u10" in ds and "v10" in ds:
        speed = (ds["u10"] ** 2 + ds["v10"] ** 2) ** 0.5
        out["wind_speed"] = speed.groupby(day).mean()
        out["wind_u"] = ds["u10"].groupby(day).mean()
        out["wind_v"] = ds["v10"].groupby(day).mean()
    if "t2m" in ds:
        out["t2m"] = ds["t2m"].groupby(day).mean() - 273.15  # K -> degC
    if "ssrd" in ds:
        # Hourly accumulation in J m-2; a day is their sum, reported in MJ m-2.
        out["ssrd"] = ds["ssrd"].groupby(day).sum() * step / 1e6
    if "tp" in ds:
        out["tp"] = ds["tp"].groupby(day).sum() * step * 1000.0  # m -> mm

    missing = [k for k in ("wind_speed", "t2m", "ssrd", "tp") if k not in out]
    if missing:
        # A silent delivery change that drops a variable would otherwise train a model on
        # an all-zero channel and look like the forcing simply not helping.
        logger.warning(
            "ERA5 reduction is missing %s -- delivered variables were %s",
            missing,
            sorted(map(str, ds.data_vars)),
        )

    daily = xr.Dataset(out)
    if "group" in daily.dims:  # older xarray names the grouped axis differently
        daily = daily.rename({"group": "time"})
    daily = daily.rename({d: "time" for d in daily.dims if str(d).startswith("floor")})
    for key, var in MET_VARIABLES.items():
        if key in daily:
            daily[key].attrs = {"units": var.units, "why": var.why, "aggregation": var.agg}
    daily.attrs["source"] = f"ECMWF ERA5 reanalysis ({CDS_DATASET}) via the Copernicus CDS"
    daily.attrs["licence"] = "Copernicus Climate Change Service (C3S) licence"
    return daily


def build_store(
    aoi: LakeAOI,
    years: list[int],
    cache_root: str | Path,
    store_root: str | Path,
    client=None,
    cube_path: str | Path | None = None,
    workers: int = 3,
) -> Path:
    """Fetch every needed chunk, reduce to daily, and write one store per lake.

    Idempotent in the same way the chlorophyll cube is: chunks already downloaded are
    reused from the cache, and the store is rewritten from the full set each time, which
    is cheap because a lake-decade of 0.25 degree weather is a few megabytes.

    Requests run concurrently because the cost is queue latency, not bandwidth: a chunk
    measured 278-506 s almost all of it waiting. CDS serialises past its own concurrency
    limit, so an over-eager pool queues rather than failing.
    """
    months = sorted(needed_months(aoi, cube_path))
    times_per_day = len(range(0, 24, SAMPLE_HOURS))
    cutoff = availability_cutoff()
    chunks = []
    for year in sorted(years):
        if year > cutoff.year:
            continue
        # The current year: only months that have started by the cut-off, the last of
        # them requested up to the cut-off day.
        usable = [m for m in months if year < cutoff.year or m <= cutoff.month]
        for group in chunk_months(usable, len(REQUESTED_VARIABLES), times_per_day):
            partial = year == cutoff.year and max(group) == cutoff.month
            chunks.append((year, group, cutoff if partial else None))
    logger.info(
        "%s: %d chunks (months %s, %d-hourly)",
        aoi.lake_id,
        len(chunks),
        ",".join(f"{m:02d}" for m in months),
        SAMPLE_HOURS,
    )

    paths: list[Path] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(fetch_chunk, aoi, year, group, cache_root, client, until=until): (
                year,
                group,
            )
            for year, group, until in chunks
        }
        for fut in as_completed(futures):
            year, group = futures[fut]
            try:
                paths.append(fut.result())
            except Exception as exc:
                logger.error("%s %d %s: %s", aoi.lake_id, year, group, exc)
                raise

    frames = []
    for nc in sorted(paths):
        ds = open_raw(nc)
        try:
            frames.append(to_daily(ds).load())
        finally:
            ds.close()
    if not frames:
        raise ValueError(f"{aoi.lake_id}: no years requested")

    daily = xr.concat(frames, dim="time").sortby("time")
    _, keep = np.unique(daily["time"].values, return_index=True)
    daily = daily.isel(time=sorted(keep))

    out = Path(store_root) / f"{aoi.lake_id}.zarr"
    out.parent.mkdir(parents=True, exist_ok=True)
    daily.attrs["lake_id"] = aoi.lake_id
    daily.chunk({"time": 512}).to_zarr(out, mode="w", consolidated=True)
    logger.info(
        "%s: met store %s -- %d days %s to %s, %.2f MB",
        aoi.lake_id,
        out,
        daily.sizes["time"],
        str(daily["time"].values[0])[:10],
        str(daily["time"].values[-1])[:10],
        sum(f.stat().st_size for f in out.rglob("*") if f.is_file()) / 1e6,
    )
    return out


# --------------------------------------------------------------------------------------
# Open-Meteo: the same ERA5, served as point time series.
#
# CDS became the bottleneck once the registry grew: a lake-year request waits in a shared
# queue for hours (2.5 h per lake measured, 2026-09-22), and past a handful of queued
# requests an account is refused outright. ARCO-ERA5 (Google's cloud Zarr) was the next
# candidate and is the wrong shape: it is chunked one *global* field per hour, so a lake
# slice still downloads the globe for every timestep -- 100+ GB per lake-decade.
#
# Open-Meteo's historical API serves ERA5 (`models=era5`, 0.25 degrees, snapped to the
# native cell) as a time series per point: a lake-decade of hourly values is one request
# of a few seconds. The model reads the lake-mean daily value (`load_daily_means`), so one
# cell at the lake centre carries what it uses; the CDS stores averaged the cells of a
# padded bbox, and synoptic weather is smooth enough at 0.25 degrees that the two agree
# closely (compared on the eight lakes that have both, scripts/compare_met_sources.py).
#
# The free tier is metered: about 10,000 calls a day, where a call of more than two weeks
# counts as several. A lake-decade is ~260 of them, so the fetch below backs off and
# retries on HTTP 429 rather than failing a multi-lake build partway.
# --------------------------------------------------------------------------------------
OPEN_METEO_URL = "https://archive-api.open-meteo.com/v1/archive"
OPEN_METEO_HOURLY = (
    "temperature_2m",
    "wind_speed_10m",
    "wind_direction_10m",
    "shortwave_radiation",
    "precipitation",
)
OPEN_METEO_MAX_WAIT_S = 26 * 3600  # a daily quota resets within a day
# A year of hourly data weighs ~26 calls against a 600-a-minute limit, so requests are
# paced to stay under it rather than tripping it and waiting.
OPEN_METEO_PACE_S = 3.0


def _open_meteo_backoff(reason: str) -> float:
    """Seconds to wait for the quota a 429 names: minutely, hourly or daily."""
    reason = reason.lower()
    if "minutely" in reason:
        return 65.0
    if "hourly" in reason:
        return 15 * 60.0
    return 60 * 60.0


def lake_point(aoi: LakeAOI) -> tuple[float, float]:
    """(lat, lon) of the bbox centre -- the ERA5 cell Open-Meteo snaps it to is the lake's."""
    west, south, east, north = aoi.bbox
    return (south + north) / 2.0, (west + east) / 2.0


def fetch_open_meteo(
    aoi: LakeAOI,
    start: date,
    end: date,
    cache_root: str | Path,
    session=None,
    force: bool = False,
) -> Path:
    """Hourly ERA5 for the lake centre from Open-Meteo, cached as the raw JSON response."""
    import json

    import requests

    out = Path(cache_root) / f"{aoi.lake_id}_om_{start:%Y%m%d}_{end:%Y%m%d}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and not force:
        logger.info("%s %s..%s: cached", aoi.lake_id, start, end)
        return out
    lat, lon = lake_point(aoi)
    params = {
        "latitude": f"{lat:.4f}",
        "longitude": f"{lon:.4f}",
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "hourly": ",".join(OPEN_METEO_HOURLY),
        "models": "era5",
        "wind_speed_unit": "ms",
        "timezone": "GMT",
    }
    http = session or requests.Session()
    waited = 0.0
    while True:
        time.sleep(OPEN_METEO_PACE_S)
        t0 = time.time()
        resp = http.get(OPEN_METEO_URL, params=params, timeout=300)
        if resp.status_code == 429 and waited < OPEN_METEO_MAX_WAIT_S:
            pause = _open_meteo_backoff(resp.text)
            logger.warning(
                "%s: Open-Meteo rate limit (%s); retrying in %.0f s",
                aoi.lake_id,
                resp.text[:120],
                pause,
            )
            time.sleep(pause)
            waited += pause
            continue
        resp.raise_for_status()
        break
    payload = resp.json()
    if "hourly" not in payload:
        raise ValueError(f"{aoi.lake_id}: Open-Meteo returned no hourly data: {str(payload)[:200]}")
    tmp = out.with_suffix(".json.part")
    tmp.write_text(json.dumps(payload), encoding="utf-8")
    tmp.replace(out)
    logger.info(
        "%s %s..%s: Open-Meteo ERA5 cell (%.2f, %.2f) in %.1fs",
        aoi.lake_id,
        start,
        end,
        payload.get("latitude", float("nan")),
        payload.get("longitude", float("nan")),
        time.time() - t0,
    )
    return out


def open_meteo_to_daily(payload: dict) -> xr.Dataset:
    """Reduce an Open-Meteo hourly response to the daily variables `to_daily` produces.

    Same rules as the CDS path: wind speed is averaged per hour, never recomputed from
    mean components; the components are rebuilt from speed and the direction the wind
    blows *from* (meteorological convention, hence the minus signs); radiation is the
    hourly mean flux (W m-2) times 3600 s summed over the day, reported in MJ m-2; rain
    is the hourly total summed. Hours with a missing value leave that day NaN rather than
    a short sum, so a gap reads as missing forcing, not as a dry, dark day.
    """
    # Pure numpy on purpose. Parsing the timestamps through pandas builds a string index,
    # which pandas 3 backs with pyarrow -- and in a process that has loaded torch that
    # segfaults with no traceback (see load_daily_means). numpy parses ISO minutes itself.
    hourly = payload["hourly"]
    hours = np.asarray(hourly["time"], dtype="datetime64[m]")
    days, inverse = np.unique(hours.astype("datetime64[D]"), return_inverse=True)

    def col(name: str) -> np.ndarray:
        return np.asarray([np.nan if v is None else v for v in hourly[name]], dtype=np.float64)

    def daily_mean(v: np.ndarray) -> np.ndarray:
        ok = np.isfinite(v)
        total = np.bincount(inverse, weights=np.where(ok, v, 0.0), minlength=days.size)
        count = np.bincount(inverse, weights=ok.astype(float), minlength=days.size)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(count > 0, total / count, np.nan)

    def daily_sum(v: np.ndarray) -> np.ndarray:
        # A day with any missing hour is missing, not a partial total.
        ok = np.isfinite(v)
        total = np.bincount(inverse, weights=np.where(ok, v, 0.0), minlength=days.size)
        count = np.bincount(inverse, weights=ok.astype(float), minlength=days.size)
        return np.where(count >= 24, total, np.nan)

    speed = col("wind_speed_10m")
    theta = np.deg2rad(col("wind_direction_10m"))
    daily = {
        "wind_speed": daily_mean(speed),
        "wind_u": daily_mean(-speed * np.sin(theta)),
        "wind_v": daily_mean(-speed * np.cos(theta)),
        "t2m": daily_mean(col("temperature_2m")),
        "ssrd": daily_sum(col("shortwave_radiation") * 3600.0 / 1e6),
        "tp": daily_sum(col("precipitation")),
    }
    lat = float(payload.get("latitude", np.nan))
    lon = float(payload.get("longitude", np.nan))
    ds = xr.Dataset(
        {
            k: (("time", "latitude", "longitude"), daily[k].astype(np.float32)[:, None, None])
            for k in MET_VARIABLES
        },
        coords={
            "time": days.astype("datetime64[ns]"),
            "latitude": [lat],
            "longitude": [lon],
        },
    )
    for key, var in MET_VARIABLES.items():
        ds[key].attrs = {"units": var.units, "why": var.why, "aggregation": var.agg}
    ds.attrs["source"] = "ECMWF ERA5 reanalysis (0.25 deg) via the Open-Meteo historical API"
    ds.attrs["licence"] = "Copernicus C3S licence (ERA5); Open-Meteo data CC BY 4.0"
    return ds


def build_store_open_meteo(
    aoi: LakeAOI,
    start: date,
    end: date,
    cache_root: str | Path,
    store_root: str | Path,
    session=None,
) -> Path:
    """One lake's met store from Open-Meteo, in the same layout the CDS path writes."""
    import json

    end = min(end, availability_cutoff())
    paths = []
    # Year-sized requests: a cached year never needs refetching, and the current year's
    # file is keyed by its end date so it refreshes as days are added.
    for year in range(start.year, end.year + 1):
        y0 = max(start, date(year, 1, 1))
        y1 = min(end, date(year, 12, 31))
        paths.append(fetch_open_meteo(aoi, y0, y1, cache_root, session=session))
    frames = [open_meteo_to_daily(json.loads(p.read_text(encoding="utf-8"))) for p in paths]
    daily = xr.concat(frames, dim="time").sortby("time")
    _, keep = np.unique(daily["time"].values, return_index=True)
    daily = daily.isel(time=sorted(keep))
    out = Path(store_root) / f"{aoi.lake_id}.zarr"
    out.parent.mkdir(parents=True, exist_ok=True)
    daily.attrs["lake_id"] = aoi.lake_id
    daily.chunk({"time": 512}).to_zarr(out, mode="w", consolidated=True)
    logger.info(
        "%s: met store %s -- %d days %s to %s (Open-Meteo)",
        aoi.lake_id,
        out,
        daily.sizes["time"],
        str(daily["time"].values[0])[:10],
        str(daily["time"].values[-1])[:10],
    )
    return out


def load_daily_means(store_path: str | Path, keys: tuple[str, ...] | None = None) -> pd.DataFrame:
    """Lake-mean daily values, indexed by date -- what the model consumes.

    The spatial mean over the lake's ERA5 cells, not a field on the analysis grid. At
    0.25 degrees a lake spans a few cells and interpolating them to 300 m would fabricate
    structure the reanalysis does not contain; the honest signal is "how windy, warm and
    bright was it over this lake today".
    """
    keys = keys or (*(k for k in MET_VARIABLES if k != "wind_speed"), "wind_speed")
    with xr.open_zarr(store_path, consolidated=True) as ds:
        present = [k for k in keys if k in ds.data_vars]
        spatial = [d for d in ds.dims if d != "time"]
        means = ds[present].mean(dim=spatial)
        values = np.stack([means[k].values for k in present], axis=1).astype(np.float32)
        dates = pd.to_datetime(means["time"].values).date
    # Built from numpy with an object-dtype column index, not via `to_dataframe()`.
    # pandas 3 backs any inferred string index with pyarrow, and in an environment where
    # pyarrow was imported before torch (the conda base this venv layers on), building that
    # index segfaults the training process -- no traceback, the run just vanishes. An
    # explicit object index never reaches pyarrow.
    frame = pd.DataFrame(
        values,
        index=pd.Index(dates, name="date", dtype=object),
        columns=pd.Index(present, dtype=object),
    )
    return frame


def fit_normalisation(frames: list[pd.DataFrame]) -> dict[str, tuple[float, float]]:
    """Per-variable mean and standard deviation, from the training lakes only.

    Same rule as chlorophyll: a held-out lake standardised by its own weather has told
    the model something about itself. Wind speed in m/s and radiation in MJ/m2 are orders
    of magnitude apart, so they cannot share constants.
    """
    stats: dict[str, tuple[float, float]] = {}
    if not frames:
        return stats
    for key in frames[0].columns:
        pooled = np.concatenate([f[key].to_numpy(dtype=np.float32) for f in frames if key in f])
        pooled = pooled[np.isfinite(pooled)]
        if not pooled.size:
            continue
        std = float(pooled.std())
        stats[key] = (float(pooled.mean()), std if std > 1e-6 else 1.0)
    return stats


def series_matrix(
    frame: pd.DataFrame,
    days: list[date],
    stats: dict[str, tuple[float, float]],
    keys: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray]:
    """Standardised values and an availability mask for `days`, shaped (len(keys), T).

    A day the store does not cover -- before the archive starts, or after it ends --
    returns zero under a zero mask, the same convention the chlorophyll channels use for
    an unobserved frame. ERA5 is continuous, so in practice this only fires at the edges,
    and the mask is what stops the model from reading those zeros as calm, freezing days.
    """
    values = np.zeros((len(keys), len(days)), dtype=np.float32)
    mask = np.zeros((len(keys), len(days)), dtype=np.float32)
    if frame.empty:
        return values, mask
    rows = frame.reindex(pd.Index(days))
    for i, key in enumerate(keys):
        if key not in rows.columns:
            continue
        col = rows[key].to_numpy(dtype=np.float32)
        ok = np.isfinite(col)
        mean, std = stats.get(key, (0.0, 1.0))
        values[i] = np.where(ok, (col - mean) / std, 0.0)
        mask[i] = ok
    return values, mask
