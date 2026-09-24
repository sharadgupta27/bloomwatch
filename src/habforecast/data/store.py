"""Per-lake Zarr datacube: the analysis-ready store the model trains from.

Layout is one Zarr group per lake, dimensioned ``(time, y, x)`` on that lake's fixed
UTM grid, with a *daily* time axis.

Why daily rather than per-scene: Sentinel-3A and 3B both overpass, and a single lake
can be clipped by two or three granules in one day. Compositing to a daily frame roughly
doubles usable coverage and gives the forecasting model a regular time step, which
matters because a ConvLSTM over irregularly spaced frames would have to learn to
interpret gaps it is given no explicit signal about.

The store is *append-only and idempotent*. Re-running an ingest over a date range that
is already present is a no-op rather than a duplicate write, so a failed run can simply
be restarted. That property is what makes the ingest safe to schedule.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from dataclasses import field as dataclass_field
from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from habforecast.aoi import LakeGrid
from habforecast.data.aux_layers import AUX_LAYERS
from habforecast.data.olci import SceneObservation

logger = logging.getLogger(__name__)

# Chunk a month of frames together: ~10 MB per chunk at these grid sizes, which keeps
# random temporal access cheap without producing thousands of tiny files.
TIME_CHUNK = 32

FILL_VALUE = np.float32(np.nan)


@dataclass(frozen=True)
class DailyFrame:
    """One day of compositted observations for a lake."""

    day: date
    log_chl: np.ndarray  # (H, W) float32, NaN where no valid observation
    n_obs: np.ndarray  # (H, W) uint8, count of scenes contributing
    scene_ids: tuple[str, ...]
    mean_valid_fraction: float
    # Auxiliary physical layers, same grid, NaN where that layer had nothing. Kept in a
    # dict rather than as named fields so adding a layer is an ingest flag and not a
    # schema change (see data.aux_layers).
    aux: dict[str, np.ndarray] = dataclass_field(default_factory=dict)
    # Mean WQSF flag frequency across the scenes composited into this day. Persisted to a
    # sidecar (see LakeCube.write) so the drift monitor can detect an upstream processing
    # change -- a jump in CLOUD_MARGIN or AC_FAIL means the data changed, not the lakes.
    flag_stats: dict[str, float] = dataclass_field(default_factory=dict)

    @property
    def valid(self) -> np.ndarray:
        return self.n_obs > 0


def composite_day(day: date, observations: Sequence[SceneObservation]) -> DailyFrame | None:
    """Merge all scenes for one day into a single frame.

    Averaging is done in log space. That is deliberate: chlorophyll is log-normally
    distributed, so the mean of logs (a geometric mean) is the appropriate central
    estimate and avoids a single glint-contaminated pixel dominating the composite.
    """
    if not observations:
        return None

    shape = observations[0].log_chl.shape
    total = np.zeros(shape, dtype=np.float64)
    count = np.zeros(shape, dtype=np.uint16)

    for obs in observations:
        if obs.log_chl.shape != shape:
            raise ValueError(f"scene {obs.item_id} shape {obs.log_chl.shape} != {shape}")
        v = obs.valid & np.isfinite(obs.log_chl)
        total[v] += obs.log_chl[v]
        count[v] += 1

    if not count.any():
        return None

    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(count > 0, total / np.maximum(count, 1), np.nan)

    # Auxiliary layers are averaged over the scenes that actually carried them, which is
    # not the same set that carried chlorophyll: a band can saturate where the retrieval
    # succeeded. Counting each separately keeps a half-observed day honest instead of
    # borrowing chlorophyll's coverage for a channel that does not have it.
    aux_keys = sorted({k for obs in observations for k in obs.aux})
    aux: dict[str, np.ndarray] = {}
    for key in aux_keys:
        a_total = np.zeros(shape, dtype=np.float64)
        a_count = np.zeros(shape, dtype=np.uint16)
        for obs in observations:
            layer = obs.aux.get(key)
            if layer is None or layer.shape != shape:
                continue
            v = np.isfinite(layer)
            a_total[v] += layer[v]
            a_count[v] += 1
        with np.errstate(invalid="ignore", divide="ignore"):
            aux[key] = np.where(a_count > 0, a_total / np.maximum(a_count, 1), np.nan).astype(
                np.float32
            )

    return DailyFrame(
        day=day,
        aux=aux,
        log_chl=mean.astype(np.float32),
        n_obs=np.minimum(count, 255).astype(np.uint8),
        scene_ids=tuple(o.item_id for o in observations),
        mean_valid_fraction=float(np.mean([o.valid_fraction for o in observations])),
        flag_stats=_mean_flag_stats(observations),
    )


def _mean_flag_stats(observations: Sequence[SceneObservation]) -> dict[str, float]:
    """Average each WQSF flag's frequency across the scenes composited into one day."""
    names: set[str] = set()
    for o in observations:
        names.update(o.flag_stats)
    if not names:
        return {}
    return {
        name: float(np.mean([o.flag_stats.get(name, 0.0) for o in observations]))
        for name in sorted(names)
    }


class LakeCube:
    """Read/write interface to one lake's Zarr datacube."""

    def __init__(self, path: str | Path, grid: LakeGrid):
        self.path = Path(path)
        self.grid = grid

    # ------------------------------------------------------------------ properties --
    @property
    def exists(self) -> bool:
        return (self.path / ".zgroup").exists() or (self.path / "zarr.json").exists()

    def open(self) -> xr.Dataset:
        """Open the cube lazily (dask-backed)."""
        if not self.exists:
            raise FileNotFoundError(f"no cube at {self.path}")
        return xr.open_zarr(self.path, consolidated=True)

    def existing_days(self) -> set[date]:
        """Days already present -- the basis for idempotent re-ingest."""
        if not self.exists:
            return set()
        with xr.open_zarr(self.path, consolidated=True) as ds:
            return {pd.Timestamp(t).date() for t in ds.time.values}

    # ---------------------------------------------------------------------- writing --
    def _empty_like(self, frames: Sequence[DailyFrame]) -> xr.Dataset:
        """Build an in-memory Dataset for a batch of daily frames."""
        grid = self.grid
        times = pd.to_datetime([f.day for f in frames])
        log_chl = np.stack([f.log_chl for f in frames]).astype(np.float32)
        n_obs = np.stack([f.n_obs for f in frames]).astype(np.uint8)
        lon, lat = grid.lonlat

        ds = xr.Dataset(
            data_vars={
                "log_chl": (("time", "y", "x"), log_chl),
                "n_obs": (("time", "y", "x"), n_obs),
            },
            coords={
                "time": times,
                "y": grid.y_coords,
                "x": grid.x_coords,
                "lat": (("y", "x"), lat.astype(np.float32)),
                "lon": (("y", "x"), lon.astype(np.float32)),
            },
        )
        # Auxiliary layers, one variable each. Written only when present, so a cube
        # ingested without them stays byte-identical to what the existing checkpoints
        # were trained on and keeps opening in the old readers.
        for key in sorted({k for f in frames for k in f.aux}):
            stack = np.stack(
                [
                    f.aux[key]
                    if key in f.aux
                    else np.full(f.log_chl.shape, np.nan, dtype=np.float32)
                    for f in frames
                ]
            ).astype(np.float32)
            ds[key] = (("time", "y", "x"), stack)
            layer = AUX_LAYERS.get(key)
            ds[key].attrs = {
                "long_name": f"auxiliary layer {key}",
                "units": layer.units if layer else "1",
                "why": layer.why if layer else "",
                "source_product": "sentinel-3-olci-wfr-l2-netcdf",
                "compositing": "mean over scenes carrying this layer within the UTC day",
            }

        ds["log_chl"].attrs = {
            "long_name": "log10 chlorophyll-a concentration (OLCI CHL_NN, daily composite)",
            "units": "log10(mg m-3)",
            "source_product": "sentinel-3-olci-wfr-l2-netcdf:chl-nn",
            "compositing": "geometric mean of QC-passing scenes within the UTC day",
        }
        ds["n_obs"].attrs = {
            "long_name": "number of QC-passing scene observations contributing to the composite",
            "units": "1",
        }
        ds["y"].attrs = {"units": "m", "standard_name": "projection_y_coordinate"}
        ds["x"].attrs = {"units": "m", "standard_name": "projection_x_coordinate"}
        return ds

    def _cube_attrs(self) -> dict:
        grid = self.grid
        return {
            "lake_id": grid.lake_id,
            "crs_epsg": grid.epsg,
            "resolution_m": grid.resolution,
            "grid": json.dumps(asdict(grid)),
            "title": f"habforecast daily chlorophyll cube -- {grid.lake_id}",
            "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
            "conventions": "CF-1.10",
        }

    def write(self, frames: Sequence[DailyFrame]) -> int:
        """Append `frames` to the cube, skipping days already stored.

        Returns the number of days actually written.
        """
        if not frames:
            return 0

        frames = sorted(frames, key=lambda f: f.day)
        already = self.existing_days()
        new = [f for f in frames if f.day not in already]
        if not new:
            logger.info(
                "%s: all %d days already present -- nothing to write",
                self.grid.lake_id,
                len(frames),
            )
            return 0

        ds = self._empty_like(new)
        chunks = {"time": TIME_CHUNK, "y": self.grid.height, "x": self.grid.width}

        if not self.exists:
            self._write_fresh(ds.chunk(chunks))
        elif new[0].day > max(already):
            # Fast path: the new block is strictly later than everything stored, so a
            # plain append keeps the time axis sorted.
            #
            # The attrs must be re-set on every append. `to_zarr(mode="a")` overwrites the
            # group attributes with those of the dataset being appended, so appending a
            # block built by `_empty_like` (which carries none) silently erases the CRS,
            # resolution and grid definition. Because the ingest appends month after
            # month, this wiped the georeferencing metadata from every production cube on
            # the very first incremental write.
            #
            # The appended block is deliberately *not* dask-chunked. Zarr chunks the time
            # axis in blocks of TIME_CHUNK, and appending a dask-backed block whose chunk
            # boundaries do not line up with a partially-filled final zarr chunk is
            # refused outright ("would overlap multiple Dask chunks"). That happens for
            # any batch whose length is not a multiple of 32 -- which is most of them.
            # A block is a few megabytes, so writing it from memory is both correct and
            # cheap, and lets zarr fill the ragged chunk itself.
            ds.attrs.update(self._cube_attrs())
            ds.load().to_zarr(self.path, mode="a", append_dim="time", consolidated=True)
        else:
            # Out-of-order write (backfilling an earlier month, or re-running a range).
            # Zarr cannot insert into the middle of a dimension, and an unsorted time
            # axis would make every subsequent .sel() slice silently wrong. These cubes
            # are small -- a few tens of MB for a full multi-year lake -- so merging and
            # rewriting is both cheap and the only way to stay correct.
            logger.info(
                "%s: out-of-order write (%s <= %s) -- merging and rewriting cube",
                self.grid.lake_id,
                new[0].day,
                max(already),
            )
            with xr.open_zarr(self.path, consolidated=True) as existing:
                merged = xr.concat([existing.load(), ds.load()], dim="time")
            merged = merged.sortby("time")
            self._write_fresh(merged.chunk(chunks))

        self._append_flag_stats(new)
        logger.info("%s: wrote %d new daily frames to %s", self.grid.lake_id, len(new), self.path)
        return len(new)

    @property
    def flag_stats_path(self) -> Path:
        """Sidecar holding per-day WQSF flag frequencies."""
        return self.path.with_suffix(".flags.json")

    def _append_flag_stats(self, frames: Sequence[DailyFrame]) -> None:
        """Record per-day flag frequencies next to the cube.

        Kept in a sidecar rather than as a Zarr variable so the cube's schema stays fixed
        -- adding a variable would break appends to cubes written by an earlier version.

        These feed `monitoring.drift.compare_flag_frequencies`, which is how an upstream
        processing-baseline change is caught: a jump in CLOUD_MARGIN or AC_FAIL frequency
        means the *data* changed, not the lakes, and the response is to investigate the
        source rather than retrain.
        """
        stats = {f.day.isoformat(): f.flag_stats for f in frames if f.flag_stats}
        if not stats:
            return
        path = self.flag_stats_path
        existing: dict = {}
        if path.exists():
            try:
                existing = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                logger.warning("%s unreadable; rewriting", path)
        existing.update(stats)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(dict(sorted(existing.items())), indent=1), encoding="utf-8")

    def read_flag_stats(self) -> dict[str, dict[str, float]]:
        """Per-day flag frequencies, keyed by ISO date. Empty if never recorded."""
        path = self.flag_stats_path
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            logger.warning("%s unreadable", path)
            return {}

    def _write_fresh(self, ds: xr.Dataset) -> None:
        """(Re)write the whole cube, setting attrs and chunk encoding."""
        ds = ds.copy()
        ds.attrs.update(self._cube_attrs())
        self.path.parent.mkdir(parents=True, exist_ok=True)
        encoding = {
            "log_chl": {
                "chunks": (TIME_CHUNK, self.grid.height, self.grid.width),
                "_FillValue": FILL_VALUE,
            },
            "n_obs": {"chunks": (TIME_CHUNK, self.grid.height, self.grid.width)},
        }
        # Auxiliary layers take the chlorophyll chunking. Chunked differently they would
        # be read on a different boundary from the variable they accompany, which turns
        # one window read into two.
        for key in AUX_LAYERS:
            if key in ds.data_vars:
                encoding[key] = {
                    "chunks": (TIME_CHUNK, self.grid.height, self.grid.width),
                    "_FillValue": FILL_VALUE,
                }
        # Drop any encoding carried over from the opened store; it can conflict with
        # the explicit chunking above after a merge.
        for var in ds.variables:
            ds[var].encoding = {}
        # Written beside the cube and swapped in, never over it: `mode="w"` deletes the
        # store before writing, so a process killed mid-rewrite (a backfill merge, an
        # out-of-memory worker) used to leave a lake with no cube at all.
        import shutil

        tmp = self.path.with_name(self.path.name + ".tmp")
        old = self.path.with_name(self.path.name + ".old")
        for leftover in (tmp, old):
            if leftover.exists():
                shutil.rmtree(leftover)
        ds.to_zarr(tmp, mode="w", consolidated=True, encoding=encoding)
        if self.path.exists():
            _rename_with_retry(self.path, old)
        _rename_with_retry(tmp, self.path)
        if old.exists():
            shutil.rmtree(old)


def _rename_with_retry(src: Path, dst: Path, attempts: int = 8) -> None:
    """Rename, retrying while Windows holds a transient lock on a freshly written store.

    Defender / the search indexer briefly open new files; a rename during that window fails
    with WinError 5 and killed a 21-lake ingest (utah_lake, 2026-09-24).
    """
    import time

    for i in range(attempts):
        try:
            src.rename(dst)
            return
        except PermissionError:
            if i == attempts - 1:
                raise
            logger.warning("rename %s -> %s locked, retrying (%d/%d)", src, dst, i + 1, attempts)
            time.sleep(min(2**i, 30))


def group_by_day(observations: Iterable[SceneObservation]) -> dict[date, list[SceneObservation]]:
    """Bucket scene observations by their UTC acquisition date."""
    out: dict[date, list[SceneObservation]] = {}
    for obs in observations:
        out.setdefault(obs.time.astimezone(UTC).date(), []).append(obs)
    return out
