"""Sequence dataset: turns per-lake Zarr cubes into (past -> future) training samples.

The defining property of this dataset is that **most of it is missing**. Optical satellite
observation of lakes runs at roughly 25-40% temporal coverage once cloud, glint and
high-solar-zenith pixels are removed, and the gaps are not random -- they cluster around
the storm systems that also drive bloom dynamics.

Three consequences shape the design:

1. Days are placed on a *regular daily calendar*. The cube only stores days that had at
   least one usable scene, so a naive window over stored frames would silently mix a
   3-day gap with a 30-day gap. Reindexing onto real dates makes the gap explicit.

2. Every sample carries **mask channels** alongside the values. The model is told which
   pixels it can trust rather than being left to infer it from a sentinel value.

3. Supervision is masked. Loss is computed only where the target was actually observed,
   so the model is never asked to reproduce a cloud.

Normalisation statistics are computed on the training lakes only and passed in, never
recomputed per split -- otherwise the validation and held-out test lakes leak their own
distribution into the standardisation.
"""

from __future__ import annotations

import logging
import mmap
import os
from collections.abc import Sequence
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xarray as xr
from torch.utils.data import Dataset

from habforecast.features.normalisation import Normalisation

logger = logging.getLogger(__name__)

# Process-wide budget for cached climatology frames, shared by every LakeSeries. A
# training epoch asks for the same (day-of-year, year) frames it asked for last epoch --
# the anchors are fixed, only the crops move -- so the first epoch computes each one
# (~90 ms on the largest training lake) and every later epoch reads it back in
# microseconds. The three training lakes plus the validation lake need about 1 GB in
# float16; the default leaves headroom for the held-out lake at evaluation.
CLIM_CACHE_MAX_BYTES = int(float(os.environ.get("HAB_CLIM_CACHE_MB", "2048")) * 2**20)
_clim_cache_bytes = 0


def set_clim_cache_budget(max_bytes: int) -> None:
    """Cap this process's climatology cache. Each DataLoader worker holds its own cache,
    so the workers split the budget rather than each taking all of it."""
    global CLIM_CACHE_MAX_BYTES
    CLIM_CACHE_MAX_BYTES = int(max_bytes)


def _is_file_map(arr) -> bool:
    """A whole-file .npy map (np.load mmap_mode="r"), not a view of one or a copy."""
    return isinstance(arr, np.memmap) and bool(arr.filename) and isinstance(arr.base, mmap.mmap)


class _MappedNpy:
    """Pickle stand-in for a mapped .npy file: its path, reopened read-only on unpickle."""

    __slots__ = ("path", "shape")

    def __init__(self, path: str, shape: tuple[int, ...]):
        self.path, self.shape = path, shape

    def open(self) -> np.ndarray:
        arr = np.load(self.path, mmap_mode="r")
        if arr.shape != self.shape:
            raise RuntimeError(f"{self.path} changed shape since it was pickled")
        return arr


def _pack(v):
    return _MappedNpy(v.filename, v.shape) if _is_file_map(v) else v


def _unpack(v):
    return v.open() if isinstance(v, _MappedNpy) else v


@dataclass
class LakeSeries:
    """One lake's observations held in RAM, indexed by calendar date.

    Stored as float16: log10 chlorophyll spans roughly -1 to 2.5, so float16's ~3
    decimal digits are far finer than the retrieval's own uncertainty, and it halves
    the memory footprint of the whole training set.
    """

    lake_id: str
    split: str
    days: np.ndarray  # (N,) datetime64[D], sorted, gaps present
    values: np.ndarray  # (N, H, W) float16, NaN where unobserved
    water: np.ndarray  # (H, W) bool -- pixels ever observed as water
    shape: tuple[int, int]
    # Auxiliary layers, same calendar and grid: {key: (N, H, W) float16}. Their NaNs are
    # their own -- a day can carry chlorophyll and no usable reflectance -- so each gets
    # its own observation mask downstream rather than borrowing chlorophyll's.
    aux: dict[str, np.ndarray] = dataclass_field(default_factory=dict)
    # Per-layer standardisation, fitted on training lakes only, same as chlorophyll.
    aux_stats: dict[str, tuple[float, float]] = dataclass_field(default_factory=dict)
    # Daily ERA5 forcing for this lake, indexed by date (see data.era5). Unlike every
    # other input here it has no gaps: reanalysis covers the days the satellite missed,
    # which is exactly where the chlorophyll channels carry nothing.
    met: pd.DataFrame | None = None
    met_stats: dict[str, tuple[float, float]] = dataclass_field(default_factory=dict)
    # Where the seasonal climatology for this lake is drawn from. None means this series
    # itself, leave-one-year-out. A temporal split points a held-out year at the same
    # lake's training years instead, so that the climatology a 2024 forecast sees was
    # built from history that existed when the forecast was issued.
    clim_source: LakeSeries | None = None
    # The lake's level in each factor of habforecast.groups, as vocabulary indices. All
    # zero ("unknown") unless the data module sets them from the registry.
    groups: tuple[int, ...] = (0, 0, 0)

    _index: dict = None  # type: ignore[assignment]
    _doys: np.ndarray = None  # type: ignore[assignment]
    _years: np.ndarray = None  # type: ignore[assignment]
    _clim_cache: dict = None  # type: ignore[assignment]
    _zero_filled: np.ndarray = None  # type: ignore[assignment]
    _observed: np.ndarray = None  # type: ignore[assignment]

    def __post_init__(self):
        object.__setattr__(
            self, "_index", {pd.Timestamp(d).date(): i for i, d in enumerate(self.days)}
        )
        ts = pd.to_datetime(self.days)
        object.__setattr__(self, "_doys", np.asarray(ts.dayofyear, dtype=np.int32))
        object.__setattr__(self, "_years", np.asarray(ts.year, dtype=np.int32))
        object.__setattr__(self, "_clim_cache", {})

    # DataLoader workers on Windows are spawned and receive the dataset by pickle. A
    # pickled np.memmap is written out as a full in-memory copy, so every worker would
    # carry the whole mapped training set (8.5 GB for the global registry) -- pass the
    # file path instead and map it again on the other side.
    def __getstate__(self) -> dict:
        state = dict(self.__dict__)
        for k in ("values", "_zero_filled", "_observed"):
            state[k] = _pack(state[k])
        state["aux"] = {k: _pack(v) for k, v in self.aux.items()}
        state["_clim_cache"] = {}  # per process; rebuilt on demand under its own budget
        return state

    def __setstate__(self, state: dict) -> None:
        for k in ("values", "_zero_filled", "_observed"):
            state[k] = _unpack(state[k])
        state["aux"] = {k: _unpack(v) for k, v in state["aux"].items()}
        self.__dict__.update(state)

    def frame(self, day: date) -> np.ndarray | None:
        """The frame for `day`, or None when that date has no observation at all."""
        i = self._index.get(day)
        return None if i is None else self.values[i]

    def index_of(self, day: date) -> int | None:
        """Row of `day` in `values`, or None when that date has no observation."""
        return self._index.get(day)

    def aux_frame(self, key: str, day: date) -> np.ndarray | None:
        i = self._index.get(day)
        layer = self.aux.get(key)
        return None if i is None or layer is None else layer[i]

    @property
    def n_days(self) -> int:
        return len(self.days)

    def subset_years(self, years: Sequence[int], split: str) -> LakeSeries:
        """The same lake restricted to `years`, for a temporal split.

        The water mask is kept from the full record on purpose: it is a property of the
        lake, and a mask recomputed from two years of a held-out slice would shrink with
        that slice's cloud cover and hand the model a different lake than it trained on.
        """
        keep = np.isin(self._years, np.asarray(list(years), dtype=np.int32))
        return LakeSeries(
            lake_id=self.lake_id,
            split=split,
            days=self.days[keep],
            values=self.values[keep],
            water=self.water,
            shape=self.shape,
            aux={k: v[keep] for k, v in self.aux.items()},
            aux_stats=dict(self.aux_stats),
            # The met frame is not sliced: it is indexed by date and only ever read
            # for the days a sample asks about, and a held-out year's *past* window
            # legitimately reaches back into the preceding training year for weather
            # -- an operator on 2 January knows what December's weather was.
            met=self.met,
            met_stats=dict(self.met_stats),
            groups=self.groups,
        )

    def climatology(
        self,
        doy: int,
        exclude_year: int,
        box: tuple[slice, slice] | None = None,
        halfwidth: int = 15,
    ) -> np.ndarray | None:
        """Per-pixel seasonal mean around `doy`, from every year except `exclude_year`.

        Drawn from `clim_source` when set, otherwise from this series. Returns log10
        chlorophyll in physical units with NaN where no year ever observed the pixel in
        that window, or None when no frame falls in the window at all. Excluding the
        evaluated year is what keeps this an honest quantity: a climatology that includes
        the days it is later scored on is a leak, and it showed as one when it was tried.

        Frames are reduced in blocks of 16: the obvious `np.nanmean(values[sel], axis=0)`
        materialises two full copies of the selection and was enough to run the machine
        out of memory on the largest lake while other jobs held RAM.
        """
        src = self.clim_source if self.clim_source is not None else self
        frame = src._climatology_full(int(doy), int(exclude_year), int(halfwidth))
        if frame is None:
            return None
        if box is not None:
            frame = frame[box[0], box[1]]
        return frame.astype(np.float32)

    def prepare_climatology(self) -> None:
        """Build the NaN-free values and observation mask the climatology sums over.

        Built once: summing them replaces the per-call isfinite / where / astype passes
        over every frame. Call it before DataLoader workers start, so they inherit the
        finished arrays instead of each building (and, when mapped, writing) its own.
        """
        if self._zero_filled is not None:
            return
        if _is_file_map(self.values):
            # A memory-mapped series keeps its derived arrays on disk beside it, built a
            # block of frames at a time: two more in-RAM copies of every cube are what the
            # mapping exists to avoid. The .npy name already encodes the cube version, so
            # finished files are reused; they are written under a temporary name and
            # renamed, so a killed build never leaves a half-written file to be reused.
            base = Path(self.values.filename)
            paths = (base.with_suffix(".zero.npy"), base.with_suffix(".obs.npy"))
            if not all(p.exists() for p in paths):
                tmps = [p.with_name(p.name + ".part") for p in paths]
                zf = np.lib.format.open_memmap(tmps[0], "w+", np.float16, self.values.shape)
                ob = np.lib.format.open_memmap(tmps[1], "w+", np.bool_, self.values.shape)
                for i in range(0, self.values.shape[0], 64):
                    blk = np.asarray(self.values[i : i + 64])
                    ok = np.isfinite(blk)
                    ob[i : i + 64] = ok
                    zf[i : i + 64] = np.where(ok, blk, np.float16(0.0))
                zf.flush()
                ob.flush()
                del zf, ob  # release the maps: Windows cannot rename a mapped file
                for tmp, p in zip(tmps, paths, strict=True):
                    tmp.replace(p)
            zf, ob = (np.load(p, mmap_mode="r") for p in paths)
            if zf.shape != self.values.shape or ob.shape != self.values.shape:
                raise RuntimeError(f"{base}: stale climatology arrays -- delete *.zero/obs.npy")
            object.__setattr__(self, "_zero_filled", zf)
            object.__setattr__(self, "_observed", ob)
        else:
            observed = np.isfinite(self.values)
            object.__setattr__(self, "_observed", observed)
            object.__setattr__(
                self, "_zero_filled", np.where(observed, self.values, np.float16(0.0))
            )

    def _climatology_full(self, doy: int, exclude_year: int, halfwidth: int) -> np.ndarray | None:
        """Whole-grid seasonal mean, float16, cached under the process-wide budget."""
        global _clim_cache_bytes
        key = (doy, exclude_year, halfwidth)
        hit = self._clim_cache.get(key)
        if hit is not None:
            return hit
        delta = np.abs(self._doys - doy)
        delta = np.minimum(delta, 366 - delta)
        rows = np.flatnonzero((delta <= halfwidth) & (self._years != exclude_year))
        if rows.size == 0:
            return None
        self.prepare_climatology()
        acc = np.take(self._zero_filled, rows, axis=0).sum(axis=0, dtype=np.float32)
        cnt = np.take(self._observed, rows, axis=0).sum(axis=0, dtype=np.int32)
        with np.errstate(invalid="ignore", divide="ignore"):
            frame = np.where(cnt > 0, acc / np.maximum(cnt, 1), np.nan).astype(np.float16)
        if _clim_cache_bytes + frame.nbytes <= CLIM_CACHE_MAX_BYTES:
            self._clim_cache[key] = frame
            _clim_cache_bytes += frame.nbytes
        return frame


def load_lake_series(
    cube_path: str | Path,
    lake_id: str,
    split: str,
    min_water_frequency: float = 0.05,
    aux: Sequence[str] = (),
    met_root: str | Path | None = None,
    mmap_dir: str | Path | None = None,
) -> LakeSeries:
    """Load a lake cube into memory and derive its static water mask.

    The water mask is observation-derived rather than taken from a land/water product:
    a pixel counts as water if it produced a valid retrieval on at least
    `min_water_frequency` of stored days. This automatically excludes land, permanent
    cloud-shadowed corners, and the sliver of the bbox that is not actually lake.

    `aux` names extra variables to load alongside. A cube written before those layers
    existed simply does not have them, and is loaded without -- so one registry can mix
    cubes ingested at different times without the loader guessing.

    `mmap_dir` keeps the decoded arrays on disk as .npy files and maps them instead of
    holding them in RAM. The global registry is ~4x the expanded one, and the in-RAM
    series plus the climatology's two derived copies of every cube would need ~14 GB on
    a 15 GB machine. Random 128 px crops of a memory-mapped float16 stack cost a few
    page reads each, and the OS cache keeps the hot lakes resident anyway.
    """
    aux_values: dict[str, np.ndarray] = {}
    with xr.open_zarr(cube_path, consolidated=True) as ds:
        values = ds["log_chl"].values.astype(np.float32)
        days = ds["time"].values.astype("datetime64[D]")
        for key in aux:
            if key in ds.data_vars:
                aux_values[key] = ds[key].values.astype(np.float32)

    observed = np.isfinite(values)
    frequency = observed.mean(axis=0)
    water = frequency >= min_water_frequency

    # Blank out non-water pixels so they can never contribute to loss or statistics.
    values[:, ~water] = np.nan
    for key in aux_values:
        aux_values[key][:, ~water] = np.nan

    missing = [k for k in aux if k not in aux_values]
    logger.info(
        "%s: %d days, grid %s, water pixels %d/%d (%.1f%%), observed density %.1f%%%s",
        lake_id,
        len(days),
        values.shape[1:],
        int(water.sum()),
        water.size,
        100 * water.mean(),
        100 * observed[:, water].mean() if water.any() else 0.0,
        f", aux {'+'.join(aux_values)}" if aux_values else "",
    )
    if missing:
        logger.warning("%s: cube has no %s -- ingested before those layers", lake_id, missing)
    for key, layer in aux_values.items():
        cover = float(np.isfinite(layer[:, water]).mean()) if water.any() else 0.0
        logger.info("%s: %s coverage %.1f%% of water-days", lake_id, key, 100 * cover)

    met = None
    if met_root is not None:
        from habforecast.data import era5

        store = Path(met_root) / f"{lake_id}.zarr"
        if store.exists():
            met = era5.load_daily_means(store)
            logger.info(
                "%s: met forcing %d days %s to %s (%s)",
                lake_id,
                len(met),
                met.index.min(),
                met.index.max(),
                ", ".join(met.columns),
            )
        else:
            logger.warning(
                "%s: no met store at %s -- run pipelines/ingest_era5.py; the forcing "
                "channels will be masked off for this lake",
                lake_id,
                store,
            )

    values16 = values.astype(np.float16)
    del values
    aux16 = {k: v.astype(np.float16) for k, v in aux_values.items()}
    if mmap_dir is not None:
        root = Path(mmap_dir)
        values16 = _to_mmap(values16, root, f"{lake_id}.values", cube_path, min_water_frequency)
        aux16 = {
            k: _to_mmap(v, root, f"{lake_id}.{k}", cube_path, min_water_frequency)
            for k, v in aux16.items()
        }
    return LakeSeries(
        lake_id=lake_id,
        split=split,
        days=days,
        values=values16,
        water=water,
        shape=values16.shape[1:],
        aux=aux16,
        met=met,
    )


def _to_mmap(
    arr: np.ndarray, root: Path, stem: str, cube_path: str | Path, min_water_frequency: float
) -> np.memmap:
    """Write `arr` under `root` and return a read-only map of it.

    The file name carries a digest of the cube's metadata and the water threshold, so a
    re-ingested cube or a different mask never reads a stale decode.
    """
    import hashlib

    meta = Path(cube_path) / ".zmetadata"
    stamp = f"{Path(cube_path).resolve()}|{meta.stat().st_mtime_ns if meta.exists() else 0}"
    stamp += f"|{min_water_frequency}|{arr.shape}"
    digest = hashlib.sha1(stamp.encode()).hexdigest()[:10]
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{stem}.{digest}.npy"
    if not path.exists():
        tmp = path.with_name(path.name + ".part")
        with open(tmp, "wb") as fh:
            np.save(fh, arr)
        tmp.replace(path)
    return np.load(path, mmap_mode="r")


def fit_aux_normalisation(series: list[LakeSeries], keys: Sequence[str]) -> dict:
    """Per-layer mean and standard deviation, from the training lakes only.

    Separate from chlorophyll's `Normalisation` because the layers are different physical
    quantities on different scales -- an MCI of 0.01 and a log10 Kd of -0.4 cannot share a
    standardisation without one of them vanishing into the other's noise.
    """
    stats: dict[str, tuple[float, float]] = {}
    for key in keys:
        pooled = []
        for s in series:
            layer = s.aux.get(key)
            if layer is None:
                continue
            v = layer[np.isfinite(layer)].astype(np.float32)
            if v.size:
                pooled.append(v)
        if not pooled:
            continue
        flat = np.concatenate(pooled)
        std = float(flat.std())
        stats[key] = (float(flat.mean()), std if std > 1e-6 else 1.0)
    return stats


def fit_normalisation(series: list[LakeSeries]) -> Normalisation:
    """Fit standardisation on the given (training) lakes."""
    pooled = []
    for s in series:
        v = s.values[np.isfinite(s.values)].astype(np.float32)
        if v.size:
            pooled.append(v)
    if not pooled:
        raise ValueError("no finite observations to fit normalisation on")
    allv = np.concatenate(pooled)
    return Normalisation(mean=float(allv.mean()), std=float(allv.std() + 1e-6))


@dataclass(frozen=True)
class SampleSpec:
    """Locator for one training sample."""

    lake_idx: int
    anchor: date  # last day of the input window


class SequenceDataset(Dataset):
    """(input_days past frames) -> (forecast at anchor + lead) samples across lakes.

    Args:
        series: lakes to draw from.
        norm: standardisation fitted on the training lakes.
        input_days: length of the input window, in calendar days.
        lead_times: forecast lead times in days, e.g. (1, 3, 7).
        crop: if set, take a random `crop x crop` spatial window during training.
        min_input_frames: reject anchors whose input window holds fewer observed frames.
        min_target_coverage: reject anchors whose target frame has less than this
            fraction of the lake's water pixels observed -- no supervision otherwise.
        aux_keys: auxiliary layers to stack alongside chlorophyll. Each contributes its
            own value and mask channels, because a day can carry chlorophyll and no
            usable reflectance and the model has to be able to tell.
        aux_stats: per-layer (mean, std) from the training lakes, as
            `fit_aux_normalisation` returns.
        climatology: also return, per lead, the leave-one-year-out seasonal mean at the
            target date (`clim`) and where it exists (`clim_mask`). Off by default: it
            costs a masked mean over ~30-60 frames per lead per sample, and only the
            `use_clim` model reads it.
        clim_halfwidth: half-width in days of the seasonal window. 15 matches the
            climatology baseline, so the model's input is exactly the baseline it is
            asked to beat.
        clim_dropout: probability, per training sample, of withholding the climatology
            entirely. This is what teaches the model to forecast a lake with no history
            -- every ad-hoc area the app is asked about -- rather than to depend on a
            channel that is sometimes absent.
        met_keys: ERA5 forcing variables fed over the input window (see data.era5).
            Lake-mean daily scalars, not fields: at 0.25 degrees the reanalysis has no
            structure inside a lake to give.
        met_future: also feed the forcing at the target dates. **Perfect prognosis** --
            future weather, so it measures a ceiling rather than a deployable system,
            and every artefact it touches is stamped with it.
    """

    def __init__(
        self,
        series: list[LakeSeries],
        norm: Normalisation,
        input_days: int = 14,
        lead_times: tuple[int, ...] = (1, 3, 7),
        crop: int | None = 128,
        min_input_frames: int = 2,
        min_target_coverage: float = 0.05,
        augment: bool = False,
        seed: int = 0,
        aux_keys: Sequence[str] = (),
        aux_stats: dict[str, tuple[float, float]] | None = None,
        climatology: bool = False,
        clim_halfwidth: int = 15,
        clim_dropout: float = 0.0,
        met_keys: Sequence[str] = (),
        met_future: bool = False,
        group_dropout: float = 0.0,
    ):
        self.series = series
        # Probability, per factor and per training sample, of withholding the lake's group
        # level (see habforecast.groups): what keeps the multilevel model's population
        # level -- the forecast for an unregistered area -- trained.
        self.group_dropout = float(group_dropout)
        self.met_keys = tuple(met_keys)
        # Perfect prognosis: also hand the model the weather at the target dates. That
        # is the future, so it is never deployable -- see data/era5.py.
        self.met_future = bool(met_future)
        self.climatology = bool(climatology)
        self.clim_halfwidth = int(clim_halfwidth)
        self.clim_dropout = float(clim_dropout)
        self.norm = norm
        self.input_days = input_days
        self.lead_times = tuple(sorted(lead_times))
        self.max_lead = max(self.lead_times)
        self.crop = crop
        self.augment = augment
        self.rng = np.random.default_rng(seed)
        self.min_input_frames = min_input_frames
        self.min_target_coverage = min_target_coverage
        self.aux_keys = tuple(aux_keys)
        self.aux_stats = dict(aux_stats or {})
        self.samples = self._build_index()
        logger.info(
            "SequenceDataset: %d samples from %d lakes (input=%dd, leads=%s)",
            len(self.samples),
            len(series),
            input_days,
            self.lead_times,
        )

    # --------------------------------------------------------------- index building --
    def _build_index(self) -> list[SampleSpec]:
        samples: list[SampleSpec] = []
        for li, s in enumerate(self.series):
            if s.n_days == 0:
                continue
            observed_days = {pd.Timestamp(d).date() for d in s.days}
            water_n = max(int(s.water.sum()), 1)
            # Fraction of water pixels observed on each stored day, computed once per
            # lake. The per-(anchor, lead) loop below then costs a dict lookup rather
            # than an `np.isfinite` pass over a full frame.
            coverage = np.isfinite(s.values).reshape(s.n_days, -1).sum(axis=1) / water_n
            usable_day = {
                pd.Timestamp(d).date(): bool(c >= self.min_target_coverage)
                for d, c in zip(s.days, coverage, strict=True)
            }

            for anchor in sorted(observed_days):
                window = [anchor - timedelta(days=k) for k in range(self.input_days)]
                n_obs = sum(1 for d in window if d in observed_days)
                if n_obs < self.min_input_frames:
                    continue

                # Require at least one usable target across the requested leads.
                if any(
                    usable_day.get(anchor + timedelta(days=lead), False) for lead in self.lead_times
                ):
                    samples.append(SampleSpec(lake_idx=li, anchor=anchor))
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    # ---------------------------------------------------------------------- fetching --
    def _gather(
        self,
        stack: np.ndarray,
        rows: list[int | None],
        box: tuple[slice, slice],
        mean: float,
        std: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Standardised values and observation masks for `rows` of `stack`, over `box`.

        Rows are gathered in a single fancy-index over the *cropped* window, so a training
        sample touches 128x128 pixels per frame rather than a whole lake; float16 to
        float32 conversion, the NaN test and the standardisation all run on that block.
        Shapes (len(rows), h, w). A `None` row is a day with no observation and stays an
        all-zero frame with an all-zero mask.
        """
        n = len(rows)
        h = box[0].stop - box[0].start
        w = box[1].stop - box[1].start
        vals = np.zeros((n, h, w), dtype=np.float32)
        mask = np.zeros((n, h, w), dtype=np.float32)
        present = [(slot, r) for slot, r in enumerate(rows) if r is not None]
        if not present:
            return vals, mask
        slots = [s for s, _ in present]
        block = stack[[r for _, r in present], box[0], box[1]].astype(np.float32)
        observed = np.isfinite(block)
        # `np.where` rather than boolean scatter: one vectorised pass, and NaN never
        # reaches the output because the unobserved branch is a literal zero.
        vals[slots] = np.where(observed, (block - mean) / std, 0.0)
        mask[slots] = observed
        return vals, mask

    def _window_rows(self, s: LakeSeries, anchor: date) -> list[int | None]:
        """Row indices of the input window, oldest first."""
        return [
            s.index_of(anchor - timedelta(days=self.input_days - 1 - i))
            for i in range(self.input_days)
        ]

    def _target_rows(self, s: LakeSeries, anchor: date) -> list[int | None]:
        return [s.index_of(anchor + timedelta(days=lead)) for lead in self.lead_times]

    def _stack_window(self, s: LakeSeries, anchor: date) -> tuple[np.ndarray, np.ndarray]:
        """Past frames oldest-first over the full grid, plus their masks. Shapes (T, H, W)."""
        h, w = s.shape
        return self._gather(
            s.values,
            self._window_rows(s, anchor),
            (slice(0, h), slice(0, w)),
            self.norm.mean,
            self.norm.std,
        )

    def _crop_box(self, shape: tuple[int, int]) -> tuple[tuple[slice, slice], tuple[int, int]]:
        """Where to read from, and how much to pad, for this sample.

        Returns the source window and the (bottom, right) padding that brings it up to
        `crop` on each axis. Uniform random placement. Water-biased sampling was
        considered and measured to be unnecessary: every registered lake fills enough of
        its own bbox that a random 128 px crop averages 57-74% water, with no crop below
        5% and none entirely dry. Add biasing only if a future AOI is a small lake inside
        a large bounding box.
        """
        h, w = shape
        if self.crop is None:
            return (slice(0, h), slice(0, w)), (0, 0)
        c = self.crop
        top = int(self.rng.integers(0, h - c + 1)) if h > c else 0
        left = int(self.rng.integers(0, w - c + 1)) if w > c else 0
        box = (slice(top, top + min(c, h)), slice(left, left + min(c, w)))
        return box, (max(0, c - h), max(0, c - w))

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        spec = self.samples[idx]
        s = self.series[spec.lake_idx]
        box, (pad_h, pad_w) = self._crop_box(s.shape)

        x, x_mask = self._gather(
            s.values, self._window_rows(s, spec.anchor), box, self.norm.mean, self.norm.std
        )
        y, y_mask = self._gather(
            s.values, self._target_rows(s, spec.anchor), box, self.norm.mean, self.norm.std
        )
        water = s.water[box].astype(np.float32)[None]

        # Auxiliary layers take their own value and mask channels, because a day can carry
        # chlorophyll and no usable reflectance and the model has to be able to tell. A
        # layer the cube does not carry becomes an all-zero channel with an all-zero mask,
        # which is exactly what the model sees for a cloudy day -- so a lake ingested
        # without reflectance trains alongside one that has it, contributing nothing to
        # those channels rather than contributing a fabricated value.
        t = self.input_days
        na = len(self.aux_keys)
        hw = x.shape[1:]
        aux = np.zeros((na, t, *hw), dtype=np.float32)
        aux_mask = np.zeros((na, t, *hw), dtype=np.float32)
        window_rows = None
        for a, key in enumerate(self.aux_keys):
            layer = s.aux.get(key)
            if layer is None:
                continue
            window_rows = window_rows or self._window_rows(s, spec.anchor)
            mean, std = self.aux_stats.get(key, (0.0, 1.0))
            aux[a], aux_mask[a] = self._gather(layer, window_rows, box, mean, std)

        # Seasonal expectation for each target date, in the same standardised units as
        # the values. Excludes the anchor's own year: the window around a target date
        # otherwise overlaps the future the model is being asked to forecast.
        n_leads = len(self.lead_times)
        clim = np.zeros((n_leads, *hw), dtype=np.float32)
        clim_mask = np.zeros((n_leads, *hw), dtype=np.float32)
        withhold = self.augment and self.clim_dropout > 0 and self.rng.random() < self.clim_dropout
        if self.climatology and not withhold:
            for li, lead in enumerate(self.lead_times):
                target = spec.anchor + timedelta(days=lead)
                frame = s.climatology(
                    target.timetuple().tm_yday, spec.anchor.year, box, self.clim_halfwidth
                )
                if frame is None:
                    continue
                ok = np.isfinite(frame)
                clim[li] = np.where(ok, (frame - self.norm.mean) / self.norm.std, 0.0)
                clim_mask[li] = ok

        # Meteorological forcing over the window, and optionally at the targets. These
        # are per-day scalars, so they are not cropped, padded or flipped -- the weather
        # over a lake does not change when the sample is mirrored.
        n_met = len(self.met_keys)
        met = np.zeros((n_met, self.input_days), dtype=np.float32)
        met_mask = np.zeros((n_met, self.input_days), dtype=np.float32)
        met_fut = np.zeros((n_met, len(self.lead_times)), dtype=np.float32)
        met_fut_mask = np.zeros((n_met, len(self.lead_times)), dtype=np.float32)
        if n_met and s.met is not None:
            from habforecast.data.era5 import series_matrix

            window_days = [
                spec.anchor - timedelta(days=self.input_days - 1 - i)
                for i in range(self.input_days)
            ]
            met, met_mask = series_matrix(s.met, window_days, s.met_stats, self.met_keys)
            if self.met_future:
                target_days = [spec.anchor + timedelta(days=lead) for lead in self.lead_times]
                met_fut, met_fut_mask = series_matrix(
                    s.met, target_days, s.met_stats, self.met_keys
                )

        # Every array is cropped from the same box above, so the spatial transforms below
        # cannot leave a channel mis-registered against the chlorophyll it explains.
        if pad_h or pad_w:
            pad = ((0, 0), (0, pad_h), (0, pad_w))
            x, x_mask, y, y_mask, water, clim, clim_mask = (
                np.pad(a, pad) for a in (x, x_mask, y, y_mask, water, clim, clim_mask)
            )
            aux, aux_mask = (np.pad(a, ((0, 0), *pad)) for a in (aux, aux_mask))
        if self.augment:
            x, x_mask, y, y_mask, water, aux, aux_mask, clim, clim_mask = self._apply_flips(
                x, x_mask, y, y_mask, water, aux, aux_mask, clim, clim_mask
            )

        # Only score where the target was observed AND the pixel is water.
        y_mask = y_mask * water

        groups = np.asarray(s.groups, dtype=np.int64)
        if self.augment and self.group_dropout > 0:
            groups = np.where(self.rng.random(groups.shape) < self.group_dropout, 0, groups)

        return {
            "x": torch.from_numpy(np.ascontiguousarray(x)),
            "x_mask": torch.from_numpy(np.ascontiguousarray(x_mask)),
            "y": torch.from_numpy(np.ascontiguousarray(y)),
            "y_mask": torch.from_numpy(np.ascontiguousarray(y_mask)),
            "water": torch.from_numpy(np.ascontiguousarray(water)),
            "aux": torch.from_numpy(np.ascontiguousarray(aux)),
            "aux_mask": torch.from_numpy(np.ascontiguousarray(aux_mask)),
            "clim": torch.from_numpy(np.ascontiguousarray(clim)),
            "clim_mask": torch.from_numpy(np.ascontiguousarray(clim_mask * water)),
            "met": torch.from_numpy(met),
            "met_mask": torch.from_numpy(met_mask),
            "met_future": torch.from_numpy(met_fut),
            "met_future_mask": torch.from_numpy(met_fut_mask),
            "lake_idx": torch.tensor(spec.lake_idx, dtype=torch.long),
            "doy": torch.tensor(spec.anchor.timetuple().tm_yday, dtype=torch.float32),
            # Year is carried so the climatology baseline can exclude it and stay honest
            # (see ClimatologyBaseline: fitting on the evaluated year is leakage).
            "year": torch.tensor(spec.anchor.year, dtype=torch.long),
            "groups": torch.from_numpy(groups),
        }

    # -------------------------------------------------------------------- transforms --
    def _apply_flips(self, *arrays):
        """Random horizontal / vertical flips, applied identically to every array."""
        if self.rng.random() < 0.5:
            arrays = tuple(np.flip(a, axis=-1) for a in arrays)
        if self.rng.random() < 0.5:
            arrays = tuple(np.flip(a, axis=-2) for a in arrays)
        return arrays
