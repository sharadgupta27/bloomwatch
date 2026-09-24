"""Inference: assemble a recent observation window and produce a bloom forecast.

Two sources for the input window:

- the local Zarr cube, when the requested date is inside the ingested archive (fast,
  and what the API serves by default);
- a live STAC fetch, for dates after the archive ends.

The distinction is exposed in the response (`source`) rather than hidden, because a
forecast built from a live fetch has different latency and completeness characteristics
than one built from the curated archive, and an operator should be able to tell.

Predictions are returned in physical units -- log10 mg/m^3 and mg/m^3 -- along with the
observation mask, so a consumer can distinguish "we predict low chlorophyll here" from
"we have no idea here". Serving a gap-filled field without that mask would be the single
most misleading thing this system could do.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from habforecast.aoi import AOIRegistry, LakeAOI, LakeGrid
from habforecast.eval.metrics import BLOOM_THRESHOLDS_MG_M3
from habforecast.features.normalisation import Normalisation

# torch is imported where a checkpoint is actually run, never at module import: the ONNX
# serving image (inference.onnx_backend) imports this module and has no torch installed.
if TYPE_CHECKING:
    from types import SimpleNamespace

    import torch

logger = logging.getLogger(__name__)

# Time-block size for the streaming water-mask reduction. Matches the cubes' own 32-frame
# chunking so each read is one chunk deep.
WATER_MASK_BLOCK = 32
# A pixel counts as water if it produced a valid retrieval on at least this fraction of
# stored days. Must match features.dataset.load_lake_series, or the model would be served
# a different water mask at inference than it saw in training.
WATER_MIN_FREQUENCY = 0.05
# Half-width of the seasonal window the climatology input is averaged over. Must match
# the training dataset's `clim_halfwidth` for the same reason as the water mask.
CLIM_HALFWIDTH_DAYS = 15


@dataclass
class ForecastResult:
    """One lake's forecast at all configured lead times."""

    lake_id: str
    lake_name: str
    anchor_date: date
    lead_times: tuple[int, ...]
    log_chl: np.ndarray  # (L, H, W) predicted log10 mg/m^3
    water: np.ndarray  # (H, W) bool
    input_observed_days: list[date] = field(default_factory=list)
    source: str = "cube"
    model_version: str = "unknown"
    # (L, K, H, W) p(chlorophyll >= threshold k) from the exceedance head, averaged over
    # the ensemble and TTA; None for a checkpoint without one.
    exceed_prob: np.ndarray | None = None
    exceed_thresholds: tuple[float, ...] = ()
    # (L, H, W) ensemble standard deviation of log_chl; None for a single model.
    spread: np.ndarray | None = None
    ensemble_size: int = 1
    # (T, H, W) the observed input window in log10 mg/m^3, NaN where unobserved. Not part
    # of any response; kept so the daily job can check the inputs for drift without
    # fetching the window twice (pipelines/advisory.py).
    input_log_chl: np.ndarray | None = field(default=None, repr=False)

    @property
    def chl_mg_m3(self) -> np.ndarray:
        return np.power(10.0, self.log_chl)

    def bloom_area_km2(self, threshold_mg_m3: float, cell_area_km2: float) -> list[float]:
        """Predicted bloom extent per lead time -- the number an advisory acts on."""
        thr = np.log10(threshold_mg_m3)
        with np.errstate(invalid="ignore"):
            hit = (self.log_chl >= thr) & self.water[None]  # NaN compares False
        return [float(n * cell_area_km2) for n in hit.sum(axis=(1, 2))]

    def exceedance(self, threshold_mg_m3: float) -> np.ndarray | None:
        """(L, H, W) p(exceed) for one advisory threshold, or None without a head."""
        if self.exceed_prob is None or threshold_mg_m3 not in self.exceed_thresholds:
            return None
        return self.exceed_prob[:, self.exceed_thresholds.index(threshold_mg_m3)]

    def summary(self, cell_area_km2: float) -> dict:
        water_km2 = float(self.water.sum() * cell_area_km2)
        areas = {
            f"{thr:g}": self.bloom_area_km2(thr, cell_area_km2) for thr in BLOOM_THRESHOLDS_MG_M3
        }
        per_lead = []
        for i, lead in enumerate(self.lead_times):
            frame = self.log_chl[i][self.water]
            frame = frame[np.isfinite(frame)]
            entry = {
                "lead_days": lead,
                "valid_date": (self.anchor_date + timedelta(days=lead)).isoformat(),
                "median_chl_mg_m3": float(10 ** np.median(frame)) if frame.size else None,
                "p90_chl_mg_m3": float(10 ** np.percentile(frame, 90)) if frame.size else None,
                "bloom_area_km2": {key: vals[i] for key, vals in areas.items()},
            }
            if self.exceed_prob is not None:
                # Expected area above each threshold (sum of p over water, a calibrated
                # quantity when p is), and the lake-wide peak probability.
                entry["exceed_probability"] = {
                    f"{thr:g}": {
                        "expected_area_km2": float(
                            np.nansum(self.exceed_prob[i, k][self.water]) * cell_area_km2
                        ),
                        "max_prob": float(np.nanmax(self.exceed_prob[i, k][self.water]))
                        if self.water.any()
                        else 0.0,
                    }
                    for k, thr in enumerate(self.exceed_thresholds)
                }
            if self.spread is not None:
                sp = self.spread[i][self.water]
                sp = sp[np.isfinite(sp)]
                entry["ensemble_spread_log10_median"] = float(np.median(sp)) if sp.size else None
            per_lead.append(entry)
        return {
            "lake_id": self.lake_id,
            "lake_name": self.lake_name,
            "anchor_date": self.anchor_date.isoformat(),
            "source": self.source,
            "model_version": self.model_version,
            "water_area_km2": water_km2,
            "input_observed_days": [d.isoformat() for d in self.input_observed_days],
            "ensemble_size": self.ensemble_size,
            "forecasts": per_lead,
        }


class Predictor:
    """Loads a trained checkpoint and serves forecasts for registered lakes.

    `checkpoint_path` may be a single `.ckpt` or a directory of them. A directory is an
    **ensemble**: every member is run and the forecast is their mean, the exceedance
    probability their mean probability, and the per-pixel spread is returned beside it.
    Members must agree on everything that defines the inputs (normalisation, leads, window,
    channels) -- they are seeds of one configuration, not a model zoo -- and `load()`
    refuses a directory where they do not.
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        aoi_path: str | Path,
        cube_root: str | Path,
        device: str | None = None,
        tta: bool = False,
        met_root: str | Path | None = None,
    ):
        self.checkpoint_path = Path(checkpoint_path)
        # Average the forecast over the four flips the model was trained under. Costs
        # four forward passes; a full lake still forecasts in well under a second.
        self.tta = bool(tta)
        self.registry = AOIRegistry.from_geojson(aoi_path)
        self.cube_root = Path(cube_root)
        self.device = self._resolve_device(device)
        self.module: SimpleNamespace | None = None
        self.norm: Normalisation  # assigned by load()
        self.input_days: int = 14
        # Auxiliary layers this checkpoint was trained with, filled in from model_config.
        # Empty for every checkpoint trained before they existed, which is what makes the
        # served model and the new ones loadable by the same code path.
        self.aux_keys: tuple[str, ...] = ()
        # Whether the checkpoint reads the seasonal climatology (models.anomaly, change 5).
        # Filled in from model_config; every checkpoint trained before it is False.
        self.use_clim: bool = False
        # ERA5 forcing the checkpoint reads, and the train-lake standardisation it was
        # fitted with (recorded in model_config). Read from `met_root/<lake>.zarr`; where
        # the store does not cover the window the channels are masked off, exactly as for
        # a lake with no store in training.
        self.met_keys: tuple[str, ...] = ()
        self.met_stats: dict[str, tuple[float, float]] = {}
        self.met_root = Path(met_root) if met_root else Path(cube_root).parent / "met"
        self._met_cache: dict[str, object] = {}
        self.members: list = []
        # Exceedance-probability recalibration (habforecast.calibration), if one was
        # fitted for this checkpoint; applied to every probability this predictor returns.
        self.calibration: dict | None = None
        self.lead_times: tuple[int, ...] = (1, 3, 5, 7)
        self.model_version = self.checkpoint_path.stem
        self._hparams: dict = {}
        # Static per-lake water masks; a time-axis reduction that must not run per request.
        self._water_cache: dict[str, np.ndarray] = {}
        # Open cube handles and their calendar index, keyed by lake and tagged with the
        # store's metadata signature so an ingest that appends to a cube while the server
        # runs is picked up on the next request rather than served stale.
        self._cube_cache: dict[str, tuple[tuple, object, dict[date, int]]] = {}

    @staticmethod
    def _resolve_device(device: str | None):
        import torch

        return torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    def load(self) -> Predictor:
        """Load the Lightning checkpoint and recover its normalisation constants.

        Only the network is rebuilt. The checkpoint was written by a LightningModule, but
        serving needs none of Lightning -- its state dict is the network's under a
        `model.` prefix plus a loss-weighting buffer -- and importing it measured 7 s of
        cold start and a few hundred megabytes in the API container. `lightning_module`
        builds the full training wrapper on demand for callers that want it.
        """
        from types import SimpleNamespace

        if self.checkpoint_path.is_dir():
            paths = sorted(self.checkpoint_path.glob("*.ckpt"))
            if not paths:
                raise FileNotFoundError(f"no .ckpt files in ensemble dir {self.checkpoint_path}")
        else:
            paths = [self.checkpoint_path]

        members = []
        signature = None
        for path in paths:
            net, sig = self._load_member(path)
            if signature is None:
                signature = sig
            elif sig != signature:
                raise ValueError(
                    f"ensemble member {path.name} disagrees with {paths[0].name} on its "
                    f"inputs: {sig} vs {signature}"
                )
            members.append(net)
        self.members = members
        from habforecast import calibration

        self.calibration = calibration.load(self.checkpoint_path)
        # The same shape `BaselinePredictor` exposes, so every consumer reads `.module.model`
        # without knowing which of the two it holds. For an ensemble it is the first
        # member; `run` is what averages them.
        self.module = SimpleNamespace(model=members[0])
        if len(members) > 1:
            self.model_version = f"{self.checkpoint_path.name}-x{len(members)}"
        logger.info(
            "loaded %s: %d member(s), input_days=%d, leads=%s, clim=%s, met=%s, exceed=%s, "
            "tta=%s on %s",
            self.checkpoint_path.name,
            len(members),
            self.input_days,
            self.lead_times,
            self.use_clim,
            self.met_keys,
            self.exceed_thresholds,
            self.tta,
            self.device,
        )
        return self

    @property
    def exceed_thresholds(self) -> tuple[float, ...]:
        if not self.members:
            return ()
        return tuple(getattr(self.members[0], "exceed_thresholds", ()) or ())

    def _load_member(self, path: Path):
        """Rebuild one checkpoint's network; returns it and its input signature."""
        import torch

        from habforecast.models.convlstm import build_model

        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        hp = self._hparams = dict(ckpt.get("hyper_parameters", {}))
        self.lead_times = tuple(hp.get("lead_times", self.lead_times))
        self.norm = Normalisation(
            mean=float(hp.get("norm_mean", 0.0)), std=float(hp.get("norm_std", 1.0))
        )

        state = ckpt["state_dict"]
        config = hp.get("model_config")
        if config:
            # Preferred path: the checkpoint records exactly how it was built.
            config = dict(config)
            name = config.pop("name")
            kwargs = config
            self.input_days = kwargs.pop("input_days", self.input_days)
            # Which auxiliary layers the model expects is provenance, not a constructor
            # argument: the network only needs to know how many channels to allocate.
            # Recording the names is what lets a serving path fetch the right ones, and
            # what makes a checkpoint self-describing rather than a channel count.
            self.aux_keys = tuple(kwargs.pop("aux_keys", ()) or ())
            self.use_clim = bool(kwargs.get("use_clim", False))
            # Same for the forcing: names and standardisation are provenance.
            self.met_keys = tuple(kwargs.pop("met_keys", ()) or ())
            self.met_stats = {
                k: (float(v[0]), float(v[1]))
                for k, v in (kwargs.pop("met_stats", {}) or {}).items()
            }
        else:
            # Legacy checkpoints predate model_config. Shape inference is ambiguous for
            # U-Nets (use_doy=True/T and use_doy=False/T+1 share an input-channel count),
            # so warn rather than pretend the result is trustworthy.
            logger.warning(
                "%s has no model_config; falling back to shape inference, which cannot "
                "distinguish a use_doy=False U-Net from a use_doy=True one. Retrain to "
                "embed the config if the prediction looks wrong.",
                path.name,
            )
            name, kwargs = _infer_architecture(state, len(self.lead_times))
            self.input_days = kwargs.pop("input_days", self.input_days)
        net = build_model(name, input_days=self.input_days, n_leads=len(self.lead_times), **kwargs)
        prefix = "model."
        net.load_state_dict({k[len(prefix) :]: v for k, v in state.items() if k.startswith(prefix)})
        net.eval().to(self.device)
        signature = (
            round(self.norm.mean, 6),
            round(self.norm.std, 6),
            self.input_days,
            self.lead_times,
            self.aux_keys,
            self.use_clim,
            self.met_keys,
            tuple(getattr(net, "exceed_thresholds", ()) or ()),
        )
        return net, signature

    def run(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        water: torch.Tensor,
        doy: torch.Tensor,
        clim: torch.Tensor | None = None,
        clim_mask: torch.Tensor | None = None,
        met: torch.Tensor | None = None,
        met_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """One forward pass through the loaded network(s), with whatever they can read.

        The single call site for serving, so the ad-hoc path and the registry path hand
        the model the same arguments, and TTA is applied in one place or not at all.
        Returns the (ensemble-mean) regression forecast; `run_full` adds the rest.
        """
        return self.run_full(x, x_mask, water, doy, clim, clim_mask, met, met_mask)[0]

    def run_full(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        water: torch.Tensor,
        doy: torch.Tensor,
        clim: torch.Tensor | None = None,
        clim_mask: torch.Tensor | None = None,
        met: torch.Tensor | None = None,
        met_mask: torch.Tensor | None = None,
        groups: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """(forecast, p(exceed) or None, ensemble spread or None), all ensemble-averaged.

        Shapes: (B, L, H, W), (B, L, K, H, W), (B, L, H, W). The spread is the standard
        deviation of the members' regression forecasts in the model's standardised units.
        """
        batch: dict[str, torch.Tensor | None] = {
            "x": x,
            "x_mask": x_mask,
            "water": water,
            "doy": doy,
        }
        if clim is not None:
            batch["clim"], batch["clim_mask"] = clim, clim_mask
        if met is not None:
            batch["met"], batch["met_mask"] = met, met_mask
        if groups is not None:
            # (B, F) lake-group indices for a multilevel checkpoint; absent, it serves
            # the population level (see habforecast.groups).
            batch["groups"] = groups
        return ensemble_forecast(self.members, batch, self.tta)

    def forward_numpy(
        self,
        x: np.ndarray,
        x_mask: np.ndarray,
        water: np.ndarray,
        doy: np.ndarray,
        clim: np.ndarray | None = None,
        clim_mask: np.ndarray | None = None,
        met: np.ndarray | None = None,
        met_mask: np.ndarray | None = None,
        groups=None,
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
        """`run_full` on numpy arrays, batch axis included -- the backend-neutral call.

        `predict` and the ad-hoc path go through this, so the ONNX backend has one method
        to implement and serves exactly what this one does.
        """
        import torch

        def t(a):
            return None if a is None else torch.from_numpy(np.ascontiguousarray(a)).to(self.device)

        ids = None if groups is None else np.asarray([groups.ids()], dtype=np.int64)
        with torch.no_grad():
            pred, prob, spread = self.run_full(
                t(x),
                t(x_mask),
                t(water),
                t(doy),
                t(clim),
                t(clim_mask),
                t(met),
                t(met_mask),
                t(ids),
            )
        return (
            pred.float().cpu().numpy(),
            self.calibrate(None if prob is None else prob.float().cpu().numpy(), groups),
            None if spread is None else spread.float().cpu().numpy(),
        )

    def calibrate(self, prob: np.ndarray | None, groups=None) -> np.ndarray | None:
        """Apply this checkpoint's recalibration to a (B, L, K, H, W) probability.

        `groups` (a `habforecast.groups.LakeGroups`) selects a per-trophic-class map where
        the calibration has one; without it, or for a class it has none for, the global
        map applies.
        """
        from habforecast import calibration

        return calibration.apply(
            prob, self.calibration, self.lead_times, self.exceed_thresholds, groups=groups
        )

    def lightning_module(self):
        """The training wrapper around the loaded network, built on first use.

        For callers that want `predict_step`, the loss, or a `Trainer`; nothing in the
        serving path does, which is why `load()` does not pay for it.
        """
        if self.module is None:
            raise RuntimeError("call load() before lightning_module()")
        from habforecast.models.lit import HABForecastModule

        keep = {
            "lr",
            "weight_decay",
            "huber_delta",
            "lead_weighting",
            "warmup_epochs",
            "max_epochs",
            "norm_mean",
            "norm_std",
        }
        module = HABForecastModule(
            model=self.module.model,
            lead_times=self.lead_times,
            **{k: v for k, v in self._hparams.items() if k in keep},
        )
        return module.eval().to(self.device)

    # ------------------------------------------------------------------ windowing --
    @staticmethod
    def _cube_signature(path: Path) -> tuple | None:
        """Cheap fingerprint of a cube: the consolidated metadata's size and mtime.

        Every append rewrites that file, so it changes exactly when the cube does. One
        `stat` per request, against ~30 ms to reopen the store through xarray.
        """
        for name in ("zarr.json", ".zmetadata"):
            meta = path / name
            if meta.exists():
                st = meta.stat()
                return (name, st.st_size, st.st_mtime_ns)
        return None

    def _open_cube(self, lake_id: str):
        """The lake's dataset and its calendar index, reopened only when the cube changed."""
        import pandas as pd
        import xarray as xr

        path = self.cube_root / f"{lake_id}.zarr"
        sig = self._cube_signature(path)
        if sig is None:
            return None, None
        hit = self._cube_cache.get(lake_id)
        if hit is not None and hit[0] == sig:
            return hit[1], hit[2]
        ds = xr.open_zarr(path, consolidated=True)
        days = {pd.Timestamp(t).date(): i for i, t in enumerate(ds.time.values)}
        self._cube_cache[lake_id] = (sig, ds, days)
        # The water mask is a reduction over the time axis, so it is stale as soon as
        # the cube has grown.
        self._water_cache.pop(lake_id, None)
        return ds, days

    def _window_from_cube(
        self, aoi: LakeAOI, grid: LakeGrid, anchor: date
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[date]] | None:
        ds, days = self._open_cube(aoi.lake_id)
        if ds is None:
            return None
        wanted = [anchor - timedelta(days=self.input_days - 1 - k) for k in range(self.input_days)]
        idx = [days.get(d) for d in wanted]
        if all(i is None for i in idx):
            return None

        h, w = grid.shape
        vals = np.zeros((self.input_days, h, w), dtype=np.float32)
        mask = np.zeros((self.input_days, h, w), dtype=np.float32)
        observed: list[date] = []

        # Read ONLY the requested days. `ds["log_chl"].values` would decode the entire
        # cube -- 568 MB for Lake Victoria -- on every single forecast request, for the
        # sake of a 14-day window.
        present = [
            (slot, i, d)
            for slot, (d, i) in enumerate(zip(wanted, idx, strict=True))
            if i is not None
        ]
        if present:
            block = ds["log_chl"].isel(time=[i for _, i, _ in present]).values.astype(np.float32)
            ok = np.isfinite(block)
            std = np.where(ok, self.norm.apply(block), 0.0)
            for pos, (slot, _, d) in enumerate(present):
                if not ok[pos].any():
                    continue
                vals[slot] = std[pos]
                mask[slot] = ok[pos]
                observed.append(d)

        water = self._water_mask(aoi.lake_id, ds)
        return vals, mask, water.astype(np.float32), observed

    def _climatology(self, lake_id: str, anchor: date) -> tuple[np.ndarray, np.ndarray] | None:
        """Per-lead seasonal expectation from the lake's archive, standardised.

        The same quantity the `use_clim` model trained on: the per-pixel mean of every
        archived day within +/-15 days of the target's day of year, from every year but
        the anchor's. Excluding the current year keeps serving identical to training --
        and, for a forecast issued mid-season, keeps the days after the anchor out of
        it, which for an archive that lags real time is not hypothetical.

        Returns (clim, mask) shaped (L, H, W), or None when the lake has no cube.
        """
        import pandas as pd

        ds, _days = self._open_cube(lake_id)
        if ds is None:
            return None
        ts = pd.to_datetime(ds.time.values)
        doys = np.asarray(ts.dayofyear, dtype=np.int32)
        years = np.asarray(ts.year, dtype=np.int32)
        h, w = ds.sizes["y"], ds.sizes["x"]
        n = len(self.lead_times)
        clim = np.zeros((n, h, w), dtype=np.float32)
        mask = np.zeros((n, h, w), dtype=np.float32)
        for li, lead in enumerate(self.lead_times):
            target = (anchor + timedelta(days=lead)).timetuple().tm_yday
            delta = np.abs(doys - target)
            delta = np.minimum(delta, 366 - delta)
            rows = np.flatnonzero((delta <= CLIM_HALFWIDTH_DAYS) & (years != anchor.year))
            if rows.size == 0:
                continue
            block = ds["log_chl"].isel(time=rows).values.astype(np.float32)
            ok = np.isfinite(block)
            cnt = ok.sum(axis=0)
            with np.errstate(invalid="ignore", divide="ignore"):
                mean = np.where(ok, block, 0.0).sum(axis=0) / np.maximum(cnt, 1)
            has = cnt > 0
            clim[li] = np.where(has, self.norm.apply(mean), 0.0)
            mask[li] = has
        return clim, mask

    def _water_mask(self, lake_id: str, ds) -> np.ndarray:
        """Static water mask for a lake, computed once and cached.

        The mask is a reduction over the whole time axis, so it is the expensive part of a
        request. It is also constant for a given cube, so it is computed lazily through
        dask (streaming chunk by chunk rather than materialising the cube) and cached for
        the lifetime of the process.
        """
        cached = self._water_cache.get(lake_id)
        if cached is not None:
            return cached

        da = ds["log_chl"]
        n_time = da.sizes["time"]
        shape = (da.sizes["y"], da.sizes["x"])
        counts = np.zeros(shape, dtype=np.int32)

        # Accumulate over explicit time blocks rather than letting dask schedule the
        # reduction. `notnull().mean(dim="time")` is lazy, but the threaded scheduler keeps
        # many chunks resident at once -- measured at a 500 MB peak for Lake Victoria's 34
        # chunks. Looping gives deterministic O(one block) memory, which is what a serving
        # container needs.
        for start in range(0, n_time, WATER_MASK_BLOCK):
            block = da.isel(time=slice(start, start + WATER_MASK_BLOCK)).values
            counts += np.isfinite(block).sum(axis=0).astype(np.int32)

        water = ((counts / max(n_time, 1)) >= WATER_MIN_FREQUENCY).astype(np.float32)
        self._water_cache[lake_id] = water
        logger.debug(
            "%s: water mask %d/%d px from %d frames", lake_id, int(water.sum()), water.size, n_time
        )
        return water

    def _window_live(
        self, aoi: LakeAOI, grid: LakeGrid, anchor: date
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[date]]:
        """Fetch the window directly for dates beyond the cube: from the EUMETSAT Data
        Store when credentials are set (hours behind real time), else Planetary Computer."""
        from habforecast.data.olci import default_reader, read_many
        from habforecast.data.resample import SwathResampler
        from habforecast.data.store import composite_day, group_by_day

        reader = default_reader()
        resampler = SwathResampler(grid)
        start = anchor - timedelta(days=self.input_days - 1)
        items = reader.search(grid, start.isoformat(), anchor.isoformat())
        observations = read_many(reader, items, grid, resampler, min_valid_fraction=0.01)

        h, w = grid.shape
        vals = np.zeros((self.input_days, h, w), dtype=np.float32)
        mask = np.zeros((self.input_days, h, w), dtype=np.float32)
        observed: list[date] = []
        frames = {d: composite_day(d, o) for d, o in group_by_day(observations).items()}
        for slot in range(self.input_days):
            d = start + timedelta(days=slot)
            frame = frames.get(d)
            if frame is None:
                continue
            ok = np.isfinite(frame.log_chl)
            vals[slot][ok] = self.norm.apply(frame.log_chl[ok])
            mask[slot] = ok.astype(np.float32)
            observed.append(d)

        # The water mask must mean the same thing at inference as it did in training:
        # "pixel produced a valid retrieval on >=5% of the lake's stored days". Deriving it
        # from this 14-day window instead (any pixel observed at least once) would hand the
        # model a mask that shrinks with cloud cover -- a silent input distribution shift,
        # and one that would blank out genuinely blooming water simply because it was
        # cloudy all fortnight. Prefer the cube's static mask; fall back to the window only
        # when no cube exists for this lake at all.
        water = None
        ds, _days = self._open_cube(aoi.lake_id)
        if ds is not None:
            water = self._water_mask(aoi.lake_id, ds)
        if water is None:
            logger.warning(
                "%s: no cube available, deriving the water mask from the live window only; "
                "coverage will be understated where the fortnight was cloudy",
                aoi.lake_id,
            )
            water = (mask.sum(axis=0) > 0).astype(np.float32)
        return vals, mask, water, observed

    # ------------------------------------------------------------------ prediction --
    def predict(
        self, lake_id: str, anchor: date | None = None, allow_live: bool = True
    ) -> ForecastResult:
        if self.module is None:
            raise RuntimeError("call load() before predict()")
        aoi = self.registry[lake_id]
        grid = aoi.grid()
        anchor = anchor or datetime.now(UTC).date()

        window = self._window_from_cube(aoi, grid, anchor)
        source = "cube"
        if window is None:
            if not allow_live:
                raise ValueError(
                    f"no archived observations for {lake_id} in the {self.input_days} days "
                    f"before {anchor}, and live fetching is disabled"
                )
            logger.info("%s: cube miss for %s -- fetching live", lake_id, anchor)
            window = self._window_live(aoi, grid, anchor)
            source = "live"

        vals, mask, water, observed = window
        if not observed:
            raise ValueError(
                f"no usable observations for {lake_id} in the {self.input_days} days before "
                f"{anchor} -- persistent cloud cover"
            )

        doy = np.array([anchor.timetuple().tm_yday], dtype=np.float32)
        clim = clim_mask = None
        if self.use_clim:
            found = self._climatology(lake_id, anchor)
            if found is not None:
                clim, clim_mask = found[0][None], found[1][None]
        met = met_mask = None
        if self.met_keys:
            found_met = self._met_window(lake_id, anchor)
            if found_met is not None:
                met, met_mask = found_met[0][None], found_met[1][None]

        pred_b, prob_b, spread_b = self.forward_numpy(
            vals[None],
            mask[None],
            water[None, None],
            doy,
            clim,
            clim_mask,
            met,
            met_mask,
            groups=aoi.groups,
        )
        log_chl = self.norm.invert(pred_b[0])
        water_bool = water.astype(bool)
        log_chl = np.where(water_bool[None], log_chl, np.nan)
        prob = spread = None
        if prob_b is not None:
            prob = np.where(water_bool[None, None], prob_b[0], np.nan)
        if spread_b is not None:
            # Standardised units -> log10: a spread scales by the std, never shifts.
            spread = np.where(water_bool[None], spread_b[0] * self.norm.std, np.nan)

        return ForecastResult(
            lake_id=lake_id,
            lake_name=aoi.name,
            anchor_date=anchor,
            lead_times=self.lead_times,
            log_chl=log_chl.astype(np.float32),
            water=water_bool,
            input_observed_days=observed,
            source=source,
            model_version=self.model_version,
            exceed_prob=None if prob is None else prob.astype(np.float32),
            exceed_thresholds=self.exceed_thresholds if prob is not None else (),
            spread=None if spread is None else spread.astype(np.float32),
            ensemble_size=len(self.members),
            input_log_chl=np.where(mask > 0, self.norm.invert(vals), np.nan).astype(np.float32),
        )

    def _met_window(self, lake_id: str, anchor: date) -> tuple[np.ndarray, np.ndarray] | None:
        """The standardised forcing over the input window, from the lake's ERA5 store.

        None without a store. ERA5 lags real time by ~5 days, so a forecast issued today
        finds the most recent days missing; they arrive masked, which the encoder's
        coverage channel expresses, but the model was trained with complete windows and
        the effect of that gap is not measured -- see docs/MODEL_CARD.md.
        """
        from habforecast.data import era5

        if lake_id not in self._met_cache:
            store = self.met_root / f"{lake_id}.zarr"
            self._met_cache[lake_id] = era5.load_daily_means(store) if store.exists() else None
        frame = self._met_cache[lake_id]
        if frame is None:
            return None
        days = [anchor - timedelta(days=self.input_days - 1 - i) for i in range(self.input_days)]
        return era5.series_matrix(frame, days, self.met_stats, self.met_keys)


def ensemble_forecast(
    members: list, batch: dict, tta: bool
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Mean forecast, mean p(exceed) and spread over `members`, each with optional TTA.

    Shared by serving and `pipelines/evaluate.py`, so the number the evaluation reports
    for an ensemble is the number the API serves. Probabilities are averaged as
    probabilities; members without an exceedance head contribute no probability, and a
    mixed ensemble returns None rather than a mean over a subset.
    """
    import torch

    from habforecast.models.lit import has_exceed_head, model_inputs, predict_tta

    preds, probs = [], []
    with torch.no_grad():
        for net in members:
            inputs = model_inputs(net, batch)
            if has_exceed_head(net):
                pred, prob = predict_tta(net, inputs, tta, with_exceed=True)
                probs.append(prob)
            else:
                pred = predict_tta(net, inputs, tta)
            preds.append(pred.float())
    stack = torch.stack(preds)
    mean = stack.mean(dim=0)
    spread = stack.std(dim=0, unbiased=True) if len(preds) > 1 else None
    prob = torch.stack(probs).mean(dim=0) if probs and len(probs) == len(preds) else None
    return mean, prob, spread


def _infer_architecture(state_dict: dict, n_leads: int) -> tuple[str, dict]:
    """Recover model type and shape hyperparameters from a checkpoint's state_dict."""
    keys = list(state_dict)
    if any("cells.0.conv.weight" in k for k in keys):
        w = state_dict[next(k for k in keys if k.endswith("cells.0.conv.weight"))]
        hidden = w.shape[0] // 4
        layers = len({k.split("cells.")[1].split(".")[0] for k in keys if "cells." in k})
        stem = state_dict[next(k for k in keys if k.endswith("stem.0.weight"))]
        use_doy = stem.shape[1] == 5
        return "convlstm", {
            "hidden": hidden,
            "layers": layers,
            "use_doy": use_doy,
            "kernel_size": w.shape[-1],
        }
    if any("enc1.0.weight" in k for k in keys):
        w = state_dict[next(k for k in keys if k.endswith("enc1.0.weight"))]
        base, in_ch = w.shape[0], w.shape[1]
        # in_ch = 2*input_days + 1 (+2 if doy)
        for use_doy in (True, False):
            extra = 1 + (2 if use_doy else 0)
            if (in_ch - extra) % 2 == 0:
                return "unet", {
                    "base": base,
                    "use_doy": use_doy,
                    "input_days": (in_ch - extra) // 2,
                }
    raise ValueError("could not infer architecture from checkpoint state_dict")


class BaselinePredictor:
    """A non-learned forecaster, for areas where no checkpoint has been validated.

    Open ocean is the case this exists for. The only products that cover it here are the
    global 4 km grids, and this project has already *measured* what a 4 km model is worth
    on this problem: negative skill against persistence at three of four leads, in a
    controlled comparison against the same sensor at 750 m. Serving that checkpoint anyway
    would present a measured non-result as a forecast.

    So the honest output is persistence -- carry the most recent observed field forward --
    labelled as persistence. It is a weak forecast, but its error is characterised, which
    is more than the alternative offers. `habforecast.routing` decides when this is what
    runs, and the app says so on the result.

    It exposes the attributes `inference.adhoc` reads from a `Predictor`, so the fetch and
    assembly path does not need to know which of the two it is holding.
    """

    def __init__(
        self,
        kind: str = "persistence",
        lead_times: tuple[int, ...] = (1, 3, 5, 7),
        input_days: int = 14,
        device: str | None = None,
    ):
        from types import SimpleNamespace

        if kind != "persistence":
            raise ValueError(f"unknown baseline {kind!r}")
        self.kind = kind
        self.lead_times = tuple(lead_times)
        self.input_days = int(input_days)
        self.aux_keys: tuple[str, ...] = ()
        self.model_version = f"baseline-{kind}"
        self.device = device or "cpu"
        # Identity standardisation. Persistence copies its input to its output, so any
        # other choice would only introduce a round-trip through constants that cancel --
        # and this way the values the model returns are the physical ones throughout.
        self.norm = Normalisation(mean=0.0, std=1.0)
        self.module = SimpleNamespace(model=None)
        self.members: list = []

    def load(self) -> BaselinePredictor:
        """Present for interface parity with `Predictor`; there is nothing to load."""
        return self

    def forward_numpy(self, x, x_mask, water, doy, *_ignored):
        """Persistence in numpy: the newest observed value per pixel, at every lead.

        The same quantity as `models.baselines.PersistenceBaseline`, without torch, so a
        labelled-baseline route runs in the ONNX serving image too.
        """
        t = x.shape[1]
        steps = np.arange(1, t + 1, dtype=np.float32).reshape(1, t, 1, 1)
        weighted = x_mask * steps
        idx = weighted.argmax(axis=1)  # (B, H, W)
        has = (weighted.max(axis=1) > 0).astype(np.float32)
        last = np.take_along_axis(x, idx[:, None], axis=1)[:, 0] * has
        pred = np.repeat((last * water[:, 0])[:, None], len(self.lead_times), axis=1)
        return pred.astype(np.float32), None, None


def load_predictor(
    checkpoint_path: str | Path,
    aoi_path: str | Path,
    cube_root: str | Path,
    device: str | None = None,
    tta: bool = False,
    met_root: str | Path | None = None,
):
    """The right backend for `checkpoint_path`, loaded.

    A directory holding `manifest.json` is an ONNX export (`scripts/export_onnx.py`) and is
    served by onnxruntime with no torch at all; anything else -- a `.ckpt`, or a directory
    of them as an ensemble -- by `Predictor`.
    """
    path = Path(checkpoint_path)
    if path.is_dir() and (path / "manifest.json").exists():
        from habforecast.inference.onnx_backend import OnnxPredictor

        return OnnxPredictor(path, aoi_path, cube_root, tta=tta, met_root=met_root).load()
    return Predictor(path, aoi_path, cube_root, device=device, tta=tta, met_root=met_root).load()
