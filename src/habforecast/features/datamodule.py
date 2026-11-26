"""Lightning DataModule wiring lake cubes to train/val/test loaders.

The split is **by lake**, defined in the AOI registry, and this is the most consequential
design decision in the project.

A random split over samples would put 2019-07-14 Lake Erie in train and 2019-07-15 Lake
Erie in validation. Those two frames share most of their pixels and nearly all of their
bloom structure, so the validation score measures memorisation, not generalisation, and
would come out far too high. The same applies to random *spatial* tiling: adjacent
300 m pixels in a lake are strongly autocorrelated.

Holding out whole lakes -- Balaton for validation, Lake Victoria for test -- answers the
question an operational user actually asks: *will this work on my lake, which you have
never seen?* Victoria is deliberately the hardest possible case: tropical, no seasonal
cycle, a different continent, and a different bloom-forming community.

Normalisation is fitted on training lakes only, then applied unchanged to val and test.

A second split exists because the first answers only one question. `split_mode="temporal"`
keeps every lake on both sides and cuts by *year*: train on seasons up to a cut-off,
validate on the next year, test on the years after. That is the operational question for
a lake already in the registry -- "will this work on Erie next summer?" -- and it is the
split the catalogue this project follows asks for. It is strictly forward in time: a
held-out year's input windows never reach back into training years, its climatology is
drawn from the training years only, and the normalisation is fitted before the cut. The
two numbers are reported side by side; the gap between them is the cost of a new lake.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, cast

import lightning as L
import numpy as np
import torch
from lightning.fabric.utilities.data import AttributeDict
from torch.utils.data import DataLoader, get_worker_info

from habforecast.aoi import AOIRegistry
from habforecast.features import dataset as dataset_mod
from habforecast.features.dataset import (
    LakeSeries,
    Normalisation,
    SequenceDataset,
    fit_aux_normalisation,
    fit_normalisation,
    load_lake_series,
)

logger = logging.getLogger(__name__)


def _init_worker(worker_id: int) -> None:
    """Per-worker setup for DataLoader processes.

    Each worker receives a copy of the dataset, random generator included, so without a
    reseed every worker would draw the same crops, flips and dropouts. torch hands each
    worker a distinct seed derived from the run's seed, so this stays reproducible.
    The workers also split the climatology cache budget: each holds its own cache.
    """
    info = get_worker_info()
    info.dataset.rng = np.random.default_rng(torch.initial_seed() % 2**32)
    dataset_mod.set_clim_cache_budget(dataset_mod.CLIM_CACHE_MAX_BYTES // info.num_workers)


class HABDataModule(L.LightningDataModule):
    """Loads per-lake cubes, fits normalisation on train, builds split datasets."""

    def __init__(
        self,
        aoi_path: str | Path,
        cube_root: str | Path,
        input_days: int = 14,
        lead_times: tuple[int, ...] = (1, 3, 5, 7),
        crop: int | None = 128,
        batch_size: int = 8,
        num_workers: int = 0,
        min_input_frames: int = 2,
        min_target_coverage: float = 0.05,
        min_water_frequency: float = 0.05,
        augment: bool = True,
        seed: int = 1337,
        norm: Normalisation | None = None,
        aux_keys: tuple[str, ...] = (),
        climatology: bool = False,
        clim_halfwidth: int = 15,
        clim_dropout: float = 0.15,
        met_root: str | Path | None = None,
        met_keys: tuple[str, ...] = (),
        met_future: bool = False,
        split_mode: str = "lake",
        val_years: tuple[int, ...] = (2023,),
        test_years: tuple[int, ...] = (2024, 2025),
        group_dropout: float = 0.0,
        mmap_dir: str | Path | None = None,
        val_num_workers: int | None = None,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.met_root = Path(met_root) if met_root else None
        self._met_stats: dict[str, tuple[float, float]] = {}
        if split_mode not in ("lake", "temporal"):
            raise ValueError(f"split_mode must be 'lake' or 'temporal', got {split_mode!r}")
        if split_mode == "temporal" and set(val_years) & set(test_years):
            raise ValueError("val_years and test_years overlap")
        self.aoi_path = Path(aoi_path)
        self.cube_root = Path(cube_root)
        self.registry = AOIRegistry.from_geojson(self.aoi_path)
        # A caller may supply the normalisation from a checkpoint. That is both lighter
        # (no need to load the training cubes just to recompute constants) and stricter:
        # evaluation then uses exactly the constants the model was trained with, rather
        # than ones refitted from whatever cubes happen to be on disk today.
        self.norm: Normalisation | None = norm
        self._norm_provided = norm is not None
        self._series: dict[str, list[LakeSeries]] = {}
        self._datasets: dict[str, SequenceDataset] = {}
        self._aux_stats: dict[str, tuple[float, float]] = {}

    @property
    def hparams(self) -> AttributeDict:
        """`save_hyperparameters` always stores an AttributeDict; Lightning's annotation
        is the wider MutableMapping, which hides every attribute read from type checkers."""
        return cast(AttributeDict, super().hparams)

    # ------------------------------------------------------------------ loading ----
    def _load_split(self, split: str) -> list[LakeSeries]:
        out = []
        for aoi in self.registry.split(split):
            path = self.cube_root / f"{aoi.lake_id}.zarr"
            if not path.exists():
                logger.warning("%s: no cube at %s -- skipping", aoi.lake_id, path)
                continue
            out.append(
                load_lake_series(
                    path,
                    aoi.lake_id,
                    split,
                    min_water_frequency=self.hparams.min_water_frequency,
                    aux=tuple(self.hparams.aux_keys or ()),
                    met_root=self.met_root if self.hparams.met_keys else None,
                    mmap_dir=self.hparams.mmap_dir,
                )
            )
            out[-1].groups = tuple(aoi.groups.ids())
        if not out:
            logger.warning("split '%s' has no loadable cubes", split)
        return out

    def _load_temporal(self) -> dict[str, list[LakeSeries]]:
        """Every lake, cut by year into train / val / test slices.

        Each held-out slice points its climatology at the same lake's training slice, so
        a 2024 forecast is compared against, and fed, a seasonal expectation built from
        history that existed when it would have been issued.
        """
        val_years = set(self.hparams.val_years)
        test_years = set(self.hparams.test_years)
        out: dict[str, list[LakeSeries]] = {"train": [], "val": [], "test": []}
        for aoi in self.registry:
            path = self.cube_root / f"{aoi.lake_id}.zarr"
            if not path.exists():
                logger.warning("%s: no cube at %s -- skipping", aoi.lake_id, path)
                continue
            full = load_lake_series(
                path,
                aoi.lake_id,
                "all",
                min_water_frequency=self.hparams.min_water_frequency,
                aux=tuple(self.hparams.aux_keys or ()),
                met_root=self.met_root if self.hparams.met_keys else None,
                mmap_dir=self.hparams.mmap_dir,
            )
            full.groups = tuple(aoi.groups.ids())
            years = {int(y) for y in full._years}
            train_years = years - val_years - test_years
            train = full.subset_years(sorted(train_years), "train")
            out["train"].append(train)
            for split, held in (("val", val_years), ("test", test_years)):
                held_years = sorted(years & held)
                if not held_years:
                    continue
                sl = full.subset_years(held_years, split)
                sl.clim_source = train
                out[split].append(sl)
            logger.info(
                "%s: temporal split train=%s val=%s test=%s frames",
                aoi.lake_id,
                train.n_days,
                sum(s.n_days for s in out["val"] if s.lake_id == aoi.lake_id),
                sum(s.n_days for s in out["test"] if s.lake_id == aoi.lake_id),
            )
        return out

    def setup(self, stage: str | None = None) -> None:
        if self._datasets:
            return

        if self.hparams.split_mode == "temporal":
            self._series = self._load_temporal()
        else:
            wanted = ("val", "test") if self._norm_provided else ("train", "val", "test")
            for split in ("train", "val", "test"):
                self._series[split] = self._load_split(split) if split in wanted else []

        if self._norm_provided:
            assert self.norm is not None
            logger.info("normalisation (from checkpoint): %s", self.norm.to_dict())
        else:
            if not self._series["train"]:
                raise RuntimeError(
                    f"no training cubes under {self.cube_root}. Run pipelines/ingest.py first."
                )
            # Fit on training lakes only -- see module docstring.
            self.norm = fit_normalisation(self._series["train"])
            logger.info("normalisation (train lakes only): %s", self.norm.to_dict())

        # Auxiliary statistics are fitted on the training lakes only, for the same reason
        # the chlorophyll normalisation is: a held-out lake standardised by its own
        # distribution has told the model something about itself.
        aux_keys = tuple(self.hparams.aux_keys or ())
        if aux_keys and not self._aux_stats:
            source = self._series["train"] or self._series["val"] or self._series["test"]
            self._aux_stats = fit_aux_normalisation(source, aux_keys)
            logger.info("auxiliary normalisation (train lakes only): %s", self._aux_stats)

        # Forcing statistics, like every other standardisation here, come from the
        # training lakes only: a held-out lake standardised by its own weather has told
        # the model something about itself.
        met_keys = tuple(self.hparams.met_keys or ())
        if met_keys and not self._met_stats:
            from habforecast.data import era5

            source = self._series["train"] or self._series["val"] or self._series["test"]
            frames = [sr.met for sr in source if sr.met is not None]
            self._met_stats = era5.fit_normalisation(frames)
            logger.info("met normalisation (train lakes only): %s", self._met_stats)
        for split_series in self._series.values():
            for sr in split_series:
                sr.met_stats = self._met_stats

        common = {
            "met_keys": met_keys,
            "met_future": bool(self.hparams.met_future),
            "aux_keys": aux_keys,
            "aux_stats": self._aux_stats,
            "norm": self.norm,
            "input_days": self.hparams.input_days,
            "lead_times": tuple(self.hparams.lead_times),
            "min_input_frames": self.hparams.min_input_frames,
            "min_target_coverage": self.hparams.min_target_coverage,
            "seed": self.hparams.seed,
            "climatology": bool(self.hparams.climatology),
            "clim_halfwidth": int(self.hparams.clim_halfwidth),
        }
        if self._series["train"]:
            self._datasets["train"] = SequenceDataset(
                self._series["train"],
                crop=self.hparams.crop,
                augment=self.hparams.augment,
                # Only the training set ever withholds the climatology; evaluation and
                # serving hand the model everything it can legitimately know.
                clim_dropout=float(self.hparams.clim_dropout),
                group_dropout=float(self.hparams.group_dropout),
                **common,
            )
        # Build the climatology arrays here, once, before any DataLoader worker exists:
        # otherwise each worker builds its own on first use -- for mapped series, all of
        # them writing the same files at once.
        if self.hparams.climatology:
            for split_series in self._series.values():
                for sr in split_series:
                    (sr.clim_source or sr).prepare_climatology()

        # Val and test use the full grid and no augmentation: the score must reflect
        # performance on the whole lake, not on a lucky crop.
        for split in ("val", "test"):
            if self._series[split]:
                self._datasets[split] = SequenceDataset(
                    self._series[split], crop=None, augment=False, **common
                )

    # ------------------------------------------------------------------ loaders ----
    def _loader(self, split: str, shuffle: bool) -> DataLoader | None:
        ds = self._datasets.get(split)
        if ds is None or len(ds) == 0:
            return None
        # Full-grid val/test batches are large; batch 1 keeps peak memory predictable
        # on a 4 GB card and costs nothing since evaluation is not the bottleneck.
        bs = self.hparams.batch_size if split == "train" else 1
        # Val/test workers are kept alive all run but work a few minutes per epoch, and each
        # holds ~1.2 GB: six of each pushed the machine to 55.8 of 58.9 GB committed
        # (2026-09-25). So evaluation gets its own, smaller count (None = same as train).
        workers = self.hparams.num_workers
        if split != "train" and self.hparams.val_num_workers is not None:
            workers = self.hparams.val_num_workers
        return DataLoader(
            ds,
            batch_size=bs,
            shuffle=shuffle,
            num_workers=workers,
            pin_memory=True,
            drop_last=(split == "train" and len(ds) > self.hparams.batch_size),
            persistent_workers=workers > 0,
            worker_init_fn=_init_worker if workers > 0 else None,
            prefetch_factor=4 if workers > 0 else None,
        )

    def train_dataloader(self):
        return self._loader("train", shuffle=True)

    def val_dataloader(self):
        return self._loader("val", shuffle=False)

    def test_dataloader(self):
        return self._loader("test", shuffle=False)

    # ------------------------------------------------------------------ reporting --
    def summary(self) -> dict[str, dict]:
        """Split composition -- logged to MLflow so every run records what it trained on."""
        out: dict[str, Any] = {"split_mode": self.hparams.split_mode}
        if self.hparams.split_mode == "temporal":
            out["val_years"] = list(self.hparams.val_years)
            out["test_years"] = list(self.hparams.test_years)
        for split, series in self._series.items():
            out[split] = {
                "lakes": [s.lake_id for s in series],
                "days": {s.lake_id: int(s.n_days) for s in series},
                "samples": len(self._datasets[split]) if split in self._datasets else 0,
            }
        return out
