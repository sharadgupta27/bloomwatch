"""Training entrypoint: Hydra config composition + Lightning + MLflow tracking.

Hydra is used here (and not in the ingest) because this is where sweeping earns its
keep:

    python pipelines/train.py                                  # defaults
    python pipelines/train.py model=unet train.lr=1e-4
    python pipelines/train.py -m model=unet,convlstm train.lr=1e-4,3e-4

Every run logs to MLflow: the resolved config, the split composition, normalisation
constants, per-lead skill against persistence, and the best checkpoint. The split
composition in particular is logged because a result on this dataset is meaningless
without knowing which lakes were held out.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, cast

import hydra
import lightning as L
import torch
from hydra.core.hydra_config import HydraConfig
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import MLFlowLogger
from omegaconf import DictConfig, OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from habforecast.features.datamodule import HABDataModule  # noqa: E402
from habforecast.models.convlstm import build_model  # noqa: E402
from habforecast.models.lit import HABForecastModule  # noqa: E402

logger = logging.getLogger(__name__)


def _to_dict(node: DictConfig) -> dict[str, Any]:
    """A config section as a plain dict (every section here is a mapping, never a list)."""
    return cast(dict[str, Any], OmegaConf.to_container(node, resolve=True))


def _count_params(model: torch.nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> float:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")
    for noisy in ("botocore", "urllib3", "fsspec", "numexpr", "git"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    L.seed_everything(cfg.seed, workers=True)
    torch.set_float32_matmul_precision("medium")  # TF32 on Ampere

    # ------------------------------------------------------------------ data ------
    dm = HABDataModule(
        aoi_path=cfg.paths.aoi,
        cube_root=cfg.paths.cubes,
        input_days=cfg.data.input_days,
        lead_times=tuple(cfg.data.lead_times),
        crop=cfg.data.crop,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        min_input_frames=cfg.data.min_input_frames,
        min_target_coverage=cfg.data.min_target_coverage,
        min_water_frequency=cfg.data.min_water_frequency,
        augment=cfg.data.augment,
        seed=cfg.seed,
        aux_keys=tuple(cfg.data.get("aux", ()) or ()),
        # The model config decides whether the climatology channels are built: a dataset
        # that computes them for a model that cannot read them is wasted work, and one
        # that omits them for a model that expects them trains on an empty channel.
        climatology=bool(cfg.model.get("use_clim", False)),
        clim_halfwidth=int(cfg.data.get("clim_halfwidth", 15)),
        clim_dropout=float(cfg.data.get("clim_dropout", 0.15)),
        met_root=cfg.paths.get("met"),
        met_keys=tuple(cfg.data.get("met", ()) or ()),
        met_future=bool(cfg.model.get("met_future", False)),
        split_mode=str(cfg.data.get("split_mode", "lake")),
        val_years=tuple(cfg.data.get("val_years", (2023,))),
        test_years=tuple(cfg.data.get("test_years", (2024, 2025))),
        group_dropout=float(cfg.data.get("group_dropout", 0.0)),
        mmap_dir=cfg.paths.get("mmap"),
    )
    dm.setup()
    split_summary = dm.summary()
    logger.info("split composition: %s", json.dumps(split_summary, indent=2))

    # ------------------------------------------------------------------ model -----
    model_kwargs = {k: v for k, v in _to_dict(cfg.model).items() if k != "name"}
    # The channel count is derived from the data config, never set independently: a model
    # built for three auxiliary layers and fed two is a shape error at best and a silent
    # mis-assignment of channels at worst.
    aux_keys = tuple(cfg.data.get("aux", ()) or ())
    if aux_keys and cfg.model.name in ("anomaly", "anomaly_deep"):
        model_kwargs["n_aux"] = len(aux_keys)
    # Same rule for the forcing: the channel count is derived from the data config, never
    # set independently, so a model built for six variables can never be fed five.
    met_keys = tuple(cfg.data.get("met", ()) or ())
    if met_keys and cfg.model.name in ("anomaly", "anomaly_deep"):
        model_kwargs["n_met"] = len(met_keys)
    # The exceedance head places absolute advisory thresholds in the model's standardised
    # units, so it is handed the constants the data module fitted on the training lakes.
    if model_kwargs.get("exceed_thresholds"):
        model_kwargs["norm_mean"] = float(dm.norm.mean)
        model_kwargs["norm_std"] = float(dm.norm.std)
    net = build_model(
        cfg.model.name,
        input_days=cfg.data.input_days,
        n_leads=len(cfg.data.lead_times),
        **model_kwargs,
    )
    total, trainable = _count_params(net)
    logger.info(
        "model=%s params: %.2fM total, %.2fM trainable",
        cfg.model.name,
        total / 1e6,
        trainable / 1e6,
    )

    module = HABForecastModule(
        model=net,
        lead_times=tuple(cfg.data.lead_times),
        lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay,
        huber_delta=cfg.train.huber_delta,
        lead_weighting=cfg.train.lead_weighting,
        warmup_epochs=cfg.train.warmup_epochs,
        max_epochs=cfg.train.max_epochs,
        norm_mean=dm.norm.mean,
        norm_std=dm.norm.std,
        exceed_weight=float(cfg.train.get("exceed_weight", 1.0)),
        focal_gamma=float(cfg.train.get("focal_gamma", 2.0)),
        # Recorded so inference rebuilds exactly this architecture instead of guessing
        # it from tensor shapes, which is ambiguous. See HABForecastModule.
        model_config={
            "name": cfg.model.name,
            "input_days": cfg.data.input_days,
            "aux_keys": list(aux_keys),
            "met_keys": list(met_keys),
            # The forcing's train-lake standardisation travels with the checkpoint, so
            # serving applies the constants the model was fitted with.
            "met_stats": {k: list(v) for k, v in dm._met_stats.items()},
            **model_kwargs,
        },
    )

    # ------------------------------------------------------------------ logging ---
    # SQLite needs its parent directory to exist before the store is opened.
    tracking_uri = str(cfg.paths.mlflow_uri)
    if tracking_uri.startswith("sqlite:///"):
        Path(tracking_uri.removeprefix("sqlite:///")).parent.mkdir(parents=True, exist_ok=True)

    mlf = MLFlowLogger(
        experiment_name=cfg.experiment_name,
        run_name=cfg.run_name,
        tracking_uri=tracking_uri,
    )
    mlf.log_hyperparams(
        {
            **{f"model/{k}": v for k, v in _to_dict(cfg.model).items()},
            **{f"train/{k}": v for k, v in _to_dict(cfg.train).items()},
            **{f"data/{k}": v for k, v in _to_dict(cfg.data).items()},
            "seed": cfg.seed,
            "params_total": total,
            "params_trainable": trainable,
            "norm_mean": dm.norm.mean,
            "norm_std": dm.norm.std,
            "split/mode": split_summary.get("split_mode", "lake"),
            "split/train_lakes": ",".join(split_summary["train"]["lakes"]),
            "split/val_lakes": ",".join(split_summary["val"]["lakes"]),
            "split/test_lakes": ",".join(split_summary["test"]["lakes"]),
            "split/train_samples": split_summary["train"]["samples"],
            "split/val_samples": split_summary.get("val", {}).get("samples", 0),
        }
    )

    # Hydra 1.2+ defaults `hydra.job.chdir` to False, so cwd is still the invocation
    # directory -- using it here silently scattered checkpoints into the repo root.
    # The runtime output_dir is the authoritative per-run location.
    run_dir = Path(HydraConfig.get().runtime.output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "split_summary.json").write_text(
        json.dumps(split_summary, indent=2), encoding="utf-8"
    )
    (run_dir / "config_resolved.yaml").write_text(
        OmegaConf.to_yaml(cfg, resolve=True), encoding="utf-8"
    )

    ckpt = ModelCheckpoint(
        dirpath=run_dir / "checkpoints",
        filename="best-{epoch:02d}-{val/skill_overall:.4f}",
        monitor=cfg.train.monitor,
        mode=cfg.train.monitor_mode,
        save_top_k=1,
        save_last=True,
        auto_insert_metric_name=False,
    )
    callbacks = [ckpt, LearningRateMonitor(logging_interval="epoch")]
    if dm.val_dataloader() is not None:
        callbacks.append(
            EarlyStopping(
                monitor=cfg.train.monitor,
                mode=cfg.train.monitor_mode,
                patience=cfg.train.early_stopping_patience,
                min_delta=1e-4,
            )
        )

    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    trainer = L.Trainer(
        max_epochs=cfg.train.max_epochs,
        accelerator=accelerator,
        devices=1,
        precision=cfg.train.precision if accelerator == "gpu" else "32-true",
        accumulate_grad_batches=cfg.train.accumulate_grad_batches,
        gradient_clip_val=cfg.train.gradient_clip_val,
        log_every_n_steps=cfg.train.log_every_n_steps,
        deterministic=cfg.train.deterministic,
        callbacks=callbacks,
        logger=mlf,
        default_root_dir=str(run_dir),
        enable_progress_bar=True,
    )

    trainer.fit(module, datamodule=dm)

    # ------------------------------------------------------------------ testing ---
    if dm.test_dataloader() is not None:
        logger.info("evaluating on held-out lake(s): %s", split_summary["test"]["lakes"])
        trainer.test(module, datamodule=dm, ckpt_path="best")

    best = ckpt.best_model_score
    logger.info("best %s = %s (%s)", cfg.train.monitor, best, ckpt.best_model_path)
    if ckpt.best_model_path:
        # A completed training run must not be thrown away because the artifact store is
        # misconfigured. The checkpoint is already safely on disk under
        # `artifacts/<run>/checkpoints/` and tracked by DVC; the MLflow copy is a
        # convenience for the model registry. This used to raise, and a forty-minute GPU
        # run died on its last line with EACCES because the tracking server's artifact
        # root pointed at a path only the *server* container can write.
        try:
            mlf.experiment.log_artifact(mlf.run_id, ckpt.best_model_path)
        except Exception as exc:  # never fatal at this point
            logger.warning(
                "could not upload the checkpoint to MLflow (%s: %s); it is still at %s",
                type(exc).__name__,
                exc,
                ckpt.best_model_path,
            )

    # Returned so Hydra multirun sweeps can optimise against it.
    return float(best) if best is not None else float("nan")


if __name__ == "__main__":
    main()
