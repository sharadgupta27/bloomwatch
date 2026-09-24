"""Evaluation: score a checkpoint per lake and lead time against both baselines.

Produces the results table that the README and model card quote. Three things are
reported for every (lake, lead) cell:

- regression error in log10 space;
- event scores at each advisory threshold, because a threshold crossing is what an
  operator acts on;
- skill against persistence *and* climatology, because either one alone flatters the
  model. Persistence is hard to beat at 1 day and easy at 7; climatology is the reverse.
  A model that beats both across the range is the only one worth deploying;
- for a checkpoint with an exceedance head, the head's own ranking (`head_thr20_pr_auc`)
  and its calibration (Brier, Brier skill against the base rate, ECE and the reliability
  table); for an ensemble (`--checkpoint` a directory of seeds), the spread beside the
  error, which a calibrated ensemble keeps roughly equal.

Scores are written per lake rather than pooled. A pooled average over lakes of very
different sizes and bloom regimes hides exactly the failure this project cares about:
degradation on an unseen lake.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from habforecast.eval.metrics import StreamingEvaluation, skill_score  # noqa: E402
from habforecast.features.datamodule import HABDataModule  # noqa: E402
from habforecast.inference.predictor import ensemble_forecast  # noqa: E402
from habforecast.models.baselines import ClimatologyBaseline, PersistenceBaseline  # noqa: E402

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="a .ckpt, or a directory of them to score as an ensemble (their mean)",
    )
    p.add_argument(
        "--met",
        type=Path,
        default=REPO_ROOT / "data" / "processed" / "met",
        help="ERA5 met stores, read only for a checkpoint trained with forcing",
    )
    p.add_argument("--aoi", type=Path, default=REPO_ROOT / "data" / "aoi" / "lakes.geojson")
    p.add_argument("--cubes", type=Path, default=REPO_ROOT / "data" / "processed" / "cubes")
    p.add_argument("--out", type=Path, default=REPO_ROOT / "artifacts" / "eval")
    p.add_argument("--splits", nargs="*", default=["val", "test"])
    p.add_argument("--device", default=None)
    p.add_argument(
        "--tta",
        action="store_true",
        help="average the forecast over horizontal/vertical flips (see models.lit.predict_tta)",
    )
    p.add_argument(
        "--split-mode",
        choices=["lake", "temporal"],
        default="lake",
        help="lake: hold out whole lakes (the default and the headline number); "
        "temporal: train on years up to a cut-off, score the years after it on every lake",
    )
    p.add_argument("--val-years", nargs="*", type=int, default=None)
    p.add_argument("--test-years", nargs="*", type=int, default=None)
    p.add_argument(
        "--no-calibration",
        action="store_true",
        help="score the raw exceedance probabilities; what scripts/calibrate_exceedance.py fits from",
    )
    p.add_argument(
        "--no-clim",
        action="store_true",
        help="withhold the seasonal climatology from a use_clim checkpoint: the number an "
        "ad-hoc area with no archive actually gets",
    )
    return p


@torch.no_grad()
def evaluate_split(
    members, dm, split: str, device, tta: bool = False, calibrate=None
) -> list[dict]:
    """Score every sample in a split, grouped by (lake, lead)."""
    loader = {"val": dm.val_dataloader, "test": dm.test_dataloader, "train": dm.train_dataloader}[
        split
    ]()
    if loader is None:
        logger.warning("split '%s' has no data", split)
        return []

    series = dm._series[split]
    lead_times = tuple(dm.hparams.lead_times)
    norm = dm.norm

    persistence = PersistenceBaseline(n_leads=len(lead_times)).to(device)
    climatology = ClimatologyBaseline(n_leads=len(lead_times)).fit(series, norm)

    # Accumulate running statistics per (lake, lead) rather than raw arrays. Collecting
    # arrays first needed 9.6 GB for Lake Victoria's test split and OOM'd the machine;
    # streaming gives identical numbers in constant memory. Averaging per-batch RMSEs
    # would be a different (and wrong) quantity, which is why this accumulates sums.
    accs: dict[tuple[str, int], StreamingEvaluation] = {}

    for batch in loader:
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        # The call serving makes (inference.predictor.ensemble_forecast), so an ensemble
        # or a head is scored exactly as the API would serve it, and every member reads
        # the channels it was trained with rather than chlorophyll alone.
        pred, prob, spread = ensemble_forecast(members, batch, tta)
        pers = persistence(batch["x"], batch["x_mask"], batch["water"])
        # `year` is passed so the climatology excludes the evaluated year -- without it
        # the baseline sees the days it is scored on and wins unfairly.
        clim = climatology(
            batch["x"],
            batch["x_mask"],
            batch["water"],
            batch["doy"],
            batch["lake_idx"],
            batch.get("year"),
        )

        # One device->host copy per tensor per batch, and the inverse standardisation
        # applied to the whole block, rather than twenty small transfers per sample.
        pred_np = norm.invert(pred.float().cpu().numpy())
        y_np = norm.invert(batch["y"].float().cpu().numpy())
        mask_np = batch["y_mask"].float().cpu().numpy()
        pers_np = norm.invert(pers.float().cpu().numpy())
        clim_np = norm.invert(clim.float().cpu().numpy())
        prob_np = prob.float().cpu().numpy() if prob is not None else None
        if calibrate is not None and prob_np is not None:
            # Per sample, with that lake's groups: a per-trophic-class calibration maps
            # the same raw probability differently on a eutrophic and an oligotrophic lake.
            for b in range(prob_np.shape[0]):
                lake = series[int(batch["lake_idx"][b])].lake_id
                groups = dm.registry[lake].groups if lake in dm.registry.lake_ids else None
                prob_np[b : b + 1] = calibrate(prob_np[b : b + 1], groups)
        # A spread is a width: it scales by the standardisation and never shifts.
        spread_np = spread.float().cpu().numpy() * norm.std if spread is not None else None
        lake_ids = batch["lake_idx"].tolist()

        for b in range(pred_np.shape[0]):
            lake_id = series[int(lake_ids[b])].lake_id
            for li, lead in enumerate(lead_times):
                acc = accs.get((lake_id, lead))
                if acc is None:
                    acc = accs[(lake_id, lead)] = StreamingEvaluation(lake_id, lead)
                acc.update(
                    pred_np[b, li],
                    y_np[b, li],
                    mask_np[b, li],
                    persistence=pers_np[b, li],
                    climatology=clim_np[b, li],
                    prob=prob_np[b, li] if prob_np is not None else None,
                    spread=spread_np[b, li] if spread_np is not None else None,
                )

    rows = []
    for _key, acc in sorted(accs.items()):
        ev = acc.result()
        row = ev.to_dict()
        row["split"] = split
        row["n_samples"] = acc.n_samples
        clim_rmse = acc.climatology.result().rmse
        row["climatology_rmse"] = clim_rmse
        row["skill_vs_climatology"] = skill_score(ev.regression.rmse, clim_rmse)
        # ROC / PR AUC of both baselines beside the model's, so the ranking metrics can
        # be read as skill rather than as absolute numbers.
        row.update(acc.baseline_rankings())
        row.update(acc.probabilistic())
        rows.append(row)
    return rows


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")

    from habforecast.inference.predictor import Predictor

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    predictor = Predictor(
        args.checkpoint, args.aoi, args.cubes, device=str(device), met_root=args.met
    ).load()
    met_future = bool(getattr(predictor.members[0], "met_future", False))

    dm = HABDataModule(
        aoi_path=args.aoi,
        cube_root=args.cubes,
        input_days=predictor.input_days,
        lead_times=predictor.lead_times,
        crop=None,
        batch_size=1,
        augment=False,
        # Use the constants the checkpoint was trained with, not refitted ones.
        norm=predictor.norm,
        # And the layers it was trained with. Evaluating a multi-layer checkpoint without
        # them would hand it all-zero channels and score the result as if that were the
        # model -- the comparison would show the extra inputs doing nothing, which is
        # exactly the conclusion the mistake would manufacture.
        aux_keys=predictor.aux_keys,
        climatology=predictor.use_clim and not args.no_clim,
        met_root=args.met if predictor.met_keys else None,
        met_keys=predictor.met_keys,
        met_future=met_future,
        split_mode=args.split_mode,
        **({"val_years": tuple(args.val_years)} if args.val_years else {}),
        **({"test_years": tuple(args.test_years)} if args.test_years else {}),
    )
    dm.setup()

    all_rows = []
    for split in args.splits:
        use_calib = predictor.calibration is not None and not args.no_calibration
        rows = evaluate_split(
            predictor.members,
            dm,
            split,
            device,
            tta=args.tta,
            calibrate=predictor.calibrate if use_calib else None,
        )
        for r in rows:
            r["split_mode"] = args.split_mode
            r["tta"] = bool(args.tta)
            r["climatology_input"] = bool(predictor.use_clim and not args.no_clim)
            r["checkpoint"] = predictor.model_version
            r["ensemble_size"] = len(predictor.members)
            r["met_keys"] = ",".join(predictor.met_keys)
            # Perfect prognosis used future weather; the row says so wherever it goes.
            r["met_future_perfect_prognosis"] = met_future
            r["calibrated"] = use_calib
            if use_calib:
                # A calibration scored on the lake it was fitted on is in-sample.
                fitted = predictor.calibration.get("fitted_on", {}).get("lakes", [])
                r["calibration_in_sample"] = r["lake_id"] in fitted
        all_rows.extend(rows)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "scores.json").write_text(json.dumps(all_rows, indent=2), encoding="utf-8")

    # A tidy CSV for DVC plots.
    import csv

    if all_rows:
        keys = sorted({k for r in all_rows for k in r})
        with (args.out / "skill_by_lead.csv").open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(all_rows)

    logger.info(
        "%-14s %-6s %5s %8s %8s %8s %8s %8s %8s",
        "lake",
        "split",
        "lead",
        "rmse",
        "skill_p",
        "skill_c",
        "csi@20",
        "auc@20",
        "prauc@20",
    )
    if any("head_thr20_pr_auc" in r for r in all_rows):
        logger.info("exceedance head: PR-AUC@20 (regression / head / persistence), BSS@20, ECE@20")
    for r in all_rows:
        logger.info(
            "%-14s %-6s %5d %8.4f %8.3f %8.3f %8.3f %8.3f %8.3f",
            r["lake_id"],
            r["split"],
            r["lead_days"],
            r["rmse"],
            r["skill_vs_persistence"],
            r["skill_vs_climatology"],
            r.get("thr20_csi", float("nan")),
            r.get("thr20_roc_auc", float("nan")),
            r.get("thr20_pr_auc", float("nan")),
        )
        if "head_thr20_pr_auc" in r:
            logger.info(
                "    head %-8s lead %d  prauc@20 %.3f / %.3f / %.3f   bss@20 %.3f  ece@20 %.3f%s",
                r["lake_id"],
                r["lead_days"],
                r.get("thr20_pr_auc", float("nan")),
                r["head_thr20_pr_auc"],
                r.get("persistence_thr20_pr_auc", float("nan")),
                r.get("thr20_bss", float("nan")),
                r.get("thr20_ece", float("nan")),
                f"  spread {r['ensemble_spread']:.3f} vs rmse {r['rmse']:.3f}"
                if "ensemble_spread" in r
                else "",
            )
    logger.info("wrote %s", args.out / "scores.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
