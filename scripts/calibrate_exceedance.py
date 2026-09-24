"""Fit the exceedance-probability recalibration on the validation split.

    python pipelines/evaluate.py --checkpoint models/ens_met --out artifacts/eval_ens_met --tta --no-calibration
    python scripts/calibrate_exceedance.py --scores artifacts/eval_ens_met/scores.json --checkpoint models/ens_met

Reads the *uncalibrated* reliability tables of the validation split (never the test
split) and writes `calibration.json` where `habforecast.calibration.load` finds it for
that checkpoint.

    python scripts/calibrate_exceedance.py --scores ... --checkpoint ... \
        --by trophic --aoi data/aoi/lakes_global.geojson

`--by trophic` adds one map per trophic class, fitted on that class's validation lakes and
pulled toward the global map with a weight of `--prior-positives` exceedances (see
`habforecast/calibration.py`). See `habforecast/calibration.py` for why the map is two parameters.
Re-evaluate afterwards (without `--no-calibration`) to measure the held-out effect; the
validation rows are then in-sample and are stamped as such.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from habforecast.calibration import calibration_path, fit_from_reliability  # noqa: E402
from habforecast.eval.metrics import BLOOM_THRESHOLDS_MG_M3  # noqa: E402


def pool_bins(tables: list[list[dict]]) -> list[dict]:
    """Merge reliability tables from several lakes, pixel-weighted. Bins are matched by
    their probability decile, which every `StreamingReliability` table shares."""
    acc: dict[int, list[float]] = {}
    for table in tables:
        for b in table:
            k = min(int(b["mean_prob"] * 10), 9)
            n, sp, so = acc.get(k, [0.0, 0.0, 0.0])
            acc[k] = [n + b["n"], sp + b["n"] * b["mean_prob"], so + b["n"] * b["obs_freq"]]
    return [
        {"n": int(n), "mean_prob": sp / n, "obs_freq": so / n}
        for _k, (n, sp, so) in sorted(acc.items())
        if n > 0
    ]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--scores", type=Path, required=True, help="uncalibrated scores.json")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--split", default="val")
    p.add_argument(
        "--by",
        choices=["none", "trophic", "thermal", "continent"],
        default="none",
        help="also fit one map per level of this lake-group factor, pooled toward the global",
    )
    p.add_argument("--aoi", type=Path, default=None, help="registry giving each lake's groups")
    p.add_argument("--prior-positives", type=float, default=500.0)
    args = p.parse_args(argv)

    rows = [r for r in json.loads(args.scores.read_text()) if r.get("split") == args.split]
    if any(r.get("calibrated") for r in rows):
        raise SystemExit("these scores were computed with a calibration applied; refit from raw")
    params, lakes = {}, sorted({r["lake_id"] for r in rows})
    if not lakes:
        raise SystemExit(f"no {args.split} rows in {args.scores}")
    # Several validation lakes are pooled pixel-weighted, bin by bin: one map for all of
    # them, since the lake a forecast will be served on is by construction none of them.
    for lead in sorted({int(r["lead_days"]) for r in rows}):
        for thr in BLOOM_THRESHOLDS_MG_M3:
            tables = [
                r.get(f"thr{thr:g}_reliability") or [] for r in rows if int(r["lead_days"]) == lead
            ]
            bins = pool_bins(tables)
            if not bins:
                continue
            ab = fit_from_reliability(bins)
            if ab is not None:
                params[f"{lead}:{thr:g}"] = ab
            print(f"  lead {lead} thr {thr:g}: {ab if ab else 'too few exceedances; identity'}")
    doc = {
        "method": "platt",
        "fitted_on": {"split": args.split, "lakes": lakes, "scores": str(args.scores)},
        "fitted_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "params": params,
    }
    if args.by != "none":
        if args.aoi is None:
            raise SystemExit("--by needs --aoi to know each lake's group")
        from habforecast.aoi import AOIRegistry

        registry = AOIRegistry.from_geojson(args.aoi)
        level_of = {lid: registry[lid].groups.key(args.by) for lid in lakes if lid in registry.lake_ids}
        groups: dict[str, dict] = {}
        for level in sorted(set(level_of.values())):
            members = [lid for lid, lv in level_of.items() if lv == level]
            fits = {}
            for lead in sorted({int(r["lead_days"]) for r in rows}):
                for thr in BLOOM_THRESHOLDS_MG_M3:
                    key = f"{lead}:{thr:g}"
                    if key not in params:
                        continue  # no global fit to pool toward: served uncalibrated
                    tables = [
                        r.get(f"thr{thr:g}_reliability") or []
                        for r in rows
                        if int(r["lead_days"]) == lead and r["lake_id"] in members
                    ]
                    bins = pool_bins(tables)
                    if not bins:
                        continue
                    ab = fit_from_reliability(
                        bins, prior=params[key], prior_positives=args.prior_positives
                    )
                    if ab is not None:
                        fits[key] = ab
            groups[level] = fits
            print(f"  {level} ({', '.join(members)}): {len(fits)} fits")
        doc.update(
            {
                "method": "platt-pooled",
                "group_factor": args.by,
                "prior_positives": args.prior_positives,
                "groups": groups,
                "group_members": {
                    lv: [lid for lid, x in level_of.items() if x == lv] for lv in groups
                },
            }
        )
    out = calibration_path(args.checkpoint)
    out.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    print(f"wrote {out} ({len(params)} global fits, {len(doc.get('groups', {}))} groups)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
