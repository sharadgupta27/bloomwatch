"""Promotion gate: replace the served checkpoint only if the candidate does not regress.

    # check only (CI, or before deciding):
    python scripts/promotion_gate.py --candidate artifacts/eval_new/scores.json \
        --incumbent artifacts/eval/scores.json
    # check and, on a pass, install the candidate as models/best.ckpt:
    python scripts/promotion_gate.py --candidate artifacts/eval_new/scores.json \
        --incumbent artifacts/eval/scores.json \
        --candidate-checkpoint artifacts/new/checkpoints/best.ckpt --promote-to models/best.ckpt

Two numbers decide it, both on the **held-out** split (`test` by default), per lead time,
over the lakes both evaluations scored:

- skill against persistence -- the regression is still the map;
- PR-AUC for exceeding 20 mg/m^3 -- what an operator acts on. For a candidate with an
  exceedance head the head's PR-AUC is used, since that is what its advisories read;
- R^2 in log10 -- how much of the held-out lake's variation the map explains. Guarded
  rather than maximised: per lake it divides by that lake's own variance, so on a
  uniformly bloomy lake it is low for a good forecast; what matters is that a new model
  does not explain *less* of the same lake than the one it replaces.

A candidate fails if either is lower than the incumbent's by more than its tolerance at
*any* lead. Per lead, not averaged: a model that gains at 1 day and loses at 7 is a
different forecast, not a better one, and an average would wave it through.

Lakes are pooled by an unweighted mean, so every held-out lake counts equally -- but a lake
with very few held-out samples must not decide a promotion on its own (user decision,
2026-09-24: on the expanded registry Hartbeespoort, 28 test samples, failed an otherwise
passing model). So by default any lake with fewer than 100 test samples on either side is
left out of the comparison and listed as `excluded` in the report; a lake whose rows carry
no sample count is kept. Both choices are recorded in the report:

- `--min-samples N` changes the threshold; `--min-samples 0` is the strict gate, every
  lake counts;
- `--weighting samples` weights each lake by its sample count instead of equally.

It also refuses to compare evaluations that are not like for like -- a different split
mode, TTA on one side only, or no lake in common -- because a gate that compares a
temporal split to a leave-lake-out one passes or fails on the difference between them.

On a pass with `--promote-to`, the incumbent is kept beside the new file as
`<name>.previous<suffix>`, the candidate's scores are installed as `<name>.scores.json` --
the incumbent the *next* candidate is gated against -- and a `PROMOTION.json` records both
evaluations and the margins, so a promotion can be audited and undone. Exit status: 0 pass, 1 fail, 2 not comparable.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path

PRAUC_KEY = "thr20_pr_auc"
HEAD_PRAUC_KEY = "head_thr20_pr_auc"
DEFAULT_MIN_SAMPLES = 100


def _rows(path: Path, split: str) -> list[dict]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    return [r for r in rows if r.get("split") == split]


def _prauc(row: dict) -> float | None:
    value = row.get(HEAD_PRAUC_KEY, row.get(PRAUC_KEY))
    return None if value is None or value != value else float(value)  # NaN -> None


def _mean(pairs: list[tuple[float, float]]) -> float:
    """Weighted mean of (value, weight) pairs; NaN when there are none."""
    total = sum(w for _, w in pairs)
    return sum(v * w for v, w in pairs) / total if total > 0 else float("nan")


def summarise(
    rows: list[dict], lakes: set[str], weighting: str = "lakes"
) -> dict[int, dict[str, float]]:
    """Per lead: mean skill vs persistence, PR-AUC@20 and R^2 over `lakes`, each lake
    counted once (`weighting="lakes"`) or by its sample count (`"samples"`)."""
    out: dict[int, dict[str, float]] = {}
    for lead in sorted({int(r["lead_days"]) for r in rows}):
        cell = [r for r in rows if int(r["lead_days"]) == lead and r["lake_id"] in lakes]

        def w(r: dict) -> float:
            return float(r.get("n_samples") or 0) if weighting == "samples" else 1.0

        skills = [(float(r["skill_vs_persistence"]), w(r)) for r in cell]
        r2s = [(float(r["r2"]), w(r)) for r in cell if r.get("r2") is not None]
        praucs = [(v, w(r)) for r in cell if (v := _prauc(r)) is not None]
        if not skills:
            continue
        out[lead] = {
            "skill_vs_persistence": _mean(skills),
            "pr_auc_20": _mean(praucs),
            "r2": _mean(r2s),
        }
    return out


def small_lakes(rows: list[dict], min_samples: int) -> set[str]:
    """Lakes with fewer than `min_samples` samples at any lead (rows without a count are kept)."""
    return {
        r["lake_id"]
        for r in rows
        if r.get("n_samples") is not None and int(r["n_samples"]) < min_samples
    }


def comparable(cand: list[dict], inc: list[dict]) -> str | None:
    """Why the two evaluations cannot be compared, or None if they can."""
    if not cand or not inc:
        return "one of the evaluations has no rows for this split"
    # Score files written before these fields existed were all leave-lake-out.
    defaults = {"split_mode": "lake", "tta": None}
    for key, default in defaults.items():
        a = {r.get(key, default) for r in cand}
        b = {r.get(key, default) for r in inc}
        if a != b:
            return (
                f"{key} differs: candidate {sorted(map(str, a))} vs incumbent {sorted(map(str, b))}"
            )
    if not {r["lake_id"] for r in cand} & {r["lake_id"] for r in inc}:
        return "no lake scored by both evaluations"
    return None


def gate(
    cand_rows: list[dict],
    inc_rows: list[dict],
    tol_skill: float = 0.01,
    tol_prauc: float = 0.01,
    tol_r2: float = 0.02,
    min_samples: int = DEFAULT_MIN_SAMPLES,
    weighting: str = "lakes",
) -> dict:
    lakes = {r["lake_id"] for r in cand_rows} & {r["lake_id"] for r in inc_rows}
    excluded: set[str] = set()
    if min_samples > 0:
        excluded = lakes & (small_lakes(cand_rows, min_samples) | small_lakes(inc_rows, min_samples))
        lakes = lakes - excluded
    cand = summarise(cand_rows, lakes, weighting)
    inc = summarise(inc_rows, lakes, weighting)
    checks, passed = [], True
    for lead in sorted(set(cand) & set(inc)):
        for metric, tol in (
            ("skill_vs_persistence", tol_skill),
            ("pr_auc_20", tol_prauc),
            ("r2", tol_r2),
        ):
            c, i = cand[lead][metric], inc[lead][metric]
            if c != c or i != i:  # NaN on either side: nothing to compare
                ok, margin = True, float("nan")
            else:
                margin = c - i
                ok = margin >= -tol
            passed &= ok
            checks.append(
                {
                    "lead_days": lead,
                    "metric": metric,
                    "candidate": c,
                    "incumbent": i,
                    "margin": margin,
                    "tolerance": tol,
                    "ok": ok,
                }
            )
    return {
        "passed": passed and bool(lakes),
        "lakes": sorted(lakes),
        "excluded": sorted(excluded),
        "min_samples": min_samples,
        "weighting": weighting,
        "checks": checks,
    }


def promote(candidate: Path, target: Path) -> Path | None:
    """Install `candidate` at `target`, keeping the incumbent beside it. Returns the backup."""
    backup = None
    if target.exists():
        backup = target.with_name(f"{target.stem}.previous{target.suffix}")
        if backup.exists():
            shutil.rmtree(backup) if backup.is_dir() else backup.unlink()
        target.replace(backup)
    if candidate.is_dir():
        shutil.copytree(candidate, target)
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_name(target.name + ".tmp")
        shutil.copy2(candidate, tmp)
        tmp.replace(target)  # atomic on one filesystem: never a half-written best.ckpt
    return backup


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--candidate", type=Path, required=True, help="candidate scores.json")
    p.add_argument("--incumbent", type=Path, required=True, help="served model's scores.json")
    p.add_argument("--split", default="test")
    p.add_argument("--tol-skill", type=float, default=0.01)
    p.add_argument("--tol-prauc", type=float, default=0.01)
    p.add_argument("--tol-r2", type=float, default=0.02)
    p.add_argument(
        "--min-samples",
        type=int,
        default=DEFAULT_MIN_SAMPLES,
        help="leave out lakes with fewer held-out samples than this, so a small lake cannot "
        f"veto on its own (default {DEFAULT_MIN_SAMPLES}; 0: strict, every lake counts)",
    )
    p.add_argument("--weighting", choices=["lakes", "samples"], default="lakes")
    p.add_argument("--candidate-checkpoint", type=Path, default=None)
    p.add_argument("--promote-to", type=Path, default=None)
    p.add_argument("--report", type=Path, default=None, help="write the gate result as JSON")
    args = p.parse_args(argv)

    cand_rows = _rows(args.candidate, args.split)
    inc_rows = _rows(args.incumbent, args.split)
    why = comparable(cand_rows, inc_rows)
    if why:
        print(f"NOT COMPARABLE: {why}", file=sys.stderr)
        return 2

    result = gate(
        cand_rows,
        inc_rows,
        args.tol_skill,
        args.tol_prauc,
        args.tol_r2,
        min_samples=args.min_samples,
        weighting=args.weighting,
    )
    result.update(
        {
            "candidate_scores": str(args.candidate),
            "incumbent_scores": str(args.incumbent),
            "split": args.split,
            "evaluated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        }
    )
    for c in result["checks"]:
        print(
            f"  {'ok  ' if c['ok'] else 'FAIL'} lead {c['lead_days']} {c['metric']:22s} "
            f"{c['candidate']:.4f} vs {c['incumbent']:.4f} ({c['margin']:+.4f}, tol {c['tolerance']})"
        )
    print(f"gate: {'PASS' if result['passed'] else 'FAIL'} on {', '.join(result['lakes'])}")
    if result["excluded"]:
        print(f"  excluded (< {args.min_samples} samples): {', '.join(result['excluded'])}")

    if result["passed"] and args.promote_to:
        if args.candidate_checkpoint is None:
            print("--promote-to needs --candidate-checkpoint", file=sys.stderr)
            return 2
        backup = promote(args.candidate_checkpoint, args.promote_to)
        shutil.copy2(
            args.candidate, args.promote_to.with_name(f"{args.promote_to.stem}.scores.json")
        )
        result["promoted"] = {
            "from": str(args.candidate_checkpoint),
            "to": str(args.promote_to),
            "previous": str(backup) if backup else None,
        }
        (args.promote_to.parent / "PROMOTION.json").write_text(
            json.dumps(result, indent=2), encoding="utf-8"
        )
        print(f"promoted {args.candidate_checkpoint} -> {args.promote_to}")
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
