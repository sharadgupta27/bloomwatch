"""Render evaluation scores as the Markdown tables the README and model card quote.

Kept as a script rather than pasted output so the documents can never drift from the
numbers actually produced by the last evaluation run. Re-run it after every evaluation:

    python scripts/make_results_table.py --scores artifacts/eval/scores.json
    python scripts/make_results_table.py --scores artifacts/eval/scores.json \\
        --temporal artifacts/eval_temporal/scores.json --model-card docs/MODEL_CARD.md

With `--model-card` the tables are written into the model card between the
`<!-- results:begin -->` / `<!-- results:end -->` markers, so the card's quantitative
section is generated from `scores.json` rather than typed. Passing `--temporal` adds the
second split the catalogue asks for beside the leave-lake-out headline.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import UTC, datetime
from pathlib import Path

# Windows consoles default to cp1252, which cannot encode the subscripts and arrows this
# table uses. Reconfiguring stdout is cheaper than degrading the Markdown to ASCII.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BEGIN, END = "<!-- results:begin -->", "<!-- results:end -->"


def fmt(v, nd=3, dash="—"):
    if v is None:
        return dash
    try:
        f = float(v)
    except (TypeError, ValueError):
        return dash
    if f != f:  # NaN
        return dash
    return f"{f:.{nd}f}"


def _sorted(rows: list[dict]) -> list[dict]:
    order = {"train": 0, "val": 1, "test": 2}
    return sorted(
        rows, key=lambda x: (order.get(x.get("split", ""), 9), x["lake_id"], x["lead_days"])
    )


def skill_table(rows: list[dict]) -> str:
    """Per lake x lead: RMSE and skill against both baselines."""
    out = [
        "| Lake | Split | Lead | RMSE (log₁₀) | vs persistence | vs climatology | CSI @20 | POD @20 | FAR @20 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in _sorted(rows):
        out.append(
            f"| {r['lake_id']} | {r.get('split', '')} | {r['lead_days']} d "
            f"| {fmt(r.get('rmse'))} "
            f"| {fmt(r.get('skill_vs_persistence'))} "
            f"| {fmt(r.get('skill_vs_climatology'))} "
            f"| {fmt(r.get('thr20_csi'))} "
            f"| {fmt(r.get('thr20_pod'))} "
            f"| {fmt(r.get('thr20_far'))} |"
        )
    return "\n".join(out)


def ranking_table(rows: list[dict], threshold: float = 20.0) -> str:
    """Threshold-free event skill: ROC-AUC and PR-AUC for exceeding `threshold`, for the
    model and both baselines. Missing from scores produced before these were added."""
    tag = f"thr{threshold:g}_"
    if not any(f"{tag}roc_auc" in r for r in rows):
        return ""
    out = [
        "| Lake | Split | Lead | Bloom fraction | ROC-AUC | PR-AUC | persistence ROC / PR | climatology ROC / PR |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in _sorted(rows):
        out.append(
            f"| {r['lake_id']} | {r.get('split', '')} | {r['lead_days']} d "
            f"| {fmt(r.get(f'{tag}prevalence'))} "
            f"| {fmt(r.get(f'{tag}roc_auc'))} "
            f"| {fmt(r.get(f'{tag}pr_auc'))} "
            f"| {fmt(r.get(f'persistence_{tag}roc_auc'))} / {fmt(r.get(f'persistence_{tag}pr_auc'))} "
            f"| {fmt(r.get(f'climatology_{tag}roc_auc'))} / {fmt(r.get(f'climatology_{tag}pr_auc'))} |"
        )
    head = (
        f"#### Advisory-threshold ranking at {threshold:g} mg/m³\n\n"
        '_ROC-AUC and PR-AUC for "chlorophyll exceeds the threshold at the target date". '
        "PR-AUC is the one to read: blooms are rare, and a constant forecast scores its "
        "prevalence. A model that cannot rank better than persistence has not learned "
        "anything persistence did not already know._\n\n"
    )
    return head + "\n".join(out)


def baseline_table(rows: list[dict]) -> str:
    """The crossover story: persistence vs climatology vs model, by lead time."""
    leads = sorted({r["lead_days"] for r in rows})
    out = [
        "| Lead | Model RMSE | Persistence RMSE | Climatology RMSE |",
        "|---:|---:|---:|---:|",
    ]
    for lead in leads:
        sel = [r for r in rows if r["lead_days"] == lead]
        if not sel:
            continue

        # `rows_for_lead` is bound as a default so the closure captures this iteration's
        # value rather than whatever `sel` happens to hold when it is finally called.
        def mean(key, rows_for_lead=sel):
            vals = [
                float(r[key])
                for r in rows_for_lead
                if r.get(key) is not None and float(r[key]) == float(r[key])
            ]
            return sum(vals) / len(vals) if vals else None

        out.append(
            f"| {lead} d | {fmt(mean('rmse'))} | {fmt(mean('persistence_rmse'))} "
            f"| {fmt(mean('climatology_rmse'))} |"
        )
    return "\n".join(out)


def split_comparison(lake_rows: list[dict], temporal_rows: list[dict]) -> str:
    """The two numbers the catalogue asks for, side by side: skill on a lake never seen
    (leave-lake-out test) against skill on a year never seen (temporal test, all lakes).
    The gap between them is the cost of a new lake."""
    leads = sorted({r["lead_days"] for r in lake_rows} | {r["lead_days"] for r in temporal_rows})

    def mean_skill(rows, key):
        vals = [
            float(r[key])
            for r in rows
            if r.get("split") == "test"
            and r.get(key) is not None
            and float(r[key]) == float(r[key])
        ]
        return sum(vals) / len(vals) if vals else None

    out = [
        "| Lead | New lake: vs persistence | New lake: vs climatology | New year: vs persistence | New year: vs climatology |",
        "|---:|---:|---:|---:|---:|",
    ]
    for lead in leads:
        lk = [r for r in lake_rows if r["lead_days"] == lead]
        tm = [r for r in temporal_rows if r["lead_days"] == lead]
        out.append(
            f"| {lead} d | {fmt(mean_skill(lk, 'skill_vs_persistence'))} "
            f"| {fmt(mean_skill(lk, 'skill_vs_climatology'))} "
            f"| {fmt(mean_skill(tm, 'skill_vs_persistence'))} "
            f"| {fmt(mean_skill(tm, 'skill_vs_climatology'))} |"
        )
    lakes = sorted({r["lake_id"] for r in temporal_rows if r.get("split") == "test"})
    return (
        "#### Leave-lake-out vs. temporal split (test, mean over lakes)\n\n"
        f"_New lake: whole lakes held out. New year: every lake ({', '.join(lakes)}) trained "
        "on seasons up to the cut-off and scored on the years after it. The gap between the "
        "columns is what a lake outside the registry costs._\n\n" + "\n".join(out)
    )


def render(rows: list[dict], temporal: list[dict] | None = None) -> str:
    parts = [
        "#### Per lake and lead time",
        skill_table(rows),
        "#### Model vs baselines (mean over lakes)",
        baseline_table(rows),
        "_Skill is the Murphy skill score: 1 = perfect, 0 = no better than the baseline, "
        "< 0 = worse. Persistence is strongest at short lead; climatology does not degrade "
        "with horizon._",
    ]
    rk = ranking_table(rows)
    if rk:
        parts.append(rk)
    if temporal:
        parts.append(split_comparison(rows, temporal))
        parts.append("#### Temporal split, per lake and lead time")
        parts.append(skill_table(temporal))
    return "\n\n".join(parts)


def inject(card: Path, body: str, scores: Path) -> None:
    """Replace the block between the result markers in the model card."""
    text = card.read_text(encoding="utf-8")
    if BEGIN not in text or END not in text:
        raise SystemExit(f"{card} has no {BEGIN} / {END} markers")
    stamp = datetime.now(UTC).strftime("%Y-%m-%d")
    block = (
        f"{BEGIN}\n"
        f"_Generated by `scripts/make_results_table.py` from `{scores.as_posix()}` on {stamp}. "
        "Do not edit by hand._\n\n"
        f"{body}\n{END}"
    )
    new = re.sub(re.escape(BEGIN) + r".*?" + re.escape(END), lambda _m: block, text, flags=re.S)
    card.write_text(new, encoding="utf-8")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scores", type=Path, default=Path("artifacts/eval/scores.json"))
    p.add_argument(
        "--temporal",
        type=Path,
        default=None,
        help="scores.json from a temporal-split evaluation, rendered beside the headline",
    )
    p.add_argument("--out", type=Path, default=None, help="write Markdown here instead of stdout")
    p.add_argument(
        "--model-card",
        type=Path,
        default=None,
        help="inject the tables into this model card between the results markers",
    )
    args = p.parse_args(argv)

    if not args.scores.exists():
        raise SystemExit(f"no scores at {args.scores} - run pipelines/evaluate.py first")

    rows = json.loads(args.scores.read_text(encoding="utf-8"))
    if not rows:
        raise SystemExit("scores file is empty")
    temporal = None
    if args.temporal:
        if not args.temporal.exists():
            raise SystemExit(f"no temporal scores at {args.temporal}")
        temporal = json.loads(args.temporal.read_text(encoding="utf-8"))

    md = render(rows, temporal)

    if args.model_card:
        inject(args.model_card, md, args.scores)
        print(f"updated {args.model_card}")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(md, encoding="utf-8")
        print(f"wrote {args.out}")
    elif not args.model_card:
        print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
