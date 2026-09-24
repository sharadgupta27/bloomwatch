"""Render the evaluation results as a self-contained HTML dashboard.

Reads every `artifacts/eval_*/scores.json` and emits one page with no external assets,
so it can be opened from disk, served by the API at `/dashboard`, or committed as a
build artefact.

    python scripts/make_dashboard.py
    python scripts/make_dashboard.py --out artifacts/dashboard.html

Design notes (the charts follow a validated method, not taste):

* Colour is assigned by *job*, not by rank. Model / persistence / climatology are three
  categorical identities and take the first three slots of a palette whose adjacent- and
  all-pairs colour-vision separation was validated with a script rather than eyeballed.
  Slot order is fixed, so a chart that drops a series never repaints the survivors.
* Single y-axis everywhere. Skill and RMSE are different scales and therefore live in
  different charts rather than on twin axes.
* The palette's aqua sits below 3:1 on the light surface, so the "relief rule" applies:
  every series carries a direct label and the full numbers are also available as a table.
  Identity is never conveyed by colour alone.
* Dark mode is a selected set of steps for the dark surface, not an inverted light theme.
"""

from __future__ import annotations

import argparse
import functools
import html
import json
import logging
import math
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# --- palette (validated: see module docstring) --------------------------------------
# Neutrals carry a slight cyan bias rather than being pure grey -- the subject is water
# and ocean-colour telemetry, and a chosen neutral reads differently from an inherited one.
# Series hues are the validated categorical slots and are NOT free to change; the surfaces
# below were re-run through the validator (light keeps a contrast WARN on aqua, which is
# why every series is also directly labelled and the full table is present).
LIGHT = {
    "surface": "#f6f8f8",
    "panel": "#ffffff",
    "line": "#dde4e5",
    "text": "#0c1517",
    "muted": "#4d5f63",
    "faint": "#8a9a9e",
    "s1": "#2a78d6",
    "s2": "#eb6834",
    "s3": "#1baf7a",
    "good": "#008300",
    "bad": "#e34948",
    "zero": "#b6c3c5",
}
DARK = {
    "surface": "#101617",
    "panel": "#182022",
    "line": "#2b383a",
    "text": "#eef4f4",
    "muted": "#a8babd",
    "faint": "#7b8c90",
    "s1": "#3987e5",
    "s2": "#d95926",
    "s3": "#199e70",
    "good": "#3cb043",
    "bad": "#e66767",
    "zero": "#41504f",
}
SERIES = [("Model", "s1"), ("Persistence", "s2"), ("Climatology", "s3")]


def esc(s) -> str:
    return html.escape(str(s))


# Kept out of the page's f-string: JS is brace-dense and doubling every one of them to
# survive `str.format` is how a working script quietly becomes a broken one.
GALLERY_JS = r"""
(function () {
  const node = document.getElementById('case-data');
  if (!node) return;
  const DATA = JSON.parse(node.textContent);
  const CASES = DATA.cases, LEADS = DATA.leads;
  if (!CASES.length) return;

  const $ = id => document.getElementById(id);
  const state = { c: 0, model: Object.keys(CASES[0].models)[0], lead: LEADS[0], strip: null };
  let timer = null;

  const num = (v, d = 1) => (v == null ? '—' : Number(v).toFixed(d));
  const pct = v => (v == null ? '—' : Math.round(v * 100) + '%');

  function stats(el, rows) {
    el.innerHTML = rows
      .filter(r => r)
      .map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`)
      .join('');
  }

  function render() {
    const c = CASES[state.c];
    const m = c.models[state.model];
    const li = LEADS.indexOf(state.lead);
    const lead = m.leads[li];
    const truth = c.truth[String(state.lead)];

    $('g-meta').innerHTML =
      `Anchor <strong>${c.anchor}</strong> · forecast issued from the ${c.input_days} days ` +
      `up to and including it, of which <strong>${c.observed_days}</strong> carried a usable ` +
      `retrieval · ${Math.round(c.water_km2).toLocaleString()} km² of water · ` +
      `grid ${c.shape[1]}×${c.shape[0]} at 300 m · checkpoint <code>${m.model_version}</code>`;

    $('im-pred').src = lead.prediction;
    $('d-pred').textContent = lead.valid_date + '  (+' + lead.lead_days + ' d)';
    const ps = lead.pred_stats;
    stats($('st-pred'), [
      ['Median', num(ps.median, 1) + ' mg/m³'],
      ['90th pct', num(ps.p90, 1) + ' mg/m³'],
      ['Above 20', num(ps.bloom_km2, 0) + ' km²'],
      ['Roughness', num(ps.roughness, 2)],
    ]);

    // Clicking an input day parks the middle pane on that observation instead of the
    // forecast target, which is how you check whether the model simply copied a day.
    const sd = state.strip == null ? null : c.window[state.strip];
    const shown = sd || truth;
    document.querySelector('#st-truth').closest('.pane').querySelector('.pane-t').textContent =
      sd ? 'Input day' : 'What actually happened';
    if (shown) {
      $('im-truth').src = shown.image;
      $('im-truth').style.visibility = '';
      $('d-truth').textContent = (sd ? sd.date : shown.date) + (sd ? '  (input)' : '');
      stats($('st-truth'), [
        ['Observed', pct(shown.coverage)],
        ['Median', num(shown.median, 1) + ' mg/m³'],
        ['90th pct', num(shown.p90, 1) + ' mg/m³'],
        ['Above 20', num(shown.bloom_km2, 0) + ' km²'],
        ['Roughness', num(shown.roughness, 2)],
      ]);
    } else {
      $('im-truth').style.visibility = 'hidden';
      $('d-truth').textContent = 'not observed';
      stats($('st-truth'), [['Observed', '0%']]);
    }

    if (sd) {
      $('im-err').style.visibility = 'hidden';
      $('d-err').textContent = '—';
      stats($('st-err'), [['—', 'not applicable while an input day is shown']]);
    } else if (lead.error) {
      $('im-err').src = lead.error;
      $('im-err').style.visibility = '';
      $('d-err').textContent = lead.valid_date;
      const es = lead.error_stats;
      stats($('st-err'), [
        ['RMSE', num(es.rmse, 3) + ' log₁₀'],
        ['Bias', (es.bias >= 0 ? '+' : '') + num(es.bias, 3) + ' log₁₀'],
        ['Scored px', es.n.toLocaleString()],
      ]);
    } else {
      $('im-err').style.visibility = 'hidden';
      $('d-err').textContent = 'no observation to score against';
      stats($('st-err'), [['Scored px', '0']]);
    }

    $('g-strip-n').textContent =
      ` · ${c.observed_days} of ${c.input_days} days usable`;
    $('g-strip').innerHTML = c.window
      .map(
        (w, i) =>
          `<button class="sf${w.observed ? '' : ' gap'}${i === state.strip ? ' on' : ''}" ` +
          `data-i="${i}" title="${w.date} — ${pct(w.coverage)} of the lake observed">` +
          `<img src="${w.image}" alt="observation ${w.date}">` +
          `<span>${w.date.slice(5)}</span></button>`
      )
      .join('');
    $('g-strip')
      .querySelectorAll('.sf')
      .forEach(b =>
        b.addEventListener('click', () => {
          const i = Number(b.dataset.i);
          state.strip = state.strip === i ? null : i;
          render();
        })
      );
  }

  function seg(id, key, cast) {
    $(id).addEventListener('click', e => {
      const b = e.target.closest('.sgb');
      if (!b) return;
      state[key] = cast(b.dataset[key]);
      $(id).querySelectorAll('.sgb').forEach(x => {
        const on = x === b;
        x.classList.toggle('on', on);
        x.setAttribute('aria-checked', on ? 'true' : 'false');
      });
      render();
    });
  }
  seg('g-model', 'model', String);
  seg('g-lead', 'lead', Number);

  function setLead(v) {
    state.lead = v;
    $('g-lead').querySelectorAll('.sgb').forEach(x => {
      const on = Number(x.dataset.lead) === v;
      x.classList.toggle('on', on);
      x.setAttribute('aria-checked', on ? 'true' : 'false');
    });
    render();
  }

  // Arrow keys inside the lead group, the behaviour a radiogroup is expected to have.
  $('g-lead').addEventListener('keydown', e => {
    const d = e.key === 'ArrowRight' ? 1 : e.key === 'ArrowLeft' ? -1 : 0;
    if (!d) return;
    e.preventDefault();
    const i = (LEADS.indexOf(state.lead) + d + LEADS.length) % LEADS.length;
    setLead(LEADS[i]);
    $('g-lead').querySelectorAll('.sgb')[i].focus();
  });

  $('g-case').addEventListener('change', e => {
    state.c = Number(e.target.value);
    state.strip = null;
    render();
  });

  function stop() {
    clearInterval(timer);
    timer = null;
    $('g-play').textContent = '▶ Play leads';
  }
  $('g-play').addEventListener('click', () => {
    if (timer) return stop();
    state.strip = null;
    $('g-play').textContent = '❚❚ Pause';
    timer = setInterval(() => {
      setLead(LEADS[(LEADS.indexOf(state.lead) + 1) % LEADS.length]);
    }, 1100);
  });

  render();
})();
"""


# ------------------------------------------------------------------ chart primitives --
def _axes(w, h, pad, xs, ylo, yhi, xlabel, ylabel, yfmt="{:.2f}", zero=False):
    """Recessive grid + axes. Returns (svg_parts, x_of, y_of)."""
    x0, y0 = pad["l"], pad["t"]
    pw, ph = w - pad["l"] - pad["r"], h - pad["t"] - pad["b"]
    span = (xs[-1] - xs[0]) or 1

    def X(v):
        return x0 + (v - xs[0]) / span * pw

    def Y(v):
        return y0 + (yhi - v) / ((yhi - ylo) or 1) * ph

    p = []
    ticks = 5
    for i in range(ticks + 1):
        v = ylo + (yhi - ylo) * i / ticks
        y = Y(v)
        p.append(f'<line x1="{x0}" y1="{y:.1f}" x2="{x0 + pw}" y2="{y:.1f}" class="grid"/>')
        p.append(f'<text x="{x0 - 8}" y="{y + 4:.1f}" class="tick tick-y">{yfmt.format(v)}</text>')
    if zero and ylo < 0 < yhi:
        # A zero line matters here: negative skill means "worse than the baseline".
        p.append(
            f'<line x1="{x0}" y1="{Y(0):.1f}" x2="{x0 + pw}" y2="{Y(0):.1f}" class="zeroline"/>'
        )
    for v in xs:
        p.append(f'<text x="{X(v):.1f}" y="{y0 + ph + 20}" class="tick tick-x">{v}</text>')
    p.append(f'<text x="{x0 + pw / 2:.1f}" y="{h - 4}" class="axis-title">{esc(xlabel)}</text>')
    p.append(
        f'<text transform="translate(13,{y0 + ph / 2:.1f}) rotate(-90)" '
        f'class="axis-title axis-title-y">{esc(ylabel)}</text>'
    )
    return p, X, Y


def line_chart(
    cid,
    series,
    xs,
    ylabel,
    xlabel="Forecast lead (days)",
    zero=False,
    w=520,
    h=300,
    yfmt="{:.2f}",
    unit="",
):
    """Multi-series line chart. `series` = [(name, slot, [values]), ...]."""
    pad = {"l": 62, "r": 96, "t": 16, "b": 44}
    vals = [v for _, _, ys in series for v in ys if v is not None and math.isfinite(v)]
    if not vals:
        return '<p class="muted">no data</p>'
    lo, hi = min(vals), max(vals)
    if zero:
        lo, hi = min(lo, 0.0), max(hi, 0.0)
    margin = (hi - lo) * 0.18 or 0.1
    lo, hi = lo - margin, hi + margin

    parts, X, Y = _axes(w, h, pad, xs, lo, hi, xlabel, ylabel, yfmt, zero)
    for name, slot, ys in series:
        pts = [
            (X(x), Y(y)) for x, y in zip(xs, ys, strict=True) if y is not None and math.isfinite(y)
        ]
        if not pts:
            continue
        d = " ".join(f"{'M' if i == 0 else 'L'}{x:.1f} {y:.1f}" for i, (x, y) in enumerate(pts))
        parts.append(f'<path d="{d}" class="ln" stroke="var(--{slot})"/>')
        for (px, py), yv in zip(
            pts, [y for y in ys if y is not None and math.isfinite(y)], strict=False
        ):
            # 2px surface ring keeps overlapping markers separable.
            parts.append(
                f'<circle cx="{px:.1f}" cy="{py:.1f}" r="5" fill="var(--{slot})" '
                f'class="mk" data-s="{esc(name)}" data-v="{yfmt.format(yv)}{esc(unit)}"/>'
            )
        # Direct label at the line end -- identity never rests on colour alone.
        lx, ly = pts[-1]
        parts.append(
            f'<text x="{lx + 10:.1f}" y="{ly + 4:.1f}" class="dlabel" '
            f'fill="var(--{slot})">{esc(name)}</text>'
        )
    return (
        f'<svg viewBox="0 0 {w} {h}" role="img" class="chart" id="{cid}">'
        + "".join(parts)
        + "</svg>"
    )


def grouped_bars(cid, groups, cats, ylabel, w=520, h=300, yfmt="{:.2f}"):
    """`groups` = [(group_label, {cat: value})]; `cats` = [(cat, slot)]."""
    pad = {"l": 62, "r": 16, "t": 16, "b": 56}
    vals = [v for _, d in groups for v in d.values() if v is not None and math.isfinite(v)]
    if not vals:
        return '<p class="muted">no data</p>'
    hi = max(vals) * 1.15 or 1.0
    x0, y0 = pad["l"], pad["t"]
    pw, ph = w - pad["l"] - pad["r"], h - pad["t"] - pad["b"]

    def Y(v):
        return y0 + (hi - v) / hi * ph

    parts = []
    for i in range(6):
        v = hi * i / 5
        y = Y(v)
        parts.append(f'<line x1="{x0}" y1="{y:.1f}" x2="{x0 + pw}" y2="{y:.1f}" class="grid"/>')
        parts.append(
            f'<text x="{x0 - 8}" y="{y + 4:.1f}" class="tick tick-y">{yfmt.format(v)}</text>'
        )

    gw = pw / max(len(groups), 1)
    bw = min(26, (gw - 14) / max(len(cats), 1))
    for gi, (label, d) in enumerate(groups):
        gx = x0 + gi * gw + (gw - bw * len(cats)) / 2
        for ci, (cat, slot) in enumerate(cats):
            v = d.get(cat)
            if v is None or not math.isfinite(v):
                continue
            # 2px gap between adjacent bars so fills never touch.
            bx, by = gx + ci * (bw + 2), Y(v)
            parts.append(
                f'<rect x="{bx:.1f}" y="{by:.1f}" width="{bw - 2:.1f}" '
                f'height="{max(y0 + ph - by, 0):.1f}" rx="4" fill="var(--{slot})" '
                f'class="mk" data-s="{esc(cat)}" data-v="{yfmt.format(v)}"/>'
            )
        parts.append(
            f'<text x="{x0 + gi * gw + gw / 2:.1f}" y="{y0 + ph + 20}" '
            f'class="tick tick-x">{esc(label)}</text>'
        )
    parts.append(
        f'<text transform="translate(13,{y0 + ph / 2:.1f}) rotate(-90)" '
        f'class="axis-title axis-title-y">{esc(ylabel)}</text>'
    )
    return (
        f'<svg viewBox="0 0 {w} {h}" role="img" class="chart" id="{cid}">'
        + "".join(parts)
        + "</svg>"
    )


def legend(items):
    sw = "".join(
        f'<span class="lg"><span class="sw" style="background:var(--{slot})"></span>{esc(n)}</span>'
        for n, slot in items
    )
    return f'<div class="legend">{sw}</div>'


def colourbar(cid, stops, labels, title):
    """A continuous scale strip with tick labels underneath."""
    grad = "".join(
        f'<stop offset="{i / (len(stops) - 1):.4f}" stop-color="{c}"/>' for i, c in enumerate(stops)
    )
    ticks = "".join(
        f'<text x="{i / (len(labels) - 1) * 240:.1f}" y="26" class="tick '
        f'{"cb-first" if i == 0 else "cb-last" if i == len(labels) - 1 else "tick-x"}">'
        f"{esc(t)}</text>"
        for i, t in enumerate(labels)
    )
    return (
        f'<div class="cb"><span class="cb-t">{esc(title)}</span>'
        f'<svg viewBox="0 0 240 32" class="cb-svg" role="img" aria-label="{esc(title)}">'
        f'<defs><linearGradient id="{cid}">{grad}</linearGradient></defs>'
        f'<rect x="0" y="0" width="240" height="11" rx="3" fill="url(#{cid})"/>'
        f"{ticks}</svg></div>"
    )


def gallery(payload: dict) -> str:
    """The interactive case-study viewer: real frames, real forecasts, real error.

    The shell is rendered here and the frames are driven from an inlined JSON blob, so the
    page stays a single file with no external requests while still letting a reader pivot
    between lake, episode, model and lead time. That pivot is the point: the generalisation
    gap this project's headline rests on is a claim about *which lake*, and a static figure
    can only assert it where an interactive one lets you go and check.
    """
    cases = payload.get("cases", [])
    if not cases:
        return ""
    models = list(cases[0]["models"])
    opts = "".join(
        f'<option value="{i}">{esc(c["lake_name"])} &middot; {esc(c["anchor"])}'
        f"{'  (held-out lake)' if c['split'] == 'test' else ''}</option>"
        for i, c in enumerate(cases)
    )
    model_btns = "".join(
        f'<button class="sgb{" on" if i == 0 else ""}" data-model="{esc(m)}" '
        f'role="radio" aria-checked="{"true" if i == 0 else "false"}">{esc(m)}</button>'
        for i, m in enumerate(models)
    )
    lead_btns = "".join(
        f'<button class="sgb{" on" if i == 0 else ""}" data-lead="{ld}" role="radio" '
        f'aria-checked="{"true" if i == 0 else "false"}">+{ld} d</button>'
        for i, ld in enumerate(payload["leads"])
    )
    chl_bar = colourbar(
        "cb-chl",
        ["#264696", "#28a08c", "#b4d23c", "#f0b428", "#c82828"],
        # The unit lives in the title, not on the last tick: an end-anchored "100 mg/m³"
        # runs straight into the "32" beside it at this width.
        ["1", "3", "10", "32", "100"],
        "Chlorophyll-a  mg/m³",
    )
    err_bar = colourbar(
        "cb-err",
        ["#2166ac", "#67a9cf", "#dee2e4", "#ef8a62", "#b2182b"],
        ["−4×", "", "correct", "", "+4×"],
        "Forecast error",
    )
    panes = "".join(
        f'<figure class="pane"><figcaption><span class="pane-t">{t}</span>'
        f'<span class="pane-d" id="d-{k}"></span></figcaption>'
        f'<div class="frame"><img id="im-{k}" alt="{esc(alt)}"></div>'
        f'<dl class="st" id="st-{k}"></dl></figure>'
        for k, t, alt in (
            ("pred", "Forecast", "predicted chlorophyll-a field"),
            ("truth", "What actually happened", "observed chlorophyll-a field"),
            ("err", "Error", "forecast minus observation"),
        )
    )
    return f"""
<div class="gal">
  <div class="ctls">
    <div class="ctl"><label for="g-case">Episode</label>
      <select id="g-case">{opts}</select></div>
    <div class="ctl"><label id="lb-m">Model</label>
      <div class="seg" id="g-model" role="radiogroup" aria-labelledby="lb-m">{model_btns}</div></div>
    <div class="ctl"><label id="lb-l">Lead time</label>
      <div class="leadrow">
        <div class="seg" id="g-lead" role="radiogroup" aria-labelledby="lb-l">{lead_btns}</div>
        <button id="g-play" class="playb" aria-label="Animate through lead times">▶ Play leads</button>
      </div></div>
  </div>
  <p class="cap" id="g-meta"></p>
  <div class="panes">{panes}</div>
  <div class="bars">{chl_bar}{err_bar}
    <div class="cb"><span class="cb-t">Water, no retrieval</span>
      <span class="nodata"></span></div></div>
  <div class="strip-h">Input window &mdash; the 14 days the model was given
    <span class="muted" id="g-strip-n"></span></div>
  <div class="strip" id="g-strip"></div>
  <p class="cap">Grey is water the sensor could not see through cloud that day; it is
  <em>not</em> clean water. Two thirds of a typical day is grey, which is the whole reason
  a forecast is built from a fortnight rather than from yesterday. Click a day to inspect it.</p>
</div>"""


# ------------------------------------------------------------------------ page build --
def load_cases(artifacts: Path) -> dict:
    f = artifacts / "case_studies.json"
    if not f.exists():
        return {}
    return json.loads(f.read_text(encoding="utf-8"))


def load_runs(artifacts: Path) -> dict[str, list[dict]]:
    runs = {}
    for d in sorted(artifacts.glob("eval_*")):
        f = d / "scores.json"
        if not f.exists():
            continue
        name = d.name.removeprefix("eval_")
        # `eval_unet_stream` was a verification re-run of `eval_unet`; prefer the newer.
        runs[name] = json.loads(f.read_text(encoding="utf-8"))
    if "unet_stream" in runs and "unet" in runs:
        runs["unet"] = runs.pop("unet_stream")
    return runs


def build(
    runs: dict[str, list[dict]], title: str, cases: dict | None = None, fragment: bool = False
) -> str:
    leads = sorted({r["lead_days"] for rows in runs.values() for r in rows}) or [1, 3, 5, 7]
    lakes = sorted(
        {(r["lake_id"], r.get("split", "")) for rows in runs.values() for r in rows},
        key=lambda t: (t[1] != "val", t[0]),
    )
    model_names = sorted(runs)

    def get(rows, lake, lead, key):
        for r in rows:
            if r["lake_id"] == lake and r["lead_days"] == lead:
                v = r.get(key)
                return None if v is None else float(v)
        return None

    css_vars = lambda p: "".join(f"--{k}:{v};" for k, v in p.items())  # noqa: E731

    # ---- hero numbers -----------------------------------------------------------
    tiles = []
    for name in model_names:
        rows = runs[name]
        val = [r for r in rows if r.get("split") == "val"]
        test = [r for r in rows if r.get("split") == "test"]
        bv = max((r["skill_vs_persistence"] for r in val), default=float("nan"))
        bt = max((r["skill_vs_persistence"] for r in test), default=float("nan"))
        tiles.append(
            f'<div class="tile"><div class="tile-h">{esc(name)}</div>'
            f'<div class="tile-n" style="color:var(--{"good" if bv > 0 else "bad"})">'
            f'{bv:+.3f}</div><div class="tile-s">best skill vs persistence &middot; validation lake</div>'
            f'<div class="tile-n2" style="color:var(--{"good" if bt > 0 else "bad"})">{bt:+.3f}</div>'
            f'<div class="tile-s">held-out test lake</div></div>'
        )

    # ---- chart 1: skill vs persistence, per lake, per model ---------------------
    skill_charts = []
    for lake, split in lakes:
        series = []
        for i, name in enumerate(model_names):
            slot = ["s1", "s2", "s3"][i % 3]
            series.append(
                (name, slot, [get(runs[name], lake, ld, "skill_vs_persistence") for ld in leads])
            )
        badge = "held out" if split == "test" else split
        skill_charts.append(
            f'<figure class="fig"><figcaption>{esc(lake)} '
            f'<span class="badge badge-{esc(split)}">{esc(badge)}</span></figcaption>'
            + legend([(n, ["s1", "s2", "s3"][i % 3]) for i, n in enumerate(model_names)])
            + line_chart(
                f"sk-{lake}", series, leads, "Skill vs persistence", zero=True, yfmt="{:+.2f}"
            )
            + '<p class="cap">Above zero beats "tomorrow looks like today". Below zero is '
            "worse than doing nothing.</p></figure>"
        )

    # ---- chart 2: the baseline crossover ----------------------------------------
    best = model_names[0]
    if len(model_names) > 1:
        best = max(
            model_names,
            key=lambda n: max(
                (r["skill_vs_persistence"] for r in runs[n] if r.get("split") == "val"), default=-9
            ),
        )
    cross = []
    for lake, split in lakes:
        rows = runs[best]
        series = [
            (f"{best}", "s1", [get(rows, lake, ld, "rmse") for ld in leads]),
            ("Persistence", "s2", [get(rows, lake, ld, "persistence_rmse") for ld in leads]),
            ("Climatology", "s3", [get(rows, lake, ld, "climatology_rmse") for ld in leads]),
        ]
        cross.append(
            f'<figure class="fig"><figcaption>{esc(lake)} '
            f'<span class="badge badge-{esc(split)}">{"held out" if split == "test" else split}'
            f"</span></figcaption>"
            + legend([(s[0], s[1]) for s in series])
            + line_chart(f"cr-{lake}", series, leads, "RMSE (log₁₀ mg/m³)")
            + '<p class="cap">Lower is better. Persistence degrades with horizon; climatology '
            "does not. Where they cross is where a learned model has room to earn its keep."
            "</p></figure>"
        )

    # ---- chart 3: event scores at the advisory threshold ------------------------
    ev_groups = []
    for lake, _ in lakes:
        rows = runs[best]
        for ld in leads:
            ev_groups.append(
                (
                    f"{lake.split('_')[0]}·{ld}d",
                    {
                        "POD": get(rows, lake, ld, "thr20_pod"),
                        "FAR": get(rows, lake, ld, "thr20_far"),
                        "CSI": get(rows, lake, ld, "thr20_csi"),
                    },
                )
            )
    ev_cats = [("POD", "s1"), ("FAR", "s2"), ("CSI", "s3")]
    events = (
        '<figure class="fig fig-wide"><figcaption>Advisory threshold, 20 mg/m³ '
        f"&mdash; {esc(best)}</figcaption>"
        + legend(ev_cats)
        + grouped_bars("ev", ev_groups, ev_cats, "Rate", w=1000, h=300)
        + '<p class="cap">POD = share of real bloom pixels caught (higher better). '
        "FAR = share of warnings that were wrong (lower better). CSI ignores the huge "
        "correct-negative class, which is why it is the operational standard.</p></figure>"
    )

    # ---- case studies: the images, and what only the images show ----------------
    cases = cases or {}
    gal = gallery(cases)
    gal_json = (
        '<script type="application/json" id="case-data">'
        + json.dumps(cases).replace("</", "<\\/")
        + "</script>"
        if gal
        else ""
    )
    texture = ""
    if gal:
        # Roughness is the defect the images make obvious and every scalar metric hides,
        # so it is stated as a number here rather than left to the reader's eye.
        obs = [
            t["roughness"]
            for c in cases["cases"]
            for t in c["truth"].values()
            if t.get("roughness")
        ]
        rough: list[tuple[str, float]] = []
        for name in cases["cases"][0]["models"]:
            vals = [
                ld["pred_stats"]["roughness"]
                for c in cases["cases"]
                for ld in c["models"][name]["leads"]
                if ld["pred_stats"].get("roughness")
            ]
            if vals:
                rough.append((name, sum(vals) / len(vals)))
        if obs and rough:
            o = sum(obs) / len(obs)
            # State the networks' *marginal* contribution from the measurement rather than
            # asserting it. Persistence is one of the rows, so the comparison is available.
            base = dict(rough).get("Persistence")
            if base is not None:
                others = [(n, v) for n, v in rough if n != "Persistence"]
                lo = min(v - base for _, v in others)
                hi = max(v - base for _, v in others)
                margin = (
                    "The tempting reading is that the networks paint speckle. Mostly they "
                    f"do not: <strong>persistence alone already sits at {base:.2f}</strong>, "
                    "and persistence contains no network at all. Against that, the models "
                    f"land between {lo:+.2f} and {hi:+.2f} &mdash; "
                    + (
                        "one of them is actually <em>smoother</em> than the baseline it is "
                        "built on. "
                        if lo < 0
                        else "a modest addition on top. "
                    )
                )
            else:
                margin = ""
            cells = "".join(
                f"<tr><td>{esc(n)}</td><td class='n'>{v:.2f}</td>"
                f"<td class='n'>{v / o:.1f}&times;</td></tr>"
                for n, v in rough
            )
            texture = (
                '<div class="note"><strong>What the pictures show that the table does '
                "not &mdash; and what a first look at them got wrong.</strong> Every "
                "forecast field is visibly grainier than the sensor's own composites. "
                "Measured as neighbour-to-neighbour difference over the field's own "
                f"spread, an observed target-day composite sits at <strong>{o:.2f}</strong>:"
                f"<div class='wrap'><table><tr><th>Field</th><th class='n'>Roughness</th>"
                f"<th class='n'>vs observed</th></tr>{cells}</table></div>"
                f"{margin}"
                "Its anchor is a mosaic &mdash; the most recent retrieval for each pixel, "
                "which can be yesterday for one pixel and nine days ago for its neighbour "
                "&mdash; and those seams are pixel-scale by construction. Every model here "
                "is built on that anchor and inherits it. So most of the graininess is a "
                "property of forecasting through 60&ndash;75% cloud gaps, not of the "
                "architectures, and the place to attack it is the gap-filling rather than "
                "the network head. It still matters operationally: RMSE barely penalises "
                "zero-mean speckle, and it stops being cosmetic the moment a product "
                "thresholds <em>area</em> above 20&nbsp;mg/m³ rather than averaging.</div>"
            )

    # ---- table (the relief rule + the accessible view) --------------------------
    head = (
        "<tr><th>Model</th><th>Lake</th><th>Split</th><th class='n'>Lead</th>"
        "<th class='n'>RMSE</th><th class='n'>vs persistence</th>"
        "<th class='n'>vs climatology</th><th class='n'>POD@20</th>"
        "<th class='n'>FAR@20</th><th class='n'>CSI@20</th></tr>"
    )

    def cell(row, k, spec="{:.3f}"):
        v = row.get(k)
        return "&mdash;" if v is None else spec.format(float(v))

    body = []
    for name in model_names:
        for r in sorted(
            runs[name], key=lambda r: (r.get("split", ""), r["lake_id"], r["lead_days"])
        ):
            f = functools.partial(cell, r)
            sp = r["skill_vs_persistence"]
            cls = "pos" if (sp is not None and sp > 0) else "neg"
            body.append(
                f"<tr><td>{esc(name)}</td><td>{esc(r['lake_id'])}</td>"
                f"<td><span class='badge badge-{esc(r.get('split', ''))}'>"
                f"{esc(r.get('split', ''))}</span></td>"
                f"<td class='n'>{r['lead_days']} d</td><td class='n'>{f('rmse')}</td>"
                f"<td class='n {cls}'>{f('skill_vs_persistence', '{:+.3f}')}</td>"
                f"<td class='n'>{f('skill_vs_climatology', '{:+.3f}')}</td>"
                f"<td class='n'>{f('thr20_pod')}</td><td class='n'>{f('thr20_far')}</td>"
                f"<td class='n'>{f('thr20_csi')}</td></tr>"
            )

    # An Artifact publish supplies its own <!doctype>/<head>/<body>, so the fragment form
    # drops the document wrapper and the link back to the API (which does not exist there).
    open_doc = (
        ""
        if fragment
        else '<!doctype html>\n<html lang="en"><head><meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
    )
    close_head = "" if fragment else "</head><body>"
    close_doc = "" if fragment else "</body></html>"
    api_link = "" if fragment else ' &middot;\n<a href="/">run a live forecast &rarr;</a>'
    return f"""{open_doc}
<title>{esc(title)}</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&amp;family=IBM+Plex+Sans:wght@400;500;600;700&amp;display=swap">
<style>
:root {{ {css_vars(LIGHT)} color-scheme: light; }}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{ {css_vars(DARK)} color-scheme: dark; }} }}
:root[data-theme="dark"] {{ {css_vars(DARK)} color-scheme: dark; }}
* {{ box-sizing: border-box; }}
/* IBM Plex was drawn for technical readouts, which is what this page is; mono carries
   the eyebrows and every digit so columns of numbers line up and read as instrument
   output rather than prose. */
body {{ margin:0; background:var(--surface); color:var(--text);
  font:14px/1.55 "IBM Plex Sans",ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif;
  -webkit-font-smoothing:antialiased; }}
header {{ padding:26px 28px 20px; border-bottom:1px solid var(--line); background:var(--panel); }}
.eyebrow {{ font-family:"IBM Plex Mono",ui-monospace,SFMono-Regular,Menlo,monospace;
  font-size:10.5px; letter-spacing:.16em; text-transform:uppercase; color:var(--s1);
  font-weight:500; }}
h1 {{ margin:6px 0 0; font-size:23px; font-weight:600; letter-spacing:-.02em;
  text-wrap:balance; }}
.sub {{ color:var(--muted); font-size:13px; margin-top:6px; max-width:66ch; }}
main {{ padding:22px 28px 56px; max-width:1160px; margin:0 auto; }}
h2 {{ font-family:"IBM Plex Mono",ui-monospace,monospace; font-size:11px;
  text-transform:uppercase; letter-spacing:.13em; color:var(--muted);
  margin:38px 0 12px; font-weight:500; display:flex; align-items:center; gap:12px; }}
h2::after {{ content:""; flex:1; height:1px; background:var(--line); }}
a {{ color:var(--s1); }}
:focus-visible {{ outline:2px solid var(--s1); outline-offset:2px; border-radius:3px; }}
@media (prefers-reduced-motion:reduce) {{ * {{ transition:none !important; }} }}
.tiles {{ display:flex; gap:14px; flex-wrap:wrap; }}
.tile {{ background:var(--panel); border:1px solid var(--line); border-radius:10px;
  padding:14px 18px; min-width:210px; }}
.tile-h {{ font-family:"IBM Plex Mono",ui-monospace,monospace; font-weight:500; font-size:11.5px; letter-spacing:.08em; text-transform:uppercase; }}
.tile-n {{ font-family:"IBM Plex Mono",ui-monospace,monospace; font-size:32px; font-weight:600; font-variant-numeric:tabular-nums; margin-top:8px; letter-spacing:-.02em; }}
.tile-n2 {{ font-family:"IBM Plex Mono",ui-monospace,monospace; font-size:21px; font-weight:600; font-variant-numeric:tabular-nums; margin-top:10px; }}
.tile-s {{ color:var(--muted); font-size:11.5px; }}
.grid2 {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(430px,1fr)); gap:18px; }}
.fig {{ margin:0; background:var(--panel); border:1px solid var(--line);
  border-radius:10px; padding:14px 16px 8px; }}
.fig-wide {{ grid-column:1/-1; }}
figcaption {{ font-weight:650; font-size:13.5px; margin-bottom:6px; }}
.chart {{ width:100%; height:auto; overflow:visible; }}
.grid {{ stroke:var(--line); stroke-width:1; }}
.zeroline {{ stroke:var(--zero); stroke-width:1.5; stroke-dasharray:4 3; }}
.ln {{ fill:none; stroke-width:2; stroke-linejoin:round; stroke-linecap:round; }}
.mk {{ stroke:var(--panel); stroke-width:2; cursor:pointer; }}
.tick {{ fill:var(--faint); font-size:10.5px; font-family:"IBM Plex Mono",ui-monospace,monospace; }}
.tick-y {{ text-anchor:end; }} .tick-x {{ text-anchor:middle; }}
.axis-title {{ fill:var(--muted); font-size:11.5px; text-anchor:middle; }}
.dlabel {{ font-size:11.5px; font-weight:600; }}
.legend {{ display:flex; gap:14px; flex-wrap:wrap; margin:2px 0 6px; font-size:12px;
  color:var(--muted); }}
.lg {{ display:inline-flex; align-items:center; gap:6px; }}
.sw {{ width:11px; height:11px; border-radius:3px; display:inline-block; }}
.cap {{ color:var(--muted); font-size:11.5px; margin:2px 0 8px; }}
.badge {{ font-family:"IBM Plex Mono",ui-monospace,monospace; font-size:9.5px; padding:2px 7px; border-radius:3px; border:1px solid var(--line);
  color:var(--muted); text-transform:uppercase; letter-spacing:.09em; font-weight:500; }}
.badge-test {{ color:var(--s2); border-color:var(--s2); }}
table {{ border-collapse:collapse; width:100%; margin-top:6px; font-size:12.5px; }}
th,td {{ text-align:left; padding:7px 10px; border-bottom:1px solid var(--line); }}
th {{ font-family:"IBM Plex Mono",ui-monospace,monospace; color:var(--muted); font-weight:500; font-size:10px; text-transform:uppercase;
  letter-spacing:.09em; }}
tbody tr:hover {{ background:color-mix(in srgb, var(--s1) 6%, transparent); }}
td.n, th.n {{ text-align:right; font-variant-numeric:tabular-nums; font-family:"IBM Plex Mono",ui-monospace,monospace; font-size:12px; }}
.pos {{ color:var(--good); font-weight:650; }} .neg {{ color:var(--bad); font-weight:650; }}
.muted {{ color:var(--muted); }}
.note {{ background:var(--panel); border:1px solid var(--line); border-left:3px solid var(--s2);
  border-radius:8px; padding:12px 16px; margin-top:14px; font-size:13px; }}
/* --- case-study viewer ---------------------------------------------------- */
.gal {{ background:var(--panel); border:1px solid var(--line); border-radius:10px;
  padding:16px 18px 14px; }}
.ctls {{ display:flex; gap:18px; flex-wrap:wrap; align-items:flex-end; }}
.ctl {{ display:flex; flex-direction:column; gap:5px; }}
.ctl label {{ font-family:"IBM Plex Mono",ui-monospace,monospace; font-size:10px;
  letter-spacing:.11em; text-transform:uppercase; color:var(--muted); }}
#g-case {{ font:inherit; font-size:13px; padding:7px 10px; min-width:290px;
  background:var(--surface); color:var(--text); border:1px solid var(--line);
  border-radius:7px; }}
.seg {{ display:flex; border:1px solid var(--line); border-radius:7px; overflow:hidden; }}
.sgb {{ font:inherit; font-size:12.5px; font-family:"IBM Plex Mono",ui-monospace,monospace;
  padding:7px 13px; background:var(--surface); color:var(--muted); border:0;
  border-right:1px solid var(--line); cursor:pointer; white-space:nowrap; }}
.sgb:last-child {{ border-right:0; }}
.sgb:hover {{ color:var(--text); }}
.sgb.on {{ background:var(--s1); color:#fff; }}
.leadrow {{ display:flex; gap:10px; align-items:center; flex-wrap:wrap; }}
.playb {{ font:inherit; font-size:12.5px; font-family:"IBM Plex Mono",ui-monospace,monospace;
  padding:7px 13px; background:var(--surface); color:var(--text);
  border:1px solid var(--line); border-radius:7px; cursor:pointer; }}
.playb:hover {{ border-color:var(--s1); color:var(--s1); }}
.panes {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(240px,1fr));
  gap:14px; margin-top:6px; }}
.pane {{ margin:0; }}
.pane figcaption {{ display:flex; justify-content:space-between; align-items:baseline;
  gap:8px; margin-bottom:6px; }}
.pane-t {{ font-size:12.5px; font-weight:650; }}
.pane-d {{ font-family:"IBM Plex Mono",ui-monospace,monospace; font-size:11px;
  color:var(--muted); font-variant-numeric:tabular-nums; }}
/* The frame is a fixed 4:3 box: lakes differ in aspect and a grid of jumping panels
   makes them hard to compare. contain preserves each lake's true shape inside it. */
.frame {{ aspect-ratio:4/3; background:var(--surface); border:1px solid var(--line);
  border-radius:8px; display:flex; align-items:center; justify-content:center;
  overflow:hidden; }}
.frame img {{ max-width:100%; max-height:100%; image-rendering:pixelated; display:block; }}
.st {{ margin:8px 0 0; display:grid; grid-template-columns:auto auto; gap:2px 10px;
  font-size:11.5px; }}
.st dt {{ color:var(--muted); }}
.st dd {{ margin:0; text-align:right; font-variant-numeric:tabular-nums;
  font-family:"IBM Plex Mono",ui-monospace,monospace; }}
.bars {{ display:flex; gap:26px; flex-wrap:wrap; margin-top:14px; align-items:flex-start; }}
.cb {{ display:flex; flex-direction:column; gap:3px; }}
.cb-t {{ font-family:"IBM Plex Mono",ui-monospace,monospace; font-size:10px;
  letter-spacing:.11em; text-transform:uppercase; color:var(--muted); }}
.cb-svg {{ width:240px; height:32px; overflow:visible; }}
.cb-first {{ text-anchor:start; }} .cb-last {{ text-anchor:end; }}
.nodata {{ width:26px; height:11px; border-radius:3px; background:#7c888c;
  margin-top:2px; }}
.strip-h {{ font-family:"IBM Plex Mono",ui-monospace,monospace; font-size:10px;
  letter-spacing:.11em; text-transform:uppercase; color:var(--muted); margin-top:18px; }}
.strip {{ display:flex; gap:5px; margin-top:7px; overflow-x:auto; padding-bottom:4px; }}
.sf {{ flex:0 0 auto; width:66px; border:0; padding:0; background:none; cursor:pointer;
  border-radius:6px; }}
.sf img {{ width:66px; height:50px; object-fit:contain; image-rendering:pixelated;
  background:var(--surface); border:1px solid var(--line); border-radius:6px;
  display:block; }}
.sf.gap img {{ opacity:.35; }}
.sf.on img {{ border-color:var(--s1); box-shadow:0 0 0 2px var(--s1); }}
.sf span {{ display:block; font-family:"IBM Plex Mono",ui-monospace,monospace;
  font-size:9px; color:var(--faint); margin-top:3px; text-align:center; }}
#tip {{ position:fixed; pointer-events:none; opacity:0; transition:opacity .1s;
  background:var(--text); color:var(--surface); padding:5px 9px; border-radius:6px;
  font-size:12px; font-variant-numeric:tabular-nums; z-index:9; white-space:nowrap;
  font-family:"IBM Plex Mono",ui-monospace,monospace; }}
.wrap {{ overflow-x:auto; }}
</style>{close_head}
<header><div class="eyebrow">Sentinel-3 OLCI &middot; leave-lake-out evaluation</div>
<h1>{esc(title)}</h1>
<div class="sub">Cyanobacterial bloom forecasting from Sentinel-3 OLCI &middot;
evaluated leave-lake-out against persistence and climatology{api_link}</div></header>
<main>
<h2>Headline</h2>
<div class="tiles">{"".join(tiles)}</div>

{"<h2>Case studies &mdash; what the model actually sees</h2>" + gal if gal else ""}
{texture}

<h2>Skill against persistence, by lead time</h2>
<div class="grid2">{"".join(skill_charts)}</div>

<h2>Why: the baseline crossover</h2>
<div class="grid2">{"".join(cross)}</div>

<h2>Event detection</h2>
<div class="grid2">{events}</div>

<h2>All numbers</h2>
<div class="wrap"><table>{head}{"".join(body)}</table></div>
<p class="cap">Skill is the Murphy skill score: 1 = perfect, 0 = no better than the
baseline, below 0 = worse. RMSE is in log₁₀ mg/m³. Every metric is masked to
pixels that were actually observed in the target frame and lie inside the lake's water
mask.</p>
<div class="note"><strong>Advisory only.</strong> Satellite chlorophyll-a is not a direct
cyanotoxin measurement. Confirm with in-situ sampling before any public-health action.</div>
</main>
<div id="tip"></div>
{gal_json}
<script>
const tip = document.getElementById('tip');
document.querySelectorAll('.mk').forEach(m => {{
  m.addEventListener('mouseenter', e => {{
    tip.textContent = e.target.dataset.s + ': ' + e.target.dataset.v;
    tip.style.opacity = 1;
  }});
  m.addEventListener('mousemove', e => {{
    tip.style.left = (e.clientX + 12) + 'px';
    tip.style.top  = (e.clientY - 30) + 'px';
  }});
  m.addEventListener('mouseleave', () => {{ tip.style.opacity = 0; }});
}});
{GALLERY_JS}
</script>
{close_doc}"""


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--artifacts", type=Path, default=REPO_ROOT / "artifacts")
    p.add_argument("--out", type=Path, default=REPO_ROOT / "artifacts" / "dashboard.html")
    p.add_argument("--title", default="habforecast — results")
    p.add_argument(
        "--fragment",
        action="store_true",
        help="emit body content only, for hosts that supply their own document shell",
    )
    args = p.parse_args(argv)

    runs = load_runs(args.artifacts)
    if not runs:
        raise SystemExit(
            f"no eval_*/scores.json under {args.artifacts} - run pipelines/evaluate.py first"
        )
    cases = load_cases(args.artifacts)
    if not cases:
        logging.warning(
            "no %s/case_studies.json -- the page will have no imagery; "
            "run scripts/make_case_studies.py first",
            args.artifacts,
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(build(runs, args.title, cases, fragment=args.fragment), encoding="utf-8")
    total = sum(len(v) for v in runs.values())
    mb = args.out.stat().st_size / 1e6
    print(
        f"wrote {args.out}  ({len(runs)} model(s), {total} rows, "
        f"{len(cases.get('cases', []))} case(s), {mb:.1f} MB)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
