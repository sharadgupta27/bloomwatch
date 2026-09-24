/* Time series for one pixel: what the satellite saw, what the model says next, and what
 * anyone has actually measured in that water.
 *
 * The whole point of the third series is that the first two are self-referential. A
 * forecast drawn beside its own input is a picture of a model agreeing with itself; a
 * laboratory chlorophyll value from a boat is the only mark on this chart that came from
 * outside the system being judged. So the in-situ marks are drawn to be *found* — they
 * are the reason the chart exists — while staying visually separate from the satellite
 * line, because merging them would claim an agreement nobody has demonstrated.
 *
 * Three rules the drawing keeps:
 *
 *   1. **Dashed means predicted.** Everything left of the anchor happened; everything
 *      right of it is the model talking. The rule is applied to the line, not only to the
 *      legend, so the distinction survives being screenshotted.
 *   2. **Hollow means a different year.** In-situ measurements taken inside the plotted
 *      window are solid diamonds and verify something. Ones taken at the same place in the
 *      same week of a *different* year are hollow circles and verify nothing; they are
 *      here because a lake's normal August is worth knowing. Their real date rides along
 *      in the tooltip.
 *   3. **One axis.** Chlorophyll only, log-scaled because a bloom is two orders of
 *      magnitude above clear water. Microcystin is a different quantity in different
 *      units and is listed beside the chart, never plotted on this scale.
 */

const NS = 'http://www.w3.org/2000/svg';
const MARGIN = { top: 10, right: 10, bottom: 22, left: 34 };

// The project's headline bloom threshold (habforecast.eval.metrics.BLOOM_THRESHOLDS_MG_M3).
const BLOOM_MG = 20;

// Log-friendly ticks. A linear axis would compress everything interesting into the
// bottom eighth of the plot: clear water is ~1 mg/m3 and a scum is ~300.
const TICKS = [0.3, 1, 3, 10, 30, 100, 300, 1000];

const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
  'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];

const DAY_MS = 86400000;

/** Day number from an ISO date, in UTC so a westerly browser does not shift the axis. */
export function dayNumber(iso) {
  const [y, m, d] = iso.split('-').map(Number);
  return Math.round(Date.UTC(y, m - 1, d) / DAY_MS);
}

function label(iso) {
  const [, m, d] = iso.split('-').map(Number);
  return `${d} ${MONTHS[m - 1]}`;
}

function el(tag, attrs, text) {
  const node = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs || {})) node.setAttribute(k, v);
  if (text !== undefined) node.textContent = text;
  return node;
}

function tip(node, text) {
  node.appendChild(el('title', {}, text));
  return node;
}

function fmt(mg) {
  return mg >= 100 ? mg.toFixed(0) : mg >= 10 ? mg.toFixed(1) : mg.toFixed(2);
}

/**
 * Build the chart.
 *
 * @param {object} o
 * @param {{date:string, mg:number|null}[]} o.observed  retrieved values at this pixel
 * @param {{date:string, mg:number|null, lead:number}[]} o.forecast  predicted values
 * @param {string} o.anchor        last observed day: the boundary between the two
 * @param {object} o.insitu        the /api/insitu payload, or null while it loads
 * @param {number} [o.width]
 * @param {number} [o.height]
 * @returns {HTMLElement} a figure holding the plot and its legend
 */
export function timeSeries(o) {
  const width = o.width || 300;
  const height = o.height || 170;
  const fig = document.createElement('figure');
  fig.className = 'ts';

  const obs = (o.observed || []).filter(p => typeof p.mg === 'number');
  const fc = (o.forecast || []).filter(p => typeof p.mg === 'number');
  const situ = insituPoints(o.insitu);

  if (!obs.length && !fc.length) {
    fig.innerHTML = '<p class="ts-empty">No values at this pixel to plot.</p>';
    return fig;
  }

  // --- scales ------------------------------------------------------------------
  const xs = [...obs, ...fc].map(p => dayNumber(p.date));
  const x0 = Math.min(...xs, ...situ.map(p => p.x));
  const x1 = Math.max(...xs, ...situ.map(p => p.x));
  const span = Math.max(1, x1 - x0);
  const plotW = width - MARGIN.left - MARGIN.right;
  const plotH = height - MARGIN.top - MARGIN.bottom;
  const X = d => MARGIN.left + ((d - x0) / span) * plotW;

  const values = [...obs, ...fc].map(p => p.mg).concat(situ.filter(p => p.mg > 0).map(p => p.mg));
  // Half a decade of headroom either way, so a flat series is not drawn as a flat line
  // pinned to the axis, and the bloom threshold stays on the chart when it is nearby.
  const lo = Math.log10(Math.max(0.05, Math.min(...values, BLOOM_MG / 3) / 1.6));
  const hi = Math.log10(Math.max(...values, BLOOM_MG * 1.2) * 1.6);
  const yspan = Math.max(0.35, hi - lo);
  const Y = mg => MARGIN.top + plotH - ((Math.log10(Math.max(mg, 0.05)) - lo) / yspan) * plotH;

  const svg = el('svg', {
    viewBox: `0 0 ${width} ${height}`, width: '100%', height,
    class: 'ts-svg', role: 'img',
    'aria-label': describe(obs, fc, situ, o.anchor),
  });

  // --- grid, ticks, reference lines ---------------------------------------------
  for (const t of TICKS) {
    const y = Y(t);
    if (y < MARGIN.top || y > MARGIN.top + plotH) continue;
    svg.appendChild(el('line', {
      class: 'ts-grid', x1: MARGIN.left, x2: width - MARGIN.right, y1: y, y2: y,
    }));
    svg.appendChild(el('text', { class: 'ts-ytick', x: MARGIN.left - 5, y: y + 3 }, String(t)));
  }
  if (Y(BLOOM_MG) > MARGIN.top + 8 && Y(BLOOM_MG) < MARGIN.top + plotH) {
    svg.appendChild(el('line', {
      class: 'ts-bloom', x1: MARGIN.left, x2: width - MARGIN.right,
      y1: Y(BLOOM_MG), y2: Y(BLOOM_MG),
    }));
    svg.appendChild(el('text', {
      class: 'ts-bloom-label', x: width - MARGIN.right, y: Y(BLOOM_MG) - 3,
    }, 'bloom'));
  }
  svg.appendChild(el('text', { class: 'ts-unit', x: 0, y: 8 }, 'mg/m³'));
  for (const d of xTicks(x0, x1)) {
    svg.appendChild(el('text', {
      class: 'ts-xtick', x: X(d), y: height - 7, 'text-anchor': 'middle',
    }, label(new Date(d * DAY_MS).toISOString().slice(0, 10))));
  }

  // The anchor is the only vertical the eye needs: left of it is measurement, right of
  // it is the model. Labelled, because an unlabelled rule invites the wrong reading.
  if (o.anchor) {
    const ax = X(dayNumber(o.anchor));
    svg.appendChild(el('line', {
      class: 'ts-anchor', x1: ax, x2: ax, y1: MARGIN.top, y2: MARGIN.top + plotH,
    }));
    svg.appendChild(el('text', {
      class: 'ts-anchor-label', x: ax + 3, y: MARGIN.top + 8,
    }, 'forecast from'));
  }

  // --- satellite: observed solid, predicted dashed -------------------------------
  if (obs.length > 1) svg.appendChild(el('path', { class: 'ts-obs-line', d: path(obs, X, Y) }));
  const bridge = obs.length ? [obs[obs.length - 1], ...fc] : fc;
  if (bridge.length > 1) {
    svg.appendChild(el('path', { class: 'ts-fc-line', d: path(bridge, X, Y) }));
  }
  for (const p of obs) {
    svg.appendChild(tip(el('circle', {
      class: 'ts-obs-dot', cx: X(dayNumber(p.date)), cy: Y(p.mg), r: 3.6,
    }), `${p.date} — satellite ${fmt(p.mg)} mg/m³`));
  }
  for (const p of fc) {
    svg.appendChild(tip(el('rect', {
      class: 'ts-fc-dot', x: X(dayNumber(p.date)) - 3.4, y: Y(p.mg) - 3.4,
      width: 6.8, height: 6.8, rx: 1,
    }), `${p.date} — forecast +${p.lead} d, ${fmt(p.mg)} mg/m³`));
  }

  // --- in-situ, drawn last so it is never hidden under the model -----------------
  for (const p of situ) {
    const cx = X(p.x);
    const cy = Y(p.mg > 0 ? p.mg : Math.pow(10, lo));
    const note = p.mg > 0 ? `${fmt(p.mg)} mg/m³` : 'below detection';
    if (p.sameWindow) {
      svg.appendChild(tip(el('path', {
        class: 'ts-situ', d: `M${cx} ${cy - 5}L${cx + 5} ${cy}L${cx} ${cy + 5}L${cx - 5} ${cy}Z`,
      }), `${p.date} — ${p.source} in situ ${note}, ${p.distance_km} km away`));
    } else {
      svg.appendChild(tip(el('circle', {
        class: 'ts-situ-past', cx, cy, r: 3.4,
      }), `${p.date} (other year) — ${p.source} in situ ${note}, ${p.distance_km} km away`));
    }
  }

  svg.appendChild(el('line', {
    class: 'ts-axis', x1: MARGIN.left, x2: width - MARGIN.right,
    y1: MARGIN.top + plotH, y2: MARGIN.top + plotH,
  }));
  fig.appendChild(svg);
  fig.appendChild(legend(obs.length, fc.length, situ));
  return fig;
}

/** Chlorophyll only, and each point told apart by which list it arrived in. */
function insituPoints(payload) {
  if (!payload || !payload.available) return [];
  const out = [];
  for (const p of payload.window || []) {
    if (p.variable !== 'chlorophyll_a') continue;
    out.push({ ...p, x: dayNumber(p.date), mg: p.value, sameWindow: true });
  }
  for (const p of payload.seasonal || []) {
    if (p.variable !== 'chlorophyll_a') continue;
    out.push({ ...p, x: dayNumber(p.plotted_date), mg: p.value, sameWindow: false });
  }
  return out;
}

function path(points, X, Y) {
  return points
    .map((p, i) => `${i ? 'L' : 'M'}${X(dayNumber(p.date)).toFixed(1)} ${Y(p.mg).toFixed(1)}`)
    .join('');
}

/** Four or five dates across the axis, always including both ends. */
function xTicks(x0, x1) {
  const span = x1 - x0;
  if (span <= 0) return [x0];
  const step = Math.max(1, Math.round(span / 4));
  const out = [];
  for (let d = x0; d <= x1; d += step) out.push(d);
  if (out[out.length - 1] !== x1) out.push(x1);
  return out;
}

function legend(nObs, nFc, situ) {
  const past = situ.filter(p => !p.sameWindow).length;
  const now = situ.length - past;
  const items = [
    ['obs', `Satellite, observed (${nObs})`],
    ['fc', `Forecast (${nFc})`],
  ];
  if (now) items.push(['situ', `In situ, this window (${now})`]);
  if (past) items.push(['situ-past', `In situ, same week other years (${past})`]);
  const box = document.createElement('figcaption');
  box.className = 'ts-legend';
  box.innerHTML = items
    .map(([k, t]) => `<span class="ts-key ts-key-${k}"></span>${t}`)
    .map(s => `<span class="ts-item">${s}</span>`)
    .join('');
  return box;
}

function describe(obs, fc, situ, anchor) {
  const parts = [`Chlorophyll-a at one pixel. ${obs.length} observed day(s)`];
  if (anchor) parts.push(`anchored ${anchor}`);
  parts.push(`${fc.length} forecast lead(s)`);
  if (situ.length) parts.push(`${situ.length} in-situ measurement(s) nearby`);
  return `${parts.join(', ')}.`;
}
