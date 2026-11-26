/* Bloom Watch — application logic.
 *
 * Four responsibilities, kept separate:
 *   1. tools    — what a click on the map means right now (explore / inspect / draw / measure)
 *   2. vectors  — everything drawn over the tiles, re-rendered whenever the map moves
 *   3. forecast — submit a job, follow its progress, show what came back
 *   4. export   — compose the view into a printable map sheet (see export.js)
 *
 * The inspect tool is the one that reaches outside this system. A click reports the
 * predicted value at that pixel, plots the fortnight of retrieval behind it against the
 * week of forecast ahead of it (see chart.js), and asks the server what anyone has
 * actually measured in that water within 10 km. That last series is the only thing on
 * screen that did not come out of the model.
 *
 * The drawing tools are not decoration. A rectangle or circle *is* the area of interest
 * the model runs on, so drawing and requesting are the same gesture rather than two. Once
 * drawn, the geometry stays editable — draggable and deletable — because the first place
 * you click is rarely the right one.
 */

import {
  BASEMAPS, MapView, formatArea, formatDistance, formatLatLon, graticule, haversine, ringArea,
} from './map.js';
import { downloadSheet } from './export.js';
import { timeSeries } from './chart.js';

const $ = id => document.getElementById(id);
const SVGNS = 'http://www.w3.org/2000/svg';
const esc = s => String(s).replace(/[&<>"']/g, c =>
  ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

const S = {
  tool: 'pan',
  aoi: null,          // {kind:'circle',lat,lon,radiusKm} | {kind:'bbox',bbox:[w,s,e,n],lat,lon}
  draft: null,        // first click of a two-click shape
  hover: null,
  measure: [],
  measuring: false,
  poly: [],            // vertices of a polygon being drawn
  result: null,
  grids: null,        // {lead: Uint16Array} decoded from the result, for the inspect tool
  history: null,      // {isoDate: Uint16Array} the input window, same encoding
  probe: null,        // {lat, lon} of the inspected pixel
  insitu: null,       // last /api/insitu answer, keyed by probeKey()
  insituKm: 10,       // how far from the click to look for a bottle sample
  lead: null,
  opacity: 0.9,
  radiusKm: 12,
  showGrid: false,
};

/* =============================================================== map bootstrap === */
const map = new MapView($('map'), { lat: 24, lon: 12, zoom: 2.4, basemap: 'satellite' });

map.on('render', () => { drawVectors(); updateReadout(); });
map.on('hover', geo => { S.hover = geo; if (S.draft || S.measuring) drawVectors(); updateReadout(); });
map.on('click', onMapClick);
map.on('basemap', key => {
  document.querySelectorAll('.tile-opt').forEach(b => b.classList.toggle('on', b.dataset.base === key));
  $('attrib').innerHTML = BASEMAPS[key].attribution;
  applyTheme();
});
$('attrib').innerHTML = BASEMAPS.satellite.attribution;

/* ==================================================================== vectors ==== */
function svg(tag, attrs) {
  const el = document.createElementNS(SVGNS, tag);
  for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, v);
  return el;
}

function label(x, y, text, cls = '') {
  const g = document.createElementNS(SVGNS, 'g');
  const t = svg('text', { x, y, class: `vlabel ${cls}` });
  t.textContent = text;
  const bg = svg('rect', { class: 'vlabel-bg', rx: 4 });
  g.append(bg, t);
  // Size the plate after the text is measurable; done on the next frame by the caller.
  g.dataset.fit = '1';
  return g;
}

function fitLabels(root) {
  root.querySelectorAll('g[data-fit]').forEach(g => {
    const t = g.querySelector('text'), bg = g.querySelector('rect');
    const b = t.getBBox();
    bg.setAttribute('x', b.x - 6); bg.setAttribute('y', b.y - 3);
    bg.setAttribute('width', b.width + 12); bg.setAttribute('height', b.height + 6);
  });
}

/** Metres between two lon degrees at a latitude — used to size the AOI circle on screen. */
function circleScreenRadius(lat, lon, radiusKm) {
  const c = map.project(lat, lon);
  const dLon = radiusKm / (111.320 * Math.cos(lat * Math.PI / 180));
  return Math.abs(map.project(lat, lon + dLon).x - c.x);
}

function ringBounds(rings) {
  const lons = rings.flat().map(p => p[0]);
  const lats = rings.flat().map(p => p[1]);
  return [Math.min(...lons), Math.min(...lats), Math.max(...lons), Math.max(...lats)];
}

function aoiBbox(aoi) {
  if (!aoi) return null;
  if (aoi.kind === 'poly') return ringBounds(aoi.rings);
  if (aoi.kind === 'bbox') return aoi.bbox;
  const dLat = aoi.radiusKm / 110.574;
  const dLon = aoi.radiusKm / (111.320 * Math.cos(aoi.lat * Math.PI / 180));
  return [aoi.lon - dLon, aoi.lat - dLat, aoi.lon + dLon, aoi.lat + dLat];
}

function drawVectors() {
  const layer = map.vectorLayer;
  layer.replaceChildren();

  if (S.showGrid) drawGraticule(layer);

  // --- area of interest -----------------------------------------------------------
  const draft = S.draftShape();
  const aoi = draft || S.aoi;
  if (aoi) {
    // A committed shape is draggable, but only while dragging is what a click means.
    // Under the inspect and measure tools the click belongs to the pixel or the ruler,
    // and a filled AOI that captured pointer events would swallow every one of them --
    // which is to say, exactly the clicks the user came to make.
    const movable = !draft && S.tool !== 'probe' && S.tool !== 'measure';
    const grab = movable ? ' aoi-grab' : '';
    if (aoi.kind === 'circle') {
      const c = map.project(aoi.lat, aoi.lon);
      const r = circleScreenRadius(aoi.lat, aoi.lon, aoi.radiusKm);
      layer.appendChild(dragify(svg('circle', { cx: c.x, cy: c.y, r, class: `aoi${grab}` }), !movable));
      layer.appendChild(dragify(svg('circle', { cx: c.x, cy: c.y, r: 5, class: `aoi-pin${grab}` }), !movable));
      layer.appendChild(label(c.x + r + 8, c.y - 6, `${aoi.radiusKm.toFixed(1)} km radius`));
    } else if (aoi.kind === 'poly') {
      for (const ring of aoi.rings) {
        const d = ring.map(([lo, la], i) => {
          const p = map.project(la, lo);
          return `${i ? 'L' : 'M'}${p.x} ${p.y}`;
        }).join(' ') + ' Z';
        layer.appendChild(dragify(svg('path', { d, class: `aoi${grab}` }), !movable));
      }
      const [w, s, e, n] = ringBounds(aoi.rings);
      const tl = map.project(n, w);
      const verts = aoi.rings.reduce((a, r) => a + r.length, 0);
      layer.appendChild(label(tl.x + 4, tl.y - 9,
        `${aoi.rings.length > 1 ? `${aoi.rings.length} rings, ` : ''}${verts} points`));
    } else {
      const [w, s, e, n] = aoi.bbox;
      const tl = map.project(n, w), br = map.project(s, e);
      layer.appendChild(dragify(svg('rect', {
        x: Math.min(tl.x, br.x), y: Math.min(tl.y, br.y),
        width: Math.abs(br.x - tl.x), height: Math.abs(br.y - tl.y), rx: 2, class: `aoi${grab}`,
      }), !movable));
      const km = haversine({ lat: n, lon: w }, { lat: n, lon: e }) / 1000;
      const kmY = haversine({ lat: s, lon: w }, { lat: n, lon: w }) / 1000;
      layer.appendChild(label(Math.min(tl.x, br.x), Math.min(tl.y, br.y) - 9,
        `${km.toFixed(1)} × ${kmY.toFixed(1)} km`));
    }
  }

  // --- measurement -----------------------------------------------------------------
  const pts = S.measure.slice();
  if (S.measuring && S.hover && pts.length) pts.push(S.hover);
  if (pts.length > 1) {
    const d = pts.map(p => map.project(p.lat, p.lon)).map((p, i) => `${i ? 'L' : 'M'}${p.x} ${p.y}`).join(' ');
    layer.appendChild(svg('path', { d, class: 'measure-line' }));
    let total = 0;
    for (let i = 1; i < pts.length; i++) {
      total += haversine(pts[i - 1], pts[i]);
      const a = map.project(pts[i - 1].lat, pts[i - 1].lon);
      const b = map.project(pts[i].lat, pts[i].lon);
      layer.appendChild(label((a.x + b.x) / 2, (a.y + b.y) / 2 - 8,
        formatDistance(haversine(pts[i - 1], pts[i])), 'seg'));
    }
    const last = map.project(pts.at(-1).lat, pts.at(-1).lon);
    layer.appendChild(label(last.x + 12, last.y + 14, `Σ ${formatDistance(total)}`, 'total'));
    if (pts.length > 2) {
      const c = pts.reduce((a, p) => ({ x: a.x + p.lon, y: a.y + p.lat }), { x: 0, y: 0 });
      const centroid = map.project(c.y / pts.length, c.x / pts.length);
      layer.appendChild(label(centroid.x, centroid.y, formatArea(ringArea(pts)), 'total'));
    }
  }
  S.measure.forEach(p => {
    const s = map.project(p.lat, p.lon);
    layer.appendChild(svg('circle', { cx: s.x, cy: s.y, r: 4, class: 'measure-pt' }));
  });
  // Vertices placed so far, so it is clear where the next click attaches.
  S.poly.forEach(p => {
    const s = map.project(p.lat, p.lon);
    layer.appendChild(svg('circle', { cx: s.x, cy: s.y, r: 4, class: 'aoi-vertex' }));
  });

  // --- inspected pixel ---------------------------------------------------------------
  if (S.probe) {
    const p = map.project(S.probe.lat, S.probe.lon);
    // Drawn twice, white under dark: one stroke colour cannot stay visible over both a
    // navy-blue clear-water pixel and a red bloom.
    for (const cls of ['probe-ring', 'probe-ring-in']) {
      layer.appendChild(svg('circle', { cx: p.x, cy: p.y, r: 7, class: cls }));
      layer.appendChild(svg('path', {
        d: `M${p.x - 12} ${p.y}H${p.x - 3}M${p.x + 3} ${p.y}H${p.x + 12}` +
           `M${p.x} ${p.y - 12}V${p.y - 3}M${p.x} ${p.y + 3}V${p.y + 12}`,
        class: cls,
      }));
    }
  }

  fitLabels(layer);
  updateMeasurePanel();
}

function drawGraticule(layer) {
  const { w, h } = map.size;
  const view = map.viewState();
  const { lats, lons } = graticule(view.bounds, 6);
  const seg = [];
  for (const lat of lats) {
    const y = map.project(lat, view.center.lon).y;
    seg.push(`M0 ${y}H${w}`);
    const t = svg('text', { x: 7, y: y - 5, class: 'grat-txt' });
    t.textContent = formatLatLon(lat, 'lat');
    layer.appendChild(t);
  }
  for (const lon of lons) {
    const x = map.project(view.center.lat, lon).x;
    seg.push(`M${x} 0V${h}`);
    const t = svg('text', { x: x + 5, y: 15, class: 'grat-txt' });
    t.textContent = formatLatLon(lon, 'lon');
    layer.appendChild(t);
  }
  layer.insertBefore(svg('path', { d: seg.join(' '), class: 'grat' }), layer.firstChild);
}

/**
 * The shape a half-finished two-click draw currently describes.
 *
 * `end` defaults to the hovered point, which is what the rubber-band preview follows.
 * The second click passes its own position instead: a touchscreen produces two taps with
 * no pointermove between them, so there is no hover to fall back on and the shape would
 * silently fail to commit.
 */
S.draftShape = function (end) {
  const to = end || S.hover;
  // A polygon is drawn vertex by vertex rather than by dragging a corner, so its preview
  // comes from the accumulated list, not from a single anchor point.
  if (S.tool === 'poly' && S.poly.length) {
    const pts = S.poly.map(p => [p.lon, p.lat]);
    if (to) pts.push([to.lon, to.lat]);
    return pts.length >= 3 ? { kind: 'poly', rings: [pts] } : null;
  }
  if (!S.draft || !to) return null;
  if (S.tool === 'circle') {
    const r = haversine(S.draft, to) / 1000;
    return { kind: 'circle', lat: S.draft.lat, lon: S.draft.lon, radiusKm: Math.max(r, 0.5) };
  }
  if (S.tool === 'rect') {
    return {
      kind: 'bbox',
      bbox: [
        Math.min(S.draft.lon, to.lon), Math.min(S.draft.lat, to.lat),
        Math.max(S.draft.lon, to.lon), Math.max(S.draft.lat, to.lat),
      ],
    };
  }
  return null;
};

/* ============================================================== moving the AOI === */
/** Make an SVG shape drag the whole area of interest with it. */
function dragify(el, isDraft) {
  if (isDraft || !S.aoi) return el;
  el.addEventListener('pointerdown', ev => {
    if (ev.button !== 0 && ev.pointerType === 'mouse') return;
    // The map pans on pointerdown anywhere in its container. Without this the shape and
    // the basemap would move together, which looks like nothing moved at all.
    ev.stopPropagation();
    ev.preventDefault();
    const rect = map.el.getBoundingClientRect();
    const from = map.unproject(ev.clientX - rect.left, ev.clientY - rect.top);
    const origin = structuredClone(S.aoi);
    map.el.classList.add('mv-editing');

    const move = e2 => {
      const to = map.unproject(e2.clientX - rect.left, e2.clientY - rect.top);
      setAoi(translateAoi(origin, to.lat - from.lat, to.lon - from.lon), origin.name);
    };
    const up = () => {
      window.removeEventListener('pointermove', move);
      window.removeEventListener('pointerup', up);
      window.removeEventListener('pointercancel', up);
      map.el.classList.remove('mv-editing');
    };
    window.addEventListener('pointermove', move);
    window.addEventListener('pointerup', up);
    window.addEventListener('pointercancel', up);
  });
  return el;
}

const wrapLon = lon => ((lon + 540) % 360) - 180;
const clampLat = lat => Math.min(85, Math.max(-85, lat));

function translateAoi(aoi, dLat, dLon) {
  if (aoi.kind === 'circle') {
    return { ...aoi, lat: clampLat(aoi.lat + dLat), lon: wrapLon(aoi.lon + dLon) };
  }
  if (aoi.kind === 'poly') {
    const rings = aoi.rings.map(r => r.map(([lo, la]) => [wrapLon(lo + dLon), clampLat(la + dLat)]));
    const [w, sLat, e, n] = ringBounds(rings);
    return { ...aoi, rings, lat: (sLat + n) / 2, lon: (w + e) / 2 };
  }
  // Clamp the shift, not the corners: clamping corners at a pole would squash the box
  // instead of stopping it.
  const [w, s, e, n] = aoi.bbox;
  const dy = Math.max(Math.min(dLat, 85 - n), -85 - s);
  const bbox = [wrapLon(w + dLon), s + dy, wrapLon(e + dLon), n + dy];
  return { ...aoi, bbox, lat: (bbox[1] + bbox[3]) / 2, lon: (bbox[0] + bbox[2]) / 2 };
}

/* ====================================================================== tools ==== */
const TOOLS = ['pan', 'probe', 'place', 'circle', 'rect', 'poly', 'measure'];
const TOOL_HINT = {
  pan: 'Drag to pan, scroll to zoom. Nothing is drawn or selected.',
  probe: 'Click the forecast to read the predicted value at that pixel.',
  place: 'Click the map to drop a point. The radius slider sets the area.',
  circle: 'Click the centre, then click again to set the radius.',
  rect: 'Click one corner, then the opposite corner.',
  poly: 'Click each corner. Double-click or press Enter to close, Backspace to undo a point.',
  measure: 'Click to add points. Double-click or press Enter to finish, Esc to clear.',
};

function toolHint() {
  const base = TOOL_HINT[S.tool];
  return S.aoi && S.tool !== 'measure' ? `${base} Drag the outline to move it.` : base;
}

function setTool(tool) {
  S.tool = tool;
  S.draft = null;
  if (tool !== 'measure') { S.measuring = false; }
  if (tool !== 'poly') { S.poly = []; }
  TOOLS.forEach(t => $(`tool-${t}`).setAttribute('aria-pressed', String(t === tool)));
  $('toolhint').textContent = toolHint();
  $('measure-panel').hidden = tool !== 'measure';
  $('probe-panel').hidden = tool !== 'probe';
  $('map').classList.toggle('crosshair', tool !== 'pan');
  drawVectors();
}

function onMapClick(geo) {
  if (S.tool === 'pan') return;
  if (S.tool === 'probe') { probeAt(geo); return; }
  if (S.tool === 'poly') { S.poly.push(geo); drawVectors(); return; }
  if (S.tool === 'measure') {
    S.measuring = true;
    S.measure.push(geo);
    drawVectors();
    return;
  }
  if (S.tool === 'place') {
    setAoi({ kind: 'circle', lat: geo.lat, lon: geo.lon, radiusKm: S.radiusKm },
      `${geo.lat.toFixed(3)}, ${geo.lon.toFixed(3)}`);
    return;
  }
  if (!S.draft) { S.draft = geo; drawVectors(); return; }
  const shape = S.draftShape(geo);
  S.draft = null;
  if (shape) {
    const b = aoiBbox(shape);
    const name = shape.kind === 'circle'
      ? `${shape.lat.toFixed(3)}, ${shape.lon.toFixed(3)}`
      : 'Drawn area';
    setAoi({ ...shape, lat: (b[1] + b[3]) / 2, lon: (b[0] + b[2]) / 2 }, name);
  }
}

function closePolygon() {
  if (S.poly.length < 3) return;
  const rings = [S.poly.map(p => [p.lon, p.lat])];
  const [w, sLat, e, n] = ringBounds(rings);
  setAoi({ kind: 'poly', rings, lat: (sLat + n) / 2, lon: (w + e) / 2 }, 'Drawn polygon');
  S.poly = [];
  setTool('pan');
}

// Mirrors habforecast.aoi.MAX_ADHOC_SIDE_KM. Checked here as well as on the server so a
// user dragging a box across an ocean is told immediately, rather than after waiting for
// a job to start and fail.
const MAX_SIDE_KM = 80;

function aoiSideKm(aoi) {
  const [w, s, e, n] = aoiBbox(aoi);
  return Math.max(
    haversine({ lat: (s + n) / 2, lon: w }, { lat: (s + n) / 2, lon: e }),
    haversine({ lat: s, lon: (w + e) / 2 }, { lat: n, lon: (w + e) / 2 }),
  ) / 1000;
}

function setAoi(aoi, name) {
  S.aoi = aoi;
  S.aoi.name = name || S.aoi.name || 'Selected area';
  if (aoi.kind === 'circle') {
    S.radiusKm = aoi.radiusKm;
    $('radius').value = String(Math.round(aoi.radiusKm));
    $('radiusval').textContent = `${aoi.radiusKm.toFixed(0)} km`;
  }
  const side = aoiSideKm(S.aoi);
  const tooBig = side > MAX_SIDE_KM;
  $('go').disabled = tooBig;
  $('clear-aoi').disabled = false;
  $('btn-erase').disabled = false;
  // Describe the shape the user actually has, not the box around it: a polygon reported
  // by its bounding box reads as a rectangle, which is the one thing it is not.
  let detail;
  if (aoi.kind === 'circle') {
    detail = `${aoi.lat.toFixed(4)}, ${aoi.lon.toFixed(4)} · ${aoi.radiusKm.toFixed(0)} km radius`;
  } else if (aoi.kind === 'poly') {
    const verts = aoi.rings.reduce((a, r) => a + r.length, 0);
    const rings = aoi.rings.length > 1 ? `${aoi.rings.length} rings, ` : '';
    detail = `polygon · ${rings}${verts} points · centre ${aoi.lat.toFixed(4)}, ${aoi.lon.toFixed(4)}`;
  } else {
    detail = aoiBbox(aoi).map(v => v.toFixed(3)).join(', ');
  }
  $('aoi-label').innerHTML =
    `<b>${esc(S.aoi.name)}</b><br><span class="mono">${detail}</span>` +
    `<br><span class="mono" style="color:var(--faint)">${side.toFixed(1)} km across` +
    `${aoi.kind === 'poly' ? ' · results clipped to the shape' : ''}</span>`;
  $('aoi-warn').innerHTML = tooBig
    ? `<div class="note bad" style="margin-top:8px">That area is <strong>${side.toFixed(0)} km</strong>
       across; the limit is ${MAX_SIDE_KM} km. Zoom in and select a single lake or bay.</div>`
    : '';
  $('toolhint').textContent = toolHint();
  previewRoute();
  drawVectors();
  return S.aoi;
}

function clearAoi() {
  S.aoi = null;
  S.draft = null;
  S.poly = [];
  $('go').disabled = true;
  $('clear-aoi').disabled = true;
  $('btn-erase').disabled = true;
  $('aoi-label').textContent =
    'Search for a lake above, or pick a drawing tool and click the map. ' +
    'Zoom out to see the whole world.';
  $('aoi-warn').innerHTML = '';
  S.route = null;
  $('route-preview').hidden = true;
  $('model-field').hidden = true;
  $('toolhint').textContent = toolHint();
  drawVectors();
}

TOOLS.forEach(t => $(`tool-${t}`).addEventListener('click', () => setTool(t)));
$('clear-aoi').addEventListener('click', clearAoi);
$('btn-erase').addEventListener('click', clearAoi);

document.addEventListener('keydown', e => {
  if (e.target.tagName === 'INPUT' || e.metaKey || e.ctrlKey) return;
  if (e.key === 'Escape') {
    if (S.poly.length) { S.poly = []; }
    else if (S.measure.length || S.measuring) { S.measure = []; S.measuring = false; }
    else if (!S.draft) closeAllPops();
    S.draft = null;
    drawVectors();
    return;
  }
  if (e.key === 'Backspace' && S.poly.length) {
    e.preventDefault();
    S.poly.pop();
    drawVectors();
    return;
  }
  if (e.key === 'Delete' || e.key === 'Backspace') {
    if (S.aoi) { e.preventDefault(); clearAoi(); }
    return;
  }
  if (e.key === 'Enter') {
    if (S.tool === 'poly') { closePolygon(); return; }
    if (S.tool === 'measure') { S.measuring = false; drawVectors(); return; }
  }
  if (e.key === 's' || e.key === 'S') {
    e.preventDefault();
    return $('sites').hidden ? openSites() : closeSites();
  }
  const keys = { v: 'pan', i: 'probe', p: 'place', c: 'circle', r: 'rect', g: 'poly', m: 'measure' };
  if (keys[e.key.toLowerCase()]) setTool(keys[e.key.toLowerCase()]);
});
$('map').addEventListener('dblclick', () => {
  if (S.tool === 'measure') { S.measuring = false; drawVectors(); }
  if (S.tool === 'poly') closePolygon();
});

function updateMeasurePanel() {
  if ($('measure-panel').hidden) return;
  let total = 0;
  for (let i = 1; i < S.measure.length; i++) total += haversine(S.measure[i - 1], S.measure[i]);
  $('measure-total').textContent = S.measure.length > 1 ? formatDistance(total) : '—';
  $('measure-area').textContent = S.measure.length > 2 ? formatArea(ringArea(S.measure)) : '—';
  $('measure-pts').textContent = String(S.measure.length);
}
$('measure-clear').addEventListener('click', () => {
  S.measure = []; S.measuring = false; drawVectors();
});
$('measure-toaoi').addEventListener('click', () => {
  if (S.measure.length < 2) return;
  const lons = S.measure.map(p => p.lon), lats = S.measure.map(p => p.lat);
  setAoi({
    kind: 'bbox',
    bbox: [Math.min(...lons), Math.min(...lats), Math.max(...lons), Math.max(...lats)],
    lat: (Math.min(...lats) + Math.max(...lats)) / 2,
    lon: (Math.min(...lons) + Math.max(...lons)) / 2,
  }, 'Measured area');
  setTool('pan');
});

/* ============================================================== upload a shape ==== */
/* Parsed on the server: two of the three accepted formats are binary containers (a
   shapefile bundle and a KMZ are both zips), and a client-side parser for them would be
   either a large vendored library or a silent "unsupported format". */
$('btn-upload').addEventListener('click', () => $('geomfile').click());
$('geomfile').addEventListener('change', async e => {
  const file = e.target.files && e.target.files[0];
  e.target.value = '';                       // let the same file be picked twice
  if (!file) return;
  status(`Reading ${file.name}…`);
  try {
    const form = new FormData();
    form.append('file', file);
    const r = await fetch('/api/geometry', { method: 'POST', body: form });
    const doc = await r.json();
    if (!r.ok) throw new Error(doc.detail || r.statusText);
    const [w, s, e2, n] = doc.bbox;
    setAoi({ kind: 'poly', rings: doc.rings, lat: (s + n) / 2, lon: (w + e2) / 2 }, doc.name);
    map.fitBounds(doc.bbox, 0.3);
    setTool('pan');
    hideStatus();
    if (doc.too_large) {
      status(`That shape is ${doc.side_km} km across; the limit is ${MAX_SIDE_KM} km.`, true);
    }
  } catch (err) {
    status(err.message, true);
  }
});

/* ============================================================ inspect a pixel ==== */
/* The server sends the forecast twice: once as a picture, once as a quantised grid of
   the values behind it. Reading colours back out of the picture would be guessing --
   the ramp is 8-bit through five stops and is not invertible near its ends. */

/** base64 little-endian uint16 -> Uint16Array. Endianness is read explicitly rather than
 *  trusted to the platform: a big-endian client would otherwise report nonsense. */
function decodeGrid(b64) {
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  const view = new DataView(bytes.buffer);
  const out = new Uint16Array(bytes.length >> 1);
  for (let i = 0; i < out.length; i++) out[i] = view.getUint16(i * 2, true);
  return out;
}

/**
 * Concentration at a point, in mg/m³, read out of one quantised grid.
 * `undefined` means outside the forecast; `null` means inside it with no retrieval.
 */
function sampleGrid(grid, lat, lon) {
  const res = S.result;
  if (!res || !grid) return undefined;
  const [west, south, east, north] = res.bbox;
  const [rows, cols] = res.grid;
  if (lat > north || lat < south || lon < west || lon > east) return undefined;
  const row = Math.min(rows - 1, Math.floor((north - lat) / (north - south) * rows));
  const col = Math.min(cols - 1, Math.floor((lon - west) / (east - west) * cols));
  const code = grid[row * cols + col];
  if (code === 0) return null;
  const [lo, hi] = res.value_scale;
  return Math.pow(10, lo + ((code - 1) / 65534) * (hi - lo));
}

const gridValue = (lead, lat, lon) => sampleGrid(S.grids && S.grids[lead], lat, lon);

/** This pixel through time: the input window behind it, the model's leads ahead of it. */
function pixelSeries(geo) {
  const observed = Object.keys(S.history || {}).sort().map(day => ({
    date: day, mg: sampleGrid(S.history[day], geo.lat, geo.lon),
  }));
  const forecast = S.result.forecasts.map(f => ({
    date: f.valid_date, lead: f.lead_days,
    mg: gridValue(String(f.lead_days), geo.lat, geo.lon),
  }));
  return { observed, forecast };
}

/** The span the chart covers: first day actually retrieved, to the furthest lead. */
function probeWindow() {
  const days = Object.keys(S.history || {}).sort();
  return {
    start: days[0] || S.result.anchor_date,
    end: S.result.forecasts.at(-1).valid_date,
  };
}

/* One key for "which question is on screen". A click, a radius change or a new forecast
   all invalidate the answer, and a reply that arrives after the user has moved on has to
   be recognised as stale rather than painted over the new pixel. */
function probeKey() {
  if (!S.probe || !S.result) return '';
  return `${S.probe.lat.toFixed(4)},${S.probe.lon.toFixed(4)},${S.insituKm},${S.result.anchor_date}`;
}

function probeAt(geo) {
  S.probe = { lat: geo.lat, lon: geo.lon };
  $('probe-panel').hidden = false;
  $('probe-coord').textContent = `${formatLatLon(geo.lat, 'lat')}  ${formatLatLon(geo.lon, 'lon')}`;

  if (!S.result) {
    $('probe-value').textContent = '—';
    $('probe-unit').textContent = 'no forecast loaded';
    $('probe-rows').innerHTML = '';
    $('probe-chart').innerHTML = '';
    $('probe-situ').innerHTML = '';
    $('probe-series').hidden = true;
    $('probe-note').textContent =
      'Select an area and generate a forecast, then click it to read values.';
    drawVectors();
    return;
  }

  const readings = S.result.forecasts.map(f => ({
    lead: f.lead_days,
    date: f.valid_date,
    mg: gridValue(String(f.lead_days), geo.lat, geo.lon),
  }));
  const here = readings.find(r => r.lead === S.lead);
  const outside = !here || here.mg === undefined;

  $('probe-value').textContent = here && typeof here.mg === 'number' ? here.mg.toFixed(1) : '—';
  $('probe-unit').textContent = outside
    ? 'outside the forecast area'
    : here.mg === null
      ? 'no retrieval here — land, cloud or ice'
      : `mg/m³ chlorophyll-a at +${S.lead} d`;

  $('probe-rows').innerHTML = readings.map(r => `
    <tr class="${r.lead === S.lead ? 'on' : ''}">
      <td class="mono">+${r.lead} d</td>
      <td class="mono" style="color:var(--faint)">${esc(r.date)}</td>
      <td class="n mono">${typeof r.mg === 'number' ? r.mg.toFixed(1) : (r.mg === null ? '—' : '·')}</td>
    </tr>`).join('');

  $('probe-note').textContent = outside ? '' :
    'Read from the model output, not from the picture. A 300 m pixel is an area average: ' +
    'a shoreline scum can be far denser than the number here.';

  $('probe-series').hidden = outside;
  if (outside) {
    $('probe-chart').innerHTML = '';
    $('probe-situ').innerHTML = '';
  } else {
    renderSeries();
    loadInsitu();
  }
  drawVectors();
}

/* ------------------------------------------------- the pixel's time series ----- */
/* A fortnight of retrieval, a week of forecast, and whatever anyone has measured in that
   water. The third series is the only one that did not come out of this system, which is
   why it is fetched on every click rather than hidden behind an option. */

function renderSeries() {
  const host = $('probe-chart');
  host.innerHTML = '';
  if (!S.result || !S.probe) return;
  const { observed, forecast } = pixelSeries(S.probe);
  host.appendChild(timeSeries({
    observed,
    forecast,
    anchor: S.result.anchor_date,
    insitu: S.insitu && S.insitu.key === probeKey() ? S.insitu.data : null,
    width: 302,
    height: 168,
  }));
}

let insituAbort = null;
async function loadInsitu() {
  const key = probeKey();
  if (!key) return;
  if (S.insitu && S.insitu.key === key) { renderInsitu(); return; }

  if (insituAbort) insituAbort.abort();
  insituAbort = new AbortController();
  $('probe-situ').innerHTML = '<p class="ts-empty">Looking for in-situ measurements…</p>';
  const { start, end } = probeWindow();
  const q = new URLSearchParams({
    lat: S.probe.lat.toFixed(5), lon: S.probe.lon.toFixed(5),
    start, end, radius_km: String(S.insituKm),
  });
  let data;
  try {
    const r = await fetch(`/api/insitu?${q}`, { signal: insituAbort.signal });
    data = await r.json();
    if (!r.ok) throw new Error(data.detail || r.statusText);
  } catch (err) {
    if (err.name === 'AbortError') return;
    data = { available: false, note: `Could not reach the validation index: ${err.message}` };
  }
  if (probeKey() !== key) return;   // the user has clicked elsewhere since
  S.insitu = { key, data };
  renderSeries();
  renderInsitu();
}

function renderInsitu() {
  const box = $('probe-situ');
  const st = S.insitu && S.insitu.key === probeKey() ? S.insitu.data : null;
  if (!st) { box.innerHTML = ''; return; }
  if (!st.available) { box.innerHTML = `<p class="ts-empty">${esc(st.note)}</p>`; return; }

  const parts = [`<p class="ts-note">${esc(st.note)}</p>`];
  if (st.total) {
    parts.push(`<p class="ts-note">Nearest sample ${st.nearest_km} km away · sources:
      ${esc(st.sources.join(', '))}.</p>`);
    parts.push(`<table class="probe-table"><tbody>${st.variables.map(v => `
      <tr><td>${v.variable === 'microcystin' ? 'Microcystin' : 'Chlorophyll-a'}</td>
          <td class="mono" style="color:var(--faint)">${esc(v.first)} – ${esc(v.last)}</td>
          <td class="n mono">${v.n}</td></tr>`).join('')}</tbody></table>`);
  }

  // Microcystin is deliberately absent from the plot: it is a different quantity in
  // different units, and putting it on a chlorophyll axis would be the exact confusion
  // this project's own disclaimer warns about. It is reported here instead, because it is
  // the measurement a public-health decision actually turns on.
  const tox = [...(st.window || []), ...(st.seasonal || [])]
    .filter(p => p.variable === 'microcystin')
    .sort((a, b) => (a.date < b.date ? 1 : -1))
    .slice(0, 6);
  if (tox.length) {
    parts.push(`<p class="ts-note"><strong>Microcystin measured nearby</strong> — a toxin
      concentration, not chlorophyll, so it is listed rather than plotted:</p>`);
    parts.push(`<table class="probe-table"><tbody>${tox.map(p => `
      <tr class="${p.exceeds_who ? 'over' : ''}">
        <td class="mono" style="color:var(--faint)">${esc(p.date)}</td>
        <td>${esc(p.site || p.source)}</td>
        <td class="n mono">${p.flag === 'nd' ? 'n.d.' : `${p.value} µg/L`}</td>
      </tr>`).join('')}</tbody></table>`);
    if (tox.some(p => p.exceeds_who)) {
      parts.push(`<p class="ts-note">Highlighted rows exceed the WHO 8 µg/L recreational
        guideline.</p>`);
    }
  }
  box.innerHTML = parts.join('');
}

$('probe-km').addEventListener('change', e => {
  S.insituKm = +e.target.value;
  S.insitu = null;
  if (S.probe && S.result) loadInsitu();
});

$('probe-close').addEventListener('click', () => {
  $('probe-panel').hidden = true;
  S.probe = null;
  drawVectors();
});

/* ==================================================================== readout ==== */
function updateReadout() {
  const g = S.hover || map.center;
  $('coords').textContent = `${g.lat.toFixed(4)}, ${g.lon.toFixed(4)}`;
  $('zoomval').textContent = map.zoom.toFixed(1);
  // Scale bar: the widest "nice" distance that fits in 120 px.
  const mpp = map.metresPerPixel();
  const targets = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1e3, 2e3, 5e3, 1e4, 2e4, 5e4, 1e5, 2e5, 5e5, 1e6, 2e6];
  let pick = targets[0];
  for (const t of targets) if (t / mpp <= 120) pick = t;
  $('scaleline').style.width = `${Math.round(pick / mpp)}px`;
  $('scaletext').textContent = formatDistance(pick);
}

/* ================================================================== popovers ==== */
const POPS = ['layers', 'info'];
function closeAllPops(except) {
  POPS.forEach(p => { if (p !== except) { $(`pop-${p}`).hidden = true; $(`btn-${p}`).setAttribute('aria-pressed', 'false'); } });
}
POPS.forEach(p => {
  $(`btn-${p}`).addEventListener('click', () => {
    const open = $(`pop-${p}`).hidden;
    closeAllPops(open ? p : null);
    $(`pop-${p}`).hidden = !open;
    $(`btn-${p}`).setAttribute('aria-pressed', String(open));
    if (open) positionPop($(`pop-${p}`), $(`btn-${p}`));
  });
  $(`pop-${p}`).querySelector('.pop-close').addEventListener('click', () => {
    $(`pop-${p}`).hidden = true;
    $(`btn-${p}`).setAttribute('aria-pressed', 'false');
  });
});
function positionPop(pop, btn) {
  // Below 760 px the stylesheet docks popovers above the tool rail; setting inline
  // coordinates here would override it and put the panel back under the rail.
  if (window.matchMedia('(max-width: 760px)').matches) {
    pop.style.top = pop.style.left = '';
    return;
  }
  const r = btn.getBoundingClientRect();
  pop.style.top = `${Math.min(r.top, window.innerHeight - 380)}px`;
  pop.style.left = `${Math.max(12, r.left - pop.offsetWidth - 10)}px`;
}

document.querySelectorAll('.tile-opt').forEach(b =>
  b.addEventListener('click', () => map.setBasemap(b.dataset.base)));

$('opt-grat').addEventListener('change', e => { S.showGrid = e.target.checked; drawVectors(); });

$('opacity').addEventListener('input', e => {
  S.opacity = e.target.value / 100;
  $('opacityval').textContent = `${e.target.value}%`;
  map.setOverlayOpacity(S.opacity);
});
$('radius').addEventListener('input', e => {
  S.radiusKm = +e.target.value;
  $('radiusval').textContent = `${S.radiusKm} km`;
  if (S.aoi && S.aoi.kind === 'circle') setAoi({ ...S.aoi, radiusKm: S.radiusKm }, S.aoi.name);
});

$('btn-zin').addEventListener('click', () => map.zoomBy(1));
$('btn-zout').addEventListener('click', () => map.zoomBy(-1));

/* ===================================================================== theme ==== */
const THEMES = ['auto', 'light', 'dark'];
let themeIdx = THEMES.indexOf(localStorage.getItem('hab-theme') || 'auto');
if (themeIdx < 0) themeIdx = 0;
function applyTheme() {
  const t = THEMES[themeIdx];
  if (t === 'auto') {
    // Auto means "match the map", not only "match the OS". Satellite and ocean imagery
    // are dark and saturated; light chrome over them washes out and competes with the
    // data. A basemap of streets or terrain is light, so there the OS preference wins.
    if (map.isDark) document.documentElement.setAttribute('data-theme', 'dark');
    else document.documentElement.removeAttribute('data-theme');
  } else {
    document.documentElement.setAttribute('data-theme', t);
  }
  $('btn-theme').title = `Theme: ${t}`;
  $('btn-theme').setAttribute('aria-label', `Theme: ${t}. Click to change.`);
  try { localStorage.setItem('hab-theme', t); } catch { /* private mode */ }
}
$('btn-theme').addEventListener('click', () => { themeIdx = (themeIdx + 1) % 3; applyTheme(); });
applyTheme();

/* ================================================================= geolocate ==== */
$('btn-locate').addEventListener('click', () => {
  if (!navigator.geolocation) return status('This browser will not share a location.', true);
  status('Locating…');
  navigator.geolocation.getCurrentPosition(
    pos => {
      const { latitude: lat, longitude: lon } = pos.coords;
      map.setView(lat, lon, 10);
      setAoi({ kind: 'circle', lat, lon, radiusKm: S.radiusKm }, 'My location');
      hideStatus();
    },
    err => status(`Could not get your location (${err.message}).`, true),
    { timeout: 10000 },
  );
});

/* ==================================================================== search ==== */
let searchTimer, hits = [], hitIdx = -1;
$('q').addEventListener('input', () => {
  clearTimeout(searchTimer);
  const q = $('q').value.trim();
  if (!q) { $('hits').replaceChildren(); hits = []; return; }
  searchTimer = setTimeout(() => runSearch(q), 170);
});
$('q').addEventListener('keydown', e => {
  if (!hits.length) { if (e.key === 'Enter') runSearch($('q').value.trim()); return; }
  if (e.key === 'ArrowDown') { e.preventDefault(); hitIdx = (hitIdx + 1) % hits.length; paintHits(); }
  if (e.key === 'ArrowUp') { e.preventDefault(); hitIdx = (hitIdx - 1 + hits.length) % hits.length; paintHits(); }
  if (e.key === 'Enter') { e.preventDefault(); pick(hits[Math.max(hitIdx, 0)]); }
  if (e.key === 'Escape') { $('hits').replaceChildren(); hits = []; }
});
document.addEventListener('click', e => { if (!e.target.closest('.search')) { $('hits').replaceChildren(); hits = []; } });

async function runSearch(q) {
  if (!q) return;
  try {
    const r = await fetch(`/api/places?q=${encodeURIComponent(q)}`);
    hits = r.ok ? await r.json() : [];
  } catch { hits = []; }
  hitIdx = hits.length ? 0 : -1;
  paintHits();
}

function paintHits() {
  const box = $('hits');
  if (!hits.length) {
    box.innerHTML = '<div class="hit-empty">No match. Enter coordinates as ' +
      '<span class="mono">lat, lon</span>, or click the map.</div>';
    return;
  }
  box.innerHTML = hits.map((p, i) =>
    `<button class="hit${i === hitIdx ? ' sel' : ''}" data-i="${i}">${esc(p.name)}` +
    `<small>${esc([p.country, p.note].filter(Boolean).join(' · '))}` +
    `${p.country || p.note ? ' · ' : ''}<span class="mono">${p.lat.toFixed(3)}, ${p.lon.toFixed(3)}</span>` +
    `</small></button>`).join('');
  box.querySelectorAll('.hit').forEach(b => b.addEventListener('click', () => pick(hits[+b.dataset.i])));
}

function pick(p) {
  if (!p) return;
  $('q').value = p.name;
  $('hits').replaceChildren();
  hits = [];
  // A geocoder that knows the feature's extent gives a much better frame than a guessed
  // zoom -- Lake Winnipeg and a village should not arrive at the same scale. The
  // selection stays a circle: the extent frames the view, it does not become the area,
  // because a lake's bounding box is mostly land.
  if (p.bbox && p.bbox.length === 4) {
    const wide = haversine({ lat: p.lat, lon: p.bbox[0] }, { lat: p.lat, lon: p.bbox[2] }) / 1000;
    map.fitBounds(p.bbox, 0.3);
    if (wide > MAX_SIDE_KM) map.setView(p.lat, p.lon, Math.max(map.zoom, 9));
  } else {
    map.setView(p.lat, p.lon, Math.max(map.zoom, 10));
  }
  setAoi({ kind: 'circle', lat: p.lat, lon: p.lon, radiusKm: S.radiusKm }, p.name);
}


/* ========================================================== validation sites ==== */
/* Where a forecast can be checked against somebody's laboratory. The list is the honest
   answer to "how would I know if this is right", so it leads with the lakes the model was
   trained and evaluated on and then offers water it has never seen -- labelled as such,
   because a forecast on unfamiliar water is the case where field data matters most.
   Selecting a site frames the map on it and arms the forecast button; it does not run
   anything, because the run is the user's decision and costs a minute of somebody's
   bandwidth. */

const SITES = { all: [], loaded: false, selected: null };

function siteBadge(s) {
  if (s.kind === 'modelled') {
    return `<span class="site-tag on" title="In the registry this model was trained and evaluated on">modelled · ${esc(s.split || 'held out')}</span>`;
  }
  return '<span class="site-tag" title="The model has never seen this water body">unseen</span>';
}

function renderSites() {
  const q = $('sites-q').value.trim().toLowerCase();
  const list = SITES.all.filter(s => !q || s.name.toLowerCase().includes(q)
    || (s.region || '').toLowerCase().includes(q));
  $('sites-list').innerHTML = list.map((s, i) => `
    <button class="site" data-i="${i}" role="option" aria-selected="${SITES.selected === s.id}">
      <div class="site-top">
        <span class="site-name">${esc(s.name)}</span>${siteBadge(s)}
      </div>
      <div class="site-meta">
        <span class="mono">${s.chlorophyll_a.toLocaleString()}</span> chlorophyll${
          s.microcystin ? ` · <span class="mono">${s.microcystin.toLocaleString()}</span> microcystin` : ''}
      </div>
      <div class="site-meta faint">${esc(s.first || '')} – ${esc(s.last || '')} · ${s.span_km} km across</div>
    </button>`).join('') || '<p class="sites-note">Nothing matches that.</p>';
  $('sites-list').querySelectorAll('.site').forEach(b =>
    b.addEventListener('click', () => pickSite(list[+b.dataset.i])));
  $('sites-note').textContent =
    `${list.length} of ${SITES.all.length} water bodies · sorted by how much has been measured`;
}

function pickSite(s) {
  if (!s) return;
  SITES.selected = s.id;
  const [west, south, east, north] = s.forecast_bbox;
  map.fitBounds(s.bbox, 0.35);
  setAoi({
    kind: 'bbox',
    bbox: [west, south, east, north],
    lat: (south + north) / 2,
    lon: (west + east) / 2,
  }, s.name);

  // The field data is the reason this list exists, so it is stated on selection rather
  // than left for the user to discover by clicking a pixel.
  const sources = (s.sources || []).join(', ') || 'none';
  const lines = [
    `<b>${esc(s.name)}</b> — <span class="mono">${s.chlorophyll_a.toLocaleString()}</span>
     chlorophyll-a and <span class="mono">${s.microcystin.toLocaleString()}</span> microcystin
     measurements, ${esc(s.first || '?')} to ${esc(s.last || '?')}.`,
    `Source: ${esc(sources)}.`,
  ];
  if (s.kind === 'modelled') {
    lines.push(`This lake is in the training registry (<span class="mono">${esc(s.split)}</span>
      split). Skill here is measured — see the <a href="/dashboard">results</a>.`);
  } else {
    lines.push('The model has never seen this water body. Treat the forecast as an extrapolation.');
  }
  if (s.clipped_to_limit) {
    lines.push(`It is ${s.span_km} km across, past the ${MAX_SIDE_KM} km limit for one
      forecast, so the selected area is a ${s.forecast_span_km} km box at its centre.
      Drag or redraw it to look somewhere else in this water body.`);
  }
  $('aoi-warn').innerHTML = `<div class="note">${lines.join(' ')}</div>`;
  renderSites();
  if (window.matchMedia('(max-width: 900px)').matches) closeSites();
}

async function loadSites() {
  if (SITES.loaded) return;
  SITES.loaded = true;
  try {
    const r = await fetch('/api/validation-sites');
    const doc = await r.json();
    if (!r.ok) throw new Error(doc.detail || r.statusText);
    SITES.all = doc.sites || [];
    if (!SITES.all.length) {
      $('sites-note').textContent =
        'No in-situ index is installed, so there is nothing to compare against. '
        + 'Build one with scripts/build_insitu_index.py.';
      return;
    }
    renderSites();
  } catch (err) {
    SITES.loaded = false;   // a failed load must be retryable
    $('sites-note').textContent = `Could not load the list: ${err.message}`;
  }
}

function openSites() {
  $('sites').hidden = false;
  $('sites-toggle').setAttribute('aria-expanded', 'true');
  document.body.classList.add('sites-open');
  loadSites();
}
function closeSites() {
  $('sites').hidden = true;
  $('sites-toggle').setAttribute('aria-expanded', 'false');
  document.body.classList.remove('sites-open');
}

$('sites-toggle').addEventListener('click', () => ($('sites').hidden ? openSites() : closeSites()));
$('sites-close').addEventListener('click', closeSites);
$('sites-q').addEventListener('input', renderSites);

/* ================================================================== forecast ==== */
function status(text, isError = false, progress = null) {
  const pill = $('pill');
  pill.hidden = false;
  pill.classList.toggle('is-error', isError);
  $('pill-spin').hidden = isError || progress === null;
  $('pill-bar').hidden = progress === null;
  $('pill-text').textContent = text;
  if (progress !== null) $('pill-fill').style.width = `${Math.round(progress * 100)}%`;
}
function hideStatus() { $('pill').hidden = true; }

$('go').addEventListener('click', runForecast);

async function runForecast() {
  if (!S.aoi) return;
  $('go').disabled = true;
  $('drawer').hidden = true;
  map.removeOverlay();
  S.result = null;
  S.grids = null;
  S.history = null;
  S.insitu = null;
  S.probe = null;
  status('Submitting…', false, 0.02);
  try {
    const body = {
      lat: S.aoi.lat, lon: S.aoi.lon, radius_km: S.radiusKm,
      name: S.aoi.name || 'Selected area',
    };
    if (S.aoi.kind === 'bbox') body.bbox = S.aoi.bbox;
    if (S.aoi.kind === 'poly') body.rings = S.aoi.rings;
    body.source = $('source').value;
    body.model = $('model').value || 'auto';
    if ($('anchor').value) body.anchor = $('anchor').value;
    const r = await fetch('/api/forecast/jobs', {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
    });
    const job = await r.json();
    if (!r.ok) throw new Error(job.detail || r.statusText);
    await pollJob(job.id);
  } catch (err) {
    status(err.message, true);
  } finally {
    $('go').disabled = !S.aoi;
  }
}

async function pollJob(id) {
  for (;;) {
    await new Promise(res => setTimeout(res, 1200));
    const r = await fetch(`/api/forecast/jobs/${id}`);
    if (!r.ok) throw new Error('Lost track of that job — it may have expired.');
    const j = await r.json();
    status(`${j.stage} · ${j.elapsed_s}s`, false, j.progress);
    if (j.status === 'error') { status(j.error, true); return; }
    if (j.status === 'done') { hideStatus(); S.jobId = id; renderResult(j.result); return; }
  }
}

const fmt = (v, d = 1) => (v == null ? '—' : Number(v).toFixed(d));

function renderResult(res) {
  S.result = res;
  S.grids = Object.fromEntries(
    Object.entries(res.values || {}).map(([k, v]) => [k, decodeGrid(v)]));
  S.history = Object.fromEntries(
    Object.entries(res.history || {}).map(([k, v]) => [k, decodeGrid(v)]));
  S.insitu = null;
  const leads = res.forecasts.map(f => f.lead_days);
  S.lead = leads.at(-1);

  $('drawer').hidden = false;
  $('drawer').classList.remove('collapsed');
  $('drawer-title').textContent = res.lake_name;

  $('leadtabs').innerHTML = leads.map(l =>
    `<button data-l="${l}" role="radio" aria-checked="${l === S.lead}" class="${l === S.lead ? 'on' : ''}">+${l} d</button>`
  ).join('');
  $('leadtabs').querySelectorAll('button').forEach(b =>
    b.addEventListener('click', () => selectLead(+b.dataset.l)));

  const peak = res.forecasts.reduce((a, f) => ((f.p90_chl_mg_m3 ?? 0) > (a.p90_chl_mg_m3 ?? 0) ? f : a), res.forecasts[0]);
  $('stats').innerHTML = `
    <div class="stat"><h4>Peak p90</h4><div class="v">${fmt(peak.p90_chl_mg_m3)}</div>
      <div class="u">mg/m³ at +${peak.lead_days} d</div></div>
    <div class="stat"><h4>Bloom &gt;20</h4><div class="v">${fmt(peak.bloom_area_km2['20'], 0)}</div>
      <div class="u">km² at +${peak.lead_days} d</div></div>
    <div class="stat"><h4>Water mapped</h4><div class="v">${fmt(res.water_area_km2, 0)}</div>
      <div class="u">km² · ${Math.round(res.water_confidence * 100)}% coverage</div></div>
    <div class="stat"><h4>Days observed</h4><div class="v">${res.observed_days.length}<span class="u"> /14</span></div>
      <div class="u">${res.scenes_used}/${res.scenes_found} scenes used</div></div>`;

  $('rows').innerHTML = res.forecasts.map(f => `
    <tr data-l="${f.lead_days}">
      <td class="n">+${f.lead_days} d</td><td class="mono">${f.valid_date}</td>
      <td class="n">${fmt(f.median_chl_mg_m3)}</td><td class="n">${fmt(f.p90_chl_mg_m3)}</td>
      <td class="n">${fmt(f.bloom_area_km2['20'], 0)}</td>
    </tr>`).join('');
  $('rows').querySelectorAll('tr').forEach(tr =>
    tr.addEventListener('click', () => selectLead(+tr.dataset.l)));

  $('routeinfo').innerHTML = routeHtml(res.route, false);
  $('caveats').innerHTML = (res.caveats || []).length
    ? `<div class="note"><strong>Read before using these numbers.</strong>
        <ul>${res.caveats.map(c => `<li>${esc(c)}</li>`).join('')}</ul></div>` : '';

  const age = res.anchor_age_days;
  const fresh = age != null && age <= 7;
  $('resfoot').innerHTML =
    `<span class="srcpill ${fresh ? 'fresh' : 'stale'}">${esc(res.source_label || 'source')}</span>
     Anchored on <strong>${res.anchor_date}</strong>` +
    (age != null ? `, <strong>${age} day${age === 1 ? '' : 's'} old</strong>` : '') +
    `. ${res.clipped ? 'Clipped to the drawn shape. ' : ''}Model
     <code class="mono">${esc(res.model_version)}</code>, grid
     ${res.grid[1]}×${res.grid[0]} at ${gridLabel(res)}.
     <a href="/dashboard">How well does this model actually do?</a>`;

  map.fitBounds(res.bbox, 0.35);
  selectLead(S.lead);
  $('btn-download').disabled = false;
  $('btn-bulletin').disabled = !S.jobId;
  // Drawing is finished. Reading the result is what anyone does next with it, and the
  // inspect tool is the only one that is useless before this moment.
  setTool('probe');
}

function selectLead(lead) {
  if (!S.result) return;
  S.lead = lead;
  $('leadtabs').querySelectorAll('button').forEach(b => {
    const on = +b.dataset.l === lead;
    b.classList.toggle('on', on);
    b.setAttribute('aria-checked', String(on));
  });
  $('rows').querySelectorAll('tr').forEach(tr => tr.classList.toggle('on', +tr.dataset.l === lead));
  map.addImageOverlay(S.result.bbox, S.result.images[String(lead)], { opacity: S.opacity });
  if (S.probe) probeAt(S.probe);   // keep the readout on the lead being displayed
}

/* ==================================================== collapsing and moving panels */
/* Both docked panels behave the same way: their header collapses them, and dragging that
   header lifts them out of the dock to wherever the user wants them. Undocking is a real
   mode change rather than a CSS nicety -- a floating panel is positioned from the
   viewport, so it has to leave the flex column that otherwise stacks the two. */

function toggleCollapse(panel, head) {
  panel.classList.toggle('collapsed');
  head.setAttribute('aria-expanded', String(!panel.classList.contains('collapsed')));
}

function undock(panel, left, top) {
  panel.classList.add('floating');
  const w = panel.offsetWidth, h = panel.offsetHeight;
  panel.style.left = `${Math.max(4, Math.min(left, window.innerWidth - w - 4))}px`;
  const y = Math.max(4, Math.min(top, window.innerHeight - Math.min(h, 120) - 4));
  panel.style.top = `${y}px`;
  // The CSS cap is the whole viewport, which only fits a panel parked at the very top.
  // Anywhere lower, its foot -- and the Generate button in it -- ran off the screen.
  // Cap it to the room below where it now sits; the body scrolls inside that.
  panel.style.maxHeight = `${window.innerHeight - y - 8}px`;
  panel.querySelector('.panel-redock').hidden = false;
}

function redock(panel) {
  panel.classList.remove('floating');
  panel.style.left = panel.style.top = panel.style.maxHeight = '';
  panel.querySelector('.panel-redock').hidden = true;
}

function makeMovable(panelId, headId) {
  const panel = $(panelId), head = $(headId);
  head.addEventListener('pointerdown', ev => {
    if (ev.target.closest('button')) return;      // the buttons in the header are not a handle
    if (ev.button !== 0 && ev.pointerType === 'mouse') return;
    const rect = panel.getBoundingClientRect();
    const dx = ev.clientX - rect.left, dy = ev.clientY - rect.top;
    let moved = 0;
    const move = e2 => {
      moved += Math.abs(e2.movementX || 0) + Math.abs(e2.movementY || 0);
      // A few pixels of slop keeps a click from turning into a one-pixel drag, which
      // would undock the panel every time somebody tried to collapse it.
      if (moved < 5) return;
      ev.target.setPointerCapture?.(ev.pointerId);
      undock(panel, e2.clientX - dx, e2.clientY - dy);
    };
    const up = () => {
      window.removeEventListener('pointermove', move);
      window.removeEventListener('pointerup', up);
    };
    window.addEventListener('pointermove', move);
    window.addEventListener('pointerup', up);
  });
  head.addEventListener('click', e => {
    if (e.target.closest('button')) return;
    toggleCollapse(panel, head);
  });
  head.addEventListener('keydown', e => {
    if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); toggleCollapse(panel, head); }
  });
  panel.querySelector('.panel-redock').addEventListener('click', () => redock(panel));
}

makeMovable('action', 'action-head');
makeMovable('drawer', 'drawer-head');

// Anything dragged off-screen by a window resize comes back rather than being lost, and
// every floating panel's height cap follows the new window size. undock() does the
// clamping, so re-running it in place is all a resize needs.
window.addEventListener('resize', () => {
  document.querySelectorAll('.floating').forEach(p => {
    const r = p.getBoundingClientRect();
    undock(p, r.left, r.top);
  });
});

/* ==================================================================== export ==== */
$('btn-download').addEventListener('click', async () => {
  if (!S.result) return;
  const res = S.result;
  const lead = res.forecasts.find(f => f.lead_days === S.lead);
  const probeMg = S.probe ? gridValue(String(S.lead), S.probe.lat, S.probe.lon) : undefined;
  $('btn-download').disabled = true;
  status('Composing the map sheet…');
  try {
    await downloadSheet(map, {
      title: `${res.lake_name} — chlorophyll-a forecast +${S.lead} d`,
      subtitle: `Valid ${lead ? lead.valid_date : '—'} · anchored on ${res.anchor_date} · ` +
                `${res.grid[1]}×${res.grid[0]} grid at 300 m`,
      overlay: { bbox: res.bbox, src: res.images[String(S.lead)], opacity: S.opacity },
      aoi: S.aoi,
      probe: S.probe && typeof probeMg === 'number'
        ? { ...S.probe, label: `${probeMg.toFixed(1)} mg/m³` }
        : null,
      source: 'Sentinel-3 OLCI L2 WFR via Microsoft Planetary Computer · model ' +
              `${res.model_version} · sheet generated ${new Date().toISOString().slice(0, 10)}`,
    }, `bloom_${res.lake_name.replace(/\W+/g, '_')}_${res.anchor_date}_plus${S.lead}d.png`);
    hideStatus();
  } catch (err) {
    status(`Could not build the sheet: ${err.message}`, true);
  } finally {
    $('btn-download').disabled = false;
  }
});

/* The bulletin is drawn server-side from the forecast object (full-resolution fields, the
   input history, the route and its caveats), so it is a download of the job's PDF rather
   than something composed here. It takes a few seconds -- the wind outlook is fetched then. */
$('btn-bulletin').addEventListener('click', async () => {
  if (!S.result || !S.jobId) return;
  $('btn-bulletin').disabled = true;
  status('Drawing the bulletin (imagery, forecast, wind & mixing)…');
  try {
    const r = await fetch(`/api/forecast/jobs/${S.jobId}/bulletin.pdf`);
    if (!r.ok) {
      let detail = r.statusText;
      try { detail = (await r.json()).detail || detail; } catch { /* not JSON */ }
      throw new Error(detail);
    }
    const blob = await r.blob();
    const name = (/filename="([^"]+)"/.exec(r.headers.get('Content-Disposition') || '') || [])[1]
      || `bloom_watch_bulletin_${S.result.anchor_date}.pdf`;
    const url = URL.createObjectURL(blob);
    const a = Object.assign(document.createElement('a'), { href: url, download: name });
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 10000);
    hideStatus();
  } catch (err) {
    status(`Could not build the bulletin: ${err.message}`, true);
  } finally {
    $('btn-bulletin').disabled = false;
  }
});

/* Open on the whole world with nothing selected, in the tool that only explores. Any
   other default would either pre-select somebody else's lake or arm a drawing tool
   before the user has decided where to look. */
setTool('pan');
updateReadout();

// --- what will be used, and why ------------------------------------------------------
// The router decides two things a user cannot see from the map: which satellite product
// covers this water, and which checkpoint was trained on water like it. Both are settled
// before anything is fetched, so both can be shown before anything is fetched -- which is
// the point, since one of the possible answers is "no validated model exists here".
//
// The preview is a separate, cheap endpoint (a few milliseconds, no imagery, no model)
// calling the same `plan` the forecast calls, so the two cannot disagree.

const WATER_LABEL = {
  inland: 'Inland water',
  coastal: 'Coastal water',
  'open-ocean': 'Open ocean',
  unknown: 'Unclassified',
};

function gridLabel(res) {
  const m = res?.route?.sources?.[0]?.resolution_m;
  if (!m) return '300 m';
  return m >= 1000 ? `${(m / 1000).toFixed(0)} km` : `${m.toFixed(0)} m`;
}

function skillLine(model) {
  const s = model?.skill_vs_persistence;
  if (!s) return '';
  const cells = Object.keys(s).sort((a, b) => a - b)
    .map(k => `+${k}d ${s[k] >= 0 ? '+' : ''}${Number(s[k]).toFixed(3)}`).join('   ');
  return `<div class="skill">skill vs persistence, held-out lakes: ${cells}</div>`;
}

function routeHtml(route, preview) {
  if (!route) return '';
  const w = route.water || {};
  const src = route.sources?.[0];
  const model = route.model;
  const cls = model ? (w.setting === 'inland' ? '' : ' marine') : ' baseline';
  const modelText = model
    ? `${esc(model.label)}`
    : `Persistence baseline — no validated model`;
  const modelSub = model
    ? `trained on ${model.train_bodies} ${esc(model.train_domain)}` +
      (model.beats_climatology ? ', beats seasonal climatology' : ', does not beat climatology')
    : 'the last observed field, carried forward';
  const why = [...(route.reasons || [])];
  return `
    <div class="rline">
      <span class="wclass${cls}">${esc(WATER_LABEL[w.setting] || w.setting || 'area')}</span>
      <span class="rsub">${w.coast_distance_km != null
        ? `${Math.round(w.coast_distance_km)} km from shore · ` : ''}${w.side_km ?? '?'} km across</span>
    </div>
    <div class="rline"><span class="rkey">Imagery</span>
      <span class="rval">${esc(src?.label || '—')}</span>
      <span class="rsub">${esc(src?.record || '')}</span></div>
    <div class="rline"><span class="rkey">Model</span>
      <span class="rval">${modelText}</span>
      <span class="rsub">${modelSub}</span></div>
    ${model ? skillLine(model) : ''}
    ${why.length ? `<details><summary>Why this combination</summary>
      <ul>${why.map(r => `<li>${esc(r)}</li>`).join('')}</ul></details>` : ''}
    ${preview && (route.caveats || []).length
      ? `<details><summary>${route.caveats.length} thing${route.caveats.length === 1 ? '' : 's'} to know first</summary>
         <ul>${route.caveats.map(c => `<li>${esc(c)}</li>`).join('')}</ul></details>` : ''}`;
}

function fillModelChoices(route) {
  const alts = route?.alternatives?.filter(a => a.servable) || [];
  const field = $('model-field');
  const sel = $('model');
  if (!alts.length) {
    // Nothing to choose between is not a choice. Keep the request explicit all the same.
    sel.innerHTML = '<option value="auto">Automatic</option>';
    field.hidden = true;
    return;
  }
  const auto = route.model
    ? `Automatic — ${route.model.label}`
    : 'Automatic — persistence baseline';
  const keep = sel.value;
  sel.innerHTML = [`<option value="auto">${esc(auto)}</option>`]
    .concat(alts.map(a => `<option value="${esc(a.key)}">${esc(a.label)}</option>`)).join('');
  sel.value = alts.some(a => a.key === keep) ? keep : 'auto';
  field.hidden = false;
  const note = () => {
    const picked = alts.find(a => a.key === sel.value);
    $('model-note').textContent = picked ? picked.why : '';
  };
  sel.onchange = note;
  note();
}

let routeToken = 0;
async function previewRoute() {
  if (!S.aoi) return;
  const [west, south, east, north] = aoiBbox(S.aoi);
  const mine = ++routeToken;
  try {
    const r = await fetch(`/api/route?west=${west}&south=${south}&east=${east}&north=${north}`);
    if (!r.ok) throw new Error(await r.text());
    const route = await r.json();
    if (mine !== routeToken) return;  // a newer shape was drawn while this was in flight
    S.route = route;
    $('route-preview').innerHTML = routeHtml(route, true);
    $('route-preview').hidden = false;
    fillModelChoices(route);
  } catch (err) {
    // A preview that cannot be fetched must not block a forecast that still can be run.
    if (mine !== routeToken) return;
    $('route-preview').hidden = true;
    $('model-field').hidden = true;
    console.warn('route preview unavailable:', err);
  }
}
