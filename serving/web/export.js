/* Composes what is on screen into a printable map sheet.
 *
 * The download button used to hand over the raw forecast PNG: a bare raster with no
 * basemap, no scale, no coordinates and no statement of where the data came from. That is
 * a picture of an array, not a map -- nobody receiving it could tell which lake it is or
 * how big the bloom is. What this builds instead is a sheet with the five things that
 * make a raster into a map: a graticule, a legend, a scale bar, a north indication, and a
 * source statement.
 *
 * Two constraints shaped the implementation:
 *
 *   1. Tiles must be re-fetched with `crossOrigin` before they can be drawn to a canvas.
 *      All four providers send `Access-Control-Allow-Origin: *`, but the on-screen tiles
 *      deliberately do NOT set the attribute: if a provider ever stopped sending the
 *      header, a crossOrigin tile fails to load at all, and a blank basemap is a much
 *      worse failure than a basemap-less export. So the export fetches its own copies and
 *      degrades per tile.
 *
 *   2. The sheet is light regardless of the app's theme. It is paper, not a screen, and
 *      dark ink on white is what survives being printed, pasted into a report, or opened
 *      by someone who never saw the application.
 */

import { BASEMAPS, formatDistance, formatLatLon, graticule } from './map.js';

export const PORTAL = 'Bloom Watch';

const PAPER = '#ffffff';
const INK = '#0d2026';
const INK_SOFT = '#5b7076';
const HAIR = 'rgba(13, 32, 38, 0.18)';
const ACCENT = '#0b7f92';
const NODATA = '#7c888c';

// Same five stops as the server's colour map, so the legend on the sheet is the ramp in
// the image rather than an approximation of it.
const RAMP = [[38, 70, 160], [40, 160, 140], [180, 210, 60], [240, 180, 40], [200, 40, 40]];
// The ramp spans log10(chl) 0..2, so these sit at log10(v)/2 along it.
const RAMP_TICKS = [1, 3, 10, 30, 100];

const SANS = '"IBM Plex Sans", system-ui, -apple-system, "Segoe UI", sans-serif';
const MONO = '"IBM Plex Mono", ui-monospace, Menlo, monospace';

const MARGIN = 34;
const GUTTER = 30;   // graticule label band outside the neatline
const HEADER = 76;
const FOOTER = 128;

/** Fetch one tile as a canvas-safe image; resolve to null rather than reject. */
function loadTile(url) {
  return new Promise(resolve => {
    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.onload = () => resolve(img);
    img.onerror = () => resolve(null);
    img.src = url;
  });
}

function loadImage(src) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error('overlay image failed to load'));
    img.src = src;
  });
}

function roundRect(ctx, x, y, w, h, r) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r);
  ctx.closePath();
}

/**
 * Render the current view as a map sheet.
 *
 * @param {MapView} map    the live map, read for its view, tiles and projection
 * @param {object}  o      sheet content: title, subtitle, aoi, overlay, probe, notes
 * @returns {Promise<HTMLCanvasElement>}
 */
export async function renderSheet(map, o = {}) {
  const { w: sw, h: sh } = map.size;
  if (!sw || !sh) throw new Error('the map has no size to export');

  // Never downscale the view: k < 1 would throw away tile detail the user can already
  // see. 900 px keeps a phone-sized window from exporting a postage stamp.
  const paneW = Math.min(1800, Math.max(900, sw));
  const k = paneW / sw;
  const paneH = Math.round(sh * k);

  const sheetW = paneW + 2 * (MARGIN + GUTTER);
  const sheetH = paneH + 2 * GUTTER + HEADER + FOOTER + 2 * MARGIN;
  const px = MARGIN + GUTTER;          // map pane origin
  const py = MARGIN + HEADER + GUTTER;

  const cv = document.createElement('canvas');
  cv.width = sheetW;
  cv.height = sheetH;
  const ctx = cv.getContext('2d');
  ctx.textBaseline = 'alphabetic';

  try { await document.fonts.ready; } catch { /* fonts are a nicety, not a blocker */ }

  ctx.fillStyle = PAPER;
  ctx.fillRect(0, 0, sheetW, sheetH);

  const view = map.viewState();
  const project = (lat, lon) => {
    const p = map.project(lat, lon);
    return { x: px + p.x * k, y: py + p.y * k };
  };

  /* ---------------------------------------------------------------- the map pane -- */
  ctx.save();
  ctx.beginPath();
  ctx.rect(px, py, paneW, paneH);
  ctx.clip();
  ctx.fillStyle = '#dfe7e9';
  ctx.fillRect(px, py, paneW, paneH);

  const { tiles } = map.tileLayout(sw, sh);
  const images = await Promise.all(tiles.map(t => loadTile(t.url)));
  let missing = 0;
  tiles.forEach((t, i) => {
    const img = images[i];
    if (!img) { missing++; return; }
    // +1 px on the size closes the hairline seams sub-pixel positions leave behind.
    ctx.drawImage(img, px + t.left * k, py + t.top * k, t.size * k + 1, t.size * k + 1);
  });

  if (o.overlay) {
    const [west, south, east, north] = o.overlay.bbox;
    const tl = project(north, west), br = project(south, east);
    try {
      const img = await loadImage(o.overlay.src);
      ctx.imageSmoothingEnabled = false;
      ctx.globalAlpha = o.overlay.opacity ?? 0.9;
      ctx.drawImage(img, tl.x, tl.y, br.x - tl.x, br.y - tl.y);
      ctx.globalAlpha = 1;
      ctx.imageSmoothingEnabled = true;
    } catch { /* a sheet without the overlay still beats no sheet */ }
  }

  drawGraticule(ctx, view, project, { px, py, paneW, paneH });
  if (o.aoi) drawAoi(ctx, o.aoi, project);
  if (o.probe) drawProbe(ctx, o.probe, project);
  drawWatermark(ctx, px + paneW, py + paneH);
  ctx.restore();

  ctx.strokeStyle = INK;
  ctx.lineWidth = 1.2;
  ctx.strokeRect(px + 0.5, py + 0.5, paneW - 1, paneH - 1);
  drawGraticuleLabels(ctx, view, project, { px, py, paneW, paneH });

  /* --------------------------------------------------------------------- header -- */
  ctx.fillStyle = INK;
  ctx.font = `600 27px ${SANS}`;
  ctx.textAlign = 'left';
  ctx.fillText(o.title || 'Bloom forecast', MARGIN, MARGIN + 26);
  ctx.fillStyle = INK_SOFT;
  ctx.font = `14px ${SANS}`;
  ctx.fillText(o.subtitle || '', MARGIN, MARGIN + 48);

  ctx.textAlign = 'right';
  ctx.fillStyle = INK;
  ctx.font = `600 15px ${SANS}`;
  ctx.fillText(PORTAL, sheetW - MARGIN, MARGIN + 20);
  ctx.textAlign = 'left';

  ctx.strokeStyle = HAIR;
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(MARGIN, MARGIN + HEADER - 14);
  ctx.lineTo(sheetW - MARGIN, MARGIN + HEADER - 14);
  ctx.stroke();

  /* --------------------------------------------------------------------- footer -- */
  const fy = py + paneH + GUTTER + 20;
  drawLegend(ctx, MARGIN, fy, Math.min(360, sheetW * 0.32));
  drawScaleBar(ctx, MARGIN + Math.min(360, sheetW * 0.32) + 52, fy, map.metresPerPixel() / k);
  drawNorth(ctx, sheetW - MARGIN - 26, fy + 6);

  ctx.fillStyle = INK_SOFT;
  ctx.font = `10.5px ${MONO}`;
  const source = [
    `Basemap: ${BASEMAPS[map.basemapKey].label} — ${stripTags(BASEMAPS[map.basemapKey].attribution)}`,
    o.source || '',
  ].filter(Boolean);
  source.forEach((line, i) => ctx.fillText(line, MARGIN, fy + 74 + i * 15));
  if (missing) {
    ctx.fillStyle = '#b4521c';
    ctx.fillText(`${missing} basemap tile(s) unavailable at export time`, MARGIN, fy + 74 + source.length * 15);
  }

  ctx.fillStyle = INK_SOFT;
  ctx.font = `10.5px ${MONO}`;
  ctx.textAlign = 'right';
  ctx.fillText('Advisory only — not a cyanotoxin measurement.', sheetW - MARGIN, fy + 89);
  ctx.textAlign = 'left';

  return cv;
}

const stripTags = html => html.replace(/<[^>]*>/g, '').replace(/&copy;/g, '©').trim();

/* ------------------------------------------------------------------- pane pieces -- */
function drawGraticule(ctx, view, project, box) {
  const { lats, lons } = graticule(view.bounds, 5);
  ctx.save();
  ctx.lineWidth = 1;
  ctx.setLineDash([5, 5]);
  ctx.strokeStyle = 'rgba(255,255,255,0.42)';
  ctx.beginPath();
  for (const lat of lats) {
    const y = project(lat, view.center.lon).y;
    ctx.moveTo(box.px, y); ctx.lineTo(box.px + box.paneW, y);
  }
  for (const lon of lons) {
    const x = project(view.center.lat, lon).x;
    ctx.moveTo(x, box.py); ctx.lineTo(x, box.py + box.paneH);
  }
  ctx.stroke();
  // A second pass in near-black underneath keeps the graticule legible over pale
  // terrain as well as over dark water; one colour cannot do both.
  ctx.strokeStyle = 'rgba(0,0,0,0.22)';
  ctx.setLineDash([5, 5]);
  ctx.lineDashOffset = 5;
  ctx.stroke();
  ctx.restore();
}

function drawGraticuleLabels(ctx, view, project, box) {
  const { lats, lons } = graticule(view.bounds, 5);
  ctx.save();
  ctx.fillStyle = INK;
  ctx.font = `10px ${MONO}`;
  ctx.strokeStyle = INK;
  ctx.lineWidth = 1;
  for (const lat of lats) {
    const y = project(lat, view.center.lon).y;
    if (y < box.py + 8 || y > box.py + box.paneH - 8) continue;
    ctx.beginPath();
    ctx.moveTo(box.px - 6, y); ctx.lineTo(box.px, y);
    ctx.moveTo(box.px + box.paneW, y); ctx.lineTo(box.px + box.paneW + 6, y);
    ctx.stroke();
    const t = formatLatLon(lat, 'lat');
    ctx.textAlign = 'right';
    ctx.fillText(t, box.px - 9, y + 3.5);
    ctx.textAlign = 'left';
    ctx.fillText(t, box.px + box.paneW + 9, y + 3.5);
  }
  ctx.textAlign = 'center';
  for (const lon of lons) {
    const x = project(view.center.lat, lon).x;
    if (x < box.px + 20 || x > box.px + box.paneW - 20) continue;
    ctx.beginPath();
    ctx.moveTo(x, box.py - 6); ctx.lineTo(x, box.py);
    ctx.moveTo(x, box.py + box.paneH); ctx.lineTo(x, box.py + box.paneH + 6);
    ctx.stroke();
    const t = formatLatLon(lon, 'lon');
    ctx.fillText(t, x, box.py - 10);
    ctx.fillText(t, x, box.py + box.paneH + 20);
  }
  ctx.restore();
}

function drawAoi(ctx, aoi, project) {
  ctx.save();
  ctx.strokeStyle = ACCENT;
  ctx.fillStyle = 'rgba(11, 127, 146, 0.16)';
  ctx.lineWidth = 2.4;
  ctx.setLineDash([9, 6]);
  if (aoi.kind === 'poly') {
    for (const ring of aoi.rings) {
      ctx.beginPath();
      ring.forEach(([lo, la], i) => {
        const p = project(la, lo);
        if (i) ctx.lineTo(p.x, p.y);
        else ctx.moveTo(p.x, p.y);
      });
      ctx.closePath();
      ctx.fill();
      ctx.stroke();
    }
  } else if (aoi.kind === 'circle') {
    const c = project(aoi.lat, aoi.lon);
    const edge = project(aoi.lat, aoi.lon + aoi.radiusKm / (111.320 * Math.cos(aoi.lat * Math.PI / 180)));
    const r = Math.abs(edge.x - c.x);
    ctx.beginPath();
    ctx.arc(c.x, c.y, r, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = ACCENT;
    ctx.beginPath();
    ctx.arc(c.x, c.y, 4.5, 0, Math.PI * 2);
    ctx.fill();
  } else {
    const [west, south, east, north] = aoi.bbox;
    const tl = project(north, west), br = project(south, east);
    ctx.beginPath();
    ctx.rect(tl.x, tl.y, br.x - tl.x, br.y - tl.y);
    ctx.fill();
    ctx.stroke();
  }
  ctx.restore();
}

function drawProbe(ctx, probe, project) {
  const p = project(probe.lat, probe.lon);
  ctx.save();
  ctx.strokeStyle = '#ffffff';
  ctx.lineWidth = 3.5;
  crossPath(ctx, p);
  ctx.stroke();
  ctx.strokeStyle = '#111';
  ctx.lineWidth = 1.6;
  crossPath(ctx, p);
  ctx.stroke();
  if (probe.label) {
    ctx.font = `600 12px ${MONO}`;
    const w = ctx.measureText(probe.label).width + 14;
    ctx.fillStyle = 'rgba(255,255,255,0.94)';
    roundRect(ctx, p.x + 14, p.y - 11, w, 22, 5);
    ctx.fill();
    ctx.strokeStyle = 'rgba(0,0,0,0.35)';
    ctx.lineWidth = 1;
    ctx.stroke();
    ctx.fillStyle = INK;
    ctx.fillText(probe.label, p.x + 21, p.y + 4);
  }
  ctx.restore();
}

function crossPath(ctx, p) {
  ctx.beginPath();
  ctx.moveTo(p.x - 11, p.y); ctx.lineTo(p.x - 3, p.y);
  ctx.moveTo(p.x + 3, p.y); ctx.lineTo(p.x + 11, p.y);
  ctx.moveTo(p.x, p.y - 11); ctx.lineTo(p.x, p.y - 3);
  ctx.moveTo(p.x, p.y + 3); ctx.lineTo(p.x, p.y + 11);
  ctx.arc(p.x, p.y, 7, 0, Math.PI * 2);
}

function drawWatermark(ctx, right, bottom) {
  const text = PORTAL.toUpperCase();
  ctx.save();
  ctx.font = `600 13px ${MONO}`;
  const w = ctx.measureText(text).width;
  const x = right - w - 34, y = bottom - 18;
  ctx.globalAlpha = 0.82;
  ctx.fillStyle = 'rgba(8, 22, 27, 0.55)';
  roundRect(ctx, x - 26, y - 16, w + 36, 25, 6);
  ctx.fill();
  ctx.globalAlpha = 1;
  ctx.fillStyle = '#7fe3f0';
  ctx.beginPath();
  ctx.arc(x - 14, y - 3.5, 4.5, 0, Math.PI * 2);
  ctx.fill();
  ctx.fillStyle = 'rgba(255,255,255,0.94)';
  ctx.fillText(text, x, y);
  ctx.restore();
}

/* ----------------------------------------------------------------- footer pieces -- */
function drawLegend(ctx, x, y, width) {
  ctx.save();
  ctx.fillStyle = INK_SOFT;
  ctx.font = `9.5px ${MONO}`;
  ctx.fillText('CHLOROPHYLL-A  mg m⁻³', x, y);

  const barY = y + 10, barH = 13;
  const grad = ctx.createLinearGradient(x, 0, x + width, 0);
  RAMP.forEach((c, i) => grad.addColorStop(i / (RAMP.length - 1), `rgb(${c[0]},${c[1]},${c[2]})`));
  ctx.fillStyle = grad;
  ctx.fillRect(x, barY, width, barH);
  ctx.strokeStyle = HAIR;
  ctx.lineWidth = 1;
  ctx.strokeRect(x + 0.5, barY + 0.5, width - 1, barH - 1);

  ctx.fillStyle = INK;
  ctx.font = `10px ${MONO}`;
  ctx.textAlign = 'center';
  for (const v of RAMP_TICKS) {
    const fx = x + (Math.log10(v) / 2) * width;
    ctx.strokeStyle = INK;
    ctx.beginPath();
    ctx.moveTo(fx, barY + barH); ctx.lineTo(fx, barY + barH + 4);
    ctx.stroke();
    ctx.fillText(String(v), fx, barY + barH + 16);
  }
  ctx.textAlign = 'left';

  // The grey is a third of a typical scene and is routinely mistaken for clean water,
  // so it gets its own legend entry rather than a footnote.
  const ny = barY + barH + 30;
  ctx.fillStyle = NODATA;
  ctx.fillRect(x, ny, 16, 11);
  ctx.fillStyle = INK_SOFT;
  ctx.font = `10.5px ${SANS}`;
  ctx.fillText('no retrieval (cloud, ice, land)', x + 23, ny + 10);

  ctx.strokeStyle = ACCENT;
  ctx.lineWidth = 2;
  ctx.setLineDash([6, 4]);
  ctx.strokeRect(x + 200.5, ny + 0.5, 16, 11);
  ctx.setLineDash([]);
  ctx.fillStyle = INK_SOFT;
  ctx.fillText('forecast area', x + 223, ny + 10);
  ctx.restore();
}

function drawScaleBar(ctx, x, y, metresPerPixel) {
  // Pick the widest round distance that fits, then divide it into four cells: the
  // alternating bar is legible after the sheet has been resized, which a single line
  // with a number beside it is not.
  const targets = [100, 200, 500, 1e3, 2e3, 5e3, 1e4, 2e4, 5e4, 1e5, 2e5, 5e5];
  let pick = targets[0];
  for (const t of targets) if (t / metresPerPixel <= 220) pick = t;
  const wpx = pick / metresPerPixel;
  const cell = wpx / 4, barY = y + 10, barH = 11;

  ctx.save();
  ctx.fillStyle = INK_SOFT;
  ctx.font = `9.5px ${MONO}`;
  ctx.fillText('SCALE', x, y);
  for (let i = 0; i < 4; i++) {
    ctx.fillStyle = i % 2 ? PAPER : INK;
    ctx.fillRect(x + i * cell, barY, cell, barH);
  }
  ctx.strokeStyle = INK;
  ctx.lineWidth = 1;
  ctx.strokeRect(x + 0.5, barY + 0.5, wpx - 1, barH - 1);
  ctx.fillStyle = INK;
  ctx.font = `10px ${MONO}`;
  ctx.textAlign = 'center';
  ctx.fillText('0', x, barY + barH + 15);
  ctx.fillText(formatDistance(pick / 2), x + wpx / 2, barY + barH + 15);
  ctx.fillText(formatDistance(pick), x + wpx, barY + barH + 15);
  ctx.textAlign = 'left';
  ctx.fillStyle = INK_SOFT;
  ctx.font = `9.5px ${MONO}`;
  ctx.fillText(`WEB MERCATOR (EPSG:3857) · ${Math.round(metresPerPixel)} m/px`, x, barY + barH + 32);
  ctx.restore();
}

function drawNorth(ctx, cx, y) {
  // The view is always north-up -- there is no map rotation to represent -- so this is a
  // plain north point rather than a declination diagram.
  ctx.save();
  ctx.translate(cx, y + 26);
  ctx.fillStyle = INK;
  ctx.beginPath();
  ctx.moveTo(0, -26); ctx.lineTo(9, 12); ctx.lineTo(0, 6); ctx.closePath();
  ctx.fill();
  ctx.beginPath();
  ctx.moveTo(0, -26); ctx.lineTo(-9, 12); ctx.lineTo(0, 6); ctx.closePath();
  ctx.strokeStyle = INK;
  ctx.lineWidth = 1.3;
  ctx.stroke();
  ctx.fillStyle = INK;
  ctx.font = `600 12px ${SANS}`;
  ctx.textAlign = 'center';
  ctx.fillText('N', 0, 28);
  ctx.restore();
}

/** Render and hand the sheet to the browser as a download. */
export async function downloadSheet(map, opts, filename) {
  const cv = await renderSheet(map, opts);
  const blob = await new Promise(res => cv.toBlob(res, 'image/png'));
  if (!blob) throw new Error('the browser refused to encode the sheet');
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 10_000);
}
