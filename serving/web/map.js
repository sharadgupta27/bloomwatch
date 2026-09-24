/* A small slippy-map engine: pan, zoom, layers, and a georeferenced raster overlay.
 *
 * Written rather than pulled from a CDN. Leaflet is ~150 KB and would make an offline
 * container depend on a third-party host at page load; vendoring a minified blob into the
 * repo trades that for an unreadable 150 KB file nobody reviews. What this app actually
 * needs -- Web Mercator arithmetic, a tile grid, pointer handling and one image overlay --
 * is a few hundred lines, and owning it means the interaction quality is a choice rather
 * than a default.
 *
 * Coordinates, in three spaces:
 *   geo     lat/lon degrees, EPSG:4326
 *   world   pixels at the *current fractional* zoom; origin at the antimeridian/85N corner
 *   screen  pixels inside the container
 * Everything is derived from `project`/`unproject`, so there is one place to be wrong.
 */

const TILE = 256;
const MAX_LAT = 85.0511287798; // where Web Mercator is clipped square
const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));

export const BASEMAPS = {
  satellite: {
    label: 'Satellite',
    url: z => `https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/${z.z}/${z.y}/${z.x}`,
    attribution: 'Imagery &copy; Esri, Maxar, Earthstar Geographics',
    maxZoom: 18,
    dark: true,
  },
  streets: {
    label: 'Streets',
    url: z => `https://tile.openstreetmap.org/${z.z}/${z.x}/${z.y}.png`,
    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright" target="_blank" rel="noopener">OpenStreetMap</a> contributors',
    maxZoom: 19,
    dark: false,
  },
  terrain: {
    label: 'Terrain',
    url: z => `https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/${z.z}/${z.y}/${z.x}`,
    attribution: 'Map data &copy; Esri, USGS, NOAA',
    maxZoom: 17,
    dark: false,
  },
  ocean: {
    label: 'Ocean',
    url: z => `https://server.arcgisonline.com/ArcGIS/rest/services/Ocean/World_Ocean_Base/MapServer/tile/${z.z}/${z.y}/${z.x}`,
    attribution: 'Map data &copy; Esri, GEBCO, NOAA',
    maxZoom: 13,
    dark: false,
  },
};

/* ------------------------------------------------------------------ projection ----- */
export function lonToWorldX(lon, zoom) {
  return ((lon + 180) / 360) * TILE * Math.pow(2, zoom);
}
export function latToWorldY(lat, zoom) {
  const r = clamp(lat, -MAX_LAT, MAX_LAT) * Math.PI / 180;
  return ((1 - Math.log(Math.tan(r) + 1 / Math.cos(r)) / Math.PI) / 2) * TILE * Math.pow(2, zoom);
}
export function worldXToLon(x, zoom) {
  return (x / (TILE * Math.pow(2, zoom))) * 360 - 180;
}
export function worldYToLat(y, zoom) {
  const n = Math.PI - 2 * Math.PI * y / (TILE * Math.pow(2, zoom));
  return (180 / Math.PI) * Math.atan(0.5 * (Math.exp(n) - Math.exp(-n)));
}

/** Great-circle distance in metres. Used for the measure tool and the scale bar. */
export function haversine(a, b) {
  const R = 6371008.8;
  const p = Math.PI / 180;
  const dLat = (b.lat - a.lat) * p, dLon = (b.lon - a.lon) * p;
  const s = Math.sin(dLat / 2) ** 2 +
    Math.cos(a.lat * p) * Math.cos(b.lat * p) * Math.sin(dLon / 2) ** 2;
  return 2 * R * Math.asin(Math.sqrt(s));
}

/** Spherical polygon area in m^2, via the spherical excess of the ring. */
export function ringArea(points) {
  if (points.length < 3) return 0;
  const R = 6371008.8, p = Math.PI / 180;
  let total = 0;
  for (let i = 0; i < points.length; i++) {
    const a = points[i], b = points[(i + 1) % points.length];
    total += (b.lon - a.lon) * p * (2 + Math.sin(a.lat * p) + Math.sin(b.lat * p));
  }
  return Math.abs(total * R * R / 2);
}

export function formatDistance(m) {
  if (m < 1000) return `${m.toFixed(0)} m`;
  const km = m / 1000;
  // Trailing zeros read as false precision on a scale bar: "50.00 km" implies a
  // measurement, "50 km" reads as the round number it is.
  const digits = km < 10 ? 2 : km < 100 ? 1 : 0;
  return `${Number(km.toFixed(digits))} km`;
}
export function formatArea(m2) {
  const km2 = m2 / 1e6;
  if (km2 < 1) return `${(m2 / 10000).toFixed(1)} ha`;
  return `${km2.toFixed(km2 < 100 ? 2 : 0)} km²`;
}

/* ------------------------------------------------------------------- graticule ----- */
// Intervals a cartographer would actually label: whole degrees down to 30", never an
// arbitrary "nice" decimal like 0.0234°, which is what a naive log10 step produces.
const GRAT_STEPS = [30, 15, 10, 5, 2, 1, 0.5, 0.25, 1 / 6, 1 / 12, 1 / 30, 1 / 60, 1 / 120];

/** Label a degree value as D°M'S", the convention on a printed map sheet. */
export function formatLatLon(deg, axis) {
  // The equator and the prime/antimeridian carry no hemisphere: "0°N" is not a thing a
  // map sheet says.
  const zeroish = Math.abs(deg) < 1e-9 || (axis === 'lon' && Math.abs(Math.abs(deg) - 180) < 1e-9);
  const hemi = zeroish ? '' : axis === 'lat' ? (deg >= 0 ? 'N' : 'S') : (deg >= 0 ? 'E' : 'W');
  let v = Math.abs(deg);
  const d = Math.floor(v + 1e-9);
  v = (v - d) * 60;
  const m = Math.floor(v + 1e-7);
  const s = Math.round((v - m) * 60);
  const [mm, ss] = s === 60 ? [m + 1, 0] : [m, s];
  if (mm === 0 && ss === 0) return `${d}°${hemi}`;
  if (ss === 0) return `${d}°${String(mm).padStart(2, '0')}′${hemi}`;
  return `${d}°${String(mm).padStart(2, '0')}′${String(ss).padStart(2, '0')}″${hemi}`;
}

/** Graticule lines crossing a [west, south, east, north] extent, aimed at ~`target` of them. */
export function graticule([west, south, east, north], target = 5) {
  const span = Math.max(Math.abs(north - south), Math.abs(east - west));
  // Walk the intervals coarse to fine and take the first that divides the extent into
  // enough pieces. Asking instead for the first step the extent fits *inside* always
  // answers 30 degrees, which over a lake draws one line through the corner of the map.
  const want = Math.max(2, target - 1);
  let step = GRAT_STEPS[GRAT_STEPS.length - 1];
  for (const s of GRAT_STEPS) if (span / s >= want) { step = s; break; }
  const series = (lo, hi) => {
    const out = [];
    for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) {
      out.push(Math.abs(v) < 1e-9 ? 0 : v); // -0 prints as "0°S"
    }
    return out;
  };
  return { step, lats: series(south, north), lons: series(west, east) };
}

/* ------------------------------------------------------------------- the map ------- */
export class MapView {
  constructor(container, { lat = 20, lon = 0, zoom = 2, minZoom = 2, basemap = 'satellite' } = {}) {
    this.el = container;
    this.el.classList.add('mv');
    this.center = { lat, lon };
    this.zoom = zoom;
    this.minZoom = minZoom;
    this.basemapKey = basemap;
    this._handlers = {};
    this._tiles = new Map(); // key -> img, reused across renders so panning does not flash
    this._overlays = [];

    this.tileLayer = document.createElement('div');
    this.tileLayer.className = 'mv-tiles';
    this.overlayLayer = document.createElement('div');
    this.overlayLayer.className = 'mv-overlays';
    this.vectorLayer = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    this.vectorLayer.setAttribute('class', 'mv-vectors');
    this.markerLayer = document.createElement('div');
    this.markerLayer.className = 'mv-markers';
    this.el.append(this.tileLayer, this.overlayLayer, this.vectorLayer, this.markerLayer);

    this._bindPointer();
    this._observer = new ResizeObserver(() => this.render());
    this._observer.observe(this.el);
    this.render();
  }

  on(event, fn) { (this._handlers[event] ||= []).push(fn); return this; }
  _emit(event, payload) { (this._handlers[event] || []).forEach(f => f(payload)); }

  get size() { return { w: this.el.clientWidth, h: this.el.clientHeight }; }
  get maxZoom() { return BASEMAPS[this.basemapKey].maxZoom; }
  get isDark() { return !!BASEMAPS[this.basemapKey].dark; }

  project(lat, lon) {
    const { w, h } = this.size;
    return {
      x: lonToWorldX(lon, this.zoom) - lonToWorldX(this.center.lon, this.zoom) + w / 2,
      y: latToWorldY(lat, this.zoom) - latToWorldY(this.center.lat, this.zoom) + h / 2,
    };
  }

  unproject(x, y) {
    const { w, h } = this.size;
    return {
      lat: worldYToLat(latToWorldY(this.center.lat, this.zoom) + y - h / 2, this.zoom),
      lon: worldXToLon(lonToWorldX(this.center.lon, this.zoom) + x - w / 2, this.zoom),
    };
  }

  setView(lat, lon, zoom = this.zoom, { emit = true } = {}) {
    this.center = { lat: clamp(lat, -MAX_LAT, MAX_LAT), lon: ((lon + 540) % 360) - 180 };
    this.zoom = clamp(zoom, this.minZoom, this.maxZoom);
    this.render();
    if (emit) this._emit('move', this.viewState());
  }

  /** Frame a [west, south, east, north] box with a little breathing room. */
  fitBounds([west, south, east, north], padding = 0.22) {
    const { w, h } = this.size;
    let best = this.minZoom;
    for (let z = this.maxZoom; z >= this.minZoom; z -= 0.25) {
      const dx = Math.abs(lonToWorldX(east, z) - lonToWorldX(west, z));
      const dy = Math.abs(latToWorldY(south, z) - latToWorldY(north, z));
      if (dx <= w * (1 - padding) && dy <= h * (1 - padding)) { best = z; break; }
    }
    this.setView((south + north) / 2, (west + east) / 2, best);
  }

  viewState() {
    const { w, h } = this.size;
    const nw = this.unproject(0, 0), se = this.unproject(w, h);
    return { center: { ...this.center }, zoom: this.zoom, bounds: [nw.lon, se.lat, se.lon, nw.lat] };
  }

  setBasemap(key) {
    if (!BASEMAPS[key]) return;
    this.basemapKey = key;
    this.zoom = clamp(this.zoom, this.minZoom, this.maxZoom);
    this._tiles.forEach(img => img.remove());
    this._tiles.clear();
    this.render();
    this._emit('basemap', key);
  }

  zoomBy(delta, anchor) {
    const target = clamp(this.zoom + delta, this.minZoom, this.maxZoom);
    if (target === this.zoom) return;
    if (anchor) {
      // Keep the point under the cursor fixed: convert it to geo before the zoom change
      // and put it back at the same screen position afterwards.
      const geo = this.unproject(anchor.x, anchor.y);
      this.zoom = target;
      const after = this.project(geo.lat, geo.lon);
      const { w, h } = this.size;
      const shift = this.unproject(w / 2 - (after.x - anchor.x), h / 2 - (after.y - anchor.y));
      this.setView(shift.lat, shift.lon, target);
    } else {
      this.setView(this.center.lat, this.center.lon, target);
    }
  }

  /* ------------------------------------------------------------- interaction ----- */
  _bindPointer() {
    const el = this.el;
    let dragging = false, moved = 0, last = null, pointers = new Map(), pinchStart = null;

    el.addEventListener('pointerdown', e => {
      if (e.button !== 0 && e.pointerType === 'mouse') return;
      pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
      if (pointers.size === 2) {
        const [a, b] = [...pointers.values()];
        pinchStart = { dist: Math.hypot(a.x - b.x, a.y - b.y), zoom: this.zoom };
        dragging = false;
        return;
      }
      dragging = true; moved = 0; last = { x: e.clientX, y: e.clientY };
      el.setPointerCapture(e.pointerId);
      el.classList.add('mv-dragging');
    });

    el.addEventListener('pointermove', e => {
      if (pointers.has(e.pointerId)) pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
      if (pointers.size === 2 && pinchStart) {
        const [a, b] = [...pointers.values()];
        const dist = Math.hypot(a.x - b.x, a.y - b.y);
        this.setView(this.center.lat, this.center.lon,
          pinchStart.zoom + Math.log2(dist / Math.max(pinchStart.dist, 1)));
        return;
      }
      if (!dragging) {
        const r = el.getBoundingClientRect();
        this._emit('hover', this.unproject(e.clientX - r.left, e.clientY - r.top));
        return;
      }
      const dx = e.clientX - last.x, dy = e.clientY - last.y;
      moved += Math.abs(dx) + Math.abs(dy);
      last = { x: e.clientX, y: e.clientY };
      const { w, h } = this.size;
      const c = this.unproject(w / 2 - dx, h / 2 - dy);
      this.setView(c.lat, c.lon, this.zoom);
    });

    const end = e => {
      pointers.delete(e.pointerId);
      if (pointers.size < 2) pinchStart = null;
      if (!dragging) return;
      dragging = false;
      el.classList.remove('mv-dragging');
      const r = el.getBoundingClientRect();
      // A drag is not a click. 4 px of slop keeps a shaky hand from losing its click.
      if (moved < 4) this._emit('click', this.unproject(e.clientX - r.left, e.clientY - r.top));
    };
    el.addEventListener('pointerup', end);
    el.addEventListener('pointercancel', end);

    el.addEventListener('wheel', e => {
      e.preventDefault();
      const r = el.getBoundingClientRect();
      const step = e.deltaMode === 1 ? e.deltaY * 0.15 : e.deltaY * 0.0035;
      this.zoomBy(-clamp(step, -1.5, 1.5), { x: e.clientX - r.left, y: e.clientY - r.top });
    }, { passive: false });

    el.addEventListener('dblclick', e => {
      const r = el.getBoundingClientRect();
      this.zoomBy(1, { x: e.clientX - r.left, y: e.clientY - r.top });
    });

    // Keyboard panning, so the map is usable without a pointer at all.
    el.tabIndex = 0;
    el.addEventListener('keydown', e => {
      const step = e.shiftKey ? 200 : 80;
      const nudge = (dx, dy) => {
        const { w, h } = this.size;
        const c = this.unproject(w / 2 + dx, h / 2 + dy);
        this.setView(c.lat, c.lon, this.zoom);
      };
      const keys = {
        ArrowLeft: () => nudge(-step, 0), ArrowRight: () => nudge(step, 0),
        ArrowUp: () => nudge(0, -step), ArrowDown: () => nudge(0, step),
        '+': () => this.zoomBy(1), '=': () => this.zoomBy(1), '-': () => this.zoomBy(-1),
      };
      if (keys[e.key]) { e.preventDefault(); keys[e.key](); }
    });
  }

  /* ----------------------------------------------------------------- overlays ---- */
  /** A georeferenced raster pinned to a [west, south, east, north] box. */
  addImageOverlay(bbox, src, { opacity = 0.9, id = 'overlay' } = {}) {
    let entry = this._overlays.find(o => o.id === id);
    if (!entry) {
      const img = document.createElement('img');
      img.className = 'mv-image-overlay';
      img.alt = '';
      this.overlayLayer.appendChild(img);
      entry = { id, img };
      this._overlays.push(entry);
    }
    entry.bbox = bbox;
    entry.img.src = src;
    entry.img.style.opacity = opacity;
    entry.img.hidden = false;
    this.render();
    return entry;
  }

  setOverlayOpacity(opacity, id = 'overlay') {
    const o = this._overlays.find(x => x.id === id);
    if (o) o.img.style.opacity = opacity;
  }

  removeOverlay(id = 'overlay') {
    const i = this._overlays.findIndex(o => o.id === id);
    if (i >= 0) { this._overlays[i].img.remove(); this._overlays.splice(i, 1); }
  }

  /* ------------------------------------------------------------------ render ----- */
  render() {
    const { w, h } = this.size;
    if (!w || !h) return;
    this._renderTiles(w, h);
    for (const o of this._overlays) {
      const [west, south, east, north] = o.bbox;
      const tl = this.project(north, west), br = this.project(south, east);
      Object.assign(o.img.style, {
        left: `${tl.x}px`, top: `${tl.y}px`,
        width: `${br.x - tl.x}px`, height: `${br.y - tl.y}px`,
      });
    }
    this.vectorLayer.setAttribute('viewBox', `0 0 ${w} ${h}`);
    this.vectorLayer.setAttribute('width', w);
    this.vectorLayer.setAttribute('height', h);
    this._emit('render', this.viewState());
  }

  /** Which tiles cover a w×h viewport at the current view, and where each one sits.
   *
   * Split out of the renderer because the PNG export has to lay the same tiles onto a
   * canvas. Two copies of this arithmetic would drift, and the failure mode -- an
   * exported map offset from the one on screen -- is silent.
   */
  tileLayout(w, h) {
    const spec = BASEMAPS[this.basemapKey];
    const tz = clamp(Math.round(this.zoom), this.minZoom, spec.maxZoom);
    const size = TILE * Math.pow(2, this.zoom - tz);
    const n = Math.pow(2, tz);
    const originX = (lonToWorldX(this.center.lon, this.zoom) - w / 2) / size;
    const originY = (latToWorldY(this.center.lat, this.zoom) - h / 2) / size;
    const tiles = [];
    for (let ty = Math.floor(originY); ty <= Math.floor(originY + h / size); ty++) {
      if (ty < 0 || ty >= n) continue;
      for (let tx = Math.floor(originX); tx <= Math.floor(originX + w / size); tx++) {
        const wrapped = ((tx % n) + n) % n; // the world repeats east-west
        tiles.push({
          key: `${tz}/${wrapped}/${ty}/${tx}`,
          url: spec.url({ z: tz, x: wrapped, y: ty }),
          left: (tx - originX) * size,
          top: (ty - originY) * size,
          size,
        });
      }
    }
    return { tiles, size, spec };
  }

  _renderTiles(w, h) {
    const { tiles } = this.tileLayout(w, h);
    const wanted = new Set();
    for (const { key, url, left, top, size } of tiles) {
      wanted.add(key);
      let img = this._tiles.get(key);
      if (!img) {
        img = document.createElement('img');
        img.alt = '';
        img.loading = 'eager';
        img.decoding = 'async';
        img.draggable = false;
        img.className = 'mv-tile';
        img.addEventListener('load', () => img.classList.add('mv-tile-in'));
        // A missing tile is normal at the edges of a provider's coverage; leave the
        // ground showing rather than a broken-image glyph.
        img.addEventListener('error', () => img.classList.add('mv-tile-missing'));
        img.src = url;
        this.tileLayer.appendChild(img);
        this._tiles.set(key, img);
      }
      Object.assign(img.style, {
        left: `${left}px`, top: `${top}px`,
        width: `${size + 1}px`, height: `${size + 1}px`,
      });
    }
    for (const [key, img] of this._tiles) {
      if (!wanted.has(key)) { img.remove(); this._tiles.delete(key); }
    }
  }

  /** Metres per screen pixel at the current centre -- drives the scale bar. */
  metresPerPixel() {
    return 156543.03392 * Math.cos(this.center.lat * Math.PI / 180) / Math.pow(2, this.zoom);
  }
}
