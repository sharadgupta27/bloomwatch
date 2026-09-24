"""Polygon handling: read a shape a user drew or uploaded, and cut a grid to it.

Three formats are accepted because those are the three a GIS user already has on disk:
GeoJSON, KML (and KMZ), and a zipped Esri shapefile. All three are parsed here rather
than in the browser -- a shapefile is binary and a KMZ is a zip, and pushing that into
client JavaScript would mean either a large vendored parser or a silent "unsupported".

Parsing is deliberately narrow. Everything is reduced to *rings of lon/lat*: the outer
boundaries of whatever polygons the file contains. Holes, z values, attributes, multi-
geometry nesting and CRS niceties are dropped, because the only question downstream is
"which 300 m cells are inside the shape".

The one CRS subtlety worth keeping: a shapefile carries its projection in a sidecar
`.prj`, and a file in a projected CRS whose coordinates are metres would otherwise be
read as absurd longitudes. When a `.prj` is present and is not already geographic, the
rings are transformed; when it is absent the coordinates are assumed to be lon/lat,
which is the overwhelmingly common case for shapes shared for web mapping.
"""

from __future__ import annotations

import io
import json
import logging
import struct
import zipfile
from xml.etree import ElementTree

import numpy as np

logger = logging.getLogger(__name__)

Ring = list[tuple[float, float]]

MAX_UPLOAD_BYTES = 8 * 1024 * 1024
MAX_RINGS = 200
MAX_VERTICES = 50_000
MIN_RING_VERTICES = 3

SHP_POLYGON_TYPES = {5, 15, 25}  # Polygon, PolygonZ, PolygonM
SHP_POLYLINE_TYPES = {3, 13, 23}


class GeometryError(ValueError):
    """A problem with an uploaded shape, phrased for the person who uploaded it."""


# ------------------------------------------------------------------------ parsing ----
def parse_geometry(filename: str, data: bytes) -> list[Ring]:
    """Rings of (lon, lat) from an uploaded geometry file.

    Dispatches on content as well as on extension: browsers label a `.geojson` as
    `application/octet-stream` about as often as not, and a renamed file should still
    work if it is obviously JSON or obviously a zip.
    """
    if len(data) > MAX_UPLOAD_BYTES:
        raise GeometryError(
            f"That file is {len(data) / 1e6:.1f} MB; the limit is "
            f"{MAX_UPLOAD_BYTES / 1e6:.0f} MB. A forecast area is at most 80 km across, so "
            "a simplified outline is enough."
        )
    if not data:
        raise GeometryError("That file is empty.")

    name = (filename or "").lower()
    head = data[:4]
    if head[:2] == b"PK":  # zip: shapefile bundle or KMZ
        rings = _from_zip(data)
    elif name.endswith(".shp") or head == b"\x00\x00\x27\x0a":
        rings = _from_shp(data, None)
    elif data.lstrip()[:1] in (b"{", b"["):
        rings = _from_geojson(data)
    elif name.endswith((".kml", ".xml")) or b"<kml" in data[:2048].lower():
        rings = _from_kml(data)
    else:
        raise GeometryError(
            "Unrecognised file. Upload GeoJSON (.geojson/.json), KML (.kml/.kmz), or a "
            "zipped shapefile (.zip containing .shp)."
        )
    return _validate(rings)


def _from_zip(data: bytes) -> list[Ring]:
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        names = zf.namelist()
        shp = next((n for n in names if n.lower().endswith(".shp")), None)
        if shp:
            prj = next((n for n in names if n.lower().endswith(".prj")), None)
            return _from_shp(zf.read(shp), zf.read(prj).decode("utf-8", "replace") if prj else None)
        kml = next((n for n in names if n.lower().endswith(".kml")), None)
        if kml:
            return _from_kml(zf.read(kml))
        gj = next((n for n in names if n.lower().endswith((".geojson", ".json"))), None)
        if gj:
            return _from_geojson(zf.read(gj))
    raise GeometryError(
        "That archive contains no .shp, .kml or .geojson. A shapefile must be zipped "
        "together with its .shx, .dbf and .prj sidecars."
    )


def _from_geojson(data: bytes) -> list[Ring]:
    try:
        doc = json.loads(data.decode("utf-8", "replace"))
    except json.JSONDecodeError as exc:
        raise GeometryError(f"That file is not valid JSON ({exc.msg}).") from exc
    rings: list[Ring] = []
    _walk_geojson(doc, rings)
    return rings


def _walk_geojson(node, rings: list[Ring]) -> None:
    if isinstance(node, list):
        for item in node:
            _walk_geojson(item, rings)
        return
    if not isinstance(node, dict):
        return
    kind = node.get("type")
    if kind == "FeatureCollection":
        _walk_geojson(node.get("features", []), rings)
    elif kind == "Feature":
        _walk_geojson(node.get("geometry"), rings)
    elif kind == "GeometryCollection":
        _walk_geojson(node.get("geometries", []), rings)
    elif kind == "Polygon":
        # coordinates[0] is the exterior ring; the rest are holes, which are dropped.
        coords = node.get("coordinates") or []
        if coords:
            rings.append(_coords(coords[0]))
    elif kind == "MultiPolygon":
        for poly in node.get("coordinates") or []:
            if poly:
                rings.append(_coords(poly[0]))
    elif kind in ("LineString", "MultiLineString"):
        coords = node.get("coordinates") or []
        parts = [coords] if kind == "LineString" else coords
        for part in parts:
            if len(part) >= MIN_RING_VERTICES:
                rings.append(_coords(part))


def _coords(seq) -> Ring:
    out: Ring = []
    for pt in seq:
        if isinstance(pt, (list, tuple)) and len(pt) >= 2:
            out.append((float(pt[0]), float(pt[1])))
    return out


def _from_kml(data: bytes) -> list[Ring]:
    try:
        root = ElementTree.fromstring(data)
    except ElementTree.ParseError as exc:
        raise GeometryError(f"That KML could not be parsed ({exc}).") from exc
    rings: list[Ring] = []
    # KML namespaces vary by producer (and Google Earth writes gx: extensions), so match
    # on the local tag name rather than on a namespace that may not be the one expected.
    for el in root.iter():
        text = el.text or ""
        if el.tag.rsplit("}", 1)[-1] != "coordinates" or not text.strip():
            continue
        ring: Ring = []
        for token in text.replace("\n", " ").split():
            parts = token.split(",")
            if len(parts) >= 2:
                try:
                    ring.append((float(parts[0]), float(parts[1])))
                except ValueError:
                    continue
        if len(ring) >= MIN_RING_VERTICES:
            rings.append(ring)
    return rings


def _from_shp(data: bytes, prj: str | None) -> list[Ring]:
    """Read polygon and polyline geometry straight out of the .shp record stream.

    The .shp format is a header plus length-prefixed records, so this needs no third-party
    reader: adding a shapefile library to the runtime image to parse ~60 lines of struct
    unpacking would be the larger liability.
    """
    if len(data) < 100:
        raise GeometryError("That .shp file is truncated.")
    rings: list[Ring] = []
    offset = 100  # fixed-size header
    while offset + 8 <= len(data):
        _, content_len = struct.unpack(">ii", data[offset : offset + 8])
        body = offset + 8
        end = body + content_len * 2
        if end > len(data):
            break
        (shape_type,) = struct.unpack("<i", data[body : body + 4])
        if shape_type in SHP_POLYGON_TYPES or shape_type in SHP_POLYLINE_TYPES:
            rings.extend(_shp_parts(data, body))
        offset = end
    if not rings:
        raise GeometryError(
            "That shapefile contains no polygon or line geometry -- a point layer cannot "
            "define an area."
        )
    if prj:
        rings = _reproject(rings, prj)
    return rings


def _shp_parts(data: bytes, body: int) -> list[Ring]:
    n_parts, n_points = struct.unpack("<ii", data[body + 36 : body + 44])
    if n_parts <= 0 or n_points <= 0 or n_points > MAX_VERTICES:
        return []
    parts_at = body + 44
    parts = list(struct.unpack(f"<{n_parts}i", data[parts_at : parts_at + 4 * n_parts]))
    points_at = parts_at + 4 * n_parts
    coords = struct.unpack(f"<{2 * n_points}d", data[points_at : points_at + 16 * n_points])
    out: list[Ring] = []
    for i, begin in enumerate(parts):
        stop = parts[i + 1] if i + 1 < n_parts else n_points
        ring = [(coords[2 * j], coords[2 * j + 1]) for j in range(begin, stop)]
        if len(ring) >= MIN_RING_VERTICES:
            out.append(ring)
    return out


def _reproject(rings: list[Ring], prj: str) -> list[Ring]:
    """Bring rings into EPSG:4326 when the .prj says they are not already there."""
    try:
        from pyproj import CRS, Transformer

        crs = CRS.from_wkt(prj)
        if crs.to_epsg() == 4326 or crs.is_geographic:
            return rings
        tf = Transformer.from_crs(crs, CRS.from_epsg(4326), always_xy=True)
    except Exception as exc:
        logger.warning("unreadable .prj (%s); assuming lon/lat", exc)
        return rings
    out: list[Ring] = []
    for ring in rings:
        xs, ys = zip(*ring, strict=True)
        lon, lat = tf.transform(xs, ys)
        out.append(list(zip(lon, lat, strict=True)))
    return out


def _validate(rings: list[Ring]) -> list[Ring]:
    kept = [r for r in rings if len(r) >= MIN_RING_VERTICES]
    if not kept:
        raise GeometryError(
            "No polygon was found in that file. It may contain only points, or only "
            "attribute records."
        )
    if len(kept) > MAX_RINGS:
        # Keep the largest by bounding-box area: an administrative layer with a thousand
        # parcels is nearly always "I meant this one big shape" plus noise.
        kept.sort(key=_bbox_area, reverse=True)
        kept = kept[:MAX_RINGS]
    total = sum(len(r) for r in kept)
    if total > MAX_VERTICES:
        raise GeometryError(
            f"That shape has {total:,} vertices; the limit is {MAX_VERTICES:,}. "
            "Simplify it before uploading -- a forecast grid is 300 m, so detail finer "
            "than that cannot change the result."
        )
    for ring in kept:
        for lon, lat in ring:
            if not (-180.0 <= lon <= 180.0 and -90.0 <= lat <= 90.0):
                raise GeometryError(
                    "Those coordinates are not longitude and latitude "
                    f"(found {lon:.1f}, {lat:.1f}). Reproject the file to EPSG:4326, or "
                    "include a .prj so the projection can be read."
                )
    return kept


def _bbox_area(ring: Ring) -> float:
    lons = [p[0] for p in ring]
    lats = [p[1] for p in ring]
    return (max(lons) - min(lons)) * (max(lats) - min(lats))


# ---------------------------------------------------------------------- rasterise ----
def rings_bounds(rings: list[Ring]) -> tuple[float, float, float, float]:
    """(west, south, east, north) enclosing every ring."""
    lons = [p[0] for r in rings for p in r]
    lats = [p[1] for r in rings for p in r]
    if not lons:
        raise GeometryError("That shape has no coordinates.")
    return min(lons), min(lats), max(lons), max(lats)


def rings_mask(rings: list[Ring], lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    """Which grid cells fall inside the rings, by the even-odd rule.

    Vectorised crossing-number test, one ring at a time: for a 300 m grid capped at 80 km
    the target is at most ~70k cells, so the whole mask costs a few million operations and
    needs no spatial index. A cell is inside when an odd number of rings contain it, which
    makes an inner ring drawn inside an outer one behave as a hole -- the conventional
    reading of a shape drawn that way.
    """
    inside = np.zeros(lon.shape, dtype=bool)
    x = lon.ravel()
    y = lat.ravel()
    for ring in rings:
        pts = np.asarray(ring, dtype=np.float64)
        if len(pts) < MIN_RING_VERTICES:
            continue
        x0, y0 = pts[:, 0], pts[:, 1]
        x1, y1 = np.roll(x0, -1), np.roll(y0, -1)
        acc = np.zeros(x.shape, dtype=bool)
        for ax, ay, bx, by in zip(x0, y0, x1, y1, strict=True):
            if ay == by:
                continue  # a horizontal edge can never be crossed by a horizontal ray
            straddles = (ay > y) != (by > y)
            # x of the edge at this y; only compare where the edge actually straddles it,
            # so the division never sees a zero denominator.
            with np.errstate(invalid="ignore", divide="ignore"):
                cross = ax + (y - ay) * (bx - ax) / (by - ay)
            acc ^= straddles & (x < cross)
        inside ^= acc.reshape(lon.shape)
    return inside


def rings_to_geojson(rings: list[Ring]) -> dict:
    """A MultiPolygon for the client to draw, with rings closed as GeoJSON requires."""
    polys = []
    for ring in rings:
        closed = list(ring)
        if closed[0] != closed[-1]:
            closed.append(closed[0])
        polys.append([[[round(x, 6), round(y, 6)] for x, y in closed]])
    return {"type": "MultiPolygon", "coordinates": polys}
