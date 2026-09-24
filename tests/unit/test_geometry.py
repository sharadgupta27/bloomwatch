"""Uploaded shapes decide which pixels get forecast, so a wrong ring is a wrong map.

The cases here are the ones that actually go wrong with geometry uploads: a shapefile in
metres read as degrees, a KML whose namespace is not the one expected, a polygon with a
hole, and a ring that straddles the grid edge.
"""

from __future__ import annotations

import io
import json
import struct
import zipfile

import numpy as np
import pytest

from habforecast.geometry import (
    GeometryError,
    parse_geometry,
    rings_bounds,
    rings_mask,
    rings_to_geojson,
)

SQUARE = [(120.1, 31.2), (120.3, 31.2), (120.3, 31.35), (120.1, 31.35)]


def _grid(west=120.0, south=31.1, east=120.4, north=31.4, n=40):
    lon, lat = np.meshgrid(
        np.linspace(west, east, n),
        np.linspace(north, south, n),  # north-up, as the analysis grid is
    )
    return lon, lat


class TestGeoJSON:
    def test_reads_a_polygon(self):
        doc = {"type": "Polygon", "coordinates": [[list(p) for p in SQUARE]]}
        rings = parse_geometry("a.geojson", json.dumps(doc).encode())
        assert len(rings) == 1
        assert rings_bounds(rings) == pytest.approx((120.1, 31.2, 120.3, 31.35))

    def test_reads_a_feature_collection_with_several_polygons(self):
        poly = {"type": "Polygon", "coordinates": [[list(p) for p in SQUARE]]}
        doc = {
            "type": "FeatureCollection",
            "features": [
                {"type": "Feature", "properties": {}, "geometry": poly},
                {"type": "Feature", "properties": {}, "geometry": poly},
            ],
        }
        assert len(parse_geometry("a.json", json.dumps(doc).encode())) == 2

    def test_drops_holes_rather_than_treating_them_as_separate_shapes(self):
        """An interior ring is a hole, not a second area to forecast."""
        outer = [list(p) for p in SQUARE]
        hole = [[120.15, 31.25], [120.20, 31.25], [120.20, 31.30], [120.15, 31.30]]
        doc = {"type": "Polygon", "coordinates": [outer, hole]}
        assert len(parse_geometry("a.geojson", json.dumps(doc).encode())) == 1

    def test_a_file_with_no_polygon_says_so(self):
        doc = {"type": "Point", "coordinates": [120.2, 31.3]}
        with pytest.raises(GeometryError, match="No polygon"):
            parse_geometry("a.geojson", json.dumps(doc).encode())

    def test_malformed_json_is_a_user_error_not_a_stack_trace(self):
        with pytest.raises(GeometryError, match="not valid JSON"):
            parse_geometry("a.geojson", b'{"type": "Polygon"')


class TestKML:
    KML = """<?xml version="1.0" encoding="UTF-8"?>
    <kml xmlns="http://www.opengis.net/kml/2.2"><Document><Placemark><Polygon>
    <outerBoundaryIs><LinearRing><coordinates>
      120.1,31.2,0 120.3,31.2,0 120.3,31.35,0 120.1,31.35,0 120.1,31.2,0
    </coordinates></LinearRing></outerBoundaryIs>
    </Polygon></Placemark></Document></kml>"""

    def test_reads_a_placemark_polygon(self):
        rings = parse_geometry("a.kml", self.KML.encode())
        assert rings_bounds(rings) == pytest.approx((120.1, 31.2, 120.3, 31.35))

    def test_altitude_is_dropped_not_read_as_a_coordinate(self):
        ring = parse_geometry("a.kml", self.KML.encode())[0]
        assert all(len(p) == 2 for p in ring)

    def test_a_different_namespace_still_parses(self):
        """Producers disagree on the KML namespace; matching on it would reject half of them."""
        odd = self.KML.replace(
            'xmlns="http://www.opengis.net/kml/2.2"', 'xmlns="http://earth.google.com/kml/2.1"'
        )
        assert parse_geometry("a.kml", odd.encode())

    def test_kmz_is_unzipped(self):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("doc.kml", self.KML)
        assert parse_geometry("a.kmz", buf.getvalue())


def _shapefile(points, prj: str | None = None) -> bytes:
    """A minimal single-polygon .shp, optionally zipped with a .prj."""
    n = len(points)
    body = struct.pack("<i", 5)  # shape type: polygon
    body += struct.pack("<4d", *(0.0, 0.0, 0.0, 0.0))  # bbox, unused by the reader
    body += struct.pack("<ii", 1, n)  # one part, n points
    body += struct.pack("<i", 0)  # part starts at index 0
    for x, y in points:
        body += struct.pack("<2d", x, y)
    shp = b"\x00" * 100 + struct.pack(">ii", 1, len(body) // 2) + body
    if prj is None:
        return shp
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("shape.shp", shp)
        zf.writestr("shape.prj", prj)
    return buf.getvalue()


class TestShapefile:
    def test_reads_a_polygon_from_the_record_stream(self):
        rings = parse_geometry("shape.shp", _shapefile(SQUARE))
        assert rings_bounds(rings) == pytest.approx((120.1, 31.2, 120.3, 31.35))

    def test_reads_one_out_of_a_zip_bundle(self):
        wgs84 = 'GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563]],PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433]]'
        rings = parse_geometry("shape.zip", _shapefile(SQUARE, prj=wgs84))
        assert rings_bounds(rings) == pytest.approx((120.1, 31.2, 120.3, 31.35))

    def test_a_projected_prj_is_transformed_rather_than_read_as_degrees(self):
        """UTM eastings are ~500000. Read as longitude they are nonsense; the .prj says so."""
        utm = (
            'PROJCS["WGS 84 / UTM zone 51N",GEOGCS["WGS 84",DATUM["WGS_1984",'
            'SPHEROID["WGS 84",6378137,298.257223563]],PRIMEM["Greenwich",0],'
            'UNIT["degree",0.0174532925199433]],PROJECTION["Transverse_Mercator"],'
            'PARAMETER["latitude_of_origin",0],PARAMETER["central_meridian",123],'
            'PARAMETER["scale_factor",0.9996],PARAMETER["false_easting",500000],'
            'PARAMETER["false_northing",0],UNIT["metre",1]]'
        )
        metres = [(228000, 3455000), (247000, 3455000), (247000, 3472000), (228000, 3472000)]
        west, south, east, north = rings_bounds(parse_geometry("s.zip", _shapefile(metres, utm)))
        assert 119.5 < west < 120.5 and 119.5 < east < 121.0
        assert 31.0 < south < 31.6 and 31.0 < north < 31.8

    def test_metre_coordinates_without_a_prj_are_refused_not_silently_wrong(self):
        metres = [(228000, 3455000), (247000, 3455000), (247000, 3472000)]
        with pytest.raises(GeometryError, match="not longitude and latitude"):
            parse_geometry("shape.shp", _shapefile(metres))


class TestGuards:
    def test_an_empty_file_is_rejected(self):
        with pytest.raises(GeometryError, match="empty"):
            parse_geometry("a.geojson", b"")

    def test_an_oversized_file_is_refused_before_parsing(self):
        with pytest.raises(GeometryError, match="the limit is"):
            parse_geometry("a.geojson", b"x" * (9 * 1024 * 1024))

    def test_an_unknown_format_names_the_ones_that_work(self):
        with pytest.raises(GeometryError, match="GeoJSON"):
            parse_geometry("a.txt", b"just some prose, not a shape at all")


class TestMask:
    def test_marks_the_inside_and_only_the_inside(self):
        lon, lat = _grid()
        mask = rings_mask([SQUARE], lon, lat)
        assert mask.any() and not mask.all()
        # Every marked cell is within the square's bounds.
        assert lon[mask].min() >= 120.1 and lon[mask].max() <= 120.3
        assert lat[mask].min() >= 31.2 and lat[mask].max() <= 31.35

    def test_area_is_about_right(self):
        """A quarter of this grid's lon span and half its lat span => about an eighth."""
        lon, lat = _grid()
        frac = rings_mask([SQUARE], lon, lat).mean()
        assert 0.20 < frac < 0.28, frac

    def test_a_ring_inside_another_reads_as_a_hole(self):
        lon, lat = _grid()
        hole = [(120.15, 31.25), (120.25, 31.25), (120.25, 31.30), (120.15, 31.30)]
        outer_only = rings_mask([SQUARE], lon, lat)
        with_hole = rings_mask([SQUARE, hole], lon, lat)
        assert with_hole.sum() < outer_only.sum()
        assert (with_hole & ~outer_only).sum() == 0

    def test_two_separate_rings_both_count(self):
        lon, lat = _grid()
        far = [(120.32, 31.12), (120.38, 31.12), (120.38, 31.18)]
        both = rings_mask([SQUARE, far], lon, lat)
        assert both.sum() > rings_mask([SQUARE], lon, lat).sum()

    def test_a_shape_entirely_off_the_grid_marks_nothing(self):
        lon, lat = _grid()
        elsewhere = [(0.0, 0.0), (0.1, 0.0), (0.1, 0.1)]
        assert not rings_mask([elsewhere], lon, lat).any()

    def test_a_horizontal_edge_does_not_break_the_crossing_count(self):
        """Edges exactly level with a scanline are the classic even-odd rasterising bug."""
        lon, lat = _grid(south=31.2, north=31.35, n=21)  # rows land exactly on the edges
        mask = rings_mask([SQUARE], lon, lat)
        assert mask.any()
        # No row should be fully marked or fully empty by accident of edge alignment.
        assert 0.15 < mask.mean() < 0.6, mask.mean()


class TestRoundTrip:
    def test_geojson_output_closes_its_rings(self):
        doc = rings_to_geojson([SQUARE])
        assert doc["type"] == "MultiPolygon"
        ring = doc["coordinates"][0][0]
        assert ring[0] == ring[-1], "GeoJSON requires a closed ring"

    def test_output_parses_back_to_the_same_bounds(self):
        doc = rings_to_geojson([SQUARE])
        back = parse_geometry("x.geojson", json.dumps(doc).encode())
        assert rings_bounds(back) == pytest.approx(rings_bounds([SQUARE]))
