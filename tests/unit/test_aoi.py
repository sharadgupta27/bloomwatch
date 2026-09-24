"""Tests for the AOI registry and the deterministic analysis grid.

Grid determinism is a data-integrity property, not a nicety: the Zarr cubes are appended
to across many separate ingest runs, and if `LakeAOI.grid()` ever returned a different
origin or shape for the same AOI, frames written by different runs would be silently
misaligned by up to one pixel with no error raised anywhere.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from habforecast.aoi import OLCI_FR_RESOLUTION_M, AOIRegistry, LakeAOI

REGISTRY_PATH = Path(__file__).resolve().parents[2] / "data" / "aoi" / "lakes.geojson"


@pytest.fixture(scope="module")
def registry() -> AOIRegistry:
    return AOIRegistry.from_geojson(REGISTRY_PATH)


@pytest.fixture
def erie(registry) -> LakeAOI:
    return registry["erie_west"]


class TestRegistry:
    def test_loads_all_lakes(self, registry):
        assert len(registry) == 5
        assert "erie_west" in registry.lake_ids

    def test_splits_are_disjoint(self, registry):
        """No lake may appear in more than one split -- the core anti-leakage invariant."""
        seen: dict[str, str] = {}
        for split in ("train", "val", "test"):
            for aoi in registry.split(split):
                assert aoi.lake_id not in seen, (
                    f"{aoi.lake_id} appears in both '{seen.get(aoi.lake_id)}' and '{split}'"
                )
                seen[aoi.lake_id] = split
        assert len(seen) == len(registry)

    def test_every_split_is_populated(self, registry):
        for split in ("train", "val", "test"):
            assert registry.split(split), f"split '{split}' is empty"

    def test_duplicate_lake_ids_rejected(self, erie):
        with pytest.raises(ValueError, match="duplicate"):
            AOIRegistry([erie, erie])

    def test_unknown_lake_raises(self, registry):
        with pytest.raises(KeyError):
            registry["not_a_lake"]


class TestBloomSeason:
    def test_normal_range(self, erie):
        assert erie.bloom_season == (6, 10)
        assert erie.in_bloom_season(8)
        assert not erie.in_bloom_season(2)

    def test_wrapping_range_spans_new_year(self):
        aoi = LakeAOI(
            lake_id="southern",
            name="S",
            country="X",
            split="train",
            trophic_state="eutrophic",
            bloom_season=(11, 3),
            utm_epsg=32736,
            bbox=(30.0, -1.0, 31.0, 0.0),
        )
        assert aoi.in_bloom_season(12)
        assert aoi.in_bloom_season(1)
        assert not aoi.in_bloom_season(6)


class TestGrid:
    def test_is_deterministic(self, erie):
        """Same AOI must always yield an identical grid across calls."""
        a, b = erie.grid(), erie.grid()
        assert (a.x_min, a.y_min, a.width, a.height) == (b.x_min, b.y_min, b.width, b.height)

    def test_origin_snapped_to_resolution_lattice(self, erie):
        g = erie.grid()
        assert g.x_min % g.resolution == pytest.approx(0.0, abs=1e-6)
        assert g.y_min % g.resolution == pytest.approx(0.0, abs=1e-6)

    def test_default_resolution_is_olci_native(self, erie):
        assert erie.grid().resolution == OLCI_FR_RESOLUTION_M

    def test_shape_matches_extent(self, erie):
        g = erie.grid()
        assert g.shape == (g.height, g.width)
        assert g.x_max - g.x_min == pytest.approx(g.width * g.resolution)
        assert g.y_max - g.y_min == pytest.approx(g.height * g.resolution)

    def test_coordinates_are_cell_centres(self, erie):
        g = erie.grid()
        assert g.x_coords[0] == pytest.approx(g.x_min + g.resolution / 2)
        assert len(g.x_coords) == g.width
        assert len(g.y_coords) == g.height

    def test_y_descends_north_up(self, erie):
        g = erie.grid()
        assert np.all(np.diff(g.y_coords) < 0), "row 0 must be the northernmost"

    def test_bounds_cover_source_bbox(self, erie):
        """The projected+snapped grid must fully contain the original lon/lat AOI."""
        west, south, east, north = erie.bbox
        gw, gs, ge, gn = erie.grid().bounds_lonlat
        assert gw <= west and gs <= south and ge >= east and gn >= north

    def test_finer_resolution_gives_more_pixels(self, erie):
        coarse, fine = erie.grid(600.0), erie.grid(300.0)
        assert fine.width > coarse.width and fine.height > coarse.height

    @pytest.mark.parametrize(
        "lake_id", ["erie_west", "okeechobee", "taihu", "balaton", "victoria_ne"]
    )
    def test_all_grids_fit_the_vram_budget(self, registry, lake_id):
        """Grids must stay small enough to train on a 4 GB card."""
        g = registry[lake_id].grid()
        assert g.width * g.height < 250_000, f"{lake_id} grid {g.shape} too large for 4 GB"


class TestRegistryFile:
    def test_geojson_is_wellformed(self):
        raw = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
        assert raw["type"] == "FeatureCollection"
        for feat in raw["features"]:
            props = feat["properties"]
            for key in ("lake_id", "name", "split", "utm_epsg", "bloom_season"):
                assert key in props, f"{props.get('lake_id')} missing '{key}'"
            assert props["split"] in {"train", "val", "test"}
            ring = feat["geometry"]["coordinates"][0]
            assert ring[0] == ring[-1], "polygon ring must be closed"
