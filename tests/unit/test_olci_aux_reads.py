"""How many times a scene opens the same reflectance band.

The spectral indices overlap: MCI wants Oa10/Oa11/Oa12, CI wants Oa08/Oa10/Oa11, and the
phycocyanin depth wants Oa06/Oa07/Oa08. Read naively that is nine opens of six files, and
each open is a signed HTTP range read against a remote granule -- on a 46-lake ingest that
already runs for days, the difference is not academic.

The failure this guards against is silent in every way that matters: reading a band twice
produces exactly the right numbers, just slower, so nothing fails and no test that only
checks values would notice.

No network: the granule is an in-memory HDF5 file per asset.
"""

from __future__ import annotations

import io

import h5py
import numpy as np
import pytest

from habforecast.data.aux_layers import AUX_LAYERS
from habforecast.data.olci import OLCIReader


def band_file(dataset: str, value: float, shape=(8, 8)) -> io.BytesIO:
    """One OLCI band as a real HDF5 file, stored scaled the way the product does."""
    buf = io.BytesIO()
    with h5py.File(buf, "w") as h5:
        ds = h5.create_dataset(dataset, data=np.full(shape, value / 1e-5, dtype=np.int32))
        ds.attrs["scale_factor"] = 1e-5
    buf.seek(0)
    return buf


class CountingReader(OLCIReader):
    """An OLCIReader whose asset opens are local and counted."""

    def __init__(self, values: dict[str, float]):
        self.values = values
        self.opens: list[str] = []

    def has_asset(self, item, asset: str) -> bool:
        return asset in self.values

    def _open_asset(self, item, asset: str):
        self.opens.append(asset)
        dataset = None
        for layer in AUX_LAYERS.values():
            if asset in layer.sources:
                dataset = layer.sources[asset]
                break
        return band_file(dataset, self.values[asset])


@pytest.fixture
def reader():
    # A flat spectrum: every index reads zero, which is fine -- this is about read counts.
    return CountingReader(
        {
            "oa06-reflectance": 0.03,
            "oa07-reflectance": 0.03,
            "oa08-reflectance": 0.03,
            "oa10-reflectance": 0.03,
            "oa11-reflectance": 0.03,
            "oa12-reflectance": 0.03,
        }
    )


class TestBandCache:
    def test_the_full_spectral_set_opens_each_band_once(self, reader):
        cache: dict = {}
        for key in ("mci", "ci", "pc620"):
            reader._read_aux(None, key, None, cache)
        assert len(reader.opens) == 6, f"expected 6 distinct opens, got {reader.opens}"
        assert sorted(set(reader.opens)) == sorted(reader.opens), "a band was opened twice"

    def test_without_a_shared_cache_each_layer_pays_its_own_way(self, reader):
        """The old behaviour, kept explicit so the saving is visible and not assumed."""
        for key in ("mci", "ci", "pc620"):
            reader._read_aux(None, key, None)
        assert len(reader.opens) == 9

    def test_a_cached_band_still_produces_the_right_number(self, reader):
        """The cache must hand back the decoded band, not a stale or empty array."""
        reader.values["oa11-reflectance"] = 0.09  # a 709 nm peak
        cache: dict = {}
        first = reader._read_aux(None, "mci", None, cache)
        second = reader._read_aux(None, "ci", None, cache)
        # Oa11 was read once but both indices must see the peak.
        assert reader.opens.count("oa11-reflectance") == 1
        assert (first > 0).all(), "MCI should see the 709 nm peak"
        assert (second > 0).all(), "CI should see it through the cache too"

    def test_a_missing_asset_is_still_refused_by_name(self, reader):
        del reader.values["oa07-reflectance"]
        with pytest.raises(KeyError, match="oa07-reflectance"):
            reader._read_aux(None, "pc620", None, {})
