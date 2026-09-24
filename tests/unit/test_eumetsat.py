"""EUMETSAT Data Store reader: naming, timeliness de-duplication, and the range file.

No network and no credentials: the Data Store and the HTTP session are faked. The live
read against the real service was checked by hand on 2026-09-23 (Lake Erie, 5 overpasses
over 3 days, ~27 range requests and ~22 MB per scene).
"""

from __future__ import annotations

import io
from datetime import UTC, datetime

import h5py
import numpy as np
import pytest
from tests.unit.test_advisory import REPO_ROOT

from habforecast.aoi import AOIRegistry
from habforecast.data import eumetsat
from habforecast.data.olci import ASSET_CHL, ASSET_GEO, make_reader

NR = "S3A_OL_2_WFR____20260921T160459_20260921T160759_20260921T175615_0179_144_154_2160_MAR_O_NR_004.SEN3"
NT = "S3A_OL_2_WFR____20260921T160459_20260921T160759_20260922T204035_0179_144_154_2160_MAR_O_NT_004.SEN3"
NR_ONLY = "S3B_OL_2_WFR____20260922T152916_20260922T153216_20260922T171818_0179_125_011_2340_MAR_O_NR_004.SEN3"


class TestNaming:
    def test_entry_files(self):
        assert eumetsat.entry_file(ASSET_CHL) == "chl_nn.nc"
        assert eumetsat.entry_file(ASSET_GEO) == "geo_coordinates.nc"
        assert eumetsat.entry_file("oa10-reflectance") == "Oa10_reflectance.nc"

    def test_item_from_granule_name(self):
        it = eumetsat.EumetsatItem.from_id(NT)
        assert it.timeliness == "NT" and it.baseline == "004"
        assert it.properties["platform"] == "sentinel-3a"
        assert it.datetime == datetime(2026, 9, 21, 16, 4, 59, tzinfo=UTC)
        assert it.overpass == eumetsat.EumetsatItem.from_id(NR).overpass

    def test_payload_round_trip(self):
        it = eumetsat.EumetsatItem.from_id(NR)
        back = eumetsat.EumetsatOLCIReader.item_from_payload(
            eumetsat.EumetsatOLCIReader.item_to_payload(it)
        )
        assert back == it

    def test_unknown_source_rejected(self):
        with pytest.raises(ValueError):
            make_reader("nowhere")


class _Product:
    def __init__(self, pid):
        self.pid = pid
        self.sensing_start = eumetsat.EumetsatItem.from_id(pid).datetime.replace(tzinfo=None)

    def __str__(self):
        return self.pid


class _Collection:
    def search(self, **_kw):
        return [_Product(NR), _Product(NT), _Product(NR_ONLY)]


class _Store:
    def get_collection(self, _cid):
        return _Collection()


def test_search_keeps_one_granule_per_overpass_preferring_nt():
    reader = eumetsat.EumetsatOLCIReader()
    reader._store = _Store()
    grid = AOIRegistry.from_geojson(REPO_ROOT / "data" / "aoi" / "lakes.geojson")[
        "erie_west"
    ].grid()
    items = reader.search(grid, "2026-09-21", "2026-09-22")
    assert [i.id for i in items] == [NT, NR_ONLY]


class _Response:
    def __init__(self, status, content=b"", headers=None):
        self.status_code, self.content, self.headers, self.text = status, content, headers or {}, ""


class _Session:
    """Serves an in-memory file through Range requests, like the Data Store does."""

    def __init__(self, payload: bytes):
        self.payload, self.calls = payload, 0

    def get(self, url, params=None, headers=None, timeout=None):
        self.calls += 1
        start, end = (int(x) for x in headers["Range"].split("=")[1].split("-"))
        chunk = self.payload[start : end + 1]
        return _Response(206, chunk, {"Content-Range": f"bytes {start}-{end}/{len(self.payload)}"})


def test_range_file_serves_hdf5_to_h5py():
    buf = io.BytesIO()
    data = np.arange(200 * 300, dtype=np.float32).reshape(200, 300)
    with h5py.File(buf, "w") as h5:
        h5.create_dataset("CHL_NN", data=data, chunks=(50, 50), compression="gzip")
    session = _Session(buf.getvalue())

    class _Tok:
        _token = None

        def __call__(self):
            return "t"

    raw = eumetsat.RangeFile("https://example", {"name": "x"}, _Tok(), session=session)
    with io.BufferedReader(raw, buffer_size=4096) as fh, h5py.File(fh, "r") as h5:
        window = h5["CHL_NN"][60:90, 120:170]
    np.testing.assert_array_equal(window, data[60:90, 120:170])
    assert raw.size == len(buf.getvalue())
    assert session.calls > 1  # ranged, not one whole-file GET


def test_credentials_lookup_is_safe_off_the_main_thread():
    """`eumdac.cli` installs a SIGPIPE handler at import on Linux, which raises in a worker
    thread -- and a forecast request runs in one. The lookup must not import it. A fresh
    interpreter, because another test may already have imported the CLI module."""
    import subprocess
    import sys

    code = (
        "import sys, threading\n"
        "from habforecast.data import eumetsat\n"
        "errors = []\n"
        "def run():\n"
        "    try: eumetsat.available()\n"
        "    except Exception as exc: errors.append(repr(exc))\n"
        "t = threading.Thread(target=run); t.start(); t.join()\n"
        "assert not errors, errors\n"
        "assert 'eumdac.cli' not in sys.modules, 'credentials lookup imported the CLI'\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True, timeout=120)
