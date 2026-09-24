"""Sentinel-3 OLCI Level-2 water scenes from the EUMETSAT Data Store -- the live source.

Why this exists
---------------
The model trains on Planetary Computer's mirror of the OLCI L2 WFR product, which lags
real time by ~6-7 months. Live forecasts read NOAA CoastWatch's 300 m sector composites
until those left the CoastWatch catalogue (checked 2026-09-23). The Data Store is
EUMETSAT's own distribution of the *same* product -- collection `EO:EUM:DAT:0407`,
`S3?_OL_2_WFR` granules with the same `chl_nn.nc` / `wqsf.nc` / `geo_coordinates.nc`
files -- available within hours of the overpass. So this is not a new data distribution
to validate the model against, as CoastWatch's pre-masked Level-3 composites were: the
full WQSF quality control in `data.qc` applies unchanged, and the scene is read by the
same `OLCIReader.read` the archive ingest uses.

What differs
------------
* **Access.** A free EUMETSAT account; the consumer key and secret live in
  `~/.eumdac/credentials` (written by `eumdac set-credentials KEY SECRET`), never in the
  repository or the environment of this process's logs. `available()` says whether they
  are present, so every caller can fall back to the archive rather than fail.
* **Two timeliness versions.** Each overpass is published as near-real-time (`NR`, ~3 h
  after sensing) and again as non-time-critical (`NT`, ~1-2 days later, better
  auxiliary data). The newest version of each overpass is not always the best, so the
  search keeps `NT` where both exist and `NR` otherwise.
* **Processing baseline.** Current products are baseline 004; the training archive is
  003. The CHL_NN algorithm and the flag table are carried forward between them, but
  that is exactly the kind of difference a skill measurement on this source exists to
  catch -- see `pipelines/evaluate_nrt.py`.
* **File access.** The download API answers HTTP range requests (206) but rejects HEAD
  (405), which fsspec's HTTP filesystem needs to size a file. `RangeFile` is the minimal
  seekable file h5py needs, sized from the first response's Content-Range, so a scene
  still costs a few megabytes of windowed reads rather than the ~70 MB granule.
"""

from __future__ import annotations

import io
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from urllib.parse import quote

from habforecast.aoi import LakeGrid
from habforecast.data.olci import (
    ASSET_CHL,
    ASSET_GEO,
    ASSET_TIE_GEO,
    ASSET_WQSF,
    HTTP_MAX_RETRIES,
    HTTP_RETRY_BACKOFF_S,
    OLCIReader,
)

logger = logging.getLogger(__name__)

SOURCE = "eumetsat"
COLLECTION = "EO:EUM:DAT:0407"  # OLCI Level 2 Ocean Colour Full Resolution
# The BC003 reprocessing of the same product: baseline 003, the one the training archive
# is on, and it starts at Sentinel-3A's first light (2016) where Planetary Computer's
# mirror starts in November 2017. It is the only source for 2016-2017 at full resolution.
SOURCE_REPROCESSED = "eumetsat-reprocessed"
COLLECTION_REPROCESSED = "EO:EUM:DAT:0556"
DOWNLOAD_URL = "https://api.eumetsat.int/data/download/1.0.0/collections/{col}/products/{pid}/entry"
# Asset keys the reader uses (Planetary Computer's names) -> file inside the SAFE granule.
ENTRY_FILES = {
    ASSET_TIE_GEO: "tie_geo_coordinates.nc",
    ASSET_GEO: "geo_coordinates.nc",
    ASSET_CHL: "chl_nn.nc",
    ASSET_WQSF: "wqsf.nc",
}
# Granule name fields: ..._<centre>_<class>_<timeliness>_<baseline>.SEN3
_NAME = re.compile(r"^(S3[AB])_OL_2_WFR____(\d{8}T\d{6})_.*_(NR|NT|ST)_(\d{3})\.SEN3$")
REQUEST_TIMEOUT_S = 90
BLOCK_SIZE = 1 << 20


def credentials_path() -> Path:
    # From `eumdac.config`, never `eumdac.cli`: importing the CLI module runs
    # `signal.signal(SIGPIPE, ...)` at import time on Linux, and the first import here
    # happens inside the forecast worker thread, where that raises "signal only works in
    # main thread of the main interpreter" -- the whole request failed with that message.
    from eumdac.config import get_credentials_path

    return Path(get_credentials_path())


def available() -> bool:
    """Whether a Data Store read can even be attempted: client installed, credentials set."""
    try:
        return credentials_path().exists()
    except ImportError:
        return False


def entry_file(asset: str) -> str:
    """The file inside the granule for a reader asset key: chl-nn -> chl_nn.nc,
    oa10-reflectance -> Oa10_reflectance.nc (the auxiliary layers' band files)."""
    if asset in ENTRY_FILES:
        return ENTRY_FILES[asset]
    name = asset.replace("-", "_")
    if name.startswith("oa"):
        name = "O" + name[1:]
    return f"{name}.nc"


# `EumetsatItem` has a field named `datetime`, which shadows the class inside its body.
_Datetime = datetime


@dataclass
class EumetsatItem:
    """The slice of a pystac Item that `OLCIReader.read` touches, for a Data Store granule."""

    id: str
    datetime: datetime
    properties: dict = field(default_factory=dict)

    @property
    def timeliness(self) -> str:
        m = _NAME.match(self.id)
        return m.group(3) if m else "?"

    @property
    def baseline(self) -> str:
        m = _NAME.match(self.id)
        return m.group(4) if m else "?"

    @property
    def overpass(self) -> tuple[str, str]:
        """(platform, sensing start): identical for the NR and NT versions of one pass."""
        m = _NAME.match(self.id)
        return (m.group(1), m.group(2)) if m else (self.id, "")

    @classmethod
    def from_id(cls, product_id: str, sensed: _Datetime | None = None) -> EumetsatItem:
        m = _NAME.match(product_id)
        if sensed is None:
            if m is None:
                raise ValueError(f"not an OLCI L2 WFR granule name: {product_id}")
            sensed = _Datetime.strptime(m.group(2), "%Y%m%dT%H%M%S").replace(tzinfo=UTC)
        platform = f"sentinel-3{m.group(1)[-1].lower()}" if m else "unknown"
        return cls(id=product_id, datetime=sensed, properties={"platform": platform})


class _Token:
    """A bearer token that renews itself shortly before expiry, shared by all reads."""

    def __init__(self):
        self._token = None

    def __call__(self) -> str:
        import eumdac

        if self._token is None:
            key, secret = credentials_path().read_text(encoding="utf-8").strip().split(",", 1)
            self._token = eumdac.AccessToken((key.strip(), secret.strip()))
        # `access_token` renews itself inside eumdac once expiry is within its request
        # margin, so reading it is all a long ingest needs. (An earlier version also called
        # a `refresh()` that eumdac does not have; every read after the first hour of a
        # run failed on it -- 59 of Taihu's scenes on 2026-09-23.)
        return self._token.access_token


class RangeFile(io.RawIOBase):
    """A read-only, seekable file over HTTP range requests, for h5py.

    Wrapped in `io.BufferedReader` by `open()`, so HDF5's many small reads of chunk
    indices become 1 MB range requests rather than one request per B-tree node.
    """

    def __init__(self, url: str, params: dict, token: _Token, session=None):
        import requests

        self.url, self.params, self.token = url, params, token
        self.session = session or requests.Session()
        self.pos = 0
        self.size = -1  # unknown until the first response; `_probe` learns it
        self._probe()

    def _get(self, start: int, end: int) -> bytes:
        last: Exception | None = None
        for attempt in range(HTTP_MAX_RETRIES):
            try:
                r = self.session.get(
                    self.url,
                    params=self.params,
                    headers={
                        "Authorization": f"Bearer {self.token()}",
                        "Range": f"bytes={start}-{end}",
                    },
                    timeout=REQUEST_TIMEOUT_S,
                )
                if r.status_code == 206:
                    if self.size < 0:
                        self.size = int(r.headers["Content-Range"].rsplit("/", 1)[1])
                    return r.content
                if r.status_code == 200:
                    # A server that ignores Range sends the whole file; take our slice.
                    self.size = len(r.content)
                    return r.content[start : end + 1]
                last = RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
                if r.status_code in (401, 403):
                    self.token._token = None  # re-authenticate once, then give up
                elif r.status_code < 500 and r.status_code != 429:
                    break
            except Exception as exc:  # timeouts, resets
                last = exc
            time.sleep(HTTP_RETRY_BACKOFF_S * (2**attempt))
        raise RuntimeError(f"range read {start}-{end} of {self.params.get('name')} failed: {last}")

    def _probe(self) -> None:
        self._get(0, 0)

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self.pos, io.SEEK_END: self.size}[whence]
        self.pos = max(0, base + offset)
        return self.pos

    def readinto(self, b) -> int:
        if self.pos >= self.size:
            return 0
        n = min(len(b), self.size - self.pos)
        data = self._get(self.pos, self.pos + n - 1)
        b[: len(data)] = data
        self.pos += len(data)
        return len(data)


class EumetsatOLCIReader(OLCIReader):
    """`OLCIReader` against the EUMETSAT Data Store: same QC, same read, fresher data."""

    source = SOURCE

    def __init__(self, collection: str = COLLECTION):
        super().__init__(collection=collection)
        # The worker processes rebuild the reader from this name (data.parallel), so a
        # reprocessed-collection reader must say so or its workers would search 0407.
        self.source = SOURCE_REPROCESSED if collection == COLLECTION_REPROCESSED else SOURCE
        self._token = _Token()
        self._store = None

    @property
    def store(self):
        if self._store is None:
            import eumdac

            key, secret = credentials_path().read_text(encoding="utf-8").strip().split(",", 1)
            self._store = eumdac.DataStore(eumdac.AccessToken((key.strip(), secret.strip())))
        return self._store

    # -- item transport for the process-pool ingest (data.parallel) ------------------
    @staticmethod
    def item_to_payload(item: EumetsatItem) -> dict:
        return {"id": item.id, "datetime": item.datetime.isoformat()}

    @staticmethod
    def item_from_payload(payload: dict) -> EumetsatItem:
        return EumetsatItem.from_id(payload["id"], datetime.fromisoformat(payload["datetime"]))

    # -- search ---------------------------------------------------------------------
    def search(self, grid: LakeGrid, start: str, end: str) -> list[EumetsatItem]:
        """Granules over the grid between two ISO dates (inclusive), one per overpass."""
        west, south, east, north = grid.bounds_lonlat
        d0 = datetime.fromisoformat(start[:10])
        d1 = datetime.fromisoformat(end[:10]) + timedelta(days=1)
        last = None
        for attempt in range(HTTP_MAX_RETRIES + 1):
            try:
                col = self.store.get_collection(self.collection)
                found = list(
                    col.search(bbox=f"{west},{south},{east},{north}", dtstart=d0, dtend=d1)
                )
                break
            except Exception as exc:
                last = exc
                self._store = None
                time.sleep(HTTP_RETRY_BACKOFF_S * (2**attempt))
        else:
            raise RuntimeError(f"{grid.lake_id}: Data Store search {start}..{end} failed: {last}")

        best: dict[tuple[str, str], EumetsatItem] = {}
        rank = {"NT": 0, "ST": 1, "NR": 2}
        for product in found:
            pid = str(product)
            if not _NAME.match(pid):
                continue
            sensed = product.sensing_start
            sensed = sensed.replace(tzinfo=UTC) if sensed.tzinfo is None else sensed
            item = EumetsatItem.from_id(pid, sensed)
            key = item.overpass
            if key not in best or rank.get(item.timeliness, 9) < rank.get(best[key].timeliness, 9):
                best[key] = item
        items = sorted(best.values(), key=lambda it: it.datetime)
        logger.info(
            "%s: %d OLCI overpasses on the Data Store between %s and %s (%d granules)",
            grid.lake_id,
            len(items),
            start,
            end,
            len(found),
        )
        return items

    def newest(self, grid: LakeGrid, lookback_days: int = 6) -> date | None:
        """The most recent day the Data Store has a granule over the grid, or None."""
        today = datetime.now(UTC).date()
        items = self.search(
            grid, (today - timedelta(days=lookback_days)).isoformat(), today.isoformat()
        )
        return max((it.datetime.date() for it in items), default=None)

    # -- file access ----------------------------------------------------------------
    def _open_asset(self, item: EumetsatItem, asset: str):
        url = DOWNLOAD_URL.format(col=quote(self.collection, safe=""), pid=quote(item.id, safe=""))
        raw = RangeFile(url, {"name": f"{item.id}/{entry_file(asset)}"}, self._token)
        return io.BufferedReader(raw, buffer_size=BLOCK_SIZE)

    def has_asset(self, item, asset: str) -> bool:
        return True  # every WFR granule carries every file; a missing one fails at read
