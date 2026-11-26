"""Windowed reader for Sentinel-3 OLCI Level-2 water products on Planetary Computer.

A full OLCI granule is 4090 x 4865 pixels covering roughly 1270 km of swath. Any one
lake occupies a fraction of a percent of that. Downloading whole granules would cost
~70 MB per scene and tens of gigabytes per lake-year, which does not fit on a laptop.

Instead we exploit two properties of the product format:

1. The tie-point geolocation file carries a column-subsampled lat/lon grid in ~1 MB.
   Reading it locates the AOI within the granule for almost nothing.
2. The science files are chunked HDF5 served over HTTP. Once the row/column window is
   known, h5py over an fsspec file object issues range requests and decompresses only
   the intersecting chunks.

Together these cut a scene read from ~70 MB to a few MB. Note that *strided* reads
(taking every Nth pixel) defeat this entirely -- they touch every chunk -- which is why
the AOI search goes through the tie-point grid rather than subsampling the full array.
"""

from __future__ import annotations

import logging
import socket
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from urllib.parse import urlsplit

import h5py
import numpy as np

from habforecast.aoi import LakeGrid
from habforecast.data.aux_layers import AUX_LAYERS
from habforecast.data.aux_layers import clean as aux_clean
from habforecast.data.qc import FlagTable, flag_frequencies, valid_water_mask
from habforecast.data.resample import SwathResampler

logger = logging.getLogger(__name__)

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
COLLECTION = "sentinel-3-olci-wfr-l2-netcdf"

# Assets we touch. Everything else in the 34-asset product is left alone.
ASSET_TIE_GEO = "tie-geo-coordinates"
ASSET_GEO = "geo-coordinates"
ASSET_CHL = "chl-nn"
ASSET_WQSF = "wqsf"

# Pad the derived window so resampling near the AOI edge still has source pixels.
WINDOW_PAD_PX = 8

# Network timeouts. fsspec's HTTP filesystem defaults to no timeout at all, which turns
# any stalled Azure Blob connection into an indefinitely hung ingest.
HTTP_CONNECT_TIMEOUT_S = 30
HTTP_READ_TIMEOUT_S = 90  # per-socket-read; the one that catches mid-transfer stalls
HTTP_TOTAL_TIMEOUT_S = 300  # ceiling for one asset
HTTP_MAX_RETRIES = 3
HTTP_RETRY_BACKOFF_S = 2.0

# STAC search retries. Planetary Computer intermittently times out under load; without a
# retry, one such response ends a multi-hour ingest.
STAC_MAX_RETRIES = 4
STAC_RETRY_BACKOFF_S = 3.0

# Outages of the machine's own network (Wi-Fi drop, sleep/resume). The retries above span
# well under a minute, so on 2026-09-25 a DNS outage ended two overnight ingests. When the
# host does not even resolve, pause until it does instead of burning the retries.
NETWORK_OUTAGE_MAX_WAIT_S = 3 * 3600
NETWORK_OUTAGE_POLL_S = 60


def wait_for_host(
    url_or_host: str,
    max_wait_s: float = NETWORK_OUTAGE_MAX_WAIT_S,
    poll_s: float = NETWORK_OUTAGE_POLL_S,
) -> bool:
    """Block while ``url_or_host`` fails DNS resolution, up to ``max_wait_s``.

    Returns immediately (True) when the name resolves, so it is free to call on every
    retry. Returns False if it still does not resolve after the wait.
    """
    host = urlsplit(url_or_host).hostname or url_or_host
    deadline = time.monotonic() + max_wait_s
    waited = False
    while True:
        try:
            socket.getaddrinfo(host, 443)
            if waited:
                logger.warning("network back: %s resolves again", host)
            return True
        except OSError:
            if time.monotonic() >= deadline:
                return False
            if not waited:
                logger.warning(
                    "network down: %s does not resolve; waiting up to %.0f min",
                    host,
                    max_wait_s / 60,
                )
                waited = True
            time.sleep(poll_s)


@dataclass
class SceneObservation:
    """One OLCI scene resampled onto a lake's analysis grid."""

    lake_id: str
    item_id: str
    time: datetime
    platform: str
    log_chl: np.ndarray  # (H, W) float32, log10(mg m^-3), NaN where invalid
    valid: np.ndarray  # (H, W) bool -- passed QC and covered by the swath
    flag_stats: dict[str, float] = field(default_factory=dict)
    # Extra physical layers read from the same granule (see data.aux_layers). Each is
    # (H, W) float32 with NaN where the layer itself was not retrievable, which is not
    # the same set of pixels as `valid`: a scene can carry chlorophyll where its
    # reflectance bands were saturated, and the model is told so per channel.
    aux: dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def valid_fraction(self) -> float:
        return float(self.valid.mean()) if self.valid.size else 0.0


def _read_scaled(ds: h5py.Dataset, window: tuple[slice, slice] | None = None) -> np.ndarray:
    """Read an HDF5 dataset applying CF scale_factor / add_offset / _FillValue.

    h5py returns raw stored values; unlike xarray it does not decode CF attributes.
    OLCI stores geolocation as scaled int32 micro-degrees, so skipping this step
    yields latitudes in the tens of millions.
    """
    raw = ds[window] if window is not None else ds[:]
    fill = ds.attrs.get("_FillValue")
    scale = ds.attrs.get("scale_factor")
    offset = ds.attrs.get("add_offset")

    out = raw.astype(np.float64)
    if fill is not None:
        out[raw == np.asarray(fill).ravel()[0]] = np.nan
    if scale is not None:
        out *= float(np.asarray(scale).ravel()[0])
    if offset is not None:
        out += float(np.asarray(offset).ravel()[0])
    return out


def _locate_window(
    tie_lat: np.ndarray,
    tie_lon: np.ndarray,
    bounds: tuple[float, float, float, float],
    full_shape: tuple[int, int],
    pad: int = WINDOW_PAD_PX,
) -> tuple[slice, slice] | None:
    """Map an AOI lon/lat box to a full-resolution (row, column) window.

    Returns None when the AOI does not intersect the granule at all.
    """
    west, south, east, north = bounds
    hit = (tie_lat >= south) & (tie_lat <= north) & (tie_lon >= west) & (tie_lon <= east)
    if not hit.any():
        return None

    rows, cols = np.where(hit)
    n_rows_full, n_cols_full = full_shape
    row_scale = n_rows_full / tie_lat.shape[0]
    col_scale = n_cols_full / tie_lat.shape[1]

    r0 = max(0, int(np.floor(rows.min() * row_scale)) - pad)
    r1 = min(n_rows_full, int(np.ceil((rows.max() + 1) * row_scale)) + pad)
    c0 = max(0, int(np.floor(cols.min() * col_scale)) - pad)
    c1 = min(n_cols_full, int(np.ceil((cols.max() + 1) * col_scale)) + pad)

    if r1 <= r0 or c1 <= c0:
        return None
    return (slice(r0, r1), slice(c0, c1))


class OLCIReader:
    """Searches and windowed-reads OLCI L2 water scenes for a lake grid.

    Planetary Computer by default. `data.eumetsat.EumetsatOLCIReader` subclasses it for
    the EUMETSAT Data Store, overriding only `search` and `_open_asset` -- the QC,
    windowing and resampling below are the same code for both sources.
    """

    source = "planetary-computer"

    def __init__(self, stac_url: str = STAC_URL, collection: str = COLLECTION):
        self.stac_url = stac_url
        self.collection = collection
        self._catalog = None
        self._fs = None

    # -- lazy handles so the class stays importable (and unit-testable) offline -------
    @property
    def catalog(self):
        if self._catalog is None:
            import pystac_client

            # Deliberately NOT `modifier=pc.sign_inplace`. Planetary Computer SAS tokens
            # live about an hour. Signing at search time means a multi-hour ingest starts
            # failing with HTTP 403 partway through -- observed as 18 simultaneous read
            # failures across every concurrently-running lake at 04:24, when the tokens
            # issued at the start of the run expired together. Assets are signed inside
            # `read()` instead, immediately before use.
            self._catalog = pystac_client.Client.open(self.stac_url)
        return self._catalog

    @staticmethod
    def _sign(href: str) -> str:
        """Sign an asset URL just before opening it.

        `planetary_computer.sign` caches and refreshes the underlying token itself, so
        calling it per asset is cheap and always yields a currently-valid signature.
        """
        import planetary_computer as pc

        return pc.sign(href)

    @property
    def fs(self):
        if self._fs is None:
            import aiohttp
            import fsspec

            # fsspec's HTTP filesystem has NO default timeout. A single stalled
            # connection to Azure Blob will block a read forever -- observed in practice
            # as an ingest that sat for 2.5 hours having logged nothing and consumed no
            # CPU. sock_read is the one that matters: the connection succeeds and then
            # goes quiet mid-transfer.
            self._fs = fsspec.filesystem(
                "http",
                block_size=2**20,
                client_kwargs={
                    "timeout": aiohttp.ClientTimeout(
                        total=HTTP_TOTAL_TIMEOUT_S,
                        sock_connect=HTTP_CONNECT_TIMEOUT_S,
                        sock_read=HTTP_READ_TIMEOUT_S,
                    )
                },
            )
        return self._fs

    def _open_retrying(self, href: str, resign: bool = True):
        """Open a remote asset, retrying transient network failures.

        Azure Blob occasionally resets a connection mid-transfer. Retrying is worth it:
        a scene lost to a transient error is a permanent hole in the time series.

        Each retry re-signs the URL. A 403 from an expired SAS token looks exactly like a
        transient failure at this layer, and retrying the *same* stale URL would just fail
        three times in a row -- which is what the `RuntimeError: failed to open ...`
        entries in the ingest logs actually were.
        """
        last: Exception | None = None
        base = href
        for attempt in range(HTTP_MAX_RETRIES):
            try:
                return self.fs.open(href)
            except Exception as exc:
                last = exc
                if attempt < HTTP_MAX_RETRIES - 1:
                    delay = HTTP_RETRY_BACKOFF_S * (2**attempt)
                    logger.debug("open failed (%s), retry %d in %.1fs", exc, attempt + 1, delay)
                    time.sleep(delay)
                    # Force a fresh session; the cached one may hold a dead connection.
                    self._fs = None
                    if resign:
                        href = self._sign(base)
        raise RuntimeError(f"failed to open {href[:80]} after {HTTP_MAX_RETRIES} attempts: {last}")

    def search(self, grid: LakeGrid, start: str, end: str) -> list:
        """All OLCI scenes intersecting the grid between start and end (ISO dates).

        Retries transient STAC failures. Planetary Computer intermittently returns
        "The request exceeded the maximum allowed time" under load, and without a retry a
        single such response aborts a multi-hour, multi-lake ingest at whatever month it
        happened to reach. Retrying costs seconds; not retrying costs the run.
        """
        last: Exception | None = None
        for attempt in range(STAC_MAX_RETRIES):
            try:
                search = self.catalog.search(
                    collections=[self.collection],
                    bbox=list(grid.bounds_lonlat),
                    datetime=f"{start}/{end}",
                )
                items = list(search.items())
                items.sort(key=lambda it: it.datetime)
                logger.info(
                    "%s: %d OLCI scenes between %s and %s", grid.lake_id, len(items), start, end
                )
                return items
            except Exception as exc:
                last = exc
                if attempt < STAC_MAX_RETRIES - 1:
                    delay = STAC_RETRY_BACKOFF_S * (2**attempt)
                    logger.warning(
                        "%s: STAC search %s..%s failed (%s); retry %d/%d in %.0fs",
                        grid.lake_id,
                        start,
                        end,
                        type(exc).__name__,
                        attempt + 1,
                        STAC_MAX_RETRIES - 1,
                        delay,
                    )
                    time.sleep(delay)
                    wait_for_host(STAC_URL)
                    # A fresh client avoids reusing a connection the server just dropped.
                    self._catalog = None
        raise RuntimeError(
            f"{grid.lake_id}: STAC search {start}..{end} failed after "
            f"{STAC_MAX_RETRIES} attempts: {last}"
        )

    # -- source hooks (overridden by data.eumetsat) -----------------------------------
    def _open_asset(self, item, asset: str):
        """A seekable file object for one file of a granule."""
        return self._open_retrying(self._sign(item.assets[asset].href))

    def has_asset(self, item, asset: str) -> bool:
        return asset in item.assets

    @staticmethod
    def item_to_payload(item) -> dict:
        """A picklable form of a search result, for the process-pool ingest."""
        return item.to_dict()

    @staticmethod
    def item_from_payload(payload: dict):
        import pystac

        return pystac.Item.from_dict(payload)

    def _read_aux(
        self, item, key: str, window, cache: dict[str, np.ndarray] | None = None
    ) -> np.ndarray:
        """Read and compute one auxiliary layer over the same pixel window as chlorophyll.

        `cache` is shared across the layers of a single scene, because the spectral
        indices overlap: MCI, CI and the phycocyanin depth between them name six distinct
        reflectance assets but nine slots, and each slot is a signed HTTP range read of a
        remote file. Reading a band once per scene rather than once per index is the
        difference between 9 and 6 fetches for the full spectral set.
        """
        layer = AUX_LAYERS[key]
        bands: dict[str, np.ndarray] = {}
        if cache is None:
            cache = {}
        for asset, dataset in layer.sources.items():
            if asset not in cache:
                if not self.has_asset(item, asset):
                    raise KeyError(f"granule has no asset {asset!r}")
                with (
                    self._open_asset(item, asset) as fh,
                    h5py.File(fh, "r") as h5,
                ):
                    cache[asset] = _read_scaled(h5[dataset], window)
            # Keyed by the asset stem so a three-band index can name its inputs
            # (oa10, oa11, oa12) without repeating the asset naming scheme.
            bands[asset.split("-")[0]] = cache[asset]
            bands[asset] = cache[asset]
        return aux_clean(key, layer.compute(bands))

    def read(
        self,
        item,
        grid: LakeGrid,
        resampler: SwathResampler | None = None,
        min_valid_fraction: float = 0.0,
        aux: Sequence[str] = (),
    ) -> SceneObservation | None:
        """Read one scene onto the grid, or None if it does not usefully cover the AOI.

        min_valid_fraction rejects scenes that clip only the edge of the lake or are
        almost entirely cloud -- keeping them would pad the time series with frames that
        carry no signal but still cost storage and training time.

        `aux` names extra layers from `data.aux_layers` to read out of the same granule.
        They are read *after* the QC early exit, so a scene that was going to be discarded
        never pays for them.
        """
        resampler = resampler or SwathResampler(grid)
        bounds = grid.bounds_lonlat

        # 1. Cheap AOI lookup via the column-subsampled tie-point grid.
        with (
            self._open_asset(item, ASSET_TIE_GEO) as fh,
            h5py.File(fh, "r") as h5,
        ):
            tie_lat = _read_scaled(h5["latitude"])
            tie_lon = _read_scaled(h5["longitude"])

        # 2. Full-resolution shape comes from the science file's own dimensions.
        with (
            self._open_asset(item, ASSET_CHL) as fh,
            h5py.File(fh, "r") as h5,
        ):
            full_shape = h5["CHL_NN"].shape
            window = _locate_window(tie_lat, tie_lon, bounds, full_shape)
            if window is None:
                logger.debug("%s: AOI outside granule", item.id)
                return None
            chl = _read_scaled(h5["CHL_NN"], window)

        # 3. Flags next -- also small -- so QC can reject before the expensive read.
        with (
            self._open_asset(item, ASSET_WQSF) as fh,
            h5py.File(fh, "r") as h5,
        ):
            wqsf_ds = h5["WQSF"]
            table = FlagTable.from_attrs(
                wqsf_ds.attrs["flag_masks"], wqsf_ds.attrs["flag_meanings"]
            )
            wqsf = np.asarray(wqsf_ds[window], dtype=np.uint64)

        good = valid_water_mask(wqsf, table) & np.isfinite(chl)

        # 4. Early exit before touching geolocation. The geo-coordinates asset is ~61 MB
        # against ~5 MB for chl-nn and ~6 MB for wqsf, so it dominates scene read time.
        # Most granules only clip the edge of a lake and would be discarded anyway; source
        # and target are both at 300 m, so the count of QC-passing source pixels is a
        # direct upper bound on achievable grid coverage. Checking it here rather than
        # after resampling avoids the dominant cost for the majority of scenes.
        if min_valid_fraction > 0:
            needed = min_valid_fraction * grid.height * grid.width
            if good.sum() < needed:
                logger.debug(
                    "%s: %d QC-passing source px < %.0f needed -- skipped before geo read",
                    item.id,
                    int(good.sum()),
                    needed,
                )
                return None

        # 5. Geolocation, only for scenes that can still qualify.
        with (
            self._open_asset(item, ASSET_GEO) as fh,
            h5py.File(fh, "r") as h5,
        ):
            lat = _read_scaled(h5["latitude"], window)
            lon = _read_scaled(h5["longitude"], window)

        # 6. Auxiliary layers, read only now that the scene is known to be worth it.
        fields: dict[str, np.ndarray] = {"log_chl": chl.astype(np.float32)}
        band_cache: dict[str, np.ndarray] = {}
        for key in aux:
            try:
                fields[key] = self._read_aux(item, key, window, band_cache)
            except Exception as exc:
                # One missing or unreadable band must not lose the chlorophyll: the aux
                # channel simply arrives empty, and the mask says so.
                logger.warning("%s: auxiliary layer %s failed: %s", item.id, key, exc)

        # 7. Resample only the pixels that survived QC.
        resampled, covered = resampler(lon, lat, fields, source_valid=good)
        log_chl = resampled["log_chl"]
        valid = covered & np.isfinite(log_chl)
        log_chl = np.where(valid, log_chl, np.nan).astype(np.float32)
        aux_out = {
            key: np.where(covered, resampled[key], np.nan).astype(np.float32)
            for key in aux
            if key in resampled
        }

        obs = SceneObservation(
            lake_id=grid.lake_id,
            item_id=item.id,
            time=item.datetime.astimezone(UTC),
            platform=item.properties.get("platform", "unknown"),
            log_chl=log_chl,
            valid=valid,
            flag_stats=flag_frequencies(wqsf, table),
            aux=aux_out,
        )

        if obs.valid_fraction < min_valid_fraction:
            logger.debug(
                "%s: valid fraction %.3f below threshold %.3f -- skipped",
                item.id,
                obs.valid_fraction,
                min_valid_fraction,
            )
            return None
        return obs


def make_reader(source: str = "planetary-computer", stac_url: str | None = None) -> OLCIReader:
    """The reader for a named source: "planetary-computer" (the training archive) or
    "eumetsat" (the Data Store, near-real-time; needs `~/.eumdac/credentials`), or
    "eumetsat-reprocessed" (the Data Store's BC003 reprocessing, 2016 onward)."""
    if source == "eumetsat":
        from habforecast.data.eumetsat import EumetsatOLCIReader

        return EumetsatOLCIReader()
    if source == "eumetsat-reprocessed":
        from habforecast.data.eumetsat import COLLECTION_REPROCESSED, EumetsatOLCIReader

        return EumetsatOLCIReader(COLLECTION_REPROCESSED)
    if source != "planetary-computer":
        raise ValueError(f"unknown OLCI source {source!r}")
    return OLCIReader(stac_url) if stac_url else OLCIReader()


def default_reader() -> OLCIReader:
    """The reader live paths should use: the EUMETSAT Data Store when credentials are set.

    The Data Store holds the same product as the training archive, both the reprocessed
    history and the near-real-time stream, so where it is reachable it is strictly the
    better source for a window ending today. Without credentials it is the Planetary
    Computer archive, exactly as before.
    """
    from habforecast.data import eumetsat

    return make_reader("eumetsat" if eumetsat.available() else "planetary-computer")


def read_many(
    reader: OLCIReader,
    items: Sequence,
    grid: LakeGrid,
    resampler: SwathResampler,
    min_valid_fraction: float = 0.01,
    workers: int = 8,
    progress=None,
) -> list[SceneObservation]:
    """Read scenes concurrently; a failing scene is logged and skipped, never fatal.

    Threads, not processes. A Data Store scene is ~27 range requests of ~1 s latency
    each (measured 2026-09-23), so these reads wait on the network far more than they
    decompress, and waiting releases the GIL -- unlike the Planetary Computer bulk ingest,
    where `data/parallel.py` measured threads at an effective concurrency of 1.2.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    from habforecast.data import scene_cache

    out: list[SceneObservation] = []
    if not items:
        return out
    # Scenes this server already read for exactly this grid come from disk
    # (`data.scene_cache`); only the rest are downloaded.
    source = type(reader).__name__
    cutoff = getattr(resampler, "max_distance_m", None)
    keys = {
        id(it): (
            scene_cache.key(source, it.id, grid, cutoff)
            if getattr(it, "id", None) and cutoff is not None
            else None
        )
        for it in items
    }
    todo = []
    done = 0
    for it in items:
        hit = scene_cache.load(keys[id(it)], min_valid_fraction) if keys[id(it)] else None
        if hit is None:
            todo.append(it)
            continue
        done += 1
        if hit[0] == "hit":
            out.append(hit[1])
        if progress is not None:
            progress(done, len(items))
    if not todo:
        return out
    with ThreadPoolExecutor(max_workers=max(1, min(workers, len(todo)))) as pool:
        futures = {
            pool.submit(reader.read, it, grid, resampler, min_valid_fraction): it for it in todo
        }
        for fut in as_completed(futures):
            it = futures[fut]
            done += 1
            try:
                obs = fut.result()
            except Exception as exc:  # one bad granule must not sink the window
                logger.warning("scene %s failed: %s", getattr(it, "id", "?"), exc)
                # A failure (timeout, server error) is not stored: next time, try again.
                if progress is not None:
                    progress(done, len(items))
                continue
            if keys[id(it)]:
                scene_cache.save(keys[id(it)], obs, min_valid_fraction)
            if obs is not None:
                out.append(obs)
            if progress is not None:
                progress(done, len(items))
    return out
