"""Ingest orchestration: STAC search -> windowed read -> daily composite -> Zarr.

Reads are parallelised across *processes*, not threads. The workload looks I/O-bound, so
threads are the intuitive choice, but measurement said otherwise: 24 threads achieved an
effective concurrency of roughly 1.2 because h5py's HDF5 calls and gzip decompression
hold the GIL. Processes cut cold-cache per-scene wall time from 5.8 s to 3.4 s. See
`habforecast.data.parallel` for the worker protocol.

Work is processed in monthly batches for three reasons: it bounds peak memory, it keeps
the Zarr append strictly time-ordered, and it means an interrupted multi-year run
resumes at the last completed month rather than starting over.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures import TimeoutError as FuturesTimeoutError
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass, field

import pandas as pd

from habforecast.aoi import LakeAOI
from habforecast.data.olci import OLCIReader, SceneObservation
from habforecast.data.parallel import init_worker, payload_to_observation, read_scene
from habforecast.data.store import LakeCube, composite_day, group_by_day

logger = logging.getLogger(__name__)

# Upper bound on how long one scene may take end-to-end, used to size the per-batch
# wall-clock budget. Generous: a cold scene read is ~3.5 s, so this only fires on a
# genuine wedge, not on a slow network.
SCENE_TIMEOUT_S = 180.0


@dataclass
class IngestReport:
    """Telemetry for one lake ingest -- logged to MLflow and used for data validation."""

    lake_id: str
    scenes_found: int = 0
    scenes_read: int = 0
    scenes_rejected: int = 0
    scenes_failed: int = 0
    days_written: int = 0
    failures: list[tuple[str, str]] = field(default_factory=list)

    @property
    def read_success_rate(self) -> float:
        return self.scenes_read / self.scenes_found if self.scenes_found else 0.0

    def summary(self) -> str:
        return (
            f"{self.lake_id}: {self.scenes_read}/{self.scenes_found} scenes read "
            f"({self.scenes_rejected} low-coverage, {self.scenes_failed} failed) "
            f"-> {self.days_written} daily frames"
        )


def _month_ranges(start: str, end: str) -> Iterator[tuple[str, str]]:
    """Yield (first_day, last_day) ISO strings for each month spanned by [start, end]."""
    t0, t1 = pd.Timestamp(start), pd.Timestamp(end)
    cursor = t0.normalize().replace(day=1)
    while cursor <= t1:
        month_end = cursor + pd.offsets.MonthEnd(0)
        lo = max(cursor, t0)
        hi = min(month_end, t1)
        yield lo.strftime("%Y-%m-%d"), hi.strftime("%Y-%m-%d")
        cursor = cursor + pd.offsets.MonthBegin(1)


def ingest_lake(
    aoi: LakeAOI,
    start: str,
    end: str,
    cube_root: str,
    *,
    aoi_path: str,
    min_valid_fraction: float = 0.02,
    aux: tuple[str, ...] = (),
    bloom_season_only: bool = True,
    max_workers: int = 12,
    batch_timeout_s: float | None = None,
    reader: OLCIReader | None = None,
    skip_days: set | None = None,
) -> IngestReport:
    """Ingest one lake over a date range into its Zarr cube.

    Args:
        aoi_path: path to the registry GeoJSON. Worker processes re-load it rather than
            receiving the AOI over a pickle, because Windows spawns fresh interpreters.
        aux: auxiliary layers to read from the same granules (see data.aux_layers).
            Each costs extra asset reads per scene, so it is opt-in rather than default.
        min_valid_fraction: drop scenes covering less than this fraction of the grid
            with QC-passing pixels. A scene clipping the lake edge costs as much to
            store as a full one but carries almost no signal.
        bloom_season_only: restrict to the lake's configured bloom months. Off-season
            frames are near-constant low chlorophyll and mostly teach the model the
            seasonal mean, which the climatology baseline already captures.
        skip_days: days to treat as already ingested although they are not in this
            cube -- held by another cube the two will later be merged with
            (scripts/merge_cubes.py). Lets a backfill write append-only into a fresh
            cube instead of rewriting a large one once per out-of-order month.
    """
    reader = reader or OLCIReader()
    grid = aoi.grid()
    cube = LakeCube(f"{cube_root}/{aoi.lake_id}.zarr", grid)
    report = IngestReport(lake_id=aoi.lake_id)
    already = cube.existing_days() | set(skip_days or ())

    # One pool for the whole lake: spawning 12 interpreters costs a few seconds and
    # would otherwise be paid again for every month.
    pool = ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=init_worker,
        initargs=(aoi_path, aoi.lake_id, None, reader.source),
    )
    try:
        for m_start, m_end in _month_ranges(start, end):
            month = pd.Timestamp(m_start).month
            if bloom_season_only and not aoi.in_bloom_season(month):
                continue

            items = reader.search(grid, m_start, m_end)
            report.scenes_found += len(items)
            if not items:
                continue

            # Skip scenes whose day is already stored -- makes re-runs cheap, not just safe.
            items = [it for it in items if it.datetime.date() not in already]
            if not items:
                logger.info("%s %s: all days already ingested", aoi.lake_id, m_start[:7])
                continue

            observations: list[SceneObservation] = []
            futures = {
                pool.submit(
                    read_scene, (reader.item_to_payload(it), min_valid_fraction, aux)
                ): it.id
                for it in items
            }
            # Wall-clock guard. The reader has its own HTTP timeouts, but a worker can
            # still wedge below that layer (a stuck HDF5 decompress, an OS-level socket
            # hang). Without this the whole multi-lake run stops silently -- which is
            # exactly what happened: 2.5 hours of zero progress and zero CPU.
            budget = batch_timeout_s or (SCENE_TIMEOUT_S * max(1, len(items) // max_workers + 1))
            try:
                for fut in as_completed(futures, timeout=budget):
                    result = fut.result()
                    if result is None:
                        report.scenes_rejected += 1
                    elif "error" in result:
                        report.scenes_failed += 1
                        report.failures.append((result["item_id"], result["error"]))
                        logger.warning("%s: read failed -- %s", result["item_id"], result["error"])
                    else:
                        report.scenes_read += 1
                        observations.append(payload_to_observation(result, aoi.lake_id))
            except (FuturesTimeoutError, BrokenProcessPool) as exc:
                # Two ways to lose a worker: it wedges (the timeout) or it dies outright --
                # a native crash inside HDF5 takes the process with it, and the pool then
                # fails every pending future. Before this was caught, one such crash on the
                # third lake of a five-lake run (2026-09-23) aborted the remaining lakes.
                broken = isinstance(exc, BrokenProcessPool)
                stalled = [
                    name
                    for f, name in futures.items()
                    if not f.done() or (broken and f.exception() is not None)
                ]
                report.scenes_failed += len(stalled)
                reason = "worker died" if broken else "worker timeout"
                report.failures.extend((s, reason) for s in stalled)
                logger.error(
                    "%s %s: %d scene(s) lost (%s) -- recycling pool",
                    aoi.lake_id,
                    m_start[:7],
                    len(stalled),
                    reason if broken else f"exceeded the {budget:.0f}s batch budget",
                )
                # A wedged child cannot be cancelled, so replace the whole pool. Frames
                # already collected in this batch are kept; the stalled days are simply
                # absent and will be retried on the next idempotent run.
                pool.shutdown(wait=False, cancel_futures=True)
                pool = ProcessPoolExecutor(
                    max_workers=max_workers,
                    initializer=init_worker,
                    initargs=(aoi_path, aoi.lake_id, None, reader.source),
                )

            if not observations:
                continue

            frames = []
            for day, obs_list in sorted(group_by_day(observations).items()):
                frame = composite_day(day, obs_list)
                if frame is not None:
                    frames.append(frame)

            written = cube.write(frames)
            report.days_written += written
            already.update(f.day for f in frames)
            logger.info(
                "%s %s: %d scenes -> %d days (%d written)",
                aoi.lake_id,
                m_start[:7],
                len(observations),
                len(frames),
                written,
            )
    finally:
        pool.shutdown(wait=True)

    logger.info(report.summary())
    return report


def ingest_all(
    aois: list[LakeAOI],
    start: str,
    end: str,
    cube_root: str,
    **kwargs,
) -> dict[str, IngestReport]:
    """Ingest every AOI sequentially (each lake already parallelises its scene reads)."""
    reports = {}
    for aoi in aois:
        logger.info("=== %s (%s, split=%s) ===", aoi.lake_id, aoi.name, aoi.split)
        reports[aoi.lake_id] = ingest_lake(aoi, start, end, cube_root, **kwargs)
    return reports
