"""Background jobs for forecasts that take minutes rather than milliseconds.

A forecast for one of the five registry lakes reads a local Zarr cube and returns in under
a second. A forecast for an arbitrary point has to search STAC, download and reproject up
to a dozen granules, and only then run the model -- typically 30 seconds to three minutes,
almost all of it waiting on the network.

Holding an HTTP request open for that is the wrong shape: proxies time out, the browser
gives no sign of life, and a reload silently starts the work again. So the request returns
a job id immediately and the client polls, which also gives somewhere for the per-stage
progress the pipeline already reports to go.

Deliberately in-process and in-memory. A Redis queue and a worker pool would be the right
answer for many concurrent users, but this serves a single container with one model in
memory; adding a broker would add an operational dependency without changing what any
current user experiences. The bounded worker count and job cap below are what keep that
honest -- they mean the process degrades by refusing work, not by exhausting memory.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# One at a time. The work is network-bound, but it ends by running a model on the shared
# predictor, and the readers hold whole granules in memory while they reproject. Two
# concurrent jobs on a 2 GB container is how you get an OOM kill instead of a queue.
MAX_WORKERS = 1
# Jobs are small (a few PNGs and some numbers) but not free. Past this many, the oldest
# finished job is dropped.
MAX_JOBS = 64
# Finished jobs stay readable this long so a slow client can still collect its result.
JOB_TTL_S = 30 * 60


@dataclass
class Job:
    """One unit of background work and everything a client needs to follow it."""

    id: str
    kind: str
    params: dict[str, Any]
    status: str = "queued"  # queued | running | done | error
    stage: str = "waiting for a worker"
    progress: float = 0.0
    result: Any = None
    error: str | None = None
    created: float = field(default_factory=time.time)
    finished: float | None = None

    def as_dict(self) -> dict:
        """The polling payload. `result` is left to the caller to serialise."""
        return {
            "id": self.id,
            "kind": self.kind,
            "status": self.status,
            "stage": self.stage,
            "progress": round(self.progress, 3),
            "error": self.error,
            "params": self.params,
            "elapsed_s": round((self.finished or time.time()) - self.created, 1),
        }


class JobStore:
    """Thread-safe job registry with a bounded worker pool."""

    def __init__(self, max_workers: int = MAX_WORKERS, max_jobs: int = MAX_JOBS):
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="hab-job")
        self._max_jobs = max_jobs

    def submit(self, kind: str, params: dict, fn: Callable[[Job], Any]) -> Job:
        """Queue `fn`, which receives the Job so it can report progress."""
        job = Job(id=uuid.uuid4().hex[:12], kind=kind, params=params)
        with self._lock:
            self._evict()
            self._jobs[job.id] = job
        self._pool.submit(self._run, job, fn)
        return job

    def _run(self, job: Job, fn: Callable[[Job], Any]) -> None:
        job.status = "running"
        job.stage = "starting"
        try:
            job.result = fn(job)
            job.status = "done"
            job.stage = "done"
            job.progress = 1.0
        except ValueError as exc:
            # ValueError is the pipeline's channel for "no usable data here", and those
            # messages are written for the person who typed the location. Pass it through.
            job.status = "error"
            job.error = str(exc)
            job.stage = "failed"
            logger.info("job %s rejected: %s", job.id, exc)
        except Exception as exc:
            job.status = "error"
            job.error = f"{type(exc).__name__}: {exc}"
            job.stage = "failed"
            logger.exception("job %s failed", job.id)
        finally:
            job.finished = time.time()

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def ids(self) -> set[str]:
        """Ids of every job still held, so a caller keeping results beside the store can
        drop the ones whose job has been evicted rather than growing without bound."""
        with self._lock:
            return set(self._jobs)

    def _evict(self) -> None:
        """Drop expired jobs, then the oldest finished ones if still over the cap.

        Running and queued jobs are never evicted: dropping a job someone is waiting on
        to make room for a newer one would lose work that is already half done.
        """
        now = time.time()
        for jid, job in list(self._jobs.items()):
            if job.finished and now - job.finished > JOB_TTL_S:
                del self._jobs[jid]
        if len(self._jobs) < self._max_jobs:
            return
        finished = sorted(
            (j for j in self._jobs.values() if j.finished),
            key=lambda j: j.finished or 0.0,
        )
        for job in finished[: len(self._jobs) - self._max_jobs + 1]:
            self._jobs.pop(job.id, None)

    def stats(self) -> dict:
        with self._lock:
            jobs = list(self._jobs.values())
        return {
            "total": len(jobs),
            "queued": sum(j.status == "queued" for j in jobs),
            "running": sum(j.status == "running" for j in jobs),
            "done": sum(j.status == "done" for j in jobs),
            "error": sum(j.status == "error" for j in jobs),
            "max_workers": MAX_WORKERS,
        }
