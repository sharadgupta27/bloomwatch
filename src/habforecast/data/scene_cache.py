"""Keep every scene a user's request downloaded, so asking again costs a disk read.

A live forecast reads 15-30 granules, each a minute or more of range requests against
the EUMETSAT Data Store (measured 2026-11-27: 97 s for one scene over western Lake Erie,
61 s of it a single stalled request). The scenes of a past fortnight never change, so a
repeat of the same area -- the same lake searched again, a box redrawn in the same place,
a second visitor -- should not pay that again.

What is stored is the scene *after* windowing, quality control and resampling onto the
request's grid: a few hundred kB, not the hundreds of MB of the granule. The key is the
scene id plus the exact grid, so a different area never receives another area's pixels.
A scene that was rejected (no coverage, or too cloudy for the threshold asked) is stored
as a marker, so it is not downloaded again just to be rejected again.

`HAB_SCENE_CACHE` sets the directory (empty disables); `HAB_SCENE_CACHE_MB` caps its size,
oldest entries removed first.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
import threading
from datetime import datetime
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_DIR = "data/cache/scenes"
DEFAULT_MB = 2048
# Prune at most this often (in writes): a directory scan is cheap, not free.
PRUNE_EVERY = 25

_writes = 0
_lock = threading.Lock()


def directory() -> Path | None:
    root = os.getenv("HAB_SCENE_CACHE", DEFAULT_DIR)
    return Path(root) if root else None


def key(source: str, item_id: str, grid, max_distance_m: float, aux: tuple[str, ...] = ()) -> str:
    """Everything that changes the resampled scene, and nothing that does not."""
    parts = [
        source,
        item_id,
        grid.epsg,
        round(grid.x_min, 3),
        round(grid.y_min, 3),
        grid.width,
        grid.height,
        round(grid.resolution, 3),
        round(float(max_distance_m), 3),
        sorted(aux),
    ]
    return hashlib.sha256(json.dumps(parts).encode("utf-8")).hexdigest()[:32]


def load(k: str, min_valid_fraction: float):
    """("hit", obs) | ("rejected", None) | None when nothing usable is stored.

    A stored rejection is reused only for a request at least as strict as the one that
    produced it; a looser request reads the scene again.
    """
    root = directory()
    if root is None:
        return None
    path = root / f"{k}.npz"
    if not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=False) as z:
            meta = json.loads(str(z["meta"]))
            if meta.get("rejected_below") is not None:
                if min_valid_fraction >= meta["rejected_below"]:
                    return ("rejected", None)
                return None
            from habforecast.data.olci import SceneObservation

            obs = SceneObservation(
                lake_id=meta["lake_id"],
                item_id=meta["item_id"],
                time=datetime.fromisoformat(meta["time"]),
                platform=meta["platform"],
                log_chl=z["log_chl"],
                valid=z["valid"],
                flag_stats=meta.get("flag_stats", {}),
                aux={name: z[f"aux_{name}"] for name in meta.get("aux", [])},
            )
    except (OSError, ValueError, KeyError) as exc:
        logger.info("unreadable scene cache entry %s (%s); reading the scene again", k, exc)
        return None
    if obs.valid_fraction < min_valid_fraction:
        return ("rejected", None)
    with contextlib.suppress(OSError):
        os.utime(path)  # recently used: pruned last
    return ("hit", obs)


def save(k: str, obs, min_valid_fraction: float) -> None:
    """Store a scene, or (obs None) the fact that it was rejected at this threshold."""
    global _writes
    root = directory()
    if root is None:
        return
    try:
        root.mkdir(parents=True, exist_ok=True)
        if obs is None:
            meta = {"rejected_below": float(min_valid_fraction)}
            arrays: dict[str, np.ndarray] = {}
        else:
            meta = {
                "lake_id": obs.lake_id,
                "item_id": obs.item_id,
                "time": obs.time.isoformat(),
                "platform": obs.platform,
                "flag_stats": {k2: float(v) for k2, v in obs.flag_stats.items()},
                "aux": sorted(obs.aux),
                "rejected_below": None,
            }
            arrays = {"log_chl": obs.log_chl, "valid": obs.valid}
            arrays |= {f"aux_{name}": a for name, a in obs.aux.items()}
        tmp = root / f"{k}.{os.getpid()}.{threading.get_ident()}.part"
        with tmp.open("wb") as fh:
            np.savez_compressed(fh, meta=np.array(json.dumps(meta)), **arrays)
        tmp.replace(root / f"{k}.npz")
    except OSError as exc:  # a full or read-only disk serves, just without the cache
        logger.info("could not store scene %s (%s)", k, exc)
        return
    with _lock:
        _writes += 1
        due = _writes % PRUNE_EVERY == 0
    if due:
        prune()


def prune(limit_mb: float | None = None) -> int:
    """Remove the least recently used entries until the cache fits. Returns files removed."""
    root = directory()
    if root is None or not root.exists():
        return 0
    limit = limit_mb if limit_mb is not None else float(os.getenv("HAB_SCENE_CACHE_MB", DEFAULT_MB))
    entries = []
    for f in root.glob("*.npz"):
        try:
            st = f.stat()
        except OSError:
            continue
        entries.append((st.st_mtime, st.st_size, f))
    total = sum(e[1] for e in entries)
    removed = 0
    for _mtime, size, f in sorted(entries):
        if total <= limit * 2**20:
            break
        try:
            f.unlink()
            total -= size
            removed += 1
        except OSError:
            continue
    return removed
