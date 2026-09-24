"""Process-pool scene reading.

Scene reads look I/O-bound, so a thread pool is the obvious choice -- and it is the
wrong one. Measured on this workload, 24 threads gave an effective concurrency of ~1.2:
h5py's HDF5 calls and gzip decompression hold the GIL for most of the wall time, so the
threads serialise behind each other. Switching to processes cut per-scene wall time from
5.8 s to 3.4 s on a cold cache.

Windows uses spawn rather than fork, so worker state cannot be inherited. Everything the
worker needs is rebuilt once per process in `init_worker` and cached at module level;
only lightweight STAC item dicts cross the process boundary on each call.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

import numpy as np

from habforecast.aoi import AOIRegistry
from habforecast.data.olci import SceneObservation, make_reader
from habforecast.data.resample import SwathResampler

logger = logging.getLogger(__name__)

# Per-process cache. Rebuilding the resampler's projected target grid and the STAC
# client for every scene would cost more than the read itself.
_WORKER: dict[str, Any] = {}


def init_worker(
    aoi_path: str, lake_id: str, stac_url: str | None = None, source: str = "planetary-computer"
) -> None:
    """Initialise per-process state. Called once by ProcessPoolExecutor per worker."""
    registry = AOIRegistry.from_geojson(aoi_path)
    grid = registry[lake_id].grid()
    _WORKER["grid"] = grid
    _WORKER["reader"] = make_reader(source, stac_url)
    _WORKER["resampler"] = SwathResampler(grid)
    # Workers inherit the root logger config from spawn; keep them quiet.
    logging.getLogger("habforecast").setLevel(logging.WARNING)


def read_scene(payload: tuple[dict, float, tuple[str, ...]]) -> dict | None:
    """Read one scene in a worker process.

    Returns a plain dict rather than a SceneObservation so the result pickles cheaply
    and the parent does not need the reader's imports to reconstruct it.
    """
    item_dict, min_valid_fraction, aux = payload
    try:
        item = _WORKER["reader"].item_from_payload(item_dict)
        obs = _WORKER["reader"].read(
            item, _WORKER["grid"], _WORKER["resampler"], min_valid_fraction, aux=aux
        )
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}", "item_id": item_dict.get("id", "?")}

    if obs is None:
        return None
    return {
        "item_id": obs.item_id,
        "time": obs.time.isoformat(),
        "platform": obs.platform,
        "log_chl": obs.log_chl,
        "valid": obs.valid,
        "flag_stats": obs.flag_stats,
        "aux": obs.aux,
    }


def payload_to_observation(payload: dict, lake_id: str) -> SceneObservation:
    """Rebuild a SceneObservation in the parent process."""
    return SceneObservation(
        lake_id=lake_id,
        item_id=payload["item_id"],
        time=datetime.fromisoformat(payload["time"]),
        platform=payload["platform"],
        log_chl=np.asarray(payload["log_chl"], dtype=np.float32),
        valid=np.asarray(payload["valid"], dtype=bool),
        flag_stats=payload.get("flag_stats", {}),
        aux={k: np.asarray(v, dtype=np.float32) for k, v in (payload.get("aux") or {}).items()},
    )
