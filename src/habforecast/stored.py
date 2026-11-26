"""Answer from stored data whenever stored data can answer.

Forty-six lakes have years of quality-controlled imagery on disk. A request for an area
inside one of them -- however it was chosen: drawn, searched, uploaded, or sent to the
API -- does not need a single scene downloaded, and from the archive it gets the stronger
model (the one that reads the lake's own seasonal climatology) in about a second instead
of minutes.

The match is deliberately conservative, because a wrong "yes" is worse than a slow one:

* **Coverage.** The requested area must lie (almost) wholly inside the lake's extent.
  Anything materially outside it is water the archive never recorded.
* **Freshness.** The archive must hold the date asked for. A request for "the newest
  imagery" is only answered from the archive when the archive *is* current (newest day
  within `FRESH_DAYS`); otherwise it would silently hand back a weeks-old forecast where
  the live path would have found last week's scenes. In that case the request goes live,
  and `StoredMatch.reason` says why and how to get the instant answer instead.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import date

from habforecast.outlook import STALE_AFTER_DAYS

# Share of the requested area that must fall inside a lake's extent.
COVER_MIN = 0.9
# "Newest imagery" is answered from the archive only if the archive is this fresh.
FRESH_DAYS = STALE_AFTER_DAYS

BBox = tuple[float, float, float, float]  # west, south, east, north


@dataclass(frozen=True)
class StoredMatch:
    lake_id: str
    lake_name: str
    coverage: float  # share of the requested area inside the lake's extent
    newest: date | None  # newest day the archive holds
    usable: bool  # True: answer from the archive
    anchor: date | None  # the day to forecast from, when usable
    reason: str  # one sentence for the route panel, either way

    def as_dict(self) -> dict:
        return {
            "lake_id": self.lake_id,
            "lake_name": self.lake_name,
            "coverage": round(self.coverage, 3),
            "newest": self.newest.isoformat() if self.newest else None,
            "usable": self.usable,
            "anchor": self.anchor.isoformat() if self.anchor else None,
            "reason": self.reason,
        }


def _area(b: BBox) -> float:
    """Approximate area of a lon/lat box (degrees², scaled by cos(latitude))."""
    w, s, e, n = b
    return max(e - w, 0.0) * max(n - s, 0.0) * math.cos(math.radians((s + n) / 2.0))


def coverage(request: BBox, lake: BBox) -> float:
    """Share of `request` that lies inside `lake`, 0..1."""
    w = max(request[0], lake[0])
    s = max(request[1], lake[1])
    e = min(request[2], lake[2])
    n = min(request[3], lake[3])
    inside = _area((w, s, e, n)) if e > w and n > s else 0.0
    whole = _area(request)
    return inside / whole if whole > 0 else 0.0


def match(
    request: BBox,
    lakes: Iterable[tuple[str, str, BBox]],
    newest_day: Callable[[str], date | None],
    anchor: date | None,
    today: date,
) -> StoredMatch | None:
    """The monitored lake that can answer `request`, or None if none covers it.

    `lakes` is (lake_id, name, bbox). When several cover the area (nested or overlapping
    registry entries), the best-covering, then the smallest, wins: it is the one whose
    grid fits the request most tightly. A returned match with `usable=False` still names
    the lake, so the caller can explain why the archive was not used.
    """
    candidates = []
    for lake_id, name, bbox in lakes:
        c = coverage(request, bbox)
        if c >= COVER_MIN:
            candidates.append((-c, _area(bbox), lake_id, name))
    if not candidates:
        return None
    neg_c, _, lake_id, name = min(candidates)
    c = -neg_c
    newest = newest_day(lake_id)

    if newest is None:
        return StoredMatch(
            lake_id,
            name,
            c,
            None,
            False,
            None,
            f"{name} is monitored, but no archive is installed for it here.",
        )
    if anchor is not None:
        if anchor <= newest:
            return StoredMatch(
                lake_id,
                name,
                c,
                newest,
                True,
                anchor,
                f"This area lies inside {name}, so it is forecast from the "
                f"stored archive — nothing is downloaded.",
            )
        return StoredMatch(
            lake_id,
            name,
            c,
            newest,
            False,
            None,
            f"This area lies inside {name}, but its stored archive ends "
            f"{newest.isoformat()}, before the date asked for, so imagery "
            "is downloaded.",
        )
    age = (today - newest).days
    if age <= FRESH_DAYS:
        return StoredMatch(
            lake_id,
            name,
            c,
            newest,
            True,
            newest,
            f"This area lies inside {name}, so it is forecast from the "
            f"stored archive (newest image {newest.isoformat()}) — nothing "
            "is downloaded.",
        )
    return StoredMatch(
        lake_id,
        name,
        c,
        newest,
        False,
        None,
        f"This area lies inside {name}, but its stored archive ends "
        f"{newest.isoformat()} ({age} days ago), so the newest imagery is "
        f"downloaded. Pick a date on or before {newest.isoformat()} to get "
        "the forecast from the archive in about a second.",
    )
