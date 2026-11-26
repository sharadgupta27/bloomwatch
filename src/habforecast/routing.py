"""Which data product, and which model, for the water the user actually selected.

The problem
-----------
Until now every request took the same path: Sentinel-3 OLCI at 300 m, forecast by the one
checkpoint trained on three inland lakes. That is the right answer for Lake Erie and the
wrong answer twice over for a point 400 km into the Gulf of Mexico -- where the 300 m
near-real-time sectors do not reach at all, and where no model in this project has ever
been evaluated.

So the area is classified first (`habforecast.water`), and this module turns the
classification into an explicit, inspectable plan: which product to fetch, which
checkpoint to run, what to fall back to, and what the user must be told. The plan is
returned as data and shown in the app, because a tool that civic authorities act on has to
be able to answer "why this number?" without reading the source.

The three facts the policy is built on
--------------------------------------
All three are measurements this project already made; none is an assumption.

1. **Band-ratio retrievals go blind in optically complex water.** VIIRS mean valid-pixel
   coverage over the ingested lakes runs from 28% on the Curonian Lagoon down to **2.6% on
   Taihu** (`docs/DATA_SOURCES.md`). OLCI's CHL_NN is a neural retrieval built for Case-2
   water and does not do this. So Case-2 water is routed to OLCI even though the OLCI
   checkpoint is the weaker of the two -- a better model on days it cannot see is worth
   nothing.

2. **The 750 m checkpoint forecasts better than the 300 m one.** On the lake held out of
   both runs it roughly triples skill against persistence, and it is the only checkpoint
   here that beats a seasonal climatology. It is offered as an alternative on every
   Case-2 route, with the coverage caveat attached, and it is the primary route wherever
   VIIRS can actually see.

3. **4 km has no skill on this problem.** Same sensor, same lakes, same period as the
   750 m arm, and negative skill against persistence at three of four leads. The 4 km
   checkpoint is therefore listed and *not served*: where 4 km is the only product that
   covers the area, this returns imagery plus a persistence baseline and says so, rather
   than dressing a measured non-result as a forecast.

What this does not do
---------------------
It does not invent a coastal model. **No checkpoint in this project has been trained or
evaluated on marine water** -- every training body is a lake. A coastal or open-ocean
route therefore always carries a caveat saying the forecast is outside its evaluated
domain, and `docs/MODEL_CARD.md` says the same. Routing can pick the best available
combination; it cannot manufacture evidence that does not exist.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path

from habforecast.water import (
    CASE_2,
    COASTAL,
    INLAND,
    OPEN_OCEAN,
    UNKNOWN,
    WaterBody,
    classify,
)

logger = logging.getLogger(__name__)

# The forecaster downsamples three times, so below roughly this many pixels across the
# coarse head has nothing to work with. The same number, for the same reason, gates the
# PACE argument in `habforecast.data.pace`.
MIN_PIXELS_ACROSS = 16

PERSISTENCE = "persistence"

# Where served checkpoints live. The catalogue below records each one at its repo-relative
# path, which is what a reader wants; the container mounts the same directory somewhere
# else, so serving resolves by filename against this root before falling back to the
# recorded path.
MODEL_ROOT = Path(os.getenv("HAB_MODEL_ROOT", "models"))


@dataclass(frozen=True)
class ServedModel:
    """One checkpoint, with the domain it was trained on and the skill it measured.

    Every number here comes from `artifacts/multires/summary.json`, produced by the
    leave-lake-out evaluation in `pipelines/evaluate.py`. They are quoted in the app, so
    they must stay tied to a run rather than to a recollection of one.
    """

    key: str
    label: str
    checkpoint: Path
    sensor: str
    grid_m: float
    product: str  # the data product this checkpoint's receptive field was trained at
    train_bodies: int
    train_domain: str
    vs_persistence: dict[int, float]  # lead days -> Murphy skill, held-out bodies
    vs_climatology: dict[int, float]
    servable: bool
    note: str
    # True for a checkpoint that reads the lake's own multi-year archive (the seasonal
    # climatology input). It serves the registry lakes, which have one, and cannot be
    # offered for an ad-hoc area, which does not: measured without that input it falls
    # back to roughly persistence, below the no-history checkpoint it would replace.
    needs_archive: bool = False

    @property
    def beats_climatology(self) -> bool:
        return all(v > 0 for v in self.vs_climatology.values())

    def as_dict(self) -> dict:
        return {
            "key": self.key,
            "label": self.label,
            "sensor": self.sensor,
            "grid_m": self.grid_m,
            "product": self.product,
            "train_bodies": self.train_bodies,
            "train_domain": self.train_domain,
            "skill_vs_persistence": {str(k): v for k, v in self.vs_persistence.items()},
            "skill_vs_climatology": {str(k): v for k, v in self.vs_climatology.items()},
            "beats_climatology": self.beats_climatology,
            "servable": self.servable,
            "needs_archive": self.needs_archive,
            "note": self.note,
        }


def resolve_checkpoint(model: ServedModel) -> Path:
    """The checkpoint file for `model` wherever this process can actually see it."""
    mounted = MODEL_ROOT / model.checkpoint.name
    return mounted if mounted.exists() else model.checkpoint


# A checkpoint is paired with the product it was trained on and cannot be moved off it.
# The network is convolutional, so its receptive field is a fixed number of *pixels*: run
# the 750 m model on 300 m imagery and every feature it learned is 2.5x the wrong size.
# That pairing is what makes this a source-and-model decision rather than two decisions.
MODELS: dict[str, ServedModel] = {
    "olci-300m-clim": ServedModel(
        key="olci-300m-clim",
        label="OLCI 300 m, 3 lakes, with the lake's own climatology",
        checkpoint=Path("models/best.ckpt"),
        sensor="Sentinel-3 OLCI",
        grid_m=300.0,
        product="olci-300m",
        train_bodies=3,
        train_domain="inland lakes with a multi-year archive",
        # artifacts/eval_anomaly_clim_tta/scores.json, leave-lake-out test (Victoria), TTA.
        vs_persistence={1: 0.255, 3: 0.283, 5: 0.298, 7: 0.318},
        vs_climatology={1: 0.213, 3: 0.189, 5: 0.178, 7: 0.163},
        servable=True,
        needs_archive=True,
        note=(
            "The same network as the 300 m checkpoint, fed the lake's own leave-one-year-out "
            "seasonal climatology for each target date and blending its anchor toward it. On "
            "the lake held out of training it is the only 300 m checkpoint that beats a "
            "seasonal climatology, at every lead. It needs years of archive to build that "
            "input, which the registry lakes have and an ad-hoc area does not; measured with "
            "the input withheld it scores -0.003 to +0.033 against persistence, so the "
            "no-history checkpoint serves those areas instead."
        ),
    ),
    "olci-300m": ServedModel(
        key="olci-300m",
        label="OLCI 300 m, 3 lakes",
        checkpoint=Path("models/olci-300m-nohistory.ckpt"),
        sensor="Sentinel-3 OLCI",
        grid_m=300.0,
        product="olci-300m",
        train_bodies=3,
        train_domain="inland lakes",
        vs_persistence={1: 0.046, 3: 0.037, 5: 0.042, 7: 0.062},
        vs_climatology={1: -0.008, 3: -0.089, 5: -0.122, 7: -0.151},
        servable=True,
        note=(
            "Trained on CHL_NN, the retrieval designed for optically complex water, so it "
            "sees turbid lakes that band-ratio products mask away. Trained on only three "
            "lakes, and it does not beat a seasonal climatology at any lead. It is what an "
            "area with no archive gets: the climatology-aware checkpoint needs history this "
            "area does not have."
        ),
    ),
    "viirs-750m": ServedModel(
        key="viirs-750m",
        label="VIIRS 750 m, 15 lakes",
        checkpoint=Path("models/viirs-750m.ckpt"),
        sensor="VIIRS (SNPP)",
        grid_m=750.0,
        product="viirs-750m",
        train_bodies=15,
        train_domain="inland lakes",
        vs_persistence={1: 0.075, 3: 0.065, 5: 0.084, 7: 0.074},
        vs_climatology={1: 0.191, 3: 0.103, 5: 0.095, 7: 0.012},
        servable=True,
        note=(
            "Five times the lakes and ten times the lake-days of the 300 m checkpoint, and "
            "the only one here that beats a seasonal climatology. Its retrieval is a "
            "band-ratio algorithm, which masks turbid inland water heavily -- 2.6% valid "
            "pixels on Taihu -- so it is not the right choice on every water body."
        ),
    ),
    "viirs-4km": ServedModel(
        key="viirs-4km",
        label="VIIRS 4 km, 15 lakes",
        checkpoint=Path("artifacts/multires/anomaly-viirs4km.ckpt"),
        sensor="VIIRS (SNPP)",
        grid_m=4000.0,
        product="viirs-4km",
        train_bodies=15,
        train_domain="inland lakes",
        vs_persistence={1: -0.015, 3: 0.002, 5: -0.032, 7: -0.011},
        vs_climatology={1: 0.006, 3: -0.179, 5: -0.082, 7: -0.179},
        servable=False,
        note=(
            "Kept for the record, not for service. Same sensor, lakes and period as the "
            "750 m arm and negative skill against persistence at three of four leads: a "
            "measured non-result, and serving it as a forecast would be dishonest."
        ),
    ),
}


@dataclass(frozen=True)
class SourceStep:
    """One product in the fetch plan, and what it is doing there."""

    product: str
    label: str
    resolution_m: float
    role: str  # primary | fallback
    reason: str
    record: str = ""

    def as_dict(self) -> dict:
        return {
            "product": self.product,
            "label": self.label,
            "resolution_m": self.resolution_m,
            "role": self.role,
            "reason": self.reason,
            "record": self.record,
        }


@dataclass(frozen=True)
class Route:
    """The plan: what to fetch, what to run it through, and what to disclose."""

    water: WaterBody
    sources: tuple[SourceStep, ...]
    model: ServedModel | None
    baseline: str | None  # set when no checkpoint is defensible here
    reasons: list[str] = field(default_factory=list)
    caveats: list[str] = field(default_factory=list)
    alternatives: list[dict] = field(default_factory=list)

    @property
    def product(self) -> str:
        return self.sources[0].product

    @property
    def resolution_m(self) -> float:
        return self.sources[0].resolution_m

    @property
    def forecasts(self) -> bool:
        """Whether a learned model runs at all, or only a baseline."""
        return self.model is not None

    def as_dict(self) -> dict:
        return {
            "water": self.water.as_dict(),
            "sources": [s.as_dict() for s in self.sources],
            "model": self.model.as_dict() if self.model else None,
            "baseline": self.baseline,
            "reasons": list(self.reasons),
            "caveats": list(self.caveats),
            "alternatives": list(self.alternatives),
        }


def _product(key: str):
    from habforecast.data.coastwatch import PRODUCTS

    return PRODUCTS[key]


def _step(key: str, role: str, reason: str) -> SourceStep:
    spec = _product(key)
    return SourceStep(
        product=key,
        label=spec.label,
        resolution_m=spec.resolution_m,
        role=role,
        reason=reason,
        record=spec.record,
    )


# Every step in one route's fetch plan must be at the route's own resolution, because the
# model is pinned to it. A coarser product is not a fallback -- it is a different route,
# with a different checkpoint, and it belongs in `alternatives` where a person chooses it.
ARCHIVE_STEP = SourceStep(
    product="planetary-computer-archive",
    label="Sentinel-3 OLCI L2 WFR archive",
    resolution_m=300.0,
    role="fallback",
    record="2018 to a few months behind today",
    reason=(
        "The same sensor and retrieval at the same 300 m, with this project's own WQSF "
        "quality control and a flag-derived water mask. Global and complete back to 2018, "
        "but the mirror runs months behind, so it answers for old dates and for outages."
    ),
)


def coarsest_useful_m(side_km: float) -> float:
    """The largest cell size that still leaves the network something to see.

    A 12 km lake at 4 km is three pixels across, and a three-pixel field has no spatial
    structure to forecast -- which is exactly what the 4 km arm measured.
    """
    return max(side_km * 1000.0 / MIN_PIXELS_ACROSS, 1.0)


def plan(
    bbox: tuple[float, float, float, float],
    when: date | None = None,
    water: WaterBody | None = None,
) -> Route:
    """Choose the product and model for `bbox` = (west, south, east, north).

    Pure and offline: no network, no imagery, no model loaded. `when` is accepted so the
    reasoning can mention recency, but the near-real-time-versus-archive choice stays with
    `habforecast.inference.adhoc`, which is the only place that knows what a source
    actually holds today.
    """
    body = water if water is not None else classify(bbox)
    limit_m = coarsest_useful_m(body.side_km)
    reasons: list[str] = []
    caveats: list[str] = []

    if body.enclosed_sea:
        reasons.append(body.evidence[0])
    elif body.setting == UNKNOWN:
        caveats.append(
            "The water body could not be classified, so the most conservative route was "
            "taken: 300 m OLCI, which is valid over both inland and coastal water."
        )
    else:
        where = {
            INLAND: "Inland water",
            COASTAL: "Coastal water",
            OPEN_OCEAN: "Open ocean",
        }[body.setting]
        distance = (
            ""
            if body.coast_distance_km is None
            else f", {body.coast_distance_km:.0f} km from shore"
        )
        reasons.append(f"{where}{distance}, about {body.side_km:.0f} km across.")

    case2 = body.optical_class == CASE_2 or body.setting == UNKNOWN
    if case2:
        route = _case2_route(body, limit_m, when, reasons, caveats)
    else:
        route = _case1_route(body, limit_m, reasons, caveats)

    if body.setting in (COASTAL, OPEN_OCEAN):
        caveats.append(
            "No model in this project has been trained or evaluated on marine water -- "
            "every training body is a lake. Treat a forecast here as an extrapolation "
            "outside its measured domain, not as a validated product."
        )
    if body.mixed:
        caveats.append(
            "The selected area straddles the shoreline, so it mixes inland and marine "
            "water. Chlorophyll retrievals and thresholds differ between the two."
        )
    return route


def _case2_route(
    body: WaterBody, limit_m: float, when: date | None, reasons: list[str], caveats: list[str]
) -> Route:
    """Optically complex water: the retrieval decides, and only one retrieval is built for it."""
    reasons.append(
        "Optically complex (Case-2) water: sediment and dissolved organic matter drive "
        "reflectance independently of chlorophyll. OLCI's CHL_NN is the retrieval designed "
        "for it; band-ratio products mask this water heavily -- measured at 2.6% valid "
        "pixels on Taihu."
    )
    sources = [
        _step(
            "olci-300m",
            "primary",
            "300 m CHL_NN, near-real-time, and the only retrieval here validated on turbid water.",
        ),
        ARCHIVE_STEP,
    ]
    model = MODELS["olci-300m"]
    clim = MODELS["olci-300m-clim"]
    caveats.append(
        "An area drawn on the map has no multi-year archive, so the checkpoint that reads "
        f"one ({clim.label}: +{clim.vs_persistence[7]:.2f} skill against persistence at "
        "7 days on the lake held out of training) cannot run here. The registry lakes are "
        "served by it; this area gets the no-history checkpoint."
    )

    alternatives = []
    if limit_m >= MODELS["viirs-750m"].grid_m:
        why = (
            f"This area is {body.side_km * 1000 / 750:.0f} pixels across at 750 m, so the "
            "better-trained checkpoint can run here. It is measurably the better forecaster "
            "-- roughly triple the skill against persistence on the lake held out of both "
            "runs, and the only checkpoint here that beats a seasonal climatology. It is not "
            "the default on Case-2 water because its band-ratio retrieval masks turbid water "
            "away; on clear water this size it is the better choice."
        )
        end = _product("viirs-750m").record_end
        wanted = when or datetime.now(UTC).date()
        if end is not None and wanted > end:
            # Found by running it: asking this route for today returned imagery from the
            # previous August. The checkpoint is still the better forecaster, but the
            # product behind it stopped publishing, so it answers history and not today.
            why += (
                f" Note that CoastWatch stopped publishing this product on {end}, so "
                f"choosing it for {wanted} returns imagery about "
                f"{(wanted - end).days} days old -- it answers questions about the past, "
                "not a live forecast."
            )
        alternatives.append({**MODELS["viirs-750m"].as_dict(), "why": why})
    else:
        caveats.append(
            f"At {body.side_km:.0f} km across, this area is "
            f"{body.side_km * 1000 / 750:.0f} pixels wide at 750 m, below the "
            f"{MIN_PIXELS_ACROSS} the forecaster needs, so the better-trained 750 m "
            "checkpoint cannot be offered here. 300 m is the only rung that resolves it."
        )
    return Route(
        water=body,
        sources=tuple(sources),
        model=model,
        baseline=None,
        reasons=reasons,
        caveats=caveats,
        alternatives=alternatives,
    )


def _case1_route(body: WaterBody, limit_m: float, reasons: list[str], caveats: list[str]) -> Route:
    """Open ocean: the coarse global grids are the only products that cover it.

    And at 4 km this project has a measurement rather than a guess -- no skill -- so what
    comes back is imagery and a persistence baseline, labelled as such.
    """
    reasons.append(
        "Case-1 water, where chlorophyll dominates the optics and the standard band-ratio "
        "retrievals are the validated ones. The 300 m and 750 m products are sectored over "
        "coastal and inland water and do not reach here; the global 4 km grids do."
    )
    sources = [
        _step("viirs-4km", "primary", "Global daily coverage, fourteen years deep."),
        _step("olci-4km", "fallback", "Same grid from the OLCI sensor family, since 2019."),
        ARCHIVE_STEP,
    ]
    if limit_m < 4000.0:
        caveats.append(
            f"This area is {body.side_km:.0f} km across, which is only "
            f"{body.side_km / 4:.0f} pixels at 4 km -- below the {MIN_PIXELS_ACROSS} the "
            "forecaster needs. Draw a larger area, or accept that the result is close to a "
            "single number."
        )
    reasons.append(
        "No forecast model is served here. The 4 km checkpoint was trained and evaluated "
        "under the same protocol as the others and scored negative skill against "
        "persistence at three of four leads, so what is returned is the imagery and a "
        "persistence baseline -- a baseline whose error is characterised beats a model "
        "whose is not."
    )
    caveats.append(
        "Persistence, not a learned forecast: the most recent observed field is carried "
        "forward unchanged to every lead time."
    )
    return Route(
        water=body,
        sources=tuple(sources),
        model=None,
        baseline=PERSISTENCE,
        reasons=reasons,
        caveats=caveats,
        alternatives=[
            {
                **MODELS["viirs-4km"].as_dict(),
                "why": (
                    "The checkpoint that exists for this grid, and the reason nothing is "
                    "served: negative skill against persistence at three of four leads."
                ),
            }
        ],
    )


REGISTRY_ARCHIVE_STEP = SourceStep(
    product="bloom-watch-archive",
    label="Bloom Watch archive (ingested Sentinel-3 OLCI L2)",
    resolution_m=300.0,
    role="primary",
    record="years of daily composites for each monitored lake",
    reason=(
        "This lake is monitored: its imagery is already ingested through the training "
        "pipeline, with full WQSF quality control and a water mask built from years of "
        "retrievals. Nothing is downloaded."
    ),
)


def registry_route(route: Route, lake_name: str, note: str | None = None) -> Route:
    """The route for a monitored lake served from its own archive.

    Such a lake has the multi-year history the climatology checkpoint needs, so it gets
    that checkpoint -- the one the outlook, /forecast and /advisory already use -- rather
    than the no-history model a drawn area falls back to. The drawn-area caveat about the
    missing archive is dropped because it is no longer true; every other caveat stays.
    """
    model = MODELS["olci-300m-clim"]
    return Route(
        water=route.water,
        sources=(REGISTRY_ARCHIVE_STEP, *[s for s in route.sources if s.role != "primary"]),
        model=model,
        baseline=None,
        reasons=[
            *route.reasons,
            note or f"{lake_name} is a monitored lake, so it is forecast from its own archive.",
            f"Model: {model.label}, which reads the lake's seasonal climatology.",
        ],
        caveats=[c for c in route.caveats if "has no multi-year archive" not in c],
        alternatives=[],
    )


def override_model(route: Route, key: str) -> Route:
    """The same area, forced onto a named checkpoint -- and onto that checkpoint's product.

    Overriding the model without moving the source would feed the network imagery at a
    scale it was never trained on, which is the failure this module exists to prevent. So
    an override moves both, and says on the route that a person chose it.
    """
    model = MODELS.get(key)
    if model is None:
        raise KeyError(f"unknown model {key!r}")
    if not model.servable:
        raise ValueError(f"{key} is catalogued but not served: {model.note}")
    if model.needs_archive:
        raise ValueError(
            f"{key} reads the lake's multi-year archive, which an ad-hoc area does not have; "
            "it serves the registry lakes through /forecast and /advisory"
        )
    return Route(
        water=route.water,
        sources=(
            _step(
                model.product,
                "primary",
                f"Chosen explicitly: {model.label} is pinned to this product.",
            ),
            ARCHIVE_STEP,
        ),
        model=model,
        baseline=None,
        reasons=[*route.reasons, f"{model.label} was selected by hand, overriding the router."],
        caveats=[*route.caveats],
        alternatives=list(route.alternatives),
    )


def available_models() -> list[dict]:
    """The catalogue, with each checkpoint's presence on disk resolved."""
    out = []
    for m in MODELS.values():
        path = resolve_checkpoint(m)
        entry = m.as_dict()
        entry["present"] = path.exists()
        entry["checkpoint"] = str(path)
        out.append(entry)
    return out
