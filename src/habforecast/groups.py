"""Lake groups: the levels of the multilevel forecaster and of its calibration.

One network is trained on every lake, and it is told three things about the lake a sample
comes from -- its trophic class, its thermal regime and its continent. Each is a learned
embedding; the embeddings are summed and modulate the network's features (see
`models.anomaly.GroupFiLM`). That is a multilevel model in the ordinary statistical sense:
the shared weights are the population level, and each group gets an offset from it that
is estimated from all of that group's lakes together.

Why these three, and why not one model per group:

* **Trophic class** sets the base rate. A hypereutrophic lake sits above 20 mg/m^3 for
  half the season; an oligotrophic one essentially never does. It is also the grouping
  the probability calibration is fitted on (`calibration.py`), because a single global
  map measurably did not transfer across base rates (Chapala, 46% prevalence).
* **Thermal regime** stands in for the growth-rate physics: cyanobacteria outcompete other
  phytoplankton above roughly 20 degC, and a tropical lake has no winter reset. It is
  derived from the lake's own ERA5 annual mean temperature, not from latitude, so a
  high-altitude tropical lake (Titicaca, Dianchi) lands where its water actually is.
* **Continent** carries what neither of those does: bloom community, land use, the
  product's regional retrieval biases.

Separate models per group were the alternative, and the data rules them out: a group of
three lakes would train a network on three lakes, which is the regime the expanded
registry was built to escape. Partial pooling lets a small group borrow from the rest.

**Index 0 is "unknown" in every vocabulary**, and training withholds each factor at
random (`group_dropout`), so the network also learns the population-level forecast. An
ad-hoc area drawn in the app, which has no registry entry, is served exactly that --
the same thing a checkpoint trained before this module existed would have given it.

The vocabularies are append-only. A checkpoint stores embeddings by index, so reordering
or removing an entry silently reassigns every trained group.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

UNKNOWN = "unknown"

TROPHIC_CLASSES: tuple[str, ...] = (
    UNKNOWN,
    "oligotrophic",
    "mesotrophic",
    "eutrophic",
    "hypereutrophic",
)

# Annual-mean 2 m air temperature over the lake, degC. Boundaries: 10 degC separates lakes
# with a real ice or cold-mixing winter from those without; 20 degC is where cyanobacteria
# hold a growth advantage for most of the year.
THERMAL_CLASSES: tuple[str, ...] = (UNKNOWN, "cold", "temperate", "warm")
THERMAL_EDGES_C: tuple[float, float] = (10.0, 20.0)

CONTINENTS: tuple[str, ...] = (
    UNKNOWN,
    "africa",
    "asia",
    "europe",
    "north_america",
    "south_america",
    "oceania",
)

FACTORS: dict[str, tuple[str, ...]] = {
    "trophic": TROPHIC_CLASSES,
    "thermal": THERMAL_CLASSES,
    "continent": CONTINENTS,
}
FACTOR_NAMES: tuple[str, ...] = tuple(FACTORS)


def thermal_class(mean_t2m_c: float | None) -> str:
    """Thermal regime from an annual mean air temperature (degC); unknown if missing."""
    if mean_t2m_c is None or mean_t2m_c != mean_t2m_c:  # None or NaN
        return UNKNOWN
    lo, hi = THERMAL_EDGES_C
    return "cold" if mean_t2m_c < lo else ("temperate" if mean_t2m_c < hi else "warm")


def _index(vocab: tuple[str, ...], value: str | None) -> int:
    value = (value or UNKNOWN).strip().lower()
    return vocab.index(value) if value in vocab else 0


@dataclass(frozen=True)
class LakeGroups:
    """A lake's level in each factor, as vocabulary names."""

    trophic: str = UNKNOWN
    thermal: str = UNKNOWN
    continent: str = UNKNOWN

    @classmethod
    def from_properties(cls, props: Mapping) -> LakeGroups:
        """From a registry feature's properties. Absent keys are unknown, never guessed."""
        return cls(
            trophic=str(props.get("trophic_state") or UNKNOWN).lower(),
            thermal=str(props.get("thermal_class") or UNKNOWN).lower(),
            continent=str(props.get("continent") or UNKNOWN).lower(),
        )

    def ids(self) -> list[int]:
        """Vocabulary indices in FACTOR_NAMES order -- what the model embeds."""
        return [
            _index(TROPHIC_CLASSES, self.trophic),
            _index(THERMAL_CLASSES, self.thermal),
            _index(CONTINENTS, self.continent),
        ]

    def key(self, factor: str) -> str:
        """`factor:level`, the key a per-group calibration is stored under."""
        return f"{factor}:{getattr(self, factor)}"


UNKNOWN_GROUPS = LakeGroups()
