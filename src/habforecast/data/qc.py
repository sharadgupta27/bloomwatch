"""Quality masking for OLCI Level-2 water products via the WQSF flag field.

WQSF ("Water Quality and Science Flags") is a uint64 bitfield carried alongside every
OLCI L2 water product. Getting this right is the highest-leverage step in the pipeline:
unmasked cloud-edge and sun-glint pixels produce spuriously high chlorophyll that looks
exactly like a bloom, and a model trained on those learns to predict cloud artefacts.

Three traps specific to *inland* water, all verified empirically against Lake Erie
(see tests/unit/test_qc.py, which pins the behaviour):

1. ``WATER`` is an open-ocean flag and is **never set** over lakes. Requiring it
   discards 100% of inland pixels. Require ``INLAND_WATER`` instead.

2. ``LAND`` **is** set over inland water. On a Lake Erie scene, 75.9% of window pixels
   carry LAND while 54.9% carry INLAND_WATER, and the 40.7% intersection is precisely
   the set with a finite CHL_NN retrieval. Rejecting LAND therefore throws away every
   usable lake pixel. Land is excluded by *requiring* INLAND_WATER, not by rejecting
   LAND.

3. ``TURBID_ATM`` and ``OC4ME_FAIL`` are **bloom-correlated**. A dense surface scum
   raises reflectance enough to trip the turbid-atmosphere test, and OC4ME_FAIL simply
   means the open-ocean band-ratio algorithm failed -- which it always does in case-2
   water, and which is exactly why we read CHL_NN instead. Rejecting either flag
   preferentially deletes the strongest blooms, biasing the training set toward clear
   water. Both are deliberately kept.

The flags we *do* reject are those that make a retrieval physically meaningless
(cloud, ice, glint) or that the processor itself marks as failed for the neural-net
product we depend on (``OCNN_FAIL``).
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np

# Flags that invalidate a pixel for inland-water chlorophyll retrieval.
# NOTE: LAND, TURBID_ATM, OC4ME_FAIL, KDM_FAIL, BPAC_ON and HIGHRW are intentionally
# absent -- see the module docstring for why each would do more harm than good.
DEFAULT_REJECT_FLAGS: tuple[str, ...] = (
    "INVALID",  # product generation failed
    "CLOUD",  # confident cloud
    "CLOUD_AMBIGUOUS",
    "CLOUD_MARGIN",  # halo around cloud -- the classic false-bloom source
    "SNOW_ICE",
    "SUSPECT",
    "HISOLZEN",  # high solar zenith; atmospheric correction degrades badly
    "SATURATED",
    "HIGHGLINT",  # specular sun glint
    "WHITECAPS",
    "AC_FAIL",  # atmospheric correction failed
    "OCNN_FAIL",  # the CHL_NN retrieval itself failed
)

# At least one of these must be set for a pixel to count as water.
# INLAND_WATER carries lakes; WATER is retained so the same code path works for the
# coastal/marine AOIs the registry may grow later.
DEFAULT_REQUIRE_ANY: tuple[str, ...] = ("INLAND_WATER", "WATER")

# A stricter profile for ablation: additionally drops adjacency-effect pixels near the
# shoreline and the turbid-atmosphere set. Useful to quantify how much bloom signal the
# conservative choice above actually retains -- report both in the model card.
STRICT_REJECT_FLAGS: tuple[str, ...] = (*DEFAULT_REJECT_FLAGS, "ADJAC", "TURBID_ATM")


@dataclass(frozen=True)
class FlagTable:
    """Maps WQSF flag names to bit masks, read from the product's own attributes.

    Reading the table from the file rather than hard-coding bit positions means the
    pipeline survives a processing-baseline change that reorders or adds flags.
    """

    masks: dict[str, int]

    @classmethod
    def from_attrs(cls, flag_masks, flag_meanings) -> FlagTable:
        if isinstance(flag_meanings, bytes):
            flag_meanings = flag_meanings.decode("utf-8")
        names = str(flag_meanings).split()
        values = [int(v) for v in np.asarray(flag_masks).ravel()]
        if len(names) != len(values):
            raise ValueError(
                f"WQSF flag table malformed: {len(names)} names vs {len(values)} masks"
            )
        return cls(masks=dict(zip(names, values, strict=True)))

    def bitmask_for(self, names: Iterable[str]) -> np.uint64:
        """OR the bit masks for `names`, ignoring any this product does not define."""
        acc = np.uint64(0)
        for n in names:
            if n in self.masks:
                acc |= np.uint64(self.masks[n])
        return acc

    def missing(self, names: Iterable[str]) -> list[str]:
        """Which of `names` this product's flag table does not define."""
        return [n for n in names if n not in self.masks]


def valid_water_mask(
    wqsf: np.ndarray,
    table: FlagTable,
    reject_flags: Sequence[str] = DEFAULT_REJECT_FLAGS,
    require_any: Sequence[str] = DEFAULT_REQUIRE_ANY,
) -> np.ndarray:
    """Boolean mask of pixels usable for inland chlorophyll retrieval.

    A pixel is valid when it carries at least one `require_any` flag AND none of the
    `reject_flags`.
    """
    wqsf = np.asarray(wqsf, dtype=np.uint64)
    reject_bits = table.bitmask_for(reject_flags)
    require_bits = table.bitmask_for(require_any)

    is_water = (wqsf & require_bits) != 0
    is_rejected = (wqsf & reject_bits) != 0
    return is_water & ~is_rejected


def flag_frequencies(wqsf: np.ndarray, table: FlagTable) -> dict[str, float]:
    """Fraction of pixels carrying each flag -- ingest telemetry and drift input.

    Tracking these over time is a cheap early warning: a jump in CLOUD_MARGIN or
    AC_FAIL frequency usually signals an upstream processing change, not a real-world
    one -- in which case the response is to investigate the source, not to retrain.

    Averaged per day by `store.composite_day` and written to the `<lake>.flags.json`
    sidecar beside the cube, where `monitoring.drift.compare_flag_frequencies` reads
    them. Cubes ingested before that sidecar existed simply have no file, and the flag
    comparison is skipped rather than silently reporting no drift.
    """
    wqsf = np.asarray(wqsf, dtype=np.uint64)
    n = wqsf.size
    if n == 0:
        return {}
    return {
        name: float(((wqsf & np.uint64(bit)) != 0).sum()) / n for name, bit in table.masks.items()
    }
