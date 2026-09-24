"""Auxiliary input layers, derived from assets already inside the OLCI granule.

The forecaster has always been univariate: fourteen daily frames of log10 chlorophyll,
an observation mask, a water mask and the day of year. Everything else that drives a
cyanobacterial bloom -- how turbid the water is, how deep the light gets, whether the
surface is a scum or a sediment plume -- has to be inferred from where chlorophyll has
been, which is a lot to ask of one channel.

**The cheap place to get more is the granule we already download.** A Sentinel-3 OLCI
L2 WFR product carries 34 assets; the ingest opens four of them. The rest include all
21 reflectance bands, total suspended matter, transparency, PAR and the meteorological
tie-point grid -- at 300 m, over inland water, with no additional credential and no
second provider to be down.

What is deliberately *not* here, having been measured on 2026-08-30:

* **Marine SST.** NOAA's ACSPO, geo-polar blended and Coral Reef Watch products all
  return cells over western Lake Erie and every one of them is masked: 0 valid of 630.
  These are sea-surface products and inland water is not in them. Lake surface
  temperature at a useful resolution needs a different family entirely (MODIS/VIIRS LST,
  or GLSEA for the Great Lakes only), and none of it is on the open ERDDAP.
* **PACE.** The mission is real and current -- the newest L3 BGC granule on 2026-08-30
  was two days old -- but a download redirects to Earthdata Login and answers 401, and
  the L3 mapped product is 0.1 degrees, about 11 km. A 12 km lake is one pixel. See
  `habforecast.data.pace` for the gated reader and the resolution argument.

Each layer below states what it contributes that chlorophyll cannot, because an input
channel that duplicates an existing one costs training data and buys nothing.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

# OLCI band centres used by the indices, in nanometres.
OA10_NM, OA11_NM, OA12_NM = 681.25, 708.75, 753.75


def _mci(bands: dict[str, np.ndarray]) -> np.ndarray:
    """Maximum Chlorophyll Index: the height of the 709 nm peak above a 681-754 baseline.

    This is the standard OLCI index for surface cyanobacteria, and it is *not* a
    restatement of CHL_NN. CHL_NN is a neural-net inversion tuned for optically complex
    water and it degrades in exactly the conditions that matter most -- a dense surface
    scum saturates it, and its training set does not contain floating vegetation. MCI is
    a three-band arithmetic on reflectance with no inversion in it: when there is
    near-infrared reflectance where clear water should absorb, MCI rises. A high MCI with
    an unremarkable CHL_NN is the signature of a scum the retrieval has under-read.
    """
    r681, r709, r754 = bands["oa10"], bands["oa11"], bands["oa12"]
    slope = (OA11_NM - OA10_NM) / (OA12_NM - OA10_NM)
    return r709 - r681 - (r754 - r681) * slope


def _identity(name: str) -> Callable[[dict[str, np.ndarray]], np.ndarray]:
    def take(bands: dict[str, np.ndarray]) -> np.ndarray:
        return bands[name]

    return take


@dataclass(frozen=True)
class AuxLayer:
    """One extra input channel: where it comes from and what it adds."""

    key: str
    # asset name -> HDF5 dataset inside it. Each asset is an extra read per scene, which
    # is why layers are opt-in per ingest rather than always on.
    sources: dict[str, str]
    compute: Callable[[dict[str, np.ndarray]], np.ndarray]
    why: str
    units: str
    # Physically plausible range. Values outside it are dropped rather than clipped: a
    # reflectance of 40 is a decoding error, and clipping would feed the model a lie
    # instead of a gap it already knows how to handle.
    valid_range: tuple[float, float]

    @property
    def assets(self) -> tuple[str, ...]:
        return tuple(self.sources)


AUX_LAYERS: dict[str, AuxLayer] = {
    "mci": AuxLayer(
        key="mci",
        sources={
            "oa10-reflectance": "Oa10_reflectance",
            "oa11-reflectance": "Oa11_reflectance",
            "oa12-reflectance": "Oa12_reflectance",
        },
        compute=_mci,
        why=(
            "surface-scum sensitivity the neural-net chlorophyll retrieval loses at the "
            "top of its range"
        ),
        units="dimensionless reflectance",
        valid_range=(-0.05, 0.2),
    ),
    "tsm": AuxLayer(
        key="tsm",
        sources={"tsm-nn": "TSM_NN"},
        compute=_identity("tsm-nn"),
        why=(
            "suspended sediment, which reddens the water like a bloom does and is the "
            "commonest false positive in a river plume"
        ),
        units="log10(g m^-3)",
        valid_range=(-2.0, 3.0),
    ),
    "kd490": AuxLayer(
        key="kd490",
        sources={"trsp": "KD490_M07"},
        compute=_identity("trsp"),
        why=(
            "how fast light is attenuated -- the depth of the lit layer a bloom can grow "
            "in, and an independent read on turbidity"
        ),
        units="log10(m^-1)",
        valid_range=(-2.0, 1.5),
    ),
}

# The reflectance-derived layers cost three extra asset reads per scene; the rest cost
# one. Stated so an ingest plan can be costed rather than discovered.
ASSET_COST = {key: len(layer.sources) for key, layer in AUX_LAYERS.items()}


def parse_aux(spec: str | None) -> tuple[str, ...]:
    """Turn a comma-separated CLI value into validated layer keys."""
    if not spec:
        return ()
    keys = tuple(k.strip() for k in spec.split(",") if k.strip())
    unknown = [k for k in keys if k not in AUX_LAYERS]
    if unknown:
        raise ValueError(
            f"unknown auxiliary layer(s): {', '.join(unknown)}. "
            f"Available: {', '.join(sorted(AUX_LAYERS))}"
        )
    return keys


def clean(key: str, values: np.ndarray) -> np.ndarray:
    """Apply the layer's plausible range, returning NaN outside it."""
    layer = AUX_LAYERS[key]
    lo, hi = layer.valid_range
    out = np.asarray(values, dtype=np.float32)
    return np.where(np.isfinite(out) & (out >= lo) & (out <= hi), out, np.nan).astype(np.float32)
