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
* **PACE.** The mission is real and current, and as of 2026-09-25 it is reachable from
  this host (`~/.netrc`, verified against L2 BGC NRT). It is still not an input channel:
  the finest swath rung is ~1.2 km, so a 12 km lake is 10 pixels across and the mapped
  rungs are far worse. Its use here is to *check* the indices below, not to join them --
  see `habforecast.data.pace`.

Each layer below states what it contributes that chlorophyll cannot, because an input
channel that duplicates an existing one costs training data and buys nothing.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

# OLCI band centres used by the indices, in nanometres.
OA06_NM, OA07_NM, OA08_NM = 560.0, 620.0, 665.0
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


def _ci(bands: dict[str, np.ndarray]) -> np.ndarray:
    """Cyanobacteria Index: the Wynne-Stumpf spectral shape at 681 nm, sign-flipped.

    This is the index NOAA runs operationally for the Lake Erie bloom bulletin, and it is
    the one layer here that claims to be *taxonomic* rather than just sensitive. MCI keys
    on near-infrared brightness, so it rises for any dense surface algae -- a duckweed mat
    and a Microcystis scum look alike to it. The 681 nm shape is different: cyanobacteria
    have negligible chlorophyll fluorescence where eukaryotic algae have a distinct peak,
    so the curvature at 681 nm reverses between the two. A positive CI beside a high MCI
    says the dense thing on the surface is cyanobacterial, which is the distinction that
    decides whether a bloom is a public-health problem.

    Reported sign-flipped (CI = -SS(681)) so that, like every other layer here, larger
    means more bloom.
    """
    r665, r681, r709 = bands["oa08"], bands["oa10"], bands["oa11"]
    slope = (OA10_NM - OA08_NM) / (OA11_NM - OA08_NM)
    return -(r681 - r665 - (r709 - r665) * slope)


def _phycocyanin(bands: dict[str, np.ndarray]) -> np.ndarray:
    """Depth of the 620 nm absorption trough below a 560-665 nm baseline.

    Phycocyanin is the one pigment no eukaryotic alga carries, so an absorption dip at
    620 nm is direct evidence of cyanobacteria rather than an inference from bloom shape.
    Measured as a baseline subtraction, not as raw Oa07 reflectance, because raw Oa07 is
    mostly brightness -- it would tell the model how bright the water is, which it can
    already see, instead of what is in it.

    **Treat this one as exploratory.** The phycocyanin feature is shallow, OLCI's 620 nm
    band is 10 nm wide, and at 300 m over a small lake the dip can sit inside the noise.
    It is cheap to carry and it is the only unambiguous cyanobacteria signal available, so
    it is worth an arm in the experiment -- but it is the layer most likely to contribute
    nothing, and the ablation should be allowed to say so.
    """
    r560, r620, r665 = bands["oa06"], bands["oa07"], bands["oa08"]
    slope = (OA07_NM - OA06_NM) / (OA08_NM - OA06_NM)
    return (r560 + (r665 - r560) * slope) - r620


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
    "ci": AuxLayer(
        key="ci",
        sources={
            "oa08-reflectance": "Oa08_reflectance",
            "oa10-reflectance": "Oa10_reflectance",
            "oa11-reflectance": "Oa11_reflectance",
        },
        compute=_ci,
        why=(
            "whether a dense bloom is cyanobacterial or merely algal -- the 681 nm "
            "curvature reverses between them, where MCI's near-infrared peak does not"
        ),
        units="dimensionless reflectance",
        valid_range=(-0.1, 0.1),
    ),
    "pc620": AuxLayer(
        key="pc620",
        sources={
            "oa06-reflectance": "Oa06_reflectance",
            "oa07-reflectance": "Oa07_reflectance",
            "oa08-reflectance": "Oa08_reflectance",
        },
        compute=_phycocyanin,
        why=(
            "phycocyanin absorption at 620 nm, the one pigment no eukaryotic alga carries "
            "-- direct taxonomic evidence rather than an inference from bloom shape"
        ),
        units="dimensionless reflectance",
        valid_range=(-0.1, 0.1),
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
    "adg443": AuxLayer(
        key="adg443",
        sources={"iop-nn": "ADG443_NN"},
        compute=_identity("iop-nn"),
        why=(
            "absorption by coloured dissolved organic matter and detritus -- the brown, "
            "peat-stained water that TSM misses because nothing is suspended in it"
        ),
        units="log10(m^-1)",
        valid_range=(-3.0, 1.0),
    ),
}

# The reflectance-derived layers cost three extra asset reads per scene; the rest cost
# one. Stated so an ingest plan can be costed rather than discovered.
ASSET_COST = {key: len(layer.sources) for key, layer in AUX_LAYERS.items()}


def asset_reads(keys: Sequence[str]) -> int:
    """Distinct assets a scene must open to produce all of `keys`.

    Not the sum of `ASSET_COST`: the spectral indices share bands (MCI and CI both want
    Oa10 and Oa11; CI and pc620 both want Oa08), and the reader caches each band per
    scene. Costing an ingest by summing the per-layer figures overstates the full
    spectral set by a third.
    """
    return len({asset for k in keys for asset in AUX_LAYERS[k].assets})


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
