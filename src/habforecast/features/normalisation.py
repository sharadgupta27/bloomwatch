"""Standardisation constants for log10 chlorophyll.

Its own module, with no torch import, because serving needs it and the ONNX serving image
has no torch: `features.dataset` (which re-exports it) imports torch at module level.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Normalisation:
    """Standardisation constants for log10 chlorophyll, fitted on training lakes only."""

    mean: float
    std: float

    def apply(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std

    def invert(self, z: np.ndarray) -> np.ndarray:
        return z * self.std + self.mean

    def to_dict(self) -> dict[str, float]:
        return {"mean": self.mean, "std": self.std}
