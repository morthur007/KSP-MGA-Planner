"""Small SPICE access wrapper.

Keeps the rest of the code from directly depending on spiceypy everywhere.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class SpiceKernels:
    bsp: Path
    tpc: Path


class SpiceEphemeris:
    def __init__(self, kernels: SpiceKernels, frame: str = "J2000", aberration: str = "NONE") -> None:
        try:
            import spiceypy as spice
        except Exception as e:  # pragma: no cover
            raise RuntimeError("spiceypy is required for SpiceEphemeris") from e
        self.spice = spice
        self.kernels = kernels
        self.frame = frame
        self.aberration = aberration
        self._loaded = False

    def load(self) -> None:
        if self._loaded:
            return
        self.spice.furnsh(str(self.kernels.tpc))
        self.spice.furnsh(str(self.kernels.bsp))
        self._loaded = True

    def state_km(self, body: str, et_s: float, center: str) -> np.ndarray:
        self.load()
        st, _lt = self.spice.spkezr(body.upper(), float(et_s), self.frame, self.aberration, center.upper())
        return np.asarray(st, dtype=float)

    def pos_km(self, body: str, et_s: float, center: str) -> np.ndarray:
        return self.state_km(body, et_s, center)[:3]

    def vel_km_s(self, body: str, et_s: float, center: str) -> np.ndarray:
        return self.state_km(body, et_s, center)[3:]
