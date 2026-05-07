# pykep_gateway_v0_1.py

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import pykep as pk


@dataclass
class LambertSolution:
    v0_km_s: np.ndarray
    v1_km_s: np.ndarray
    path_label: str
    revs: int
    index: int


def solve_lambert_pykep(
    r0_km,
    r1_km,
    tof_s: float,
    mu_km3_s2: float,
    *,
    cw: bool = False,
    max_revs: int = 0,
) -> List[LambertSolution]:
    """
    Thin compatibility layer around PyKEP/kep3 Lambert.

    Units:
      r0/r1: km
      tof: seconds
      mu: km^3/s^2
      output velocities: km/s
    """
    r0 = np.asarray(r0_km, dtype=float)
    r1 = np.asarray(r1_km, dtype=float)

    # Chamada posicional blindada: (r1, r2, tof, mu, cw, max_revs)
    # Evita conflitos de nomes (kwargs) com a assinatura do C++
    lp = pk.lambert_problem(
        r0.tolist(),
        r1.tolist(),
        float(tof_s),
        float(mu_km3_s2),
        bool(cw),
        int(max_revs)
    )

    if hasattr(lp, "get_v1"):
        v0s = lp.get_v1()
        v1s = lp.get_v2()
    else:
        v0s = lp.v0
        v1s = lp.v1

    out: List[LambertSolution] = []
    for i, (v0, v1) in enumerate(zip(v0s, v1s)):
        if i == 0:
            revs = 0
            branch = "0rev"
        else:
            revs = (i + 1) // 2
            branch = "left" if i % 2 == 1 else "right"

        out.append(
            LambertSolution(
                v0_km_s=np.asarray(v0, dtype=float),
                v1_km_s=np.asarray(v1, dtype=float),
                path_label=f"{branch}_cw{int(cw)}",
                revs=revs,
                index=i,
            )
        )

    return out