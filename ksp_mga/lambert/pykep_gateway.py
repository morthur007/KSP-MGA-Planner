"""Stable wrapper around PyKEP/kep3 Lambert.

Do not import PyKEP directly across the project. Use this module so we can keep
version differences and solution indexing in one place.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np


@dataclass(frozen=True)
class LambertSolution:
    v0_km_s: np.ndarray
    v1_km_s: np.ndarray
    path_label: str
    revs: int
    index: int
    cw: bool


def _label_for_index(i: int, cw: bool) -> tuple[str, int]:
    if i == 0:
        return f"0rev_cw{int(cw)}", 0
    # PyKEP convention: for each revolution there are two solutions. The exact
    # left/right ordering can vary in wrappers, but this label is kept stable for
    # our own pipeline because we consume the saved velocities, not the label.
    revs = (i + 1) // 2
    branch = "left" if i % 2 == 1 else "right"
    return f"{branch}_cw{int(cw)}", revs


def solve_lambert_pykep(
    r0_km: Sequence[float],
    r1_km: Sequence[float],
    tof_s: float,
    mu_km3_s2: float,
    *,
    cw: bool = False,
    max_revs: int = 0,
) -> List[LambertSolution]:
    try:
        import pykep as pk
    except Exception as e:  # pragma: no cover
        raise RuntimeError("pykep is required for solve_lambert_pykep") from e

    lp = pk.lambert_problem(
        r0=list(map(float, r0_km)),
        r1=list(map(float, r1_km)),
        tof=float(tof_s),
        mu=float(mu_km3_s2),
        cw=bool(cw),
        max_revs=int(max_revs),
    )

    if hasattr(lp, "get_v1"):
        v0s = lp.get_v1()
        v1s = lp.get_v2()
    else:
        v0s = lp.v0
        v1s = lp.v1

    out: list[LambertSolution] = []
    for i, (v0, v1) in enumerate(zip(v0s, v1s)):
        label, revs = _label_for_index(i, cw)
        out.append(
            LambertSolution(
                v0_km_s=np.asarray(v0, dtype=float),
                v1_km_s=np.asarray(v1, dtype=float),
                path_label=label,
                revs=revs,
                index=i,
                cw=bool(cw),
            )
        )
    return out
