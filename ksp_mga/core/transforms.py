"""Frame transform helpers.

Known convention from current project:
  Principia raw -> canonical/SPK/LevelA = (-Y, +Z, +X)
  canonical/SPK/LevelA -> Principia raw = (+Z, -X, +Y)

A transform label is a comma-separated signed axis list, for example:
  "+Z,-X,+Y"
meaning:
  out_x = +in_z
  out_y = -in_x
  out_z = +in_y
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

AXIS_INDEX = {"X": 0, "Y": 1, "Z": 2}

RAW_TO_CANONICAL = "-Y,+Z,+X"
CANONICAL_TO_RAW = "+Z,-X,+Y"


@dataclass(frozen=True)
class AxisTransform:
    label: str
    indices: tuple[int, int, int]
    signs: tuple[float, float, float]

    def apply(self, vec: Sequence[float]) -> np.ndarray:
        v = np.asarray(vec, dtype=float)
        if v.shape[-1] != 3:
            raise ValueError(f"expected 3-vector, got shape {v.shape}")
        return np.asarray([self.signs[i] * v[self.indices[i]] for i in range(3)], dtype=float)

    def apply_many(self, arr: np.ndarray) -> np.ndarray:
        a = np.asarray(arr, dtype=float)
        if a.shape[-1] != 3:
            raise ValueError(f"expected last dimension 3, got shape {a.shape}")
        out = np.empty_like(a, dtype=float)
        for i in range(3):
            out[..., i] = self.signs[i] * a[..., self.indices[i]]
        return out


def parse_transform(label: str) -> AxisTransform:
    parts = [p.strip().upper() for p in label.split(",")]
    if len(parts) != 3:
        raise ValueError(f"transform must have 3 axis terms: {label!r}")

    indices: list[int] = []
    signs: list[float] = []
    used: set[int] = set()
    for p in parts:
        if len(p) != 2 or p[0] not in "+-" or p[1] not in AXIS_INDEX:
            raise ValueError(f"invalid transform term {p!r} in {label!r}")
        idx = AXIS_INDEX[p[1]]
        if idx in used:
            raise ValueError(f"axis repeated in transform {label!r}")
        used.add(idx)
        indices.append(idx)
        signs.append(1.0 if p[0] == "+" else -1.0)
    return AxisTransform(label=label, indices=tuple(indices), signs=tuple(signs))


def apply_transform(vec: Sequence[float], label: str) -> np.ndarray:
    return parse_transform(label).apply(vec)


def inverse_transform(label: str) -> str:
    t = parse_transform(label)
    inv_terms = [None, None, None]
    axis_letters = ["X", "Y", "Z"]
    for out_axis, (in_axis, sign) in enumerate(zip(t.indices, t.signs)):
        inv_terms[in_axis] = ("+" if sign > 0 else "-") + axis_letters[out_axis]
    return ",".join(inv_terms)  # type: ignore[arg-type]
