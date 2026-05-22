#!/usr/bin/env python3
"""
fix_active_vessel_state_frame_v0.py

Corrige/audita o frame do active_vessel_state exportado pela DLL.

Problema detectado:
  QP.q/QP.p podem sair na ordem [raw_x, raw_z, raw_y], enquanto o solver
  espera [raw_x, raw_y, raw_z]. O sintoma é:
    normalized(rel_v_raw_m_s) != principia_basis.tangent_raw
  mas:
    normalized([p0, p2, p1]) == principia_basis.tangent_raw

Este script:
  1. testa permutações de eixos da velocidade contra principia_basis.tangent_raw;
  2. aplica a melhor permutação também na posição;
  3. grava um active_state corrigido;
  4. preenche defaults de thrust/Isp se vierem null, evitando TypeError no event writer.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path
from typing import Sequence

import numpy as np


def norm(v: Sequence[float]) -> float:
    return float(np.linalg.norm(np.asarray(v, dtype=float)))


def unit(v: Sequence[float]) -> np.ndarray:
    a = np.asarray(v, dtype=float)
    n = norm(a)
    if n <= 0 or not math.isfinite(n):
        raise ValueError(f"cannot normalize {v!r}")
    return a / n


def angle_deg(a: Sequence[float], b: Sequence[float]) -> float:
    au = unit(a)
    bu = unit(b)
    return math.degrees(math.acos(max(-1.0, min(1.0, float(np.dot(au, bu))))))


def permute(v: Sequence[float], p: tuple[int, int, int]) -> list[float]:
    return [float(v[p[0]]), float(v[p[1]]), float(v[p[2]])]


def safe_default(d: dict, key: str, default):
    if key not in d or d[key] is None:
        d[key] = default


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--force-permutation", default=None, help="Ex.: 0,2,1. Se omitido, escolhe pela tangente.")
    ap.add_argument("--default-mass-tonnes", type=float, default=2.6)
    ap.add_argument("--default-thrust-kN", type=float, default=2686.87701225281)
    ap.add_argument("--default-isp-s-g0", type=float, default=1000.0)
    args = ap.parse_args()

    d = json.loads(args.input.read_text())

    rel_r = d.get("rel_r_raw_m")
    rel_v = d.get("rel_v_raw_m_s")
    if rel_r is None or rel_v is None:
        raise SystemExit("input missing rel_r_raw_m or rel_v_raw_m_s")

    tangent = (((d.get("principia_basis") or {}).get("tangent_raw")) or
               ((d.get("debug") or {}).get("vessel_tangent_raw")))
    if tangent is None:
        raise SystemExit("input missing principia_basis.tangent_raw; cannot audit permutation")

    if args.force_permutation:
        p = tuple(int(x.strip()) for x in args.force_permutation.split(","))
        if len(p) != 3:
            raise SystemExit("--force-permutation must have 3 indices")
        candidates = [(angle_deg(permute(rel_v, p), tangent), p)]
    else:
        candidates = []
        for p in itertools.permutations((0, 1, 2)):
            try:
                candidates.append((angle_deg(permute(rel_v, p), tangent), p))
            except Exception:
                pass
        candidates.sort(key=lambda x: x[0])

    best_ang, best_p = candidates[0]
    old_ang = angle_deg(rel_v, tangent)

    fixed = dict(d)
    fixed["rel_r_raw_m_original_export"] = rel_r
    fixed["rel_v_raw_m_s_original_export"] = rel_v
    fixed["rel_r_raw_m"] = permute(rel_r, best_p)
    fixed["rel_v_raw_m_s"] = permute(rel_v, best_p)
    fixed["state_source"] = f"{d.get('state_source', 'unknown')}+frame_permutation_{best_p[0]}{best_p[1]}{best_p[2]}"
    fixed["frame_fix"] = {
        "schema": "active_vessel_state_frame_fix_v0",
        "chosen_permutation": list(best_p),
        "old_tangent_angle_deg": old_ang,
        "new_tangent_angle_deg": best_ang,
        "candidate_angles_deg": [
            {"permutation": list(p), "angle_deg": a}
            for a, p in candidates[:6]
        ],
        "note": "Applied same axis permutation to rel_r_raw_m and rel_v_raw_m_s.",
    }

    safe_default(fixed, "mass_tonnes", args.default_mass_tonnes)
    safe_default(fixed, "available_thrust_kN", args.default_thrust_kN)
    safe_default(fixed, "specific_impulse_s_g0", args.default_isp_s_g0)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(fixed, indent=2) + "\n")

    print("=== ACTIVE VESSEL STATE FRAME FIX ===")
    print(f"input                 : {args.input}")
    print(f"output                : {args.output}")
    print(f"chosen_permutation    : {best_p}")
    print(f"old_tangent_angle_deg : {old_ang:.12g}")
    print(f"new_tangent_angle_deg : {best_ang:.12g}")
    print(f"old_rel_r             : {rel_r}")
    print(f"new_rel_r             : {fixed['rel_r_raw_m']}")
    print(f"old_rel_v             : {rel_v}")
    print(f"new_rel_v             : {fixed['rel_v_raw_m_s']}")
    print("top candidates:")
    for a, p in candidates[:6]:
        print(f"  p={p} angle={a:.12g} deg")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
