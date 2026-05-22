#!/usr/bin/env python3
"""
diagnose_spice_frame_time_parity_v0.py

Compara o estado absoluto de um corpo no BSP/SPICE, após transform para
pipeline raw, contra um estado absoluto de referência vindo de VCAREL_NAV.

Exemplo com Kerbin implícito no burn debug:

python scripts/diagnose_spice_frame_time_parity_v0.py \
  --bsp data/kernels/jnsq/v0_3/jnsq_principia_native_v0_3_1000y_1h_deg9.bsp \
  --tpc data/kernels/jnsq/v0_3/jnsq_principia_native_v0_3_1000y_1h_deg9.ids.tpc \
  --body-catalog data/catalogs/jnsq/body_catalog.json \
  --body KERBIN \
  --observer SUN \
  --t-s 822531441.851686 \
  --ref-r "17879123212.85027,32560972746.20639,845044.3507884003" \
  --ref-v "-13106.815230119213,7511.284845378739,1.3868677171226027" \
  --scan-half-s 7200 \
  --scan-step-s 60
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from spice_vcarelnav_targeter_v0_2 import SpiceVcarelNavTargeter, parse_body_list


def vec3(s: str):
    v = [float(x.strip()) for x in s.replace(";", ",").split(",") if x.strip()]
    if len(v) != 3:
        raise argparse.ArgumentTypeError(f"expected x,y,z vector, got {s!r}")
    return np.asarray(v, dtype=float)


def n(v):
    return float(np.linalg.norm(v))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bsp", type=Path, required=True)
    ap.add_argument("--tpc", type=Path, default=None)
    ap.add_argument("--body-catalog", type=Path, required=True)
    ap.add_argument("--body", required=True)
    ap.add_argument("--observer", default="SUN")
    ap.add_argument("--attractors", default=None)
    ap.add_argument("--t-s", type=float, required=True)
    ap.add_argument("--ref-r", type=vec3, required=True)
    ap.add_argument("--ref-v", type=vec3, required=True)
    ap.add_argument("--scan-half-s", type=float, default=7200.0)
    ap.add_argument("--scan-step-s", type=float, default=60.0)
    args = ap.parse_args()

    with SpiceVcarelNavTargeter(
        bsp=args.bsp,
        tpc=args.tpc,
        body_catalog=args.body_catalog,
        attractors=parse_body_list(args.attractors),
        observer=args.observer,
    ) as targeter:
        r0, v0 = targeter.body_state_abs_m(args.body, args.t_s)

        print("=== SPICE FRAME/TIME PARITY ===")
        print(f"body        : {args.body}")
        print(f"observer    : {args.observer}")
        print(f"t_s         : {args.t_s}")
        print(f"spice_r     : {r0.tolist()}")
        print(f"ref_r       : {args.ref_r.tolist()}")
        print(f"dr_km       : {n(r0 - args.ref_r)/1000:.6f}")
        print(f"dr          : {(r0 - args.ref_r).tolist()}")
        print(f"spice_v     : {v0.tolist()}")
        print(f"ref_v       : {args.ref_v.tolist()}")
        print(f"dv_m_s      : {n(v0 - args.ref_v):.9f}")
        print(f"dv          : {(v0 - args.ref_v).tolist()}")

        best = None
        steps = int(round((2.0 * args.scan_half_s) / args.scan_step_s))
        for i in range(steps + 1):
            dt = -args.scan_half_s + i * args.scan_step_s
            r, v = targeter.body_state_abs_m(args.body, args.t_s + dt)
            dr = n(r - args.ref_r)
            dv = n(v - args.ref_v)
            score = dr + 1000.0 * dv
            if best is None or score < best[0]:
                best = (score, dt, dr, dv, r, v)

        _score, dt, dr, dv, r, v = best
        print("\nBest time offset scan:")
        print(f"dt_s        : {dt:.6f}")
        print(f"dr_km       : {dr/1000:.6f}")
        print(f"dv_m_s      : {dv:.9f}")
        print(f"r_at_best   : {r.tolist()}")
        print(f"v_at_best   : {v.tolist()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
