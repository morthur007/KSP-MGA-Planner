#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import spiceypy as spice


def norm(v):
    return math.sqrt(sum(x*x for x in v))


def sub(a, b):
    return [a[i] - b[i] for i in range(3)]


def levela_to_raw(v):
    # Validated Principia mapping:
    # raw -> LevelA = [-Y, +Z, +X]
    # therefore LevelA -> raw = [+Z, -X, +Y]
    x, y, z = v
    return [z, -x, y]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--leg-optimizations", type=Path, required=True)
    ap.add_argument("--leg", type=int, default=1)
    ap.add_argument("--bsp", type=Path, required=True)
    ap.add_argument("--tpc", type=Path, required=True)
    ap.add_argument("--dep-body", default="KERBIN")
    ap.add_argument("--center", default="SUN")
    ap.add_argument("--frame", default="J2000")
    ap.add_argument("--output-json", type=Path, required=True)
    args = ap.parse_args()

    spice.kclear()
    spice.furnsh(str(args.tpc))
    spice.furnsh(str(args.bsp))

    rows = list(csv.DictReader(args.leg_optimizations.open()))
    row = next(r for r in rows if int(float(r["leg"])) == args.leg)

    t = float(row["t_start_s"])

    sc_r = [
        float(row["start_x_raw_m"]),
        float(row["start_y_raw_m"]),
        float(row["start_z_raw_m"]),
    ]
    sc_v = [
        float(row["start_vx_raw_m_s"]),
        float(row["start_vy_raw_m_s"]),
        float(row["start_vz_raw_m_s"]),
    ]

    # SPICE returns km and km/s.
    st, _ = spice.spkezr(args.dep_body, t, args.frame, "NONE", args.center)
    # SPICE/J2000 values are in the LevelA-like frame used by the BSP.
    # The optimizer CSV uses raw Principia coordinates, so rotate body state
    # into raw before subtracting.
    body_r_levela = [1000.0 * st[0], 1000.0 * st[1], 1000.0 * st[2]]
    body_v_levela = [1000.0 * st[3], 1000.0 * st[4], 1000.0 * st[5]]
    body_r = levela_to_raw(body_r_levela)
    body_v = levela_to_raw(body_v_levela)

    vinf = sub(sc_v, body_v)
    r_rel = sub(sc_r, body_r)

    out = {
        "leg": args.leg,
        "dep_body": row["dep_body"],
        "arr_body": row["arr_body"],
        "t_start_s": t,
        "t_dep_s": float(row["t_dep_s"]),
        "t_arr_s": float(row["t_arr_s"]),
        "spacecraft_start_raw_m": sc_r,
        "spacecraft_start_raw_m_s": sc_v,
        "body_raw_m": body_r,
        "body_raw_m_s": body_v,
        "relative_start_m": r_rel,
        "relative_start_km_norm": norm(r_rel) / 1000.0,
        "vinf_raw_m_s": vinf,
        "vinf_norm_m_s": norm(vinf),
        "vinf_norm_km_s": norm(vinf) / 1000.0,
        "correction_dv_raw_m_s": [
            float(row["dvx_m_s"]),
            float(row["dvy_m_s"]),
            float(row["dvz_m_s"]),
        ],
        "correction_dv_norm_m_s": float(row["dv_norm_m_s"]),
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(out, indent=2) + "\n")

    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
