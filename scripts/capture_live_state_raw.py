#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import krpc
import spiceypy as spice


def norm(v):
    return math.sqrt(sum(x*x for x in v))


def levela_to_raw(v):
    # LevelA -> Principia raw = [+Z, -X, +Y]
    x, y, z = v
    return [z, -x, y]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bsp", type=Path, required=True)
    ap.add_argument("--tpc", type=Path, required=True)
    ap.add_argument("--center", default="SUN")
    ap.add_argument("--frame", default="J2000")
    ap.add_argument("--output-json", type=Path, required=True)
    args = ap.parse_args()

    conn = krpc.connect(name="MGA capture live state raw")
    sc = conn.space_center
    vessel = sc.active_vessel

    ut = sc.ut
    body = vessel.orbit.body
    sun = sc.bodies["Sun"]

    # kRPC state in Sun non-rotating frame. We treat it as LevelA-like and rotate.
    r_levela = list(vessel.position(sun.non_rotating_reference_frame))
    v_levela = list(vessel.velocity(sun.non_rotating_reference_frame))
    r_raw = levela_to_raw(r_levela)
    v_raw = levela_to_raw(v_levela)

    # Sanity: compare relative distance to active body against SPICE body state.
    spice.kclear()
    spice.furnsh(str(args.tpc))
    spice.furnsh(str(args.bsp))
    st, _ = spice.spkezr(body.name.upper(), ut, args.frame, "NONE", args.center)
    body_r_raw = levela_to_raw([1000*st[0], 1000*st[1], 1000*st[2]])
    body_v_raw = levela_to_raw([1000*st[3], 1000*st[4], 1000*st[5]])

    rel_raw = [r_raw[i] - body_r_raw[i] for i in range(3)]
    relv_raw = [v_raw[i] - body_v_raw[i] for i in range(3)]

    r_body_ksp = list(vessel.position(body.non_rotating_reference_frame))
    v_body_ksp = list(vessel.velocity(body.non_rotating_reference_frame))

    out = {
        "ut_s": ut,
        "vessel_name": vessel.name,
        "body": body.name,
        "r_raw_m": r_raw,
        "v_raw_m_s": v_raw,
        "body_r_raw_m": body_r_raw,
        "body_v_raw_m_s": body_v_raw,
        "relative_r_raw_m": rel_raw,
        "relative_v_raw_m_s": relv_raw,
        "relative_r_raw_norm_m": norm(rel_raw),
        "relative_v_raw_norm_m_s": norm(relv_raw),
        "ksp_body_relative_r_norm_m": norm(r_body_ksp),
        "ksp_body_relative_v_norm_m_s": norm(v_body_ksp),
        "sanity_note": "relative raw norm should be close to kRPC body-relative norm",
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
