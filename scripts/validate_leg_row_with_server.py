#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from ksp_mga.native.impulse_server_client_v0_2 import PrincipiaImpulseServerV2


def norm(v):
    return float(np.linalg.norm(v))


def row_for_leg(path: Path, leg: int):
    with path.open() as f:
        for row in csv.DictReader(f):
            if int(float(row["leg"])) == leg:
                return row
    raise SystemExit(f"[FAIL] leg {leg} not found")


def arr(row, *names):
    return np.array([float(row[n]) for n in names], dtype=float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plugin-b64", required=True)
    ap.add_argument("--server", default="principia_impulsive_particle_server")
    ap.add_argument("--leg-optimizations", type=Path, required=True)
    ap.add_argument("--leg", type=int, default=1)
    ap.add_argument("--output-json", type=Path, required=True)
    args = ap.parse_args()

    row = row_for_leg(args.leg_optimizations, args.leg)

    t0 = float(row["t_start_s"])
    t1 = float(row["t_end_s"])

    r0 = arr(row, "start_x_raw_m", "start_y_raw_m", "start_z_raw_m")
    v0 = arr(row, "start_vx_raw_m_s", "start_vy_raw_m_s", "start_vz_raw_m_s")

    dv = arr(row, "dvx_m_s", "dvy_m_s", "dvz_m_s")

    target_r = arr(row, "target_x_raw_m", "target_y_raw_m", "target_z_raw_m")
    target_v = arr(row, "target_vx_raw_m_s", "target_vy_raw_m_s", "target_vz_raw_m_s")

    with PrincipiaImpulseServerV2(args.server, args.plugin_b64) as srv:
        print("ready:", srv.ready_line)
        assert srv.ping()

        res = srv.propagate_n(
            req_id="validate_leg",
            t0_s=t0,
            t1_s=t1,
            r0_m=r0,
            v0_m_s=v0,
            impulses=[(t0, dv)],
        )

    out = {
        "status": res.status,
        "message": res.message,
        "t_start_s": t0,
        "t_end_s": t1,
        "dv_raw_m_s": dv.tolist(),
        "dv_norm_m_s": norm(dv),
        "final_pos_err_km": None,
        "final_vel_err_m_s": None,
    }

    if res.status == "ok":
        out["final_pos_err_km"] = norm(res.final_r_m - target_r) / 1000.0
        out["final_vel_err_m_s"] = norm(res.final_v_m_s - target_v)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))

    return 0 if res.status == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
