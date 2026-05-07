#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import spiceypy as spice

from scripts.smoke_impulse_server import (
    DAY_S,
    kepler_universal_propagate,
    norm,
    norm_name,
    spk_state,
)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--candidate-seed", type=Path, required=True)
    p.add_argument("--rank", type=int, default=1)
    p.add_argument("--leg", type=int, default=1)
    p.add_argument("--bsp", type=Path, required=True)
    p.add_argument("--tpc", type=Path, required=True)
    p.add_argument("--central-body", default="Sun")
    p.add_argument("--central-mu-km3-s2", type=float, required=True)
    p.add_argument("--buffer-days", type=float, default=0.235)
    args = p.parse_args()

    spice.furnsh(str(args.tpc))
    spice.furnsh(str(args.bsp))

    rows = list(csv.DictReader(args.candidate_seed.open()))
    row = rows[args.rank - 1]

    leg = args.leg
    dep = norm_name(row[f"leg{leg}_dep"])
    arr = norm_name(row[f"leg{leg}_arr"])

    dep_i = leg - 1
    arr_i = leg

    t_dep = float(row[f"event{dep_i}_et_s"])
    t_arr = float(row[f"event{arr_i}_et_s"])
    buffer_s = args.buffer_days * DAY_S

    st_dep = spk_state(dep, t_dep, args.central_body)
    st_arr = spk_state(arr, t_arr, args.central_body)

    vdep = np.array([
        float(row[f"leg{leg}_vdep_x_km_s"]),
        float(row[f"leg{leg}_vdep_y_km_s"]),
        float(row[f"leg{leg}_vdep_z_km_s"]),
    ])

    varr_saved = np.array([
        float(row[f"leg{leg}_varr_x_km_s"]),
        float(row[f"leg{leg}_varr_y_km_s"]),
        float(row[f"leg{leg}_varr_z_km_s"]),
    ])

    # Full Lambert closure: dep epoch -> arr epoch.
    r_full, v_full = kepler_universal_propagate(
        st_dep[:3], vdep, t_arr - t_dep, args.central_mu_km3_s2
    )

    full_pos_miss_km = norm(r_full - st_arr[:3])
    full_vel_miss_m_s = norm(v_full - varr_saved) * 1000.0

    # Buffered closure: spacecraft at t_arr-buffer vs planet at t_arr-buffer.
    t_start = t_dep + buffer_s
    t_end = t_arr - buffer_s

    r_start, v_start = kepler_universal_propagate(
        st_dep[:3], vdep, buffer_s, args.central_mu_km3_s2
    )

    r_end, v_end = kepler_universal_propagate(
        r_start, v_start, t_end - t_start, args.central_mu_km3_s2
    )

    st_target = spk_state(arr, t_end, args.central_body)

    buffer_pos_miss_km = norm(r_end - st_target[:3])
    buffer_vel_miss_m_s = norm(v_end - st_target[3:]) * 1000.0

    print("=== KEPLER CLOSURE CHECK ===")
    print(f"rank      : {args.rank}")
    print(f"leg       : {leg} {dep}->{arr}")
    print(f"path      : {row.get(f'leg{leg}_path')}")
    print(f"tof_days  : {(t_arr - t_dep) / DAY_S:.6f}")
    print("")
    print("Full endpoint closure:")
    print(f"  pos miss: {full_pos_miss_km:.6f} km")
    print(f"  vel miss: {full_vel_miss_m_s:.6f} m/s")
    print("")
    print("Buffered pre-encounter miss:")
    print(f"  pos miss: {buffer_pos_miss_km:.6f} km")
    print(f"  vel miss: {buffer_vel_miss_m_s:.6f} m/s")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
