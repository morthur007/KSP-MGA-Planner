#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import spiceypy as spice

from ksp_mga.native.impulse_server_client import PrincipiaImpulseServer
from scripts.smoke_impulse_server import (
    DAY_S,
    apply_transform,
    kepler_universal_propagate,
    norm,
    norm_name,
    parse_transform,
    sample_raw_body_state,
    spk_state,
)


def read_candidate(path: Path, rank: int) -> dict[str, str]:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    return rows[rank - 1]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--candidate-seed", type=Path, required=True)
    p.add_argument("--rank", type=int, default=1)
    p.add_argument("--leg", type=int, default=1)

    p.add_argument("--bsp", type=Path, required=True)
    p.add_argument("--tpc", type=Path, required=True)
    p.add_argument("--plugin-b64", type=Path, required=True)
    p.add_argument("--server", default="principia_impulsive_particle_server")
    p.add_argument("--sampler", default="sample_principia_ephemeris")

    p.add_argument("--central-body", default="Sun")
    p.add_argument("--central-mu-km3-s2", type=float, required=True)
    p.add_argument("--plugin-base-et-s", type=float, default=81.65168640136972)
    p.add_argument("--raw-origin-body", default="Sun")
    p.add_argument("--raw-cache-dir", type=Path, default=Path("data/runs/frame_debug/raw_cache"))

    p.add_argument("--transform", default="+Z,-X,+Y")
    p.add_argument("--buffer-days", type=float, default=0.235)
    p.add_argument(
        "--dt-days",
        nargs="+",
        type=float,
        default=[1/24, 0.25, 1, 3, 7, 14, 30, 60, 120, 229.53],
    )
    p.add_argument("--output-csv", type=Path, required=True)
    args = p.parse_args()

    spice.furnsh(str(args.tpc))
    spice.furnsh(str(args.bsp))

    row = read_candidate(args.candidate_seed, args.rank)
    leg = args.leg

    dep = norm_name(row[f"leg{leg}_dep"])
    arr = norm_name(row[f"leg{leg}_arr"])
    dep_i = leg - 1
    arr_i = leg

    t_dep = float(row[f"event{dep_i}_et_s"])
    t_arr = float(row[f"event{arr_i}_et_s"])
    buffer_s = args.buffer_days * DAY_S
    t_start = t_dep + buffer_s
    t_end_max = t_arr - buffer_s

    st_dep = spk_state(dep, t_dep, args.central_body)

    vdep_km_s = np.array([
        float(row[f"leg{leg}_vdep_x_km_s"]),
        float(row[f"leg{leg}_vdep_y_km_s"]),
        float(row[f"leg{leg}_vdep_z_km_s"]),
    ])

    # Kepler state at buffer start, still LevelA/Sun-centered.
    r0_rel_km, v0_rel_km_s = kepler_universal_propagate(
        st_dep[:3],
        vdep_km_s,
        buffer_s,
        args.central_mu_km3_s2,
    )

    transform = parse_transform(args.transform)

    # Convert start state to Principia raw/Barycentric absolute.
    origin_start_r_raw_m, origin_start_v_raw_m_s = sample_raw_body_state(
        sampler=args.sampler,
        plugin_b64=args.plugin_b64,
        target_body=args.raw_origin_body,
        sampler_central_body=args.raw_origin_body,
        et_s=t_start,
        plugin_base_et_s=args.plugin_base_et_s,
        work_dir=args.raw_cache_dir,
    )

    r0_raw_abs_m = origin_start_r_raw_m + apply_transform(r0_rel_km * 1000.0, transform)
    v0_raw_abs_m_s = origin_start_v_raw_m_s + apply_transform(v0_rel_km_s * 1000.0, transform)

    print("=== IMPULSE VS KEPLER SWEEP ===")
    print(f"candidate : {row.get('candidate_id')} rank={args.rank}")
    print(f"leg       : {leg} {dep}->{arr}")
    print(f"t_start   : {t_start:.9f}")
    print(f"t_end_max : {t_end_max:.9f}")
    print(f"transform : {args.transform}")
    print(f"raw origin: {args.raw_origin_body}")

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    out_rows = []

    with PrincipiaImpulseServer(args.server, args.plugin_b64) as srv:
        for dt_day in args.dt_days:
            dt_s = dt_day * DAY_S
            t_eval = t_start + dt_s
            if t_eval > t_end_max:
                continue

            # Kepler expected particle state at t_eval, LevelA/Sun-centered.
            r_kep_rel_km, v_kep_rel_km_s = kepler_universal_propagate(
                r0_rel_km,
                v0_rel_km_s,
                dt_s,
                args.central_mu_km3_s2,
            )

            # Convert Kepler expected endpoint to raw/Barycentric absolute.
            origin_eval_r_raw_m, origin_eval_v_raw_m_s = sample_raw_body_state(
                sampler=args.sampler,
                plugin_b64=args.plugin_b64,
                target_body=args.raw_origin_body,
                sampler_central_body=args.raw_origin_body,
                et_s=t_eval,
                plugin_base_et_s=args.plugin_base_et_s,
                work_dir=args.raw_cache_dir,
            )

            expected_r_raw_m = origin_eval_r_raw_m + apply_transform(r_kep_rel_km * 1000.0, transform)
            expected_v_raw_m_s = origin_eval_v_raw_m_s + apply_transform(v_kep_rel_km_s * 1000.0, transform)

            res = srv.propagate(
                req_id=f"rank{args.rank}_leg{leg}_dt{dt_day}",
                t0_s=t_start,
                burn_t_s=t_start,
                t1_s=t_eval,
                r0_m=r0_raw_abs_m,
                v0_m_s=v0_raw_abs_m_s,
                burn_dv_m_s=np.zeros(3),
            )

            if res.status != "ok":
                print(f"dt={dt_day:10.4f} d | server error: {res.status} {res.message}")
                out_rows.append({
                    "dt_days": dt_day,
                    "status": res.status,
                    "message": res.message,
                })
                continue

            # Also compare to arrival body at same t_eval, for reference.
            target_st = spk_state(arr, t_eval, args.central_body)
            target_r_raw_m = origin_eval_r_raw_m + apply_transform(target_st[:3] * 1000.0, transform)
            target_v_raw_m_s = origin_eval_v_raw_m_s + apply_transform(target_st[3:] * 1000.0, transform)

            p_vs_kep_km = norm(res.final_r_m - expected_r_raw_m) / 1000.0
            v_vs_kep_m_s = norm(res.final_v_m_s - expected_v_raw_m_s)

            p_vs_target_km = norm(res.final_r_m - target_r_raw_m) / 1000.0
            v_vs_target_m_s = norm(res.final_v_m_s - target_v_raw_m_s)

            print(
                f"dt={dt_day:10.4f} d | "
                f"Nbody-Kepler={p_vs_kep_km:12.3f} km {v_vs_kep_m_s:10.3f} m/s | "
                f"Nbody-target={p_vs_target_km:12.3f} km {v_vs_target_m_s:10.3f} m/s"
            )

            out_rows.append({
                "dt_days": dt_day,
                "t_eval_s": t_eval,
                "status": "ok",
                "nbody_vs_kepler_km": p_vs_kep_km,
                "nbody_vs_kepler_m_s": v_vs_kep_m_s,
                "nbody_vs_target_km": p_vs_target_km,
                "nbody_vs_target_m_s": v_vs_target_m_s,
            })

    fields = [
        "dt_days", "t_eval_s", "status",
        "nbody_vs_kepler_km", "nbody_vs_kepler_m_s",
        "nbody_vs_target_km", "nbody_vs_target_m_s",
        "message",
    ]

    with args.output_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(out_rows)

    print(f"[OK] wrote {args.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
