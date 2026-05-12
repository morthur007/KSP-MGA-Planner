#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from ksp_mga.native.impulse_server_client import PrincipiaImpulseServer


def n(v):
    return float(np.linalg.norm(v))


def raw_to_levela(v):
    x, y, z = v
    return [-y, z, x]


def read_leg_row(path: Path, leg: int) -> dict:
    rows = list(csv.DictReader(path.open()))
    return next(r for r in rows if int(float(r["leg"])) == leg)


def arr(row, *names):
    return np.array([float(row[n]) for n in names], dtype=float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plugin-b64", required=True)
    ap.add_argument("--server", default="bin/x64/principia_impulsive_particle_server")
    ap.add_argument("--live-state-json", type=Path, required=True)
    ap.add_argument("--leg-optimizations", type=Path, required=True)
    ap.add_argument("--leg", type=int, default=1)
    ap.add_argument("--output-dir", type=Path, required=True)

    ap.add_argument("--corr-time", type=float, default=None,
                    help="Default: leg t_start_s")
    ap.add_argument("--final-time", type=float, default=None,
                    help="Default: leg t_end_s")
    ap.add_argument("--max-nfev", type=int, default=80)
    ap.add_argument("--pos-scale-km", type=float, default=10000.0)
    ap.add_argument("--vel-scale-m-s", type=float, default=100.0)
    ap.add_argument("--dv-scale-m-s", type=float, default=5000.0)
    ap.add_argument("--initial-dv0-nav-m-s", type=float, default=0.0,
                    help="Optional scalar prograde seed is not used here; kept for CLI compatibility.")
    args = ap.parse_args()

    live = json.loads(args.live_state_json.read_text())
    row = read_leg_row(args.leg_optimizations, args.leg)

    t0 = float(live["ut_s"])
    t_corr = args.corr_time if args.corr_time is not None else float(row["t_start_s"])
    t_final = args.final_time if args.final_time is not None else float(row["t_end_s"])

    r0 = np.array(live["r_raw_m"], dtype=float)
    v0 = np.array(live["v_raw_m_s"], dtype=float)

    target_r = arr(row, "target_x_raw_m", "target_y_raw_m", "target_z_raw_m")
    target_v = arr(row, "target_vx_raw_m_s", "target_vy_raw_m_s", "target_vz_raw_m_s")

    old_corr = arr(row, "dvx_m_s", "dvy_m_s", "dvz_m_s")

    # Seed: no departure guess, old correction as correction seed.
    # For better convergence, replace x0[:3] with a real departure estimate later.
    x0 = np.zeros(6, dtype=float)
    x0[3:6] = old_corr

    pos_scale_m = args.pos_scale_km * 1000.0
    vel_scale = args.vel_scale_m_s
    dv_scale = args.dv_scale_m_s

    evals = {"n": 0}
    best = {"obj": float("inf"), "x": None, "pos": None, "vel": None}

    def evaluate(x):
        evals["n"] += 1
        dv0 = np.array(x[:3], dtype=float)
        dv1 = np.array(x[3:6], dtype=float)

        with PrincipiaImpulseServer(args.server, args.plugin_b64) as srv:
            a = srv.propagate(
                f"a{evals['n']}",
                t0, t0, t_corr,
                r0, v0, dv0,
            )
            if a.status != "ok":
                return None, None, f"first:{a.status}:{a.message}"

            b = srv.propagate(
                f"b{evals['n']}",
                t_corr, t_corr, t_final,
                a.final_r_m, a.final_v_m_s, dv1,
            )
            if b.status != "ok":
                return None, None, f"second:{b.status}:{b.message}"

        return b.final_r_m, b.final_v_m_s, "ok"

    def residual(x):
        rf, vf, status = evaluate(x)
        if status != "ok":
            return np.ones(12) * 1e6

        dv0 = np.array(x[:3], dtype=float)
        dv1 = np.array(x[3:6], dtype=float)

        pos_err = rf - target_r
        vel_err = vf - target_v

        res = np.concatenate([
            pos_err / pos_scale_m,
            vel_err / vel_scale,
            dv0 / dv_scale * 0.02,
            dv1 / dv_scale * 0.02,
        ])

        obj = float(np.linalg.norm(res))
        if obj < best["obj"]:
            best.update({
                "obj": obj,
                "x": np.array(x, dtype=float),
                "pos": n(pos_err) / 1000.0,
                "vel": n(vel_err),
            })
            print(
                f"[best {evals['n']:04d}] "
                f"obj={obj:.6g} pos={best['pos']:.3f} km "
                f"vel={best['vel']:.3f} m/s "
                f"dv0={n(dv0):.3f} dv1={n(dv1):.3f}",
                flush=True,
            )

        return res

    print("=== TWO-IMPULSE LIVE DEPARTURE OPT ===")
    print(f"t0      : {t0}")
    print(f"t_corr  : {t_corr}")
    print(f"t_final : {t_final}")
    print(f"old_corr_norm_m_s: {n(old_corr):.3f}")

    res = least_squares(
        residual,
        x0,
        max_nfev=args.max_nfev,
        x_scale=np.array([1000,1000,1000,200,200,200], dtype=float),
        diff_step=1e-3,
        verbose=2,
    )

    x = best["x"] if best["x"] is not None else res.x
    rf, vf, status = evaluate(x)

    dv0_raw = np.array(x[:3], dtype=float)
    dv1_raw = np.array(x[3:6], dtype=float)

    out = {
        "success": status == "ok",
        "status": status,
        "t0_s": t0,
        "t_corr_s": t_corr,
        "t_final_s": t_final,
        "dv0_raw_m_s": dv0_raw.tolist(),
        "dv1_raw_m_s": dv1_raw.tolist(),
        "dv0_levela_m_s": raw_to_levela(dv0_raw).tolist(),
        "dv1_levela_m_s": raw_to_levela(dv1_raw).tolist(),
        "dv0_norm_m_s": n(dv0_raw),
        "dv1_norm_m_s": n(dv1_raw),
        "total_dv_m_s": n(dv0_raw) + n(dv1_raw),
        "final_pos_err_km": None if rf is None else n(rf - target_r) / 1000.0,
        "final_vel_err_m_s": None if vf is None else n(vf - target_v),
        "optimizer_cost": float(res.cost),
        "optimizer_status": int(res.status),
        "optimizer_message": str(res.message),
        "nfev": int(res.nfev),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "result.json").write_text(json.dumps(out, indent=2) + "\n")

    # Mission events for DLL.
    vessel_guid = "60735c81-7e29-4c06-9551-9e5283e37586"
    base = {
        "enabled": True,
        "vessel_guid": vessel_guid,
        "ensure_flight_plan": True,
        "extend_existing_flight_plan": True,
        "mass_tonnes": 2.6,
        "insert_index": -1,
        "burn_template": "json",
        "thrust_kN": 2686.87701225281,
        "specific_impulse_s_g0": 1000.0,
        "is_inertially_fixed": False,
        "frame_extension": 6000,
        "frame_centre_from_active_body": True,
        "frame_centre_index": -1,
        "frame_primary_index": -1,
        "frame_secondary_index": -1,
        "placeholder_dv_m_s": 0.001,
        "require_status_ok": True,
        "cleanup_on_error": True,
        "tolerance_time_s": 0.01,
        "tolerance_dv_m_s": 1e-6,
        "one_shot": True,
        "disable_after_success": True,
    }

    ev0 = dict(base)
    ev0.update({
        "request_id": "rank12_twoimp_departure_attempt0",
        "dedupe_tag": "rank12_twoimp_departure",
        "mode": "insert_levela",
        "initial_time": t0,
        "plan_final_time": t0 + 3600.0,
        "delta_v_levela_m_s": out["dv0_levela_m_s"],
    })

    ev1 = dict(base)
    ev1.update({
        "request_id": "rank12_twoimp_correction_attempt0",
        "dedupe_tag": "rank12_twoimp_correction",
        "mode": "insert_levela",
        "initial_time": t_corr,
        "plan_final_time": t_corr + 3600.0,
        "delta_v_levela_m_s": out["dv1_levela_m_s"],
    })

    with (args.output_dir / "mission_events.jsonl").open("w") as f:
        f.write(json.dumps(ev0) + "\n")
        f.write(json.dumps(ev1) + "\n")

    print(json.dumps(out, indent=2))
    print("[OK] wrote", args.output_dir / "result.json")
    print("[OK] wrote", args.output_dir / "mission_events.jsonl")


if __name__ == "__main__":
    main()
