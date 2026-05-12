#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from ksp_mga.native.impulse_server_client_v0_2 import PrincipiaImpulseServerV2


def norm(v: np.ndarray) -> float:
    return float(np.linalg.norm(v))


def raw_to_levela(v: np.ndarray) -> list[float]:
    x, y, z = [float(a) for a in v]
    return [-y, z, x]


def parse_vec3(s: str) -> np.ndarray:
    parts = [float(x.strip()) for x in s.replace(";", ",").split(",") if x.strip()]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(f"expected x,y,z vector, got {s!r}")
    return np.array(parts, dtype=float)


def read_leg_row(path: Path, leg: int) -> dict:
    rows = list(csv.DictReader(path.open()))
    for row in rows:
        if int(float(row["leg"])) == leg:
            return row
    raise SystemExit(f"[FAIL] leg {leg} not found in {path}")


def arr(row: dict, *names: str) -> np.ndarray:
    return np.array([float(row[n]) for n in names], dtype=float)


def load_impulse_times(args, t0: float, row: dict) -> list[float]:
    if args.impulse_time_s:
        times = [float(x) for x in args.impulse_time_s]
    elif args.impulse_fraction:
        t_final = args.final_time if args.final_time is not None else float(row["t_end_s"])
        times = [t0 + float(frac) * (t_final - t0) for frac in args.impulse_fraction]
    else:
        # Robust default for live departure refinement:
        # burn 0 now/current epoch, burn 1 at original leg start/DSM epoch.
        times = [t0, float(row["t_start_s"])]

    if sorted(times) != times:
        raise SystemExit(f"[FAIL] impulse times are not monotonic: {times}")
    if times and times[0] < t0 - 1e-6:
        raise SystemExit(f"[FAIL] first impulse time {times[0]} is before t0 {t0}")
    return times


def load_seed(args, n_impulses: int, row: dict) -> np.ndarray:
    seed = np.zeros((n_impulses, 3), dtype=float)
    if n_impulses > 0:
        # Use old leg correction as final impulse seed. It is only a seed, not a constraint.
        seed[-1, :] = arr(row, "dvx_m_s", "dvy_m_s", "dvz_m_s")

    for spec in args.seed_dv_raw or []:
        # format: index:x,y,z   e.g. 0:-500,2000,-400
        idx_s, vec_s = spec.split(":", 1)
        idx = int(idx_s)
        if not (0 <= idx < n_impulses):
            raise SystemExit(f"[FAIL] seed index {idx} out of range for {n_impulses} impulses")
        seed[idx, :] = parse_vec3(vec_s)
    return seed.reshape(-1)


def make_event(base: dict, request_id: str, dedupe_tag: str, burn_t: float, dv_raw: np.ndarray, plan_duration_s: float) -> dict:
    ev = dict(base)
    ev.update({
        "enabled": True,
        "request_id": request_id,
        "dedupe_tag": dedupe_tag,
        "mode": "insert_levela",
        "initial_time": float(burn_t),
        "plan_final_time": float(burn_t + plan_duration_s),
        "delta_v_levela_m_s": raw_to_levela(dv_raw),
    })
    return ev


def main() -> int:
    ap = argparse.ArgumentParser(description="Optimize N-impulse Principia raw trajectory to a leg target state.")
    ap.add_argument("--plugin-b64", required=True)
    ap.add_argument("--server", default="bin/x64/principia_impulsive_particle_server")
    ap.add_argument("--live-state-json", type=Path, required=True)
    ap.add_argument("--leg-optimizations", type=Path, required=True)
    ap.add_argument("--leg", type=int, default=1)
    ap.add_argument("--output-dir", type=Path, required=True)

    ap.add_argument("--final-time", type=float, default=None, help="Default: leg t_end_s")
    ap.add_argument("--impulse-time-s", action="append", default=None, help="Repeatable absolute impulse time in seconds.")
    ap.add_argument("--impulse-fraction", action="append", default=None, help="Repeatable fraction between t0 and final_time.")
    ap.add_argument("--seed-dv-raw", action="append", default=None, help="Repeatable index:x,y,z seed in raw m/s.")

    ap.add_argument("--max-nfev", type=int, default=120)
    ap.add_argument("--pos-scale-km", type=float, default=100000.0)
    ap.add_argument("--vel-scale-m-s", type=float, default=500.0)
    ap.add_argument("--dv-scale-m-s", type=float, default=3000.0)
    ap.add_argument("--dv-penalty", type=float, default=0.01)
    ap.add_argument("--ignore-velocity", action="store_true", help="Optimize only final position + dv penalty.")

    ap.add_argument("--vessel-guid", default="60735c81-7e29-4c06-9551-9e5283e37586")
    ap.add_argument("--mass-tonnes", type=float, default=2.6)
    ap.add_argument("--thrust-kN", type=float, default=2686.87701225281)
    ap.add_argument("--isp-s", type=float, default=1000.0)
    ap.add_argument("--event-prefix", default="rank12_propn")
    ap.add_argument("--plan-duration-s", type=float, default=3600.0)
    args = ap.parse_args()

    live = json.loads(args.live_state_json.read_text())
    row = read_leg_row(args.leg_optimizations, args.leg)

    t0 = float(live["ut_s"])
    t_final = args.final_time if args.final_time is not None else float(row["t_end_s"])
    impulse_times = load_impulse_times(args, t0, row)
    n_impulses = len(impulse_times)
    if n_impulses < 1:
        raise SystemExit("[FAIL] at least one impulse is required")

    r0 = np.array(live["r_raw_m"], dtype=float)
    v0 = np.array(live["v_raw_m_s"], dtype=float)
    target_r = arr(row, "target_x_raw_m", "target_y_raw_m", "target_z_raw_m")
    target_v = arr(row, "target_vx_raw_m_s", "target_vy_raw_m_s", "target_vz_raw_m_s")

    x0 = load_seed(args, n_impulses, row)

    pos_scale_m = args.pos_scale_km * 1000.0
    vel_scale = args.vel_scale_m_s
    dv_scale = args.dv_scale_m_s

    best = {"obj": float("inf"), "x": None, "pos_km": None, "vel_m_s": None}
    eval_count = 0

    print("=== PROPN N-IMPULSE TARGET OPT ===")
    print(f"server    : {args.server}")
    print(f"t0        : {t0}")
    print(f"t_final   : {t_final}")
    print(f"impulses  : {impulse_times}")
    print(f"x0 norms  : {[norm(x0.reshape(n_impulses, 3)[i]) for i in range(n_impulses)]}")
    print(f"target_r  : {target_r.tolist()}")
    print(f"target_v  : {target_v.tolist()}")

    with PrincipiaImpulseServerV2(args.server, args.plugin_b64) as srv:
        print(f"ready     : {srv.ready_line}")
        if not srv.ping():
            raise SystemExit("[FAIL] server PING failed")

        def residual(x: np.ndarray) -> np.ndarray:
            nonlocal eval_count
            eval_count += 1
            dvs = np.asarray(x, dtype=float).reshape(n_impulses, 3)
            impulses = [(impulse_times[i], dvs[i]) for i in range(n_impulses)]
            result = srv.propagate_n(
                req_id=f"eval{eval_count}",
                t0_s=t0,
                t1_s=t_final,
                r0_m=r0,
                v0_m_s=v0,
                impulses=impulses,
            )
            if result.status != "ok" or result.final_r_m is None or result.final_v_m_s is None:
                print(f"[eval {eval_count:04d}] server_error {result.status} {result.message}", flush=True)
                return np.ones(6 + 3 * n_impulses) * 1e6

            pos_err = result.final_r_m - target_r
            vel_err = result.final_v_m_s - target_v
            pieces = [pos_err / pos_scale_m]
            if not args.ignore_velocity:
                pieces.append(vel_err / vel_scale)
            pieces.append((dvs.reshape(-1) / dv_scale) * args.dv_penalty)
            res = np.concatenate(pieces)
            obj = float(np.linalg.norm(res))

            if obj < best["obj"]:
                best.update({
                    "obj": obj,
                    "x": np.array(x, dtype=float),
                    "pos_km": norm(pos_err) / 1000.0,
                    "vel_m_s": norm(vel_err),
                })
                print(
                    f"[best {eval_count:04d}] obj={obj:.6g} "
                    f"pos={best['pos_km']:.3f} km vel={best['vel_m_s']:.3f} m/s "
                    f"dv={[round(norm(dvs[i]),3) for i in range(n_impulses)]} "
                    f"total={sum(norm(dvs[i]) for i in range(n_impulses)):.3f}",
                    flush=True,
                )
            return res

        result_opt = least_squares(
            residual,
            x0,
            max_nfev=args.max_nfev,
            x_scale=np.full(3 * n_impulses, args.dv_scale_m_s, dtype=float),
            diff_step=1e-4,
            verbose=2,
        )

        x_best = best["x"] if best["x"] is not None else result_opt.x
        dvs_best = x_best.reshape(n_impulses, 3)
        final = srv.propagate_n(
            req_id="final",
            t0_s=t0,
            t1_s=t_final,
            r0_m=r0,
            v0_m_s=v0,
            impulses=[(impulse_times[i], dvs_best[i]) for i in range(n_impulses)],
        )

    if final.status != "ok" or final.final_r_m is None or final.final_v_m_s is None:
        final_pos_err_km = None
        final_vel_err_m_s = None
    else:
        final_pos_err_km = norm(final.final_r_m - target_r) / 1000.0
        final_vel_err_m_s = norm(final.final_v_m_s - target_v)

    out = {
        "success": final.status == "ok",
        "status": final.status,
        "message": final.message,
        "t0_s": t0,
        "t_final_s": t_final,
        "impulse_times_s": [float(t) for t in impulse_times],
        "dv_raw_m_s": [dvs_best[i].tolist() for i in range(n_impulses)],
        "dv_levela_m_s": [raw_to_levela(dvs_best[i]) for i in range(n_impulses)],
        "dv_norms_m_s": [norm(dvs_best[i]) for i in range(n_impulses)],
        "total_dv_m_s": sum(norm(dvs_best[i]) for i in range(n_impulses)),
        "final_pos_err_km": final_pos_err_km,
        "final_vel_err_m_s": final_vel_err_m_s,
        "best_objective": best["obj"],
        "optimizer_cost": float(result_opt.cost),
        "optimizer_status": int(result_opt.status),
        "optimizer_message": str(result_opt.message),
        "nfev": int(result_opt.nfev),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "result.json").write_text(json.dumps(out, indent=2) + "\n")

    base_event = {
        "enabled": True,
        "vessel_guid": args.vessel_guid,
        "ensure_flight_plan": True,
        "extend_existing_flight_plan": True,
        "mass_tonnes": args.mass_tonnes,
        "insert_index": -1,
        "burn_template": "json",
        "thrust_kN": args.thrust_kN,
        "specific_impulse_s_g0": args.isp_s,
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

    with (args.output_dir / "mission_events.jsonl").open("w") as f:
        for i in range(n_impulses):
            ev = dict(base_event)
            ev.update({
                "request_id": f"{args.event_prefix}_impulse{i}_attempt0",
                "dedupe_tag": f"{args.event_prefix}_impulse{i}",
                "event_key": f"{args.event_prefix}_impulse{i}",
                "attempt": 0,
                "mode": "insert_levela",
                "initial_time": float(impulse_times[i]),
                "plan_final_time": float(impulse_times[i] + args.plan_duration_s),
                "delta_v_levela_m_s": raw_to_levela(dvs_best[i]),
            })
            f.write(json.dumps(ev, separators=(",", ":")) + "\n")

    print(json.dumps(out, indent=2))
    print("[OK] wrote", args.output_dir / "result.json")
    print("[OK] wrote", args.output_dir / "mission_events.jsonl")
    return 0 if out["success"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
