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
import spiceypy as spice

from ksp_mga.native.impulse_server_client_v0_2 import PrincipiaImpulseServerV2


def norm(v: np.ndarray) -> float:
    return float(np.linalg.norm(v))


def unit(v: np.ndarray) -> np.ndarray:
    m = norm(v)
    if m == 0:
        raise ValueError("zero vector")
    return v / m


def levela_to_raw(v: np.ndarray) -> np.ndarray:
    # LevelA/SPICE -> Principia raw = [+Z, -X, +Y]
    x, y, z = v
    return np.array([z, -x, y], dtype=float)


def raw_to_levela(v: np.ndarray) -> np.ndarray:
    x, y, z = v
    return np.array([-y, z, x], dtype=float)


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


def body_state_raw(body: str, t_s: float, center: str, frame: str) -> tuple[np.ndarray, np.ndarray]:
    st, _ = spice.spkezr(body, t_s, frame, "NONE", center)
    r_levela = np.array([1000.0 * st[0], 1000.0 * st[1], 1000.0 * st[2]])
    v_levela = np.array([1000.0 * st[3], 1000.0 * st[4], 1000.0 * st[5]])
    return levela_to_raw(r_levela), levela_to_raw(v_levela)


def body_mu_m3_s2(body: str) -> float:
    # TPC gerado pelo nosso conversor escreve BODY*_GM em km^3/s^2.
    _, vals = spice.bodvrd(body, "GM", 1)
    return float(vals[0]) * 1.0e9


def rtn_basis(r_rel: np.ndarray, v_rel: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    R = unit(r_rel)
    H = np.cross(r_rel, v_rel)
    N = unit(H)
    T = unit(np.cross(N, R))
    return R, T, N


def load_impulse_times(args, t0: float, row: dict) -> list[float]:
    if not args.impulse_time_s:
        t_dep = float(row["t_dep_s"])
        t_start = float(row["t_start_s"])

        if t_dep <= t0:
            raise SystemExit(
                f"[FAIL] planned departure t_dep={t_dep} is not after live t0={t0}. "
                "Reload an earlier save or pass --impulse-time-s explicitly."
            )

        impulse_times = [t_dep, t_start]
    else:
        impulse_times = [float(x) for x in args.impulse_time_s]

    for tb in impulse_times:
        if tb < t0:
            raise SystemExit(
                f"[FAIL] impulse_time_s={tb} is before live initial state t0={t0}"
            )

    if any(impulse_times[i] >= impulse_times[i + 1] for i in range(len(impulse_times) - 1)):
        raise SystemExit(f"[FAIL] impulse times must be strictly increasing: {impulse_times}")

    return impulse_times


def load_seed(args, n_impulses: int, row: dict) -> np.ndarray:
    seed = np.zeros((n_impulses, 3), dtype=float)
    if n_impulses > 0:
        # Use old leg correction as final impulse seed. It is only a seed, not a constraint.
        if "dvx_m_s" in row:
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
        "delta_v_levela_m_s": raw_to_levela(dv_raw).tolist(),
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

    ap.add_argument("--bsp", type=Path, required=True)
    ap.add_argument("--tpc", type=Path, required=True)
    ap.add_argument("--dep-body", default="KERBIN")
    ap.add_argument("--arr-body", default="EVE")
    ap.add_argument("--center", default="SUN")
    ap.add_argument("--frame", default="J2000")

    ap.add_argument("--dv0-t-min", type=float, default=500.0)
    ap.add_argument("--dv0-t-max", type=float, default=5000.0)
    ap.add_argument("--dv0-r-max", type=float, default=600.0)
    ap.add_argument("--dv0-n-max", type=float, default=600.0)

    ap.add_argument("--dv1-soft-max", type=float, default=500.0)
    ap.add_argument("--dv1-hard-max", type=float, default=900.0)

    ap.add_argument("--burn1-min-kerbin-distance-km", type=float, default=20000.0)
    ap.add_argument("--require-escape", action="store_true", default=True)

    ap.add_argument("--export-only-if-valid", action="store_true", default=True)
    
    ap.add_argument("--max-preburn-coast-s", type=float, default=900.0)
    ap.add_argument("--allow-long-preburn-coast", action="store_true")
    ap.add_argument("--burn0-max-kerbin-distance-km", type=float, default=10000.0)

    ap.add_argument("--final-pos-max-km", type=float, default=100000.0)
    ap.add_argument("--final-vel-max-m-s", type=float, default=1000.0)

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

    spice.kclear()
    spice.furnsh(str(args.tpc))
    spice.furnsh(str(args.bsp))
    mu_dep = body_mu_m3_s2(args.dep_body)

    live = json.loads(args.live_state_json.read_text())
    row = read_leg_row(args.leg_optimizations, args.leg)

    t0 = float(live["ut_s"])
    t_final = args.final_time if args.final_time is not None else float(row["t_end_s"])
    impulse_times = load_impulse_times(args, t0, row)
    n_impulses = len(impulse_times)
    
    tb0 = impulse_times[0]
    tb1 = impulse_times[1]
    
    preburn_coast_s = tb0 - t0

    if preburn_coast_s > args.max_preburn_coast_s and not args.allow_long_preburn_coast:
        raise SystemExit(
            f"[FAIL] first impulse is {preburn_coast_s:.1f} s after live_state t0. "
            f"Warp closer to the burn and recapture live_state_raw, or pass "
            f"--allow-long-preburn-coast explicitly. "
            f"Default max is {args.max_preburn_coast_s:.1f} s."
        )

    r0 = np.array(live["r_raw_m"], dtype=float)
    v0 = np.array(live["v_raw_m_s"], dtype=float)
    target_r = arr(row, "target_x_raw_m", "target_y_raw_m", "target_z_raw_m")
    target_v = arr(row, "target_vx_raw_m_s", "target_vy_raw_m_s", "target_vz_raw_m_s")

    old_corr_raw = np.zeros(3)
    if "dvx_m_s" in row:
        old_corr_raw = arr(row, "dvx_m_s", "dvy_m_s", "dvz_m_s")

    x0 = np.zeros(6, dtype=float)
    # fisicamente plausível:
    x0[0] = 1800.0   # tangencial
    x0[1] = 0.0      # radial
    x0[2] = 0.0      # normal
    x0[3:6] = old_corr_raw

    lower = np.array([
        args.dv0_t_min,
        -args.dv0_r_max,
        -args.dv0_n_max,
        -args.dv1_hard_max,
        -args.dv1_hard_max,
        -args.dv1_hard_max,
    ], dtype=float)

    upper = np.array([
        args.dv0_t_max,
        args.dv0_r_max,
        args.dv0_n_max,
        args.dv1_hard_max,
        args.dv1_hard_max,
        args.dv1_hard_max,
    ], dtype=float)

    pos_scale_m = args.pos_scale_km * 1000.0
    vel_scale = args.vel_scale_m_s
    dv_scale = args.dv_scale_m_s

    best = {"obj": float("inf"), "x": None, "pos_km": None, "vel_m_s": None}
    eval_count = 0

    def escape_penalty(burn0_r, burn0_v_after, tb0):
        body_r, body_v = body_state_raw(args.dep_body, tb0, args.center, args.frame)
        r_rel = burn0_r - body_r
        v_rel = burn0_v_after - body_v
        eps = 0.5 * norm(v_rel)**2 - mu_dep / norm(r_rel)
        if eps >= 0:
            return 0.0
        return (-eps) / 1.0e6

    def burn1_distance_penalty(burn1_r, tb1):
        body_r, _ = body_state_raw(args.dep_body, tb1, args.center, args.frame)
        d_km = norm(burn1_r - body_r) / 1000.0
        if d_km >= args.burn1_min_kerbin_distance_km:
            return 0.0
        return (args.burn1_min_kerbin_distance_km - d_km) / 5000.0

    def dv1_penalty(dv1_raw):
        d = norm(dv1_raw)
        pen = 0.0
        if d > args.dv1_soft_max:
            pen += (d - args.dv1_soft_max) / 200.0
        if d > args.dv1_hard_max:
            pen += (d - args.dv1_hard_max) * 50.0
        return pen

    def normal_fraction_penalty(dv0_t, dv0_r, dv0_n):
        pen_r = dv0_r / 300.0
        pen_n = dv0_n / 300.0
        
        d0_norm = math.sqrt(dv0_t**2 + dv0_r**2 + dv0_n**2) + 1e-9
        n_frac = abs(dv0_n) / d0_norm
        if n_frac > 0.33:
            pen_n += (n_frac - 0.33) * 500.0
            
        return np.array([pen_r, pen_n])

    print("=== PROPN 2-IMPULSE TARGET OPT (RTN) ===")
    print(f"server    : {args.server}")
    print(f"t0        : {t0}")
    print(f"t_final   : {t_final}")
    print(f"impulses  : {impulse_times}")
    print(f"target_r  : {target_r.tolist()}")
    print(f"target_v  : {target_v.tolist()}")

    with PrincipiaImpulseServerV2(args.server, args.plugin_b64) as srv:
        print(f"ready     : {srv.ready_line}")
        if not srv.ping():
            raise SystemExit("[FAIL] server PING failed")
            
        # Coast real da nave até a primeira queima.
        # A base RTN da departure deve ser calculada no estado pré-burn real,
        # não misturando r0(t0) com Kerbin(tb0).
        preburn = srv.propagate_n(
            req_id="preburn0",
            t0_s=t0,
            t1_s=tb0,
            r0_m=r0,
            v0_m_s=v0,
            impulses=[(tb0, np.zeros(3))],
        )

        if preburn.status != "ok":
            raise SystemExit(
                f"[FAIL] preburn coast failed: {preburn.status} {preburn.message}"
            )

        if not preburn.burns:
            raise SystemExit("[FAIL] preburn coast returned no burn records")

        preburn_r0 = np.array(preburn.burns[0].r_m, dtype=float)
        preburn_v0 = np.array(preburn.burns[0].v_before_m_s, dtype=float)

        dep_r_tb0, dep_v_tb0 = body_state_raw(
            args.dep_body, tb0, args.center, args.frame
        )

        preburn_rel_r = preburn_r0 - dep_r_tb0
        preburn_rel_v = preburn_v0 - dep_v_tb0
        preburn_distance_km = norm(preburn_rel_r) / 1000.0
        preburn_speed_m_s = norm(preburn_rel_v)

        print(
            f"preburn_distance_from_{args.dep_body}_km: "
            f"{preburn_distance_km:.6f}"
        )
        print(f"preburn_rel_speed_m_s: {preburn_speed_m_s:.6f}")

        if preburn_distance_km > args.burn0_max_kerbin_distance_km:
            raise SystemExit(
                f"[FAIL] preburn is not in parking orbit: "
                f"{preburn_distance_km:.3f} km from {args.dep_body}. "
                f"Expected <= {args.burn0_max_kerbin_distance_km:.3f} km."
            )

        R0_const, T0_const, N0_const = rtn_basis(preburn_rel_r, preburn_rel_v)

        def residual(x: np.ndarray) -> np.ndarray:
            nonlocal eval_count
            eval_count += 1
            
            dv0_t, dv0_r, dv0_n = x[0], x[1], x[2]
            dv0_raw = (
                dv0_t * T0_const +
                dv0_r * R0_const +
                dv0_n * N0_const
            )
            dv1_raw = np.array(x[3:6], dtype=float)
            
            impulses = [(tb0, dv0_raw), (tb1, dv1_raw)]
            
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
                n_res = 10 if args.ignore_velocity else 13
                return np.ones(n_res) * 1e6

            try:
                burn0_r = np.array(result.burns[0].r_m)
                burn0_v_after = np.array(result.burns[0].v_after_m_s)
                burn1_r = np.array(result.burns[1].r_m)
            except IndexError:
                print(f"[eval {eval_count:04d}] missing burns in response", flush=True)
                n_res = 10 if args.ignore_velocity else 13
                return np.ones(n_res) * 1e6

            pos_err = (result.final_r_m - target_r) / pos_scale_m
            vel_err = (result.final_v_m_s - target_v) / vel_scale
            
            res_list = [
                *pos_err,
            ]
            if not args.ignore_velocity:
                res_list.extend(vel_err)
                
            res_list.extend([
                norm(dv0_raw) / 8000.0 * 0.05,
                norm(dv1_raw) / 1000.0 * 0.10,
                escape_penalty(burn0_r, burn0_v_after, tb0) * 5.0,
                burn1_distance_penalty(burn1_r, tb1) * 3.0,
                dv1_penalty(dv1_raw) * 3.0,
                *(normal_fraction_penalty(dv0_t, dv0_r, dv0_n) * 0.5),
            ])
            res = np.array(res_list, dtype=float)
            obj = float(np.linalg.norm(res))

            if obj < best["obj"]:
                best.update({
                    "obj": obj,
                    "x": np.array(x, dtype=float),
                    "pos_km": norm(result.final_r_m - target_r) / 1000.0,
                    "vel_m_s": norm(result.final_v_m_s - target_v),
                })
                print(
                    f"[best {eval_count:04d}] obj={obj:.6g} "
                    f"pos={best['pos_km']:.3f} km vel={best['vel_m_s']:.3f} m/s "
                    f"dv0={norm(dv0_raw):.1f} dv1={norm(dv1_raw):.1f} "
                    f"(t={dv0_t:.1f} r={dv0_r:.1f} n={dv0_n:.1f})",
                    flush=True,
                )
            return res

        result_opt = least_squares(
            residual,
            x0,
            bounds=(lower, upper),
            max_nfev=args.max_nfev,
            x_scale=np.array([1500, 300, 300, 300, 300, 300], dtype=float),
            diff_step=1e-4,
            verbose=2,
        )

        x_best = best["x"] if best["x"] is not None else result_opt.x
        
        dv0_t, dv0_r, dv0_n = x_best[0], x_best[1], x_best[2]
        dvs_best = [
            dv0_t * T0_const + dv0_r * R0_const + dv0_n * N0_const,
            np.array(x_best[3:6], dtype=float),
        ]
        
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
        diagnostics = {}
        valid = False
        reasons = ["server_error"]
        dv0_t = dv0_r = dv0_n = 0.0
        burn0_distance_km = burn1_distance_km = escape_energy = 0.0
        dv0_raw = dvs_best[0] if 'dvs_best' in locals() else np.zeros(3)
        dv1_raw = dvs_best[1] if 'dvs_best' in locals() else np.zeros(3)
    else:
        final_pos_err_km = norm(final.final_r_m - target_r) / 1000.0
        final_vel_err_m_s = norm(final.final_v_m_s - target_v)

        try:
            burn0_r = np.array(final.burns[0].r_m)
            burn0_v_after = np.array(final.burns[0].v_after_m_s)
            burn1_r = np.array(final.burns[1].r_m)
            
            body0_r, body0_v = body_state_raw(args.dep_body, tb0, args.center, args.frame)
            burn0_distance_km = norm(burn0_r - body0_r) / 1000.0
            r_rel0 = burn0_r - body0_r
            v_rel0 = burn0_v_after - body0_v
            escape_energy = 0.5 * norm(v_rel0)**2 - mu_dep / norm(r_rel0)

            body1_r, _ = body_state_raw(args.dep_body, tb1, args.center, args.frame)
            burn1_distance_km = norm(burn1_r - body1_r) / 1000.0
        except AttributeError:
            burn0_distance_km = 0.0
            escape_energy = 0.0
            burn1_distance_km = 0.0

        dv0_raw = dvs_best[0]
        dv1_raw = dvs_best[1]
        
        diagnostics = {
            "dv0_t_m_s": float(dv0_t),
            "dv0_r_m_s": float(dv0_r),
            "dv0_n_m_s": float(dv0_n),
            "dv0_norm_m_s": norm(dv0_raw),
            "dv0_radial_fraction": abs(dv0_r) / max(norm(dv0_raw), 1e-9),
            "dv0_normal_fraction": abs(dv0_n) / max(norm(dv0_raw), 1e-9),

            "burn0_distance_from_kerbin_km": burn0_distance_km,
            "burn0_escape_energy_m2_s2": escape_energy,
            "burn0_escape": escape_energy > 0,

            "burn1_distance_from_kerbin_km": burn1_distance_km,
            "dv1_norm_m_s": norm(dv1_raw),

            "final_pos_err_km": final_pos_err_km,
            "final_vel_err_m_s": final_vel_err_m_s,
        }

        valid = True
        reasons = []

        if diagnostics["burn0_escape_energy_m2_s2"] <= 0:
            valid = False
            reasons.append("burn0_not_escape")

        if diagnostics["dv0_normal_fraction"] > 0.35:
            valid = False
            reasons.append("burn0_normal_fraction_too_high")

        if diagnostics["dv1_norm_m_s"] > args.dv1_hard_max:
            valid = False
            reasons.append("dv1_too_large")

        if diagnostics["burn1_distance_from_kerbin_km"] < args.burn1_min_kerbin_distance_km:
            valid = False
            reasons.append("burn1_too_close_to_kerbin")

        if final_pos_err_km is None or final_pos_err_km > args.final_pos_max_km:
            valid = False
            reasons.append("final_pos_err_too_large")

        if final_vel_err_m_s is None or final_vel_err_m_s > args.final_vel_max_m_s:
            valid = False
            reasons.append("final_vel_err_too_large")

        if abs(dv1_norm := norm(dv1_raw)) > args.dv1_soft_max * 1.6:
            valid = False
            reasons.append("dv1_saturated_or_too_high")

        if abs(dv0_t - args.dv0_t_min) < 1e-3:
            valid = False
            reasons.append("dv0_tangent_at_lower_bound")

        if abs(abs(dv0_r) - args.dv0_r_max) < 1e-3:
            valid = False
            reasons.append("dv0_radial_at_bound")

        if abs(abs(dv0_n) - args.dv0_n_max) < 1e-3:
            valid = False
            reasons.append("dv0_normal_at_bound")

    out = {
        "success": final.status == "ok",
        "physically_valid": valid,
        "invalid_reasons": reasons,
        "status": final.status,
        "message": final.message,
        "t0_s": t0,
        "t_final_s": t_final,
        "impulse_times_s": [float(t) for t in impulse_times],
        "dv_raw_m_s": [dvs_best[i].tolist() for i in range(n_impulses)],
        "dv_levela_m_s": [raw_to_levela(dvs_best[i]).tolist() for i in range(n_impulses)],
        "dv_norms_m_s": [norm(dvs_best[i]) for i in range(n_impulses)],
        "total_dv_m_s": sum(norm(dvs_best[i]) for i in range(n_impulses)),
        "final_pos_err_km": final_pos_err_km,
        "final_vel_err_m_s": final_vel_err_m_s,
        "best_objective": best["obj"],
        "optimizer_cost": float(result_opt.cost) if 'result_opt' in locals() else None,
        "optimizer_status": int(result_opt.status) if 'result_opt' in locals() else None,
        "optimizer_message": str(result_opt.message) if 'result_opt' in locals() else None,
        "nfev": int(result_opt.nfev) if 'result_opt' in locals() else None,
        "diagnostics": diagnostics,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "result.json").write_text(json.dumps(out, indent=2) + "\n")

    if args.export_only_if_valid and not valid:
        print(f"[FAIL] refusing to export mission_events.jsonl")
        print(f"invalid_reasons: {reasons}")
        print(json.dumps(out, indent=2))
        print("[OK] wrote", args.output_dir / "result.json")
        return 2

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
                "delta_v_levela_m_s": raw_to_levela(dvs_best[i]).tolist(),
            })
            f.write(json.dumps(ev, separators=(",", ":")) + "\n")

    print(json.dumps(out, indent=2))
    print("[OK] wrote", args.output_dir / "result.json")
    print("[OK] wrote", args.output_dir / "mission_events.jsonl")
    return 0 if out["success"] else 2


if __name__ == "__main__":
    raise SystemExit(main())