#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
import spiceypy as spice

from ksp_mga.native.impulse_server_client_v0_2 import PrincipiaImpulseServerV2


def norm(v):
    return float(np.linalg.norm(v))


def unit(v):
    n = norm(v)
    if n == 0:
        raise ValueError("zero vector")
    return v / n


def raw_to_levela(v):
    x, y, z = v
    return np.array([-y, z, x], dtype=float)


def levela_to_raw(v):
    x, y, z = v
    return np.array([z, -x, y], dtype=float)


def arr(row, *names):
    return np.array([float(row[n]) for n in names], dtype=float)


def row_for_leg(path: Path, leg: int):
    with path.open() as f:
        for row in csv.DictReader(f):
            if int(float(row["leg"])) == leg:
                return row
    raise SystemExit(f"[FAIL] leg {leg} not found")


def body_state_raw(body, t_s, center, frame):
    st, _ = spice.spkezr(body, t_s, frame, "NONE", center)
    r_levela = np.array([1000 * st[0], 1000 * st[1], 1000 * st[2]])
    v_levela = np.array([1000 * st[3], 1000 * st[4], 1000 * st[5]])
    return levela_to_raw(r_levela), levela_to_raw(v_levela)


def body_mu_m3_s2(body):
    _, vals = spice.bodvrd(body, "GM", 1)
    return float(vals[0]) * 1e9


def rtn_basis(r_rel, v_rel):
    R = unit(r_rel)
    N = unit(np.cross(r_rel, v_rel))
    T = unit(np.cross(N, R))
    return R, T, N


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plugin-b64", required=True)
    ap.add_argument("--server", default="principia_impulsive_particle_server")
    ap.add_argument("--live-state-json", type=Path, required=True)
    ap.add_argument("--leg-optimizations", type=Path, required=True)
    ap.add_argument("--leg", type=int, default=1)
    ap.add_argument("--bsp", type=Path, required=True)
    ap.add_argument("--tpc", type=Path, required=True)
    ap.add_argument("--dep-body", default="KERBIN")
    ap.add_argument("--center", default="SUN")
    ap.add_argument("--frame", default="J2000")
    ap.add_argument("--tb0", type=float, required=True)
    ap.add_argument("--tb1", type=float, default=None, help="Default: row t_start_s")
    ap.add_argument("--output-dir", type=Path, required=True)

    ap.add_argument("--dv0-t-min", type=float, default=1300)
    ap.add_argument("--dv0-t-max", type=float, default=3500)
    ap.add_argument("--dv0-r-max", type=float, default=300)
    ap.add_argument("--dv0-n-max", type=float, default=300)
    ap.add_argument("--dv1-max", type=float, default=600)

    ap.add_argument("--pos-scale-km", type=float, default=1000)
    ap.add_argument("--vel-scale-m-s", type=float, default=50)
    ap.add_argument("--max-nfev", type=int, default=200)

    ap.add_argument("--vessel-guid", default="60735c81-7e29-4c06-9551-9e5283e37586")
    ap.add_argument("--event-prefix", default="rank12_patchpoint")
    ap.add_argument("--plan-duration-s", type=float, default=600)
    args = ap.parse_args()

    spice.kclear()
    spice.furnsh(str(args.tpc))
    spice.furnsh(str(args.bsp))

    mu = body_mu_m3_s2(args.dep_body)

    live = json.loads(args.live_state_json.read_text())
    row = row_for_leg(args.leg_optimizations, args.leg)

    t0 = float(live["ut_s"])
    tb0 = args.tb0
    tb1 = args.tb1 if args.tb1 is not None else float(row["t_start_s"])

    r0 = np.array(live["r_raw_m"], dtype=float)
    v0 = np.array(live["v_raw_m_s"], dtype=float)

    target_r = arr(row, "start_x_raw_m", "start_y_raw_m", "start_z_raw_m")

    target_v_pre = arr(
        row,
        "start_vx_raw_m_s",
        "start_vy_raw_m_s",
        "start_vz_raw_m_s",
    )

    leg_correction_dv = arr(row, "dvx_m_s", "dvy_m_s", "dvz_m_s")

    # O estado final depois da segunda manobra deve ser o estado pós-correção
    # que a leg validada usa para chegar em Eve.
    target_v = target_v_pre + leg_correction_dv

    pos_scale = args.pos_scale_km * 1000
    vel_scale = args.vel_scale_m_s

    best = {"obj": float("inf"), "x": None, "pos_km": None, "vel_m_s": None}
    eval_count = 0

    lower = np.array([
        args.dv0_t_min,
        -args.dv0_r_max,
        -args.dv0_n_max,
        -args.dv1_max,
        -args.dv1_max,
        -args.dv1_max,
    ], dtype=float)

    upper = np.array([
        args.dv0_t_max,
        args.dv0_r_max,
        args.dv0_n_max,
        args.dv1_max,
        args.dv1_max,
        args.dv1_max,
    ], dtype=float)

    x0 = np.array([
        1800.0,
        0.0,
        0.0,
        leg_correction_dv[0],
        leg_correction_dv[1],
        leg_correction_dv[2],
    ], dtype=float)

    print("=== OPTIMIZE DEPARTURE TO LEG START ===")
    print("t0 :", t0)
    print("tb0:", tb0)
    print("tb1:", tb1)
    print("target_r:", target_r.tolist())
    print("target_v:", target_v.tolist())

    with PrincipiaImpulseServerV2(args.server, args.plugin_b64) as srv:
        print("ready:", srv.ready_line)
        assert srv.ping()

        preburn = srv.propagate_n(
            req_id="preburn",
            t0_s=t0,
            t1_s=tb0,
            r0_m=r0,
            v0_m_s=v0,
            impulses=[(tb0, np.zeros(3))],
        )
        if preburn.status != "ok":
            raise SystemExit(f"[FAIL] preburn failed: {preburn.status} {preburn.message}")

        burn0_r = np.array(preburn.burns[0].r_m)
        burn0_v_before = np.array(preburn.burns[0].v_before_m_s)

        dep_r, dep_v = body_state_raw(args.dep_body, tb0, args.center, args.frame)
        rel_r = burn0_r - dep_r
        rel_v = burn0_v_before - dep_v
        R0, T0, N0 = rtn_basis(rel_r, rel_v)

        print("preburn_distance_km:", norm(rel_r) / 1000)
        print("preburn_speed_m_s:", norm(rel_v))

        def residual(x):
            nonlocal eval_count
            eval_count += 1

            dv0_t, dv0_r, dv0_n = x[0], x[1], x[2]
            dv0 = dv0_t * T0 + dv0_r * R0 + dv0_n * N0
            dv1 = np.array(x[3:6], dtype=float)

            res = srv.propagate_n(
                req_id=f"eval{eval_count}",
                t0_s=t0,
                t1_s=tb1,
                r0_m=r0,
                v0_m_s=v0,
                impulses=[(tb0, dv0), (tb1, dv1)],
            )

            if res.status != "ok":
                return np.ones(11) * 1e6

            final_r = np.array(res.final_r_m)
            final_v = np.array(res.final_v_m_s)

            pos_err = final_r - target_r
            vel_err = final_v - target_v

            # Burn0 escape check.
            b0_r = np.array(res.burns[0].r_m)
            b0_va = np.array(res.burns[0].v_after_m_s)
            body_r, body_v = body_state_raw(args.dep_body, tb0, args.center, args.frame)
            rr = b0_r - body_r
            vv = b0_va - body_v
            eps = 0.5 * norm(vv)**2 - mu / norm(rr)
            escape_pen = max(0.0, -eps / 1e6)

            out = np.concatenate([
                pos_err / pos_scale,
                vel_err / vel_scale,
                np.array([
                    norm(dv0) / 4000 * 0.05,
                    norm(dv1) / args.dv1_max * 0.20,
                    escape_pen * 5.0,
                    dv0_r / 300 * 0.2,
                    dv0_n / 300 * 0.2,
                ])
            ])

            obj = norm(out)
            if obj < best["obj"]:
                best.update({
                    "obj": obj,
                    "x": x.copy(),
                    "pos_km": norm(pos_err) / 1000,
                    "vel_m_s": norm(vel_err),
                })
                print(
                    f"[best {eval_count:04d}] obj={obj:.6g} "
                    f"pos={best['pos_km']:.3f} km "
                    f"vel={best['vel_m_s']:.3f} m/s "
                    f"dv0={norm(dv0):.1f} dv1={norm(dv1):.1f} "
                    f"(t={dv0_t:.1f} r={dv0_r:.1f} n={dv0_n:.1f})",
                    flush=True,
                )

            return out

        opt = least_squares(
            residual,
            x0,
            bounds=(lower, upper),
            x_scale=np.array([1500, 300, 300, 200, 200, 200], dtype=float),
            max_nfev=args.max_nfev,
            verbose=2,
            diff_step=1e-4,
        )

        x = best["x"] if best["x"] is not None else opt.x
        dv0_t, dv0_r, dv0_n = x[0], x[1], x[2]
        dv0 = dv0_t * T0 + dv0_r * R0 + dv0_n * N0
        dv1 = np.array(x[3:6], dtype=float)

        final = srv.propagate_n(
            req_id="final",
            t0_s=t0,
            t1_s=tb1,
            r0_m=r0,
            v0_m_s=v0,
            impulses=[(tb0, dv0), (tb1, dv1)],
        )

    final_r = np.array(final.final_r_m)
    final_v = np.array(final.final_v_m_s)

    pos_err_km = norm(final_r - target_r) / 1000
    vel_err = norm(final_v - target_v)

    valid = (
        final.status == "ok"
        and pos_err_km < 1000
        and vel_err < 50
        and norm(dv1) < args.dv1_max
    )

    out = {
        "success": final.status == "ok",
        "physically_valid": valid,
        "status": final.status,
        "message": final.message,
        "t0_s": t0,
        "tb0_s": tb0,
        "tb1_s": tb1,
        "target_v_pre_raw_m_s": target_v_pre.tolist(),
        "leg_correction_dv_raw_m_s": leg_correction_dv.tolist(),
        "target_v_post_raw_m_s": target_v.tolist(),
        "dv0_raw_m_s": dv0.tolist(),
        "dv1_raw_m_s": dv1.tolist(),
        "dv0_levela_m_s": raw_to_levela(dv0).tolist(),
        "dv1_levela_m_s": raw_to_levela(dv1).tolist(),
        "dv0_norm_m_s": norm(dv0),
        "dv1_norm_m_s": norm(dv1),
        "total_dv_m_s": norm(dv0) + norm(dv1),
        "patchpoint_pos_err_km": pos_err_km,
        "patchpoint_vel_err_m_s": vel_err,
        "optimizer_cost": float(opt.cost),
        "optimizer_status": int(opt.status),
        "optimizer_message": str(opt.message),
        "nfev": int(opt.nfev),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "result.json").write_text(json.dumps(out, indent=2) + "\n")

    base = {
        "enabled": True,
        "vessel_guid": args.vessel_guid,
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
        "request_id": f"{args.event_prefix}_departure_attempt0",
        "dedupe_tag": f"{args.event_prefix}_departure",
        "event_key": f"{args.event_prefix}_departure",
        "attempt": 0,
        "mode": "insert_levela",
        "initial_time": tb0,
        "plan_final_time": tb0 + args.plan_duration_s,
        "delta_v_levela_m_s": raw_to_levela(dv0).tolist(),
    })

    ev1 = dict(base)
    ev1.update({
        "request_id": f"{args.event_prefix}_patch_attempt0",
        "dedupe_tag": f"{args.event_prefix}_patch",
        "event_key": f"{args.event_prefix}_patch",
        "attempt": 0,
        "mode": "insert_levela",
        "initial_time": tb1,
        "plan_final_time": tb1 + args.plan_duration_s,
        "delta_v_levela_m_s": raw_to_levela(dv1).tolist(),
    })
    if not valid:
        print("[FAIL] refusing to export mission_events.jsonl")
        print(json.dumps(out, indent=2))
        print("[OK] wrote", args.output_dir / "result.json")
        return 2
    with (args.output_dir / "mission_events.jsonl").open("w") as f:
        f.write(json.dumps(ev0, separators=(",", ":")) + "\n")
        f.write(json.dumps(ev1, separators=(",", ":")) + "\n")

    print(json.dumps(out, indent=2))
    print("[OK] wrote", args.output_dir / "result.json")
    print("[OK] wrote", args.output_dir / "mission_events.jsonl")

    return 0 if valid else 2


if __name__ == "__main__":
    raise SystemExit(main())
