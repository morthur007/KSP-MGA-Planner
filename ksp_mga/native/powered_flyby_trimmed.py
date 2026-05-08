from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import least_squares

from ksp_mga.native.leg_optimizer import norm, sample_raw_body_state
from ksp_mga.native.powered_flyby_bridge import (
    ExtendedImpulseServer,
    LinearBodyEphemerisCache,
    build_setup,
)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def load_initial_x_from_history(path: Path, eval_id: int | None = None) -> np.ndarray:
    rows = read_csv_rows(path)

    if eval_id is not None:
        rows = [r for r in rows if int(r["eval"]) == eval_id]
        if not rows:
            raise KeyError(f"eval {eval_id} not found in {path}")

    def score(r: dict[str, str]) -> float:
        try:
            pos = float(r.get("pos_err_km", "inf"))
            vel = float(r.get("vel_err_m_s", "inf"))
            alt = float(r.get("burn_altitude_km", "inf"))
            vr = abs(float(r.get("burn_radial_v_km_s", "inf")))
            low_alt = max(0.0, 50.0 - alt)
            return pos + 0.05 * vel + 100.0 * vr + 10.0 * low_alt
        except Exception:
            return float("inf")

    row = min(rows, key=score)

    return np.array([
        float(row["dv0_x_m_s"]),
        float(row["dv0_y_m_s"]),
        float(row["dv0_z_m_s"]),
        float(row["burn_dv_x_m_s"]),
        float(row["burn_dv_y_m_s"]),
        float(row["burn_dv_z_m_s"]),
        float(row["burn_dt_s"]),
    ], dtype=float)


def write_history(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    fields: list[str] = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)

    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def evaluate(server: ExtendedImpulseServer, setup, x: np.ndarray):
    dv0 = np.asarray(x[:3], dtype=float)
    burn = np.asarray(x[3:6], dtype=float)
    burn_dt = float(x[6])
    burn_t = setup.t_event_s + burn_dt

    resp = server.propagate(
        req_id="trim_eval",
        t0_s=setup.t0_s,
        burn_t_s=burn_t,
        t1_s=setup.t1_s,
        r0_m=setup.r0_m,
        v0_m_s=setup.v0_m_s + dv0,
        burn_dv_m_s=burn,
    )

    return resp, dv0, burn, burn_t


def optimize_trimmed(args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    setup = build_setup(args)

    lower_dt = (setup.t0_s + args.time_margin_s) - setup.t_event_s
    upper_dt = (setup.t1_s - args.time_margin_s) - setup.t_event_s

    if lower_dt >= upper_dt:
        raise ValueError("time_margin_s leaves no valid burn window")

    x0 = np.zeros(7, dtype=float)

    if args.initial_burn_dt_s is not None:
        x0[6] = float(args.initial_burn_dt_s)

    if args.initial_burn_guess_m_s:
        dv_guess_dir = setup.target_v_m_s - setup.v0_m_s
        n = norm(dv_guess_dir)
        if n > 0:
            x0[3:6] = dv_guess_dir / n * args.initial_burn_guess_m_s

    if args.initial_history_csv is not None:
        x0 = load_initial_x_from_history(args.initial_history_csv, args.initial_history_eval)

    lb = np.array(
        [-args.max_dv0_m_s] * 3
        + [-args.max_burn_m_s] * 3
        + [lower_dt],
        dtype=float,
    )
    ub = np.array(
        [args.max_dv0_m_s] * 3
        + [args.max_burn_m_s] * 3
        + [upper_dt],
        dtype=float,
    )

    x0 = np.minimum(np.maximum(x0, lb), ub)

    x_scale = np.array(
        [args.dv0_x_scale_m_s] * 3
        + [args.burn_x_scale_m_s] * 3
        + [args.time_x_scale_s],
        dtype=float,
    )

    # One body sample only. No disk I/O inside the residual loop.
    body_r_ref, body_v_ref = sample_raw_body_state(
        sampler=args.sampler,
        plugin_b64=args.plugin_b64,
        target_body=setup.body,
        sampler_central_body=args.raw_origin_body,
        et_s=setup.t_event_s,
        plugin_base_et_s=args.plugin_base_et_s,
        work_dir=args.raw_cache_dir,
    )
    body_cache = LinearBodyEphemerisCache(setup.t_event_s, body_r_ref, body_v_ref)

    pos_scale_m = args.pos_scale_km * 1000.0

    history: list[dict[str, Any]] = []
    counter = {"n": 0}
    best = {
        "objective": float("inf"),
        "x": None,
        "metrics": None,
    }

    with ExtendedImpulseServer(args.server, args.plugin_b64) as server:

        def residual(x: np.ndarray) -> np.ndarray:
            counter["n"] += 1

            resp, dv0, burn, burn_t = evaluate(server, setup, x)

            if resp.status != "ok":
                r = np.full(8, 1e9, dtype=float)
                row = {
                    "eval": counter["n"],
                    "status": resp.status,
                    "message": resp.message,
                    "objective": float("inf"),
                    "dv0_x_m_s": dv0[0],
                    "dv0_y_m_s": dv0[1],
                    "dv0_z_m_s": dv0[2],
                    "dv0_norm_m_s": norm(dv0),
                    "burn_dv_x_m_s": burn[0],
                    "burn_dv_y_m_s": burn[1],
                    "burn_dv_z_m_s": burn[2],
                    "burn_dv_norm_m_s": norm(burn),
                    "burn_dt_s": float(x[6]),
                    "burn_t_s": burn_t,
                    "pos_err_km": math.inf,
                    "cleanup_dv_m_s": math.inf,
                    "burn_altitude_km": math.inf,
                    "burn_radial_v_km_s": math.inf,
                    "total_bridge_dv_m_s": math.inf,
                }
                history.append(row)
                return r

            pos_err = resp.final_r_m - setup.target_r_m
            cleanup_dv = setup.target_v_m_s - resp.final_v_m_s

            pos_err_km = norm(pos_err) / 1000.0
            cleanup_norm = norm(cleanup_dv)

            body_r, body_v = body_cache.state(burn_t)
            rel_r = resp.burn_r_m - body_r
            rel_v_before = resp.burn_v_before_m_s - body_v

            burn_altitude_km = norm(rel_r) / 1000.0 - setup.radius_km
            burn_radial_v_km_s = float(np.dot(rel_r, rel_v_before) / max(norm(rel_r), 1.0)) / 1000.0

            low_altitude_km = max(0.0, setup.min_altitude_km - burn_altitude_km)

            # Core trimmed objective:
            # - close position at outbound buffer;
            # - burn near periapsis: radial velocity near zero;
            # - never prefer unsafe low altitude;
            # - keep total maneuver cost small via soft regularization.
            core = [
                *(pos_err / pos_scale_m),
                burn_radial_v_km_s / args.radial_scale_km_s,
                low_altitude_km / args.low_altitude_scale_km,
            ]

            if args.target_burn_altitude_km is not None and args.target_altitude_weight > 0:
                core.append(
                    args.target_altitude_weight
                    * (burn_altitude_km - args.target_burn_altitude_km)
                    / args.periapsis_scale_km
                )

            reg = [
                *(args.regularization_weight * dv0 / args.dv0_regularization_m_s),
                *(args.regularization_weight * burn / args.burn_regularization_m_s),
                args.cleanup_weight * cleanup_norm / args.cleanup_scale_m_s,
            ]

            r = np.array(core + reg, dtype=float)
            objective = float(np.linalg.norm(r))

            total_bridge_dv = norm(dv0) + norm(burn) + cleanup_norm

            row = {
                "eval": counter["n"],
                "status": resp.status,
                "message": resp.message,
                "objective": objective,

                "dv0_x_m_s": dv0[0],
                "dv0_y_m_s": dv0[1],
                "dv0_z_m_s": dv0[2],
                "dv0_norm_m_s": norm(dv0),

                "burn_dv_x_m_s": burn[0],
                "burn_dv_y_m_s": burn[1],
                "burn_dv_z_m_s": burn[2],
                "burn_dv_norm_m_s": norm(burn),

                "burn_dt_s": float(x[6]),
                "burn_t_s": burn_t,

                "pos_err_km": pos_err_km,
                "cleanup_dv_m_s": cleanup_norm,
                "burn_altitude_km": burn_altitude_km,
                "burn_radial_v_km_s": burn_radial_v_km_s,
                "low_altitude_km": low_altitude_km,
                "total_bridge_dv_m_s": total_bridge_dv,
            }

            history.append(row)

            if (
                np.isfinite(objective)
                and objective < best["objective"]
                and burn_altitude_km >= setup.min_altitude_km
            ):
                best["objective"] = objective
                best["x"] = np.array(x, dtype=float)
                best["metrics"] = row

            print(
                f"[eval {counter['n']:04d}] "
                f"pos={pos_err_km:10.3f} km "
                f"cleanup={cleanup_norm:9.3f} m/s "
                f"dv0={norm(dv0):8.3f} "
                f"burn={norm(burn):8.3f} "
                f"total={total_bridge_dv:9.3f} "
                f"dt={x[6]:9.1f} "
                f"alt={burn_altitude_km:9.1f} "
                f"vr={burn_radial_v_km_s:8.3f}"
            )

            return r

        def jac_abs(x: np.ndarray) -> np.ndarray:
            f0 = residual(x)
            J = np.zeros((len(f0), len(x)), dtype=float)

            steps = np.array([
                args.fd_dv0_step_m_s,
                args.fd_dv0_step_m_s,
                args.fd_dv0_step_m_s,
                args.fd_burn_step_m_s,
                args.fd_burn_step_m_s,
                args.fd_burn_step_m_s,
                args.fd_time_step_s,
            ], dtype=float)

            for j, h in enumerate(steps):
                xp = np.array(x, dtype=float)
                xm = np.array(x, dtype=float)

                xp[j] = min(xp[j] + h, ub[j])
                xm[j] = max(xm[j] - h, lb[j])

                if xp[j] == xm[j]:
                    continue

                fp = residual(xp)
                fm = residual(xm)
                J[:, j] = (fp - fm) / (xp[j] - xm[j])

            return J

        sol = least_squares(
            residual,
            x0=x0,
            jac=jac_abs,
            bounds=(lb, ub),
            method="trf",
            x_scale=x_scale,
            max_nfev=args.max_nfev,
            ftol=args.ftol,
            xtol=args.xtol,
            gtol=args.gtol,
        )

        final_x = best["x"] if best["x"] is not None else sol.x
        final_resp, final_dv0, final_burn, final_burn_t = evaluate(server, setup, final_x)

    if final_resp.status == "ok":
        final_pos_err_km = norm(final_resp.final_r_m - setup.target_r_m) / 1000.0
        final_cleanup_vec = setup.target_v_m_s - final_resp.final_v_m_s
        final_cleanup_dv_m_s = norm(final_cleanup_vec)

        body_r, body_v = body_cache.state(final_burn_t)
        rel_r = final_resp.burn_r_m - body_r
        rel_v_before = final_resp.burn_v_before_m_s - body_v

        final_alt_km = norm(rel_r) / 1000.0 - setup.radius_km
        final_radial_km_s = float(np.dot(rel_r, rel_v_before) / max(norm(rel_r), 1.0)) / 1000.0
    else:
        final_pos_err_km = math.inf
        final_cleanup_dv_m_s = math.inf
        final_alt_km = math.inf
        final_radial_km_s = math.inf

    total_bridge_dv = norm(final_dv0) + norm(final_burn) + final_cleanup_dv_m_s

    success = (
        final_resp.status == "ok"
        and final_pos_err_km <= args.accept_pos_km
        and final_cleanup_dv_m_s <= args.accept_cleanup_m_s
        and abs(final_radial_km_s) <= args.accept_radial_km_s
        and final_alt_km >= setup.min_altitude_km
    )

    result = {
        "success": bool(success),
        "raw_solver_success": bool(sol.success),
        "solver_message": str(sol.message),
        "nfev": int(sol.nfev),
        "best_objective": float(best["objective"]),

        "flyby_index": setup.flyby_index,
        "body": setup.body,
        "leg_in": setup.leg_in,
        "leg_out": setup.leg_out,

        "t0_s": setup.t0_s,
        "t_event_s": setup.t_event_s,
        "t1_s": setup.t1_s,
        "burn_t_s": final_burn_t,
        "burn_dt_s": float(final_x[6]),

        "dv0_x_m_s": final_dv0[0],
        "dv0_y_m_s": final_dv0[1],
        "dv0_z_m_s": final_dv0[2],
        "dv0_norm_m_s": norm(final_dv0),

        "burn_dv_x_m_s": final_burn[0],
        "burn_dv_y_m_s": final_burn[1],
        "burn_dv_z_m_s": final_burn[2],
        "burn_dv_norm_m_s": norm(final_burn),

        "dv1_cleanup_norm_m_s": final_cleanup_dv_m_s,
        "total_bridge_dv_m_s": total_bridge_dv,

        "final_pos_err_km": final_pos_err_km,
        "burn_altitude_km": final_alt_km,
        "burn_radial_v_before_km_s": final_radial_km_s,
        "geometry_status": "OK" if final_alt_km >= setup.min_altitude_km else "LOW_ALTITUDE",

        "min_altitude_km": setup.min_altitude_km,
        "radius_km": setup.radius_km,
        "audit_vinf_mismatch_km_s": setup.vinf_mismatch_km_s,
        "audit_turn_margin_deg": setup.turn_margin_deg,
        "audit_alt_required_km": setup.alt_required_km,
    }

    return result, history


def main_cli() -> int:
    p = argparse.ArgumentParser(description="Trimmed native powered flyby bridge.")
    p.add_argument("--candidate-seed", type=Path, default=None)
    p.add_argument("--rank", type=int, default=None)

    p.add_argument("--leg-optimizations", type=Path, required=True)
    p.add_argument("--flyby-audit", type=Path, required=True)
    p.add_argument("--flyby-index", type=int, required=True)

    p.add_argument("--body-catalog", type=Path, required=True)
    p.add_argument("--plugin-b64", type=Path, required=True)
    p.add_argument("--server", default="principia_impulsive_particle_server")
    p.add_argument("--sampler", default="sample_principia_ephemeris")

    p.add_argument("--plugin-base-et-s", type=float, default=81.65168640136972)
    p.add_argument("--raw-origin-body", default="Sun")
    p.add_argument("--raw-cache-dir", type=Path, default=Path("data/runs/frame_debug/raw_cache"))

    p.add_argument("--min-altitude-km", type=float, default=50.0)
    p.add_argument("--atmosphere-margin-km", type=float, default=10.0)

    p.add_argument("--max-dv0-m-s", type=float, default=600.0)
    p.add_argument("--max-burn-m-s", type=float, default=800.0)
    p.add_argument("--time-margin-s", type=float, default=60.0)

    p.add_argument("--initial-burn-dt-s", type=float, default=None)
    p.add_argument("--initial-burn-guess-m-s", type=float, default=0.0)
    p.add_argument("--initial-history-csv", type=Path, default=None)
    p.add_argument("--initial-history-eval", type=int, default=None)

    p.add_argument("--pos-scale-km", type=float, default=20.0)
    p.add_argument("--radial-scale-km-s", type=float, default=0.1)
    p.add_argument("--low-altitude-scale-km", type=float, default=50.0)

    p.add_argument("--target-burn-altitude-km", type=float, default=None)
    p.add_argument("--periapsis-scale-km", type=float, default=200.0)
    p.add_argument("--target-altitude-weight", type=float, default=0.0)

    p.add_argument("--regularization-weight", type=float, default=0.02)
    p.add_argument("--cleanup-weight", type=float, default=0.05)
    p.add_argument("--dv0-regularization-m-s", type=float, default=300.0)
    p.add_argument("--burn-regularization-m-s", type=float, default=500.0)
    p.add_argument("--cleanup-scale-m-s", type=float, default=200.0)

    p.add_argument("--dv0-x-scale-m-s", type=float, default=100.0)
    p.add_argument("--burn-x-scale-m-s", type=float, default=150.0)
    p.add_argument("--time-x-scale-s", type=float, default=500.0)

    p.add_argument("--fd-dv0-step-m-s", type=float, default=10.0)
    p.add_argument("--fd-burn-step-m-s", type=float, default=10.0)
    p.add_argument("--fd-time-step-s", type=float, default=60.0)

    p.add_argument("--max-nfev", type=int, default=100)
    p.add_argument("--ftol", type=float, default=1e-10)
    p.add_argument("--xtol", type=float, default=1e-10)
    p.add_argument("--gtol", type=float, default=1e-10)

    p.add_argument("--accept-pos-km", type=float, default=20.0)
    p.add_argument("--accept-cleanup-m-s", type=float, default=250.0)
    p.add_argument("--accept-radial-km-s", type=float, default=0.1)

    p.add_argument("--work-dir", type=Path, required=True)
    p.add_argument("--output-json", type=Path, required=True)
    p.add_argument("--output-history-csv", type=Path, required=True)

    args = p.parse_args()

    if args.work_dir:
        args.work_dir.mkdir(parents=True, exist_ok=True)

    result, history = optimize_trimmed(args)

    write_history(args.output_history_csv, history)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2))

    print("")
    print("=== TRIMMED POWERED FLYBY BRIDGE RESULT ===")
    print(f"success             : {result['success']} {result['solver_message']}")
    print(f"nfev                : {result['nfev']}")
    print(f"final pos err km    : {result['final_pos_err_km']}")
    print(f"dv0 m/s             : {result['dv0_norm_m_s']}")
    print(f"burn dv m/s         : {result['burn_dv_norm_m_s']}")
    print(f"dv1 cleanup m/s     : {result['dv1_cleanup_norm_m_s']}")
    print(f"total bridge dv m/s : {result['total_bridge_dv_m_s']}")
    print(f"burn alt km         : {result['burn_altitude_km']}")
    print(f"burn radial km/s    : {result['burn_radial_v_before_km_s']}")
    print(f"burn dt s           : {result['burn_dt_s']}")
    print(f"geometry            : {result['geometry_status']}")
    print(f"[OK] result : {args.output_json}")
    print(f"[OK] history: {args.output_history_csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main_cli())
