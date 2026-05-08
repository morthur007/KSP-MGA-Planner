from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import least_squares

from ksp_mga.native.leg_optimizer import norm, norm_name, sample_raw_body_state
from ksp_mga.native.flyby_audit import (
    load_json,
    get_body_record,
    body_radius_km,
    body_atmosphere_km,
)


@dataclass
class BridgeResponse:
    status: str
    message: str

    burn_r_m: np.ndarray
    burn_v_before_m_s: np.ndarray
    burn_v_after_m_s: np.ndarray

    final_r_m: np.ndarray
    final_v_m_s: np.ndarray


class LinearBodyEphemerisCache:
    """Fast local body ephemeris approximation to avoid disk I/O inside optimizer loops."""
    def __init__(self, t_ref_s: float, r_ref_m: np.ndarray, v_ref_m_s: np.ndarray):
        self.t_ref_s = float(t_ref_s)
        self.r_ref_m = np.asarray(r_ref_m, dtype=float)
        self.v_ref_m_s = np.asarray(v_ref_m_s, dtype=float)

    def state(self, t_s: float) -> tuple[np.ndarray, np.ndarray]:
        dt = float(t_s) - self.t_ref_s
        return self.r_ref_m + self.v_ref_m_s * dt, self.v_ref_m_s


class ExtendedImpulseServer:
    """
    Client for principia_impulsive_particle_server TSV protocol.

    Expected OK response layout:
      0  OK
      1  id
      2  t0_s
      3  burn_t_s
      4  t1_s
      5  burn_x_m
      6  burn_y_m
      7  burn_z_m
      8  burn_vx_before_m_s
      9  burn_vy_before_m_s
      10 burn_vz_before_m_s
      11 burn_vx_after_m_s
      12 burn_vy_after_m_s
      13 burn_vz_after_m_s
      14 final_x_m
      15 final_y_m
      16 final_z_m
      17 final_vx_m_s
      18 final_vy_m_s
      19 final_vz_m_s
    """

    def __init__(self, executable: str, plugin_b64: Path):
        cmd = executable.split() + [str(plugin_b64)]
        self.proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )

        assert self.proc.stdout is not None
        ready = self.proc.stdout.readline().strip()
        if not ready.startswith("READY"):
            raise RuntimeError(f"failed to start impulse server: {ready}")

    def close(self) -> None:
        if self.proc.poll() is None:
            try:
                assert self.proc.stdin is not None
                self.proc.stdin.write("QUIT\n")
                self.proc.stdin.flush()
                self.proc.wait(timeout=2)
            except Exception:
                self.proc.kill()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def propagate(
        self,
        *,
        req_id: str,
        t0_s: float,
        burn_t_s: float,
        t1_s: float,
        r0_m: np.ndarray,
        v0_m_s: np.ndarray,
        burn_dv_m_s: np.ndarray,
    ) -> BridgeResponse:
        assert self.proc.stdin is not None
        assert self.proc.stdout is not None

        req = (
            f"PROP\t{req_id}\t{t0_s:.17g}\t{burn_t_s:.17g}\t{t1_s:.17g}\t"
            f"{r0_m[0]:.17g}\t{r0_m[1]:.17g}\t{r0_m[2]:.17g}\t"
            f"{v0_m_s[0]:.17g}\t{v0_m_s[1]:.17g}\t{v0_m_s[2]:.17g}\t"
            f"{burn_dv_m_s[0]:.17g}\t{burn_dv_m_s[1]:.17g}\t{burn_dv_m_s[2]:.17g}\n"
        )

        self.proc.stdin.write(req)
        self.proc.stdin.flush()

        resp = self.proc.stdout.readline().strip()
        if not resp:
            return BridgeResponse(
                status="crash",
                message="empty response",
                burn_r_m=np.full(3, np.nan),
                burn_v_before_m_s=np.full(3, np.nan),
                burn_v_after_m_s=np.full(3, np.nan),
                final_r_m=np.full(3, np.nan),
                final_v_m_s=np.full(3, np.nan),
            )

        parts = resp.split("\t")

        if parts[0] != "OK":
            return BridgeResponse(
                status="error",
                message=parts[2] if len(parts) > 2 else resp,
                burn_r_m=np.full(3, np.nan),
                burn_v_before_m_s=np.full(3, np.nan),
                burn_v_after_m_s=np.full(3, np.nan),
                final_r_m=np.full(3, np.nan),
                final_v_m_s=np.full(3, np.nan),
            )

        if len(parts) < 20:
            return BridgeResponse(
                status="error",
                message=f"short OK response with {len(parts)} fields: {resp}",
                burn_r_m=np.full(3, np.nan),
                burn_v_before_m_s=np.full(3, np.nan),
                burn_v_after_m_s=np.full(3, np.nan),
                final_r_m=np.full(3, np.nan),
                final_v_m_s=np.full(3, np.nan),
            )

        return BridgeResponse(
            status="ok",
            message="",
            burn_r_m=np.array([float(parts[5]), float(parts[6]), float(parts[7])]),
            burn_v_before_m_s=np.array([float(parts[8]), float(parts[9]), float(parts[10])]),
            burn_v_after_m_s=np.array([float(parts[11]), float(parts[12]), float(parts[13])]),
            final_r_m=np.array([float(parts[14]), float(parts[15]), float(parts[16])]),
            final_v_m_s=np.array([float(parts[17]), float(parts[18]), float(parts[19])]),
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
            if "objective" in r and r["objective"]:
                return float(r["objective"])
            return float(r["pos_err_km"]) + 0.01 * float(r["vel_err_m_s"])
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


def read_leg_rows(path: Path) -> list[dict[str, str]]:
    rows = read_csv_rows(path)
    rows.sort(key=lambda r: int(r["leg"]))
    return rows


def get_float(row: dict[str, str], key: str) -> float:
    return float(row[key])


def vec(row: dict[str, str], keys: list[str]) -> np.ndarray:
    return np.array([float(row[k]) for k in keys], dtype=float)


def find_flyby_row(path: Path, flyby_index: int) -> dict[str, str]:
    for r in read_csv_rows(path):
        if int(r["flyby_index"]) == flyby_index:
            return r
    raise KeyError(f"flyby_index {flyby_index} not found in {path}")


@dataclass
class BridgeSetup:
    flyby_index: int
    body: str

    leg_in: int
    leg_out: int

    t0_s: float
    t_event_s: float
    t1_s: float

    r0_m: np.ndarray
    v0_m_s: np.ndarray

    target_r_m: np.ndarray
    target_v_m_s: np.ndarray

    radius_km: float
    atmosphere_km: float
    min_altitude_km: float

    vinf_mismatch_km_s: float
    turn_margin_deg: float
    alt_required_km: float


def build_setup(args: argparse.Namespace) -> BridgeSetup:
    legs = read_leg_rows(args.leg_optimizations)
    flyby = find_flyby_row(args.flyby_audit, args.flyby_index)

    leg_in_n = int(flyby["leg_in"])
    leg_out_n = int(flyby["leg_out"])
    body = norm_name(flyby["body"])

    leg_in = legs[leg_in_n - 1]
    leg_out = legs[leg_out_n - 1]

    if norm_name(leg_in["arr_body"]) != body:
        raise ValueError(f"leg_in does not arrive at {body}")
    if norm_name(leg_out["dep_body"]) != body:
        raise ValueError(f"leg_out does not depart from {body}")

    t0 = get_float(leg_in, "t_end_s")
    t1 = get_float(leg_out, "t_start_s")

    # For continuous patched-conic semantics, these should be the same event epoch.
    t_event_in = get_float(leg_in, "t_arr_s")
    t_event_out = get_float(leg_out, "t_dep_s")
    t_event = 0.5 * (t_event_in + t_event_out)

    r0 = vec(leg_in, ["final_x_raw_m", "final_y_raw_m", "final_z_raw_m"])
    v0 = vec(leg_in, ["final_vx_raw_m_s", "final_vy_raw_m_s", "final_vz_raw_m_s"])

    target_r = vec(leg_out, ["start_x_raw_m", "start_y_raw_m", "start_z_raw_m"])
    target_v = vec(leg_out, [
        "optimized_vx_raw_m_s",
        "optimized_vy_raw_m_s",
        "optimized_vz_raw_m_s",
    ])

    catalog = load_json(args.body_catalog)
    body_rec = get_body_record(catalog, body)
    radius = body_radius_km(body_rec)
    atm = body_atmosphere_km(body_rec)

    min_alt = max(args.min_altitude_km, atm + args.atmosphere_margin_km)

    return BridgeSetup(
        flyby_index=args.flyby_index,
        body=body,
        leg_in=leg_in_n,
        leg_out=leg_out_n,
        t0_s=t0,
        t_event_s=t_event,
        t1_s=t1,
        r0_m=r0,
        v0_m_s=v0,
        target_r_m=target_r,
        target_v_m_s=target_v,
        radius_km=radius,
        atmosphere_km=atm,
        min_altitude_km=min_alt,
        vinf_mismatch_km_s=float(flyby.get("vinf_mismatch_km_s", "nan")),
        turn_margin_deg=float(flyby.get("turn_margin_deg", "nan")),
        alt_required_km=float(flyby.get("alt_required_km", "nan")),
    )


def evaluate(
    *,
    setup: BridgeSetup,
    server: ExtendedImpulseServer,
    x: np.ndarray,
    req_id: str,
) -> tuple[BridgeResponse, np.ndarray, np.ndarray, float]:
    dv0 = np.asarray(x[:3], dtype=float)
    burn_dv = np.asarray(x[3:6], dtype=float)
    burn_dt = float(x[6])
    burn_t = setup.t_event_s + burn_dt

    resp = server.propagate(
        req_id=req_id,
        t0_s=setup.t0_s,
        burn_t_s=burn_t,
        t1_s=setup.t1_s,
        r0_m=setup.r0_m,
        v0_m_s=setup.v0_m_s + dv0,
        burn_dv_m_s=burn_dv,
    )

    return resp, dv0, burn_dv, burn_t


def final_body_metrics(args: argparse.Namespace, setup: BridgeSetup, resp: BridgeResponse, burn_t: float) -> dict[str, Any]:
    body_r, body_v = sample_raw_body_state(
        sampler=args.sampler,
        plugin_b64=args.plugin_b64,
        target_body=setup.body,
        sampler_central_body=args.raw_origin_body,
        et_s=burn_t,
        plugin_base_et_s=args.plugin_base_et_s,
        work_dir=args.raw_cache_dir,
    )

    rel_r = resp.burn_r_m - body_r
    rel_v_before = resp.burn_v_before_m_s - body_v
    rel_v_after = resp.burn_v_after_m_s - body_v

    alt_km = norm(rel_r) / 1000.0 - setup.radius_km
    radial_v_before_km_s = float(np.dot(rel_r, rel_v_before) / max(norm(rel_r), 1.0)) / 1000.0
    radial_v_after_km_s = float(np.dot(rel_r, rel_v_after) / max(norm(rel_r), 1.0)) / 1000.0

    return {
        "burn_altitude_km": alt_km,
        "burn_radius_km": norm(rel_r) / 1000.0,
        "burn_radial_v_before_km_s": radial_v_before_km_s,
        "burn_radial_v_after_km_s": radial_v_after_km_s,
        "burn_vinf_before_km_s": norm(rel_v_before) / 1000.0,
        "burn_vinf_after_km_s": norm(rel_v_after) / 1000.0,
    }


def optimize_bridge(args: argparse.Namespace, setup: BridgeSetup) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    pos_scale_m = args.pos_scale_km * 1000.0
    vel_scale_m_s = args.vel_scale_m_s

    if args.target_burn_altitude_km is None:
        target_burn_altitude_km = setup.alt_required_km
    else:
        target_burn_altitude_km = args.target_burn_altitude_km

    # Time bounds must stay inside the propagation arc.
    lower_dt = (setup.t0_s + args.time_margin_s) - setup.t_event_s
    upper_dt = (setup.t1_s - args.time_margin_s) - setup.t_event_s

    if lower_dt >= upper_dt:
        raise ValueError("time_margin_s leaves no valid burn window")

    x0 = np.zeros(7, dtype=float)

    if args.initial_burn_dt_s is not None:
        x0[6] = float(args.initial_burn_dt_s)
    else:
        x0[6] = 0.0

    x0[6] = min(max(x0[6], lower_dt), upper_dt)

    if args.initial_burn_guess_m_s != 0.0:
        # A weak guess along the required velocity difference at the bridge boundaries.
        dv_guess_dir = setup.target_v_m_s - setup.v0_m_s
        n = norm(dv_guess_dir)
        if n > 0.0:
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

    history: list[dict[str, Any]] = []
    counter = {"n": 0}
    best_seen = {
        "objective": float("inf"),
        "x": None,
    }

    # Avoid disk I/O in every optimizer evaluation. The flyby window is short,
    # so a linear ephemeris around the event is sufficient for local residuals.
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

    with ExtendedImpulseServer(args.server, args.plugin_b64) as server:
        def residual(x: np.ndarray) -> np.ndarray:
            counter["n"] += 1

            resp, dv0, burn_dv, burn_t = evaluate(
                setup=setup,
                server=server,
                x=x,
                req_id=f"bridge_eval_{counter['n']}",
            )

            burn_altitude_km = math.inf
            burn_radial_v_km_s = math.inf

            if resp.status != "ok":
                pos_err_km = math.inf
                vel_err_m_s = math.inf
                base = np.full(9, 1e9, dtype=float)
            else:
                pos_err = resp.final_r_m - setup.target_r_m
                vel_err = resp.final_v_m_s - setup.target_v_m_s

                pos_err_km = norm(pos_err) / 1000.0
                vel_err_m_s = norm(vel_err)

                body_r, body_v = body_cache.state(burn_t)

                rel_r = resp.burn_r_m - body_r
                rel_v = resp.burn_v_before_m_s - body_v

                burn_altitude_km = norm(rel_r) / 1000.0 - setup.radius_km
                burn_radial_v_km_s = float(np.dot(rel_r, rel_v) / max(norm(rel_r), 1.0)) / 1000.0

                alt_res = (burn_altitude_km - target_burn_altitude_km) / args.periapsis_scale_km
                radial_res = burn_radial_v_km_s / args.radial_scale_km_s

                # Hard-ish safety residual: never let the optimizer prefer a
                # sub-surface or too-low periapsis just to improve endpoint closure.
                low_altitude_res = max(0.0, setup.min_altitude_km - burn_altitude_km) / args.periapsis_scale_km

                base = np.concatenate([
                    pos_err / pos_scale_m,
                    vel_err / vel_scale_m_s,
                    np.array([alt_res, radial_res, low_altitude_res], dtype=float),
                ])

            # Soft regularization: prefer small pre-trim and small flyby burn,
            # but do not let it dominate final-state closure.
            reg = np.concatenate([
                dv0 / args.dv0_regularization_m_s,
                burn_dv / args.burn_regularization_m_s,
                np.array([x[6] / args.time_regularization_s]),
            ])

            r = np.concatenate([
                base,
                args.regularization_weight * reg,
            ])

            objective = float(np.linalg.norm(r))
            if np.isfinite(objective) and objective < best_seen["objective"]:
                best_seen["objective"] = objective
                best_seen["x"] = np.array(x, dtype=float)

            row = {
                "eval": counter["n"],
                "status": resp.status,
                "message": resp.message,
                "dv0_x_m_s": dv0[0],
                "dv0_y_m_s": dv0[1],
                "dv0_z_m_s": dv0[2],
                "dv0_norm_m_s": norm(dv0),
                "burn_dv_x_m_s": burn_dv[0],
                "burn_dv_y_m_s": burn_dv[1],
                "burn_dv_z_m_s": burn_dv[2],
                "burn_dv_norm_m_s": norm(burn_dv),
                "burn_t_s": burn_t,
                "burn_dt_s": x[6],
                "pos_err_km": pos_err_km,
                "vel_err_m_s": vel_err_m_s,
                "burn_altitude_km": burn_altitude_km,
                "burn_radial_v_km_s": burn_radial_v_km_s,
                "target_burn_altitude_km": target_burn_altitude_km,
                "low_altitude_penalty_km": max(0.0, setup.min_altitude_km - burn_altitude_km),
                "objective": objective,
            }
            history.append(row)

            print(
                f"[eval {counter['n']:03d}] "
                f"pos={pos_err_km:12.3f} km "
                f"vel={vel_err_m_s:10.3f} m/s "
                f"dv0={norm(dv0):8.3f} m/s "
                f"burn={norm(burn_dv):8.3f} m/s "
                f"dt={x[6]:9.1f} s "
                f"alt={burn_altitude_km:9.1f} km "
                f"vr={burn_radial_v_km_s:8.3f} km/s "
                f"{resp.status}"
            )

            return r

        def jac_abs(x: np.ndarray) -> np.ndarray:
            # Absolute finite-difference steps. Relative FD steps are too small
            # for a C++ N-body propagator near close-approach events.
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

        final_x = best_seen["x"] if best_seen["x"] is not None else sol.x

        final_resp, final_dv0, final_burn, final_burn_t = evaluate(
            setup=setup,
            server=server,
            x=final_x,
            req_id="bridge_final",
        )

    if final_resp.status == "ok":
        final_pos_err_km = norm(final_resp.final_r_m - setup.target_r_m) / 1000.0
        final_vel_err_m_s = norm(final_resp.final_v_m_s - setup.target_v_m_s)
    else:
        final_pos_err_km = math.inf
        final_vel_err_m_s = math.inf

    body_metrics: dict[str, Any] = {}
    if final_resp.status == "ok":
        body_metrics = final_body_metrics(args, setup, final_resp, final_burn_t)

    success = (
        bool(sol.success)
        and final_resp.status == "ok"
        and final_pos_err_km <= args.accept_pos_km
        and final_vel_err_m_s <= args.accept_vel_m_s
    )

    result = {
        "success": success,
        "raw_solver_success": bool(sol.success),
        "solver_message": str(sol.message),
        "nfev": int(sol.nfev),
        "best_objective": float(best_seen["objective"]),

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

        "total_bridge_dv_m_s": norm(final_dv0) + norm(final_burn),

        "final_status": final_resp.status,
        "final_message": final_resp.message,
        "final_pos_err_km": final_pos_err_km,
        "final_vel_err_m_s": final_vel_err_m_s,

        "audit_vinf_mismatch_km_s": setup.vinf_mismatch_km_s,
        "audit_turn_margin_deg": setup.turn_margin_deg,
        "audit_alt_required_km": setup.alt_required_km,

        "radius_km": setup.radius_km,
        "atmosphere_km": setup.atmosphere_km,
        "min_altitude_km": setup.min_altitude_km,
    }

    result.update(body_metrics)

    if "burn_altitude_km" in result:
        if result["burn_altitude_km"] < setup.min_altitude_km:
            result["geometry_status"] = "LOW_ALTITUDE"
        else:
            result["geometry_status"] = "OK"
    else:
        result["geometry_status"] = "UNKNOWN"

    return result, history


def write_history(path: Path, history: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    fields: list[str] = []
    for row in history:
        for k in row:
            if k not in fields:
                fields.append(k)

    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(history)


def main_cli() -> int:
    p = argparse.ArgumentParser(description="Native Principia powered flyby bridge.")
    p.add_argument("--candidate-seed", type=Path, default=None)  # kept for provenance/CLI symmetry
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

    p.add_argument("--max-dv0-m-s", type=float, default=500.0)
    p.add_argument("--max-burn-m-s", type=float, default=2000.0)
    p.add_argument("--time-margin-s", type=float, default=60.0)

    p.add_argument("--initial-burn-dt-s", type=float, default=None)
    p.add_argument("--initial-burn-guess-m-s", type=float, default=0.0)
    p.add_argument("--initial-history-csv", type=Path, default=None)
    p.add_argument("--initial-history-eval", type=int, default=None)

    p.add_argument("--pos-scale-km", type=float, default=100.0)
    p.add_argument("--vel-scale-m-s", type=float, default=10.0)

    # Optional local flyby constraints. These turn the bridge into a real
    # periapsis/aimpoint targeter instead of a pure endpoint matcher.
    p.add_argument("--target-burn-altitude-km", type=float, default=None)
    p.add_argument("--periapsis-scale-km", type=float, default=100.0)
    p.add_argument("--radial-scale-km-s", type=float, default=0.1)

    p.add_argument("--dv0-x-scale-m-s", type=float, default=50.0)
    p.add_argument("--burn-x-scale-m-s", type=float, default=100.0)
    p.add_argument("--time-x-scale-s", type=float, default=3600.0)

    p.add_argument("--dv0-regularization-m-s", type=float, default=100.0)
    p.add_argument("--burn-regularization-m-s", type=float, default=200.0)
    p.add_argument("--time-regularization-s", type=float, default=21600.0)
    p.add_argument("--regularization-weight", type=float, default=1e-4)

    p.add_argument("--fd-dv0-step-m-s", type=float, default=10.0)
    p.add_argument("--fd-burn-step-m-s", type=float, default=10.0)
    p.add_argument("--fd-time-step-s", type=float, default=30.0)

    p.add_argument("--max-nfev", type=int, default=120)
    p.add_argument("--ftol", type=float, default=1e-10)
    p.add_argument("--xtol", type=float, default=1e-10)
    p.add_argument("--gtol", type=float, default=1e-10)

    p.add_argument("--accept-pos-km", type=float, default=1.0)
    p.add_argument("--accept-vel-m-s", type=float, default=1.0)

    p.add_argument("--work-dir", type=Path, required=True)
    p.add_argument("--output-json", type=Path, required=True)
    p.add_argument("--output-history-csv", type=Path, required=True)

    args = p.parse_args()

    args.work_dir.mkdir(parents=True, exist_ok=True)

    setup = build_setup(args)

    print("=== NATIVE POWERED FLYBY BRIDGE ===")
    if args.rank is not None:
        print(f"rank        : {args.rank}")
    print(f"flyby      : {setup.body} index={setup.flyby_index}")
    print(f"legs       : {setup.leg_in} -> {setup.leg_out}")
    print(f"arc        : t0={setup.t0_s:.6f} event={setup.t_event_s:.6f} t1={setup.t1_s:.6f}")
    print(f"audit      : mismatch={setup.vinf_mismatch_km_s:.6f} km/s margin={setup.turn_margin_deg:.3f} deg alt_req={setup.alt_required_km:.3f} km")
    print(f"body       : radius={setup.radius_km:.3f} km min_alt={setup.min_altitude_km:.3f} km")
    print("")

    result, history = optimize_bridge(args, setup)

    write_history(args.output_history_csv, history)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2))

    print("")
    print("=== POWERED FLYBY BRIDGE RESULT ===")
    print(f"success          : {result['success']} {result['solver_message']}")
    print(f"nfev             : {result['nfev']}")
    print(f"final pos err km : {result['final_pos_err_km']}")
    print(f"final vel err m/s: {result['final_vel_err_m_s']}")
    print(f"dv0 m/s          : {result['dv0_norm_m_s']}")
    print(f"burn dv m/s      : {result['burn_dv_norm_m_s']}")
    print(f"total dv m/s     : {result['total_bridge_dv_m_s']}")
    print(f"burn_t_s         : {result['burn_t_s']}")
    print(f"burn_dt_s        : {result['burn_dt_s']}")
    if "burn_altitude_km" in result:
        print(f"burn alt km      : {result['burn_altitude_km']}")
        print(f"burn radial km/s : {result['burn_radial_v_before_km_s']}")
        print(f"geometry status  : {result['geometry_status']}")

    print(f"[OK] result : {args.output_json}")
    print(f"[OK] history: {args.output_history_csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main_cli())
