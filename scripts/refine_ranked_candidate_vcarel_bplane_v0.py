#!/usr/bin/env python3
"""
refine_ranked_candidate_vcarel_bplane_v0.py

B-plane / flyby-target refiner for one ranked departure candidate.

Why this exists:
  The old DSM refiner minimized distance to the flyby body's centre. That finds
  impacts. This targeter instead aims at the patched-conics/Lambert flyby
  geometry: the side of the planet, periapsis radius, and outgoing v∞ direction
  implied by the route.

Inputs:
  - candidate_departure_executability_rank.json from
    rank_pykep_candidates_by_departure_executability_v0_1.py
  - anchor_packet.json for the selected route
  - body_catalog.json

VCAREL convention used here:
  state_dt_s is absolute game time, while scan_start_dt_s/scan_end_dt_s are
  relative to the synthetic state epoch.

Method:
  Keep burn0 fixed.
  For each DSM epoch fraction, use finite-difference sensitivity of a residual:
    [ (r_ca - r_target) / pos_scale_m,
      out_dir_weight * (unit(vinf_out_natural) - unit(vinf_out_required)),
      mag_weight * (|vinf_out_natural|-|vinf_out_required|)/vel_scale_m_s ]
  Solve damped least squares for a DSM update, clip to --dsm-max-m-s, and accept
  only improvements.

This is a deterministic local targeter, not a global optimizer.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Sequence

import concurrent.futures
import functools

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from principia_targeter_client import PrincipiaTargeterClient

DAY_S = 86400.0


def worker_optimize_fraction(
    frac: float,
    args: argparse.Namespace,
    tof_rel_s: float,
    dep_body: str,
    arr_body: str,
    state_abs_s: float,
    scan_start_rel_s: float,
    scan_end_rel_s: float,
    rel_r: list[float],
    rel_v: list[float],
    burn0_raw: list[float],
    mu: float,
    target: dict[str, Any],
    safe_radius_m: float
) -> list[dict[str, Any]]:
    rows = []
    eval_counter = 0
    dsm_dt_s = max(1.0, min(float(frac) * tof_rel_s, tof_rel_s - DAY_S))
    dsm = np.zeros(3, dtype=float)
    
    print(f"\n[Worker PID {os.getpid()}] [fraction {frac:.4f}] dsm_dt={dsm_dt_s:.3f}s ({dsm_dt_s/DAY_S:.3f} d)")

    # Cada worker abre seu próprio client
    with PrincipiaTargeterClient(
        args.server,
        args.plugin_b64,
        plugin_arg_mode=args.plugin_arg_mode,
        response_timeout_s=args.server_timeout_s,
        quiet_stderr=args.quiet_stderr,
    ) as client:
        for it in range(args.iterations):
            try:
                cur = eval_candidate(
                    client, f"bplane_{os.getpid()}_{eval_counter}",
                    dep_body, arr_body, state_abs_s, scan_start_rel_s, scan_end_rel_s,
                    args.vca_samples, rel_r, rel_v, burn0_raw, dsm_dt_s, dsm.tolist(),
                    args.server_timeout_s, mu, target,
                    args.pos_scale_km * 1000.0, args.vel_scale_m_s,
                    args.out_dir_weight, args.mag_weight, args.dsm_weight,
                    args.unsafe_penalty, safe_radius_m,
                )
                eval_counter += 1
                cur.update({"kind": "iterate_base", "fraction": frac, "iteration": it})
                rows.append(cur)
            except Exception as exc:
                rows.append({"ok": False, "error": str(exc), "kind": "iterate_base", "fraction": frac, "iteration": it,
                             "dsm_dt_s": dsm_dt_s, "dsm_raw_m_s": dsm.tolist(), "dsm_norm_m_s": norm(dsm)})
                print(f"  [frac {frac:.4f}] iter {it}: base failed: {exc}")
                break

            r0 = np.asarray(cur["residual"], dtype=float)
            J = np.zeros((len(r0), 3), dtype=float)

            for j in range(3):
                dsm_p = dsm.copy()
                dsm_p[j] += args.fd_step_m_s
                try:
                    rp = eval_candidate(
                        client, f"bplane_{os.getpid()}_{eval_counter}",
                        dep_body, arr_body, state_abs_s, scan_start_rel_s, scan_end_rel_s,
                        args.vca_samples, rel_r, rel_v, burn0_raw, dsm_dt_s, dsm_p.tolist(),
                        args.server_timeout_s, mu, target,
                        args.pos_scale_km * 1000.0, args.vel_scale_m_s,
                        args.out_dir_weight, args.mag_weight, args.dsm_weight,
                        args.unsafe_penalty, safe_radius_m,
                    )
                    eval_counter += 1
                    rp.update({"kind": f"fd_axis_{j}", "fraction": frac, "iteration": it})
                    rows.append(rp)
                    J[:, j] = (np.asarray(rp["residual"], dtype=float) - r0) / args.fd_step_m_s
                except Exception as exc:
                    rows.append({"ok": False, "error": str(exc), "kind": f"fd_axis_{j}", "fraction": frac, "iteration": it,
                                 "dsm_dt_s": dsm_dt_s, "dsm_raw_m_s": dsm_p.tolist(), "dsm_norm_m_s": norm(dsm_p)})

            JTJ = J.T @ J
            rhs = -J.T @ r0
            lam = args.regularization * max(1.0, float(np.trace(JTJ)) / 3.0)
            try:
                delta = np.linalg.solve(JTJ + lam * np.eye(3), rhs)
            except np.linalg.LinAlgError:
                delta = np.linalg.lstsq(JTJ + lam * np.eye(3), rhs, rcond=None)[0]

            best_local = cur
            best_local_dsm = dsm.copy()
            for alpha in (1.0, 0.5, 0.25, 0.1):
                dsm_trial = clip_norm(dsm + alpha * delta, args.dsm_max_m_s)
                try:
                    trial = eval_candidate(
                        client, f"bplane_{os.getpid()}_{eval_counter}",
                        dep_body, arr_body, state_abs_s, scan_start_rel_s, scan_end_rel_s,
                        args.vca_samples, rel_r, rel_v, burn0_raw, dsm_dt_s, dsm_trial.tolist(),
                        args.server_timeout_s, mu, target,
                        args.pos_scale_km * 1000.0, args.vel_scale_m_s,
                        args.out_dir_weight, args.mag_weight, args.dsm_weight,
                        args.unsafe_penalty, safe_radius_m,
                    )
                    eval_counter += 1
                    trial.update({"kind": f"trial_alpha_{alpha}", "fraction": frac, "iteration": it})
                    rows.append(trial)
                    if trial["score"] < best_local["score"]:
                        best_local = trial
                        best_local_dsm = dsm_trial
                except Exception as exc:
                    rows.append({"ok": False, "error": str(exc), "kind": f"trial_alpha_{alpha}", "fraction": frac, "iteration": it,
                                 "dsm_dt_s": dsm_dt_s, "dsm_raw_m_s": dsm_trial.tolist(), "dsm_norm_m_s": norm(dsm_trial)})

            print(
                f"  [frac {frac:.4f}] iter {it}: score={cur['score']:10.3f} -> {best_local['score']:10.3f} "
                f"pos={best_local['target_pos_err_km']:10.3f}km "
                f"rp={best_local['periapsis_radius_km']:9.3f}km "
                f"safe_margin={best_local['periapsis_altitude_over_safe_km']:9.3f}km "
                f"out_ang={best_local['natural_out_angle_deg']:8.3f} "
                f"dsm={best_local['dsm_norm_m_s']:8.3f}"
            )

            if best_local["score"] < cur["score"]:
                dsm = best_local_dsm
            else:
                break

    return rows

def norm(v: Sequence[float]) -> float:
    return float(np.linalg.norm(np.asarray(v, dtype=float)))


def unit(v: Sequence[float]) -> np.ndarray:
    a = np.asarray(v, dtype=float)
    n = norm(a)
    if n <= 0 or not math.isfinite(n):
        raise ValueError(f"cannot normalize vector: {v}")
    return a / n


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def angle_deg(a: Sequence[float], b: Sequence[float]) -> float:
    na = norm(a)
    nb = norm(b)
    if na <= 0 or nb <= 0:
        return math.nan
    return math.degrees(math.acos(clamp(float(np.dot(a, b)) / (na * nb), -1.0, 1.0)))


def raw_to_levela(v: Sequence[float]) -> list[float]:
    x, y, z = map(float, v)
    return [-y, z, x]


def levela_to_raw(v: Sequence[float]) -> list[float]:
    x, y, z = map(float, v)
    return [z, -x, y]


def parse_float_list(s: str) -> list[float]:
    return [float(x.strip()) for x in str(s).replace(";", ",").split(",") if x.strip()]


def read_live_t(path: Path) -> float:
    r = json.loads(path.read_text())
    for k in ("ut_s", "t_s", "spice_t_s", "et_s"):
        if k in r:
            return float(r[k])
    raise KeyError(f"could not find time in {path}")


def find_body_record(obj: Any, body: str):
    body_l = body.lower()
    if isinstance(obj, dict):
        name = str(obj.get("name", obj.get("body", obj.get("id", "")))).lower()
        if name == body_l:
            return obj
        for k, v in obj.items():
            if str(k).lower() == body_l and isinstance(v, dict):
                return v
        for v in obj.values():
            found = find_body_record(v, body)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for x in obj:
            found = find_body_record(x, body)
            if found is not None:
                return found
    return None


def body_radius_mu(path: Path, body: str) -> tuple[float, float]:
    data = json.loads(path.read_text())
    rec = find_body_record(data, body)
    if rec is None:
        raise RuntimeError(f"body {body} not found in {path}")

    radius = None
    for k in ("radius_km", "mean_radius_km", "equatorial_radius_km"):
        if k in rec:
            radius = float(rec[k])
            break
    if radius is None:
        # Added "equatorial_radius" to the list of keys to check
        for k in ("radius_m", "mean_radius_m", "equatorial_radius_m", "radius", "equatorial_radius"):
            if k in rec:
                val = float(rec[k])
                # This logic cleanly handles both meters and km depending on the magnitude
                radius = val / 1000.0 if val > 1e5 else val 
                break
    if radius is None:
        raise RuntimeError(f"radius not found for {body}")

    mu = None
    for k in ("mu_m3_s2", "gm_m3_s2", "gravitational_parameter_m3_s2"):
        if k in rec:
            mu = float(rec[k])
            break
    if mu is None:
        for k in ("mu", "gm", "gravitational_parameter"):
            if k in rec:
                val = float(rec[k])
                mu = val * 1e9 if val < 1e12 else val
                break
    if mu is None:
        raise RuntimeError(f"mu/gm not found for {body}")

    return radius, mu


def select_candidate(rank_json: Path, top_index: int) -> dict[str, Any]:
    data = json.loads(rank_json.read_text())
    top = data.get("top", [])
    if top_index < 0 or top_index >= len(top):
        raise SystemExit(f"--top-index out of range: {top_index}; top has {len(top)} rows")
    c = dict(top[top_index])
    required = ["burn_abs_s", "burn_dt_s", "t_arr_s", "dv_raw_m_s", "burn_rel_r_raw_m", "burn_rel_v_raw_m_s"]
    missing = [k for k in required if k not in c or c[k] is None]
    if missing:
        raise SystemExit("selected candidate lacks: " + ", ".join(missing))
    if "dv_levela_m_s" not in c or c["dv_levela_m_s"] is None:
        c["dv_levela_m_s"] = raw_to_levela(c["dv_raw_m_s"])
    return c


def anchor_legs(anchor_json: Path) -> list[dict[str, Any]]:
    data = json.loads(anchor_json.read_text())
    if "legs" in data and data["legs"]:
        return data["legs"]
    legs = []
    for i in range(1, 20):
        key = f"leg{i}"
        if key in data:
            legs.append(data[key])
    if not legs:
        raise RuntimeError(f"no legs found in {anchor_json}")
    return legs


def get_vinf_raw_m_s(leg: dict[str, Any], kind: str) -> np.ndarray:
    # kind: "arr" or "dep"
    candidates = [
        f"vinf_{kind}_raw_m_s",
        f"vinf_{kind}_m_s_raw",
        f"vinf_{kind}_raw",
    ]
    for k in candidates:
        if k in leg:
            return np.asarray(leg[k], dtype=float)

    levela_km = f"vinf_{kind}_levela_km_s"
    if levela_km in leg:
        return np.asarray(levela_to_raw([1000.0 * float(x) for x in leg[levela_km]]), dtype=float)

    levela_m = f"vinf_{kind}_levela_m_s"
    if levela_m in leg:
        return np.asarray(levela_to_raw(leg[levela_m]), dtype=float)

    raise RuntimeError(f"cannot find vinf_{kind} vector in leg keys={sorted(leg.keys())}")


def compute_route_bplane_target(
    vinf_in_raw: Sequence[float],
    vinf_out_req_raw: Sequence[float],
    mu: float,
    safe_radius_m: float,
    target_altitude_km: float | None,
) -> dict[str, Any]:
    vin = np.asarray(vinf_in_raw, dtype=float)
    vout = np.asarray(vinf_out_req_raw, dtype=float)
    vin_hat = unit(vin)
    vout_hat = unit(vout)

    turn_req_rad = math.acos(clamp(float(np.dot(vin_hat, vout_hat)), -1.0, 1.0))
    vin_mag = norm(vin)

    # For an unpowered flyby with fixed incoming v∞:
    # delta = 2 asin(1/e), e = 1 + rp*v_inf^2/mu
    if turn_req_rad <= 1e-12:
        rp_req_m = float("inf")
    else:
        rp_req_m = mu / (vin_mag * vin_mag) * (1.0 / math.sin(turn_req_rad / 2.0) - 1.0)

    if target_altitude_km is not None:
        rp_target_m = target_altitude_km * 1000.0
    else:
        rp_target_m = max(safe_radius_m, rp_req_m)

    # If route requires more turning than safe radius can deliver, target safe
    # altitude on the correct side. Powered flyby/cleanup may be needed.
    e_safe = 1.0 + safe_radius_m * vin_mag * vin_mag / mu
    max_turn_safe_rad = 2.0 * math.asin(clamp(1.0 / e_safe, -1.0, 1.0))
    ballistic_possible_at_safe = turn_req_rad <= max_turn_safe_rad + 1e-12

    # Periapsis side implied by route. With our hyperbola convention:
    # v_inf_in_hat - v_inf_out_hat = 2/e * rhat_p.
    side_vec = vin_hat - vout_hat
    if norm(side_vec) < 1e-9:
        # Degenerate no-turn case: choose a deterministic perpendicular-ish side.
        side_vec = np.cross(vin_hat, np.array([0.0, 0.0, 1.0]))
        if norm(side_vec) < 1e-9:
            side_vec = np.cross(vin_hat, np.array([0.0, 1.0, 0.0]))
    rhat_target = unit(side_vec)
    target_r = rp_target_m * rhat_target

    return {
        "route_vinf_in_raw_m_s": vin.tolist(),
        "route_vinf_out_req_raw_m_s": vout.tolist(),
        "route_vinf_in_mag_m_s": vin_mag,
        "route_vinf_out_req_mag_m_s": norm(vout),
        "route_turn_required_deg": math.degrees(turn_req_rad),
        "safe_max_turn_deg": math.degrees(max_turn_safe_rad),
        "rp_required_unpowered_km": rp_req_m / 1000.0 if math.isfinite(rp_req_m) else math.inf,
        "rp_target_km": rp_target_m / 1000.0,
        "ballistic_possible_at_safe": ballistic_possible_at_safe,
        "target_rhat_raw": rhat_target.tolist(),
        "target_r_raw_m": target_r.tolist(),
    }


def parse_okcarel(line: str) -> dict[str, Any]:
    fields = line.rstrip("\n").split("\t")
    if not fields or fields[0] != "OKCAREL":
        raise RuntimeError(f"expected OKCAREL response, got: {line[:500]}")
    if len(fields) < 32:
        raise RuntimeError(f"OKCAREL response too short: {len(fields)} fields")

    out = {
        "id": fields[1],
        "dep_body": fields[2],
        "arr_body": fields[3],
        "state_dt_s": float(fields[4]),
        "state_t_game_s": float(fields[5]),
        "ca_dt_s": float(fields[6]),
        "ca_t_game_s": float(fields[7]),
        "ca_rel_r_raw_m": list(map(float, fields[8:11])),
        "ca_rel_v_raw_m_s": list(map(float, fields[11:14])),
        "ca_distance_m": float(fields[14]),
        "ca_speed_m_s": float(fields[15]),
        "ca_radial_v_m_s": float(fields[16]),
        "samples": int(float(fields[17])),
        "status": fields[18],
        "ca_abs_debug_r_raw_m": list(map(float, fields[19:22])),
        "ca_abs_debug_v_raw_m_s": list(map(float, fields[22:25])),
        "arr_abs_debug_r_raw_m": list(map(float, fields[25:28])),
        "arr_abs_debug_v_raw_m_s": list(map(float, fields[28:31])),
        "n_burns": int(float(fields[31])),
        "burns": [],
    }
    idx = 32
    for _ in range(out["n_burns"]):
        if idx + 10 > len(fields):
            raise RuntimeError(f"OKCAREL burn diagnostics truncated at field {idx}")
        out["burns"].append({
            "burn_dt_s": float(fields[idx + 0]),
            "burn_r_raw_m": list(map(float, fields[idx + 1:idx + 4])),
            "burn_v_before_raw_m_s": list(map(float, fields[idx + 4:idx + 7])),
            "burn_v_after_raw_m_s": list(map(float, fields[idx + 7:idx + 10])),
        })
        idx += 10
    return out


def vcarel(
    client: PrincipiaTargeterClient,
    rid: str,
    dep_body: str,
    arr_body: str,
    state_abs_s: float,
    scan_start_rel_s: float,
    scan_end_rel_s: float,
    samples: int,
    rel_r: Sequence[float],
    rel_v: Sequence[float],
    impulses: Sequence[tuple[float, float, float, float]],
    timeout_s: float,
) -> dict[str, Any]:
    fields: list[Any] = [
        "VCAREL", rid, dep_body, arr_body,
        float(state_abs_s), float(scan_start_rel_s), float(scan_end_rel_s),
        int(samples),
        float(rel_r[0]), float(rel_r[1]), float(rel_r[2]),
        float(rel_v[0]), float(rel_v[1]), float(rel_v[2]),
        int(len(impulses)),
    ]
    for dt, dvx, dvy, dvz in impulses:
        fields += [float(dt), float(dvx), float(dvy), float(dvz)]
    line = client.command_fields(fields, timeout_s=timeout_s)
    return parse_okcarel(line)


def asymptotes_from_ca(r_m: Sequence[float], v_m_s: Sequence[float], mu: float) -> dict[str, Any]:
    r = np.asarray(r_m, dtype=float)
    v = np.asarray(v_m_s, dtype=float)
    rp = norm(r)
    vp = norm(v)
    eps = 0.5 * vp * vp - mu / rp
    if eps <= 0:
        return {"hyperbolic": False, "specific_energy_m2_s2": eps, "vinf_mag_m_s": math.nan}

    vinf = math.sqrt(2.0 * eps)
    e = 1.0 + rp * vinf * vinf / mu
    if e <= 1.0:
        return {"hyperbolic": False, "specific_energy_m2_s2": eps, "vinf_mag_m_s": vinf, "eccentricity": e}

    rhat = unit(r)
    vt = v - float(np.dot(v, rhat)) * rhat
    that = unit(vt if norm(vt) > 1e-9 else v)

    c = 1.0 / e
    s = math.sqrt(max(0.0, 1.0 - c * c))
    vinf_in_hat = c * rhat + s * that
    vinf_out_hat = -c * rhat + s * that

    return {
        "hyperbolic": True,
        "specific_energy_m2_s2": eps,
        "vinf_mag_m_s": vinf,
        "eccentricity": e,
        "turn_angle_deg": math.degrees(2.0 * math.asin(clamp(1.0 / e, -1.0, 1.0))),
        "natural_vinf_in_raw_m_s": (vinf * vinf_in_hat).tolist(),
        "natural_vinf_out_raw_m_s": (vinf * vinf_out_hat).tolist(),
    }


def eval_candidate(
    client: PrincipiaTargeterClient,
    rid: str,
    dep_body: str,
    arr_body: str,
    state_abs_s: float,
    scan_start_rel_s: float,
    scan_end_rel_s: float,
    samples: int,
    rel_r: Sequence[float],
    rel_v: Sequence[float],
    burn0_raw: Sequence[float],
    dsm_dt_s: float | None,
    dsm_raw: Sequence[float] | None,
    timeout_s: float,
    mu: float,
    target: dict[str, Any],
    pos_scale_m: float,
    vel_scale_m_s: float,
    out_dir_weight: float,
    mag_weight: float,
    dsm_weight: float,
    unsafe_penalty: float,
    safe_radius_m: float,
) -> dict[str, Any]:
    impulses: list[tuple[float, float, float, float]] = [
        (0.0, float(burn0_raw[0]), float(burn0_raw[1]), float(burn0_raw[2])),
    ]
    if dsm_dt_s is not None and dsm_raw is not None and norm(dsm_raw) > 0:
        impulses.append((float(dsm_dt_s), float(dsm_raw[0]), float(dsm_raw[1]), float(dsm_raw[2])))

    res = vcarel(client, rid, dep_body, arr_body, state_abs_s, scan_start_rel_s, scan_end_rel_s,
                 samples, rel_r, rel_v, impulses, timeout_s)

    r = np.asarray(res["ca_rel_r_raw_m"], dtype=float)
    v = np.asarray(res["ca_rel_v_raw_m_s"], dtype=float)
    hyp = asymptotes_from_ca(r, v, mu)

    target_r = np.asarray(target["target_r_raw_m"], dtype=float)
    req_out = np.asarray(target["route_vinf_out_req_raw_m_s"], dtype=float)
    pos_err_m = norm(r - target_r)
    rp_m = norm(r)
    alt_margin_m = rp_m - safe_radius_m

    if hyp.get("hyperbolic"):
        natural_out = np.asarray(hyp["natural_vinf_out_raw_m_s"], dtype=float)
        out_angle = angle_deg(natural_out, req_out)
        out_vec_err = norm(natural_out - req_out)
        out_mag_mis = norm(natural_out) - norm(req_out)
        out_dir_res = unit(natural_out) - unit(req_out)
    else:
        natural_out = np.array([math.nan, math.nan, math.nan], dtype=float)
        out_angle = math.nan
        out_vec_err = math.inf
        out_mag_mis = math.inf
        out_dir_res = np.array([1e3, 1e3, 1e3], dtype=float)

    dsm_norm = norm(dsm_raw or [0.0, 0.0, 0.0])
    residual = np.concatenate([
        (r - target_r) / pos_scale_m,
        out_dir_weight * out_dir_res,
        np.array([mag_weight * (0.0 if not math.isfinite(out_mag_mis) else out_mag_mis) / vel_scale_m_s]),
    ])

    unsafe = max(0.0, -alt_margin_m / 1000.0)
    score = (
        pos_err_m / 1000.0
        + 1000.0 * out_dir_weight * (0.0 if not math.isfinite(out_angle) else math.radians(out_angle))
        + mag_weight * abs(0.0 if not math.isfinite(out_mag_mis) else out_mag_mis) / 10.0
        + dsm_weight * dsm_norm
        + unsafe_penalty * unsafe
    )

    out = {
        "ok": True,
        "error": "",
        "ca_distance_km": res["ca_distance_m"] / 1000.0,
        "ca_speed_m_s": res["ca_speed_m_s"],
        "ca_radial_v_m_s": res["ca_radial_v_m_s"],
        "ca_dt_s": res["ca_dt_s"],
        "ca_t_game_s": res["ca_t_game_s"],
        "ca_rel_r_raw_m": res["ca_rel_r_raw_m"],
        "ca_rel_v_raw_m_s": res["ca_rel_v_raw_m_s"],
        "status": res["status"],
        "samples": res["samples"],
        "n_burns": res["n_burns"],
        "burns": res["burns"],
        "dsm_dt_s": dsm_dt_s,
        "dsm_raw_m_s": list(map(float, dsm_raw)) if dsm_raw is not None else [0.0, 0.0, 0.0],
        "dsm_norm_m_s": dsm_norm,
        "periapsis_radius_km": rp_m / 1000.0,
        "periapsis_altitude_over_safe_km": alt_margin_m / 1000.0,
        "target_rp_km": target["rp_target_km"],
        "target_pos_err_km": pos_err_m / 1000.0,
        "natural_out_vec_err_m_s": out_vec_err,
        "natural_out_angle_deg": out_angle,
        "natural_out_mag_mismatch_m_s": out_mag_mis,
        "vinf_mag_m_s": hyp.get("vinf_mag_m_s", math.nan),
        "turn_angle_deg": hyp.get("turn_angle_deg", math.nan),
        "hyperbolic": hyp.get("hyperbolic", False),
        "residual": residual.tolist(),
        "residual_norm": norm(residual),
        "score": score,
    }
    return out


def clip_norm(v: np.ndarray, max_norm: float) -> np.ndarray:
    n = norm(v)
    if n > max_norm and n > 0:
        return v * (max_norm / n)
    return v


def make_events(c: dict[str, Any], vessel_guid: str, out_dir: Path, best: dict[str, Any]) -> None:
    burn_abs = float(c["burn_abs_s"])
    burn0_levela = c.get("dv_levela_m_s") or raw_to_levela(c["dv_raw_m_s"])
    dsm_raw = best.get("dsm_raw_m_s", [0.0, 0.0, 0.0])
    dsm_levela = raw_to_levela(dsm_raw)
    dsm_abs = burn_abs + float(best["dsm_dt_s"])

    common = {
        "enabled": True,
        "vessel_guid": vessel_guid,
        "ensure_flight_plan": True,
        "extend_existing_flight_plan": True,
        "mass_tonnes": 2.6,
        "insert_index": 0,
        "burn_template": "json",
        "thrust_kN": 2686.87701225281,
        "specific_impulse_s_g0": 1000.0,
        "is_inertially_fixed": True,
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
        "mode": "insert_levela",
        "require_warp_close": True,
        "max_lead_before_insert_s": 600.0,
        "reject_long_plan_before_first_burn": True,
        "max_first_plan_duration_s": 900.0,
        "rollback_on_status_error": True,
        "auto_sort_insert_index": True,
    }
    event1 = dict(common)
    event1.update({
        "request_id": f"row{c.get('row_index0','x')}_bplane_burn0_attempt0",
        "dedupe_tag": f"row{c.get('row_index0','x')}_bplane_burn0",
        "event_key": f"row{c.get('row_index0','x')}_bplane_burn0",
        "initial_time": burn_abs,
        "plan_final_time": burn_abs + 600.0,
        "delta_v_levela_m_s": [float(x) for x in burn0_levela],
    })
    event2 = dict(common)
    event2.update({
        "request_id": f"row{c.get('row_index0','x')}_bplane_dsm_attempt0",
        "dedupe_tag": f"row{c.get('row_index0','x')}_bplane_dsm",
        "event_key": f"row{c.get('row_index0','x')}_bplane_dsm",
        "initial_time": dsm_abs,
        "plan_final_time": dsm_abs + 600.0,
        "delta_v_levela_m_s": [float(x) for x in dsm_levela],
    })
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "event1_burn0_bplane_inertial_levela.json").write_text(json.dumps(event1, indent=2) + "\n")
    (out_dir / "event2_dsm_bplane_inertial_levela.json").write_text(json.dumps(event2, indent=2) + "\n")


def flatten_row(row: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for k, v in row.items():
        if isinstance(v, list):
            if all(not isinstance(x, (list, dict)) for x in v):
                for i, x in enumerate(v):
                    out[f"{k}_{i}"] = x
            else:
                out[k] = json.dumps(v)
        elif isinstance(v, dict):
            out[k] = json.dumps(v)
        else:
            out[k] = v
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rank-json", type=Path, required=True)
    ap.add_argument("--anchor-json", type=Path, required=True)
    ap.add_argument("--top-index", type=int, default=0)
    ap.add_argument("--leg-in", type=int, default=1)
    ap.add_argument("--leg-out", type=int, default=2)

    ap.add_argument("--server", required=True)
    ap.add_argument("--plugin-b64", type=Path, required=True)
    ap.add_argument("--plugin-arg-mode", choices=["option", "positional"], default="positional")
    ap.add_argument("--vessel-guid", required=True)
    ap.add_argument("--live-state-json", type=Path, required=True)
    ap.add_argument("--dep-body", default=None)
    ap.add_argument("--body", required=True)
    ap.add_argument("--body-catalog", type=Path, required=True)

    ap.add_argument("--min-altitude-km", type=float, default=50.0)
    ap.add_argument("--atmosphere-margin-km", type=float, default=10.0)
    ap.add_argument("--target-altitude-km", type=float, default=None,
                    help="Override target periapsis radius by altitude above body radius. Default uses route turn if safe, else safe altitude.")
    ap.add_argument("--dsm-fractions", default="0.03,0.05,0.08,0.1,0.15,0.2,0.35,0.5,0.7")
    ap.add_argument("--dsm-max-m-s", type=float, default=500.0)
    ap.add_argument("--fd-step-m-s", type=float, default=1.0)
    ap.add_argument("--regularization", type=float, default=1e-6)
    ap.add_argument("--iterations", type=int, default=4)
    ap.add_argument("--scan-half-width-days", type=float, default=20.0)
    ap.add_argument("--arrival-offset-days", type=float, default=0.0)
    ap.add_argument("--vca-samples", type=int, default=101)
    ap.add_argument("--pos-scale-km", type=float, default=1000.0)
    ap.add_argument("--vel-scale-m-s", type=float, default=1000.0)
    ap.add_argument("--out-dir-weight", type=float, default=5.0)
    ap.add_argument("--mag-weight", type=float, default=0.1)
    ap.add_argument("--dsm-weight", type=float, default=0.001)
    ap.add_argument("--unsafe-penalty", type=float, default=10000.0)
    ap.add_argument("--server-timeout-s", type=float, default=900.0)
    ap.add_argument("--quiet-stderr", action="store_true")
    ap.add_argument("--write-events", action="store_true")
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    c = select_candidate(args.rank_json, args.top_index)
    live_t = read_live_t(args.live_state_json)
    radius_km, mu = body_radius_mu(args.body_catalog, args.body)
    safe_alt_km = args.min_altitude_km + args.atmosphere_margin_km
    safe_radius_m = (radius_km + safe_alt_km) * 1000.0

    legs = anchor_legs(args.anchor_json)
    leg_in = legs[args.leg_in - 1]
    leg_out = legs[args.leg_out - 1]
    vinf_in_route = get_vinf_raw_m_s(leg_in, "arr")
    vinf_out_req = get_vinf_raw_m_s(leg_out, "dep")

    target_alt_abs_km = None
    if args.target_altitude_km is not None:
        target_alt_abs_km = radius_km + args.target_altitude_km

    target = compute_route_bplane_target(
        vinf_in_route,
        vinf_out_req,
        mu,
        safe_radius_m,
        target_alt_abs_km,
    )

    sequence = str(c.get("sequence", "")).split()
    dep_body = (args.dep_body or c.get("dep_body") or (sequence[0] if sequence else "KERBIN")).upper()
    arr_body = args.body.upper()

    state_abs_s = float(c["burn_abs_s"])
    t_arr_s = float(c["t_arr_s"])
    tof_rel_s = t_arr_s - state_abs_s
    scan_center_rel_s = tof_rel_s + args.arrival_offset_days * DAY_S
    scan_start_rel_s = scan_center_rel_s - args.scan_half_width_days * DAY_S
    scan_end_rel_s = scan_center_rel_s + args.scan_half_width_days * DAY_S

    rel_r = [float(x) for x in c["burn_rel_r_raw_m"]]
    rel_v = [float(x) for x in c["burn_rel_v_raw_m_s"]]
    burn0_raw = [float(x) for x in c["dv_raw_m_s"]]
    fractions = parse_float_list(args.dsm_fractions)

    print("=== REFINE RANKED CANDIDATE VCAREL B-PLANE V0 ===")
    print(f"row_index0      : {c.get('row_index0')}")
    print(f"sequence        : {c.get('sequence')}")
    print(f"dep -> flyby    : {dep_body} -> {arr_body}")
    print(f"state_abs_s     : {state_abs_s}")
    print(f"tof_from_burn   : {tof_rel_s:.3f}s = {tof_rel_s/DAY_S:.3f} d")
    print(f"body radius     : {radius_km:.3f} km safe_alt={safe_alt_km:.3f} km")
    print(f"route turn      : {target['route_turn_required_deg']:.6f} deg")
    print(f"safe max turn   : {target['safe_max_turn_deg']:.6f} deg")
    print(f"rp route req    : {target['rp_required_unpowered_km']:.3f} km")
    print(f"rp target       : {target['rp_target_km']:.3f} km")
    print(f"ballistic safe  : {target['ballistic_possible_at_safe']}")
    print(f"target rhat raw : {target['target_rhat_raw']}")
    print(f"out req |v|     : {target['route_vinf_out_req_mag_m_s']/1000:.6f} km/s")
    print(f"dsm fractions   : {fractions}")
    print(f"output_dir      : {args.output_dir}")

    rows: list[dict[str, Any]] = []

    # Configura os argumentos congelados para a função do worker
    worker_partial = functools.partial(
        worker_optimize_fraction,
        args=args,
        tof_rel_s=tof_rel_s,
        dep_body=dep_body,
        arr_body=arr_body,
        state_abs_s=state_abs_s,
        scan_start_rel_s=scan_start_rel_s,
        scan_end_rel_s=scan_end_rel_s,
        rel_r=rel_r,
        rel_v=rel_v,
        burn0_raw=burn0_raw,
        mu=mu,
        target=target,
        safe_radius_m=safe_radius_m
    )

    # Inicia o multiprocessing
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # Envia as frações para os workers
        futures = {executor.submit(worker_partial, frac): frac for frac in fractions}

        for future in concurrent.futures.as_completed(futures):
            frac = futures[future]
            try:
                frac_rows = future.result()
                rows.extend(frac_rows)
            except Exception as exc:
                print(f"[!] Erro crítico no worker da fração {frac}: {exc}")

    ok_rows = [r for r in rows if r.get("ok") and math.isfinite(float(r.get("score", math.nan)))]
    ok_rows.sort(key=lambda r: r["score"])
    best = ok_rows[0] if ok_rows else None

    print("\n=== TOP B-PLANE REFINEMENT RESULTS ===")
    for i, r in enumerate(ok_rows[:20], 1):
        print(
            f"{i:2d} score={r['score']:12.3f} "
            f"pos={r['target_pos_err_km']:10.3f}km "
            f"rp={r['periapsis_radius_km']:9.3f}km "
            f"safe={r['periapsis_altitude_over_safe_km']:9.3f}km "
            f"out_ang={r['natural_out_angle_deg']:8.3f} "
            f"out_err={r['natural_out_vec_err_m_s']:8.1f} "
            f"dsm={r['dsm_norm_m_s']:8.3f} "
            f"frac={r.get('fraction')} it={r.get('iteration')} kind={r.get('kind')}"
        )

    out = {
        "schema": "ranked_candidate_vcarel_bplane_refine_v0",
        "rank_json": str(args.rank_json),
        "anchor_json": str(args.anchor_json),
        "top_index": args.top_index,
        "candidate": c,
        "target": target,
        "body": args.body.upper(),
        "body_radius_km": radius_km,
        "safe_altitude_km": safe_alt_km,
        "safe_radius_km": safe_radius_m / 1000.0,
        "config": {
            "dep_body": dep_body,
            "state_abs_s": state_abs_s,
            "t_arr_s": t_arr_s,
            "tof_rel_s": tof_rel_s,
            "scan_start_rel_s": scan_start_rel_s,
            "scan_end_rel_s": scan_end_rel_s,
            "dsm_fractions": fractions,
            "dsm_max_m_s": args.dsm_max_m_s,
            "fd_step_m_s": args.fd_step_m_s,
            "iterations": args.iterations,
            "pos_scale_km": args.pos_scale_km,
            "vel_scale_m_s": args.vel_scale_m_s,
            "out_dir_weight": args.out_dir_weight,
            "mag_weight": args.mag_weight,
        },
        "n_rows": len(rows),
        "n_ok": len(ok_rows),
        "best": best,
        "top": ok_rows[:50],
        "rows": rows,
    }

    json_path = args.output_dir / "ranked_candidate_vcarel_bplane_refine.json"
    csv_path = args.output_dir / "ranked_candidate_vcarel_bplane_refine.csv"
    json_path.write_text(json.dumps(out, indent=2) + "\n")

    flat = [flatten_row(r) for r in rows]
    if flat:
        fields = sorted({k for r in flat for k in r.keys()})
        with csv_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(flat)

    print(f"[OK] wrote {json_path}")
    print(f"[OK] wrote {csv_path}")

    if args.write_events and best and best.get("ok"):
        make_events(c, args.vessel_guid, args.output_dir, best)
        print("[OK] wrote event1_burn0_bplane_inertial_levela.json")
        print("[OK] wrote event2_dsm_bplane_inertial_levela.json")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
