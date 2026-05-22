#!/usr/bin/env python3
"""
refine_ranked_candidate_vcarelnav_bplane_spice_v0.py

Refina primeira queima em T/N/B usando backend SPICE, sem plugin-b64/save.

Entrada:
  - rank-json com candidato executável contendo:
      burn_abs_s, burn_rel_r_raw_m, burn_rel_v_raw_m_s,
      dv_tangent_m_s, dv_normal_m_s opcional, dv_binormal_m_s
  - anchor-json com pernas PyKEP para calcular B-plane alvo.
  - BSP/TPC/body_catalog.

Saída:
  - ranked_candidate_vcarelnav_bplane_spice_refine.json
  - ranked_candidate_vcarelnav_bplane_spice_refine.csv
  - event1_spice_vcarelnav_burn0_navigation.json, se --write-event

O evento final usa:
  mode = insert_navigation
  delta_v_navigation_m_s = [T, N, B]
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from spice_vcarelnav_targeter_v0_3 import (
    SpiceVcarelNavTargeter,
    NavImpulse,
    parse_body_list,
    load_body_catalog,
    DAY_S,
)


def norm(v: Sequence[float]) -> float:
    return float(np.linalg.norm(np.asarray(v, dtype=float)))


def unit(v: Sequence[float]) -> np.ndarray:
    a = np.asarray(v, dtype=float)
    n = norm(a)
    if n <= 0 or not math.isfinite(n):
        raise ValueError(f"cannot normalize vector {v!r}")
    return a / n


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def angle_deg(a: Sequence[float], b: Sequence[float]) -> float:
    na, nb = norm(a), norm(b)
    if na <= 0 or nb <= 0:
        return math.nan
    return math.degrees(math.acos(clamp(float(np.dot(a, b)) / (na * nb), -1.0, 1.0)))


def levela_to_raw(v: Sequence[float]) -> list[float]:
    x, y, z = map(float, v)
    return [z, -x, y]


def anchor_legs(anchor_json: Path) -> list[dict[str, Any]]:
    data = json.loads(anchor_json.read_text())
    if data.get("legs"):
        return data["legs"]
    legs = []
    for i in range(1, 50):
        key = f"leg{i}"
        if key in data:
            legs.append(data[key])
    if not legs:
        raise RuntimeError(f"no legs found in {anchor_json}")
    return legs


def get_vinf_raw_m_s(leg: dict[str, Any], kind: str) -> np.ndarray:
    for k in (f"vinf_{kind}_raw_m_s", f"vinf_{kind}_m_s_raw", f"vinf_{kind}_raw"):
        if k in leg:
            return np.asarray(leg[k], dtype=float)

    if f"vinf_{kind}_levela_km_s" in leg:
        return np.asarray(levela_to_raw([1000.0 * float(x) for x in leg[f"vinf_{kind}_levela_km_s"]]), dtype=float)

    if f"vinf_{kind}_levela_m_s" in leg:
        return np.asarray(levela_to_raw(leg[f"vinf_{kind}_levela_m_s"]), dtype=float)

    raise RuntimeError(f"cannot find vinf_{kind} vector in leg keys={sorted(leg.keys())}")


def compute_route_bplane_target(vinf_in_raw, vinf_out_req_raw, mu: float, safe_radius_m: float, target_altitude_abs_km: float | None = None) -> dict[str, Any]:
    vin = np.asarray(vinf_in_raw, dtype=float)
    vout = np.asarray(vinf_out_req_raw, dtype=float)

    vin_hat = unit(vin)
    vout_hat = unit(vout)
    turn_rad = math.acos(clamp(float(np.dot(vin_hat, vout_hat)), -1.0, 1.0))
    vin_mag = norm(vin)

    if turn_rad <= 1e-12:
        rp_req_m = float("inf")
    else:
        rp_req_m = mu / (vin_mag * vin_mag) * (1.0 / math.sin(turn_rad / 2.0) - 1.0)

    rp_target_m = target_altitude_abs_km * 1000.0 if target_altitude_abs_km is not None else max(safe_radius_m, rp_req_m)

    e_safe = 1.0 + safe_radius_m * vin_mag * vin_mag / mu
    safe_turn_rad = 2.0 * math.asin(clamp(1.0 / e_safe, -1.0, 1.0))

    # Convenção usada nos scripts anteriores:
    # v_inf_in_hat - v_inf_out_hat aponta para o lado do periapsis.
    side = vin_hat - vout_hat
    if norm(side) < 1e-9:
        side = np.cross(vin_hat, np.array([0.0, 0.0, 1.0]))
        if norm(side) < 1e-9:
            side = np.cross(vin_hat, np.array([0.0, 1.0, 0.0]))
    rhat = unit(side)

    return {
        "route_vinf_in_raw_m_s": vin.tolist(),
        "route_vinf_out_req_raw_m_s": vout.tolist(),
        "route_vinf_in_mag_m_s": vin_mag,
        "route_vinf_out_req_mag_m_s": norm(vout),
        "route_turn_required_deg": math.degrees(turn_rad),
        "safe_max_turn_deg": math.degrees(safe_turn_rad),
        "rp_required_unpowered_km": rp_req_m / 1000.0 if math.isfinite(rp_req_m) else math.inf,
        "rp_target_km": rp_target_m / 1000.0,
        "ballistic_possible_at_safe": turn_rad <= safe_turn_rad + 1e-12,
        "target_rhat_raw": rhat.tolist(),
        "target_r_raw_m": (rp_target_m * rhat).tolist(),
    }


def asymptotes_from_ca(r_m, v_m_s, mu: float) -> dict[str, Any]:
    r = np.asarray(r_m, dtype=float)
    v = np.asarray(v_m_s, dtype=float)
    rp = norm(r)
    vp = norm(v)
    eps = 0.5 * vp * vp - mu / rp

    if eps <= 0:
        return {"hyperbolic": False, "vinf_mag_m_s": math.nan, "specific_energy_m2_s2": eps}

    vinf = math.sqrt(2.0 * eps)
    e = 1.0 + rp * vinf * vinf / mu
    if e <= 1.0:
        return {"hyperbolic": False, "vinf_mag_m_s": vinf, "eccentricity": e, "specific_energy_m2_s2": eps}

    rhat = unit(r)
    vt = v - float(np.dot(v, rhat)) * rhat
    that = unit(vt if norm(vt) > 1e-9 else v)

    c = 1.0 / e
    s = math.sqrt(max(0.0, 1.0 - c * c))
    vin_hat = c * rhat + s * that
    vout_hat = -c * rhat + s * that

    return {
        "hyperbolic": True,
        "specific_energy_m2_s2": eps,
        "vinf_mag_m_s": vinf,
        "eccentricity": e,
        "turn_angle_deg": math.degrees(2.0 * math.asin(clamp(1.0 / e, -1.0, 1.0))),
        "natural_vinf_in_raw_m_s": (vinf * vin_hat).tolist(),
        "natural_vinf_out_raw_m_s": (vinf * vout_hat).tolist(),
    }


def eval_nav(targeter: SpiceVcarelNavTargeter, rid: str, cfg: dict[str, Any], x: Sequence[float]) -> dict[str, Any]:
    dvt, dvn, dvb = map(float, x)

    res = targeter.vcarel_nav_spice(
        rid=rid,
        dep_body=cfg["dep_body"],
        arr_body=cfg["arr_body"],
        nav_body=cfg["nav_body"],
        state_abs_s=cfg["state_abs_s"],
        scan_start_rel_s=cfg["scan_start_rel_s"],
        scan_end_rel_s=cfg["scan_end_rel_s"],
        samples=cfg["samples"],
        rel_r_raw_m=cfg["rel_r_raw_m"],
        rel_v_raw_m_s=cfg["rel_v_raw_m_s"],
        impulses_nav=[NavImpulse(0.0, dvt, dvn, dvb)],
    )

    r = np.asarray(res["ca_rel_r_raw_m"], dtype=float)
    v = np.asarray(res["ca_rel_v_raw_m_s"], dtype=float)
    hyp = asymptotes_from_ca(r, v, cfg["mu_m3_s2"])

    target_r = np.asarray(cfg["target"]["target_r_raw_m"], dtype=float)
    req_out = np.asarray(cfg["target"]["route_vinf_out_req_raw_m_s"], dtype=float)

    pos_err_m = norm(r - target_r)
    rp_m = norm(r)
    alt_margin_m = rp_m - cfg["safe_radius_m"]

    if hyp.get("hyperbolic"):
        natural_out = np.asarray(hyp["natural_vinf_out_raw_m_s"], dtype=float)
        out_angle = angle_deg(natural_out, req_out)
        out_vec_err = norm(natural_out - req_out)
        out_mag_mis = norm(natural_out) - norm(req_out)
        out_dir_res = unit(natural_out) - unit(req_out)
    else:
        out_angle = math.inf
        out_vec_err = math.inf
        out_mag_mis = math.inf
        out_dir_res = np.array([1000.0, 1000.0, 1000.0])

    residual = np.concatenate([
        (r - target_r) / cfg["pos_scale_m"],
        cfg["out_dir_weight"] * out_dir_res,
        np.array([cfg["mag_weight"] * (0.0 if not math.isfinite(out_mag_mis) else out_mag_mis) / cfg["vel_scale_m_s"]]),
    ])

    unsafe_km = max(0.0, -alt_margin_m / 1000.0)

    score = (
        pos_err_m / 1000.0
        + 1000.0 * cfg["out_dir_weight"] * (0.0 if not math.isfinite(out_angle) else math.radians(out_angle))
        + cfg["mag_weight"] * abs(0.0 if not math.isfinite(out_mag_mis) else out_mag_mis) / 10.0
        + cfg["dv_weight"] * norm(x)
        + cfg["unsafe_penalty"] * unsafe_km
    )

    out = dict(res)
    out.update({
        "ok": True,
        "error": "",
        "dvt_m_s": dvt,
        "dvn_m_s": dvn,
        "dvb_m_s": dvb,
        "dv_navigation_m_s": [dvt, dvn, dvb],
        "dv_norm_m_s": norm(x),
        "ca_distance_km": res["ca_distance_m"] / 1000.0,
        "periapsis_radius_km": rp_m / 1000.0,
        "periapsis_altitude_over_safe_km": alt_margin_m / 1000.0,
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
    })

    if out.get("burns"):
        b = out["burns"][0]
        for k in ("burn_r_raw_m", "burn_rel_r_raw_m", "burn_v_before_raw_m_s", "burn_rel_v_before_raw_m_s", "dv_navigation_m_s", "tangent_raw", "normal_raw", "binormal_raw", "dv_raw", "burn_v_after_raw_m_s"):
            if k in b:
                out[k] = b[k]

    return out


def clip_bounds(x: np.ndarray, lb: np.ndarray, ub: np.ndarray) -> np.ndarray:
    return np.minimum(np.maximum(x, lb), ub)


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


def make_event(out_dir: Path, c: dict[str, Any], best: dict[str, Any], vessel_guid: str, planned_from_state: dict[str, Any]) -> Path:
    burn_abs_s = float(c.get("burn_abs_s", best["state_t_game_s"]))

    event = {
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
        "request_id": f"row{c.get('row_index0','x')}_spice_vcarelnav_burn0_attempt0",
        "dedupe_tag": f"row{c.get('row_index0','x')}_spice_vcarelnav_burn0",
        "event_key": f"row{c.get('row_index0','x')}_spice_vcarelnav_burn0",
        "attempt": 0,
        "mode": "insert_navigation",
        "initial_time": burn_abs_s,
        "plan_final_time": burn_abs_s + 600.0,
        "delta_v_navigation_m_s": [
            float(best["dvt_m_s"]),
            float(best["dvn_m_s"]),
            float(best["dvb_m_s"]),
        ],
        "planned_from_state": planned_from_state,
        "require_warp_close": True,
        "max_lead_before_insert_s": 600.0,
        "reject_long_plan_before_first_burn": True,
        "max_first_plan_duration_s": 900.0,
        "rollback_on_status_error": True,
        "auto_sort_insert_index": True,
    }

    p = out_dir / "event1_spice_vcarelnav_burn0_navigation.json"
    p.write_text(json.dumps(event, indent=2) + "\n")
    return p


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rank-json", type=Path, required=True)
    ap.add_argument("--anchor-json", type=Path, required=True)
    ap.add_argument("--top-index", type=int, default=0)
    ap.add_argument("--leg-in", type=int, default=1)
    ap.add_argument("--leg-out", type=int, default=2)

    ap.add_argument("--bsp", type=Path, required=True)
    ap.add_argument("--tpc", type=Path, default=None)
    ap.add_argument("--body-catalog", type=Path, required=True)
    ap.add_argument("--attractors", default="SUN,KERBIN,DUNA")
    ap.add_argument("--observer", default="SUN")
    ap.add_argument("--frame", default="J2000")
    ap.add_argument("--spice-time-offset-s", type=float, default=0.0)

    ap.add_argument("--vessel-guid", required=True)
    ap.add_argument("--dep-body", default=None)
    ap.add_argument("--arr-body", default=None)
    ap.add_argument("--nav-body", default=None)

    ap.add_argument("--min-altitude-km", type=float, default=50.0)
    ap.add_argument("--atmosphere-margin-km", type=float, default=10.0)
    ap.add_argument("--target-altitude-km", type=float, default=None)

    ap.add_argument("--tangent-trust-m-s", type=float, default=600.0)
    ap.add_argument("--normal-max-abs-m-s", type=float, default=600.0)
    ap.add_argument("--binormal-max-abs-m-s", type=float, default=1200.0)

    ap.add_argument("--fd-step-m-s", type=float, default=1.0)
    ap.add_argument("--iterations", type=int, default=8)
    ap.add_argument("--regularization", type=float, default=1e-6)

    ap.add_argument("--arrival-offset-days", type=float, default=0.0)
    ap.add_argument("--scan-half-width-days", type=float, default=20.0)
    ap.add_argument("--samples", type=int, default=81)

    ap.add_argument("--pos-scale-km", type=float, default=1000.0)
    ap.add_argument("--vel-scale-m-s", type=float, default=1000.0)
    ap.add_argument("--out-dir-weight", type=float, default=5.0)
    ap.add_argument("--mag-weight", type=float, default=0.1)
    ap.add_argument("--dv-weight", type=float, default=0.001)
    ap.add_argument("--unsafe-penalty", type=float, default=10000.0)

    ap.add_argument("--rtol", type=float, default=1e-9)
    ap.add_argument("--max-step-s", type=float, default=7200.0)

    ap.add_argument("--write-event", action="store_true")
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    rank = json.loads(args.rank_json.read_text())
    c = dict(rank["top"][args.top_index] if "top" in rank else rank.get("candidate", rank))

    seq = str(c.get("sequence", "")).split()
    dep_body = (args.dep_body or c.get("dep_body") or (seq[0] if len(seq) > 0 else "KERBIN")).upper()
    arr_body = args.arr_body or c.get("arr_body") or (seq[1] if len(seq) > 1 else None)
    if arr_body is None:
        raise SystemExit("cannot infer arr_body; pass --arr-body")
    arr_body = arr_body.upper()
    nav_body = (args.nav_body or dep_body).upper()

    for k in ("burn_abs_s", "burn_rel_r_raw_m", "burn_rel_v_raw_m_s"):
        if k not in c:
            raise SystemExit(f"candidate missing {k}")

    state_abs_s = float(c["burn_abs_s"])
    t_arr_s = float(c.get("t_arr_s", state_abs_s + 200 * DAY_S))
    scan_center = (t_arr_s - state_abs_s) + args.arrival_offset_days * DAY_S
    scan_start = scan_center - args.scan_half_width_days * DAY_S
    scan_end = scan_center + args.scan_half_width_days * DAY_S

    rel_r = [float(x) for x in c["burn_rel_r_raw_m"]]
    rel_v = [float(x) for x in c["burn_rel_v_raw_m_s"]]
    x0 = np.array([
        float(c.get("dv_tangent_m_s", c.get("T", 0.0))),
        float(c.get("dv_normal_m_s", c.get("N", 0.0))),
        float(c.get("dv_binormal_m_s", c.get("B", 0.0))),
    ], dtype=float)

    lb = np.array([x0[0] - args.tangent_trust_m_s, -args.normal_max_abs_m_s, -args.binormal_max_abs_m_s], dtype=float)
    ub = np.array([x0[0] + args.tangent_trust_m_s, +args.normal_max_abs_m_s, +args.binormal_max_abs_m_s], dtype=float)
    x = clip_bounds(x0, lb, ub)

    bodies = load_body_catalog(args.body_catalog)
    if arr_body not in bodies:
        raise SystemExit(f"{arr_body} missing in body catalog")
    radius_km = bodies[arr_body]["radius_km"]
    if radius_km is None:
        raise SystemExit(f"{arr_body} radius missing in body catalog")
    mu = bodies[arr_body]["mu_m3_s2"]

    safe_alt_km = args.min_altitude_km + args.atmosphere_margin_km
    safe_radius_m = (radius_km + safe_alt_km) * 1000.0

    legs = anchor_legs(args.anchor_json)
    vinf_in = get_vinf_raw_m_s(legs[args.leg_in - 1], "arr")
    vinf_out = get_vinf_raw_m_s(legs[args.leg_out - 1], "dep")
    target_alt_abs_km = None if args.target_altitude_km is None else radius_km + args.target_altitude_km
    target = compute_route_bplane_target(vinf_in, vinf_out, mu, safe_radius_m, target_alt_abs_km)

    cfg = {
        "dep_body": dep_body,
        "arr_body": arr_body,
        "nav_body": nav_body,
        "state_abs_s": state_abs_s,
        "scan_start_rel_s": scan_start,
        "scan_end_rel_s": scan_end,
        "samples": args.samples,
        "rel_r_raw_m": rel_r,
        "rel_v_raw_m_s": rel_v,
        "mu_m3_s2": mu,
        "target": target,
        "safe_radius_m": safe_radius_m,
        "pos_scale_m": args.pos_scale_km * 1000.0,
        "vel_scale_m_s": args.vel_scale_m_s,
        "out_dir_weight": args.out_dir_weight,
        "mag_weight": args.mag_weight,
        "dv_weight": args.dv_weight,
        "unsafe_penalty": args.unsafe_penalty,
    }

    print("=== REFINE RANKED CANDIDATE VCAREL_NAV SPICE B-PLANE V0 ===")
    print(f"row_index0       : {c.get('row_index0')}")
    print(f"sequence         : {c.get('sequence')}")
    print(f"dep -> arr/nav   : {dep_body} -> {arr_body} / {nav_body}")
    print(f"state_abs_s      : {state_abs_s}")
    print(f"t_arr_s          : {t_arr_s}")
    print(f"scan_rel_s       : {scan_start} .. {scan_end}")
    print(f"x0 nav           : {x0.tolist()} norm={norm(x0):.6f}")
    print(f"bounds           : lb={lb.tolist()} ub={ub.tolist()}")
    print(f"target rp        : {target['rp_target_km']:.6f} km")
    print(f"route turn       : {target['route_turn_required_deg']:.6f} deg")
    print(f"attractors       : {args.attractors}")
    print(f"observer         : {args.observer}")
    print(f"output_dir       : {args.output_dir}")

    rows: list[dict[str, Any]] = []
    eval_counter = 0

    with SpiceVcarelNavTargeter(
        bsp=args.bsp,
        tpc=args.tpc,
        body_catalog=args.body_catalog,
        attractors=parse_body_list(args.attractors),
        frame=args.frame,
        observer=args.observer,
        spice_time_offset_s=args.spice_time_offset_s,
        rtol=args.rtol,
        max_step_s=args.max_step_s,
    ) as targeter:

        for it in range(args.iterations):
            try:
                cur = eval_nav(targeter, f"spice_ref_{eval_counter}", cfg, x)
                eval_counter += 1
                cur.update({"kind": "iterate_base", "iteration": it})
                rows.append(cur)
            except Exception as exc:
                rows.append({"ok": False, "error": str(exc), "kind": "iterate_base", "iteration": it, "dv_navigation_m_s": x.tolist(), "dv_norm_m_s": norm(x)})
                print(f"iter {it:02d}: base failed: {exc}")
                break

            r0 = np.asarray(cur["residual"], dtype=float)
            J = np.zeros((len(r0), 3), dtype=float)

            for j in range(3):
                xp = clip_bounds(x + np.eye(3)[j] * args.fd_step_m_s, lb, ub)
                step = xp[j] - x[j]
                if abs(step) < 1e-12:
                    continue
                try:
                    rp = eval_nav(targeter, f"spice_ref_{eval_counter}", cfg, xp)
                    eval_counter += 1
                    rp.update({"kind": f"fd_axis_{j}", "iteration": it})
                    rows.append(rp)
                    J[:, j] = (np.asarray(rp["residual"], dtype=float) - r0) / step
                except Exception as exc:
                    rows.append({"ok": False, "error": str(exc), "kind": f"fd_axis_{j}", "iteration": it, "dv_navigation_m_s": xp.tolist(), "dv_norm_m_s": norm(xp)})

            JTJ = J.T @ J
            rhs = -J.T @ r0
            lam = args.regularization * max(1.0, float(np.trace(JTJ)) / 3.0)
            try:
                delta = np.linalg.solve(JTJ + lam * np.eye(3), rhs)
            except np.linalg.LinAlgError:
                delta = np.linalg.lstsq(JTJ + lam * np.eye(3), rhs, rcond=None)[0]

            best_local = cur
            best_x = x.copy()
            for alpha in (1.0, 0.5, 0.25, 0.1, 0.05):
                xt = clip_bounds(x + alpha * delta, lb, ub)
                try:
                    tr = eval_nav(targeter, f"spice_ref_{eval_counter}", cfg, xt)
                    eval_counter += 1
                    tr.update({"kind": f"trial_alpha_{alpha}", "iteration": it})
                    rows.append(tr)
                    if tr["score"] < best_local["score"]:
                        best_local = tr
                        best_x = xt
                except Exception as exc:
                    rows.append({"ok": False, "error": str(exc), "kind": f"trial_alpha_{alpha}", "iteration": it, "dv_navigation_m_s": xt.tolist(), "dv_norm_m_s": norm(xt)})

            print(
                f"iter {it:02d}: score={cur['score']:12.3f}->{best_local['score']:12.3f} "
                f"pos={best_local['target_pos_err_km']:10.3f}km "
                f"rp={best_local['periapsis_radius_km']:9.3f}km "
                f"safe={best_local['periapsis_altitude_over_safe_km']:9.3f}km "
                f"out_ang={best_local['natural_out_angle_deg']:8.3f} "
                f"dv={best_local['dv_norm_m_s']:8.3f} "
                f"TNB={best_local['dv_navigation_m_s']}"
            )

            if best_local["score"] + 1e-12 < cur["score"]:
                x = best_x
            else:
                break

    ok_rows = [r for r in rows if r.get("ok") and math.isfinite(float(r.get("score", math.nan)))]
    ok_rows.sort(key=lambda r: r["score"])
    best = ok_rows[0] if ok_rows else None

    print("\n=== TOP SPICE VCAREL_NAV RESULTS ===")
    for i, r in enumerate(ok_rows[:20], 1):
        print(
            f"{i:2d} score={r['score']:12.3f} "
            f"pos={r['target_pos_err_km']:10.3f}km "
            f"rp={r['periapsis_radius_km']:9.3f}km "
            f"safe={r['periapsis_altitude_over_safe_km']:9.3f}km "
            f"out_ang={r['natural_out_angle_deg']:8.3f} "
            f"out_err={r['natural_out_vec_err_m_s']:9.1f} "
            f"dv={r['dv_norm_m_s']:8.3f} "
            f"TNB={r['dv_navigation_m_s']} "
            f"it={r.get('iteration')} kind={r.get('kind')}"
        )

    planned_from_state = {
        "backend": "spice_vcarelnav",
        "bsp": str(args.bsp),
        "tpc": None if args.tpc is None else str(args.tpc),
        "body_catalog": str(args.body_catalog),
        "attractors": args.attractors,
        "observer": args.observer,
        "state_abs_s": state_abs_s,
        "nav_body": nav_body,
        "rel_r_raw_m": rel_r,
        "rel_v_raw_m_s": rel_v,
    }

    out = {
        "schema": "ranked_candidate_vcarelnav_bplane_spice_refine_v0",
        "rank_json": str(args.rank_json),
        "anchor_json": str(args.anchor_json),
        "top_index": args.top_index,
        "candidate": c,
        "target": target,
        "body_radius_km": radius_km,
        "safe_altitude_km": safe_alt_km,
        "safe_radius_km": safe_radius_m / 1000.0,
        "config": {
            "dep_body": dep_body,
            "arr_body": arr_body,
            "nav_body": nav_body,
            "state_abs_s": state_abs_s,
            "t_arr_s": t_arr_s,
            "scan_start_rel_s": scan_start,
            "scan_end_rel_s": scan_end,
            "samples": args.samples,
            "rel_r_raw_m": rel_r,
            "rel_v_raw_m_s": rel_v,
            "x0_navigation_m_s": x0.tolist(),
            "lower_bounds": lb.tolist(),
            "upper_bounds": ub.tolist(),
            "fd_step_m_s": args.fd_step_m_s,
            "iterations": args.iterations,
            "rtol": args.rtol,
            "max_step_s": args.max_step_s,
            "spice_time_offset_s": args.spice_time_offset_s,
            "attractors": args.attractors,
            "observer": args.observer,
        },
        "planned_from_state": planned_from_state,
        "n_rows": len(rows),
        "n_ok": len(ok_rows),
        "best": best,
        "top": ok_rows[:50],
        "rows": rows,
    }

    json_path = args.output_dir / "ranked_candidate_vcarelnav_bplane_spice_refine.json"
    csv_path = args.output_dir / "ranked_candidate_vcarelnav_bplane_spice_refine.csv"
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

    if args.write_event and best:
        event_path = make_event(args.output_dir, c, best, args.vessel_guid, planned_from_state)
        print(f"[OK] wrote {event_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
