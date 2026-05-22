#!/usr/bin/env python3
"""
refine_ranked_candidate_vcarelnav_bplane_v0_2.py

Refina a primeira queima usando VCAREL_NAV, isto é, em componentes
Principia/FlightPlan:

    Δv_navigation = [tangent, normal, binormal]

Não usa raw/LevelA para a manobra operacional. O raw fica apenas como debug,
retornado pelo binário.

Entrada típica:
  - candidate_departure_executability_rank.json gerado pelo hunter VCAREL/VCAREL_NAV
  - anchor_packet.json da rota PyKEP/PyKEP-anchor
  - body_catalog.json

Saída:
  - ranked_candidate_vcarelnav_bplane_refine.json
  - ranked_candidate_vcarelnav_bplane_refine.csv
  - event1_vcarelnav_burn0_navigation.json, se --write-event

Importante:
  VCAREL_NAV trata state_dt_s como tempo ABSOLUTO de jogo. Este script usa
  burn_abs_s quando disponível, e cai para live_t + burn_dt_s se necessário.
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
from functools import partial

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from principia_targeter_client import PrincipiaTargeterClient


DAY_S = 86400.0


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
    # Convenção antiga, usada apenas para ler anchors que estejam em LevelA.
    x, y, z = map(float, v)
    return [z, -x, y]


def worker_eval_nav(
    task_info: dict[str, Any],
    args: argparse.Namespace,
    cfg: dict[str, Any]
) -> dict[str, Any]:
    """Worker isolado para rodar uma avaliação do VCAREL_NAV em paralelo."""
    x = task_info["x"]
    eval_id = task_info["eval_id"]
    
    with PrincipiaTargeterClient(
        args.server,
        args.plugin_b64,
        plugin_arg_mode=args.plugin_arg_mode,
        response_timeout_s=args.server_timeout_s,
        quiet_stderr=args.quiet_stderr,
    ) as client:
        try:
            res = eval_nav(client, f"vcarelnav_ref_{os.getpid()}_{eval_id}", cfg, x)
            return {"ok": True, "task": task_info, "result": res}
        except Exception as exc:
            return {"ok": False, "task": task_info, "error": str(exc)}

def read_live_t(path: Path) -> float:
    d = json.loads(path.read_text())
    for k in ("ut_s", "t_s", "spice_t_s", "et_s", "current_ut", "time_s"):
        if k in d:
            return float(d[k])
    raise KeyError(f"cannot find live time in {path}")


def find_body_record(obj: Any, body: str):
    bl = body.lower()
    if isinstance(obj, dict):
        name = str(obj.get("name", obj.get("body", obj.get("id", "")))).lower()
        if name == bl:
            return obj
        for k, v in obj.items():
            if str(k).lower() == bl and isinstance(v, dict):
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
        for k in ("radius_m", "mean_radius_m", "equatorial_radius_m", "radius", "equatorial_radius"):
            if k in rec:
                val = float(rec[k])
                radius = val / 1000.0 if val > 1e5 else val
                break

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

    if radius is None or mu is None:
        raise RuntimeError(f"missing radius/mu for {body}; keys={sorted(rec.keys())}")
    return radius, mu


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


def compute_route_bplane_target(
    vinf_in_raw: Sequence[float],
    vinf_out_req_raw: Sequence[float],
    mu: float,
    safe_radius_m: float,
    target_altitude_abs_km: float | None = None,
) -> dict[str, Any]:
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

    # Para a convenção de periapsis usada aqui:
    # v_inf_in_hat - v_inf_out_hat ~= (2/e) * rhat_periapsis.
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


def asymptotes_from_ca(r_m: Sequence[float], v_m_s: Sequence[float], mu: float) -> dict[str, Any]:
    r = np.asarray(r_m, dtype=float)
    v = np.asarray(v_m_s, dtype=float)

    rp = norm(r)
    vp = norm(v)
    eps = 0.5 * vp * vp - mu / rp

    if eps <= 0:
        return {
            "hyperbolic": False,
            "vinf_mag_m_s": math.nan,
            "specific_energy_m2_s2": eps,
        }

    vinf = math.sqrt(2.0 * eps)
    e = 1.0 + rp * vinf * vinf / mu

    if e <= 1.0:
        return {
            "hyperbolic": False,
            "vinf_mag_m_s": vinf,
            "eccentricity": e,
            "specific_energy_m2_s2": eps,
        }

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


def eval_nav(client: PrincipiaTargeterClient, rid: str, cfg: dict[str, Any], x: Sequence[float]) -> dict[str, Any]:
    dvt, dvn, dvb = map(float, x)

    res = client.vcarel_nav(
        rid,
        cfg["dep_body"],
        cfg["arr_body"],
        cfg["nav_body"],
        cfg["state_abs_s"],
        cfg["scan_start_rel_s"],
        cfg["scan_end_rel_s"],
        cfg["samples"],
        cfg["rel_r_raw_m"],
        cfg["rel_v_raw_m_s"],
        [(0.0, dvt, dvn, dvb)],
        timeout_s=cfg["timeout_s"],
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
        out_dir_res = np.array([1000.0, 1000.0, 1000.0], dtype=float)

    residual = np.concatenate([
        (r - target_r) / cfg["pos_scale_m"],
        cfg["out_dir_weight"] * out_dir_res,
        np.array([
            cfg["mag_weight"] * (0.0 if not math.isfinite(out_mag_mis) else out_mag_mis) / cfg["vel_scale_m_s"]
        ]),
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
        "ca_radial_v_m_s": res.get("ca_radial_velocity_m_s"),
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
        for k in ("burn_r_raw_m", "burn_v_before_raw_m_s", "dv_navigation_m_s", "tangent_raw", "normal_raw", "binormal_raw", "dv_raw", "burn_v_after_raw_m_s"):
            if k in b:
                out[k] = b[k]

    return out


def clip_bounds(x: np.ndarray, lb: np.ndarray, ub: np.ndarray) -> np.ndarray:
    return np.minimum(np.maximum(x, lb), ub)


def flatten_row(row: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
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


def make_event(out_dir: Path, c: dict[str, Any], best: dict[str, Any], vessel_guid: str) -> Path:
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
        "request_id": f"row{c.get('row_index0','x')}_vcarelnav_burn0_attempt0",
        "dedupe_tag": f"row{c.get('row_index0','x')}_vcarelnav_burn0",
        "event_key": f"row{c.get('row_index0','x')}_vcarelnav_burn0",
        "attempt": 0,
        "mode": "insert_navigation",
        "initial_time": burn_abs_s,
        "plan_final_time": burn_abs_s + 600.0,
        "delta_v_navigation_m_s": [
            float(best["dvt_m_s"]),
            float(best["dvn_m_s"]),
            float(best["dvb_m_s"]),
        ],
        "require_warp_close": True,
        "max_lead_before_insert_s": 600.0,
        "reject_long_plan_before_first_burn": True,
        "max_first_plan_duration_s": 900.0,
        "rollback_on_status_error": True,
        "auto_sort_insert_index": True,
    }

    p = out_dir / "event1_vcarelnav_burn0_navigation.json"
    p.write_text(json.dumps(event, indent=2) + "\n")
    return p


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
    ap.add_argument("--arr-body", default=None)
    ap.add_argument("--nav-body", default=None)
    ap.add_argument("--body-catalog", type=Path, required=True)

    ap.add_argument("--min-altitude-km", type=float, default=50.0)
    ap.add_argument("--atmosphere-margin-km", type=float, default=10.0)
    ap.add_argument("--target-altitude-km", type=float, default=None)

    ap.add_argument("--tangent-trust-m-s", type=float, default=600.0)
    ap.add_argument("--normal-max-abs-m-s", type=float, default=600.0)
    ap.add_argument("--binormal-max-abs-m-s", type=float, default=1200.0)

    ap.add_argument("--fd-step-m-s", type=float, default=1.0)
    ap.add_argument("--iterations", type=int, default=10)
    ap.add_argument("--regularization", type=float, default=1e-6)

    ap.add_argument("--arrival-offset-days", type=float, default=0.0)
    ap.add_argument("--scan-half-width-days", type=float, default=20.0)
    ap.add_argument("--vca-samples", type=int, default=101)

    ap.add_argument("--pos-scale-km", type=float, default=1000.0)
    ap.add_argument("--vel-scale-m-s", type=float, default=1000.0)
    ap.add_argument("--out-dir-weight", type=float, default=5.0)
    ap.add_argument("--mag-weight", type=float, default=0.1)
    ap.add_argument("--dv-weight", type=float, default=0.001)
    ap.add_argument("--unsafe-penalty", type=float, default=10000.0)

    ap.add_argument("--server-timeout-s", type=float, default=900.0)
    ap.add_argument("--quiet-stderr", action="store_true")
    ap.add_argument("--write-event", action="store_true")
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    data = json.loads(args.rank_json.read_text())
    c = dict(data["top"][args.top_index])
    live_t = read_live_t(args.live_state_json)

    seq = str(c.get("sequence", "")).split()
    dep_body = (args.dep_body or c.get("dep_body") or (seq[0] if len(seq) > 0 else "KERBIN")).upper()
    arr_body = (args.arr_body or c.get("arr_body") or (seq[1] if len(seq) > 1 else None))
    if arr_body is None:
        raise SystemExit("cannot infer arr_body; pass --arr-body")
    arr_body = arr_body.upper()
    nav_body = (args.nav_body or dep_body).upper()

    required = ["burn_dt_s", "burn_rel_r_raw_m", "burn_rel_v_raw_m_s"]
    missing = [k for k in required if k not in c]
    if missing:
        raise SystemExit(f"candidate missing required fields: {missing}")

    burn_dt_rel_s = float(c["burn_dt_s"])
    state_abs_s = float(c.get("burn_abs_s", live_t + burn_dt_rel_s))
    t_arr_s = float(c.get("t_arr_s", state_abs_s + 200.0 * DAY_S))

    # VCAREL_NAV usa scan relativo ao state_abs_s/state_t_game_s.
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

    lb = np.array([
        x0[0] - args.tangent_trust_m_s,
        -args.normal_max_abs_m_s,
        -args.binormal_max_abs_m_s,
    ], dtype=float)

    ub = np.array([
        x0[0] + args.tangent_trust_m_s,
        +args.normal_max_abs_m_s,
        +args.binormal_max_abs_m_s,
    ], dtype=float)

    x = clip_bounds(x0, lb, ub)

    radius_km, mu = body_radius_mu(args.body_catalog, arr_body)
    safe_alt_km = args.min_altitude_km + args.atmosphere_margin_km
    safe_radius_m = (radius_km + safe_alt_km) * 1000.0

    legs = anchor_legs(args.anchor_json)
    leg_in = legs[args.leg_in - 1]
    leg_out = legs[args.leg_out - 1]

    vinf_in = get_vinf_raw_m_s(leg_in, "arr")
    vinf_out = get_vinf_raw_m_s(leg_out, "dep")

    target_alt_abs_km = None if args.target_altitude_km is None else radius_km + args.target_altitude_km
    target = compute_route_bplane_target(vinf_in, vinf_out, mu, safe_radius_m, target_alt_abs_km)

    cfg = {
        "dep_body": dep_body,
        "arr_body": arr_body,
        "nav_body": nav_body,
        "state_abs_s": state_abs_s,
        "scan_start_rel_s": scan_start,
        "scan_end_rel_s": scan_end,
        "samples": args.vca_samples,
        "rel_r_raw_m": rel_r,
        "rel_v_raw_m_s": rel_v,
        "timeout_s": args.server_timeout_s,
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

    print("=== REFINE RANKED CANDIDATE VCAREL_NAV B-PLANE V0.2 ===")
    print(f"row_index0       : {c.get('row_index0')}")
    print(f"sequence         : {c.get('sequence')}")
    print(f"dep -> arr / nav : {dep_body} -> {arr_body} / {nav_body}")
    print(f"burn_dt_rel_s    : {burn_dt_rel_s}")
    print(f"state_abs_s      : {state_abs_s}")
    print(f"t_arr_s          : {t_arr_s}")
    print(f"scan_rel_s       : {scan_start} .. {scan_end}")
    print(f"x0 nav           : {x0.tolist()} norm={norm(x0):.6f}")
    print(f"bounds           : lb={lb.tolist()} ub={ub.tolist()}")
    print(f"target rp        : {target['rp_target_km']:.6f} km")
    print(f"target rhat raw  : {target['target_rhat_raw']}")
    print(f"route turn       : {target['route_turn_required_deg']:.6f} deg")
    print(f"safe max turn    : {target['safe_max_turn_deg']:.6f} deg")
    print(f"output_dir       : {args.output_dir}")

    rows: list[dict[str, Any]] = []
    eval_counter = 0

    # Congelamos args e cfg para os workers
    worker_func = partial(worker_eval_nav, args=args, cfg=cfg)

    print("=== INICIANDO MULTIPROCESSING (FD e Line Search) ===")
    
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for it in range(args.iterations):
            
            # 1. AVALIAÇÃO BASE (Sequencial, serve de pivô para o cálculo de erro)
            base_task = {"kind": "iterate_base", "iteration": it, "x": x, "eval_id": eval_counter}
            eval_counter += 1
            base_res = worker_func(base_task)
            
            if not base_res["ok"]:
                rows.append({"ok": False, "error": base_res["error"], "kind": "iterate_base", "iteration": it, "dv_navigation_m_s": x.tolist(), "dv_norm_m_s": norm(x)})
                print(f"iter {it:02d}: base failed: {base_res['error']}")
                break
                
            cur = base_res["result"]
            cur.update({"kind": "iterate_base", "iteration": it})
            rows.append(cur)

            r0 = np.asarray(cur["residual"], dtype=float)
            J = np.zeros((len(r0), 3), dtype=float)

            # 2. DIFERENÇAS FINITAS (Paralelo - 3 eixos simulâneos)
            fd_tasks = []
            for j in range(3):
                xp = x.copy()
                xp[j] += args.fd_step_m_s
                xp = clip_bounds(xp, lb, ub)
                step = xp[j] - x[j]
                if abs(step) < 1e-12:
                    continue
                fd_tasks.append({"kind": f"fd_axis_{j}", "axis": j, "x": xp, "step": step, "eval_id": eval_counter})
                eval_counter += 1
                
            if fd_tasks:
                futures = [executor.submit(worker_func, t) for t in fd_tasks]
                for future in concurrent.futures.as_completed(futures):
                    res = future.result()
                    task = res["task"]
                    j = task["axis"]
                    xp = task["x"]
                    
                    if res["ok"]:
                        rp = res["result"]
                        rp.update({"kind": task["kind"], "iteration": it})
                        rows.append(rp)
                        J[:, j] = (np.asarray(rp["residual"], dtype=float) - r0) / task["step"]
                    else:
                        rows.append({"ok": False, "error": res["error"], "kind": task["kind"], "iteration": it, "dv_navigation_m_s": xp.tolist(), "dv_norm_m_s": norm(xp)})

            # Cálculo de Resolução Matemática
            JTJ = J.T @ J
            rhs = -J.T @ r0
            lam = args.regularization * max(1.0, float(np.trace(JTJ)) / 3.0)

            try:
                delta = np.linalg.solve(JTJ + lam * np.eye(3), rhs)
            except np.linalg.LinAlgError:
                delta = np.linalg.lstsq(JTJ + lam * np.eye(3), rhs, rcond=None)[0]

            best_local = cur
            best_x = x.copy()

            # 3. LINE SEARCH (Paralelo - 5 tentativas simultâneas)
            alpha_tasks = []
            for alpha in (1.0, 0.5, 0.25, 0.1, 0.05):
                xt = clip_bounds(x + alpha * delta, lb, ub)
                alpha_tasks.append({"kind": f"trial_alpha_{alpha}", "alpha": alpha, "x": xt, "eval_id": eval_counter})
                eval_counter += 1
                
            alpha_futures = [executor.submit(worker_func, t) for t in alpha_tasks]
            
            for future in concurrent.futures.as_completed(alpha_futures):
                res = future.result()
                task = res["task"]
                xt = task["x"]
                
                if res["ok"]:
                    tr = res["result"]
                    tr.update({"kind": task["kind"], "iteration": it})
                    rows.append(tr)
                    if tr["score"] < best_local["score"]:
                        best_local = tr
                        best_x = xt
                else:
                    rows.append({"ok": False, "error": res["error"], "kind": task["kind"], "iteration": it, "dv_navigation_m_s": xt.tolist(), "dv_norm_m_s": norm(xt)})

            print(
                f"iter {it:02d}: score={cur['score']:12.3f}->{best_local['score']:12.3f} "
                f"pos={best_local['target_pos_err_km']:10.3f}km "
                f"rp={best_local['periapsis_radius_km']:9.3f}km "
                f"safe={best_local['periapsis_altitude_over_safe_km']:9.3f}km "
                f"out_ang={best_local['natural_out_angle_deg']:8.3f} "
                f"dv={best_local['dv_norm_m_s']:8.3f} "
                f"TNB={best_local['dv_navigation_m_s']}"
            )

            # Define se continua baseando-se no avanço do score
            if best_local["score"] + 1e-12 < cur["score"]:
                x = best_x
            else:
                break

    ok_rows = [r for r in rows if r.get("ok") and math.isfinite(float(r.get("score", math.nan)))]
    # O restante do seu código de ordenação e exports continua igual
    ok_rows.sort(key=lambda r: r["score"])
    best = ok_rows[0] if ok_rows else None

    print("\n=== TOP VCAREL_NAV RESULTS ===")
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

    out = {
        "schema": "ranked_candidate_vcarelnav_bplane_refine_v0_2",
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
            "burn_dt_rel_s": burn_dt_rel_s,
            "state_abs_s": state_abs_s,
            "t_arr_s": t_arr_s,
            "scan_start_rel_s": scan_start,
            "scan_end_rel_s": scan_end,
            "samples": args.vca_samples,
            "rel_r_raw_m": rel_r,
            "rel_v_raw_m_s": rel_v,
            "x0_navigation_m_s": x0.tolist(),
            "lower_bounds": lb.tolist(),
            "upper_bounds": ub.tolist(),
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

    json_path = args.output_dir / "ranked_candidate_vcarelnav_bplane_refine.json"
    csv_path = args.output_dir / "ranked_candidate_vcarelnav_bplane_refine.csv"

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
        event_path = make_event(args.output_dir, c, best, args.vessel_guid)
        print(f"[OK] wrote {event_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
