#!/usr/bin/env python3
"""
polish_active_vinf_seed_bplane_spice_v0.py

Polidor rápido local para uma seed boa gerada por scan_active_vinf_ejection_spice_v0.py.

Entrada:
  - active-state corrigido (rawfix)
  - anchor-json PyKEP
  - CSV de seeds validadas por SPICE
  - BSP/TPC/body_catalog

Escolhe a melhor seed por ca_distance_km e faz least_squares local em:
  burn_dt_s, dvt, dvn, dvb

Objetivo principal:
  r_CA_rel_to_arr_body ~= target_r_bplane

Ou seja, sai de "acertei Duna por ~10.000 km" para "acertei o B-plane/periapsis alvo".
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy.optimize import least_squares

sys.path.insert(0, str(Path(__file__).resolve().parent))
from spice_vcarelnav_targeter_v0_3 import (
    SpiceVcarelNavTargeter,
    NavImpulse,
    parse_body_list,
    load_body_catalog,
    DAY_S,
)


def norm(v: Sequence[float]) -> float:
    return float(np.linalg.norm(np.asarray(v, dtype=float)))


def unit(v: Sequence[float], name: str = "vector") -> np.ndarray:
    a = np.asarray(v, dtype=float)
    n = norm(a)
    if n <= 0.0 or not math.isfinite(n):
        raise ValueError(f"cannot normalize {name}: {v!r}")
    return a / n


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def angle_deg(a: Sequence[float], b: Sequence[float]) -> float:
    na, nb = norm(a), norm(b)
    if na <= 0.0 or nb <= 0.0:
        return math.nan
    return math.degrees(math.acos(clamp(float(np.dot(a, b)) / (na * nb), -1.0, 1.0)))


def levela_to_raw(v: Sequence[float]) -> list[float]:
    x, y, z = map(float, v)
    return [z, -x, y]


def anchor_legs(anchor_json: Path) -> list[dict[str, Any]]:
    data = json.loads(anchor_json.read_text())
    if isinstance(data.get("legs"), list):
        return data["legs"]
    legs = []
    for i in range(1, 100):
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
    mu_m3_s2: float,
    safe_radius_m: float,
    target_altitude_abs_km: float | None = None,
) -> dict[str, Any]:
    vin = np.asarray(vinf_in_raw, dtype=float)
    vout = np.asarray(vinf_out_req_raw, dtype=float)

    vin_hat = unit(vin, "vinf_in")
    vout_hat = unit(vout, "vinf_out")

    turn_rad = math.acos(clamp(float(np.dot(vin_hat, vout_hat)), -1.0, 1.0))
    vin_mag = norm(vin)

    if turn_rad <= 1e-12:
        rp_req_m = float("inf")
    else:
        rp_req_m = mu_m3_s2 / (vin_mag * vin_mag) * (1.0 / math.sin(turn_rad / 2.0) - 1.0)

    rp_target_m = target_altitude_abs_km * 1000.0 if target_altitude_abs_km is not None else max(safe_radius_m, rp_req_m)

    e_safe = 1.0 + safe_radius_m * vin_mag * vin_mag / mu_m3_s2
    safe_turn_rad = 2.0 * math.asin(clamp(1.0 / e_safe, -1.0, 1.0))

    side = vin_hat - vout_hat
    if norm(side) < 1e-9:
        side = np.cross(vin_hat, np.array([0.0, 0.0, 1.0]))
        if norm(side) < 1e-9:
            side = np.cross(vin_hat, np.array([0.0, 1.0, 0.0]))
    rhat = unit(side, "target_rhat")

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


def asymptotes_from_ca(r_m: Sequence[float], v_m_s: Sequence[float], mu_m3_s2: float) -> dict[str, Any]:
    r = np.asarray(r_m, dtype=float)
    v = np.asarray(v_m_s, dtype=float)
    rp = norm(r)
    vp = norm(v)

    eps = 0.5 * vp * vp - mu_m3_s2 / rp
    if eps <= 0.0:
        return {"hyperbolic": False, "specific_energy_m2_s2": eps, "vinf_mag_m_s": math.nan}

    vinf = math.sqrt(2.0 * eps)
    e = 1.0 + rp * vinf * vinf / mu_m3_s2
    if e <= 1.0:
        return {"hyperbolic": False, "specific_energy_m2_s2": eps, "vinf_mag_m_s": vinf, "eccentricity": e}

    rhat = unit(r, "r_ca")
    vt = v - float(np.dot(v, rhat)) * rhat
    that = unit(vt if norm(vt) > 1e-9 else v, "tangent_ca")

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


def load_seed_from_csv(path: Path, index: int = 0) -> dict[str, Any]:
    rows = []
    with path.open(newline="") as f:
        for r in csv.DictReader(f):
            try:
                ca = float(r["ca_distance_km"])
                if math.isfinite(ca):
                    rows.append((ca, r))
            except Exception:
                pass
    if not rows:
        raise RuntimeError(f"no finite ca_distance_km rows in {path}")
    rows.sort(key=lambda x: x[0])
    row = rows[index][1]
    return {
        "ca_distance_km": float(row["ca_distance_km"]),
        "burn_dt_s": float(row["burn_dt_s"]),
        "dvt_m_s": float(row["dvt_m_s"]),
        "dvn_m_s": float(row["dvn_m_s"]),
        "dvb_m_s": float(row["dvb_m_s"]),
        "dv_norm_m_s": float(row["dv_norm_m_s"]),
        "phase_angle_deg": float(row.get("phase_angle_deg", "nan")),
    }


def safe_float(value, default: float) -> float:
    if value is None:
        return float(default)
    try:
        return float(value)
    except Exception:
        return float(default)


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


def write_event(out_dir: Path, active: dict[str, Any], best: dict[str, Any], cfg: dict[str, Any]) -> Path:
    initial_time = float(cfg["state_abs_s"]) + float(best["burn_dt_s"])
    event = {
        "enabled": True,
        "vessel_guid": active.get("vessel_guid", ""),
        "ensure_flight_plan": True,
        "extend_existing_flight_plan": True,
        "mass_tonnes": safe_float(active.get("mass_tonnes"), 2.6),
        "insert_index": 0,
        "burn_template": "json",
        "thrust_kN": safe_float(active.get("available_thrust_kN"), 2686.87701225281),
        "specific_impulse_s_g0": safe_float(active.get("specific_impulse_s_g0"), 1000.0),
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
        "request_id": "active_polished_bplane_burn0_attempt0",
        "dedupe_tag": "active_polished_bplane_burn0",
        "event_key": "active_polished_bplane_burn0",
        "attempt": 0,
        "mode": "insert_navigation",
        "initial_time": initial_time,
        "plan_final_time": initial_time + 600.0,
        "delta_v_navigation_m_s": [
            float(best["dvt_m_s"]),
            float(best["dvn_m_s"]),
            float(best["dvb_m_s"]),
        ],
        "planned_from_state": {
            "schema": "planned_from_polished_active_seed_v0",
            "active_state_schema": active.get("schema"),
            "state_source": active.get("state_source"),
            "t_game_s": active.get("t_game_s"),
            "t_spice_s": active.get("t_spice_s"),
            "vessel_guid": active.get("vessel_guid"),
            "vessel_name": active.get("vessel_name"),
            "nav_body": cfg["nav_body"],
            "rel_r_raw_m": cfg["rel_r_raw_m"],
            "rel_v_raw_m_s": cfg["rel_v_raw_m_s"],
            "backend": "spice_vcarelnav_bplane_polish",
            "bsp": cfg["bsp"],
            "tpc": cfg["tpc"],
            "body_catalog": cfg["body_catalog"],
            "attractors": cfg["attractors"],
            "observer": cfg["observer"],
        },
        "require_warp_close": True,
        "max_lead_before_insert_s": 600.0,
        "reject_long_plan_before_first_burn": True,
        "max_first_plan_duration_s": 900.0,
        "rollback_on_status_error": True,
        "auto_sort_insert_index": True,
    }

    p = out_dir / "event1_active_polished_bplane_burn0_navigation.json"
    p.write_text(json.dumps(event, indent=2) + "\n")
    return p


class Polisher:
    def __init__(self, targeter: SpiceVcarelNavTargeter, cfg: dict[str, Any]):
        self.targeter = targeter
        self.cfg = cfg
        self.rows: list[dict[str, Any]] = []
        self.best: dict[str, Any] | None = None
        self.eval_count = 0

    def eval(self, x_in: Sequence[float], kind: str = "lsq") -> dict[str, Any]:
        x = np.asarray(x_in, dtype=float)
        x = np.minimum(np.maximum(x, self.cfg["lb"]), self.cfg["ub"])
        burn_dt_s, dvt, dvn, dvb = map(float, x)

        base = {
            "kind": kind,
            "burn_dt_s": burn_dt_s,
            "dvt_m_s": dvt,
            "dvn_m_s": dvn,
            "dvb_m_s": dvb,
            "dv_navigation_m_s": [dvt, dvn, dvb],
            "dv_norm_m_s": norm([dvt, dvn, dvb]),
        }

        try:
            res = self.targeter.vcarel_nav_spice(
                rid=f"polish_{self.eval_count}",
                dep_body=self.cfg["dep_body"],
                arr_body=self.cfg["arr_body"],
                nav_body=self.cfg["nav_body"],
                state_abs_s=self.cfg["state_abs_s"],
                scan_start_rel_s=self.cfg["scan_start_rel_s"],
                scan_end_rel_s=self.cfg["scan_end_rel_s"],
                samples=self.cfg["samples"],
                rel_r_raw_m=self.cfg["rel_r_raw_m"],
                rel_v_raw_m_s=self.cfg["rel_v_raw_m_s"],
                impulses_nav=[NavImpulse(burn_dt_s, dvt, dvn, dvb)],
            )
            self.eval_count += 1

            ca_r = np.asarray(res["ca_rel_r_raw_m"], dtype=float)
            ca_v = np.asarray(res["ca_rel_v_raw_m_s"], dtype=float)
            target_r = np.asarray(self.cfg["target"]["target_r_raw_m"], dtype=float)
            err = ca_r - target_r

            hyp = asymptotes_from_ca(ca_r, ca_v, self.cfg["mu_m3_s2"])
            req_out = np.asarray(self.cfg["target"]["route_vinf_out_req_raw_m_s"], dtype=float)

            if hyp.get("hyperbolic"):
                out = np.asarray(hyp["natural_vinf_out_raw_m_s"], dtype=float)
                out_angle = angle_deg(out, req_out)
                out_vec_err = norm(out - req_out)
                out_mag_mis = norm(out) - norm(req_out)
                out_dir_resid = unit(out, "out") - unit(req_out, "req_out")
            else:
                out_angle = math.inf
                out_vec_err = math.inf
                out_mag_mis = math.inf
                out_dir_resid = np.array([10.0, 10.0, 10.0])

            rp_m = norm(ca_r)
            safe_margin_m = rp_m - self.cfg["safe_radius_m"]
            unsafe_m = max(0.0, -safe_margin_m)

            residual = np.concatenate([
                err / self.cfg["pos_scale_m"],
                self.cfg["out_dir_weight"] * out_dir_resid,
                np.array([
                    self.cfg["mag_weight"] * (0.0 if not math.isfinite(out_mag_mis) else out_mag_mis) / self.cfg["vel_scale_m_s"],
                    math.sqrt(self.cfg["unsafe_weight"]) * unsafe_m / self.cfg["pos_scale_m"],
                    self.cfg["dv_reg_weight"] * (norm([dvt, dvn, dvb]) - self.cfg["seed_dv_norm_m_s"]) / self.cfg["vel_scale_m_s"],
                    self.cfg["burn_reg_weight"] * (burn_dt_s - self.cfg["seed_burn_dt_s"]) / max(1.0, self.cfg["burn_trust_s"]),
                ])
            ])

            score = float(np.dot(residual, residual))

            row = dict(base)
            row.update(res)
            row.update({
                "ok": True,
                "error": "",
                "score": score,
                "ca_distance_km": res["ca_distance_m"] / 1000.0,
                "periapsis_radius_km": rp_m / 1000.0,
                "periapsis_altitude_over_safe_km": safe_margin_m / 1000.0,
                "target_pos_err_km": norm(err) / 1000.0,
                "natural_out_angle_deg": out_angle,
                "natural_out_vec_err_m_s": out_vec_err,
                "natural_out_mag_mismatch_m_s": out_mag_mis,
                "residual": residual.tolist(),
                "residual_norm": norm(residual),
            })

            if row.get("burns"):
                b = row["burns"][0]
                for k in ("tangent_raw", "normal_raw", "binormal_raw", "dv_raw", "burn_r_raw_m", "burn_rel_r_raw_m", "burn_v_before_raw_m_s", "burn_v_after_raw_m_s"):
                    if k in b:
                        row[k] = b[k]

            self.rows.append(row)
            if self.best is None or row["score"] < self.best["score"]:
                self.best = row
            return row

        except Exception as exc:
            self.eval_count += 1
            row = dict(base)
            row.update({
                "ok": False,
                "error": str(exc),
                "score": 1e30,
                "residual": [1e12, 1e12, 1e12, 1e6],
                "residual_norm": 1e30,
            })
            self.rows.append(row)
            return row


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--active-state", type=Path, required=True)
    ap.add_argument("--anchor-json", type=Path, required=True)
    ap.add_argument("--validated-csv", type=Path, required=True)
    ap.add_argument("--seed-index", type=int, default=0)
    ap.add_argument("--leg-in", type=int, default=1)
    ap.add_argument("--leg-out", type=int, default=2)

    ap.add_argument("--bsp", type=Path, required=True)
    ap.add_argument("--tpc", type=Path, default=None)
    ap.add_argument("--body-catalog", type=Path, required=True)
    ap.add_argument("--attractors", default="SUN,KERBIN,DUNA")
    ap.add_argument("--observer", default="SUN")
    ap.add_argument("--frame", default="J2000")
    ap.add_argument("--spice-time-offset-s", type=float, default=0.0)

    ap.add_argument("--dep-body", default=None)
    ap.add_argument("--arr-body", default=None)
    ap.add_argument("--nav-body", default=None)
    ap.add_argument("--t-arr-s", type=float, default=None)

    ap.add_argument("--min-altitude-km", type=float, default=50.0)
    ap.add_argument("--atmosphere-margin-km", type=float, default=10.0)
    ap.add_argument("--target-altitude-km", type=float, default=None)

    ap.add_argument("--burn-trust-s", type=float, default=600.0)
    ap.add_argument("--tangent-trust-m-s", type=float, default=300.0)
    ap.add_argument("--normal-trust-m-s", type=float, default=300.0)
    ap.add_argument("--binormal-trust-m-s", type=float, default=500.0)

    ap.add_argument("--scan-half-width-days", type=float, default=20.0)
    ap.add_argument("--arrival-offset-days", type=float, default=0.0)
    ap.add_argument("--samples", type=int, default=81)

    ap.add_argument("--pos-scale-km", type=float, default=1000.0)
    ap.add_argument("--vel-scale-m-s", type=float, default=1000.0)
    ap.add_argument("--out-dir-weight", type=float, default=1.0)
    ap.add_argument("--mag-weight", type=float, default=0.05)
    ap.add_argument("--unsafe-weight", type=float, default=10000.0)
    ap.add_argument("--dv-reg-weight", type=float, default=0.01)
    ap.add_argument("--burn-reg-weight", type=float, default=0.001)

    ap.add_argument("--max-nfev", type=int, default=45)
    ap.add_argument("--rtol", type=float, default=1e-9)
    ap.add_argument("--max-step-s", type=float, default=21600.0)

    ap.add_argument("--write-event", action="store_true")
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    active = json.loads(args.active_state.read_text())
    seed = load_seed_from_csv(args.validated_csv, args.seed_index)

    legs = anchor_legs(args.anchor_json)
    leg_in = legs[args.leg_in - 1]
    leg_out = legs[args.leg_out - 1]

    dep_body = (args.dep_body or leg_in.get("dep") or leg_in.get("dep_body") or active.get("nav_body") or "KERBIN").upper()
    arr_body = (args.arr_body or leg_in.get("arr") or leg_in.get("arr_body")).upper()
    nav_body = (args.nav_body or active.get("nav_body") or dep_body).upper()

    state_abs_s = float(active.get("t_spice_s", active.get("t_game_s")))
    rel_r = [float(x) for x in active["rel_r_raw_m"]]
    rel_v = [float(x) for x in active["rel_v_raw_m_s"]]

    if args.t_arr_s is not None:
        t_arr_s = float(args.t_arr_s)
    else:
        t_arr_s = float(leg_in.get("t_arr_s"))

    bodies = load_body_catalog(args.body_catalog)
    mu_arr = bodies[arr_body]["mu_m3_s2"]
    radius_km = bodies[arr_body]["radius_km"]
    if radius_km is None:
        raise SystemExit(f"{arr_body} radius missing in body catalog")

    safe_alt_km = args.min_altitude_km + args.atmosphere_margin_km
    safe_radius_m = (radius_km + safe_alt_km) * 1000.0

    vinf_in = get_vinf_raw_m_s(leg_in, "arr")
    vinf_out = get_vinf_raw_m_s(leg_out, "dep")
    target_alt_abs_km = None if args.target_altitude_km is None else radius_km + args.target_altitude_km
    target = compute_route_bplane_target(vinf_in, vinf_out, mu_arr, safe_radius_m, target_alt_abs_km)

    scan_center = (t_arr_s - state_abs_s) + args.arrival_offset_days * DAY_S
    scan_start = scan_center - args.scan_half_width_days * DAY_S
    scan_end = scan_center + args.scan_half_width_days * DAY_S

    x0 = np.array([
        seed["burn_dt_s"],
        seed["dvt_m_s"],
        seed["dvn_m_s"],
        seed["dvb_m_s"],
    ], dtype=float)

    lb = np.array([
        x0[0] - args.burn_trust_s,
        x0[1] - args.tangent_trust_m_s,
        x0[2] - args.normal_trust_m_s,
        x0[3] - args.binormal_trust_m_s,
    ], dtype=float)
    ub = np.array([
        x0[0] + args.burn_trust_s,
        x0[1] + args.tangent_trust_m_s,
        x0[2] + args.normal_trust_m_s,
        x0[3] + args.binormal_trust_m_s,
    ], dtype=float)
    lb[0] = max(0.0, lb[0])

    cfg = {
        "dep_body": dep_body,
        "arr_body": arr_body,
        "nav_body": nav_body,
        "state_abs_s": state_abs_s,
        "rel_r_raw_m": rel_r,
        "rel_v_raw_m_s": rel_v,
        "scan_start_rel_s": scan_start,
        "scan_end_rel_s": scan_end,
        "samples": args.samples,
        "mu_m3_s2": mu_arr,
        "safe_radius_m": safe_radius_m,
        "target": target,
        "pos_scale_m": args.pos_scale_km * 1000.0,
        "vel_scale_m_s": args.vel_scale_m_s,
        "out_dir_weight": args.out_dir_weight,
        "mag_weight": args.mag_weight,
        "unsafe_weight": args.unsafe_weight,
        "dv_reg_weight": args.dv_reg_weight,
        "burn_reg_weight": args.burn_reg_weight,
        "seed_dv_norm_m_s": seed["dv_norm_m_s"],
        "seed_burn_dt_s": seed["burn_dt_s"],
        "burn_trust_s": args.burn_trust_s,
        "lb": lb,
        "ub": ub,
        "bsp": str(args.bsp),
        "tpc": None if args.tpc is None else str(args.tpc),
        "body_catalog": str(args.body_catalog),
        "attractors": args.attractors,
        "observer": args.observer,
    }

    print("=== POLISH ACTIVE VINF SEED B-PLANE SPICE V0 ===")
    print(f"active_state     : {args.active_state}")
    print(f"validated_csv    : {args.validated_csv}")
    print(f"seed_index       : {args.seed_index}")
    print(f"seed             : {seed}")
    print(f"dep -> arr/nav   : {dep_body} -> {arr_body} / {nav_body}")
    print(f"state_abs_s      : {state_abs_s}")
    print(f"t_arr_s          : {t_arr_s}")
    print(f"scan_rel_s       : {scan_start} .. {scan_end}")
    print(f"target_r_km      : {target['rp_target_km']:.6f}")
    print(f"target_rhat      : {target['target_rhat_raw']}")
    print(f"x0               : {x0.tolist()}")
    print(f"lb               : {lb.tolist()}")
    print(f"ub               : {ub.tolist()}")
    print(f"output_dir       : {args.output_dir}")

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
        pol = Polisher(targeter, cfg)

        def fun(x):
            row = pol.eval(x)
            if pol.eval_count % 5 == 0 and pol.best:
                b = pol.best
                print(
                    f"[eval {pol.eval_count:3d}] score={b['score']:12.6g} "
                    f"pos={b['target_pos_err_km']:10.3f}km "
                    f"rp={b['periapsis_radius_km']:9.3f}km "
                    f"out={b['natural_out_angle_deg']:8.3f}deg "
                    f"burn={b['burn_dt_s']:9.3f}s "
                    f"TNB={[b['dvt_m_s'], b['dvn_m_s'], b['dvb_m_s']]}"
                )
            return np.asarray(row["residual"], dtype=float)

        result = least_squares(
            fun,
            x0,
            bounds=(lb, ub),
            method="trf",
            x_scale=[args.burn_trust_s, args.tangent_trust_m_s, args.normal_trust_m_s, args.binormal_trust_m_s],
            ftol=1e-6,
            xtol=1e-4,
            gtol=1e-6,
            max_nfev=args.max_nfev,
            verbose=1,
        )

        final_row = pol.eval(result.x, kind="least_squares_final")

    rows = pol.rows
    ok_rows = [r for r in rows if r.get("ok") and math.isfinite(float(r.get("score", math.inf)))]
    ok_rows.sort(key=lambda r: r["score"])
    best = ok_rows[0] if ok_rows else None

    print("\n=== TOP POLISHED RESULTS ===")
    for i, r in enumerate(ok_rows[:20], 1):
        print(
            f"{i:2d} score={r['score']:12.6g} "
            f"pos={r['target_pos_err_km']:10.3f}km "
            f"rp={r['periapsis_radius_km']:9.3f}km "
            f"safe={r['periapsis_altitude_over_safe_km']:9.3f}km "
            f"out={r['natural_out_angle_deg']:8.3f}deg "
            f"out_err={r['natural_out_vec_err_m_s']:9.1f}m/s "
            f"burn={r['burn_dt_s']:9.3f}s "
            f"dv={r['dv_norm_m_s']:9.3f} "
            f"T={r['dvt_m_s']:9.3f} N={r['dvn_m_s']:9.3f} B={r['dvb_m_s']:9.3f}"
        )

    out = {
        "schema": "polish_active_vinf_seed_bplane_spice_v0",
        "active_state_path": str(args.active_state),
        "validated_csv": str(args.validated_csv),
        "anchor_json": str(args.anchor_json),
        "seed": seed,
        "target": target,
        "body_radius_km": radius_km,
        "safe_altitude_km": safe_alt_km,
        "safe_radius_km": safe_radius_m / 1000.0,
        "active_state": active,
        "config": {
            "dep_body": dep_body,
            "arr_body": arr_body,
            "nav_body": nav_body,
            "state_abs_s": state_abs_s,
            "t_arr_s": t_arr_s,
            "scan_start_rel_s": scan_start,
            "scan_end_rel_s": scan_end,
            "samples": args.samples,
            "x0": x0.tolist(),
            "lower_bounds": lb.tolist(),
            "upper_bounds": ub.tolist(),
            "attractors": args.attractors,
            "observer": args.observer,
            "rtol": args.rtol,
            "max_step_s": args.max_step_s,
            "least_squares": {
                "success": bool(result.success),
                "status": int(result.status),
                "message": str(result.message),
                "cost": float(result.cost),
                "optimality": float(result.optimality),
                "nfev": int(result.nfev),
                "x": [float(v) for v in result.x],
            },
        },
        "n_rows": len(rows),
        "n_ok": len(ok_rows),
        "best": best,
        "top": ok_rows[:50],
    }

    result_json = args.output_dir / "polish_active_vinf_seed_bplane_spice_result.json"
    rows_csv = args.output_dir / "polish_active_vinf_seed_bplane_spice_rows.csv"
    result_json.write_text(json.dumps(out, indent=2) + "\n")

    flat = [flatten_row(r) for r in rows]
    if flat:
        fields = sorted({k for r in flat for k in r.keys()})
        with rows_csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(flat)

    print(f"[OK] wrote {result_json}")
    print(f"[OK] wrote {rows_csv}")

    if args.write_event and best:
        event_path = write_event(args.output_dir, active, best, cfg)
        print(f"[OK] wrote {event_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
