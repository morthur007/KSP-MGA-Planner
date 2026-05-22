#!/usr/bin/env python3
"""
refine_active_vessel_state_vcarelnav_bplane_spice_v0.py

Refina uma queima de departure a partir do estado REAL atual da nave exportado
pela DLL/Principia, sem plugin-b64 e sem estado antigo do rank-json.

Variáveis otimizadas:
  burn_dt_s, dvt, dvn, dvb

O backend propaga:
  active_state.t_spice_s
  active_state.rel_r_raw_m
  active_state.rel_v_raw_m_s
  impulso em T/N/B no burn_dt_s
  closest approach/B-plane no corpo de chegada

Entrada recomendada:
  data/live/active_vessel_state_principia.json

Saída:
  active_vessel_vcarelnav_bplane_spice_refine.json
  active_vessel_vcarelnav_bplane_spice_refine.csv
  event1_active_spice_vcarelnav_burn0_navigation.json

Requer:
  scripts/spice_vcarelnav_targeter_v0_3.py
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy.optimize import minimize

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
    if n <= 0.0 or not math.isfinite(n):
        raise ValueError(f"cannot normalize vector {v!r}")
    return a / n


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def angle_deg(a: Sequence[float], b: Sequence[float]) -> float:
    na = norm(a)
    nb = norm(b)
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

    vin_hat = unit(vin)
    vout_hat = unit(vout)

    turn_rad = math.acos(clamp(float(np.dot(vin_hat, vout_hat)), -1.0, 1.0))
    vin_mag = norm(vin)

    if turn_rad <= 1e-12:
        rp_req_m = float("inf")
    else:
        rp_req_m = mu_m3_s2 / (vin_mag * vin_mag) * (1.0 / math.sin(turn_rad / 2.0) - 1.0)

    rp_target_m = target_altitude_abs_km * 1000.0 if target_altitude_abs_km is not None else max(safe_radius_m, rp_req_m)

    e_safe = 1.0 + safe_radius_m * vin_mag * vin_mag / mu_m3_s2
    safe_turn_rad = 2.0 * math.asin(clamp(1.0 / e_safe, -1.0, 1.0))

    # Mesma convenção usada no refinador anterior.
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


def asymptotes_from_ca(r_m: Sequence[float], v_m_s: Sequence[float], mu_m3_s2: float) -> dict[str, Any]:
    r = np.asarray(r_m, dtype=float)
    v = np.asarray(v_m_s, dtype=float)
    rp = norm(r)
    vp = norm(v)

    eps = 0.5 * vp * vp - mu_m3_s2 / rp
    if eps <= 0.0:
        return {
            "hyperbolic": False,
            "specific_energy_m2_s2": eps,
            "vinf_mag_m_s": math.nan,
        }

    vinf = math.sqrt(2.0 * eps)
    e = 1.0 + rp * vinf * vinf / mu_m3_s2
    if e <= 1.0:
        return {
            "hyperbolic": False,
            "specific_energy_m2_s2": eps,
            "vinf_mag_m_s": vinf,
            "eccentricity": e,
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


def load_optional_rank_seed(rank_json: Path | None, top_index: int) -> dict[str, Any] | None:
    if rank_json is None:
        return None
    data = json.loads(rank_json.read_text())
    if "top" in data:
        return dict(data["top"][top_index])
    if "candidate" in data:
        return dict(data["candidate"])
    return dict(data)


def two_body_period_s(r_m: Sequence[float], v_m_s: Sequence[float], mu_m3_s2: float) -> float | None:
    r = np.asarray(r_m, dtype=float)
    v = np.asarray(v_m_s, dtype=float)
    rn = norm(r)
    vn = norm(v)
    eps = 0.5 * vn * vn - mu_m3_s2 / rn
    if eps >= 0.0:
        return None
    a = -mu_m3_s2 / (2.0 * eps)
    if a <= 0.0 or not math.isfinite(a):
        return None
    return 2.0 * math.pi * math.sqrt(a ** 3 / mu_m3_s2)


class ActiveRefiner:
    def __init__(self, targeter: SpiceVcarelNavTargeter, cfg: dict[str, Any]):
        self.targeter = targeter
        self.cfg = cfg
        self.rows: list[dict[str, Any]] = []
        self.eval_count = 0
        self.best: dict[str, Any] | None = None

    def _bounded(self, x: Sequence[float]) -> np.ndarray:
        a = np.asarray(x, dtype=float)
        return np.minimum(np.maximum(a, self.cfg["lb"]), self.cfg["ub"])

    def evaluate(self, x_in: Sequence[float], *, kind: str = "eval", iteration: int | None = None) -> float:
        x = self._bounded(x_in)
        burn_dt_s, dvt, dvn, dvb = map(float, x)

        row_base = {
            "kind": kind,
            "iteration": iteration,
            "burn_dt_s": burn_dt_s,
            "dvt_m_s": dvt,
            "dvn_m_s": dvn,
            "dvb_m_s": dvb,
            "dv_navigation_m_s": [dvt, dvn, dvb],
            "dv_norm_m_s": norm([dvt, dvn, dvb]),
        }

        try:
            res = self.targeter.vcarel_nav_spice(
                rid=f"active_spice_{self.eval_count}",
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

            r = np.asarray(res["ca_rel_r_raw_m"], dtype=float)
            v = np.asarray(res["ca_rel_v_raw_m_s"], dtype=float)
            hyp = asymptotes_from_ca(r, v, self.cfg["mu_m3_s2"])

            target_r = np.asarray(self.cfg["target"]["target_r_raw_m"], dtype=float)
            req_out = np.asarray(self.cfg["target"]["route_vinf_out_req_raw_m_s"], dtype=float)

            pos_err_m = norm(r - target_r)
            rp_m = norm(r)
            alt_margin_m = rp_m - self.cfg["safe_radius_m"]

            if hyp.get("hyperbolic"):
                natural_out = np.asarray(hyp["natural_vinf_out_raw_m_s"], dtype=float)
                out_angle = angle_deg(natural_out, req_out)
                out_vec_err = norm(natural_out - req_out)
                out_mag_mis = norm(natural_out) - norm(req_out)
            else:
                out_angle = math.inf
                out_vec_err = math.inf
                out_mag_mis = math.inf

            unsafe_km = max(0.0, -alt_margin_m / 1000.0)
            finite_out_angle = 180.0 if not math.isfinite(out_angle) else out_angle
            finite_out_mag = 1.0e6 if not math.isfinite(out_mag_mis) else out_mag_mis

            score = (
                pos_err_m / 1000.0
                + self.cfg["out_angle_weight"] * finite_out_angle
                + self.cfg["mag_weight"] * abs(finite_out_mag) / 10.0
                + self.cfg["dv_weight"] * norm([dvt, dvn, dvb])
                + self.cfg["burn_time_weight"] * abs(burn_dt_s - self.cfg["burn_dt_seed_s"])
                + self.cfg["unsafe_penalty"] * unsafe_km
            )

            row = dict(row_base)
            row.update(res)
            row.update({
                "ok": True,
                "error": "",
                "score": score,
                "ca_distance_km": res["ca_distance_m"] / 1000.0,
                "periapsis_radius_km": rp_m / 1000.0,
                "periapsis_altitude_over_safe_km": alt_margin_m / 1000.0,
                "target_pos_err_km": pos_err_m / 1000.0,
                "natural_out_angle_deg": out_angle,
                "natural_out_vec_err_m_s": out_vec_err,
                "natural_out_mag_mismatch_m_s": out_mag_mis,
                "vinf_mag_m_s": hyp.get("vinf_mag_m_s", math.nan),
                "turn_angle_deg": hyp.get("turn_angle_deg", math.nan),
                "hyperbolic": hyp.get("hyperbolic", False),
            })

            if row.get("burns"):
                b = row["burns"][0]
                for k in (
                    "burn_r_raw_m",
                    "burn_rel_r_raw_m",
                    "burn_v_before_raw_m_s",
                    "burn_rel_v_before_raw_m_s",
                    "dv_navigation_m_s",
                    "tangent_raw",
                    "normal_raw",
                    "binormal_raw",
                    "dv_raw",
                    "burn_v_after_raw_m_s",
                    "burn_rel_v_after_raw_m_s",
                ):
                    if k in b:
                        row[k] = b[k]

            self.rows.append(row)

            if self.best is None or row["score"] < self.best["score"]:
                self.best = row

            return float(score)

        except Exception as exc:
            self.eval_count += 1
            row = dict(row_base)
            row.update({
                "ok": False,
                "error": str(exc),
                "score": self.cfg["fail_score"],
            })
            self.rows.append(row)
            return self.cfg["fail_score"]


def make_event(out_dir: Path, active: dict[str, Any], cfg: dict[str, Any], best: dict[str, Any]) -> Path:
    initial_time = float(cfg["state_abs_s"]) + float(best["burn_dt_s"])
    event = {
        "enabled": True,
        "vessel_guid": active.get("vessel_guid", cfg.get("vessel_guid", "")),
        "ensure_flight_plan": True,
        "extend_existing_flight_plan": True,
        "mass_tonnes": float(active.get("mass_tonnes", 2.6)),
        "insert_index": 0,
        "burn_template": "json",
        "thrust_kN": float(active.get("available_thrust_kN", 2686.87701225281)),
        "specific_impulse_s_g0": float(active.get("specific_impulse_s_g0", 1000.0)),
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
        "request_id": "active_spice_vcarelnav_burn0_attempt0",
        "dedupe_tag": "active_spice_vcarelnav_burn0",
        "event_key": "active_spice_vcarelnav_burn0",
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
            "schema": "planned_from_active_vessel_state_v0",
            "active_state_schema": active.get("schema"),
            "state_source": active.get("state_source"),
            "t_game_s": active.get("t_game_s"),
            "t_spice_s": active.get("t_spice_s"),
            "vessel_guid": active.get("vessel_guid"),
            "vessel_name": active.get("vessel_name"),
            "nav_body": cfg["nav_body"],
            "rel_r_raw_m": cfg["rel_r_raw_m"],
            "rel_v_raw_m_s": cfg["rel_v_raw_m_s"],
            "backend": "spice_vcarelnav",
            "bsp": cfg.get("bsp"),
            "tpc": cfg.get("tpc"),
            "body_catalog": cfg.get("body_catalog"),
            "attractors": cfg.get("attractors"),
            "observer": cfg.get("observer"),
        },
        "require_warp_close": True,
        "max_lead_before_insert_s": 600.0,
        "reject_long_plan_before_first_burn": True,
        "max_first_plan_duration_s": 900.0,
        "rollback_on_status_error": True,
        "auto_sort_insert_index": True,
    }

    p = out_dir / "event1_active_spice_vcarelnav_burn0_navigation.json"
    p.write_text(json.dumps(event, indent=2) + "\n")
    return p


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--active-state", type=Path, required=True)
    ap.add_argument("--anchor-json", type=Path, required=True)
    ap.add_argument("--rank-json", type=Path, default=None, help="Opcional: usado só para seed e t_arr se tiver candidato antigo.")
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

    ap.add_argument("--dep-body", default=None)
    ap.add_argument("--arr-body", default=None)
    ap.add_argument("--nav-body", default=None)
    ap.add_argument("--t-arr-s", type=float, default=None)

    ap.add_argument("--min-altitude-km", type=float, default=50.0)
    ap.add_argument("--atmosphere-margin-km", type=float, default=10.0)
    ap.add_argument("--target-altitude-km", type=float, default=None)

    ap.add_argument("--burn-dt-min-s", type=float, default=0.0)
    ap.add_argument("--burn-dt-max-s", type=float, default=None)
    ap.add_argument("--burn-dt-initial-s", type=float, default=None)
    ap.add_argument("--burn-samples", type=int, default=9)

    ap.add_argument("--tangent-initial-m-s", type=float, default=None)
    ap.add_argument("--normal-initial-m-s", type=float, default=None)
    ap.add_argument("--binormal-initial-m-s", type=float, default=None)
    ap.add_argument("--tangent-trust-m-s", type=float, default=1000.0)
    ap.add_argument("--normal-max-abs-m-s", type=float, default=800.0)
    ap.add_argument("--binormal-max-abs-m-s", type=float, default=1400.0)

    ap.add_argument("--scan-half-width-days", type=float, default=20.0)
    ap.add_argument("--arrival-offset-days", type=float, default=0.0)
    ap.add_argument("--samples", type=int, default=81)

    ap.add_argument("--out-angle-weight", type=float, default=100.0)
    ap.add_argument("--mag-weight", type=float, default=0.1)
    ap.add_argument("--dv-weight", type=float, default=0.001)
    ap.add_argument("--burn-time-weight", type=float, default=0.0)
    ap.add_argument("--unsafe-penalty", type=float, default=10000.0)

    ap.add_argument("--rtol", type=float, default=1e-9)
    ap.add_argument("--max-step-s", type=float, default=7200.0)

    ap.add_argument("--maxiter", type=int, default=80)
    ap.add_argument("--maxfev", type=int, default=180)
    ap.add_argument("--fail-score", type=float, default=1e12)
    ap.add_argument("--write-event", action="store_true")
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    active = json.loads(args.active_state.read_text())
    rank_seed = load_optional_rank_seed(args.rank_json, args.top_index)

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
    elif rank_seed and "t_arr_s" in rank_seed:
        t_arr_s = float(rank_seed["t_arr_s"])
    else:
        t_arr_s = float(leg_in.get("t_arr_s"))

    bodies = load_body_catalog(args.body_catalog)
    if arr_body not in bodies:
        raise SystemExit(f"{arr_body} missing in body catalog")
    if nav_body not in bodies:
        raise SystemExit(f"{nav_body} missing in body catalog")

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

    # Burn window.
    nav_mu = bodies[nav_body]["mu_m3_s2"]
    period_s = two_body_period_s(rel_r, rel_v, nav_mu)
    burn_dt_max = args.burn_dt_max_s
    if burn_dt_max is None:
        burn_dt_max = max(7200.0, min(86400.0, 2.0 * period_s if period_s else 21600.0))

    if args.burn_dt_initial_s is not None:
        burn_dt_seed = args.burn_dt_initial_s
    elif rank_seed and "burn_abs_s" in rank_seed:
        burn_dt_seed = max(args.burn_dt_min_s, min(burn_dt_max, float(rank_seed["burn_abs_s"]) - state_abs_s))
    else:
        burn_dt_seed = min(max(args.burn_dt_min_s, 300.0), burn_dt_max)

    # TNB seed.
    if args.tangent_initial_m_s is not None:
        t_seed = args.tangent_initial_m_s
    elif rank_seed:
        t_seed = float(rank_seed.get("dv_tangent_m_s", rank_seed.get("T", 1900.0)))
    else:
        # Aproximação de magnitude de ejeção prograde.
        rnorm = norm(rel_r)
        vnorm = norm(rel_v)
        vinf_dep = norm(get_vinf_raw_m_s(leg_in, "dep"))
        vp_req = math.sqrt(vinf_dep * vinf_dep + 2.0 * nav_mu / rnorm)
        t_seed = max(500.0, vp_req - vnorm)

    n_seed = (
        args.normal_initial_m_s
        if args.normal_initial_m_s is not None
        else float(rank_seed.get("dv_normal_m_s", rank_seed.get("N", 0.0))) if rank_seed else 0.0
    )
    b_seed = (
        args.binormal_initial_m_s
        if args.binormal_initial_m_s is not None
        else float(rank_seed.get("dv_binormal_m_s", rank_seed.get("B", 0.0))) if rank_seed else 0.0
    )

    x0 = np.array([burn_dt_seed, t_seed, n_seed, b_seed], dtype=float)
    lb = np.array([
        float(args.burn_dt_min_s),
        t_seed - args.tangent_trust_m_s,
        -args.normal_max_abs_m_s,
        -args.binormal_max_abs_m_s,
    ], dtype=float)
    ub = np.array([
        float(burn_dt_max),
        t_seed + args.tangent_trust_m_s,
        args.normal_max_abs_m_s,
        args.binormal_max_abs_m_s,
    ], dtype=float)
    x0 = np.minimum(np.maximum(x0, lb), ub)

    scan_center = (t_arr_s - state_abs_s) + args.arrival_offset_days * DAY_S
    scan_start = scan_center - args.scan_half_width_days * DAY_S
    scan_end = scan_center + args.scan_half_width_days * DAY_S

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
        "out_angle_weight": args.out_angle_weight,
        "mag_weight": args.mag_weight,
        "dv_weight": args.dv_weight,
        "burn_time_weight": args.burn_time_weight,
        "unsafe_penalty": args.unsafe_penalty,
        "fail_score": args.fail_score,
        "lb": lb,
        "ub": ub,
        "burn_dt_seed_s": burn_dt_seed,
        "vessel_guid": active.get("vessel_guid", ""),
        "bsp": str(args.bsp),
        "tpc": None if args.tpc is None else str(args.tpc),
        "body_catalog": str(args.body_catalog),
        "attractors": args.attractors,
        "observer": args.observer,
    }

    print("=== REFINE ACTIVE VESSEL VCAREL_NAV SPICE B-PLANE V0 ===")
    print(f"active_state    : {args.active_state}")
    print(f"state_source    : {active.get('state_source')}")
    print(f"vessel          : {active.get('vessel_name')} {active.get('vessel_guid')}")
    print(f"dep -> arr/nav  : {dep_body} -> {arr_body} / {nav_body}")
    print(f"state_abs_s     : {state_abs_s}")
    print(f"t_arr_s         : {t_arr_s}")
    print(f"tof_remaining_d : {(t_arr_s - state_abs_s) / DAY_S:.6f}")
    print(f"rel_r_norm_km   : {norm(rel_r) / 1000.0:.6f}")
    print(f"rel_v_norm_m_s  : {norm(rel_v):.6f}")
    print(f"period_s        : {period_s}")
    print(f"burn bounds     : {lb[0]} .. {ub[0]}")
    print(f"x0              : {x0.tolist()}")
    print(f"bounds          : lb={lb.tolist()} ub={ub.tolist()}")
    print(f"target rp       : {target['rp_target_km']:.6f} km")
    print(f"route turn      : {target['route_turn_required_deg']:.6f} deg")
    print(f"scan_rel_s      : {scan_start} .. {scan_end}")
    print(f"attractors      : {args.attractors}")
    print(f"output_dir      : {args.output_dir}")

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
        ref = ActiveRefiner(targeter, cfg)

        # Coarse burn-time scan around seed with the seed TNB.
        if args.burn_samples > 1:
            for dt in np.linspace(lb[0], ub[0], args.burn_samples):
                xs = x0.copy()
                xs[0] = float(dt)
                ref.evaluate(xs, kind="coarse_burn_scan", iteration=0)

            if ref.best is not None:
                x0 = np.array([
                    ref.best["burn_dt_s"],
                    ref.best["dvt_m_s"],
                    ref.best["dvn_m_s"],
                    ref.best["dvb_m_s"],
                ], dtype=float)

        print(f"best after coarse: {x0.tolist()} score={ref.best['score'] if ref.best else None}")

        def obj(x):
            val = ref.evaluate(x, kind="powell", iteration=None)
            if ref.eval_count % 10 == 0 and ref.best:
                b = ref.best
                print(
                    f"[eval {ref.eval_count:4d}] best_score={b['score']:12.3f} "
                    f"pos={b.get('target_pos_err_km', math.nan):10.3f}km "
                    f"rp={b.get('periapsis_radius_km', math.nan):9.3f}km "
                    f"out={b.get('natural_out_angle_deg', math.nan):8.3f}deg "
                    f"burn={b['burn_dt_s']:9.1f}s "
                    f"TNB={[b['dvt_m_s'], b['dvn_m_s'], b['dvb_m_s']]}"
                )
            return val

        result = minimize(
            obj,
            x0,
            method="Powell",
            bounds=list(zip(lb, ub)),
            options={
                "maxiter": args.maxiter,
                "maxfev": args.maxfev,
                "xtol": 1e-3,
                "ftol": 1e-3,
                "disp": True,
            },
        )

    rows = ref.rows
    ok_rows = [r for r in rows if r.get("ok") and math.isfinite(float(r.get("score", math.nan)))]
    ok_rows.sort(key=lambda r: r["score"])
    best = ok_rows[0] if ok_rows else None

    print("\n=== TOP ACTIVE SPICE RESULTS ===")
    for i, r in enumerate(ok_rows[:20], 1):
        print(
            f"{i:2d} score={r['score']:12.3f} "
            f"pos={r['target_pos_err_km']:10.3f}km "
            f"rp={r['periapsis_radius_km']:9.3f}km "
            f"safe={r['periapsis_altitude_over_safe_km']:9.3f}km "
            f"out_ang={r['natural_out_angle_deg']:8.3f} "
            f"out_err={r['natural_out_vec_err_m_s']:9.1f} "
            f"burn={r['burn_dt_s']:9.1f}s "
            f"dv={r['dv_norm_m_s']:8.3f} "
            f"TNB={r['dv_navigation_m_s']} "
            f"kind={r.get('kind')}"
        )

    out = {
        "schema": "active_vessel_vcarelnav_bplane_spice_refine_v0",
        "active_state_path": str(args.active_state),
        "anchor_json": str(args.anchor_json),
        "rank_json": None if args.rank_json is None else str(args.rank_json),
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
            "rel_r_raw_m": rel_r,
            "rel_v_raw_m_s": rel_v,
            "x0": x0.tolist(),
            "lower_bounds": lb.tolist(),
            "upper_bounds": ub.tolist(),
            "attractors": args.attractors,
            "observer": args.observer,
            "rtol": args.rtol,
            "max_step_s": args.max_step_s,
            "scipy_result": {
                "success": bool(result.success),
                "status": int(result.status),
                "message": str(result.message),
                "fun": float(result.fun) if math.isfinite(float(result.fun)) else None,
                "x": [float(v) for v in result.x],
                "nfev": int(result.nfev),
                "nit": int(result.nit),
            },
        },
        "n_rows": len(rows),
        "n_ok": len(ok_rows),
        "best": best,
        "top": ok_rows[:50],
        "rows": rows,
    }

    json_path = args.output_dir / "active_vessel_vcarelnav_bplane_spice_refine.json"
    csv_path = args.output_dir / "active_vessel_vcarelnav_bplane_spice_refine.csv"
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
        event_path = make_event(args.output_dir, active, cfg, best)
        print(f"[OK] wrote {event_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
