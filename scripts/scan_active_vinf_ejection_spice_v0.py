#!/usr/bin/env python3
"""
scan_active_vinf_ejection_spice_v0.py

Seed rápido para departure a partir do estado REAL atual da nave.

Ideia:
  1. Propaga a órbita de estacionamento em dois-corpos ao redor do nav_body.
  2. Para milhares de possíveis burn_dt, calcula analiticamente a queima que
     produz o v_inf_dep da perna PyKEP/Lambert.
  3. Mantém os melhores por erro de fase/asymptote.
  4. Valida só os melhores em SPICE/N-body.
  5. Exporta evento insert_navigation.

Isso substitui "Powell cego em burn_dt,T,N,B". O otimizador local não deve
descobrir sozinho a fase de ejeção; ela vem do Lambert/v_inf.
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
from scipy.integrate import solve_ivp

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
    if not math.isfinite(n) or n <= 0:
        raise ValueError(f"cannot normalize {name}: {v!r}")
    return a / n


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def angle_deg(a: Sequence[float], b: Sequence[float]) -> float:
    na = norm(a)
    nb = norm(b)
    if na <= 0 or nb <= 0:
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


def tnb_basis(r: Sequence[float], v: Sequence[float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r = np.asarray(r, dtype=float)
    v = np.asarray(v, dtype=float)
    T = unit(v, "T")
    h = np.cross(r, v)
    if norm(h) < 1e-12:
        radial = unit(r, "radial")
        B = unit(np.cross(radial, T), "fallback B")
    else:
        B = unit(h, "B")
    N = unit(np.cross(B, T), "N")
    return T, N, B


def two_body_period_s(r: Sequence[float], v: Sequence[float], mu: float) -> float | None:
    rn = norm(r)
    vn = norm(v)
    eps = 0.5 * vn * vn - mu / rn
    if eps >= 0:
        return None
    a = -mu / (2.0 * eps)
    if a <= 0 or not math.isfinite(a):
        return None
    return 2.0 * math.pi * math.sqrt(a ** 3 / mu)


def propagate_twobody_dense(r0: np.ndarray, v0: np.ndarray, mu: float, t_final_s: float, rtol: float = 1e-11):
    def rhs(t, y):
        r = y[:3]
        v = y[3:]
        rn = norm(r)
        return np.r_[v, -mu * r / rn**3]

    sol = solve_ivp(
        rhs,
        (0.0, float(t_final_s)),
        np.r_[r0, v0],
        method="DOP853",
        rtol=rtol,
        atol=[1e-3, 1e-3, 1e-3, 1e-9, 1e-9, 1e-9],
        dense_output=True,
        max_step=120.0,
    )
    if not sol.success:
        raise RuntimeError(f"two-body propagation failed: {sol.message}")
    return sol


def ejection_solution_at_state(r: np.ndarray, v: np.ndarray, mu: float, vinf_raw: np.ndarray) -> dict[str, Any]:
    rp = norm(r)
    vinf_mag = norm(vinf_raw)
    vinf_hat = unit(vinf_raw, "vinf")
    rhat = unit(r, "r")

    e = 1.0 + rp * vinf_mag * vinf_mag / mu
    c = 1.0 / e
    s = math.sqrt(max(0.0, 1.0 - c * c))
    vp = math.sqrt(vinf_mag * vinf_mag + 2.0 * mu / rp)

    # Para uma saída hiperbólica em que o burn point é periapsis:
    # vinf_hat = -c*rhat + s*that.
    # Se a fase não for exata, projetamos a direção candidata no plano tangente.
    raw_that = vinf_hat + c * rhat
    tangential = raw_that - float(np.dot(raw_that, rhat)) * rhat
    if norm(tangential) <= 1e-12:
        raise RuntimeError("degenerate tangential direction")

    that = unit(tangential, "hyperbola periapsis tangent")
    predicted_vinf_hat = unit(-c * rhat + s * that, "predicted_vinf_hat")

    v_post = vp * that
    dv_raw = v_post - v

    T, N, B = tnb_basis(r, v)
    dvt = float(np.dot(dv_raw, T))
    dvn = float(np.dot(dv_raw, N))
    dvb = float(np.dot(dv_raw, B))

    return {
        "rp_km": rp / 1000.0,
        "vinf_mag_m_s": vinf_mag,
        "ejection_eccentricity": e,
        "required_dot_rhat_vinfhat": -c,
        "actual_dot_rhat_vinfhat": float(np.dot(rhat, vinf_hat)),
        "phase_scalar_error": float(np.dot(rhat, vinf_hat) + c),
        "phase_angle_deg": angle_deg(predicted_vinf_hat, vinf_hat),
        "vp_required_m_s": vp,
        "v_pre_norm_m_s": norm(v),
        "v_post_norm_m_s": norm(v_post),
        "dv_raw_m_s": dv_raw.tolist(),
        "dv_navigation_m_s": [dvt, dvn, dvb],
        "dvt_m_s": dvt,
        "dvn_m_s": dvn,
        "dvb_m_s": dvb,
        "dv_norm_m_s": norm(dv_raw),
        "predicted_vinf_hat": predicted_vinf_hat.tolist(),
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


def safe_float(value, default: float) -> float:
    if value is None:
        return float(default)
    try:
        return float(value)
    except Exception:
        return float(default)


def normalize_active_state(active: dict[str, Any]) -> dict[str, Any]:
    """Aceita tanto active_state antigo quanto snapshot vivo da DLL."""
    if "vessel" not in active:
        return active

    v = active["vessel"]
    out = dict(active)
    for k in (
        "vessel_guid",
        "vessel_name",
        "nav_body",
        "rel_r_raw_m",
        "rel_v_raw_m_s",
        "mass_tonnes",
        "available_thrust_kN",
        "specific_impulse_s_g0",
        "state_source",
    ):
        if k in v:
            out[k] = v[k]

    # Preserva tempo do snapshot no topo.
    if "t_spice_s" not in out and "t_spice_s" in active:
        out["t_spice_s"] = active["t_spice_s"]
    if "t_game_s" not in out and "t_game_s" in active:
        out["t_game_s"] = active["t_game_s"]

    return out


def find_rank_candidate(rank_json: Path | None, top_index: int, row_index0: int | None = None) -> dict[str, Any] | None:
    if rank_json is None:
        return None

    data = json.loads(rank_json.read_text())

    candidates: list[dict[str, Any]] = []

    def walk(x):
        if isinstance(x, dict):
            if "row_index0" in x and ("vinf_dep_raw_m_s" in x or "t_arr_s" in x):
                candidates.append(x)
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)

    walk(data)

    if row_index0 is not None:
        for c in candidates:
            if int(c.get("row_index0", -999999)) == int(row_index0):
                return c

    if "top" in data and isinstance(data["top"], list) and data["top"]:
        return data["top"][top_index]

    if candidates:
        return candidates[top_index]

    if isinstance(data, dict) and ("vinf_dep_raw_m_s" in data or "t_arr_s" in data):
        return data

    return None


def get_vinf_dep_with_rank_fallback(leg: Any, rank_candidate: dict[str, Any] | None) -> np.ndarray:
    if isinstance(leg, dict):
        try:
            return get_vinf_raw_m_s(leg, "dep")
        except Exception:
            pass

    if rank_candidate is not None:
        for k in ("vinf_dep_raw_m_s", "vinf_dep_m_s_raw", "vinf_dep_raw"):
            if k in rank_candidate:
                return np.asarray(rank_candidate[k], dtype=float)
        if "vinf_dep_levela_m_s" in rank_candidate:
            return np.asarray(levela_to_raw(rank_candidate["vinf_dep_levela_m_s"]), dtype=float)
        if "vinf_dep_levela_km_s" in rank_candidate:
            return np.asarray(levela_to_raw([1000.0 * float(x) for x in rank_candidate["vinf_dep_levela_km_s"]]), dtype=float)

    raise RuntimeError("cannot find vinf_dep in anchor leg or rank candidate")


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
        "request_id": "active_vinf_seed_burn0_attempt0",
        "dedupe_tag": "active_vinf_seed_burn0",
        "event_key": "active_vinf_seed_burn0",
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
            "schema": "planned_from_active_vessel_state_vinf_seed_v0",
            "active_state_schema": active.get("schema"),
            "state_source": active.get("state_source"),
            "t_game_s": active.get("t_game_s"),
            "t_spice_s": active.get("t_spice_s"),
            "vessel_guid": active.get("vessel_guid"),
            "vessel_name": active.get("vessel_name"),
            "nav_body": cfg["nav_body"],
            "rel_r_raw_m": cfg["rel_r_raw_m"],
            "rel_v_raw_m_s": cfg["rel_v_raw_m_s"],
            "vinf_dep_raw_m_s": cfg["vinf_dep_raw_m_s"],
            "backend": "twobody_vinf_seed_plus_spice_validation",
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

    p = out_dir / "event1_active_vinf_seed_burn0_navigation.json"
    p.write_text(json.dumps(event, indent=2) + "\n")
    return p


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--active-state", type=Path, required=True)
    ap.add_argument("--anchor-json", type=Path, required=True)
    ap.add_argument("--rank-json", type=Path, default=None)
    ap.add_argument("--top-index", type=int, default=0)
    ap.add_argument("--row-index0", type=int, default=None)
    ap.add_argument("--leg", type=int, default=1)
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

    ap.add_argument("--burn-dt-min-s", type=float, default=0.0)
    ap.add_argument("--burn-dt-max-s", type=float, default=None)
    ap.add_argument("--burn-grid", type=int, default=1441)
    ap.add_argument("--phase-top-n", type=int, default=80)
    ap.add_argument("--validate-top-n", type=int, default=25)

    ap.add_argument("--dv-min-m-s", type=float, default=500.0)
    ap.add_argument("--dv-max-m-s", type=float, default=5000.0)
    ap.add_argument("--max-abs-phase-deg", type=float, default=25.0)

    ap.add_argument("--scan-half-width-days", type=float, default=30.0)
    ap.add_argument("--arrival-offset-days", type=float, default=0.0)
    ap.add_argument("--samples", type=int, default=61)

    ap.add_argument("--rtol", type=float, default=1e-8)
    ap.add_argument("--max-step-s", type=float, default=21600.0)
    ap.add_argument("--write-event", action="store_true")
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    active_raw = json.loads(args.active_state.read_text())
    active = normalize_active_state(active_raw)
    rank_candidate = find_rank_candidate(args.rank_json, args.top_index, args.row_index0)

    legs = anchor_legs(args.anchor_json)
    leg = legs[args.leg - 1]
    leg_dict = leg if isinstance(leg, dict) else {}

    dep_body = (args.dep_body or leg_dict.get("dep") or leg_dict.get("dep_body") or active.get("nav_body") or "KERBIN").upper()
    arr_body = (args.arr_body or leg_dict.get("arr") or leg_dict.get("arr_body") or (rank_candidate or {}).get("arr_body") or "EVE").upper()
    nav_body = (args.nav_body or active.get("nav_body") or dep_body).upper()

    state_abs_s = float(active.get("t_spice_s", active.get("t_game_s")))
    rel_r = np.asarray(active["rel_r_raw_m"], dtype=float)
    rel_v = np.asarray(active["rel_v_raw_m_s"], dtype=float)

    if args.t_arr_s is not None:
        t_arr_s = float(args.t_arr_s)
    elif isinstance(leg, dict) and "t_arr_s" in leg:
        t_arr_s = float(leg["t_arr_s"])
    elif rank_candidate is not None and "t_arr_s" in rank_candidate:
        t_arr_s = float(rank_candidate["t_arr_s"])
    else:
        raise SystemExit("cannot infer t_arr_s; pass --t-arr-s or use anchor/rank containing t_arr_s")

    bodies = load_body_catalog(args.body_catalog)
    mu_nav = bodies[nav_body]["mu_m3_s2"]

    vinf_dep = get_vinf_dep_with_rank_fallback(leg, rank_candidate)

    period = two_body_period_s(rel_r, rel_v, mu_nav)
    burn_dt_max = args.burn_dt_max_s
    if burn_dt_max is None:
        burn_dt_max = max(7200.0, min(3.0 * period if period else 21600.0, 21600.0))

    print("=== SCAN ACTIVE VINF EJECTION SPICE V0 ===")
    print(f"active_state       : {args.active_state}")
    print(f"state_source       : {active.get('state_source')}")
    print(f"dep -> arr/nav     : {dep_body} -> {arr_body} / {nav_body}")
    print(f"state_abs_s        : {state_abs_s}")
    print(f"t_arr_s            : {t_arr_s}")
    print(f"tof_remaining_days : {(t_arr_s-state_abs_s)/DAY_S:.6f}")
    print(f"rel_r_norm_km      : {norm(rel_r)/1000:.6f}")
    print(f"rel_v_norm_m_s     : {norm(rel_v):.6f}")
    print(f"period_s           : {period}")
    print(f"vinf_dep_raw_m_s   : {vinf_dep.tolist()} |v|={norm(vinf_dep):.6f}")
    print(f"burn_dt range      : {args.burn_dt_min_s} .. {burn_dt_max}")
    print(f"burn_grid          : {args.burn_grid}")
    print(f"phase_top_n        : {args.phase_top_n}")
    print(f"validate_top_n     : {args.validate_top_n}")
    print(f"output_dir         : {args.output_dir}")

    sol = propagate_twobody_dense(rel_r, rel_v, mu_nav, burn_dt_max, rtol=1e-11)

    phase_rows: list[dict[str, Any]] = []
    for burn_dt in np.linspace(args.burn_dt_min_s, burn_dt_max, args.burn_grid):
        y = np.asarray(sol.sol(float(burn_dt)), dtype=float)
        r = y[:3]
        v = y[3:]
        try:
            ej = ejection_solution_at_state(r, v, mu_nav, vinf_dep)
        except Exception as exc:
            phase_rows.append({
                "ok": False,
                "kind": "phase",
                "burn_dt_s": float(burn_dt),
                "error": str(exc),
            })
            continue

        gate_ok = (
            abs(ej["phase_angle_deg"]) <= args.max_abs_phase_deg
            and args.dv_min_m_s <= ej["dv_norm_m_s"] <= args.dv_max_m_s
        )
        row = {
            "ok": True,
            "kind": "phase",
            "gate_ok": gate_ok,
            "burn_dt_s": float(burn_dt),
            "burn_abs_s": state_abs_s + float(burn_dt),
            "rmag_km": norm(r) / 1000.0,
            "vmag_m_s": norm(v),
            **ej,
        }
        # Prioridade de seed: primeiro fase, depois dv.
        row["phase_score"] = abs(row["phase_angle_deg"]) + 0.0001 * row["dv_norm_m_s"]
        phase_rows.append(row)

    phase_ok = [r for r in phase_rows if r.get("ok") and r.get("gate_ok")]
    if not phase_ok:
        phase_ok = [r for r in phase_rows if r.get("ok")]

    phase_ok.sort(key=lambda r: r["phase_score"])
    top_phase = phase_ok[: args.phase_top_n]

    print("\n=== TOP TWO-BODY VINF PHASE CANDIDATES ===")
    for i, r in enumerate(top_phase[:20], 1):
        print(
            f"{i:2d} burn={r['burn_dt_s']:9.1f}s "
            f"phase={r['phase_angle_deg']:9.4f}deg "
            f"dv={r['dv_norm_m_s']:9.3f} "
            f"T={r['dvt_m_s']:9.3f} N={r['dvn_m_s']:9.3f} B={r['dvb_m_s']:9.3f} "
            f"r={r['rmag_km']:9.3f}km"
        )

    scan_center = (t_arr_s - state_abs_s) + args.arrival_offset_days * DAY_S
    scan_start = scan_center - args.scan_half_width_days * DAY_S
    scan_end = scan_center + args.scan_half_width_days * DAY_S

    validate_rows: list[dict[str, Any]] = []
    cfg = {
        "state_abs_s": state_abs_s,
        "nav_body": nav_body,
        "rel_r_raw_m": rel_r.tolist(),
        "rel_v_raw_m_s": rel_v.tolist(),
        "vinf_dep_raw_m_s": vinf_dep.tolist(),
        "bsp": str(args.bsp),
        "tpc": None if args.tpc is None else str(args.tpc),
        "body_catalog": str(args.body_catalog),
        "attractors": args.attractors,
        "observer": args.observer,
    }

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
        for idx, seed in enumerate(top_phase[: args.validate_top_n]):
            try:
                res = targeter.vcarel_nav_spice(
                    rid=f"active_vinf_seed_{idx}",
                    dep_body=dep_body,
                    arr_body=arr_body,
                    nav_body=nav_body,
                    state_abs_s=state_abs_s,
                    scan_start_rel_s=scan_start,
                    scan_end_rel_s=scan_end,
                    samples=args.samples,
                    rel_r_raw_m=rel_r.tolist(),
                    rel_v_raw_m_s=rel_v.tolist(),
                    impulses_nav=[
                        NavImpulse(
                            float(seed["burn_dt_s"]),
                            float(seed["dvt_m_s"]),
                            float(seed["dvn_m_s"]),
                            float(seed["dvb_m_s"]),
                        )
                    ],
                )
                row = dict(seed)
                row.update(res)
                row.update({
                    "kind": "spice_validation",
                    "ca_distance_km": res["ca_distance_m"] / 1000.0,
                    "validation_score": res["ca_distance_m"] / 1000.0 + 0.001 * seed["dv_norm_m_s"],
                })
                if row.get("burns"):
                    b = row["burns"][0]
                    for k in ("tangent_raw", "normal_raw", "binormal_raw", "dv_raw", "burn_r_raw_m", "burn_rel_r_raw_m", "burn_v_before_raw_m_s", "burn_v_after_raw_m_s"):
                        if k in b:
                            row[k] = b[k]
                validate_rows.append(row)
                print(
                    f"[val {idx+1:3d}/{min(args.validate_top_n, len(top_phase)):3d}] "
                    f"ca={row['ca_distance_km']:12.3f}km "
                    f"phase={row['phase_angle_deg']:8.3f} "
                    f"burn={row['burn_dt_s']:9.1f}s "
                    f"dv={row['dv_norm_m_s']:9.3f}"
                )
            except Exception as exc:
                row = dict(seed)
                row.update({
                    "kind": "spice_validation",
                    "validation_ok": False,
                    "error": str(exc),
                    "validation_score": float("inf"),
                })
                validate_rows.append(row)
                print(f"[val {idx+1:3d}] ERR {exc}")

    valid = [r for r in validate_rows if math.isfinite(float(r.get("validation_score", math.inf)))]
    valid.sort(key=lambda r: r["validation_score"])
    best = valid[0] if valid else None

    print("\n=== TOP SPICE VALIDATED VINF SEEDS ===")
    for i, r in enumerate(valid[:20], 1):
        print(
            f"{i:2d} ca={r['ca_distance_km']:12.3f}km "
            f"phase={r['phase_angle_deg']:8.3f}deg "
            f"burn={r['burn_dt_s']:9.1f}s "
            f"dv={r['dv_norm_m_s']:9.3f} "
            f"T={r['dvt_m_s']:9.3f} N={r['dvn_m_s']:9.3f} B={r['dvb_m_s']:9.3f} "
            f"ca_t={r['ca_t_game_s']:.3f}"
        )

    out = {
        "schema": "active_vinf_ejection_spice_scan_v0",
        "active_state_path": str(args.active_state),
        "anchor_json": str(args.anchor_json),
        "rank_json": None if args.rank_json is None else str(args.rank_json),
        "leg_index": args.leg,
        "dep_body": dep_body,
        "arr_body": arr_body,
        "nav_body": nav_body,
        "state_abs_s": state_abs_s,
        "t_arr_s": t_arr_s,
        "scan_start_rel_s": scan_start,
        "scan_end_rel_s": scan_end,
        "active_state": active,
        "vinf_dep_raw_m_s": vinf_dep.tolist(),
        "vinf_dep_norm_m_s": norm(vinf_dep),
        "config": {
            "burn_dt_min_s": args.burn_dt_min_s,
            "burn_dt_max_s": burn_dt_max,
            "burn_grid": args.burn_grid,
            "phase_top_n": args.phase_top_n,
            "validate_top_n": args.validate_top_n,
            "dv_min_m_s": args.dv_min_m_s,
            "dv_max_m_s": args.dv_max_m_s,
            "max_abs_phase_deg": args.max_abs_phase_deg,
            "attractors": args.attractors,
            "observer": args.observer,
            "rtol": args.rtol,
            "max_step_s": args.max_step_s,
        },
        "n_phase_rows": len(phase_rows),
        "n_phase_ok": len([r for r in phase_rows if r.get("ok")]),
        "n_phase_gate_ok": len([r for r in phase_rows if r.get("ok") and r.get("gate_ok")]),
        "n_validated": len(validate_rows),
        "n_valid": len(valid),
        "best": best,
        "top_phase": top_phase[:50],
        "top_validated": valid[:50],
    }

    result_json = args.output_dir / "active_vinf_ejection_spice_scan_result.json"
    phase_csv = args.output_dir / "active_vinf_ejection_phase_rows.csv"
    valid_csv = args.output_dir / "active_vinf_ejection_validated_rows.csv"
    result_json.write_text(json.dumps(out, indent=2) + "\n")

    for path, rows in ((phase_csv, phase_rows), (valid_csv, validate_rows)):
        flat = [flatten_row(r) for r in rows]
        if flat:
            fields = sorted({k for r in flat for k in r.keys()})
            with path.open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fields)
                w.writeheader()
                w.writerows(flat)

    print(f"[OK] wrote {result_json}")
    print(f"[OK] wrote {phase_csv}")
    print(f"[OK] wrote {valid_csv}")

    if args.write_event and best:
        event_path = write_event(args.output_dir, active, best, cfg)
        print(f"[OK] wrote {event_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
