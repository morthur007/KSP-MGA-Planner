#!/usr/bin/env python3
"""
scan_pykep_vinf_ejection_v0_1.py

PyKEP-anchored departure scan.

This script does NOT let the N-body targeter invent an arbitrary transfer.
It reconstructs the physical LKO ejection burn implied by the PyKEP departure
v-infinity vector, then evaluates that nominal burn in Principia N-body via VCA.

For each burn epoch near the PyKEP t_dep:
  1. Query vessel relative state to dep body with VREL.
  2. Treat current position as hyperbolic periapsis of the ejection.
  3. Use PyKEP vinf magnitude and direction to compute an ideal two-body
     periapsis burn vector.
  4. Evaluate closest approach to arrival body with VCA.

This is a diagnostic/seed generator, not a free optimizer.

Frame:
  Anchor packet stores vinf_dep_raw_m_s in Principia raw axes.
  VCA/VREL burn vectors are Principia raw inertial m/s.

Output:
  - CSV of candidate ejection burns.
  - JSON summary with best candidates.
  - Optional FlightPlan legacy event for the best candidate, as inertially fixed.
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

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from principia_targeter_client import PrincipiaTargeterClient


DAY_S = 86400.0


def norm(v: Sequence[float]) -> float:
    return float(np.linalg.norm(np.asarray(v, dtype=float)))


def unit(v: Sequence[float]) -> np.ndarray:
    a = np.asarray(v, dtype=float)
    n = np.linalg.norm(a)
    if not np.isfinite(n) or n <= 0:
        raise ValueError(f"cannot normalize vector {v}")
    return a / n


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def angle_deg(a: Sequence[float], b: Sequence[float]) -> float:
    ua = unit(a)
    ub = unit(b)
    return math.degrees(math.acos(clamp(float(np.dot(ua, ub)), -1.0, 1.0)))


def raw_to_levela(v: Sequence[float]) -> list[float]:
    x, y, z = map(float, v)
    return [-y, z, x]


def read_live_t(path: Path) -> float:
    r = json.loads(path.read_text())
    for k in ("ut_s", "t_s", "spice_t_s", "et_s"):
        if k in r:
            return float(r[k])
    raise KeyError(f"could not find time in {path}")


def parse_offsets(args: argparse.Namespace) -> list[float]:
    if args.burn_offsets_s:
        return [float(x.strip()) for x in args.burn_offsets_s.replace(";", ",").split(",") if x.strip()]
    out = []
    x = float(args.burn_offset_min_s)
    while x <= float(args.burn_offset_max_s) + 1e-9:
        out.append(x)
        x += float(args.burn_offset_step_s)
    return out


def parse_arrival_offsets_days(args: argparse.Namespace) -> list[float]:
    if args.arrival_offsets_days:
        return [float(x.strip()) for x in args.arrival_offsets_days.replace(";", ",").split(",") if x.strip()]
    return [0.0]


def find_leg(anchor: dict[str, Any], leg_index: int) -> dict[str, Any]:
    for leg in anchor["legs"]:
        if int(leg["leg_index"]) == int(leg_index):
            return leg
    raise KeyError(f"leg_index {leg_index} not found in anchor")


def find_mu_from_catalog(path: Path, body: str) -> float | None:
    if not path:
        return None
    data = json.loads(path.read_text())
    body_l = body.lower()

    def walk(obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                if str(k).lower() == body_l and isinstance(v, dict):
                    yield v
            name = str(obj.get("name", obj.get("body", obj.get("id", "")))).lower()
            if name == body_l:
                yield obj
            for v in obj.values():
                yield from walk(v)
        elif isinstance(obj, list):
            for x in obj:
                yield from walk(x)

    for obj in walk(data):
        for key in (
            "mu_m3_s2", "gm_m3_s2", "gravitational_parameter_m3_s2",
            "mu", "gm", "gravitational_parameter",
        ):
            if key in obj:
                val = float(obj[key])
                if val < 1e12:  # likely km^3/s^2
                    val *= 1e9
                return val
    return None


def compute_ejection_dv(
    r_raw_m: Sequence[float],
    v_raw_m_s: Sequence[float],
    vinf_raw_m_s: Sequence[float],
    mu_m3_s2: float,
) -> dict[str, Any]:
    r = np.asarray(r_raw_m, dtype=float)
    v = np.asarray(v_raw_m_s, dtype=float)
    s = unit(vinf_raw_m_s)
    vinf_mag = norm(vinf_raw_m_s)

    rmag = norm(r)
    vmag = norm(v)
    rhat = r / rmag

    # Hyperbolic periapsis energy.
    vp_mag = math.sqrt(vinf_mag * vinf_mag + 2.0 * mu_m3_s2 / rmag)

    # Hyperbolic eccentricity if current point is periapsis.
    ecc = 1.0 + rmag * vinf_mag * vinf_mag / mu_m3_s2
    nu_inf = math.acos(clamp(-1.0 / ecc, -1.0, 1.0))

    theta = math.acos(clamp(float(np.dot(rhat, s)), -1.0, 1.0))
    phase_error = theta - nu_inf

    # Velocity at periapsis lies in the plane spanned by r and vinf, perpendicular to r.
    # This is exact only when theta == nu_inf. When not exact, it is still the closest
    # local periapsis direction pointing toward the desired asymptote.
    proj = s - float(np.dot(s, rhat)) * rhat
    if norm(proj) < 1e-12:
        # Degenerate: choose current transverse direction.
        proj = v - float(np.dot(v, rhat)) * rhat
    t_hat = unit(proj)

    # Prefer the branch close to current prograde direction if both projections are ambiguous.
    transverse_current = v - float(np.dot(v, rhat)) * rhat
    if norm(transverse_current) > 1e-9 and np.dot(t_hat, transverse_current) < 0:
        # Flipping changes outgoing asymptote, so keep the PyKEP projection by default.
        # We record branch_prograde_dot to diagnose.
        pass

    v_after = vp_mag * t_hat
    dv = v_after - v

    # Local components for diagnostics.
    T = unit(transverse_current) if norm(transverse_current) > 1e-9 else unit(v)
    B = unit(np.cross(r, v))
    N = unit(np.cross(B, T))
    dv_tnb = np.array([np.dot(dv, T), np.dot(dv, N), np.dot(dv, B)], dtype=float)

    eps_after = 0.5 * norm(v_after) ** 2 - mu_m3_s2 / rmag
    vinf_after_mag = math.sqrt(max(0.0, 2.0 * eps_after))

    return {
        "rmag_m": rmag,
        "vmag_m_s": vmag,
        "vinf_mag_m_s": vinf_mag,
        "vp_mag_m_s": vp_mag,
        "ecc": ecc,
        "nu_inf_deg": math.degrees(nu_inf),
        "theta_to_vinf_deg": math.degrees(theta),
        "phase_error_deg": math.degrees(phase_error),
        "dv_raw_m_s": dv.tolist(),
        "dv_norm_m_s": norm(dv),
        "dv_tnb_diag_m_s": dv_tnb.tolist(),
        "v_after_raw_m_s": v_after.tolist(),
        "vinf_after_mag_m_s": vinf_after_mag,
        "branch_prograde_dot": float(np.dot(t_hat, unit(transverse_current))) if norm(transverse_current) > 1e-9 else float("nan"),
    }


def make_event(
    vessel_guid: str,
    burn_abs_s: float,
    dv_raw_m_s: Sequence[float],
    out_path: Path,
    request_id: str,
) -> dict[str, Any]:
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
        "request_id": request_id,
        "dedupe_tag": request_id,
        "event_key": request_id,
        "attempt": 0,
        "mode": "insert_levela",
        "initial_time": float(burn_abs_s),
        "plan_final_time": float(burn_abs_s) + 600.0,
        "delta_v_levela_m_s": raw_to_levela(dv_raw_m_s),
        "require_warp_close": True,
        "max_lead_before_insert_s": 600.0,
        "reject_long_plan_before_first_burn": True,
        "max_first_plan_duration_s": 900.0,
        "rollback_on_status_error": True,
        "auto_sort_insert_index": True,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(event, indent=2) + "\n")
    return event


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--server", required=True)
    ap.add_argument("--plugin-b64", type=Path, required=True)
    ap.add_argument("--plugin-arg-mode", choices=["option", "positional"], default="positional")
    ap.add_argument("--vessel-guid", required=True)
    ap.add_argument("--live-state-json", type=Path, required=True)
    ap.add_argument("--anchor-json", type=Path, required=True)
    ap.add_argument("--leg", type=int, default=1)
    ap.add_argument("--dep-body", default=None)
    ap.add_argument("--arr-body", default=None)

    ap.add_argument("--body-catalog", type=Path, default=None)
    ap.add_argument("--dep-mu-m3-s2", type=float, default=None)

    ap.add_argument("--burn-offsets-s", default=None)
    ap.add_argument("--burn-offset-min-s", type=float, default=-7200.0)
    ap.add_argument("--burn-offset-max-s", type=float, default=7200.0)
    ap.add_argument("--burn-offset-step-s", type=float, default=600.0)

    ap.add_argument("--arrival-offsets-days", default="-10,-7,-5,-3,-1,0,1,3,5,7,10")
    ap.add_argument("--scan-half-width-days", type=float, default=5.0)
    ap.add_argument("--vca-samples", type=int, default=41)

    ap.add_argument("--dv-min-m-s", type=float, default=1000.0)
    ap.add_argument("--dv-max-m-s", type=float, default=3200.0)
    ap.add_argument("--max-abs-phase-error-deg", type=float, default=90.0)
    ap.add_argument("--server-timeout-s", type=float, default=600.0)
    ap.add_argument("--quiet-stderr", action="store_true")
    ap.add_argument("--top-n", type=int, default=20)
    ap.add_argument("--phase-top-n", type=int, default=40,
                    help="Evaluate VCA only for this many best phase candidates. 0 means all gate-ok candidates.")
    ap.add_argument("--phase-target-deg", type=float, default=0.0,
                    help="Desired phase error in degrees, normally 0.")
    ap.add_argument("--phase-weight-dv", type=float, default=0.0005,
                    help="Small ranking weight for dv_norm in phase preselection.")
    ap.add_argument("--write-best-event", action="store_true")
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    anchor = json.loads(args.anchor_json.read_text())
    leg = find_leg(anchor, args.leg)
    live_t = read_live_t(args.live_state_json)

    dep_body = (args.dep_body or leg["dep_body"]).upper()
    arr_body = (args.arr_body or leg["arr_body"]).upper()

    mu = args.dep_mu_m3_s2
    if mu is None and args.body_catalog:
        mu = find_mu_from_catalog(args.body_catalog, dep_body)
    if mu is None:
        raise SystemExit("Provide --dep-mu-m3-s2 or --body-catalog containing the departure body's GM/mu.")

    t_dep_abs = float(leg["t_dep_s"])
    t_arr_abs = float(leg["t_arr_s"])
    nominal_burn_dt = t_dep_abs - live_t
    nominal_arrival_dt = t_arr_abs - live_t
    vinf_raw = np.asarray(leg["vinf_dep_raw_m_s"], dtype=float)

    offsets = parse_offsets(args)
    arr_offsets_days = parse_arrival_offsets_days(args)

    rows: list[dict[str, Any]] = []
    phase_rows: list[dict[str, Any]] = []

    print("=== SCAN PYKEP VINF EJECTION V0.1 / TWO-STAGE PHASE SCAN ===")
    print(f"sequence leg       : {dep_body} -> {arr_body}")
    print(f"live_t             : {live_t}")
    print(f"t_dep_abs          : {t_dep_abs}")
    print(f"nominal_burn_dt    : {nominal_burn_dt}")
    print(f"t_arr_abs          : {t_arr_abs}")
    print(f"nominal_arrival_dt : {nominal_arrival_dt}")
    print(f"vinf_raw_m_s       : {vinf_raw.tolist()} |v|={norm(vinf_raw):.6f} m/s")
    print(f"mu_m3_s2           : {mu}")
    print(f"burn offsets       : {len(offsets)}")
    print(f"arrival offsets    : {arr_offsets_days}")
    print(f"phase_top_n        : {args.phase_top_n}")
    print(f"output_dir         : {args.output_dir}")

    def finite(x: float) -> bool:
        return math.isfinite(float(x))

    with PrincipiaTargeterClient(
        args.server,
        args.plugin_b64,
        plugin_arg_mode=args.plugin_arg_mode,
        response_timeout_s=args.server_timeout_s,
        quiet_stderr=args.quiet_stderr,
    ) as client:
        # Stage 1: cheap phase/ejection scan. No VCA yet.
        for idx, off in enumerate(offsets):
            burn_dt = nominal_burn_dt + off
            try:
                st = client.vrel(
                    f"basis_{os.getpid()}_{idx}",
                    args.vessel_guid,
                    dep_body,
                    burn_dt,
                    [],
                    timeout_s=args.server_timeout_s,
                )
                r = st["final_rel_r_raw_m"]
                v = st["final_rel_v_raw_m_s"]
                ej = compute_ejection_dv(r, v, vinf_raw, float(mu))
                dv_norm = float(ej["dv_norm_m_s"])
                phase_error = float(ej["phase_error_deg"])
                phase_abs = abs(phase_error - args.phase_target_deg)
                gate_ok = (
                    args.dv_min_m_s <= dv_norm <= args.dv_max_m_s
                    and phase_abs <= args.max_abs_phase_error_deg
                )
                phase_score = phase_abs + args.phase_weight_dv * dv_norm
                row = {
                    "phase_scan_ok": True,
                    "gate_ok": bool(gate_ok),
                    "error": "",
                    "burn_offset_s": float(off),
                    "burn_dt_s": float(burn_dt),
                    "burn_abs_s": float(live_t + burn_dt),
                    "rmag_km": ej["rmag_m"] / 1000.0,
                    "vmag_m_s": ej["vmag_m_s"],
                    "vinf_mag_m_s": ej["vinf_mag_m_s"],
                    "vp_mag_m_s": ej["vp_mag_m_s"],
                    "ecc": ej["ecc"],
                    "nu_inf_deg": ej["nu_inf_deg"],
                    "theta_to_vinf_deg": ej["theta_to_vinf_deg"],
                    "phase_error_deg": ej["phase_error_deg"],
                    "phase_abs_deg": phase_abs,
                    "phase_score": phase_score,
                    "dv_norm_m_s": ej["dv_norm_m_s"],
                    "dv_raw_m_s": ej["dv_raw_m_s"],
                    "dv_levela_m_s": raw_to_levela(ej["dv_raw_m_s"]),
                    "dv_tnb_diag_m_s": ej["dv_tnb_diag_m_s"],
                    "branch_prograde_dot": ej["branch_prograde_dot"],
                }
            except Exception as exc:
                row = {
                    "phase_scan_ok": False,
                    "gate_ok": False,
                    "error": str(exc),
                    "burn_offset_s": float(off),
                    "burn_dt_s": float(burn_dt),
                    "phase_abs_deg": float("inf"),
                    "phase_score": float("inf"),
                }
            phase_rows.append(row)

        phase_candidates = [r for r in phase_rows if r.get("gate_ok") and finite(r.get("phase_score", float("nan")))]
        phase_candidates.sort(key=lambda r: (r["phase_score"], r["phase_abs_deg"], r["dv_norm_m_s"]))

        print("")
        print("=== TOP PHASE CANDIDATES BEFORE VCA ===")
        for i, r in enumerate(phase_candidates[: max(args.top_n, 20)], start=1):
            print(
                f"{i:3d} phase={r['phase_error_deg']:9.4f} deg "
                f"abs={r['phase_abs_deg']:8.4f} "
                f"burn_off={r['burn_offset_s']:9.2f}s "
                f"dv={r['dv_norm_m_s']:8.2f} "
                f"r={r['rmag_km']:8.1f} km "
                f"tnb={r['dv_tnb_diag_m_s']}"
            )

        selected = phase_candidates if args.phase_top_n <= 0 else phase_candidates[: args.phase_top_n]

        # Stage 2: VCA only for the best phase candidates.
        for cand in selected:
            for arr_off_d in arr_offsets_days:
                scan_center = nominal_arrival_dt + arr_off_d * DAY_S
                scan_start = scan_center - args.scan_half_width_days * DAY_S
                scan_end = scan_center + args.scan_half_width_days * DAY_S

                vca_ok = False
                vca_error = ""
                try:
                    vca = client.vca(
                        f"vca_{os.getpid()}_{len(rows)}",
                        args.vessel_guid,
                        arr_body,
                        scan_start,
                        scan_end,
                        args.vca_samples,
                        [(cand["burn_dt_s"], *cand["dv_raw_m_s"])],
                        timeout_s=args.server_timeout_s,
                    )
                    vca_ok = True
                except Exception as exc:
                    vca = {}
                    vca_error = str(exc)

                row = dict(cand)
                row.update({
                    "ok": bool(vca_ok),
                    "error": vca_error,
                    "arrival_offset_days": float(arr_off_d),
                    "scan_start_dt_s": float(scan_start),
                    "scan_end_dt_s": float(scan_end),
                    "ca_distance_km": float(vca.get("ca_distance_m", float("nan"))) / 1000.0 if vca_ok else float("nan"),
                    "ca_dt_s": float(vca.get("ca_dt_s", float("nan"))) if vca_ok else float("nan"),
                    "ca_t_game_s": float(vca.get("ca_t_game_s", float("nan"))) if vca_ok else float("nan"),
                    "ca_speed_m_s": float(vca.get("ca_speed_m_s", float("nan"))) if vca_ok else float("nan"),
                    "vca_status": vca.get("status", ""),
                })
                rows.append(row)

    ok_rows = [
        r for r in rows
        if r.get("ok") and math.isfinite(float(r.get("ca_distance_km", float("nan"))))
    ]
    ok_rows.sort(key=lambda r: (r["ca_distance_km"], r["phase_abs_deg"], r["dv_norm_m_s"]))

    print("")
    print("=== TOP PYKEP-ANCHORED EJECTION RESULTS ===")
    for i, r in enumerate(ok_rows[: args.top_n], start=1):
        print(
            f"{i:3d} ca={r['ca_distance_km']:12.3f} km "
            f"arr_off={r['arrival_offset_days']:6.1f} d "
            f"burn_off={r['burn_offset_s']:8.1f}s "
            f"dv={r['dv_norm_m_s']:8.2f} "
            f"phase={r['phase_error_deg']:8.3f} deg "
            f"r={r['rmag_km']:8.1f} km "
            f"tnb={r['dv_tnb_diag_m_s']}"
        )

    summary = {
        "schema": "pykep_vinf_ejection_scan_v0_1",
        "anchor_json": str(args.anchor_json),
        "leg": {
            "leg_index": args.leg,
            "dep_body": dep_body,
            "arr_body": arr_body,
            "t_dep_s": t_dep_abs,
            "t_arr_s": t_arr_abs,
            "vinf_dep_raw_m_s": vinf_raw.tolist(),
            "vinf_dep_norm_m_s": norm(vinf_raw),
        },
        "live_t_s": live_t,
        "mu_m3_s2": mu,
        "n_phase_rows": len(phase_rows),
        "n_phase_gate_ok": len([r for r in phase_rows if r.get("gate_ok")]),
        "top_phase": phase_candidates[: args.top_n],
        "n_rows": len(rows),
        "n_ok": len(ok_rows),
        "n_gate_ok": sum(1 for r in rows if r.get("gate_ok")),
        "best": ok_rows[0] if ok_rows else None,
        "top": ok_rows[: args.top_n],
    }

    json_path = args.output_dir / "pykep_vinf_ejection_scan_v0_1_result.json"
    csv_path = args.output_dir / "pykep_vinf_ejection_scan_v0_1_rows.csv"

    # Flatten vector fields for CSV.
    flat_rows = []
    for r in rows:
        rr = {}
        for k, v in r.items():
            if isinstance(v, list):
                for j, x in enumerate(v):
                    rr[f"{k}_{j}"] = x
            else:
                rr[k] = v
        flat_rows.append(rr)
    if flat_rows:
        fieldnames = sorted({k for r in flat_rows for k in r.keys()})
        with csv_path.open("w", newline="") as fp:
            w = csv.DictWriter(fp, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(flat_rows)

    phase_csv_path = args.output_dir / "pykep_vinf_ejection_phase_scan_v0_1_rows.csv"
    flat_phase = []
    for r in phase_rows:
        rr = {}
        for k, v in r.items():
            if isinstance(v, list):
                for j, x in enumerate(v):
                    rr[f"{k}_{j}"] = x
            else:
                rr[k] = v
        flat_phase.append(rr)
    if flat_phase:
        fieldnames = sorted({k for r in flat_phase for k in r.keys()})
        with phase_csv_path.open("w", newline="") as fp:
            w = csv.DictWriter(fp, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(flat_phase)

    json_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"[OK] wrote {json_path}")
    print(f"[OK] wrote {csv_path}")
    print(f"[OK] wrote {phase_csv_path}")

    if args.write_best_event and ok_rows:
        event_path = args.output_dir / "event1_pykep_vinf_burn0_inertial_levela.json"
        event = make_event(
            args.vessel_guid,
            ok_rows[0]["burn_abs_s"],
            ok_rows[0]["dv_raw_m_s"],
            event_path,
            request_id="pykep_vinf_ejection_burn0_attempt0",
        )
        print(f"[OK] wrote {event_path}")
        print(json.dumps({
            "event_initial_time": event["initial_time"],
            "event_delta_v_levela_m_s": event["delta_v_levela_m_s"],
            "event_dv_norm_m_s": norm(event["delta_v_levela_m_s"]),
        }, indent=2))

    if not ok_rows:
        print("[WARN] no VCA-ok rows. Loosen gates or inspect CSV.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
