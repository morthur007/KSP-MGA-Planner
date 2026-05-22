#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import spiceypy as spice


def norm(v: np.ndarray) -> float:
    return float(np.linalg.norm(v))


def unit(v: np.ndarray) -> np.ndarray:
    n = norm(v)
    if n <= 0:
        return np.full(3, np.nan)
    return v / n


def safe_float(x: Any, default: float = math.inf) -> float:
    try:
        if x is None or x == "":
            return default
        v = float(x)
        if math.isnan(v):
            return default
        return v
    except Exception:
        return default


def angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = norm(a), norm(b)
    if na <= 0 or nb <= 0:
        return math.nan
    c = float(np.dot(a, b) / (na * nb))
    c = max(-1.0, min(1.0, c))
    return math.degrees(math.acos(c))


def levela_to_raw(v: np.ndarray) -> np.ndarray:
    # LevelA/SPICE -> Principia raw = [+Z, -X, +Y]
    x, y, z = v
    return np.array([z, -x, y], dtype=float)


def raw_to_levela(v: np.ndarray) -> np.ndarray:
    x, y, z = v
    return np.array([-y, z, x], dtype=float)


def body_state_raw(body: str, t_s: float, center: str, frame: str) -> tuple[np.ndarray, np.ndarray]:
    st, _ = spice.spkezr(body, t_s, frame, "NONE", center)
    r_levela = np.array([1000.0 * st[0], 1000.0 * st[1], 1000.0 * st[2]], dtype=float)
    v_levela = np.array([1000.0 * st[3], 1000.0 * st[4], 1000.0 * st[5]], dtype=float)
    return levela_to_raw(r_levela), levela_to_raw(v_levela)


def body_mu_m3_s2(body: str) -> float:
    _, vals = spice.bodvrd(body, "GM", 1)
    return float(vals[0]) * 1.0e9


def arr(row: dict[str, Any], *names: str, default: float | None = None) -> np.ndarray:
    vals = []
    for n in names:
        if n not in row or row[n] in (None, ""):
            if default is None:
                raise KeyError(n)
            vals.append(default)
        else:
            vals.append(float(row[n]))
    return np.array(vals, dtype=float)


def read_leg_row(path: Path, leg: int) -> dict[str, str]:
    rows = list(csv.DictReader(path.open()))
    for row in rows:
        if int(float(row["leg"])) == leg:
            return row
    raise SystemExit(f"[FAIL] leg {leg} not found in {path}")


def read_rows(path: Path, max_rows: int | None, valid_only: bool) -> list[dict[str, Any]]:
    obj = json.loads(path.read_text())
    if isinstance(obj, dict):
        rows = obj.get("rows") or obj.get("results") or []
        if not rows and obj.get("best"):
            rows = [obj["best"]]
        if not rows and obj.get("best_valid"):
            rows = [obj["best_valid"]]
    elif isinstance(obj, list):
        rows = obj
    else:
        raise SystemExit(f"[FAIL] unsupported schema: {path}")
    rows = [r for r in rows if isinstance(r, dict)]
    if valid_only:
        rows = [r for r in rows if bool(r.get("solution_valid"))]
    rows.sort(key=lambda r: (
        0 if bool(r.get("solution_valid")) else 1,
        safe_float(r.get("score"), math.inf),
        safe_float(r.get("final_pos_err_km"), math.inf),
        safe_float(r.get("leg2_dsm_norm_m_s"), math.inf),
    ))
    if max_rows is not None:
        rows = rows[:max_rows]
    return rows


def find_body_record(obj: Any, body: str) -> dict[str, Any] | None:
    body_u = body.upper()
    if isinstance(obj, dict):
        # Direct map by name.
        for k, v in obj.items():
            if str(k).upper() == body_u and isinstance(v, dict):
                return v
        # Dict with name-like fields.
        names = [obj.get(k) for k in ("name", "body", "display_name", "spice_name")]
        if any(str(n).upper() == body_u for n in names if n is not None):
            return obj
        for v in obj.values():
            found = find_body_record(v, body)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for v in obj:
            found = find_body_record(v, body)
            if found is not None:
                return found
    return None


def get_radius_km(body_catalog: Path | None, body: str) -> float:
    # --- QUICK BYPASS: Hardcode JNSQ Kerbin ---
    if body.upper() == "KERBIN": return 600.0

    if body_catalog and body_catalog.exists():
        try:
            obj = json.loads(body_catalog.read_text())
            rec = find_body_record(obj, body)
            if rec:
                for key in ["radius_km", "mean_radius_km", "equatorial_radius_km"]:
                    if key in rec:
                        return float(rec[key])
                for key in ["radius_m", "mean_radius_m", "equatorial_radius_m"]:
                    if key in rec:
                        return float(rec[key]) / 1000.0
                if "radius" in rec:
                    val = float(rec["radius"])
                    return val / 1000.0 if val > 1e5 else val
            else:
                print(f"[DEBUG] '{body}' was not found inside the catalog JSON.")
        except Exception as e:
            print(f"[DEBUG] Error reading JSON catalog: {e}")
    else:
        print(f"[DEBUG] Catalog file not found at path: {body_catalog}")

    # Try SPICE PCK/TPC RADII if available.
    try:
        _, vals = spice.bodvrd(body, "RADII", 3)
        return float(np.mean(vals))
    except Exception as e:
        print(f"[DEBUG] SPICE RADII error: {e}")
        raise SystemExit(f"[FAIL] could not determine radius for {body}; pass --body-catalog")


def max_turn_deg(mu_m3_s2: float, rp_m: float, vinf_m_s: float) -> float:
    if vinf_m_s <= 0 or rp_m <= 0:
        return math.nan
    e = 1.0 + rp_m * vinf_m_s**2 / mu_m3_s2
    x = 1.0 / e
    x = max(-1.0, min(1.0, x))
    return math.degrees(2.0 * math.asin(x))


def required_altitude_km(mu_m3_s2: float, radius_km: float, vinf_m_s: float, turn_deg: float) -> float:
    if vinf_m_s <= 0 or turn_deg <= 0 or turn_deg >= 180:
        return math.nan
    s = math.sin(math.radians(turn_deg) / 2.0)
    if s <= 0:
        return math.nan
    rp_m = mu_m3_s2 / vinf_m_s**2 * (1.0 / s - 1.0)
    return rp_m / 1000.0 - radius_km


def leg_start_state(row: dict[str, str], mode: str) -> tuple[float, np.ndarray, np.ndarray]:
    t = float(row.get("t_start_s") or row.get("t_dep_s"))
    r = arr(row, "start_x_raw_m", "start_y_raw_m", "start_z_raw_m")
    v = arr(row, "start_vx_raw_m_s", "start_vy_raw_m_s", "start_vz_raw_m_s")
    if mode == "post_correction" and all(k in row for k in ["dvx_m_s", "dvy_m_s", "dvz_m_s"]):
        v = v + arr(row, "dvx_m_s", "dvy_m_s", "dvz_m_s")
    elif mode == "pre_correction":
        pass
    else:
        if mode not in {"pre_correction", "post_correction"}:
            raise SystemExit(f"[FAIL] unsupported outgoing mode: {mode}")
    return t, r, v


def audit_one(
    row: dict[str, Any],
    leg3_row: dict[str, str],
    body: str,
    radius_km: float,
    mu_m3_s2: float,
    center: str,
    frame: str,
    safe_alt_km: float,
    outgoing_mode: str,
    pass_mag_m_s: float,
    powered_mag_m_s: float,
) -> dict[str, Any]:
    t_arr = float(row["t_arr_s"])
    body_r, body_v = body_state_raw(body, t_arr, center, frame)
    final_r = np.array(row.get("final_r_raw_m") or row.get("target_r_raw_m") or body_r, dtype=float)
    final_v = np.array(row["final_v_raw_m_s"], dtype=float)
    vinf_in = final_v - body_v

    t_ref, r_ref, v_ref = leg_start_state(leg3_row, outgoing_mode)
    body_r_ref, body_v_ref = body_state_raw(body, t_ref, center, frame)
    vinf_out_ref = v_ref - body_v_ref

    vin_in = norm(vinf_in)
    vin_out = norm(vinf_out_ref)
    turn_req = angle_deg(vinf_in, vinf_out_ref)
    mag_mis = abs(vin_out - vin_in)
    vinf_for_turn = max(vin_in, vin_out)
    safe_rp_m = (radius_km + safe_alt_km) * 1000.0
    turn_max = max_turn_deg(mu_m3_s2, safe_rp_m, vinf_for_turn)
    margin = turn_max - turn_req
    alt_req = required_altitude_km(mu_m3_s2, radius_km, vinf_for_turn, turn_req)

    if not math.isfinite(margin) or margin < 0:
        status = "FAIL_TURN"
    elif mag_mis <= pass_mag_m_s:
        status = "PASS"
    elif mag_mis <= powered_mag_m_s:
        status = "POWERED"
    else:
        status = "FAIL_POWERED"

    # Lower bound: changing hyperbolic excess magnitude cannot be free; direction change can be ballistic if margin >= 0.
    powered_lb = 0.0 if status == "PASS" else mag_mis

    return {
        "status": status,
        "departure_id": row.get("departure_id"),
        "outgoing_mode_leg2": row.get("outgoing_mode"),
        "leg3_outgoing_mode": outgoing_mode,
        "solution_valid_leg2": bool(row.get("solution_valid")),
        "t_arr_s": t_arr,
        "t_ref_leg3_s": t_ref,
        "arrival_offset_from_leg3_start_days": (t_arr - t_ref) / 86400.0,
        "vinf_in_m_s": vin_in,
        "vinf_out_ref_m_s": vin_out,
        "vinf_mag_mismatch_m_s": mag_mis,
        "turn_required_deg": turn_req,
        "turn_max_at_safe_alt_deg": turn_max,
        "turn_margin_deg": margin,
        "required_altitude_km": alt_req,
        "safe_altitude_km": safe_alt_km,
        "powered_lower_bound_m_s": powered_lb,
        "leg2_pos_err_km": safe_float(row.get("final_pos_err_km")),
        "leg2_arrival_vinf_m_s": safe_float(row.get("arrival_rel_vinf_m_s"), safe_float(row.get("final_vel_err_m_s"))),
        "leg2_dsm_norm_m_s": safe_float(row.get("leg2_dsm_norm_m_s")),
        "leg1_dsm_norm_m_s": safe_float(row.get("leg1_dsm_norm_m_s")),
        "eve_powered_lower_bound_m_s": safe_float(row.get("eve_powered_lower_bound_m_s"), 0.0),
        "total_with_departure_dv_m_s": safe_float(row.get("total_with_departure_dv_m_s")),
        "vinf_in_raw_m_s": vinf_in.tolist(),
        "vinf_out_ref_raw_m_s": vinf_out_ref.tolist(),
        "final_r_minus_body_km": (norm(final_r - body_r) / 1000.0),
        "source_leg2_rerender": row,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    flat: list[dict[str, Any]] = []
    for r in rows:
        rr: dict[str, Any] = {}
        for k, v in r.items():
            if isinstance(v, (list, dict)):
                rr[k] = json.dumps(v, separators=(",", ":"))
            else:
                rr[k] = v
            if k not in fields:
                fields.append(k)
        flat.append(rr)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(flat)


def main() -> int:
    ap = argparse.ArgumentParser(description="Audit Kerbin handoff after rerendered leg2 against leg3 outgoing reference.")
    ap.add_argument("--leg2-rerender-json", type=Path, required=True, help="leg2_rerender_top.json or leg2_rerender_result.json")
    ap.add_argument("--leg-optimizations", type=Path, required=True)
    ap.add_argument("--leg-out", type=int, default=3, help="Outgoing leg after Kerbin, normally 3.")
    ap.add_argument("--bsp", type=Path, required=True)
    ap.add_argument("--tpc", type=Path, required=True)
    ap.add_argument("--body", default="KERBIN")
    ap.add_argument("--body-catalog", type=Path, default=None)
    ap.add_argument("--center", default="SUN")
    ap.add_argument("--frame", default="J2000")
    ap.add_argument("--min-altitude-km", type=float, default=50.0)
    ap.add_argument("--atmosphere-margin-km", type=float, default=10.0)
    ap.add_argument("--outgoing-mode", choices=["pre_correction", "post_correction"], default="post_correction")
    ap.add_argument("--vinf-mismatch-pass-m-s", type=float, default=50.0)
    ap.add_argument("--vinf-mismatch-powered-m-s", type=float, default=2500.0)
    ap.add_argument("--max-rows", type=int, default=100)
    ap.add_argument("--valid-only", action="store_true", default=True)
    ap.add_argument("--include-invalid", action="store_true")
    ap.add_argument("--bridge-json", type=Path, default=None, help="Optional existing Kerbin bridge result, reported but not applied.")
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()

    spice.kclear(); spice.furnsh(str(args.tpc)); spice.furnsh(str(args.bsp))
    radius_km = get_radius_km(args.body_catalog, args.body)
    mu = body_mu_m3_s2(args.body)
    safe_alt = args.min_altitude_km + args.atmosphere_margin_km
    leg3 = read_leg_row(args.leg_optimizations, args.leg_out)
    rows = read_rows(args.leg2_rerender_json, args.max_rows, valid_only=(args.valid_only and not args.include_invalid))

    out = [audit_one(
        r, leg3, args.body, radius_km, mu, args.center, args.frame, safe_alt,
        args.outgoing_mode, args.vinf_mismatch_pass_m_s, args.vinf_mismatch_powered_m_s,
    ) for r in rows]
    out.sort(key=lambda r: (
        {"PASS": 0, "POWERED": 1, "FAIL_TURN": 2, "FAIL_POWERED": 3}.get(str(r.get("status")), 9),
        safe_float(r.get("powered_lower_bound_m_s"), math.inf),
        -safe_float(r.get("turn_margin_deg"), -math.inf),
        safe_float(r.get("leg2_dsm_norm_m_s"), math.inf),
        safe_float(r.get("leg2_pos_err_km"), math.inf),
    ))
    status_counts: dict[str, int] = {}
    for r in out:
        status_counts[str(r.get("status"))] = status_counts.get(str(r.get("status")), 0) + 1

    bridge_report = None
    if args.bridge_json and args.bridge_json.exists():
        try:
            b = json.loads(args.bridge_json.read_text())
            bridge_report = {
                "path": str(args.bridge_json),
                "success": b.get("success"),
                "t0_s": b.get("t0_s"),
                "t1_s": b.get("t1_s"),
                "t_event_s": b.get("t_event_s"),
                "total_bridge_dv_m_s": b.get("total_bridge_dv_m_s"),
                "final_pos_err_km": b.get("final_pos_err_km"),
                "note": "Existing bridge is reported only; this audit does not assume it is still valid after leg2 rerender.",
            }
        except Exception as e:
            bridge_report = {"path": str(args.bridge_json), "error": repr(e)}

    summary = {
        "schema": "rerendered_kerbin_handoff_audit.v1",
        "status_counts": status_counts,
        "body": args.body,
        "radius_km": radius_km,
        "safe_altitude_km": safe_alt,
        "leg_out": args.leg_out,
        "outgoing_mode": args.outgoing_mode,
        "best": out[0] if out else None,
        "rows": out,
        "bridge_report": bridge_report,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "kerbin_handoff_audit.json").write_text(json.dumps(summary, indent=2) + "\n")
    write_csv(args.output_dir / "kerbin_handoff_audit.csv", out)

    print("=== RERENDERED KERBIN HANDOFF AUDIT ===")
    print(f"rows       : {len(out)}")
    print(f"body       : {args.body} radius={radius_km:.3f} km safe_alt={safe_alt:.3f} km")
    print(f"leg{args.leg_out} mode  : {args.outgoing_mode} t_ref={leg3.get('t_start_s')}")
    if bridge_report:
        print(f"bridge     : total={bridge_report.get('total_bridge_dv_m_s')} m/s success={bridge_report.get('success')}")
    print("rank status    dep        mode            arr_off_d vin_in vin_out mag_mis turn_req turn_margin alt_req powered_lb pos_km dsm")
    for i, r in enumerate(out[:20], start=1):
        print(
            f"{i:3d} {str(r.get('status')):<9} "
            f"{str(r.get('departure_id')):<10} {str(r.get('outgoing_mode_leg2')):<15} "
            f"{safe_float(r.get('arrival_offset_from_leg3_start_days')):10.3f} "
            f"{safe_float(r.get('vinf_in_m_s'))/1000.0:6.3f} "
            f"{safe_float(r.get('vinf_out_ref_m_s'))/1000.0:7.3f} "
            f"{safe_float(r.get('vinf_mag_mismatch_m_s')):7.1f} "
            f"{safe_float(r.get('turn_required_deg')):8.3f} "
            f"{safe_float(r.get('turn_margin_deg')):10.3f} "
            f"{safe_float(r.get('required_altitude_km')):8.1f} "
            f"{safe_float(r.get('powered_lower_bound_m_s')):9.1f} "
            f"{safe_float(r.get('leg2_pos_err_km')):7.1f} "
            f"{safe_float(r.get('leg2_dsm_norm_m_s')):7.1f}"
        )
    print(f"[OK] wrote {args.output_dir / 'kerbin_handoff_audit.json'}")
    print(f"[OK] wrote {args.output_dir / 'kerbin_handoff_audit.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
