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


def angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = norm(a), norm(b)
    if na <= 0 or nb <= 0:
        return math.nan
    c = float(np.dot(a, b) / (na * nb))
    c = max(-1.0, min(1.0, c))
    return math.degrees(math.acos(c))


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


def maybe_get(d: dict[str, Any], keys: list[str]) -> Any:
    for k in keys:
        if k in d and d[k] not in (None, ""):
            return d[k]
    return None


def find_body_record(obj: Any, name: str) -> dict[str, Any] | None:
    lname = name.lower()
    if isinstance(obj, dict):
        # Direct keyed catalog: {"EVE": {...}}
        for k, v in obj.items():
            if str(k).lower() == lname and isinstance(v, dict):
                return v
        # Record with name field.
        nm = obj.get("name") or obj.get("body") or obj.get("id") or obj.get("spice_name")
        if nm is not None and str(nm).lower() == lname:
            return obj
        # Common containers.
        for key in ("bodies", "items", "catalog", "body_catalog"):
            if key in obj:
                rec = find_body_record(obj[key], name)
                if rec is not None:
                    return rec
        for v in obj.values():
            rec = find_body_record(v, name)
            if rec is not None:
                return rec
    elif isinstance(obj, list):
        for v in obj:
            rec = find_body_record(v, name)
            if rec is not None:
                return rec
    return None


def load_radius_km(body: str, body_catalog: Path | None, override: float | None) -> float:
    if override is not None:
        return float(override)
    if body_catalog is None:
        raise SystemExit("[FAIL] pass --body-radius-km or --body-catalog")
    obj = json.loads(body_catalog.read_text())
    rec = find_body_record(obj, body)
    if rec is None:
        raise SystemExit(f"[FAIL] body {body!r} not found in {body_catalog}; pass --body-radius-km")
    # Accept several possible schemas.
    km = maybe_get(rec, [
        "radius_km", "mean_radius_km", "equatorial_radius_km", "body_radius_km",
        "radius", "mean_radius", "equatorial_radius",
    ])
    if km is not None:
        km = float(km)
        # Heuristic: if field was radius in metres, convert. Most KSP catalogs use km/m clearly.
        if km > 1.0e5:
            return km / 1000.0
        return km
    m = maybe_get(rec, ["radius_m", "mean_radius_m", "equatorial_radius_m"])
    if m is not None:
        return float(m) / 1000.0
    raise SystemExit(f"[FAIL] could not infer radius for {body!r} from {body_catalog}; pass --body-radius-km")


def read_leg_row(path: Path, leg: int) -> dict[str, str]:
    rows = list(csv.DictReader(path.open()))
    for row in rows:
        if int(float(row["leg"])) == leg:
            return row
    raise SystemExit(f"[FAIL] leg {leg} not found in {path}")


def arr(row: dict[str, str], *names: str) -> np.ndarray:
    return np.array([float(row[n]) for n in names], dtype=float)


def read_rerender_rows(path: Path, max_rows: int | None, only_valid: bool) -> list[dict[str, Any]]:
    obj = json.loads(path.read_text())
    if isinstance(obj, dict):
        if "best_valid" in obj and obj.get("best_valid") and max_rows == 1:
            rows = [obj["best_valid"]]
        elif "best" in obj and obj.get("best") and max_rows == 1:
            rows = [obj["best"]]
        elif "rows" in obj:
            rows = obj["rows"]
        else:
            # leg1_rerender_top.json is normally a list; result.json has best/best_valid.
            rows = []
            if obj.get("best_valid"):
                rows.append(obj["best_valid"])
            if obj.get("best") and obj.get("best") != obj.get("best_valid"):
                rows.append(obj["best"])
    elif isinstance(obj, list):
        rows = obj
    else:
        raise SystemExit(f"[FAIL] unsupported rerender schema in {path}")
    rows = [r for r in rows if isinstance(r, dict) and r.get("status", "OK") == "OK"]
    if only_valid:
        rows = [r for r in rows if bool(r.get("solution_valid"))]
    rows.sort(key=lambda r: (
        not bool(r.get("solution_valid")),
        safe_float(r.get("final_pos_err_km"), math.inf),
        safe_float(r.get("dsm_norm_m_s"), math.inf),
    ))
    if max_rows is not None:
        rows = rows[:max_rows]
    return rows


def max_turn_deg(vinf_m_s: float, rp_km: float, mu_m3_s2: float) -> float:
    rp_m = rp_km * 1000.0
    if vinf_m_s <= 0 or rp_m <= 0:
        return math.nan
    arg = 1.0 / (rp_m * vinf_m_s * vinf_m_s / mu_m3_s2 + 1.0)
    arg = max(-1.0, min(1.0, arg))
    return math.degrees(2.0 * math.asin(arg))


def required_rp_km(turn_deg: float, vinf_m_s: float, mu_m3_s2: float) -> float:
    if not math.isfinite(turn_deg) or turn_deg <= 0 or vinf_m_s <= 0:
        return math.inf
    s = math.sin(math.radians(turn_deg) / 2.0)
    if s <= 0:
        return math.inf
    rp_m = mu_m3_s2 / (vinf_m_s * vinf_m_s) * (1.0 / s - 1.0)
    return rp_m / 1000.0


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    flat_rows: list[dict[str, Any]] = []
    for r in rows:
        rr = {}
        for k, v in r.items():
            if isinstance(v, (list, dict)):
                rr[k] = json.dumps(v, separators=(",", ":"))
            else:
                rr[k] = v
            if k not in fields:
                fields.append(k)
        flat_rows.append(rr)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(flat_rows)


def main() -> int:
    ap = argparse.ArgumentParser(description="Audit whether rerendered Kerbin->Eve arrival can hand off to the old/new leg2 via an Eve flyby.")
    ap.add_argument("--rerender-json", type=Path, required=True, help="leg1_rerender_top.json or leg1_rerender_result.json")
    ap.add_argument("--leg-optimizations", type=Path, required=True)
    ap.add_argument("--leg-out", type=int, default=2, help="Outgoing leg after the flyby; default leg2")
    ap.add_argument("--bsp", type=Path, required=True)
    ap.add_argument("--tpc", type=Path, required=True)
    ap.add_argument("--body", default="EVE")
    ap.add_argument("--center", default="SUN")
    ap.add_argument("--frame", default="J2000")
    ap.add_argument("--body-catalog", type=Path, default=None)
    ap.add_argument("--body-radius-km", type=float, default=None)
    ap.add_argument("--min-altitude-km", type=float, default=50.0)
    ap.add_argument("--atmosphere-margin-km", type=float, default=10.0)
    ap.add_argument("--outgoing-mode", choices=["pre_correction", "post_correction"], default="post_correction")
    ap.add_argument("--max-rows", type=int, default=50)
    ap.add_argument("--only-valid", action="store_true", default=True)
    ap.add_argument("--vinf-mismatch-pass-m-s", type=float, default=100.0)
    ap.add_argument("--vinf-mismatch-powered-m-s", type=float, default=2000.0)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()

    spice.kclear(); spice.furnsh(str(args.tpc)); spice.furnsh(str(args.bsp))
    mu = body_mu_m3_s2(args.body)
    radius_km = load_radius_km(args.body, args.body_catalog, args.body_radius_km)
    safe_alt_km = args.min_altitude_km + args.atmosphere_margin_km
    rp_min_km = radius_km + safe_alt_km

    rows = read_rerender_rows(args.rerender_json, args.max_rows, args.only_valid)
    leg2 = read_leg_row(args.leg_optimizations, args.leg_out)
    t_ref = float(leg2.get("t_start_s") or leg2.get("t_dep_s"))
    v_start = arr(leg2, "start_vx_raw_m_s", "start_vy_raw_m_s", "start_vz_raw_m_s")
    dv_corr = np.zeros(3)
    if args.outgoing_mode == "post_correction" and "dvx_m_s" in leg2:
        dv_corr = arr(leg2, "dvx_m_s", "dvy_m_s", "dvz_m_s")
    body_r_ref, body_v_ref = body_state_raw(args.body, t_ref, args.center, args.frame)
    vinf_out_ref = v_start + dv_corr - body_v_ref
    vinf_out_ref_mag = norm(vinf_out_ref)

    out_rows: list[dict[str, Any]] = []
    for i, r in enumerate(rows):
        dep_id = str(r.get("departure_id", f"row_{i:05d}"))
        t_arr = float(r["t_arr_s"])
        final_v = np.array(r["final_v_raw_m_s"], dtype=float)
        # Prefer stored target_v from rerender script; it is the body's velocity at t_arr.
        if "target_v_raw_m_s" in r:
            body_v_arr = np.array(r["target_v_raw_m_s"], dtype=float)
        else:
            _, body_v_arr = body_state_raw(args.body, t_arr, args.center, args.frame)
        vinf_in = final_v - body_v_arr
        vinf_in_mag = norm(vinf_in)
        turn_req = angle_deg(vinf_in, vinf_out_ref)
        vinf_mag_mismatch = abs(vinf_out_ref_mag - vinf_in_mag)
        vinf_vec_mismatch = norm(vinf_out_ref - vinf_in)
        # Conservative turn uses larger speed because it permits less turning.
        vinf_turn = max(vinf_in_mag, vinf_out_ref_mag)
        rp_req = required_rp_km(turn_req, vinf_turn, mu)
        alt_req = rp_req - radius_km
        turn_max = max_turn_deg(vinf_turn, rp_min_km, mu)
        turn_margin = turn_max - turn_req
        powered_lower_bound = vinf_mag_mismatch

        status = "PASS"
        reasons: list[str] = []
        if turn_margin < 0:
            status = "FAIL_TURN"
            reasons.append("turn_exceeds_safe_altitude")
        if vinf_mag_mismatch > args.vinf_mismatch_pass_m_s:
            if vinf_mag_mismatch <= args.vinf_mismatch_powered_m_s and status == "PASS":
                status = "POWERED"
                reasons.append("vinf_magnitude_mismatch_powered")
            elif vinf_mag_mismatch > args.vinf_mismatch_powered_m_s:
                status = "FAIL_VINF"
                reasons.append("vinf_magnitude_mismatch_too_large")
        out_rows.append({
            "row_index": i,
            "departure_id": dep_id,
            "status": status,
            "reasons": reasons,
            "t_arr_s": t_arr,
            "arrival_offset_from_leg2_start_days": (t_arr - t_ref) / 86400.0,
            "leg2_reference_t_start_s": t_ref,
            "outgoing_mode": args.outgoing_mode,
            "leg1_pos_err_km": safe_float(r.get("final_pos_err_km")),
            "leg1_dsm_norm_m_s": safe_float(r.get("dsm_norm_m_s")),
            "leg1_dv0_norm_m_s": safe_float(r.get("dv0_norm_m_s")),
            "vinf_in_m_s": vinf_in_mag,
            "vinf_out_ref_m_s": vinf_out_ref_mag,
            "vinf_mag_mismatch_m_s": vinf_mag_mismatch,
            "vinf_vec_mismatch_m_s": vinf_vec_mismatch,
            "turn_required_deg": turn_req,
            "turn_max_at_safe_alt_deg": turn_max,
            "turn_margin_deg": turn_margin,
            "required_rp_km": rp_req,
            "required_altitude_km": alt_req,
            "safe_altitude_km": safe_alt_km,
            "radius_km": radius_km,
            "powered_lower_bound_m_s": powered_lower_bound,
            "vinf_in_raw_m_s": vinf_in.tolist(),
            "vinf_out_ref_raw_m_s": vinf_out_ref.tolist(),
            "source_rerender": r,
        })

    status_rank = {"PASS": 0, "POWERED": 1, "FAIL_TURN": 2, "FAIL_VINF": 3}
    out_rows.sort(key=lambda x: (
        status_rank.get(str(x.get("status")), 9),
        safe_float(x.get("powered_lower_bound_m_s"), math.inf),
        -safe_float(x.get("turn_margin_deg"), -math.inf),
        safe_float(x.get("leg1_pos_err_km"), math.inf),
    ))
    summary = {
        "schema": "rerendered_eve_handoff_audit.v1",
        "body": args.body,
        "radius_km": radius_km,
        "mu_m3_s2": mu,
        "safe_altitude_km": safe_alt_km,
        "rp_min_km": rp_min_km,
        "outgoing_mode": args.outgoing_mode,
        "n_rows": len(out_rows),
        "status_counts": {s: sum(1 for r in out_rows if r.get("status") == s) for s in sorted(set(str(r.get("status")) for r in out_rows))},
        "best": out_rows[0] if out_rows else None,
        "rows": out_rows,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "eve_handoff_audit.json").write_text(json.dumps(summary, indent=2) + "\n")
    write_csv(args.output_dir / "eve_handoff_audit.csv", out_rows)

    print("=== RERENDERED EVE HANDOFF AUDIT ===")
    print(f"rows       : {len(out_rows)}")
    print(f"body       : {args.body} radius={radius_km:.3f} km safe_alt={safe_alt_km:.3f} km")
    print(f"leg2 mode  : {args.outgoing_mode} t_ref={t_ref}")
    print("rank status    dep        arr_off_d vin_in vin_out mag_mis turn_req turn_margin alt_req powered_lb pos_km dsm")
    for i, r in enumerate(out_rows[:20], start=1):
        print(
            f"{i:>3} {r['status']:<9} {r['departure_id']:<10} "
            f"{r['arrival_offset_from_leg2_start_days']:9.3f} "
            f"{r['vinf_in_m_s']/1000:6.3f} {r['vinf_out_ref_m_s']/1000:7.3f} "
            f"{r['vinf_mag_mismatch_m_s']:7.1f} "
            f"{r['turn_required_deg']:8.3f} {r['turn_margin_deg']:10.3f} "
            f"{r['required_altitude_km']:8.1f} {r['powered_lower_bound_m_s']:10.1f} "
            f"{r['leg1_pos_err_km']:8.1f} {r['leg1_dsm_norm_m_s']:7.1f}"
        )
    print(f"[OK] wrote {args.output_dir / 'eve_handoff_audit.json'}")
    print(f"[OK] wrote {args.output_dir / 'eve_handoff_audit.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
