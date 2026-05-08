from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from ksp_mga.native.leg_optimizer import (
    norm,
    norm_name,
    sample_raw_body_state,
)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def walk_json(obj):
    if isinstance(obj, dict):
        yield obj
        for v in obj.values():
            yield from walk_json(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from walk_json(v)


def get_body_record(catalog: Any, body: str) -> dict[str, Any]:
    target = norm_name(body)

    name_keys = ["name", "body", "body_name", "id", "naif_name"]
    for d in walk_json(catalog):
        names = [str(d.get(k, "")).upper() for k in name_keys]
        if target in names:
            return d

    raise KeyError(f"body {body!r} not found in body catalog")


def get_number(d: dict[str, Any], keys: list[str], default: float | None = None) -> float:
    for k in keys:
        if k in d and d[k] not in ("", None):
            return float(d[k])
    if default is not None:
        return float(default)
    raise KeyError(f"missing numeric key among {keys}")


def body_mu_km3_s2(d: dict[str, Any]) -> float:
    mu = get_number(d, [
        "mu_km3_s2",
        "gm_km3_s2",
        "GM_km3_s2",
        "gravitational_parameter_km3_s2",
        "mu",
        "gm",
        "GM",
        "gravitational_parameter",
    ])

    # Heuristic: if value is in m^3/s^2, convert to km^3/s^2.
    if mu > 1e12:
        mu /= 1e9
    return mu


def body_radius_km(d: dict[str, Any]) -> float:
    r = get_number(d, [
        "radius_km",
        "equatorial_radius_km",
        "mean_radius_km",
        "equatorial_radius",
        "mean_radius",
        "radius_m",
        "equatorial_radius_m",
        "mean_radius_m",
        "radius",
        "Radius",
        "RADIUS",
        "body_radius",
        "body_radius_km",
        "body_radius_m",
    ])

    # Heuristic: if radius is in metres, convert to km.
    if r > 1e5:
        r /= 1000.0
    return r


def body_atmosphere_km(d: dict[str, Any]) -> float:
    for k in [
        "atmosphere_depth_km",
        "atmosphere_height_km",
        "atmosphere_km",
        "atmosphere_depth_m",
        "atmosphere_height_m",
        "atmosphere_m",
        "atmosphere_depth",
        "atmosphere_height",
        "atmosphere",
        "AtmosphereDepth",
        "atmosphereDepth",
    ]:
        if k in d and d[k] not in ("", None):
            v = float(d[k])
            if v > 1e5:
                v /= 1000.0
            return max(0.0, v)
    return 0.0


def angle_between_deg(a: np.ndarray, b: np.ndarray) -> float:
    na = norm(a)
    nb = norm(b)
    if na == 0.0 or nb == 0.0:
        return float("nan")
    c = float(np.dot(a, b) / (na * nb))
    c = max(-1.0, min(1.0, c))
    return math.degrees(math.acos(c))


def max_turn_deg(mu_km3_s2: float, vinf_km_s: float, rp_min_km: float) -> float:
    if vinf_km_s <= 0.0:
        return 180.0
    e = 1.0 + rp_min_km * vinf_km_s * vinf_km_s / mu_km3_s2
    s = 1.0 / e
    s = max(0.0, min(1.0, s))
    return math.degrees(2.0 * math.asin(s))


def required_rp_km(mu_km3_s2: float, vinf_km_s: float, turn_deg: float) -> float:
    if vinf_km_s <= 0.0:
        return float("inf")
    half = math.radians(turn_deg) / 2.0
    s = math.sin(half)
    if s <= 0.0:
        return float("inf")
    return mu_km3_s2 / (vinf_km_s * vinf_km_s) * (1.0 / s - 1.0)


def load_leg_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    rows.sort(key=lambda r: int(r["leg"]))
    return rows


def vec(row: dict[str, str], keys: list[str]) -> np.ndarray:
    return np.array([float(row[k]) for k in keys], dtype=float)


def audit_flybys(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    catalog = load_json(args.body_catalog)
    legs = load_leg_rows(args.leg_optimizations)

    if len(legs) < 2:
        raise ValueError("need at least two optimized legs for flyby audit")

    rows: list[dict[str, Any]] = []

    for i in range(len(legs) - 1):
        leg_in = legs[i]
        leg_out = legs[i + 1]

        leg_in_n = int(leg_in["leg"])
        leg_out_n = int(leg_out["leg"])

        flyby_body = norm_name(leg_in["arr_body"])
        dep_next = norm_name(leg_out["dep_body"])

        if flyby_body != dep_next:
            status = "FAIL_SEQUENCE"
            row = {
                "flyby_index": i + 1,
                "body": flyby_body,
                "leg_in": leg_in_n,
                "leg_out": leg_out_n,
                "status": status,
                "message": f"leg {leg_in_n} arrives at {flyby_body}, leg {leg_out_n} departs from {dep_next}",
            }
            rows.append(row)
            continue

        t_pre = float(leg_in["t_end_s"])
        t_post = float(leg_out["t_start_s"])
        t_event_in = float(leg_in["t_arr_s"])
        t_event_out = float(leg_out["t_dep_s"])

        body_rec = get_body_record(catalog, flyby_body)
        mu = body_mu_km3_s2(body_rec)
        radius = body_radius_km(body_rec)
        atm = body_atmosphere_km(body_rec)

        min_alt = max(args.min_altitude_km, atm + args.atmosphere_margin_km)
        rp_min = radius + min_alt

        body_pre_r, body_pre_v = sample_raw_body_state(
            sampler=args.sampler,
            plugin_b64=args.plugin_b64,
            target_body=flyby_body,
            sampler_central_body=args.raw_origin_body,
            et_s=t_pre,
            plugin_base_et_s=args.plugin_base_et_s,
            work_dir=args.raw_cache_dir,
        )

        body_post_r, body_post_v = sample_raw_body_state(
            sampler=args.sampler,
            plugin_b64=args.plugin_b64,
            target_body=flyby_body,
            sampler_central_body=args.raw_origin_body,
            et_s=t_post,
            plugin_base_et_s=args.plugin_base_et_s,
            work_dir=args.raw_cache_dir,
        )

        sc_pre_r = vec(leg_in, ["final_x_raw_m", "final_y_raw_m", "final_z_raw_m"])
        sc_pre_v = vec(leg_in, ["final_vx_raw_m_s", "final_vy_raw_m_s", "final_vz_raw_m_s"])

        sc_post_r = vec(leg_out, ["start_x_raw_m", "start_y_raw_m", "start_z_raw_m"])
        sc_post_v = vec(leg_out, [
            "optimized_vx_raw_m_s",
            "optimized_vy_raw_m_s",
            "optimized_vz_raw_m_s",
        ])

        vinf_in_m_s = sc_pre_v - body_pre_v
        vinf_out_m_s = sc_post_v - body_post_v

        vinf_in = norm(vinf_in_m_s) / 1000.0
        vinf_out = norm(vinf_out_m_s) / 1000.0
        vinf_avg = 0.5 * (vinf_in + vinf_out)
        vinf_max = max(vinf_in, vinf_out)

        vinf_mismatch = abs(vinf_out - vinf_in)
        powered_dv_lower_bound = vinf_mismatch

        turn_req = angle_between_deg(vinf_in_m_s, vinf_out_m_s)
        turn_max = max_turn_deg(mu, vinf_max, rp_min)
        turn_margin = turn_max - turn_req

        rp_req = required_rp_km(mu, vinf_avg, turn_req)
        alt_req = rp_req - radius

        pre_alt_km = norm(sc_pre_r - body_pre_r) / 1000.0 - radius
        post_alt_km = norm(sc_post_r - body_post_r) / 1000.0 - radius

        if not np.isfinite(turn_req):
            status = "FAIL_NUMERIC"
        elif turn_margin < -args.turn_tolerance_deg:
            status = "FAIL_TURN"
        elif alt_req < min_alt - args.altitude_tolerance_km:
            status = "FAIL_ALTITUDE"
        elif vinf_mismatch <= args.vinf_mismatch_pass_km_s:
            status = "PASS"
        elif vinf_mismatch <= args.vinf_mismatch_powered_km_s:
            status = "POWERED"
        else:
            status = "CHECK_POWERED"

        rows.append({
            "flyby_index": i + 1,
            "body": flyby_body,
            "leg_in": leg_in_n,
            "leg_out": leg_out_n,

            "t_event_in_s": t_event_in,
            "t_event_out_s": t_event_out,
            "t_pre_s": t_pre,
            "t_post_s": t_post,
            "buffer_pre_days": (t_event_in - t_pre) / 86400.0,
            "buffer_post_days": (t_post - t_event_out) / 86400.0,

            "mu_km3_s2": mu,
            "radius_km": radius,
            "atmosphere_km": atm,
            "min_altitude_km": min_alt,
            "rp_min_km": rp_min,

            "pre_altitude_km": pre_alt_km,
            "post_altitude_km": post_alt_km,

            "vinf_in_km_s": vinf_in,
            "vinf_out_km_s": vinf_out,
            "vinf_avg_km_s": vinf_avg,
            "vinf_mismatch_km_s": vinf_mismatch,
            "powered_dv_lower_bound_km_s": powered_dv_lower_bound,

            "turn_required_deg": turn_req,
            "turn_max_deg": turn_max,
            "turn_margin_deg": turn_margin,

            "rp_required_km": rp_req,
            "alt_required_km": alt_req,

            "status": status,
        })

    total_powered = sum(float(r.get("powered_dv_lower_bound_km_s", 0.0)) for r in rows)
    max_mismatch = max(float(r.get("vinf_mismatch_km_s", 0.0)) for r in rows)
    min_turn_margin = min(float(r.get("turn_margin_deg", float("inf"))) for r in rows)

    if any(str(r["status"]).startswith("FAIL") for r in rows):
        overall = "FAIL"
    elif any(r["status"] in ("POWERED", "CHECK_POWERED") for r in rows):
        overall = "POWERED"
    else:
        overall = "PASS"

    summary = {
        "overall_status": overall,
        "n_flybys": len(rows),
        "total_powered_dv_lower_bound_km_s": total_powered,
        "max_vinf_mismatch_km_s": max_mismatch,
        "min_turn_margin_deg": min_turn_margin,
    }

    return rows, summary


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    fields: list[str] = []
    for row in rows:
        for k in row:
            if k not in fields:
                fields.append(k)

    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main_cli() -> int:
    p = argparse.ArgumentParser(description="Audit corrected N-body legs for flyby feasibility.")
    p.add_argument("--leg-optimizations", type=Path, required=True)
    p.add_argument("--body-catalog", type=Path, required=True)

    p.add_argument("--plugin-b64", type=Path, required=True)
    p.add_argument("--sampler", default="sample_principia_ephemeris")
    p.add_argument("--plugin-base-et-s", type=float, default=81.65168640136972)
    p.add_argument("--raw-origin-body", default="Sun")
    p.add_argument("--raw-cache-dir", type=Path, default=Path("data/runs/frame_debug/raw_cache"))

    p.add_argument("--min-altitude-km", type=float, default=50.0)
    p.add_argument("--atmosphere-margin-km", type=float, default=10.0)
    p.add_argument("--altitude-tolerance-km", type=float, default=1e-3)
    p.add_argument("--turn-tolerance-deg", type=float, default=1e-6)

    p.add_argument("--vinf-mismatch-pass-km-s", type=float, default=0.05)
    p.add_argument("--vinf-mismatch-powered-km-s", type=float, default=2.0)

    p.add_argument("--output-csv", type=Path, required=True)
    p.add_argument("--output-json", type=Path, required=True)

    args = p.parse_args()

    rows, summary = audit_flybys(args)

    write_csv(args.output_csv, rows)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps({
        "summary": summary,
        "flybys": rows,
    }, indent=2))

    print("=== CORRECTED FLYBY AUDIT ===")
    for r in rows:
        print(
            f"{r['body']:<8} "
            f"vinf_in={r['vinf_in_km_s']:8.4f} km/s "
            f"vinf_out={r['vinf_out_km_s']:8.4f} km/s "
            f"mismatch={r['vinf_mismatch_km_s']:8.4f} km/s "
            f"turn={r['turn_required_deg']:8.3f}/{r['turn_max_deg']:8.3f} deg "
            f"margin={r['turn_margin_deg']:8.3f} deg "
            f"alt_req={r['alt_required_km']:10.3f} km "
            f"{r['status']}"
        )

    print("")
    print("=== SUMMARY ===")
    for k, v in summary.items():
        print(f"{k:40}: {v}")

    print(f"[OK] CSV : {args.output_csv}")
    print(f"[OK] JSON: {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main_cli())
