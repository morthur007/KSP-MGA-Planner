#!/usr/bin/env python3
"""
audit_vcarel_dsm_handoff_v0.py

Audit DSM-refined VCAREL flyby candidates against the next PyKEP leg.

This script does NOT optimize. It classifies each VCAREL row by:
  - impact / safe periapsis altitude;
  - inbound hyperbolic energy at closest approach;
  - natural outgoing v∞ from the osculating flyby at CA;
  - mismatch against the next leg's required departure v∞.

It is intended to run after refine_ranked_candidate_vcarel_dsm_v0.py.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np


def norm(v: Sequence[float]) -> float:
    return float(np.linalg.norm(np.asarray(v, dtype=float)))


def unit(v: Sequence[float]) -> np.ndarray:
    a = np.asarray(v, dtype=float)
    n = norm(a)
    if n <= 0 or not math.isfinite(n):
        raise ValueError(f"cannot normalize {v}")
    return a / n


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def levela_to_raw(v: Sequence[float]) -> list[float]:
    x, y, z = map(float, v)
    return [z, -x, y]


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


def body_radius_mu(path: Path, body: str) -> tuple[float, float, dict[str, Any]]:
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
        raise RuntimeError(f"radius not found for {body}; record keys={sorted(rec.keys())}")

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
        raise RuntimeError(f"mu/gm not found for {body}; record keys={sorted(rec.keys())}")

    return radius, mu, rec


def load_anchor_leg(anchor_json: Path, leg_out: int) -> dict[str, Any]:
    data = json.loads(anchor_json.read_text())
    legs = data.get("legs")
    if not legs:
        # Fallback for old anchor packet with leg1, leg2 keys.
        key = f"leg{leg_out}"
        if key in data:
            return data[key]
        raise RuntimeError(f"no legs array and no leg{leg_out} in {anchor_json}")
    idx = leg_out - 1
    if idx < 0 or idx >= len(legs):
        raise RuntimeError(f"leg_out {leg_out} out of range; anchor has {len(legs)} legs")
    return legs[idx]


def leg_vinf_dep_raw_m_s(leg: dict[str, Any]) -> np.ndarray:
    if "vinf_dep_raw_m_s" in leg:
        return np.asarray(leg["vinf_dep_raw_m_s"], dtype=float)
    if "vinf_dep_levela_km_s" in leg:
        return np.asarray(levela_to_raw([1000.0*x for x in leg["vinf_dep_levela_km_s"]]), dtype=float)
    if "vinf_dep_levela_m_s" in leg:
        return np.asarray(levela_to_raw(leg["vinf_dep_levela_m_s"]), dtype=float)
    raise RuntimeError(f"cannot find vinf_dep vector in leg keys={sorted(leg.keys())}")


def asymptotes_from_ca(r_m: Sequence[float], v_m_s: Sequence[float], mu: float) -> dict[str, Any]:
    r = np.asarray(r_m, dtype=float)
    v = np.asarray(v_m_s, dtype=float)
    rp = norm(r)
    vp = norm(v)

    eps = 0.5 * vp * vp - mu / rp
    if eps <= 0:
        return {
            "hyperbolic": False,
            "specific_energy_m2_s2": eps,
            "vinf_mag_m_s": math.nan,
        }

    vinf = math.sqrt(2.0 * eps)
    e = 1.0 + rp * vinf * vinf / mu
    if e <= 1.0:
        return {
            "hyperbolic": False,
            "specific_energy_m2_s2": eps,
            "vinf_mag_m_s": vinf,
            "eccentricity": e,
        }

    rhat = unit(r)
    # Use the transverse direction at closest approach as the periapsis tangent.
    vt = v - float(np.dot(v, rhat)) * rhat
    if norm(vt) < 1e-9:
        that = unit(v)
    else:
        that = unit(vt)

    c = 1.0 / e
    s = math.sqrt(max(0.0, 1.0 - c * c))

    # Direction of incoming velocity at -infinity and natural outgoing velocity
    # at +infinity in the osculating two-body flyby plane.
    vinf_in_hat = c * rhat + s * that
    vinf_out_hat = -c * rhat + s * that

    turn_deg = math.degrees(2.0 * math.asin(clamp(1.0 / e, -1.0, 1.0)))

    return {
        "hyperbolic": True,
        "specific_energy_m2_s2": eps,
        "vinf_mag_m_s": vinf,
        "eccentricity": e,
        "turn_angle_deg": turn_deg,
        "natural_vinf_in_raw_m_s": (vinf * vinf_in_hat).tolist(),
        "natural_vinf_out_raw_m_s": (vinf * vinf_out_hat).tolist(),
    }


def angle_deg(a: Sequence[float], b: Sequence[float]) -> float:
    na = norm(a)
    nb = norm(b)
    if na <= 0 or nb <= 0:
        return math.nan
    return math.degrees(math.acos(clamp(float(np.dot(a, b)) / (na * nb), -1.0, 1.0)))


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
    ap.add_argument("--dsm-refine-json", type=Path, required=True)
    ap.add_argument("--anchor-json", type=Path, required=True)
    ap.add_argument("--leg-out", type=int, default=2, help="1-based outgoing leg index, e.g. 2 for DUNA->KERBIN after leg1 arrival")
    ap.add_argument("--body", required=True)
    ap.add_argument("--body-catalog", type=Path, required=True)
    ap.add_argument("--min-altitude-km", type=float, default=50.0)
    ap.add_argument("--atmosphere-margin-km", type=float, default=10.0)
    ap.add_argument("--max-powered-m-s", type=float, default=250.0)
    ap.add_argument("--top-n", type=int, default=50)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    radius_km, mu, body_rec = body_radius_mu(args.body_catalog, args.body)
    safe_alt_km = args.min_altitude_km + args.atmosphere_margin_km
    safe_radius_km = radius_km + safe_alt_km

    leg = load_anchor_leg(args.anchor_json, args.leg_out)
    req_out = leg_vinf_dep_raw_m_s(leg)

    data = json.loads(args.dsm_refine_json.read_text())
    rows_in = data.get("top") or data.get("rows") or []
    audited: list[dict[str, Any]] = []

    for i, row in enumerate(rows_in):
        if not row.get("ok", True):
            continue
        try:
            rp_km = float(row["ca_distance_km"])
            r = row["ca_rel_r_raw_m"]
            v = row["ca_rel_v_raw_m_s"]
            alt_km = rp_km - radius_km
            safe = alt_km >= safe_alt_km

            hyp = asymptotes_from_ca(r, v, mu)
            out = dict(row)
            out.update({
                "audit_index": i,
                "body_radius_km": radius_km,
                "safe_altitude_km": safe_alt_km,
                "safe_radius_km": safe_radius_km,
                "periapsis_radius_km": rp_km,
                "periapsis_altitude_km": alt_km,
                "safe_altitude_pass": safe,
                "required_out_vinf_raw_m_s": req_out.tolist(),
                "required_out_vinf_mag_m_s": norm(req_out),
                **hyp,
            })

            if not safe:
                out["handoff_status"] = "IMPACT_OR_TOO_LOW"
                out["natural_out_vec_err_m_s"] = math.nan
                out["natural_out_angle_deg"] = math.nan
                out["natural_out_mag_mismatch_m_s"] = math.nan
                out["powered_lower_bound_m_s"] = math.nan
            elif not hyp.get("hyperbolic"):
                out["handoff_status"] = "BOUND_OR_NONHYPERBOLIC"
                out["natural_out_vec_err_m_s"] = math.nan
                out["natural_out_angle_deg"] = math.nan
                out["natural_out_mag_mismatch_m_s"] = math.nan
                out["powered_lower_bound_m_s"] = math.nan
            else:
                natural_out = np.asarray(hyp["natural_vinf_out_raw_m_s"], dtype=float)
                vec_err = norm(natural_out - req_out)
                ang = angle_deg(natural_out, req_out)
                mag_mis = abs(norm(natural_out) - norm(req_out))
                powered_lb = mag_mis
                out["natural_out_vec_err_m_s"] = vec_err
                out["natural_out_angle_deg"] = ang
                out["natural_out_mag_mismatch_m_s"] = mag_mis
                out["powered_lower_bound_m_s"] = powered_lb
                if vec_err <= args.max_powered_m_s:
                    out["handoff_status"] = "PASS_APPROX"
                else:
                    out["handoff_status"] = "POWERED_OR_RETARGET"

            audited.append(out)

        except Exception as exc:
            audited.append({
                "audit_index": i,
                "handoff_status": "AUDIT_ERROR",
                "error": str(exc),
            })

    def sort_key(r):
        status_rank = {
            "PASS_APPROX": 0,
            "POWERED_OR_RETARGET": 1,
            "BOUND_OR_NONHYPERBOLIC": 2,
            "IMPACT_OR_TOO_LOW": 3,
            "AUDIT_ERROR": 4,
        }.get(r.get("handoff_status"), 9)
        vec = r.get("natural_out_vec_err_m_s")
        if vec is None or not isinstance(vec, (int, float)) or not math.isfinite(vec):
            vec = 1e99
        alt_pen = 0.0 if r.get("safe_altitude_pass") else 1e8
        dsm = float(r.get("dsm_norm_m_s", 0.0) or 0.0)
        return (status_rank, alt_pen, vec, dsm)

    audited.sort(key=sort_key)

    counts: dict[str, int] = {}
    for r in audited:
        counts[r.get("handoff_status", "UNKNOWN")] = counts.get(r.get("handoff_status", "UNKNOWN"), 0) + 1

    out_json = {
        "schema": "vcarel_dsm_handoff_audit_v0",
        "body": args.body.upper(),
        "body_radius_km": radius_km,
        "safe_altitude_km": safe_alt_km,
        "safe_radius_km": safe_radius_km,
        "mu_m3_s2": mu,
        "leg_out": leg,
        "required_out_vinf_raw_m_s": req_out.tolist(),
        "required_out_vinf_mag_m_s": norm(req_out),
        "status_counts": counts,
        "best": audited[0] if audited else None,
        "top": audited[:args.top_n],
        "rows": audited,
    }

    json_path = args.output_dir / "vcarel_dsm_handoff_audit.json"
    csv_path = args.output_dir / "vcarel_dsm_handoff_audit.csv"
    json_path.write_text(json.dumps(out_json, indent=2) + "\n")

    flat = [flatten_row(r) for r in audited]
    if flat:
        fields = sorted({k for r in flat for k in r.keys()})
        with csv_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(flat)

    print("=== VCAREL DSM HANDOFF AUDIT ===")
    print(f"body          : {args.body.upper()} radius={radius_km:.3f} km safe_alt={safe_alt_km:.3f} km")
    print(f"required out  : |vinf|={norm(req_out)/1000:.6f} km/s raw={req_out.tolist()}")
    print(f"status_counts : {counts}")
    print("rank status              alt_km      rp_km    vinf_in  req_out  out_err  out_ang  dsm    ca_t")
    for i, r in enumerate(audited[:args.top_n], 1):
        print(
            f"{i:3d} {r.get('handoff_status','?'):<18} "
            f"{float(r.get('periapsis_altitude_km', math.nan)):10.3f} "
            f"{float(r.get('periapsis_radius_km', math.nan)):10.3f} "
            f"{float(r.get('vinf_mag_m_s', math.nan))/1000:8.3f} "
            f"{norm(req_out)/1000:8.3f} "
            f"{float(r.get('natural_out_vec_err_m_s', math.nan)):8.1f} "
            f"{float(r.get('natural_out_angle_deg', math.nan)):8.3f} "
            f"{float(r.get('dsm_norm_m_s', 0.0) or 0.0):7.2f} "
            f"{float(r.get('ca_t_game_s', math.nan)):14.3f}"
        )

    print(f"[OK] wrote {json_path}")
    print(f"[OK] wrote {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
