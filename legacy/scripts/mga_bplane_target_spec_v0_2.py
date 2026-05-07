#!/usr/bin/env python3
"""
mga_bplane_target_spec_v0_2.py

Convert MGA B-plane packets into compact local-targeting specs.

Input:
  - JSON packet from mga_make_bplane_packet_v0_1.py, or JSONL of RoutePacket rows.
  - Body catalog from kRPC/config patch.

Output:
  - CSV summary per flyby target.
  - JSONL one target spec per route.
  - JSON summary plus optional best-spec JSON.

Scope:
  This is not a differential corrector. It is the explicit handoff from global/coarse
  MGA into a local B-plane targeter: periapsis radius/altitude, B-plane coordinates,
  incoming/outgoing v-infinity vectors, hyperbola scalar parameters, and TCM metadata.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

SCHEMA_VERSION = "mga_bplane_target_spec.v0.2"
Vec3 = Tuple[float, float, float]


def finite(x: Any, default: float = math.nan) -> float:
    try:
        y = float(x)
    except (TypeError, ValueError):
        return default
    return y if math.isfinite(y) else default


def finite_or_none(x: Any) -> Optional[float]:
    y = finite(x)
    return y if math.isfinite(y) else None


def vec_or_none(x: Any) -> Optional[List[float]]:
    if isinstance(x, Sequence) and not isinstance(x, (str, bytes)) and len(x) >= 3:
        out = []
        for i in range(3):
            v = finite(x[i])
            if not math.isfinite(v):
                return None
            out.append(v)
        return out
    return None


def norm(v: Optional[Sequence[float]]) -> Optional[float]:
    if v is None or len(v) < 3:
        return None
    return math.sqrt(sum(float(v[i]) ** 2 for i in range(3)))


def stable_id(prefix: str, payload: Mapping[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=True, separators=(",", ":"), default=str)
    return f"{prefix}_{hashlib.sha1(raw.encode('utf-8')).hexdigest()[:16]}"


def json_sanitize(x: Any) -> Any:
    if isinstance(x, Mapping):
        return {str(k): json_sanitize(v) for k, v in x.items()}
    if isinstance(x, list):
        return [json_sanitize(v) for v in x]
    if isinstance(x, tuple):
        return [json_sanitize(v) for v in x]
    if isinstance(x, float):
        return x if math.isfinite(x) else None
    return x


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return data


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if not isinstance(obj, Mapping):
                raise ValueError(f"Expected object at {path}:{line_no}")
            rows.append(dict(obj))
    return rows


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(json_sanitize(payload), f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write("\n")


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(json_sanitize(row), ensure_ascii=False, separators=(",", ":"), allow_nan=False))
            f.write("\n")


def load_routes(input_path: Path) -> List[Dict[str, Any]]:
    if input_path.suffix.lower() == ".jsonl":
        return read_jsonl(input_path)
    data = load_json(input_path)
    routes = data.get("routes")
    if isinstance(routes, list):
        return [dict(r) for r in routes if isinstance(r, Mapping)]
    # Allow a single RoutePacket JSON object.
    if "flybys" in data and "legs" in data:
        return [data]
    raise ValueError(f"Could not find routes in {input_path}")


def body_lookup(catalog: Mapping[str, Any], name: str) -> Dict[str, Any]:
    bodies = catalog.get("bodies") or {}
    if not isinstance(bodies, Mapping):
        return {}
    ent = bodies.get(name)
    if ent is None:
        for k, v in bodies.items():
            if str(k).lower() == name.lower():
                ent = v
                break
    return dict(ent) if isinstance(ent, Mapping) else {}


def hyperbola_from_flyby(fb: Mapping[str, Any], body: Mapping[str, Any]) -> Dict[str, Any]:
    mu = finite(fb.get("mu_km3_s2", body.get("mu_km3_s2", body.get("gm_km3_s2"))))
    vin = finite(fb.get("vinf_in_km_s"))
    vout = finite(fb.get("vinf_out_km_s"))
    v_eff = 0.5 * (vin + vout) if math.isfinite(vin) and math.isfinite(vout) else math.nan
    rp = finite(fb.get("rp_required_km"))
    radius = finite(body.get("radius_km", body.get("equatorial_radius_km")))
    atm = finite(body.get("atmosphere_top_km", body.get("atmosphere_depth_km", 0.0)), 0.0)

    alt = rp - radius if math.isfinite(rp) and math.isfinite(radius) else math.nan
    alt_above_atm = alt - atm if math.isfinite(alt) and math.isfinite(atm) else math.nan

    if mu > 0 and v_eff > 0 and rp > 0:
        a_km = -mu / (v_eff * v_eff)
        ecc = 1.0 + rp * v_eff * v_eff / mu
        vp = math.sqrt(v_eff * v_eff + 2.0 * mu / rp)
        h = rp * vp
        turn_from_rp = 2.0 * math.degrees(math.asin(1.0 / ecc)) if ecc >= 1.0 else math.nan
        c3_rel = v_eff * v_eff
    else:
        a_km = ecc = vp = h = turn_from_rp = c3_rel = math.nan

    return {
        "mu_km3_s2": finite_or_none(mu),
        "radius_km": finite_or_none(radius),
        "atmosphere_top_km": finite_or_none(atm),
        "v_eff_km_s": finite_or_none(v_eff),
        "c3_relative_km2_s2": finite_or_none(c3_rel),
        "rp_required_km": finite_or_none(rp),
        "periapsis_altitude_km": finite_or_none(alt),
        "periapsis_altitude_above_atmosphere_km": finite_or_none(alt_above_atm),
        "semimajor_axis_km": finite_or_none(a_km),
        "eccentricity": finite_or_none(ecc),
        "periapsis_speed_km_s": finite_or_none(vp),
        "specific_angular_momentum_km2_s": finite_or_none(h),
        "turn_angle_from_rp_deg": finite_or_none(turn_from_rp),
    }


def route_score(route: Mapping[str, Any], rp_soft_margin_km: float, vinf_soft_m_s: float) -> float:
    corr = finite(route.get("total_departure_correction_m_s"), 1e6)
    miss = finite(route.get("max_miss_after_km"), 1e6)
    rp = finite(route.get("min_rp_margin_km"), -1e6)
    vinf = finite(route.get("max_vinf_mismatch_m_s"), 1e6)
    rp_penalty = max(0.0, rp_soft_margin_km - rp) / max(1.0, rp_soft_margin_km)
    vinf_penalty = max(0.0, vinf - vinf_soft_m_s) / max(1.0, vinf_soft_m_s)
    return corr + 0.001 * miss + 2.0 * rp_penalty + 0.5 * vinf_penalty


def make_spec(route: Mapping[str, Any], catalog: Mapping[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    flybys = route.get("flybys") or []
    legs = route.get("legs") or []
    if not isinstance(flybys, list):
        flybys = []
    if not isinstance(legs, list):
        legs = []

    flyby_targets: List[Dict[str, Any]] = []
    for fb in flybys:
        if not isinstance(fb, Mapping):
            continue
        body_name = str(fb.get("body"))
        body = body_lookup(catalog, body_name)
        hyp = hyperbola_from_flyby(fb, body)
        flyby_targets.append({
            "body": body_name,
            "flyby_index": fb.get("flyby_index"),
            "encounter_et": finite_or_none(fb.get("encounter_et")),
            "vinf_in_vec_km_s": vec_or_none(fb.get("vinf_in_vec_km_s")),
            "vinf_out_vec_km_s": vec_or_none(fb.get("vinf_out_vec_km_s")),
            "vinf_in_km_s": finite_or_none(fb.get("vinf_in_km_s")),
            "vinf_out_km_s": finite_or_none(fb.get("vinf_out_km_s")),
            "vinf_mismatch_m_s": finite_or_none(fb.get("vinf_mismatch_m_s")),
            "turn_angle_deg": finite_or_none(fb.get("turn_angle_deg")),
            "rp_min_km": finite_or_none(fb.get("rp_min_km")),
            "rp_margin_km": finite_or_none(fb.get("rp_margin_km")),
            "b_magnitude_km": finite_or_none(fb.get("b_magnitude_km")),
            "b_dot_t_km": finite_or_none(fb.get("b_dot_t_km")),
            "b_dot_r_km": finite_or_none(fb.get("b_dot_r_km")),
            "s_hat": vec_or_none(fb.get("s_hat")),
            "t_hat": vec_or_none(fb.get("t_hat")),
            "r_hat": vec_or_none(fb.get("r_hat")),
            "h_hat": vec_or_none(fb.get("h_hat")),
            "b_hat": vec_or_none(fb.get("b_hat")),
            "pass_flyby": bool(fb.get("pass_flyby")),
            "hyperbola": hyp,
        })

    tcm_plan: List[Dict[str, Any]] = []
    for leg in legs:
        if not isinstance(leg, Mapping):
            continue
        dv = vec_or_none(leg.get("dv_correction_km_s")) or [0.0, 0.0, 0.0]
        tcm_plan.append({
            "leg_index": leg.get("leg_index"),
            "origin": leg.get("origin"),
            "target": leg.get("target"),
            "depart_et": finite_or_none(leg.get("depart_et")),
            "arrive_et": finite_or_none(leg.get("arrive_et")),
            "tof_days": finite_or_none(leg.get("tof_days")),
            "dv_correction_km_s": dv,
            "dv_correction_m_s": finite_or_none(leg.get("dv_correction_m_s")),
            "miss_after_km": finite_or_none(leg.get("miss_after_km")),
            "suggested_execution": "at_leg_departure_patch_point",
        })

    score = route_score(route, args.rp_soft_margin_km, args.vinf_soft_m_s)
    spec_id = stable_id("bptgt", {
        "packet_id": route.get("packet_id"),
        "route_id": route.get("route_id"),
        "score": round(score, 9),
    })
    return {
        "schema_version": SCHEMA_VERSION,
        "target_spec_id": spec_id,
        "packet_id": route.get("packet_id"),
        "route_id": route.get("route_id"),
        "route_rank": route.get("route_rank"),
        "sequence": route.get("sequence"),
        "status": "ok" if route.get("pass_all_flybys") else "not_passed",
        "score": score,
        "objective": finite_or_none(route.get("objective")),
        "total_tof_days": finite_or_none(route.get("total_tof_days")),
        "total_departure_correction_m_s": finite_or_none(route.get("total_departure_correction_m_s")),
        "max_miss_after_km": finite_or_none(route.get("max_miss_after_km")),
        "min_rp_margin_km": finite_or_none(route.get("min_rp_margin_km")),
        "max_vinf_mismatch_m_s": finite_or_none(route.get("max_vinf_mismatch_m_s")),
        "central_body": args.central_body,
        "frame": args.frame,
        "flyby_targets": flyby_targets,
        "tcm_plan": tcm_plan,
        "source_packet": route if args.include_source_packet else None,
    }


def flatten_rows(specs: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for s in specs:
        for fb in s.get("flyby_targets", []) or []:
            if not isinstance(fb, Mapping):
                continue
            hyp = fb.get("hyperbola") if isinstance(fb.get("hyperbola"), Mapping) else {}
            rows.append({
                "target_spec_id": s.get("target_spec_id"),
                "packet_id": s.get("packet_id"),
                "route_id": s.get("route_id"),
                "route_rank": s.get("route_rank"),
                "sequence": s.get("sequence"),
                "score": s.get("score"),
                "total_tof_days": s.get("total_tof_days"),
                "total_departure_correction_m_s": s.get("total_departure_correction_m_s"),
                "max_miss_after_km": s.get("max_miss_after_km"),
                "body": fb.get("body"),
                "encounter_et": fb.get("encounter_et"),
                "vinf_in_km_s": fb.get("vinf_in_km_s"),
                "vinf_out_km_s": fb.get("vinf_out_km_s"),
                "vinf_mismatch_m_s": fb.get("vinf_mismatch_m_s"),
                "turn_angle_deg": fb.get("turn_angle_deg"),
                "rp_min_km": fb.get("rp_min_km"),
                "rp_required_km": hyp.get("rp_required_km"),
                "rp_margin_km": fb.get("rp_margin_km"),
                "periapsis_altitude_km": hyp.get("periapsis_altitude_km"),
                "periapsis_altitude_above_atmosphere_km": hyp.get("periapsis_altitude_above_atmosphere_km"),
                "eccentricity": hyp.get("eccentricity"),
                "periapsis_speed_km_s": hyp.get("periapsis_speed_km_s"),
                "b_magnitude_km": fb.get("b_magnitude_km"),
                "b_dot_t_km": fb.get("b_dot_t_km"),
                "b_dot_r_km": fb.get("b_dot_r_km"),
                "pass_flyby": int(bool(fb.get("pass_flyby"))),
            })
    return rows


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "target_spec_id", "packet_id", "route_id", "route_rank", "sequence", "score", "total_tof_days",
        "total_departure_correction_m_s", "max_miss_after_km", "body", "encounter_et", "vinf_in_km_s",
        "vinf_out_km_s", "vinf_mismatch_m_s", "turn_angle_deg", "rp_min_km", "rp_required_km",
        "rp_margin_km", "periapsis_altitude_km", "periapsis_altitude_above_atmosphere_km", "eccentricity",
        "periapsis_speed_km_s", "b_magnitude_km", "b_dot_t_km", "b_dot_r_km", "pass_flyby",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k) for k in fields})


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build local B-plane target specs from MGA B-plane packets.")
    p.add_argument("--input-packet", required=True, type=Path, help="JSON packet or JSONL from mga_make_bplane_packet_v0_1.py")
    p.add_argument("--body-catalog", required=True, type=Path)
    p.add_argument("--central-body", default="Sun")
    p.add_argument("--frame", default="J2000")
    p.add_argument("--top-n", type=int, default=10)
    p.add_argument("--min-rp-margin-km", type=float, default=50.0)
    p.add_argument("--max-vinf-mismatch-m-s", type=float, default=25.0)
    p.add_argument("--max-correction-m-s", type=float, default=20.0)
    p.add_argument("--max-miss-after-km", type=float, default=10.0, help="Reject route packets whose max_miss_after_km exceeds this threshold. Use a large value to disable.")
    p.add_argument("--rp-soft-margin-km", type=float, default=500.0)
    p.add_argument("--vinf-soft-m-s", type=float, default=5.0)
    p.add_argument("--include-source-packet", action="store_true")
    p.add_argument("--output-csv", required=True, type=Path)
    p.add_argument("--output-jsonl", required=True, type=Path)
    p.add_argument("--output-json", required=True, type=Path)
    p.add_argument("--output-best-json", required=True, type=Path)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    catalog = load_json(args.body_catalog)
    routes = load_routes(args.input_packet)

    eligible = []
    reject_counts: Dict[str, int] = {}
    for r in routes:
        if not bool(r.get("pass_all_flybys")):
            reject_counts["not_pass_all_flybys"] = reject_counts.get("not_pass_all_flybys", 0) + 1
            continue
        if finite(r.get("min_rp_margin_km"), -1e99) < args.min_rp_margin_km:
            reject_counts["rp_margin"] = reject_counts.get("rp_margin", 0) + 1
            continue
        if finite(r.get("max_vinf_mismatch_m_s"), 1e99) > args.max_vinf_mismatch_m_s:
            reject_counts["vinf_mismatch"] = reject_counts.get("vinf_mismatch", 0) + 1
            continue
        if finite(r.get("total_departure_correction_m_s"), 1e99) > args.max_correction_m_s:
            reject_counts["correction"] = reject_counts.get("correction", 0) + 1
            continue
        if finite(r.get("max_miss_after_km"), 1e99) > args.max_miss_after_km:
            reject_counts["miss_after"] = reject_counts.get("miss_after", 0) + 1
            continue
        eligible.append(r)

    specs = [make_spec(r, catalog, args) for r in eligible]
    specs.sort(key=lambda s: finite(s.get("score"), 1e99))
    if args.top_n > 0:
        specs = specs[: args.top_n]

    rows = flatten_rows(specs)
    write_csv(args.output_csv, rows)
    write_jsonl(args.output_jsonl, specs)
    best = specs[0] if specs else {}
    write_json(args.output_best_json, best)

    summary = {
        "schema_version": SCHEMA_VERSION,
        "input_packet": str(args.input_packet),
        "body_catalog": str(args.body_catalog),
        "routes_input": len(routes),
        "routes_eligible": len(eligible),
        "specs_written": len(specs),
        "reject_counts": reject_counts,
        "thresholds": {
            "min_rp_margin_km": args.min_rp_margin_km,
            "max_vinf_mismatch_m_s": args.max_vinf_mismatch_m_s,
            "max_correction_m_s": args.max_correction_m_s,
            "max_miss_after_km": args.max_miss_after_km,
            "rp_soft_margin_km": args.rp_soft_margin_km,
            "vinf_soft_m_s": args.vinf_soft_m_s,
        },
        "top_specs": [
            {
                "target_spec_id": s.get("target_spec_id"),
                "sequence": s.get("sequence"),
                "score": s.get("score"),
                "total_departure_correction_m_s": s.get("total_departure_correction_m_s"),
                "min_rp_margin_km": s.get("min_rp_margin_km"),
                "max_vinf_mismatch_m_s": s.get("max_vinf_mismatch_m_s"),
                "total_tof_days": s.get("total_tof_days"),
            }
            for s in specs[:20]
        ],
    }
    write_json(args.output_json, summary)

    print("=" * 80)
    print("MGA B-PLANE TARGET SPEC V0.2")
    print("=" * 80)
    print(f"Routes input:     {len(routes)}")
    print(f"Eligible routes:  {len(eligible)}")
    print(f"Specs written:    {len(specs)}")
    if reject_counts:
        print("\nReject reasons:")
        for k, v in sorted(reject_counts.items(), key=lambda kv: (-kv[1], kv[0])):
            print(f"  - {k:<18} {v}")
    print("\nTop specs:")
    for i, s in enumerate(specs[:10], start=1):
        flyby_desc = []
        for fb in (s.get("flyby_targets") or []):
            if not isinstance(fb, Mapping):
                continue
            hyp = fb.get("hyperbola") if isinstance(fb.get("hyperbola"), Mapping) else {}
            flyby_desc.append(
                f"{fb.get('body')}:alt={finite(hyp.get('periapsis_altitude_km')):.1f}km,"
                f"rpM={finite(fb.get('rp_margin_km')):.1f}km,"
                f"vinfMis={finite(fb.get('vinf_mismatch_m_s')):.1f}m/s"
            )
        print(
            f" {i}. {s.get('sequence')} | score={finite(s.get('score')):.3f} | "
            f"corr={finite(s.get('total_departure_correction_m_s')):.3f} m/s | "
            f"miss={finite(s.get('max_miss_after_km')):.3g} km | "
            f"flybys=[{' ; '.join(flyby_desc)}]"
        )
    print("=" * 80)
    print(f"[OK] wrote CSV:       {args.output_csv}")
    print(f"[OK] wrote JSONL:     {args.output_jsonl}")
    print(f"[OK] wrote JSON:      {args.output_json}")
    print(f"[OK] wrote best JSON: {args.output_best_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
