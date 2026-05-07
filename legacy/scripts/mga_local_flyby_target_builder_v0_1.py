#!/usr/bin/env python3
"""
mga_local_flyby_target_builder_v0_1.py

Build a local, body-centered hyperbolic flyby target from B-plane target specs.

Input:
  - JSON best-spec or JSONL emitted by mga_bplane_target_spec_v0_1.py
  - Body catalog emitted by kRPC extractor

Output:
  - CSV summary
  - JSONL local flyby target rows
  - JSON summary
  - optional best local target JSON

Scope:
  This is a geometry/handoff builder, not a numerical corrector. It converts a
  validated B-plane target into local hyperbola quantities useful for the next
  targeter: rp, periapsis altitude, v_p, B-vector, approximate periapsis state
  in a Duna-centered inertial frame, approximate SOI entry/exit states and time
  from SOI to periapsis.

Conventions:
  - All distances in km, velocities in km/s, epochs in ET seconds.
  - The flyby encounter_et from the target spec is treated as the nominal
    periapsis epoch of the patched-conic local flyby.
  - Local frame axes are the packet's J2000-like S/T/R/H basis. This is a
    patched-conic handoff convention, not yet a Principia maneuver frame.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

SCHEMA_VERSION = "mga_local_flyby_target.v0.1"
Vec3 = Tuple[float, float, float]


def finite(x: Any, default: float = math.nan) -> float:
    try:
        y = float(x)
    except (TypeError, ValueError):
        return default
    return y if math.isfinite(y) else default


def opt_float(x: Any) -> Optional[float]:
    y = finite(x)
    return y if math.isfinite(y) else None


def vec3(x: Any) -> Optional[Vec3]:
    if not isinstance(x, Sequence) or isinstance(x, (str, bytes)) or len(x) < 3:
        return None
    vals = []
    for i in range(3):
        y = finite(x[i])
        if not math.isfinite(y):
            return None
        vals.append(y)
    return (vals[0], vals[1], vals[2])


def vdot(a: Sequence[float], b: Sequence[float]) -> float:
    return float(a[0])*float(b[0]) + float(a[1])*float(b[1]) + float(a[2])*float(b[2])


def vcross(a: Sequence[float], b: Sequence[float]) -> Vec3:
    return (
        float(a[1])*float(b[2]) - float(a[2])*float(b[1]),
        float(a[2])*float(b[0]) - float(a[0])*float(b[2]),
        float(a[0])*float(b[1]) - float(a[1])*float(b[0]),
    )


def vnorm(a: Sequence[float]) -> float:
    return math.sqrt(max(0.0, vdot(a, a)))


def vunit(a: Sequence[float], fallback: Optional[Vec3] = None) -> Vec3:
    n = vnorm(a)
    if n <= 0.0 or not math.isfinite(n):
        if fallback is not None:
            return fallback
        raise ValueError(f"Cannot normalize vector {a!r}")
    return (float(a[0])/n, float(a[1])/n, float(a[2])/n)


def vadd(a: Sequence[float], b: Sequence[float]) -> Vec3:
    return (float(a[0])+float(b[0]), float(a[1])+float(b[1]), float(a[2])+float(b[2]))


def vsub(a: Sequence[float], b: Sequence[float]) -> Vec3:
    return (float(a[0])-float(b[0]), float(a[1])-float(b[1]), float(a[2])-float(b[2]))


def vmul(s: float, a: Sequence[float]) -> Vec3:
    return (s*float(a[0]), s*float(a[1]), s*float(a[2]))


def angle_deg(a: Sequence[float], b: Sequence[float]) -> float:
    na = vnorm(a); nb = vnorm(b)
    if na <= 0 or nb <= 0:
        return math.nan
    c = max(-1.0, min(1.0, vdot(a, b)/(na*nb)))
    return math.degrees(math.acos(c))


def stable_id(prefix: str, payload: Mapping[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=True, separators=(",", ":"), default=str)
    return f"{prefix}_{hashlib.sha1(raw.encode('utf-8')).hexdigest()[:16]}"


def sanitize(x: Any) -> Any:
    if isinstance(x, Mapping):
        return {str(k): sanitize(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [sanitize(v) for v in x]
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
    out: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if not isinstance(obj, Mapping):
                raise ValueError(f"Expected object at {path}:{line_no}")
            out.append(dict(obj))
    return out


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(sanitize(payload), f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write("\n")


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(sanitize(r), ensure_ascii=False, separators=(",", ":"), allow_nan=False))
            f.write("\n")


def load_specs(path: Path) -> List[Dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        return read_jsonl(path)
    data = load_json(path)
    if "flyby_targets" in data:
        return [data]
    specs = data.get("specs") or data.get("routes") or data.get("target_specs")
    if isinstance(specs, list):
        return [dict(s) for s in specs if isinstance(s, Mapping)]
    raise ValueError(f"Could not find specs in {path}")


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


def hyperbolic_anomaly_from_true_anomaly(e: float, f_rad: float) -> float:
    # For hyperbolic e > 1: tanh(H/2) = sqrt((e-1)/(e+1)) tan(f/2)
    q = math.sqrt(max(0.0, (e - 1.0)/(e + 1.0))) * math.tan(0.5*f_rad)
    q = max(-0.999999999999, min(0.999999999999, q))
    return 2.0 * math.atanh(q)


def time_from_periapsis(mu: float, a_abs: float, e: float, f_rad: float) -> float:
    H = hyperbolic_anomaly_from_true_anomaly(e, f_rad)
    M = e * math.sinh(H) - H
    return math.sqrt((a_abs**3)/mu) * M


def local_state_at_true_anomaly(mu: float, rp: float, e: float, f_rad: float, p_hat: Vec3, q_hat: Vec3, h_hat: Vec3) -> Tuple[Vec3, Vec3, float, float]:
    # Perifocal: r = p/(1+e cos f); p = rp(1+e)
    p = rp * (1.0 + e)
    cf = math.cos(f_rad); sf = math.sin(f_rad)
    rmag = p/(1.0 + e*cf)
    r_hat = vadd(vmul(cf, p_hat), vmul(sf, q_hat))
    r_vec = vmul(rmag, r_hat)
    # v = sqrt(mu/p) * [-sin f p_hat + (e + cos f) q_hat]
    scale = math.sqrt(mu/p)
    v_vec = vadd(vmul(-scale*sf, p_hat), vmul(scale*(e + cf), q_hat))
    return r_vec, v_vec, rmag, vnorm(v_vec)


def build_local_target(spec: Mapping[str, Any], catalog: Mapping[str, Any], args: argparse.Namespace) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    flybys = spec.get("flyby_targets") or []
    if not isinstance(flybys, list):
        return out
    for fb in flybys:
        if not isinstance(fb, Mapping):
            continue
        body_name = str(fb.get("body"))
        body = body_lookup(catalog, body_name)
        hyp = fb.get("hyperbola") if isinstance(fb.get("hyperbola"), Mapping) else {}
        mu = finite(hyp.get("mu_km3_s2", fb.get("mu_km3_s2", body.get("mu_km3_s2"))))
        rp = finite(hyp.get("rp_required_km", fb.get("rp_required_km")))
        vinf_in_vec = vec3(fb.get("vinf_in_vec_km_s"))
        vinf_out_vec = vec3(fb.get("vinf_out_vec_km_s"))
        h_hat_in = vec3(fb.get("h_hat"))
        b_hat = vec3(fb.get("b_hat"))
        s_hat = vec3(fb.get("s_hat"))
        t_hat = vec3(fb.get("t_hat"))
        r_axis_hat = vec3(fb.get("r_hat"))
        if mu <= 0 or rp <= 0 or vinf_in_vec is None or vinf_out_vec is None:
            continue
        vin = vnorm(vinf_in_vec); vout = vnorm(vinf_out_vec)
        v_eff = 0.5*(vin + vout)
        if v_eff <= 0:
            continue
        # Use packet h_hat when available, otherwise rebuild from vinf vectors.
        h_hat = vunit(h_hat_in if h_hat_in is not None else vcross(vinf_in_vec, vinf_out_vec), fallback=(0.0, 0.0, 1.0))
        # Periapsis velocity is approximately the bisector of incoming/outgoing v-infinity directions.
        vp_dir = vunit(vadd(vunit(vinf_in_vec), vunit(vinf_out_vec)), fallback=vunit(vinf_in_vec))
        # At periapsis h = r x v => r_hat = v_hat x h_hat.
        rp_hat = vunit(vcross(vp_dir, h_hat), fallback=b_hat or (1.0, 0.0, 0.0))
        # q_hat completes prograde perifocal basis: h = p x q.
        q_hat = vunit(vcross(h_hat, rp_hat), fallback=vp_dir)
        # Ensure q_hat roughly aligns with periapsis velocity direction.
        if vdot(q_hat, vp_dir) < 0:
            q_hat = vmul(-1.0, q_hat)
            h_hat = vmul(-1.0, h_hat)
        a = -mu/(v_eff*v_eff)
        a_abs = -a
        ecc = 1.0 + rp*v_eff*v_eff/mu
        vp = math.sqrt(v_eff*v_eff + 2.0*mu/rp)
        h = rp*vp
        p = h*h/mu
        bmag = a_abs*math.sqrt(max(0.0, ecc*ecc - 1.0))
        turn = 2.0*math.degrees(math.asin(max(0.0, min(1.0, 1.0/ecc)))) if ecc > 1 else math.nan
        radius = finite(body.get("radius_km", body.get("equatorial_radius_km")))
        atm = finite(body.get("atmosphere_top_km", body.get("atmosphere_depth_km", 0.0)), 0.0)
        soi = finite(body.get("soi_km", body.get("sphere_of_influence_km")))
        alt = rp - radius if math.isfinite(radius) else math.nan
        alt_atm = alt - atm if math.isfinite(alt) else math.nan
        # SOI entry/exit estimates, clipped if SOI absent or inside asymptote domain.
        soi_block: Dict[str, Any] = {}
        if math.isfinite(soi) and soi > rp and ecc > 1.0:
            cosf = (p/soi - 1.0)/ecc
            cosf = max(-1.0, min(1.0, cosf))
            f_soi = math.acos(cosf)
            dt_soi = abs(time_from_periapsis(mu, a_abs, ecc, f_soi))
            r_in, v_in, _, _ = local_state_at_true_anomaly(mu, rp, ecc, -f_soi, rp_hat, q_hat, h_hat)
            r_out, v_out, _, _ = local_state_at_true_anomaly(mu, rp, ecc, f_soi, rp_hat, q_hat, h_hat)
            soi_block = {
                "soi_radius_km": opt_float(soi),
                "true_anomaly_at_soi_deg": math.degrees(f_soi),
                "time_from_soi_to_periapsis_s": dt_soi,
                "time_from_soi_to_periapsis_days": dt_soi/86400.0,
                "entry_state_body_centered": {"dt_from_periapsis_s": -dt_soi, "r_km": r_in, "v_km_s": v_in},
                "exit_state_body_centered": {"dt_from_periapsis_s": dt_soi, "r_km": r_out, "v_km_s": v_out},
            }
        target_id = stable_id("localflyby", {
            "target_spec_id": spec.get("target_spec_id"),
            "body": body_name,
            "encounter_et": fb.get("encounter_et"),
            "rp": round(rp, 9),
            "bt": fb.get("b_dot_t_km"),
            "br": fb.get("b_dot_r_km"),
        })
        out.append({
            "schema_version": SCHEMA_VERSION,
            "local_target_id": target_id,
            "target_spec_id": spec.get("target_spec_id"),
            "packet_id": spec.get("packet_id"),
            "route_id": spec.get("route_id"),
            "sequence": spec.get("sequence"),
            "body": body_name,
            "central_body": spec.get("central_body", args.central_body),
            "frame": spec.get("frame", args.frame),
            "encounter_et": opt_float(fb.get("encounter_et")),
            "nominal_periapsis_et": opt_float(fb.get("encounter_et")),
            "b_plane": {
                "b_magnitude_km": opt_float(fb.get("b_magnitude_km", bmag)),
                "b_magnitude_recomputed_km": bmag,
                "b_dot_t_km": opt_float(fb.get("b_dot_t_km")),
                "b_dot_r_km": opt_float(fb.get("b_dot_r_km")),
                "s_hat": s_hat,
                "t_hat": t_hat,
                "r_hat": r_axis_hat,
                "b_hat": b_hat,
                "h_hat": h_hat,
            },
            "asymptotes": {
                "vinf_in_vec_km_s": vinf_in_vec,
                "vinf_out_vec_km_s": vinf_out_vec,
                "vinf_in_km_s": vin,
                "vinf_out_km_s": vout,
                "vinf_effective_km_s": v_eff,
                "vinf_mismatch_m_s": opt_float(fb.get("vinf_mismatch_m_s")),
                "turn_angle_deg": opt_float(fb.get("turn_angle_deg", turn)),
                "turn_angle_recomputed_deg": turn,
            },
            "hyperbola": {
                "mu_km3_s2": mu,
                "radius_km": opt_float(radius),
                "atmosphere_top_km": opt_float(atm),
                "rp_km": rp,
                "rp_min_km": opt_float(fb.get("rp_min_km")),
                "rp_margin_km": opt_float(fb.get("rp_margin_km")),
                "periapsis_altitude_km": opt_float(alt),
                "periapsis_altitude_above_atmosphere_km": opt_float(alt_atm),
                "semimajor_axis_km": a,
                "eccentricity": ecc,
                "semilatus_rectum_km": p,
                "specific_angular_momentum_km2_s": h,
                "periapsis_speed_km_s": vp,
                "turn_angle_from_rp_deg": turn,
            },
            "periapsis_state_body_centered": {
                "r_km": vmul(rp, rp_hat),
                "v_km_s": vmul(vp, q_hat),
                "r_hat": rp_hat,
                "v_hat": q_hat,
                "h_hat": h_hat,
            },
            "soi_patch_estimate": soi_block,
            "tcm_plan": spec.get("tcm_plan", []),
            "quality": {
                "source_score": opt_float(spec.get("score")),
                "total_departure_correction_m_s": opt_float(spec.get("total_departure_correction_m_s")),
                "max_miss_after_km": opt_float(spec.get("max_miss_after_km")),
                "min_rp_margin_km": opt_float(spec.get("min_rp_margin_km")),
                "max_vinf_mismatch_m_s": opt_float(spec.get("max_vinf_mismatch_m_s")),
            },
        })
    return out


def flatten_rows(targets: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for t in targets:
        hyp = t.get("hyperbola") if isinstance(t.get("hyperbola"), Mapping) else {}
        bp = t.get("b_plane") if isinstance(t.get("b_plane"), Mapping) else {}
        asym = t.get("asymptotes") if isinstance(t.get("asymptotes"), Mapping) else {}
        soi = t.get("soi_patch_estimate") if isinstance(t.get("soi_patch_estimate"), Mapping) else {}
        qual = t.get("quality") if isinstance(t.get("quality"), Mapping) else {}
        rows.append({
            "local_target_id": t.get("local_target_id"),
            "target_spec_id": t.get("target_spec_id"),
            "sequence": t.get("sequence"),
            "body": t.get("body"),
            "encounter_et": t.get("encounter_et"),
            "rp_km": hyp.get("rp_km"),
            "periapsis_altitude_km": hyp.get("periapsis_altitude_km"),
            "periapsis_altitude_above_atmosphere_km": hyp.get("periapsis_altitude_above_atmosphere_km"),
            "rp_margin_km": hyp.get("rp_margin_km"),
            "vinf_effective_km_s": asym.get("vinf_effective_km_s"),
            "vinf_mismatch_m_s": asym.get("vinf_mismatch_m_s"),
            "turn_angle_deg": asym.get("turn_angle_deg"),
            "eccentricity": hyp.get("eccentricity"),
            "periapsis_speed_km_s": hyp.get("periapsis_speed_km_s"),
            "b_magnitude_km": bp.get("b_magnitude_km"),
            "b_magnitude_recomputed_km": bp.get("b_magnitude_recomputed_km"),
            "b_dot_t_km": bp.get("b_dot_t_km"),
            "b_dot_r_km": bp.get("b_dot_r_km"),
            "soi_time_days": soi.get("time_from_soi_to_periapsis_days"),
            "total_departure_correction_m_s": qual.get("total_departure_correction_m_s"),
            "source_score": qual.get("source_score"),
        })
    return rows


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "local_target_id", "target_spec_id", "sequence", "body", "encounter_et",
        "rp_km", "periapsis_altitude_km", "periapsis_altitude_above_atmosphere_km", "rp_margin_km",
        "vinf_effective_km_s", "vinf_mismatch_m_s", "turn_angle_deg", "eccentricity", "periapsis_speed_km_s",
        "b_magnitude_km", "b_magnitude_recomputed_km", "b_dot_t_km", "b_dot_r_km", "soi_time_days",
        "total_departure_correction_m_s", "source_score",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fields})


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build local Duna/body-centered flyby targets from B-plane specs.")
    p.add_argument("--input-spec", required=True, type=Path, help="Best JSON or JSONL from mga_bplane_target_spec_v0_1.py")
    p.add_argument("--body-catalog", required=True, type=Path)
    p.add_argument("--central-body", default="Sun")
    p.add_argument("--frame", default="J2000")
    p.add_argument("--top-n", type=int, default=10)
    p.add_argument("--output-csv", required=True, type=Path)
    p.add_argument("--output-jsonl", required=True, type=Path)
    p.add_argument("--output-json", required=True, type=Path)
    p.add_argument("--output-best-json", required=True, type=Path)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    catalog = load_json(args.body_catalog)
    specs = load_specs(args.input_spec)
    specs.sort(key=lambda s: finite(s.get("score"), 1e99))
    if args.top_n > 0:
        specs = specs[: args.top_n]
    targets: List[Dict[str, Any]] = []
    for s in specs:
        targets.extend(build_local_target(s, catalog, args))
    targets.sort(key=lambda t: (
        finite(((t.get("quality") or {}).get("source_score")), 1e99),
        -finite(((t.get("hyperbola") or {}).get("rp_margin_km")), -1e99),
    ))
    rows = flatten_rows(targets)
    write_csv(args.output_csv, rows)
    write_jsonl(args.output_jsonl, targets)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "input_spec": str(args.input_spec),
        "body_catalog": str(args.body_catalog),
        "specs_input": len(specs),
        "targets_written": len(targets),
        "top_targets": rows[:20],
    }
    write_json(args.output_json, summary)
    write_json(args.output_best_json, targets[0] if targets else {})
    print("="*80)
    print("MGA LOCAL FLYBY TARGET BUILDER V0.1")
    print("="*80)
    print(f"Specs input:     {len(specs)}")
    print(f"Targets written: {len(targets)}")
    print("\nTop local targets:")
    for i, r in enumerate(rows[:10], start=1):
        print(
            f" {i}. {r.get('sequence')} @ {r.get('body')} | "
            f"alt={finite(r.get('periapsis_altitude_km')):.1f} km | "
            f"rp_margin={finite(r.get('rp_margin_km')):.1f} km | "
            f"vinf={finite(r.get('vinf_effective_km_s')):.3f} km/s | "
            f"turn={finite(r.get('turn_angle_deg')):.3f}° | "
            f"B.T={finite(r.get('b_dot_t_km')):.1f} km | B.R={finite(r.get('b_dot_r_km')):.1f} km | "
            f"SOI→Pe={finite(r.get('soi_time_days')):.3f} d"
        )
    print("="*80)
    print(f"[OK] wrote CSV:       {args.output_csv}")
    print(f"[OK] wrote JSONL:     {args.output_jsonl}")
    print(f"[OK] wrote JSON:      {args.output_json}")
    print(f"[OK] wrote best JSON: {args.output_best_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
