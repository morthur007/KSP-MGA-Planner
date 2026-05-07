#!/usr/bin/env python3
"""
mga_local_flyby_validate_v0_1.py

Numerically validate local, body-centered hyperbolic flyby targets emitted by
mga_local_flyby_target_builder_v0_1.py.

Scope:
  - Duna/body-centered two-body validation only;
  - integrate from SOI-entry state to SOI-exit state;
  - verify endpoint closure, periapsis radius/epoch, energy and angular momentum;
  - produce CSV/JSONL/JSON summaries and an optional best JSON.

This is intentionally not a global N-body validator. It validates that the
B-plane/local handoff is self-consistent as a local hyperbolic conic before the
next stage stitches it back to heliocentric arcs or Principia validation.

Units:
  km, km/s, seconds.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

SCHEMA_VERSION = "mga_local_flyby_validation.v0.1"
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
    vals: List[float] = []
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


def vsub(a: Sequence[float], b: Sequence[float]) -> Vec3:
    return (float(a[0])-float(b[0]), float(a[1])-float(b[1]), float(a[2])-float(b[2]))


def vnorm(a: Sequence[float]) -> float:
    return math.sqrt(max(0.0, vdot(a, a)))


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        x = json.load(f)
    if not isinstance(x, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return x


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if not isinstance(obj, Mapping):
                raise ValueError(f"Expected JSON object at {path}:{line_no}")
            rows.append(dict(obj))
    return rows


def load_targets(path: Path) -> List[Dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        return read_jsonl(path)
    data = load_json(path)
    if data.get("schema_version") == "mga_local_flyby_target.v0.1" or "local_target_id" in data:
        return [data]
    targets = data.get("targets") or data.get("local_targets") or data.get("top_targets")
    if isinstance(targets, list):
        return [dict(t) for t in targets if isinstance(t, Mapping)]
    raise ValueError(f"Could not find local flyby targets in {path}")


def sanitize(x: Any) -> Any:
    if isinstance(x, Mapping):
        return {str(k): sanitize(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [sanitize(v) for v in x]
    if isinstance(x, float):
        return x if math.isfinite(x) else None
    return x


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


def flatten_result(r: Mapping[str, Any]) -> Dict[str, Any]:
    t = r.get("target", {}) if isinstance(r.get("target"), Mapping) else {}
    hyp = t.get("hyperbola", {}) if isinstance(t.get("hyperbola"), Mapping) else {}
    asym = t.get("asymptotes", {}) if isinstance(t.get("asymptotes"), Mapping) else {}
    val = r.get("validation", {}) if isinstance(r.get("validation"), Mapping) else {}
    return {
        "local_target_id": t.get("local_target_id"),
        "sequence": t.get("sequence"),
        "body": t.get("body"),
        "pass_validation": val.get("pass_validation"),
        "class": val.get("class"),
        "endpoint_position_miss_km": val.get("endpoint_position_miss_km"),
        "endpoint_velocity_miss_m_s": val.get("endpoint_velocity_miss_m_s"),
        "periapsis_radius_error_km": val.get("periapsis_radius_error_km"),
        "periapsis_time_error_s": val.get("periapsis_time_error_s"),
        "min_radius_km": val.get("min_radius_km"),
        "target_rp_km": hyp.get("rp_km"),
        "periapsis_altitude_km": hyp.get("periapsis_altitude_km"),
        "rp_margin_km": hyp.get("rp_margin_km"),
        "vinf_effective_km_s": asym.get("vinf_effective_km_s"),
        "turn_angle_deg": asym.get("turn_angle_deg"),
        "energy_drift_abs_km2_s2": val.get("energy_drift_abs_km2_s2"),
        "energy_expected_error_km2_s2": val.get("energy_expected_error_km2_s2"),
        "h_drift_abs_km2_s": val.get("h_drift_abs_km2_s"),
        "integration_success": val.get("integration_success"),
        "message": val.get("message"),
    }


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "local_target_id", "sequence", "body", "pass_validation", "class",
        "endpoint_position_miss_km", "endpoint_velocity_miss_m_s",
        "periapsis_radius_error_km", "periapsis_time_error_s", "min_radius_km", "target_rp_km",
        "periapsis_altitude_km", "rp_margin_km", "vinf_effective_km_s", "turn_angle_deg",
        "energy_drift_abs_km2_s2", "energy_expected_error_km2_s2", "h_drift_abs_km2_s",
        "integration_success", "message",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fields})


def two_body_rhs(mu: float):
    def f(_t: float, y: Sequence[float]) -> List[float]:
        rx, ry, rz, vx, vy, vz = map(float, y)
        r2 = rx*rx + ry*ry + rz*rz
        r = math.sqrt(r2)
        if r <= 0.0:
            return [vx, vy, vz, 0.0, 0.0, 0.0]
        fac = -mu/(r*r*r)
        return [vx, vy, vz, fac*rx, fac*ry, fac*rz]
    return f


def energy(mu: float, r: Sequence[float], v: Sequence[float]) -> float:
    rn = vnorm(r)
    return 0.5*vdot(v, v) - mu/rn


def validate_one(target: Mapping[str, Any], args_dict: Mapping[str, Any]) -> Dict[str, Any]:
    import numpy as np
    from scipy.integrate import solve_ivp

    t = dict(target)
    hyp = t.get("hyperbola") if isinstance(t.get("hyperbola"), Mapping) else {}
    soi = t.get("soi_patch_estimate") if isinstance(t.get("soi_patch_estimate"), Mapping) else {}
    peri = t.get("periapsis_state_body_centered") if isinstance(t.get("periapsis_state_body_centered"), Mapping) else {}
    asym = t.get("asymptotes") if isinstance(t.get("asymptotes"), Mapping) else {}

    mu = finite(hyp.get("mu_km3_s2"))
    rp = finite(hyp.get("rp_km"))
    vinf_eff = finite(asym.get("vinf_effective_km_s"))
    entry = soi.get("entry_state_body_centered") if isinstance(soi.get("entry_state_body_centered"), Mapping) else {}
    exit_ = soi.get("exit_state_body_centered") if isinstance(soi.get("exit_state_body_centered"), Mapping) else {}
    r0 = vec3(entry.get("r_km"))
    v0 = vec3(entry.get("v_km_s"))
    r1_expected = vec3(exit_.get("r_km"))
    v1_expected = vec3(exit_.get("v_km_s"))
    t0 = finite(entry.get("dt_from_periapsis_s"))
    t1 = finite(exit_.get("dt_from_periapsis_s"))
    rp_state = vec3(peri.get("r_km"))

    failures: List[str] = []
    if mu <= 0: failures.append("missing_mu")
    if rp <= 0: failures.append("missing_rp")
    if r0 is None or v0 is None: failures.append("missing_entry_state")
    if r1_expected is None or v1_expected is None: failures.append("missing_exit_state")
    if not math.isfinite(t0) or not math.isfinite(t1) or t1 <= t0: failures.append("invalid_time_span")
    if failures:
        return {
            "schema_version": SCHEMA_VERSION,
            "target": t,
            "validation": {
                "pass_validation": False,
                "class": "invalid_input",
                "integration_success": False,
                "message": ";".join(failures),
            },
        }

    y0 = np.array([*r0, *v0], dtype=float)
    max_step_s = finite(args_dict.get("max_step_hours"), 1.0)*3600.0
    rtol = finite(args_dict.get("rtol"), 1e-11)
    atol = finite(args_dict.get("atol"), 1e-13)
    method = str(args_dict.get("integrator") or "DOP853")
    try:
        sol = solve_ivp(
            two_body_rhs(mu),
            (t0, t1),
            y0,
            method=method,
            rtol=rtol,
            atol=atol,
            max_step=max_step_s,
            dense_output=True,
        )
    except Exception as exc:
        return {
            "schema_version": SCHEMA_VERSION,
            "target": t,
            "validation": {
                "pass_validation": False,
                "class": "integration_error",
                "integration_success": False,
                "message": repr(exc),
            },
        }

    if sol.y.shape[1] == 0:
        return {
            "schema_version": SCHEMA_VERSION,
            "target": t,
            "validation": {
                "pass_validation": False,
                "class": "integration_error",
                "integration_success": bool(sol.success),
                "message": "empty_solution",
            },
        }

    y_end = sol.y[:, -1]
    r_end = (float(y_end[0]), float(y_end[1]), float(y_end[2]))
    v_end = (float(y_end[3]), float(y_end[4]), float(y_end[5]))
    pos_miss = vnorm(vsub(r_end, r1_expected))
    vel_miss_km_s = vnorm(vsub(v_end, v1_expected))
    vel_miss_m_s = vel_miss_km_s*1000.0

    # Evaluate periapsis at t=0 when covered; otherwise use dense/sample minimum.
    if sol.success and getattr(sol, "sol", None) is not None and t0 <= 0.0 <= t1:
        y_pe = sol.sol(0.0)
        r_pe = (float(y_pe[0]), float(y_pe[1]), float(y_pe[2]))
        v_pe = (float(y_pe[3]), float(y_pe[4]), float(y_pe[5]))
        rpe_num = vnorm(r_pe)
        tpe_num = 0.0
        if rp_state is not None:
            pe_state_error = vnorm(vsub(r_pe, rp_state))
        else:
            pe_state_error = math.nan
    else:
        n = max(101, int(args_dict.get("samples", 401)))
        ts = np.linspace(t0, t1, n)
        vals = sol.sol(ts) if sol.success and getattr(sol, "sol", None) is not None else sol.y
        radii = np.linalg.norm(vals[:3, :], axis=0)
        idx = int(np.argmin(radii))
        rpe_num = float(radii[idx])
        tpe_num = float(ts[idx]) if vals.shape[1] == n else float(sol.t[idx])
        pe_state_error = math.nan

    rp_error = abs(rpe_num - rp)
    tpe_error = abs(tpe_num)

    e0 = energy(mu, r0, v0)
    e1 = energy(mu, r_end, v_end)
    expected_energy = 0.5*vinf_eff*vinf_eff if math.isfinite(vinf_eff) else math.nan
    energy_drift = abs(e1 - e0)
    expected_energy_error = abs(e0 - expected_energy) if math.isfinite(expected_energy) else math.nan
    h0 = vnorm(vcross(r0, v0))
    h1 = vnorm(vcross(r_end, v_end))
    h_drift = abs(h1 - h0)

    pos_thr = finite(args_dict.get("endpoint_position_threshold_km"), 1e-3)
    vel_thr = finite(args_dict.get("endpoint_velocity_threshold_m_s"), 1e-3)
    rp_thr = finite(args_dict.get("periapsis_radius_threshold_km"), 1e-3)
    time_thr = finite(args_dict.get("periapsis_time_threshold_s"), 1e-2)
    pass_val = bool(sol.success and pos_miss <= pos_thr and vel_miss_m_s <= vel_thr and rp_error <= rp_thr and tpe_error <= time_thr)
    if pass_val:
        klass = "A"
    elif sol.success and pos_miss <= 1.0 and vel_miss_m_s <= 1.0:
        klass = "B"
    elif sol.success:
        klass = "C"
    else:
        klass = "failed"

    return {
        "schema_version": SCHEMA_VERSION,
        "target": t,
        "validation": {
            "pass_validation": pass_val,
            "class": klass,
            "integration_success": bool(sol.success),
            "message": str(sol.message),
            "endpoint_position_miss_km": pos_miss,
            "endpoint_velocity_miss_km_s": vel_miss_km_s,
            "endpoint_velocity_miss_m_s": vel_miss_m_s,
            "min_radius_km": rpe_num,
            "target_rp_km": rp,
            "periapsis_radius_error_km": rp_error,
            "periapsis_time_error_s": tpe_error,
            "periapsis_state_position_error_km": pe_state_error,
            "energy_initial_km2_s2": e0,
            "energy_final_km2_s2": e1,
            "energy_drift_abs_km2_s2": energy_drift,
            "energy_expected_km2_s2": expected_energy,
            "energy_expected_error_km2_s2": expected_energy_error,
            "h_initial_km2_s": h0,
            "h_final_km2_s": h1,
            "h_drift_abs_km2_s": h_drift,
            "t_span_s": t1 - t0,
            "nfev": int(getattr(sol, "nfev", -1)),
            "steps": int(len(sol.t)),
        },
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate local two-body hyperbolic flyby targets.")
    p.add_argument("--input-target", required=True, type=Path, help="JSON/JSONL from mga_local_flyby_target_builder_v0_1.py")
    p.add_argument("--body-catalog", type=Path, help="Optional body catalog, kept for provenance")
    p.add_argument("--top-n", type=int, default=0)
    p.add_argument("--workers", type=int, default=1, help="0 = os.cpu_count(); 1 = serial")
    p.add_argument("--multiprocessing-start-method", default="spawn", choices=["spawn", "fork", "forkserver"])
    p.add_argument("--integrator", default="DOP853")
    p.add_argument("--rtol", type=float, default=1e-11)
    p.add_argument("--atol", type=float, default=1e-13)
    p.add_argument("--max-step-hours", type=float, default=0.25)
    p.add_argument("--samples", type=int, default=401)
    p.add_argument("--endpoint-position-threshold-km", type=float, default=1e-3)
    p.add_argument("--endpoint-velocity-threshold-m-s", type=float, default=1e-3)
    p.add_argument("--periapsis-radius-threshold-km", type=float, default=1e-3)
    p.add_argument("--periapsis-time-threshold-s", type=float, default=1e-2)
    p.add_argument("--output-csv", required=True, type=Path)
    p.add_argument("--output-jsonl", required=True, type=Path)
    p.add_argument("--output-json", required=True, type=Path)
    p.add_argument("--output-best-json", required=True, type=Path)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    targets = load_targets(args.input_target)
    targets.sort(key=lambda t: finite(((t.get("quality") or {}).get("source_score")), 1e99))
    if args.top_n > 0:
        targets = targets[:args.top_n]
    args_dict = vars(args).copy()
    # Convert Paths to strings so args_dict is picklable and JSON-safe-ish.
    for k, v in list(args_dict.items()):
        if isinstance(v, Path):
            args_dict[k] = str(v)

    if args.workers == 0:
        workers = os.cpu_count() or 1
    else:
        workers = max(1, args.workers)

    results: List[Dict[str, Any]] = []
    if workers == 1 or len(targets) <= 1:
        for t in targets:
            results.append(validate_one(t, args_dict))
    else:
        ctx = mp.get_context(args.multiprocessing_start_method)
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
            futs = [ex.submit(validate_one, t, args_dict) for t in targets]
            for fut in as_completed(futs):
                results.append(fut.result())

    results.sort(key=lambda r: (
        not bool(((r.get("validation") or {}).get("pass_validation"))),
        finite(((r.get("validation") or {}).get("endpoint_position_miss_km")), 1e99),
        finite(((r.get("validation") or {}).get("endpoint_velocity_miss_m_s")), 1e99),
    ))
    flat = [flatten_result(r) for r in results]
    write_csv(args.output_csv, flat)
    write_jsonl(args.output_jsonl, results)
    pass_count = sum(1 for r in results if ((r.get("validation") or {}).get("pass_validation")))
    pos_vals = [finite(r.get("endpoint_position_miss_km")) for r in flat if math.isfinite(finite(r.get("endpoint_position_miss_km")))]
    vel_vals = [finite(r.get("endpoint_velocity_miss_m_s")) for r in flat if math.isfinite(finite(r.get("endpoint_velocity_miss_m_s")))]
    rp_vals = [finite(r.get("periapsis_radius_error_km")) for r in flat if math.isfinite(finite(r.get("periapsis_radius_error_km")))]
    summary = {
        "schema_version": SCHEMA_VERSION,
        "input_target": str(args.input_target),
        "body_catalog": str(args.body_catalog) if args.body_catalog else None,
        "targets_input": len(targets),
        "targets_validated": len(results),
        "pass_count": pass_count,
        "workers": workers,
        "thresholds": {
            "endpoint_position_threshold_km": args.endpoint_position_threshold_km,
            "endpoint_velocity_threshold_m_s": args.endpoint_velocity_threshold_m_s,
            "periapsis_radius_threshold_km": args.periapsis_radius_threshold_km,
            "periapsis_time_threshold_s": args.periapsis_time_threshold_s,
        },
        "stats": {
            "endpoint_position_miss_km_min": min(pos_vals) if pos_vals else None,
            "endpoint_position_miss_km_median": sorted(pos_vals)[len(pos_vals)//2] if pos_vals else None,
            "endpoint_position_miss_km_max": max(pos_vals) if pos_vals else None,
            "endpoint_velocity_miss_m_s_min": min(vel_vals) if vel_vals else None,
            "endpoint_velocity_miss_m_s_median": sorted(vel_vals)[len(vel_vals)//2] if vel_vals else None,
            "endpoint_velocity_miss_m_s_max": max(vel_vals) if vel_vals else None,
            "periapsis_radius_error_km_max": max(rp_vals) if rp_vals else None,
        },
        "top_results": flat[:20],
    }
    write_json(args.output_json, summary)
    write_json(args.output_best_json, results[0] if results else {})

    print("="*80)
    print("MGA LOCAL FLYBY VALIDATION V0.1")
    print("="*80)
    print(f"Targets input:     {len(targets)}")
    print(f"Validated targets: {len(results)}")
    print(f"Pass validation:   {pass_count}")
    print(f"Workers:           {workers}")
    if pos_vals:
        print(f"Endpoint pos miss: min={min(pos_vals):.6g} km median={sorted(pos_vals)[len(pos_vals)//2]:.6g} km max={max(pos_vals):.6g} km")
    if vel_vals:
        print(f"Endpoint vel miss: min={min(vel_vals):.6g} m/s median={sorted(vel_vals)[len(vel_vals)//2]:.6g} m/s max={max(vel_vals):.6g} m/s")
    if rp_vals:
        print(f"Periapsis r error: max={max(rp_vals):.6g} km")
    print("\nTop local validations:")
    for i, r in enumerate(flat[:10], start=1):
        print(
            f" {i}. {r.get('sequence')} @ {r.get('body')} | pass={r.get('pass_validation')} | "
            f"class={r.get('class')} | pos_miss={finite(r.get('endpoint_position_miss_km')):.3g} km | "
            f"vel_miss={finite(r.get('endpoint_velocity_miss_m_s')):.3g} m/s | "
            f"rp_err={finite(r.get('periapsis_radius_error_km')):.3g} km | "
            f"alt={finite(r.get('periapsis_altitude_km')):.1f} km"
        )
    print("="*80)
    print(f"[OK] wrote CSV:       {args.output_csv}")
    print(f"[OK] wrote JSONL:     {args.output_jsonl}")
    print(f"[OK] wrote JSON:      {args.output_json}")
    print(f"[OK] wrote best JSON: {args.output_best_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
