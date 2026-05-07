#!/usr/bin/env python3
"""
fast_principia_observer_v3.py

Observer kRPC/Principia V3: warp-as-transport, read-at-1x.

Motivation
----------
V2/fast observers can produce contaminated samples when positions/velocities are
read while KSP/Principia is still in high rails warp or immediately after a warp
transition. V3 uses time warp only to move the game clock near the target epoch,
then explicitly returns to 1x, waits for stabilization, performs two direct
state reads, checks kinematic consistency, and only then writes the sample.

Key design choices
------------------
- kRPC is DAQ only, not a solver.
- The exported epoch is actual_ut_s, not target_ut_s.
- Default export is the raw kRPC frame. Do NOT enable right-handed export unless
  the snapshot was generated with the same transform.
- By default, the first stored sample is start_ut + step_seconds. A warmup read
  is performed and discarded.
- Rejected samples are logged, not written to states.csv.

Expected outputs
----------------
output_dir/
  states.csv             Accepted states, compatible with rebound_level_a_cache.py
  sample_log.csv          Per-target status, lag, validation metrics
  body_catalog.json       Names and physical parameters available via kRPC
  manifest.json           Acquisition metadata
  relative_monitor.csv    Optional coarse relative monitor for common families

Example
-------
python fast_principia_observer_v3.py \
  --output-dir data/jnsq_gate0/ksp_5d_v3 \
  --reference-body Sun \
  --duration-seconds 216000 \
  --step-seconds 3600 \
  --rails-warp-factor 6 \
  --settle-seconds 5 \
  --verify-dt-seconds 0.25
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    import krpc  # type: ignore
except ImportError as exc:
    raise SystemExit("O pacote 'krpc' não está instalado neste ambiente Python.") from exc


Vec3 = Tuple[float, float, float]


def norm3(v: Vec3) -> float:
    return math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])


def sub3(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def add3(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def mul3(a: Vec3, s: float) -> Vec3:
    return (a[0] * s, a[1] * s, a[2] * s)


def vec3(v: Any) -> Vec3:
    return (float(v[0]), float(v[1]), float(v[2]))


def get_body_names(bodies: Any) -> List[str]:
    try:
        return sorted(str(k) for k in bodies.keys())
    except Exception:
        return sorted(str(getattr(b, "name")) for b in bodies)


def get_body(bodies: Any, name: str) -> Any:
    try:
        return bodies[name]
    except Exception:
        for b in bodies:
            if getattr(b, "name", None) == name:
                return b
        raise KeyError(name)


def pick_reference_body(bodies: Any, preferred: Optional[str]) -> Any:
    names = get_body_names(bodies)
    if preferred and preferred in names:
        return get_body(bodies, preferred)
    for candidate in ("Sun", "Kerbol"):
        if candidate in names:
            return get_body(bodies, candidate)
    return get_body(bodies, names[0])


def transform_vec(v: Vec3, mode: str) -> Vec3:
    """Optional handedness transforms. Default mode 'raw' leaves kRPC untouched.

    'swap_yz' is a common left/right-handed conversion by swapping Y/Z.
    'negate_z' keeps axes but mirrors Z. Use only if snapshot uses same mode.
    """
    if mode == "raw":
        return v
    if mode == "jnsq_canonical":
        # kRPC raw -> LevelA = (+Z, -X, +Y)
        return (v[2], -v[0], v[1])
    if mode == "swap_yz":
        return (v[0], v[2], v[1])
    if mode == "negate_z":
        return (v[0], v[1], -v[2])
    raise ValueError(f"unknown transform mode: {mode}")


def stop_warp(sc: Any, sleep_s: float = 0.1) -> None:
    """Best-effort return to 1x / no warp."""
    for attr in ("rails_warp_factor", "physics_warp_factor"):
        try:
            setattr(sc, attr, 0)
        except Exception:
            pass
    if sleep_s > 0:
        time.sleep(sleep_s)


def warp_to_target(sc: Any, target_ut: float, rails_warp_factor: int, lag_acumulado: float = 0.0, poll_s: float = 0.05) -> None:
    """Warp Balístico ADAPTATIVO: Aprende com o overshoot passado."""
    current = float(sc.ut)
    if target_ut <= current:
        stop_warp(sc)
        return

    rates = [1, 5, 10, 50, 100, 1000, 10000, 100000, 1000000]
    max_rate = rates[rails_warp_factor] if rails_warp_factor < len(rates) else 1000000

    inercia_s = 0.15 
    margem_final = 180.0 
    
    # A MÁGICA AQUI: O ponto de freio empurra para trás baseado no último erro!
    brake_distance = (max_rate * inercia_s) + margem_final + lag_acumulado

    try:
        sc.rails_warp_factor = rails_warp_factor
    except Exception:
        pass

    while True:
        current = float(sc.ut)
        delta = target_ut - current
        if delta <= brake_distance:
            break
        time.sleep(poll_s)

    try:
        sc.rails_warp_factor = 0
    except Exception:
        pass

    while float(sc.ut) < target_ut:
        time.sleep(0.01)
        
    stop_warp(sc)


def read_all_body_states(
    sc: Any,
    bodies: Any,
    body_names: List[str],
    frame: Any,
    transform_mode: str,
) -> Tuple[float, Dict[str, Dict[str, Vec3]], List[str]]:
    """Read position/velocity of all bodies directly."""
    actual_ut = float(sc.ut)
    states: Dict[str, Dict[str, Vec3]] = {}
    failed: List[str] = []

    for name in body_names:
        body = get_body(bodies, name)
        try:
            p = transform_vec(vec3(body.position(frame)), transform_mode)
            v = transform_vec(vec3(body.velocity(frame)), transform_mode)
            states[name] = {"r": p, "v": v}
        except Exception:
            failed.append(name)

    return actual_ut, states, failed


def validate_double_read(
    a_ut: float,
    a_states: Dict[str, Dict[str, Vec3]],
    b_ut: float,
    b_states: Dict[str, Dict[str, Vec3]],
    max_kinematic_error_m: float,
    max_bad_bodies: int,
) -> Tuple[bool, float, str, int, Dict[str, float]]:
    """Check p2 ~= p1 + v1*dt for each body in two stabilized reads."""
    dt = max(0.0, b_ut - a_ut)
    errors: Dict[str, float] = {}
    max_err = 0.0
    max_body = ""
    bad = 0

    common = sorted(set(a_states.keys()) & set(b_states.keys()))
    for name in common:
        p1 = a_states[name]["r"]
        v1 = a_states[name]["v"]
        p2 = b_states[name]["r"]
        pred = add3(p1, mul3(v1, dt))
        err = norm3(sub3(p2, pred))
        errors[name] = err
        if err > max_err:
            max_err = err
            max_body = name
        if err > max_kinematic_error_m:
            bad += 1

    ok = bad <= max_bad_bodies
    return ok, max_err, max_body, bad, errors


def body_catalog(bodies: Any, body_names: List[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for name in body_names:
        b = get_body(bodies, name)
        entry: Dict[str, Any] = {"name": name}
        for attr in [
            "gravitational_parameter",
            "mass",
            "equatorial_radius",
            "radius",
            "rotational_period",
            "sphere_of_influence",
        ]:
            try:
                entry[attr] = float(getattr(b, attr))
            except Exception:
                pass
        try:
            entry["reference_frame"] = str(b.reference_frame)
        except Exception:
            pass
        out[name] = entry
    return out


def write_relative_monitor(
    writer: csv.DictWriter,
    et: float,
    sample_index: int,
    states: Dict[str, Dict[str, Vec3]],
    pairs: Iterable[Tuple[str, str]],
) -> None:
    for parent, child in pairs:
        if parent not in states or child not in states:
            continue
        rel = sub3(states[child]["r"], states[parent]["r"])
        writer.writerow({
            "sample_index": sample_index,
            "et_seconds": f"{et:.16e}",
            "parent": parent,
            "child": child,
            "rel_distance_m": f"{norm3(rel):.16e}",
            "rel_x_m": f"{rel[0]:.16e}",
            "rel_y_m": f"{rel[1]:.16e}",
            "rel_z_m": f"{rel[2]:.16e}",
        })


def parse_pairs(texts: List[str]) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    for text in texts:
        if ":" not in text:
            continue
        parent, children = text.split(":", 1)
        parent = parent.strip()
        for child in children.split(","):
            child = child.strip()
            if parent and child:
                pairs.append((parent, child))
    return pairs


def main() -> None:
    ap = argparse.ArgumentParser(description="Reliable stop-settle-read observer for KSP/Principia via kRPC.")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--reference-body", default="Sun")
    ap.add_argument("--duration-seconds", type=float, default=None)
    ap.add_argument("--duration-days", type=float, default=None)
    ap.add_argument("--day-seconds", type=float, default=86400.0)
    ap.add_argument("--step-seconds", type=float, default=21600.0)
    ap.add_argument("--start-delay-seconds", type=float, default=0.0)
    ap.add_argument("--include-start", action="store_true", help="Also sample at start+start_delay; default begins at +step.")
    ap.add_argument("--rails-warp-factor", type=int, default=6)
    ap.add_argument("--prestop-margin-seconds", type=float, default=0.0,
                    help="If >0, warp to target-margin then wait at 1x until target. More stable but slower.")
    ap.add_argument("--settle-seconds", type=float, default=5.0,
                    help="Real seconds to wait at 1x before first read.")
    ap.add_argument("--verify-dt-seconds", type=float, default=0.25,
                    help="Real seconds between two validation reads.")
    ap.add_argument("--max-kinematic-error-m", type=float, default=1000.0)
    ap.add_argument("--max-bad-bodies", type=int, default=0)
    ap.add_argument("--max-sample-attempts", type=int, default=3)
    ap.add_argument("--retry-settle-seconds", type=float, default=2.0)
    ap.add_argument("--transform-mode", 
                choices=["raw", "swap_yz", "negate_z", "jnsq_canonical"], # Adicione aqui
                default="raw")
    ap.add_argument("--right-handed-export", action="store_true",
                    help="Deprecated alias for --transform-mode swap_yz. Prefer explicit --transform-mode.")
    ap.add_argument("--family", action="append", default=[],
                    help="Relative monitor family, e.g. Kerbin:Mun,Minmus or Jool:Laythe,Vall,Tylo.")
    ap.add_argument("--connection-name", default="Principia_Observer_V3")
    args = ap.parse_args()

    if args.duration_seconds is None:
        if args.duration_days is None:
            raise SystemExit("Forneça --duration-seconds ou --duration-days.")
        args.duration_seconds = float(args.duration_days) * float(args.day_seconds)

    if args.right_handed_export and args.transform_mode == "raw":
        args.transform_mode = "swap_yz"

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    conn = krpc.connect(name=args.connection_name)
    try:
        sc = conn.space_center
        bodies = sc.bodies
        body_names = get_body_names(bodies)
        reference_body = pick_reference_body(bodies, args.reference_body)
        frame = reference_body.non_rotating_reference_frame

        start_ut = float(sc.ut) + args.start_delay_seconds
        end_ut = start_ut + float(args.duration_seconds)

        # Warmup at 1x. This read is intentionally discarded to initialize streams/caches.
        stop_warp(sc, sleep_s=0.2)
        if args.settle_seconds > 0:
            time.sleep(args.settle_seconds)
        warm_ut, _, warm_failed = read_all_body_states(sc, bodies, body_names, frame, args.transform_mode)

        states_path = outdir / "states.csv"
        log_path = outdir / "sample_log.csv"
        rel_path = outdir / "relative_monitor.csv"
        catalog_path = outdir / "body_catalog.json"
        manifest_path = outdir / "manifest.json"

        with catalog_path.open("w", encoding="utf-8") as f:
            json.dump(body_catalog(bodies, body_names), f, indent=2, ensure_ascii=False)

        state_fields = [
            "sample_index", "target_ut_s", "actual_ut_s", "et_seconds", "body",
            "x_m", "y_m", "z_m", "vx_m_s", "vy_m_s", "vz_m_s",
        ]
        log_fields = [
            "sample_index", "target_ut_s", "actual_ut_s", "lag_s", "accepted",
            "attempts", "ok_bodies", "failed_bodies", "max_kinematic_error_m",
            "max_kinematic_error_body", "bad_kinematic_bodies", "message",
        ]
        rel_fields = [
            "sample_index", "et_seconds", "parent", "child", "rel_distance_m",
            "rel_x_m", "rel_y_m", "rel_z_m",
        ]

        rel_pairs = parse_pairs(args.family)
        accepted_samples = 0
        rejected_samples = 0
        rows_written = 0

        with states_path.open("w", newline="", encoding="utf-8") as sf, \
             log_path.open("w", newline="", encoding="utf-8") as lf, \
             rel_path.open("w", newline="", encoding="utf-8") as rf:
            sw = csv.DictWriter(sf, fieldnames=state_fields)
            lw = csv.DictWriter(lf, fieldnames=log_fields)
            rw = csv.DictWriter(rf, fieldnames=rel_fields)
            sw.writeheader()
            lw.writeheader()
            rw.writeheader()

            sample_index = 0
            target_ut = start_ut if args.include_start else start_ut + float(args.step_seconds)
            
            # --- NOVA VARIÁVEL AQUI ---
            memoria_lag_s = 0.0

            while target_ut <= end_ut + 1e-9:
                warp_target = target_ut
                if args.prestop_margin_seconds > 0:
                    warp_target = max(float(sc.ut), target_ut - args.prestop_margin_seconds)

                warp_to_target(sc, warp_target, args.rails_warp_factor, memoria_lag_s)
                stop_warp(sc, sleep_s=0.2)

                # If we stopped before target, wait in real/1x time until target.
                while float(sc.ut) < target_ut:
                    time.sleep(min(0.25, max(0.01, target_ut - float(sc.ut))))

                accepted = False
                message = ""
                actual_ut = float(sc.ut)
                states_b: Dict[str, Dict[str, Vec3]] = {}
                failed_b: List[str] = []
                attempts_used = 0
                max_err = float("nan")
                max_body = ""
                bad_count = 0

                for attempt in range(1, args.max_sample_attempts + 1):
                    attempts_used = attempt
                    if args.settle_seconds > 0:
                        time.sleep(args.settle_seconds if attempt == 1 else args.retry_settle_seconds)

                    a_ut, states_a, failed_a = read_all_body_states(sc, bodies, body_names, frame, args.transform_mode)
                    if args.verify_dt_seconds > 0:
                        time.sleep(args.verify_dt_seconds)
                    b_ut, states_b, failed_b = read_all_body_states(sc, bodies, body_names, frame, args.transform_mode)

                    ok, max_err, max_body, bad_count, _ = validate_double_read(
                        a_ut, states_a, b_ut, states_b,
                        args.max_kinematic_error_m, args.max_bad_bodies,
                    )

                    actual_ut = b_ut
                    failed = sorted(set(failed_a) | set(failed_b))
                    if failed:
                        message = "failed bodies: " + ",".join(failed)
                        ok = False

                    if ok:
                        accepted = True
                        message = "accepted"
                        break
                    else:
                        message = (
                            f"rejected attempt {attempt}: max_kinematic_error_m={max_err:.6g} "
                            f"body={max_body} bad_bodies={bad_count}"
                        )

                lag = actual_ut - target_ut

                if lag > 0:
                    memoria_lag_s += lag

                lw.writerow({
                    "sample_index": sample_index,
                    "target_ut_s": f"{target_ut:.16e}",
                    "actual_ut_s": f"{actual_ut:.16e}",
                    "lag_s": f"{lag:.16e}",
                    "accepted": int(accepted),
                    "attempts": attempts_used,
                    "ok_bodies": len(states_b),
                    "failed_bodies": len(failed_b),
                    "max_kinematic_error_m": f"{max_err:.16e}" if math.isfinite(max_err) else "",
                    "max_kinematic_error_body": max_body,
                    "bad_kinematic_bodies": bad_count,
                    "message": message,
                })

                if accepted:
                    et = actual_ut
                    for name in body_names:
                        if name not in states_b:
                            continue
                        r = states_b[name]["r"]
                        v = states_b[name]["v"]
                        sw.writerow({
                            "sample_index": sample_index,
                            "target_ut_s": f"{target_ut:.16e}",
                            "actual_ut_s": f"{actual_ut:.16e}",
                            "et_seconds": f"{et:.16e}",
                            "body": name,
                            "x_m": f"{r[0]:.16e}",
                            "y_m": f"{r[1]:.16e}",
                            "z_m": f"{r[2]:.16e}",
                            "vx_m_s": f"{v[0]:.16e}",
                            "vy_m_s": f"{v[1]:.16e}",
                            "vz_m_s": f"{v[2]:.16e}",
                        })
                        rows_written += 1

                    write_relative_monitor(rw, et, sample_index, states_b, rel_pairs)
                    accepted_samples += 1
                else:
                    rejected_samples += 1

                progress = 100.0 * max(0.0, min(1.0, (actual_ut - start_ut) / float(args.duration_seconds)))
                print(
                    f"[{sample_index}] target={target_ut:.2f} actual={actual_ut:.2f} "
                    f"lag={lag:.2f}s accepted={int(accepted)} attempts={attempts_used} "
                    f"max_kin={max_err:.3f}m body={max_body} progress={progress:.1f}%"
                )

                sample_index += 1
                target_ut += float(args.step_seconds)

        manifest = {
            "tool": "fast_principia_observer_v3",
            "design": "warp-as-transport; stop-settle-read-at-1x",
            "output_dir": str(outdir),
            "reference_body": getattr(reference_body, "name", args.reference_body),
            "reference_frame": "reference_body.non_rotating_reference_frame",
            "transform_mode": args.transform_mode,
            "start_ut_s": start_ut,
            "end_ut_s": end_ut,
            "duration_seconds_effective": float(args.duration_seconds),
            "duration_days_input": args.duration_days,
            "day_seconds": args.day_seconds,
            "step_seconds": args.step_seconds,
            "include_start": bool(args.include_start),
            "warmup_ut_s": warm_ut,
            "warmup_failed_bodies": warm_failed,
            "rails_warp_factor": args.rails_warp_factor,
            "prestop_margin_seconds": args.prestop_margin_seconds,
            "settle_seconds": args.settle_seconds,
            "verify_dt_seconds": args.verify_dt_seconds,
            "max_kinematic_error_m": args.max_kinematic_error_m,
            "max_bad_bodies": args.max_bad_bodies,
            "max_sample_attempts": args.max_sample_attempts,
            "accepted_samples": accepted_samples,
            "rejected_samples": rejected_samples,
            "rows_written": rows_written,
            "states_csv": str(states_path),
            "sample_log_csv": str(log_path),
            "relative_monitor_csv": str(rel_path),
            "body_catalog_json": str(catalog_path),
        }
        with manifest_path.open("w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)

        print(json.dumps(manifest, indent=2, ensure_ascii=False))

    finally:
        conn.close()


if __name__ == "__main__":
    main()
