#!/usr/bin/env python3
"""
principia_jool_stability_sentinel.py

Fast stability gate for KSP + Principia planet packs.

Goal
----
Do NOT build a full ephemeris.  This is a quick certification probe for one
planetary subsystem, typically Jool + moons, after changing Kopernicus/
ModuleManager patches.

It warps through time, samples only the parent and selected moons, and stops as
soon as the system violates configurable stability criteria:

- moon distance from parent grows above escape_multiple * its initial distance;
- moon distance exceeds a fraction of the parent's KSP sphere of influence;
- moon distance becomes non-finite;
- pairwise moon separation drops below a safety multiple of radii.

This is intentionally much faster than ephemeris acquisition because it samples
few bodies, fewer epochs, and stops early on failure.

Notes
-----
- kRPC is treated as an observatory of the live game state, not as a solver.
- We avoid body.orbit as truth.  The main stability metrics are direct positions
  in parent.non_rotating_reference_frame.
- Velocity/osculating elements are optional diagnostics only.
- KSP/Principia cannot be rewound by kRPC.  Adaptive refinement backward in time
  requires saving before the run and rerunning with a smaller step.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import krpc  # type: ignore
except ImportError as exc:
    raise SystemExit("Instale o pacote 'krpc' no Python que conecta ao KSP.") from exc

Vec3 = Tuple[float, float, float]


def vec3(x: Any) -> Vec3:
    return (float(x[0]), float(x[1]), float(x[2]))


def norm3(v: Vec3) -> float:
    return math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])


def sub3(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def safe_get(obj: Any, attr: str, default: Any = None) -> Any:
    try:
        return getattr(obj, attr)
    except Exception:
        return default


def classify_distance(r: float, r0: float, parent_soi: Optional[float], args: argparse.Namespace) -> Tuple[str, str]:
    if not math.isfinite(r):
        return "FAIL", "non_finite_distance"

    if r0 > 0 and r > args.escape_multiple * r0:
        return "FAIL", f"distance_gt_{args.escape_multiple:g}x_initial"

    if parent_soi and math.isfinite(parent_soi) and parent_soi > 0:
        if r > args.parent_soi_fraction_fail * parent_soi:
            return "FAIL", f"distance_gt_{args.parent_soi_fraction_fail:g}_parent_soi"
        if r > args.parent_soi_fraction_warn * parent_soi:
            return "WARN", f"distance_gt_{args.parent_soi_fraction_warn:g}_parent_soi"

    if r0 > 0 and r > args.warn_multiple * r0:
        return "WARN", f"distance_gt_{args.warn_multiple:g}x_initial"

    return "OK", ""


def wait_for_ut(sc: Any, target_ut: float, timeout_s: float, poll_s: float, tolerance_s: float, post_pause_s: float) -> Dict[str, float]:
    sc.warp_to(target_ut)
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        ut = float(sc.ut)
        warp_rate = float(safe_get(sc, "warp_rate", 1.0) or 1.0)
        if abs(ut - target_ut) <= tolerance_s and warp_rate <= 1.01:
            break
        time.sleep(poll_s)
    if post_pause_s > 0:
        time.sleep(post_pause_s)
    return {
        "target_ut_s": target_ut,
        "actual_ut_s": float(sc.ut),
        "ut_error_s": float(sc.ut) - target_ut,
        "warp_rate": float(safe_get(sc, "warp_rate", 1.0) or 1.0),
    }


def make_streams(conn: Any, bodies: Dict[str, Any], frame: Any) -> Dict[str, Any]:
    streams: Dict[str, Any] = {}
    for name, body in bodies.items():
        try:
            streams[name] = conn.add_stream(body.position, frame)
        except Exception:
            pass
    return streams


def read_positions(bodies: Dict[str, Any], streams: Dict[str, Any], frame: Any) -> Dict[str, Vec3]:
    out: Dict[str, Vec3] = {}
    for name, body in bodies.items():
        try:
            raw = streams[name]() if name in streams else body.position(frame)
            out[name] = vec3(raw)
        except Exception:
            pass
    return out


def maybe_osculating(parent_mu: float, moon_mu: float, r_vec: Vec3, v_vec: Vec3) -> Dict[str, Optional[float]]:
    """Diagnostic two-body osculating elements from instantaneous r/v.

    This is not used as the primary stability criterion because kRPC velocity can
    be noisy under Principia.  Still useful as a rough indicator if available.
    """
    mu = parent_mu + moon_mu
    r = norm3(r_vec)
    v2 = v_vec[0] ** 2 + v_vec[1] ** 2 + v_vec[2] ** 2
    if r <= 0 or mu <= 0:
        return {"specific_energy_j_kg": None, "semi_major_axis_m": None}
    eps = 0.5 * v2 - mu / r
    a = -mu / (2.0 * eps) if abs(eps) > 0 else math.inf
    return {"specific_energy_j_kg": eps, "semi_major_axis_m": a}


@dataclass
class MoonStats:
    initial_r_m: float
    max_r_m: float
    min_r_m: float
    max_ratio: float
    first_warn_ut_s: Optional[float] = None
    first_fail_ut_s: Optional[float] = None
    first_fail_reason: Optional[str] = None


def main() -> int:
    p = argparse.ArgumentParser(description="Fast live stability gate for a Principia subsystem")
    p.add_argument("--parent", default="Jool")
    p.add_argument("--moons", nargs="+", default=["Laythe", "Vall", "Tylo", "Bop", "Pol"])
    p.add_argument("--duration-days", type=float, default=90.0)
    p.add_argument("--step-hours", type=float, default=12.0)
    p.add_argument("--output-dir", type=Path, default=Path("data/jool_stability_sentinel"))
    p.add_argument("--escape-multiple", type=float, default=3.0, help="FAIL if r > this * initial moon-parent distance")
    p.add_argument("--warn-multiple", type=float, default=1.5, help="WARN if r > this * initial moon-parent distance")
    p.add_argument("--parent-soi-fraction-warn", type=float, default=0.25)
    p.add_argument("--parent-soi-fraction-fail", type=float, default=0.75)
    p.add_argument("--min-separation-radius-multiple", type=float, default=3.0, help="FAIL if moon-moon distance < factor*(r1+r2)")
    p.add_argument("--max-rails-rate", type=float, default=100000.0)
    p.add_argument("--max-physics-rate", type=float, default=2.0)
    p.add_argument("--settle-timeout-s", type=float, default=20.0)
    p.add_argument("--settle-poll-s", type=float, default=0.10)
    p.add_argument("--post-warp-pause-s", type=float, default=0.15)
    p.add_argument("--ut-tolerance-s", type=float, default=0.5)
    p.add_argument("--stop-on-fail", action="store_true", default=True)
    p.add_argument("--do-not-stop-on-fail", dest="stop_on_fail", action="store_false")
    p.add_argument("--include-velocity-diagnostic", action="store_true")
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    samples_csv = args.output_dir / "stability_samples.csv"
    pairs_csv = args.output_dir / "pairwise_separations.csv"
    summary_json = args.output_dir / "stability_summary.json"

    conn = krpc.connect(name="Jool_Stability_Sentinel")
    try:
        sc = conn.space_center
        # Keep this explicit; warp_to may use these limits.
        try:
            sc.maximum_rails_warp_factor = args.max_rails_rate
        except Exception:
            pass

        bodies_obj = sc.bodies
        if args.parent not in bodies_obj:
            raise SystemExit(f"Parent body not found: {args.parent}")

        parent_body = bodies_obj[args.parent]
        frame = parent_body.non_rotating_reference_frame

        selected_names = [args.parent] + [m for m in args.moons if m in bodies_obj]
        missing = [m for m in args.moons if m not in bodies_obj]
        if missing:
            print(f"[WARN] Moons not found and skipped: {', '.join(missing)}")

        bodies = {name: bodies_obj[name] for name in selected_names}
        streams = make_streams(conn, bodies, frame)

        parent_soi = safe_get(parent_body, "sphere_of_influence", None)
        parent_mu = float(safe_get(parent_body, "gravitational_parameter", 0.0) or 0.0)
        radii = {name: float(safe_get(b, "equatorial_radius", 0.0) or 0.0) for name, b in bodies.items()}
        mus = {name: float(safe_get(b, "gravitational_parameter", 0.0) or 0.0) for name, b in bodies.items()}

        start_ut = float(sc.ut)
        step_s = args.step_hours * 3600.0
        n_steps = int(math.floor(args.duration_days * 86400.0 / step_s))
        target_uts = [start_ut + i * step_s for i in range(n_steps + 1)]

        # Initial sample.
        positions0 = read_positions(bodies, streams, frame)
        initial_r: Dict[str, float] = {}
        stats: Dict[str, MoonStats] = {}
        for moon in args.moons:
            if moon not in positions0:
                continue
            r0 = norm3(positions0[moon])
            initial_r[moon] = r0
            stats[moon] = MoonStats(initial_r_m=r0, max_r_m=r0, min_r_m=r0, max_ratio=1.0)

        manifest = {
            "schema": "principia_subsystem_stability_sentinel.v1",
            "purpose": "Fast live stability gate; not a full ephemeris acquisition.",
            "parent": args.parent,
            "moons": list(stats.keys()),
            "start_ut_s": start_ut,
            "duration_days_requested": args.duration_days,
            "step_hours": args.step_hours,
            "parent_soi_m": parent_soi,
            "criteria": {
                "escape_multiple": args.escape_multiple,
                "warn_multiple": args.warn_multiple,
                "parent_soi_fraction_warn": args.parent_soi_fraction_warn,
                "parent_soi_fraction_fail": args.parent_soi_fraction_fail,
                "min_separation_radius_multiple": args.min_separation_radius_multiple,
            },
            "initial_distances_m": initial_r,
            "body_mu_m3_s2": mus,
            "body_radius_m": radii,
        }

        sample_fields = [
            "sample_index", "target_ut_s", "actual_ut_s", "day", "body",
            "r_parent_m", "r_over_initial", "status", "reason",
            "x_m", "y_m", "z_m",
        ]
        pair_fields = [
            "sample_index", "actual_ut_s", "day", "body_a", "body_b",
            "separation_m", "separation_over_radii", "status", "reason",
        ]

        failure_events: List[Dict[str, Any]] = []
        warning_events: List[Dict[str, Any]] = []

        with samples_csv.open("w", newline="", encoding="utf-8") as sf, pairs_csv.open("w", newline="", encoding="utf-8") as pf:
            sw = csv.DictWriter(sf, fieldnames=sample_fields)
            pw = csv.DictWriter(pf, fieldnames=pair_fields)
            sw.writeheader()
            pw.writeheader()

            print("=== Principia subsystem stability sentinel ===")
            print(f"Parent: {args.parent}; moons: {', '.join(stats.keys())}")
            print(f"Duration: {args.duration_days:g} d; step: {args.step_hours:g} h; samples: {len(target_uts)}")
            print("This is a fast stability gate, not a dense ephemeris.")

            for i, target_ut in enumerate(target_uts):
                if i == 0:
                    settle = {"actual_ut_s": start_ut, "ut_error_s": 0.0, "warp_rate": 1.0}
                else:
                    # kRPC warp_to supports max rates in many versions, but not all wrappers.
                    try:
                        sc.warp_to(target_ut, max_rails_rate=args.max_rails_rate, max_physics_rate=args.max_physics_rate)
                        deadline = time.time() + args.settle_timeout_s
                        while time.time() < deadline:
                            ut = float(sc.ut)
                            warp_rate = float(safe_get(sc, "warp_rate", 1.0) or 1.0)
                            if abs(ut - target_ut) <= args.ut_tolerance_s and warp_rate <= 1.01:
                                break
                            time.sleep(args.settle_poll_s)
                        if args.post_warp_pause_s > 0:
                            time.sleep(args.post_warp_pause_s)
                        settle = {
                            "target_ut_s": target_ut,
                            "actual_ut_s": float(sc.ut),
                            "ut_error_s": float(sc.ut) - target_ut,
                            "warp_rate": float(safe_get(sc, "warp_rate", 1.0) or 1.0),
                        }
                    except TypeError:
                        settle = wait_for_ut(sc, target_ut, args.settle_timeout_s, args.settle_poll_s, args.ut_tolerance_s, args.post_warp_pause_s)

                actual_ut = float(settle["actual_ut_s"])
                day = (actual_ut - start_ut) / 86400.0
                positions = read_positions(bodies, streams, frame)

                step_failed = False
                for moon, st in stats.items():
                    if moon not in positions:
                        status, reason = "FAIL", "position_unavailable"
                        r = math.nan
                        ratio = math.nan
                    else:
                        r = norm3(positions[moon])
                        ratio = r / st.initial_r_m if st.initial_r_m > 0 else math.nan
                        st.max_r_m = max(st.max_r_m, r)
                        st.min_r_m = min(st.min_r_m, r)
                        st.max_ratio = max(st.max_ratio, ratio if math.isfinite(ratio) else st.max_ratio)
                        status, reason = classify_distance(r, st.initial_r_m, parent_soi, args)

                    if status == "WARN" and st.first_warn_ut_s is None:
                        st.first_warn_ut_s = actual_ut
                        warning_events.append({"ut_s": actual_ut, "day": day, "body": moon, "reason": reason, "r_m": r, "ratio": ratio})
                    if status == "FAIL" and st.first_fail_ut_s is None:
                        st.first_fail_ut_s = actual_ut
                        st.first_fail_reason = reason
                        failure_events.append({"ut_s": actual_ut, "day": day, "body": moon, "reason": reason, "r_m": r, "ratio": ratio})
                        step_failed = True

                    pos = positions.get(moon, (math.nan, math.nan, math.nan))
                    sw.writerow({
                        "sample_index": i,
                        "target_ut_s": f"{target_ut:.9f}",
                        "actual_ut_s": f"{actual_ut:.9f}",
                        "day": f"{day:.9f}",
                        "body": moon,
                        "r_parent_m": f"{r:.16e}" if math.isfinite(r) else "nan",
                        "r_over_initial": f"{ratio:.16e}" if math.isfinite(ratio) else "nan",
                        "status": status,
                        "reason": reason,
                        "x_m": f"{pos[0]:.16e}",
                        "y_m": f"{pos[1]:.16e}",
                        "z_m": f"{pos[2]:.16e}",
                    })

                # Pairwise close approach checks.
                moon_list = [m for m in stats if m in positions]
                for ia in range(len(moon_list)):
                    for ib in range(ia + 1, len(moon_list)):
                        a, b = moon_list[ia], moon_list[ib]
                        sep = norm3(sub3(positions[a], positions[b]))
                        denom = radii.get(a, 0.0) + radii.get(b, 0.0)
                        sep_over_radii = sep / denom if denom > 0 else math.inf
                        status = "OK"
                        reason = ""
                        if sep_over_radii < args.min_separation_radius_multiple:
                            status = "FAIL"
                            reason = "moon_moon_close_approach"
                            failure_events.append({"ut_s": actual_ut, "day": day, "body": f"{a}-{b}", "reason": reason, "separation_m": sep, "separation_over_radii": sep_over_radii})
                            step_failed = True
                        pw.writerow({
                            "sample_index": i,
                            "actual_ut_s": f"{actual_ut:.9f}",
                            "day": f"{day:.9f}",
                            "body_a": a,
                            "body_b": b,
                            "separation_m": f"{sep:.16e}",
                            "separation_over_radii": f"{sep_over_radii:.16e}" if math.isfinite(sep_over_radii) else "inf",
                            "status": status,
                            "reason": reason,
                        })

                sf.flush(); pf.flush()
                worst = max((st.max_ratio for st in stats.values()), default=1.0)
                print(f"[{i+1:04d}/{len(target_uts):04d}] day={day:8.3f} worst_r/r0={worst:8.3f} failures={len(failure_events)}", flush=True)

                if step_failed and args.stop_on_fail:
                    print("[STOP] Failure criterion reached; stopping early.")
                    break

        summary = {
            "manifest": manifest,
            "status": "FAIL" if failure_events else "PASS",
            "warnings": warning_events,
            "failures": failure_events,
            "stats_by_moon": {name: asdict(st) for name, st in stats.items()},
            "files": {
                "samples_csv": str(samples_csv),
                "pairwise_separations_csv": str(pairs_csv),
                "summary_json": str(summary_json),
            },
        }
        summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

        print("\n=== Summary ===")
        print(f"Status: {summary['status']}")
        if failure_events:
            first = failure_events[0]
            print(f"First failure: day={first.get('day'):.3f}, body={first.get('body')}, reason={first.get('reason')}")
        print(f"Wrote: {summary_json}")
        return 1 if failure_events else 0

    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
