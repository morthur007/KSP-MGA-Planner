#!/usr/bin/env python3
"""
MGA kRPC Route Probe V0.1

Safety-first helper for in-game validation of a precomputed MGA route.
This is NOT yet a full autonomous multi-year route executor.

It can:
  - connect to kRPC and print vessel/orbit telemetry;
  - inspect predicted patched-conic encounters via next_orbit;
  - optionally execute the currently existing maneuver node using a cautious two-phase burn;
  - optionally tune a target body's predicted periapsis altitude after the node burn.

Workflow for now:
  1. Load route packet for metadata/reference.
  2. Create or place a maneuver node manually/in another script.
  3. Run this probe with --execute-current-node to burn it.
  4. Log the predicted encounter/periapsis via kRPC.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

try:
    import krpc
except ImportError:  # pragma: no cover
    krpc = None

Vector = Tuple[float, float, float]


def normalize(v: Vector) -> Vector:
    mag = math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
    if mag == 0:
        return (0.0, 1.0, 0.0)
    return (v[0] / mag, v[1] / mag, v[2] / mag)


def dot(a: Vector, b: Vector) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def angle_between(a: Vector, b: Vector) -> float:
    dp = max(-1.0, min(1.0, dot(normalize(a), normalize(b))))
    return math.degrees(math.acos(dp))


def read_packet(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}
    p = Path(path)
    text = p.read_text(encoding="utf-8").strip()
    if not text:
        return {}
    if p.suffix == ".jsonl":
        return json.loads(text.splitlines()[0])
    return json.loads(text)


def get_sequence(packet: Dict[str, Any]) -> str:
    def scan(x: Any, depth: int = 0) -> Optional[Any]:
        if depth > 8:
            return None
        if isinstance(x, dict):
            for k in ("sequence", "bodies", "body_sequence"):
                if k in x:
                    return x[k]
            for v in x.values():
                got = scan(v, depth + 1)
                if got is not None:
                    return got
        elif isinstance(x, list):
            for v in x:
                got = scan(v, depth + 1)
                if got is not None:
                    return got
        return None
    s = scan(packet)
    if isinstance(s, list):
        return " -> ".join(str(x) for x in s)
    if isinstance(s, str):
        return " -> ".join(x.strip() for x in s.replace(",", "->").split("->") if x.strip())
    return "unknown"


def iter_orbit_patches(vessel, max_patches: int = 20):
    orb = vessel.orbit
    for _ in range(max_patches):
        if orb is None:
            break
        yield orb
        try:
            orb = orb.next_orbit
        except Exception:
            break


def get_target_periapsis(vessel, target_body_name: str) -> Optional[float]:
    for orb in iter_orbit_patches(vessel):
        try:
            if orb.body.name.lower() == target_body_name.lower():
                return orb.periapsis_altitude
        except Exception:
            pass
    return None


def print_patch_chain(vessel, max_patches: int = 12) -> None:
    print("\nPredicted orbit patch chain:")
    for i, orb in enumerate(iter_orbit_patches(vessel, max_patches=max_patches), start=1):
        try:
            print(
                f" {i:02d}. body={orb.body.name:<12} "
                f"Pe={orb.periapsis_altitude/1000:12.3f} km "
                f"Ap={orb.apoapsis_altitude/1000:12.3f} km "
                f"period={getattr(orb, 'period', float('nan')):12.3f} s"
            )
        except Exception as e:
            print(f" {i:02d}. [could not read patch: {e}]")


def active_engines_with_fuel(vessel):
    return [e for e in vessel.parts.engines if e.active and e.has_fuel]


def estimate_burn_time(vessel, node) -> float:
    m0 = vessel.mass
    isp = vessel.specific_impulse
    dv = node.delta_v
    thrust = vessel.available_thrust
    g0 = 9.80665
    if thrust <= 0 or isp <= 0 or dv <= 0:
        return 0.0
    mdot = thrust / (isp * g0)
    m1 = m0 / math.exp(dv / (isp * g0))
    return max(0.0, (m0 - m1) / mdot)


def execute_current_node(
    conn,
    target_body_name: Optional[str] = None,
    target_pe_altitude_m: Optional[float] = None,
    fine_tune: bool = False,
    pe_tolerance_m: float = 50.0,
) -> None:
    sc = conn.space_center
    vessel = sc.active_vessel
    ap = vessel.auto_pilot

    if len(vessel.control.nodes) == 0:
        raise RuntimeError("No maneuver node found on active vessel.")
    node = vessel.control.nodes[0]

    engines = active_engines_with_fuel(vessel)
    original_limits = {e: e.thrust_limit for e in engines}

    rf = vessel.orbit.body.reference_frame
    ap.reference_frame = rf
    ap.engage()

    burn_time = estimate_burn_time(vessel, node)
    print(f"Estimated burn time: {burn_time:.3f} s | node Δv={node.delta_v:.3f} m/s")

    if node.ut - sc.ut > burn_time / 2.0 + 5.0:
        print("Warping to maneuver...")
        sc.warp_to(node.ut - burn_time / 2.0 - 5.0)

    while node.time_to > burn_time / 2.0:
        ap.target_direction = node.remaining_burn_vector(rf)
        time.sleep(0.05)

    print("IGNITION: coarse node execution")
    locked_vector = None
    last_vec = node.remaining_burn_vector(rf)
    try:
        while True:
            remaining = node.remaining_delta_v
            if remaining < 1.0 and locked_vector is None:
                locked_vector = node.remaining_burn_vector(rf)
                print(f"[node] locked vector below 1 m/s: {locked_vector}")
            target_vec = locked_vector if locked_vector else node.remaining_burn_vector(rf)
            last_vec = target_vec
            ap.target_direction = target_vec

            current_dir = vessel.flight(rf).direction
            if angle_between(current_dir, target_vec) > 2.0:
                vessel.control.throttle = 0.0
                time.sleep(0.02)
                continue

            if remaining < 0.05:
                vessel.control.throttle = 0.0
                break

            twr = vessel.available_thrust / max(vessel.mass * 9.80665, 1e-9)
            if remaining < twr / 3:
                throttle = 0.05
            elif remaining < twr / 2:
                throttle = 0.10
            elif remaining < twr:
                throttle = 0.25
            else:
                throttle = 1.0
            vessel.control.throttle = throttle
            time.sleep(0.01)
    finally:
        vessel.control.throttle = 0.0
        try:
            node.remove()
        except Exception:
            pass

    print("Node complete. Waiting for thrust spool-down...")
    while vessel.thrust > 0.01:
        time.sleep(0.05)
    time.sleep(0.5)

    if fine_tune and target_body_name and target_pe_altitude_m is not None:
        print(f"\nFine-tuning predicted Pe at {target_body_name} to {target_pe_altitude_m/1000:.3f} km")
        pe_now = get_target_periapsis(vessel, target_body_name)
        if pe_now is None:
            print(f"[WARN] No predicted encounter with {target_body_name}; fine tune skipped.")
        else:
            diff = pe_now - target_pe_altitude_m
            forward = normalize(last_vec)
            backward = (-forward[0], -forward[1], -forward[2])
            tune_vec = backward if diff < 0 else forward
            ap.target_direction = tune_vec
            while angle_between(vessel.flight(rf).direction, tune_vec) > 0.5:
                time.sleep(0.05)
            for e in engines:
                e.thrust_limit = 0.01
            best = abs(diff)
            worsening = 0
            while True:
                pe_now = get_target_periapsis(vessel, target_body_name)
                if pe_now is None:
                    print(f"[WARN] Lost predicted encounter with {target_body_name}; stopping.")
                    break
                err = abs(pe_now - target_pe_altitude_m)
                print(f"[fine] Pe={pe_now/1000:.3f} km | err={err:.1f} m")
                if err <= pe_tolerance_m:
                    break
                if err > best + 2.0:
                    worsening += 1
                else:
                    worsening = 0
                    best = min(best, err)
                if worsening > 5:
                    print("[fine] Error worsening; stopping burn.")
                    break
                if angle_between(vessel.flight(rf).direction, tune_vec) > 1.0:
                    vessel.control.throttle = 0.0
                else:
                    vessel.control.throttle = 1.0
                time.sleep(0.02)
            vessel.control.throttle = 0.0
            while vessel.thrust > 0.01:
                time.sleep(0.05)

    for e, limit in original_limits.items():
        try:
            e.thrust_limit = limit
        except Exception:
            pass
    ap.disengage()
    print_patch_chain(vessel)


def main() -> int:
    ap = argparse.ArgumentParser(description="kRPC visual/telemetry probe for an MGA route packet.")
    ap.add_argument("--packet", help="Route packet JSON/JSONL for metadata only")
    ap.add_argument("--connect-name", default="MGA Route Probe")
    ap.add_argument("--target-body", help="Body to inspect/tune predicted periapsis for, e.g. Eve")
    ap.add_argument("--target-pe-altitude-m", type=float)
    ap.add_argument("--execute-current-node", action="store_true")
    ap.add_argument("--fine-tune-pe", action="store_true")
    ap.add_argument("--pe-tolerance-m", type=float, default=50.0)
    ap.add_argument("--max-patches", type=int, default=12)
    args = ap.parse_args()

    if krpc is None:
        raise RuntimeError("krpc is not installed/importable in this Python environment.")

    packet = read_packet(args.packet)
    print("=" * 80)
    print("MGA kRPC ROUTE PROBE V0.1")
    print("=" * 80)
    if packet:
        print(f"Packet sequence: {get_sequence(packet)}")
    conn = krpc.connect(name=args.connect_name)
    sc = conn.space_center
    vessel = sc.active_vessel
    print(f"UT:      {sc.ut:.3f} s")
    print(f"Vessel:  {vessel.name}")
    print(f"Body:    {vessel.orbit.body.name}")
    print(f"Mass:    {vessel.mass:.3f} kg")
    print(f"Thrust:  {vessel.available_thrust:.3f} N")
    print_patch_chain(vessel, max_patches=args.max_patches)

    if args.target_body:
        pe = get_target_periapsis(vessel, args.target_body)
        if pe is None:
            print(f"No predicted encounter with {args.target_body} in current patch chain.")
        else:
            print(f"Predicted Pe at {args.target_body}: {pe/1000:.3f} km")

    if args.execute_current_node:
        execute_current_node(
            conn,
            target_body_name=args.target_body,
            target_pe_altitude_m=args.target_pe_altitude_m,
            fine_tune=args.fine_tune_pe,
            pe_tolerance_m=args.pe_tolerance_m,
        )
    else:
        print("Dry telemetry only. Use --execute-current-node to burn the active maneuver node.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
