#!/usr/bin/env python3
"""
MGA kRPC Create Departure Node V0.1

Safe first bridge from an offline MGA route packet to a KSP/kRPC maneuver node.

Scope V0.1:
- Connects to kRPC and reads the active vessel/parking orbit.
- Loads an optional B6D/route packet and tries to infer a departure v_inf magnitude.
- Creates an approximate prograde ejection node at next periapsis, sized to achieve
  the requested hyperbolic excess speed from the current parking orbit.

Important limitation:
- This does NOT yet target the full heliocentric v_inf vector/B-plane direction.
- It is meant to create a practical first departure node so the existing node executor
  and kRPC patch-chain telemetry can be used for visual/game validation.
- A V0.2 should solve the parking-orbit injection geometry to match the desired v_inf vector.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable


def norm(v):
    return math.sqrt(sum(float(x) * float(x) for x in v))


def finite(x: Any) -> bool:
    try:
        return math.isfinite(float(x))
    except Exception:
        return False


def load_json(path: str | None) -> Any:
    if not path:
        return None
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def walk(obj: Any, path: str = "") -> Iterable[tuple[str, Any]]:
    yield path, obj
    if isinstance(obj, dict):
        for k, v in obj.items():
            p = f"{path}.{k}" if path else str(k)
            yield from walk(v, p)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            p = f"{path}[{i}]"
            yield from walk(v, p)


def find_vinf_candidates(packet: Any) -> list[dict[str, Any]]:
    """Find plausible v_inf values/vectors in arbitrary route-packet schemas."""
    out: list[dict[str, Any]] = []
    if packet is None:
        return out

    for path, obj in walk(packet):
        key = path.lower()
        if "vinf" not in key and "v_inf" not in key and "hyperbolic" not in key:
            continue

        if isinstance(obj, (int, float)) and finite(obj):
            val = float(obj)
            # Guess units by magnitude and key name.
            if "m_s" in key or "mps" in key or "m/s" in key:
                val_m_s = val
            elif "km_s" in key or "kmps" in key or "km/s" in key:
                val_m_s = val * 1000.0
            else:
                # Most project internals use km/s for vectors and m/s for mismatch.
                # v_inf magnitude for interplanetary likely < 20 km/s if value is small.
                val_m_s = val * 1000.0 if abs(val) < 100.0 else val
            if 1.0 <= abs(val_m_s) <= 50000.0:
                out.append({"path": path, "kind": "scalar", "value_m_s": abs(val_m_s), "raw": obj})

        elif isinstance(obj, (list, tuple)) and len(obj) == 3 and all(finite(x) for x in obj):
            vals = [float(x) for x in obj]
            mag = norm(vals)
            if "m_s" in key or "mps" in key or "m/s" in key:
                mag_m_s = mag
            elif "km_s" in key or "kmps" in key or "km/s" in key:
                mag_m_s = mag * 1000.0
            else:
                mag_m_s = mag * 1000.0 if mag < 100.0 else mag
            if 1.0 <= mag_m_s <= 50000.0:
                out.append({"path": path, "kind": "vector", "value_m_s": mag_m_s, "raw": vals})

    # Prefer departure / first-segment / Kerbin-ish paths, avoid final-arrival mismatch.
    def score(c):
        p = c["path"].lower()
        s = 0
        for token in ("depart", "departure", "launch", "seg0", "segments[0]", "kerbin", "origin"):
            if token in p:
                s -= 10
        for token in ("final", "arrival", "jool", "mismatch", "diagnostic", "after"):
            if token in p:
                s += 5
        # Prefer plausible route v_inf 500..8000 m/s over tiny mismatch values.
        v = c["value_m_s"]
        if 500 <= v <= 10000:
            s -= 3
        if v < 100:
            s += 20
        return s

    out.sort(key=score)
    return out


def pick_vinf(packet: Any, explicit_vinf_m_s: float | None, candidate_index: int) -> tuple[float, list[dict[str, Any]], str]:
    if explicit_vinf_m_s is not None:
        return float(explicit_vinf_m_s), [], "explicit --vinf-m-s"
    candidates = find_vinf_candidates(packet)
    if not candidates:
        raise RuntimeError(
            "Could not infer departure v_inf from packet. Pass --vinf-m-s explicitly."
        )
    if candidate_index < 0 or candidate_index >= len(candidates):
        raise RuntimeError(f"--vinf-candidate-index out of range: {candidate_index}; candidates={len(candidates)}")
    c = candidates[candidate_index]
    return float(c["value_m_s"]), candidates, c["path"]


def compute_node_delta_v(vessel, space_center, body, vinf_m_s: float, burn_at: str, safety_margin_m_s: float):
    orbit = vessel.orbit
    mu = float(body.gravitational_parameter)

    # Burn at next periapsis by default. This is the usual lowest-cost place to escape.
    if burn_at == "periapsis":
        r = float(orbit.periapsis)  # distance from body center, m
        a = float(orbit.semi_major_axis)
        ut = float(space_center.ut + orbit.time_to_periapsis)
        if math.isfinite(a) and abs(a) > 1.0:
            v_parking = math.sqrt(max(0.0, mu * (2.0 / r - 1.0 / a)))
        else:
            v_parking = float(orbit.speed)
    else:
        # Current point fallback.
        r = float(orbit.radius)
        v_parking = float(orbit.speed)
        ut = float(space_center.ut + 60.0)

    v_escape_at_r = math.sqrt(2.0 * mu / r)
    v_hyp_periapsis = math.sqrt(vinf_m_s * vinf_m_s + v_escape_at_r * v_escape_at_r)
    dv_prograde = max(0.0, v_hyp_periapsis - v_parking + safety_margin_m_s)

    return {
        "ut": ut,
        "r_m": r,
        "mu_m3_s2": mu,
        "v_parking_m_s": v_parking,
        "v_escape_at_r_m_s": v_escape_at_r,
        "v_hyp_periapsis_m_s": v_hyp_periapsis,
        "dv_prograde_m_s": dv_prograde,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Create an approximate kRPC departure maneuver node from an MGA route packet.")
    ap.add_argument("--packet", help="Route/B6D packet JSON. Optional if --vinf-m-s is provided.")
    ap.add_argument("--vinf-m-s", type=float, help="Explicit desired hyperbolic excess speed at departure, m/s.")
    ap.add_argument("--vinf-candidate-index", type=int, default=0, help="Which inferred packet v_inf candidate to use.")
    ap.add_argument("--burn-at", choices=["periapsis", "now"], default="periapsis")
    ap.add_argument("--safety-margin-m-s", type=float, default=0.0, help="Extra prograde m/s added to node.")
    ap.add_argument("--dry-run", action="store_true", help="Print computed node but do not create it.")
    ap.add_argument("--list-vinf-candidates", action="store_true", help="List packet v_inf candidates and exit before node creation.")
    ap.add_argument("--remove-existing-nodes", action="store_true", help="Remove existing maneuver nodes before creating the new one.")
    ap.add_argument("--connection-name", default="MGA Departure Node Creator")
    args = ap.parse_args(argv)

    packet = load_json(args.packet) if args.packet else None
    vinf_m_s, candidates, source = pick_vinf(packet, args.vinf_m_s, args.vinf_candidate_index)

    print("=" * 80)
    print("MGA kRPC CREATE DEPARTURE NODE V0.1")
    print("=" * 80)
    if args.packet:
        print(f"Packet:        {args.packet}")
    print(f"v_inf used:    {vinf_m_s:.3f} m/s")
    print(f"v_inf source:  {source}")

    if candidates:
        print("\nTop inferred v_inf candidates:")
        for i, c in enumerate(candidates[:12]):
            mark = "*" if i == args.vinf_candidate_index else " "
            print(f" {mark} [{i:02d}] {c['value_m_s']:10.3f} m/s | {c['kind']:<6} | {c['path']}")
    if args.list_vinf_candidates:
        return 0

    try:
        import krpc
    except Exception as e:
        print(f"[ERROR] Could not import krpc: {e}", file=sys.stderr)
        return 2

    conn = krpc.connect(name=args.connection_name)
    sc = conn.space_center
    vessel = sc.active_vessel
    body = vessel.orbit.body

    print("\nActive vessel:")
    print(f"  vessel:       {vessel.name}")
    print(f"  body:         {body.name}")
    print(f"  UT:           {sc.ut:.3f} s")
    print(f"  mass:         {vessel.mass:.3f} kg")
    print(f"  thrust:       {vessel.available_thrust:.3f} N")
    print(f"  orbit Pe:     {vessel.orbit.periapsis_altitude / 1000.0:.3f} km alt")
    print(f"  orbit Ap:     {vessel.orbit.apoapsis_altitude / 1000.0:.3f} km alt")

    calc = compute_node_delta_v(vessel, sc, body, vinf_m_s, args.burn_at, args.safety_margin_m_s)
    dt = calc["ut"] - sc.ut

    print("\nApproximate ejection node:")
    print(f"  burn_at:      {args.burn_at}")
    print(f"  node UT:      {calc['ut']:.3f} s  (T+{dt:.3f} s)")
    print(f"  radius:       {calc['r_m'] / 1000.0:.3f} km")
    print(f"  parking v:    {calc['v_parking_m_s']:.3f} m/s")
    print(f"  escape v:     {calc['v_escape_at_r_m_s']:.3f} m/s")
    print(f"  hyp peri v:   {calc['v_hyp_periapsis_m_s']:.3f} m/s")
    print(f"  node dV pro:  {calc['dv_prograde_m_s']:.3f} m/s")
    print("\nWARNING: V0.1 matches v_inf magnitude only, not full MGA v_inf vector/B-plane direction.")

    if args.dry_run:
        print("\nDry run: node not created.")
        return 0

    if args.remove_existing_nodes:
        for n in list(vessel.control.nodes):
            n.remove()

    node = vessel.control.add_node(calc["ut"], prograde=calc["dv_prograde_m_s"], normal=0.0, radial=0.0)
    print("\n[OK] Created maneuver node:")
    print(f"  UT:       {node.ut:.3f}")
    print(f"  prograde: {calc['dv_prograde_m_s']:.3f} m/s")
    print("\nNext: execute with your node executor/probe, then inspect predicted patch chain for Eve.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
