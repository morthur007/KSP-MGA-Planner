#!/usr/bin/env python3
"""
MGA kRPC Create Departure Node V0.2

Safe bridge from an offline MGA route packet to a first KSP/kRPC maneuver node.

Scope V0.2:
- Connects to kRPC and reads the active vessel/parking orbit.
- Loads an optional B6D/route packet and tries to infer a *physical* v_inf magnitude.
- Filters out diagnostic/count/mismatch fields that are not trajectory v_inf values.
- Handles km/s vs m/s consistently.
- Creates an approximate prograde ejection node at next periapsis, sized to achieve
  the requested hyperbolic excess speed from the current parking orbit.

Important limitation:
- This still matches v_inf magnitude only, not the full heliocentric v_inf vector or B-plane.
- It is meant to create a practical first departure node for visual/game validation.
- A later version should solve the parking-orbit injection geometry to match the desired vector.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable

BAD_PATH_TOKENS = (
    "diagnosis", "diagnostic", "diagnosis_counts", "counts", "count",
    "mismatch", "miss", "error", "residual", "reject", "flag",
    "score", "class", "final_arrival", "arrival_mismatch",
)

GOOD_PATH_TOKENS = (
    "departure", "depart", "launch", "origin", "kerbin", "seg0", "segments[0]",
    "leg0", "legs[0]", "pre_leg", "first", "vinf_out", "vinf_depart",
)

PHYSICAL_VINF_TOKENS = (
    "vinf", "v_inf", "hyperbolic_excess", "asymptote", "asymptotes"
)


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


def is_vinf_like_path(path: str) -> bool:
    key = path.lower()
    return any(tok in key for tok in PHYSICAL_VINF_TOKENS)


def is_bad_path(path: str) -> bool:
    key = path.lower()
    return any(tok in key for tok in BAD_PATH_TOKENS)


def infer_units_to_m_s(value_or_mag: float, path: str) -> tuple[float, str]:
    """Return value in m/s and a unit note."""
    key = path.lower()
    val = float(value_or_mag)

    # Strong explicit units first.
    if "m_s" in key or "mps" in key or "m/s" in key:
        return val, "m/s explicit"
    if "km_s" in key or "kmps" in key or "km/s" in key:
        return val * 1000.0, "km/s explicit"

    # Project convention fallback: most state vectors use km/s; counts/mismatch are filtered out.
    # Interplanetary v_inf stored as a small scalar is almost certainly km/s.
    if abs(val) < 100.0:
        return val * 1000.0, "km/s inferred"
    return val, "m/s inferred"


def route_sequence(packet: Any) -> str:
    def find_seq(obj: Any, depth=0):
        if depth > 8:
            return None
        if isinstance(obj, dict):
            s = obj.get("sequence")
            if isinstance(s, list):
                return " -> ".join(map(str, s))
            if isinstance(s, str):
                return s.replace(",", " -> ")
            for v in obj.values():
                got = find_seq(v, depth + 1)
                if got:
                    return got
        elif isinstance(obj, list):
            for v in obj:
                got = find_seq(v, depth + 1)
                if got:
                    return got
        return None
    return find_seq(packet) or "unknown"


def find_vinf_candidates(packet: Any, include_diagnostics: bool = False) -> list[dict[str, Any]]:
    """Find plausible physical v_inf values/vectors in arbitrary route-packet schemas."""
    out: list[dict[str, Any]] = []
    if packet is None:
        return out

    for path, obj in walk(packet):
        if not is_vinf_like_path(path):
            continue
        if is_bad_path(path) and not include_diagnostics:
            continue

        if isinstance(obj, (int, float)) and finite(obj):
            val = float(obj)
            val_m_s, units = infer_units_to_m_s(val, path)
            if 50.0 <= abs(val_m_s) <= 50000.0:
                out.append({
                    "path": path,
                    "kind": "scalar",
                    "value_m_s": abs(val_m_s),
                    "raw": obj,
                    "units": units,
                    "excluded": False,
                })

        elif isinstance(obj, (list, tuple)) and len(obj) == 3 and all(finite(x) for x in obj):
            vals = [float(x) for x in obj]
            mag = norm(vals)
            mag_m_s, units = infer_units_to_m_s(mag, path)
            if 50.0 <= mag_m_s <= 50000.0:
                out.append({
                    "path": path,
                    "kind": "vector",
                    "value_m_s": mag_m_s,
                    "raw": vals,
                    "units": units,
                    "excluded": False,
                })

    def score(c):
        p = c["path"].lower()
        s = 0.0

        # Prefer actual departure/origin/first-leg fields.
        for token in GOOD_PATH_TOKENS:
            if token in p:
                s -= 20.0

        # Asymptotes of first flyby are not departure v_inf, but are more physical than diagnostics.
        if "flybys[0]" in p or "flybys.0" in p:
            s -= 5.0
        if "flybys[1]" in p or "flybys.1" in p:
            s += 5.0

        # Prefer scalar named vinf_* over vector components only if both exist.
        if c["kind"] == "scalar":
            s -= 1.0
        if "vec" in p:
            s += 0.5

        # Avoid values that look too small/huge for KSP interplanetary departure unless explicit.
        v = c["value_m_s"]
        if 500.0 <= v <= 8000.0:
            s -= 5.0
        elif v < 300.0:
            s += 10.0
        elif v > 15000.0:
            s += 10.0

        return s

    out.sort(key=score)
    return out


def pick_vinf(packet: Any, explicit_vinf_m_s: float | None, candidate_index: int, include_diagnostics: bool) -> tuple[float, list[dict[str, Any]], str]:
    if explicit_vinf_m_s is not None:
        return float(explicit_vinf_m_s), [], "explicit --vinf-m-s"
    candidates = find_vinf_candidates(packet, include_diagnostics=include_diagnostics)
    if not candidates:
        raise RuntimeError(
            "Could not infer a physical departure v_inf from packet. "
            "Pass --vinf-m-s explicitly, or rerun with --include-diagnostic-vinf-candidates for inspection."
        )
    if candidate_index < 0 or candidate_index >= len(candidates):
        raise RuntimeError(f"--vinf-candidate-index out of range: {candidate_index}; candidates={len(candidates)}")
    c = candidates[candidate_index]
    return float(c["value_m_s"]), candidates, c["path"]


def compute_node_delta_v(space_center, vessel, body, vinf_m_s: float, burn_at: str, safety_margin_m_s: float):
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


def print_candidates(candidates: list[dict[str, Any]], selected: int):
    print("\nTop inferred physical v_inf candidates:")
    if not candidates:
        print("  none")
        return
    for i, c in enumerate(candidates[:24]):
        mark = "*" if i == selected else " "
        print(f" {mark} [{i:02d}] {c['value_m_s']:10.3f} m/s | {c['value_m_s']/1000.0:8.4f} km/s | {c['kind']:<6} | {c.get('units','?'):<14} | {c['path']}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Create an approximate kRPC departure maneuver node from an MGA route packet.")
    ap.add_argument("--packet", help="Route/B6D packet JSON. Optional if --vinf-m-s is provided.")
    ap.add_argument("--vinf-m-s", type=float, help="Explicit desired hyperbolic excess speed at departure, m/s.")
    ap.add_argument("--vinf-candidate-index", type=int, default=0, help="Which inferred packet v_inf candidate to use.")
    ap.add_argument("--include-diagnostic-vinf-candidates", action="store_true", help="Include diagnostic/count/mismatch fields in candidate listing. Not recommended for node creation.")
    ap.add_argument("--burn-at", choices=["periapsis", "now"], default="periapsis")
    ap.add_argument("--safety-margin-m-s", type=float, default=0.0, help="Extra prograde m/s added to node.")
    ap.add_argument("--dry-run", action="store_true", help="Print computed node but do not create it.")
    ap.add_argument("--list-vinf-candidates", action="store_true", help="List packet v_inf candidates and exit before node creation.")
    ap.add_argument("--remove-existing-nodes", action="store_true", help="Remove existing maneuver nodes before creating the new one.")
    ap.add_argument("--connection-name", default="MGA Departure Node Creator")
    args = ap.parse_args(argv)

    packet = load_json(args.packet) if args.packet else None
    vinf_m_s, candidates, source = pick_vinf(
        packet,
        args.vinf_m_s,
        args.vinf_candidate_index,
        include_diagnostics=args.include_diagnostic_vinf_candidates,
    )

    print("=" * 80)
    print("MGA kRPC CREATE DEPARTURE NODE V0.2")
    print("=" * 80)
    if args.packet:
        print(f"Packet:        {args.packet}")
    if packet is not None:
        print(f"Sequence:      {route_sequence(packet)}")
    print(f"v_inf used:    {vinf_m_s:.3f} m/s ({vinf_m_s/1000.0:.6f} km/s)")
    print(f"v_inf source:  {source}")

    if candidates:
        print_candidates(candidates, args.vinf_candidate_index)
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

    calc = compute_node_delta_v(sc, vessel, body, vinf_m_s, args.burn_at, args.safety_margin_m_s)
    dt = calc["ut"] - sc.ut

    print("\nApproximate ejection node:")
    print(f"  burn_at:      {args.burn_at}")
    print(f"  node UT:      {calc['ut']:.3f} s  (T+{dt:.3f} s)")
    print(f"  radius:       {calc['r_m'] / 1000.0:.3f} km")
    print(f"  parking v:    {calc['v_parking_m_s']:.3f} m/s")
    print(f"  escape v:     {calc['v_escape_at_r_m_s']:.3f} m/s")
    print(f"  hyp peri v:   {calc['v_hyp_periapsis_m_s']:.3f} m/s")
    print(f"  node dV pro:  {calc['dv_prograde_m_s']:.3f} m/s")
    print("\nWARNING: V0.2 matches v_inf magnitude only, not full MGA v_inf vector/B-plane direction.")

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
