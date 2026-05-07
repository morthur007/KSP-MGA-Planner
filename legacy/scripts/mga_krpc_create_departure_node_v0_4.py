#!/usr/bin/env python3
"""
MGA kRPC Create Departure Node V0.4

Safe diagnostic/utility script for creating an approximate KSP maneuver node from a
B6D/MGA route packet.

V0.4 fixes two important V0.3 pitfalls:
  - time candidates are restricted to real epoch-like fields such as depart_et/arrival_et;
    fields such as correction_m_s are never treated as time.
  - v_inf candidates respect key units. Fields ending in *_km_s are converted to m/s;
    vector components are not offered as scalar v_inf candidates.

This is still a magnitude-only ejection helper. It does NOT solve the full
heliocentric v_inf vector, parking-orbit phase, or B-plane direction.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

try:
    import krpc  # type: ignore
except Exception:  # allow --list modes without kRPC if possible
    krpc = None

G0 = 9.80665
DAY_S = 86400.0

BAD_NUMERIC_PATH_TOKENS = (
    "count", "counts", "diagnosis", "diagnostics", "mismatch", "miss",
    "margin", "rp", "alt", "angle", "turn", "score", "class", "flag",
    "correction", "dv", "delta_v", "tof", "duration", "period", "radius",
    "mass", "thrust", "isp", "ratio", "index", "rank", "segments",
)

TIME_KEY_ALLOW = ("depart_et", "arrival_et", "event_et", "epoch_et", "time_et", "node_ut", "ut")
TIME_KEY_DENY = (
    "correction", "m_s", "km_s", "vinf", "dv", "delta", "mismatch", "miss",
    "score", "angle", "turn", "margin", "rp", "alt", "radius", "count", "tof",
)

@dataclass
class Candidate:
    value_m_s: float
    source: str
    kind: str
    unit_note: str

@dataclass
class TimeCandidate:
    et_s: float
    source: str
    kind: str


def load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def walk(obj: Any, prefix: str = "") -> Iterable[tuple[str, Any]]:
    if isinstance(obj, dict):
        for k, v in obj.items():
            p = f"{prefix}.{k}" if prefix else str(k)
            yield p, v
            yield from walk(v, p)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            p = f"{prefix}[{i}]"
            yield p, v
            yield from walk(v, p)


def is_finite_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(float(x))


def path_leaf(path: str) -> str:
    return path.replace("]", "").split(".")[-1].split("[")[0]


def has_bad_path_token(path: str) -> bool:
    p = path.lower()
    return any(tok in p for tok in BAD_NUMERIC_PATH_TOKENS)


def vector_norm(v: Any) -> float | None:
    if not isinstance(v, list) or len(v) != 3:
        return None
    if not all(is_finite_number(x) for x in v):
        return None
    return math.sqrt(sum(float(x) ** 2 for x in v))


def collect_vinf_candidates(packet: Any, include_components: bool = False, include_debug: bool = False) -> list[Candidate]:
    cands: list[Candidate] = []

    for path, val in walk(packet):
        pl = path.lower()
        leaf = path_leaf(path).lower()

        if not include_debug and has_bad_path_token(path):
            # do not allow turn_angle, corrections, scores, counts etc. as v_inf
            # exception: explicit vinf path contains vinf and is handled below
            if "vinf" not in pl:
                continue

        # Vector fields: accept only full vectors with km_s or m_s in key/path.
        n = vector_norm(val)
        if n is not None and "vinf" in pl:
            if "km_s" in pl or "km/s" in pl:
                cands.append(Candidate(n * 1000.0, path, "vector", "km/s -> m/s"))
            elif "m_s" in pl or "m/s" in pl:
                cands.append(Candidate(n, path, "vector", "m/s"))
            # no unit => ignore to stay safe
            continue

        if not is_finite_number(val):
            continue

        # Scalar v_inf: accept exact scalar magnitudes, not components.
        if "vinf" not in pl:
            continue
        if "vec" in pl and not include_components:
            continue

        x = float(val)
        if x <= 0:
            continue

        # Explicit unit from key/path.
        if "km_s" in pl or "km/s" in pl:
            cands.append(Candidate(x * 1000.0, path, "scalar", "km/s -> m/s"))
        elif "m_s" in pl or "m/s" in pl:
            cands.append(Candidate(x, path, "scalar", "m/s"))

    # de-duplicate by rounded value and source preference
    seen: set[tuple[int, str]] = set()
    out: list[Candidate] = []
    for c in sorted(cands, key=lambda c: (c.value_m_s, c.source)):
        key = (round(c.value_m_s), c.source)
        if key in seen:
            continue
        seen.add(key)
        # practical bounds: 10 m/s to 30 km/s unless debug requested
        if include_debug or (10.0 <= c.value_m_s <= 30000.0):
            out.append(c)
    return out


def collect_time_candidates(packet: Any, include_debug: bool = False) -> list[TimeCandidate]:
    out: list[TimeCandidate] = []
    for path, val in walk(packet):
        if not is_finite_number(val):
            continue
        pl = path.lower()
        leaf = path_leaf(path).lower()
        if not any(leaf.endswith(k) or leaf == k for k in TIME_KEY_ALLOW):
            continue
        if not include_debug and any(tok in pl for tok in TIME_KEY_DENY):
            continue
        x = float(val)
        # Real mission epochs should be non-negative and not tiny correction-like values.
        if not include_debug and x < 1000.0:
            continue
        out.append(TimeCandidate(x, path, "epoch"))

    # prioritize depart_et before later event times
    def priority(t: TimeCandidate) -> tuple[int, float, str]:
        p = t.source.lower()
        pri = 0 if "segment_corrections[0].depart_et" in p else 1
        if "depart_et" in p:
            pri = min(pri, 1)
        elif "arrival_et" in p:
            pri = 2
        else:
            pri = 3
        return (pri, t.et_s, t.source)

    # de-dup values close to millisecond
    seen: set[tuple[int, str]] = set()
    dedup: list[TimeCandidate] = []
    for t in sorted(out, key=priority):
        key = (round(t.et_s * 1000), t.source)
        if key not in seen:
            seen.add(key)
            dedup.append(t)
    return dedup


def get_sequence(packet: Any) -> str:
    for path, val in walk(packet):
        if path.endswith("sequence"):
            if isinstance(val, list):
                return " -> ".join(map(str, val))
            if isinstance(val, str):
                return val.replace(",", " -> ")
    return "unknown"


def get_body_mu(body: Any) -> float:
    # kRPC body.gravitational_parameter is m^3/s^2
    return float(body.gravitational_parameter)


def compute_ejection_delta_v(space_center: Any, vessel: Any, vinf_m_s: float, node_ut: float, burn_at: str, safety_margin_m_s: float) -> dict[str, float]:
    orbit = vessel.orbit
    body = orbit.body
    mu = get_body_mu(body)

    if burn_at == "periapsis":
        r = float(orbit.periapsis)
    elif burn_at == "current":
        # approximate current orbital radius from body center
        pos = vessel.position(body.reference_frame)
        r = math.sqrt(sum(float(x) ** 2 for x in pos))
    else:
        raise ValueError(f"Unsupported burn_at={burn_at}")

    if r <= 0:
        raise ValueError("Invalid orbital radius")

    v_circ_like = math.sqrt(mu * (2.0 / r - 1.0 / float(orbit.semi_major_axis)))
    v_escape = math.sqrt(2.0 * mu / r)
    v_hyp_peri = math.sqrt(vinf_m_s ** 2 + v_escape ** 2)
    dv = max(0.0, v_hyp_peri - v_circ_like + safety_margin_m_s)

    return {
        "node_ut": float(node_ut),
        "radius_m": r,
        "parking_v_m_s": v_circ_like,
        "escape_v_m_s": v_escape,
        "hyp_peri_v_m_s": v_hyp_peri,
        "prograde_m_s": dv,
        "radial_m_s": 0.0,
        "normal_m_s": 0.0,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--packet", required=True)
    ap.add_argument("--vinf-candidate-index", type=int, default=None)
    ap.add_argument("--vinf-m-s", type=float, default=None)
    ap.add_argument("--vinf-km-s", type=float, default=None)
    ap.add_argument("--list-vinf-candidates", action="store_true")
    ap.add_argument("--list-time-candidates", action="store_true")
    ap.add_argument("--include-debug-candidates", action="store_true")
    ap.add_argument("--include-vector-components", action="store_true")

    ap.add_argument("--use-inferred-departure-time", action="store_true")
    ap.add_argument("--departure-time-candidate-index", type=int, default=0)
    ap.add_argument("--et-to-ut-offset-s", type=float, default=0.0, help="KSP UT = packet ET + offset. Default assumes identity.")
    ap.add_argument("--node-ut", type=float, default=None)
    ap.add_argument("--node-in-seconds", type=float, default=None)
    ap.add_argument("--allow-far-future-node", action="store_true")
    ap.add_argument("--max-node-lead-days-without-confirm", type=float, default=30.0)
    ap.add_argument("--allow-past-node", action="store_true")

    ap.add_argument("--burn-at", choices=["periapsis", "current"], default="periapsis")
    ap.add_argument("--safety-margin-m-s", type=float, default=0.0)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--remove-existing-nodes", action="store_true")
    ap.add_argument("--connect-name", default="MGA Create Departure Node V0.4")
    args = ap.parse_args()

    packet = load_json(args.packet)
    sequence = get_sequence(packet)
    vinf_cands = collect_vinf_candidates(packet, include_components=args.include_vector_components, include_debug=args.include_debug_candidates)
    time_cands = collect_time_candidates(packet, include_debug=args.include_debug_candidates)

    # Choose vinf
    if args.vinf_m_s is not None:
        vinf_m_s = float(args.vinf_m_s)
        vinf_source = "explicit --vinf-m-s"
    elif args.vinf_km_s is not None:
        vinf_m_s = float(args.vinf_km_s) * 1000.0
        vinf_source = "explicit --vinf-km-s"
    elif args.vinf_candidate_index is not None:
        if args.vinf_candidate_index < 0 or args.vinf_candidate_index >= len(vinf_cands):
            raise SystemExit(f"Invalid --vinf-candidate-index {args.vinf_candidate_index}; have {len(vinf_cands)} candidates")
        c = vinf_cands[args.vinf_candidate_index]
        vinf_m_s = c.value_m_s
        vinf_source = c.source
    elif vinf_cands:
        vinf_m_s = vinf_cands[0].value_m_s
        vinf_source = vinf_cands[0].source
    else:
        raise SystemExit("No v_inf candidate found. Pass --vinf-m-s or --vinf-km-s explicitly.")

    print("=" * 80)
    print("MGA kRPC CREATE DEPARTURE NODE V0.4")
    print("=" * 80)
    print(f"Packet:        {args.packet}")
    print(f"Sequence:      {sequence}")
    print(f"v_inf used:    {vinf_m_s:.3f} m/s ({vinf_m_s/1000.0:.6f} km/s)")
    print(f"v_inf source:  {vinf_source}")

    if args.list_vinf_candidates or True:
        print("\nTop inferred physical v∞ candidates:")
        if not vinf_cands:
            print("  none")
        for i, c in enumerate(vinf_cands[:25]):
            mark = "*" if args.vinf_candidate_index == i or (args.vinf_candidate_index is None and args.vinf_m_s is None and args.vinf_km_s is None and i == 0) else " "
            print(f" {mark} [{i:02d}] {c.value_m_s:10.3f} m/s | {c.value_m_s/1000.0:8.4f} km/s | {c.kind:<6} | {c.unit_note:<12} | {c.source}")

    if args.list_time_candidates or True:
        print("\nInferred packet time candidates:")
        if not time_cands:
            print("  none")
        for i, t in enumerate(time_cands[:25]):
            mark = "*" if args.use_inferred_departure_time and args.departure_time_candidate_index == i else " "
            print(f" {mark} [{i:02d}] {t.et_s:14.3f} s | {t.et_s/DAY_S:10.3f} d | {t.source}")

    if args.list_vinf_candidates and args.list_time_candidates and not (args.dry_run or args.node_ut is not None or args.node_in_seconds is not None or args.use_inferred_departure_time):
        return 0

    if krpc is None:
        raise SystemExit("kRPC is not available in this Python environment.")
    conn = krpc.connect(name=args.connect_name)
    sc = conn.space_center
    vessel = sc.active_vessel
    orbit = vessel.orbit
    body = orbit.body
    now_ut = float(sc.ut)

    print("\nActive vessel:")
    print(f"  vessel:       {vessel.name}")
    print(f"  body:         {body.name}")
    print(f"  UT:           {now_ut:.3f} s")
    print(f"  mass:         {vessel.mass:.3f} kg")
    print(f"  thrust:       {vessel.available_thrust:.3f} N")
    print(f"  orbit Pe:     {orbit.periapsis_altitude/1000.0:.3f} km alt")
    print(f"  orbit Ap:     {orbit.apoapsis_altitude/1000.0:.3f} km alt")

    # Choose node UT
    time_source = "next periapsis/test timing"
    if args.node_ut is not None:
        node_ut = float(args.node_ut)
        time_source = "explicit --node-ut"
    elif args.node_in_seconds is not None:
        node_ut = now_ut + float(args.node_in_seconds)
        time_source = "explicit --node-in-seconds"
    elif args.use_inferred_departure_time:
        if args.departure_time_candidate_index < 0 or args.departure_time_candidate_index >= len(time_cands):
            raise SystemExit(f"Invalid --departure-time-candidate-index {args.departure_time_candidate_index}; have {len(time_cands)} candidates")
        t = time_cands[args.departure_time_candidate_index]
        node_ut = t.et_s + float(args.et_to_ut_offset_s)
        time_source = f"inferred packet ET + offset from {t.source}"
    else:
        # test mode: next periapsis/current not mission timing
        if args.burn_at == "periapsis":
            node_ut = now_ut + float(orbit.time_to_periapsis)
        else:
            node_ut = now_ut + 60.0

    calc = compute_ejection_delta_v(sc, vessel, vinf_m_s, node_ut, args.burn_at, args.safety_margin_m_s)
    lead_s = calc["node_ut"] - now_ut

    print("\nApproximate ejection node:")
    print(f"  node time source: {time_source}")
    print(f"  node UT:          {calc['node_ut']:.3f} s  (T{lead_s:+.3f} s / {lead_s/DAY_S:+.3f} d)")
    print(f"  radius:           {calc['radius_m']/1000.0:.3f} km")
    print(f"  parking v:        {calc['parking_v_m_s']:.3f} m/s")
    print(f"  escape v:         {calc['escape_v_m_s']:.3f} m/s")
    print(f"  hyp peri v:       {calc['hyp_peri_v_m_s']:.3f} m/s")
    print(f"  node prograde:    {calc['prograde_m_s']:.3f} m/s")
    print(f"  node radial:      {calc['radial_m_s']:.3f} m/s")
    print(f"  node normal:      {calc['normal_m_s']:.3f} m/s")

    print("\nWARNING: V0.4 still matches v∞ magnitude only unless radial/normal are supplied by a future vector solver.")
    print("For operational departure, solve full heliocentric v∞ vector and parking-orbit phase.")

    if lead_s < 0 and not args.allow_past_node:
        print("\nWARNING: computed node UT is in the past. Node not created. Use --allow-past-node only for debugging.")
        return 0
    if lead_s > args.max_node_lead_days_without_confirm * DAY_S and not args.allow_far_future_node:
        print(f"\nWARNING: node is {lead_s/DAY_S:.3f} days in the future. This is likely the actual launch window.")
        print("Node not created unless you pass --allow-far-future-node, or use --node-in-seconds for a test burn.")
        return 0

    if args.dry_run:
        print("\nDry run: node not created.")
        return 0

    if args.remove_existing_nodes:
        for node in list(vessel.control.nodes):
            node.remove()

    node = vessel.control.add_node(calc["node_ut"], prograde=calc["prograde_m_s"], normal=calc["normal_m_s"], radial=calc["radial_m_s"])
    print("\n[OK] Created maneuver node:")
    print(f"  UT:       {calc['node_ut']:.3f}")
    print(f"  prograde: {calc['prograde_m_s']:.3f} m/s")
    print(f"  radial:   {calc['radial_m_s']:.3f} m/s")
    print(f"  normal:   {calc['normal_m_s']:.3f} m/s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
