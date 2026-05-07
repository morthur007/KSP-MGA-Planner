#!/usr/bin/env python3
"""
MGA kRPC Create Departure Node V0.3

Purpose
-------
Create a *guarded* approximate departure maneuver node from a B6D/MGA route packet.

V0.3 fixes two important V0.2 problems:
  1. It no longer treats diagnostic fields, counters, turn angles, or mismatch values as v∞.
  2. It separates "next periapsis test burn" from "mission departure epoch".

This is still NOT a full departure targeter. It matches hyperbolic excess magnitude only
unless you explicitly provide node components. For a real operational injection, a later
version should solve the parking-orbit phase and prograde/radial/normal components against
 the heliocentric v∞ vector of the first segment.
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
except Exception:  # pragma: no cover
    krpc = None

G0 = 9.80665
DAY_S = 86400.0

BAD_VINF_PATH_TOKENS = (
    "turn_angle",
    "angle_deg",
    "diagnosis",
    "diagnostic",
    "count",
    "counts",
    "mismatch",
    "miss",
    "error",
    "score",
    "rank",
    "class",
    "margin",
    "alt",
    "radius",
    "rp",
    "periapsis",
    "tof",
    "duration",
    "layover",
)

GOOD_VINF_TOKENS = ("vinf", "v_inf", "v-infinity", "hyperbolic_excess")

TIME_GOOD_TOKENS = (
    "depart",
    "departure",
    "launch",
    "start_et",
    "start_ut",
    "t_depart",
    "epoch_depart",
    "segment_start",
    "et0",
    "t0",
)
TIME_BAD_TOKENS = (
    "tof",
    "duration",
    "layover",
    "period",
    "time_to",
    "periapsis_time",
    "soi_to_pe",
    "window",
    "span",
)


@dataclass
class VinfCandidate:
    value_m_s: float
    path: str
    source_type: str
    unit_note: str


@dataclass
class TimeCandidate:
    value_s: float
    path: str
    note: str


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
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


def finite_float(x: Any) -> float | None:
    try:
        v = float(x)
    except Exception:
        return None
    if not math.isfinite(v):
        return None
    return v


def mag3(xs: Any) -> float | None:
    if not isinstance(xs, list) or len(xs) != 3:
        return None
    vals = [finite_float(x) for x in xs]
    if any(v is None for v in vals):
        return None
    return math.sqrt(sum(float(v) ** 2 for v in vals))


def path_has_any(path: str, tokens: tuple[str, ...]) -> bool:
    p = path.lower()
    return any(t in p for t in tokens)


def infer_unit_to_m_s(value: float, path: str) -> tuple[float, str]:
    p = path.lower()
    if "m_s" in p or "mps" in p or "m/s" in p:
        return value, "m/s explicit"
    if "km_s" in p or "kmps" in p or "km/s" in p:
        return value * 1000.0, "km/s explicit"
    # In route packets, v∞ values are usually km/s. Values under 100 are overwhelmingly km/s.
    if 0.01 <= abs(value) <= 100.0:
        return value * 1000.0, "km/s inferred"
    return value, "m/s inferred"


def find_vinf_candidates(packet: Any, include_diagnostic: bool = False) -> list[VinfCandidate]:
    out: list[VinfCandidate] = []
    seen: set[tuple[int, str]] = set()

    for path, obj in walk(packet):
        lower = path.lower()
        if not path_has_any(lower, GOOD_VINF_TOKENS):
            continue
        if (not include_diagnostic) and path_has_any(lower, BAD_VINF_PATH_TOKENS):
            continue

        m = mag3(obj)
        if m is not None:
            val_m_s, unit = infer_unit_to_m_s(m, path)
            # discard absurd values for KSP planetary v∞ candidate list
            if 1.0 <= val_m_s <= 20000.0:
                key = (round(val_m_s), path)
                if key not in seen:
                    out.append(VinfCandidate(val_m_s, path, "vector", unit))
                    seen.add(key)
            continue

        v = finite_float(obj)
        if v is None:
            continue
        val_m_s, unit = infer_unit_to_m_s(v, path)
        if 1.0 <= val_m_s <= 20000.0:
            key = (round(val_m_s), path)
            if key not in seen:
                out.append(VinfCandidate(val_m_s, path, "scalar", unit))
                seen.add(key)

    # Prefer departure/first-segment-like paths, then source_packet, then smaller v∞.
    def sort_key(c: VinfCandidate):
        p = c.path.lower()
        dep_bonus = 0 if any(t in p for t in ("depart", "departure", "leg0", "segment_corrections[0]", "segments[0]")) else 1
        flyby_penalty = 1 if "flybys" in p else 0
        return (dep_bonus, flyby_penalty, c.value_m_s)

    out.sort(key=sort_key)
    return out


def find_time_candidates(packet: Any) -> list[TimeCandidate]:
    out: list[TimeCandidate] = []
    for path, obj in walk(packet):
        lower = path.lower()
        if path_has_any(lower, TIME_BAD_TOKENS):
            continue
        if not path_has_any(lower, TIME_GOOD_TOKENS):
            continue
        v = finite_float(obj)
        if v is None:
            continue
        # Route ET/UT seconds should be positive and within a few decades.
        if 0 <= v <= 2.0e9:
            note = "seconds"
            out.append(TimeCandidate(v, path, note))
    # prefer explicit departure/launch, then segment 0 start
    def sort_key(c: TimeCandidate):
        p = c.path.lower()
        if "depart" in p or "launch" in p:
            return (0, c.value_s)
        if "segment" in p and ("[0]" in p or ".0" in p):
            return (1, c.value_s)
        return (2, c.value_s)
    out.sort(key=sort_key)
    # de-dup by rounded seconds/path-ish
    dedup: list[TimeCandidate] = []
    seen_vals: set[int] = set()
    for c in out:
        r = int(round(c.value_s))
        if r not in seen_vals:
            dedup.append(c)
            seen_vals.add(r)
    return dedup


def get_body_mu_m3_s2(body: Any) -> float:
    # kRPC returns SI gravitational parameter.
    return float(body.gravitational_parameter)


def current_parking_radius_m(vessel: Any) -> float:
    # For nearly circular parking orbit use current radius around body.
    body_radius = float(vessel.orbit.body.equatorial_radius)
    altitude = float(vessel.flight(vessel.orbit.body.reference_frame).mean_altitude)
    if not math.isfinite(altitude) or altitude < -0.9 * body_radius:
        altitude = 0.5 * (float(vessel.orbit.periapsis_altitude) + float(vessel.orbit.apoapsis_altitude))
    return body_radius + altitude


def parking_speed_at_radius_m_s(mu: float, sma: float, r: float) -> float:
    # vis-viva for current bound orbit
    return math.sqrt(max(0.0, mu * (2.0 / r - 1.0 / sma)))


def compute_ejection_delta_v(vessel: Any, vinf_m_s: float, safety_margin_m_s: float = 0.0) -> dict[str, float]:
    orbit = vessel.orbit
    body = orbit.body
    mu = get_body_mu_m3_s2(body)

    # Use periapsis as default burn radius for Oberth efficiency.
    r = float(orbit.periapsis)
    if not math.isfinite(r) or r <= 0:
        r = current_parking_radius_m(vessel)
    sma = float(orbit.semi_major_axis)
    if not math.isfinite(sma) or sma <= 0:
        sma = r

    v_parking = parking_speed_at_radius_m_s(mu, sma, r)
    v_escape = math.sqrt(2.0 * mu / r)
    v_hyp_peri = math.sqrt(vinf_m_s * vinf_m_s + v_escape * v_escape)
    dv = max(0.0, v_hyp_peri - v_parking + safety_margin_m_s)
    return {
        "radius_m": r,
        "parking_speed_m_s": v_parking,
        "escape_speed_m_s": v_escape,
        "hyp_peri_speed_m_s": v_hyp_peri,
        "dv_prograde_m_s": dv,
    }


def node_ut_from_args(space_center: Any, vessel: Any, args: argparse.Namespace, time_candidates: list[TimeCandidate]) -> tuple[float, str]:
    now = float(space_center.ut)
    orbit = vessel.orbit

    if args.node_ut is not None:
        return float(args.node_ut), "explicit --node-ut"

    if args.node_in_seconds is not None:
        return now + float(args.node_in_seconds), "explicit --node-in-seconds"

    if args.use_inferred_departure_time:
        if not time_candidates:
            raise RuntimeError("--use-inferred-departure-time was set, but no departure time candidate was found in packet")
        raw = time_candidates[int(args.departure_time_candidate_index)].value_s
        return raw + float(args.packet_to_ksp_ut_offset_s), f"inferred packet time + offset from {time_candidates[int(args.departure_time_candidate_index)].path}"

    # Default is intentionally only a test burn at next periapsis, not a mission departure.
    try:
        dt = float(orbit.time_to_periapsis)
    except Exception:
        dt = 60.0
    return now + max(30.0, dt), "next periapsis test burn"


def create_node(vessel: Any, ut: float, prograde: float, radial: float = 0.0, normal: float = 0.0, remove_existing: bool = False) -> Any:
    if remove_existing:
        for n in list(vessel.control.nodes):
            n.remove()
    return vessel.control.add_node(float(ut), prograde=float(prograde), radial=float(radial), normal=float(normal))


def positive_int(x: str) -> int:
    v = int(x)
    if v < 0:
        raise argparse.ArgumentTypeError("must be >= 0")
    return v


def main() -> int:
    ap = argparse.ArgumentParser(description="Create guarded approximate KSP departure node from MGA route packet")
    ap.add_argument("--packet", required=True, type=Path)
    ap.add_argument("--vinf-candidate-index", type=positive_int, default=0)
    ap.add_argument("--vinf-m-s", type=float, default=None, help="Explicit v∞ magnitude in m/s")
    ap.add_argument("--include-diagnostic-vinf", action="store_true")
    ap.add_argument("--list-vinf-candidates", action="store_true")
    ap.add_argument("--list-time-candidates", action="store_true")

    ap.add_argument("--node-ut", type=float, default=None, help="Explicit KSP UT for node")
    ap.add_argument("--node-in-seconds", type=float, default=None, help="Create node this many seconds from current UT")
    ap.add_argument("--use-inferred-departure-time", action="store_true")
    ap.add_argument("--departure-time-candidate-index", type=positive_int, default=0)
    ap.add_argument("--packet-to-ksp-ut-offset-s", type=float, default=0.0)
    ap.add_argument("--warn-if-node-soon-s", type=float, default=3600.0)

    ap.add_argument("--safety-margin-m-s", type=float, default=0.0)
    ap.add_argument("--radial-m-s", type=float, default=0.0, help="Optional radial component; default 0")
    ap.add_argument("--normal-m-s", type=float, default=0.0, help="Optional normal component; default 0")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--remove-existing-nodes", action="store_true")
    ap.add_argument("--connect-name", default="MGA Create Departure Node V0.3")

    args = ap.parse_args()
    packet = load_json(args.packet)
    vinfs = find_vinf_candidates(packet, include_diagnostic=args.include_diagnostic_vinf)
    times = find_time_candidates(packet)

    seq = packet.get("sequence") or packet.get("route", {}).get("sequence") or "?"
    if isinstance(seq, list):
        seq_s = " -> ".join(map(str, seq))
    else:
        seq_s = str(seq)

    if args.vinf_m_s is not None:
        vinf_m_s = float(args.vinf_m_s)
        vinf_source = "explicit --vinf-m-s"
    else:
        if not vinfs:
            raise RuntimeError("No physical v∞ candidates found; pass --vinf-m-s explicitly")
        if args.vinf_candidate_index >= len(vinfs):
            raise RuntimeError(f"v∞ candidate index {args.vinf_candidate_index} out of range 0..{len(vinfs)-1}")
        c = vinfs[args.vinf_candidate_index]
        vinf_m_s = c.value_m_s
        vinf_source = c.path

    print("=" * 80)
    print("MGA kRPC CREATE DEPARTURE NODE V0.3")
    print("=" * 80)
    print(f"Packet:        {args.packet}")
    print(f"Sequence:      {seq_s}")
    print(f"v_inf used:    {vinf_m_s:.3f} m/s ({vinf_m_s/1000:.6f} km/s)")
    print(f"v_inf source:  {vinf_source}")

    if vinfs:
        print("\nTop inferred physical v∞ candidates:")
        for i, c in enumerate(vinfs[:20]):
            mark = "*" if (args.vinf_m_s is None and i == args.vinf_candidate_index) else " "
            print(f" {mark} [{i:02d}] {c.value_m_s:10.3f} m/s | {c.value_m_s/1000:8.4f} km/s | {c.source_type:<6} | {c.unit_note:<13} | {c.path}")
    else:
        print("\nNo physical v∞ candidates found in packet.")

    if times:
        print("\nInferred packet time candidates:")
        for i, t in enumerate(times[:20]):
            mark = "*" if (args.use_inferred_departure_time and i == args.departure_time_candidate_index) else " "
            print(f" {mark} [{i:02d}] {t.value_s:14.3f} s | {t.value_s/DAY_S:10.3f} d | {t.path}")
    else:
        print("\nNo departure/launch time candidates found in packet.")

    if args.list_vinf_candidates or args.list_time_candidates:
        return 0

    if krpc is None:
        raise RuntimeError("krpc Python package is not available")

    conn = krpc.connect(name=args.connect_name)
    sc = conn.space_center
    vessel = sc.active_vessel
    orbit = vessel.orbit
    body = orbit.body
    now = float(sc.ut)

    print("\nActive vessel:")
    print(f"  vessel:       {vessel.name}")
    print(f"  body:         {body.name}")
    print(f"  UT:           {now:.3f} s")
    print(f"  mass:         {float(vessel.mass):.3f} kg")
    print(f"  thrust:       {float(vessel.available_thrust):.3f} N")
    print(f"  orbit Pe:     {float(orbit.periapsis_altitude)/1000:.3f} km alt")
    print(f"  orbit Ap:     {float(orbit.apoapsis_altitude)/1000:.3f} km alt")

    node_ut, node_source = node_ut_from_args(sc, vessel, args, times)
    calc = compute_ejection_delta_v(vessel, vinf_m_s, args.safety_margin_m_s)
    dt = node_ut - now

    print("\nApproximate ejection node:")
    print(f"  node time source: {node_source}")
    print(f"  node UT:          {node_ut:.3f} s  (T{dt:+.3f} s / {dt/DAY_S:+.3f} d)")
    print(f"  radius:           {calc['radius_m']/1000:.3f} km")
    print(f"  parking v:        {calc['parking_speed_m_s']:.3f} m/s")
    print(f"  escape v:         {calc['escape_speed_m_s']:.3f} m/s")
    print(f"  hyp peri v:       {calc['hyp_peri_speed_m_s']:.3f} m/s")
    print(f"  node prograde:    {calc['dv_prograde_m_s']:.3f} m/s")
    print(f"  node radial:      {args.radial_m_s:.3f} m/s")
    print(f"  node normal:      {args.normal_m_s:.3f} m/s")

    if dt < 0:
        print("\nWARNING: computed node UT is in the past. Node will not be created unless you pass a future --node-ut/--node-in-seconds.")
    elif dt < args.warn_if_node_soon_s and not (args.node_ut or args.node_in_seconds):
        print("\nWARNING: node is soon because no route departure epoch was used. This is a test-burn timing, not the MGA launch window.")

    print("\nWARNING: V0.3 still matches v∞ magnitude only unless radial/normal are supplied.")
    print("For operational departure, solve full heliocentric v∞ vector and parking-orbit phase.")

    if args.dry_run:
        print("\nDry run: node not created.")
        return 0

    if dt < 0:
        raise RuntimeError("Refusing to create a maneuver node in the past")

    node = create_node(
        vessel,
        node_ut,
        calc["dv_prograde_m_s"],
        radial=args.radial_m_s,
        normal=args.normal_m_s,
        remove_existing=args.remove_existing_nodes,
    )
    print("\n[OK] Created maneuver node:")
    print(f"  UT:       {node.ut:.3f}")
    print(f"  prograde: {calc['dv_prograde_m_s']:.3f} m/s")
    print(f"  radial:   {args.radial_m_s:.3f} m/s")
    print(f"  normal:   {args.normal_m_s:.3f} m/s")
    print("\nNext: execute with node executor, then inspect patch chain. If no Eve encounter, do not treat that as route failure; V0.3 is not vector-targeted.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
