#!/usr/bin/env python3
"""
MGA kRPC Manual Burn Executor V0.1

Purpose
-------
Execute a burn WITHOUT relying on KSP maneuver nodes. This is intended for
Principia-heavy installs where stock maneuver nodes are unreliable as an
execution API. It points the vessel along a requested burn direction in the
current orbit/body frame and integrates delivered delta-v from thrust/mass.

This is NOT yet a full MGA departure-vector solver. It is a robust burn
executor primitive: given UT + desired delta-v vector components, execute the
burn by attitude + throttle + telemetry.

Direction convention
--------------------
The requested vector is in the local orbital basis at execution time:
  prograde  : vessel orbital velocity direction around current body
  radial    : radial-out from current body to vessel
  normal    : h = r x v orbital angular momentum direction

The script creates no maneuver nodes and does not read node.remaining_delta_v.

Examples
--------
Dry run only:
  python mga_krpc_manual_burn_executor_v0_1.py --burn-in 600 --prograde 100 --dry-run

Execute a prograde ejection-like burn:
  python mga_krpc_manual_burn_executor_v0_1.py --burn-ut 114478918.652 --prograde 2479.391 --auto-warp --execute

Execute a custom PRN vector:
  python mga_krpc_manual_burn_executor_v0_1.py --burn-in 1200 --prograde 2400 --radial 20 --normal -5 --execute
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from typing import Iterable, Tuple

import krpc

Vec3 = Tuple[float, float, float]


def vadd(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def vsub(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def vmul(a: Vec3, s: float) -> Vec3:
    return (a[0] * s, a[1] * s, a[2] * s)


def dot(a: Vec3, b: Vec3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def cross(a: Vec3, b: Vec3) -> Vec3:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def norm(a: Vec3) -> float:
    return math.sqrt(dot(a, a))


def unit(a: Vec3, fallback: Vec3 = (0.0, 1.0, 0.0)) -> Vec3:
    n = norm(a)
    if n <= 0.0 or not math.isfinite(n):
        return fallback
    return (a[0] / n, a[1] / n, a[2] / n)


def angle_deg(a: Vec3, b: Vec3) -> float:
    ua = unit(a)
    ub = unit(b)
    c = max(-1.0, min(1.0, dot(ua, ub)))
    return math.degrees(math.acos(c))


def fmt_t(seconds: float) -> str:
    sign = "-" if seconds < 0 else ""
    s = abs(float(seconds))
    d = int(s // 86400)
    s -= d * 86400
    h = int(s // 3600)
    s -= h * 3600
    m = int(s // 60)
    s -= m * 60
    if d:
        return f"{sign}{d}d {h:02d}h {m:02d}m {s:04.1f}s"
    if h:
        return f"{sign}{h}h {m:02d}m {s:04.1f}s"
    if m:
        return f"{sign}{m}m {s:04.1f}s"
    return f"{sign}{s:.1f}s"


@dataclass
class BurnPlan:
    burn_ut: float
    dv_prn: Vec3
    dv_mag: float
    start_ut: float
    estimated_burn_time_s: float
    direction_body_rf: Vec3


def get_body_basis(vessel, rf) -> Tuple[Vec3, Vec3, Vec3]:
    """Return prograde, radial-out, normal in the current body inertial/orbital RF."""
    pos = vessel.position(rf)
    vel = vessel.velocity(rf)
    radial = unit(pos, fallback=(1.0, 0.0, 0.0))
    prograde = unit(vel, fallback=(0.0, 1.0, 0.0))
    normal = unit(cross(pos, vel), fallback=(0.0, 0.0, 1.0))

    # Re-orthogonalize prograde against radial a bit for near-circular numerical stability.
    prograde = unit(vsub(prograde, vmul(radial, dot(prograde, radial))), fallback=prograde)
    normal = unit(cross(radial, prograde), fallback=normal)
    return prograde, radial, normal


def estimate_burn_time(vessel, dv_m_s: float) -> float:
    """Rocket-equation estimate for full-throttle burn time."""
    g0 = 9.80665
    m0 = float(vessel.mass)
    isp = float(vessel.specific_impulse)
    thrust = float(vessel.available_thrust)
    if dv_m_s <= 0:
        return 0.0
    if m0 <= 0 or isp <= 0 or thrust <= 0:
        raise RuntimeError("Cannot estimate burn time: vessel mass/isp/thrust invalid or no active thrust.")
    mdot = thrust / (isp * g0)
    m1 = m0 / math.exp(dv_m_s / (isp * g0))
    return max(0.0, (m0 - m1) / mdot)


def build_burn_plan(space_center, vessel, burn_ut: float, dv_prn: Vec3, center_burn: bool) -> BurnPlan:
    rf = vessel.orbit.body.reference_frame
    prograde, radial, normal = get_body_basis(vessel, rf)
    direction = vadd(vadd(vmul(prograde, dv_prn[0]), vmul(radial, dv_prn[1])), vmul(normal, dv_prn[2]))
    dv_mag = norm(dv_prn)
    burn_time = estimate_burn_time(vessel, dv_mag)
    start_ut = burn_ut - 0.5 * burn_time if center_burn else burn_ut
    return BurnPlan(
        burn_ut=burn_ut,
        dv_prn=dv_prn,
        dv_mag=dv_mag,
        start_ut=start_ut,
        estimated_burn_time_s=burn_time,
        direction_body_rf=unit(direction),
    )


def compute_throttle(remaining_dv: float, current_accel: float, min_throttle: float, fine_dv: float) -> float:
    if remaining_dv <= 0:
        return 0.0
    if current_accel <= 0:
        return 0.0
    # Keep burns stable near cutoff. We integrate delivered dv ourselves, so tapering helps overshoot.
    if remaining_dv < fine_dv:
        return min_throttle
    if remaining_dv < current_accel * 0.4:
        return max(min_throttle, 0.05)
    if remaining_dv < current_accel * 1.0:
        return max(min_throttle, 0.15)
    if remaining_dv < current_accel * 3.0:
        return 0.35
    return 1.0


def active_engines(vessel):
    return [e for e in vessel.parts.engines if e.active and e.has_fuel]


def execute_manual_burn(space_center, vessel, plan: BurnPlan, args) -> None:
    rf = vessel.orbit.body.reference_frame
    ap = vessel.auto_pilot
    ap.reference_frame = rf
    ap.engage()

    engines = active_engines(vessel)
    original_limits = {e: e.thrust_limit for e in engines}

    print("\nManual burn execution")
    print(f"  target burn UT:      {plan.burn_ut:.3f}")
    print(f"  start UT:            {plan.start_ut:.3f}")
    print(f"  estimated duration:  {plan.estimated_burn_time_s:.3f} s")
    print(f"  target Δv:           {plan.dv_mag:.3f} m/s")
    print(f"  direction body RF:   ({plan.direction_body_rf[0]:+.6f}, {plan.direction_body_rf[1]:+.6f}, {plan.direction_body_rf[2]:+.6f})")

    try:
        if args.auto_warp:
            warp_to = plan.start_ut - args.preburn_settle_s
            now = float(space_center.ut)
            if warp_to > now:
                print(f"Warping to T-{args.preburn_settle_s:.1f}s before burn start: UT {warp_to:.3f}")
                space_center.warp_to(warp_to)

        print("Pointing vessel...")
        ap.target_direction = plan.direction_body_rf
        while True:
            err = angle_deg(vessel.flight(rf).direction, plan.direction_body_rf)
            t_to_start = plan.start_ut - float(space_center.ut)
            if err <= args.pointing_tolerance_deg and t_to_start <= args.preburn_settle_s:
                break
            if t_to_start <= 0 and err > args.max_burn_angle_deg:
                print(f"[HOLD] Burn start reached but pointing error is {err:.3f} deg; waiting.")
            time.sleep(0.05)

        while float(space_center.ut) < plan.start_ut:
            ap.target_direction = plan.direction_body_rf
            time.sleep(0.02)

        print("IGNITION: manual node-free burn")
        delivered = 0.0
        last_t = float(space_center.ut)
        best_remaining = plan.dv_mag
        worsening_frames = 0

        while delivered < plan.dv_mag:
            now = float(space_center.ut)
            dt = max(0.0, now - last_t)
            last_t = now

            ap.target_direction = plan.direction_body_rf
            err = angle_deg(vessel.flight(rf).direction, plan.direction_body_rf)
            if err > args.max_burn_angle_deg:
                vessel.control.throttle = 0.0
                time.sleep(0.03)
                continue

            # delivered Δv is integrated from actual thrust / current mass.
            # For spool-up engines, vessel.thrust is better than available_thrust.
            actual_thrust = max(0.0, float(vessel.thrust))
            mass = max(1e-9, float(vessel.mass))
            if actual_thrust > 0.0 and dt > 0.0:
                delivered += (actual_thrust / mass) * dt

            remaining = max(0.0, plan.dv_mag - delivered)
            if remaining <= args.cutoff_tolerance_m_s:
                break

            if remaining > best_remaining + args.worsen_margin_m_s:
                worsening_frames += 1
            else:
                best_remaining = min(best_remaining, remaining)
                worsening_frames = 0
            if worsening_frames > args.max_worsening_frames:
                print("[CUT] Remaining Δv estimate worsened repeatedly; cutting throttle.")
                break

            current_accel_full = max(1e-9, float(vessel.available_thrust) / mass)
            throttle = compute_throttle(remaining, current_accel_full, args.min_throttle, args.fine_dv_m_s)
            vessel.control.throttle = throttle

            if args.verbose and int(now * 2) % 2 == 0:
                print(f"  delivered={delivered:9.3f} m/s remaining={remaining:9.3f} m/s throttle={throttle:.2f} err={err:.3f} deg")

            time.sleep(args.loop_dt_s)

        vessel.control.throttle = 0.0
        print("Throttle zero. Waiting for thrust spool-down...")
        while float(vessel.thrust) > args.spooldown_thrust_n:
            time.sleep(0.05)

        print(f"[OK] Burn complete. Delivered estimate: {delivered:.3f} m/s / target {plan.dv_mag:.3f} m/s")

    finally:
        vessel.control.throttle = 0.0
        for e, limit in original_limits.items():
            try:
                e.thrust_limit = limit
            except Exception:
                pass
        try:
            ap.disengage()
        except Exception:
            pass


def print_patch_chain(vessel, target_body: str | None, max_patches: int) -> None:
    print("\nPredicted orbit patch chain after current state:")
    orb = vessel.orbit
    found = False
    for i in range(1, max_patches + 1):
        if orb is None:
            break
        body = orb.body.name
        try:
            pe_km = orb.periapsis_altitude / 1000.0
            ap_km = orb.apoapsis_altitude / 1000.0
            period = orb.period
            print(f" {i:02d}. body={body:<12} Pe={pe_km:12.3f} km Ap={ap_km:12.3f} km period={period:12.3f} s")
        except Exception:
            print(f" {i:02d}. body={body:<12} <orbit data unavailable>")
        if target_body and body.lower() == target_body.lower():
            found = True
        orb = orb.next_orbit
    if target_body:
        print(f"Target encounter {target_body}: {'FOUND' if found else 'not present'}")


def main() -> int:
    p = argparse.ArgumentParser(description="Execute a node-free kRPC burn using attitude + integrated Δv.")
    t = p.add_mutually_exclusive_group(required=True)
    t.add_argument("--burn-ut", type=float, help="Burn epoch in KSP UT seconds.")
    t.add_argument("--burn-in", type=float, help="Burn after this many seconds from current UT.")

    p.add_argument("--prograde", type=float, default=0.0, help="Prograde component in m/s.")
    p.add_argument("--radial", type=float, default=0.0, help="Radial-out component in m/s.")
    p.add_argument("--normal", type=float, default=0.0, help="Normal component in m/s.")
    p.add_argument("--center-burn", action="store_true", help="Start half burn duration before burn UT.")
    p.add_argument("--auto-warp", action="store_true", help="Warp to shortly before burn start.")
    p.add_argument("--execute", action="store_true", help="Actually execute. Without this, dry-run only.")
    p.add_argument("--target-body", default=None, help="Optional target body to look for in predicted patches after burn.")
    p.add_argument("--max-patches", type=int, default=8)
    p.add_argument("--preburn-settle-s", type=float, default=10.0)
    p.add_argument("--pointing-tolerance-deg", type=float, default=0.5)
    p.add_argument("--max-burn-angle-deg", type=float, default=2.0)
    p.add_argument("--cutoff-tolerance-m-s", type=float, default=0.2)
    p.add_argument("--fine-dv-m-s", type=float, default=5.0)
    p.add_argument("--min-throttle", type=float, default=0.03)
    p.add_argument("--loop-dt-s", type=float, default=0.03)
    p.add_argument("--spooldown-thrust-n", type=float, default=0.1)
    p.add_argument("--worsen-margin-m-s", type=float, default=1.0)
    p.add_argument("--max-worsening-frames", type=int, default=30)
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    conn = krpc.connect(name="MGA Manual Node-Free Burn Executor V0.1")
    sc = conn.space_center
    vessel = sc.active_vessel

    burn_ut = float(args.burn_ut) if args.burn_ut is not None else float(sc.ut) + float(args.burn_in)
    dv_prn = (float(args.prograde), float(args.radial), float(args.normal))
    if norm(dv_prn) <= 0:
        raise SystemExit("Requested Δv is zero. Pass --prograde/--radial/--normal in m/s.")

    plan = build_burn_plan(sc, vessel, burn_ut, dv_prn, center_burn=args.center_burn)

    print("=" * 80)
    print("MGA kRPC MANUAL BURN EXECUTOR V0.1")
    print("=" * 80)
    print(f"Vessel:       {vessel.name}")
    print(f"Body:         {vessel.orbit.body.name}")
    print(f"UT now:       {float(sc.ut):.3f}")
    print(f"Burn UT:      {plan.burn_ut:.3f} ({fmt_t(plan.burn_ut - float(sc.ut))} from now)")
    print(f"Start UT:     {plan.start_ut:.3f} ({fmt_t(plan.start_ut - float(sc.ut))} from now)")
    print(f"Δv PRN:       pro={dv_prn[0]:.3f} radial={dv_prn[1]:.3f} normal={dv_prn[2]:.3f} m/s")
    print(f"Δv magnitude: {plan.dv_mag:.3f} m/s")
    print(f"Burn est.:    {plan.estimated_burn_time_s:.3f} s")
    print("\nNo stock maneuver nodes are created or read.")

    if not args.execute:
        print("\nDry run only. Pass --execute to burn.")
        return 0

    execute_manual_burn(sc, vessel, plan, args)
    time.sleep(1.0)
    print_patch_chain(vessel, args.target_body, args.max_patches)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
