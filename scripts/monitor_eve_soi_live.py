#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import time

import krpc


def norm(v):
    return math.sqrt(sum(x*x for x in v))


def sub(a, b):
    return tuple(a[i] - b[i] for i in range(3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--end-ut", type=float, required=True)
    ap.add_argument("--chunk-hours", type=float, default=6.0)
    ap.add_argument("--max-rails-rate", type=float, default=100000.0)
    ap.add_argument("--max-physics-rate", type=float, default=2.0)
    ap.add_argument("--body", default="Eve")
    args = ap.parse_args()

    conn = krpc.connect(name="MGA monitor Eve SOI")
    sc = conn.space_center
    vessel = sc.active_vessel
    bodies = sc.bodies

    target = bodies[args.body]
    sun = bodies["Sun"]
    ref = sun.non_rotating_reference_frame

    soi = target.sphere_of_influence
    best_d = float("inf")
    best_ut = None

    print("=== MGA LIVE SOI MONITOR ===")
    print(f"target      : {args.body}")
    print(f"target SOI  : {soi/1000:.3f} km")
    print(f"start UT    : {sc.ut:.3f}")
    print(f"end UT      : {args.end_ut:.3f}")

    chunk_s = max(60.0, args.chunk_hours * 3600.0)

    while sc.ut < args.end_ut:
        ut = sc.ut
        rv = vessel.position(ref)
        rb = target.position(ref)
        d = norm(sub(rv, rb))

        if d < best_d:
            best_d = d
            best_ut = ut

        print(
            f"UT={ut:14.3f} "
            f"dist={d/1000:12.1f} km "
            f"best={best_d/1000:12.1f} km "
            f"SOI_ratio={d/soi:8.3f}"
        )

        if d <= soi:
            print()
            print("[HIT] vessel is inside target SOI")
            print(f"hit_ut={ut:.6f}")
            print(f"distance_km={d/1000:.6f}")
            return 0

        next_ut = min(args.end_ut, ut + chunk_s)
        sc.warp_to(next_ut, args.max_rails_rate, args.max_physics_rate)
        time.sleep(0.25)

    print()
    print("[MISS] did not enter target SOI before end_ut")
    print(f"best_ut={best_ut:.6f}")
    print(f"best_distance_km={best_d/1000:.6f}")
    print(f"target_soi_km={soi/1000:.6f}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
