#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import krpc


def dot(a, b):
    return sum(a[i] * b[i] for i in range(3))


def norm(a):
    return math.sqrt(dot(a, a))


def scale(s, a):
    return [s * x for x in a]


def sub(a, b):
    return [a[i] - b[i] for i in range(3)]


def unit(a):
    n = norm(a)
    if n == 0:
        raise ValueError("zero vector")
    return scale(1.0 / n, a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--departure-target-json", type=Path, required=True)
    ap.add_argument("--vessel-guid", required=True)

    # Stable protocol controls.
    ap.add_argument("--event-key", default=None,
                    help="Stable logical key. Default: rank12_departure_smoke_leg<leg>")
    ap.add_argument("--attempt", type=int, default=0,
                    help="Increment only when you intentionally want a new event.")
    ap.add_argument("--request-id", default=None,
                    help="Override request_id. Avoid timestamps unless debugging.")

    # Timing.
    ap.add_argument("--burn-ut", type=float, default=None,
                    help="Explicit burn UT. If omitted, uses current UT + lead-seconds.")
    ap.add_argument("--lead-seconds", type=float, default=300.0)
    ap.add_argument("--plan-duration-s", type=float, default=7200.0)

    # Burn model.
    ap.add_argument("--mass-tonnes", type=float, default=2.6)
    ap.add_argument("--thrust-kN", type=float, default=2686.87701225281)
    ap.add_argument("--isp-s", type=float, default=1000.0)

    ap.add_argument("--output-json", type=Path, required=True)
    args = ap.parse_args()

    target = json.loads(args.departure_target_json.read_text())
    leg = int(target.get("leg", 1))
    vinf_raw = target["vinf_raw_m_s"]
    vinf_mag = norm(vinf_raw)

    event_key = args.event_key or f"rank12_departure_smoke_leg{leg}"

    if args.request_id:
        request_id = args.request_id
    else:
        request_id = f"{event_key}_attempt{args.attempt}"

    conn = krpc.connect(name="MGA departure smoke event")
    sc = conn.space_center
    vessel = sc.active_vessel
    body = vessel.orbit.body

    ut_now = sc.ut
    burn_ut = args.burn_ut if args.burn_ut is not None else ut_now + args.lead_seconds

    mu = body.gravitational_parameter
    r = norm(vessel.position(body.non_rotating_reference_frame))
    v_now = vessel.velocity(body.non_rotating_reference_frame)

    v_escape = math.sqrt(2.0 * mu / r)
    v_hyp = math.sqrt(v_escape * v_escape + vinf_mag * vinf_mag)

    # Smoke-test only: prograde energy match.
    # This is NOT the final Eve targeter.
    v_dir = unit(v_now)
    desired_v = scale(v_hyp, v_dir)
    dv_body = sub(desired_v, v_now)

    dv_nav = [norm(dv_body), 0.0, 0.0]

    ev = {
        "enabled": True,
        "request_id": request_id,
        "mode": "insert_navigation",
        "dedupe_tag": event_key,
        "event_key": event_key,
        "attempt": args.attempt,

        "vessel_guid": args.vessel_guid,

        "ensure_flight_plan": True,
        "extend_existing_flight_plan": True,
        "plan_final_time": burn_ut + args.plan_duration_s,
        "mass_tonnes": args.mass_tonnes,

        "insert_index": -1,
        "burn_template": "json",
        "thrust_kN": args.thrust_kN,
        "specific_impulse_s_g0": args.isp_s,
        "is_inertially_fixed": False,

        "frame_extension": 6000,
        "frame_centre_from_active_body": True,
        "frame_centre_index": -1,
        "frame_primary_index": -1,
        "frame_secondary_index": -1,

        "initial_time": burn_ut,
        "delta_v_navigation_m_s": dv_nav,

        "placeholder_dv_m_s": 0.001,
        "require_status_ok": True,
        "cleanup_on_error": True,
        "tolerance_time_s": 0.01,
        "tolerance_dv_m_s": 1e-6,

        # For future bridge versions / human debugging.
        "one_shot": True,
        "disable_after_success": True,

        "_diagnostic": {
            "ut_now": ut_now,
            "body": body.name,
            "r_m": r,
            "v_now_m_s": norm(v_now),
            "v_escape_m_s": v_escape,
            "vinf_target_m_s": vinf_mag,
            "vinf_target_raw_m_s": vinf_raw,
            "v_hyp_m_s": v_hyp,
            "dv_body_m_s": dv_body,
            "dv_body_norm_m_s": norm(dv_body),
            "note": (
                "Energy/prograde departure smoke only. "
                "Use attempt=N to intentionally create a new event. "
                "This is not the final Eve B-plane targeter."
            ),
        },
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(ev, indent=2) + "\n")

    print(json.dumps(ev, indent=2))


if __name__ == "__main__":
    main()
