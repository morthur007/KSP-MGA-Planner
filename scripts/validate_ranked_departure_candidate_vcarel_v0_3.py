#!/usr/bin/env python3
"""
validate_ranked_departure_candidate_vcarel_v0_3.py

Validate a candidate produced by
rank_pykep_candidates_by_departure_executability_v0_1.py using the new Principia
server command VCAREL.

v0.3 adds the observed hybrid VCAREL convention: state_dt_s is absolute game time, while scan_start_dt_s/scan_end_dt_s are relative to that synthetic state time.

VCAREL starts from a synthetic relative state at the departure burn epoch instead
of long-coasting the serialized vessel for hundreds of days before burn0.

Expected VCAREL command format:

VCAREL <rid> <dep_body> <arr_body> <state_dt_s> <scan_start_dt_s> <scan_end_dt_s>
       <samples>
       <rel_r[3]> <rel_v[3]>
       <n_impulses>
       <impulse_dt_s> <dv_raw[3]> ...

Expected OKCAREL response fields are parsed according to the protocol supplied in
the project notes.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from principia_targeter_client import PrincipiaTargeterClient


DAY_S = 86400.0


def norm(v: Sequence[float]) -> float:
    return float(np.linalg.norm(np.asarray(v, dtype=float)))


def raw_to_levela(v: Sequence[float]) -> list[float]:
    x, y, z = map(float, v)
    return [-y, z, x]


def parse_days(s: str) -> list[float]:
    return [float(x.strip()) for x in str(s).replace(";", ",").split(",") if x.strip()]


def read_live_t(path: Path) -> float:
    r = json.loads(path.read_text())
    for k in ("ut_s", "t_s", "spice_t_s", "et_s"):
        if k in r:
            return float(r[k])
    raise KeyError(f"could not find time in {path}")


def select_candidate(rank_json: Path, top_index: int) -> dict[str, Any]:
    data = json.loads(rank_json.read_text())
    top = data.get("top", [])
    if top_index < 0 or top_index >= len(top):
        raise SystemExit(f"--top-index out of range: {top_index}; top has {len(top)} rows")
    c = dict(top[top_index])

    required = [
        "burn_dt_s",
        "burn_abs_s",
        "t_arr_s",
        "dv_raw_m_s",
        "burn_rel_r_raw_m",
        "burn_rel_v_raw_m_s",
    ]
    missing = [k for k in required if k not in c or c[k] is None]
    if missing:
        raise SystemExit(
            "selected candidate is missing fields needed by VCAREL: "
            + ", ".join(missing)
            + "\nRe-run rank_pykep_candidates_by_departure_executability_v0_1.py."
        )

    if "dv_levela_m_s" not in c or c["dv_levela_m_s"] is None:
        c["dv_levela_m_s"] = raw_to_levela(c["dv_raw_m_s"])

    return c


def parse_okcarel(line: str) -> dict[str, Any]:
    fields = line.rstrip("\n").split("\t")
    if not fields or fields[0] != "OKCAREL":
        raise RuntimeError(f"expected OKCAREL response, got: {line[:500]}")
    if len(fields) < 32:
        raise RuntimeError(f"OKCAREL response too short: {len(fields)} fields: {line[:500]}")

    out = {
        "id": fields[1],
        "dep_body": fields[2],
        "arr_body": fields[3],
        "state_dt_s": float(fields[4]),
        "state_t_game_s": float(fields[5]),
        "ca_dt_s": float(fields[6]),
        "ca_t_game_s": float(fields[7]),

        "ca_rel_r_raw_m": list(map(float, fields[8:11])),
        "ca_rel_v_raw_m_s": list(map(float, fields[11:14])),

        "ca_distance_m": float(fields[14]),
        "ca_speed_m_s": float(fields[15]),
        "ca_radial_v_m_s": float(fields[16]),

        "samples": int(float(fields[17])),
        "status": fields[18],

        "ca_abs_debug_r_raw_m": list(map(float, fields[19:22])),
        "ca_abs_debug_v_raw_m_s": list(map(float, fields[22:25])),

        "arr_abs_debug_r_raw_m": list(map(float, fields[25:28])),
        "arr_abs_debug_v_raw_m_s": list(map(float, fields[28:31])),

        "n_burns": int(float(fields[31])),
        "burns": [],
    }

    idx = 32
    for _ in range(out["n_burns"]):
        if idx + 10 > len(fields):
            raise RuntimeError(f"OKCAREL burn diagnostics truncated at field {idx}")
        burn = {
            "burn_dt_s": float(fields[idx + 0]),
            "burn_r_raw_m": list(map(float, fields[idx + 1:idx + 4])),
            "burn_v_before_raw_m_s": list(map(float, fields[idx + 4:idx + 7])),
            "burn_v_after_raw_m_s": list(map(float, fields[idx + 7:idx + 10])),
        }
        out["burns"].append(burn)
        idx += 10

    return out


def vcarel(
    client: PrincipiaTargeterClient,
    rid: str,
    dep_body: str,
    arr_body: str,
    state_dt_s: float,
    scan_start_dt_s: float,
    scan_end_dt_s: float,
    samples: int,
    rel_r: Sequence[float],
    rel_v: Sequence[float],
    impulses: Sequence[tuple[float, float, float, float]],
    timeout_s: float,
) -> dict[str, Any]:
    fields: list[Any] = [
        "VCAREL",
        rid,
        dep_body,
        arr_body,
        float(state_dt_s),
        float(scan_start_dt_s),
        float(scan_end_dt_s),
        int(samples),
        float(rel_r[0]), float(rel_r[1]), float(rel_r[2]),
        float(rel_v[0]), float(rel_v[1]), float(rel_v[2]),
        int(len(impulses)),
    ]
    for dt, dvx, dvy, dvz in impulses:
        fields += [float(dt), float(dvx), float(dvy), float(dvz)]

    line = client.command_fields(fields, timeout_s=timeout_s)
    return parse_okcarel(line)


def make_event(c: dict[str, Any], vessel_guid: str, out_path: Path, request_id: str) -> dict[str, Any]:
    burn_abs = float(c["burn_abs_s"])
    dv_levela = c.get("dv_levela_m_s") or raw_to_levela(c["dv_raw_m_s"])

    event = {
        "enabled": True,
        "vessel_guid": vessel_guid,
        "ensure_flight_plan": True,
        "extend_existing_flight_plan": True,
        "mass_tonnes": 2.6,
        "insert_index": 0,
        "burn_template": "json",
        "thrust_kN": 2686.87701225281,
        "specific_impulse_s_g0": 1000.0,
        "is_inertially_fixed": True,
        "frame_extension": 6000,
        "frame_centre_from_active_body": True,
        "frame_centre_index": -1,
        "frame_primary_index": -1,
        "frame_secondary_index": -1,
        "placeholder_dv_m_s": 0.001,
        "require_status_ok": True,
        "cleanup_on_error": True,
        "tolerance_time_s": 0.01,
        "tolerance_dv_m_s": 1e-6,
        "one_shot": True,
        "disable_after_success": True,
        "request_id": request_id,
        "dedupe_tag": request_id,
        "event_key": request_id,
        "attempt": 0,
        "mode": "insert_levela",
        "initial_time": burn_abs,
        "plan_final_time": burn_abs + 600.0,
        "delta_v_levela_m_s": [float(x) for x in dv_levela],
        "require_warp_close": True,
        "max_lead_before_insert_s": 600.0,
        "reject_long_plan_before_first_burn": True,
        "max_first_plan_duration_s": 900.0,
        "rollback_on_status_error": True,
        "auto_sort_insert_index": True,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(event, indent=2) + "\n")
    return event


def flatten_row(row: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in row.items():
        if isinstance(v, list):
            # list of scalars only; nested lists are JSON-stringified.
            if all(not isinstance(x, (list, dict)) for x in v):
                for i, x in enumerate(v):
                    out[f"{k}_{i}"] = x
            else:
                out[k] = json.dumps(v)
        elif isinstance(v, dict):
            out[k] = json.dumps(v)
        else:
            out[k] = v
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rank-json", type=Path, required=True)
    ap.add_argument("--top-index", type=int, default=0)

    ap.add_argument("--server", required=True)
    ap.add_argument("--plugin-b64", type=Path, required=True)
    ap.add_argument("--plugin-arg-mode", choices=["option", "positional"], default="positional")
    ap.add_argument("--vessel-guid", required=True)
    ap.add_argument("--live-state-json", type=Path, required=True)
    ap.add_argument("--dep-body", default=None)
    ap.add_argument("--arr-body", default=None)

    ap.add_argument("--arrival-offsets-days", default="-90,-60,-45,-30,-20,-15,-10,-7,-5,-3,-1,0,1,3,5,7,10,15,20,30,45,60,90")
    ap.add_argument("--scan-half-width-days", type=float, default=10.0)
    ap.add_argument("--vcarel-timebase",
                    choices=["absolute_state_relative_scan", "absolute", "live", "state"],
                    default="absolute_state_relative_scan",
                    help=(
                        "'absolute_state_relative_scan': observed current binary convention; "
                        "send state_dt_s=burn_abs_s and scan windows relative to the synthetic state. "
                        "'absolute': send state_dt_s=burn_abs_s and scan windows as absolute game times. "
                        "'live': legacy, send state_dt_s=burn_dt_s and scan windows relative to live_t. "
                        "'state': legacy, send state_dt_s=burn_dt_s and scan windows relative to synthetic state time."
                    ))
    ap.add_argument("--max-calls", type=int, default=0,
                    help="Debug limiter. 0 means all arrival offsets.")
    ap.add_argument("--vca-samples", type=int, default=81)
    ap.add_argument("--server-timeout-s", type=float, default=900.0)
    ap.add_argument("--quiet-stderr", action="store_true")
    ap.add_argument("--write-event", action="store_true")
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    c = select_candidate(args.rank_json, args.top_index)
    live_t = read_live_t(args.live_state_json)

    sequence = str(c.get("sequence", "")).split()
    dep_body = (args.dep_body or c.get("dep_body") or (sequence[0] if sequence else "KERBIN")).upper()
    arr_body = (args.arr_body or c.get("arr_body") or (sequence[1] if len(sequence) > 1 else "")).upper()
    if not arr_body:
        raise SystemExit("could not determine arr_body; pass --arr-body")

    burn_dt_s = float(c["burn_dt_s"])
    burn_abs_s = float(c["burn_abs_s"])
    t_arr_s = float(c["t_arr_s"])

    if args.vcarel_timebase == "absolute_state_relative_scan":
        # Observed from OKCAREL:
        #   state_t_game_s == state_dt_s
        #   ca_t_game_s    == state_t_game_s + ca_dt_s
        # Therefore state_dt_s must be absolute game time, while scan windows
        # are relative to that synthetic state epoch.
        state_dt_s = burn_abs_s
        state_t_game_s = burn_abs_s
        nominal_arrival_dt = t_arr_s - state_t_game_s
    elif args.vcarel_timebase == "absolute":
        # Legacy/diagnostic mode: absolute state and absolute scan values.
        # This is usually wrong for the current binary because it double-counts
        # the state epoch into ca_t_game_s.
        state_dt_s = burn_abs_s
        state_t_game_s = burn_abs_s
        nominal_arrival_dt = t_arr_s
    elif args.vcarel_timebase == "live":
        state_dt_s = burn_dt_s
        state_t_game_s = live_t + state_dt_s
        nominal_arrival_dt = t_arr_s - live_t
    else:
        state_dt_s = burn_dt_s
        state_t_game_s = live_t + state_dt_s
        nominal_arrival_dt = t_arr_s - state_t_game_s

    rel_r = [float(x) for x in c["burn_rel_r_raw_m"]]
    rel_v = [float(x) for x in c["burn_rel_v_raw_m_s"]]
    dv_raw = [float(x) for x in c["dv_raw_m_s"]]
    impulses = [(0.0, dv_raw[0], dv_raw[1], dv_raw[2])]

    offsets = parse_days(args.arrival_offsets_days)

    print("=== VALIDATE RANKED DEPARTURE CANDIDATE VCAREL V0 ===")
    print(f"rank_json       : {args.rank_json}")
    print(f"top_index       : {args.top_index}")
    print(f"row_index0      : {c.get('row_index0')}")
    print(f"sequence        : {c.get('sequence')}")
    print(f"dep -> arr      : {dep_body} -> {arr_body}")
    print(f"live_t          : {live_t}")
    print(f"state_dt_s      : {state_dt_s}")
    print(f"state_t_game_s  : {state_t_game_s}")
    print(f"t_arr_s         : {t_arr_s}")
    print(f"vcarel_timebase : {args.vcarel_timebase}")
    print(f"nom_arrival_dt  : {nominal_arrival_dt}")
    print(f"rel_r_norm_km   : {norm(rel_r)/1000:.6f}")
    print(f"rel_v_norm_m_s  : {norm(rel_v):.6f}")
    print(f"dv_raw_m_s      : {dv_raw} |v|={norm(dv_raw):.6f}")
    print(f"hunter T/B/phase: T={c.get('dv_tangent_m_s')} B={c.get('dv_binormal_m_s')} phase={c.get('phase_error_deg')}")
    print(f"output_dir      : {args.output_dir}")

    rows: list[dict[str, Any]] = []

    with PrincipiaTargeterClient(
        args.server,
        args.plugin_b64,
        plugin_arg_mode=args.plugin_arg_mode,
        response_timeout_s=args.server_timeout_s,
        quiet_stderr=args.quiet_stderr,
    ) as client:
        if args.max_calls and len(offsets) > args.max_calls:
            offsets = offsets[:args.max_calls]

        for i, off_d in enumerate(offsets):
            center = nominal_arrival_dt + off_d * DAY_S
            start = center - args.scan_half_width_days * DAY_S
            end = center + args.scan_half_width_days * DAY_S
            rid = f"vcarel_{os.getpid()}_{i}"
            try:
                if i == 0:
                    print("[debug] first VCAREL call:")
                    print(f"        state_dt={state_dt_s}")
                    print(f"        scan_start={start}")
                    print(f"        scan_end={end}")
                    print(f"        impulse_dt=0.0 dv={dv_raw}")
                res = vcarel(
                    client=client,
                    rid=rid,
                    dep_body=dep_body,
                    arr_body=arr_body,
                    state_dt_s=state_dt_s,
                    scan_start_dt_s=start,
                    scan_end_dt_s=end,
                    samples=args.vca_samples,
                    rel_r=rel_r,
                    rel_v=rel_v,
                    impulses=impulses,
                    timeout_s=args.server_timeout_s,
                )
                row = {
                    "ok": True,
                    "error": "",
                    "arrival_offset_days": off_d,
                    "scan_start_dt_s": start,
                    "scan_end_dt_s": end,
                    "ca_distance_km": res["ca_distance_m"] / 1000.0,
                    "ca_speed_m_s": res["ca_speed_m_s"],
                    "ca_radial_v_m_s": res["ca_radial_v_m_s"],
                    "ca_dt_s": res["ca_dt_s"],
                    "ca_t_game_s": res["ca_t_game_s"],
                    "state_dt_s": res["state_dt_s"],
                    "state_t_game_s": res["state_t_game_s"],
                    "status": res["status"],
                    "samples": res["samples"],
                    "n_burns": res["n_burns"],
                    "ca_rel_r_raw_m": res["ca_rel_r_raw_m"],
                    "ca_rel_v_raw_m_s": res["ca_rel_v_raw_m_s"],
                    "burns": res["burns"],
                }
            except Exception as exc:
                row = {
                    "ok": False,
                    "error": str(exc),
                    "arrival_offset_days": off_d,
                    "scan_start_dt_s": start,
                    "scan_end_dt_s": end,
                    "ca_distance_km": math.nan,
                }
            rows.append(row)

    finite = [
        r for r in rows
        if r.get("ok") and math.isfinite(float(r.get("ca_distance_km", math.nan)))
    ]
    finite.sort(key=lambda r: r["ca_distance_km"])

    print("")
    print("=== TOP VCAREL VALIDATION RESULTS ===")
    for i, r in enumerate(finite[:20], 1):
        print(
            f"{i:2d} ca={r['ca_distance_km']:12.3f} km "
            f"arr_off={r['arrival_offset_days']:7.1f} d "
            f"ca_dt={r['ca_dt_s']:14.3f} "
            f"speed={r['ca_speed_m_s']:9.2f} "
            f"radial={r['ca_radial_v_m_s']:9.2f} "
            f"status={r['status']}"
        )

    out = {
        "schema": "ranked_departure_candidate_vcarel_validation_v0_3",
        "rank_json": str(args.rank_json),
        "top_index": args.top_index,
        "candidate": c,
        "live_t_s": live_t,
        "dep_body": dep_body,
        "arr_body": arr_body,
        "vcarel_timebase": args.vcarel_timebase,
        "state_dt_s": state_dt_s,
        "state_t_game_s": state_t_game_s,
        "rel_r_raw_m": rel_r,
        "rel_v_raw_m_s": rel_v,
        "impulses": impulses,
        "n_rows": len(rows),
        "n_ok": len(finite),
        "best": finite[0] if finite else None,
        "rows": rows,
    }

    json_path = args.output_dir / "ranked_candidate_vcarel_validation.json"
    csv_path = args.output_dir / "ranked_candidate_vcarel_validation.csv"
    json_path.write_text(json.dumps(out, indent=2) + "\n")

    flat = [flatten_row(r) for r in rows]
    if flat:
        fields = sorted({k for r in flat for k in r.keys()})
        with csv_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(flat)

    print(f"[OK] wrote {json_path}")
    print(f"[OK] wrote {csv_path}")

    if args.write_event:
        event_path = args.output_dir / "event1_ranked_candidate_burn0_inertial_levela.json"
        event = make_event(
            c,
            args.vessel_guid,
            event_path,
            request_id=f"ranked_candidate_row{c.get('row_index0','x')}_vcarel_burn0_attempt0",
        )
        print(f"[OK] wrote {event_path}")
        print(json.dumps({
            "initial_time": event["initial_time"],
            "plan_final_time": event["plan_final_time"],
            "is_inertially_fixed": event["is_inertially_fixed"],
            "delta_v_levela_m_s": event["delta_v_levela_m_s"],
            "dv_norm_m_s": norm(event["delta_v_levela_m_s"]),
        }, indent=2))

    if not finite:
        print("[WARN] no finite VCAREL rows. Inspect errors in CSV/JSON.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
