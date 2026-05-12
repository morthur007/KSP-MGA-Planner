#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

def f(row, *names, default=None):
    for n in names:
        if n in row and str(row[n]).strip() not in ("", "nan", "None"):
            return float(row[n])
    if default is not None:
        return default
    raise KeyError(f"missing any of {names}")

def s(row, *names, default=""):
    for n in names:
        if n in row and str(row[n]).strip():
            return str(row[n]).strip()
    return default

def raw_to_levela(v):
    x, y, z = v
    return [-y, z, x]

def vec_from(row, prefix):
    return [
        f(row, f"{prefix}_x_m_s", f"{prefix}_x", f"{prefix}x_m_s", f"{prefix}x"),
        f(row, f"{prefix}_y_m_s", f"{prefix}_y", f"{prefix}y_m_s", f"{prefix}y"),
        f(row, f"{prefix}_z_m_s", f"{prefix}_z", f"{prefix}z_m_s", f"{prefix}z"),
    ]

def has_any(row, names):
    return any(n in row and str(row[n]).strip() not in ("", "nan", "None") for n in names)

def detect_time(row):
    return f(
        row,
        "burn_t_s",
        "burn_time_s",
        "t_burn_s",
        "t_start_s",
        "start_time_s",
        "initial_time",
        "event_time_s",
    )

def detect_levela_or_raw(row):
    # Prefer explicit LevelA columns.
    levela_candidates = [
        ("dv_levela", ["dv_levela_x_m_s", "dv_levela_y_m_s", "dv_levela_z_m_s"]),
        ("dv0_levela", ["dv0_levela_x_m_s", "dv0_levela_y_m_s", "dv0_levela_z_m_s"]),
        ("burn_dv_levela", ["burn_dv_levela_x_m_s", "burn_dv_levela_y_m_s", "burn_dv_levela_z_m_s"]),
    ]
    for prefix, cols in levela_candidates:
        if all(c in row and str(row[c]).strip() for c in cols):
            return [float(row[c]) for c in cols], "levela:" + prefix

    # Raw Principia inertial columns; convert raw -> LevelA.
    raw_candidates = [
        ("dv_raw", ["dv_raw_x_m_s", "dv_raw_y_m_s", "dv_raw_z_m_s"]),
        ("dv0_raw", ["dv0_raw_x_m_s", "dv0_raw_y_m_s", "dv0_raw_z_m_s"]),
        ("dv", ["dv_x_m_s", "dv_y_m_s", "dv_z_m_s"]),
        ("dv_compact", ["dvx_m_s", "dvy_m_s", "dvz_m_s"]),
        ("dv0", ["dv0_x_m_s", "dv0_y_m_s", "dv0_z_m_s"]),
        ("burn_dv", ["burn_dv_x_m_s", "burn_dv_y_m_s", "burn_dv_z_m_s"]),
    ]
    for prefix, cols in raw_candidates:
        if all(c in row and str(row[c]).strip() for c in cols):
            raw = [float(row[c]) for c in cols]
            return raw_to_levela(raw), "raw:" + prefix

    raise KeyError("could not detect dv vector columns")

def norm(v):
    return math.sqrt(sum(x*x for x in v))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--leg-optimizations", type=Path, required=True)
    ap.add_argument("--vessel-guid", required=True)
    ap.add_argument("--request-prefix", default="route")
    ap.add_argument("--output-jsonl", type=Path, required=True)
    ap.add_argument("--max-events", type=int, default=999)
    ap.add_argument("--min-dv-m-s", type=float, default=0.001)
    ap.add_argument("--time-offset-s", type=float, default=0.0,
                    help="Use only for interface tests. Do not use to claim physical route validity.")
    args = ap.parse_args()

    events = []
    with args.leg_optimizations.open(newline="") as fcsv:
        reader = csv.DictReader(fcsv)
        for row in reader:
            success = s(row, "success", "status", default="").lower()
            if success and success not in ("true", "ok", "success", "1"):
                # Don't skip POWERED/PASS summaries accidentally.
                if "fail" in success or success == "false":
                    continue

            try:
                t = detect_time(row) + args.time_offset_s
                dv_levela, source = detect_levela_or_raw(row)
            except Exception as e:
                continue

            if norm(dv_levela) < args.min_dv_m_s:
                continue

            leg = s(row, "leg", "leg_index", default=str(len(events) + 1))
            dep = s(row, "dep_body", "dep", "leg_dep", default="")
            arr = s(row, "arr_body", "arr", "leg_arr", default="")

            ev = {
                "enabled": True,
                "request_id": f"{args.request_prefix}_leg{leg}_{len(events)+1:03d}",
                "mode": "insert_levela",
                "dedupe_tag": f"{args.request_prefix}_leg{leg}_{dep}_{arr}",
                "vessel_guid": args.vessel_guid,
                "insert_index": -1,
                "clone_from_index": 0,
                "initial_time": t,
                "delta_v_levela_m_s": dv_levela,
                "placeholder_dv_m_s": 0.001,
                "tolerance_time_s": 0.01,
                "tolerance_dv_m_s": 1e-6,
                "_source": {
                    "file": str(args.leg_optimizations),
                    "vector_source": source,
                    "leg": leg,
                    "dep": dep,
                    "arr": arr,
                    "dv_norm_m_s": norm(dv_levela),
                },
            }
            events.append(ev)
            if len(events) >= args.max_events:
                break

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("w") as f:
        for ev in events:
            f.write(json.dumps(ev, separators=(",", ":")) + "\n")

    print(f"[OK] wrote {len(events)} events -> {args.output_jsonl}")
    for ev in events:
        src = ev["_source"]
        print(
            f"{ev['request_id']} t={ev['initial_time']:.6f} "
            f"dv={src['dv_norm_m_s']:.3f} m/s "
            f"{src['dep']}->{src['arr']} source={src['vector_source']}"
        )

if __name__ == "__main__":
    main()
