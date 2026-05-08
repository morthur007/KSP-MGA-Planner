#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

def dot(a, b):
    return sum(x*y for x, y in zip(a, b))

def norm(v):
    return math.sqrt(sum(x*x for x in v))

def add(a, b):
    return [x+y for x, y in zip(a, b)]

def scale(s, v):
    return [s*x for x in v]

def raw_to_levela(v):
    return [-v[1], v[2], v[0]]

def nav_to_raw(dv_nav, tri):
    return add(
        add(scale(dv_nav[0], tri["tangent"]), scale(dv_nav[1], tri["normal"])),
        scale(dv_nav[2], tri["binormal"]),
    )

def raw_to_nav(dv_raw, tri):
    return [
        dot(dv_raw, tri["tangent"]),
        dot(dv_raw, tri["normal"]),
        dot(dv_raw, tri["binormal"]),
    ]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("probe_json", type=Path)
    ap.add_argument("--output-json", type=Path, default=None)
    args = ap.parse_args()

    probe = json.load(open(args.probe_json))
    out = {
        "vessel_guid": probe["vessel_guid"],
        "manoeuvres": [],
    }

    for m in probe["manoeuvres"]:
        dv_nav = m["delta_v_navigation_m_s"]
        tri = m["frenet_trihedron"]

        dv_raw = nav_to_raw(dv_nav, tri)
        dv_levela = raw_to_levela(dv_raw)
        roundtrip_nav = raw_to_nav(dv_raw, tri)

        rec = {
            "index": m["index"],
            "initial_time": m["initial_time"],
            "time_of_half_delta_v": m["time_of_half_delta_v"],
            "duration": m["duration"],
            "dv_navigation_m_s": dv_nav,
            "dv_raw_m_s": dv_raw,
            "dv_levela_m_s": dv_levela,
            "norm_navigation_m_s": norm(dv_nav),
            "norm_raw_m_s": norm(dv_raw),
            "norm_levela_m_s": norm(dv_levela),
            "roundtrip_navigation_m_s": roundtrip_nav,
            "roundtrip_error_m_s": norm([a-b for a, b in zip(dv_nav, roundtrip_nav)]),
        }
        out["manoeuvres"].append(rec)

    text = json.dumps(out, indent=2)
    print(text)

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text + "\n")
        print(f"[OK] wrote {args.output_json}")

if __name__ == "__main__":
    main()
