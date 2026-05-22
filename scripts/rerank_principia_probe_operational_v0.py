#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


def f(x: Any, default: float = math.nan) -> float:
    try:
        if x is None or x == "":
            return default
        return float(x)
    except Exception:
        return default


def get_probe(row: dict[str, Any]) -> dict[str, Any]:
    p = row.get("probe")
    return p if isinstance(p, dict) else {}


def get_base(row: dict[str, Any]) -> dict[str, Any]:
    b = row.get("base")
    return b if isinstance(b, dict) else {}


def compute_score(row: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    probe = get_probe(row)
    base = get_base(row)
    ca = f(probe.get("ca_distance_km"), math.inf)
    base_ca = f(base.get("ca_distance_km"), math.inf)
    rv = abs(f(probe.get("ca_radial_velocity_m_s"), 0.0))
    edge = str(probe.get("edge", ""))
    status = str(probe.get("status", ""))

    tnb = row.get("probe_tnb_m_s") or row.get("tnb_m_s") or [math.nan, math.nan, math.nan]
    T, N, B = [f(x) for x in tnb[:3]]
    dv = math.sqrt(T*T + N*N + B*B) if all(math.isfinite(x) for x in (T, N, B)) else math.inf
    oop = f(row.get("probe_oop_m_s"), math.hypot(N, B) if math.isfinite(N) and math.isfinite(B) else math.inf)
    angle = f(row.get("probe_plane_angle_deg"), math.degrees(math.atan2(oop, abs(T))) if math.isfinite(oop) and math.isfinite(T) else math.inf)

    score = 0.0
    terms: dict[str, float] = {}

    # CA band: being within [ca_min, ca_good_max] is fine. This avoids ranking a
    # candidate first merely because it hit exactly 200,000 km while another is
    # cleanly close at 176,000 km.
    if ca > args.ca_good_max_km:
        terms["ca_high"] = (ca - args.ca_good_max_km) * args.ca_high_weight
    elif ca < args.ca_min_km:
        terms["ca_low"] = (args.ca_min_km - ca) * args.ca_low_weight
    else:
        terms["ca_band"] = 0.0

    # Base CA is weakly useful: prefer candidates that already arrive near the flyby
    # before correction, but do not let this dominate geometry.
    if math.isfinite(base_ca):
        terms["base_ca"] = max(0.0, base_ca - args.base_ca_free_km) * args.base_ca_weight

    terms["dv"] = dv * args.dv_weight if math.isfinite(dv) else 1e12
    terms["oop"] = oop * args.oop_weight if math.isfinite(oop) else 1e12
    terms["radial"] = rv * args.radial_weight if math.isfinite(rv) else 1e12

    for name, value, soft, weight in [
        ("normal_soft", abs(N), args.normal_soft_m_s, args.normal_soft_weight),
        ("binormal_soft", abs(B), args.binormal_soft_m_s, args.binormal_soft_weight),
        ("oop_soft", oop, args.oop_soft_m_s, args.oop_soft_weight),
        ("angle_soft", angle, args.plane_angle_soft_deg, args.plane_angle_soft_weight),
    ]:
        excess = max(0.0, value - soft)
        terms[name] = weight * excess * excess

    if edge and edge != "none":
        terms["edge"] = args.edge_penalty
    if status == "scan_best":
        terms["scan_best"] = args.scan_best_penalty

    score = sum(terms.values())

    out = dict(row)
    out["operational_probe_score"] = score
    out["operational_probe_terms"] = terms
    out["operational_ca_km"] = ca
    out["operational_base_ca_km"] = base_ca
    out["operational_dv_m_s"] = dv
    out["operational_oop_m_s"] = oop
    out["operational_plane_angle_deg"] = angle
    out["operational_edge"] = edge
    out["operational_status"] = status
    return out


def flatten(row: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for k, v in row.items():
        if isinstance(v, dict):
            if k in ("base", "probe"):
                for kk, vv in v.items():
                    out[f"{k}_{kk}"] = vv
            elif k == "operational_probe_terms":
                for kk, vv in v.items():
                    out[f"term_{kk}"] = vv
            else:
                out[k] = json.dumps(v, ensure_ascii=False)
        elif isinstance(v, list):
            for i, vv in enumerate(v):
                out[f"{k}_{i}"] = vv
        else:
            out[k] = v
    return out


def main() -> int:
    p = argparse.ArgumentParser(description="Rerank Principia probe candidates with an operational flyby score.")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output-json", type=Path, required=True)
    p.add_argument("--output-csv", type=Path, default=None)
    p.add_argument("--top-n", type=int, default=100)

    p.add_argument("--ca-min-km", type=float, default=50000.0)
    p.add_argument("--ca-good-max-km", type=float, default=300000.0)
    p.add_argument("--ca-high-weight", type=float, default=1.0)
    p.add_argument("--ca-low-weight", type=float, default=5.0)
    p.add_argument("--base-ca-free-km", type=float, default=500000.0)
    p.add_argument("--base-ca-weight", type=float, default=0.02)

    p.add_argument("--dv-weight", type=float, default=0.2)
    p.add_argument("--oop-weight", type=float, default=2.0)
    p.add_argument("--radial-weight", type=float, default=1.0)

    p.add_argument("--normal-soft-m-s", type=float, default=150.0)
    p.add_argument("--binormal-soft-m-s", type=float, default=300.0)
    p.add_argument("--oop-soft-m-s", type=float, default=300.0)
    p.add_argument("--plane-angle-soft-deg", type=float, default=8.0)
    p.add_argument("--normal-soft-weight", type=float, default=0.5)
    p.add_argument("--binormal-soft-weight", type=float, default=1.0)
    p.add_argument("--oop-soft-weight", type=float, default=0.5)
    p.add_argument("--plane-angle-soft-weight", type=float, default=200.0)

    p.add_argument("--edge-penalty", type=float, default=1e8)
    p.add_argument("--scan-best-penalty", type=float, default=1e6)
    args = p.parse_args()

    data = json.loads(args.input.read_text())
    rows = data.get("top") or data.get("rows") or data.get("candidates")
    if not isinstance(rows, list):
        raise SystemExit("input must have a list field: top, rows, or candidates")

    out_rows = [compute_score(r, args) for r in rows]
    out_rows.sort(key=lambda r: f(r.get("operational_probe_score"), math.inf))
    out_rows = out_rows[: args.top_n]

    out = dict(data)
    out["schema"] = "principia_probe_operational_rerank_v0"
    out["rerank_config"] = vars(args) | {"input": str(args.input), "output_json": str(args.output_json), "output_csv": None if args.output_csv is None else str(args.output_csv)}
    out["top"] = out_rows

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")

    if args.output_csv:
        flat = [flatten(r) for r in out_rows]
        fields = []
        preferred = [
            "source_row_index0", "probe_dep_body", "probe_arr_body", "sequence",
            "operational_probe_score", "operational_ca_km", "operational_base_ca_km",
            "operational_dv_m_s", "operational_oop_m_s", "operational_plane_angle_deg",
            "operational_edge", "operational_status",
            "probe_tnb_m_s_0", "probe_tnb_m_s_1", "probe_tnb_m_s_2",
        ]
        for k in preferred:
            if any(k in r for r in flat) and k not in fields:
                fields.append(k)
        for r in flat:
            for k in r:
                if k not in fields:
                    fields.append(k)
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.output_csv.open("w", newline="") as fcsv:
            w = csv.DictWriter(fcsv, fieldnames=fields)
            w.writeheader()
            w.writerows(flat)

    print(f"[OK] wrote {args.output_json}")
    if args.output_csv:
        print(f"[OK] wrote {args.output_csv}")
    print("=== TOP OPERATIONAL PROBE RERANK ===")
    for i, r in enumerate(out_rows[:20], 1):
        tnb = r.get("probe_tnb_m_s") or [math.nan, math.nan, math.nan]
        print(
            f"{i:02d} row={r.get('source_row_index0')} "
            f"{r.get('probe_dep_body')}->{r.get('probe_arr_body')} "
            f"score={r['operational_probe_score']:12.3f} "
            f"ca={r['operational_ca_km']:10.1f} km "
            f"T={f(tnb[0]):8.1f} N={f(tnb[1]):8.1f} B={f(tnb[2]):8.1f} "
            f"oop={r['operational_oop_m_s']:8.1f} angle={r['operational_plane_angle_deg']:6.2f} "
            f"edge={r['operational_edge']} status={r['operational_status']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
