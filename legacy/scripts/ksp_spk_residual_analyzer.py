#!/usr/bin/env python3
"""
ksp_spk_residual_analyzer.py

Analyze KSP/Principia observed states against a SPK kernel and decompose residuals.
This is for fact-grade per-body error measurement AFTER the SPK dense audit passes.

It reports:
  - position/velocity max/RMS/final errors
  - radial / transverse / normal residual components using the SPK state as RTN basis
  - an apparent time-offset estimate: delta_t ~= dot(dr, v) / dot(v, v)

A constant delta_t across many bodies is an epoch mapping issue.
A residual that is body-family dependent points to mu/model/missing-body/parent-system issues.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Any

try:
    import spiceypy as spice
except ImportError as exc:
    raise SystemExit("Instale spiceypy: pip install spiceypy") from exc


def dot(a, b): return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]
def sub(a, b): return (a[0]-b[0], a[1]-b[1], a[2]-b[2])
def cross(a, b): return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])
def norm(a): return math.sqrt(dot(a, a))
def scale(a, s): return (a[0]*s, a[1]*s, a[2]*s)

def unit(a):
    n = norm(a)
    if n == 0: return (float('nan'), float('nan'), float('nan'))
    return scale(a, 1.0/n)

def spice_name(name: str) -> str:
    return name.upper().replace(" ", "_").replace("-", "_")

def percentile(vals: List[float], q: float) -> float:
    if not vals: return float('nan')
    xs = sorted(vals)
    if len(xs) == 1: return xs[0]
    pos = (len(xs)-1) * q
    lo = int(math.floor(pos)); hi = int(math.ceil(pos))
    if lo == hi: return xs[lo]
    return xs[lo] * (hi-pos) + xs[hi] * (pos-lo)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Per-body KSP vs SPK residual analyzer.")
    ap.add_argument("--ksp-csv", type=Path, required=True)
    ap.add_argument("--bsp", type=Path, required=True)
    ap.add_argument("--tpc", type=Path, required=True)
    ap.add_argument("--central-body", required=True)
    ap.add_argument("--frame", default="J2000")
    ap.add_argument("--output-csv", type=Path, default=None)
    ap.add_argument("--output-json", type=Path, default=None)
    ap.add_argument("--max-print", type=int, default=200)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    spice.furnsh(str(args.tpc)); spice.furnsh(str(args.bsp))
    center = spice_name(args.central_body)

    residual_rows: List[Dict[str, Any]] = []
    grouped = defaultdict(lambda: {
        "pos": [], "vel": [], "radial": [], "transverse": [], "normal": [], "dt": [],
        "first_pos": None, "last_pos": None, "first_et": None, "last_et": None
    })

    try:
        with args.ksp_csv.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("read_error") or not row.get("x_m"):
                    continue
                body = row["body"]
                if body.upper() == args.central_body.upper():
                    continue
                et = float(row["et_seconds"])
                rk = (float(row["x_m"]), float(row["y_m"]), float(row["z_m"]))
                vk = (float(row["vx_m_s"]), float(row["vy_m_s"]), float(row["vz_m_s"]))
                try:
                    st, _ = spice.spkezr(spice_name(body), et, args.frame, "NONE", center)
                except spice.stypes.SpiceyError as exc:
                    continue
                rs = (st[0]*1000.0, st[1]*1000.0, st[2]*1000.0)
                vs = (st[3]*1000.0, st[4]*1000.0, st[5]*1000.0)
                dr = sub(rk, rs)
                dv = sub(vk, vs)
                pe = norm(dr); ve = norm(dv)
                rhat = unit(rs)
                hhat = unit(cross(rs, vs))
                that = unit(cross(hhat, rhat))
                radial = dot(dr, rhat)
                transverse = dot(dr, that)
                normal = dot(dr, hhat)
                v2 = dot(vs, vs)
                dt_est = dot(dr, vs) / v2 if v2 > 0 else float('nan')
                residual_rows.append({
                    "body": body, "et_seconds": et, "actual_ut_s": row.get("actual_ut_s"),
                    "pos_err_m": pe, "vel_err_m_s": ve,
                    "radial_m": radial, "transverse_m": transverse, "normal_m": normal,
                    "apparent_epoch_offset_s": dt_est,
                })
                g = grouped[body]
                for k, v in (("pos", pe), ("vel", ve), ("radial", radial), ("transverse", transverse), ("normal", normal), ("dt", dt_est)):
                    if math.isfinite(v): g[k].append(v)
                if g["first_pos"] is None:
                    g["first_pos"] = pe; g["first_et"] = et
                g["last_pos"] = pe; g["last_et"] = et
    finally:
        spice.kclear()

    summary = {}
    for body, g in grouped.items():
        pos = g["pos"]; vel = g["vel"]
        if not pos: continue
        summary[body] = {
            "samples": len(pos),
            "first_et": g["first_et"], "last_et": g["last_et"],
            "first_pos_err_m": g["first_pos"], "final_pos_err_m": g["last_pos"],
            "max_pos_err_m": max(pos), "rms_pos_err_m": math.sqrt(sum(x*x for x in pos)/len(pos)),
            "p50_pos_err_m": percentile(pos, 0.50), "p95_pos_err_m": percentile(pos, 0.95),
            "max_vel_err_m_s": max(vel) if vel else float('nan'),
            "rms_vel_err_m_s": math.sqrt(sum(x*x for x in vel)/len(vel)) if vel else float('nan'),
            "median_radial_m": percentile(g["radial"], 0.50),
            "median_transverse_m": percentile(g["transverse"], 0.50),
            "median_normal_m": percentile(g["normal"], 0.50),
            "median_apparent_epoch_offset_s": percentile(g["dt"], 0.50),
            "p95_abs_apparent_epoch_offset_s": percentile([abs(x) for x in g["dt"]], 0.95),
        }

    print("Per-body residuals: KSP/Principia CSV vs SPK")
    print(f"{'Corpo':<16} | {'N':>5} | {'Max km':>12} | {'RMS km':>12} | {'Final km':>12} | {'Max m/s':>10} | {'med dt s':>10}")
    print("-" * 95)
    for body, s in sorted(summary.items(), key=lambda kv: kv[1]["max_pos_err_m"], reverse=True)[:args.max_print]:
        print(f"{body:<16} | {s['samples']:5d} | {s['max_pos_err_m']/1000:12.3f} | {s['rms_pos_err_m']/1000:12.3f} | {s['final_pos_err_m']/1000:12.3f} | {s['max_vel_err_m_s']:10.3f} | {s['median_apparent_epoch_offset_s']:10.3f}")

    if args.output_csv:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.output_csv.open("w", newline="", encoding="utf-8") as f:
            fieldnames = sorted({k for r in residual_rows for k in r.keys()})
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader(); w.writerows(residual_rows)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps({"summary": summary}, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
