#!/usr/bin/env python3
"""
spk_rebound_dense_audit.py

Audit numerical integrity of the chain:
    input snapshot -> REBOUND direct state -> SPK Type 3 -> SpiceyPy readback

This does NOT compare against KSP/Principia.  It isolates the SPK interpolation / writer layer.
If this audit is not clean, do not tune masses, epochs, frames, or integrator settings.

Usage:
  python spk_rebound_dense_audit.py \
    --input-json data/true_snapshot_v2.json \
    --bsp data/opm_mpe_1y_rebound/opm_mpe_1y_true_v2_1d.bsp \
    --tpc data/opm_mpe_1y_rebound/opm_mpe_1y_true_v2_1d.ids.tpc \
    --central-body Sun \
    --duration-years 1 \
    --samples-per-record 5 \
    --record-span-days 1 \
    --output-csv data/spk_dense_audit.csv
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

try:
    import spiceypy as spice
except ImportError as exc:
    raise SystemExit("Instale spiceypy: pip install spiceypy") from exc

DAY_S = 86400.0
JULIAN_YEAR_S = 365.25 * DAY_S
M_TO_KM = 1e-3
KM_TO_M = 1000.0


def norm3(v: Iterable[float]) -> float:
    a, b, c = v
    return math.sqrt(a*a + b*b + c*c)


def spice_name(name: str) -> str:
    return name.upper().replace(" ", "_").replace("-", "_")


def load_rebound_writer_module(module_path: Path):
    spec = importlib.util.spec_from_file_location("rebound_writer_v2", str(module_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Não consegui importar {module_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def direct_state_relative_m(sim: Any, body_index: int, center_index: int) -> Tuple[float, float, float, float, float, float]:
    p = sim.particles[body_index]
    c = sim.particles[center_index]
    return (
        p.x - c.x, p.y - c.y, p.z - c.z,
        p.vx - c.vx, p.vy - c.vy, p.vz - c.vz,
    )


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Dense audit: REBOUND direct vs SPK readback.")
    ap.add_argument("--input-json", type=Path, required=True)
    ap.add_argument("--bsp", type=Path, required=True)
    ap.add_argument("--tpc", type=Path, required=True)
    ap.add_argument("--central-body", required=True)
    ap.add_argument("--duration-years", type=float, required=True)
    ap.add_argument("--record-span-days", type=float, required=True)
    ap.add_argument("--samples-per-record", type=int, default=5, help="Dense interior samples per record, endpoints excluded by default.")
    ap.add_argument("--include-endpoints", action="store_true")
    ap.add_argument("--writer-module", type=Path, default=Path("rebound_ephemeris_to_spk_type3_v2.py"))
    ap.add_argument("--integrator", default="ias15", choices=["ias15", "whfast", "mercurius", "trace", "leapfrog", "sei", "saba"])
    ap.add_argument("--ias15-epsilon", type=float, default=1e-11)
    ap.add_argument("--whfast-dt-seconds", type=float, default=None)
    ap.add_argument("--flip-z-input", action="store_true")
    ap.add_argument("--frame", default="J2000")
    ap.add_argument("--output-csv", type=Path, default=None)
    ap.add_argument("--output-json", type=Path, default=None)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    mod = load_rebound_writer_module(args.writer_module)

    eph = mod.load_legacy_json(
        input_json=args.input_json,
        body_catalog_path=None,
        central_body=args.central_body,
        initial_sample_index=0,
        et_offset_seconds=None,
        flip_z_input=args.flip_z_input,
    )
    config = mod.ExportConfig(
        duration_years=args.duration_years,
        record_span_days=args.record_span_days,
        cheby_degree=15,  # not used here
        samples_per_record=16,  # not used here
        integrator=args.integrator,
        whfast_dt_seconds=args.whfast_dt_seconds,
        ias15_epsilon=args.ias15_epsilon,
        central_body=args.central_body,
        frame_name=args.frame,
        center_naif_code=990000,
        target_naif_code_base=990001,
        archive_path=None,
        archive_every_records=50,
    )
    sim, ordered_names = mod.make_rebound_simulation(eph, config)
    name_to_index = {name: i for i, name in enumerate(ordered_names)}
    center_index = name_to_index[args.central_body]
    target_names = [n for n in ordered_names if n != args.central_body]

    start_et = eph.start_ut_s + eph.et_offset_seconds
    duration_s = args.duration_years * JULIAN_YEAR_S
    record_span_s = args.record_span_days * DAY_S
    n_records = math.ceil(duration_s / record_span_s)

    # Times are monotonic. Use interior points to detect interpolation error between Chebyshev nodes.
    rel_times: List[float] = []
    for rec in range(n_records):
        t0 = rec * record_span_s
        t1 = min((rec + 1) * record_span_s, duration_s)
        if args.include_endpoints:
            denom = max(args.samples_per_record - 1, 1)
            rel_times.extend([t0 + (t1 - t0) * i / denom for i in range(args.samples_per_record)])
        else:
            denom = args.samples_per_record + 1
            rel_times.extend([t0 + (t1 - t0) * i / denom for i in range(1, args.samples_per_record + 1)])
    rel_times = sorted(set(t for t in rel_times if 0.0 <= t <= duration_s))

    spice.furnsh(str(args.tpc))
    spice.furnsh(str(args.bsp))

    rows: List[Dict[str, Any]] = []
    stats: Dict[str, Dict[str, float]] = {
        name: {"samples": 0, "max_pos_m": 0.0, "rms_pos_accum": 0.0, "max_vel_m_s": 0.0, "rms_vel_accum": 0.0}
        for name in target_names
    }

    try:
        for t_rel in rel_times:
            try:
                sim.integrate(t_rel, exact_finish_time=1)
            except TypeError:
                sim.integrate(t_rel)
            et = start_et + t_rel
            for name in target_names:
                d = direct_state_relative_m(sim, name_to_index[name], center_index)
                try:
                    st, _ = spice.spkezr(spice_name(name), et, args.frame, "NONE", spice_name(args.central_body))
                except spice.stypes.SpiceyError as exc:
                    rows.append({"body": name, "et_seconds": et, "spice_error": str(exc).splitlines()[0]})
                    continue
                s = (st[0]*KM_TO_M, st[1]*KM_TO_M, st[2]*KM_TO_M, st[3]*KM_TO_M, st[4]*KM_TO_M, st[5]*KM_TO_M)
                pe = norm3((d[0]-s[0], d[1]-s[1], d[2]-s[2]))
                ve = norm3((d[3]-s[3], d[4]-s[4], d[5]-s[5]))
                rows.append({"body": name, "et_seconds": et, "t_rel_s": t_rel, "pos_err_m": pe, "vel_err_m_s": ve})
                stt = stats[name]
                stt["samples"] += 1
                stt["max_pos_m"] = max(stt["max_pos_m"], pe)
                stt["rms_pos_accum"] += pe * pe
                stt["max_vel_m_s"] = max(stt["max_vel_m_s"], ve)
                stt["rms_vel_accum"] += ve * ve
    finally:
        spice.kclear()

    summary = {}
    for name, stt in sorted(stats.items()):
        n = int(stt["samples"])
        rms_pos = math.sqrt(stt["rms_pos_accum"] / n) if n else float("nan")
        rms_vel = math.sqrt(stt["rms_vel_accum"] / n) if n else float("nan")
        summary[name] = {
            "samples": n,
            "max_pos_err_m": stt["max_pos_m"],
            "rms_pos_err_m": rms_pos,
            "max_vel_err_m_s": stt["max_vel_m_s"],
            "rms_vel_err_m_s": rms_vel,
        }

    print(f"Audit REBOUND direto vs SPK: {len(rel_times)} epochs, {len(target_names)} corpos")
    print(f"{'Corpo':<16} | {'N':>5} | {'Max pos (m)':>14} | {'RMS pos (m)':>14} | {'Max vel (m/s)':>14} | Status")
    print("-" * 92)
    for name, s in summary.items():
        maxp = s["max_pos_err_m"]
        maxv = s["max_vel_err_m_s"]
        if maxp <= 1.0 and maxv <= 1e-6:
            status = "EXCELENTE"
        elif maxp <= 100.0 and maxv <= 1e-4:
            status = "OK"
        elif maxp <= 1000.0 and maxv <= 1e-3:
            status = "SUSPEITO"
        else:
            status = "FALHA"
        print(f"{name:<16} | {s['samples']:5d} | {maxp:14.6g} | {s['rms_pos_err_m']:14.6g} | {maxv:14.6g} | {status}")

    if args.output_csv:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.output_csv.open("w", newline="", encoding="utf-8") as f:
            fieldnames = sorted({k for r in rows for k in r.keys()})
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps({"summary": summary, "n_epochs": len(rel_times)}, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
