#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path

import krpc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-output-dir", type=Path, required=True)
    ap.add_argument("--live-state-json", required=True)
    ap.add_argument("--leg-optimizations", required=True)
    ap.add_argument("--plugin-b64", required=True)
    ap.add_argument("--server", default="principia_impulsive_particle_server")
    ap.add_argument("--bsp", required=True)
    ap.add_argument("--tpc", required=True)
    ap.add_argument("--leg", default="1")
    ap.add_argument("--dep-body", default="KERBIN")
    ap.add_argument("--arr-body", default="EVE")
    ap.add_argument("--t-dep", type=float, required=True)
    ap.add_argument("--t-start", type=float, required=True)
    ap.add_argument("--samples", type=int, default=25)
    ap.add_argument("--span-s", type=float, default=None)
    ap.add_argument("--max-nfev", type=int, default=80)
    args = ap.parse_args()

    conn = krpc.connect(name="MGA scan departure phase")
    sc = conn.space_center
    vessel = sc.active_vessel
    period = vessel.orbit.period

    span = args.span_s if args.span_s is not None else period
    dt_corr = args.t_start - args.t_dep

    args.base_output_dir.mkdir(parents=True, exist_ok=True)

    summary = []
    print("=== DEPARTURE PHASE SCAN ===")
    print("orbital_period_s:", period)
    print("span_s:", span)
    print("samples:", args.samples)

    for i in range(args.samples):
        if args.samples == 1:
            offset = 0.0
        else:
            offset = -0.5 * span + span * i / (args.samples - 1)

        tb0 = args.t_dep + offset
        tb1 = tb0 + dt_corr

        outdir = args.base_output_dir / f"phase_{i:03d}_dt_{offset:+.1f}s"
        cmd = [
            "python", "scripts/optimize_propn_to_target.py",
            "--plugin-b64", args.plugin_b64,
            "--server", args.server,
            "--live-state-json", args.live_state_json,
            "--leg-optimizations", args.leg_optimizations,
            "--leg", args.leg,
            "--bsp", args.bsp,
            "--tpc", args.tpc,
            "--dep-body", args.dep_body,
            "--arr-body", args.arr_body,
            "--impulse-time-s", str(tb0),
            "--impulse-time-s", str(tb1),
            "--dv0-t-min", "1450",
            "--dv0-t-max", "4200",
            "--dv0-r-max", "300",
            "--dv0-n-max", "300",
            "--dv1-soft-max", "400",
            "--dv1-hard-max", "900",
            "--burn1-min-kerbin-distance-km", "40000",
            "--final-pos-max-km", "100000",
            "--final-vel-max-m-s", "1000",
            "--output-dir", str(outdir),
            "--max-nfev", str(args.max_nfev),
        ]

        print()
        print(f"[SCAN {i+1}/{args.samples}] offset={offset:+.1f}s tb0={tb0}")
        rc = subprocess.call(cmd)

        result_path = outdir / "result.json"
        if result_path.exists():
            r = json.loads(result_path.read_text())
            summary.append({
                "i": i,
                "offset_s": offset,
                "tb0": tb0,
                "tb1": tb1,
                "rc": rc,
                "physically_valid": r.get("physically_valid"),
                "invalid_reasons": r.get("invalid_reasons"),
                "final_pos_err_km": r.get("final_pos_err_km"),
                "final_vel_err_m_s": r.get("final_vel_err_m_s"),
                "dv0": (r.get("dv_norms_m_s") or [None])[0],
                "dv1": (r.get("dv_norms_m_s") or [None, None])[1],
                "total_dv": r.get("total_dv_m_s"),
                "burn0_escape": (r.get("diagnostics") or {}).get("burn0_escape"),
            })

    summary_path = args.base_output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    ranked = sorted(
        summary,
        key=lambda x: (
            not bool(x.get("physically_valid")),
            float("inf") if x.get("final_pos_err_km") is None else x["final_pos_err_km"],
            float("inf") if x.get("total_dv") is None else x["total_dv"],
        ),
    )

    print()
    print("=== TOP RESULTS ===")
    for r in ranked[:10]:
        print(
            f"i={r['i']:03d} dt={r['offset_s']:+9.1f}s "
            f"valid={r['physically_valid']} "
            f"escape={r['burn0_escape']} "
            f"pos={r['final_pos_err_km']} km "
            f"vel={r['final_vel_err_m_s']} m/s "
            f"dv0={r['dv0']} dv1={r['dv1']} "
            f"reasons={r['invalid_reasons']}"
        )

    print("[OK] wrote", summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
