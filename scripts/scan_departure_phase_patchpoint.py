#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
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
    ap.add_argument("--t-dep", type=float, required=True)
    ap.add_argument("--t-start", type=float, required=True)
    ap.add_argument("--samples", type=int, default=41)
    ap.add_argument("--span-s", type=float, default=None)
    ap.add_argument("--max-nfev", type=int, default=120)
    args = ap.parse_args()

    conn = krpc.connect(name="MGA scan patchpoint phase")
    sc = conn.space_center
    period = sc.active_vessel.orbit.period

    span = args.span_s if args.span_s is not None else period
    args.base_output_dir.mkdir(parents=True, exist_ok=True)

    summary = []

    print("=== PATCHPOINT DEPARTURE PHASE SCAN ===")
    print("orbital_period_s:", period)
    print("span_s:", span)
    print("samples:", args.samples)

    for i in range(args.samples):
        if args.samples == 1:
            offset = 0.0
        else:
            offset = -0.5 * span + span * i / (args.samples - 1)

        tb0 = args.t_dep + offset
        outdir = args.base_output_dir / f"phase_{i:03d}_dt_{offset:+.1f}s"

        cmd = [
            "python", "scripts/optimize_departure_to_leg_start.py",
            "--plugin-b64", args.plugin_b64,
            "--server", args.server,
            "--live-state-json", args.live_state_json,
            "--leg-optimizations", args.leg_optimizations,
            "--leg", args.leg,
            "--bsp", args.bsp,
            "--tpc", args.tpc,
            "--tb0", str(tb0),
            "--tb1", str(args.t_start),
            "--dv0-t-min", "1400",
            "--dv0-t-max", "3800",
            "--dv0-r-max", "500",
            "--dv0-n-max", "500",
            "--dv1-max", "1200",
            "--pos-scale-km", "1000",
            "--vel-scale-m-s", "50",
            "--output-dir", str(outdir),
            "--max-nfev", str(args.max_nfev),
        ]

        print()
        print(f"[PATCH SCAN {i+1}/{args.samples}] offset={offset:+.1f}s tb0={tb0}")
        rc = subprocess.call(cmd)

        rp = outdir / "result.json"
        if rp.exists():
            r = json.loads(rp.read_text())
            summary.append({
                "i": i,
                "offset_s": offset,
                "tb0": tb0,
                "rc": rc,
                "physically_valid": r.get("physically_valid"),
                "patchpoint_pos_err_km": r.get("patchpoint_pos_err_km"),
                "patchpoint_vel_err_m_s": r.get("patchpoint_vel_err_m_s"),
                "dv0_norm_m_s": r.get("dv0_norm_m_s"),
                "dv1_norm_m_s": r.get("dv1_norm_m_s"),
                "total_dv_m_s": r.get("total_dv_m_s"),
                "result_path": str(rp),
            })

    summary_path = args.base_output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    ranked = sorted(
        summary,
        key=lambda r: (
            float("inf") if r.get("patchpoint_pos_err_km") is None else r["patchpoint_pos_err_km"],
            float("inf") if r.get("patchpoint_vel_err_m_s") is None else r["patchpoint_vel_err_m_s"],
            float("inf") if r.get("total_dv_m_s") is None else r["total_dv_m_s"],
        ),
    )

    print()
    print("=== TOP PATCHPOINT RESULTS ===")
    for r in ranked[:10]:
        print(
            f"i={r['i']:03d} dt={r['offset_s']:+9.1f}s "
            f"valid={r['physically_valid']} "
            f"pos={r['patchpoint_pos_err_km']} km "
            f"vel={r['patchpoint_vel_err_m_s']} m/s "
            f"dv0={r['dv0_norm_m_s']} "
            f"dv1={r['dv1_norm_m_s']} "
            f"total={r['total_dv_m_s']}"
        )

    print("[OK] wrote", summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
