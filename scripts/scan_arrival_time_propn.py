#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


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
    ap.add_argument("--tb0", type=float, required=True)
    ap.add_argument("--tb1", type=float, required=True)
    ap.add_argument("--t-final-center", type=float, required=True)
    ap.add_argument("--span-days", type=float, default=10.0)
    ap.add_argument("--samples", type=int, default=21)
    ap.add_argument("--max-nfev", type=int, default=100)
    args = ap.parse_args()

    args.base_output_dir.mkdir(parents=True, exist_ok=True)
    span_s = args.span_days * 86400.0

    summary = []

    for i in range(args.samples):
        if args.samples == 1:
            offset = 0.0
        else:
            offset = -0.5 * span_s + span_s * i / (args.samples - 1)

        tf = args.t_final_center + offset
        outdir = args.base_output_dir / f"arr_{i:03d}_dt_{offset:+.0f}s"

        cmd = [
            "python", "scripts/optimize_propn_to_target.py",
            "--plugin-b64", args.plugin_b64,
            "--server", args.server,
            "--live-state-json", args.live_state_json,
            "--leg-optimizations", args.leg_optimizations,
            "--leg", args.leg,
            "--bsp", args.bsp,
            "--tpc", args.tpc,
            "--dep-body", "KERBIN",
            "--arr-body", "EVE",
            "--impulse-time-s", str(args.tb0),
            "--impulse-time-s", str(args.tb1),
            "--final-time", str(tf),
            "--dv0-t-min", "1450",
            "--dv0-t-max", "3400",
            "--dv0-r-max", "300",
            "--dv0-n-max", "300",
            "--dv1-soft-max", "700",
            "--dv1-hard-max", "1200",
            "--burn1-min-kerbin-distance-km", "40000",
            "--final-pos-max-km", "100000",
            "--final-vel-max-m-s", "1000",
            "--output-dir", str(outdir),
            "--max-nfev", str(args.max_nfev),
        ]

        print()
        print(f"[ARR {i+1}/{args.samples}] tf={tf} offset={offset:+.0f}s")
        rc = subprocess.call(cmd)

        rp = outdir / "result.json"
        if rp.exists():
            r = json.loads(rp.read_text())
            summary.append({
                "i": i,
                "offset_s": offset,
                "t_final": tf,
                "rc": rc,
                "physically_valid": r.get("physically_valid"),
                "invalid_reasons": r.get("invalid_reasons"),
                "final_pos_err_km": r.get("final_pos_err_km"),
                "final_vel_err_m_s": r.get("final_vel_err_m_s"),
                "dv_norms_m_s": r.get("dv_norms_m_s"),
                "total_dv_m_s": r.get("total_dv_m_s"),
                "diagnostics": r.get("diagnostics"),
            })

    (args.base_output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )

    ranked = sorted(
        summary,
        key=lambda x: (
            float("inf") if x.get("final_pos_err_km") is None else x["final_pos_err_km"],
            float("inf") if x.get("total_dv_m_s") is None else x["total_dv_m_s"],
        ),
    )

    print()
    print("=== TOP ARRIVAL RESULTS ===")
    for r in ranked[:10]:
        print(
            f"i={r['i']:03d} dt={r['offset_s']:+.0f}s "
            f"valid={r['physically_valid']} "
            f"pos={r['final_pos_err_km']} km "
            f"vel={r['final_vel_err_m_s']} m/s "
            f"dv={r['dv_norms_m_s']} "
            f"reasons={r['invalid_reasons']}"
        )

    print("[OK] wrote", args.base_output_dir / "summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
