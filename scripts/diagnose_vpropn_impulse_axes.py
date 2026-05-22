#!/usr/bin/env python3
"""
Diagnose what frame VPROPN actually applies impulse components in.

For each axis impulse, this script sends a single burn and compares:

  applied_dv = burn_v_after_raw_m_s - burn_v_before_raw_m_s

against the commanded vector.

If applied_dv != commanded_dv, the optimizer is using the wrong burn frame.
"""

from __future__ import annotations

import argparse
import inspect
import json
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from vessel_server_client import Burn, VesselPropnClient


def norm(v):
    return float(np.linalg.norm(np.asarray(v, dtype=float)))


def make_client(args):
    kwargs = dict(response_timeout_s=args.server_timeout_s, quiet_stderr=args.quiet_stderr)
    sig = inspect.signature(VesselPropnClient)
    if "plugin_arg_mode" in sig.parameters:
        return VesselPropnClient(args.server, args.plugin_b64, plugin_arg_mode=args.plugin_arg_mode, **kwargs)
    if args.plugin_arg_mode == "positional":
        return VesselPropnClient([str(args.server), str(args.plugin_b64)], plugin_b64=None, **kwargs)
    return VesselPropnClient(args.server, args.plugin_b64, **kwargs)


def main():
    ap = argparse.ArgumentParser(description="Diagnose VPROPN impulse axes.")
    ap.add_argument("--server", required=True)
    ap.add_argument("--plugin-b64", type=Path, required=True)
    ap.add_argument("--plugin-arg-mode", choices=["option", "positional"], default="option")
    ap.add_argument("--vessel-guid", required=True)
    ap.add_argument("--burn-dt-s", type=float, default=263.5379168987274)
    ap.add_argument("--final-dt-s", type=float, default=600.0)
    ap.add_argument("--impulse-m-s", type=float, default=100.0)
    ap.add_argument("--server-timeout-s", type=float, default=120.0)
    ap.add_argument("--quiet-stderr", action="store_true")
    ap.add_argument("--output-json", type=Path, required=True)
    args = ap.parse_args()

    axes = {
        "raw_x": np.array([args.impulse_m_s, 0.0, 0.0]),
        "raw_y": np.array([0.0, args.impulse_m_s, 0.0]),
        "raw_z": np.array([0.0, 0.0, args.impulse_m_s]),
        "diag_xyz": np.array([args.impulse_m_s, args.impulse_m_s * 0.3, -args.impulse_m_s * 0.2]),
    }

    rows = []
    with make_client(args) as client:
        for name, cmd in axes.items():
            res = client.vpropn(
                f"axis_{name}",
                args.vessel_guid,
                max(args.final_dt_s, args.burn_dt_s + 1.0),
                [Burn(args.burn_dt_s, cmd.tolist())],
                timeout_s=args.server_timeout_s,
            )
            b = res["burns"][0]
            v_before = np.asarray(b["burn_v_before_raw_m_s"], dtype=float)
            v_after = np.asarray(b["burn_v_after_raw_m_s"], dtype=float)
            applied = v_after - v_before
            err = applied - cmd
            rows.append({
                "axis": name,
                "commanded_raw_m_s": cmd.tolist(),
                "applied_from_burn_v_after_minus_before_m_s": applied.tolist(),
                "commanded_norm_m_s": norm(cmd),
                "applied_norm_m_s": norm(applied),
                "error_m_s": err.tolist(),
                "error_norm_m_s": norm(err),
                "cos_angle_commanded_applied": float(np.dot(cmd, applied) / max(1e-12, norm(cmd) * norm(applied))),
                "burn_dt_s_returned": b["burn_dt_s"],
                "burn_r_raw_m": b["burn_r_raw_m"],
                "burn_v_before_raw_m_s": b["burn_v_before_raw_m_s"],
                "burn_v_after_raw_m_s": b["burn_v_after_raw_m_s"],
            })

    out = {
        "burn_dt_s": args.burn_dt_s,
        "impulse_m_s": args.impulse_m_s,
        "rows": rows,
        "interpretation": (
            "If error_norm_m_s is ~0 for all rows, VPROPN impulse input is raw inertial delta-v. "
            "If not, the optimizer must transform decision variables into the frame expected by VPROPN."
        ),
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(out, indent=2) + "\n")

    print("=== VPROPN IMPULSE AXIS DIAGNOSTIC ===")
    for r in rows:
        print(
            f"{r['axis']:8s} cmd={r['commanded_raw_m_s']} "
            f"applied={r['applied_from_burn_v_after_minus_before_m_s']} "
            f"err={r['error_norm_m_s']:.6g} m/s cos={r['cos_angle_commanded_applied']:.9f}"
        )
    print(f"[OK] wrote {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
