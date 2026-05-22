#!/usr/bin/env python3
"""
Diagnose frame/epoch mismatch between:
  - VPROPN vessel propagation output,
  - leg_optimizations.csv matchpoint columns,
  - SPICE body states.

This does not optimize anything. It answers:
  "Is the matchpoint target in the same frame/epoch as the VPROPN state?"

Typical usage:
  python scripts/diagnose_matchpoint_frame_epoch.py \
    --server /home/matheus/Principia/bin/x64/principia_impulsive_particle_server \
    --plugin-b64 data/principia/live_probe/principia_serialized_plugin_rocket.b64 \
    --plugin-arg-mode positional \
    --vessel-guid 60735c81-7e29-4c06-9551-9e5283e37586 \
    --live-state-json data/runs/game_export/rank12_real/live_state_raw_near_tdep.json \
    --leg-optimizations data/runs/finalists/rank12_kekj/leg_optimizations.csv \
    --leg 1 \
    --bsp data/kernels/jnsq/v0_3/jnsq_principia_native_v0_3_1000y_1h_deg9.bsp \
    --tpc data/kernels/jnsq/v0_3/jnsq_principia_native_v0_3_1000y_1h_deg9.ids.tpc \
    --dep-body KERBIN \
    --arr-body EVE \
    --quiet-stderr \
    --output-dir data/runs/game_export/rank12_real/matchpoint_frame_diag01
"""

from __future__ import annotations

import argparse
import csv
import inspect
import json
import math
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import spiceypy as spice

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from vessel_server_client import VesselPropnClient


RAW_TO_LEVELA = np.array([
    [0.0, -1.0, 0.0],
    [0.0,  0.0, 1.0],
    [1.0,  0.0, 0.0],
], dtype=float)

LEVELA_TO_RAW = np.array([
    [0.0, 0.0, 1.0],
    [-1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
], dtype=float)


def norm(v: Sequence[float]) -> float:
    return float(np.linalg.norm(np.asarray(v, dtype=float)))


def raw_to_levela(v: Sequence[float]) -> np.ndarray:
    return RAW_TO_LEVELA @ np.asarray(v, dtype=float)


def levela_to_raw(v: Sequence[float]) -> np.ndarray:
    return LEVELA_TO_RAW @ np.asarray(v, dtype=float)


def read_live_spice_t0(path: Path) -> float:
    r = json.loads(path.read_text())
    for key in ("ut_s", "t_s", "spice_t_s", "et_s"):
        if key in r:
            return float(r[key])
    raise KeyError(f"Could not find time field in {path}")


def read_leg_row(path: Path, leg: int) -> dict[str, str]:
    rows = list(csv.DictReader(open(path, newline="")))
    for row in rows:
        if "leg" in row and row["leg"]:
            try:
                val = int(float(row["leg"]))
            except Exception:
                continue
            if val == leg or val == leg - 1:
                return row
    return rows[leg - 1]


def frow(row: dict[str, str], key: str) -> float:
    return float(row[key])


def row_vec(row: dict[str, str], prefix: str, suffix: str = "") -> np.ndarray:
    # e.g. prefix="start", suffix="_raw_m" -> start_x_raw_m...
    return np.array([
        frow(row, f"{prefix}_x{suffix}"),
        frow(row, f"{prefix}_y{suffix}"),
        frow(row, f"{prefix}_z{suffix}"),
    ], dtype=float)


def row_v(row: dict[str, str], mode: str) -> np.ndarray:
    if mode == "optimized":
        p = "optimized"
    elif mode == "start":
        p = "start"
    elif mode == "initial":
        p = "initial"
    else:
        raise ValueError(mode)
    return np.array([
        frow(row, f"{p}_vx_raw_m_s"),
        frow(row, f"{p}_vy_raw_m_s"),
        frow(row, f"{p}_vz_raw_m_s"),
    ], dtype=float)


def spice_body_state_raw(body: str, et_s: float, center: str = "SUN", frame: str = "J2000") -> tuple[np.ndarray, np.ndarray]:
    state, _ = spice.spkezr(body.upper(), float(et_s), frame, "NONE", center.upper())
    r_levela_m = np.asarray(state[:3], dtype=float) * 1000.0
    v_levela_m_s = np.asarray(state[3:], dtype=float) * 1000.0
    return levela_to_raw(r_levela_m), levela_to_raw(v_levela_m_s)


def make_client(args) -> VesselPropnClient:
    kwargs = dict(response_timeout_s=args.server_timeout_s, quiet_stderr=args.quiet_stderr)
    sig = inspect.signature(VesselPropnClient)
    if "plugin_arg_mode" in sig.parameters:
        return VesselPropnClient(args.server, args.plugin_b64, plugin_arg_mode=args.plugin_arg_mode, **kwargs)
    if args.plugin_arg_mode == "positional":
        return VesselPropnClient([str(args.server), str(args.plugin_b64)], plugin_b64=None, **kwargs)
    return VesselPropnClient(args.server, args.plugin_b64, **kwargs)


def err_block(name: str, a: np.ndarray, b: np.ndarray) -> dict[str, Any]:
    d = a - b
    return {
        "name": name,
        "err_m": norm(d),
        "err_km": norm(d) / 1000.0,
        "a_norm_km": norm(a) / 1000.0,
        "b_norm_km": norm(b) / 1000.0,
        "delta": d.tolist(),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Diagnose matchpoint frame/epoch mismatches.")
    ap.add_argument("--server", required=True)
    ap.add_argument("--plugin-b64", type=Path, required=True)
    ap.add_argument("--plugin-arg-mode", choices=["option", "positional"], default="option")
    ap.add_argument("--vessel-guid", required=True)
    ap.add_argument("--live-state-json", type=Path, required=True)
    ap.add_argument("--leg-optimizations", type=Path, required=True)
    ap.add_argument("--leg", type=int, default=1)
    ap.add_argument("--bsp", type=Path, required=True)
    ap.add_argument("--tpc", type=Path, required=True)
    ap.add_argument("--dep-body", default="KERBIN")
    ap.add_argument("--arr-body", default="EVE")
    ap.add_argument("--center", default="SUN")
    ap.add_argument("--frame", default="J2000")
    ap.add_argument("--quiet-stderr", action="store_true")
    ap.add_argument("--server-timeout-s", type=float, default=180.0)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()

    spice.kclear()
    spice.furnsh(str(args.tpc))
    spice.furnsh(str(args.bsp))

    live_t = read_live_spice_t0(args.live_state_json)
    row = read_leg_row(args.leg_optimizations, args.leg)

    t_dep = frow(row, "t_dep_s")
    t_start = frow(row, "t_start_s")
    match_dt = t_start - live_t

    row_start = row_vec(row, "start", "_raw_m")
    row_target = row_vec(row, "target", "_raw_m")
    row_start_v = row_v(row, "start")
    row_opt_v = row_v(row, "optimized")

    dep_live_r, dep_live_v = spice_body_state_raw(args.dep_body, live_t, args.center, args.frame)
    dep_tdep_r, dep_tdep_v = spice_body_state_raw(args.dep_body, t_dep, args.center, args.frame)
    dep_tstart_r, dep_tstart_v = spice_body_state_raw(args.dep_body, t_start, args.center, args.frame)
    arr_tstart_r, arr_tstart_v = spice_body_state_raw(args.arr_body, t_start, args.center, args.frame)

    with make_client(args) as client:
        no_burn = client.vpropn("diag_match_noburn", args.vessel_guid, match_dt, [], timeout_s=args.server_timeout_s)
        # Also request a 1-second state for the canonical initial fields.
        initial = client.vpropn("diag_initial", args.vessel_guid, 1.0, [], timeout_s=args.server_timeout_s)

    final_r = np.asarray(no_burn["final_r_raw_m"], dtype=float)
    final_v = np.asarray(no_burn["final_v_raw_m_s"], dtype=float)
    server_parent_final = np.asarray(no_burn["final_parent_r_m"], dtype=float)
    server_parent_initial = np.asarray(initial["initial_parent_r_m"], dtype=float)
    server_initial_r = np.asarray(initial["initial_r_raw_m"], dtype=float)

    # Hypotheses:
    # H0: CSV "raw" is same raw frame.
    # H1: CSV is actually LevelA and should be converted to raw.
    # H2: VPROPN final raw converted to LevelA should compare to CSV.
    csv_as_raw = row_start
    csv_levela_to_raw = levela_to_raw(row_start)
    final_raw_to_levela = raw_to_levela(final_r)

    comparisons = [
        err_block("VPROPN_final_raw vs CSV_start_raw_as_is", final_r, csv_as_raw),
        err_block("VPROPN_final_raw vs CSV_start_if_CSV_were_LevelA", final_r, csv_levela_to_raw),
        err_block("VPROPN_final_LevelA vs CSV_start_as_if_LevelA", final_raw_to_levela, row_start),
        err_block("CSV_start_raw_as_is vs SPICE_dep_t_start_raw", csv_as_raw, dep_tstart_r),
        err_block("CSV_start_if_LevelA_to_raw vs SPICE_dep_t_start_raw", csv_levela_to_raw, dep_tstart_r),
        err_block("VPROPN_final_raw vs SPICE_dep_t_start_raw", final_r, dep_tstart_r),
        err_block("server_final_parent_raw vs SPICE_dep_t_start_raw", server_parent_final, dep_tstart_r),
        err_block("server_initial_parent_raw vs SPICE_dep_live_raw", server_parent_initial, dep_live_r),
        err_block("server_initial_vessel_raw vs SPICE_dep_live_raw", server_initial_r, dep_live_r),
        err_block("CSV_start_raw_as_is vs SPICE_arr_t_start_raw", csv_as_raw, arr_tstart_r),
    ]

    relative = {
        "csv_start_minus_spice_dep_tstart_km": norm(csv_as_raw - dep_tstart_r) / 1000.0,
        "csv_levela_to_raw_minus_spice_dep_tstart_km": norm(csv_levela_to_raw - dep_tstart_r) / 1000.0,
        "vpropn_final_minus_spice_dep_tstart_km": norm(final_r - dep_tstart_r) / 1000.0,
        "vpropn_final_minus_server_parent_final_km": norm(final_r - server_parent_final) / 1000.0,
        "csv_start_minus_server_parent_final_km": norm(csv_as_raw - server_parent_final) / 1000.0,
        "kerbin_to_eve_at_tstart_km": norm(dep_tstart_r - arr_tstart_r) / 1000.0,
    }

    summary = {
        "live_t_s": live_t,
        "t_dep_s": t_dep,
        "t_start_s": t_start,
        "match_dt_s": match_dt,
        "server_t0_game_s": initial["t0_game_s"],
        "server_t1_game_s_match": no_burn["t1_game_s"],
        "row": {
            "dep_body": row.get("dep_body"),
            "arr_body": row.get("arr_body"),
            "transform": row.get("transform"),
            "path": row.get("path"),
        },
        "vectors": {
            "csv_start_raw_m": csv_as_raw.tolist(),
            "csv_start_if_levela_to_raw_m": csv_levela_to_raw.tolist(),
            "csv_start_raw_to_levela_m": raw_to_levela(csv_as_raw).tolist(),
            "vpropn_final_raw_m": final_r.tolist(),
            "vpropn_final_levela_m": final_raw_to_levela.tolist(),
            "spice_dep_tstart_raw_m": dep_tstart_r.tolist(),
            "spice_arr_tstart_raw_m": arr_tstart_r.tolist(),
            "server_parent_final_raw_m": server_parent_final.tolist(),
            "server_parent_initial_raw_m": server_parent_initial.tolist(),
            "server_initial_vessel_raw_m": server_initial_r.tolist(),
            "csv_start_v_raw_m_s": row_start_v.tolist(),
            "csv_optimized_v_raw_m_s": row_opt_v.tolist(),
            "vpropn_final_v_raw_m_s": final_v.tolist(),
            "spice_dep_tstart_v_raw_m_s": dep_tstart_v.tolist(),
        },
        "relative_diagnostics": relative,
        "comparisons": comparisons,
        "raw_server_no_burn": no_burn,
        "raw_server_initial": initial,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = args.output_dir / "matchpoint_frame_epoch_diagnostic.json"
    out.write_text(json.dumps(summary, indent=2) + "\n")

    print("=== MATCHPOINT FRAME/EPOCH DIAGNOSTIC ===")
    print(f"live_t_s       : {live_t}")
    print(f"t_start_s      : {t_start}")
    print(f"match_dt_s     : {match_dt}")
    print(f"row transform  : {row.get('transform')}")
    print("")
    print("Key distances:")
    for k, v in relative.items():
        print(f"  {k:52s}: {v:14.3f} km")
    print("")
    print("Best comparison hypotheses:")
    for c in sorted(comparisons, key=lambda x: x["err_km"])[:10]:
        print(f"  {c['name']:58s}: {c['err_km']:14.3f} km")
    print(f"[OK] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
