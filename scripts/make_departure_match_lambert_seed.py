#!/usr/bin/env python3
"""
Create a Lambert-based x0 seed for departure_relative_match_ipopt_v0_4_dsm.py.

This is not the final optimizer. It produces a physically informed first guess:
  1. Use VPROPN no-burn to get the real vessel relative state at burn_dt.
  2. Solve a two-body Lambert problem around the departure body from
     r_burn_rel to target_rel_r at t_start.
  3. Set dv0 = v_lambert_departure - v_burn_rel.
  4. Write x0 JSON compatible with v0.4.

Requires pykep.
"""

from __future__ import annotations

import argparse
import csv
import inspect
import json
import math
import sys
from pathlib import Path

import numpy as np
import spiceypy as spice

try:
    import pykep as pk
except Exception as exc:
    pk = None
    _PYKEP_IMPORT_ERROR = exc
else:
    _PYKEP_IMPORT_ERROR = None

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from vessel_server_client import VesselPropnClient


RAW_TO_LEVELA = np.array([[0.0, -1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]], dtype=float)
LEVELA_TO_RAW = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=float)


def norm(v):
    return float(np.linalg.norm(np.asarray(v, dtype=float)))


def levela_to_raw(v):
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
        if row.get("leg"):
            try:
                val = int(float(row["leg"]))
            except Exception:
                continue
            if val == leg or val == leg - 1:
                return row
    return rows[leg - 1]


def frow(row, key):
    return float(row[key])


def row_r(row, prefix):
    return np.array([frow(row, f"{prefix}_x_raw_m"), frow(row, f"{prefix}_y_raw_m"), frow(row, f"{prefix}_z_raw_m")], dtype=float)


def row_v(row, mode):
    p = {"optimized": "optimized", "start": "start", "initial": "initial"}[mode]
    return np.array([frow(row, f"{p}_vx_raw_m_s"), frow(row, f"{p}_vy_raw_m_s"), frow(row, f"{p}_vz_raw_m_s")], dtype=float)


def spice_body_state_raw(body, et_s, center="SUN", frame="J2000"):
    state, _ = spice.spkezr(body.upper(), float(et_s), frame, "NONE", center.upper())
    r_levela_m = np.asarray(state[:3], dtype=float) * 1000.0
    v_levela_m_s = np.asarray(state[3:], dtype=float) * 1000.0
    return levela_to_raw(r_levela_m), levela_to_raw(v_levela_m_s)


def make_client(args):
    kwargs = dict(response_timeout_s=args.server_timeout_s, quiet_stderr=args.quiet_stderr)
    sig = inspect.signature(VesselPropnClient)
    if "plugin_arg_mode" in sig.parameters:
        return VesselPropnClient(args.server, args.plugin_b64, plugin_arg_mode=args.plugin_arg_mode, **kwargs)
    if args.plugin_arg_mode == "positional":
        return VesselPropnClient([str(args.server), str(args.plugin_b64)], plugin_b64=None, **kwargs)
    return VesselPropnClient(args.server, args.plugin_b64, **kwargs)


def find_mu(body_catalog: Path, body: str) -> float:
    data = json.loads(body_catalog.read_text())
    body_l = body.lower()

    candidates = []
    if isinstance(data, dict):
        candidates.append(data)
        for key in ("bodies", "Bodies", "catalog", "items"):
            if key in data:
                candidates.append(data[key])

    def walk(obj):
        if isinstance(obj, dict):
            # dictionary keyed by body name
            for k, v in obj.items():
                if str(k).lower() == body_l and isinstance(v, dict):
                    yield v
            # object with name/body fields
            name = str(obj.get("name", obj.get("body", obj.get("id", "")))).lower()
            if name == body_l:
                yield obj
            for v in obj.values():
                yield from walk(v)
        elif isinstance(obj, list):
            for x in obj:
                yield from walk(x)

    for obj in walk(data):
        for key in (
            "mu_m3_s2", "gm_m3_s2", "gravitational_parameter_m3_s2",
            "mu", "gm", "gravitational_parameter",
        ):
            if key in obj:
                val = float(obj[key])
                # Heuristic: if given in km^3/s^2, convert to m^3/s^2.
                if val < 1e12:
                    val *= 1e9
                return val

    raise KeyError(f"Could not find gravitational parameter for {body} in {body_catalog}")


def solve_lambert_all(r1, r2, tof_s, mu, max_revs):
    sols = []
    for cw in (False, True):
        try:
            lp = pk.lambert_problem(r1=r1.tolist(), r2=r2.tolist(), tof=float(tof_s), mu=float(mu), cw=cw, max_revs=int(max_revs))
            v1s = lp.get_v1()
            v2s = lp.get_v2()
            for i, (v1, v2) in enumerate(zip(v1s, v2s)):
                sols.append({"cw": cw, "index": i, "v1": np.asarray(v1, dtype=float), "v2": np.asarray(v2, dtype=float)})
        except Exception as exc:
            # Some geometries/cw/rev combinations may fail.
            continue
    return sols


def main():
    ap = argparse.ArgumentParser(description="Generate Lambert x0 seed for v0.4 relative match IPOPT.")
    ap.add_argument("--server", required=True)
    ap.add_argument("--plugin-b64", type=Path, required=True)
    ap.add_argument("--plugin-arg-mode", choices=["option", "positional"], default="option")
    ap.add_argument("--vessel-guid", required=True)
    ap.add_argument("--live-state-json", type=Path, required=True)
    ap.add_argument("--leg-optimizations", type=Path, required=True)
    ap.add_argument("--leg", type=int, default=1)
    ap.add_argument("--target-velocity-mode", choices=["optimized", "start", "initial"], default="optimized")
    ap.add_argument("--bsp", type=Path, required=True)
    ap.add_argument("--tpc", type=Path, required=True)
    ap.add_argument("--body-catalog", type=Path, required=True)
    ap.add_argument("--dep-body", default="KERBIN")
    ap.add_argument("--center", default="SUN")
    ap.add_argument("--frame", default="J2000")
    ap.add_argument("--burn-dt-s", type=float, default=None)
    ap.add_argument("--burn-dt-offset-s", type=float, default=0.0)
    ap.add_argument("--dsm-dt-fraction", type=float, default=0.35)
    ap.add_argument("--max-revs", type=int, default=0)
    ap.add_argument("--select", choices=["min_dv", "min_arrival_velocity_error"], default="min_dv")
    ap.add_argument("--dsm-initial-raw-m-s", default="0,0,0")
    ap.add_argument("--server-timeout-s", type=float, default=180.0)
    ap.add_argument("--quiet-stderr", action="store_true")
    ap.add_argument("--output-json", type=Path, required=True)
    args = ap.parse_args()

    if pk is None:
        raise SystemExit(f"[FAIL] pykep is not importable: {_PYKEP_IMPORT_ERROR}")

    spice.kclear()
    spice.furnsh(str(args.tpc))
    spice.furnsh(str(args.bsp))

    live_t = read_live_spice_t0(args.live_state_json)
    row = read_leg_row(args.leg_optimizations, args.leg)
    t_dep = frow(row, "t_dep_s")
    t_start = frow(row, "t_start_s")
    match_dt = t_start - live_t
    burn_dt = args.burn_dt_s if args.burn_dt_s is not None else (t_dep - live_t + args.burn_dt_offset_s)
    tof = match_dt - burn_dt
    if tof <= 0:
        raise SystemExit(f"[FAIL] Lambert TOF <= 0: match_dt={match_dt}, burn_dt={burn_dt}")

    target_abs_r = row_r(row, "start")
    target_abs_v = row_v(row, args.target_velocity_mode)
    dep_r_match, dep_v_match = spice_body_state_raw(args.dep_body, t_start, args.center, args.frame)
    target_rel_r = target_abs_r - dep_r_match
    target_rel_v = target_abs_v - dep_v_match
    mu = find_mu(args.body_catalog, args.dep_body)

    with make_client(args) as client:
        state = client.vpropn("lambert_seed_burn_state", args.vessel_guid, burn_dt, [], timeout_s=args.server_timeout_s)
    r1 = np.asarray(state["final_parent_r_m"], dtype=float)
    v1_current = np.asarray(state["final_parent_v_m_s"], dtype=float)

    sols = solve_lambert_all(r1, target_rel_r, tof, mu, args.max_revs)
    if not sols:
        raise SystemExit("[FAIL] no Lambert solutions found")

    ranked = []
    for s in sols:
        dv0 = s["v1"] - v1_current
        arr_vel_err = norm(s["v2"] - target_rel_v)
        ranked.append({
            **s,
            "dv0": dv0,
            "dv0_norm": norm(dv0),
            "arrival_velocity_error_m_s": arr_vel_err,
            "v1_current": v1_current,
        })

    if args.select == "min_arrival_velocity_error":
        best = min(ranked, key=lambda z: (z["arrival_velocity_error_m_s"], z["dv0_norm"]))
    else:
        best = min(ranked, key=lambda z: (z["dv0_norm"], z["arrival_velocity_error_m_s"]))

    dsm_vals = [float(x.strip()) for x in args.dsm_initial_raw_m_s.replace(";", ",").split(",") if x.strip()]
    if len(dsm_vals) != 3:
        raise SystemExit("--dsm-initial-raw-m-s must have 3 comma-separated floats")

    dsm_dt = burn_dt + args.dsm_dt_fraction * (match_dt - burn_dt)
    x = [float(burn_dt), *best["dv0"].tolist(), float(dsm_dt), *dsm_vals]

    out = {
        "x": x,
        "best": {
            "burn_dt_s": float(burn_dt),
            "dsm_dt_s": float(dsm_dt),
            "dv0_raw_m_s": best["dv0"].tolist(),
            "dv0_norm_m_s": best["dv0_norm"],
            "dsm_raw_m_s": dsm_vals,
            "dsm_norm_m_s": norm(dsm_vals),
            "tof_s": float(tof),
            "match_dt_s": float(match_dt),
            "target_rel_r_raw_m": target_rel_r.tolist(),
            "target_rel_v_raw_m_s": target_rel_v.tolist(),
            "burn_rel_r_raw_m": r1.tolist(),
            "burn_rel_v_raw_m_s": v1_current.tolist(),
            "lambert_v_depart_raw_m_s": best["v1"].tolist(),
            "lambert_v_arrive_raw_m_s": best["v2"].tolist(),
            "lambert_arrival_velocity_error_m_s": best["arrival_velocity_error_m_s"],
            "mu_m3_s2": mu,
            "cw": best["cw"],
            "solution_index": best["index"],
        },
        "ranked": [
            {
                "cw": z["cw"],
                "index": z["index"],
                "dv0_norm_m_s": z["dv0_norm"],
                "arrival_velocity_error_m_s": z["arrival_velocity_error_m_s"],
                "dv0_raw_m_s": z["dv0"].tolist(),
                "lambert_v_depart_raw_m_s": z["v1"].tolist(),
                "lambert_v_arrive_raw_m_s": z["v2"].tolist(),
            }
            for z in ranked
        ],
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(out, indent=2) + "\n")

    print("=== LAMBERT DEPARTURE MATCH SEED ===")
    print(f"live_t       : {live_t}")
    print(f"burn_dt_s    : {burn_dt}")
    print(f"match_dt_s   : {match_dt}")
    print(f"tof_s        : {tof}")
    print(f"mu           : {mu}")
    print(f"target_r_km  : {norm(target_rel_r)/1000:.3f}")
    print(f"target_v_ms  : {norm(target_rel_v):.3f}")
    print(f"burn_r_km    : {norm(r1)/1000:.3f}")
    print(f"burn_v_ms    : {norm(v1_current):.3f}")
    print(f"best dv0     : {best['dv0_norm']:.3f} m/s")
    print(f"arr vel err  : {best['arrival_velocity_error_m_s']:.3f} m/s")
    print(f"x            : {x}")
    print(f"[OK] wrote {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
