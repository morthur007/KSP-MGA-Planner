#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import spiceypy as spice


def parse_csv_floats(s: str | None) -> list[float]:
    if not s:
        return []
    return [float(x.strip()) for x in str(s).replace(";", ",").split(",") if x.strip()]


def load_helpers():
    # This script is intended to live in scripts/ beside the v0.6 solver.
    # Reusing its server/SPICE helpers keeps the protocol and frame contract identical.
    here = Path(__file__).resolve().parent
    sys.path.insert(0, str(here))
    return importlib.import_module("solve_leg1_lambert_trust_mp_de_v0_6_parking_gate")


def specific_energy(v_rel: np.ndarray, r_rel: np.ndarray, mu: float) -> float:
    r = float(np.linalg.norm(r_rel))
    v = float(np.linalg.norm(v_rel))
    return 0.5 * v * v - mu / r


def maybe_radius_km(mod, body: str, catalog: Path | None) -> float:
    try:
        return float(mod.load_radius_km(body, catalog, None))
    except Exception:
        return math.nan


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Diagnose whether live_state is really in parking orbit and compare simple prograde transfer baselines."
    )
    ap.add_argument("--live-state-json", type=Path, required=True)
    ap.add_argument("--plugin-b64", type=Path, required=True)
    ap.add_argument("--server", required=True)
    ap.add_argument("--bsp", type=Path, required=True)
    ap.add_argument("--tpc", type=Path, required=True)
    ap.add_argument("--body-catalog", type=Path, default=None)
    ap.add_argument("--dep-body", default="KERBIN")
    ap.add_argument("--arr-body", default="EVE")
    ap.add_argument("--leg-optimizations", type=Path, default=None)
    ap.add_argument("--leg", type=int, default=1)
    ap.add_argument("--arrival-time-s", type=float, default=None)
    ap.add_argument("--center", default="SUN")
    ap.add_argument("--frame", default="J2000")

    ap.add_argument("--phase-offsets-s", default="0,600,1200,1800,2400,3000,3600,5400,7200,10800,14400,21600")
    ap.add_argument("--prograde-dv-m-s", default="1600,1800,2000,2200,2400")
    ap.add_argument("--arrival-offset-days", default="-10,-7,-5,-3,-1,0,1,3,5,7,10")
    ap.add_argument("--parking-radius-min-km", type=float, default=0.0)
    ap.add_argument("--parking-radius-max-km", type=float, default=10000.0)
    ap.add_argument("--parking-max-abs-radial-v-m-s", type=float, default=250.0)
    ap.add_argument("--quiet-stderr", action="store_true")
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()

    mod = load_helpers()

    spice.kclear()
    spice.furnsh(str(args.tpc))
    spice.furnsh(str(args.bsp))

    live = json.loads(args.live_state_json.read_text())
    live_t = float(live["ut_s"])
    live_r = np.array(live["r_raw_m"], dtype=float)
    live_v = np.array(live["v_raw_m_s"], dtype=float)

    mu_dep = float(mod.body_mu_m3_s2(args.dep_body))
    radius_km = maybe_radius_km(mod, args.dep_body, args.body_catalog)

    if args.arrival_time_s is not None:
        arrival_nom = float(args.arrival_time_s)
    elif args.leg_optimizations is not None:
        leg_row = mod.read_leg_row(args.leg_optimizations, args.leg)
        arrival_nom = float(leg_row["t_end_s"])
    else:
        raise SystemExit("[FAIL] pass --arrival-time-s or --leg-optimizations")

    phase_offsets = parse_csv_floats(args.phase_offsets_s)
    dvs = parse_csv_floats(args.prograde_dv_m_s)
    arrival_offsets_s = [d * 86400.0 for d in parse_csv_floats(args.arrival_offset_days)]

    args.output_dir.mkdir(parents=True, exist_ok=True)

    def rel_state_at(t: float, srv) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if abs(t - live_t) < 1e-6:
            r = live_r.copy()
            v = live_v.copy()
        else:
            # zero impulse at t, so the server reports the pre-burn state at t.
            res = srv.propn(f"state_{t:.3f}", live_t, t, live_r, live_v, [(t, np.zeros(3))])
            if res.status != "ok" or not res.burns:
                raise RuntimeError(f"propagate to {t} failed: {res.status} {res.message}")
            r = res.burns[0].r_m
            v = res.burns[0].v_before_m_s
        body_r, body_v = mod.body_state_raw(args.dep_body, t, args.center, args.frame)
        return r - body_r, v - body_v, r, v

    parking_rows: list[dict[str, Any]] = []
    baseline_rows: list[dict[str, Any]] = []

    with mod.ServerSession(str(args.server), str(args.plugin_b64), bool(args.quiet_stderr)) as srv:
        if not srv.ping():
            raise SystemExit("[FAIL] server ping failed")

        print("=== PARKING STATE / NO-BURN DIAGNOSTIC ===")
        print(f"live_t       : {live_t}")
        print(f"dep_body     : {args.dep_body}")
        print(f"arr_body     : {args.arr_body}")
        print(f"radius_km    : {radius_km}")
        print(f"arrival_nom  : {arrival_nom}")

        for off in phase_offsets:
            t = live_t + float(off)
            rel_r, rel_v, abs_r, abs_v = rel_state_at(t, srv)
            dist_km = float(np.linalg.norm(rel_r)) / 1000.0
            alt_km = dist_km - radius_km if math.isfinite(radius_km) else math.nan
            speed = float(np.linalg.norm(rel_v))
            vr = float(mod.radial_velocity(rel_r, rel_v))
            eps = specific_energy(rel_v, rel_r, mu_dep)
            h = np.cross(rel_r, rel_v)
            h_norm = float(np.linalg.norm(h))
            ecc_vec = np.cross(rel_v, h) / mu_dep - rel_r / np.linalg.norm(rel_r)
            ecc = float(np.linalg.norm(ecc_vec))
            parking_like = (
                args.parking_radius_min_km <= dist_km <= args.parking_radius_max_km
                and abs(vr) <= args.parking_max_abs_radial_v_m_s
                and eps < 0.0
            )
            row = {
                "offset_s": off,
                "t_s": t,
                "dist_km": dist_km,
                "alt_km": alt_km,
                "speed_m_s": speed,
                "radial_v_m_s": vr,
                "specific_energy_m2_s2": eps,
                "eccentricity": ecc,
                "parking_like": parking_like,
            }
            parking_rows.append(row)

        print("offset_s      dist_km      alt_km    speed  radial_v        eps        ecc parking")
        for r in parking_rows[:30]:
            print(f"{r['offset_s']:8.0f} {r['dist_km']:12.1f} {r['alt_km']:11.1f} {r['speed_m_s']:8.1f} {r['radial_v_m_s']:9.1f} {r['specific_energy_m2_s2']:10.3e} {r['eccentricity']:9.4f} {str(r['parking_like']):>7}")

        print("\n=== PURE PROGRADE BASELINE TO ARRIVAL WINDOW ===")
        print("This is a sanity baseline: if pure prograde beats the optimizer, the optimizer/export basis is wrong.")

        for off in phase_offsets:
            tb = live_t + float(off)
            try:
                rel_r, rel_v, abs_r, abs_v = rel_state_at(tb, srv)
            except Exception as e:
                baseline_rows.append({"offset_s": off, "status": f"preburn_error:{e}"})
                continue
            dist_km = float(np.linalg.norm(rel_r)) / 1000.0
            vr = float(mod.radial_velocity(rel_r, rel_v))
            _, T, _ = mod.tangent_angle_basis(rel_r, rel_v)

            for dv in dvs:
                dv_vec = float(dv) * T
                best = None
                for arr_off in arrival_offsets_s:
                    t_arr = arrival_nom + arr_off
                    if t_arr <= tb:
                        continue
                    res = srv.propn(
                        f"prog_{off:.0f}_{dv:.0f}_{arr_off:.0f}",
                        live_t,
                        t_arr,
                        live_r,
                        live_v,
                        [(tb, dv_vec)],
                    )
                    if res.status != "ok":
                        row = {
                            "offset_s": off, "tb_s": tb, "dv_m_s": dv,
                            "arrival_offset_days": arr_off / 86400.0,
                            "status": res.status, "message": res.message,
                        }
                    else:
                        arr_r, arr_v = mod.body_state_raw(args.arr_body, t_arr, args.center, args.frame)
                        pos_err_km = float(np.linalg.norm(res.final_r_m - arr_r)) / 1000.0
                        vinf_m_s = float(np.linalg.norm(res.final_v_m_s - arr_v))
                        row = {
                            "offset_s": off,
                            "tb_s": tb,
                            "burn_dist_km": dist_km,
                            "burn_radial_v_m_s": vr,
                            "dv_m_s": dv,
                            "arrival_offset_days": arr_off / 86400.0,
                            "t_arr_s": t_arr,
                            "status": "ok",
                            "final_pos_err_km": pos_err_km,
                            "arrival_vinf_m_s": vinf_m_s,
                        }
                    if row.get("status") == "ok":
                        if best is None or row["final_pos_err_km"] < best["final_pos_err_km"]:
                            best = row
                if best is not None:
                    baseline_rows.append(best)

    # write outputs
    with (args.output_dir / "parking_state_diagnostic.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(parking_rows[0].keys()))
        w.writeheader(); w.writerows(parking_rows)

    if baseline_rows:
        keys = sorted({k for r in baseline_rows for k in r.keys()})
        with (args.output_dir / "prograde_baseline.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader(); w.writerows(baseline_rows)

    ok_rows = [r for r in baseline_rows if r.get("status") == "ok"]
    ok_rows.sort(key=lambda r: float(r["final_pos_err_km"]))
    summary = {
        "n_parking_rows": len(parking_rows),
        "n_parking_like": sum(bool(r["parking_like"]) for r in parking_rows),
        "n_baseline_ok": len(ok_rows),
        "best_prograde": ok_rows[0] if ok_rows else None,
        "parking_gate": {
            "radius_min_km": args.parking_radius_min_km,
            "radius_max_km": args.parking_radius_max_km,
            "max_abs_radial_v_m_s": args.parking_max_abs_radial_v_m_s,
        },
    }
    (args.output_dir / "parking_departure_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    print("\n=== BEST PURE PROGRADE BASELINES ===")
    for i, r in enumerate(ok_rows[:20], 1):
        print(f"{i:2d} off={r['offset_s']:8.0f}s dv={r['dv_m_s']:7.1f} r={r['burn_dist_km']:9.1f}km vr={r['burn_radial_v_m_s']:8.1f} pos={r['final_pos_err_km']:12.1f}km vinf={r['arrival_vinf_m_s']:8.1f} arr_off={r['arrival_offset_days']:6.1f}d")

    print(f"[OK] wrote {args.output_dir / 'parking_state_diagnostic.csv'}")
    print(f"[OK] wrote {args.output_dir / 'prograde_baseline.csv'}")
    print(f"[OK] wrote {args.output_dir / 'parking_departure_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
