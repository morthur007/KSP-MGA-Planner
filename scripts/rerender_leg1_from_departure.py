#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
from typing import Any

import numpy as np
import spiceypy as spice

from ksp_mga.native.impulse_server_client_v0_2 import PrincipiaImpulseServerV2


def norm(v: np.ndarray) -> float:
    return float(np.linalg.norm(v))


def raw_to_levela(v: np.ndarray) -> np.ndarray:
    x, y, z = v
    return np.array([-y, z, x], dtype=float)


def levela_to_raw(v: np.ndarray) -> np.ndarray:
    x, y, z = v
    return np.array([z, -x, y], dtype=float)


def safe_float(x: Any, default: float = math.inf) -> float:
    try:
        if x is None or x == "":
            return default
        v = float(x)
        if math.isnan(v):
            return default
        return v
    except Exception:
        return default


def arr(row: dict[str, str], *names: str) -> np.ndarray:
    return np.array([float(row[n]) for n in names], dtype=float)


def read_leg_row(path: Path, leg: int) -> dict[str, str]:
    rows = list(csv.DictReader(path.open()))
    for row in rows:
        if int(float(row["leg"])) == leg:
            return row
    raise SystemExit(f"[FAIL] leg {leg} not found in {path}")


def parse_float_list(s: str | None) -> list[float]:
    if s is None or str(s).strip() == "":
        return []
    return [float(x.strip()) for x in str(s).replace(";", ",").split(",") if x.strip()]


def body_state_raw(body: str, t_s: float, center: str, frame: str) -> tuple[np.ndarray, np.ndarray]:
    st, _ = spice.spkezr(body, t_s, frame, "NONE", center)
    r_levela = np.array([1000.0 * st[0], 1000.0 * st[1], 1000.0 * st[2]], dtype=float)
    v_levela = np.array([1000.0 * st[3], 1000.0 * st[4], 1000.0 * st[5]], dtype=float)
    return levela_to_raw(r_levela), levela_to_raw(v_levela)


def load_departures(path: Path, max_departures: int | None = None) -> list[dict[str, Any]]:
    obj = json.loads(path.read_text())
    if isinstance(obj, dict):
        rows = obj.get("departures", [])
    elif isinstance(obj, list):
        rows = obj
    else:
        raise SystemExit(f"[FAIL] unsupported reachable departures schema in {path}")
    rows = [r for r in rows if r.get("status", "OK") == "OK"]
    rows.sort(key=lambda r: safe_float(r.get("reference_vinf_vec_err_m_s"), math.inf))
    if max_departures is not None:
        rows = rows[:max_departures]
    return rows


def propagate_state(srv: PrincipiaImpulseServerV2, req_id: str, t0: float, t1: float, r0: np.ndarray, v0: np.ndarray, impulses: list[tuple[float, np.ndarray]]) -> tuple[np.ndarray, np.ndarray, str, str]:
    res = srv.propagate_n(
        req_id=req_id,
        t0_s=float(t0),
        t1_s=float(t1),
        r0_m=r0,
        v0_m_s=v0,
        impulses=impulses,
    )
    if res.status != "ok" or res.final_r_m is None or res.final_v_m_s is None:
        return np.full(3, np.nan), np.full(3, np.nan), res.status, res.message
    return np.array(res.final_r_m, dtype=float), np.array(res.final_v_m_s, dtype=float), res.status, res.message


def solve_dsm_to_target_position(
    srv: PrincipiaImpulseServerV2,
    cfg: dict[str, Any],
    departure: dict[str, Any],
    t_arr: float,
    dsm_fraction: float,
) -> dict[str, Any]:
    live_t = float(cfg["live_t"])
    live_r = np.array(cfg["live_r"], dtype=float)
    live_v = np.array(cfg["live_v"], dtype=float)
    tb0 = float(departure["tb0_s"])
    dv0 = np.array(departure["dv0_raw_m_s"], dtype=float)
    dep_id = str(departure.get("departure_id", "dep"))

    min_dsm_after_burn_s = float(cfg["min_dsm_after_burn_s"])
    max_dsm_before_arrival_s = float(cfg["max_dsm_before_arrival_s"])
    t_dsm = tb0 + dsm_fraction * (t_arr - tb0)
    t_dsm = max(t_dsm, tb0 + min_dsm_after_burn_s)
    t_dsm = min(t_dsm, t_arr - max_dsm_before_arrival_s)
    if t_dsm <= tb0 or t_dsm >= t_arr:
        return {
            "status": "BAD_TIMING",
            "departure_id": dep_id,
            "tb0_s": tb0,
            "t_dsm_s": t_dsm,
            "t_arr_s": t_arr,
            "dsm_fraction": dsm_fraction,
        }

    # State just before DSM, after real departure burn0.
    r_dsm, v_dsm, st, msg = propagate_state(
        srv,
        req_id=f"pre_dsm_{os.getpid()}_{dep_id}_{t_arr:.1f}_{dsm_fraction:.3f}",
        t0=live_t,
        t1=t_dsm,
        r0=live_r,
        v0=live_v,
        impulses=[(tb0, dv0)],
    )
    if st != "ok":
        return {"status": "PRE_DSM_FAILED", "server_status": st, "server_message": msg, "departure_id": dep_id}

    target_r, target_v = body_state_raw(cfg["arr_body"], t_arr, cfg["center"], cfg["frame"])

    fd_step = float(cfg["fd_step_m_s"])
    max_dsm = float(cfg["max_dsm_m_s"])
    dsm = np.zeros(3, dtype=float)
    last_base_r = None
    last_base_v = None
    clipped = False

    for it in range(int(cfg["iterations"])):
        base_r, base_v, st, msg = propagate_state(
            srv,
            req_id=f"base_{os.getpid()}_{dep_id}_{it}_{t_arr:.1f}_{dsm_fraction:.3f}",
            t0=t_dsm,
            t1=t_arr,
            r0=r_dsm,
            v0=v_dsm + dsm,
            impulses=[],
        )
        if st != "ok":
            return {"status": "BASE_FAILED", "server_status": st, "server_message": msg, "departure_id": dep_id}
        last_base_r, last_base_v = base_r, base_v
        err = target_r - base_r
        J = np.zeros((3, 3), dtype=float)
        for j in range(3):
            dd = np.zeros(3, dtype=float); dd[j] = fd_step
            rp, vp, stp, msgp = propagate_state(
                srv,
                req_id=f"fd_{os.getpid()}_{dep_id}_{it}_{j}_{t_arr:.1f}_{dsm_fraction:.3f}",
                t0=t_dsm,
                t1=t_arr,
                r0=r_dsm,
                v0=v_dsm + dsm + dd,
                impulses=[],
            )
            if stp != "ok":
                return {"status": "FD_FAILED", "server_status": stp, "server_message": msgp, "departure_id": dep_id}
            J[:, j] = (rp - base_r) / fd_step
        try:
            delta, *_ = np.linalg.lstsq(J, err, rcond=None)
        except np.linalg.LinAlgError:
            return {"status": "SINGULAR_J", "departure_id": dep_id}
        dsm = dsm + delta
        dsm_norm = norm(dsm)
        if dsm_norm > max_dsm:
            dsm = dsm * (max_dsm / dsm_norm)
            clipped = True
            break
        if norm(delta) < float(cfg["delta_stop_m_s"]):
            break

    final_r, final_v, st, msg = propagate_state(
        srv,
        req_id=f"final_{os.getpid()}_{dep_id}_{t_arr:.1f}_{dsm_fraction:.3f}",
        t0=t_dsm,
        t1=t_arr,
        r0=r_dsm,
        v0=v_dsm + dsm,
        impulses=[],
    )
    if st != "ok":
        return {"status": "FINAL_FAILED", "server_status": st, "server_message": msg, "departure_id": dep_id}

    pos_err_km = norm(final_r - target_r) / 1000.0
    vel_err_m_s = norm(final_v - target_v)
    rel_vinf_arr_m_s = norm(final_v - target_v)
    dsm_norm = norm(dsm)
    dv0_norm = norm(dv0)
    score = (
        pos_err_km / float(cfg["pos_scale_km"])
        + vel_err_m_s / float(cfg["vel_scale_m_s"]) * float(cfg["vel_weight"])
        + dsm_norm / float(cfg["dsm_scale_m_s"]) * float(cfg["dsm_weight"])
        + dv0_norm / float(cfg["dv0_scale_m_s"]) * float(cfg["dv0_weight"])
    )

    valid = True
    reasons: list[str] = []
    if pos_err_km > float(cfg["accept_pos_km"]):
        valid = False; reasons.append("pos_err_too_large")
    if dsm_norm > float(cfg["accept_dsm_m_s"]):
        valid = False; reasons.append("dsm_too_large")
    if clipped:
        valid = False; reasons.append("dsm_clipped")

    return {
        "status": "OK",
        "solution_valid": valid,
        "invalid_reasons": reasons,
        "score": score,
        "departure_id": dep_id,
        "selection_sources": departure.get("selection_sources", []),
        "tb0_s": tb0,
        "t_dsm_s": t_dsm,
        "t_arr_s": t_arr,
        "dsm_fraction": dsm_fraction,
        "dv0_raw_m_s": dv0.tolist(),
        "dv0_levela_m_s": raw_to_levela(dv0).tolist(),
        "dv0_norm_m_s": dv0_norm,
        "dsm_raw_m_s": dsm.tolist(),
        "dsm_levela_m_s": raw_to_levela(dsm).tolist(),
        "dsm_norm_m_s": dsm_norm,
        "dsm_clipped": clipped,
        "final_pos_err_km": pos_err_km,
        "final_vel_err_m_s": vel_err_m_s,
        "arrival_rel_vinf_m_s": rel_vinf_arr_m_s,
        "target_r_raw_m": target_r.tolist(),
        "target_v_raw_m_s": target_v.tolist(),
        "final_r_raw_m": final_r.tolist(),
        "final_v_raw_m_s": final_v.tolist(),
        "r_dsm_raw_m": r_dsm.tolist(),
        "v_dsm_before_raw_m_s": v_dsm.tolist(),
        "v_dsm_after_raw_m_s": (v_dsm + dsm).tolist(),
        "source_departure": departure,
    }


def worker(payload: tuple[list[dict[str, Any]], dict[str, Any]]) -> list[dict[str, Any]]:
    departures, cfg = payload
    spice.kclear(); spice.furnsh(str(cfg["tpc"])); spice.furnsh(str(cfg["bsp"]))
    out: list[dict[str, Any]] = []
    t_arrivals = cfg["arrival_times_s"]
    fractions = cfg["dsm_fractions"]
    with PrincipiaImpulseServerV2(cfg["server"], cfg["plugin_b64"]) as srv:
        if not srv.ping():
            raise RuntimeError("server PING failed in worker")
        for dep in departures:
            for t_arr in t_arrivals:
                for frac in fractions:
                    try:
                        out.append(solve_dsm_to_target_position(srv, cfg, dep, float(t_arr), float(frac)))
                    except Exception as e:
                        out.append({
                            "status": "EXCEPTION",
                            "departure_id": dep.get("departure_id"),
                            "error": repr(e),
                            "t_arr_s": t_arr,
                            "dsm_fraction": frac,
                        })
    return out


def chunked(xs: list[Any], n: int) -> list[list[Any]]:
    if n <= 1:
        return [[x] for x in xs]
    return [xs[i:i+n] for i in range(0, len(xs), n)]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    flat: list[dict[str, Any]] = []
    for r in rows:
        rr: dict[str, Any] = {}
        for k, v in r.items():
            if isinstance(v, (list, dict)):
                rr[k] = json.dumps(v, separators=(",", ":"))
            else:
                rr[k] = v
            if k not in fields:
                fields.append(k)
        flat.append(rr)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(flat)


def make_mission_events(best: dict[str, Any], args: argparse.Namespace) -> list[dict[str, Any]]:
    base = {
        "enabled": True,
        "vessel_guid": args.vessel_guid,
        "ensure_flight_plan": True,
        "extend_existing_flight_plan": True,
        "mass_tonnes": args.mass_tonnes,
        "insert_index": -1,
        "burn_template": "json",
        "thrust_kN": args.thrust_kN,
        "specific_impulse_s_g0": args.isp_s,
        "is_inertially_fixed": False,
        "frame_extension": 6000,
        "frame_centre_from_active_body": True,
        "frame_centre_index": -1,
        "frame_primary_index": -1,
        "frame_secondary_index": -1,
        "placeholder_dv_m_s": 0.001,
        "require_status_ok": True,
        "cleanup_on_error": True,
        "tolerance_time_s": 0.01,
        "tolerance_dv_m_s": 1e-6,
        "one_shot": True,
        "disable_after_success": True,
    }
    events = []
    for key, t_field, dv_field in [
        ("departure_burn0", "tb0_s", "dv0_levela_m_s"),
        ("leg1_dsm", "t_dsm_s", "dsm_levela_m_s"),
    ]:
        ev = dict(base)
        ev.update({
            "request_id": f"{args.event_prefix}_{key}_attempt0",
            "dedupe_tag": f"{args.event_prefix}_{key}",
            "event_key": f"{args.event_prefix}_{key}",
            "attempt": 0,
            "mode": "insert_levela",
            "initial_time": float(best[t_field]),
            "plan_final_time": float(best[t_field]) + args.plan_duration_s,
            "delta_v_levela_m_s": best[dv_field],
        })
        events.append(ev)
    return events


def main() -> int:
    ap = argparse.ArgumentParser(description="Re-render leg 1 from real reachable LKO departures using one DSM cleanup.")
    ap.add_argument("--reachable-departures", type=Path, required=True)
    ap.add_argument("--plugin-b64", required=True)
    ap.add_argument("--server", default="principia_impulsive_particle_server")
    ap.add_argument("--live-state-json", type=Path, required=True)
    ap.add_argument("--leg-optimizations", type=Path, required=True)
    ap.add_argument("--leg", type=int, default=1)
    ap.add_argument("--bsp", type=Path, required=True)
    ap.add_argument("--tpc", type=Path, required=True)
    ap.add_argument("--arr-body", default="EVE")
    ap.add_argument("--center", default="SUN")
    ap.add_argument("--frame", default="J2000")
    ap.add_argument("--max-departures", type=int, default=40)
    ap.add_argument("--arrival-offset-days", default="-5,-3,-1,0,1,3,5")
    ap.add_argument("--dsm-fractions", default="0.05,0.10,0.20,0.35")
    ap.add_argument("--min-dsm-after-burn-s", type=float, default=1800.0)
    ap.add_argument("--max-dsm-before-arrival-s", type=float, default=86400.0)
    ap.add_argument("--fd-step-m-s", type=float, default=1.0)
    ap.add_argument("--iterations", type=int, default=2)
    ap.add_argument("--delta-stop-m-s", type=float, default=0.05)
    ap.add_argument("--max-dsm-m-s", type=float, default=2500.0)
    ap.add_argument("--accept-pos-km", type=float, default=100000.0)
    ap.add_argument("--accept-dsm-m-s", type=float, default=1200.0)
    ap.add_argument("--pos-scale-km", type=float, default=100000.0)
    ap.add_argument("--vel-scale-m-s", type=float, default=1000.0)
    ap.add_argument("--dsm-scale-m-s", type=float, default=1000.0)
    ap.add_argument("--dv0-scale-m-s", type=float, default=5000.0)
    ap.add_argument("--vel-weight", type=float, default=0.15)
    ap.add_argument("--dsm-weight", type=float, default=0.50)
    ap.add_argument("--dv0-weight", type=float, default=0.05)
    ap.add_argument("--workers", type=int, default=max(1, min(4, os.cpu_count() or 1)))
    ap.add_argument("--chunk-size", type=int, default=1)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--vessel-guid", default="60735c81-7e29-4c06-9551-9e5283e37586")
    ap.add_argument("--mass-tonnes", type=float, default=2.6)
    ap.add_argument("--thrust-kN", type=float, default=2686.87701225281)
    ap.add_argument("--isp-s", type=float, default=1000.0)
    ap.add_argument("--event-prefix", default="rank12_rerender_leg1")
    ap.add_argument("--plan-duration-s", type=float, default=3600.0)
    args = ap.parse_args()

    live = json.loads(args.live_state_json.read_text())
    leg_row = read_leg_row(args.leg_optimizations, args.leg)
    t_end = float(leg_row["t_end_s"])
    arrival_offsets_days = parse_float_list(args.arrival_offset_days)
    arrival_times_s = [t_end + d * 86400.0 for d in arrival_offsets_days]
    dsm_fractions = parse_float_list(args.dsm_fractions)
    departures = load_departures(args.reachable_departures, args.max_departures)

    cfg = {
        "plugin_b64": args.plugin_b64,
        "server": args.server,
        "bsp": str(args.bsp),
        "tpc": str(args.tpc),
        "live_t": float(live["ut_s"]),
        "live_r": live["r_raw_m"],
        "live_v": live["v_raw_m_s"],
        "arr_body": args.arr_body,
        "center": args.center,
        "frame": args.frame,
        "arrival_times_s": arrival_times_s,
        "dsm_fractions": dsm_fractions,
        "min_dsm_after_burn_s": args.min_dsm_after_burn_s,
        "max_dsm_before_arrival_s": args.max_dsm_before_arrival_s,
        "fd_step_m_s": args.fd_step_m_s,
        "iterations": args.iterations,
        "delta_stop_m_s": args.delta_stop_m_s,
        "max_dsm_m_s": args.max_dsm_m_s,
        "accept_pos_km": args.accept_pos_km,
        "accept_dsm_m_s": args.accept_dsm_m_s,
        "pos_scale_km": args.pos_scale_km,
        "vel_scale_m_s": args.vel_scale_m_s,
        "dsm_scale_m_s": args.dsm_scale_m_s,
        "dv0_scale_m_s": args.dv0_scale_m_s,
        "vel_weight": args.vel_weight,
        "dsm_weight": args.dsm_weight,
        "dv0_weight": args.dv0_weight,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print("=== RERENDER LEG1 FROM REACHABLE DEPARTURES ===")
    print(f"departures     : {len(departures)}")
    print(f"arrivals       : {len(arrival_times_s)} offsets_days={arrival_offsets_days}")
    print(f"dsm fractions  : {dsm_fractions}")
    print(f"tasks          : {len(departures) * len(arrival_times_s) * len(dsm_fractions)}")
    print(f"workers        : {args.workers}")
    print(f"output_dir     : {args.output_dir}")

    rows: list[dict[str, Any]] = []
    chunks = chunked(departures, max(1, args.chunk_size))
    if args.workers <= 1:
        for ch in chunks:
            rows.extend(worker((ch, cfg)))
    else:
        with mp.Pool(processes=args.workers) as pool:
            for part in pool.imap_unordered(worker, [(ch, cfg) for ch in chunks]):
                rows.extend(part)
                ok = sum(1 for r in rows if r.get("status") == "OK")
                print(f"[progress] rows={len(rows)} ok={ok}", flush=True)

    ok_rows = [r for r in rows if r.get("status") == "OK"]
    ok_rows.sort(key=lambda r: safe_float(r.get("score"), math.inf))
    valid_rows = [r for r in ok_rows if r.get("solution_valid")]

    (args.output_dir / "leg1_rerender_all.json").write_text(json.dumps(rows, indent=2) + "\n")
    (args.output_dir / "leg1_rerender_top.json").write_text(json.dumps(ok_rows[:100], indent=2) + "\n")
    write_csv(args.output_dir / "leg1_rerender_summary.csv", ok_rows)

    best = ok_rows[0] if ok_rows else None
    summary = {
        "schema": "leg1_rerender_from_departure.v1",
        "n_rows": len(rows),
        "n_ok": len(ok_rows),
        "n_valid": len(valid_rows),
        "best": best,
        "best_valid": valid_rows[0] if valid_rows else None,
        "config": {k: v for k, v in vars(args).items() if k not in {"plugin_b64"}},
    }
    (args.output_dir / "leg1_rerender_result.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")

    if best:
        print("\n=== TOP RERENDER RESULTS ===")
        for i, r in enumerate(ok_rows[:20], start=1):
            print(
                f"{i:3d} valid={str(r.get('solution_valid')):<5} "
                f"dep={r.get('departure_id'):<10} "
                f"arr_off={(safe_float(r.get('t_arr_s')) - t_end)/86400.0:7.2f} d "
                f"frac={safe_float(r.get('dsm_fraction')):5.2f} "
                f"pos={safe_float(r.get('final_pos_err_km')):10.1f} km "
                f"vel={safe_float(r.get('final_vel_err_m_s')):8.1f} m/s "
                f"dv0={safe_float(r.get('dv0_norm_m_s')):8.1f} "
                f"dsm={safe_float(r.get('dsm_norm_m_s')):8.1f} "
                f"reasons={','.join(r.get('invalid_reasons', []))}"
            )
        events = make_mission_events(best, args)
        (args.output_dir / "mission_events_preview.json").write_text(json.dumps(events, indent=2) + "\n")
        with (args.output_dir / "mission_events_preview.jsonl").open("w") as f:
            for ev in events:
                f.write(json.dumps(ev, separators=(",", ":")) + "\n")
        print(f"[OK] wrote {args.output_dir / 'mission_events_preview.json'}")

    print(f"[OK] wrote {args.output_dir / 'leg1_rerender_result.json'}")
    print(f"[OK] wrote {args.output_dir / 'leg1_rerender_summary.csv'}")
    return 0 if valid_rows else 2


if __name__ == "__main__":
    raise SystemExit(main())
