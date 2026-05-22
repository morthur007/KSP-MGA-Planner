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


def unit(v: np.ndarray) -> np.ndarray:
    n = norm(v)
    if n <= 0:
        return np.full(3, np.nan)
    return v / n


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


def levela_to_raw(v: np.ndarray) -> np.ndarray:
    # LevelA/SPICE -> Principia raw = [+Z, -X, +Y]
    x, y, z = v
    return np.array([z, -x, y], dtype=float)


def raw_to_levela(v: np.ndarray) -> np.ndarray:
    x, y, z = v
    return np.array([-y, z, x], dtype=float)


def body_state_raw(body: str, t_s: float, center: str, frame: str) -> tuple[np.ndarray, np.ndarray]:
    st, _ = spice.spkezr(body, t_s, frame, "NONE", center)
    r_levela = np.array([1000.0 * st[0], 1000.0 * st[1], 1000.0 * st[2]], dtype=float)
    v_levela = np.array([1000.0 * st[3], 1000.0 * st[4], 1000.0 * st[5]], dtype=float)
    return levela_to_raw(r_levela), levela_to_raw(v_levela)


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


def read_handoff_rows(path: Path, max_rows: int | None, statuses: set[str]) -> list[dict[str, Any]]:
    obj = json.loads(path.read_text())
    if isinstance(obj, dict):
        rows = obj.get("rows", [])
        if not rows and obj.get("best"):
            rows = [obj["best"]]
    elif isinstance(obj, list):
        rows = obj
    else:
        raise SystemExit(f"[FAIL] unsupported handoff schema: {path}")
    rows = [r for r in rows if isinstance(r, dict)]
    if statuses:
        rows = [r for r in rows if str(r.get("status")) in statuses]
    rows.sort(key=lambda r: (
        0 if r.get("status") == "PASS" else 1 if r.get("status") == "POWERED" else 2,
        safe_float(r.get("powered_lower_bound_m_s"), math.inf),
        safe_float(r.get("leg1_pos_err_km"), math.inf),
    ))
    if max_rows is not None:
        rows = rows[:max_rows]
    return rows


def propagate_state(
    srv: PrincipiaImpulseServerV2,
    req_id: str,
    t0: float,
    t1: float,
    r0: np.ndarray,
    v0: np.ndarray,
    impulses: list[tuple[float, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray, str, str]:
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


def make_start_state(row: dict[str, Any], mode: str, cfg: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, float, float, list[str]]:
    """Return r_start, v_start, t_start, powered_cost_m_s, notes.

    Modes:
      ref_powered: use the outgoing v∞ vector from the audit reference; cost is |vinf_out|-|vinf_in| lower bound.
      same_mag_ref_dir: use the reference outgoing direction but set magnitude to incoming v∞ magnitude; cost 0 in magnitude, still assumes turn feasibility from audit.
    """
    t_event = float(row["t_arr_s"])
    body_r, body_v = body_state_raw(cfg["dep_body"], t_event, cfg["center"], cfg["frame"])
    # Prefer actual final position from rerendered leg1 if available, but default to body centre.
    src = row.get("source_rerender", {}) if isinstance(row.get("source_rerender"), dict) else {}
    if cfg.get("start_at_actual_arrival_position") and src.get("final_r_raw_m") is not None:
        r_start = np.array(src["final_r_raw_m"], dtype=float)
    else:
        r_start = body_r

    vinf_in = np.array(row.get("vinf_in_raw_m_s"), dtype=float)
    vinf_ref = np.array(row.get("vinf_out_ref_raw_m_s"), dtype=float)
    if vinf_ref.shape != (3,) or vinf_in.shape != (3,):
        raise ValueError("handoff row missing vinf_in_raw_m_s or vinf_out_ref_raw_m_s")
    notes: list[str] = []
    if mode == "ref_powered":
        vinf_out = vinf_ref
        powered_cost = safe_float(row.get("powered_lower_bound_m_s"), abs(norm(vinf_ref) - norm(vinf_in)))
        notes.append("vinf_out_ref_from_audit")
    elif mode == "same_mag_ref_dir":
        vinf_out = unit(vinf_ref) * norm(vinf_in)
        powered_cost = 0.0
        notes.append("vinf_out_ref_direction_same_incoming_magnitude")
        if safe_float(row.get("turn_margin_deg"), -math.inf) < 0:
            notes.append("warning_negative_turn_margin")
    else:
        raise ValueError(f"unsupported outgoing mode {mode!r}")
    return r_start, body_v + vinf_out, t_event, powered_cost, notes


def solve_dsm_to_target_position(
    srv: PrincipiaImpulseServerV2,
    cfg: dict[str, Any],
    handoff: dict[str, Any],
    mode: str,
    t_arr: float,
    dsm_fraction: float,
) -> dict[str, Any]:
    dep_id = str(handoff.get("departure_id", "dep"))
    row_idx = int(handoff.get("row_index", 0))
    r_start, v_start, t_start, powered_cost, notes = make_start_state(handoff, mode, cfg)

    min_dsm_after_start_s = float(cfg["min_dsm_after_start_s"])
    max_dsm_before_arrival_s = float(cfg["max_dsm_before_arrival_s"])
    t_dsm = t_start + dsm_fraction * (t_arr - t_start)
    t_dsm = max(t_dsm, t_start + min_dsm_after_start_s)
    t_dsm = min(t_dsm, t_arr - max_dsm_before_arrival_s)
    if t_dsm <= t_start or t_dsm >= t_arr:
        return {
            "status": "BAD_TIMING",
            "departure_id": dep_id,
            "row_index": row_idx,
            "mode": mode,
            "t_start_s": t_start,
            "t_dsm_s": t_dsm,
            "t_arr_s": t_arr,
            "dsm_fraction": dsm_fraction,
        }

    r_dsm, v_dsm, st, msg = propagate_state(
        srv,
        req_id=f"leg2_pre_dsm_{os.getpid()}_{dep_id}_{row_idx}_{mode}_{t_arr:.1f}_{dsm_fraction:.3f}",
        t0=t_start,
        t1=t_dsm,
        r0=r_start,
        v0=v_start,
        impulses=[],
    )
    if st != "ok":
        return {"status": "PRE_DSM_FAILED", "server_status": st, "server_message": msg, "departure_id": dep_id, "row_index": row_idx, "mode": mode}

    target_r, target_v = body_state_raw(cfg["arr_body"], t_arr, cfg["center"], cfg["frame"])

    fd_step = float(cfg["fd_step_m_s"])
    max_dsm = float(cfg["max_dsm_m_s"])
    dsm = np.zeros(3, dtype=float)
    clipped = False
    for it in range(int(cfg["iterations"])):
        base_r, base_v, st, msg = propagate_state(
            srv,
            req_id=f"leg2_base_{os.getpid()}_{dep_id}_{row_idx}_{mode}_{it}_{t_arr:.1f}_{dsm_fraction:.3f}",
            t0=t_dsm,
            t1=t_arr,
            r0=r_dsm,
            v0=v_dsm + dsm,
            impulses=[],
        )
        if st != "ok":
            return {"status": "BASE_FAILED", "server_status": st, "server_message": msg, "departure_id": dep_id, "row_index": row_idx, "mode": mode}
        err = target_r - base_r
        J = np.zeros((3, 3), dtype=float)
        for j in range(3):
            dd = np.zeros(3, dtype=float); dd[j] = fd_step
            rp, vp, stp, msgp = propagate_state(
                srv,
                req_id=f"leg2_fd_{os.getpid()}_{dep_id}_{row_idx}_{mode}_{it}_{j}_{t_arr:.1f}_{dsm_fraction:.3f}",
                t0=t_dsm,
                t1=t_arr,
                r0=r_dsm,
                v0=v_dsm + dsm + dd,
                impulses=[],
            )
            if stp != "ok":
                return {"status": "FD_FAILED", "server_status": stp, "server_message": msgp, "departure_id": dep_id, "row_index": row_idx, "mode": mode}
            J[:, j] = (rp - base_r) / fd_step
        try:
            delta, *_ = np.linalg.lstsq(J, err, rcond=None)
        except np.linalg.LinAlgError:
            return {"status": "SINGULAR_J", "departure_id": dep_id, "row_index": row_idx, "mode": mode}
        dsm = dsm + delta
        if norm(dsm) > max_dsm:
            dsm = dsm * (max_dsm / norm(dsm))
            clipped = True
            break
        if norm(delta) < float(cfg["delta_stop_m_s"]):
            break

    final_r, final_v, st, msg = propagate_state(
        srv,
        req_id=f"leg2_final_{os.getpid()}_{dep_id}_{row_idx}_{mode}_{t_arr:.1f}_{dsm_fraction:.3f}",
        t0=t_dsm,
        t1=t_arr,
        r0=r_dsm,
        v0=v_dsm + dsm,
        impulses=[],
    )
    if st != "ok":
        return {"status": "FINAL_FAILED", "server_status": st, "server_message": msg, "departure_id": dep_id, "row_index": row_idx, "mode": mode}

    pos_err_km = norm(final_r - target_r) / 1000.0
    vel_err_m_s = norm(final_v - target_v)
    dsm_norm = norm(dsm)
    leg1_dv0 = safe_float(handoff.get("leg1_dv0_norm_m_s"), 0.0)
    leg1_dsm = safe_float(handoff.get("leg1_dsm_norm_m_s"), 0.0)
    total_after_departure = leg1_dsm + powered_cost + dsm_norm
    total_with_departure = leg1_dv0 + total_after_departure

    score = (
        pos_err_km / float(cfg["pos_scale_km"])
        + vel_err_m_s / float(cfg["vel_scale_m_s"]) * float(cfg["vel_weight"])
        + dsm_norm / float(cfg["dsm_scale_m_s"]) * float(cfg["dsm_weight"])
        + powered_cost / float(cfg["powered_scale_m_s"]) * float(cfg["powered_weight"])
    )

    valid = True
    reasons: list[str] = []
    if pos_err_km > float(cfg["accept_pos_km"]):
        valid = False; reasons.append("pos_err_too_large")
    if dsm_norm > float(cfg["accept_dsm_m_s"]):
        valid = False; reasons.append("leg2_dsm_too_large")
    if powered_cost > float(cfg["accept_powered_m_s"]):
        valid = False; reasons.append("eve_powered_too_large")
    if clipped:
        valid = False; reasons.append("leg2_dsm_clipped")

    return {
        "status": "OK",
        "solution_valid": valid,
        "invalid_reasons": reasons,
        "score": score,
        "departure_id": dep_id,
        "handoff_row_index": row_idx,
        "handoff_status": handoff.get("status"),
        "outgoing_mode": mode,
        "notes": notes,
        "t_start_s": t_start,
        "t_dsm_s": t_dsm,
        "t_arr_s": t_arr,
        "dsm_fraction": dsm_fraction,
        "leg1_arrival_offset_days": safe_float(handoff.get("arrival_offset_from_leg2_start_days")),
        "leg2_arrival_offset_days": (t_arr - float(cfg["leg2_t_end_s"])) / 86400.0,
        "eve_powered_lower_bound_m_s": powered_cost,
        "leg1_dv0_norm_m_s": leg1_dv0,
        "leg1_dsm_norm_m_s": leg1_dsm,
        "leg2_dsm_raw_m_s": dsm.tolist(),
        "leg2_dsm_levela_m_s": raw_to_levela(dsm).tolist(),
        "leg2_dsm_norm_m_s": dsm_norm,
        "leg2_dsm_clipped": clipped,
        "total_post_departure_dv_m_s": total_after_departure,
        "total_with_departure_dv_m_s": total_with_departure,
        "final_pos_err_km": pos_err_km,
        "final_vel_err_m_s": vel_err_m_s,
        "arrival_rel_vinf_m_s": vel_err_m_s,
        "start_r_raw_m": r_start.tolist(),
        "start_v_raw_m_s": v_start.tolist(),
        "r_dsm_raw_m": r_dsm.tolist(),
        "v_dsm_before_raw_m_s": v_dsm.tolist(),
        "v_dsm_after_raw_m_s": (v_dsm + dsm).tolist(),
        "target_r_raw_m": target_r.tolist(),
        "target_v_raw_m_s": target_v.tolist(),
        "final_r_raw_m": final_r.tolist(),
        "final_v_raw_m_s": final_v.tolist(),
        "source_handoff": handoff,
    }


def worker(payload: tuple[list[dict[str, Any]], dict[str, Any]]) -> list[dict[str, Any]]:
    handoffs, cfg = payload
    spice.kclear(); spice.furnsh(str(cfg["tpc"])); spice.furnsh(str(cfg["bsp"]))
    out: list[dict[str, Any]] = []
    modes = cfg["outgoing_modes"]
    arrivals = cfg["arrival_times_s"]
    fractions = cfg["dsm_fractions"]
    with PrincipiaImpulseServerV2(cfg["server"], cfg["plugin_b64"]) as srv:
        if not srv.ping():
            raise RuntimeError("server PING failed in worker")
        for h in handoffs:
            for mode in modes:
                for t_arr in arrivals:
                    for frac in fractions:
                        try:
                            out.append(solve_dsm_to_target_position(srv, cfg, h, mode, float(t_arr), float(frac)))
                        except Exception as e:
                            out.append({
                                "status": "EXCEPTION",
                                "departure_id": h.get("departure_id"),
                                "handoff_row_index": h.get("row_index"),
                                "mode": mode,
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


def main() -> int:
    ap = argparse.ArgumentParser(description="Re-render leg2 Eve->Kerbin from rerendered Eve handoff rows using a seeded outgoing v-infinity and one DSM.")
    ap.add_argument("--eve-handoff-json", type=Path, required=True)
    ap.add_argument("--plugin-b64", required=True)
    ap.add_argument("--server", default="principia_impulsive_particle_server")
    ap.add_argument("--leg-optimizations", type=Path, required=True)
    ap.add_argument("--leg", type=int, default=2)
    ap.add_argument("--bsp", type=Path, required=True)
    ap.add_argument("--tpc", type=Path, required=True)
    ap.add_argument("--dep-body", default="EVE")
    ap.add_argument("--arr-body", default="KERBIN")
    ap.add_argument("--center", default="SUN")
    ap.add_argument("--frame", default="J2000")
    ap.add_argument("--max-handoffs", type=int, default=20)
    ap.add_argument("--statuses", default="PASS,POWERED", help="Comma list of handoff statuses to use.")
    ap.add_argument("--outgoing-modes", default="ref_powered,same_mag_ref_dir")
    ap.add_argument("--arrival-offset-days", default="-10,-7,-5,-3,-1,0,1,3,5,7,10")
    ap.add_argument("--dsm-fractions", default="0.05,0.10,0.20,0.35,0.50")
    ap.add_argument("--min-dsm-after-start-s", type=float, default=3600.0)
    ap.add_argument("--max-dsm-before-arrival-s", type=float, default=86400.0)
    ap.add_argument("--fd-step-m-s", type=float, default=1.0)
    ap.add_argument("--iterations", type=int, default=2)
    ap.add_argument("--delta-stop-m-s", type=float, default=0.05)
    ap.add_argument("--max-dsm-m-s", type=float, default=2500.0)
    ap.add_argument("--accept-pos-km", type=float, default=100000.0)
    ap.add_argument("--accept-dsm-m-s", type=float, default=1200.0)
    ap.add_argument("--accept-powered-m-s", type=float, default=1200.0)
    ap.add_argument("--pos-scale-km", type=float, default=100000.0)
    ap.add_argument("--vel-scale-m-s", type=float, default=1000.0)
    ap.add_argument("--dsm-scale-m-s", type=float, default=1000.0)
    ap.add_argument("--powered-scale-m-s", type=float, default=1000.0)
    ap.add_argument("--vel-weight", type=float, default=0.05)
    ap.add_argument("--dsm-weight", type=float, default=0.60)
    ap.add_argument("--powered-weight", type=float, default=0.40)
    ap.add_argument("--start-at-actual-arrival-position", action="store_true", help="Use actual leg1 final_r instead of the body centre as leg2 start position.")
    ap.add_argument("--workers", type=int, default=max(1, min(4, os.cpu_count() or 1)))
    ap.add_argument("--chunk-size", type=int, default=1)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()

    statuses = {s.strip() for s in args.statuses.replace(";", ",").split(",") if s.strip()}
    modes = [s.strip() for s in args.outgoing_modes.replace(";", ",").split(",") if s.strip()]
    handoffs = read_handoff_rows(args.eve_handoff_json, args.max_handoffs, statuses)
    leg = read_leg_row(args.leg_optimizations, args.leg)
    t_end = float(leg["t_end_s"])
    arrival_offsets_days = parse_float_list(args.arrival_offset_days)
    arrival_times_s = [t_end + d * 86400.0 for d in arrival_offsets_days]
    dsm_fractions = parse_float_list(args.dsm_fractions)

    cfg = {
        "plugin_b64": args.plugin_b64,
        "server": args.server,
        "bsp": str(args.bsp),
        "tpc": str(args.tpc),
        "dep_body": args.dep_body,
        "arr_body": args.arr_body,
        "center": args.center,
        "frame": args.frame,
        "outgoing_modes": modes,
        "arrival_times_s": arrival_times_s,
        "dsm_fractions": dsm_fractions,
        "min_dsm_after_start_s": args.min_dsm_after_start_s,
        "max_dsm_before_arrival_s": args.max_dsm_before_arrival_s,
        "fd_step_m_s": args.fd_step_m_s,
        "iterations": args.iterations,
        "delta_stop_m_s": args.delta_stop_m_s,
        "max_dsm_m_s": args.max_dsm_m_s,
        "accept_pos_km": args.accept_pos_km,
        "accept_dsm_m_s": args.accept_dsm_m_s,
        "accept_powered_m_s": args.accept_powered_m_s,
        "pos_scale_km": args.pos_scale_km,
        "vel_scale_m_s": args.vel_scale_m_s,
        "dsm_scale_m_s": args.dsm_scale_m_s,
        "powered_scale_m_s": args.powered_scale_m_s,
        "vel_weight": args.vel_weight,
        "dsm_weight": args.dsm_weight,
        "powered_weight": args.powered_weight,
        "leg2_t_end_s": t_end,
        "start_at_actual_arrival_position": bool(args.start_at_actual_arrival_position),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print("=== RERENDER LEG2 FROM EVE HANDOFF ===")
    print(f"handoffs       : {len(handoffs)} statuses={sorted(statuses)}")
    print(f"outgoing modes : {modes}")
    print(f"arrivals       : {len(arrival_times_s)} offsets_days={arrival_offsets_days}")
    print(f"dsm fractions  : {dsm_fractions}")
    print(f"tasks          : {len(handoffs) * len(modes) * len(arrival_times_s) * len(dsm_fractions)}")
    print(f"workers        : {args.workers}")
    print(f"output_dir     : {args.output_dir}")

    rows: list[dict[str, Any]] = []
    chunks = chunked(handoffs, max(1, args.chunk_size))
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
    best = ok_rows[0] if ok_rows else None
    best_valid = valid_rows[0] if valid_rows else None

    summary = {
        "schema": "leg2_rerender_from_eve_handoff.v1",
        "n_rows": len(rows),
        "n_ok": len(ok_rows),
        "n_valid": len(valid_rows),
        "best": best,
        "best_valid": best_valid,
        "config": {k: v for k, v in vars(args).items() if k not in {"plugin_b64"}},
    }
    (args.output_dir / "leg2_rerender_all.json").write_text(json.dumps(rows, indent=2, default=str) + "\n")
    (args.output_dir / "leg2_rerender_top.json").write_text(json.dumps(ok_rows[:100], indent=2, default=str) + "\n")
    (args.output_dir / "leg2_rerender_result.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    write_csv(args.output_dir / "leg2_rerender_summary.csv", ok_rows)

    if best:
        print("\n=== TOP LEG2 RERENDER RESULTS ===")
        print("rank valid mode            dep        arr2_off_d pos_km vel_m_s eve_pow leg1_dsm leg2_dsm total_post total_all reasons")
        for i, r in enumerate(ok_rows[:20], start=1):
            print(
                f"{i:3d} {str(r.get('solution_valid')):<5} "
                f"{str(r.get('outgoing_mode')):<15} {str(r.get('departure_id')):<10} "
                f"{safe_float(r.get('leg2_arrival_offset_days')):10.2f} "
                f"{safe_float(r.get('final_pos_err_km')):8.1f} "
                f"{safe_float(r.get('final_vel_err_m_s')):8.1f} "
                f"{safe_float(r.get('eve_powered_lower_bound_m_s')):7.1f} "
                f"{safe_float(r.get('leg1_dsm_norm_m_s')):8.1f} "
                f"{safe_float(r.get('leg2_dsm_norm_m_s')):8.1f} "
                f"{safe_float(r.get('total_post_departure_dv_m_s')):10.1f} "
                f"{safe_float(r.get('total_with_departure_dv_m_s')):9.1f} "
                f"{','.join(r.get('invalid_reasons', []))}"
            )
    print(f"[OK] wrote {args.output_dir / 'leg2_rerender_result.json'}")
    print(f"[OK] wrote {args.output_dir / 'leg2_rerender_summary.csv'}")
    return 0 if valid_rows else 2


if __name__ == "__main__":
    raise SystemExit(main())
