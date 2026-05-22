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


def levela_to_raw(v: np.ndarray) -> np.ndarray:
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


def body_mu_m3_s2(body: str) -> float:
    _, vals = spice.bodvrd(body, "GM", 1)
    return float(vals[0]) * 1.0e9


def specific_energy(r_rel: np.ndarray, v_rel: np.ndarray, mu: float) -> float:
    r = norm(r_rel)
    if r <= 0:
        return float("nan")
    return 0.5 * norm(v_rel) ** 2 - mu / r


def radial_velocity(r_rel: np.ndarray, v_rel: np.ndarray) -> float:
    r = norm(r_rel)
    if r <= 0:
        return float("nan")
    return float(np.dot(r_rel, v_rel) / r)


def safe_float(x: Any, default: float = float("nan")) -> float:
    try:
        if x is None or x == "":
            return default
        return float(x)
    except Exception:
        return default


def load_departures(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    data = json.loads(path.read_text())
    if isinstance(data, dict) and isinstance(data.get("departures"), list):
        return data, list(data["departures"])
    if isinstance(data, list):
        return {"schema": "list"}, list(data)
    raise SystemExit(f"[FAIL] unsupported reachable departures JSON: {path}")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    flat_rows: list[dict[str, Any]] = []
    for r in rows:
        rr: dict[str, Any] = {}
        for k, v in r.items():
            if isinstance(v, (list, dict)):
                rr[k] = json.dumps(v, separators=(",", ":"))
            else:
                rr[k] = v
            if k not in fields:
                fields.append(k)
        flat_rows.append(rr)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(flat_rows)


_WORKER_ARGS: dict[str, Any] = {}


def _init_worker(args_dict: dict[str, Any]) -> None:
    global _WORKER_ARGS
    _WORKER_ARGS = args_dict
    spice.kclear()
    spice.furnsh(str(args_dict["tpc"]))
    spice.furnsh(str(args_dict["bsp"]))


def evaluate_departure(dep: dict[str, Any]) -> dict[str, Any]:
    a = _WORKER_ARGS
    live = a["live"]
    t0 = float(dep.get("t0_s", live["ut_s"]))
    r0 = np.array(live["r_raw_m"], dtype=float)
    v0 = np.array(live["v_raw_m_s"], dtype=float)
    tb0 = safe_float(dep.get("tb0_s"))
    dv0 = np.array(dep.get("dv0_raw_m_s", [float("nan")] * 3), dtype=float)

    out = dict(dep)
    gate: dict[str, Any] = {
        "gate_schema": "transfer_gate.v1",
        "exit_radius_km": a["exit_radius_km"],
        "max_exit_time_s": a["max_exit_time_s"],
        "sample_step_s": a["sample_step_s"],
    }

    if not math.isfinite(tb0) or not np.all(np.isfinite(dv0)):
        gate.update({"transfer_gate_status": "FAIL_BAD_INPUT"})
        out["transfer_gate"] = gate
        out["transfer_gate_status"] = gate["transfer_gate_status"]
        return out

    sample_times = []
    t = tb0 + a["sample_step_s"]
    while t <= tb0 + a["max_exit_time_s"] + 1e-9:
        sample_times.append(float(t))
        t += a["sample_step_s"]
    if not sample_times:
        sample_times = [tb0 + a["max_exit_time_s"]]

    max_dist_km = -1.0
    max_radial_m_s = float("nan")
    max_eps = float("nan")
    first_exit: dict[str, Any] | None = None
    burn_metrics: dict[str, Any] = {}

    try:
        with PrincipiaImpulseServerV2(a["server"], a["plugin_b64"]) as srv:
            if not srv.ping():
                gate.update({"transfer_gate_status": "FAIL_SERVER_PING"})
                out["transfer_gate"] = gate
                out["transfer_gate_status"] = gate["transfer_gate_status"]
                return out

            # Short propagation to get the actual burn state/v_after from the server.
            burn_probe_t1 = max(tb0 + 1.0, tb0 + min(a["sample_step_s"], 60.0))
            burn_probe = srv.propagate_n(
                req_id=f"gate_burn_{dep.get('departure_id','dep')}_{os.getpid()}",
                t0_s=t0,
                t1_s=burn_probe_t1,
                r0_m=r0,
                v0_m_s=v0,
                impulses=[(tb0, dv0)],
            )
            if burn_probe.status != "ok" or not burn_probe.burns:
                gate.update({
                    "transfer_gate_status": "FAIL_BURN_PROBE",
                    "server_status": burn_probe.status,
                    "server_message": burn_probe.message,
                })
                out["transfer_gate"] = gate
                out["transfer_gate_status"] = gate["transfer_gate_status"]
                return out

            b = burn_probe.burns[0]
            burn_r = np.array(b.r_m, dtype=float)
            burn_v_after = np.array(b.v_after_m_s, dtype=float)
            body_r, body_v = body_state_raw(a["dep_body"], tb0, a["center"], a["frame"])
            burn_rel_r = burn_r - body_r
            burn_rel_v = burn_v_after - body_v
            burn_eps = specific_energy(burn_rel_r, burn_rel_v, a["mu_dep"])
            burn_radial = radial_velocity(burn_rel_r, burn_rel_v)
            burn_dist = norm(burn_rel_r) / 1000.0
            burn_metrics = {
                "burn_distance_from_body_km": burn_dist,
                "burn_radial_velocity_m_s": burn_radial,
                "burn_escape_energy_m2_s2": burn_eps,
                "burn_escape": bool(burn_eps > a["min_escape_energy_m2_s2"]),
            }

            for ts in sample_times:
                res = srv.propagate_n(
                    req_id=f"gate_{dep.get('departure_id','dep')}_{int(ts)}_{os.getpid()}",
                    t0_s=t0,
                    t1_s=ts,
                    r0_m=r0,
                    v0_m_s=v0,
                    impulses=[(tb0, dv0)],
                )
                if res.status != "ok" or res.final_r_m is None or res.final_v_m_s is None:
                    continue
                r = np.array(res.final_r_m, dtype=float)
                v = np.array(res.final_v_m_s, dtype=float)
                br, bv = body_state_raw(a["dep_body"], ts, a["center"], a["frame"])
                rel_r = r - br
                rel_v = v - bv
                d_km = norm(rel_r) / 1000.0
                rv = radial_velocity(rel_r, rel_v)
                eps = specific_energy(rel_r, rel_v, a["mu_dep"])
                if d_km > max_dist_km:
                    max_dist_km = d_km
                    max_radial_m_s = rv
                    max_eps = eps
                if first_exit is None and d_km >= a["exit_radius_km"]:
                    if (not a["require_positive_radial_at_exit"]) or rv >= a["min_exit_radial_m_s"]:
                        first_exit = {
                            "exit_t_s": ts,
                            "exit_dt_after_burn_s": ts - tb0,
                            "exit_distance_km": d_km,
                            "exit_radial_velocity_m_s": rv,
                            "exit_escape_energy_m2_s2": eps,
                            "exit_r_raw_m": r.tolist(),
                            "exit_v_raw_m_s": v.tolist(),
                            "exit_r_levela_m": raw_to_levela(r).tolist(),
                            "exit_v_levela_m_s": raw_to_levela(v).tolist(),
                        }
                        break

    except Exception as e:
        gate.update({"transfer_gate_status": "FAIL_EXCEPTION", "exception": repr(e)})
        out["transfer_gate"] = gate
        out["transfer_gate_status"] = gate["transfer_gate_status"]
        return out

    reasons = []
    if not burn_metrics.get("burn_escape", False):
        reasons.append("burn0_not_escape")
    if first_exit is None:
        reasons.append("no_exit_radius_crossing")
    status = "PASS" if not reasons else "FAIL"
    gate.update({
        "transfer_gate_status": status,
        "transfer_gate_reasons": reasons,
        **burn_metrics,
        "max_distance_from_body_km": max_dist_km,
        "max_radial_velocity_m_s": max_radial_m_s,
        "max_escape_energy_m2_s2": max_eps,
        "first_exit": first_exit,
    })
    out["transfer_gate"] = gate
    out["transfer_gate_status"] = status
    out["transfer_gate_reasons"] = reasons
    out["exit_t_s"] = None if first_exit is None else first_exit["exit_t_s"]
    out["exit_dt_after_burn_s"] = None if first_exit is None else first_exit["exit_dt_after_burn_s"]
    out["exit_distance_km"] = None if first_exit is None else first_exit["exit_distance_km"]
    out["burn0_escape_energy_m2_s2"] = burn_metrics.get("burn_escape_energy_m2_s2")
    out["burn0_distance_from_body_km"] = burn_metrics.get("burn_distance_from_body_km")
    out["burn0_radial_velocity_m_s"] = burn_metrics.get("burn_radial_velocity_m_s")
    out["max_distance_from_body_km"] = max_dist_km
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Filter reachable departures by a hard transfer-burn gate: burn0 must escape and cross an exit sphere before any DSM is allowed.")
    ap.add_argument("--reachable-departures", type=Path, required=True)
    ap.add_argument("--plugin-b64", required=True)
    ap.add_argument("--server", default="principia_impulsive_particle_server")
    ap.add_argument("--live-state-json", type=Path, required=True)
    ap.add_argument("--bsp", type=Path, required=True)
    ap.add_argument("--tpc", type=Path, required=True)
    ap.add_argument("--dep-body", default="KERBIN")
    ap.add_argument("--center", default="SUN")
    ap.add_argument("--frame", default="J2000")
    ap.add_argument("--exit-radius-km", type=float, default=40000.0)
    ap.add_argument("--max-exit-time-s", type=float, default=86400.0)
    ap.add_argument("--sample-step-s", type=float, default=1800.0)
    ap.add_argument("--min-escape-energy-m2-s2", type=float, default=0.0)
    ap.add_argument("--require-positive-radial-at-exit", action="store_true", default=True)
    ap.add_argument("--min-exit-radial-m-s", type=float, default=0.0)
    ap.add_argument("--max-departures", type=int, default=0, help="0 means all")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()

    spice.kclear(); spice.furnsh(str(args.tpc)); spice.furnsh(str(args.bsp))
    meta, departures = load_departures(args.reachable_departures)
    if args.max_departures and args.max_departures > 0:
        departures = departures[: args.max_departures]
    live = json.loads(args.live_state_json.read_text())
    mu_dep = body_mu_m3_s2(args.dep_body)

    worker_args = {
        "plugin_b64": args.plugin_b64,
        "server": args.server,
        "live": live,
        "bsp": str(args.bsp),
        "tpc": str(args.tpc),
        "dep_body": args.dep_body,
        "center": args.center,
        "frame": args.frame,
        "exit_radius_km": args.exit_radius_km,
        "max_exit_time_s": args.max_exit_time_s,
        "sample_step_s": args.sample_step_s,
        "min_escape_energy_m2_s2": args.min_escape_energy_m2_s2,
        "require_positive_radial_at_exit": args.require_positive_radial_at_exit,
        "min_exit_radial_m_s": args.min_exit_radial_m_s,
        "mu_dep": mu_dep,
    }

    print("=== FILTER DEPARTURES BY TRANSFER GATE ===")
    print(f"input          : {args.reachable_departures}")
    print(f"departures     : {len(departures)}")
    print(f"dep_body       : {args.dep_body}")
    print(f"exit_radius_km : {args.exit_radius_km}")
    print(f"max_exit_time_s: {args.max_exit_time_s}")
    print(f"sample_step_s  : {args.sample_step_s}")
    print(f"workers        : {args.workers}")
    print(f"output_dir     : {args.output_dir}")

    if args.workers <= 1:
        _init_worker(worker_args)
        results = [evaluate_departure(d) for d in departures]
    else:
        with mp.Pool(processes=args.workers, initializer=_init_worker, initargs=(worker_args,)) as pool:
            results = list(pool.imap_unordered(evaluate_departure, departures, chunksize=1))

    results.sort(key=lambda d: (
        0 if d.get("transfer_gate_status") == "PASS" else 1,
        safe_float(d.get("reference_vinf_vec_err_m_s"), math.inf),
        safe_float(d.get("dv0_norm_m_s"), math.inf),
    ))
    passed = [d for d in results if d.get("transfer_gate_status") == "PASS"]
    failed = [d for d in results if d.get("transfer_gate_status") != "PASS"]

    print("\n=== TOP TRANSFER-GATED DEPARTURES ===")
    print("rank status dep        tb0          dv0    eps_burn      exit_dt_s exit_km max_km reasons")
    for i, d in enumerate(results[:30], start=1):
        gate = d.get("transfer_gate", {})
        first_exit = gate.get("first_exit") or {}
        print(
            f"{i:4d} {d.get('transfer_gate_status','?'):<6} {d.get('departure_id','?'):<10} "
            f"{safe_float(d.get('tb0_s')):12.3f} "
            f"{safe_float(d.get('dv0_norm_m_s')):7.1f} "
            f"{safe_float(gate.get('burn_escape_energy_m2_s2')):12.3e} "
            f"{safe_float(first_exit.get('exit_dt_after_burn_s')):9.1f} "
            f"{safe_float(first_exit.get('exit_distance_km')):7.1f} "
            f"{safe_float(gate.get('max_distance_from_body_km')):7.1f} "
            f"{','.join(gate.get('transfer_gate_reasons', []))}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = {
        "schema": "reachable_departures.transfer_gated.v1",
        "source_reachable_departures": str(args.reachable_departures),
        "source_meta": meta,
        "dep_body": args.dep_body,
        "exit_radius_km": args.exit_radius_km,
        "max_exit_time_s": args.max_exit_time_s,
        "sample_step_s": args.sample_step_s,
        "n_input": len(departures),
        "n_pass": len(passed),
        "n_fail": len(failed),
        "departures": passed,
        "failed_departures": failed,
    }
    (args.output_dir / "reachable_departures_transfer_gated.json").write_text(json.dumps(out, indent=2) + "\n")
    write_csv(args.output_dir / "reachable_departures_transfer_gated.csv", passed)
    write_csv(args.output_dir / "reachable_departures_transfer_failed.csv", failed)
    print(json.dumps({"n_input": len(departures), "n_pass": len(passed), "n_fail": len(failed)}, indent=2))
    print(f"[OK] wrote {args.output_dir / 'reachable_departures_transfer_gated.json'}")
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
