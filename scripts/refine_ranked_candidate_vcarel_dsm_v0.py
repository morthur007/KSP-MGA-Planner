#!/usr/bin/env python3
"""
refine_ranked_candidate_vcarel_dsm_v0.py

Deterministic DSM refiner for a ranked candidate validated with VCAREL.

Input:
  candidate_departure_executability_rank.json produced by
  rank_pykep_candidates_by_departure_executability_v0_1.py.

Method:
  Keep burn0 fixed.
  Try several DSM epochs as fractions of the first-leg TOF.
  At each DSM epoch, use finite-difference sensitivity of VCAREL closest-approach
  relative position to a DSM impulse in raw XYZ.
  Solve a damped least-squares correction, clip to --dsm-max-m-s, evaluate, and
  iterate.

This is not a global optimizer; it is a deterministic midcourse targeting step
designed to turn a plausible single-burn miss into a close flyby.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from principia_targeter_client import PrincipiaTargeterClient


DAY_S = 86400.0


def norm(v: Sequence[float]) -> float:
    return float(np.linalg.norm(np.asarray(v, dtype=float)))


def raw_to_levela(v: Sequence[float]) -> list[float]:
    x, y, z = map(float, v)
    return [-y, z, x]


def parse_float_list(s: str) -> list[float]:
    return [float(x.strip()) for x in str(s).replace(";", ",").split(",") if x.strip()]


def read_live_t(path: Path) -> float:
    r = json.loads(path.read_text())
    for k in ("ut_s", "t_s", "spice_t_s", "et_s"):
        if k in r:
            return float(r[k])
    raise KeyError(f"could not find time in {path}")


def select_candidate(rank_json: Path, top_index: int) -> dict[str, Any]:
    data = json.loads(rank_json.read_text())
    top = data.get("top", [])
    if top_index < 0 or top_index >= len(top):
        raise SystemExit(f"--top-index out of range: {top_index}; top has {len(top)} rows")
    c = dict(top[top_index])

    required = [
        "burn_dt_s",
        "burn_abs_s",
        "t_arr_s",
        "dv_raw_m_s",
        "burn_rel_r_raw_m",
        "burn_rel_v_raw_m_s",
    ]
    missing = [k for k in required if k not in c or c[k] is None]
    if missing:
        raise SystemExit(
            "selected candidate is missing fields needed by VCAREL: "
            + ", ".join(missing)
            + "\nRe-run rank_pykep_candidates_by_departure_executability_v0_1.py."
        )

    if "dv_levela_m_s" not in c or c["dv_levela_m_s"] is None:
        c["dv_levela_m_s"] = raw_to_levela(c["dv_raw_m_s"])

    return c


def parse_okcarel(line: str) -> dict[str, Any]:
    fields = line.rstrip("\n").split("\t")
    if not fields or fields[0] != "OKCAREL":
        raise RuntimeError(f"expected OKCAREL response, got: {line[:500]}")
    if len(fields) < 32:
        raise RuntimeError(f"OKCAREL response too short: {len(fields)} fields: {line[:500]}")

    out = {
        "id": fields[1],
        "dep_body": fields[2],
        "arr_body": fields[3],
        "state_dt_s": float(fields[4]),
        "state_t_game_s": float(fields[5]),
        "ca_dt_s": float(fields[6]),
        "ca_t_game_s": float(fields[7]),

        "ca_rel_r_raw_m": list(map(float, fields[8:11])),
        "ca_rel_v_raw_m_s": list(map(float, fields[11:14])),

        "ca_distance_m": float(fields[14]),
        "ca_speed_m_s": float(fields[15]),
        "ca_radial_v_m_s": float(fields[16]),

        "samples": int(float(fields[17])),
        "status": fields[18],

        "ca_abs_debug_r_raw_m": list(map(float, fields[19:22])),
        "ca_abs_debug_v_raw_m_s": list(map(float, fields[22:25])),

        "arr_abs_debug_r_raw_m": list(map(float, fields[25:28])),
        "arr_abs_debug_v_raw_m_s": list(map(float, fields[28:31])),

        "n_burns": int(float(fields[31])),
        "burns": [],
    }

    idx = 32
    for _ in range(out["n_burns"]):
        if idx + 10 > len(fields):
            raise RuntimeError(f"OKCAREL burn diagnostics truncated at field {idx}")
        out["burns"].append({
            "burn_dt_s": float(fields[idx + 0]),
            "burn_r_raw_m": list(map(float, fields[idx + 1:idx + 4])),
            "burn_v_before_raw_m_s": list(map(float, fields[idx + 4:idx + 7])),
            "burn_v_after_raw_m_s": list(map(float, fields[idx + 7:idx + 10])),
        })
        idx += 10

    return out


def vcarel(
    client: PrincipiaTargeterClient,
    rid: str,
    dep_body: str,
    arr_body: str,
    state_abs_s: float,
    scan_start_rel_s: float,
    scan_end_rel_s: float,
    samples: int,
    rel_r: Sequence[float],
    rel_v: Sequence[float],
    impulses: Sequence[tuple[float, float, float, float]],
    timeout_s: float,
) -> dict[str, Any]:
    fields: list[Any] = [
        "VCAREL",
        rid,
        dep_body,
        arr_body,
        float(state_abs_s),
        float(scan_start_rel_s),
        float(scan_end_rel_s),
        int(samples),
        float(rel_r[0]), float(rel_r[1]), float(rel_r[2]),
        float(rel_v[0]), float(rel_v[1]), float(rel_v[2]),
        int(len(impulses)),
    ]
    for dt, dvx, dvy, dvz in impulses:
        fields += [float(dt), float(dvx), float(dvy), float(dvz)]
    line = client.command_fields(fields, timeout_s=timeout_s)
    return parse_okcarel(line)


def eval_candidate(
    client: PrincipiaTargeterClient,
    rid: str,
    dep_body: str,
    arr_body: str,
    state_abs_s: float,
    scan_start_rel_s: float,
    scan_end_rel_s: float,
    samples: int,
    rel_r: Sequence[float],
    rel_v: Sequence[float],
    burn0_raw: Sequence[float],
    dsm_dt_s: float | None,
    dsm_raw: Sequence[float] | None,
    timeout_s: float,
) -> dict[str, Any]:
    impulses: list[tuple[float, float, float, float]] = [
        (0.0, float(burn0_raw[0]), float(burn0_raw[1]), float(burn0_raw[2])),
    ]
    if dsm_dt_s is not None and dsm_raw is not None and norm(dsm_raw) > 0:
        impulses.append((float(dsm_dt_s), float(dsm_raw[0]), float(dsm_raw[1]), float(dsm_raw[2])))

    res = vcarel(
        client=client,
        rid=rid,
        dep_body=dep_body,
        arr_body=arr_body,
        state_abs_s=state_abs_s,
        scan_start_rel_s=scan_start_rel_s,
        scan_end_rel_s=scan_end_rel_s,
        samples=samples,
        rel_r=rel_r,
        rel_v=rel_v,
        impulses=impulses,
        timeout_s=timeout_s,
    )

    out = {
        "ok": True,
        "error": "",
        "ca_distance_km": res["ca_distance_m"] / 1000.0,
        "ca_speed_m_s": res["ca_speed_m_s"],
        "ca_radial_v_m_s": res["ca_radial_v_m_s"],
        "ca_dt_s": res["ca_dt_s"],
        "ca_t_game_s": res["ca_t_game_s"],
        "ca_rel_r_raw_m": res["ca_rel_r_raw_m"],
        "ca_rel_v_raw_m_s": res["ca_rel_v_raw_m_s"],
        "status": res["status"],
        "samples": res["samples"],
        "n_burns": res["n_burns"],
        "burns": res["burns"],
        "dsm_dt_s": dsm_dt_s,
        "dsm_raw_m_s": list(map(float, dsm_raw)) if dsm_raw is not None else [0.0, 0.0, 0.0],
        "dsm_norm_m_s": norm(dsm_raw or [0.0, 0.0, 0.0]),
    }
    return out


def clip_norm(v: np.ndarray, max_norm: float) -> np.ndarray:
    n = norm(v)
    if n > max_norm and n > 0:
        return v * (max_norm / n)
    return v


def row_score(row: dict[str, Any], dsm_weight: float) -> float:
    return float(row["ca_distance_km"]) + dsm_weight * float(row.get("dsm_norm_m_s", 0.0))


def make_event(
    c: dict[str, Any],
    vessel_guid: str,
    out_dir: Path,
    best: dict[str, Any],
) -> None:
    burn_abs = float(c["burn_abs_s"])
    burn0_levela = c.get("dv_levela_m_s") or raw_to_levela(c["dv_raw_m_s"])

    event1 = {
        "enabled": True,
        "vessel_guid": vessel_guid,
        "ensure_flight_plan": True,
        "extend_existing_flight_plan": True,
        "mass_tonnes": 2.6,
        "insert_index": 0,
        "burn_template": "json",
        "thrust_kN": 2686.87701225281,
        "specific_impulse_s_g0": 1000.0,
        "is_inertially_fixed": True,
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
        "request_id": f"row{c.get('row_index0','x')}_burn0_attempt0",
        "dedupe_tag": f"row{c.get('row_index0','x')}_burn0",
        "event_key": f"row{c.get('row_index0','x')}_burn0",
        "attempt": 0,
        "mode": "insert_levela",
        "initial_time": burn_abs,
        "plan_final_time": burn_abs + 600.0,
        "delta_v_levela_m_s": [float(x) for x in burn0_levela],
        "require_warp_close": True,
        "max_lead_before_insert_s": 600.0,
        "reject_long_plan_before_first_burn": True,
        "max_first_plan_duration_s": 900.0,
        "rollback_on_status_error": True,
        "auto_sort_insert_index": True,
    }

    dsm_raw = best.get("dsm_raw_m_s", [0.0, 0.0, 0.0])
    dsm_levela = raw_to_levela(dsm_raw)
    dsm_abs = burn_abs + float(best["dsm_dt_s"])

    event2 = dict(event1)
    event2.update({
        "request_id": f"row{c.get('row_index0','x')}_dsm_attempt0",
        "dedupe_tag": f"row{c.get('row_index0','x')}_dsm",
        "event_key": f"row{c.get('row_index0','x')}_dsm",
        "initial_time": dsm_abs,
        "plan_final_time": dsm_abs + 600.0,
        "delta_v_levela_m_s": [float(x) for x in dsm_levela],
    })

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "event1_burn0_inertial_levela.json").write_text(json.dumps(event1, indent=2) + "\n")
    (out_dir / "event2_dsm_inertial_levela.json").write_text(json.dumps(event2, indent=2) + "\n")


def flatten_row(row: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in row.items():
        if isinstance(v, list):
            if all(not isinstance(x, (list, dict)) for x in v):
                for i, x in enumerate(v):
                    out[f"{k}_{i}"] = x
            else:
                out[k] = json.dumps(v)
        elif isinstance(v, dict):
            out[k] = json.dumps(v)
        else:
            out[k] = v
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rank-json", type=Path, required=True)
    ap.add_argument("--top-index", type=int, default=0)

    ap.add_argument("--server", required=True)
    ap.add_argument("--plugin-b64", type=Path, required=True)
    ap.add_argument("--plugin-arg-mode", choices=["option", "positional"], default="positional")
    ap.add_argument("--vessel-guid", required=True)
    ap.add_argument("--live-state-json", type=Path, required=True)
    ap.add_argument("--dep-body", default=None)
    ap.add_argument("--arr-body", default=None)

    ap.add_argument("--dsm-fractions", default="0.05,0.1,0.2,0.35,0.5,0.7")
    ap.add_argument("--dsm-max-m-s", type=float, default=500.0)
    ap.add_argument("--fd-step-m-s", type=float, default=1.0)
    ap.add_argument("--regularization", type=float, default=1e-6)
    ap.add_argument("--iterations", type=int, default=3)
    ap.add_argument("--scan-half-width-days", type=float, default=20.0)
    ap.add_argument("--arrival-offset-days", type=float, default=0.0)
    ap.add_argument("--vca-samples", type=int, default=101)
    ap.add_argument("--dsm-weight", type=float, default=0.001)
    ap.add_argument("--server-timeout-s", type=float, default=900.0)
    ap.add_argument("--quiet-stderr", action="store_true")
    ap.add_argument("--write-events", action="store_true")
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    c = select_candidate(args.rank_json, args.top_index)
    live_t = read_live_t(args.live_state_json)

    sequence = str(c.get("sequence", "")).split()
    dep_body = (args.dep_body or c.get("dep_body") or (sequence[0] if sequence else "KERBIN")).upper()
    arr_body = (args.arr_body or c.get("arr_body") or (sequence[1] if len(sequence) > 1 else "")).upper()
    if not arr_body:
        raise SystemExit("could not determine arr_body; pass --arr-body")

    state_abs_s = float(c["burn_abs_s"])
    t_arr_s = float(c["t_arr_s"])
    tof_rel_s = t_arr_s - state_abs_s
    if tof_rel_s <= 0:
        raise SystemExit(f"invalid first-leg TOF from burn to arrival: {tof_rel_s}")

    scan_center_rel_s = tof_rel_s + args.arrival_offset_days * DAY_S
    scan_start_rel_s = scan_center_rel_s - args.scan_half_width_days * DAY_S
    scan_end_rel_s = scan_center_rel_s + args.scan_half_width_days * DAY_S

    rel_r = [float(x) for x in c["burn_rel_r_raw_m"]]
    rel_v = [float(x) for x in c["burn_rel_v_raw_m_s"]]
    burn0_raw = [float(x) for x in c["dv_raw_m_s"]]
    fractions = parse_float_list(args.dsm_fractions)

    print("=== REFINE RANKED CANDIDATE VCAREL DSM V0 ===")
    print(f"row_index0      : {c.get('row_index0')}")
    print(f"sequence        : {c.get('sequence')}")
    print(f"dep -> arr      : {dep_body} -> {arr_body}")
    print(f"state_abs_s     : {state_abs_s}")
    print(f"t_arr_s         : {t_arr_s}")
    print(f"tof_from_burn   : {tof_rel_s:.3f}s = {tof_rel_s/DAY_S:.3f} d")
    print(f"scan_rel        : {scan_start_rel_s:.3f} .. {scan_end_rel_s:.3f}")
    print(f"burn0_norm      : {norm(burn0_raw):.3f} m/s")
    print(f"dsm fractions   : {fractions}")
    print(f"dsm max         : {args.dsm_max_m_s} m/s")
    print(f"output_dir      : {args.output_dir}")

    rows: list[dict[str, Any]] = []
    eval_counter = 0
    best: dict[str, Any] | None = None

    with PrincipiaTargeterClient(
        args.server,
        args.plugin_b64,
        plugin_arg_mode=args.plugin_arg_mode,
        response_timeout_s=args.server_timeout_s,
        quiet_stderr=args.quiet_stderr,
    ) as client:
        # Baseline no DSM.
        try:
            base0 = eval_candidate(
                client, f"dsmref_{os.getpid()}_{eval_counter}",
                dep_body, arr_body, state_abs_s, scan_start_rel_s, scan_end_rel_s,
                args.vca_samples, rel_r, rel_v, burn0_raw, None, None, args.server_timeout_s
            )
            eval_counter += 1
            base0.update({"kind": "baseline", "fraction": None, "iteration": 0, "score": row_score(base0, args.dsm_weight)})
            rows.append(base0)
            best = base0
            print(f"[baseline] ca={base0['ca_distance_km']:.3f} km speed={base0['ca_speed_m_s']:.3f}")
        except Exception as exc:
            print(f"[baseline] failed: {exc}")

        for frac in fractions:
            dsm_dt_s = max(1.0, min(float(frac) * tof_rel_s, tof_rel_s - DAY_S))
            if dsm_dt_s <= 0:
                continue

            dsm = np.zeros(3, dtype=float)
            current: dict[str, Any] | None = None

            print(f"\n[fraction {frac:.4f}] dsm_dt={dsm_dt_s:.3f}s ({dsm_dt_s/DAY_S:.3f} d)")

            for it in range(args.iterations):
                try:
                    cur = eval_candidate(
                        client, f"dsmref_{os.getpid()}_{eval_counter}",
                        dep_body, arr_body, state_abs_s, scan_start_rel_s, scan_end_rel_s,
                        args.vca_samples, rel_r, rel_v, burn0_raw, dsm_dt_s, dsm.tolist(), args.server_timeout_s
                    )
                    eval_counter += 1
                    cur.update({
                        "kind": "iterate_base",
                        "fraction": frac,
                        "iteration": it,
                        "score": row_score(cur, args.dsm_weight),
                    })
                    rows.append(cur)
                    current = cur
                except Exception as exc:
                    rows.append({
                        "ok": False, "error": str(exc), "kind": "iterate_base",
                        "fraction": frac, "iteration": it,
                        "dsm_dt_s": dsm_dt_s, "dsm_raw_m_s": dsm.tolist(),
                        "dsm_norm_m_s": norm(dsm),
                    })
                    print(f"  iter {it}: base failed: {exc}")
                    break

                r0 = np.asarray(cur["ca_rel_r_raw_m"], dtype=float)
                ca0_km = norm(r0) / 1000.0

                A = np.zeros((3, 3), dtype=float)
                for j in range(3):
                    dsm_p = dsm.copy()
                    dsm_p[j] += args.fd_step_m_s
                    try:
                        rp = eval_candidate(
                            client, f"dsmref_{os.getpid()}_{eval_counter}",
                            dep_body, arr_body, state_abs_s, scan_start_rel_s, scan_end_rel_s,
                            args.vca_samples, rel_r, rel_v, burn0_raw, dsm_dt_s, dsm_p.tolist(), args.server_timeout_s
                        )
                        eval_counter += 1
                        rp.update({
                            "kind": f"fd_axis_{j}",
                            "fraction": frac,
                            "iteration": it,
                            "score": row_score(rp, args.dsm_weight),
                        })
                        rows.append(rp)
                        A[:, j] = (np.asarray(rp["ca_rel_r_raw_m"], dtype=float) - r0) / args.fd_step_m_s
                    except Exception as exc:
                        rows.append({
                            "ok": False, "error": str(exc), "kind": f"fd_axis_{j}",
                            "fraction": frac, "iteration": it,
                            "dsm_dt_s": dsm_dt_s, "dsm_raw_m_s": dsm_p.tolist(),
                            "dsm_norm_m_s": norm(dsm_p),
                        })
                        A[:, j] = 0.0

                # Damped least squares: minimize |r0 + A delta|^2 + lambda |delta|^2
                ATA = A.T @ A
                rhs = -A.T @ r0
                lam = args.regularization * max(1.0, float(np.trace(ATA)) / 3.0)
                try:
                    delta = np.linalg.solve(ATA + lam * np.eye(3), rhs)
                except np.linalg.LinAlgError:
                    delta = np.linalg.lstsq(ATA + lam * np.eye(3), rhs, rcond=None)[0]

                # Do not allow a single linearized step to consume all budget too violently.
                dsm_trial = clip_norm(dsm + delta, args.dsm_max_m_s)
                try:
                    trial = eval_candidate(
                        client, f"dsmref_{os.getpid()}_{eval_counter}",
                        dep_body, arr_body, state_abs_s, scan_start_rel_s, scan_end_rel_s,
                        args.vca_samples, rel_r, rel_v, burn0_raw, dsm_dt_s, dsm_trial.tolist(), args.server_timeout_s
                    )
                    eval_counter += 1
                    trial.update({
                        "kind": "trial",
                        "fraction": frac,
                        "iteration": it,
                        "score": row_score(trial, args.dsm_weight),
                    })
                    rows.append(trial)

                    print(
                        f"  iter {it}: ca0={ca0_km:12.3f} km "
                        f"dsm={norm(dsm):8.3f} -> {norm(dsm_trial):8.3f} m/s "
                        f"trial_ca={trial['ca_distance_km']:12.3f} km"
                    )

                    # Accept if it improves score; otherwise damp the step a few times.
                    best_local = trial
                    best_local_dsm = dsm_trial
                    if row_score(trial, args.dsm_weight) >= row_score(cur, args.dsm_weight):
                        for alpha in (0.5, 0.25, 0.1):
                            dsm_half = clip_norm(dsm + alpha * delta, args.dsm_max_m_s)
                            try:
                                half = eval_candidate(
                                    client, f"dsmref_{os.getpid()}_{eval_counter}",
                                    dep_body, arr_body, state_abs_s, scan_start_rel_s, scan_end_rel_s,
                                    args.vca_samples, rel_r, rel_v, burn0_raw, dsm_dt_s, dsm_half.tolist(), args.server_timeout_s
                                )
                                eval_counter += 1
                                half.update({
                                    "kind": f"trial_alpha_{alpha}",
                                    "fraction": frac,
                                    "iteration": it,
                                    "score": row_score(half, args.dsm_weight),
                                })
                                rows.append(half)
                                if row_score(half, args.dsm_weight) < row_score(best_local, args.dsm_weight):
                                    best_local = half
                                    best_local_dsm = dsm_half
                            except Exception as exc:
                                rows.append({
                                    "ok": False, "error": str(exc), "kind": f"trial_alpha_{alpha}",
                                    "fraction": frac, "iteration": it,
                                    "dsm_dt_s": dsm_dt_s, "dsm_raw_m_s": dsm_half.tolist(),
                                    "dsm_norm_m_s": norm(dsm_half),
                                })

                    if row_score(best_local, args.dsm_weight) < row_score(cur, args.dsm_weight):
                        dsm = best_local_dsm
                        current = best_local
                    else:
                        current = cur
                        print("  no improvement; stopping this fraction")
                        break

                    if best is None or row_score(current, args.dsm_weight) < row_score(best, args.dsm_weight):
                        best = current

                except Exception as exc:
                    rows.append({
                        "ok": False, "error": str(exc), "kind": "trial",
                        "fraction": frac, "iteration": it,
                        "dsm_dt_s": dsm_dt_s, "dsm_raw_m_s": dsm_trial.tolist(),
                        "dsm_norm_m_s": norm(dsm_trial),
                    })
                    print(f"  iter {it}: trial failed: {exc}")
                    break

    ok_rows = [r for r in rows if r.get("ok") and math.isfinite(float(r.get("ca_distance_km", math.nan)))]
    ok_rows.sort(key=lambda r: row_score(r, args.dsm_weight))
    best = ok_rows[0] if ok_rows else best

    print("\n=== TOP DSM REFINEMENT RESULTS ===")
    for i, r in enumerate(ok_rows[:20], 1):
        print(
            f"{i:2d} ca={r['ca_distance_km']:12.3f} km "
            f"dsm={r.get('dsm_norm_m_s', 0.0):8.3f} m/s "
            f"dt={float(r['dsm_dt_s'])/DAY_S if r.get('dsm_dt_s') is not None else 0.0:8.3f} d "
            f"score={row_score(r, args.dsm_weight):12.3f} "
            f"kind={r.get('kind')} frac={r.get('fraction')} it={r.get('iteration')}"
        )

    out = {
        "schema": "ranked_candidate_vcarel_dsm_refine_v0",
        "rank_json": str(args.rank_json),
        "top_index": args.top_index,
        "candidate": c,
        "config": {
            "dep_body": dep_body,
            "arr_body": arr_body,
            "state_abs_s": state_abs_s,
            "t_arr_s": t_arr_s,
            "tof_rel_s": tof_rel_s,
            "scan_start_rel_s": scan_start_rel_s,
            "scan_end_rel_s": scan_end_rel_s,
            "dsm_fractions": fractions,
            "dsm_max_m_s": args.dsm_max_m_s,
            "fd_step_m_s": args.fd_step_m_s,
            "iterations": args.iterations,
            "vca_samples": args.vca_samples,
        },
        "n_rows": len(rows),
        "n_ok": len(ok_rows),
        "best": best,
        "top": ok_rows[:50],
        "rows": rows,
    }

    json_path = args.output_dir / "ranked_candidate_vcarel_dsm_refine.json"
    csv_path = args.output_dir / "ranked_candidate_vcarel_dsm_refine.csv"
    json_path.write_text(json.dumps(out, indent=2) + "\n")

    flat = [flatten_row(r) for r in rows]
    if flat:
        fields = sorted({k for r in flat for k in r.keys()})
        with csv_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(flat)

    print(f"[OK] wrote {json_path}")
    print(f"[OK] wrote {csv_path}")

    if args.write_events and best and best.get("ok"):
        make_event(c, args.vessel_guid, args.output_dir, best)
        print(f"[OK] wrote event1_burn0_inertial_levela.json")
        print(f"[OK] wrote event2_dsm_inertial_levela.json")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
