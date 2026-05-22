#!/usr/bin/env python3
"""
validate_rank12_full_chain.py

Read-only continuity/propagation validator for the KSP-MGA + Principia pipeline.

Goal
----
Find where a promoted MGA candidate stops being the same trajectory that the
N-body leg/bridge/departure tools validated.

This script does NOT optimize anything and does NOT touch a save/FlightPlan.  It
only launches principia_impulsive_particle_server instances, replays existing
impulses, and writes diagnostics.

Main checks
-----------
1. propn_replay:
   Replays optimize_propn_to_target.py/result.json from the captured live state.

2. leg_replay_i:
   Replays each leg_optimizations.csv row independently:
       row start state + row dv at t_start_s -> row target state at t_end_s

3. stitch_i_to_i+1:
   Pure CSV continuity check from one optimized leg target to next optimized leg
   start.  This often reveals whether an expected flyby/bridge reset is missing.

4. naive_chain:
   Sequential propagation without resetting to each row start.  This is the most
   useful "where does it explode?" check.  It can start from either:
       - row 1 start state + row 1 dv, or
       - live_state + propn_result impulses, then continue from leg 2.

5. ca_scan_i, optional:
   Samples arrival distance to the arrival body around t_end_s.  Useful when a
   fixed-epoch miss is huge but closest approach may happen earlier/later.

Notes
-----
- Every worker process owns its own PrincipiaImpulseServerV2 instance.  Do not
  share a server across processes.
- The script is intentionally schema-tolerant, but expects the canonical columns
  already used by your current pipeline when available:
    start_x_raw_m/start_y_raw_m/start_z_raw_m
    start_vx_raw_m_s/start_vy_raw_m_s/start_vz_raw_m_s
    target_x_raw_m/target_y_raw_m/target_z_raw_m
    target_vx_raw_m_s/target_vy_raw_m_s/target_vz_raw_m_s
    dvx_m_s/dvy_m_s/dvz_m_s
    t_start_s/t_end_s
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import csv
import json
import math
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Iterable

import numpy as np

try:
    import spiceypy as spice
except Exception:  # pragma: no cover - optional at runtime if no CA scans/body checks.
    spice = None  # type: ignore


RAW_R_COLS = [
    ("start_x_raw_m", "start_y_raw_m", "start_z_raw_m"),
    ("r0_x_raw_m", "r0_y_raw_m", "r0_z_raw_m"),
    ("initial_x_raw_m", "initial_y_raw_m", "initial_z_raw_m"),
    ("x0_raw_m", "y0_raw_m", "z0_raw_m"),
]
RAW_V_COLS = [
    ("start_vx_raw_m_s", "start_vy_raw_m_s", "start_vz_raw_m_s"),
    ("v0_x_raw_m_s", "v0_y_raw_m_s", "v0_z_raw_m_s"),
    ("initial_vx_raw_m_s", "initial_vy_raw_m_s", "initial_vz_raw_m_s"),
    ("vx0_raw_m_s", "vy0_raw_m_s", "vz0_raw_m_s"),
]
TARGET_R_COLS = [
    ("target_x_raw_m", "target_y_raw_m", "target_z_raw_m"),
    ("final_x_raw_m", "final_y_raw_m", "final_z_raw_m"),
    ("arr_x_raw_m", "arr_y_raw_m", "arr_z_raw_m"),
]
TARGET_V_COLS = [
    ("target_vx_raw_m_s", "target_vy_raw_m_s", "target_vz_raw_m_s"),
    ("final_vx_raw_m_s", "final_vy_raw_m_s", "final_vz_raw_m_s"),
    ("arr_vx_raw_m_s", "arr_vy_raw_m_s", "arr_vz_raw_m_s"),
]
DV_COLS = [
    ("dvx_m_s", "dvy_m_s", "dvz_m_s"),
    ("delta_vx_raw_m_s", "delta_vy_raw_m_s", "delta_vz_raw_m_s"),
    ("dv_raw_x_m_s", "dv_raw_y_m_s", "dv_raw_z_m_s"),
]

TIME_START_KEYS = ["t_start_s", "start_time_s", "t0_s", "dep_time_s", "t_dep_s"]
TIME_END_KEYS = ["t_end_s", "end_time_s", "t1_s", "arr_time_s", "t_arr_s"]


def norm(v: np.ndarray) -> float:
    return float(np.linalg.norm(v))


def safe_float(x: Any, default: float = math.inf) -> float:
    try:
        if x in ("", None):
            return default
        v = float(x)
        if math.isnan(v):
            return default
        return v
    except Exception:
        return default


def finite_or_none(x: Any) -> float | None:
    v = safe_float(x, math.inf)
    return v if math.isfinite(v) else None


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def json_load_optional(path: Path | None) -> Any | None:
    if path is None or not path.exists():
        return None
    return json.loads(path.read_text())


def first_existing(row: dict[str, Any], keys: Iterable[str]) -> tuple[str, Any] | tuple[None, None]:
    for k in keys:
        if k in row and row[k] not in ("", None):
            return k, row[k]
    return None, None


def vec_from_cols(row: dict[str, Any], candidates: list[tuple[str, str, str]], *, required: bool = True) -> tuple[np.ndarray | None, tuple[str, str, str] | None]:
    for cols in candidates:
        if all(c in row and row[c] not in ("", None) for c in cols):
            return np.array([float(row[c]) for c in cols], dtype=float), cols
    if required:
        raise KeyError(f"missing vector columns; tried {candidates}")
    return None, None


def time_from_row(row: dict[str, Any], keys: list[str]) -> tuple[float, str]:
    k, v = first_existing(row, keys)
    if k is None:
        raise KeyError(f"missing time column; tried {keys}")
    return float(v), str(k)


def leg_number(row: dict[str, Any], fallback: int) -> int:
    for k in ("leg", "leg_index", "i"):
        if k in row and row[k] not in ("", None):
            try:
                return int(float(row[k]))
            except Exception:
                pass
    return fallback


def raw_to_levela(v: np.ndarray) -> np.ndarray:
    x, y, z = v
    return np.array([-y, z, x], dtype=float)


def levela_to_raw(v: np.ndarray) -> np.ndarray:
    x, y, z = v
    return np.array([z, -x, y], dtype=float)


def body_state_raw(body: str, t_s: float, center: str, frame: str) -> tuple[np.ndarray, np.ndarray]:
    if spice is None:
        raise RuntimeError("spiceypy is not importable; cannot compute body_state_raw")
    st, _ = spice.spkezr(body, t_s, frame, "NONE", center)
    r_levela = np.array([1000.0 * st[0], 1000.0 * st[1], 1000.0 * st[2]], dtype=float)
    v_levela = np.array([1000.0 * st[3], 1000.0 * st[4], 1000.0 * st[5]], dtype=float)
    return levela_to_raw(r_levela), levela_to_raw(v_levela)


def infer_sequence(args: argparse.Namespace, rows: list[dict[str, str]]) -> list[str]:
    if args.sequence:
        return [x.strip().upper() for x in args.sequence.replace("-", " ").replace(",", " ").split() if x.strip()]

    # Try to infer from explicit leg dep/arr columns.
    bodies: list[str] = []
    for i, row in enumerate(rows):
        dep = (row.get("dep_body") or row.get("leg_dep") or row.get("departure_body") or "").strip().upper()
        arr = (row.get("arr_body") or row.get("leg_arr") or row.get("arrival_body") or row.get("target_body") or "").strip().upper()
        if i == 0 and dep:
            bodies.append(dep)
        if arr:
            bodies.append(arr)
    if bodies:
        return bodies
    return []


def arrival_body_for_leg(args: argparse.Namespace, rows: list[dict[str, str]], leg_idx0: int) -> str | None:
    row = rows[leg_idx0]
    for k in ("arr_body", "leg_arr", "arrival_body", "target_body"):
        v = row.get(k, "").strip()
        if v:
            return v.upper()
    seq = infer_sequence(args, rows)
    if len(seq) > leg_idx0 + 1:
        return seq[leg_idx0 + 1].upper()
    return None


def row_state(row: dict[str, Any], idx0: int) -> dict[str, Any]:
    r0, rcols = vec_from_cols(row, RAW_R_COLS)
    v0, vcols = vec_from_cols(row, RAW_V_COLS)
    rt, rtcols = vec_from_cols(row, TARGET_R_COLS)
    vt, vtcols = vec_from_cols(row, TARGET_V_COLS)
    dv, dvcols = vec_from_cols(row, DV_COLS, required=False)
    if dv is None:
        dv = np.zeros(3, dtype=float)
        dvcols = ("<zero>", "<zero>", "<zero>")
    t0, t0_key = time_from_row(row, TIME_START_KEYS)
    t1, t1_key = time_from_row(row, TIME_END_KEYS)
    return {
        "idx0": idx0,
        "leg": leg_number(row, idx0 + 1),
        "t0": t0,
        "t1": t1,
        "r0": r0,
        "v0": v0,
        "target_r": rt,
        "target_v": vt,
        "dv": dv,
        "cols": {
            "r0": rcols,
            "v0": vcols,
            "target_r": rtcols,
            "target_v": vtcols,
            "dv": dvcols,
            "t0": t0_key,
            "t1": t1_key,
        },
    }


def load_leg_states(leg_optimizations: Path) -> list[dict[str, Any]]:
    rows = read_csv_rows(leg_optimizations)
    out = [row_state(r, i) for i, r in enumerate(rows)]
    out.sort(key=lambda x: (x["t0"], x["t1"], x["leg"]))
    return out


def result_template(kind: str, name: str, status: str = "UNKNOWN") -> dict[str, Any]:
    return {
        "kind": kind,
        "name": name,
        "status": status,
        "pos_err_km": None,
        "vel_err_m_s": None,
        "message": "",
    }


def import_server_class():
    from ksp_mga.native.impulse_server_client_v0_2 import PrincipiaImpulseServerV2
    return PrincipiaImpulseServerV2


def propagate_once(server: str, plugin_b64: str, req_id: str, t0: float, t1: float, r0: np.ndarray, v0: np.ndarray, impulses: list[tuple[float, np.ndarray]]) -> dict[str, Any]:
    PrincipiaImpulseServerV2 = import_server_class()
    impulses = sorted(impulses, key=lambda x: x[0])
    with PrincipiaImpulseServerV2(server, plugin_b64) as srv:
        if not srv.ping():
            raise RuntimeError("Principia impulse server PING failed")
        res = srv.propagate_n(
            req_id=req_id,
            t0_s=float(t0),
            t1_s=float(t1),
            r0_m=np.asarray(r0, dtype=float),
            v0_m_s=np.asarray(v0, dtype=float),
            impulses=[(float(t), np.asarray(dv, dtype=float)) for t, dv in impulses],
        )
    return {
        "status": res.status,
        "message": res.message,
        "final_r_m": None if res.final_r_m is None else np.asarray(res.final_r_m, dtype=float).tolist(),
        "final_v_m_s": None if res.final_v_m_s is None else np.asarray(res.final_v_m_s, dtype=float).tolist(),
        "n_burns": len(getattr(res, "burns", []) or []),
    }


def worker_leg_replay(task: dict[str, Any]) -> dict[str, Any]:
    leg = task["leg_state"]
    out = result_template("leg_replay", f"leg{leg['leg']}_replay")
    out.update({"leg": leg["leg"], "t0_s": leg["t0"], "t1_s": leg["t1"], "dv_norm_m_s": norm(leg["dv"])})
    try:
        prop = propagate_once(
            task["server"],
            task["plugin_b64"],
            out["name"],
            leg["t0"],
            leg["t1"],
            leg["r0"],
            leg["v0"],
            [(leg["t0"], leg["dv"])],
        )
        out.update({"server_status": prop["status"], "server_message": prop["message"], "n_burns": prop["n_burns"]})
        if prop["status"] != "ok" or prop["final_r_m"] is None or prop["final_v_m_s"] is None:
            out["status"] = "SERVER_FAIL"
            return out
        rf = np.asarray(prop["final_r_m"], dtype=float)
        vf = np.asarray(prop["final_v_m_s"], dtype=float)
        out["pos_err_km"] = norm(rf - leg["target_r"]) / 1000.0
        out["vel_err_m_s"] = norm(vf - leg["target_v"])
        out["final_r_m"] = rf.tolist()
        out["final_v_m_s"] = vf.tolist()
        out["status"] = "PASS" if out["pos_err_km"] <= task["pass_pos_km"] and out["vel_err_m_s"] <= task["pass_vel_m_s"] else "MISS"
        return out
    except Exception as e:
        out["status"] = "EXCEPTION"
        out["message"] = repr(e)
        out["traceback"] = traceback.format_exc()
        return out


def worker_propn_replay(task: dict[str, Any]) -> dict[str, Any]:
    out = result_template("propn_replay", "propn_result_replay")
    try:
        live = json.loads(Path(task["live_state_json"]).read_text())
        propn = json.loads(Path(task["propn_result_json"]).read_text())
        leg = task["leg_state"]

        t0 = float(live.get("ut_s", propn.get("t0_s")))
        r0 = np.asarray(live["r_raw_m"], dtype=float)
        v0 = np.asarray(live["v_raw_m_s"], dtype=float)
        t1 = float(propn.get("t_final_s", leg["t1"]))
        times = [float(x) for x in propn["impulse_times_s"]]
        dvs = [np.asarray(x, dtype=float) for x in propn["dv_raw_m_s"]]
        impulses = list(zip(times, dvs))

        out.update({
            "t0_s": t0,
            "t1_s": t1,
            "impulse_times_s": times,
            "dv_norms_m_s": [norm(dv) for dv in dvs],
            "stored_final_pos_err_km": propn.get("final_pos_err_km"),
            "stored_final_vel_err_m_s": propn.get("final_vel_err_m_s"),
            "stored_physically_valid": propn.get("physically_valid"),
            "stored_invalid_reasons": propn.get("invalid_reasons"),
        })

        prop = propagate_once(
            task["server"], task["plugin_b64"], out["name"], t0, t1, r0, v0, impulses
        )
        out.update({"server_status": prop["status"], "server_message": prop["message"], "n_burns": prop["n_burns"]})
        if prop["status"] != "ok" or prop["final_r_m"] is None or prop["final_v_m_s"] is None:
            out["status"] = "SERVER_FAIL"
            return out

        rf = np.asarray(prop["final_r_m"], dtype=float)
        vf = np.asarray(prop["final_v_m_s"], dtype=float)
        out["pos_err_km"] = norm(rf - leg["target_r"]) / 1000.0
        out["vel_err_m_s"] = norm(vf - leg["target_v"])
        out["final_r_m"] = rf.tolist()
        out["final_v_m_s"] = vf.tolist()
        out["status"] = "PASS" if out["pos_err_km"] <= task["pass_pos_km"] and out["vel_err_m_s"] <= task["pass_vel_m_s"] else "MISS"
        return out
    except Exception as e:
        out["status"] = "EXCEPTION"
        out["message"] = repr(e)
        out["traceback"] = traceback.format_exc()
        return out


def worker_ca_scan(task: dict[str, Any]) -> dict[str, Any]:
    leg = task["leg_state"]
    body = task["arrival_body"]
    out = result_template("ca_scan", f"leg{leg['leg']}_ca_scan")
    out.update({"leg": leg["leg"], "arrival_body": body, "nominal_t_end_s": leg["t1"]})
    try:
        if spice is None:
            out["status"] = "SKIP"
            out["message"] = "spiceypy is not importable"
            return out
        spice.kclear()
        spice.furnsh(task["tpc"])
        spice.furnsh(task["bsp"])

        half = float(task["ca_window_days"]) * 86400.0
        n = int(task["ca_samples"])
        if n < 3:
            n = 3
        offsets = np.linspace(-half, half, n)

        PrincipiaImpulseServerV2 = import_server_class()
        samples: list[dict[str, Any]] = []
        best: dict[str, Any] | None = None
        with PrincipiaImpulseServerV2(task["server"], task["plugin_b64"]) as srv:
            if not srv.ping():
                raise RuntimeError("Principia impulse server PING failed")
            for j, off in enumerate(offsets):
                ts = float(leg["t1"] + off)
                if ts <= leg["t0"]:
                    continue
                res = srv.propagate_n(
                    req_id=f"{out['name']}_{j:03d}",
                    t0_s=float(leg["t0"]),
                    t1_s=ts,
                    r0_m=leg["r0"],
                    v0_m_s=leg["v0"],
                    impulses=[(float(leg["t0"]), leg["dv"])],
                )
                if res.status != "ok" or res.final_r_m is None or res.final_v_m_s is None:
                    samples.append({"offset_s": float(off), "t_s": ts, "status": res.status, "message": res.message})
                    continue
                rf = np.asarray(res.final_r_m, dtype=float)
                vf = np.asarray(res.final_v_m_s, dtype=float)
                rb, vb = body_state_raw(body, ts, task["center"], task["frame"])
                d_km = norm(rf - rb) / 1000.0
                relv = norm(vf - vb)
                sample = {"offset_s": float(off), "t_s": ts, "status": "ok", "distance_km": d_km, "relv_m_s": relv}
                samples.append(sample)
                if best is None or d_km < best["distance_km"]:
                    best = sample

        out["samples"] = samples
        if best is None:
            out["status"] = "SERVER_FAIL"
            out["message"] = "no successful samples"
            return out
        out["status"] = "OK"
        out["min_distance_km"] = best["distance_km"]
        out["min_relv_m_s"] = best["relv_m_s"]
        out["best_t_s"] = best["t_s"]
        out["best_offset_s"] = best["offset_s"]
        out["best_offset_days"] = best["offset_s"] / 86400.0
        return out
    except Exception as e:
        out["status"] = "EXCEPTION"
        out["message"] = repr(e)
        out["traceback"] = traceback.format_exc()
        return out


def build_stitch_checks(legs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    for a, b in zip(legs, legs[1:]):
        out = result_template("stitch_csv", f"leg{a['leg']}_to_leg{b['leg']}_csv_stitch")
        out.update({
            "from_leg": a["leg"],
            "to_leg": b["leg"],
            "from_t_end_s": a["t1"],
            "to_t_start_s": b["t0"],
            "time_gap_s": b["t0"] - a["t1"],
            "pos_err_km": norm(a["target_r"] - b["r0"]) / 1000.0,
            "vel_err_m_s": norm(a["target_v"] - b["v0"]),
            "status": "INFO",
            "message": "CSV target of previous leg versus CSV start of next leg; nonzero can be expected across modeled flyby/bridge boundaries.",
        })
        checks.append(out)
    return checks


def load_extra_impulse_events(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return []
    data = json.loads(path.read_text())
    if isinstance(data, dict) and "events" in data:
        data = data["events"]
    if not isinstance(data, list):
        raise ValueError("extra impulse JSON must be a list or an object with an 'events' list")
    events = []
    for i, ev in enumerate(data):
        if not isinstance(ev, dict):
            raise ValueError(f"extra impulse event {i} is not an object")
        t = ev.get("time_s", ev.get("t_s", ev.get("epoch_s", ev.get("initial_time"))))
        dv = ev.get("dv_raw_m_s", ev.get("delta_v_raw_m_s", ev.get("delta_v_m_s")))
        if t is None or dv is None:
            raise ValueError(f"extra impulse event {i} must contain time_s/t_s/epoch_s and dv_raw_m_s")
        events.append({
            "name": ev.get("name", f"extra_impulse_{i}"),
            "time_s": float(t),
            "dv_raw_m_s": [float(x) for x in dv],
        })
    return sorted(events, key=lambda e: e["time_s"])


def bridge_schema_report(path: Path | None) -> dict[str, Any] | None:
    data = json_load_optional(path)
    if data is None:
        return None

    vectors: list[str] = []
    scalars: list[str] = []

    def walk(prefix: str, obj: Any) -> None:
        if isinstance(obj, dict):
            for k, v in obj.items():
                walk(f"{prefix}.{k}" if prefix else k, v)
        elif isinstance(obj, list):
            if len(obj) == 3 and all(isinstance(x, (int, float)) for x in obj):
                vectors.append(prefix)
            elif len(obj) <= 20:
                for i, v in enumerate(obj):
                    walk(f"{prefix}[{i}]", v)
        elif isinstance(obj, (int, float)):
            low = prefix.lower()
            if any(token in low for token in ("time", "epoch", "_t_", "dv", "altitude", "err", "dt")):
                scalars.append(prefix)

    walk("", data)
    return {
        "path": str(path),
        "top_level_keys": sorted(data.keys()) if isinstance(data, dict) else None,
        "vector3_paths": vectors[:200],
        "interesting_scalar_paths": scalars[:300],
        "note": "This is a schema report only. Pass explicit --extra-impulses-json to include bridge impulses in naive_chain.",
    }


def worker_naive_chain(task: dict[str, Any]) -> dict[str, Any]:
    out = result_template("naive_chain", "naive_no_reset_chain")
    try:
        legs = task["legs"]
        extra_impulses = [
            (float(ev["time_s"]), np.asarray(ev["dv_raw_m_s"], dtype=float), ev.get("name", "extra"))
            for ev in task.get("extra_impulses", [])
        ]
        segments: list[dict[str, Any]] = []

        PrincipiaImpulseServerV2 = import_server_class()
        with PrincipiaImpulseServerV2(task["server"], task["plugin_b64"]) as srv:
            if not srv.ping():
                raise RuntimeError("Principia impulse server PING failed")

            start_mode = task.get("chain_start", "leg")
            if start_mode == "propn":
                if not task.get("live_state_json") or not task.get("propn_result_json"):
                    raise ValueError("chain_start=propn requires live_state_json and propn_result_json")
                live = json.loads(Path(task["live_state_json"]).read_text())
                propn = json.loads(Path(task["propn_result_json"]).read_text())
                current_t = float(live.get("ut_s", propn.get("t0_s")))
                current_r = np.asarray(live["r_raw_m"], dtype=float)
                current_v = np.asarray(live["v_raw_m_s"], dtype=float)
                first_leg_index = 1  # propn result covers leg 1 target.
                leg0 = legs[0]
                impulses = [(float(t), np.asarray(dv, dtype=float)) for t, dv in zip(propn["impulse_times_s"], propn["dv_raw_m_s"])]
                res = srv.propagate_n(
                    req_id="chain_propn_to_leg1_target",
                    t0_s=current_t,
                    t1_s=float(propn.get("t_final_s", leg0["t1"])),
                    r0_m=current_r,
                    v0_m_s=current_v,
                    impulses=impulses,
                )
                seg = {
                    "name": "propn_to_leg1_target",
                    "status": res.status,
                    "message": res.message,
                    "from_t_s": current_t,
                    "to_t_s": float(propn.get("t_final_s", leg0["t1"])),
                    "leg": leg0["leg"],
                    "target_kind": "leg1_target",
                    "n_impulses": len(impulses),
                }
                if res.status == "ok" and res.final_r_m is not None and res.final_v_m_s is not None:
                    current_t = seg["to_t_s"]
                    current_r = np.asarray(res.final_r_m, dtype=float)
                    current_v = np.asarray(res.final_v_m_s, dtype=float)
                    seg["pos_err_km"] = norm(current_r - leg0["target_r"]) / 1000.0
                    seg["vel_err_m_s"] = norm(current_v - leg0["target_v"])
                segments.append(seg)
            else:
                first_leg_index = 0
                current_t = float(legs[0]["t0"])
                current_r = np.asarray(legs[0]["r0"], dtype=float)
                current_v = np.asarray(legs[0]["v0"], dtype=float)

            for leg in legs[first_leg_index:]:
                # Coast from current epoch to the row start, if necessary.
                if current_t < leg["t0"] - task["time_tol_s"]:
                    interval_extras = [(t, dv) for t, dv, _name in extra_impulses if current_t <= t <= leg["t0"]]
                    res = srv.propagate_n(
                        req_id=f"chain_coast_to_leg{leg['leg']}_start",
                        t0_s=current_t,
                        t1_s=leg["t0"],
                        r0_m=current_r,
                        v0_m_s=current_v,
                        impulses=interval_extras,
                    )
                    seg = {
                        "name": f"coast_to_leg{leg['leg']}_start",
                        "status": res.status,
                        "message": res.message,
                        "from_t_s": current_t,
                        "to_t_s": leg["t0"],
                        "leg": leg["leg"],
                        "target_kind": "next_leg_start",
                        "n_impulses": len(interval_extras),
                    }
                    if res.status == "ok" and res.final_r_m is not None and res.final_v_m_s is not None:
                        current_t = leg["t0"]
                        current_r = np.asarray(res.final_r_m, dtype=float)
                        current_v = np.asarray(res.final_v_m_s, dtype=float)
                        seg["pos_err_km"] = norm(current_r - leg["r0"]) / 1000.0
                        seg["vel_err_m_s"] = norm(current_v - leg["v0"])
                    segments.append(seg)
                elif current_t > leg["t0"] + task["time_tol_s"]:
                    segments.append({
                        "name": f"leg{leg['leg']}_start_time_overlap",
                        "status": "TIME_OVERLAP",
                        "message": "current chain time is after this row start; cannot apply row start impulse consistently",
                        "current_t_s": current_t,
                        "row_t_start_s": leg["t0"],
                        "dt_s": current_t - leg["t0"],
                        "leg": leg["leg"],
                    })
                    continue

                interval_impulses: list[tuple[float, np.ndarray]] = []
                # Apply the row correction if the chain is at or before t_start.
                if current_t <= leg["t0"] + task["time_tol_s"]:
                    interval_impulses.append((leg["t0"], leg["dv"]))
                interval_impulses += [(t, dv) for t, dv, _name in extra_impulses if current_t <= t <= leg["t1"]]
                # Remove duplicate exact impulses only by object identity is not useful; keep all.
                interval_impulses = sorted(interval_impulses, key=lambda x: x[0])

                res = srv.propagate_n(
                    req_id=f"chain_leg{leg['leg']}_to_target",
                    t0_s=current_t,
                    t1_s=leg["t1"],
                    r0_m=current_r,
                    v0_m_s=current_v,
                    impulses=interval_impulses,
                )
                seg = {
                    "name": f"leg{leg['leg']}_to_target",
                    "status": res.status,
                    "message": res.message,
                    "from_t_s": current_t,
                    "to_t_s": leg["t1"],
                    "leg": leg["leg"],
                    "target_kind": "leg_target",
                    "n_impulses": len(interval_impulses),
                    "dv_norms_m_s": [norm(dv) for _, dv in interval_impulses],
                }
                if res.status == "ok" and res.final_r_m is not None and res.final_v_m_s is not None:
                    current_t = leg["t1"]
                    current_r = np.asarray(res.final_r_m, dtype=float)
                    current_v = np.asarray(res.final_v_m_s, dtype=float)
                    seg["pos_err_km"] = norm(current_r - leg["target_r"]) / 1000.0
                    seg["vel_err_m_s"] = norm(current_v - leg["target_v"])
                segments.append(seg)

        worst_pos = None
        worst_vel = None
        first_large = None
        for seg in segments:
            pk = finite_or_none(seg.get("pos_err_km"))
            vk = finite_or_none(seg.get("vel_err_m_s"))
            if pk is not None:
                worst_pos = pk if worst_pos is None else max(worst_pos, pk)
                if first_large is None and pk > task["large_pos_km"]:
                    first_large = seg["name"]
            if vk is not None:
                worst_vel = vk if worst_vel is None else max(worst_vel, vk)

        out.update({
            "status": "OK" if first_large is None else "LARGE_ERROR",
            "segments": segments,
            "worst_pos_err_km": worst_pos,
            "worst_vel_err_m_s": worst_vel,
            "first_large_error_segment": first_large,
            "chain_start": task.get("chain_start", "leg"),
        })
        return out
    except Exception as e:
        out["status"] = "EXCEPTION"
        out["message"] = repr(e)
        out["traceback"] = traceback.format_exc()
        return out


def run_tasks(tasks: list[dict[str, Any]], workers: int) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    if workers <= 1:
        for t in tasks:
            results.append(dispatch_task(t))
        return results

    with cf.ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(dispatch_task, t) for t in tasks]
        for fut in cf.as_completed(futs):
            results.append(fut.result())
    return results


def dispatch_task(task: dict[str, Any]) -> dict[str, Any]:
    kind = task["kind"]
    if kind == "leg_replay":
        return worker_leg_replay(task)
    if kind == "propn_replay":
        return worker_propn_replay(task)
    if kind == "ca_scan":
        return worker_ca_scan(task)
    if kind == "naive_chain":
        return worker_naive_chain(task)
    raise ValueError(f"unknown task kind {kind}")


def flatten_for_csv(check: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {}
    for k, v in check.items():
        if isinstance(v, (str, int, float, type(None), bool)):
            row[k] = v
        elif isinstance(v, list) and k in ("impulse_times_s", "dv_norms_m_s"):
            row[k] = ";".join(str(x) for x in v)
    return row


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    flat = [flatten_for_csv(r) for r in rows]
    for r in flat:
        for k in r:
            if k not in fields:
                fields.append(k)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(flat)


def summarize(checks: list[dict[str, Any]]) -> dict[str, Any]:
    worst_pos = -math.inf
    worst_pos_name = None
    worst_vel = -math.inf
    worst_vel_name = None
    counts: dict[str, int] = {}

    for c in checks:
        counts[c.get("status", "UNKNOWN")] = counts.get(c.get("status", "UNKNOWN"), 0) + 1
        p = finite_or_none(c.get("pos_err_km"))
        if p is None:
            p = finite_or_none(c.get("worst_pos_err_km"))
        if p is not None and p > worst_pos:
            worst_pos = p
            worst_pos_name = c.get("name")
        v = finite_or_none(c.get("vel_err_m_s"))
        if v is None:
            v = finite_or_none(c.get("worst_vel_err_m_s"))
        if v is not None and v > worst_vel:
            worst_vel = v
            worst_vel_name = c.get("name")

    return {
        "n_checks": len(checks),
        "status_counts": counts,
        "worst_pos_err_km": None if worst_pos == -math.inf else worst_pos,
        "worst_pos_err_check": worst_pos_name,
        "worst_vel_err_m_s": None if worst_vel == -math.inf else worst_vel,
        "worst_vel_err_check": worst_vel_name,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate promoted rank12/rankN artifacts by replaying N-body blocks and locating continuity breaks.")

    p.add_argument("--plugin-b64", type=Path, required=True)
    p.add_argument("--server", default="principia_impulsive_particle_server")
    p.add_argument("--leg-optimizations", type=Path, default=Path("data/runs/finalists/rank12_kekj/leg_optimizations.csv"))
    p.add_argument("--propn-result-json", type=Path, default=Path("data/runs/game_export/rank12_real/propn_best_phase_dv1_diagnostic/result.json"))
    p.add_argument("--live-state-json", type=Path, default=Path("data/runs/game_export/rank12_real/live_state_raw_near_tdep.json"))
    p.add_argument("--bridge-json", type=Path, default=Path("data/runs/finalists/rank12_kekj/kerbin_powered_bridge_trimmed.json"))
    p.add_argument("--extra-impulses-json", type=Path, default=None, help="Optional explicit raw impulse events to include in naive_chain. Format: list of {name,time_s,dv_raw_m_s}.")

    p.add_argument("--bsp", type=Path, default=None)
    p.add_argument("--tpc", type=Path, default=None)
    p.add_argument("--center", default="SUN")
    p.add_argument("--frame", default="J2000")
    p.add_argument("--sequence", default="KERBIN EVE KERBIN JOOL", help="Body sequence used for arrival-body CA scans.")

    p.add_argument("--output-dir", type=Path, default=Path("data/runs/finalists/rank12_kekj/full_chain_validation"))
    p.add_argument("--workers", type=int, default=max(1, min(4, os.cpu_count() or 1)))

    p.add_argument("--pass-pos-km", type=float, default=100.0, help="PASS threshold for leg/propn replay position error.")
    p.add_argument("--pass-vel-m-s", type=float, default=10.0, help="PASS threshold for leg/propn replay velocity error.")
    p.add_argument("--large-pos-km", type=float, default=1.0e6, help="Naive-chain segment is flagged after this position error.")
    p.add_argument("--time-tol-s", type=float, default=1.0e-6)

    p.add_argument("--skip-propn", action="store_true")
    p.add_argument("--skip-leg-replays", action="store_true")
    p.add_argument("--skip-naive-chain", action="store_true")
    p.add_argument("--chain-start", choices=["leg", "propn"], default="propn")
    p.add_argument("--closest-approach-scan", action="store_true")
    p.add_argument("--ca-window-days", type=float, default=10.0)
    p.add_argument("--ca-samples", type=int, default=41)

    return p.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not args.leg_optimizations.exists():
        raise SystemExit(f"[FAIL] leg optimizations not found: {args.leg_optimizations}")
    if not args.plugin_b64.exists():
        raise SystemExit(f"[FAIL] plugin b64 not found: {args.plugin_b64}")

    legs = load_leg_states(args.leg_optimizations)
    extra_impulses = load_extra_impulse_events(args.extra_impulses_json)

    # Make numpy arrays serializable for process pool payloads.
    leg_payloads: list[dict[str, Any]] = []
    for leg in legs:
        lp = dict(leg)
        for k in ("r0", "v0", "target_r", "target_v", "dv"):
            lp[k] = np.asarray(lp[k], dtype=float)
        leg_payloads.append(lp)

    common = {
        "server": str(args.server),
        "plugin_b64": str(args.plugin_b64),
        "pass_pos_km": args.pass_pos_km,
        "pass_vel_m_s": args.pass_vel_m_s,
    }

    checks: list[dict[str, Any]] = []
    checks.extend(build_stitch_checks(legs))

    tasks: list[dict[str, Any]] = []

    if not args.skip_propn:
        if args.live_state_json.exists() and args.propn_result_json.exists():
            tasks.append({
                "kind": "propn_replay",
                **common,
                "live_state_json": str(args.live_state_json),
                "propn_result_json": str(args.propn_result_json),
                "leg_state": leg_payloads[0],
            })
        else:
            checks.append({
                **result_template("propn_replay", "propn_result_replay", "SKIP"),
                "message": f"missing live or propn result: {args.live_state_json} / {args.propn_result_json}",
            })

    if not args.skip_leg_replays:
        for leg in leg_payloads:
            tasks.append({"kind": "leg_replay", **common, "leg_state": leg})

    if not args.skip_naive_chain:
        if args.chain_start == "propn" and (not args.live_state_json.exists() or not args.propn_result_json.exists()):
            checks.append({
                **result_template("naive_chain", "naive_no_reset_chain", "SKIP"),
                "message": "chain_start=propn requested but live/propn JSON is missing",
            })
        else:
            tasks.append({
                "kind": "naive_chain",
                **common,
                "legs": leg_payloads,
                "extra_impulses": extra_impulses,
                "chain_start": args.chain_start,
                "live_state_json": str(args.live_state_json),
                "propn_result_json": str(args.propn_result_json),
                "time_tol_s": args.time_tol_s,
                "large_pos_km": args.large_pos_km,
            })

    if args.closest_approach_scan:
        if args.bsp is None or args.tpc is None:
            checks.append({
                **result_template("ca_scan", "all_ca_scans", "SKIP"),
                "message": "--closest-approach-scan requires --bsp and --tpc",
            })
        else:
            for i, leg in enumerate(leg_payloads):
                body = arrival_body_for_leg(args, read_csv_rows(args.leg_optimizations), i)
                if not body:
                    checks.append({
                        **result_template("ca_scan", f"leg{leg['leg']}_ca_scan", "SKIP"),
                        "message": "could not infer arrival body",
                    })
                    continue
                tasks.append({
                    "kind": "ca_scan",
                    **common,
                    "leg_state": leg,
                    "arrival_body": body,
                    "bsp": str(args.bsp),
                    "tpc": str(args.tpc),
                    "center": args.center,
                    "frame": args.frame,
                    "ca_window_days": args.ca_window_days,
                    "ca_samples": args.ca_samples,
                })

    print("=== VALIDATE FULL CHAIN / BLOCKS ===")
    print(f"leg_optimizations : {args.leg_optimizations}")
    print(f"plugin_b64        : {args.plugin_b64}")
    print(f"workers           : {args.workers}")
    print(f"tasks             : {len(tasks)}")
    print(f"output_dir        : {args.output_dir}")
    print("")

    task_results = run_tasks(tasks, args.workers)
    checks.extend(task_results)

    # Sort for readable output.
    kind_order = {"propn_replay": 0, "leg_replay": 1, "stitch_csv": 2, "naive_chain": 3, "ca_scan": 4}
    checks.sort(key=lambda c: (kind_order.get(c.get("kind", ""), 99), c.get("leg", 999), c.get("name", "")))

    report = {
        "config": {
            "plugin_b64": str(args.plugin_b64),
            "server": str(args.server),
            "leg_optimizations": str(args.leg_optimizations),
            "propn_result_json": str(args.propn_result_json),
            "live_state_json": str(args.live_state_json),
            "bridge_json": str(args.bridge_json),
            "extra_impulses_json": None if args.extra_impulses_json is None else str(args.extra_impulses_json),
            "sequence": infer_sequence(args, read_csv_rows(args.leg_optimizations)),
            "workers": args.workers,
            "thresholds": {
                "pass_pos_km": args.pass_pos_km,
                "pass_vel_m_s": args.pass_vel_m_s,
                "large_pos_km": args.large_pos_km,
            },
        },
        "bridge_schema_report": bridge_schema_report(args.bridge_json),
        "extra_impulses": extra_impulses,
        "summary": summarize(checks),
        "checks": checks,
    }

    out_json = args.output_dir / "full_chain_validation.json"
    out_csv = args.output_dir / "full_chain_validation_summary.csv"
    out_json.write_text(json.dumps(report, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)) + "\n")
    write_csv(out_csv, checks)

    print("kind              status           name                               pos_err_km       vel_err_m_s")
    print("-" * 104)
    for c in checks:
        print(
            f"{c.get('kind',''):<17} "
            f"{c.get('status',''):<16} "
            f"{c.get('name',''):<34} "
            f"{str(c.get('pos_err_km', c.get('worst_pos_err_km', ''))):>14} "
            f"{str(c.get('vel_err_m_s', c.get('worst_vel_err_m_s', ''))):>16}"
        )
        if c.get("kind") == "naive_chain" and c.get("segments"):
            for s in c["segments"]:
                print(
                    f"  segment          {s.get('status',''):<16} "
                    f"{s.get('name',''):<34} "
                    f"{str(s.get('pos_err_km','')):>14} "
                    f"{str(s.get('vel_err_m_s','')):>16}"
                )

    print("")
    print(json.dumps(report["summary"], indent=2))
    print(f"[OK] wrote {out_json}")
    print(f"[OK] wrote {out_csv}")

    # Return nonzero only if something operational failed to run, not just if the route has a large miss.
    fatal_statuses = {"EXCEPTION", "SERVER_FAIL"}
    return 2 if any(c.get("status") in fatal_statuses for c in checks) else 0


if __name__ == "__main__":
    raise SystemExit(main())
