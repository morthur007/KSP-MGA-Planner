#!/usr/bin/env python3
"""
fast_principia_vcarelnav_refine_v0.py

Fast local refiner for a PyKEP/patched-conics ranked candidate using the
Principia impulsive particle server as the truth model.

Design goal
-----------
Use the PyKEP/SPICE/rank JSON as a seed, but refine directly against
Principia VCAREL_NAV without re-launching the server for every evaluation.

This script is meant for the current row15 situation:
  - The candidate has a future burn_abs_s and a burn_rel_* state.
  - We do NOT propagate the active vessel for ~860 days before the burn.
  - We start the VCAREL_NAV state at burn_abs_s using burn_rel_r/v from rank.
  - We optimize TNB/Frenet components and optionally arrival offset.
  - Optional DSM can be enabled as a second impulse in TNB/Frenet.

Server command used
-------------------
VCAREL_NAV rid dep_body arr_body nav_body state_abs_s scan_start_rel_s scan_end_rel_s samples \
           rel_r[3] rel_v[3] n_burns [burn_dt dvt dvn dvb]...

Important
---------
For the Principia server, do not pass nav_body=AUTO. Use KERBIN, etc.
The server converts dvt,dvn,dvb internally using its Frenet/NavigationFrame.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.optimize import minimize, differential_evolution

DAY_S = 86400.0


def norm3(v: Iterable[float]) -> float:
    return float(np.linalg.norm(np.asarray(list(v), dtype=float)))


def finite(x: float) -> bool:
    return math.isfinite(float(x))


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def load_json(path: Path) -> Any:
    with path.open() as f:
        return json.load(f)


def dump_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")


def find_candidate(data: Any, row_index0: int | None, top_index: int) -> dict[str, Any]:
    """Finds a ranked candidate robustly in nested JSONs."""
    if isinstance(data, dict):
        if row_index0 is None:
            if isinstance(data.get("top"), list) and data["top"]:
                return data["top"][top_index]
            if isinstance(data.get("ranked"), list) and data["ranked"]:
                return data["ranked"][top_index]
            if isinstance(data.get("candidate"), dict):
                return data["candidate"]
            if "row_index0" in data:
                return data

    found: list[dict[str, Any]] = []

    def walk(x: Any) -> None:
        if isinstance(x, dict):
            if "row_index0" in x and (row_index0 is None or int(x.get("row_index0")) == int(row_index0)):
                # Prefer entries with real burn-state fields.
                if "burn_abs_s" in x or "burn_rel_r_raw_m" in x or "dv_tangent_m_s" in x:
                    found.append(x)
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)

    walk(data)
    if not found:
        raise RuntimeError(f"candidate not found: row_index0={row_index0!r}, top_index={top_index}")

    # Prefer the richest row.
    found.sort(key=lambda c: (
        "burn_rel_r_raw_m" in c,
        "burn_rel_v_raw_m_s" in c,
        "t_arr_s" in c,
        "dv_tangent_m_s" in c,
        len(c),
    ), reverse=True)
    return found[0]


def nested_vessel(snapshot_or_active: dict[str, Any]) -> dict[str, Any]:
    if isinstance(snapshot_or_active.get("vessel"), dict):
        out = dict(snapshot_or_active)
        out.update(snapshot_or_active["vessel"])
        return out
    return snapshot_or_active


def read_float(c: dict[str, Any], *keys: str, default: float | None = None) -> float:
    for k in keys:
        if k in c and c[k] is not None:
            return float(c[k])
    if default is not None:
        return float(default)
    raise KeyError(f"none of keys found: {keys}")


def read_vec3(c: dict[str, Any], *keys: str) -> list[float]:
    for k in keys:
        if k in c and c[k] is not None:
            v = c[k]
            if len(v) != 3:
                raise ValueError(f"{k} is not vec3: {v!r}")
            return [float(v[0]), float(v[1]), float(v[2])]
    raise KeyError(f"none of vec3 keys found: {keys}")


@dataclass
class CandidateSeed:
    row_index0: int | None
    sequence: Any
    dep_body: str
    arr_body: str
    nav_body: str
    state_abs_s: float
    t_arr_s: float
    rel_r_raw_m: list[float]
    rel_v_raw_m_s: list[float]
    dvt0_m_s: float
    dvn0_m_s: float
    dvb0_m_s: float
    dv0_norm_m_s: float


def build_seed(
    cand: dict[str, Any],
    live_state: dict[str, Any] | None,
    dep_body_arg: str | None,
    arr_body_arg: str | None,
    nav_body_arg: str | None,
    flip_normal: bool,
    flip_binormal: bool,
) -> CandidateSeed:
    live = nested_vessel(live_state or {})

    dep = (dep_body_arg or cand.get("dep_body") or live.get("nav_body") or "KERBIN").upper()
    arr = (arr_body_arg or cand.get("arr_body") or "EVE").upper()
    nav = (nav_body_arg or cand.get("nav_body") or dep).upper()
    if nav == "AUTO":
        nav = dep

    # Fast path: ranked candidate already has the state at burn_abs_s.
    if "burn_rel_r_raw_m" in cand and "burn_rel_v_raw_m_s" in cand:
        state_abs_s = read_float(cand, "burn_abs_s")
        rel_r = read_vec3(cand, "burn_rel_r_raw_m")
        rel_v = read_vec3(cand, "burn_rel_v_raw_m_s")
    else:
        # Fallback only for immediate/current-state refinement.
        if not live:
            raise RuntimeError("rank has no burn_rel_* state and no live-state-json was provided")
        state_abs_s = read_float(live, "t_spice_s", "t_game_s")
        rel_r = read_vec3(live, "rel_r_raw_m")
        rel_v = read_vec3(live, "rel_v_raw_m_s")

    t_arr_s = read_float(cand, "t_arr_s")
    dvt = read_float(cand, "dv_tangent_m_s", "dvt_m_s", "T", default=0.0)
    dvn = read_float(cand, "dv_normal_m_s", "dvn_m_s", "N", default=0.0)
    dvb = read_float(cand, "dv_binormal_m_s", "dvb_m_s", "B", default=0.0)

    if flip_normal:
        dvn = -dvn
    if flip_binormal:
        dvb = -dvb

    return CandidateSeed(
        row_index0=cand.get("row_index0"),
        sequence=cand.get("sequence"),
        dep_body=dep,
        arr_body=arr,
        nav_body=nav,
        state_abs_s=float(state_abs_s),
        t_arr_s=float(t_arr_s),
        rel_r_raw_m=rel_r,
        rel_v_raw_m_s=rel_v,
        dvt0_m_s=float(dvt),
        dvn0_m_s=float(dvn),
        dvb0_m_s=float(dvb),
        dv0_norm_m_s=norm3([dvt, dvn, dvb]),
    )


class PrincipiaVcarelNavClient:
    def __init__(self, server: str, plugin_b64: Path, plugin_arg_mode: str = "positional", timeout_s: float = 900.0, quiet_stderr: bool = False):
        self.server = server
        self.plugin_b64 = plugin_b64
        self.plugin_arg_mode = plugin_arg_mode
        self.timeout_s = timeout_s
        self.quiet_stderr = quiet_stderr
        self.proc: subprocess.Popen[str] | None = None
        self.ready_line: str | None = None

    def __enter__(self):
        args = [self.server]
        if self.plugin_arg_mode == "positional":
            args.append(str(self.plugin_b64))
        elif self.plugin_arg_mode == "option":
            args.extend(["--plugin-b64", str(self.plugin_b64)])
        elif self.plugin_arg_mode == "none":
            pass
        else:
            raise ValueError(f"bad plugin_arg_mode: {self.plugin_arg_mode}")

        stderr = subprocess.DEVNULL if self.quiet_stderr else None
        self.proc = subprocess.Popen(
            args,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=stderr,
            text=True,
            bufsize=1,
        )
        self.ready_line = self._readline_timeout(self.timeout_s).strip()
        if not self.ready_line.startswith("READY"):
            raise RuntimeError(f"server did not report READY: {self.ready_line!r}")
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.proc is not None:
            try:
                self.proc.terminate()
                self.proc.wait(timeout=3)
            except Exception:
                try:
                    self.proc.kill()
                except Exception:
                    pass
        self.proc = None

    def _readline_timeout(self, timeout_s: float) -> str:
        assert self.proc is not None and self.proc.stdout is not None
        # Portable enough for Linux terminal use. Avoids blocking forever.
        fd = self.proc.stdout.fileno()
        import select
        end = time.time() + timeout_s
        while True:
            if self.proc.poll() is not None:
                raise RuntimeError(f"server exited with code {self.proc.returncode}")
            rem = end - time.time()
            if rem <= 0:
                raise TimeoutError("timeout waiting for server response")
            r, _, _ = select.select([fd], [], [], min(0.25, rem))
            if r:
                line = self.proc.stdout.readline()
                if line == "":
                    raise RuntimeError("server stdout closed")
                return line

    def command(self, line: str) -> str:
        assert self.proc is not None and self.proc.stdin is not None
        self.proc.stdin.write(line.rstrip("\n") + "\n")
        self.proc.stdin.flush()
        return self._readline_timeout(self.timeout_s).strip()

    def ping(self) -> str:
        return self.command("PING")

    def vcarel_nav(
        self,
        rid: str,
        dep_body: str,
        arr_body: str,
        nav_body: str,
        state_abs_s: float,
        scan_start_rel_s: float,
        scan_end_rel_s: float,
        samples: int,
        rel_r_raw_m: list[float],
        rel_v_raw_m_s: list[float],
        impulses: list[tuple[float, float, float, float]],
    ) -> dict[str, Any]:
        fields: list[Any] = [
            "VCAREL_NAV", rid, dep_body, arr_body, nav_body,
            f"{state_abs_s:.17g}", f"{scan_start_rel_s:.17g}", f"{scan_end_rel_s:.17g}", int(samples),
            *[f"{x:.17g}" for x in rel_r_raw_m],
            *[f"{x:.17g}" for x in rel_v_raw_m_s],
            len(impulses),
        ]
        for dt, dvt, dvn, dvb in impulses:
            fields.extend([f"{dt:.17g}", f"{dvt:.17g}", f"{dvn:.17g}", f"{dvb:.17g}"])
        line = "\t".join(map(str, fields))
        resp = self.command(line)
        if resp.startswith("ERR"):
            return {"ok": False, "error": resp, "raw_response": resp}
        return parse_okcarelnav(resp)


def parse_okcarelnav(resp: str) -> dict[str, Any]:
    p = resp.strip().split()
    if not p:
        return {"ok": False, "error": "empty response", "raw_response": resp}
    if p[0] != "OKCARELNAV":
        return {"ok": False, "error": f"unexpected response: {p[0]}", "raw_response": resp}
    try:
        out: dict[str, Any] = {
            "ok": True,
            "raw_response": resp,
            "id": p[1],
            "dep_body": p[2],
            "arr_body": p[3],
            "nav_body": p[4],
            "state_dt_s": float(p[5]),
            "state_t_game_s": float(p[6]),
            "ca_dt_s": float(p[7]),
            "ca_t_game_s": float(p[8]),
            "ca_rel_r_raw_m": [float(p[9]), float(p[10]), float(p[11])],
            "ca_rel_v_raw_m_s": [float(p[12]), float(p[13]), float(p[14])],
            "ca_distance_m": float(p[15]),
            "ca_distance_km": float(p[15]) / 1000.0,
            "ca_speed_m_s": float(p[16]),
            "ca_radial_velocity_m_s": float(p[17]),
            "samples": int(float(p[18])),
            "status": p[19],
        }
        i = 20
        if len(p) >= i + 12:
            out["ca_abs_debug_r_raw_m"] = [float(p[i]), float(p[i+1]), float(p[i+2])]; i += 3
            out["ca_abs_debug_v_raw_m_s"] = [float(p[i]), float(p[i+1]), float(p[i+2])]; i += 3
            out["arr_abs_debug_r_raw_m"] = [float(p[i]), float(p[i+1]), float(p[i+2])]; i += 3
            out["arr_abs_debug_v_raw_m_s"] = [float(p[i]), float(p[i+1]), float(p[i+2])]; i += 3
        burns = []
        if len(p) > i:
            n_burns = int(float(p[i])); i += 1
            out["n_burns"] = n_burns
            for _ in range(n_burns):
                if len(p) < i + 25:
                    break
                b = {
                    "burn_dt_s": float(p[i]),
                    "burn_r_raw_m": [float(p[i+1]), float(p[i+2]), float(p[i+3])],
                    "burn_v_before_raw_m_s": [float(p[i+4]), float(p[i+5]), float(p[i+6])],
                    "dv_tnb_cmd_m_s": [float(p[i+7]), float(p[i+8]), float(p[i+9])],
                    "tangent_raw": [float(p[i+10]), float(p[i+11]), float(p[i+12])],
                    "normal_raw": [float(p[i+13]), float(p[i+14]), float(p[i+15])],
                    "binormal_raw": [float(p[i+16]), float(p[i+17]), float(p[i+18])],
                    "dv_raw_m_s": [float(p[i+19]), float(p[i+20]), float(p[i+21])],
                    "burn_v_after_raw_m_s": [float(p[i+22]), float(p[i+23]), float(p[i+24])],
                }
                i += 25
                burns.append(b)
        out["burns"] = burns
        return out
    except Exception as e:
        return {"ok": False, "error": f"parse failed: {e}", "raw_response": resp, "tokens": p}


def flatten(row: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in row.items():
        if isinstance(v, (dict, list)):
            out[k] = json.dumps(v, separators=(",", ":"))
        else:
            out[k] = v
    return out


class Stopper:
    def __init__(self, max_evals: int, max_time_s: float, target_ca_km: float, min_evals_before_stop: int, stall_evals: int):
        self.max_evals = max_evals
        self.max_time_s = max_time_s
        self.target_ca_km = target_ca_km
        self.min_evals_before_stop = min_evals_before_stop
        self.stall_evals = stall_evals
        self.t0 = time.time()
        self.evals = 0
        self.best_ca = float("inf")
        self.best_eval = 0
        self.stop_reason: str | None = None

    def update(self, ca_km: float) -> None:
        self.evals += 1
        if finite(ca_km) and ca_km < self.best_ca:
            self.best_ca = ca_km
            self.best_eval = self.evals

    def should_stop(self) -> bool:
        if self.max_evals and self.evals >= self.max_evals:
            self.stop_reason = f"max_evals={self.max_evals}"
            return True
        if self.max_time_s and (time.time() - self.t0) >= self.max_time_s:
            self.stop_reason = f"max_time_s={self.max_time_s}"
            return True
        if self.target_ca_km > 0 and self.evals >= self.min_evals_before_stop and self.best_ca <= self.target_ca_km:
            self.stop_reason = f"target_ca_km={self.target_ca_km}"
            return True
        if self.stall_evals and self.evals - self.best_eval >= self.stall_evals:
            self.stop_reason = f"stall_evals={self.stall_evals}"
            return True
        return False


class EarlyStop(Exception):
    pass


def make_event(seed: CandidateSeed, best: dict[str, Any], live_state: dict[str, Any] | None, output_dir: Path, event_note: str) -> Path:
    live = nested_vessel(live_state or {})
    path = output_dir / "event1_fast_refined_vcarelnav_navigation.json"
    event = {
        "enabled": True,
        "vessel_guid": live.get("vessel_guid", ""),
        "ensure_flight_plan": True,
        "extend_existing_flight_plan": True,
        "mass_tonnes": live.get("mass_tonnes", 0),
        "insert_index": 0,
        "burn_template": "json",
        "thrust_kN": live.get("available_thrust_kN", 0),
        "specific_impulse_s_g0": live.get("specific_impulse_s_g0", 0),
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
        "request_id": f"row{seed.row_index0}_fast_refined_vcarelnav_attempt0",
        "dedupe_tag": f"row{seed.row_index0}_fast_refined_vcarelnav",
        "event_key": f"row{seed.row_index0}_fast_refined_vcarelnav",
        "attempt": 0,
        "mode": "insert_navigation",
        "initial_time": seed.state_abs_s,
        "plan_final_time": seed.state_abs_s + 600.0,
        "delta_v_navigation_m_s": [float(best["dvt_m_s"]), float(best["dvn_m_s"]), float(best["dvb_m_s"])],
        "planned_from_state": {
            "schema": "planned_from_fast_principia_vcarelnav_refine_v0",
            "warning": event_note,
            "row_index0": seed.row_index0,
            "sequence": seed.sequence,
            "dep_body": seed.dep_body,
            "arr_body": seed.arr_body,
            "nav_body": seed.nav_body,
            "state_abs_s": seed.state_abs_s,
            "t_arr_s": seed.t_arr_s,
            "ca_distance_km": best.get("ca_distance_km"),
            "ca_t_game_s": best.get("ca_t_game_s"),
            "ca_speed_m_s": best.get("ca_speed_m_s"),
            "status": best.get("status"),
        },
        "require_warp_close": True,
        "max_lead_before_insert_s": 600.0,
        "reject_long_plan_before_first_burn": True,
        "max_first_plan_duration_s": 900.0,
        "rollback_on_status_error": True,
        "auto_sort_insert_index": True,
    }
    dump_json(path, event)
    return path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rank-json", type=Path, required=True)
    ap.add_argument("--row-index0", type=int, default=None)
    ap.add_argument("--top-index", type=int, default=0)
    ap.add_argument("--live-state-json", type=Path, default=None, help="Used for vessel metadata and as fallback state only.")

    ap.add_argument("--server", required=True)
    ap.add_argument("--plugin-b64", type=Path, required=True)
    ap.add_argument("--plugin-arg-mode", choices=["positional", "option", "none"], default="positional")

    ap.add_argument("--dep-body", default=None)
    ap.add_argument("--arr-body", default=None)
    ap.add_argument("--nav-body", default=None)

    ap.add_argument("--flip-normal", action="store_true", help="Flip initial normal seed sign before optimizing.")
    ap.add_argument("--flip-binormal", action="store_true", help="Flip initial binormal seed sign before optimizing. Use for row15 server convention.")

    ap.add_argument("--optimize-arrival-offset", action="store_true")
    ap.add_argument("--arrival-offset-initial-days", type=float, default=0.0)
    ap.add_argument("--arrival-offset-min-days", type=float, default=-60.0)
    ap.add_argument("--arrival-offset-max-days", type=float, default=60.0)
    ap.add_argument("--scan-half-width-days", type=float, default=20.0)
    ap.add_argument("--samples", type=int, default=61)

    ap.add_argument("--t-min", type=float, default=None)
    ap.add_argument("--t-max", type=float, default=None)
    ap.add_argument("--n-max-abs", type=float, default=350.0)
    ap.add_argument("--b-min", type=float, default=None)
    ap.add_argument("--b-max", type=float, default=None)
    ap.add_argument("--t-trust", type=float, default=350.0)
    ap.add_argument("--b-trust", type=float, default=500.0)

    ap.add_argument("--enable-dsm", action="store_true")
    ap.add_argument("--dsm-frac-initial", type=float, default=0.50)
    ap.add_argument("--dsm-frac-min", type=float, default=0.15)
    ap.add_argument("--dsm-frac-max", type=float, default=0.85)
    ap.add_argument("--dsm-max-abs", type=float, default=150.0)
    ap.add_argument("--dsm-initial", default="0,0,0", help="DSM T,N,B initial m/s.")

    ap.add_argument("--method", choices=["powell", "nelder-mead", "de-powell"], default="powell")
    ap.add_argument("--multistart", type=int, default=1)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--ms-dv-spread", type=float, default=80.0)
    ap.add_argument("--ms-arrival-spread-days", type=float, default=8.0)
    ap.add_argument("--max-iter", type=int, default=20)
    ap.add_argument("--max-evals", type=int, default=120)
    ap.add_argument("--max-time-s", type=float, default=0.0)
    ap.add_argument("--stall-evals", type=int, default=35)
    ap.add_argument("--target-ca-km", type=float, default=100000.0)
    ap.add_argument("--min-evals-before-stop", type=int, default=12)
    ap.add_argument("--tol", type=float, default=1e-3)

    ap.add_argument("--ca-scale-km", type=float, default=100000.0)
    ap.add_argument("--dv-scale-m-s", type=float, default=1000.0)
    ap.add_argument("--dv-weight", type=float, default=0.01)
    ap.add_argument("--arrival-offset-weight", type=float, default=0.02)
    ap.add_argument("--dsm-weight", type=float, default=0.03)
    ap.add_argument("--boundary-penalty", type=float, default=20.0)
    ap.add_argument("--server-timeout-s", type=float, default=900.0)
    ap.add_argument("--quiet-stderr", action="store_true")
    ap.add_argument("--print-every", type=int, default=1)
    ap.add_argument("--write-event", action="store_true")
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rank = load_json(args.rank_json)
    cand = find_candidate(rank, args.row_index0, args.top_index)
    live = load_json(args.live_state_json) if args.live_state_json else None
    seed = build_seed(cand, live, args.dep_body, args.arr_body, args.nav_body, args.flip_normal, args.flip_binormal)

    t_min = args.t_min if args.t_min is not None else seed.dvt0_m_s - args.t_trust
    t_max = args.t_max if args.t_max is not None else seed.dvt0_m_s + args.t_trust
    b_min = args.b_min if args.b_min is not None else seed.dvb0_m_s - args.b_trust
    b_max = args.b_max if args.b_max is not None else seed.dvb0_m_s + args.b_trust

    dims = ["T", "N", "B"]
    x0 = [seed.dvt0_m_s, seed.dvn0_m_s, seed.dvb0_m_s]
    bounds: list[tuple[float, float]] = [
        (float(t_min), float(t_max)),
        (-float(args.n_max_abs), float(args.n_max_abs)),
        (float(b_min), float(b_max)),
    ]

    if args.optimize_arrival_offset:
        dims.append("arrival_offset_days")
        x0.append(float(args.arrival_offset_initial_days))
        bounds.append((float(args.arrival_offset_min_days), float(args.arrival_offset_max_days)))
    else:
        fixed_arrival_offset_days = float(args.arrival_offset_initial_days)

    if args.enable_dsm:
        dsm0 = [float(x) for x in args.dsm_initial.split(",")]
        if len(dsm0) != 3:
            raise SystemExit("--dsm-initial must be T,N,B")
        dims += ["dsm_frac", "dsm_T", "dsm_N", "dsm_B"]
        x0 += [float(args.dsm_frac_initial), *dsm0]
        bounds += [
            (float(args.dsm_frac_min), float(args.dsm_frac_max)),
            (-float(args.dsm_max_abs), float(args.dsm_max_abs)),
            (-float(args.dsm_max_abs), float(args.dsm_max_abs)),
            (-float(args.dsm_max_abs), float(args.dsm_max_abs)),
        ]

    x0 = np.asarray(x0, dtype=float)
    lo = np.asarray([b[0] for b in bounds], dtype=float)
    hi = np.asarray([b[1] for b in bounds], dtype=float)

    rows: list[dict[str, Any]] = []
    cache: dict[tuple[float, ...], float] = {}
    stopper = Stopper(args.max_evals, args.max_time_s, args.target_ca_km, args.min_evals_before_stop, args.stall_evals)

    print("=== FAST PRINCIPIA VCAREL_NAV REFINE V0 ===")
    print(f"server        : {args.server}")
    print(f"rank_json     : {args.rank_json}")
    print(f"row_index0    : {seed.row_index0}")
    print(f"sequence      : {seed.sequence}")
    print(f"dep/arr/nav   : {seed.dep_body} -> {seed.arr_body} / {seed.nav_body}")
    print(f"state_abs_s   : {seed.state_abs_s}")
    print(f"t_arr_s       : {seed.t_arr_s}")
    print(f"tof_days      : {(seed.t_arr_s - seed.state_abs_s) / DAY_S:.6f}")
    print(f"rel_r_km      : {norm3(seed.rel_r_raw_m) / 1000.0:.6f}")
    print(f"rel_v_m_s     : {norm3(seed.rel_v_raw_m_s):.6f}")
    print(f"x dims        : {dims}")
    print(f"x0            : {x0.tolist()}")
    print(f"bounds        : {bounds}")
    print(f"scan_half_days: {args.scan_half_width_days}, samples={args.samples}")
    print(f"output_dir    : {args.output_dir}")

    def unpack(x_raw: np.ndarray) -> tuple[list[tuple[float, float, float, float]], float, dict[str, float]]:
        x = np.asarray(x_raw, dtype=float)
        x = np.minimum(np.maximum(x, lo), hi)
        vals = {name: float(x[i]) for i, name in enumerate(dims)}
        arr_off = vals.get("arrival_offset_days", fixed_arrival_offset_days if not args.optimize_arrival_offset else 0.0)
        impulses = [(0.0, vals["T"], vals["N"], vals["B"])]
        if args.enable_dsm:
            tof = seed.t_arr_s - seed.state_abs_s
            dsm_dt = vals["dsm_frac"] * tof
            impulses.append((dsm_dt, vals["dsm_T"], vals["dsm_N"], vals["dsm_B"]))
        return impulses, arr_off, vals

    def penalty_for(vals: dict[str, float], arr_off: float) -> float:
        dv0 = norm3([vals["T"], vals["N"], vals["B"]])
        p = args.dv_weight * ((dv0 - seed.dv0_norm_m_s) / args.dv_scale_m_s) ** 2
        if args.optimize_arrival_offset:
            p += args.arrival_offset_weight * (arr_off / 30.0) ** 2
        if args.enable_dsm:
            dsm = norm3([vals["dsm_T"], vals["dsm_N"], vals["dsm_B"]])
            p += args.dsm_weight * (dsm / max(1.0, args.dsm_max_abs)) ** 2
        # Soft boundary penalty so Powell does not happily sit at clipped edges.
        for i, name in enumerate(dims):
            span = max(1e-9, hi[i] - lo[i])
            z = min((vals[name] - lo[i]) / span, (hi[i] - vals[name]) / span)
            if z < 0.03:
                p += args.boundary_penalty * (0.03 - z) ** 2
        return float(p)

    def objective_factory(client: PrincipiaVcarelNavClient):
        def obj(x_raw: np.ndarray) -> float:
            x_clipped = np.minimum(np.maximum(np.asarray(x_raw, dtype=float), lo), hi)
            key = tuple(round(float(v), 6) for v in x_clipped)
            if key in cache:
                return cache[key]
            if stopper.should_stop():
                raise EarlyStop(stopper.stop_reason or "stop")

            impulses, arr_off, vals = unpack(x_clipped)
            scan_center = (seed.t_arr_s - seed.state_abs_s) + arr_off * DAY_S
            scan_start = scan_center - args.scan_half_width_days * DAY_S
            scan_end = scan_center + args.scan_half_width_days * DAY_S

            rid = f"eval_{stopper.evals + 1:05d}"
            t_eval_start = time.time()
            res = client.vcarel_nav(
                rid=rid,
                dep_body=seed.dep_body,
                arr_body=seed.arr_body,
                nav_body=seed.nav_body,
                state_abs_s=seed.state_abs_s,
                scan_start_rel_s=scan_start,
                scan_end_rel_s=scan_end,
                samples=args.samples,
                rel_r_raw_m=seed.rel_r_raw_m,
                rel_v_raw_m_s=seed.rel_v_raw_m_s,
                impulses=impulses,
            )
            elapsed = time.time() - t_eval_start

            if res.get("ok"):
                ca_km = float(res["ca_distance_km"])
                score = (ca_km / args.ca_scale_km) + penalty_for(vals, arr_off)
            else:
                ca_km = float("inf")
                score = 1e9

            row = {
                "eval": stopper.evals + 1,
                "ok": bool(res.get("ok")),
                "score": float(score),
                "ca_distance_km": ca_km,
                "elapsed_s": elapsed,
                "arrival_offset_days": arr_off,
                "scan_start_rel_s": scan_start,
                "scan_end_rel_s": scan_end,
                "dvt_m_s": vals["T"],
                "dvn_m_s": vals["N"],
                "dvb_m_s": vals["B"],
                "dv_norm_m_s": norm3([vals["T"], vals["N"], vals["B"]]),
                "impulses": impulses,
                **{f"x_{k}": v for k, v in vals.items()},
            }
            if res.get("ok"):
                for k in ("ca_t_game_s", "ca_dt_s", "ca_speed_m_s", "ca_radial_velocity_m_s", "status", "ca_rel_r_raw_m", "ca_rel_v_raw_m_s", "burns"):
                    if k in res:
                        row[k] = res[k]
            else:
                row["error"] = res.get("error")
                row["raw_response"] = res.get("raw_response")

            rows.append(row)
            stopper.update(ca_km)
            cache[key] = float(score)

            if args.print_every and (row["eval"] % args.print_every == 0 or ca_km <= stopper.best_ca):
                print(
                    f"[{row['eval']:04d}] score={score:10.5g} ca={ca_km:12.3f}km "
                    f"T={vals['T']:9.3f} N={vals['N']:9.3f} B={vals['B']:9.3f} "
                    f"arr={arr_off:7.2f}d status={row.get('status')} elapsed={elapsed:6.2f}s"
                )

            if stopper.should_stop():
                raise EarlyStop(stopper.stop_reason or "stop")
            return float(score)
        return obj

    def write_outputs(best: dict[str, Any] | None, result_note: str) -> None:
        ok_rows = [r for r in rows if r.get("ok") and finite(float(r.get("ca_distance_km", math.inf)))]
        ok_rows.sort(key=lambda r: (float(r["ca_distance_km"]), float(r.get("score", math.inf))))
        all_rows = sorted(rows, key=lambda r: (0 if r.get("ok") else 1, float(r.get("ca_distance_km", math.inf)), float(r.get("score", math.inf))))
        out = {
            "schema": "fast_principia_vcarelnav_refine_v0",
            "note": result_note,
            "rank_json": str(args.rank_json),
            "live_state_json": None if args.live_state_json is None else str(args.live_state_json),
            "server": args.server,
            "plugin_b64": str(args.plugin_b64),
            "seed": asdict(seed),
            "config": vars(args),
            "dims": dims,
            "x0": x0.tolist(),
            "bounds": bounds,
            "n_rows": len(rows),
            "n_ok": len(ok_rows),
            "stop_reason": stopper.stop_reason,
            "best": best or (ok_rows[0] if ok_rows else None),
            "top": ok_rows[:50],
            "rows": rows,
        }
        dump_json(args.output_dir / "fast_principia_vcarelnav_refine_result.json", out)

        if rows:
            flat = [flatten(r) for r in rows]
            keys = sorted({k for r in flat for k in r.keys()})
            with (args.output_dir / "fast_principia_vcarelnav_refine_rows.csv").open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=keys)
                w.writeheader()
                w.writerows(flat)

        if ok_rows:
            print("\n=== TOP OK ROWS ===")
            for i, r in enumerate(ok_rows[:15], 1):
                print(
                    f"{i:02d} ca={r['ca_distance_km']:12.3f}km score={r['score']:10.5g} "
                    f"T={r['dvt_m_s']:9.3f} N={r['dvn_m_s']:9.3f} B={r['dvb_m_s']:9.3f} "
                    f"arr={r['arrival_offset_days']:7.2f}d speed={r.get('ca_speed_m_s', math.nan):9.3f}"
                )

    best_row: dict[str, Any] | None = None
    note = "completed"

    with PrincipiaVcarelNavClient(args.server, args.plugin_b64, args.plugin_arg_mode, args.server_timeout_s, args.quiet_stderr) as client:
        print(f"READY line    : {client.ready_line}")
        try:
            print(f"PING          : {client.ping()}")
        except Exception as e:
            print(f"PING failed   : {e}")

        obj = objective_factory(client)

        # Always evaluate seed first.
        try:
            obj(x0)
        except EarlyStop as e:
            note = f"early_stop_after_seed: {e}"

        starts = [x0]
        for _ in range(max(0, args.multistart - 1)):
            xs = x0.copy()
            xs[0] += random.uniform(-args.ms_dv_spread, args.ms_dv_spread)
            xs[1] += random.uniform(-args.ms_dv_spread, args.ms_dv_spread)
            xs[2] += random.uniform(-args.ms_dv_spread, args.ms_dv_spread)
            if args.optimize_arrival_offset:
                idx = dims.index("arrival_offset_days")
                xs[idx] += random.uniform(-args.ms_arrival_spread_days, args.ms_arrival_spread_days)
            if args.enable_dsm:
                for name in ("dsm_T", "dsm_N", "dsm_B"):
                    idx = dims.index(name)
                    xs[idx] += random.uniform(-args.ms_dv_spread, args.ms_dv_spread)
                idx = dims.index("dsm_frac")
                xs[idx] += random.uniform(-0.1, 0.1)
            starts.append(np.minimum(np.maximum(xs, lo), hi))

        try:
            if args.method == "de-powell":
                # Small global-ish pre-pass, then Powell from the best DE point.
                print("\n=== differential_evolution pre-pass ===")
                de_res = differential_evolution(
                    obj,
                    bounds=bounds,
                    maxiter=max(1, min(8, args.max_iter)),
                    popsize=5,
                    polish=False,
                    seed=args.seed,
                    updating="immediate",
                    workers=1,
                    tol=0.05,
                )
                starts.insert(0, np.asarray(de_res.x, dtype=float))

            for si, start in enumerate(starts, 1):
                if stopper.should_stop():
                    break
                print(f"\n=== local optimize start {si}/{len(starts)} ===")
                print("start:", start.tolist())
                if args.method in ("powell", "de-powell"):
                    minimize(
                        obj,
                        start,
                        method="Powell",
                        bounds=bounds,
                        options={"maxiter": args.max_iter, "maxfev": max(1, args.max_evals - stopper.evals), "xtol": args.tol, "ftol": args.tol, "disp": False},
                    )
                elif args.method == "nelder-mead":
                    # Nelder-Mead has no true bounds; objective clips.
                    minimize(
                        obj,
                        start,
                        method="Nelder-Mead",
                        options={"maxiter": args.max_iter, "maxfev": max(1, args.max_evals - stopper.evals), "xatol": args.tol, "fatol": args.tol, "disp": False},
                    )
        except EarlyStop as e:
            note = f"early_stop: {e}"
        except KeyboardInterrupt:
            note = "keyboard_interrupt"
            print("\n[INTERRUPTED] writing partial results...")

    ok_rows = [r for r in rows if r.get("ok") and finite(float(r.get("ca_distance_km", math.inf)))]
    ok_rows.sort(key=lambda r: (float(r["ca_distance_km"]), float(r.get("score", math.inf))))
    best_row = ok_rows[0] if ok_rows else None
    write_outputs(best_row, note)

    if args.write_event and best_row:
        event_note = "TNB is in server/Principia VCAREL_NAV convention. Confirm DLL insert_navigation binormal convention before pushing."
        p = make_event(seed, best_row, live, args.output_dir, event_note)
        print(f"[OK] wrote event: {p}")

    print(f"[OK] wrote: {args.output_dir / 'fast_principia_vcarelnav_refine_result.json'}")
    print(f"[OK] wrote: {args.output_dir / 'fast_principia_vcarelnav_refine_rows.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
