#!/usr/bin/env python3
"""
IPOPT v0.4 departure relative matchpoint with one DSM.

Why v0.4
--------
v0.3 fixed the frame/epoch issue, but tried to match a 6D target state
(position + velocity) with only 4 decision variables:

    burn_dt + dv0_xyz

That is generally overdetermined and can converge to a local infeasibility.
v0.4 adds one DSM before the matchpoint:

    x = [
      burn_dt_s,
      dv0x_raw_m_s, dv0y_raw_m_s, dv0z_raw_m_s,
      dsm_dt_s,
      dsmx_raw_m_s, dsmy_raw_m_s, dsmz_raw_m_s,
    ]

Evaluation:
    VPROPN id vessel_guid match_dt_s 2 burn0 dsm

Target:
    relative state at leg t_start_s:
      target_rel_r = CSV start_r_raw - SPICE(dep_body, t_start_s)
      target_rel_v = CSV optimized/start/initial_v_raw - SPICE(dep_body, t_start_s)

This is the first useful multiple-shooting block:
    real vessel parking state -> patchpoint of abstract optimized leg
with enough control authority to match both position and velocity.
"""

from __future__ import annotations

import argparse
import csv
import inspect
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import spiceypy as spice

try:
    import cyipopt
except Exception as exc:  # pragma: no cover
    cyipopt = None
    _CYIPOPT_IMPORT_ERROR = exc
else:
    _CYIPOPT_IMPORT_ERROR = None

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from vessel_server_client import Burn, VesselPropnClient


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


def unit(v: Sequence[float], fallback: Sequence[float] = (1.0, 0.0, 0.0)) -> np.ndarray:
    a = np.asarray(v, dtype=float)
    n = np.linalg.norm(a)
    if not np.isfinite(n) or n <= 0:
        return np.asarray(fallback, dtype=float)
    return a / n


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
    if not rows:
        raise RuntimeError(f"No rows in {path}")
    for row in rows:
        if row.get("leg"):
            try:
                val = int(float(row["leg"]))
            except Exception:
                continue
            if val == leg or val == leg - 1:
                return row
    return rows[leg - 1]


def frow(row: dict[str, str], key: str) -> float:
    if key not in row:
        raise KeyError(f"Missing column {key!r}. Available: {list(row.keys())}")
    return float(row[key])


def row_r(row: dict[str, str], prefix: str) -> np.ndarray:
    return np.array([
        frow(row, f"{prefix}_x_raw_m"),
        frow(row, f"{prefix}_y_raw_m"),
        frow(row, f"{prefix}_z_raw_m"),
    ], dtype=float)


def row_v(row: dict[str, str], mode: str) -> np.ndarray:
    if mode == "optimized":
        p = "optimized"
    elif mode == "start":
        p = "start"
    elif mode == "initial":
        p = "initial"
    else:
        raise ValueError(f"Unknown velocity mode: {mode}")
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


def tangent_direction_from_relative(rel_r: np.ndarray, rel_v: np.ndarray) -> np.ndarray:
    radial = unit(rel_r)
    tang = rel_v - float(np.dot(rel_v, radial)) * radial
    return unit(tang, fallback=rel_v)


def direction_from_mode(mode: str, initial_rel_r: np.ndarray, initial_rel_v: np.ndarray) -> np.ndarray:
    mode = mode.lower()
    if mode in ("prograde", "tangent"):
        return tangent_direction_from_relative(initial_rel_r, initial_rel_v)
    if mode in ("velocity", "relative_velocity"):
        return unit(initial_rel_v)
    if mode == "raw_x":
        return np.array([1.0, 0.0, 0.0])
    if mode == "raw_y":
        return np.array([0.0, 1.0, 0.0])
    if mode == "raw_z":
        return np.array([0.0, 0.0, 1.0])
    raise ValueError(f"Unknown direction mode: {mode}")


@dataclass
class EvalResult:
    f: float
    g: np.ndarray
    data: dict[str, Any]


class DepartureRelativeMatchDSM_NLP:
    def __init__(self, cfg: dict[str, Any], client: VesselPropnClient):
        self.cfg = cfg
        self.client = client
        self.cache: dict[tuple[float, ...], EvalResult] = {}
        self.eval_count = 0
        self.last: EvalResult | None = None
        self.best_objective: EvalResult | None = None
        self.best_position: EvalResult | None = None
        self.best_lexicographic: EvalResult | None = None

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        c = self.cfg
        lb = np.array([
            c["burn_dt_min_s"],
            -c["dv_component_bound_m_s"],
            -c["dv_component_bound_m_s"],
            -c["dv_component_bound_m_s"],
            c["dsm_dt_min_s"],
            -c["dsm_component_bound_m_s"],
            -c["dsm_component_bound_m_s"],
            -c["dsm_component_bound_m_s"],
        ], dtype=float)
        ub = np.array([
            c["burn_dt_max_s"],
            c["dv_component_bound_m_s"],
            c["dv_component_bound_m_s"],
            c["dv_component_bound_m_s"],
            c["dsm_dt_max_s"],
            c["dsm_component_bound_m_s"],
            c["dsm_component_bound_m_s"],
            c["dsm_component_bound_m_s"],
        ], dtype=float)
        return lb, ub

    def constraint_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        c = self.cfg
        gl = np.array([
            c["min_dsm_after_burn_s"],
            c["min_match_after_dsm_s"],
            c["dv0_min_m_s"],
            0.0,
            0.0,
            0.0,
        ], dtype=float)
        gu = np.array([
            1.0e12,
            1.0e12,
            c["dv0_max_m_s"],
            c["dsm_max_m_s"],
            c["pos_constraint_max_km"],
            c["vel_constraint_max_m_s"],
        ], dtype=float)
        return gl, gu

    def _key(self, x: np.ndarray) -> tuple[float, ...]:
        return tuple(float(f"{v:.12g}") for v in np.asarray(x, dtype=float))

    def _update_best(self, ev: EvalResult) -> None:
        if not ev.data.get("server_ok", False):
            return
        if self.best_objective is None or ev.f < self.best_objective.f:
            self.best_objective = ev
        if self.best_position is None or ev.data["final_pos_err_km"] < self.best_position.data["final_pos_err_km"]:
            self.best_position = ev

        # Lexicographic: first get below pos threshold, then minimize velocity; otherwise minimize position.
        pos_thr = self.cfg["lexicographic_pos_threshold_km"]
        if self.best_lexicographic is None:
            self.best_lexicographic = ev
            return
        a = ev.data
        b = self.best_lexicographic.data
        a_good = a["final_pos_err_km"] <= pos_thr
        b_good = b["final_pos_err_km"] <= pos_thr
        if a_good and not b_good:
            self.best_lexicographic = ev
        elif a_good and b_good and a["final_vel_err_m_s"] < b["final_vel_err_m_s"]:
            self.best_lexicographic = ev
        elif (not a_good) and (not b_good) and a["final_pos_err_km"] < b["final_pos_err_km"]:
            self.best_lexicographic = ev

    def evaluate(self, x: np.ndarray) -> EvalResult:
        x = np.asarray(x, dtype=float)
        key = self._key(x)
        if key in self.cache:
            return self.cache[key]

        c = self.cfg
        self.eval_count += 1

        burn_dt = float(x[0])
        dv0 = np.asarray(x[1:4], dtype=float)
        dsm_dt = float(x[4])
        dsm = np.asarray(x[5:8], dtype=float)

        dv0_norm = norm(dv0)
        dsm_norm = norm(dsm)
        burn_dt_call = max(0.0, burn_dt)
        dsm_dt_call = max(0.0, dsm_dt)

        request_id = f"ipopt_relmatch_dsm_{os.getpid()}_{self.eval_count}"

        try:
            res = self.client.vpropn(
                request_id,
                c["vessel_guid"],
                c["match_dt_s"],
                [
                    Burn(burn_dt_call, dv0.tolist()),
                    Burn(dsm_dt_call, dsm.tolist()),
                ],
                timeout_s=c["server_timeout_s"],
            )
            server_ok = True
            server_error = ""
            final_rel_r = np.asarray(res["final_parent_r_m"], dtype=float)
            final_rel_v = np.asarray(res["final_parent_v_m_s"], dtype=float)
            server_final_abs_r = np.asarray(res["final_r_raw_m"], dtype=float)
            server_final_abs_v = np.asarray(res["final_v_raw_m_s"], dtype=float)
        except Exception as exc:
            server_ok = False
            server_error = str(exc)
            final_rel_r = c["target_rel_r_raw_m"] + np.array([c["fail_soft_pos_km"] * 1000.0, 0.0, 0.0])
            final_rel_v = c["target_rel_v_raw_m_s"] + np.array([c["fail_soft_vel_m_s"], 0.0, 0.0])
            server_final_abs_r = np.array([np.nan, np.nan, np.nan])
            server_final_abs_v = np.array([np.nan, np.nan, np.nan])
            res = {"burns": []}

        target_rel_r = c["target_rel_r_raw_m"]
        target_rel_v = c["target_rel_v_raw_m_s"]

        pos_err_km = norm(final_rel_r - target_rel_r) / 1000.0
        vel_err_m_s = norm(final_rel_v - target_rel_v)

        final_abs_reconstructed_r = c["dep_r_at_match_raw_m"] + final_rel_r
        final_abs_reconstructed_v = c["dep_v_at_match_raw_m_s"] + final_rel_v

        f = (
            0.5 * (pos_err_km / c["pos_scale_km"]) ** 2
            + 0.5 * c["vel_weight"] * (vel_err_m_s / c["vel_scale_m_s"]) ** 2
            + c["dv0_weight"] * (dv0_norm / c["dv_scale_m_s"])
            + c["dsm_weight"] * (dsm_norm / max(1.0, c["dsm_max_m_s"]))
            + c["burn_time_weight"] * abs(burn_dt - c["burn_dt_nominal_s"]) / max(1.0, c["burn_dt_trust_s"])
        )
        if not server_ok:
            f += c["server_error_penalty"]

        g = np.array([
            dsm_dt - burn_dt,
            c["match_dt_s"] - dsm_dt,
            dv0_norm,
            dsm_norm,
            pos_err_km,
            vel_err_m_s,
        ], dtype=float)

        data = {
            "server_ok": server_ok,
            "server_error": server_error,
            "x": x.tolist(),
            "burn_dt_s": burn_dt,
            "dsm_dt_s": dsm_dt,
            "match_dt_s": c["match_dt_s"],
            "t_match_spice_s": c["spice_t0_s"] + c["match_dt_s"],

            "dv0_raw_m_s": dv0.tolist(),
            "dv0_levela_m_s": raw_to_levela(dv0).tolist(),
            "dv0_norm_m_s": dv0_norm,
            "dsm_raw_m_s": dsm.tolist(),
            "dsm_levela_m_s": raw_to_levela(dsm).tolist(),
            "dsm_norm_m_s": dsm_norm,
            "total_impulsive_dv_m_s": dv0_norm + dsm_norm,

            "final_pos_err_km": pos_err_km,
            "final_vel_err_m_s": vel_err_m_s,
            "arrival_vinf_in_m_s": vel_err_m_s,
            "vinf_mag_mismatch_m_s": vel_err_m_s,

            "final_rel_r_raw_m": final_rel_r.tolist(),
            "final_rel_v_raw_m_s": final_rel_v.tolist(),
            "target_rel_r_raw_m": target_rel_r.tolist(),
            "target_rel_v_raw_m_s": target_rel_v.tolist(),

            "final_r_raw_m": final_abs_reconstructed_r.tolist(),
            "final_v_raw_m_s": final_abs_reconstructed_v.tolist(),
            "target_r_raw_m": c["target_abs_r_raw_m"].tolist(),
            "target_v_raw_m_s": c["target_abs_v_raw_m_s"].tolist(),

            "server_final_abs_r_raw_m": server_final_abs_r.tolist(),
            "server_final_abs_v_raw_m_s": server_final_abs_v.tolist(),

            "initial_parent_distance_m": res.get("initial_parent_distance_m"),
            "initial_parent_speed_m_s": res.get("initial_parent_speed_m_s"),
            "initial_parent_radial_velocity_m_s": res.get("initial_parent_radial_velocity_m_s"),
            "final_parent_distance_m": res.get("final_parent_distance_m"),
            "final_parent_speed_m_s": res.get("final_parent_speed_m_s"),
            "final_parent_radial_velocity_m_s": res.get("final_parent_radial_velocity_m_s"),
            "burns": res.get("burns", []),
            "raw_server": res,
            "objective": float(f),
            "constraints": g.tolist(),
        }

        ev = EvalResult(float(f), g, data)
        self.cache[key] = ev
        self.last = ev
        self._update_best(ev)
        return ev

    def objective(self, x: np.ndarray) -> float:
        return self.evaluate(x).f

    def constraints(self, x: np.ndarray) -> np.ndarray:
        return self.evaluate(x).g

    def gradient(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        grad = np.zeros_like(x)
        f0 = self.objective(x)
        steps = np.asarray(self.cfg["fd_steps"], dtype=float)
        lb, ub = self.bounds()
        for i, h in enumerate(steps):
            xp = x.copy()
            xp[i] += h
            if xp[i] > ub[i]:
                xp[i] = x[i] - h
                grad[i] = (f0 - self.objective(xp)) / h
            else:
                grad[i] = (self.objective(xp) - f0) / h
        return grad

    def jacobian(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        g0 = self.constraints(x)
        jac = np.zeros((len(g0), len(x)), dtype=float)
        steps = np.asarray(self.cfg["fd_steps"], dtype=float)
        lb, ub = self.bounds()
        for j, h in enumerate(steps):
            xp = x.copy()
            xp[j] += h
            if xp[j] > ub[j]:
                xp[j] = x[j] - h
                jac[:, j] = (g0 - self.constraints(xp)) / h
            else:
                jac[:, j] = (self.constraints(xp) - g0) / h
        return jac.reshape(-1)

    def intermediate(self, alg_mod, iter_count, obj_value, inf_pr, inf_du, mu, d_norm, regularization_size, alpha_du, alpha_pr, ls_trials):
        if iter_count % max(1, self.cfg["print_every"]) == 0:
            last = self.last.data if self.last else {}
            best = self.best_objective.data if self.best_objective else {}
            print(
                f"[ipopt-relmatch-dsm] it={iter_count:4d} obj={obj_value:12.6g} "
                f"pos={last.get('final_pos_err_km', float('nan')):10.3f} km "
                f"vel={last.get('final_vel_err_m_s', float('nan')):9.3f} "
                f"dv0={last.get('dv0_norm_m_s', float('nan')):8.2f} "
                f"dsm={last.get('dsm_norm_m_s', float('nan')):8.2f} "
                f"best_obj={best.get('objective', float('nan')):10.4g} "
                f"evals={self.eval_count}",
                flush=True,
            )
        return True


def build_x0(cfg: dict[str, Any], client: VesselPropnClient) -> np.ndarray:
    smoke = client.vpropn("ipopt_relmatch_dsm_init", cfg["vessel_guid"], 1.0, [], timeout_s=cfg["server_timeout_s"])
    cfg["server_t0_game_s"] = float(smoke["t0_game_s"])

    initial_rel_r = np.asarray(smoke["initial_parent_r_m"], dtype=float)
    initial_rel_v = np.asarray(smoke["initial_parent_v_m_s"], dtype=float)
    direction = direction_from_mode(cfg["initial_dv0_direction"], initial_rel_r, initial_rel_v)
    dv0 = cfg["dv0_initial_m_s"] * direction

    dsm = np.asarray(cfg["dsm_initial_raw_m_s"], dtype=float)

    x0 = np.array([
        cfg["burn_dt_initial_s"],
        dv0[0], dv0[1], dv0[2],
        cfg["dsm_dt_initial_s"],
        dsm[0], dsm[1], dsm[2],
    ], dtype=float)
    lb, ub = DepartureRelativeMatchDSM_NLP(cfg, client).bounds()
    return np.minimum(np.maximum(x0, lb), ub)


def load_x0_json(path: Path, cfg: dict[str, Any]) -> np.ndarray:
    r = json.loads(path.read_text())
    if "x" in r and len(r["x"]) == 8:
        return np.asarray(r["x"], dtype=float)

    best = r.get("best", r)
    if "burn_dt_s" in best and "dv0_raw_m_s" in best:
        dsm_dt = best.get("dsm_dt_s", cfg["dsm_dt_initial_s"])
        dsm = best.get("dsm_raw_m_s", [0.0, 0.0, 0.0])
        return np.array([best["burn_dt_s"], *best["dv0_raw_m_s"], dsm_dt, *dsm], dtype=float)

    raise RuntimeError(f"Cannot parse x0 from {path}")


def parse_vec3(s: str) -> list[float]:
    vals = [float(x.strip()) for x in s.replace(";", ",").split(",") if x.strip()]
    if len(vals) != 3:
        raise argparse.ArgumentTypeError("expected 3 comma-separated floats")
    return vals


def main() -> int:
    ap = argparse.ArgumentParser(description="IPOPT v0.4 relative LKO -> leg start matchpoint with one DSM.")
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
    ap.add_argument("--dep-body", default="KERBIN")
    ap.add_argument("--spice-center", default="SUN")
    ap.add_argument("--spice-frame", default="J2000")

    ap.add_argument("--burn-dt-min-s", type=float, default=None)
    ap.add_argument("--burn-dt-max-s", type=float, default=None)
    ap.add_argument("--burn-dt-trust-s", type=float, default=7200.0)
    ap.add_argument("--burn-dt-initial-s", type=float, default=None)
    ap.add_argument("--burn-dt-initial-offset-s", type=float, default=0.0)

    ap.add_argument("--dv0-initial-m-s", type=float, default=2000.0)
    ap.add_argument("--dv0-min-m-s", type=float, default=1400.0)
    ap.add_argument("--dv0-max-m-s", type=float, default=3200.0)
    ap.add_argument("--dv-component-bound-m-s", type=float, default=5000.0)
    ap.add_argument("--initial-dv0-direction", choices=["prograde", "tangent", "velocity", "raw_x", "raw_y", "raw_z"], default="prograde")

    ap.add_argument("--dsm-dt-initial-s", type=float, default=None)
    ap.add_argument("--dsm-dt-fraction", type=float, default=0.35)
    ap.add_argument("--dsm-dt-min-s", type=float, default=None)
    ap.add_argument("--dsm-dt-max-s", type=float, default=None)
    ap.add_argument("--min-dsm-after-burn-s", type=float, default=3600.0)
    ap.add_argument("--min-match-after-dsm-s", type=float, default=3600.0)
    ap.add_argument("--dsm-max-m-s", type=float, default=1500.0)
    ap.add_argument("--dsm-component-bound-m-s", type=float, default=2000.0)
    ap.add_argument("--dsm-initial-raw-m-s", type=parse_vec3, default=[0.0, 0.0, 0.0])

    ap.add_argument("--pos-scale-km", type=float, default=1000.0)
    ap.add_argument("--vel-scale-m-s", type=float, default=100.0)
    ap.add_argument("--vel-weight", type=float, default=1.0)
    ap.add_argument("--dv-scale-m-s", type=float, default=1000.0)
    ap.add_argument("--dv0-weight", type=float, default=0.001)
    ap.add_argument("--dsm-weight", type=float, default=0.001)
    ap.add_argument("--burn-time-weight", type=float, default=0.0001)

    ap.add_argument("--pos-constraint-max-km", type=float, default=1.0e9)
    ap.add_argument("--vel-constraint-max-m-s", type=float, default=1.0e9)
    ap.add_argument("--lexicographic-pos-threshold-km", type=float, default=1000.0)
    ap.add_argument("--best-selection", choices=["objective", "position", "lexicographic", "ipopt"], default="objective")

    ap.add_argument("--fd-time-step-s", type=float, default=10.0)
    ap.add_argument("--fd-dv-step-m-s", type=float, default=0.5)

    ap.add_argument("--fail-soft-pos-km", type=float, default=1.0e8)
    ap.add_argument("--fail-soft-vel-m-s", type=float, default=1.0e5)
    ap.add_argument("--server-error-penalty", type=float, default=1.0e6)
    ap.add_argument("--server-timeout-s", type=float, default=180.0)
    ap.add_argument("--quiet-stderr", action="store_true")

    ap.add_argument("--max-iter", type=int, default=150)
    ap.add_argument("--tol", type=float, default=1e-4)
    ap.add_argument("--acceptable-tol", type=float, default=1e-2)
    ap.add_argument("--print-level", type=int, default=5)
    ap.add_argument("--print-every", type=int, default=1)

    ap.add_argument("--x0-json", type=Path, default=None)
    ap.add_argument("--output-dir", type=Path, required=True)

    args = ap.parse_args()

    if cyipopt is None:
        raise SystemExit(
            "[FAIL] cyipopt is not importable. Install cyipopt/Ipopt first. "
            f"Import error: {_CYIPOPT_IMPORT_ERROR}"
        )

    spice.kclear()
    spice.furnsh(str(args.tpc))
    spice.furnsh(str(args.bsp))

    spice_t0_s = read_live_spice_t0(args.live_state_json)
    row = read_leg_row(args.leg_optimizations, args.leg)

    target_t_s = frow(row, "t_start_s")
    match_dt_s = target_t_s - spice_t0_s
    nominal_burn_dt_s = frow(row, "t_dep_s") - spice_t0_s if row.get("t_dep_s") else 0.0

    target_abs_r = row_r(row, "start")
    target_abs_v = row_v(row, args.target_velocity_mode)
    dep_r_match, dep_v_match = spice_body_state_raw(args.dep_body, target_t_s, args.spice_center, args.spice_frame)
    target_rel_r = target_abs_r - dep_r_match
    target_rel_v = target_abs_v - dep_v_match

    burn_dt_initial = args.burn_dt_initial_s
    if burn_dt_initial is None:
        burn_dt_initial = nominal_burn_dt_s + args.burn_dt_initial_offset_s

    burn_dt_min = args.burn_dt_min_s
    burn_dt_max = args.burn_dt_max_s
    if burn_dt_min is None:
        burn_dt_min = max(0.0, nominal_burn_dt_s - args.burn_dt_trust_s)
    if burn_dt_max is None:
        burn_dt_max = max(burn_dt_min + 1.0, nominal_burn_dt_s + args.burn_dt_trust_s)

    dsm_dt_initial = args.dsm_dt_initial_s
    if dsm_dt_initial is None:
        dsm_dt_initial = burn_dt_initial + args.dsm_dt_fraction * max(1.0, match_dt_s - burn_dt_initial)

    dsm_dt_min = args.dsm_dt_min_s
    dsm_dt_max = args.dsm_dt_max_s
    if dsm_dt_min is None:
        dsm_dt_min = max(0.0, burn_dt_min + args.min_dsm_after_burn_s)
    if dsm_dt_max is None:
        dsm_dt_max = max(dsm_dt_min + 1.0, match_dt_s - args.min_match_after_dsm_s)

    cfg: dict[str, Any] = {
        "vessel_guid": args.vessel_guid,
        "spice_t0_s": spice_t0_s,
        "dep_body": args.dep_body.upper(),
        "spice_center": args.spice_center.upper(),
        "spice_frame": args.spice_frame,
        "target_velocity_mode": args.target_velocity_mode,
        "target_t_s": target_t_s,
        "match_dt_s": match_dt_s,

        "dep_r_at_match_raw_m": dep_r_match,
        "dep_v_at_match_raw_m_s": dep_v_match,
        "target_abs_r_raw_m": target_abs_r,
        "target_abs_v_raw_m_s": target_abs_v,
        "target_rel_r_raw_m": target_rel_r,
        "target_rel_v_raw_m_s": target_rel_v,

        "burn_dt_nominal_s": nominal_burn_dt_s,
        "burn_dt_initial_s": burn_dt_initial,
        "burn_dt_min_s": burn_dt_min,
        "burn_dt_max_s": burn_dt_max,
        "burn_dt_trust_s": args.burn_dt_trust_s,

        "dv0_initial_m_s": args.dv0_initial_m_s,
        "dv0_min_m_s": args.dv0_min_m_s,
        "dv0_max_m_s": args.dv0_max_m_s,
        "dv_component_bound_m_s": args.dv_component_bound_m_s,
        "initial_dv0_direction": args.initial_dv0_direction,

        "dsm_dt_initial_s": dsm_dt_initial,
        "dsm_dt_min_s": dsm_dt_min,
        "dsm_dt_max_s": dsm_dt_max,
        "min_dsm_after_burn_s": args.min_dsm_after_burn_s,
        "min_match_after_dsm_s": args.min_match_after_dsm_s,
        "dsm_max_m_s": args.dsm_max_m_s,
        "dsm_component_bound_m_s": args.dsm_component_bound_m_s,
        "dsm_initial_raw_m_s": args.dsm_initial_raw_m_s,

        "pos_scale_km": args.pos_scale_km,
        "vel_scale_m_s": args.vel_scale_m_s,
        "vel_weight": args.vel_weight,
        "dv_scale_m_s": args.dv_scale_m_s,
        "dv0_weight": args.dv0_weight,
        "dsm_weight": args.dsm_weight,
        "burn_time_weight": args.burn_time_weight,

        "pos_constraint_max_km": args.pos_constraint_max_km,
        "vel_constraint_max_m_s": args.vel_constraint_max_m_s,
        "lexicographic_pos_threshold_km": args.lexicographic_pos_threshold_km,

        "fd_steps": [
            args.fd_time_step_s,
            args.fd_dv_step_m_s,
            args.fd_dv_step_m_s,
            args.fd_dv_step_m_s,
            args.fd_time_step_s,
            args.fd_dv_step_m_s,
            args.fd_dv_step_m_s,
            args.fd_dv_step_m_s,
        ],

        "fail_soft_pos_km": args.fail_soft_pos_km,
        "fail_soft_vel_m_s": args.fail_soft_vel_m_s,
        "server_error_penalty": args.server_error_penalty,
        "server_timeout_s": args.server_timeout_s,
        "print_every": args.print_every,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)

    with make_client(args) as client:
        nlp = DepartureRelativeMatchDSM_NLP(cfg, client)
        lb, ub = nlp.bounds()
        gl, gu = nlp.constraint_bounds()

        if args.x0_json is not None:
            x0 = load_x0_json(args.x0_json, cfg)
            x0 = np.minimum(np.maximum(x0, lb), ub)
        else:
            x0 = build_x0(cfg, client)

        print("=== DEPARTURE RELATIVE MATCH IPOPT V0.4 DSM / VPROPN ===")
        print(f"server              : {args.server}")
        print(f"vessel_guid         : {args.vessel_guid}")
        print(f"spice_t0_s          : {spice_t0_s}")
        print(f"target_t_start_s    : {target_t_s}")
        print(f"match_dt_s          : {match_dt_s}")
        print(f"nominal_burn_dt_s   : {nominal_burn_dt_s}")
        print(f"burn_dt bounds      : {burn_dt_min} .. {burn_dt_max}")
        print(f"dsm_dt bounds       : {dsm_dt_min} .. {dsm_dt_max}")
        print(f"target_velocity_mode: {args.target_velocity_mode}")
        print(f"target_rel_r_norm_km: {norm(target_rel_r) / 1000.0}")
        print(f"target_rel_v_norm_ms: {norm(target_rel_v)}")
        print(f"output_dir          : {args.output_dir}")
        print("x0:", json.dumps(x0.tolist()))

        x0_eval = nlp.evaluate(x0)
        print("x0_eval:", json.dumps({
            "objective": x0_eval.f,
            "final_pos_err_km": x0_eval.data["final_pos_err_km"],
            "final_vel_err_m_s": x0_eval.data["final_vel_err_m_s"],
            "dv0_norm_m_s": x0_eval.data["dv0_norm_m_s"],
            "dsm_norm_m_s": x0_eval.data["dsm_norm_m_s"],
            "server_ok": x0_eval.data["server_ok"],
            "server_error": x0_eval.data["server_error"],
        }, indent=2))

        problem = cyipopt.Problem(n=8, m=6, problem_obj=nlp, lb=lb, ub=ub, cl=gl, cu=gu)
        problem.add_option("max_iter", int(args.max_iter))
        problem.add_option("tol", float(args.tol))
        problem.add_option("acceptable_tol", float(args.acceptable_tol))
        problem.add_option("print_level", int(args.print_level))
        problem.add_option("mu_strategy", "adaptive")
        problem.add_option("nlp_scaling_method", "gradient-based")
        problem.add_option("hessian_approximation", "limited-memory")

        x_opt, info = problem.solve(x0)
        ipopt_ev = nlp.evaluate(np.asarray(x_opt, dtype=float))

        if args.best_selection == "ipopt":
            best_ev = ipopt_ev
        elif args.best_selection == "position":
            best_ev = nlp.best_position or ipopt_ev
        elif args.best_selection == "lexicographic":
            best_ev = nlp.best_lexicographic or ipopt_ev
        else:
            best_ev = nlp.best_objective or ipopt_ev

        best = best_ev.data

    out = {
        "status": "ok",
        "solver": "ipopt",
        "problem": "departure_relative_match_dsm_v0_4",
        "best_selection": args.best_selection,
        "ipopt_info": {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in dict(info).items()},
        "config": {
            **{k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in cfg.items() if k != "fd_steps"},
            "leg_optimizations": str(args.leg_optimizations),
            "leg": args.leg,
        },
        "x0": x0.tolist(),
        "x0_eval": x0_eval.data,
        "ipopt_solution": ipopt_ev.data,
        "best_by_objective": (nlp.best_objective.data if nlp.best_objective else None),
        "best_by_position": (nlp.best_position.data if nlp.best_position else None),
        "best_by_lexicographic": (nlp.best_lexicographic.data if nlp.best_lexicographic else None),
        "best": best,
        "champion_f": best["objective"],
        "best_fitness": best["objective"],

        # Required downstream aliases.
        "arrival_vinf_in_m_s": best["arrival_vinf_in_m_s"],
        "final_v_raw_m_s": best["final_v_raw_m_s"],
        "target_v_raw_m_s": best["target_v_raw_m_s"],
        "dv0_norm_m_s": best["dv0_norm_m_s"],
        "vinf_mag_mismatch_m_s": best["vinf_mag_mismatch_m_s"],
        "n_evaluations": nlp.eval_count,
    }

    result_path = args.output_dir / "departure_relative_match_ipopt_v0_4_dsm_result.json"
    result_path.write_text(json.dumps(out, indent=2, default=lambda x: x.decode('utf-8') if isinstance(x, bytes) else str(x)) + "\n")

    summary_path = args.output_dir / "departure_relative_match_ipopt_v0_4_dsm_summary.txt"
    summary_path.write_text(
        "\n".join([
            "=== DEPARTURE RELATIVE MATCH IPOPT V0.4 DSM SUMMARY ===",
            f"selection          : {args.best_selection}",
            f"objective          : {out['champion_f']}",
            f"burn_dt_s          : {best['burn_dt_s']}",
            f"dsm_dt_s           : {best['dsm_dt_s']}",
            f"dv0_norm_m_s       : {best['dv0_norm_m_s']}",
            f"dsm_norm_m_s       : {best['dsm_norm_m_s']}",
            f"pos_err_km         : {best['final_pos_err_km']}",
            f"vel_err_m_s        : {best['final_vel_err_m_s']}",
            f"match_dt_s         : {best['match_dt_s']}",
            f"server_ok          : {best['server_ok']}",
            f"n_evaluations      : {nlp.eval_count}",
            "",
        ])
    )

    print("=== RESULT ===")
    print(json.dumps({
        "selection": args.best_selection,
        "champion_f": out["champion_f"],
        "burn_dt_s": best["burn_dt_s"],
        "dsm_dt_s": best["dsm_dt_s"],
        "dv0_norm_m_s": best["dv0_norm_m_s"],
        "dsm_norm_m_s": best["dsm_norm_m_s"],
        "final_pos_err_km": best["final_pos_err_km"],
        "final_vel_err_m_s": best["final_vel_err_m_s"],
        "server_ok": best["server_ok"],
        "ipopt_status": out["ipopt_info"].get("status"),
        "n_evaluations": nlp.eval_count,
    }, indent=2))
    print(f"[OK] wrote {result_path}")
    print(f"[OK] wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
