#!/usr/bin/env python3
"""
IPOPT v0.1 departure corrector for KSP-MGA-Planner + Principia.

This is intentionally a small first integration step:
  - one real vessel state from VPROPN canonical vessel psychohistory
  - one departure impulse
  - one optional DSM impulse
  - target: final position near arrival body at final_dt_s

Decision vector, all in seconds and raw Principia m/s:
  x = [
    burn_dt_s,
    dv0x_raw_m_s, dv0y_raw_m_s, dv0z_raw_m_s,
    dsm_dt_s,
    dsmx_raw_m_s, dsmy_raw_m_s, dsmz_raw_m_s,
    final_dt_s,
  ]

Constraints:
  g[0] = dsm_dt_s - burn_dt_s
  g[1] = final_dt_s - dsm_dt_s
  g[2] = ||dv0||
  g[3] = ||dsm||
  g[4] = final_pos_err_km

IPOPT receives finite-difference Jacobians. This is expensive but simple and
correct enough for the first integration test. A later version should batch
Jacobian perturbations or move the NLP closer to the native evaluator.

Important v0.1 fixes:
  - separates VPROPN vessel time from SPICE/route time via --spice-t0-s or --live-state-json;
  - uses finite fail-soft penalties instead of 1e30 walls when VPROPN errors;
  - keeps target epochs tied to the route/patch-conics time anchor, not server t0_game_s.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
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


def raw_to_levela(v: Sequence[float]) -> np.ndarray:
    return RAW_TO_LEVELA @ np.asarray(v, dtype=float)


def levela_to_raw(v: Sequence[float]) -> np.ndarray:
    return LEVELA_TO_RAW @ np.asarray(v, dtype=float)


def norm(v: Sequence[float]) -> float:
    return float(np.linalg.norm(np.asarray(v, dtype=float)))


def unit(v: Sequence[float], fallback: Sequence[float] = (1.0, 0.0, 0.0)) -> np.ndarray:
    a = np.asarray(v, dtype=float)
    n = np.linalg.norm(a)
    if not np.isfinite(n) or n <= 0:
        return np.asarray(fallback, dtype=float)
    return a / n


def read_leg_times(path: Path, leg: int) -> dict[str, float]:
    rows: list[dict[str, str]] = list(csv.DictReader(open(path, newline="")))
    if not rows:
        raise RuntimeError(f"No rows in {path}")

    selected: dict[str, str] | None = None
    for row in rows:
        for key in ("leg", "leg_index", "i", "index"):
            if key in row and row[key] not in ("", None):
                try:
                    val = int(float(row[key]))
                except Exception:
                    continue
                if val == leg or val == leg - 1:
                    selected = row
                    break
        if selected is not None:
            break
    row = selected if selected is not None else rows[leg - 1]

    def get_float(*names: str) -> float:
        for name in names:
            if name in row and row[name] not in ("", None):
                return float(row[name])
        raise KeyError(f"None of {names} in leg row columns={list(row.keys())}")

    return {
        "t_start_s": get_float("t_start_s", "t0_s", "dep_t_s", "departure_time_s", "t_depart_s"),
        "t_end_s": get_float("t_end_s", "t1_s", "arr_t_s", "arrival_time_s", "t_arrive_s"),
    }


def read_time_from_live_state(path: Path) -> float:
    """Read the route/SPICE epoch anchor from a saved live_state JSON.

    Older project files use keys such as ut_s, t_s or et_s. This value is the
    epoch that should correspond physically to VPROPN's canonical vessel t0.
    It is NOT necessarily equal to the server's t0_game_s token.
    """
    data = json.loads(path.read_text())
    for key in ("ut_s", "t_s", "et_s", "time_s", "ut", "time"):
        if key in data and data[key] is not None:
            return float(data[key])
    raise KeyError(f"Could not find a time field in {path}; tried ut_s,t_s,et_s,time_s,ut,time")


def spice_body_state_raw(body: str, et_s: float, center: str = "SUN", frame: str = "J2000") -> tuple[np.ndarray, np.ndarray]:
    # SPICE returns km and km/s in LevelA/J2000 canonical; convert to raw m and raw m/s.
    state, _ = spice.spkezr(body.upper(), float(et_s), frame, "NONE", center.upper())
    r_levela_m = np.asarray(state[:3], dtype=float) * 1000.0
    v_levela_m_s = np.asarray(state[3:], dtype=float) * 1000.0
    return levela_to_raw(r_levela_m), levela_to_raw(v_levela_m_s)


def vector_from_mode(mode: str, initial_r: np.ndarray, initial_v: np.ndarray, dep_r: np.ndarray, dep_v: np.ndarray) -> np.ndarray:
    rel_r = initial_r - dep_r
    rel_v = initial_v - dep_v
    mode = mode.lower()
    if mode in ("prograde", "tangent"):
        radial = unit(rel_r)
        tang = rel_v - float(np.dot(rel_v, radial)) * radial
        return unit(tang, fallback=rel_v)
    if mode in ("velocity", "inertial_velocity"):
        return unit(initial_v)
    if mode in ("raw_x", "x"):
        return np.array([1.0, 0.0, 0.0])
    if mode in ("raw_y", "y"):
        return np.array([0.0, 1.0, 0.0])
    if mode in ("raw_z", "z"):
        return np.array([0.0, 0.0, 1.0])
    raise ValueError(f"Unknown dv0 direction mode: {mode}")


@dataclass
class EvalResult:
    f: float
    g: np.ndarray
    data: dict[str, Any]


class DepartureNLP:
    def __init__(self, cfg: dict[str, Any], client: VesselPropnClient):
        self.cfg = cfg
        self.client = client
        self.cache: dict[tuple[float, ...], EvalResult] = {}
        self.eval_count = 0
        self.last: EvalResult | None = None

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        cfg = self.cfg
        lb = np.array([
            cfg["burn_dt_min_s"],
            -cfg["dv0_component_bound_m_s"],
            -cfg["dv0_component_bound_m_s"],
            -cfg["dv0_component_bound_m_s"],
            cfg["dsm_dt_min_s"],
            -cfg["dsm_component_bound_m_s"],
            -cfg["dsm_component_bound_m_s"],
            -cfg["dsm_component_bound_m_s"],
            cfg["final_dt_min_s"],
        ], dtype=float)
        ub = np.array([
            cfg["burn_dt_max_s"],
            cfg["dv0_component_bound_m_s"],
            cfg["dv0_component_bound_m_s"],
            cfg["dv0_component_bound_m_s"],
            cfg["dsm_dt_max_s"],
            cfg["dsm_component_bound_m_s"],
            cfg["dsm_component_bound_m_s"],
            cfg["dsm_component_bound_m_s"],
            cfg["final_dt_max_s"],
        ], dtype=float)
        return lb, ub

    def constraint_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        cfg = self.cfg
        gl = np.array([
            cfg["min_dsm_after_burn_s"],
            cfg["min_final_after_dsm_s"],
            cfg["dv0_min_m_s"],
            0.0,
            0.0,
        ], dtype=float)
        gu = np.array([
            1.0e12,
            1.0e12,
            cfg["dv0_max_m_s"],
            cfg["dsm_max_m_s"],
            cfg["final_pos_constraint_max_km"],
        ], dtype=float)
        return gl, gu

    def _key(self, x: np.ndarray) -> tuple[float, ...]:
        return tuple(float(f"{v:.12g}") for v in np.asarray(x, dtype=float))

    def evaluate(self, x: np.ndarray) -> EvalResult:
        x = np.asarray(x, dtype=float)
        key = self._key(x)
        if key in self.cache:
            return self.cache[key]

        cfg = self.cfg
        self.eval_count += 1

        burn_dt = float(x[0])
        dv0 = np.asarray(x[1:4], dtype=float)
        dsm_dt = float(x[4])
        dsm = np.asarray(x[5:8], dtype=float)
        final_dt = float(x[8])

        dv0_norm = norm(dv0)
        dsm_norm = norm(dsm)

        # Clamp times used for the server call to avoid undefined server behavior
        # during derivative probing. IPOPT still sees the original x in constraints.
        burn_dt_call = max(0.0, burn_dt)
        dsm_dt_call = max(0.0, dsm_dt)
        final_dt_call = max(max(0.0, final_dt), dsm_dt_call + 1.0)

        # Route/SPICE time and VPROPN vessel time are related by a fixed
        # anchor, but they are not necessarily numerically equal.  The body
        # target must be evaluated at spice_t0_s + final_dt, not at the
        # server's t1_game_s token.
        target_et_s = cfg["spice_t0_s"] + final_dt_call
        target_r, target_v = spice_body_state_raw(cfg["arr_body"], target_et_s, cfg["spice_center"], cfg["spice_frame"])

        request_id = f"ipopt_{os.getpid()}_{self.eval_count}"
        try:
            res = self.client.vpropn(
                request_id,
                cfg["vessel_guid"],
                final_dt_call,
                [
                    Burn(burn_dt_call, dv0.tolist()),
                    Burn(dsm_dt_call, dsm.tolist()),
                ],
                timeout_s=cfg["server_timeout_s"],
            )
            server_ok = True
            server_error = ""
            final_r = np.asarray(res["final_r_raw_m"], dtype=float)
            final_v = np.asarray(res["final_v_raw_m_s"], dtype=float)
            final_pos_err_km = norm(final_r - target_r) / 1000.0
            arrival_vinf_in_m_s = norm(final_v - target_v)
        except Exception as exc:
            # IPOPT cannot deal with a discontinuous 1e30 wall.  Use a finite,
            # deterministic fail-soft residual so gradients remain bounded.
            server_ok = False
            server_error = str(exc)
            fail_pos_km = float(cfg["fail_soft_pos_km"])
            final_r = target_r + np.array([fail_pos_km * 1000.0, 0.0, 0.0])
            final_v = target_v.copy()
            final_pos_err_km = fail_pos_km
            arrival_vinf_in_m_s = cfg["arrival_vinf_target_m_s"]
            res = {
                "final_r_raw_m": final_r.tolist(),
                "final_v_raw_m_s": final_v.tolist(),
                "t0_game_s": cfg.get("t0_game_s", 0.0),
                "t1_game_s": cfg.get("t0_game_s", 0.0) + final_dt_call,
                "burns": [],
                "final_parent_distance_m": float("nan"),
                "final_parent_speed_m_s": float("nan"),
                "final_parent_radial_velocity_m_s": float("nan"),
                "initial_parent_distance_m": float("nan"),
                "initial_parent_speed_m_s": float("nan"),
                "initial_parent_radial_velocity_m_s": float("nan"),
            }
        vinf_mag_mismatch_m_s = abs(arrival_vinf_in_m_s - cfg["arrival_vinf_target_m_s"])

        g = np.array([
            dsm_dt - burn_dt,
            final_dt - dsm_dt,
            dv0_norm,
            dsm_norm,
            final_pos_err_km,
        ], dtype=float)

        f = (
            final_pos_err_km / cfg["pos_scale_km"]
            + cfg["dv_weight"] * ((dv0_norm + dsm_norm) / cfg["dv_scale_m_s"])
            + cfg["dsm_weight"] * (dsm_norm / max(1.0, cfg["dsm_max_m_s"]))
            + cfg["time_weight"] * abs(final_dt - cfg["final_dt_nominal_s"]) / max(1.0, cfg["arrival_trust_s"])
            + cfg["vinf_weight"] * (vinf_mag_mismatch_m_s / cfg["vinf_scale_m_s"])
        )
        if not server_ok:
            f += cfg["server_error_penalty"]

        data = {
            "server_ok": server_ok,
            "server_error": server_error,
            "x": x.tolist(),
            "burn_dt_s": burn_dt,
            "dsm_dt_s": dsm_dt,
            "final_dt_s": final_dt,
            "t0_game_s": res.get("t0_game_s"),
            "t_arr_game_s": res.get("t1_game_s"),
            "t_arr_spice_s": target_et_s,
            "dv0_raw_m_s": dv0.tolist(),
            "dsm_raw_m_s": dsm.tolist(),
            "dv0_norm_m_s": dv0_norm,
            "dsm_norm_m_s": dsm_norm,
            "total_impulsive_dv_m_s": dv0_norm + dsm_norm,
            "final_pos_err_km": final_pos_err_km,
            "arrival_vinf_in_m_s": arrival_vinf_in_m_s,
            "vinf_mag_mismatch_m_s": vinf_mag_mismatch_m_s,
            "final_r_raw_m": final_r.tolist(),
            "final_v_raw_m_s": final_v.tolist(),
            "target_r_raw_m": target_r.tolist(),
            "target_v_raw_m_s": target_v.tolist(),
            "final_parent_distance_m": res.get("final_parent_distance_m"),
            "final_parent_speed_m_s": res.get("final_parent_speed_m_s"),
            "final_parent_radial_velocity_m_s": res.get("final_parent_radial_velocity_m_s"),
            "initial_parent_distance_m": res.get("initial_parent_distance_m"),
            "initial_parent_speed_m_s": res.get("initial_parent_speed_m_s"),
            "initial_parent_radial_velocity_m_s": res.get("initial_parent_radial_velocity_m_s"),
            "burns": res.get("burns", []),
            "raw_server": res,
            "objective": float(f),
            "constraints": g.tolist(),
        }

        ev = EvalResult(float(f), g, data)
        self.cache[key] = ev
        self.last = ev
        return ev

    def objective(self, x: np.ndarray) -> float:
        return self.evaluate(x).f

    def constraints(self, x: np.ndarray) -> np.ndarray:
        return self.evaluate(x).g

    def gradient(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        grad = np.zeros_like(x)
        steps = np.asarray(self.cfg["fd_steps"], dtype=float)
        f0 = self.objective(x)
        lb, ub = self.bounds()
        for i in range(len(x)):
            h = steps[i]
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
        m, n = len(g0), len(x)
        jac = np.zeros((m, n), dtype=float)
        steps = np.asarray(self.cfg["fd_steps"], dtype=float)
        lb, ub = self.bounds()
        for j in range(n):
            h = steps[j]
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
            print(
                f"[ipopt] it={iter_count:4d} obj={obj_value:12.6g} "
                f"inf_pr={inf_pr:9.3g} pos={last.get('final_pos_err_km', float('nan')):12.3f} km "
                f"dv0={last.get('dv0_norm_m_s', float('nan')):8.2f} "
                f"dsm={last.get('dsm_norm_m_s', float('nan')):8.2f} "
                f"evals={self.eval_count}",
                flush=True,
            )
        return True


def build_initial_guess(cfg: dict[str, Any], client: VesselPropnClient) -> np.ndarray:
    smoke = client.vpropn("ipopt_init_state", cfg["vessel_guid"], 1.0, [], timeout_s=cfg["server_timeout_s"])
    cfg["t0_game_s"] = float(smoke["t0_game_s"])

    initial_r = np.asarray(smoke["initial_r_raw_m"], dtype=float)
    initial_v = np.asarray(smoke["initial_v_raw_m_s"], dtype=float)
    dep_r, dep_v = spice_body_state_raw(cfg["dep_body"], cfg["spice_t0_s"], cfg["spice_center"], cfg["spice_frame"])
    dv_dir = vector_from_mode(cfg["initial_dv0_direction"], initial_r, initial_v, dep_r, dep_v)

    burn_dt = cfg["burn_dt_initial_s"]
    dv0 = cfg["dv0_initial_m_s"] * dv_dir
    dsm_dt = cfg["dsm_dt_initial_s"]
    dsm = np.asarray(cfg["dsm_initial_raw_m_s"], dtype=float)
    final_dt = cfg["final_dt_initial_s"]

    x0 = np.array([burn_dt, dv0[0], dv0[1], dv0[2], dsm_dt, dsm[0], dsm[1], dsm[2], final_dt], dtype=float)
    lb, ub = DepartureNLP(cfg, client).bounds()
    return np.minimum(np.maximum(x0, lb), ub)


def load_x0_json(path: Path) -> np.ndarray:
    r = json.loads(path.read_text())
    if "x" in r:
        return np.asarray(r["x"], dtype=float)
    best = r.get("best", r)
    if all(k in best for k in ("burn_dt_s", "dv0_raw_m_s", "dsm_dt_s", "dsm_raw_m_s", "final_dt_s")):
        return np.array([best["burn_dt_s"], *best["dv0_raw_m_s"], best["dsm_dt_s"], *best["dsm_raw_m_s"], best["final_dt_s"]], dtype=float)
    if all(k in best for k in ("tb1_s", "dv0_raw_m_s", "t_dsm_s", "dsm_raw_m_s", "t_arr_s")):
        # Absolute SPICE route times; caller must pass --spice-t0-s/--live-state-json.
        spice_t0 = float(r.get("spice_t0_s", best.get("spice_t0_s", 0.0)))
        return np.array([best["tb1_s"] - spice_t0, *best["dv0_raw_m_s"], best["t_dsm_s"] - spice_t0, *best["dsm_raw_m_s"], best["t_arr_s"] - spice_t0], dtype=float)
    raise RuntimeError(f"Cannot parse x0 from {path}")


def main() -> int:
    ap = argparse.ArgumentParser(description="IPOPT v0 real-vessel departure corrector using VPROPN.")
    ap.add_argument("--server", required=True)
    ap.add_argument("--plugin-b64", type=Path, required=True)
    ap.add_argument("--plugin-arg-mode", choices=["option", "positional", "none"], default="option")
    ap.add_argument("--vessel-guid", required=True)
    ap.add_argument("--live-state-json", type=Path, default=None, help="Optional route/SPICE time anchor; reads ut_s/t_s/et_s.")
    ap.add_argument("--spice-t0-s", type=float, default=None, help="SPICE/route epoch corresponding to VPROPN vessel t0.")

    ap.add_argument("--bsp", type=Path, required=True)
    ap.add_argument("--tpc", type=Path, required=True)
    ap.add_argument("--leg-optimizations", type=Path, required=True)
    ap.add_argument("--leg", type=int, default=1)
    ap.add_argument("--dep-body", default="KERBIN")
    ap.add_argument("--arr-body", default="EVE")
    ap.add_argument("--spice-center", default="SUN")
    ap.add_argument("--spice-frame", default="J2000")

    ap.add_argument("--burn-dt-min-s", type=float, default=0.0)
    ap.add_argument("--burn-dt-max-s", type=float, default=7200.0)
    ap.add_argument("--burn-dt-initial-s", type=float, default=0.0)

    ap.add_argument("--dv0-initial-m-s", type=float, default=2000.0)
    ap.add_argument("--dv0-min-m-s", type=float, default=1400.0)
    ap.add_argument("--dv0-max-m-s", type=float, default=2800.0)
    ap.add_argument("--dv0-component-bound-m-s", type=float, default=4000.0)
    ap.add_argument("--initial-dv0-direction", default="prograde", choices=["prograde", "tangent", "velocity", "raw_x", "raw_y", "raw_z"])

    ap.add_argument("--dsm-dt-initial-days", type=float, default=50.0)
    ap.add_argument("--dsm-dt-min-days", type=float, default=1.0)
    ap.add_argument("--dsm-dt-max-days", type=float, default=120.0)
    ap.add_argument("--dsm-max-m-s", type=float, default=1500.0)
    ap.add_argument("--dsm-component-bound-m-s", type=float, default=2000.0)
    ap.add_argument("--dsm-initial-raw-m-s", default="0,0,0")

    ap.add_argument("--min-dsm-after-burn-s", type=float, default=21600.0)
    ap.add_argument("--min-final-after-dsm-s", type=float, default=86400.0)

    ap.add_argument("--arrival-trust-days", type=float, default=20.0)
    ap.add_argument("--final-dt-initial-days-offset", type=float, default=0.0)

    ap.add_argument("--pos-scale-km", type=float, default=10000.0)
    ap.add_argument("--dv-scale-m-s", type=float, default=1000.0)
    ap.add_argument("--dv-weight", type=float, default=0.005)
    ap.add_argument("--dsm-weight", type=float, default=0.005)
    ap.add_argument("--time-weight", type=float, default=0.001)
    ap.add_argument("--vinf-weight", type=float, default=0.0)
    ap.add_argument("--vinf-scale-m-s", type=float, default=1000.0)
    ap.add_argument("--arrival-vinf-target-m-s", type=float, default=0.0)
    ap.add_argument("--final-pos-constraint-max-km", type=float, default=1.0e9)
    ap.add_argument("--fail-soft-pos-km", type=float, default=1.0e8, help="Finite residual used when VPROPN fails instead of 1e30.")
    ap.add_argument("--server-error-penalty", type=float, default=1.0e4, help="Finite additive objective penalty when VPROPN fails.")

    ap.add_argument("--fd-time-step-s", type=float, default=30.0)
    ap.add_argument("--fd-dv-step-m-s", type=float, default=1.0)

    ap.add_argument("--max-iter", type=int, default=80)
    ap.add_argument("--tol", type=float, default=1e-4)
    ap.add_argument("--acceptable-tol", type=float, default=1e-2)
    ap.add_argument("--print-level", type=int, default=5)
    ap.add_argument("--print-every", type=int, default=1)

    ap.add_argument("--x0-json", type=Path, default=None)
    ap.add_argument("--server-timeout-s", type=float, default=300.0)
    ap.add_argument("--quiet-stderr", action="store_true")
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()

    if cyipopt is None:
        raise SystemExit("[FAIL] cyipopt is not importable. Install cyipopt/Ipopt first. Import error: " + repr(_CYIPOPT_IMPORT_ERROR))

    spice.kclear()
    spice.furnsh(str(args.tpc))
    spice.furnsh(str(args.bsp))

    leg_times = read_leg_times(args.leg_optimizations, args.leg)
    dsm_initial_raw = [float(x.strip()) for x in args.dsm_initial_raw_m_s.replace(";", ",").split(",") if x.strip()]
    if len(dsm_initial_raw) != 3:
        raise SystemExit("--dsm-initial-raw-m-s must be 3 comma-separated floats")

    cfg: dict[str, Any] = {
        "vessel_guid": args.vessel_guid,
        "dep_body": args.dep_body.upper(),
        "arr_body": args.arr_body.upper(),
        "spice_center": args.spice_center.upper(),
        "spice_frame": args.spice_frame,
        "server_timeout_s": args.server_timeout_s,
        "burn_dt_min_s": args.burn_dt_min_s,
        "burn_dt_max_s": args.burn_dt_max_s,
        "burn_dt_initial_s": args.burn_dt_initial_s,
        "dv0_initial_m_s": args.dv0_initial_m_s,
        "dv0_min_m_s": args.dv0_min_m_s,
        "dv0_max_m_s": args.dv0_max_m_s,
        "dv0_component_bound_m_s": args.dv0_component_bound_m_s,
        "initial_dv0_direction": args.initial_dv0_direction,
        "dsm_dt_initial_s": args.dsm_dt_initial_days * 86400.0,
        "dsm_dt_min_s": args.dsm_dt_min_days * 86400.0,
        "dsm_dt_max_s": args.dsm_dt_max_days * 86400.0,
        "dsm_max_m_s": args.dsm_max_m_s,
        "dsm_component_bound_m_s": args.dsm_component_bound_m_s,
        "dsm_initial_raw_m_s": dsm_initial_raw,
        "min_dsm_after_burn_s": args.min_dsm_after_burn_s,
        "min_final_after_dsm_s": args.min_final_after_dsm_s,
        "pos_scale_km": args.pos_scale_km,
        "dv_scale_m_s": args.dv_scale_m_s,
        "dv_weight": args.dv_weight,
        "dsm_weight": args.dsm_weight,
        "time_weight": args.time_weight,
        "vinf_weight": args.vinf_weight,
        "vinf_scale_m_s": args.vinf_scale_m_s,
        "arrival_vinf_target_m_s": args.arrival_vinf_target_m_s,
        "final_pos_constraint_max_km": args.final_pos_constraint_max_km,
        "server_error_penalty": args.server_error_penalty,
        "fail_soft_pos_km": args.fail_soft_pos_km,
        "fd_steps": [args.fd_time_step_s, args.fd_dv_step_m_s, args.fd_dv_step_m_s, args.fd_dv_step_m_s, args.fd_time_step_s, args.fd_dv_step_m_s, args.fd_dv_step_m_s, args.fd_dv_step_m_s, args.fd_time_step_s],
        "arrival_trust_s": args.arrival_trust_days * 86400.0,
        "print_every": args.print_every,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)

    with VesselPropnClient(
        args.server,
        args.plugin_b64,
        response_timeout_s=args.server_timeout_s,
        quiet_stderr=args.quiet_stderr,
        plugin_arg_mode=args.plugin_arg_mode,
    ) as client:
        smoke = client.vpropn("ipopt_t0_probe", args.vessel_guid, 1.0, [], timeout_s=args.server_timeout_s)
        t0_game_s = float(smoke["t0_game_s"])
        cfg["t0_game_s"] = t0_game_s
        if args.spice_t0_s is not None:
            spice_t0_s = float(args.spice_t0_s)
        elif args.live_state_json is not None:
            spice_t0_s = read_time_from_live_state(args.live_state_json)
        else:
            spice_t0_s = t0_game_s
        cfg["spice_t0_s"] = spice_t0_s
        cfg["time_anchor_offset_s"] = spice_t0_s - t0_game_s

        final_dt_nominal_s = float(leg_times["t_end_s"] - spice_t0_s)
        cfg["final_dt_nominal_s"] = final_dt_nominal_s
        cfg["final_dt_initial_s"] = final_dt_nominal_s + args.final_dt_initial_days_offset * 86400.0
        cfg["final_dt_min_s"] = final_dt_nominal_s - args.arrival_trust_days * 86400.0
        cfg["final_dt_max_s"] = final_dt_nominal_s + args.arrival_trust_days * 86400.0
        cfg["final_dt_min_s"] = max(cfg["final_dt_min_s"], cfg["dsm_dt_min_s"] + args.min_final_after_dsm_s)
        cfg["final_dt_max_s"] = max(cfg["final_dt_max_s"], cfg["final_dt_min_s"] + 86400.0)

        print("=== DEPARTURE IPOPT V0.1 / VPROPN ===")
        print(f"server         : {args.server}")
        print(f"vessel_guid    : {args.vessel_guid}")
        print(f"t0_game_s      : {t0_game_s}")
        print(f"spice_t0_s     : {spice_t0_s}")
        print(f"anchor_offset : {cfg['time_anchor_offset_s']}")
        print(f"leg t_end_s    : {leg_times['t_end_s']}")
        print(f"final_dt_nom   : {final_dt_nominal_s}")
        print(f"burn_dt bounds : {cfg['burn_dt_min_s']} .. {cfg['burn_dt_max_s']}")
        print(f"dsm_dt bounds  : {cfg['dsm_dt_min_s']} .. {cfg['dsm_dt_max_s']}")
        print(f"final bounds   : {cfg['final_dt_min_s']} .. {cfg['final_dt_max_s']}")
        print(f"output_dir     : {args.output_dir}")

        nlp = DepartureNLP(cfg, client)
        lb, ub = nlp.bounds()
        gl, gu = nlp.constraint_bounds()

        if args.x0_json is not None:
            x0 = load_x0_json(args.x0_json)
            x0 = np.minimum(np.maximum(x0, lb), ub)
        else:
            x0 = build_initial_guess(cfg, client)

        print("x0:", json.dumps(x0.tolist()))
        x0_eval = nlp.evaluate(x0)
        print("x0_eval:", json.dumps({
            "objective": x0_eval.f,
            "final_pos_err_km": x0_eval.data["final_pos_err_km"],
            "dv0_norm_m_s": x0_eval.data["dv0_norm_m_s"],
            "dsm_norm_m_s": x0_eval.data["dsm_norm_m_s"],
            "arrival_vinf_in_m_s": x0_eval.data["arrival_vinf_in_m_s"],
        }, indent=2))

        problem = cyipopt.Problem(n=9, m=5, problem_obj=nlp, lb=lb, ub=ub, cl=gl, cu=gu)
        problem.add_option("max_iter", int(args.max_iter))
        problem.add_option("tol", float(args.tol))
        problem.add_option("acceptable_tol", float(args.acceptable_tol))
        problem.add_option("print_level", int(args.print_level))
        problem.add_option("mu_strategy", "adaptive")
        problem.add_option("nlp_scaling_method", "gradient-based")
        problem.add_option("hessian_approximation", "limited-memory")

        x_opt, info = problem.solve(x0)
        best = nlp.evaluate(np.asarray(x_opt, dtype=float)).data

    out = {
        "status": "ok",
        "ipopt_info": {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in dict(info).items()},
        "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in cfg.items() if k != "fd_steps"},
        "x0": x0.tolist(),
        "x0_eval": x0_eval.data,
        "best": best,
        "champion_f": best["objective"],
        "best_fitness": best["objective"],
        "arrival_vinf_in_m_s": best["arrival_vinf_in_m_s"],
        "final_v_raw_m_s": best["final_v_raw_m_s"],
        "target_v_raw_m_s": best["target_v_raw_m_s"],
        "dv0_norm_m_s": best["dv0_norm_m_s"],
        "vinf_mag_mismatch_m_s": best["vinf_mag_mismatch_m_s"],
        "n_evaluations": nlp.eval_count,
    }

    result_path = args.output_dir / "departure_ipopt_v0_1_result.json"
    result_path.write_text(json.dumps(out, indent=2, default=lambda x: x.decode('utf-8') if isinstance(x, bytes) else str(x)) + "\n")

    summary_path = args.output_dir / "departure_ipopt_v0_1_summary.txt"
    summary_path.write_text("\n".join([
        "=== DEPARTURE IPOPT V0.1 SUMMARY ===",
        f"objective              : {out['champion_f']}",
        f"final_pos_err_km       : {best['final_pos_err_km']}",
        f"arrival_vinf_in_m_s    : {best['arrival_vinf_in_m_s']}",
        f"dv0_norm_m_s           : {best['dv0_norm_m_s']}",
        f"dsm_norm_m_s           : {best['dsm_norm_m_s']}",
        f"total_impulsive_dv_m_s : {best['total_impulsive_dv_m_s']}",
        f"burn_dt_s              : {best['burn_dt_s']}",
        f"dsm_dt_s               : {best['dsm_dt_s']}",
        f"final_dt_s             : {best['final_dt_s']}",
        f"n_evaluations          : {nlp.eval_count}",
        "",
    ]))

    print("=== RESULT ===")
    print(json.dumps({
        "champion_f": out["champion_f"],
        "final_pos_err_km": best["final_pos_err_km"],
        "arrival_vinf_in_m_s": best["arrival_vinf_in_m_s"],
        "dv0_norm_m_s": best["dv0_norm_m_s"],
        "dsm_norm_m_s": best["dsm_norm_m_s"],
        "burn_dt_s": best["burn_dt_s"],
        "dsm_dt_s": best["dsm_dt_s"],
        "final_dt_s": best["final_dt_s"],
        "n_evaluations": nlp.eval_count,
    }, indent=2))
    print(f"[OK] wrote {result_path}")
    print(f"[OK] wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
