#!/usr/bin/env python3
"""
First VCA-based N-body leg targeter for KSP-MGA-Planner + Principia.

This script intentionally does NOT try to match a fixed 6D patchpoint.
It minimizes closest approach to the arrival body using the new VCA command.

Decision vector:
  x = [
    burn_dt_s,
    dv0_tangent_m_s, dv0_normal_m_s, dv0_binormal_m_s,
    dsm_fraction_unbounded,
    dsmx_raw_m_s, dsmy_raw_m_s, dsmz_raw_m_s,
    arrival_scan_center_dt_s,
  ]

Burn0 is parameterized directly in the FlightPlan T/N/B basis, then converted
to Principia raw inertial m/s for VCA evaluation.

DSM time is derived safely:
  dsm_low  = burn_dt + min_dsm_after_burn_s
  dsm_high = arrival_scan_center_dt - min_scan_after_dsm_s
  dsm_dt   = dsm_low + sigmoid(dsm_fraction_unbounded) * (dsm_high - dsm_low)

Evaluation:
  VCA id vessel_guid arr_body scan_start scan_end samples [burn0, dsm]

Objective:
  ca_distance_m / distance_scale_m
  + dv_weight * (||dv0|| + ||dsm||) / dv_scale_m_s
  + soft penalties for invalid ordering, dv norms, scan boundary hits

For a first run, use --method powell or --method ipopt. Powell is often more
robust for this non-smooth closest-approach objective; IPOPT is available for
local refinement.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import time
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from principia_targeter_client import PrincipiaTargeterClient, PrincipiaServerError

try:
    import scipy.optimize as spo
except Exception as exc:
    spo = None
    _SCIPY_IMPORT_ERROR = exc
else:
    _SCIPY_IMPORT_ERROR = None

try:
    import cyipopt
except Exception as exc:
    cyipopt = None
    _CYIPOPT_IMPORT_ERROR = exc
else:
    _CYIPOPT_IMPORT_ERROR = None


def norm(v: Sequence[float]) -> float:
    return float(np.linalg.norm(np.asarray(v, dtype=float)))


def unit(v: Sequence[float], fallback: Sequence[float] = (1.0, 0.0, 0.0)) -> np.ndarray:
    a = np.asarray(v, dtype=float)
    n = np.linalg.norm(a)
    if not np.isfinite(n) or n <= 0:
        return np.asarray(fallback, dtype=float)
    return a / n


def tnb_basis_from_rel_state(r_raw_m: Sequence[float], v_raw_m_s: Sequence[float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Principia FlightPlan-style local TNB basis in raw coordinates.

    T = tangent/prograde, along relative velocity.
    B = binormal/out-of-plane, approximately orbital angular momentum.
    N = normal/curvature direction, B x T. For a circular orbit this points
        roughly inward to the body; negative N is outward.

    raw_delta_v = T*dv_t + N*dv_n + B*dv_b
    """
    r = np.asarray(r_raw_m, dtype=float)
    v = np.asarray(v_raw_m_s, dtype=float)
    T = unit(v)
    B = unit(np.cross(r, v), fallback=(0.0, 0.0, 1.0))
    N = unit(np.cross(B, T), fallback=-unit(r))
    return T, N, B


def tnb_to_raw(dv_tnb_m_s: Sequence[float], r_raw_m: Sequence[float], v_raw_m_s: Sequence[float]) -> np.ndarray:
    T, N, B = tnb_basis_from_rel_state(r_raw_m, v_raw_m_s)
    t, n, b = np.asarray(dv_tnb_m_s, dtype=float)
    return T * t + N * n + B * b


def raw_to_tnb(dv_raw_m_s: Sequence[float], r_raw_m: Sequence[float], v_raw_m_s: Sequence[float]) -> np.ndarray:
    T, N, B = tnb_basis_from_rel_state(r_raw_m, v_raw_m_s)
    dv = np.asarray(dv_raw_m_s, dtype=float)
    return np.array([float(np.dot(dv, T)), float(np.dot(dv, N)), float(np.dot(dv, B))], dtype=float)


def sigmoid(u: float) -> float:
    # Stable enough for our bounded u range.
    if u >= 0:
        z = math.exp(-u)
        return 1.0 / (1.0 + z)
    z = math.exp(u)
    return z / (1.0 + z)


def read_time_json(path: Path) -> float:
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


def frow(row: dict[str, str], key: str, default: float | None = None) -> float:
    if key in row and row[key] not in ("", None):
        return float(row[key])
    if default is not None:
        return float(default)
    raise KeyError(f"Missing CSV field {key!r}")


def parse_vec3_csv(s: str) -> list[float]:
    vals = [float(x.strip()) for x in s.replace(";", ",").split(",") if x.strip()]
    if len(vals) != 3:
        raise argparse.ArgumentTypeError("expected 3 comma-separated floats")
    return vals


def x_from_seed_json(seed: Path, arrival_center_dt_s: float, dsm_fraction_u: float) -> np.ndarray:
    r = json.loads(seed.read_text())
    if "x_vca" in r:
        return np.asarray(r["x_vca"], dtype=float)

    best = r.get("best", r)
    if "burn_dt_s" not in best or "dv0_raw_m_s" not in best:
        raise RuntimeError(f"Seed {seed} does not contain best.burn_dt_s and best.dv0_raw_m_s")

    burn_dt = float(best["burn_dt_s"])
    dv0 = [float(x) for x in best["dv0_raw_m_s"]]
    dsm = [float(x) for x in best.get("dsm_raw_m_s", [0.0, 0.0, 0.0])]
    return np.array([burn_dt, *dv0, dsm_fraction_u, *dsm, arrival_center_dt_s], dtype=float)


@dataclass
class EvalResult:
    objective: float
    data: dict[str, Any]


class StopOptimization(RuntimeError):
    """Raised internally when a target CA, max evals, max time, or stall limit is reached."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


class VCATargetProblem:
    def __init__(self, cfg: dict[str, Any], client: PrincipiaTargeterClient):
        self.cfg = cfg
        self.client = client
        self.eval_count = 0
        self.cache: dict[tuple[float, ...], EvalResult] = {}
        self.best: EvalResult | None = None
        self.best_distance: EvalResult | None = None
        self.start_wall_time_s = time.monotonic()
        self.best_distance_eval_count = 0
        self.stop_reason: str | None = None

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        c = self.cfg
        lb = np.array([
            c["burn_dt_min_s"],
            c["dv0_t_min_m_s"],
            -c["dv0_n_max_abs_m_s"],
            -c["dv0_b_max_abs_m_s"],
            c["dsm_fraction_min"],
            -c["dsm_component_bound_m_s"],
            -c["dsm_component_bound_m_s"],
            -c["dsm_component_bound_m_s"],
            c["arrival_center_min_dt_s"],
        ], dtype=float)
        ub = np.array([
            c["burn_dt_max_s"],
            c["dv0_t_max_m_s"],
            c["dv0_n_max_abs_m_s"],
            c["dv0_b_max_abs_m_s"],
            c["dsm_fraction_max"],
            c["dsm_component_bound_m_s"],
            c["dsm_component_bound_m_s"],
            c["dsm_component_bound_m_s"],
            c["arrival_center_max_dt_s"],
        ], dtype=float)
        return lb, ub

    def _key(self, x: np.ndarray) -> tuple[float, ...]:
        return tuple(float(f"{v:.10g}") for v in np.asarray(x, dtype=float))

    def decode(self, x: np.ndarray) -> dict[str, Any]:
        c = self.cfg
        x = np.asarray(x, dtype=float)
        burn_dt = float(x[0])
        dv0_tnb = np.asarray(x[1:4], dtype=float)
        frac_u = float(x[4])
        dsm = np.asarray(x[5:8], dtype=float)
        arrival_center = float(x[8])

        dsm_low = burn_dt + c["min_dsm_after_burn_s"]
        dsm_high = arrival_center - c["min_scan_after_dsm_s"]
        if dsm_high > dsm_low:
            frac = sigmoid(frac_u)
            dsm_dt = dsm_low + frac * (dsm_high - dsm_low)
            ordering_ok = True
            ordering_violation_s = 0.0
        else:
            frac = sigmoid(frac_u)
            dsm_dt = dsm_low
            ordering_ok = False
            ordering_violation_s = dsm_low - dsm_high

        scan_start = arrival_center - c["scan_half_width_s"]
        scan_end = arrival_center + c["scan_half_width_s"]

        # Ensure scan begins after DSM by at least a tiny numerical margin.
        min_scan_start = dsm_dt + c["min_scan_after_dsm_s"]
        scan_order_violation_s = max(0.0, min_scan_start - scan_start)
        if scan_start < min_scan_start:
            scan_start = min_scan_start
            scan_end = max(scan_end, scan_start + max(1.0, c["scan_min_width_s"]))

        return {
            "burn_dt_s": burn_dt,
            "dv0_tnb_m_s": dv0_tnb,
            "dsm_fraction_u": frac_u,
            "dsm_fraction": frac,
            "dsm_dt_s": float(dsm_dt),
            "dsm_raw_m_s": dsm,
            "arrival_center_dt_s": arrival_center,
            "scan_start_dt_s": float(scan_start),
            "scan_end_dt_s": float(scan_end),
            "ordering_ok": ordering_ok,
            "ordering_violation_s": ordering_violation_s,
            "scan_order_violation_s": scan_order_violation_s,
        }

    def _maybe_stop(self) -> None:
        c = self.cfg
        if self.best_distance is None:
            return

        best_ca_km = self.best_distance.data["ca_distance_km"]
        if self.eval_count >= c["min_evals_before_stop"]:
            if best_ca_km <= c["target_ca_km"]:
                self.stop_reason = f"target_ca_reached: {best_ca_km:.6f} km <= {c['target_ca_km']:.6f} km"
                raise StopOptimization(self.stop_reason)

        if c["max_evals"] > 0 and self.eval_count >= c["max_evals"]:
            self.stop_reason = f"max_evals_reached: {self.eval_count} >= {c['max_evals']}"
            raise StopOptimization(self.stop_reason)

        if c["max_time_s"] > 0:
            elapsed = time.monotonic() - self.start_wall_time_s
            if elapsed >= c["max_time_s"]:
                self.stop_reason = f"max_time_reached: {elapsed:.3f}s >= {c['max_time_s']:.3f}s"
                raise StopOptimization(self.stop_reason)

        if c["stall_evals"] > 0:
            since = self.eval_count - self.best_distance_eval_count
            if since >= c["stall_evals"]:
                self.stop_reason = (
                    f"stall_evals_reached: no new best distance for {since} evals; "
                    f"best_ca={best_ca_km:.6f} km"
                )
                raise StopOptimization(self.stop_reason)

    def evaluate(self, x: np.ndarray) -> EvalResult:
        x = np.asarray(x, dtype=float)
        key = self._key(x)
        if key in self.cache:
            return self.cache[key]

        c = self.cfg
        self.eval_count += 1
        d = self.decode(x)
        dv0_tnb = np.asarray(d["dv0_tnb_m_s"], dtype=float)
        dsm = np.asarray(d["dsm_raw_m_s"], dtype=float)

        basis_ok = True
        basis_error = ""
        try:
            basis_state = self.client.vrel(
                f"basis_{os.getpid()}_{self.eval_count}",
                c["vessel_guid"],
                c["dep_body"],
                d["burn_dt_s"],
                [],
                timeout_s=c["server_timeout_s"],
            )
            burn_rel_r = np.asarray(basis_state["final_rel_r_raw_m"], dtype=float)
            burn_rel_v = np.asarray(basis_state["final_rel_v_raw_m_s"], dtype=float)
            dv0 = tnb_to_raw(dv0_tnb, burn_rel_r, burn_rel_v)
            tnb_basis = tnb_basis_from_rel_state(burn_rel_r, burn_rel_v)
        except Exception as exc:
            basis_ok = False
            basis_error = str(exc)
            basis_state = {}
            burn_rel_r = np.array([float("nan"), float("nan"), float("nan")])
            burn_rel_v = np.array([float("nan"), float("nan"), float("nan")])
            dv0 = np.array([float("nan"), float("nan"), float("nan")])
            tnb_basis = (np.zeros(3), np.zeros(3), np.zeros(3))

        dv0_norm = norm(dv0_tnb)
        dsm_norm = norm(dsm)

        penalty = 0.0
        if not d["ordering_ok"]:
            penalty += c["ordering_penalty"] * (d["ordering_violation_s"] / c["time_penalty_scale_s"]) ** 2
        if d["scan_order_violation_s"] > 0:
            penalty += c["ordering_penalty"] * (d["scan_order_violation_s"] / c["time_penalty_scale_s"]) ** 2

        # Soft norm penalties, because component bounds are not enough.
        if dv0_norm < c["dv0_min_m_s"]:
            penalty += c["dv_norm_penalty"] * ((c["dv0_min_m_s"] - dv0_norm) / c["dv_scale_m_s"]) ** 2
        if dv0_norm > c["dv0_max_m_s"]:
            penalty += c["dv_norm_penalty"] * ((dv0_norm - c["dv0_max_m_s"]) / c["dv_scale_m_s"]) ** 2
        if dsm_norm > c["dsm_max_m_s"]:
            penalty += c["dv_norm_penalty"] * ((dsm_norm - c["dsm_max_m_s"]) / c["dv_scale_m_s"]) ** 2

        burns = [
            (d["burn_dt_s"], float(dv0[0]), float(dv0[1]), float(dv0[2])),
            (d["dsm_dt_s"], float(dsm[0]), float(dsm[1]), float(dsm[2])),
        ]

        if not basis_ok:
            penalty += c["server_error_penalty"]

        server_ok = True
        server_error = ""
        try:
            if not basis_ok:
                raise RuntimeError(f"basis error: {basis_error}")
            vca = self.client.vca(
                f"vca_opt_{os.getpid()}_{self.eval_count}",
                c["vessel_guid"],
                c["arr_body"],
                d["scan_start_dt_s"],
                d["scan_end_dt_s"],
                c["vca_samples"],
                burns,
                timeout_s=c["server_timeout_s"],
            )
            ca_distance_m = float(vca["ca_distance_m"])
            ca_speed_m_s = float(vca["ca_speed_m_s"])
            ca_dt_s = float(vca["ca_dt_s"])
            ca_radial_v_m_s = float(vca["ca_radial_v_m_s"])
            status = vca.get("status", "")
        except Exception as exc:
            server_ok = False
            server_error = str(exc)
            vca = {}
            ca_distance_m = c["fail_distance_m"]
            ca_speed_m_s = c["fail_speed_m_s"]
            ca_dt_s = float("nan")
            ca_radial_v_m_s = float("nan")
            status = "server_error"
            penalty += c["server_error_penalty"]

        # Penalize if closest approach is on scan boundary; likely center/half-width is wrong.
        boundary_penalty = 0.0
        if server_ok and math.isfinite(ca_dt_s):
            span = max(1.0, d["scan_end_dt_s"] - d["scan_start_dt_s"])
            margin = min(ca_dt_s - d["scan_start_dt_s"], d["scan_end_dt_s"] - ca_dt_s)
            if margin < c["scan_boundary_margin_fraction"] * span:
                boundary_penalty = c["scan_boundary_penalty"] * (
                    (c["scan_boundary_margin_fraction"] * span - margin) / span
                ) ** 2

        total_dv = dv0_norm + dsm_norm
        objective = (
            ca_distance_m / c["distance_scale_m"]
            + c["dv_weight"] * total_dv / c["dv_scale_m_s"]
            + c["speed_weight"] * ca_speed_m_s / c["speed_scale_m_s"]
            + boundary_penalty
            + penalty
        )

        data = {
            "x": x.tolist(),
            **{k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in d.items()},
            "dv0_tnb_m_s": dv0_tnb.tolist(),
            "dv0_raw_m_s": dv0.tolist(),
            "dv0_norm_m_s": dv0_norm,
            "basis_ok": basis_ok,
            "basis_error": basis_error,
            "burn_rel_r_raw_m": burn_rel_r.tolist(),
            "burn_rel_v_raw_m_s": burn_rel_v.tolist(),
            "tnb_basis_raw": {
                "T": tnb_basis[0].tolist(),
                "N": tnb_basis[1].tolist(),
                "B": tnb_basis[2].tolist(),
            },
            "dsm_norm_m_s": dsm_norm,
            "total_impulsive_dv_m_s": total_dv,
            "server_ok": server_ok,
            "server_error": server_error,
            "ca_distance_m": ca_distance_m,
            "ca_distance_km": ca_distance_m / 1000.0,
            "ca_speed_m_s": ca_speed_m_s,
            "ca_radial_v_m_s": ca_radial_v_m_s,
            "ca_dt_s": ca_dt_s,
            "ca_t_game_s": vca.get("ca_t_game_s"),
            "ca_rel_r_raw_m": vca.get("ca_rel_r_raw_m"),
            "ca_rel_v_raw_m_s": vca.get("ca_rel_v_raw_m_s"),
            "vca_status": status,
            "scan_boundary_penalty": boundary_penalty,
            "soft_penalty": penalty,
            "objective": objective,
            "raw_vca": vca,
        }

        ev = EvalResult(float(objective), data)
        self.cache[key] = ev
        if server_ok:
            if self.best is None or ev.objective < self.best.objective:
                self.best = ev
            if self.best_distance is None or ca_distance_m < self.best_distance.data["ca_distance_m"]:
                self.best_distance = ev
                self.best_distance_eval_count = self.eval_count

        if c["print_every_eval"] and self.eval_count % c["print_every_eval"] == 0:
            print(
                f"[eval {self.eval_count:5d}] J={objective:12.6g} "
                f"ca={ca_distance_m/1000:12.3f} km dv={total_dv:8.2f} "
                f"burn={d['burn_dt_s']:9.1f} dsm={d['dsm_dt_s']:12.1f} "
                f"arr={d['arrival_center_dt_s']:12.1f} ok={server_ok}",
                flush=True,
            )

        # Stop only after the current evaluation has been cached and best states were updated.
        self._maybe_stop()
        return ev

    def objective(self, x: np.ndarray) -> float:
        return self.evaluate(x).objective

    def gradient(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        lb, ub = self.bounds()
        steps = np.asarray(self.cfg["fd_steps"], dtype=float)
        grad = np.zeros_like(x)
        f0 = self.objective(x)
        for i, h in enumerate(steps):
            xp = x.copy()
            xm = x.copy()
            xp[i] = min(ub[i], x[i] + h)
            xm[i] = max(lb[i], x[i] - h)
            if xp[i] == xm[i]:
                grad[i] = 0.0
            elif xp[i] != x[i] and xm[i] != x[i]:
                grad[i] = (self.objective(xp) - self.objective(xm)) / (xp[i] - xm[i])
            elif xp[i] != x[i]:
                grad[i] = (self.objective(xp) - f0) / (xp[i] - x[i])
            else:
                grad[i] = (f0 - self.objective(xm)) / (x[i] - xm[i])
        return grad


class CyIpoptWrapper:
    def __init__(self, problem: VCATargetProblem):
        self.problem = problem

    def objective(self, x):
        return self.problem.objective(np.asarray(x, dtype=float))

    def gradient(self, x):
        return self.problem.gradient(np.asarray(x, dtype=float))

    def intermediate(self, alg_mod, iter_count, obj_value, inf_pr, inf_du, mu, d_norm, regularization_size, alpha_du, alpha_pr, ls_trials):
        best = self.problem.best.data if self.problem.best else {}
        print(
            f"[ipopt] it={iter_count:4d} obj={obj_value:12.6g} "
            f"best_ca={best.get('ca_distance_km', float('nan')):12.3f} km "
            f"best_dv={best.get('total_impulsive_dv_m_s', float('nan')):8.2f} "
            f"evals={self.problem.eval_count}",
            flush=True,
        )
        return True


def run_ipopt(problem: VCATargetProblem, x0: np.ndarray, max_iter: int, print_level: int, tol: float) -> tuple[np.ndarray, dict[str, Any]]:
    if cyipopt is None:
        raise RuntimeError(f"cyipopt is not importable: {_CYIPOPT_IMPORT_ERROR}")
    lb, ub = problem.bounds()
    nlp = cyipopt.Problem(
        n=len(x0),
        m=0,
        problem_obj=CyIpoptWrapper(problem),
        lb=lb,
        ub=ub,
        cl=np.array([], dtype=float),
        cu=np.array([], dtype=float),
    )
    nlp.add_option("max_iter", int(max_iter))
    nlp.add_option("tol", float(tol))
    nlp.add_option("print_level", int(print_level))
    nlp.add_option("mu_strategy", "adaptive")
    nlp.add_option("hessian_approximation", "limited-memory")
    x_opt, info = nlp.solve(x0)
    return np.asarray(x_opt, dtype=float), {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in dict(info).items()}


def run_scipy(problem: VCATargetProblem, x0: np.ndarray, method: str, max_iter: int, max_evals: int) -> tuple[np.ndarray, dict[str, Any]]:
    if spo is None:
        raise RuntimeError(f"scipy is not importable: {_SCIPY_IMPORT_ERROR}")
    lb, ub = problem.bounds()
    bounds = list(zip(lb.tolist(), ub.tolist()))
    opt: dict[str, Any] = {"maxiter": int(max_iter), "disp": True}
    if max_evals > 0:
        if method.lower() in ("powell", "nelder-mead"):
            opt["maxfev"] = int(max_evals)
        elif method.lower() == "l-bfgs-b":
            opt["maxfun"] = int(max_evals)
    res = spo.minimize(
        lambda z: problem.objective(np.asarray(z, dtype=float)),
        x0,
        method=method,
        bounds=bounds if method.lower() in ("powell", "nelder-mead", "l-bfgs-b") else None,
        options=opt,
    )
    return np.asarray(res.x, dtype=float), {
        "success": bool(res.success),
        "status": int(res.status) if hasattr(res, "status") else None,
        "message": str(res.message),
        "fun": float(res.fun),
        "nit": int(res.nit) if hasattr(res, "nit") else None,
        "nfev": int(res.nfev) if hasattr(res, "nfev") else None,
    }


def random_multistart(problem: VCATargetProblem, x0: np.ndarray, n: int, seed: int, spread: dict[str, float]) -> np.ndarray:
    if n <= 0:
        problem.evaluate(x0)
        return x0
    rng = random.Random(seed)
    lb, ub = problem.bounds()

    candidates = [np.asarray(x0, dtype=float)]
    for _ in range(n):
        z = np.asarray(x0, dtype=float).copy()
        z[0] += rng.uniform(-spread["burn_dt"], spread["burn_dt"])
        z[1:4] += np.array([rng.uniform(-spread["dv0"], spread["dv0"]) for _ in range(3)])
        z[4] += rng.uniform(-spread["frac"], spread["frac"])
        z[5:8] += np.array([rng.uniform(-spread["dsm"], spread["dsm"]) for _ in range(3)])
        z[8] += rng.uniform(-spread["arrival"], spread["arrival"])
        candidates.append(np.minimum(np.maximum(z, lb), ub))

    best_x = candidates[0]
    best_f = float("inf")
    for z in candidates:
        f = problem.objective(z)
        if f < best_f:
            best_f = f
            best_x = z
    print(f"[multistart] best initial objective={best_f:.6g} from {len(candidates)} candidates")
    return best_x


def main() -> int:
    ap = argparse.ArgumentParser(description="VCA-based leg targeter with TNB departure burn using Principia VCA.")
    ap.add_argument("--server", required=True)
    ap.add_argument("--plugin-b64", type=Path, required=True)
    ap.add_argument("--plugin-arg-mode", choices=["option", "positional"], default="positional")
    ap.add_argument("--vessel-guid", required=True)

    ap.add_argument("--live-state-json", type=Path, required=True)
    ap.add_argument("--leg-optimizations", type=Path, required=True)
    ap.add_argument("--leg", type=int, default=1)
    ap.add_argument("--arr-body", default=None)
    ap.add_argument("--dep-body", default="KERBIN")

    ap.add_argument("--dv0-initial-tnb-m-s", type=parse_vec3_csv, default=None,
                    help="Initial burn0 in FlightPlan T,N,B components: tangent, normal, binormal.")
    ap.add_argument("--dv0-t-min-m-s", type=float, default=1000.0)
    ap.add_argument("--dv0-t-max-m-s", type=float, default=5000.0)
    ap.add_argument("--dv0-n-max-abs-m-s", type=float, default=500.0)
    ap.add_argument("--dv0-b-max-abs-m-s", type=float, default=1000.0)

    ap.add_argument("--x0-json", type=Path, default=None)
    ap.add_argument("--dv0-initial-raw-m-s", type=parse_vec3_csv, default=None)
    ap.add_argument("--dsm-initial-raw-m-s", type=parse_vec3_csv, default=[0.0, 0.0, 0.0])

    ap.add_argument("--burn-dt-initial-s", type=float, default=None)
    ap.add_argument("--burn-dt-min-s", type=float, default=0.0)
    ap.add_argument("--burn-dt-max-s", type=float, default=12000.0)

    ap.add_argument("--dv0-component-bound-m-s", type=float, default=6000.0)
    ap.add_argument("--dv0-min-m-s", type=float, default=0.0)
    ap.add_argument("--dv0-max-m-s", type=float, default=6000.0)

    ap.add_argument("--dsm-component-bound-m-s", type=float, default=2500.0)
    ap.add_argument("--dsm-max-m-s", type=float, default=2500.0)
    ap.add_argument("--dsm-fraction-u-initial", type=float, default=0.0)
    ap.add_argument("--dsm-fraction-min", type=float, default=-8.0)
    ap.add_argument("--dsm-fraction-max", type=float, default=8.0)
    ap.add_argument("--min-dsm-after-burn-s", type=float, default=21600.0)
    ap.add_argument("--min-scan-after-dsm-s", type=float, default=86400.0)

    ap.add_argument("--arrival-center-initial-s", type=float, default=None)
    ap.add_argument("--arrival-trust-days", type=float, default=30.0)
    ap.add_argument("--scan-half-width-days", type=float, default=10.0)
    ap.add_argument("--scan-min-width-s", type=float, default=3600.0)
    ap.add_argument("--vca-samples", type=int, default=41)

    ap.add_argument("--distance-scale-m", type=float, default=1.0e7)
    ap.add_argument("--dv-scale-m-s", type=float, default=1000.0)
    ap.add_argument("--dv-weight", type=float, default=0.03)
    ap.add_argument("--speed-scale-m-s", type=float, default=10000.0)
    ap.add_argument("--speed-weight", type=float, default=0.0)
    ap.add_argument("--ordering-penalty", type=float, default=1000.0)
    ap.add_argument("--time-penalty-scale-s", type=float, default=86400.0)
    ap.add_argument("--dv-norm-penalty", type=float, default=100.0)
    ap.add_argument("--scan-boundary-margin-fraction", type=float, default=0.05)
    ap.add_argument("--scan-boundary-penalty", type=float, default=10.0)
    ap.add_argument("--fail-distance-m", type=float, default=1.0e12)
    ap.add_argument("--fail-speed-m-s", type=float, default=1.0e6)
    ap.add_argument("--server-error-penalty", type=float, default=1.0e6)

    ap.add_argument("--fd-time-step-s", type=float, default=60.0)
    ap.add_argument("--fd-dv-step-m-s", type=float, default=2.0)
    ap.add_argument("--fd-frac-step", type=float, default=0.05)

    ap.add_argument("--method", choices=["powell", "nelder-mead", "l-bfgs-b", "ipopt"], default="powell")
    ap.add_argument("--max-iter", type=int, default=120)
    ap.add_argument("--max-evals", type=int, default=0,
                    help="Hard evaluation limit. 0 disables. For scipy this is also passed as maxfev/maxfun when supported.")
    ap.add_argument("--max-time-s", type=float, default=0.0,
                    help="Wall-clock time limit in seconds. 0 disables.")
    ap.add_argument("--target-ca-km", type=float, default=10.0,
                    help="Stop once best closest approach is below this distance, after --min-evals-before-stop.")
    ap.add_argument("--min-evals-before-stop", type=int, default=1,
                    help="Minimum evaluations before target-ca stopping may trigger.")
    ap.add_argument("--stall-evals", type=int, default=0,
                    help="Stop after this many evaluations without a new best CA distance. 0 disables.")
    ap.add_argument("--ipopt-print-level", type=int, default=5)
    ap.add_argument("--tol", type=float, default=1e-4)

    ap.add_argument("--multistart", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ms-burn-spread-s", type=float, default=3600.0)
    ap.add_argument("--ms-dv0-spread-m-s", type=float, default=500.0)
    ap.add_argument("--ms-dsm-spread-m-s", type=float, default=100.0)
    ap.add_argument("--ms-frac-spread", type=float, default=2.0)
    ap.add_argument("--ms-arrival-spread-days", type=float, default=5.0)

    ap.add_argument("--server-timeout-s", type=float, default=600.0)
    ap.add_argument("--quiet-stderr", action="store_true")
    ap.add_argument("--print-every-eval", type=int, default=25)
    ap.add_argument("--output-dir", type=Path, required=True)

    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    live_t = read_time_json(args.live_state_json)
    row = read_leg_row(args.leg_optimizations, args.leg)
    arr_body = (args.arr_body or row.get("arr_body") or "EVE").upper()

    t_dep = frow(row, "t_dep_s", live_t)
    t_arr = frow(row, "t_arr_s", frow(row, "t_end_s"))
    nominal_burn_dt = t_dep - live_t
    nominal_arrival_dt = t_arr - live_t

    arrival_center_initial = args.arrival_center_initial_s
    if arrival_center_initial is None:
        arrival_center_initial = nominal_arrival_dt

    burn_dt_initial = args.burn_dt_initial_s
    if burn_dt_initial is None:
        burn_dt_initial = max(args.burn_dt_min_s, min(args.burn_dt_max_s, nominal_burn_dt))

    if args.dv0_initial_tnb_m_s is not None:
        dv0_tnb = args.dv0_initial_tnb_m_s
        dsm0 = args.dsm_initial_raw_m_s
        if args.x0_json:
            # Reuse times/DSM from seed if present, but NEVER reuse raw dv0 as TNB.
            try:
                seed_x = x_from_seed_json(args.x0_json, arrival_center_initial, args.dsm_fraction_u_initial)
                burn_dt_initial = float(seed_x[0])
                dsm0 = [float(seed_x[5]), float(seed_x[6]), float(seed_x[7])]
                arrival_center_initial = float(seed_x[8])
            except Exception:
                pass
        x0 = np.array([
            burn_dt_initial,
            *dv0_tnb,
            args.dsm_fraction_u_initial,
            *dsm0,
            arrival_center_initial,
        ], dtype=float)
    elif args.x0_json:
        raise SystemExit(
            "--x0-json from previous raw solver is not directly usable as TNB. "
            "Pass --dv0-initial-tnb-m-s, e.g. '3200,0,0'."
        )
    else:
        if args.dv0_initial_raw_m_s is not None:
            raise SystemExit("--dv0-initial-raw-m-s is disabled in TNB mode. Use --dv0-initial-tnb-m-s.")
        dv0_tnb = [3000.0, 0.0, 0.0]
        x0 = np.array([
            burn_dt_initial,
            *dv0_tnb,
            args.dsm_fraction_u_initial,
            *args.dsm_initial_raw_m_s,
            arrival_center_initial,
        ], dtype=float)

    arrival_trust_s = args.arrival_trust_days * 86400.0
    cfg = {
        "vessel_guid": args.vessel_guid,
        "arr_body": arr_body,
        "dep_body": args.dep_body.upper(),
        "dv0_t_min_m_s": args.dv0_t_min_m_s,
        "dv0_t_max_m_s": args.dv0_t_max_m_s,
        "dv0_n_max_abs_m_s": args.dv0_n_max_abs_m_s,
        "dv0_b_max_abs_m_s": args.dv0_b_max_abs_m_s,
        "burn_dt_min_s": args.burn_dt_min_s,
        "burn_dt_max_s": args.burn_dt_max_s,
        "dv0_component_bound_m_s": args.dv0_component_bound_m_s,
        "dv0_min_m_s": args.dv0_min_m_s,
        "dv0_max_m_s": args.dv0_max_m_s,
        "dsm_fraction_min": args.dsm_fraction_min,
        "dsm_fraction_max": args.dsm_fraction_max,
        "dsm_component_bound_m_s": args.dsm_component_bound_m_s,
        "dsm_max_m_s": args.dsm_max_m_s,
        "min_dsm_after_burn_s": args.min_dsm_after_burn_s,
        "min_scan_after_dsm_s": args.min_scan_after_dsm_s,
        "arrival_center_min_dt_s": nominal_arrival_dt - arrival_trust_s,
        "arrival_center_max_dt_s": nominal_arrival_dt + arrival_trust_s,
        "scan_half_width_s": args.scan_half_width_days * 86400.0,
        "scan_min_width_s": args.scan_min_width_s,
        "vca_samples": args.vca_samples,
        "distance_scale_m": args.distance_scale_m,
        "dv_scale_m_s": args.dv_scale_m_s,
        "dv_weight": args.dv_weight,
        "speed_scale_m_s": args.speed_scale_m_s,
        "speed_weight": args.speed_weight,
        "ordering_penalty": args.ordering_penalty,
        "time_penalty_scale_s": args.time_penalty_scale_s,
        "dv_norm_penalty": args.dv_norm_penalty,
        "scan_boundary_margin_fraction": args.scan_boundary_margin_fraction,
        "scan_boundary_penalty": args.scan_boundary_penalty,
        "fail_distance_m": args.fail_distance_m,
        "fail_speed_m_s": args.fail_speed_m_s,
        "server_error_penalty": args.server_error_penalty,
        "server_timeout_s": args.server_timeout_s,
        "print_every_eval": args.print_every_eval,
        "max_evals": args.max_evals,
        "max_time_s": args.max_time_s,
        "target_ca_km": args.target_ca_km,
        "min_evals_before_stop": args.min_evals_before_stop,
        "stall_evals": args.stall_evals,
        "fd_steps": [
            args.fd_time_step_s,
            args.fd_dv_step_m_s,
            args.fd_dv_step_m_s,
            args.fd_dv_step_m_s,
            args.fd_frac_step,
            args.fd_dv_step_m_s,
            args.fd_dv_step_m_s,
            args.fd_dv_step_m_s,
            args.fd_time_step_s,
        ],
    }

    with PrincipiaTargeterClient(
        args.server,
        args.plugin_b64,
        plugin_arg_mode=args.plugin_arg_mode,
        response_timeout_s=args.server_timeout_s,
        quiet_stderr=args.quiet_stderr,
    ) as client:
        problem = VCATargetProblem(cfg, client)
        lb, ub = problem.bounds()
        x0 = np.minimum(np.maximum(np.asarray(x0, dtype=float), lb), ub)

        print("=== SOLVE LEG VCA TARGETER V0 ===")
        print(f"ready_line            : {client.ready_line}")
        print(f"arr_body              : {arr_body}")
        print(f"dep_body              : {args.dep_body.upper()}")
        print("burn0 mode            : TNB/FlightPlan components")
        print(f"dv0 T bounds          : {args.dv0_t_min_m_s} .. {args.dv0_t_max_m_s}")
        print(f"dv0 |N|/|B| bounds    : {args.dv0_n_max_abs_m_s} / {args.dv0_b_max_abs_m_s}")
        print(f"live_t                : {live_t}")
        print(f"nominal_burn_dt       : {nominal_burn_dt}")
        print(f"nominal_arrival_dt    : {nominal_arrival_dt}")
        print(f"arrival bounds        : {cfg['arrival_center_min_dt_s']} .. {cfg['arrival_center_max_dt_s']}")
        print(f"scan_half_width_s     : {cfg['scan_half_width_s']}")
        print(f"method                : {args.method}")
        print(f"x0                    : {x0.tolist()}")

        x0_eval = problem.evaluate(x0)
        print("x0_eval:", json.dumps({
            "objective": x0_eval.objective,
            "ca_distance_km": x0_eval.data["ca_distance_km"],
            "ca_dt_s": x0_eval.data["ca_dt_s"],
            "dv0_norm_m_s": x0_eval.data["dv0_norm_m_s"],
            "dsm_norm_m_s": x0_eval.data["dsm_norm_m_s"],
            "total_dv_m_s": x0_eval.data["total_impulsive_dv_m_s"],
            "server_ok": x0_eval.data["server_ok"],
            "server_error": x0_eval.data["server_error"],
        }, indent=2))

        spread = {
            "burn_dt": args.ms_burn_spread_s,
            "dv0": args.ms_dv0_spread_m_s,
            "dsm": args.ms_dsm_spread_m_s,
            "frac": args.ms_frac_spread,
            "arrival": args.ms_arrival_spread_days * 86400.0,
        }
        x_start = random_multistart(problem, x0, args.multistart, args.seed, spread)

        stopped_early = False
        stop_reason = None
        try:
            if args.method == "ipopt":
                x_opt, solver_info = run_ipopt(problem, x_start, args.max_iter, args.ipopt_print_level, args.tol)
            else:
                scipy_method = {
                    "powell": "Powell",
                    "nelder-mead": "Nelder-Mead",
                    "l-bfgs-b": "L-BFGS-B",
                }[args.method]
                x_opt, solver_info = run_scipy(problem, x_start, scipy_method, args.max_iter, args.max_evals)
            final_eval = problem.evaluate(x_opt)
        except StopOptimization as exc:
            stopped_early = True
            stop_reason = exc.reason
            solver_info = {
                "success": True,
                "status": "stopped_early",
                "message": exc.reason,
                "nfev": problem.eval_count,
            }
            final_eval = problem.best or problem.best_distance or x0_eval
            x_opt = np.asarray(final_eval.data["x"], dtype=float)
            print(f"[STOP] {exc.reason}", flush=True)
        except KeyboardInterrupt:
            stopped_early = True
            stop_reason = "keyboard_interrupt"
            solver_info = {
                "success": False,
                "status": "keyboard_interrupt",
                "message": "Interrupted by user; writing best result found so far.",
                "nfev": problem.eval_count,
            }
            final_eval = problem.best or problem.best_distance or x0_eval
            x_opt = np.asarray(final_eval.data["x"], dtype=float)
            print("[STOP] keyboard_interrupt; writing best result found so far", flush=True)

        best = problem.best or final_eval
        best_distance = problem.best_distance or best

    out = {
        "status": "ok",
        "problem": "leg_vca_targeter_v0_3_tnb",
        "method": args.method,
        "stopped_early": stopped_early,
        "stop_reason": stop_reason,
        "solver_info": solver_info,
        "config": cfg,
        "live_t_s": live_t,
        "leg_row": {k: row[k] for k in row.keys()},
        "x0": x0.tolist(),
        "x0_eval": x0_eval.data,
        "x_start": x_start.tolist(),
        "ipopt_or_scipy_solution": final_eval.data,
        "best": best.data,
        "best_by_distance": best_distance.data,
        "n_evaluations": problem.eval_count,
    }

    result_path = args.output_dir / "leg_vca_targeter_v0_3_tnb_result.json"
    result_path.write_text(json.dumps(out, indent=2) + "\n")

    event_preview = {
        "burn0": {
            "initial_time_dt_s": best.data["burn_dt_s"],
            "delta_v_tnb_m_s": best.data["dv0_tnb_m_s"],
            "delta_v_raw_m_s": best.data["dv0_raw_m_s"],
            "delta_v_norm_m_s": best.data["dv0_norm_m_s"],
        },
        "dsm": {
            "initial_time_dt_s": best.data["dsm_dt_s"],
            "delta_v_raw_m_s": best.data["dsm_raw_m_s"],
            "delta_v_norm_m_s": best.data["dsm_norm_m_s"],
        },
        "vca": {
            "target_body": arr_body,
            "scan_start_dt_s": best.data["scan_start_dt_s"],
            "scan_end_dt_s": best.data["scan_end_dt_s"],
            "ca_dt_s": best.data["ca_dt_s"],
            "ca_distance_m": best.data["ca_distance_m"],
            "ca_distance_km": best.data["ca_distance_km"],
            "ca_speed_m_s": best.data["ca_speed_m_s"],
            "status": best.data["vca_status"],
        },
    }
    preview_path = args.output_dir / "leg_vca_targeter_v0_3_tnb_event_preview.json"
    preview_path.write_text(json.dumps(event_preview, indent=2) + "\n")

    print("=== RESULT ===")
    print(json.dumps({
        "objective": best.objective,
        "ca_distance_km": best.data["ca_distance_km"],
        "ca_dt_s": best.data["ca_dt_s"],
        "burn_dt_s": best.data["burn_dt_s"],
        "dsm_dt_s": best.data["dsm_dt_s"],
        "dv0_tnb_m_s": best.data["dv0_tnb_m_s"],
        "dv0_raw_m_s": best.data["dv0_raw_m_s"],
        "dv0_norm_m_s": best.data["dv0_norm_m_s"],
        "dsm_norm_m_s": best.data["dsm_norm_m_s"],
        "total_dv_m_s": best.data["total_impulsive_dv_m_s"],
        "vca_status": best.data["vca_status"],
        "n_evaluations": problem.eval_count,
        "stopped_early": stopped_early,
        "stop_reason": stop_reason,
    }, indent=2))
    print(f"[OK] wrote {result_path}")
    print(f"[OK] wrote {preview_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
