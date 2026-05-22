#!/usr/bin/env python3
"""
outer_epoch_stm_commander_v1_seeded.py

Two-level targeter for KSP/Principia MGA departures.

Architecture
------------
Outer loop ("commander"):
  Optimizes only epoch scalars:
    - dt0_days: burn epoch shift relative to candidate burn_abs_s
    - dtf_days: target/final epoch shift relative to candidate t_arr_s

Inner loop ("navigator"):
  For each (t0, tf), solves the actual dynamics correction with a finite-difference
  STM/least-squares corrector over:
    - departure burn T,N,B components
    - optional DSM LevelA XYZ components at fixed fraction of the leg

This is intended as a fast mid-fidelity targeter:
  PyKEP/Lambert seed -> TudatPy/SPICE + STM corrector -> Principia VCAREL_NAV validation

It uses the project's Principia-generated SPICE BSP/TPC as the moving-body ephemeris.
It does not replace final validation in the Principia server.

Frame contract used by the project:
  Principia raw -> LevelA/SPICE canonical: (X,Y,Z)->(-Y,+Z,+X)
  LevelA/SPICE canonical -> Principia raw: (X,Y,Z)->(+Z,-X,+Y)

Typical row15 use:
  --state-source burnstate
  --flip-binormal
  --tf-offset-initial-days -21.81279426069721
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize, differential_evolution

try:
    from tudatpy.interface import spice
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "Could not import tudatpy.interface.spice. Activate your TudatPy environment first. "
        f"Original error: {exc}"
    )

DAY_S = 86400.0


def norm(v: Sequence[float]) -> float:
    return float(np.linalg.norm(np.asarray(v, dtype=float)))


def unit(v: Sequence[float], name: str = "vector") -> np.ndarray:
    a = np.asarray(v, dtype=float)
    n = norm(a)
    if not math.isfinite(n) or n <= 0.0:
        raise ValueError(f"cannot normalize {name}: {v!r}")
    return a / n


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def raw_to_levela(v: Sequence[float]) -> np.ndarray:
    x, y, z = map(float, v)
    return np.asarray([-y, z, x], dtype=float)


def levela_to_raw(v: Sequence[float]) -> np.ndarray:
    x, y, z = map(float, v)
    return np.asarray([z, -x, y], dtype=float)


def parse_body_list(s: str) -> list[str]:
    return [x.strip().upper() for x in s.split(",") if x.strip()]


def load_json(path: Path) -> Any:
    with path.open() as f:
        return json.load(f)


def json_safe(x: Any) -> Any:
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, float):
        return x if math.isfinite(x) else None
    if isinstance(x, dict):
        return {str(k): json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [json_safe(v) for v in x]
    return x


def load_body_catalog(path: Path) -> dict[str, dict[str, Any]]:
    data = load_json(path)
    if isinstance(data, dict) and "bodies" in data and isinstance(data["bodies"], list):
        out: dict[str, dict[str, Any]] = {}
        for b in data["bodies"]:
            name = str(b.get("name") or b.get("body") or b.get("id") or "").upper()
            if name:
                out[name] = b
        return out
    if isinstance(data, dict):
        return {str(k).upper(): v for k, v in data.items()}
    raise ValueError(f"unsupported body catalog schema: {path}")


def mu_for(body_catalog: dict[str, dict[str, Any]], name: str) -> float:
    key = name.upper()
    if key not in body_catalog:
        raise KeyError(f"body {key} not found in body catalog")
    b = body_catalog[key]
    for k in (
        "mu_m3_s2", "gravitational_parameter_m3_s2", "gravitational_parameter",
        "mu", "gm_m3_s2", "GM", "gm",
    ):
        if k in b:
            return float(b[k])
    raise KeyError(f"cannot find mu for {key}; keys={sorted(b.keys())}")


def normalize_live_state(snapshot: dict[str, Any]) -> dict[str, Any]:
    if "vessel" not in snapshot:
        return snapshot
    v = snapshot["vessel"]
    out = dict(snapshot)
    for k in (
        "vessel_guid", "vessel_name", "nav_body", "reference_body", "reference_body_index",
        "rel_r_raw_m", "rel_v_raw_m_s", "mass_tonnes", "available_thrust_kN",
        "specific_impulse_s_g0", "state_source",
    ):
        if k in v:
            out[k] = v[k]
    return out


def find_candidate(rank: Any, row_index0: int | None, top_index: int) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []

    def walk(x: Any) -> None:
        if isinstance(x, dict):
            if "row_index0" in x and (
                "burn_rel_r_raw_m" in x or "dv_tangent_m_s" in x or "t_arr_s" in x
            ):
                candidates.append(x)
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)

    walk(rank)

    if row_index0 is not None:
        matches = [c for c in candidates if int(c.get("row_index0", -999999)) == int(row_index0)]
        if not matches:
            raise KeyError(f"row_index0={row_index0} not found")
        matches.sort(
            key=lambda c: sum(k in c for k in (
                "burn_abs_s", "burn_rel_r_raw_m", "burn_rel_v_raw_m_s", "t_arr_s",
                "dv_tangent_m_s", "dv_normal_m_s", "dv_binormal_m_s",
            )),
            reverse=True,
        )
        return matches[0]

    if isinstance(rank, dict) and "top" in rank and isinstance(rank["top"], list):
        return rank["top"][top_index]
    if candidates:
        return candidates[top_index]
    if isinstance(rank, dict):
        return rank
    raise ValueError("cannot find candidate in rank-json")


def tnb_basis_raw(
    r_raw: Sequence[float],
    v_raw: Sequence[float],
    binormal_sign: float = 1.0,
    normal_sign: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r = np.asarray(r_raw, dtype=float)
    v = np.asarray(v_raw, dtype=float)
    T = unit(v, "T")
    h = np.cross(r, v)
    if norm(h) <= 1e-14:
        radial = unit(r, "radial")
        B = unit(np.cross(radial, T), "fallback B")
    else:
        B = unit(h, "B")
    B = float(binormal_sign) * B
    N = unit(np.cross(B, T), "N")
    N = float(normal_sign) * N
    return T, N, B


def dv_nav_to_raw(
    r_raw: Sequence[float],
    v_raw: Sequence[float],
    dvt: float,
    dvn: float,
    dvb: float,
    binormal_basis_sign: float = 1.0,
    normal_basis_sign: float = 1.0,
) -> np.ndarray:
    T, N, B = tnb_basis_raw(r_raw, v_raw, binormal_basis_sign, normal_basis_sign)
    return float(dvt) * T + float(dvn) * N + float(dvb) * B


def get_tnb_seed(c: dict[str, Any], flip_normal: bool, flip_binormal: bool) -> np.ndarray:
    dvt = float(c.get("dv_tangent_m_s", c.get("dvt_m_s", 0.0)))
    dvn = float(c.get("dv_normal_m_s", c.get("dvn_m_s", 0.0)))
    dvb = float(c.get("dv_binormal_m_s", c.get("dvb_m_s", 0.0)))
    if flip_normal:
        dvn = -dvn
    if flip_binormal:
        dvb = -dvb
    return np.asarray([dvt, dvn, dvb], dtype=float)




def parse_vec3_csv(s: str | None, default: np.ndarray | None = None) -> np.ndarray | None:
    if s is None or str(s).strip() == "":
        return None if default is None else np.asarray(default, dtype=float)
    parts = [float(x.strip()) for x in str(s).split(",")]
    if len(parts) != 3:
        raise ValueError(f"expected 3 comma-separated values, got {s!r}")
    return np.asarray(parts, dtype=float)


def best_from_result_json(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    data = load_json(path)
    if isinstance(data, dict):
        b = data.get("best")
        if isinstance(b, dict) and b.get("ok", True):
            return b
        top = data.get("top") or data.get("top_validated")
        if isinstance(top, list) and top:
            return top[0]
    return None

def spice_state_safe(target: str, observer: str, frame: str, et: float) -> np.ndarray:
    target = target.upper()
    observer = observer.upper()
    if target == observer:
        return np.zeros(6)
    try:
        return np.asarray(spice.get_body_cartesian_state_at_epoch(target, observer, frame, "NONE", float(et)), dtype=float)
    except Exception:
        return np.asarray(spice.get_body_cartesian_state_at_epoch(target.title(), observer.title(), frame, "NONE", float(et)), dtype=float)


@dataclass
class EphemerisCache:
    observer: str
    frame: str
    times: np.ndarray
    splines: dict[str, list[CubicSpline]]

    def state(self, body: str, t: float) -> np.ndarray:
        key = body.upper()
        if key == self.observer.upper():
            return np.zeros(6)
        if key not in self.splines:
            raise KeyError(f"body {key} not cached")
        return np.asarray([cs(float(t)) for cs in self.splines[key]], dtype=float)


def build_ephemeris_cache(
    bodies: Iterable[str],
    observer: str,
    frame: str,
    t0: float,
    t1: float,
    step_s: float,
) -> EphemerisCache:
    observer = observer.upper()
    body_list = sorted({b.upper() for b in bodies if b.upper() != observer})
    n = max(8, int(math.ceil((float(t1) - float(t0)) / float(step_s))) + 1)
    times = np.linspace(float(t0), float(t1), n)
    splines: dict[str, list[CubicSpline]] = {}
    for body in body_list:
        arr = np.vstack([spice_state_safe(body, observer, frame, t) for t in times])
        splines[body] = [CubicSpline(times, arr[:, i], extrapolate=True) for i in range(6)]
    return EphemerisCache(observer=observer, frame=frame, times=times, splines=splines)


def propagate_twobody_relative(
    r0: np.ndarray,
    v0: np.ndarray,
    dt: float,
    mu: float,
    rtol: float,
    max_step_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    if abs(float(dt)) <= 1e-9:
        return np.asarray(r0, dtype=float).copy(), np.asarray(v0, dtype=float).copy()

    def rhs(_t: float, y: np.ndarray) -> np.ndarray:
        r = y[:3]
        v = y[3:]
        rn = norm(r)
        return np.r_[v, -mu * r / rn**3]

    sol = solve_ivp(
        rhs,
        (0.0, float(dt)),
        np.r_[r0, v0],
        method="DOP853",
        rtol=rtol,
        atol=[1e-3, 1e-3, 1e-3, 1e-8, 1e-8, 1e-8],
        max_step=max_step_s,
    )
    if not sol.success:
        raise RuntimeError(f"parking/two-body propagation failed: {sol.message}")
    y = np.asarray(sol.y[:, -1], dtype=float)
    return y[:3], y[3:]


@dataclass
class CommanderConfig:
    candidate: dict[str, Any]
    live_state: dict[str, Any]
    body_catalog: dict[str, dict[str, Any]]
    ephem: EphemerisCache
    attractors: list[str]
    dep_body: str
    arr_body: str
    nav_body: str
    observer: str
    frame: str

    # Nominal source state.
    state_source: str
    source_abs_s: float
    source_rel_r_raw_m: np.ndarray
    source_rel_v_raw_m_s: np.ndarray
    nominal_burn_abs_s: float
    nominal_arr_s: float
    seed_tnb: np.ndarray

    # Dynamics/settings.
    binormal_basis_sign: float
    normal_basis_sign: float
    max_step_s: float
    rtol: float
    atol_pos_m: float
    atol_vel_m_s: float
    parking_rtol: float
    parking_max_step_s: float

    # Inner STM/corrector.
    inner_iterations: int
    fd_dv_step_m_s: float
    fd_dsm_step_m_s: float
    inner_position_scale_m: float
    inner_burn_reg: float
    inner_dsm_reg: float
    inner_burn_step_max_m_s: float
    inner_dsm_step_max_m_s: float
    inner_target_km: float
    enable_dsm: bool
    dsm_frac: float
    dsm_max_abs_m_s: float
    initial_dsm_levela_m_s: np.ndarray | None

    # Bounds/cost.
    t_min: float
    t_max: float
    n_max_abs: float
    b_min: float
    b_max: float
    dv_max_m_s: float
    cost_position_scale_km: float
    cost_dv_weight: float
    cost_dsm_weight: float
    cost_epoch_weight: float


class CommanderObjective:
    def __init__(self, cfg: CommanderConfig, max_outer_evals: int, print_every: int):
        self.cfg = cfg
        self.max_outer_evals = max_outer_evals
        self.print_every = print_every
        self.evals = 0
        self.rows: list[dict[str, Any]] = []
        self.inner_rows: list[dict[str, Any]] = []
        self.best: dict[str, Any] | None = None
        self.stop_reason: str | None = None

    def relative_state_at_t0(self, t0: float) -> tuple[np.ndarray, np.ndarray]:
        mu_nav = mu_for(self.cfg.body_catalog, self.cfg.nav_body)
        dt = float(t0) - float(self.cfg.source_abs_s)
        return propagate_twobody_relative(
            self.cfg.source_rel_r_raw_m,
            self.cfg.source_rel_v_raw_m_s,
            dt,
            mu_nav,
            rtol=self.cfg.parking_rtol,
            max_step_s=self.cfg.parking_max_step_s,
        )

    def spacecraft_endpoint(
        self,
        t0: float,
        tf: float,
        rel_r_raw: np.ndarray,
        rel_v_raw: np.ndarray,
        tnb: np.ndarray,
        dsm_levela: np.ndarray | None = None,
    ) -> tuple[np.ndarray, float | None]:
        cfg = self.cfg
        dep_state = cfg.ephem.state(cfg.dep_body, t0)
        r0 = dep_state[:3] + raw_to_levela(rel_r_raw)
        v0_pre = dep_state[3:] + raw_to_levela(rel_v_raw)
        dv_raw = dv_nav_to_raw(
            rel_r_raw,
            rel_v_raw,
            float(tnb[0]),
            float(tnb[1]),
            float(tnb[2]),
            binormal_basis_sign=cfg.binormal_basis_sign,
            normal_basis_sign=cfg.normal_basis_sign,
        )
        y0 = np.r_[r0, v0_pre + raw_to_levela(dv_raw)]
        mus = {b: mu_for(cfg.body_catalog, b) for b in cfg.attractors}

        def rhs(t: float, y: np.ndarray) -> np.ndarray:
            r = y[:3]
            a = np.zeros(3)
            for body, mu in mus.items():
                rb = cfg.ephem.state(body, t)[:3]
                dr = rb - r
                d = float(np.linalg.norm(dr))
                if d > 0.0:
                    a += mu * dr / d**3
            return np.r_[y[3:], a]

        def integrate_segment(ta: float, tb: float, ya: np.ndarray):
            if tb <= ta:
                raise RuntimeError(f"invalid propagation interval: {ta} -> {tb}")
            return solve_ivp(
                rhs,
                (float(ta), float(tb)),
                ya,
                method="DOP853",
                rtol=cfg.rtol,
                atol=[cfg.atol_pos_m] * 3 + [cfg.atol_vel_m_s] * 3,
                dense_output=False,
                max_step=cfg.max_step_s,
            )

        t_dsm = None
        if cfg.enable_dsm and dsm_levela is not None and norm(dsm_levela) > 0.0:
            t_dsm = float(t0) + clamp(cfg.dsm_frac, 0.01, 0.99) * (float(tf) - float(t0))
            sol1 = integrate_segment(t0, t_dsm, y0)
            if not sol1.success:
                raise RuntimeError(f"integrator failed before DSM: {sol1.message}")
            y_dsm = np.asarray(sol1.y[:, -1], dtype=float)
            y_dsm[3:] += np.asarray(dsm_levela, dtype=float)
            sol2 = integrate_segment(t_dsm, tf, y_dsm)
            if not sol2.success:
                raise RuntimeError(f"integrator failed after DSM: {sol2.message}")
            return np.asarray(sol2.y[:, -1], dtype=float), t_dsm

        sol = integrate_segment(t0, tf, y0)
        if not sol.success:
            raise RuntimeError(f"integrator failed: {sol.message}")
        return np.asarray(sol.y[:, -1], dtype=float), t_dsm

    def endpoint_error(
        self,
        t0: float,
        tf: float,
        rel_r_raw: np.ndarray,
        rel_v_raw: np.ndarray,
        tnb: np.ndarray,
        dsm_levela: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray, float | None]:
        y = self.spacecraft_endpoint(t0, tf, rel_r_raw, rel_v_raw, tnb, dsm_levela)[0]
        arr = self.cfg.ephem.state(self.cfg.arr_body, tf)
        err = y[:3] - arr[:3]
        relv = y[3:] - arr[3:]
        return err, relv, None

    def inner_correct(self, t0: float, tf: float, outer_eval: int) -> dict[str, Any]:
        cfg = self.cfg
        rel_r, rel_v = self.relative_state_at_t0(t0)
        tnb = np.asarray(cfg.seed_tnb, dtype=float).copy()
        dsm = (np.asarray(cfg.initial_dsm_levela_m_s, dtype=float).copy() if (cfg.enable_dsm and cfg.initial_dsm_levela_m_s is not None) else (np.zeros(3, dtype=float) if cfg.enable_dsm else None))

        def clip_controls() -> None:
            tnb[0] = clamp(float(tnb[0]), cfg.t_min, cfg.t_max)
            tnb[1] = clamp(float(tnb[1]), -cfg.n_max_abs, cfg.n_max_abs)
            tnb[2] = clamp(float(tnb[2]), cfg.b_min, cfg.b_max)
            if dsm is not None:
                for i in range(3):
                    dsm[i] = clamp(float(dsm[i]), -cfg.dsm_max_abs_m_s, cfg.dsm_max_abs_m_s)

        best_iter: dict[str, Any] | None = None

        for it in range(cfg.inner_iterations + 1):
            clip_controls()
            err, relv, _ = self.endpoint_error(t0, tf, rel_r, rel_v, tnb, dsm)
            err_m = norm(err)
            dsm_norm = 0.0 if dsm is None else norm(dsm)
            row = {
                "outer_eval": outer_eval,
                "inner_iter": it,
                "t0_s": float(t0),
                "tf_s": float(tf),
                "dt0_days": (float(t0) - cfg.nominal_burn_abs_s) / DAY_S,
                "dtf_days": (float(tf) - cfg.nominal_arr_s) / DAY_S,
                "endpoint_error_km": err_m / 1000.0,
                "rel_speed_m_s": norm(relv),
                "dvt_m_s": float(tnb[0]),
                "dvn_m_s": float(tnb[1]),
                "dvb_m_s": float(tnb[2]),
                "dv_norm_m_s": norm(tnb),
                "dsm_norm_m_s": dsm_norm,
                "dsm_levela_m_s": None if dsm is None else dsm.tolist(),
                "rel_r_raw_m": rel_r.tolist(),
                "rel_v_raw_m_s": rel_v.tolist(),
            }
            self.inner_rows.append(row)
            if best_iter is None or row["endpoint_error_km"] < best_iter["endpoint_error_km"]:
                best_iter = dict(row)

            if err_m / 1000.0 <= cfg.inner_target_km or it >= cfg.inner_iterations:
                break

            # Build finite-difference sensitivity A = d(error_position)/d(control).
            cols: list[np.ndarray] = []
            labels: list[str] = []
            base_err = err.copy()

            for j, label in enumerate(("dT", "dN", "dB")):
                step = cfg.fd_dv_step_m_s
                tnb_p = tnb.copy()
                tnb_p[j] += step
                try:
                    err_p, _relv_p, _ = self.endpoint_error(t0, tf, rel_r, rel_v, tnb_p, dsm)
                    cols.append((err_p - base_err) / step)
                    labels.append(label)
                except Exception:
                    pass

            if cfg.enable_dsm and dsm is not None:
                for j, label in enumerate(("dDSMx", "dDSMy", "dDSMz")):
                    step = cfg.fd_dsm_step_m_s
                    dsm_p = dsm.copy()
                    dsm_p[j] += step
                    try:
                        err_p, _relv_p, _ = self.endpoint_error(t0, tf, rel_r, rel_v, tnb, dsm_p)
                        cols.append((err_p - base_err) / step)
                        labels.append(label)
                    except Exception:
                        pass

            if not cols:
                break
            A = np.column_stack(cols)  # m per (m/s)
            b = -base_err

            # Weighted damped least-squares:
            #   min ||(A delta - b)/pos_scale||² + reg ||delta/control_scale||²
            pos_scale = max(float(cfg.inner_position_scale_m), 1.0)
            Aw = A / pos_scale
            bw = b / pos_scale
            reg_rows = []
            reg_rhs = []
            for label in labels:
                if label.startswith("dDSM"):
                    reg = cfg.inner_dsm_reg
                    scale = max(cfg.inner_dsm_step_max_m_s, 1.0)
                else:
                    reg = cfg.inner_burn_reg
                    scale = max(cfg.inner_burn_step_max_m_s, 1.0)
                row_reg = np.zeros(len(labels))
                row_reg[len(reg_rows)] = math.sqrt(max(reg, 0.0)) / scale
                reg_rows.append(row_reg)
                reg_rhs.append(0.0)
            if reg_rows:
                Aw2 = np.vstack([Aw, np.vstack(reg_rows)])
                bw2 = np.r_[bw, np.asarray(reg_rhs)]
            else:
                Aw2, bw2 = Aw, bw

            delta, *_ = np.linalg.lstsq(Aw2, bw2, rcond=None)

            # Apply trust-region clipping per group.
            for k, label in enumerate(labels):
                val = float(delta[k])
                if label == "dT":
                    val = clamp(val, -cfg.inner_burn_step_max_m_s, cfg.inner_burn_step_max_m_s)
                    tnb[0] += val
                elif label == "dN":
                    val = clamp(val, -cfg.inner_burn_step_max_m_s, cfg.inner_burn_step_max_m_s)
                    tnb[1] += val
                elif label == "dB":
                    val = clamp(val, -cfg.inner_burn_step_max_m_s, cfg.inner_burn_step_max_m_s)
                    tnb[2] += val
                elif label == "dDSMx" and dsm is not None:
                    dsm[0] += clamp(val, -cfg.inner_dsm_step_max_m_s, cfg.inner_dsm_step_max_m_s)
                elif label == "dDSMy" and dsm is not None:
                    dsm[1] += clamp(val, -cfg.inner_dsm_step_max_m_s, cfg.inner_dsm_step_max_m_s)
                elif label == "dDSMz" and dsm is not None:
                    dsm[2] += clamp(val, -cfg.inner_dsm_step_max_m_s, cfg.inner_dsm_step_max_m_s)

        assert best_iter is not None
        # Recompute a final score from the best observed inner row.
        score = (
            best_iter["endpoint_error_km"] / cfg.cost_position_scale_km
            + cfg.cost_dv_weight * best_iter["dv_norm_m_s"]
            + cfg.cost_dsm_weight * best_iter["dsm_norm_m_s"]
            + cfg.cost_epoch_weight * (abs(best_iter["dt0_days"]) + abs(best_iter["dtf_days"]))
        )
        best_iter["score"] = float(score)
        best_iter["ok"] = True
        return best_iter

    def __call__(self, x: Sequence[float]) -> float:
        if self.max_outer_evals > 0 and self.evals >= self.max_outer_evals:
            self.stop_reason = f"max_outer_evals={self.max_outer_evals}"
            return 1e12
        xx = np.asarray(x, dtype=float)
        if len(xx) != 2:
            raise ValueError("outer variable must be [dt0_days, dtf_days]")
        dt0_days, dtf_days = map(float, xx)
        t0 = self.cfg.nominal_burn_abs_s + dt0_days * DAY_S
        tf = self.cfg.nominal_arr_s + dtf_days * DAY_S
        if tf <= t0 + 3600.0:
            return 1e11 + abs(tf - t0)

        self.evals += 1
        try:
            row = self.inner_correct(t0, tf, self.evals)
        except Exception as exc:
            row = {
                "ok": False,
                "score": 1e10,
                "error": str(exc),
                "dt0_days": dt0_days,
                "dtf_days": dtf_days,
                "outer_eval": self.evals,
            }
        self.rows.append(row)
        if row.get("ok") and (self.best is None or row["score"] < self.best["score"]):
            self.best = dict(row)

        if self.print_every > 0 and (self.evals == 1 or self.evals % self.print_every == 0 or row.get("ok")):
            if row.get("ok"):
                print(
                    f"[outer {self.evals:04d}] score={row['score']:10.5g} "
                    f"err={row['endpoint_error_km']:11.3f} km "
                    f"dt0={dt0_days:8.4f} d dtf={dtf_days:8.4f} d "
                    f"TNB=[{row['dvt_m_s']:.3f},{row['dvn_m_s']:.3f},{row['dvb_m_s']:.3f}] "
                    f"dsm={row['dsm_norm_m_s']:.3f}"
                )
            else:
                print(f"[outer {self.evals:04d}] ERR {row.get('error')}")
        return float(row["score"])


def patch_rank_with_best(rank: Any, row_index0: int, best: dict[str, Any], out_path: Path) -> None:
    data = json.loads(json.dumps(rank))
    patched = 0
    T = float(best["dvt_m_s"])
    N = float(best["dvn_m_s"])
    B = float(best["dvb_m_s"])
    dvn = math.sqrt(T*T + N*N + B*B)

    def walk(x: Any) -> None:
        nonlocal patched
        if isinstance(x, dict):
            if int(x.get("row_index0", -999999)) == int(row_index0):
                x["burn_abs_s"] = float(best["t0_s"])
                x["t_arr_s"] = float(best["tf_s"])
                x["burn_rel_r_raw_m"] = best["rel_r_raw_m"]
                x["burn_rel_v_raw_m_s"] = best["rel_v_raw_m_s"]
                x["dv_tangent_m_s"] = T
                x["dv_normal_m_s"] = N
                x["dv_binormal_m_s"] = B
                x["dv_norm_m_s"] = dvn
                x["outer_epoch_stm_patch"] = {
                    "endpoint_error_km": float(best["endpoint_error_km"]),
                    "dt0_days": float(best["dt0_days"]),
                    "dtf_days": float(best["dtf_days"]),
                    "score": float(best["score"]),
                    "dsm_norm_m_s": float(best.get("dsm_norm_m_s", 0.0)),
                }
                patched += 1
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)

    walk(data)
    if patched <= 0:
        raise RuntimeError(f"failed to patch row_index0={row_index0}")
    out_path.write_text(json.dumps(json_safe(data), indent=2) + "\n")


def write_outputs(out_dir: Path, rank: Any, row_index0: int, cfg: CommanderConfig, obj: CommanderObjective, args: argparse.Namespace, x0: np.ndarray, outer_bounds: list[tuple[float, float]]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = sorted([r for r in obj.rows if r.get("ok")], key=lambda r: r["score"])
    best = obj.best or (rows[0] if rows else None)
    result = {
        "schema": "outer_epoch_stm_commander_v1_seeded",
        "description": "Outer epoch NLP + inner finite-difference STM/least-squares corrector",
        "config": json_safe(vars(args)),
        "x0_outer": x0.tolist(),
        "outer_bounds": outer_bounds,
        "n_outer_evals": obj.evals,
        "n_ok": len(rows),
        "stop_reason": obj.stop_reason,
        "best": best,
        "top": rows[:50],
        "rows": obj.rows,
    }
    (out_dir / "outer_epoch_stm_commander_result.json").write_text(json.dumps(json_safe(result), indent=2) + "\n")

    def write_csv(path: Path, row_list: list[dict[str, Any]]) -> None:
        if not row_list:
            path.write_text("")
            return
        flat = []
        for r in row_list:
            rr = {}
            for k, v in r.items():
                if isinstance(v, (list, dict)):
                    rr[k] = json.dumps(json_safe(v))
                else:
                    rr[k] = v
            flat.append(rr)
        fields = sorted({k for r in flat for k in r.keys()})
        with path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(flat)

    write_csv(out_dir / "outer_epoch_stm_commander_rows.csv", obj.rows)
    write_csv(out_dir / "outer_epoch_stm_inner_iterations.csv", obj.inner_rows)

    if best:
        patch_rank_with_best(rank, row_index0, best, out_dir / f"rank_row{row_index0}_outer_epoch_stm_seed.json")
        # Human-readable summary/next validation hint.
        cmd_lines = [
            "# Best single-burn candidate for Principia VCAREL_NAV validation",
            f"# t0_s={best['t0_s']}",
            f"# tf_s={best['tf_s']}",
            f"# endpoint_error_km={best['endpoint_error_km']}",
            f"# TNB=[{best['dvt_m_s']}, {best['dvn_m_s']}, {best['dvb_m_s']}]",
            f"# burn_rel_r_raw_m={best['rel_r_raw_m']}",
            f"# burn_rel_v_raw_m_s={best['rel_v_raw_m_s']}",
            "# Use the patched rank JSON with validate_snapshot_ranked_candidate_burnstate_vcarelnav_v0.py",
        ]
        (out_dir / "README_next_validation.txt").write_text("\n".join(cmd_lines) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--rank-json", type=Path, required=True)
    ap.add_argument("--row-index0", type=int, required=True)
    ap.add_argument("--top-index", type=int, default=0)
    ap.add_argument("--live-state-json", type=Path, required=True)
    ap.add_argument("--bsp", type=Path, required=True)
    ap.add_argument("--tpc", type=Path, default=None)
    ap.add_argument("--body-catalog", type=Path, required=True)

    ap.add_argument("--dep-body", default="KERBIN")
    ap.add_argument("--arr-body", default="EVE")
    ap.add_argument("--nav-body", default="KERBIN")
    ap.add_argument("--attractors", default="SUN,KERBIN,EVE")
    ap.add_argument("--observer", default="SUN")
    ap.add_argument("--frame", default="J2000")
    ap.add_argument("--state-source", choices=["burnstate", "live-twobody"], default="burnstate")
    ap.add_argument("--flip-binormal", action="store_true")
    ap.add_argument("--flip-normal", action="store_true")
    ap.add_argument("--seed-result-json", type=Path, default=None,
                    help="Optional previous Tudat/outer result JSON. If provided, use its best/top[0] TNB as inner seed.")
    ap.add_argument("--seed-dsm-from-result", action="store_true",
                    help="If --seed-result-json has dsm_levela_m_s, use it as initial DSM for every inner solve.")
    ap.add_argument("--seed-dsm-levela", default=None,
                    help="Optional initial DSM LevelA as 'x,y,z' m/s, overrides --seed-dsm-from-result.")

    ap.add_argument("--dt0-initial-days", type=float, default=0.0)
    ap.add_argument("--dt0-min-days", type=float, default=-1.0)
    ap.add_argument("--dt0-max-days", type=float, default=1.0)
    ap.add_argument("--dtf-initial-days", type=float, default=0.0)
    ap.add_argument("--dtf-min-days", type=float, default=-80.0)
    ap.add_argument("--dtf-max-days", type=float, default=20.0)

    ap.add_argument("--t-trust", type=float, default=600.0)
    ap.add_argument("--n-max-abs", type=float, default=900.0)
    ap.add_argument("--b-trust", type=float, default=1200.0)
    ap.add_argument("--dv-max-m-s", type=float, default=5000.0)

    ap.add_argument("--enable-dsm", action="store_true")
    ap.add_argument("--dsm-frac", type=float, default=0.5)
    ap.add_argument("--dsm-max-abs-m-s", type=float, default=300.0)

    ap.add_argument("--inner-iterations", type=int, default=5)
    ap.add_argument("--fd-dv-step-m-s", type=float, default=1.0)
    ap.add_argument("--fd-dsm-step-m-s", type=float, default=1.0)
    ap.add_argument("--inner-position-scale-m", type=float, default=1.0e8)
    ap.add_argument("--inner-burn-reg", type=float, default=1e-3)
    ap.add_argument("--inner-dsm-reg", type=float, default=1e-3)
    ap.add_argument("--inner-burn-step-max-m-s", type=float, default=80.0)
    ap.add_argument("--inner-dsm-step-max-m-s", type=float, default=80.0)
    ap.add_argument("--inner-target-km", type=float, default=50000.0)

    ap.add_argument("--rtol", type=float, default=1e-9)
    ap.add_argument("--max-step-s", type=float, default=21600.0)
    ap.add_argument("--atol-pos-m", type=float, default=1e-2)
    ap.add_argument("--atol-vel-m-s", type=float, default=1e-8)
    ap.add_argument("--parking-rtol", type=float, default=1e-11)
    ap.add_argument("--parking-max-step-s", type=float, default=300.0)
    ap.add_argument("--ephem-cache-step-s", type=float, default=21600.0)
    ap.add_argument("--cache-padding-days", type=float, default=5.0)

    ap.add_argument("--cost-position-scale-km", type=float, default=100000.0)
    ap.add_argument("--cost-dv-weight", type=float, default=0.001)
    ap.add_argument("--cost-dsm-weight", type=float, default=0.005)
    ap.add_argument("--cost-epoch-weight", type=float, default=0.001)

    ap.add_argument("--outer-method", choices=["slsqp", "powell", "de"], default="slsqp")
    ap.add_argument("--outer-maxiter", type=int, default=12)
    ap.add_argument("--outer-maxevals", type=int, default=60)
    ap.add_argument("--outer-eps", type=float, default=0.05)
    ap.add_argument("--outer-ftol", type=float, default=1e-3)
    ap.add_argument("--de-popsize", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--print-every", type=int, default=1)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    spice.load_kernel(str(args.bsp))
    if args.tpc is not None:
        spice.load_kernel(str(args.tpc))

    rank = load_json(args.rank_json)
    candidate = find_candidate(rank, args.row_index0, args.top_index)
    live = normalize_live_state(load_json(args.live_state_json))
    body_catalog = load_body_catalog(args.body_catalog)

    dep_body = args.dep_body.upper()
    arr_body = args.arr_body.upper()
    nav_body = args.nav_body.upper()
    observer = args.observer.upper()
    attractors = parse_body_list(args.attractors)

    if args.state_source == "burnstate":
        source_abs_s = float(candidate["burn_abs_s"])
        source_rel_r = np.asarray(candidate["burn_rel_r_raw_m"], dtype=float)
        source_rel_v = np.asarray(candidate["burn_rel_v_raw_m_s"], dtype=float)
        nominal_burn_abs_s = source_abs_s
    else:
        source_abs_s = float(live.get("t_spice_s", live.get("t_game_s")))
        source_rel_r = np.asarray(live["rel_r_raw_m"], dtype=float)
        source_rel_v = np.asarray(live["rel_v_raw_m_s"], dtype=float)
        nominal_burn_abs_s = float(candidate.get("burn_abs_s", source_abs_s))

    nominal_arr_s = float(candidate["t_arr_s"])
    seed_tnb = get_tnb_seed(candidate, args.flip_normal, args.flip_binormal)
    seed_result = best_from_result_json(args.seed_result_json)
    initial_dsm_levela = parse_vec3_csv(args.seed_dsm_levela)
    if seed_result is not None:
        if all(k in seed_result for k in ("dvt_m_s", "dvn_m_s", "dvb_m_s")):
            seed_tnb = np.asarray([
                float(seed_result["dvt_m_s"]),
                float(seed_result["dvn_m_s"]),
                float(seed_result["dvb_m_s"]),
            ], dtype=float)
        if args.seed_dsm_from_result and initial_dsm_levela is None and seed_result.get("dsm_levela_m_s") is not None:
            initial_dsm_levela = np.asarray(seed_result["dsm_levela_m_s"], dtype=float)

    # Cache range must cover all outer bounds plus propagation interval.
    t_min = nominal_burn_abs_s + min(args.dt0_min_days, 0.0) * DAY_S - args.cache_padding_days * DAY_S
    t_max = nominal_arr_s + max(args.dtf_max_days, args.dtf_initial_days, 0.0) * DAY_S + args.cache_padding_days * DAY_S
    t_min = min(t_min, source_abs_s - args.cache_padding_days * DAY_S)
    bodies_to_cache = set(attractors + [dep_body, arr_body, nav_body, observer])

    print("=== OUTER EPOCH STM COMMANDER V0 ===")
    print(f"row_index0       : {args.row_index0}")
    print(f"state_source     : {args.state_source}")
    print(f"source_abs_s     : {source_abs_s}")
    print(f"nominal burn/arr : {nominal_burn_abs_s} -> {nominal_arr_s} ({(nominal_arr_s-nominal_burn_abs_s)/DAY_S:.6f} d)")
    print(f"seed TNB         : {seed_tnb.tolist()} |dv|={norm(seed_tnb):.6f}")
    print(f"outer x0 days    : [{args.dt0_initial_days}, {args.dtf_initial_days}]")
    print(f"outer bounds     : dt0=[{args.dt0_min_days},{args.dt0_max_days}], dtf=[{args.dtf_min_days},{args.dtf_max_days}]")
    print(f"enable_dsm       : {args.enable_dsm} frac={args.dsm_frac}")
    print(f"seed_result      : {args.seed_result_json}")
    print(f"initial DSM      : {None if initial_dsm_levela is None else initial_dsm_levela.tolist()}")
    print(f"cache            : {t_min:.3f} .. {t_max:.3f} ({(t_max-t_min)/DAY_S:.3f} d)")

    ephem = build_ephemeris_cache(
        bodies=bodies_to_cache,
        observer=observer,
        frame=args.frame,
        t0=t_min,
        t1=t_max,
        step_s=args.ephem_cache_step_s,
    )

    # Control bounds around the seed, in navigation components.
    t_lo = float(seed_tnb[0] - args.t_trust)
    t_hi = float(seed_tnb[0] + args.t_trust)
    b_lo = float(seed_tnb[2] - args.b_trust)
    b_hi = float(seed_tnb[2] + args.b_trust)

    cfg = CommanderConfig(
        candidate=candidate,
        live_state=live,
        body_catalog=body_catalog,
        ephem=ephem,
        attractors=attractors,
        dep_body=dep_body,
        arr_body=arr_body,
        nav_body=nav_body,
        observer=observer,
        frame=args.frame,
        state_source=args.state_source,
        source_abs_s=source_abs_s,
        source_rel_r_raw_m=source_rel_r,
        source_rel_v_raw_m_s=source_rel_v,
        nominal_burn_abs_s=nominal_burn_abs_s,
        nominal_arr_s=nominal_arr_s,
        seed_tnb=seed_tnb,
        binormal_basis_sign=-1.0 if args.flip_binormal else 1.0,
        normal_basis_sign=-1.0 if args.flip_normal else 1.0,
        max_step_s=args.max_step_s,
        rtol=args.rtol,
        atol_pos_m=args.atol_pos_m,
        atol_vel_m_s=args.atol_vel_m_s,
        parking_rtol=args.parking_rtol,
        parking_max_step_s=args.parking_max_step_s,
        inner_iterations=args.inner_iterations,
        fd_dv_step_m_s=args.fd_dv_step_m_s,
        fd_dsm_step_m_s=args.fd_dsm_step_m_s,
        inner_position_scale_m=args.inner_position_scale_m,
        inner_burn_reg=args.inner_burn_reg,
        inner_dsm_reg=args.inner_dsm_reg,
        inner_burn_step_max_m_s=args.inner_burn_step_max_m_s,
        inner_dsm_step_max_m_s=args.inner_dsm_step_max_m_s,
        inner_target_km=args.inner_target_km,
        enable_dsm=args.enable_dsm,
        dsm_frac=args.dsm_frac,
        dsm_max_abs_m_s=args.dsm_max_abs_m_s,
        initial_dsm_levela_m_s=initial_dsm_levela,
        t_min=t_lo,
        t_max=t_hi,
        n_max_abs=args.n_max_abs,
        b_min=b_lo,
        b_max=b_hi,
        dv_max_m_s=args.dv_max_m_s,
        cost_position_scale_km=args.cost_position_scale_km,
        cost_dv_weight=args.cost_dv_weight,
        cost_dsm_weight=args.cost_dsm_weight,
        cost_epoch_weight=args.cost_epoch_weight,
    )

    obj = CommanderObjective(cfg, max_outer_evals=args.outer_maxevals, print_every=args.print_every)
    x0 = np.asarray([args.dt0_initial_days, args.dtf_initial_days], dtype=float)
    outer_bounds = [(args.dt0_min_days, args.dt0_max_days), (args.dtf_min_days, args.dtf_max_days)]

    t_start = time.time()
    if args.outer_method == "slsqp":
        res = minimize(
            obj,
            x0,
            method="SLSQP",
            bounds=outer_bounds,
            options={"maxiter": args.outer_maxiter, "ftol": args.outer_ftol, "eps": args.outer_eps, "disp": True},
        )
        print(f"[outer result] success={res.success} message={res.message} x={res.x} fun={res.fun}")
    elif args.outer_method == "powell":
        res = minimize(
            obj,
            x0,
            method="Powell",
            bounds=outer_bounds,
            options={"maxiter": args.outer_maxiter, "maxfev": args.outer_maxevals, "ftol": args.outer_ftol, "xtol": args.outer_eps, "disp": True},
        )
        print(f"[outer result] success={res.success} message={res.message} x={res.x} fun={res.fun}")
    else:
        res = differential_evolution(
            obj,
            bounds=outer_bounds,
            maxiter=args.outer_maxiter,
            popsize=args.de_popsize,
            polish=False,
            seed=args.seed,
            updating="immediate",
            workers=1,
            disp=True,
        )
        print(f"[outer result] success={res.success} message={res.message} x={res.x} fun={res.fun}")

    elapsed = time.time() - t_start
    if obj.best:
        print("\n=== BEST OUTER/INNER STM RESULT ===")
        print(json.dumps(json_safe(obj.best), indent=2))
    print(f"elapsed_s: {elapsed:.3f}")

    write_outputs(args.output_dir, rank, args.row_index0, cfg, obj, args, x0, outer_bounds)
    print(f"[OK] wrote {args.output_dir / 'outer_epoch_stm_commander_result.json'}")
    print(f"[OK] wrote {args.output_dir / f'rank_row{args.row_index0}_outer_epoch_stm_seed.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
