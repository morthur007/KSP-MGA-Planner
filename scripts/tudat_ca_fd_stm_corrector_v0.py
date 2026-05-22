#!/usr/bin/env python3
"""
tudat_spice_vcarelnav_fast_refine_v0.py

Fast SPICE-backed N-body refiner for KSP/Principia MGA departure seeds.

Purpose
-------
Use TudatPy as the SPICE provider for the Principia-generated BSP/TPC kernels,
then run a fast local N-body propagation/optimization in Python.

This is intended as a fast mid-fidelity targeter between:
  PyKEP/Lambert seed  ->  TudatPy/SPICE fast refine  ->  Principia VCAREL_NAV final validation

It does NOT replace the final Principia validator. It is a fast way to search
TNB/arrival/optional DSM space without repeatedly calling the Principia server.

Inputs expected from the current project:
  --rank-json       candidate_departure_executability_rank.json
  --row-index0      e.g. 15
  --live-state-json DLL snapshot JSON, used for metadata/mass and fallback state
  --bsp             Principia synthetic BSP
  --tpc             Principia ids/constants TPC
  --body-catalog    body_catalog.json with mu_m3_s2

For row15-style future burn states, the script uses candidate fields:
  burn_abs_s, burn_rel_r_raw_m, burn_rel_v_raw_m_s, t_arr_s,
  dv_tangent_m_s, dv_normal_m_s, dv_binormal_m_s

Frame convention
----------------
The SPICE kernel is assumed to be in LevelA/J2000 canonical frame.
Project raw -> LevelA conversion:
  raw (X,Y,Z) -> levela (-Y,+Z,+X)
LevelA -> raw:
  levela (X,Y,Z) -> raw (+Z,-X,+Y)

The TNB basis is computed in raw relative state, then dv_raw is converted to LevelA.
Use --flip-binormal for the known server/Principia binormal-sign mismatch.
Use --flip-normal only for diagnostic runs.

Notes
-----
This script uses TudatPy's SPICE interface, not TudatPy's dynamics simulator.
That keeps it lightweight and robust across TudatPy versions. Body states are
sampled once from SPICE and cached with cubic interpolation; spacecraft dynamics
are propagated with scipy.integrate.solve_ivp(DOP853).
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
from scipy.optimize import minimize, differential_evolution, minimize_scalar

try:
    from tudatpy.interface import spice
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "Could not import tudatpy.interface.spice. Install/activate tudatpy first. "
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


def parse_vec3(s: str | None, default: Sequence[float] | None = None) -> np.ndarray:
    if s is None:
        if default is None:
            raise ValueError("missing vector")
        return np.asarray(default, dtype=float)
    vals = [float(x.strip()) for x in s.split(",")]
    if len(vals) != 3:
        raise ValueError(f"expected 3 comma-separated values, got {s!r}")
    return np.asarray(vals, dtype=float)


def parse_body_list(s: str) -> list[str]:
    return [x.strip().upper() for x in s.split(",") if x.strip()]


def load_json(path: Path) -> Any:
    with path.open() as f:
        return json.load(f)


def load_body_catalog(path: Path) -> dict[str, dict[str, Any]]:
    data = load_json(path)
    if isinstance(data, dict) and "bodies" in data and isinstance(data["bodies"], list):
        out = {}
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
    for k in ("mu_m3_s2", "gravitational_parameter_m3_s2", "gravitational_parameter", "mu", "gm_m3_s2", "GM", "gm"):
        if k in b:
            return float(b[k])
    raise KeyError(f"cannot find mu for {key}; keys={sorted(b.keys())}")


def normalize_live_state(snapshot: dict[str, Any]) -> dict[str, Any]:
    """Accept either old flat active state or DLL snapshot with .vessel."""
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
        if matches:
            # Prefer the most complete record.
            matches.sort(key=lambda c: sum(k in c for k in (
                "burn_abs_s", "burn_rel_r_raw_m", "burn_rel_v_raw_m_s", "t_arr_s",
                "dv_tangent_m_s", "dv_normal_m_s", "dv_binormal_m_s",
            )), reverse=True)
            return matches[0]
        raise KeyError(f"row_index0={row_index0} not found")

    if isinstance(rank, dict) and "top" in rank and isinstance(rank["top"], list):
        return rank["top"][top_index]
    if candidates:
        return candidates[top_index]
    if isinstance(rank, dict):
        return rank
    raise ValueError("cannot find candidate in rank-json")


def tnb_basis_raw(r_raw: Sequence[float], v_raw: Sequence[float], binormal_sign: float = 1.0, normal_sign: float = 1.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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


def spice_state_safe(target: str, observer: str, frame: str, et: float) -> np.ndarray:
    target = target.upper()
    observer = observer.upper()
    if target == observer:
        return np.zeros(6)
    try:
        return np.asarray(spice.get_body_cartesian_state_at_epoch(target, observer, frame, "NONE", float(et)), dtype=float)
    except Exception:
        # Try title-case fallback for kernels that use conventional names.
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


@dataclass
class EvalConfig:
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
    state_abs_s: float
    rel_r_raw_m: np.ndarray
    rel_v_raw_m_s: np.ndarray
    t_arr_s: float
    scan_half_width_days: float
    binormal_basis_sign: float
    normal_basis_sign: float
    max_step_s: float
    rtol: float
    atol_pos_m: float
    atol_vel_m_s: float
    samples: int
    optimize_arrival_offset: bool
    enable_dsm: bool
    dsm_frac_bounds: tuple[float, float]
    score_ca_scale_km: float
    dv_weight: float
    dsm_weight: float
    arrival_weight: float


def initial_inertial_state_levela(cfg: EvalConfig, dvt: float, dvn: float, dvb: float) -> np.ndarray:
    dep_state = cfg.ephem.state(cfg.dep_body, cfg.state_abs_s)
    r0 = dep_state[:3] + raw_to_levela(cfg.rel_r_raw_m)
    v0_pre = dep_state[3:] + raw_to_levela(cfg.rel_v_raw_m_s)
    dv_raw = dv_nav_to_raw(
        cfg.rel_r_raw_m,
        cfg.rel_v_raw_m_s,
        dvt,
        dvn,
        dvb,
        binormal_basis_sign=cfg.binormal_basis_sign,
        normal_basis_sign=cfg.normal_basis_sign,
    )
    v0 = v0_pre + raw_to_levela(dv_raw)
    return np.r_[r0, v0]


def propagate_and_ca(
    cfg: EvalConfig,
    dvt: float,
    dvn: float,
    dvb: float,
    arrival_offset_days: float,
    dsm_frac: float | None = None,
    dsm_levela_m_s: np.ndarray | None = None,
) -> dict[str, Any]:
    t0 = cfg.state_abs_s
    scan_center = (cfg.t_arr_s - t0) + float(arrival_offset_days) * DAY_S
    scan_start = max(0.0, scan_center - cfg.scan_half_width_days * DAY_S)
    scan_end = max(scan_start + 60.0, scan_center + cfg.scan_half_width_days * DAY_S)
    t_end = t0 + scan_end

    y0 = initial_inertial_state_levela(cfg, dvt, dvn, dvb)
    mus = {b: mu_for(cfg.body_catalog, b) for b in cfg.attractors}

    def rhs(t: float, y: np.ndarray) -> np.ndarray:
        r = y[:3]
        a = np.zeros(3)
        for body, mu in mus.items():
            rb = cfg.ephem.state(body, t)[:3]
            dr = rb - r
            d = float(np.linalg.norm(dr))
            if d <= 0:
                continue
            a += mu * dr / d**3
        return np.r_[y[3:], a]

    def integrate_segment(ta: float, tb: float, ya: np.ndarray):
        return solve_ivp(
            rhs,
            (float(ta), float(tb)),
            ya,
            method="DOP853",
            rtol=cfg.rtol,
            atol=[cfg.atol_pos_m] * 3 + [cfg.atol_vel_m_s] * 3,
            dense_output=True,
            max_step=cfg.max_step_s,
        )

    if cfg.enable_dsm and dsm_frac is not None and dsm_levela_m_s is not None and norm(dsm_levela_m_s) > 0:
        dsm_frac_clamped = clamp(float(dsm_frac), cfg.dsm_frac_bounds[0], cfg.dsm_frac_bounds[1])
        t_dsm = t0 + dsm_frac_clamped * (t_end - t0)
        sol1 = integrate_segment(t0, t_dsm, y0)
        if not sol1.success:
            raise RuntimeError(f"integrator failed before DSM: {sol1.message}")
        y_dsm = np.asarray(sol1.y[:, -1], dtype=float)
        y_dsm[3:] += np.asarray(dsm_levela_m_s, dtype=float)
        sol2 = integrate_segment(t_dsm, t_end, y_dsm)
        if not sol2.success:
            raise RuntimeError(f"integrator failed after DSM: {sol2.message}")

        def state_at(t: float) -> np.ndarray:
            if t <= t_dsm:
                return np.asarray(sol1.sol(t), dtype=float)
            return np.asarray(sol2.sol(t), dtype=float)
    else:
        sol = integrate_segment(t0, t_end, y0)
        if not sol.success:
            raise RuntimeError(f"integrator failed: {sol.message}")

        def state_at(t: float) -> np.ndarray:
            return np.asarray(sol.sol(t), dtype=float)

    ca_t0 = t0 + scan_start
    ca_t1 = t0 + scan_end
    ts = np.linspace(ca_t0, ca_t1, max(5, int(cfg.samples)))

    def dist2_at(t: float) -> float:
        y = state_at(t)
        arr = cfg.ephem.state(cfg.arr_body, t)
        dr = y[:3] - arr[:3]
        return float(np.dot(dr, dr))

    coarse = np.asarray([dist2_at(t) for t in ts], dtype=float)
    i = int(np.argmin(coarse))
    lo = ts[max(0, i - 1)]
    hi = ts[min(len(ts) - 1, i + 1)]
    if hi <= lo:
        ca_t = float(ts[i])
    else:
        opt = minimize_scalar(dist2_at, bounds=(float(lo), float(hi)), method="bounded", options={"xatol": 1e-3})
        ca_t = float(opt.x if opt.success else ts[i])

    y_ca = state_at(ca_t)
    arr_ca = cfg.ephem.state(cfg.arr_body, ca_t)
    rel = y_ca[:3] - arr_ca[:3]
    relv = y_ca[3:] - arr_ca[3:]
    ca_m = norm(rel)
    speed = norm(relv)
    radial = float(np.dot(rel, relv) / max(ca_m, 1e-12))

    dv0_norm = math.sqrt(dvt * dvt + dvn * dvn + dvb * dvb)
    dsm_norm = 0.0 if dsm_levela_m_s is None else norm(dsm_levela_m_s)
    score = (
        ca_m / 1000.0 / cfg.score_ca_scale_km
        + cfg.dv_weight * dv0_norm
        + cfg.dsm_weight * dsm_norm
        + cfg.arrival_weight * abs(float(arrival_offset_days))
    )

    return {
        "ok": True,
        "score": float(score),
        "ca_distance_m": float(ca_m),
        "ca_distance_km": float(ca_m / 1000.0),
        "ca_t_game_s": float(ca_t),
        "ca_speed_m_s": float(speed),
        "ca_radial_velocity_m_s": float(radial),
        "ca_rel_levela_m": [float(x) for x in rel],
        "ca_relv_levela_m_s": [float(x) for x in relv],
        "arrival_offset_days": float(arrival_offset_days),
        "scan_start_rel_s": float(scan_start),
        "scan_end_rel_s": float(scan_end),
        "dvt_m_s": float(dvt),
        "dvn_m_s": float(dvn),
        "dvb_m_s": float(dvb),
        "dv_norm_m_s": float(dv0_norm),
        "dsm_norm_m_s": float(dsm_norm),
        "dsm_frac": None if dsm_frac is None else float(dsm_frac),
        "dsm_levela_m_s": None if dsm_levela_m_s is None else [float(x) for x in dsm_levela_m_s],
    }



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


def flatten(row: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in row.items():
        if isinstance(v, list):
            if all(not isinstance(a, (list, dict)) for a in v):
                for i, a in enumerate(v):
                    out[f"{k}_{i}"] = a
            else:
                out[k] = json.dumps(v)
        elif isinstance(v, dict):
            out[k] = json.dumps(v)
        else:
            out[k] = v
    return out


def get_best_record(result: dict[str, Any]) -> dict[str, Any]:
    if isinstance(result.get("best"), dict) and result["best"].get("ok", True):
        return result["best"]
    if isinstance(result.get("top"), list) and result["top"]:
        return result["top"][0]
    if isinstance(result, dict) and "dvt_m_s" in result:
        return result
    raise ValueError("cannot find best record in seed-result-json")


def solve_damped_ls(A: np.ndarray, b: np.ndarray, lm: float, reg_weights: np.ndarray | None = None) -> np.ndarray:
    H = A.T @ A
    g = A.T @ b
    diag_scale = max(1.0, float(np.max(np.diag(H))) if H.size else 1.0)
    R = np.eye(A.shape[1])
    if reg_weights is not None:
        R = np.diag(reg_weights)
    return -np.linalg.solve(H + lm * diag_scale * R, g)


def clip_vec_norm(v: np.ndarray, max_norm: float) -> np.ndarray:
    n = norm(v)
    if max_norm > 0 and n > max_norm:
        return v * (max_norm / n)
    return v


def main() -> int:
    ap = argparse.ArgumentParser(description="CA-event finite-difference STM corrector over Tudat/SPICE propagation.")
    ap.add_argument("--rank-json", type=Path, required=True)
    ap.add_argument("--row-index0", type=int, default=None)
    ap.add_argument("--top-index", type=int, default=0)
    ap.add_argument("--live-state-json", type=Path, required=True)
    ap.add_argument("--seed-result-json", type=Path, default=None, help="Result JSON from tudat_spice_vcarelnav_fast_refine_v0.py; uses .best or .top[0] as warm start.")
    ap.add_argument("--bsp", type=Path, required=True)
    ap.add_argument("--tpc", type=Path, default=None)
    ap.add_argument("--body-catalog", type=Path, required=True)
    ap.add_argument("--dep-body", default="KERBIN")
    ap.add_argument("--arr-body", default="EVE")
    ap.add_argument("--nav-body", default="KERBIN")
    ap.add_argument("--attractors", default="SUN,KERBIN,EVE")
    ap.add_argument("--observer", default="SUN")
    ap.add_argument("--frame", default="J2000")
    ap.add_argument("--spice-time-offset-s", type=float, default=0.0)
    ap.add_argument("--state-source", choices=["burnstate", "snapshot"], default="burnstate")
    ap.add_argument("--flip-normal", action="store_true")
    ap.add_argument("--flip-binormal", action="store_true")
    ap.add_argument("--binormal-basis-sign", type=float, default=1.0)
    ap.add_argument("--normal-basis-sign", type=float, default=1.0)
    ap.add_argument("--arrival-offset-days", type=float, default=None, help="Override arrival offset; default from seed result or 0.")
    ap.add_argument("--scan-half-width-days", type=float, default=20.0)
    ap.add_argument("--samples", type=int, default=61)
    ap.add_argument("--t-trust", type=float, default=500.0)
    ap.add_argument("--n-max-abs", type=float, default=1200.0)
    ap.add_argument("--b-trust", type=float, default=1200.0)
    ap.add_argument("--enable-dsm", action="store_true")
    ap.add_argument("--dsm-frac", type=float, default=None, help="Fixed DSM fraction. Default from seed result, else 0.5.")
    ap.add_argument("--dsm-max-abs", type=float, default=300.0)
    ap.add_argument("--dsm-initial-levela", default=None, help="Override DSM seed as x,y,z in LevelA m/s.")
    ap.add_argument("--iterations", type=int, default=8)
    ap.add_argument("--fd-dv-step-m-s", type=float, default=2.0)
    ap.add_argument("--fd-dsm-step-m-s", type=float, default=2.0)
    ap.add_argument("--burn-step-max-m-s", type=float, default=80.0)
    ap.add_argument("--dsm-step-max-m-s", type=float, default=80.0)
    ap.add_argument("--lm", type=float, default=1e-6)
    ap.add_argument("--lm-up", type=float, default=10.0)
    ap.add_argument("--lm-down", type=float, default=0.3)
    ap.add_argument("--line-search", default="1,0.5,0.25,0.125,0.0625")
    ap.add_argument("--target-ca-km", type=float, default=50000.0)
    ap.add_argument("--ephem-cache-step-s", type=float, default=21600.0)
    ap.add_argument("--max-step-s", type=float, default=21600.0)
    ap.add_argument("--rtol", type=float, default=1e-9)
    ap.add_argument("--atol-pos-m", type=float, default=1.0)
    ap.add_argument("--atol-vel-m-s", type=float, default=1e-6)
    ap.add_argument("--print-every-eval", action="store_true")
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    spice.clear_kernels()
    if args.tpc:
        spice.load_kernel(str(args.tpc))
    spice.load_kernel(str(args.bsp))

    rank = load_json(args.rank_json)
    candidate = find_candidate(rank, args.row_index0, args.top_index)
    live_state = normalize_live_state(load_json(args.live_state_json))
    body_catalog = load_body_catalog(args.body_catalog)

    dep_body = args.dep_body.upper(); arr_body = args.arr_body.upper(); nav_body = args.nav_body.upper()
    attractors = parse_body_list(args.attractors); observer = args.observer.upper()

    if args.state_source == "burnstate" and "burn_rel_r_raw_m" in candidate:
        state_abs_s = float(candidate.get("burn_abs_s", candidate.get("t_dep_s"))) + args.spice_time_offset_s
        rel_r_raw = np.asarray(candidate["burn_rel_r_raw_m"], dtype=float)
        rel_v_raw = np.asarray(candidate["burn_rel_v_raw_m_s"], dtype=float)
    else:
        state_abs_s = float(live_state.get("t_spice_s", live_state.get("t_game_s"))) + args.spice_time_offset_s
        rel_r_raw = np.asarray(live_state["rel_r_raw_m"], dtype=float)
        rel_v_raw = np.asarray(live_state["rel_v_raw_m_s"], dtype=float)
    t_arr_s = float(candidate["t_arr_s"]) + args.spice_time_offset_s

    seed_tnb = get_tnb_seed(candidate, args.flip_normal, args.flip_binormal)
    arrival_offset = 0.0
    dsm_frac = 0.5
    dsm = np.zeros(3)
    if args.seed_result_json:
        rec = get_best_record(load_json(args.seed_result_json))
        if "dvt_m_s" in rec:
            seed_tnb = np.asarray([rec["dvt_m_s"], rec["dvn_m_s"], rec["dvb_m_s"]], dtype=float)
        if "arrival_offset_days" in rec and rec["arrival_offset_days"] is not None:
            arrival_offset = float(rec["arrival_offset_days"])
        if "dsm_frac" in rec and rec["dsm_frac"] is not None:
            dsm_frac = float(rec["dsm_frac"])
        if "dsm_levela_m_s" in rec and rec["dsm_levela_m_s"] is not None:
            dsm = np.asarray(rec["dsm_levela_m_s"], dtype=float)
    if args.arrival_offset_days is not None:
        arrival_offset = float(args.arrival_offset_days)
    if args.dsm_frac is not None:
        dsm_frac = float(args.dsm_frac)
    if args.dsm_initial_levela is not None:
        dsm = parse_vec3(args.dsm_initial_levela, [0,0,0])

    # Build cache with enough padding around fixed scan window.
    scan_center = t_arr_s + arrival_offset * DAY_S
    t_cache_end = max(scan_center + args.scan_half_width_days * DAY_S, state_abs_s + 60.0)
    bodies_to_cache = set(attractors + [dep_body, arr_body, nav_body, observer])
    print("=== CA-FD STM CACHE ===")
    print(f"bodies     : {sorted(bodies_to_cache)}")
    print(f"t0..t1     : {state_abs_s:.3f} .. {t_cache_end:.3f} ({(t_cache_end-state_abs_s)/DAY_S:.3f} d)")
    ephem = build_ephemeris_cache(bodies_to_cache, observer, args.frame, state_abs_s, t_cache_end, args.ephem_cache_step_s)

    cfg = EvalConfig(
        candidate=candidate, live_state=live_state, body_catalog=body_catalog, ephem=ephem,
        attractors=attractors, dep_body=dep_body, arr_body=arr_body, nav_body=nav_body,
        observer=observer, frame=args.frame, state_abs_s=state_abs_s, rel_r_raw_m=rel_r_raw,
        rel_v_raw_m_s=rel_v_raw, t_arr_s=t_arr_s, scan_half_width_days=args.scan_half_width_days,
        binormal_basis_sign=args.binormal_basis_sign, normal_basis_sign=args.normal_basis_sign,
        max_step_s=args.max_step_s, rtol=args.rtol, atol_pos_m=args.atol_pos_m,
        atol_vel_m_s=args.atol_vel_m_s, samples=args.samples, optimize_arrival_offset=False,
        enable_dsm=args.enable_dsm, dsm_frac_bounds=(max(0.0, dsm_frac), min(1.0, dsm_frac)),
        score_ca_scale_km=100000.0, dv_weight=0.0, dsm_weight=0.0, arrival_weight=0.0,
    )

    x0 = seed_tnb.copy()
    if args.enable_dsm:
        x0 = np.r_[x0, dsm]
    bounds = [(seed_tnb[0]-args.t_trust, seed_tnb[0]+args.t_trust), (-args.n_max_abs, args.n_max_abs), (seed_tnb[2]-args.b_trust, seed_tnb[2]+args.b_trust)]
    if args.enable_dsm:
        bounds += [(-args.dsm_max_abs, args.dsm_max_abs)]*3

    def clip_x(x: np.ndarray) -> np.ndarray:
        y = np.asarray(x, dtype=float).copy()
        for i,(lo,hi) in enumerate(bounds):
            y[i] = clamp(y[i], lo, hi)
        return y

    eval_rows: list[dict[str, Any]] = []
    iter_rows: list[dict[str, Any]] = []
    eval_count = 0

    def evaluate(x: np.ndarray, label: str) -> dict[str, Any]:
        nonlocal eval_count
        eval_count += 1
        x = clip_x(x)
        tnb = x[:3]
        dsm_vec = x[3:6] if args.enable_dsm else None
        row = propagate_and_ca(cfg, float(tnb[0]), float(tnb[1]), float(tnb[2]), arrival_offset, dsm_frac if args.enable_dsm else None, dsm_vec)
        row.update({"eval": eval_count, "label": label, "x": [float(v) for v in x]})
        eval_rows.append(row)
        if args.print_every_eval:
            print(f"[eval {eval_count:04d}] {label:12s} ca={row['ca_distance_km']:12.3f} km TNB=[{tnb[0]:.3f},{tnb[1]:.3f},{tnb[2]:.3f}] dsm={row['dsm_norm_m_s']:.3f}")
        return row

    x = clip_x(x0)
    lm = float(args.lm)
    alphas = [float(a) for a in args.line_search.split(',') if a.strip()]
    best_row = evaluate(x, "initial")
    best_x = x.copy()
    print("=== INITIAL ===")
    print(json.dumps({k: best_row.get(k) for k in ("ca_distance_km","ca_t_game_s","ca_speed_m_s","dvt_m_s","dvn_m_s","dvb_m_s","dsm_norm_m_s","dsm_frac")}, indent=2))

    for it in range(1, args.iterations+1):
        base = evaluate(x, f"base_{it}")
        rel0 = np.asarray(base["ca_rel_levela_m"], dtype=float)
        base_ca = float(base["ca_distance_km"])
        nvar = len(x)
        A = np.zeros((3, nvar))
        for j in range(nvar):
            step = args.fd_dv_step_m_s if j < 3 else args.fd_dsm_step_m_s
            xp = x.copy(); xp[j] += step; xp = clip_x(xp)
            rp = evaluate(xp, f"fd{it}_{j}")
            relp = np.asarray(rp["ca_rel_levela_m"], dtype=float)
            actual = xp[j] - x[j]
            if abs(actual) < 1e-12:
                A[:, j] = 0.0
            else:
                A[:, j] = (relp - rel0) / actual
        reg_weights = np.ones(nvar)
        if args.enable_dsm:
            reg_weights[3:] = 0.5
        delta = solve_damped_ls(A, rel0, lm, reg_weights)
        delta[:3] = clip_vec_norm(delta[:3], args.burn_step_max_m_s)
        if args.enable_dsm:
            delta[3:6] = clip_vec_norm(delta[3:6], args.dsm_step_max_m_s)
        chosen = None
        for alpha in alphas:
            xt = clip_x(x + alpha * delta)
            tr = evaluate(xt, f"trial{it}_{alpha:g}")
            if chosen is None or float(tr["ca_distance_km"]) < float(chosen["row"]["ca_distance_km"]):
                chosen = {"alpha": alpha, "x": xt, "row": tr}
        assert chosen is not None
        improved = float(chosen["row"]["ca_distance_km"]) < base_ca
        if improved:
            x = chosen["x"]
            lm = max(1e-12, lm * args.lm_down)
            if float(chosen["row"]["ca_distance_km"]) < float(best_row["ca_distance_km"]):
                best_row = chosen["row"]
                best_x = x.copy()
        else:
            lm *= args.lm_up
        iter_row = {
            "iteration": it, "base_ca_km": base_ca, "chosen_ca_km": float(chosen["row"]["ca_distance_km"]),
            "best_ca_km": float(best_row["ca_distance_km"]), "improved": improved, "alpha": chosen["alpha"],
            "lm": lm, "delta_norm_burn_m_s": norm(delta[:3]), "delta_norm_dsm_m_s": norm(delta[3:6]) if args.enable_dsm else 0.0,
            "x": [float(v) for v in x],
        }
        iter_rows.append(iter_row)
        print(f"[iter {it:02d}] base={base_ca:12.3f} km chosen={chosen['row']['ca_distance_km']:12.3f} km best={best_row['ca_distance_km']:12.3f} km alpha={chosen['alpha']} lm={lm:.3g} dBurn={iter_row['delta_norm_burn_m_s']:.3f} dDSM={iter_row['delta_norm_dsm_m_s']:.3f}")
        if float(best_row["ca_distance_km"]) <= args.target_ca_km:
            break

    # Write patched rank seed.
    rank_out = json.loads(json.dumps(rank))
    def patch_rank(obj: Any) -> None:
        if isinstance(obj, dict):
            if obj.get("row_index0") == candidate.get("row_index0"):
                obj["dv_tangent_m_s"] = float(best_row["dvt_m_s"])
                obj["dv_normal_m_s"] = float(best_row["dvn_m_s"])
                obj["dv_binormal_m_s"] = float(best_row["dvb_m_s"])
                obj["dv_norm_m_s"] = float(best_row["dv_norm_m_s"])
                obj["tudat_ca_fd_stm_patch"] = {
                    "ca_distance_km": float(best_row["ca_distance_km"]),
                    "ca_t_game_s": float(best_row["ca_t_game_s"]),
                    "arrival_offset_days": arrival_offset,
                    "dsm_frac": dsm_frac if args.enable_dsm else None,
                    "dsm_levela_m_s": best_row.get("dsm_levela_m_s"),
                }
            for v in obj.values(): patch_rank(v)
        elif isinstance(obj, list):
            for v in obj: patch_rank(v)
    patch_rank(rank_out)

    top = sorted([r for r in eval_rows if r.get("ok")], key=lambda r: r["ca_distance_km"])
    result = {
        "schema": "tudat_ca_fd_stm_corrector_v0",
        "note": "Finite-difference CA-event STM corrector. TudatPy is used as SPICE provider; spacecraft dynamics via scipy DOP853.",
        "config": json_safe(vars(args)),
        "state_abs_s": state_abs_s,
        "t_arr_s": t_arr_s,
        "arrival_offset_days": arrival_offset,
        "dsm_frac": dsm_frac if args.enable_dsm else None,
        "n_eval": eval_count,
        "best": best_row,
        "top": top[:50],
        "iterations": iter_rows,
        "rows": eval_rows,
    }
    (args.output_dir / "tudat_ca_fd_stm_corrector_result.json").write_text(json.dumps(json_safe(result), indent=2)+"\n")
    (args.output_dir / "rank_row15_tudat_ca_fd_stm_seed.json").write_text(json.dumps(json_safe(rank_out), indent=2)+"\n")
    flat = [flatten(r) for r in eval_rows]
    if flat:
        fields = sorted({k for r in flat for k in r})
        with (args.output_dir / "tudat_ca_fd_stm_corrector_rows.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(flat)
    flat_iter = [flatten(r) for r in iter_rows]
    if flat_iter:
        fields = sorted({k for r in flat_iter for k in r})
        with (args.output_dir / "tudat_ca_fd_stm_corrector_iterations.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(flat_iter)

    print("\n=== BEST CA-FD STM RESULT ===")
    print(json.dumps(json_safe({k: best_row.get(k) for k in ("ca_distance_km","ca_t_game_s","ca_speed_m_s","ca_radial_velocity_m_s","dvt_m_s","dvn_m_s","dvb_m_s","dv_norm_m_s","dsm_norm_m_s","dsm_frac","dsm_levela_m_s")}), indent=2))
    print(f"[OK] wrote {args.output_dir / 'tudat_ca_fd_stm_corrector_result.json'}")
    print(f"[OK] wrote {args.output_dir / 'rank_row15_tudat_ca_fd_stm_seed.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
