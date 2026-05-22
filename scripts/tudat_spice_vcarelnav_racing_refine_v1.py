#!/usr/bin/env python3
"""
tudat_spice_vcarelnav_racing_refine_v1.py

Fast staged/racing refiner for KSP/Principia MGA departure seeds.

This is designed to replace long Powell-only runs. It uses the same backend as
`tudat_spice_vcarelnav_fast_refine_v0.py` but changes the search strategy:

  1. Coarse Sobol/Latin-hypercube screening over T,N,B, arrival offset and optional DSM.
  2. Successive-halving: only the best candidates are re-evaluated at higher fidelity.
  3. Local jitter around the best candidates.
  4. Very limited Powell polish only for the best few, not every random start.

The goal is to spend almost no time polishing bad burns.

Assumptions:
  - Put this file in scripts/ next to tudat_spice_vcarelnav_fast_refine_v0.py.
  - The base script is the fixed version that accepts `gravitational_parameter` in body_catalog.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy.optimize import minimize
try:
    from scipy.stats import qmc
except Exception:  # pragma: no cover
    qmc = None

# Import the existing fixed fast-refine backend from the same scripts/ directory.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import tudat_spice_vcarelnav_fast_refine_v0 as base  # type: ignore

DAY_S = base.DAY_S


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


def parse_float_list(s: str | None) -> list[float]:
    if not s:
        return []
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def clip_to_bounds(x: np.ndarray, bounds: list[tuple[float, float]]) -> np.ndarray:
    y = np.asarray(x, dtype=float).copy()
    for i, (lo, hi) in enumerate(bounds):
        y[i] = min(max(float(y[i]), float(lo)), float(hi))
    return y


def bounds_width(bounds: list[tuple[float, float]]) -> np.ndarray:
    return np.asarray([hi - lo for lo, hi in bounds], dtype=float)


def scale_unit_to_bounds(u: np.ndarray, bounds: list[tuple[float, float]]) -> np.ndarray:
    lo = np.asarray([b[0] for b in bounds], dtype=float)
    hi = np.asarray([b[1] for b in bounds], dtype=float)
    return lo + np.asarray(u, dtype=float) * (hi - lo)


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


def find_best_from_result(path: Path) -> dict[str, Any] | None:
    if not path or not path.exists():
        return None
    data = json.loads(path.read_text())
    if isinstance(data.get("best"), dict) and data["best"].get("ok"):
        return data["best"]
    top = data.get("top")
    if isinstance(top, list) and top:
        return top[0]
    return None


def make_x_from_row(row: dict[str, Any], cfg: base.EvalConfig, enable_dsm: bool) -> np.ndarray:
    parts = [
        float(row["dvt_m_s"]),
        float(row["dvn_m_s"]),
        float(row["dvb_m_s"]),
    ]
    if cfg.optimize_arrival_offset:
        parts.append(float(row.get("arrival_offset_days", 0.0)))
    if enable_dsm:
        parts.append(float(row.get("dsm_frac", 0.5) if row.get("dsm_frac") is not None else 0.5))
        dsm = row.get("dsm_levela_m_s") or [0.0, 0.0, 0.0]
        parts.extend([float(v) for v in dsm])
    return np.asarray(parts, dtype=float)


def make_dsm_ball_vector(rng: np.random.Generator, max_norm: float) -> np.ndarray:
    if max_norm <= 0:
        return np.zeros(3)
    v = rng.normal(size=3)
    n = np.linalg.norm(v)
    if n <= 0:
        return np.zeros(3)
    # Uniform-ish in volume, not just on shell.
    radius = float(max_norm) * (rng.random() ** (1.0 / 3.0))
    return radius * v / n


def sobol_or_random(n: int, dim: int, seed: int) -> np.ndarray:
    if n <= 0:
        return np.zeros((0, dim))
    if qmc is not None:
        # Sobol likes powers of two, but random_base2 would force exact powers.
        sampler = qmc.Sobol(d=dim, scramble=True, seed=seed)
        m = int(math.ceil(math.log2(max(1, n))))
        return sampler.random_base2(m)[:n]
    rng = np.random.default_rng(seed)
    return rng.random((n, dim))


def build_x0_bounds_cfg(args: argparse.Namespace):
    # Load SPICE through the base module's TudatPy spice object.
    base.spice.clear_kernels()
    if args.tpc:
        base.spice.load_kernel(str(args.tpc))
    base.spice.load_kernel(str(args.bsp))
    print(f"[spice] kernels loaded: {base.spice.get_total_count_of_kernels_loaded()}")

    rank = base.load_json(args.rank_json)
    candidate = base.find_candidate(rank, args.row_index0, args.top_index)
    live_state = base.normalize_live_state(base.load_json(args.live_state_json))
    body_catalog = base.load_body_catalog(args.body_catalog)

    dep_body = args.dep_body.upper()
    arr_body = args.arr_body.upper()
    nav_body = args.nav_body.upper()
    attractors = base.parse_body_list(args.attractors)
    observer = args.observer.upper()

    if args.state_source == "burnstate" and "burn_rel_r_raw_m" in candidate:
        state_abs_s = float(candidate.get("burn_abs_s", candidate.get("t_dep_s"))) + args.spice_time_offset_s
        rel_r_raw = np.asarray(candidate["burn_rel_r_raw_m"], dtype=float)
        rel_v_raw = np.asarray(candidate["burn_rel_v_raw_m_s"], dtype=float)
    else:
        state_abs_s = float(live_state.get("t_spice_s", live_state.get("t_game_s"))) + args.spice_time_offset_s
        rel_r_raw = np.asarray(live_state["rel_r_raw_m"], dtype=float)
        rel_v_raw = np.asarray(live_state["rel_v_raw_m_s"], dtype=float)

    if "t_arr_s" not in candidate:
        raise SystemExit("candidate lacks t_arr_s")
    t_arr_s = float(candidate["t_arr_s"]) + args.spice_time_offset_s

    seed_tnb = base.get_tnb_seed(candidate, args.flip_normal, args.flip_binormal)
    x0_parts = [seed_tnb]
    if args.optimize_arrival_offset:
        x0_parts.append(np.asarray([args.arrival_offset_initial_days], dtype=float))
    if args.enable_dsm:
        x0_parts.append(np.asarray([args.dsm_frac_initial], dtype=float))
        x0_parts.append(base.parse_vec3(args.dsm_initial_levela, [0, 0, 0]))
    x0 = np.concatenate(x0_parts)
    bounds = base.make_bounds(seed_tnb, args)
    x0 = clip_to_bounds(x0, bounds)

    t_cache_end = max(
        t_arr_s + args.arrival_offset_max_days * DAY_S + args.scan_half_width_days * DAY_S,
        state_abs_s + 60.0,
    )
    bodies_to_cache = set(attractors + [dep_body, arr_body, nav_body, observer])
    print("=== TUDAT/SPICE CACHE ===")
    print(f"bodies     : {sorted(bodies_to_cache)}")
    print(f"t0..t1     : {state_abs_s:.3f} .. {t_cache_end:.3f} ({(t_cache_end-state_abs_s)/DAY_S:.3f} d)")
    print(f"cache step : {args.ephem_cache_step_s} s")
    ephem = base.build_ephemeris_cache(
        bodies_to_cache,
        observer,
        args.frame,
        state_abs_s,
        t_cache_end,
        args.ephem_cache_step_s,
    )

    cfg = base.EvalConfig(
        candidate=candidate,
        live_state=live_state,
        body_catalog=body_catalog,
        ephem=ephem,
        attractors=attractors,
        dep_body=dep_body,
        arr_body=arr_body,
        nav_body=nav_body,
        observer=observer,
        frame=args.frame,
        state_abs_s=state_abs_s,
        rel_r_raw_m=rel_r_raw,
        rel_v_raw_m_s=rel_v_raw,
        t_arr_s=t_arr_s,
        scan_half_width_days=args.scan_half_width_days,
        binormal_basis_sign=args.binormal_basis_sign,
        normal_basis_sign=args.normal_basis_sign,
        max_step_s=args.max_step_s,
        rtol=args.rtol,
        atol_pos_m=args.atol_pos_m,
        atol_vel_m_s=args.atol_vel_m_s,
        samples=args.samples,
        optimize_arrival_offset=args.optimize_arrival_offset,
        enable_dsm=args.enable_dsm,
        dsm_frac_bounds=(args.dsm_frac_min, args.dsm_frac_max),
        score_ca_scale_km=args.ca_scale_km,
        dv_weight=args.dv_weight,
        dsm_weight=args.dsm_weight,
        arrival_weight=args.arrival_weight,
    )

    print("=== INITIAL SEED ===")
    print(f"row_index0    : {candidate.get('row_index0')}")
    print(f"state_abs_s   : {state_abs_s}")
    print(f"t_arr_s       : {t_arr_s}  TOF={(t_arr_s-state_abs_s)/DAY_S:.6f} d")
    print(f"seed TNB      : {seed_tnb.tolist()} |dv|={base.norm(seed_tnb):.6f} m/s")
    print(f"bounds        : {bounds}")
    print("=== RACING CONFIG ===")
    print(f"screen_n      : {args.screen_n}")
    print(f"screen_top_k  : {args.screen_top_k}")
    print(f"jitter_n      : {args.jitter_n}")
    print(f"polish_top_k  : {args.polish_top_k}")
    return cfg, x0, bounds, candidate, live_state


class RacingEvaluator:
    def __init__(self, args: argparse.Namespace, base_cfg: base.EvalConfig, bounds: list[tuple[float, float]]):
        self.args = args
        self.base_cfg = base_cfg
        self.bounds = bounds
        self.rows: list[dict[str, Any]] = []
        self.best: dict[str, Any] | None = None
        self.evals = 0
        self.stop_reason: str | None = None
        self.start_wall = time.time()
        self.seen: set[tuple[float, ...]] = set()

    def cfg_for_stage(self, stage: str) -> base.EvalConfig:
        if stage == "screen":
            return replace(
                self.base_cfg,
                samples=self.args.screen_samples,
                max_step_s=self.args.screen_max_step_s,
                rtol=self.args.screen_rtol,
                atol_pos_m=self.args.screen_atol_pos_m,
                atol_vel_m_s=self.args.screen_atol_vel_m_s,
                scan_half_width_days=self.args.screen_scan_half_width_days or self.base_cfg.scan_half_width_days,
            )
        if stage == "medium":
            return replace(
                self.base_cfg,
                samples=self.args.medium_samples,
                max_step_s=self.args.medium_max_step_s,
                rtol=self.args.medium_rtol,
                atol_pos_m=self.args.medium_atol_pos_m,
                atol_vel_m_s=self.args.medium_atol_vel_m_s,
            )
        return self.base_cfg

    def unpack(self, cfg: base.EvalConfig, x: Sequence[float]):
        xx = np.asarray(x, dtype=float)
        dvt, dvn, dvb = map(float, xx[:3])
        idx = 3
        if cfg.optimize_arrival_offset:
            arr = float(xx[idx]); idx += 1
        else:
            arr = 0.0
        dsm_frac = None
        dsm = None
        if cfg.enable_dsm:
            dsm_frac = float(xx[idx]); idx += 1
            dsm = np.asarray(xx[idx:idx + 3], dtype=float)
        return dvt, dvn, dvb, arr, dsm_frac, dsm

    def key(self, x: Sequence[float]) -> tuple[float, ...]:
        return tuple(round(float(v), 6) for v in x)

    def evaluate(self, x: Sequence[float], stage: str, origin: str = "") -> dict[str, Any]:
        if self.args.max_total_evals > 0 and self.evals >= self.args.max_total_evals:
            self.stop_reason = f"max_total_evals={self.args.max_total_evals}"
            return {"ok": False, "score": 1e99, "stage": stage, "origin": origin, "error": self.stop_reason}
        if self.args.max_wall_s > 0 and (time.time() - self.start_wall) >= self.args.max_wall_s:
            self.stop_reason = f"max_wall_s={self.args.max_wall_s}"
            return {"ok": False, "score": 1e99, "stage": stage, "origin": origin, "error": self.stop_reason}

        xx = clip_to_bounds(np.asarray(x, dtype=float), self.bounds)
        k = self.key(xx)
        if k in self.seen and not self.args.allow_duplicate_evals:
            return {"ok": False, "score": 1e99, "stage": stage, "origin": origin, "error": "duplicate", "x": xx.tolist()}
        self.seen.add(k)

        self.evals += 1
        cfg = self.cfg_for_stage(stage)
        try:
            dvt, dvn, dvb, arr, dsm_frac, dsm = self.unpack(cfg, xx)
            row = base.propagate_and_ca(cfg, dvt, dvn, dvb, arr, dsm_frac, dsm)
        except Exception as exc:
            row = {"ok": False, "score": 1e99, "error": str(exc)}
        row["eval"] = self.evals
        row["stage"] = stage
        row["origin"] = origin
        row["x"] = [float(v) for v in xx]
        row["wall_s"] = time.time() - self.start_wall
        self.rows.append(row)

        if row.get("ok"):
            if self.best is None or float(row["ca_distance_km"]) < float(self.best.get("ca_distance_km", math.inf)):
                self.best = row
            if self.args.print_every and (self.evals % self.args.print_every == 0 or self.best is row):
                print(
                    f"[{stage:6s} {self.evals:04d}] ca={row['ca_distance_km']:12.3f} km "
                    f"TNB=[{row['dvt_m_s']:.3f},{row['dvn_m_s']:.3f},{row['dvb_m_s']:.3f}] "
                    f"arr={row['arrival_offset_days']:.3f} d "
                    f"dsm={row['dsm_norm_m_s']:.3f} origin={origin}"
                )
            if self.args.target_ca_km > 0 and row["ca_distance_km"] <= self.args.target_ca_km:
                self.stop_reason = f"target_ca_km={self.args.target_ca_km}"
        elif self.args.print_every and self.evals % self.args.print_every == 0:
            print(f"[{stage:6s} {self.evals:04d}] ERR {row.get('error')} origin={origin}")
        return row

    def ok_sorted(self, stage: str | None = None) -> list[dict[str, Any]]:
        rows = [r for r in self.rows if r.get("ok")]
        if stage:
            rows = [r for r in rows if r.get("stage") == stage]
        rows.sort(key=lambda r: float(r.get("ca_distance_km", math.inf)))
        return rows


def make_seed_cloud(args: argparse.Namespace, x0: np.ndarray, bounds: list[tuple[float, float]], cfg: base.EvalConfig) -> list[tuple[str, np.ndarray]]:
    rng = np.random.default_rng(args.seed)
    seeds: list[tuple[str, np.ndarray]] = []
    seeds.append(("x0", x0.copy()))

    # Include previous result seeds first, if available.
    for p in args.seed_result_json:
        best = find_best_from_result(Path(p))
        if best:
            seeds.append((f"prev:{Path(p).name}", clip_to_bounds(make_x_from_row(best, cfg, args.enable_dsm), bounds)))

    # Manual seeds: comma vector matching dimension.
    for i, s in enumerate(args.manual_x):
        vals = [float(v.strip()) for v in s.split(",") if v.strip()]
        if len(vals) != len(bounds):
            raise SystemExit(f"--manual-x #{i} has {len(vals)} values, expected {len(bounds)}")
        seeds.append((f"manual{i}", clip_to_bounds(np.asarray(vals, dtype=float), bounds)))

    n_random = max(0, args.screen_n - len(seeds))
    if n_random:
        u = sobol_or_random(n_random, len(bounds), args.seed)
        for i, ui in enumerate(u):
            x = scale_unit_to_bounds(ui, bounds)
            if cfg.enable_dsm and args.dsm_sample_ball:
                # Replace DSM components by a bounded ball vector to avoid wasting most samples in cube corners.
                idx = 3 + (1 if cfg.optimize_arrival_offset else 0)
                x[idx] = min(max(float(x[idx]), args.dsm_frac_min), args.dsm_frac_max)
                x[idx + 1:idx + 4] = make_dsm_ball_vector(rng, args.dsm_sample_max_norm or args.dsm_max_abs)
            seeds.append((f"sobol{i}", clip_to_bounds(x, bounds)))

    return seeds


def jitter_around(args: argparse.Namespace, x: np.ndarray, bounds: list[tuple[float, float]], n: int, label: str, seed_offset: int = 0) -> list[tuple[str, np.ndarray]]:
    rng = np.random.default_rng(args.seed + 1000 + seed_offset)
    w = bounds_width(bounds)
    out: list[tuple[str, np.ndarray]] = []
    for i in range(n):
        scale = args.jitter_scale
        y = np.asarray(x, dtype=float).copy()
        y += rng.normal(0.0, scale, size=len(bounds)) * w
        # Keep DSM fraction and arrival less noisy than DV components by default.
        idx = 3
        if args.optimize_arrival_offset:
            y[idx] = x[idx] + rng.normal(0.0, args.jitter_arrival_days); idx += 1
        if args.enable_dsm:
            y[idx] = x[idx] + rng.normal(0.0, args.jitter_dsm_frac); idx += 1
            if args.dsm_sample_ball:
                y[idx:idx+3] = x[idx:idx+3] + make_dsm_ball_vector(rng, args.jitter_dsm_m_s)
            else:
                y[idx:idx+3] = x[idx:idx+3] + rng.normal(0.0, args.jitter_dsm_m_s, size=3)
        out.append((f"{label}_j{i}", clip_to_bounds(y, bounds)))
    return out


def bounded_penalty(x: np.ndarray, bounds: list[tuple[float, float]]) -> float:
    pen = 0.0
    for v, (lo, hi) in zip(x, bounds):
        if v < lo:
            pen += (lo - v) ** 2
        elif v > hi:
            pen += (v - hi) ** 2
    return pen


def run_limited_powell(ev: RacingEvaluator, start: np.ndarray, maxfev: int, label: str) -> None:
    local_evals = 0

    def fun(x: Sequence[float]) -> float:
        nonlocal local_evals
        if local_evals >= maxfev or ev.stop_reason:
            return 1e99
        xx = np.asarray(x, dtype=float)
        pen = bounded_penalty(xx, ev.bounds)
        if pen > 0:
            return 1e9 + pen
        row = ev.evaluate(xx, "polish", label)
        local_evals += 1
        return float(row.get("score", 1e99))

    minimize(
        fun,
        clip_to_bounds(start, ev.bounds),
        method="Powell",
        bounds=ev.bounds,
        options={"maxfev": max(1, int(maxfev)), "maxiter": max(1, int(maxfev)), "xtol": 1e-3, "ftol": 1e-6},
    )


def write_outputs(args: argparse.Namespace, cfg: base.EvalConfig, x0: np.ndarray, bounds: list[tuple[float, float]], ev: RacingEvaluator) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ok_rows = ev.ok_sorted()
    result = {
        "schema": "tudat_spice_vcarelnav_racing_refine_v1",
        "note": "Successive-halving/Sobol racing over the same TudatPy-SPICE backend as fast_refine_v0.",
        "config": json_safe(vars(args)),
        "state_abs_s": cfg.state_abs_s,
        "t_arr_s": cfg.t_arr_s,
        "x0": x0.tolist(),
        "bounds": [[float(a), float(b)] for a, b in bounds],
        "n_rows": len(ev.rows),
        "n_ok": len(ok_rows),
        "stop_reason": ev.stop_reason,
        "best": ok_rows[0] if ok_rows else None,
        "top": ok_rows[:100],
        "rows": ev.rows,
    }
    (args.output_dir / "tudat_spice_vcarelnav_racing_refine_result.json").write_text(json.dumps(json_safe(result), indent=2) + "\n")

    flat = [flatten(r) for r in ev.rows]
    if flat:
        fields = sorted({k for r in flat for k in r.keys()})
        with (args.output_dir / "tudat_spice_vcarelnav_racing_refine_rows.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(flat)

    print("\n=== BEST RACING REFINE ===")
    if ok_rows:
        b = ok_rows[0]
        print(json.dumps({
            "ca_distance_km": b.get("ca_distance_km"),
            "ca_t_game_s": b.get("ca_t_game_s"),
            "ca_speed_m_s": b.get("ca_speed_m_s"),
            "dvt_m_s": b.get("dvt_m_s"),
            "dvn_m_s": b.get("dvn_m_s"),
            "dvb_m_s": b.get("dvb_m_s"),
            "dv_norm_m_s": b.get("dv_norm_m_s"),
            "arrival_offset_days": b.get("arrival_offset_days"),
            "dsm_norm_m_s": b.get("dsm_norm_m_s"),
            "dsm_frac": b.get("dsm_frac"),
            "dsm_levela_m_s": b.get("dsm_levela_m_s"),
            "stage": b.get("stage"),
            "origin": b.get("origin"),
        }, indent=2))
    else:
        print("No valid evaluations.")
    print(f"[OK] wrote {args.output_dir / 'tudat_spice_vcarelnav_racing_refine_result.json'}")
    print(f"[OK] wrote {args.output_dir / 'tudat_spice_vcarelnav_racing_refine_rows.csv'}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)

    # Same core inputs as fast_refine_v0.
    ap.add_argument("--rank-json", type=Path, required=True)
    ap.add_argument("--row-index0", type=int, default=None)
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
    ap.add_argument("--spice-time-offset-s", type=float, default=0.0)

    ap.add_argument("--state-source", choices=["burnstate", "snapshot"], default="burnstate")
    ap.add_argument("--arrival-offset-initial-days", type=float, default=0.0)
    ap.add_argument("--optimize-arrival-offset", action="store_true")
    ap.add_argument("--arrival-offset-min-days", type=float, default=-60.0)
    ap.add_argument("--arrival-offset-max-days", type=float, default=60.0)
    ap.add_argument("--scan-half-width-days", type=float, default=20.0)
    ap.add_argument("--samples", type=int, default=61)

    ap.add_argument("--flip-normal", action="store_true")
    ap.add_argument("--flip-binormal", action="store_true")
    ap.add_argument("--binormal-basis-sign", type=float, default=1.0)
    ap.add_argument("--normal-basis-sign", type=float, default=1.0)

    ap.add_argument("--t-trust", type=float, default=500.0)
    ap.add_argument("--n-max-abs", type=float, default=900.0)
    ap.add_argument("--b-trust", type=float, default=1000.0)

    ap.add_argument("--enable-dsm", action="store_true")
    ap.add_argument("--dsm-frac-initial", type=float, default=0.5)
    ap.add_argument("--dsm-frac-min", type=float, default=0.15)
    ap.add_argument("--dsm-frac-max", type=float, default=0.85)
    ap.add_argument("--dsm-max-abs", type=float, default=250.0)
    ap.add_argument("--dsm-initial-levela", default="0,0,0")

    ap.add_argument("--ephem-cache-step-s", type=float, default=21600.0)
    ap.add_argument("--max-step-s", type=float, default=21600.0)
    ap.add_argument("--rtol", type=float, default=1e-9)
    ap.add_argument("--atol-pos-m", type=float, default=1.0)
    ap.add_argument("--atol-vel-m-s", type=float, default=1e-6)

    ap.add_argument("--ca-scale-km", type=float, default=100000.0)
    ap.add_argument("--dv-weight", type=float, default=1e-4)
    ap.add_argument("--dsm-weight", type=float, default=1e-3)
    ap.add_argument("--arrival-weight", type=float, default=1e-3)

    # Racing-specific options.
    ap.add_argument("--screen-n", type=int, default=192)
    ap.add_argument("--screen-top-k", type=int, default=24)
    ap.add_argument("--screen-samples", type=int, default=17)
    ap.add_argument("--screen-scan-half-width-days", type=float, default=0.0, help="0 = use --scan-half-width-days")
    ap.add_argument("--screen-max-step-s", type=float, default=43200.0)
    ap.add_argument("--screen-rtol", type=float, default=1e-7)
    ap.add_argument("--screen-atol-pos-m", type=float, default=100.0)
    ap.add_argument("--screen-atol-vel-m-s", type=float, default=1e-4)

    ap.add_argument("--medium-top-k", type=int, default=12)
    ap.add_argument("--medium-samples", type=int, default=35)
    ap.add_argument("--medium-max-step-s", type=float, default=21600.0)
    ap.add_argument("--medium-rtol", type=float, default=1e-8)
    ap.add_argument("--medium-atol-pos-m", type=float, default=10.0)
    ap.add_argument("--medium-atol-vel-m-s", type=float, default=1e-5)

    ap.add_argument("--jitter-n", type=int, default=8)
    ap.add_argument("--jitter-scale", type=float, default=0.08, help="fraction of each bound width for T/N/B Gaussian jitter")
    ap.add_argument("--jitter-arrival-days", type=float, default=5.0)
    ap.add_argument("--jitter-dsm-frac", type=float, default=0.08)
    ap.add_argument("--jitter-dsm-m-s", type=float, default=80.0)

    ap.add_argument("--dsm-sample-ball", action="store_true", help="sample DSM components inside a norm-limited ball instead of the full cube")
    ap.add_argument("--dsm-sample-max-norm", type=float, default=0.0, help="0 = use --dsm-max-abs")

    ap.add_argument("--polish-top-k", type=int, default=4)
    ap.add_argument("--polish-maxfev", type=int, default=35)
    ap.add_argument("--no-polish", action="store_true")

    ap.add_argument("--seed-result-json", action="append", default=[], help="prior result JSON whose best/top seed should be included")
    ap.add_argument("--manual-x", action="append", default=[], help="manual vector matching dimension, comma-separated")

    ap.add_argument("--target-ca-km", type=float, default=50000.0)
    ap.add_argument("--max-total-evals", type=int, default=420)
    ap.add_argument("--max-wall-s", type=float, default=0.0)
    ap.add_argument("--allow-duplicate-evals", action="store_true")
    ap.add_argument("--print-every", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cfg, x0, bounds, candidate, live_state = build_x0_bounds_cfg(args)
    ev = RacingEvaluator(args, cfg, bounds)

    # Stage 0: evaluate seed at fine and coarse, so the comparison is anchored.
    print("=== BASELINE/FINE SEED ===")
    ev.evaluate(x0, "fine", "x0")
    if ev.stop_reason:
        write_outputs(args, cfg, x0, bounds, ev)
        return 0

    print("=== SCREENING ===")
    seeds = make_seed_cloud(args, x0, bounds, cfg)
    for origin, x in seeds:
        ev.evaluate(x, "screen", origin)
        if ev.stop_reason:
            break

    # Re-evaluate best screen candidates at medium fidelity.
    print("=== MEDIUM RECHECK ===")
    screen_top = ev.ok_sorted("screen")[:max(0, args.screen_top_k)]
    for i, r in enumerate(screen_top):
        ev.evaluate(np.asarray(r["x"], dtype=float), "medium", f"screen_top{i}")
        if ev.stop_reason:
            break

    # Jitter around best medium candidates, evaluated at medium fidelity.
    print("=== LOCAL JITTER ===")
    medium_top = ev.ok_sorted("medium")[:max(0, args.medium_top_k)]
    jitter_jobs: list[tuple[str, np.ndarray]] = []
    for i, r in enumerate(medium_top):
        jitter_jobs.extend(jitter_around(args, np.asarray(r["x"], dtype=float), bounds, args.jitter_n, f"m{i}", i))
    for origin, x in jitter_jobs:
        ev.evaluate(x, "medium", origin)
        if ev.stop_reason:
            break

    # Fine recheck of the best medium/jitter rows.
    print("=== FINE RECHECK ===")
    candidates_for_fine = ev.ok_sorted()[:max(args.polish_top_k * 3, args.medium_top_k)]
    used: set[tuple[float, ...]] = set()
    for i, r in enumerate(candidates_for_fine):
        x = np.asarray(r["x"], dtype=float)
        k = tuple(round(float(v), 6) for v in x)
        if k in used:
            continue
        used.add(k)
        ev.evaluate(x, "fine", f"recheck{i}")
        if ev.stop_reason:
            break

    # Limited local polish only around the best few fine/global rows.
    if not args.no_polish and not ev.stop_reason:
        print("=== LIMITED POWELL POLISH ===")
        polish_starts = ev.ok_sorted()[:max(0, args.polish_top_k)]
        for i, r in enumerate(polish_starts):
            if ev.stop_reason:
                break
            run_limited_powell(ev, np.asarray(r["x"], dtype=float), args.polish_maxfev, f"polish{i}")

    write_outputs(args, cfg, x0, bounds, ev)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
