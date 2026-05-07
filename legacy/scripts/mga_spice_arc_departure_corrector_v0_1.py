#!/usr/bin/env python3
"""
mga_spice_arc_departure_corrector_v0_1.py

Single-leg departure-velocity corrector for Lambert/MGA route candidates.

Purpose
-------
This script consumes selected/refined MGA routes containing `leg_evals` and computes
small per-leg inertial departure velocity corrections that make each Lambert arc
hit its target under the same SPICE-driven dynamics used by
mga_spice_nbody_validate_arcs_v0_2.py.

This is a *diagnostic / first correction* stage, not final multiple shooting:
  - each leg is corrected independently;
  - flyby continuity is reported but not globally enforced;
  - corrections are modeled as impulsive TCMs at leg departure;
  - patched_heliocentric mode excludes the current origin and target bodies as
    perturbing point masses because each leg starts at a body center.

Dependencies
------------
  conda install -c conda-forge spiceypy scipy numpy

Example
-------
  python mga_spice_arc_departure_corrector_v0_1.py \
    --bsp data/spice_v0_1_33y/opm_mpe_principia_rebound_v0_1_33y.bsp \
    --tpc data/spice_v0_1_33y/opm_mpe_principia_rebound_v0_1_33y.ids.tpc \
    --body-catalog data/mga_v0_1/body_catalog_v0_1.krpc.json \
    --input-jsonl data/mga_v0_1/mga_refined_kdj_selected_v0_1.jsonl \
    --central-body Sun \
    --mu-central-km3-s2 1172332794.83249 \
    --gravitating-bodies Kerbin Duna Jool Sarnus Urlum Neidon Plock Soden \
    --workers 4 \
    --max-correction-m-s 20 \
    --target-miss-km 10 \
    --output-csv data/mga_v0_1/mga_arc_corrected_kdj_v0_1.csv \
    --output-jsonl data/mga_v0_1/mga_arc_corrected_kdj_v0_1.jsonl \
    --output-json data/mga_v0_1/mga_arc_corrected_kdj_v0_1.summary.json
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import math
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

SECONDS_PER_DAY = 86400.0
SCHEMA_VERSION = "mga_spice_arc_departure_corrector.v0.1"
Vec3 = Tuple[float, float, float]

_WORKER_CONFIG: Dict[str, Any] = {}
_WORKER_SPICE = None
_WORKER_NP = None


def json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, tuple):
        return list(obj)
    return str(obj)


def finite(x: Any, default: float = math.nan) -> float:
    try:
        y = float(x)
    except (TypeError, ValueError):
        return default
    return y if math.isfinite(y) else default


def opt_float(x: Any) -> Optional[float]:
    y = finite(x)
    return y if math.isfinite(y) else None


def vec3(x: Any, name: str = "vector") -> Vec3:
    if not isinstance(x, Sequence) or isinstance(x, (str, bytes)) or len(x) < 3:
        raise ValueError(f"Expected 3-vector for {name}, got {x!r}")
    return (float(x[0]), float(x[1]), float(x[2]))


def vadd(a: Sequence[float], b: Sequence[float]) -> Vec3:
    return (float(a[0]) + float(b[0]), float(a[1]) + float(b[1]), float(a[2]) + float(b[2]))


def vsub(a: Sequence[float], b: Sequence[float]) -> Vec3:
    return (float(a[0]) - float(b[0]), float(a[1]) - float(b[1]), float(a[2]) - float(b[2]))


def vnorm(a: Sequence[float]) -> float:
    return math.sqrt(float(a[0]) ** 2 + float(a[1]) ** 2 + float(a[2]) ** 2)


def stable_id(prefix: str, payload: Mapping[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=True, separators=(",", ":"), default=str)
    return f"{prefix}_{hashlib.sha1(raw.encode('utf-8')).hexdigest()[:16]}"


def seq_text(route: Mapping[str, Any]) -> str:
    seq = route.get("sequence", [])
    if isinstance(seq, str):
        if "->" in seq:
            return " -> ".join(s.strip() for s in seq.split("->") if s.strip())
        if "," in seq:
            return " -> ".join(s.strip() for s in seq.split(",") if s.strip())
        return seq
    if isinstance(seq, Sequence):
        return " -> ".join(str(x) for x in seq)
    return str(seq)


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        x = json.load(f)
    if not isinstance(x, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return x


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                x = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_no}: {exc}") from exc
            if isinstance(x, Mapping):
                rows.append(dict(x))
    return rows


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=json_default)
        f.write("\n")


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":"), default=json_default))
            f.write("\n")


@dataclass
class BodyMu:
    name: str
    mu_km3_s2: float
    radius_km: Optional[float] = None
    rp_min_km: Optional[float] = None
    soi_km: Optional[float] = None


@dataclass
class LegCorrection:
    route_id: str
    route_rank: Optional[int]
    sequence: str
    leg_index: int
    origin: str
    target: str
    depart_et: float
    arrive_et: float
    tof_days: float
    miss_before_km: Optional[float]
    miss_after_km: Optional[float]
    dvx_km_s: Optional[float]
    dvy_km_s: Optional[float]
    dvz_km_s: Optional[float]
    dv_norm_km_s: Optional[float]
    dv_norm_m_s: Optional[float]
    optimizer_success: bool
    optimizer_status: str
    nfev: Optional[int]
    cost: Optional[float]
    pass_target: bool
    notes: str


@dataclass
class RouteCorrection:
    schema_version: str
    correction_id: str
    route_id: str
    route_rank: Optional[int]
    sequence: str
    objective: Optional[float]
    robust_score: Optional[float]
    n_legs: int
    max_miss_before_km: Optional[float]
    max_miss_after_km: Optional[float]
    total_departure_correction_m_s: Optional[float]
    max_departure_correction_m_s: Optional[float]
    all_legs_pass: bool
    leg_corrections: List[Dict[str, Any]]
    source_route: Dict[str, Any]


def _worker_init(config: Mapping[str, Any]) -> None:
    global _WORKER_CONFIG, _WORKER_SPICE, _WORKER_NP
    _WORKER_CONFIG = dict(config)
    _WORKER_SPICE = importlib.import_module("spiceypy")
    _WORKER_NP = importlib.import_module("numpy")
    _WORKER_SPICE.kclear()
    for k in _WORKER_CONFIG.get("kernels", []):
        _WORKER_SPICE.furnsh(str(k))


def _spice_state(body: str, et: float, central: str, frame: str) -> Tuple[Vec3, Vec3]:
    global _WORKER_SPICE
    st, _lt = _WORKER_SPICE.spkezr(str(body), float(et), str(frame), "NONE", str(central))
    return (float(st[0]), float(st[1]), float(st[2])), (float(st[3]), float(st[4]), float(st[5]))


def _accel_central_frame(t: float, y: Any, central: str, frame: str, mu_c: float, perturbers: Sequence[BodyMu]) -> List[float]:
    global _WORKER_NP
    np = _WORKER_NP
    r = np.array([y[0], y[1], y[2]], dtype=float)
    v = np.array([y[3], y[4], y[5]], dtype=float)
    nr = float(np.linalg.norm(r))
    if nr <= 0.0:
        a = np.zeros(3)
    else:
        a = -mu_c * r / (nr ** 3)

    for b in perturbers:
        if b.mu_km3_s2 <= 0.0 or b.name == central:
            continue
        try:
            rb, _vb = _spice_state(b.name, t, central, frame)
        except Exception:
            continue
        rbv = np.array(rb, dtype=float)
        dr = rbv - r
        ndr = float(np.linalg.norm(dr))
        nrb = float(np.linalg.norm(rbv))
        if ndr > 0.0:
            a += b.mu_km3_s2 * dr / (ndr ** 3)
        if nrb > 0.0:
            a -= b.mu_km3_s2 * rbv / (nrb ** 3)
    return [float(v[0]), float(v[1]), float(v[2]), float(a[0]), float(a[1]), float(a[2])]


def _body_from_catalog(catalog: Mapping[str, Any], name: str) -> Optional[BodyMu]:
    bodies = catalog.get("bodies", {}) if isinstance(catalog, Mapping) else {}
    ent = bodies.get(name) if isinstance(bodies, Mapping) else None
    if not isinstance(ent, Mapping):
        return None
    mu = opt_float(ent.get("mu_km3_s2"))
    if mu is None:
        return None
    return BodyMu(
        name=name,
        mu_km3_s2=mu,
        radius_km=opt_float(ent.get("radius_km", ent.get("equatorial_radius_km"))),
        rp_min_km=opt_float(ent.get("rp_min_km")),
        soi_km=opt_float(ent.get("sphere_of_influence_km", ent.get("soi_km"))),
    )


def _extract_legs(route: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    legs = route.get("leg_evals", [])
    if isinstance(legs, Sequence) and not isinstance(legs, (str, bytes)):
        return [x for x in legs if isinstance(x, Mapping)]
    raise ValueError("Route does not contain leg_evals. Use selected/refined JSONL.")


def _propagate_endpoint(
    scipy_integrate: Any,
    r0: Vec3,
    v0: Vec3,
    depart_et: float,
    arrive_et: float,
    central: str,
    frame: str,
    mu_c: float,
    perturbers: Sequence[BodyMu],
) -> Tuple[bool, Optional[Vec3], Optional[Vec3], str, int]:
    cfg = _WORKER_CONFIG
    y0 = [r0[0], r0[1], r0[2], v0[0], v0[1], v0[2]]
    try:
        sol = scipy_integrate.solve_ivp(
            lambda t, y: _accel_central_frame(t, y, central, frame, mu_c, perturbers),
            (depart_et, arrive_et),
            y0,
            method=str(cfg["integrator"]),
            rtol=float(cfg["rtol"]),
            atol=[float(cfg["atol_position_km"])] * 3 + [float(cfg["atol_velocity_km_s"])] * 3,
            max_step=float(cfg["max_step_days"]) * SECONDS_PER_DAY,
            dense_output=False,
        )
    except Exception as exc:
        return False, None, None, f"integrator_exception:{exc}", 0
    nfev = int(getattr(sol, "nfev", -1))
    if sol.y is None or sol.y.shape[1] == 0:
        return False, None, None, "empty_solution", nfev
    yf = sol.y[:, -1]
    return bool(sol.success), (float(yf[0]), float(yf[1]), float(yf[2])), (float(yf[3]), float(yf[4]), float(yf[5])), str(getattr(sol, "message", "")), nfev


def _correct_leg(
    route: Mapping[str, Any],
    leg: Mapping[str, Any],
    leg_index: int,
    catalog: Mapping[str, Any],
    perturbers_all: Sequence[BodyMu],
    route_id: str,
    route_rank_i: Optional[int],
    seq: str,
) -> LegCorrection:
    global _WORKER_NP
    np = _WORKER_NP
    scipy_integrate = importlib.import_module("scipy.integrate")
    scipy_optimize = importlib.import_module("scipy.optimize")

    cfg = _WORKER_CONFIG
    central = str(cfg["central_body"])
    frame = str(cfg["frame"])
    mu_c = float(cfg["mu_central_km3_s2"])
    target_miss_km = float(cfg["target_miss_km"])
    max_corr_km_s = float(cfg["max_correction_m_s"]) / 1000.0
    dynamics_mode = str(cfg.get("dynamics_mode", "patched_heliocentric"))

    origin = str(leg.get("origin"))
    target = str(leg.get("target"))
    depart_et = finite(leg.get("depart_et"))
    arrive_et = finite(leg.get("arrive_et"))
    tof_days = (arrive_et - depart_et) / SECONDS_PER_DAY if math.isfinite(depart_et) and math.isfinite(arrive_et) else math.nan

    try:
        r0, _vb0 = _spice_state(origin, depart_et, central, frame)
        v_lam = vec3(leg.get("sc_v_depart_km_s"), "sc_v_depart_km_s")
        r_target, _v_target = _spice_state(target, arrive_et, central, frame)
    except Exception as exc:
        return LegCorrection(route_id, route_rank_i, seq, leg_index, origin, target, depart_et, arrive_et, tof_days,
                             None, None, None, None, None, None, None, False, "bad_state", None, None, False, str(exc))

    if dynamics_mode == "two_body":
        perturbers: List[BodyMu] = []
    elif dynamics_mode == "full_nbody":
        perturbers = list(perturbers_all)
    elif dynamics_mode == "patched_heliocentric":
        perturbers = [b for b in perturbers_all if b.name not in {origin, target}]
    else:
        return LegCorrection(route_id, route_rank_i, seq, leg_index, origin, target, depart_et, arrive_et, tof_days,
                             None, None, None, None, None, None, None, False, "bad_dynamics_mode", None, None, False, dynamics_mode)

    def eval_residual(dv: Sequence[float]) -> Tuple[np.ndarray, bool, str, int]:
        v = (v_lam[0] + float(dv[0]), v_lam[1] + float(dv[1]), v_lam[2] + float(dv[2]))
        ok, rf, _vf, msg, nfev = _propagate_endpoint(scipy_integrate, r0, v, depart_et, arrive_et, central, frame, mu_c, perturbers)
        if not ok or rf is None:
            # Large finite residual keeps least_squares alive but discourages this point.
            return np.array([1e12, 1e12, 1e12], dtype=float), False, msg, nfev
        return np.array([rf[0] - r_target[0], rf[1] - r_target[1], rf[2] - r_target[2]], dtype=float), True, msg, nfev

    res0, ok0, msg0, nfev0 = eval_residual([0.0, 0.0, 0.0])
    miss_before = float(np.linalg.norm(res0)) if ok0 else None

    # If already good, do not run an expensive optimizer.
    if miss_before is not None and miss_before <= target_miss_km:
        return LegCorrection(route_id, route_rank_i, seq, leg_index, origin, target, depart_et, arrive_et, tof_days,
                             miss_before, miss_before, 0.0, 0.0, 0.0, 0.0, 0.0, True, "already_within_target", nfev0, 0.5 * miss_before * miss_before, True,
                             f"dynamics_mode={dynamics_mode}; perturbers={len(perturbers)}")

    # Scale residual to km. Bounds are in km/s. diff_step is in km/s-ish through x_scale.
    def fun(x: Any) -> Any:
        rr, _ok, _msg, _n = eval_residual(x)
        return rr

    try:
        sol = scipy_optimize.least_squares(
            fun,
            x0=np.zeros(3, dtype=float),
            bounds=(-max_corr_km_s * np.ones(3), max_corr_km_s * np.ones(3)),
            xtol=float(cfg["ls_xtol"]),
            ftol=float(cfg["ls_ftol"]),
            gtol=float(cfg["ls_gtol"]),
            x_scale=max(max_corr_km_s / 5.0, 1e-6),
            max_nfev=int(cfg["max_nfev"]),
            verbose=0,
        )
    except Exception as exc:
        return LegCorrection(route_id, route_rank_i, seq, leg_index, origin, target, depart_et, arrive_et, tof_days,
                             miss_before, None, None, None, None, None, None, False, "optimizer_exception", None, None, False, str(exc))

    dv = [float(sol.x[0]), float(sol.x[1]), float(sol.x[2])]
    resf, okf, msgf, nfevf = eval_residual(dv)
    miss_after = float(np.linalg.norm(resf)) if okf else None
    dv_norm = float(np.linalg.norm(np.array(dv)))
    pass_target = bool(miss_after is not None and miss_after <= target_miss_km and dv_norm <= max_corr_km_s + 1e-12)
    return LegCorrection(
        route_id=route_id,
        route_rank=route_rank_i,
        sequence=seq,
        leg_index=leg_index,
        origin=origin,
        target=target,
        depart_et=depart_et,
        arrive_et=arrive_et,
        tof_days=tof_days,
        miss_before_km=miss_before,
        miss_after_km=miss_after,
        dvx_km_s=dv[0],
        dvy_km_s=dv[1],
        dvz_km_s=dv[2],
        dv_norm_km_s=dv_norm,
        dv_norm_m_s=1000.0 * dv_norm,
        optimizer_success=bool(sol.success and okf),
        optimizer_status=f"{sol.status}:{sol.message}; final_integrator={msgf}",
        nfev=int(getattr(sol, "nfev", -1)),
        cost=float(getattr(sol, "cost", math.nan)),
        pass_target=pass_target,
        notes=f"dynamics_mode={dynamics_mode}; perturbers={len(perturbers)}; before_integrator={msg0}",
    )


def _correct_route_worker(payload: Tuple[Dict[str, Any], Dict[str, Any]]) -> Dict[str, Any]:
    route, catalog = payload
    include_bodies = list(_WORKER_CONFIG["gravitating_bodies"])
    central = str(_WORKER_CONFIG["central_body"])
    perturbers_all: List[BodyMu] = []
    for name in include_bodies:
        b = _body_from_catalog(catalog, name)
        if b is not None and b.name != central:
            perturbers_all.append(b)

    route_id = str(route.get("refined_id") or route.get("route_id") or stable_id("route", route))
    try:
        route_rank_i = int(route.get("rank")) if route.get("rank") is not None else None
    except Exception:
        route_rank_i = None
    seq = seq_text(route)

    try:
        legs = _extract_legs(route)
    except Exception as exc:
        out = RouteCorrection(
            schema_version=SCHEMA_VERSION,
            correction_id=stable_id("arccorr", {"route_id": route_id, "error": str(exc)}),
            route_id=route_id,
            route_rank=route_rank_i,
            sequence=seq,
            objective=opt_float(route.get("objective")),
            robust_score=opt_float(route.get("robust_score")),
            n_legs=0,
            max_miss_before_km=None,
            max_miss_after_km=None,
            total_departure_correction_m_s=None,
            max_departure_correction_m_s=None,
            all_legs_pass=False,
            leg_corrections=[],
            source_route=dict(route),
        )
        d = asdict(out)
        d["status"] = "missing_leg_evals"
        d["error"] = str(exc)
        return d

    leg_corrs: List[LegCorrection] = []
    for i, leg in enumerate(legs):
        leg_corrs.append(_correct_leg(route, leg, i, catalog, perturbers_all, route_id, route_rank_i, seq))

    misses_before = [x.miss_before_km for x in leg_corrs if x.miss_before_km is not None and math.isfinite(x.miss_before_km)]
    misses_after = [x.miss_after_km for x in leg_corrs if x.miss_after_km is not None and math.isfinite(x.miss_after_km)]
    dvs = [x.dv_norm_m_s for x in leg_corrs if x.dv_norm_m_s is not None and math.isfinite(x.dv_norm_m_s)]
    all_pass = bool(leg_corrs and all(x.pass_target for x in leg_corrs))
    out = RouteCorrection(
        schema_version=SCHEMA_VERSION,
        correction_id=stable_id("arccorr", {"route_id": route_id, "miss_after": max(misses_after) if misses_after else None, "legs": len(leg_corrs)}),
        route_id=route_id,
        route_rank=route_rank_i,
        sequence=seq,
        objective=opt_float(route.get("objective")),
        robust_score=opt_float(route.get("robust_score")),
        n_legs=len(leg_corrs),
        max_miss_before_km=max(misses_before) if misses_before else None,
        max_miss_after_km=max(misses_after) if misses_after else None,
        total_departure_correction_m_s=sum(dvs) if dvs else None,
        max_departure_correction_m_s=max(dvs) if dvs else None,
        all_legs_pass=all_pass,
        leg_corrections=[asdict(x) for x in leg_corrs],
        source_route=dict(route),
    )
    d = asdict(out)
    d["status"] = "ok" if leg_corrs else "no_legs"
    return d


def flatten_rows(route_corr: Mapping[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for leg in route_corr.get("leg_corrections", []) or []:
        if not isinstance(leg, Mapping):
            continue
        rows.append({
            "correction_id": route_corr.get("correction_id"),
            "route_id": route_corr.get("route_id"),
            "route_rank": route_corr.get("route_rank"),
            "sequence": route_corr.get("sequence"),
            "route_all_legs_pass": int(bool(route_corr.get("all_legs_pass"))),
            "route_max_miss_before_km": route_corr.get("max_miss_before_km"),
            "route_max_miss_after_km": route_corr.get("max_miss_after_km"),
            "route_total_departure_correction_m_s": route_corr.get("total_departure_correction_m_s"),
            "route_max_departure_correction_m_s": route_corr.get("max_departure_correction_m_s"),
            "leg_index": leg.get("leg_index"),
            "origin": leg.get("origin"),
            "target": leg.get("target"),
            "tof_days": leg.get("tof_days"),
            "miss_before_km": leg.get("miss_before_km"),
            "miss_after_km": leg.get("miss_after_km"),
            "dvx_km_s": leg.get("dvx_km_s"),
            "dvy_km_s": leg.get("dvy_km_s"),
            "dvz_km_s": leg.get("dvz_km_s"),
            "dv_norm_m_s": leg.get("dv_norm_m_s"),
            "pass_target": int(bool(leg.get("pass_target"))),
            "optimizer_success": int(bool(leg.get("optimizer_success"))),
            "optimizer_status": leg.get("optimizer_status"),
            "nfev": leg.get("nfev"),
            "notes": leg.get("notes"),
        })
    return rows


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "correction_id", "route_id", "route_rank", "sequence", "route_all_legs_pass",
        "route_max_miss_before_km", "route_max_miss_after_km",
        "route_total_departure_correction_m_s", "route_max_departure_correction_m_s",
        "leg_index", "origin", "target", "tof_days", "miss_before_km", "miss_after_km",
        "dvx_km_s", "dvy_km_s", "dvz_km_s", "dv_norm_m_s", "pass_target",
        "optimizer_success", "optimizer_status", "nfev", "notes",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k) for k in fieldnames})


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Per-leg departure velocity correction for SPICE N-body patched arcs.")
    p.add_argument("--bsp", required=True, type=Path)
    p.add_argument("--tpc", type=Path, default=None)
    p.add_argument("--body-catalog", required=True, type=Path)
    p.add_argument("--input-jsonl", required=True, type=Path)
    p.add_argument("--central-body", default="Sun")
    p.add_argument("--frame", default="J2000")
    p.add_argument("--mu-central-km3-s2", required=True, type=float)
    p.add_argument("--gravitating-bodies", nargs="+", default=[])
    p.add_argument("--dynamics-mode", default="patched_heliocentric", choices=["two_body", "patched_heliocentric", "full_nbody"])
    p.add_argument("--route-rank", type=int, default=None, help="Optional route rank to process.")
    p.add_argument("--max-routes", type=int, default=0, help="Limit number of routes after filtering; 0 means all.")
    p.add_argument("--workers", type=int, default=1, help="0=os.cpu_count, 1=serial, N=processes.")
    p.add_argument("--multiprocessing-start-method", default="spawn", choices=["spawn", "fork", "forkserver"])
    p.add_argument("--integrator", default="DOP853", choices=["DOP853", "RK45", "LSODA", "Radau", "BDF"])
    p.add_argument("--rtol", type=float, default=1e-10)
    p.add_argument("--atol-position-km", type=float, default=1e-6)
    p.add_argument("--atol-velocity-km-s", type=float, default=1e-12)
    p.add_argument("--max-step-days", type=float, default=2.0)
    p.add_argument("--max-correction-m-s", type=float, default=20.0, help="Component-wise bound is +/- this value in m/s.")
    p.add_argument("--target-miss-km", type=float, default=10.0)
    p.add_argument("--max-nfev", type=int, default=30)
    p.add_argument("--ls-xtol", type=float, default=1e-9)
    p.add_argument("--ls-ftol", type=float, default=1e-9)
    p.add_argument("--ls-gtol", type=float, default=1e-9)
    p.add_argument("--output-csv", required=True, type=Path)
    p.add_argument("--output-jsonl", required=True, type=Path)
    p.add_argument("--output-json", required=True, type=Path)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    catalog = load_json(args.body_catalog)
    routes = read_jsonl(args.input_jsonl)
    if args.route_rank is not None:
        routes = [r for r in routes if int(r.get("rank", -999999)) == args.route_rank]
    if args.max_routes and args.max_routes > 0:
        routes = routes[: args.max_routes]
    if not routes:
        raise SystemExit("No routes to process.")

    kernels = [args.bsp]
    if args.tpc is not None:
        kernels.append(args.tpc)
    grav_bodies = args.gravitating_bodies or sorted((catalog.get("bodies") or {}).keys())

    config = {
        "kernels": [str(k) for k in kernels],
        "central_body": args.central_body,
        "frame": args.frame,
        "mu_central_km3_s2": args.mu_central_km3_s2,
        "gravitating_bodies": grav_bodies,
        "dynamics_mode": args.dynamics_mode,
        "integrator": args.integrator,
        "rtol": args.rtol,
        "atol_position_km": args.atol_position_km,
        "atol_velocity_km_s": args.atol_velocity_km_s,
        "max_step_days": args.max_step_days,
        "max_correction_m_s": args.max_correction_m_s,
        "target_miss_km": args.target_miss_km,
        "max_nfev": args.max_nfev,
        "ls_xtol": args.ls_xtol,
        "ls_ftol": args.ls_ftol,
        "ls_gtol": args.ls_gtol,
    }

    workers = args.workers
    if workers == 0:
        workers = os.cpu_count() or 1
    workers = max(1, workers)

    payloads = [(r, catalog) for r in routes]
    results: List[Dict[str, Any]] = []
    if workers == 1:
        _worker_init(config)
        for pld in payloads:
            results.append(_correct_route_worker(pld))
    else:
        ctx = mp.get_context(args.multiprocessing_start_method)
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx, initializer=_worker_init, initargs=(config,)) as ex:
            futs = [ex.submit(_correct_route_worker, pld) for pld in payloads]
            for fut in as_completed(futs):
                results.append(fut.result())

    results.sort(key=lambda r: (
        0 if r.get("all_legs_pass") else 1,
        finite(r.get("total_departure_correction_m_s"), 1e99),
        finite(r.get("max_miss_after_km"), 1e99),
    ))

    flat: List[Dict[str, Any]] = []
    for rr in results:
        flat.extend(flatten_rows(rr))

    write_csv(args.output_csv, flat)
    write_jsonl(args.output_jsonl, results)

    pass_count = sum(1 for r in results if r.get("all_legs_pass"))
    total_dvs = [finite(r.get("total_departure_correction_m_s")) for r in results]
    total_dvs = [x for x in total_dvs if math.isfinite(x)]
    miss_after = [finite(r.get("max_miss_after_km")) for r in results]
    miss_after = [x for x in miss_after if math.isfinite(x)]
    summary = {
        "schema_version": SCHEMA_VERSION,
        "input_jsonl": str(args.input_jsonl),
        "body_catalog": str(args.body_catalog),
        "dynamics_mode": args.dynamics_mode,
        "workers": workers,
        "routes_processed": len(results),
        "routes_all_legs_pass": pass_count,
        "target_miss_km": args.target_miss_km,
        "max_correction_m_s": args.max_correction_m_s,
        "total_departure_correction_m_s": {
            "min": min(total_dvs) if total_dvs else None,
            "median": sorted(total_dvs)[len(total_dvs)//2] if total_dvs else None,
            "max": max(total_dvs) if total_dvs else None,
        },
        "max_miss_after_km": {
            "min": min(miss_after) if miss_after else None,
            "median": sorted(miss_after)[len(miss_after)//2] if miss_after else None,
            "max": max(miss_after) if miss_after else None,
        },
        "top_routes": [
            {
                "route_rank": r.get("route_rank"),
                "sequence": r.get("sequence"),
                "all_legs_pass": r.get("all_legs_pass"),
                "max_miss_before_km": r.get("max_miss_before_km"),
                "max_miss_after_km": r.get("max_miss_after_km"),
                "total_departure_correction_m_s": r.get("total_departure_correction_m_s"),
                "max_departure_correction_m_s": r.get("max_departure_correction_m_s"),
            }
            for r in results[:10]
        ],
        "outputs": {
            "csv": str(args.output_csv),
            "jsonl": str(args.output_jsonl),
            "json": str(args.output_json),
        },
    }
    write_json(args.output_json, summary)

    print("=" * 80)
    print("MGA SPICE ARC DEPARTURE CORRECTOR V0.1")
    print("=" * 80)
    print(f"Routes processed: {len(results)}")
    print(f"All legs pass:    {pass_count}")
    print(f"Workers:          {workers}")
    print(f"Dynamics mode:    {args.dynamics_mode}")
    print(f"Target miss:      {args.target_miss_km:g} km")
    print(f"Max corr bound:   {args.max_correction_m_s:g} m/s per component")
    if total_dvs:
        print(f"Total correction: min={min(total_dvs):.4g} m/s median={sorted(total_dvs)[len(total_dvs)//2]:.4g} m/s max={max(total_dvs):.4g} m/s")
    if miss_after:
        print(f"Max miss after:   min={min(miss_after):.4g} km median={sorted(miss_after)[len(miss_after)//2]:.4g} km max={max(miss_after):.4g} km")
    print("\nTop corrected routes:")
    for i, r in enumerate(results[:10], start=1):
        print(
            f" {i}. rank={r.get('route_rank')} | {r.get('sequence')} | pass={r.get('all_legs_pass')} | "
            f"miss {finite(r.get('max_miss_before_km')):.4g} -> {finite(r.get('max_miss_after_km')):.4g} km | "
            f"total Δv_corr={finite(r.get('total_departure_correction_m_s')):.4g} m/s | "
            f"max leg Δv_corr={finite(r.get('max_departure_correction_m_s')):.4g} m/s"
        )
    print("=" * 80)
    print(f"[OK] wrote CSV:   {args.output_csv}")
    print(f"[OK] wrote JSONL: {args.output_jsonl}")
    print(f"[OK] wrote JSON:  {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
