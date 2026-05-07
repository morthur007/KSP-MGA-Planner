#!/usr/bin/env python3
"""
mga_spice_nbody_validate_arcs_v0_1.py

Planning-grade open-loop N-body validation for Lambert/MGA route candidates.

This script consumes selected refined route JSONL produced by
mga_select_refined_routes_v0_1.py (or compatible refined-route JSONL) and
validates each Lambert leg by integrating the spacecraft as a massless particle
under a point-mass N-body model driven by SPICE ephemerides.

Important scope
---------------
This is NOT a flyby targeter and NOT a continuous trajectory corrector.
It validates each Lambert arc independently:

  initial state:  body position at leg departure + Lambert departure velocity
  propagation:    central gravity + perturbing bodies from SPICE
  endpoint check: spacecraft state vs target body at leg arrival epoch

For a route like Kerbin -> Duna -> Jool, leg 1 and leg 2 are validated as
separate open-loop arcs. The flyby is still a kinematic patch from the Lambert
solution; B-plane closure / powered flyby / multiple shooting comes later.

Dependencies
------------
  conda install -c conda-forge spiceypy scipy numpy

Example
-------
  python mga_spice_nbody_validate_arcs_v0_1.py \
    --bsp data/spice_v0_1_33y/opm_mpe_principia_rebound_v0_1_33y.bsp \
    --tpc data/spice_v0_1_33y/opm_mpe_principia_rebound_v0_1_33y.ids.tpc \
    --body-catalog data/mga_v0_1/body_catalog_v0_1.krpc.json \
    --input-jsonl data/mga_v0_1/mga_refined_kdj_selected_v0_1.jsonl \
    --central-body Sun \
    --mu-central-km3-s2 1172332794.83249 \
    --gravitating-bodies Duna Jool Kerbin Sarnus Urlum Neidon Plock Soden \
    --workers 4 \
    --output-csv data/mga_v0_1/mga_nbody_validate_kdj_v0_1.csv \
    --output-jsonl data/mga_v0_1/mga_nbody_validate_kdj_v0_1.jsonl \
    --output-json data/mga_v0_1/mga_nbody_validate_kdj_v0_1.summary.json
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
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

SECONDS_PER_DAY = 86400.0
SCHEMA_VERSION = "mga_spice_nbody_validate_arcs.v0.1"
Vec3 = Tuple[float, float, float]


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


def classify_miss(miss_km: float, soi_km: Optional[float], rp_min_km: Optional[float]) -> str:
    """Coarse planning classes. These are not operational targeting grades."""
    if not math.isfinite(miss_km):
        return "invalid"
    # Absolute tight classes for direct endpoint closure.
    if miss_km <= 10.0:
        return "A"
    if miss_km <= 100.0:
        return "B"
    if miss_km <= 1000.0:
        return "C"
    # Relative classes useful for big outer-system bodies.
    if soi_km and soi_km > 0:
        frac = miss_km / soi_km
        if frac <= 1.0e-3:
            return "D-soi-tight"
        if frac <= 1.0e-2:
            return "E-soi-coarse"
        if frac <= 5.0e-2:
            return "F-soi-wide"
    if rp_min_km and rp_min_km > 0 and miss_km <= 10.0 * rp_min_km:
        return "G-near-planet"
    return "missed"


@dataclass
class BodyMu:
    name: str
    mu_km3_s2: float
    radius_km: Optional[float] = None
    rp_min_km: Optional[float] = None
    soi_km: Optional[float] = None


@dataclass
class LegValidation:
    route_id: str
    route_rank: Optional[int]
    sequence: str
    leg_index: int
    origin: str
    target: str
    depart_et: float
    arrive_et: float
    tof_days: float
    integrator_status: str
    success: bool
    nfev: Optional[int]
    endpoint_miss_km: Optional[float]
    endpoint_speed_error_km_s: Optional[float]
    endpoint_vrel_km_s: Optional[float]
    closest_approach_km: Optional[float]
    closest_approach_dt_days: Optional[float]
    closest_approach_et: Optional[float]
    target_soi_km: Optional[float]
    target_rp_min_km: Optional[float]
    miss_class: str
    notes: str


@dataclass
class RouteValidation:
    schema_version: str
    validation_id: str
    route_id: str
    route_rank: Optional[int]
    sequence: str
    objective: Optional[float]
    robust_score: Optional[float]
    valid_input: bool
    n_legs: int
    max_endpoint_miss_km: Optional[float]
    max_closest_approach_km: Optional[float]
    max_endpoint_speed_error_km_s: Optional[float]
    worst_miss_class: str
    pass_endpoint_threshold: bool
    leg_validations: List[Dict[str, Any]]


# Globals in workers.
_WORKER_CONFIG: Dict[str, Any] = {}
_WORKER_SPICE = None
_WORKER_NP = None


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
    global _WORKER_SPICE, _WORKER_NP
    np = _WORKER_NP
    r = np.array([y[0], y[1], y[2]], dtype=float)
    v = np.array([y[3], y[4], y[5]], dtype=float)
    nr = float(np.linalg.norm(r))
    if nr <= 0.0:
        a = np.zeros(3)
    else:
        a = -mu_c * r / (nr ** 3)

    # Non-inertial correction for central-body-centered coordinates:
    # a_rel = central attraction + sum mu_i[(r_i-r)/|r_i-r|^3 - r_i/|r_i|^3]
    for b in perturbers:
        if b.mu_km3_s2 <= 0.0 or b.name == central:
            continue
        try:
            rb, _vb = _spice_state(b.name, t, central, frame)
        except Exception:
            # Missing SPICE state: skip this perturbing body for robustness.
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
    raise ValueError("Route does not contain leg_evals. Use selected/refined JSONL, not compact packet JSON.")


def _validate_route_worker(payload: Tuple[Dict[str, Any], Dict[str, Any]]) -> Dict[str, Any]:
    global _WORKER_CONFIG, _WORKER_NP
    route, catalog = payload
    scipy_integrate = importlib.import_module("scipy.integrate")
    np = _WORKER_NP

    central = str(_WORKER_CONFIG["central_body"])
    frame = str(_WORKER_CONFIG["frame"])
    mu_c = float(_WORKER_CONFIG["mu_central_km3_s2"])
    endpoint_threshold_km = float(_WORKER_CONFIG["endpoint_miss_threshold_km"])
    search_window_s = float(_WORKER_CONFIG["arrival_search_window_days"]) * SECONDS_PER_DAY
    search_samples = int(_WORKER_CONFIG["arrival_search_samples"])
    rtol = float(_WORKER_CONFIG["rtol"])
    atol_pos = float(_WORKER_CONFIG["atol_position_km"])
    atol_vel = float(_WORKER_CONFIG["atol_velocity_km_s"])
    max_step_s = float(_WORKER_CONFIG["max_step_days"]) * SECONDS_PER_DAY
    include_bodies = list(_WORKER_CONFIG["gravitating_bodies"])

    perturbers: List[BodyMu] = []
    for name in include_bodies:
        b = _body_from_catalog(catalog, name)
        if b is not None and b.name != central:
            perturbers.append(b)

    route_id = str(route.get("refined_id") or route.get("route_id") or stable_id("route", route))
    seq = seq_text(route)
    route_rank = route.get("selected_rank")
    try:
        route_rank_i = int(route_rank) if route_rank is not None else None
    except Exception:
        route_rank_i = None

    leg_results: List[LegValidation] = []
    notes_route: List[str] = []

    try:
        legs = _extract_legs(route)
    except Exception as exc:
        rv = RouteValidation(
            schema_version=SCHEMA_VERSION,
            validation_id=stable_id("nbval", {"route_id": route_id, "error": str(exc)}),
            route_id=route_id,
            route_rank=route_rank_i,
            sequence=seq,
            objective=opt_float(route.get("objective")),
            robust_score=opt_float(route.get("robust_score")),
            valid_input=False,
            n_legs=0,
            max_endpoint_miss_km=None,
            max_closest_approach_km=None,
            max_endpoint_speed_error_km_s=None,
            worst_miss_class="invalid",
            pass_endpoint_threshold=False,
            leg_validations=[],
        )
        out = asdict(rv)
        out["status"] = "missing_leg_evals"
        out["error"] = str(exc)
        return out

    for i, leg in enumerate(legs):
        origin = str(leg.get("origin"))
        target = str(leg.get("target"))
        depart_et = finite(leg.get("depart_et"))
        arrive_et = finite(leg.get("arrive_et"))
        tof_days = (arrive_et - depart_et) / SECONDS_PER_DAY if math.isfinite(depart_et) and math.isfinite(arrive_et) else math.nan
        target_info = _body_from_catalog(catalog, target)
        target_soi = target_info.soi_km if target_info else None
        target_rp = target_info.rp_min_km if target_info else None

        if not (math.isfinite(depart_et) and math.isfinite(arrive_et) and arrive_et > depart_et):
            leg_results.append(LegValidation(
                route_id, route_rank_i, seq, i, origin, target, depart_et, arrive_et, tof_days,
                "bad_time", False, None, None, None, None, None, None, None, target_soi, target_rp,
                "invalid", "depart/arrive ET invalid"
            ))
            continue
        try:
            # Recompute origin position from SPICE at the exact epoch; use the Lambert inertial departure velocity.
            r0, _vbody0 = _spice_state(origin, depart_et, central, frame)
            v0 = vec3(leg.get("sc_v_depart_km_s"), "sc_v_depart_km_s")
        except Exception as exc:
            leg_results.append(LegValidation(
                route_id, route_rank_i, seq, i, origin, target, depart_et, arrive_et, tof_days,
                "bad_initial_state", False, None, None, None, None, None, None, None, target_soi, target_rp,
                "invalid", str(exc)
            ))
            continue

        y0 = [r0[0], r0[1], r0[2], v0[0], v0[1], v0[2]]
        try:
            sol = scipy_integrate.solve_ivp(
                lambda t, y: _accel_central_frame(t, y, central, frame, mu_c, perturbers),
                (depart_et, arrive_et),
                y0,
                method=str(_WORKER_CONFIG["integrator"]),
                rtol=rtol,
                atol=[atol_pos, atol_pos, atol_pos, atol_vel, atol_vel, atol_vel],
                max_step=max_step_s,
                dense_output=bool(search_window_s > 0.0),
            )
        except Exception as exc:
            leg_results.append(LegValidation(
                route_id, route_rank_i, seq, i, origin, target, depart_et, arrive_et, tof_days,
                "integrator_exception", False, None, None, None, None, None, None, None, target_soi, target_rp,
                "invalid", str(exc)
            ))
            continue

        success = bool(sol.success)
        nfev = int(getattr(sol, "nfev", -1))
        if sol.y is None or sol.y.shape[1] == 0:
            leg_results.append(LegValidation(
                route_id, route_rank_i, seq, i, origin, target, depart_et, arrive_et, tof_days,
                "empty_solution", False, nfev, None, None, None, None, None, None, target_soi, target_rp,
                "invalid", str(getattr(sol, "message", "empty"))
            ))
            continue

        yf = sol.y[:, -1]
        r_sc = (float(yf[0]), float(yf[1]), float(yf[2]))
        v_sc = (float(yf[3]), float(yf[4]), float(yf[5]))
        try:
            r_t, v_t = _spice_state(target, arrive_et, central, frame)
            miss = vnorm(vsub(r_sc, r_t))
            vrel = vnorm(vsub(v_sc, v_t))
        except Exception as exc:
            leg_results.append(LegValidation(
                route_id, route_rank_i, seq, i, origin, target, depart_et, arrive_et, tof_days,
                "bad_target_state", success, nfev, None, None, None, None, None, None, target_soi, target_rp,
                "invalid", str(exc)
            ))
            continue

        v_err = None
        try:
            v_lam_arr = vec3(leg.get("sc_v_arrive_km_s"), "sc_v_arrive_km_s")
            v_err = vnorm(vsub(v_sc, v_lam_arr))
        except Exception:
            v_err = None

        ca_km: Optional[float] = miss
        ca_dt: Optional[float] = 0.0
        ca_et: Optional[float] = arrive_et
        if search_window_s > 0.0 and getattr(sol, "sol", None) is not None and search_samples >= 3 and sol.success:
            t0 = max(depart_et, arrive_et - search_window_s)
            t1 = arrive_et + search_window_s
            # Dense output only valid on integration span. Do not extrapolate past arrive_et.
            t1 = min(t1, arrive_et)
            if t1 > t0:
                ts = np.linspace(t0, t1, search_samples)
                vals = sol.sol(ts)
                best_d = math.inf
                best_t = arrive_et
                for j, tt in enumerate(ts):
                    rj = (float(vals[0, j]), float(vals[1, j]), float(vals[2, j]))
                    try:
                        rtj, _vtj = _spice_state(target, float(tt), central, frame)
                    except Exception:
                        continue
                    d = vnorm(vsub(rj, rtj))
                    if d < best_d:
                        best_d = d
                        best_t = float(tt)
                if math.isfinite(best_d):
                    ca_km = best_d
                    ca_et = best_t
                    ca_dt = (best_t - arrive_et) / SECONDS_PER_DAY

        miss_class = classify_miss(float(miss), target_soi, target_rp)
        leg_results.append(LegValidation(
            route_id=route_id,
            route_rank=route_rank_i,
            sequence=seq,
            leg_index=i,
            origin=origin,
            target=target,
            depart_et=depart_et,
            arrive_et=arrive_et,
            tof_days=tof_days,
            integrator_status="ok" if success else "integrator_failed",
            success=success,
            nfev=nfev,
            endpoint_miss_km=float(miss),
            endpoint_speed_error_km_s=float(v_err) if v_err is not None else None,
            endpoint_vrel_km_s=float(vrel),
            closest_approach_km=float(ca_km) if ca_km is not None else None,
            closest_approach_dt_days=float(ca_dt) if ca_dt is not None else None,
            closest_approach_et=float(ca_et) if ca_et is not None else None,
            target_soi_km=target_soi,
            target_rp_min_km=target_rp,
            miss_class=miss_class,
            notes=str(getattr(sol, "message", "")),
        ))

    endpoint_misses = [x.endpoint_miss_km for x in leg_results if x.endpoint_miss_km is not None and math.isfinite(x.endpoint_miss_km)]
    ca_misses = [x.closest_approach_km for x in leg_results if x.closest_approach_km is not None and math.isfinite(x.closest_approach_km)]
    v_errors = [x.endpoint_speed_error_km_s for x in leg_results if x.endpoint_speed_error_km_s is not None and math.isfinite(x.endpoint_speed_error_km_s)]
    max_miss = max(endpoint_misses) if endpoint_misses else None
    max_ca = max(ca_misses) if ca_misses else None
    max_verr = max(v_errors) if v_errors else None

    # Worst class: simple order from best to worst.
    order = {"A": 0, "B": 1, "C": 2, "D-soi-tight": 3, "E-soi-coarse": 4, "F-soi-wide": 5, "G-near-planet": 6, "missed": 7, "invalid": 8}
    worst = "invalid"
    if leg_results:
        worst = max((x.miss_class for x in leg_results), key=lambda c: order.get(c, 99))

    pass_threshold = bool(max_miss is not None and max_miss <= endpoint_threshold_km and all(x.success for x in leg_results))
    rv = RouteValidation(
        schema_version=SCHEMA_VERSION,
        validation_id=stable_id("nbval", {"route_id": route_id, "max_miss": max_miss, "legs": len(leg_results)}),
        route_id=route_id,
        route_rank=route_rank_i,
        sequence=seq,
        objective=opt_float(route.get("objective")),
        robust_score=opt_float(route.get("robust_score")),
        valid_input=bool(route.get("valid", True)),
        n_legs=len(leg_results),
        max_endpoint_miss_km=max_miss,
        max_closest_approach_km=max_ca,
        max_endpoint_speed_error_km_s=max_verr,
        worst_miss_class=worst,
        pass_endpoint_threshold=pass_threshold,
        leg_validations=[asdict(x) for x in leg_results],
    )
    out = asdict(rv)
    out["status"] = "ok" if leg_results else "no_legs"
    return out


def flatten_for_csv(route_val: Mapping[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for leg in route_val.get("leg_validations", []) or []:
        if not isinstance(leg, Mapping):
            continue
        rows.append({
            "validation_id": route_val.get("validation_id"),
            "route_id": route_val.get("route_id"),
            "route_rank": route_val.get("route_rank"),
            "sequence": route_val.get("sequence"),
            "route_objective": route_val.get("objective"),
            "route_robust_score": route_val.get("robust_score"),
            "route_pass_endpoint_threshold": int(bool(route_val.get("pass_endpoint_threshold"))),
            "route_max_endpoint_miss_km": route_val.get("max_endpoint_miss_km"),
            "route_worst_miss_class": route_val.get("worst_miss_class"),
            "leg_index": leg.get("leg_index"),
            "origin": leg.get("origin"),
            "target": leg.get("target"),
            "tof_days": leg.get("tof_days"),
            "success": int(bool(leg.get("success"))),
            "integrator_status": leg.get("integrator_status"),
            "nfev": leg.get("nfev"),
            "endpoint_miss_km": leg.get("endpoint_miss_km"),
            "endpoint_speed_error_km_s": leg.get("endpoint_speed_error_km_s"),
            "endpoint_vrel_km_s": leg.get("endpoint_vrel_km_s"),
            "closest_approach_km": leg.get("closest_approach_km"),
            "closest_approach_dt_days": leg.get("closest_approach_dt_days"),
            "target_soi_km": leg.get("target_soi_km"),
            "target_rp_min_km": leg.get("target_rp_min_km"),
            "miss_class": leg.get("miss_class"),
            "notes": leg.get("notes"),
        })
    return rows


def write_csv(path: Path, route_vals: Sequence[Mapping[str, Any]]) -> None:
    rows: List[Dict[str, Any]] = []
    for rv in route_vals:
        rows.extend(flatten_for_csv(rv))
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "validation_id", "route_id", "route_rank", "sequence", "route_objective", "route_robust_score",
        "route_pass_endpoint_threshold", "route_max_endpoint_miss_km", "route_worst_miss_class",
        "leg_index", "origin", "target", "tof_days", "success", "integrator_status", "nfev",
        "endpoint_miss_km", "endpoint_speed_error_km_s", "endpoint_vrel_km_s", "closest_approach_km",
        "closest_approach_dt_days", "target_soi_km", "target_rp_min_km", "miss_class", "notes",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


def discover_gravitating_bodies(catalog: Mapping[str, Any], explicit: Optional[Sequence[str]]) -> List[str]:
    if explicit:
        return [str(x) for x in explicit]
    bodies = catalog.get("bodies", {}) if isinstance(catalog, Mapping) else {}
    out: List[str] = []
    if isinstance(bodies, Mapping):
        for name, ent in bodies.items():
            if isinstance(ent, Mapping) and opt_float(ent.get("mu_km3_s2")) is not None:
                out.append(str(name))
    return sorted(set(out))


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bsp", required=True, type=Path)
    p.add_argument("--tpc", required=True, type=Path)
    p.add_argument("--fk", action="append", type=Path, default=[])
    p.add_argument("--lsk", action="append", type=Path, default=[])
    p.add_argument("--body-catalog", required=True, type=Path)
    p.add_argument("--input-jsonl", required=True, type=Path, help="Selected/refined route JSONL containing leg_evals.")
    p.add_argument("--central-body", default="Sun")
    p.add_argument("--frame", default="J2000")
    p.add_argument("--mu-central-km3-s2", required=True, type=float)
    p.add_argument("--gravitating-bodies", nargs="*", default=None)
    p.add_argument("--max-routes", type=int, default=0, help="0 means all routes.")
    p.add_argument("--route-rank", type=int, action="append", default=[], help="Validate only selected rank(s).")
    p.add_argument("--integrator", default="DOP853", choices=["DOP853", "RK45", "LSODA", "Radau", "BDF"])
    p.add_argument("--rtol", type=float, default=1e-10)
    p.add_argument("--atol-position-km", type=float, default=1e-6)
    p.add_argument("--atol-velocity-km-s", type=float, default=1e-12)
    p.add_argument("--max-step-days", type=float, default=2.0)
    p.add_argument("--arrival-search-window-days", type=float, default=10.0)
    p.add_argument("--arrival-search-samples", type=int, default=201)
    p.add_argument("--endpoint-miss-threshold-km", type=float, default=10000.0)
    p.add_argument("--workers", type=int, default=1, help="1 serial, 0 os.cpu_count(), >1 process pool.")
    p.add_argument("--multiprocessing-start-method", default="spawn", choices=["spawn", "fork", "forkserver"])
    p.add_argument("--output-csv", required=True, type=Path)
    p.add_argument("--output-jsonl", required=True, type=Path)
    p.add_argument("--output-json", required=True, type=Path)
    args = p.parse_args(argv)

    catalog = load_json(args.body_catalog)
    routes = read_jsonl(args.input_jsonl)
    if args.route_rank:
        keep = set(args.route_rank)
        routes = [r for r in routes if int(finite(r.get("selected_rank"), -999999)) in keep]
    if args.max_routes and args.max_routes > 0:
        routes = routes[: args.max_routes]

    grav_bodies = discover_gravitating_bodies(catalog, args.gravitating_bodies)
    kernels = [str(p) for p in list(args.lsk) + [args.tpc] + list(args.fk) + [args.bsp]]
    config = {
        "kernels": kernels,
        "central_body": args.central_body,
        "frame": args.frame,
        "mu_central_km3_s2": args.mu_central_km3_s2,
        "gravitating_bodies": grav_bodies,
        "integrator": args.integrator,
        "rtol": args.rtol,
        "atol_position_km": args.atol_position_km,
        "atol_velocity_km_s": args.atol_velocity_km_s,
        "max_step_days": args.max_step_days,
        "arrival_search_window_days": args.arrival_search_window_days,
        "arrival_search_samples": args.arrival_search_samples,
        "endpoint_miss_threshold_km": args.endpoint_miss_threshold_km,
    }

    t_start = time.time()
    workers = args.workers
    if workers == 0:
        workers = max(1, os.cpu_count() or 1)

    results: List[Dict[str, Any]] = []
    if workers == 1:
        _worker_init(config)
        for r in routes:
            results.append(_validate_route_worker((r, catalog)))
    else:
        ctx = mp.get_context(args.multiprocessing_start_method)
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx, initializer=_worker_init, initargs=(config,)) as ex:
            futs = [ex.submit(_validate_route_worker, (r, catalog)) for r in routes]
            for fut in as_completed(futs):
                results.append(fut.result())
        # Deterministic order by selected rank then route id.
        results.sort(key=lambda x: (finite(x.get("route_rank"), 1e9), str(x.get("route_id", ""))))

    write_csv(args.output_csv, results)
    write_jsonl(args.output_jsonl, results)

    max_misses = [finite(r.get("max_endpoint_miss_km")) for r in results]
    max_misses = [x for x in max_misses if math.isfinite(x)]
    class_counts: Dict[str, int] = {}
    pass_count = 0
    for r in results:
        class_counts[str(r.get("worst_miss_class", "unknown"))] = class_counts.get(str(r.get("worst_miss_class", "unknown")), 0) + 1
        if bool(r.get("pass_endpoint_threshold")):
            pass_count += 1

    summary = {
        "schema_version": SCHEMA_VERSION,
        "input_jsonl": str(args.input_jsonl),
        "body_catalog": str(args.body_catalog),
        "central_body": args.central_body,
        "frame": args.frame,
        "mu_central_km3_s2": args.mu_central_km3_s2,
        "gravitating_bodies": grav_bodies,
        "integrator": {
            "method": args.integrator,
            "rtol": args.rtol,
            "atol_position_km": args.atol_position_km,
            "atol_velocity_km_s": args.atol_velocity_km_s,
            "max_step_days": args.max_step_days,
            "arrival_search_window_days": args.arrival_search_window_days,
            "arrival_search_samples": args.arrival_search_samples,
        },
        "routes_validated": len(results),
        "endpoint_miss_threshold_km": args.endpoint_miss_threshold_km,
        "routes_passing_endpoint_threshold": pass_count,
        "worst_class_counts": dict(sorted(class_counts.items())),
        "max_endpoint_miss_km": max(max_misses) if max_misses else None,
        "min_endpoint_miss_km": min(max_misses) if max_misses else None,
        "median_endpoint_miss_km": sorted(max_misses)[len(max_misses) // 2] if max_misses else None,
        "top_routes": [
            {
                "route_rank": r.get("route_rank"),
                "route_id": r.get("route_id"),
                "sequence": r.get("sequence"),
                "max_endpoint_miss_km": r.get("max_endpoint_miss_km"),
                "max_closest_approach_km": r.get("max_closest_approach_km"),
                "max_endpoint_speed_error_km_s": r.get("max_endpoint_speed_error_km_s"),
                "worst_miss_class": r.get("worst_miss_class"),
                "pass_endpoint_threshold": r.get("pass_endpoint_threshold"),
            }
            for r in sorted(results, key=lambda x: finite(x.get("max_endpoint_miss_km"), 1e99))[:10]
        ],
        "runtime_s": time.time() - t_start,
        "outputs": {
            "csv": str(args.output_csv),
            "jsonl": str(args.output_jsonl),
            "summary_json": str(args.output_json),
        },
    }
    write_json(args.output_json, summary)

    print("=" * 80)
    print("MGA SPICE N-BODY ARC VALIDATION V0.1")
    print("=" * 80)
    print(f"Input routes:      {len(routes)}")
    print(f"Validated routes:  {len(results)}")
    print(f"Workers:           {workers}")
    print(f"Gravitating bodies:{' ' if grav_bodies else ' none'}{', '.join(grav_bodies)}")
    print(f"Endpoint threshold:{args.endpoint_miss_threshold_km:.3g} km")
    print(f"Pass threshold:    {pass_count}")
    if max_misses:
        print(f"Endpoint miss km:  min={min(max_misses):.6g} median={summary['median_endpoint_miss_km']:.6g} max={max(max_misses):.6g}")
    print("\nWorst class counts:")
    for k, v in sorted(class_counts.items()):
        print(f"  - {k:<14} {v}")
    print("\nTop routes by max endpoint miss:")
    for i, r in enumerate(summary["top_routes"], start=1):
        print(
            f" {i:>2}. rank={r.get('route_rank')} | {r.get('sequence')} | "
            f"max_miss={finite(r.get('max_endpoint_miss_km')):.6g} km | "
            f"max_CA={finite(r.get('max_closest_approach_km')):.6g} km | "
            f"class={r.get('worst_miss_class')} | pass={r.get('pass_endpoint_threshold')}"
        )
    print("=" * 80)
    print(f"[OK] wrote CSV:   {args.output_csv}")
    print(f"[OK] wrote JSONL: {args.output_jsonl}")
    print(f"[OK] wrote JSON:  {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
