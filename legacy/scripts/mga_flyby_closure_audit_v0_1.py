#!/usr/bin/env python3
"""
mga_flyby_closure_audit_v0_1.py

Audit whether independently corrected patched-heliocentric Lambert legs can be
interpreted as an unpowered gravity-assist at each intermediate body.

Input is the JSONL emitted by mga_spice_arc_departure_corrector_v0_1.py.
For each corrected route, the script repropagates each corrected leg, computes
incoming and outgoing v-infinity vectors at intermediate bodies, and reports:
  - v_inf magnitude mismatch,
  - required turn angle,
  - required periapsis radius for an unpowered flyby,
  - margin against rp_min from BodyCatalog,
  - layover duration versus a simple SOI crossing-time proxy.

This is still a coarse patched-conic audit, not a B-plane targeter. It answers:
"Are these corrected arcs still compatible with a physical unpowered flyby?"
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
SCHEMA_VERSION = "mga_flyby_closure_audit.v0.1"
Vec3 = Tuple[float, float, float]
_WORKER_CONFIG: Dict[str, Any] = {}
_WORKER_SPICE = None


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


def vdot(a: Sequence[float], b: Sequence[float]) -> float:
    return float(a[0]) * float(b[0]) + float(a[1]) * float(b[1]) + float(a[2]) * float(b[2])


def vnorm(a: Sequence[float]) -> float:
    return math.sqrt(vdot(a, a))


def angle_deg(a: Sequence[float], b: Sequence[float]) -> float:
    na = vnorm(a)
    nb = vnorm(b)
    if na <= 0.0 or nb <= 0.0:
        return math.nan
    c = max(-1.0, min(1.0, vdot(a, b) / (na * nb)))
    return math.degrees(math.acos(c))


def stable_id(prefix: str, payload: Mapping[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=True, separators=(",", ":"), default=str)
    return f"{prefix}_{hashlib.sha1(raw.encode('utf-8')).hexdigest()[:16]}"


def json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, tuple):
        return list(obj)
    return str(obj)


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        x = json.load(f)
    if not isinstance(x, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return x


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
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
                out.append(dict(x))
    return out


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=json_default)
        f.write("\n")


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, separators=(",", ":"), default=json_default))
            f.write("\n")


@dataclass
class BodyInfo:
    name: str
    mu_km3_s2: float
    radius_km: Optional[float]
    rp_min_km: Optional[float]
    soi_km: Optional[float]


@dataclass
class LegEndpoint:
    leg_index: int
    origin: str
    target: str
    depart_et: float
    arrive_et: float
    tof_days: float
    r_arrive_km: Optional[Vec3]
    v_arrive_km_s: Optional[Vec3]
    ok: bool
    message: str


@dataclass
class FlybyAudit:
    route_id: str
    correction_id: str
    route_rank: Optional[int]
    sequence: str
    flyby_index: int
    body: str
    incoming_leg_index: int
    outgoing_leg_index: int
    incoming_epoch_et: float
    outgoing_epoch_et: float
    layover_days: float
    vinf_in_km_s: Optional[float]
    vinf_out_km_s: Optional[float]
    vinf_mag_mismatch_m_s: Optional[float]
    turn_angle_deg: Optional[float]
    rp_required_km: Optional[float]
    rp_min_km: Optional[float]
    rp_margin_km: Optional[float]
    soi_km: Optional[float]
    soi_crossing_time_proxy_days: Optional[float]
    layover_to_soi_time_ratio: Optional[float]
    pass_vinf_magnitude: bool
    pass_turn_geometry: bool
    pass_layover_proxy: bool
    pass_flyby: bool
    status: str


def _worker_init(config: Mapping[str, Any]) -> None:
    global _WORKER_CONFIG, _WORKER_SPICE
    _WORKER_CONFIG = dict(config)
    spice = importlib.import_module("spiceypy")
    for k in _WORKER_CONFIG.get("kernels", []):
        spice.furnsh(str(k))
    _WORKER_SPICE = spice


def _spice_state(body: str, et: float, center: str, frame: str) -> Tuple[Vec3, Vec3]:
    st, _lt = _WORKER_SPICE.spkezr(str(body), float(et), str(frame), "NONE", str(center))
    return (float(st[0]), float(st[1]), float(st[2])), (float(st[3]), float(st[4]), float(st[5]))


def _body_from_catalog(catalog: Mapping[str, Any], name: str) -> Optional[BodyInfo]:
    bodies = catalog.get("bodies") or {}
    ent = None
    if isinstance(bodies, Mapping):
        ent = bodies.get(name)
        if ent is None:
            for k, v in bodies.items():
                if str(k).lower() == name.lower():
                    ent = v
                    name = str(k)
                    break
    if not isinstance(ent, Mapping):
        return None
    mu = opt_float(ent.get("mu_km3_s2", ent.get("gm_km3_s2")))
    if mu is None:
        return None
    return BodyInfo(
        name=name,
        mu_km3_s2=mu,
        radius_km=opt_float(ent.get("radius_km", ent.get("equatorial_radius_km"))),
        rp_min_km=opt_float(ent.get("rp_min_km")),
        soi_km=opt_float(ent.get("sphere_of_influence_km", ent.get("soi_km"))),
    )


def _accel_central_frame(t: float, y: Sequence[float], central: str, frame: str, mu_c: float, perturbers: Sequence[BodyInfo]) -> List[float]:
    r = (float(y[0]), float(y[1]), float(y[2]))
    v = (float(y[3]), float(y[4]), float(y[5]))
    rn = vnorm(r)
    if rn <= 0.0:
        raise FloatingPointError("spacecraft at central body singularity")
    a = [-mu_c * r[0] / rn**3, -mu_c * r[1] / rn**3, -mu_c * r[2] / rn**3]
    for b in perturbers:
        rb, _vb = _spice_state(b.name, t, central, frame)
        dr = (rb[0] - r[0], rb[1] - r[1], rb[2] - r[2])
        drn = vnorm(dr)
        rbn = vnorm(rb)
        if drn <= 0.0 or rbn <= 0.0:
            continue
        mu = b.mu_km3_s2
        # Indirect term keeps equations in the non-inertial central-body frame.
        a[0] += mu * (dr[0] / drn**3 - rb[0] / rbn**3)
        a[1] += mu * (dr[1] / drn**3 - rb[1] / rbn**3)
        a[2] += mu * (dr[2] / drn**3 - rb[2] / rbn**3)
    return [v[0], v[1], v[2], a[0], a[1], a[2]]


def _propagate_endpoint(
    scipy_integrate: Any,
    r0: Vec3,
    v0: Vec3,
    depart_et: float,
    arrive_et: float,
    central: str,
    frame: str,
    mu_c: float,
    perturbers: Sequence[BodyInfo],
) -> Tuple[bool, Optional[Vec3], Optional[Vec3], str]:
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
        return False, None, None, f"integrator_exception:{exc}"
    if sol.y is None or sol.y.shape[1] == 0:
        return False, None, None, "empty_solution"
    yf = sol.y[:, -1]
    return bool(sol.success), (float(yf[0]), float(yf[1]), float(yf[2])), (float(yf[3]), float(yf[4]), float(yf[5])), str(getattr(sol, "message", ""))


def _extract_source_legs(corr_route: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    src = corr_route.get("source_route")
    if not isinstance(src, Mapping):
        raise ValueError("corrected route has no source_route")
    legs = src.get("leg_evals", [])
    if not isinstance(legs, Sequence) or isinstance(legs, (str, bytes)):
        raise ValueError("source_route has no leg_evals")
    out = [x for x in legs if isinstance(x, Mapping)]
    if not out:
        raise ValueError("empty leg_evals")
    return out


def _extract_corr_by_leg(corr_route: Mapping[str, Any]) -> Dict[int, Mapping[str, Any]]:
    out: Dict[int, Mapping[str, Any]] = {}
    for x in corr_route.get("leg_corrections", []) or []:
        if isinstance(x, Mapping):
            try:
                out[int(x.get("leg_index"))] = x
            except Exception:
                continue
    return out


def _perturbers_for_leg(catalog: Mapping[str, Any], origin: str, target: str) -> List[BodyInfo]:
    cfg = _WORKER_CONFIG
    out: List[BodyInfo] = []
    mode = str(cfg.get("dynamics_mode", "patched_heliocentric"))
    for name in cfg.get("gravitating_bodies", []):
        b = _body_from_catalog(catalog, str(name))
        if b is None or b.name == cfg.get("central_body"):
            continue
        if mode == "two_body":
            continue
        if mode == "patched_heliocentric" and b.name in {origin, target}:
            continue
        out.append(b)
    return out


def _required_rp(mu: float, vinf_km_s: float, turn_deg: float) -> Optional[float]:
    if mu <= 0.0 or vinf_km_s <= 0.0 or not math.isfinite(turn_deg):
        return None
    if turn_deg <= 0.0:
        return math.inf
    half = math.radians(turn_deg) / 2.0
    s = math.sin(half)
    if s <= 0.0:
        return math.inf
    val = (mu / (vinf_km_s * vinf_km_s)) * (1.0 / s - 1.0)
    return val if math.isfinite(val) else None


def _audit_route_worker(payload: Tuple[Dict[str, Any], Dict[str, Any]]) -> Dict[str, Any]:
    corr_route, catalog = payload
    scipy_integrate = importlib.import_module("scipy.integrate")
    cfg = _WORKER_CONFIG
    central = str(cfg["central_body"])
    frame = str(cfg["frame"])
    mu_c = float(cfg["mu_central_km3_s2"])
    max_vinf_mismatch_m_s = float(cfg["max_vinf_mismatch_m_s"])
    max_layover_to_soi_ratio = float(cfg["max_layover_to_soi_ratio"])
    max_abs_layover_days = float(cfg["max_abs_layover_days"])

    route_id = str(corr_route.get("route_id") or stable_id("route", corr_route))
    correction_id = str(corr_route.get("correction_id") or "")
    seq = str(corr_route.get("sequence") or "")
    try:
        rank = int(corr_route.get("route_rank")) if corr_route.get("route_rank") is not None else None
    except Exception:
        rank = None

    try:
        legs = _extract_source_legs(corr_route)
        corr_by_leg = _extract_corr_by_leg(corr_route)
    except Exception as exc:
        return {
            "schema_version": SCHEMA_VERSION,
            "route_id": route_id,
            "correction_id": correction_id,
            "route_rank": rank,
            "sequence": seq,
            "pass_all_flybys": False,
            "status": "bad_input",
            "error": str(exc),
            "flyby_audits": [],
            "source_corrected_route": corr_route,
        }

    endpoints: List[LegEndpoint] = []
    for i, leg in enumerate(legs):
        origin = str(leg.get("origin"))
        target = str(leg.get("target"))
        depart_et = finite(leg.get("depart_et"))
        arrive_et = finite(leg.get("arrive_et"))
        tof_days = (arrive_et - depart_et) / SECONDS_PER_DAY if math.isfinite(depart_et) and math.isfinite(arrive_et) else math.nan
        try:
            r0, _vb0 = _spice_state(origin, depart_et, central, frame)
            v_lam = vec3(leg.get("sc_v_depart_km_s"), "sc_v_depart_km_s")
            c = corr_by_leg.get(i, {})
            dv = (finite(c.get("dvx_km_s"), 0.0), finite(c.get("dvy_km_s"), 0.0), finite(c.get("dvz_km_s"), 0.0))
            v0 = vadd(v_lam, dv)
            pert = _perturbers_for_leg(catalog, origin, target)
            ok, rf, vf, msg = _propagate_endpoint(scipy_integrate, r0, v0, depart_et, arrive_et, central, frame, mu_c, pert)
            endpoints.append(LegEndpoint(i, origin, target, depart_et, arrive_et, tof_days, rf, vf, ok, msg))
        except Exception as exc:
            endpoints.append(LegEndpoint(i, origin, target, depart_et, arrive_et, tof_days, None, None, False, str(exc)))

    audits: List[FlybyAudit] = []
    for j in range(len(legs) - 1):
        leg_in = legs[j]
        leg_out = legs[j + 1]
        body = str(leg_in.get("target"))
        if body != str(leg_out.get("origin")):
            audits.append(FlybyAudit(route_id, correction_id, rank, seq, j, body, j, j + 1,
                                     finite(leg_in.get("arrive_et")), finite(leg_out.get("depart_et")), math.nan,
                                     None, None, None, None, None, None, None, None, None, None,
                                     False, False, False, False, "non_contiguous_body"))
            continue
        ep_in = endpoints[j]
        ep_out = endpoints[j + 1]
        body_info = _body_from_catalog(catalog, body)
        t_in = finite(leg_in.get("arrive_et"))
        t_out = finite(leg_out.get("depart_et"))
        layover_days = (t_out - t_in) / SECONDS_PER_DAY if math.isfinite(t_in) and math.isfinite(t_out) else math.nan
        status = "ok"
        vin = vout = mismatch = turn = rp_req = rp_min = rp_margin = soi = soi_proxy = ratio = None
        pass_mag = pass_turn = pass_lay = False
        try:
            if not ep_in.ok or ep_in.v_arrive_km_s is None:
                raise ValueError(f"incoming_endpoint_failed:{ep_in.message}")
            _rb_in, vb_in = _spice_state(body, t_in, central, frame)
            _rb_out, vb_out = _spice_state(body, t_out, central, frame)
            v_inf_in_vec = vsub(ep_in.v_arrive_km_s, vb_in)
            # Outgoing v_inf is the corrected departure state of the next leg.
            v_lam_out = vec3(leg_out.get("sc_v_depart_km_s"), "next sc_v_depart_km_s")
            c_out = corr_by_leg.get(j + 1, {})
            dv_out = (finite(c_out.get("dvx_km_s"), 0.0), finite(c_out.get("dvy_km_s"), 0.0), finite(c_out.get("dvz_km_s"), 0.0))
            v_sc_out = vadd(v_lam_out, dv_out)
            v_inf_out_vec = vsub(v_sc_out, vb_out)
            vin = vnorm(v_inf_in_vec)
            vout = vnorm(v_inf_out_vec)
            mismatch = abs(vout - vin) * 1000.0
            turn = angle_deg(tuple(-x for x in v_inf_in_vec), v_inf_out_vec)
            # Use mean v_inf for required periapsis; conservative audits can also inspect max(vin,vout).
            v_eff = 0.5 * (vin + vout)
            if body_info is None:
                raise ValueError("missing_body_catalog_entry")
            rp_min = body_info.rp_min_km
            soi = body_info.soi_km
            rp_req = _required_rp(body_info.mu_km3_s2, v_eff, turn)
            if rp_req is not None and rp_min is not None:
                rp_margin = rp_req - rp_min
            if soi is not None and v_eff > 1e-12:
                soi_proxy = 2.0 * soi / v_eff / SECONDS_PER_DAY
                ratio = layover_days / soi_proxy if soi_proxy > 0.0 and math.isfinite(layover_days) else None
            pass_mag = mismatch <= max_vinf_mismatch_m_s
            pass_turn = (rp_margin is not None and rp_margin >= float(cfg["min_rp_margin_km"]))
            pass_lay = bool(math.isfinite(layover_days) and layover_days >= 0.0 and layover_days <= max_abs_layover_days)
            if ratio is not None:
                pass_lay = pass_lay and (ratio <= max_layover_to_soi_ratio)
        except Exception as exc:
            status = str(exc)
        pass_all = bool(pass_mag and pass_turn and pass_lay and status == "ok")
        audits.append(FlybyAudit(
            route_id=route_id,
            correction_id=correction_id,
            route_rank=rank,
            sequence=seq,
            flyby_index=j,
            body=body,
            incoming_leg_index=j,
            outgoing_leg_index=j + 1,
            incoming_epoch_et=t_in,
            outgoing_epoch_et=t_out,
            layover_days=layover_days,
            vinf_in_km_s=vin,
            vinf_out_km_s=vout,
            vinf_mag_mismatch_m_s=mismatch,
            turn_angle_deg=turn,
            rp_required_km=rp_req,
            rp_min_km=rp_min,
            rp_margin_km=rp_margin,
            soi_km=soi,
            soi_crossing_time_proxy_days=soi_proxy,
            layover_to_soi_time_ratio=ratio,
            pass_vinf_magnitude=pass_mag,
            pass_turn_geometry=pass_turn,
            pass_layover_proxy=pass_lay,
            pass_flyby=pass_all,
            status=status,
        ))

    pass_all_flybys = bool(audits and all(a.pass_flyby for a in audits))
    min_rp_margin = min([a.rp_margin_km for a in audits if a.rp_margin_km is not None and math.isfinite(a.rp_margin_km)], default=None)
    max_vinf_mismatch = max([a.vinf_mag_mismatch_m_s for a in audits if a.vinf_mag_mismatch_m_s is not None and math.isfinite(a.vinf_mag_mismatch_m_s)], default=None)
    max_lay_ratio = max([a.layover_to_soi_time_ratio for a in audits if a.layover_to_soi_time_ratio is not None and math.isfinite(a.layover_to_soi_time_ratio)], default=None)
    out = {
        "schema_version": SCHEMA_VERSION,
        "closure_id": stable_id("fbclose", {"corr": correction_id, "route": route_id, "pass": pass_all_flybys}),
        "route_id": route_id,
        "correction_id": correction_id,
        "route_rank": rank,
        "sequence": seq,
        "pass_all_flybys": pass_all_flybys,
        "n_flybys": len(audits),
        "min_rp_margin_km": min_rp_margin,
        "max_vinf_mag_mismatch_m_s": max_vinf_mismatch,
        "max_layover_to_soi_time_ratio": max_lay_ratio,
        "total_departure_correction_m_s": corr_route.get("total_departure_correction_m_s"),
        "max_departure_correction_m_s": corr_route.get("max_departure_correction_m_s"),
        "max_miss_after_km": corr_route.get("max_miss_after_km"),
        "status": "ok" if audits else "no_flybys",
        "flyby_audits": [asdict(a) for a in audits],
        "source_corrected_route": corr_route,
    }
    return out


def flatten_rows(route_audit: Mapping[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for a in route_audit.get("flyby_audits", []) or []:
        if not isinstance(a, Mapping):
            continue
        rows.append({
            "closure_id": route_audit.get("closure_id"),
            "route_id": route_audit.get("route_id"),
            "correction_id": route_audit.get("correction_id"),
            "route_rank": route_audit.get("route_rank"),
            "sequence": route_audit.get("sequence"),
            "pass_all_flybys": int(bool(route_audit.get("pass_all_flybys"))),
            "total_departure_correction_m_s": route_audit.get("total_departure_correction_m_s"),
            "max_miss_after_km": route_audit.get("max_miss_after_km"),
            "flyby_index": a.get("flyby_index"),
            "body": a.get("body"),
            "layover_days": a.get("layover_days"),
            "vinf_in_km_s": a.get("vinf_in_km_s"),
            "vinf_out_km_s": a.get("vinf_out_km_s"),
            "vinf_mag_mismatch_m_s": a.get("vinf_mag_mismatch_m_s"),
            "turn_angle_deg": a.get("turn_angle_deg"),
            "rp_required_km": a.get("rp_required_km"),
            "rp_min_km": a.get("rp_min_km"),
            "rp_margin_km": a.get("rp_margin_km"),
            "soi_km": a.get("soi_km"),
            "soi_crossing_time_proxy_days": a.get("soi_crossing_time_proxy_days"),
            "layover_to_soi_time_ratio": a.get("layover_to_soi_time_ratio"),
            "pass_vinf_magnitude": int(bool(a.get("pass_vinf_magnitude"))),
            "pass_turn_geometry": int(bool(a.get("pass_turn_geometry"))),
            "pass_layover_proxy": int(bool(a.get("pass_layover_proxy"))),
            "pass_flyby": int(bool(a.get("pass_flyby"))),
            "status": a.get("status"),
        })
    return rows


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "closure_id", "route_id", "correction_id", "route_rank", "sequence", "pass_all_flybys",
        "total_departure_correction_m_s", "max_miss_after_km", "flyby_index", "body", "layover_days",
        "vinf_in_km_s", "vinf_out_km_s", "vinf_mag_mismatch_m_s", "turn_angle_deg",
        "rp_required_km", "rp_min_km", "rp_margin_km", "soi_km", "soi_crossing_time_proxy_days",
        "layover_to_soi_time_ratio", "pass_vinf_magnitude", "pass_turn_geometry", "pass_layover_proxy",
        "pass_flyby", "status",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k) for k in fields})


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Audit corrected MGA routes for physical flyby closure.")
    p.add_argument("--bsp", required=True, type=Path)
    p.add_argument("--tpc", type=Path, default=None)
    p.add_argument("--body-catalog", required=True, type=Path)
    p.add_argument("--input-jsonl", required=True, type=Path, help="JSONL from mga_spice_arc_departure_corrector_v0_1.py")
    p.add_argument("--central-body", default="Sun")
    p.add_argument("--frame", default="J2000")
    p.add_argument("--mu-central-km3-s2", required=True, type=float)
    p.add_argument("--gravitating-bodies", nargs="+", default=[])
    p.add_argument("--dynamics-mode", default="patched_heliocentric", choices=["two_body", "patched_heliocentric", "full_nbody"])
    p.add_argument("--workers", type=int, default=1, help="0=os.cpu_count, 1=serial, N=processes")
    p.add_argument("--multiprocessing-start-method", default="spawn", choices=["spawn", "fork", "forkserver"])
    p.add_argument("--integrator", default="DOP853", choices=["DOP853", "RK45", "LSODA", "Radau", "BDF"])
    p.add_argument("--rtol", type=float, default=1e-10)
    p.add_argument("--atol-position-km", type=float, default=1e-6)
    p.add_argument("--atol-velocity-km-s", type=float, default=1e-12)
    p.add_argument("--max-step-days", type=float, default=2.0)
    p.add_argument("--max-vinf-mismatch-m-s", type=float, default=25.0)
    p.add_argument("--min-rp-margin-km", type=float, default=50.0)
    p.add_argument("--max-abs-layover-days", type=float, default=3.0)
    p.add_argument("--max-layover-to-soi-ratio", type=float, default=2.0,
                   help="Reject if layover exceeds this multiple of 2*SOI/vinf proxy. Use large value to disable.")
    p.add_argument("--route-rank", type=int, default=None)
    p.add_argument("--max-routes", type=int, default=0)
    p.add_argument("--output-csv", required=True, type=Path)
    p.add_argument("--output-jsonl", required=True, type=Path)
    p.add_argument("--output-json", required=True, type=Path)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    catalog = load_json(args.body_catalog)
    routes = read_jsonl(args.input_jsonl)
    if args.route_rank is not None:
        routes = [r for r in routes if int(r.get("route_rank", -999999)) == args.route_rank]
    if args.max_routes and args.max_routes > 0:
        routes = routes[: args.max_routes]
    if not routes:
        raise SystemExit("No corrected routes to audit.")
    kernels = [str(args.bsp)]
    if args.tpc is not None:
        kernels.append(str(args.tpc))
    grav_bodies = args.gravitating_bodies or sorted((catalog.get("bodies") or {}).keys())
    cfg = {
        "kernels": kernels,
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
        "max_vinf_mismatch_m_s": args.max_vinf_mismatch_m_s,
        "min_rp_margin_km": args.min_rp_margin_km,
        "max_abs_layover_days": args.max_abs_layover_days,
        "max_layover_to_soi_ratio": args.max_layover_to_soi_ratio,
    }
    workers = args.workers
    if workers == 0:
        workers = os.cpu_count() or 1
    workers = max(1, workers)
    payloads = [(r, catalog) for r in routes]
    results: List[Dict[str, Any]] = []
    if workers == 1:
        _worker_init(cfg)
        for pld in payloads:
            results.append(_audit_route_worker(pld))
    else:
        ctx = mp.get_context(args.multiprocessing_start_method)
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx, initializer=_worker_init, initargs=(cfg,)) as ex:
            futs = [ex.submit(_audit_route_worker, p) for p in payloads]
            for fut in as_completed(futs):
                results.append(fut.result())
    results.sort(key=lambda r: (
        0 if r.get("pass_all_flybys") else 1,
        finite(r.get("total_departure_correction_m_s"), 1e99),
        -finite(r.get("min_rp_margin_km"), -1e99),
    ))
    flat: List[Dict[str, Any]] = []
    for r in results:
        flat.extend(flatten_rows(r))
    write_csv(args.output_csv, flat)
    write_jsonl(args.output_jsonl, results)
    pass_count = sum(1 for r in results if r.get("pass_all_flybys"))
    rp_margins = [finite(r.get("min_rp_margin_km")) for r in results if math.isfinite(finite(r.get("min_rp_margin_km")))]
    mismatches = [finite(r.get("max_vinf_mag_mismatch_m_s")) for r in results if math.isfinite(finite(r.get("max_vinf_mag_mismatch_m_s")))]
    layratios = [finite(r.get("max_layover_to_soi_time_ratio")) for r in results if math.isfinite(finite(r.get("max_layover_to_soi_time_ratio")))]
    summary = {
        "schema_version": SCHEMA_VERSION,
        "input_jsonl": str(args.input_jsonl),
        "routes_processed": len(results),
        "routes_pass_all_flybys": pass_count,
        "workers": workers,
        "dynamics_mode": args.dynamics_mode,
        "thresholds": {
            "max_vinf_mismatch_m_s": args.max_vinf_mismatch_m_s,
            "min_rp_margin_km": args.min_rp_margin_km,
            "max_abs_layover_days": args.max_abs_layover_days,
            "max_layover_to_soi_ratio": args.max_layover_to_soi_ratio,
        },
        "min_rp_margin_km": {"min": min(rp_margins) if rp_margins else None, "median": sorted(rp_margins)[len(rp_margins)//2] if rp_margins else None, "max": max(rp_margins) if rp_margins else None},
        "max_vinf_mismatch_m_s": {"min": min(mismatches) if mismatches else None, "median": sorted(mismatches)[len(mismatches)//2] if mismatches else None, "max": max(mismatches) if mismatches else None},
        "max_layover_to_soi_time_ratio": {"min": min(layratios) if layratios else None, "median": sorted(layratios)[len(layratios)//2] if layratios else None, "max": max(layratios) if layratios else None},
        "top_routes": [{k: r.get(k) for k in ["closure_id", "route_id", "route_rank", "sequence", "pass_all_flybys", "min_rp_margin_km", "max_vinf_mag_mismatch_m_s", "max_layover_to_soi_time_ratio", "total_departure_correction_m_s"]} for r in results[:20]],
    }
    write_json(args.output_json, summary)

    print("=" * 80)
    print("MGA FLYBY CLOSURE AUDIT V0.1")
    print("=" * 80)
    print(f"Routes processed: {len(results)}")
    print(f"Pass flyby audit:{pass_count}")
    print(f"Workers:          {workers}")
    print(f"Dynamics mode:    {args.dynamics_mode}")
    if rp_margins:
        print(f"rp margin km:     min={min(rp_margins):.6g} median={sorted(rp_margins)[len(rp_margins)//2]:.6g} max={max(rp_margins):.6g}")
    if mismatches:
        print(f"v∞ mismatch m/s:  min={min(mismatches):.6g} median={sorted(mismatches)[len(mismatches)//2]:.6g} max={max(mismatches):.6g}")
    if layratios:
        print(f"lay/SOI proxy:    min={min(layratios):.6g} median={sorted(layratios)[len(layratios)//2]:.6g} max={max(layratios):.6g}")
    print("\nTop routes:")
    for i, r in enumerate(results[:10], start=1):
        print(f" {i}. {r.get('sequence')} | pass={r.get('pass_all_flybys')} | rp_margin={finite(r.get('min_rp_margin_km')):.3f} km | "
              f"v∞ mismatch={finite(r.get('max_vinf_mag_mismatch_m_s')):.3f} m/s | lay_ratio={finite(r.get('max_layover_to_soi_time_ratio')):.3f} | "
              f"corr={finite(r.get('total_departure_correction_m_s')):.3f} m/s")
    print("=" * 80)
    print(f"[OK] wrote CSV:   {args.output_csv}")
    print(f"[OK] wrote JSONL: {args.output_jsonl}")
    print(f"[OK] wrote JSON:  {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
