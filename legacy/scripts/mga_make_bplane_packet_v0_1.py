#!/usr/bin/env python3
"""
mga_make_bplane_packet_v0_1.py

Build a compact B-plane / local-targeting packet from MGA flyby-closure audit
records. This consumes JSONL emitted by mga_flyby_closure_audit_v0_2.py and
reconstructs corrected patched-heliocentric endpoints so the packet contains
vector information, not only scalar audit metrics.

Scope:
  - connected unpowered flyby candidates that already passed closure audit;
  - approximate B-plane coordinates from incoming/outgoing v-infinity vectors;
  - route/leg correction metadata for downstream B-plane targeter / local N-body.

This is not yet a high-fidelity B-plane targeter. It creates the handoff packet.
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
SCHEMA_VERSION = "mga_bplane_packet.v0.1"
Vec3 = Tuple[float, float, float]
_WORKER_CONFIG: Dict[str, Any] = {}
_WORKER_SPICE = None
_WORKER_NP = None


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


def vmul(s: float, a: Sequence[float]) -> Vec3:
    return (float(s) * float(a[0]), float(s) * float(a[1]), float(s) * float(a[2]))


def vdot(a: Sequence[float], b: Sequence[float]) -> float:
    return float(a[0]) * float(b[0]) + float(a[1]) * float(b[1]) + float(a[2]) * float(b[2])


def vcross(a: Sequence[float], b: Sequence[float]) -> Vec3:
    return (
        float(a[1]) * float(b[2]) - float(a[2]) * float(b[1]),
        float(a[2]) * float(b[0]) - float(a[0]) * float(b[2]),
        float(a[0]) * float(b[1]) - float(a[1]) * float(b[0]),
    )


def vnorm(a: Sequence[float]) -> float:
    return math.sqrt(max(0.0, vdot(a, a)))


def vunit(a: Sequence[float], fallback: Optional[Vec3] = None) -> Vec3:
    n = vnorm(a)
    if n <= 0.0 or not math.isfinite(n):
        if fallback is not None:
            return fallback
        raise ValueError(f"Cannot normalize vector {a!r}")
    return (float(a[0]) / n, float(a[1]) / n, float(a[2]) / n)


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
class BodyMu:
    name: str
    mu_km3_s2: float


@dataclass
class PropEndpoint:
    ok: bool
    r_km: Optional[Vec3]
    v_km_s: Optional[Vec3]
    message: str


@dataclass
class LegPacket:
    leg_index: int
    origin: str
    target: str
    depart_et: float
    arrive_et: float
    tof_days: float
    dv_correction_km_s: Vec3
    dv_correction_m_s: float
    miss_after_km: Optional[float]
    sc_v_depart_corrected_km_s: Vec3
    sc_v_arrive_corrected_km_s: Optional[Vec3]


@dataclass
class FlybyPacket:
    flyby_index: int
    body: str
    encounter_et: float
    vinf_in_vec_km_s: Vec3
    vinf_out_vec_km_s: Vec3
    vinf_in_km_s: float
    vinf_out_km_s: float
    vinf_mismatch_m_s: float
    turn_angle_deg: float
    mu_km3_s2: float
    rp_min_km: Optional[float]
    rp_required_km: Optional[float]
    rp_margin_km: Optional[float]
    b_magnitude_km: Optional[float]
    b_dot_t_km: Optional[float]
    b_dot_r_km: Optional[float]
    s_hat: Vec3
    t_hat: Vec3
    r_hat: Vec3
    h_hat: Vec3
    b_hat: Vec3
    pass_flyby: bool


@dataclass
class RoutePacket:
    packet_id: str
    closure_id: str
    correction_id: str
    route_id: str
    route_rank: Optional[int]
    sequence: str
    pass_all_flybys: bool
    objective: Optional[float]
    total_tof_days: Optional[float]
    total_departure_correction_m_s: Optional[float]
    max_miss_after_km: Optional[float]
    min_rp_margin_km: Optional[float]
    max_vinf_mismatch_m_s: Optional[float]
    legs: List[Dict[str, Any]]
    flybys: List[Dict[str, Any]]
    source_closure: Dict[str, Any]


def _worker_init(config: Mapping[str, Any]) -> None:
    global _WORKER_CONFIG, _WORKER_SPICE, _WORKER_NP
    _WORKER_CONFIG = dict(config)
    _WORKER_SPICE = importlib.import_module("spiceypy")
    _WORKER_NP = importlib.import_module("numpy")
    _WORKER_SPICE.kclear()
    for k in _WORKER_CONFIG.get("kernels", []):
        _WORKER_SPICE.furnsh(str(k))


def _spice_state(body: str, et: float, center: str, frame: str) -> Tuple[Vec3, Vec3]:
    st, _lt = _WORKER_SPICE.spkezr(str(body), float(et), str(frame), "NONE", str(center))
    return (float(st[0]), float(st[1]), float(st[2])), (float(st[3]), float(st[4]), float(st[5]))


def _body_from_catalog(catalog: Mapping[str, Any], name: str) -> Optional[BodyInfo]:
    bodies = catalog.get("bodies") or {}
    ent = None
    key_name = name
    if isinstance(bodies, Mapping):
        ent = bodies.get(name)
        if ent is None:
            for k, v in bodies.items():
                if str(k).lower() == name.lower():
                    ent = v
                    key_name = str(k)
                    break
    if not isinstance(ent, Mapping):
        return None
    mu = opt_float(ent.get("mu_km3_s2", ent.get("gm_km3_s2")))
    if mu is None:
        return None
    return BodyInfo(
        name=key_name,
        mu_km3_s2=mu,
        radius_km=opt_float(ent.get("radius_km", ent.get("equatorial_radius_km"))),
        rp_min_km=opt_float(ent.get("rp_min_km")),
        soi_km=opt_float(ent.get("soi_km", ent.get("sphere_of_influence_km"))),
    )


def _body_mu_from_catalog(catalog: Mapping[str, Any], name: str) -> Optional[BodyMu]:
    b = _body_from_catalog(catalog, name)
    if b is None:
        return None
    return BodyMu(b.name, b.mu_km3_s2)


def _accel_central_frame(t: float, y: Any, central: str, frame: str, mu_c: float, perturbers: Sequence[BodyMu]) -> List[float]:
    np = _WORKER_NP
    r = np.array([y[0], y[1], y[2]], dtype=float)
    v = np.array([y[3], y[4], y[5]], dtype=float)
    nr = float(np.linalg.norm(r))
    if nr <= 0.0:
        a = np.zeros(3)
    else:
        a = -mu_c * r / (nr ** 3)
    for p in perturbers:
        try:
            rp_tuple, _vp = _spice_state(p.name, t, central, frame)
        except Exception:
            continue
        rp = np.array(rp_tuple, dtype=float)
        dr = rp - r
        ndr = float(np.linalg.norm(dr))
        nrp = float(np.linalg.norm(rp))
        if ndr > 0.0:
            a += p.mu_km3_s2 * dr / (ndr ** 3)
        if nrp > 0.0:
            # indirect term because this frame is centered on the accelerating central body
            a -= p.mu_km3_s2 * rp / (nrp ** 3)
    return [float(v[0]), float(v[1]), float(v[2]), float(a[0]), float(a[1]), float(a[2])]


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
) -> PropEndpoint:
    cfg = _WORKER_CONFIG
    y0 = [r0[0], r0[1], r0[2], v0[0], v0[1], v0[2]]
    try:
        sol = scipy_integrate.solve_ivp(
            lambda t, y: _accel_central_frame(t, y, central, frame, mu_c, perturbers),
            (float(depart_et), float(arrive_et)),
            y0,
            method=str(cfg["integrator"]),
            rtol=float(cfg["rtol"]),
            atol=[float(cfg["atol_position_km"])] * 3 + [float(cfg["atol_velocity_km_s"])] * 3,
            max_step=float(cfg["max_step_days"]) * SECONDS_PER_DAY,
            dense_output=False,
        )
    except Exception as exc:
        return PropEndpoint(False, None, None, f"integrator_exception:{exc}")
    if sol.y is None or sol.y.shape[1] == 0:
        return PropEndpoint(False, None, None, "empty_solution")
    yf = sol.y[:, -1]
    return PropEndpoint(bool(sol.success), (float(yf[0]), float(yf[1]), float(yf[2])), (float(yf[3]), float(yf[4]), float(yf[5])), str(getattr(sol, "message", "")))


def _extract_source_route(closure: Mapping[str, Any]) -> Mapping[str, Any]:
    corr = closure.get("source_corrected_route")
    if not isinstance(corr, Mapping):
        raise ValueError("closure record has no source_corrected_route")
    src = corr.get("source_route")
    if not isinstance(src, Mapping):
        raise ValueError("source_corrected_route has no source_route")
    return src


def _extract_corrected_route(closure: Mapping[str, Any]) -> Mapping[str, Any]:
    corr = closure.get("source_corrected_route")
    if not isinstance(corr, Mapping):
        raise ValueError("closure record has no source_corrected_route")
    return corr


def _legs_from_source(src: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    legs = src.get("leg_evals", [])
    if not isinstance(legs, Sequence) or isinstance(legs, (str, bytes)):
        raise ValueError("source route has no leg_evals list")
    out = [x for x in legs if isinstance(x, Mapping)]
    if not out:
        raise ValueError("empty leg_evals")
    return out


def _corr_by_leg(corr_route: Mapping[str, Any]) -> Dict[int, Mapping[str, Any]]:
    out: Dict[int, Mapping[str, Any]] = {}
    for x in corr_route.get("leg_corrections", []) or []:
        if isinstance(x, Mapping):
            idx = x.get("leg_index")
            if idx is not None:
                try:
                    out[int(idx)] = x
                except Exception:
                    pass
    return out


def _perturbers_for_leg(catalog: Mapping[str, Any], origin: str, target: str) -> List[BodyMu]:
    cfg = _WORKER_CONFIG
    out: List[BodyMu] = []
    mode = str(cfg.get("dynamics_mode", "patched_heliocentric"))
    for name in cfg.get("gravitating_bodies", []):
        b = _body_mu_from_catalog(catalog, str(name))
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


def _impact_parameter(mu: float, vinf_km_s: float, turn_deg: float) -> Optional[float]:
    if mu <= 0.0 or vinf_km_s <= 0.0 or not math.isfinite(turn_deg) or turn_deg <= 0.0:
        return None
    half = math.radians(turn_deg) / 2.0
    tan_half = math.tan(half)
    if tan_half <= 0.0:
        return None
    b = mu / (vinf_km_s * vinf_km_s * tan_half)
    return b if math.isfinite(b) else None


def _bplane_axes(vinf_in: Vec3, vinf_out: Vec3) -> Tuple[Vec3, Vec3, Vec3, Vec3, Vec3]:
    # Convention used here: S is the incoming v-infinity direction in the central frame.
    # T/R complete a right-handed basis on the plane normal to S. T is anchored to J2000 +Z
    # unless that is near-singular. Bhat is chosen from the turn-plane angular momentum.
    s_hat = vunit(vinf_in)
    k_hat = (0.0, 0.0, 1.0)
    if vnorm(vcross(k_hat, s_hat)) < 1e-8:
        k_hat = (0.0, 1.0, 0.0)
    t_hat = vunit(vcross(k_hat, s_hat))
    r_hat = vunit(vcross(s_hat, t_hat))
    h = vcross(vinf_in, vinf_out)
    if vnorm(h) < 1e-12:
        h_hat = r_hat
    else:
        h_hat = vunit(h)
    # This B direction is perpendicular to S and lies in the turn plane.
    b_hat = vunit(vcross(s_hat, h_hat), fallback=t_hat)
    return s_hat, t_hat, r_hat, h_hat, b_hat


def _route_score(packet: Mapping[str, Any]) -> Tuple[float, float, float]:
    # lower is better; prefer low correction, high rp margin, low mismatch
    return (
        finite(packet.get("total_departure_correction_m_s"), 1e99),
        -finite(packet.get("min_rp_margin_km"), -1e99),
        finite(packet.get("max_vinf_mismatch_m_s"), 1e99),
    )


def _make_packet_worker(payload: Tuple[Dict[str, Any], Dict[str, Any]]) -> Dict[str, Any]:
    closure, catalog = payload
    scipy_integrate = importlib.import_module("scipy.integrate")
    cfg = _WORKER_CONFIG
    central = str(cfg["central_body"])
    frame = str(cfg["frame"])
    mu_c = float(cfg["mu_central_km3_s2"])
    try:
        if not bool(closure.get("pass_all_flybys")) and bool(cfg.get("require_pass", True)):
            raise ValueError("closure did not pass flyby audit")
        src = _extract_source_route(closure)
        corr = _extract_corrected_route(closure)
        legs_src = _legs_from_source(src)
        corr_by_idx = _corr_by_leg(corr)
        endpoints: List[PropEndpoint] = []
        leg_packets: List[LegPacket] = []
        for i, leg in enumerate(legs_src):
            origin = str(leg.get("origin"))
            target = str(leg.get("target"))
            depart_et = finite(leg.get("depart_et"))
            arrive_et = finite(leg.get("arrive_et"))
            tof_days = (arrive_et - depart_et) / SECONDS_PER_DAY
            r0, _vo = _spice_state(origin, depart_et, central, frame)
            v_lam = vec3(leg.get("sc_v_depart_km_s"), "sc_v_depart_km_s")
            c = corr_by_idx.get(i, {})
            dv = (finite(c.get("dvx_km_s"), 0.0), finite(c.get("dvy_km_s"), 0.0), finite(c.get("dvz_km_s"), 0.0))
            v0 = vadd(v_lam, dv)
            pert = _perturbers_for_leg(catalog, origin, target)
            ep = _propagate_endpoint(scipy_integrate, r0, v0, depart_et, arrive_et, central, frame, mu_c, pert)
            endpoints.append(ep)
            leg_packets.append(LegPacket(
                leg_index=i,
                origin=origin,
                target=target,
                depart_et=depart_et,
                arrive_et=arrive_et,
                tof_days=tof_days,
                dv_correction_km_s=dv,
                dv_correction_m_s=finite(c.get("dv_norm_m_s"), vnorm(dv) * 1000.0),
                miss_after_km=opt_float(c.get("miss_after_km")),
                sc_v_depart_corrected_km_s=v0,
                sc_v_arrive_corrected_km_s=ep.v_km_s,
            ))

        flyby_packets: List[FlybyPacket] = []
        for j in range(len(legs_src) - 1):
            lin = legs_src[j]
            lout = legs_src[j + 1]
            body = str(lin.get("target"))
            if body != str(lout.get("origin")):
                raise ValueError(f"non-contiguous flyby body at index {j}: {body} vs {lout.get('origin')}")
            body_info = _body_from_catalog(catalog, body)
            if body_info is None:
                raise ValueError(f"missing body catalog entry for {body}")
            t_enc = finite(lin.get("arrive_et"))
            if not endpoints[j].ok or endpoints[j].v_km_s is None:
                raise ValueError(f"incoming leg endpoint failed at flyby {body}: {endpoints[j].message}")
            _rb, vb = _spice_state(body, t_enc, central, frame)
            vinf_in_vec = vsub(endpoints[j].v_km_s, vb)
            vout_sc = leg_packets[j + 1].sc_v_depart_corrected_km_s
            vinf_out_vec = vsub(vout_sc, vb)
            vin = vnorm(vinf_in_vec)
            vout = vnorm(vinf_out_vec)
            mismatch = abs(vout - vin) * 1000.0
            turn = angle_deg(vinf_in_vec, vinf_out_vec)
            v_eff = 0.5 * (vin + vout)
            rp_req = _required_rp(body_info.mu_km3_s2, v_eff, turn)
            rp_margin = None
            if rp_req is not None and body_info.rp_min_km is not None:
                rp_margin = rp_req - body_info.rp_min_km
            bmag = _impact_parameter(body_info.mu_km3_s2, v_eff, turn)
            s_hat, t_hat, r_hat, h_hat, b_hat = _bplane_axes(vinf_in_vec, vinf_out_vec)
            if bmag is not None:
                bvec = vmul(bmag, b_hat)
                bdt = vdot(bvec, t_hat)
                bdr = vdot(bvec, r_hat)
            else:
                bdt = bdr = None
            flyby_packets.append(FlybyPacket(
                flyby_index=j,
                body=body,
                encounter_et=t_enc,
                vinf_in_vec_km_s=vinf_in_vec,
                vinf_out_vec_km_s=vinf_out_vec,
                vinf_in_km_s=vin,
                vinf_out_km_s=vout,
                vinf_mismatch_m_s=mismatch,
                turn_angle_deg=turn,
                mu_km3_s2=body_info.mu_km3_s2,
                rp_min_km=body_info.rp_min_km,
                rp_required_km=rp_req,
                rp_margin_km=rp_margin,
                b_magnitude_km=bmag,
                b_dot_t_km=bdt,
                b_dot_r_km=bdr,
                s_hat=s_hat,
                t_hat=t_hat,
                r_hat=r_hat,
                h_hat=h_hat,
                b_hat=b_hat,
                pass_flyby=bool((rp_margin is None or rp_margin >= float(cfg["min_rp_margin_km"])) and mismatch <= float(cfg["max_vinf_mismatch_m_s"])),
            ))

        objective = opt_float(src.get("objective"))
        total_tof = opt_float(src.get("total_tof_days"))
        packet = RoutePacket(
            packet_id=stable_id("bppkt", {"closure_id": closure.get("closure_id"), "corr": closure.get("correction_id"), "seq": closure.get("sequence")}),
            closure_id=str(closure.get("closure_id") or ""),
            correction_id=str(closure.get("correction_id") or corr.get("correction_id") or ""),
            route_id=str(closure.get("route_id") or corr.get("route_id") or src.get("refined_id") or ""),
            route_rank=int(closure.get("route_rank")) if closure.get("route_rank") is not None else None,
            sequence=str(closure.get("sequence") or corr.get("sequence") or ""),
            pass_all_flybys=bool(closure.get("pass_all_flybys")) and all(f.pass_flyby for f in flyby_packets),
            objective=objective,
            total_tof_days=total_tof,
            total_departure_correction_m_s=opt_float(closure.get("total_departure_correction_m_s", corr.get("total_departure_correction_m_s"))),
            max_miss_after_km=opt_float(closure.get("max_miss_after_km", corr.get("max_miss_after_km"))),
            min_rp_margin_km=min([f.rp_margin_km for f in flyby_packets if f.rp_margin_km is not None and math.isfinite(f.rp_margin_km)], default=None),
            max_vinf_mismatch_m_s=max([f.vinf_mismatch_m_s for f in flyby_packets if math.isfinite(f.vinf_mismatch_m_s)], default=None),
            legs=[asdict(x) for x in leg_packets],
            flybys=[asdict(x) for x in flyby_packets],
            source_closure=dict(closure),
        )
        d = asdict(packet)
        d["status"] = "ok"
        return d
    except Exception as exc:
        return {
            "schema_version": SCHEMA_VERSION,
            "packet_id": stable_id("bppkt_bad", {"closure": closure.get("closure_id"), "error": str(exc)}),
            "closure_id": closure.get("closure_id"),
            "correction_id": closure.get("correction_id"),
            "route_id": closure.get("route_id"),
            "route_rank": closure.get("route_rank"),
            "sequence": closure.get("sequence"),
            "pass_all_flybys": False,
            "status": "error",
            "error": str(exc),
            "source_closure": closure,
        }


def flatten_packet_rows(p: Mapping[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    flybys = p.get("flybys") or []
    if not isinstance(flybys, list):
        return rows
    for fb in flybys:
        if not isinstance(fb, Mapping):
            continue
        rows.append({
            "packet_id": p.get("packet_id"),
            "route_id": p.get("route_id"),
            "route_rank": p.get("route_rank"),
            "sequence": p.get("sequence"),
            "status": p.get("status"),
            "pass_all_flybys": int(bool(p.get("pass_all_flybys"))),
            "objective": p.get("objective"),
            "total_tof_days": p.get("total_tof_days"),
            "total_departure_correction_m_s": p.get("total_departure_correction_m_s"),
            "max_miss_after_km": p.get("max_miss_after_km"),
            "flyby_index": fb.get("flyby_index"),
            "body": fb.get("body"),
            "encounter_et": fb.get("encounter_et"),
            "vinf_in_km_s": fb.get("vinf_in_km_s"),
            "vinf_out_km_s": fb.get("vinf_out_km_s"),
            "vinf_mismatch_m_s": fb.get("vinf_mismatch_m_s"),
            "turn_angle_deg": fb.get("turn_angle_deg"),
            "rp_required_km": fb.get("rp_required_km"),
            "rp_min_km": fb.get("rp_min_km"),
            "rp_margin_km": fb.get("rp_margin_km"),
            "b_magnitude_km": fb.get("b_magnitude_km"),
            "b_dot_t_km": fb.get("b_dot_t_km"),
            "b_dot_r_km": fb.get("b_dot_r_km"),
            "pass_flyby": int(bool(fb.get("pass_flyby"))),
        })
    return rows


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "packet_id", "route_id", "route_rank", "sequence", "status", "pass_all_flybys", "objective",
        "total_tof_days", "total_departure_correction_m_s", "max_miss_after_km", "flyby_index", "body",
        "encounter_et", "vinf_in_km_s", "vinf_out_km_s", "vinf_mismatch_m_s", "turn_angle_deg",
        "rp_required_km", "rp_min_km", "rp_margin_km", "b_magnitude_km", "b_dot_t_km", "b_dot_r_km",
        "pass_flyby",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k) for k in fields})


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build B-plane/local-targeting packet from closure-audited MGA routes.")
    p.add_argument("--bsp", required=True, type=Path)
    p.add_argument("--tpc", type=Path, default=None)
    p.add_argument("--body-catalog", required=True, type=Path)
    p.add_argument("--input-jsonl", required=True, type=Path, help="JSONL from mga_flyby_closure_audit_v0_2.py")
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
    p.add_argument("--require-pass", action="store_true", default=True)
    p.add_argument("--include-failed", action="store_true", help="Do not filter out failed closure records before packet generation.")
    p.add_argument("--top-n", type=int, default=10)
    p.add_argument("--route-rank", type=int, default=None)
    p.add_argument("--output-csv", required=True, type=Path)
    p.add_argument("--output-jsonl", required=True, type=Path)
    p.add_argument("--output-json", required=True, type=Path)
    p.add_argument("--output-packet-json", required=True, type=Path)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    catalog = load_json(args.body_catalog)
    closures = read_jsonl(args.input_jsonl)
    if args.route_rank is not None:
        closures = [r for r in closures if int(r.get("route_rank", -999999)) == args.route_rank]
    if not args.include_failed:
        closures = [r for r in closures if bool(r.get("pass_all_flybys"))]
    if args.top_n and args.top_n > 0:
        closures = sorted(closures, key=lambda r: (
            finite(r.get("total_departure_correction_m_s"), 1e99),
            -finite(r.get("min_rp_margin_km"), -1e99),
            finite(r.get("max_vinf_mag_mismatch_m_s"), 1e99),
        ))[: args.top_n]
    if not closures:
        raise SystemExit("No closure records selected for packet generation.")
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
        "require_pass": bool(args.require_pass),
    }
    workers = args.workers
    if workers == 0:
        workers = os.cpu_count() or 1
    workers = max(1, workers)
    payloads = [(r, catalog) for r in closures]
    packets: List[Dict[str, Any]] = []
    if workers == 1:
        _worker_init(cfg)
        for pld in payloads:
            packets.append(_make_packet_worker(pld))
    else:
        ctx = mp.get_context(args.multiprocessing_start_method)
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx, initializer=_worker_init, initargs=(cfg,)) as ex:
            futs = [ex.submit(_make_packet_worker, p) for p in payloads]
            for fut in as_completed(futs):
                packets.append(fut.result())
    packets.sort(key=_route_score)
    rows: List[Dict[str, Any]] = []
    for p in packets:
        rows.extend(flatten_packet_rows(p))
    write_csv(args.output_csv, rows)
    write_jsonl(args.output_jsonl, packets)
    ok_packets = [p for p in packets if p.get("status") == "ok" and p.get("pass_all_flybys")]
    packet_payload = {
        "schema_version": SCHEMA_VERSION,
        "source_input_jsonl": str(args.input_jsonl),
        "central_body": args.central_body,
        "frame": args.frame,
        "dynamics_mode": args.dynamics_mode,
        "thresholds": {
            "max_vinf_mismatch_m_s": args.max_vinf_mismatch_m_s,
            "min_rp_margin_km": args.min_rp_margin_km,
        },
        "routes": ok_packets,
    }
    summary = {
        "schema_version": SCHEMA_VERSION,
        "input_jsonl": str(args.input_jsonl),
        "closures_selected": len(closures),
        "packets_written": len(packets),
        "packets_ok": len(ok_packets),
        "workers": workers,
        "top_packets": [{k: p.get(k) for k in ["packet_id", "route_id", "sequence", "pass_all_flybys", "total_departure_correction_m_s", "max_miss_after_km", "min_rp_margin_km", "max_vinf_mismatch_m_s"]} for p in packets[:20]],
    }
    write_json(args.output_json, summary)
    write_json(args.output_packet_json, packet_payload)

    print("=" * 80)
    print("MGA B-PLANE PACKET BUILDER V0.1")
    print("=" * 80)
    print(f"Closure records selected: {len(closures)}")
    print(f"Packets built:             {len(packets)}")
    print(f"Packets OK:                {len(ok_packets)}")
    print(f"Workers:                   {workers}")
    print("\nTop packets:")
    for i, p in enumerate(packets[:10], start=1):
        print(
            f" {i}. {p.get('sequence')} | pass={p.get('pass_all_flybys')} | "
            f"corr={finite(p.get('total_departure_correction_m_s')):.3f} m/s | "
            f"rp_margin={finite(p.get('min_rp_margin_km')):.3f} km | "
            f"vinf_mis={finite(p.get('max_vinf_mismatch_m_s')):.3f} m/s | "
            f"miss={finite(p.get('max_miss_after_km')):.3g} km"
        )
    print("=" * 80)
    print(f"[OK] wrote CSV:    {args.output_csv}")
    print(f"[OK] wrote JSONL:  {args.output_jsonl}")
    print(f"[OK] wrote JSON:   {args.output_json}")
    print(f"[OK] wrote packet: {args.output_packet_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
