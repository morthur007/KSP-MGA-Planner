#!/usr/bin/env python3
"""
mga_stitched_patch_corrector_v0_1.py

Patch-point corrector for stitched global/local MGA packets.

Input:
  - JSON/JSONL from mga_stitch_global_local_packet_v0_1.py
  - JSON/JSONL packet from mga_make_bplane_packet_v0_1.py

It solves two small impulsive corrections in the patched heliocentric model:

  1) optional pre-flyby correction at original departure epoch:
       origin center -> Duna SOI-in patch position

  2) post-flyby correction at Duna SOI-out patch epoch:
       Duna SOI-out patch state -> final target center at arrival epoch

This is not final multiple-shooting. It is a pragmatic stitched-packet corrector
that measures whether the stitched route can be made consistent with small TCMs.

Units: km, km/s, ET seconds. CLI correction bounds are in m/s per component.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

SCHEMA_VERSION = "mga_stitched_patch_corrector.v0.1"
SECONDS_PER_DAY = 86400.0
Vec3 = Tuple[float, float, float]
_WORKER_CFG: Dict[str, Any] = {}
_WORKER_SPICE = None
_WORKER_SOLVE_IVP = None
_WORKER_LSQ = None


def finite(x: Any, default: float = math.nan) -> float:
    try:
        y = float(x)
    except (TypeError, ValueError):
        return default
    return y if math.isfinite(y) else default


def opt_float(x: Any) -> Optional[float]:
    y = finite(x)
    return y if math.isfinite(y) else None


def vec3(x: Any) -> Optional[Vec3]:
    if not isinstance(x, Sequence) or isinstance(x, (str, bytes)) or len(x) < 3:
        return None
    vals: List[float] = []
    for i in range(3):
        y = finite(x[i])
        if not math.isfinite(y):
            return None
        vals.append(y)
    return (vals[0], vals[1], vals[2])


def vadd(a: Sequence[float], b: Sequence[float]) -> Vec3:
    return (float(a[0]) + float(b[0]), float(a[1]) + float(b[1]), float(a[2]) + float(b[2]))


def vsub(a: Sequence[float], b: Sequence[float]) -> Vec3:
    return (float(a[0]) - float(b[0]), float(a[1]) - float(b[1]), float(a[2]) - float(b[2]))


def vscale(a: Sequence[float], s: float) -> Vec3:
    return (float(a[0])*s, float(a[1])*s, float(a[2])*s)


def vdot(a: Sequence[float], b: Sequence[float]) -> float:
    return float(a[0])*float(b[0]) + float(a[1])*float(b[1]) + float(a[2])*float(b[2])


def vnorm(a: Sequence[float]) -> float:
    return math.sqrt(max(0.0, vdot(a, a)))


def stable_id(prefix: str, payload: Mapping[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=True, separators=(",", ":"), default=str)
    return f"{prefix}_{hashlib.sha1(raw.encode('utf-8')).hexdigest()[:16]}"


def sanitize(x: Any) -> Any:
    if isinstance(x, Mapping):
        return {str(k): sanitize(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [sanitize(v) for v in x]
    if isinstance(x, float):
        return x if math.isfinite(x) else None
    if isinstance(x, Path):
        return str(x)
    return x


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return data


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if not isinstance(obj, Mapping):
                raise ValueError(f"Expected JSON object at {path}:{line_no}")
            rows.append(dict(obj))
    return rows


def load_stitched(path: Path) -> List[Dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        return read_jsonl(path)
    data = load_json(path)
    if "patch_states" in data and "target" in data:
        return [data]
    for key in ("packets", "routes", "records", "results", "top_packets"):
        rows = data.get(key)
        if isinstance(rows, list):
            return [dict(r) for r in rows if isinstance(r, Mapping)]
    raise ValueError(f"Could not find stitched packets in {path}")


def load_bplane_packets(path: Path) -> List[Dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        return read_jsonl(path)
    data = load_json(path)
    if "legs" in data and "flybys" in data:
        return [data]
    rows = data.get("routes") or data.get("packets") or data.get("records")
    if isinstance(rows, list):
        return [dict(r) for r in rows if isinstance(r, Mapping)]
    raise ValueError(f"Could not find B-plane packets in {path}")


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(sanitize(payload), f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write("\n")


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(sanitize(r), ensure_ascii=False, separators=(",", ":"), allow_nan=False))
            f.write("\n")


def body_lookup(catalog: Mapping[str, Any], name: str) -> Dict[str, Any]:
    bodies = catalog.get("bodies") or {}
    if not isinstance(bodies, Mapping):
        return {}
    ent = bodies.get(name)
    if ent is None:
        for k, v in bodies.items():
            if str(k).lower() == name.lower():
                ent = v
                break
    return dict(ent) if isinstance(ent, Mapping) else {}


def index_bplane_packets(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Mapping[str, Any]]:
    idx: Dict[str, Mapping[str, Any]] = {}
    for p in rows:
        for key in (p.get("packet_id"), p.get("route_id"), p.get("closure_id"), p.get("correction_id")):
            if key is not None:
                idx[str(key)] = p
    return idx


def find_bplane_for_stitched(stitched: Mapping[str, Any], idx: Mapping[str, Mapping[str, Any]]) -> Optional[Mapping[str, Any]]:
    target = stitched.get("target") if isinstance(stitched.get("target"), Mapping) else {}
    for key in (target.get("packet_id"), target.get("route_id"), stitched.get("packet_id"), stitched.get("route_id")):
        if key is not None and str(key) in idx:
            return idx[str(key)]
    # Fallback: match by sequence and approximate periapsis epoch if ids were lost.
    seq = target.get("sequence")
    pe = finite((stitched.get("patch_epochs") or {}).get("periapsis_et")) if isinstance(stitched.get("patch_epochs"), Mapping) else math.nan
    for p in idx.values():
        if seq is not None and p.get("sequence") == seq:
            return p
        flys = p.get("flybys") if isinstance(p.get("flybys"), list) else []
        for fb in flys:
            if isinstance(fb, Mapping) and math.isfinite(pe):
                fpe = finite(fb.get("encounter_et") or fb.get("periapsis_et"))
                if math.isfinite(fpe) and abs(fpe - pe) < 10.0:
                    return p
    return None


def state_at(packet_state: Mapping[str, Any]) -> Tuple[Vec3, Vec3]:
    sc = packet_state.get("spacecraft_state_central") if isinstance(packet_state.get("spacecraft_state_central"), Mapping) else {}
    r = vec3(sc.get("r_km"))
    v = vec3(sc.get("v_km_s"))
    if r is None or v is None:
        raise ValueError("missing spacecraft_state_central r/v")
    return r, v


def _sequence_list(x: Any) -> List[str]:
    if isinstance(x, str):
        return [p.strip() for p in x.replace("->", ",").split(",") if p.strip()]
    if isinstance(x, Sequence) and not isinstance(x, (str, bytes)):
        return [str(p) for p in x]
    return []


def _init_worker(cfg: Mapping[str, Any]) -> None:
    global _WORKER_CFG, _WORKER_SPICE, _WORKER_SOLVE_IVP, _WORKER_LSQ
    _WORKER_CFG = dict(cfg)
    import spiceypy as spice  # type: ignore
    from scipy.integrate import solve_ivp  # type: ignore
    from scipy.optimize import least_squares  # type: ignore
    _WORKER_SPICE = spice
    _WORKER_SOLVE_IVP = solve_ivp
    _WORKER_LSQ = least_squares
    spice.kclear()
    for kernel in cfg.get("kernels", []):
        spice.furnsh(str(kernel))


def _spice() -> Any:
    global _WORKER_SPICE
    if _WORKER_SPICE is None:
        _init_worker(_WORKER_CFG)
    return _WORKER_SPICE


def _solve_ivp() -> Any:
    global _WORKER_SOLVE_IVP
    if _WORKER_SOLVE_IVP is None:
        _init_worker(_WORKER_CFG)
    return _WORKER_SOLVE_IVP


def _least_squares() -> Any:
    global _WORKER_LSQ
    if _WORKER_LSQ is None:
        _init_worker(_WORKER_CFG)
    return _WORKER_LSQ


def spice_state(body: str, et: float, central: str, frame: str) -> Tuple[Vec3, Vec3]:
    st, _lt = _spice().spkezr(str(body), float(et), str(frame), "NONE", str(central))
    return (float(st[0]), float(st[1]), float(st[2])), (float(st[3]), float(st[4]), float(st[5]))


def accel(t: float, y: Sequence[float], central: str, frame: str, mu_c: float, perturbers: Sequence[Tuple[str, float]]) -> List[float]:
    r = (float(y[0]), float(y[1]), float(y[2]))
    v = (float(y[3]), float(y[4]), float(y[5]))
    rn = max(vnorm(r), 1e-30)
    a = [-mu_c * r[0]/(rn**3), -mu_c * r[1]/(rn**3), -mu_c * r[2]/(rn**3)]
    for name, mu in perturbers:
        rb, _vb = spice_state(name, t, central, frame)
        ds = (rb[0] - r[0], rb[1] - r[1], rb[2] - r[2])
        dsn = max(vnorm(ds), 1e-30)
        rbn = max(vnorm(rb), 1e-30)
        a[0] += mu * (ds[0]/(dsn**3) - rb[0]/(rbn**3))
        a[1] += mu * (ds[1]/(dsn**3) - rb[1]/(rbn**3))
        a[2] += mu * (ds[2]/(dsn**3) - rb[2]/(rbn**3))
    return [v[0], v[1], v[2], a[0], a[1], a[2]]


def select_perturbers(catalog: Mapping[str, Any], all_names: Sequence[str], exclude: Sequence[str]) -> List[Tuple[str, float]]:
    ex = {str(x).lower() for x in exclude}
    out: List[Tuple[str, float]] = []
    for name in all_names:
        if str(name).lower() in ex:
            continue
        info = body_lookup(catalog, str(name))
        mu = finite(info.get("mu_km3_s2"))
        if mu > 0:
            out.append((str(name), mu))
    return out


def propagate(r0: Vec3, v0: Vec3, t0: float, t1: float, cfg: Mapping[str, Any], perturbers: Sequence[Tuple[str, float]]) -> Dict[str, Any]:
    if not math.isfinite(t0) or not math.isfinite(t1) or t1 <= t0:
        return {"ok": False, "message": "invalid_time_span"}
    y0 = [r0[0], r0[1], r0[2], v0[0], v0[1], v0[2]]
    atol = [float(cfg["atol_position_km"])]*3 + [float(cfg["atol_velocity_km_s"])]*3
    try:
        sol = _solve_ivp()(
            lambda t, y: accel(t, y, str(cfg["central_body"]), str(cfg["frame"]), float(cfg["mu_central_km3_s2"]), perturbers),
            (float(t0), float(t1)),
            y0,
            method=str(cfg["integrator"]),
            rtol=float(cfg["rtol"]),
            atol=atol,
            max_step=float(cfg["max_step_days"]) * SECONDS_PER_DAY,
            dense_output=False,
        )
    except Exception as exc:
        return {"ok": False, "message": repr(exc)}
    yf = [float(sol.y[i, -1]) for i in range(6)]
    return {
        "ok": bool(sol.success),
        "message": str(sol.message),
        "r_km": (yf[0], yf[1], yf[2]),
        "v_km_s": (yf[3], yf[4], yf[5]),
        "nfev": int(getattr(sol, "nfev", -1)),
        "status": int(getattr(sol, "status", 0)),
    }


def solve_velocity_correction(
    *,
    r0: Vec3,
    v0: Vec3,
    t0: float,
    t1: float,
    target_r: Vec3,
    cfg: Mapping[str, Any],
    perturbers: Sequence[Tuple[str, float]],
    max_component_m_s: float,
    position_scale_km: float,
    max_nfev: int,
) -> Dict[str, Any]:
    bound_km_s = abs(max_component_m_s) / 1000.0
    scale = max(float(position_scale_km), 1e-9)
    cache: Dict[Tuple[float, float, float], Dict[str, Any]] = {}

    def eval_dv(x: Sequence[float]) -> Dict[str, Any]:
        key = (float(x[0]), float(x[1]), float(x[2]))
        if key not in cache:
            vv = vadd(v0, key)
            cache[key] = propagate(r0, vv, t0, t1, cfg, perturbers)
        return cache[key]

    def residual(x: Sequence[float]) -> List[float]:
        res = eval_dv(x)
        if not res.get("ok"):
            return [1e9, 1e9, 1e9]
        rr = vec3(res.get("r_km"))
        if rr is None:
            return [1e9, 1e9, 1e9]
        d = vsub(rr, target_r)
        return [d[0]/scale, d[1]/scale, d[2]/scale]

    x0 = [0.0, 0.0, 0.0]
    try:
        opt = _least_squares()(residual, x0, bounds=([-bound_km_s]*3, [bound_km_s]*3), max_nfev=int(max_nfev), xtol=1e-12, ftol=1e-12, gtol=1e-12)
        x = [float(v) for v in opt.x]
        final = eval_dv(x)
        rr = vec3(final.get("r_km")); vv = vec3(final.get("v_km_s"))
        if rr is None or vv is None:
            raise RuntimeError("missing corrected propagation state")
        miss = vnorm(vsub(rr, target_r))
        dv_m_s = vnorm(x) * 1000.0
        hit_bound = any(abs(xi) >= 0.999*bound_km_s for xi in x) if bound_km_s > 0 else False
        return {
            "ok": bool(final.get("ok")),
            "success": bool(getattr(opt, "success", False)) and bool(final.get("ok")),
            "message": str(getattr(opt, "message", "")),
            "dv_km_s": tuple(x),
            "dv_m_s": dv_m_s,
            "hit_component_bound": bool(hit_bound),
            "position_miss_km": miss,
            "final_r_km": rr,
            "final_v_km_s": vv,
            "nfev_opt": int(getattr(opt, "nfev", -1)),
            "nfev_prop": final.get("nfev"),
            "cost": float(getattr(opt, "cost", math.nan)),
        }
    except Exception as exc:
        base = propagate(r0, v0, t0, t1, cfg, perturbers)
        rr = vec3(base.get("r_km"))
        miss = vnorm(vsub(rr, target_r)) if rr is not None else math.inf
        return {"ok": False, "success": False, "message": repr(exc), "dv_km_s": (0.0, 0.0, 0.0), "dv_m_s": 0.0, "position_miss_km": miss}


def correct_one(payload: Tuple[Mapping[str, Any], Optional[Mapping[str, Any]], Mapping[str, Any]]) -> Dict[str, Any]:
    stitched, bplane, catalog = payload
    cfg = _WORKER_CFG
    try:
        if not stitched.get("ok", False):
            raise ValueError("stitched packet not ok")
        if bplane is None:
            raise ValueError("no matching b-plane packet")
        target = stitched.get("target") if isinstance(stitched.get("target"), Mapping) else {}
        seq = _sequence_list(target.get("sequence") or bplane.get("sequence"))
        flyby_body = str(target.get("body") or (seq[1] if len(seq) >= 3 else ""))
        legs = bplane.get("legs") or []
        if not isinstance(legs, list) or len(legs) < 2:
            raise ValueError("matching b-plane packet has fewer than 2 legs")
        leg0, leg1 = legs[0], legs[1]
        if not isinstance(leg0, Mapping) or not isinstance(leg1, Mapping):
            raise ValueError("invalid leg records")
        patch_epochs = stitched.get("patch_epochs") if isinstance(stitched.get("patch_epochs"), Mapping) else {}
        patch_states = stitched.get("patch_states") if isinstance(stitched.get("patch_states"), Mapping) else {}
        entry_state = patch_states.get("entry_soi") if isinstance(patch_states.get("entry_soi"), Mapping) else {}
        exit_state = patch_states.get("exit_soi") if isinstance(patch_states.get("exit_soi"), Mapping) else {}
        entry_et = finite(patch_epochs.get("entry_et"))
        exit_et = finite(patch_epochs.get("exit_et"))
        pe_et = finite(patch_epochs.get("periapsis_et"))
        entry_r, entry_v = state_at(entry_state)
        exit_r, exit_v = state_at(exit_state)
        central = str(cfg["central_body"])
        frame = str(cfg["frame"])
        mode = str(cfg.get("dynamics_mode", "patched_heliocentric"))
        all_g = [str(x) for x in cfg.get("gravitating_bodies", [])]

        # Pre arc correction: original origin center -> SOI-in patch position.
        origin0 = str(leg0.get("origin"))
        target0 = str(leg0.get("target"))
        depart0 = finite(leg0.get("depart_et"))
        r0, _vb0 = spice_state(origin0, depart0, central, frame)
        v0 = vec3(leg0.get("sc_v_depart_corrected_km_s")) or vec3(leg0.get("sc_v_depart_km_s"))
        if v0 is None:
            raise ValueError("leg0 missing departure velocity")
        if mode == "patched_heliocentric":
            pert_pre = select_perturbers(catalog, all_g, exclude=[origin0, target0])
        elif mode == "full_nbody":
            pert_pre = select_perturbers(catalog, all_g, exclude=[])
        else:
            pert_pre = []
        pre_base = propagate(r0, v0, depart0, entry_et, cfg, pert_pre)
        base_pre_r = vec3(pre_base.get("r_km")); base_pre_v = vec3(pre_base.get("v_km_s"))
        base_entry_pos_miss = vnorm(vsub(base_pre_r, entry_r)) if base_pre_r else math.inf
        base_entry_vel_miss = vnorm(vsub(base_pre_v, entry_v))*1000.0 if base_pre_v else math.inf
        if bool(cfg.get("disable_pre_correction")):
            pre_corr = {
                "ok": bool(pre_base.get("ok")), "success": bool(pre_base.get("ok")),
                "dv_km_s": (0.0, 0.0, 0.0), "dv_m_s": 0.0,
                "position_miss_km": base_entry_pos_miss,
                "final_r_km": base_pre_r, "final_v_km_s": base_pre_v,
                "hit_component_bound": False, "nfev_prop": pre_base.get("nfev"),
            }
        else:
            pre_corr = solve_velocity_correction(
                r0=r0, v0=v0, t0=depart0, t1=entry_et, target_r=entry_r, cfg=cfg,
                perturbers=pert_pre,
                max_component_m_s=float(cfg["max_pre_correction_m_s"]),
                position_scale_km=float(cfg["pre_position_scale_km"]),
                max_nfev=int(cfg["max_nfev"]),
            )
        pre_vf = vec3(pre_corr.get("final_v_km_s"))
        entry_vel_after_m_s = vnorm(vsub(pre_vf, entry_v))*1000.0 if pre_vf else math.inf

        # Post arc correction: SOI-out patch state -> final target center.
        origin1 = str(leg1.get("origin"))
        target1 = str(leg1.get("target"))
        arrive1 = finite(leg1.get("arrive_et"))
        target_r, target_v = spice_state(target1, arrive1, central, frame)
        if mode == "patched_heliocentric":
            pert_post = select_perturbers(catalog, all_g, exclude=[origin1, target1])
        elif mode == "full_nbody":
            pert_post = select_perturbers(catalog, all_g, exclude=[])
        else:
            pert_post = []
        post_base = propagate(exit_r, exit_v, exit_et, arrive1, cfg, pert_post)
        base_post_r = vec3(post_base.get("r_km")); base_post_v = vec3(post_base.get("v_km_s"))
        base_arrival_miss = vnorm(vsub(base_post_r, target_r)) if base_post_r else math.inf
        base_arrival_vinf = vnorm(vsub(base_post_v, target_v)) if base_post_v else math.inf
        post_corr = solve_velocity_correction(
            r0=exit_r, v0=exit_v, t0=exit_et, t1=arrive1, target_r=target_r, cfg=cfg,
            perturbers=pert_post,
            max_component_m_s=float(cfg["max_post_correction_m_s"]),
            position_scale_km=float(cfg["post_position_scale_km"]),
            max_nfev=int(cfg["max_nfev"]),
        )
        post_vf = vec3(post_corr.get("final_v_km_s"))
        arrival_vinf_after = vnorm(vsub(post_vf, target_v)) if post_vf else math.inf

        pre_pass = (pre_corr.get("position_miss_km", math.inf) <= float(cfg["target_entry_miss_km"]) and
                    entry_vel_after_m_s <= float(cfg["max_entry_velocity_miss_m_s"]))
        post_pass = post_corr.get("position_miss_km", math.inf) <= float(cfg["target_arrival_miss_km"])
        total_dv = finite(pre_corr.get("dv_m_s"), 0.0) + finite(post_corr.get("dv_m_s"), 0.0)
        pass_all = bool(pre_pass and post_pass and pre_corr.get("ok") and post_corr.get("ok"))
        out = {
            "schema_version": SCHEMA_VERSION,
            "correction_id": stable_id("stcorr", {"stitched": stitched.get("stitched_packet_id"), "pre": round(finite(pre_corr.get("dv_m_s"),0), 6), "post": round(finite(post_corr.get("dv_m_s"),0), 6)}),
            "ok": True,
            "pass_correction": pass_all,
            "sequence": " -> ".join(seq) if seq else target.get("sequence"),
            "flyby_body": flyby_body,
            "stitched_packet_id": stitched.get("stitched_packet_id"),
            "packet_id": target.get("packet_id") or bplane.get("packet_id"),
            "route_id": target.get("route_id") or bplane.get("route_id"),
            "epochs": {"depart_et": depart0, "entry_et": entry_et, "periapsis_et": pe_et, "exit_et": exit_et, "arrival_et": arrive1},
            "pre_arc": {
                "origin": origin0, "target_patch": f"{flyby_body}_SOI_in", "tof_days": (entry_et - depart0)/SECONDS_PER_DAY,
                "perturbers": [x[0] for x in pert_pre],
                "miss_before_km": base_entry_pos_miss,
                "velocity_miss_before_m_s": base_entry_vel_miss,
                "dv_correction_km_s": pre_corr.get("dv_km_s"),
                "dv_correction_m_s": pre_corr.get("dv_m_s"),
                "miss_after_km": pre_corr.get("position_miss_km"),
                "velocity_miss_after_m_s": entry_vel_after_m_s,
                "hit_component_bound": pre_corr.get("hit_component_bound"),
                "pass_patch": pre_pass,
                "nfev_opt": pre_corr.get("nfev_opt"),
                "nfev_prop": pre_corr.get("nfev_prop"),
            },
            "local_flyby": stitched.get("local_validation") if isinstance(stitched.get("local_validation"), Mapping) else {},
            "post_arc": {
                "origin_patch": f"{flyby_body}_SOI_out", "target": target1, "tof_days": (arrive1 - exit_et)/SECONDS_PER_DAY,
                "perturbers": [x[0] for x in pert_post],
                "miss_before_km": base_arrival_miss,
                "arrival_vinf_before_km_s": base_arrival_vinf,
                "dv_correction_km_s": post_corr.get("dv_km_s"),
                "dv_correction_m_s": post_corr.get("dv_m_s"),
                "miss_after_km": post_corr.get("position_miss_km"),
                "arrival_vinf_after_km_s": arrival_vinf_after,
                "hit_component_bound": post_corr.get("hit_component_bound"),
                "pass_patch": post_pass,
                "nfev_opt": post_corr.get("nfev_opt"),
                "nfev_prop": post_corr.get("nfev_prop"),
            },
            "quality": {
                "total_patch_correction_m_s": total_dv,
                "pre_correction_m_s": pre_corr.get("dv_m_s"),
                "post_correction_m_s": post_corr.get("dv_m_s"),
                "rp_margin_km": (((target.get("hyperbola") or {}) if isinstance(target.get("hyperbola"), Mapping) else {}).get("rp_margin_km")),
                "periapsis_altitude_km": (((target.get("hyperbola") or {}) if isinstance(target.get("hyperbola"), Mapping) else {}).get("periapsis_altitude_km")),
                "source_total_departure_correction_m_s": (((target.get("quality") or {}) if isinstance(target.get("quality"), Mapping) else {}).get("total_departure_correction_m_s")),
            },
            "source_stitched_packet": stitched,
        }
        return out
    except Exception as exc:
        return {"schema_version": SCHEMA_VERSION, "ok": False, "pass_correction": False, "message": repr(exc), "source_stitched_packet": stitched}


def _worker(payload: Tuple[Mapping[str, Any], Optional[Mapping[str, Any]], Mapping[str, Any]]) -> Dict[str, Any]:
    return correct_one(payload)


def flat_row(r: Mapping[str, Any]) -> Dict[str, Any]:
    pre = r.get("pre_arc") if isinstance(r.get("pre_arc"), Mapping) else {}
    post = r.get("post_arc") if isinstance(r.get("post_arc"), Mapping) else {}
    q = r.get("quality") if isinstance(r.get("quality"), Mapping) else {}
    return {
        "correction_id": r.get("correction_id"),
        "stitched_packet_id": r.get("stitched_packet_id"),
        "packet_id": r.get("packet_id"),
        "sequence": r.get("sequence"),
        "flyby_body": r.get("flyby_body"),
        "ok": int(bool(r.get("ok"))),
        "pass_correction": int(bool(r.get("pass_correction"))),
        "entry_miss_before_km": pre.get("miss_before_km"),
        "entry_miss_after_km": pre.get("miss_after_km"),
        "entry_vel_miss_after_m_s": pre.get("velocity_miss_after_m_s"),
        "pre_dv_m_s": pre.get("dv_correction_m_s"),
        "arrival_miss_before_km": post.get("miss_before_km"),
        "arrival_miss_after_km": post.get("miss_after_km"),
        "post_dv_m_s": post.get("dv_correction_m_s"),
        "total_patch_correction_m_s": q.get("total_patch_correction_m_s"),
        "rp_margin_km": q.get("rp_margin_km"),
        "periapsis_altitude_km": q.get("periapsis_altitude_km"),
        "message": r.get("message"),
    }


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = [
        "correction_id", "stitched_packet_id", "packet_id", "sequence", "flyby_body", "ok", "pass_correction",
        "entry_miss_before_km", "entry_miss_after_km", "entry_vel_miss_after_m_s", "pre_dv_m_s",
        "arrival_miss_before_km", "arrival_miss_after_km", "post_dv_m_s", "total_patch_correction_m_s",
        "rp_margin_km", "periapsis_altitude_km", "message",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k) for k in fields})


def stats(vals: Sequence[float]) -> Dict[str, Optional[float]]:
    xs = sorted([x for x in vals if math.isfinite(x)])
    if not xs:
        return {"min": None, "median": None, "max": None}
    return {"min": xs[0], "median": xs[len(xs)//2], "max": xs[-1]}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Correct stitched MGA route patch points with small pre/post flyby TCMs.")
    p.add_argument("--bsp", required=True, type=Path)
    p.add_argument("--tpc", type=Path, default=None)
    p.add_argument("--body-catalog", required=True, type=Path)
    p.add_argument("--stitched-jsonl", required=True, type=Path)
    p.add_argument("--bplane-packet", required=True, type=Path)
    p.add_argument("--central-body", default="Sun")
    p.add_argument("--frame", default="J2000")
    p.add_argument("--mu-central-km3-s2", required=True, type=float)
    p.add_argument("--gravitating-bodies", nargs="+", default=[])
    p.add_argument("--dynamics-mode", default="patched_heliocentric", choices=["two_body", "patched_heliocentric", "full_nbody"])
    p.add_argument("--workers", type=int, default=1, help="0=os.cpu_count(); 1=serial")
    p.add_argument("--multiprocessing-start-method", default="spawn", choices=["spawn", "fork", "forkserver"])
    p.add_argument("--integrator", default="DOP853", choices=["DOP853", "RK45", "LSODA", "Radau", "BDF"])
    p.add_argument("--rtol", type=float, default=1e-10)
    p.add_argument("--atol-position-km", type=float, default=1e-6)
    p.add_argument("--atol-velocity-km-s", type=float, default=1e-12)
    p.add_argument("--max-step-days", type=float, default=1.0)
    p.add_argument("--max-nfev", type=int, default=30)
    p.add_argument("--disable-pre-correction", action="store_true")
    p.add_argument("--max-pre-correction-m-s", type=float, default=20.0, help="Per-component bound")
    p.add_argument("--max-post-correction-m-s", type=float, default=20.0, help="Per-component bound")
    p.add_argument("--pre-position-scale-km", type=float, default=1000.0)
    p.add_argument("--post-position-scale-km", type=float, default=100000.0)
    p.add_argument("--target-entry-miss-km", type=float, default=10.0)
    p.add_argument("--target-arrival-miss-km", type=float, default=10.0)
    p.add_argument("--max-entry-velocity-miss-m-s", type=float, default=100.0)
    p.add_argument("--top-n", type=int, default=0)
    p.add_argument("--output-csv", required=True, type=Path)
    p.add_argument("--output-jsonl", required=True, type=Path)
    p.add_argument("--output-json", required=True, type=Path)
    p.add_argument("--output-best-json", required=True, type=Path)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    catalog = load_json(args.body_catalog)
    stitched = load_stitched(args.stitched_jsonl)
    bplanes = load_bplane_packets(args.bplane_packet)
    idx = index_bplane_packets(bplanes)
    stitched = [s for s in stitched if isinstance(s, Mapping)]
    if args.top_n > 0:
        stitched = stitched[:args.top_n]
    kernels = [str(args.bsp)]
    if args.tpc:
        kernels.insert(0, str(args.tpc))
    cfg = {
        "kernels": kernels,
        "central_body": args.central_body,
        "frame": args.frame,
        "mu_central_km3_s2": args.mu_central_km3_s2,
        "gravitating_bodies": args.gravitating_bodies,
        "dynamics_mode": args.dynamics_mode,
        "integrator": args.integrator,
        "rtol": args.rtol,
        "atol_position_km": args.atol_position_km,
        "atol_velocity_km_s": args.atol_velocity_km_s,
        "max_step_days": args.max_step_days,
        "max_nfev": args.max_nfev,
        "disable_pre_correction": args.disable_pre_correction,
        "max_pre_correction_m_s": args.max_pre_correction_m_s,
        "max_post_correction_m_s": args.max_post_correction_m_s,
        "pre_position_scale_km": args.pre_position_scale_km,
        "post_position_scale_km": args.post_position_scale_km,
        "target_entry_miss_km": args.target_entry_miss_km,
        "target_arrival_miss_km": args.target_arrival_miss_km,
        "max_entry_velocity_miss_m_s": args.max_entry_velocity_miss_m_s,
    }
    payloads: List[Tuple[Mapping[str, Any], Optional[Mapping[str, Any]], Mapping[str, Any]]] = []
    for s in stitched:
        payloads.append((s, find_bplane_for_stitched(s, idx), catalog))
    workers = (os.cpu_count() or 1) if args.workers == 0 else max(1, args.workers)
    results: List[Dict[str, Any]] = []
    if workers == 1 or len(payloads) <= 1:
        _init_worker(cfg)
        for pld in payloads:
            results.append(correct_one(pld))
    else:
        ctx = mp.get_context(args.multiprocessing_start_method)
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx, initializer=_init_worker, initargs=(cfg,)) as ex:
            futs = [ex.submit(_worker, pld) for pld in payloads]
            for fut in as_completed(futs):
                results.append(fut.result())
    results.sort(key=lambda r: (
        not bool(r.get("ok")),
        not bool(r.get("pass_correction")),
        finite(((r.get("quality") or {}).get("total_patch_correction_m_s") if isinstance(r.get("quality"), Mapping) else None), 1e99),
        finite(((r.get("post_arc") or {}).get("miss_after_km") if isinstance(r.get("post_arc"), Mapping) else None), 1e99),
    ))
    flat = [flat_row(r) for r in results]
    write_csv(args.output_csv, flat)
    write_jsonl(args.output_jsonl, results)
    entry_before = [finite(((r.get("pre_arc") or {}).get("miss_before_km") if isinstance(r.get("pre_arc"), Mapping) else None)) for r in results]
    entry_after = [finite(((r.get("pre_arc") or {}).get("miss_after_km") if isinstance(r.get("pre_arc"), Mapping) else None)) for r in results]
    arrival_before = [finite(((r.get("post_arc") or {}).get("miss_before_km") if isinstance(r.get("post_arc"), Mapping) else None)) for r in results]
    arrival_after = [finite(((r.get("post_arc") or {}).get("miss_after_km") if isinstance(r.get("post_arc"), Mapping) else None)) for r in results]
    total_dv = [finite(((r.get("quality") or {}).get("total_patch_correction_m_s") if isinstance(r.get("quality"), Mapping) else None)) for r in results]
    summary = {
        "schema_version": SCHEMA_VERSION,
        "stitched_input": str(args.stitched_jsonl),
        "bplane_packet": str(args.bplane_packet),
        "routes_processed": len(results),
        "pass_correction": sum(1 for r in results if r.get("pass_correction")),
        "workers": workers,
        "dynamics_mode": args.dynamics_mode,
        "target_entry_miss_km": args.target_entry_miss_km,
        "target_arrival_miss_km": args.target_arrival_miss_km,
        "max_pre_correction_m_s": args.max_pre_correction_m_s,
        "max_post_correction_m_s": args.max_post_correction_m_s,
        "entry_miss_before_km": stats(entry_before),
        "entry_miss_after_km": stats(entry_after),
        "arrival_miss_before_km": stats(arrival_before),
        "arrival_miss_after_km": stats(arrival_after),
        "total_patch_correction_m_s": stats(total_dv),
        "top_results": flat[:20],
    }
    write_json(args.output_json, summary)
    write_json(args.output_best_json, results[0] if results else {})

    print("="*80)
    print("MGA STITCHED PATCH CORRECTOR V0.1")
    print("="*80)
    print(f"Routes processed: {len(results)}")
    print(f"Pass correction:  {summary['pass_correction']}")
    print(f"Workers:          {workers}")
    print(f"Dynamics mode:    {args.dynamics_mode}")
    print(f"Entry miss before: min={summary['entry_miss_before_km']['min']} median={summary['entry_miss_before_km']['median']} max={summary['entry_miss_before_km']['max']} km")
    print(f"Entry miss after:  min={summary['entry_miss_after_km']['min']} median={summary['entry_miss_after_km']['median']} max={summary['entry_miss_after_km']['max']} km")
    print(f"Arrival before:    min={summary['arrival_miss_before_km']['min']} median={summary['arrival_miss_before_km']['median']} max={summary['arrival_miss_before_km']['max']} km")
    print(f"Arrival after:     min={summary['arrival_miss_after_km']['min']} median={summary['arrival_miss_after_km']['median']} max={summary['arrival_miss_after_km']['max']} km")
    print(f"Total correction:  min={summary['total_patch_correction_m_s']['min']} median={summary['total_patch_correction_m_s']['median']} max={summary['total_patch_correction_m_s']['max']} m/s")
    print("\nTop corrected stitched routes:")
    for i, row in enumerate(flat[:10], start=1):
        print(f" {i}. {row.get('sequence')} | pass={bool(row.get('pass_correction'))} | entry {finite(row.get('entry_miss_before_km')):.3g}->{finite(row.get('entry_miss_after_km')):.3g} km | arrival {finite(row.get('arrival_miss_before_km')):.3g}->{finite(row.get('arrival_miss_after_km')):.3g} km | dv={finite(row.get('total_patch_correction_m_s')):.3f} m/s")
    print("="*80)
    print(f"[OK] wrote CSV:       {args.output_csv}")
    print(f"[OK] wrote JSONL:     {args.output_jsonl}")
    print(f"[OK] wrote JSON:      {args.output_json}")
    print(f"[OK] wrote best JSON: {args.output_best_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
