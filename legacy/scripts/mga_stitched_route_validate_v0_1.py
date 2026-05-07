#!/usr/bin/env python3
"""
mga_stitched_route_validate_v0_1.py

Validate a stitched global/local MGA packet.

This stage consumes:
  - JSON/JSONL from mga_stitch_global_local_packet_v0_1.py
  - JSON/JSONL packet from mga_make_bplane_packet_v0_1.py, to recover
    corrected heliocentric leg departure velocities.

It checks the first stitched local flyby by validating two patched arcs:

  1. pre-flyby arc:
       origin body center at leg0.depart_et + corrected Lambert velocity
       -> stitched SOI-entry state at entry_et

  2. post-flyby arc:
       stitched SOI-exit state at exit_et
       -> final target body center at leg1.arrive_et

The local SOI-in -> periapsis -> SOI-out segment is assumed to have already
passed mga_local_flyby_validate_v0_1.py and is reported, not re-solved here.

This is still a patched validation, not a full Principia continuous trajectory.
It answers: did the global legs and the local Duna flyby patch points actually
stitch together with the current corrections?

Units: km, km/s, ET seconds.
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

SCHEMA_VERSION = "mga_stitched_route_validation.v0.1"
SECONDS_PER_DAY = 86400.0
Vec3 = Tuple[float, float, float]
_WORKER_CFG: Dict[str, Any] = {}
_WORKER_SPICE = None
_WORKER_INTEGRATE = None


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
    out = []
    for i in range(3):
        y = finite(x[i])
        if not math.isfinite(y):
            return None
        out.append(y)
    return (out[0], out[1], out[2])


def vadd(a: Sequence[float], b: Sequence[float]) -> Vec3:
    return (float(a[0]) + float(b[0]), float(a[1]) + float(b[1]), float(a[2]) + float(b[2]))


def vsub(a: Sequence[float], b: Sequence[float]) -> Vec3:
    return (float(a[0]) - float(b[0]), float(a[1]) - float(b[1]), float(a[2]) - float(b[2]))


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
    return None


def state_at(packet_state: Mapping[str, Any]) -> Tuple[Vec3, Vec3]:
    sc = packet_state.get("spacecraft_state_central") if isinstance(packet_state.get("spacecraft_state_central"), Mapping) else {}
    r = vec3(sc.get("r_km"))
    v = vec3(sc.get("v_km_s"))
    if r is None or v is None:
        raise ValueError("missing spacecraft_state_central r/v")
    return r, v


def _init_worker(cfg: Mapping[str, Any]) -> None:
    global _WORKER_CFG, _WORKER_SPICE, _WORKER_INTEGRATE
    _WORKER_CFG = dict(cfg)
    import spiceypy as spice  # type: ignore
    from scipy.integrate import solve_ivp  # type: ignore
    _WORKER_SPICE = spice
    _WORKER_INTEGRATE = solve_ivp
    spice.kclear()
    for kernel in cfg.get("kernels", []):
        spice.furnsh(str(kernel))


def _spice() -> Any:
    global _WORKER_SPICE
    if _WORKER_SPICE is None:
        _init_worker(_WORKER_CFG)
    return _WORKER_SPICE


def _solve_ivp() -> Any:
    global _WORKER_INTEGRATE
    if _WORKER_INTEGRATE is None:
        _init_worker(_WORKER_CFG)
    return _WORKER_INTEGRATE


def spice_state(body: str, et: float, central: str, frame: str) -> Tuple[Vec3, Vec3]:
    st, _lt = _spice().spkezr(str(body), float(et), str(frame), "NONE", str(central))
    return (float(st[0]), float(st[1]), float(st[2])), (float(st[3]), float(st[4]), float(st[5]))


def accel(t: float, y: Sequence[float], central: str, frame: str, mu_c: float, perturbers: Sequence[Tuple[str, float]]) -> List[float]:
    r = (float(y[0]), float(y[1]), float(y[2]))
    v = (float(y[3]), float(y[4]), float(y[5]))
    rn = max(vnorm(r), 1e-30)
    a = [-mu_c * r[0]/(rn**3), -mu_c * r[1]/(rn**3), -mu_c * r[2]/(rn**3)]
    # Indirect third-body acceleration in central-body coordinates.
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
        sol = _solve_ivp()(lambda t, y: accel(t, y, str(cfg["central_body"]), str(cfg["frame"]), float(cfg["mu_central_km3_s2"]), perturbers),
                           (float(t0), float(t1)), y0,
                           method=str(cfg["integrator"]), rtol=float(cfg["rtol"]), atol=atol,
                           max_step=float(cfg["max_step_days"])*SECONDS_PER_DAY)
    except Exception as exc:
        return {"ok": False, "message": repr(exc)}
    if sol.y is None or len(sol.y) < 6:
        return {"ok": False, "message": "no_solution"}
    yf = [float(sol.y[i, -1]) for i in range(6)]
    return {"ok": bool(sol.success), "message": str(sol.message), "nfev": int(getattr(sol, "nfev", -1)),
            "r_km": (yf[0], yf[1], yf[2]), "v_km_s": (yf[3], yf[4], yf[5])}


def _sequence_list(seq: Any) -> List[str]:
    if isinstance(seq, str):
        if "->" in seq:
            return [x.strip() for x in seq.split("->") if x.strip()]
        if "," in seq:
            return [x.strip() for x in seq.split(",") if x.strip()]
        return [seq]
    if isinstance(seq, Sequence) and not isinstance(seq, (bytes, bytearray)):
        return [str(x) for x in seq]
    return []


def validate_one(payload: Tuple[Mapping[str, Any], Optional[Mapping[str, Any]], Mapping[str, Any]]) -> Dict[str, Any]:
    stitched, bplane, catalog = payload
    cfg = _WORKER_CFG
    try:
        if not stitched.get("ok", False):
            raise ValueError("stitched packet not ok")
        if bplane is None:
            raise ValueError("no matching b-plane packet; pass --bplane-packet from mga_make_bplane_packet_v0_1.py")
        target = stitched.get("target") if isinstance(stitched.get("target"), Mapping) else {}
        seq = _sequence_list(target.get("sequence") or bplane.get("sequence"))
        flyby_body = str(target.get("body") or (seq[1] if len(seq) >= 3 else ""))
        legs = bplane.get("legs") or []
        if not isinstance(legs, list) or len(legs) < 2:
            raise ValueError("matching b-plane packet has fewer than 2 legs")
        leg0 = legs[0]
        leg1 = legs[1]
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
        all_g = [str(x) for x in cfg.get("gravitating_bodies", [])]
        # Pre-flyby: from origin center to Duna SOI-in.
        origin0 = str(leg0.get("origin"))
        target0 = str(leg0.get("target"))
        depart0 = finite(leg0.get("depart_et"))
        r0, _vb0 = spice_state(origin0, depart0, central, frame)
        v0 = vec3(leg0.get("sc_v_depart_corrected_km_s")) or vec3(leg0.get("sc_v_depart_km_s"))
        if v0 is None:
            raise ValueError("leg0 missing corrected departure velocity")
        pert_pre: List[Tuple[str, float]] = []
        mode = str(cfg.get("dynamics_mode", "patched_heliocentric"))
        if mode == "patched_heliocentric":
            pert_pre = select_perturbers(catalog, all_g, exclude=[origin0, target0])
        elif mode == "full_nbody":
            pert_pre = select_perturbers(catalog, all_g, exclude=[])
        pre = propagate(r0, v0, depart0, entry_et, cfg, pert_pre)
        if not pre.get("ok"):
            raise ValueError(f"pre arc failed: {pre.get('message')}")
        pre_r = vec3(pre.get("r_km")); pre_v = vec3(pre.get("v_km_s"))
        if pre_r is None or pre_v is None:
            raise ValueError("pre arc missing final state")
        entry_pos_miss = vnorm(vsub(pre_r, entry_r))
        entry_vel_miss = vnorm(vsub(pre_v, entry_v))*1000.0
        # Post-flyby: from Duna SOI-out to final target center.
        origin1 = str(leg1.get("origin"))
        target1 = str(leg1.get("target"))
        arrive1 = finite(leg1.get("arrive_et"))
        pert_post: List[Tuple[str, float]] = []
        if mode == "patched_heliocentric":
            pert_post = select_perturbers(catalog, all_g, exclude=[origin1, target1])
        elif mode == "full_nbody":
            pert_post = select_perturbers(catalog, all_g, exclude=[])
        post = propagate(exit_r, exit_v, exit_et, arrive1, cfg, pert_post)
        if not post.get("ok"):
            raise ValueError(f"post arc failed: {post.get('message')}")
        post_r = vec3(post.get("r_km")); post_v = vec3(post.get("v_km_s"))
        if post_r is None or post_v is None:
            raise ValueError("post arc missing final state")
        target_r, target_v = spice_state(target1, arrive1, central, frame)
        arrival_pos_miss = vnorm(vsub(post_r, target_r))
        arrival_vinf = vnorm(vsub(post_v, target_v))
        # Local segment was already validated; pass through its metrics.
        local_val = stitched.get("local_validation") if isinstance(stitched.get("local_validation"), Mapping) else {}
        entry_pass = entry_pos_miss <= float(cfg["entry_position_threshold_km"]) and entry_vel_miss <= float(cfg["entry_velocity_threshold_m_s"])
        arrival_pass = arrival_pos_miss <= float(cfg["arrival_position_threshold_km"])
        pass_all = bool(entry_pass and arrival_pass and local_val.get("pass_validation", True))
        return {
            "schema_version": SCHEMA_VERSION,
            "validation_id": stable_id("stval", {"stitched": stitched.get("stitched_packet_id"), "entry": round(entry_pos_miss, 6), "arr": round(arrival_pos_miss, 6)}),
            "ok": True,
            "pass_validation": pass_all,
            "sequence": " -> ".join(seq) if seq else target.get("sequence"),
            "flyby_body": flyby_body,
            "stitched_packet_id": stitched.get("stitched_packet_id"),
            "packet_id": target.get("packet_id") or bplane.get("packet_id"),
            "route_id": target.get("route_id") or bplane.get("route_id"),
            "epochs": {"depart_et": depart0, "entry_et": entry_et, "periapsis_et": pe_et, "exit_et": exit_et, "arrival_et": arrive1},
            "pre_arc": {
                "origin": origin0, "target_patch": f"{flyby_body}_SOI_in",
                "tof_days": (entry_et - depart0)/SECONDS_PER_DAY,
                "perturbers": [x[0] for x in pert_pre],
                "entry_position_miss_km": entry_pos_miss,
                "entry_velocity_miss_m_s": entry_vel_miss,
                "pass_entry_patch": entry_pass,
                "nfev": pre.get("nfev"),
            },
            "local_flyby": {
                "pass_local_validation": bool(local_val.get("pass_validation", True)),
                "endpoint_position_miss_km": opt_float(local_val.get("endpoint_position_miss_km")),
                "endpoint_velocity_miss_m_s": opt_float(local_val.get("endpoint_velocity_miss_m_s")),
                "periapsis_radius_error_km": opt_float(local_val.get("periapsis_radius_error_km")),
                "soi_to_periapsis_days": opt_float(patch_epochs.get("soi_to_periapsis_days")),
            },
            "post_arc": {
                "origin_patch": f"{flyby_body}_SOI_out", "target": target1,
                "tof_days": (arrive1 - exit_et)/SECONDS_PER_DAY,
                "perturbers": [x[0] for x in pert_post],
                "arrival_position_miss_km": arrival_pos_miss,
                "arrival_vinf_km_s": arrival_vinf,
                "pass_arrival_patch": arrival_pass,
                "nfev": post.get("nfev"),
            },
            "quality": {
                "total_departure_correction_m_s": (((target.get("quality") or {}) if isinstance(target.get("quality"), Mapping) else {}).get("total_departure_correction_m_s")),
                "rp_margin_km": (((target.get("hyperbola") or {}) if isinstance(target.get("hyperbola"), Mapping) else {}).get("rp_margin_km")),
                "periapsis_altitude_km": (((target.get("hyperbola") or {}) if isinstance(target.get("hyperbola"), Mapping) else {}).get("periapsis_altitude_km")),
            },
            "source_stitched_packet": stitched,
        }
    except Exception as exc:
        return {
            "schema_version": SCHEMA_VERSION,
            "ok": False,
            "pass_validation": False,
            "message": repr(exc),
            "stitched_packet_id": stitched.get("stitched_packet_id"),
            "source_stitched_packet": stitched,
        }


def _worker_validate(payload: Tuple[Mapping[str, Any], Optional[Mapping[str, Any]], Mapping[str, Any]]) -> Dict[str, Any]:
    return validate_one(payload)


def flat_row(r: Mapping[str, Any]) -> Dict[str, Any]:
    pre = r.get("pre_arc") if isinstance(r.get("pre_arc"), Mapping) else {}
    loc = r.get("local_flyby") if isinstance(r.get("local_flyby"), Mapping) else {}
    post = r.get("post_arc") if isinstance(r.get("post_arc"), Mapping) else {}
    q = r.get("quality") if isinstance(r.get("quality"), Mapping) else {}
    return {
        "validation_id": r.get("validation_id"),
        "stitched_packet_id": r.get("stitched_packet_id"),
        "packet_id": r.get("packet_id"),
        "sequence": r.get("sequence"),
        "flyby_body": r.get("flyby_body"),
        "ok": int(bool(r.get("ok"))),
        "pass_validation": int(bool(r.get("pass_validation"))),
        "entry_position_miss_km": pre.get("entry_position_miss_km"),
        "entry_velocity_miss_m_s": pre.get("entry_velocity_miss_m_s"),
        "arrival_position_miss_km": post.get("arrival_position_miss_km"),
        "arrival_vinf_km_s": post.get("arrival_vinf_km_s"),
        "local_endpoint_position_miss_km": loc.get("endpoint_position_miss_km"),
        "periapsis_radius_error_km": loc.get("periapsis_radius_error_km"),
        "total_departure_correction_m_s": q.get("total_departure_correction_m_s"),
        "rp_margin_km": q.get("rp_margin_km"),
        "periapsis_altitude_km": q.get("periapsis_altitude_km"),
        "message": r.get("message"),
    }


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "validation_id", "stitched_packet_id", "packet_id", "sequence", "flyby_body", "ok", "pass_validation",
        "entry_position_miss_km", "entry_velocity_miss_m_s", "arrival_position_miss_km", "arrival_vinf_km_s",
        "local_endpoint_position_miss_km", "periapsis_radius_error_km",
        "total_departure_correction_m_s", "rp_margin_km", "periapsis_altitude_km", "message",
    ]
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
    p = argparse.ArgumentParser(description="Validate stitched global/local MGA route packet against patched heliocentric arcs.")
    p.add_argument("--bsp", required=True, type=Path)
    p.add_argument("--tpc", type=Path, default=None)
    p.add_argument("--body-catalog", required=True, type=Path)
    p.add_argument("--stitched-jsonl", required=True, type=Path, help="JSON/JSONL from mga_stitch_global_local_packet_v0_1.py")
    p.add_argument("--bplane-packet", required=True, type=Path, help="JSON/JSONL from mga_make_bplane_packet_v0_1.py")
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
    p.add_argument("--entry-position-threshold-km", type=float, default=10000.0)
    p.add_argument("--entry-velocity-threshold-m-s", type=float, default=100.0)
    p.add_argument("--arrival-position-threshold-km", type=float, default=10000.0)
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
    stitched.sort(key=lambda s: (
        not bool(s.get("ok")),
        finite((((s.get("target") or {}).get("quality") or {}).get("total_departure_correction_m_s")), 1e99) if isinstance(s.get("target"), Mapping) else 1e99,
    ))
    if args.top_n > 0:
        stitched = stitched[:args.top_n]
    grav = args.gravitating_bodies or sorted((catalog.get("bodies") or {}).keys())
    kernels = [str(args.bsp)]
    if args.tpc:
        kernels.append(str(args.tpc))
    cfg = {
        "kernels": kernels,
        "central_body": args.central_body,
        "frame": args.frame,
        "mu_central_km3_s2": args.mu_central_km3_s2,
        "gravitating_bodies": grav,
        "dynamics_mode": args.dynamics_mode,
        "integrator": args.integrator,
        "rtol": args.rtol,
        "atol_position_km": args.atol_position_km,
        "atol_velocity_km_s": args.atol_velocity_km_s,
        "max_step_days": args.max_step_days,
        "entry_position_threshold_km": args.entry_position_threshold_km,
        "entry_velocity_threshold_m_s": args.entry_velocity_threshold_m_s,
        "arrival_position_threshold_km": args.arrival_position_threshold_km,
    }
    payloads = [(s, find_bplane_for_stitched(s, idx), catalog) for s in stitched]
    workers = os.cpu_count() or 1 if args.workers == 0 else max(1, args.workers)
    results: List[Dict[str, Any]] = []
    if workers == 1 or len(payloads) <= 1:
        _init_worker(cfg)
        for pld in payloads:
            results.append(validate_one(pld))
    else:
        ctx = mp.get_context(args.multiprocessing_start_method)
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx, initializer=_init_worker, initargs=(cfg,)) as ex:
            futs = [ex.submit(_worker_validate, p) for p in payloads]
            for fut in as_completed(futs):
                results.append(fut.result())
    results.sort(key=lambda r: (
        not bool(r.get("pass_validation")),
        finite(((r.get("pre_arc") or {}).get("entry_position_miss_km")), 1e99) if isinstance(r.get("pre_arc"), Mapping) else 1e99,
        finite(((r.get("post_arc") or {}).get("arrival_position_miss_km")), 1e99) if isinstance(r.get("post_arc"), Mapping) else 1e99,
    ))
    rows = [flat_row(r) for r in results]
    pass_count = sum(1 for r in results if r.get("pass_validation"))
    entry_pos = [finite(row.get("entry_position_miss_km")) for row in rows]
    entry_vel = [finite(row.get("entry_velocity_miss_m_s")) for row in rows]
    arrival_pos = [finite(row.get("arrival_position_miss_km")) for row in rows]
    write_csv(args.output_csv, rows)
    write_jsonl(args.output_jsonl, results)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "stitched_input": str(args.stitched_jsonl),
        "bplane_packet": str(args.bplane_packet),
        "records_input": len(stitched),
        "records_validated": len(results),
        "pass_validation": pass_count,
        "workers": workers,
        "dynamics_mode": args.dynamics_mode,
        "thresholds": {
            "entry_position_threshold_km": args.entry_position_threshold_km,
            "entry_velocity_threshold_m_s": args.entry_velocity_threshold_m_s,
            "arrival_position_threshold_km": args.arrival_position_threshold_km,
        },
        "entry_position_miss_km": stats(entry_pos),
        "entry_velocity_miss_m_s": stats(entry_vel),
        "arrival_position_miss_km": stats(arrival_pos),
        "top_results": rows[:20],
    }
    write_json(args.output_json, summary)
    write_json(args.output_best_json, results[0] if results else {})
    print("="*80)
    print("MGA STITCHED ROUTE VALIDATION V0.1")
    print("="*80)
    print(f"Stitched packets: {len(stitched)}")
    print(f"Validated:        {len(results)}")
    print(f"Pass validation:  {pass_count}")
    print(f"Workers:          {workers}")
    print(f"Dynamics mode:    {args.dynamics_mode}")
    st_ep = stats(entry_pos); st_ev = stats(entry_vel); st_ar = stats(arrival_pos)
    print(f"Entry pos miss:   min={st_ep['min']} median={st_ep['median']} max={st_ep['max']} km")
    print(f"Entry vel miss:   min={st_ev['min']} median={st_ev['median']} max={st_ev['max']} m/s")
    print(f"Arrival miss:     min={st_ar['min']} median={st_ar['median']} max={st_ar['max']} km")
    print("\nTop stitched validations:")
    for i, row in enumerate(rows[:10], start=1):
        print(
            f" {i}. {row.get('sequence')} @ {row.get('flyby_body')} | pass={bool(row.get('pass_validation'))} | "
            f"entry={finite(row.get('entry_position_miss_km')):.3g} km / {finite(row.get('entry_velocity_miss_m_s')):.3g} m/s | "
            f"arrival={finite(row.get('arrival_position_miss_km')):.3g} km | "
            f"rp_margin={finite(row.get('rp_margin_km')):.1f} km"
        )
    print("="*80)
    print(f"[OK] wrote CSV:       {args.output_csv}")
    print(f"[OK] wrote JSONL:     {args.output_jsonl}")
    print(f"[OK] wrote JSON:      {args.output_json}")
    print(f"[OK] wrote best JSON: {args.output_best_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
