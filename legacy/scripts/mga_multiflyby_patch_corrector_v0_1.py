#!/usr/bin/env python3
"""
mga_multiflyby_patch_corrector_v0_1.py

Correct stitched multi-flyby route packets by solving small impulsive patch-point
corrections on the heliocentric arcs between local flyby SOI patch states.

Input:
  - JSONL/JSON from mga_stitch_multiflyby_packet_v0_1.py
  - SPICE BSP/TPC
  - body catalog JSON

For a route like Kerbin -> Eve -> Kerbin -> Jool, this creates/corrects:
  0. Kerbin center @ departure -> Eve SOI-in
  1. Eve SOI-out -> Kerbin SOI-in
  2. Kerbin SOI-out -> Jool center @ final arrival

The flyby segments themselves are assumed already locally validated two-body
hyperbolas. This stage corrects only the inter-patch heliocentric arcs.

Dynamics:
  central-body frame in km/km/s/ET seconds, with optional prescribed SPICE
  perturbations using the standard indirect term:
      a = -mu0 r/|r|^3 + sum_j mu_j[(r_j-r)/|r_j-r|^3 - r_j/|r_j|^3]
  Endpoint bodies are excluded as perturbing bodies on that arc by default,
  matching the earlier patched_heliocentric semantics.

Outputs:
  - corrected route JSONL
  - CSV segment summary
  - JSON summary
  - best route JSON
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

SCHEMA_VERSION = "mga_multiflyby_patch_corrector.v0.1"
Vec3 = Tuple[float, float, float]
_WORKER_CFG: Dict[str, Any] = {}
_SPICE = None
_NP = None
_SCIPY_INTEGRATE = None
_SCIPY_OPT = None


def finite(x: Any, default: float = math.nan) -> float:
    try:
        y = float(x)
    except (TypeError, ValueError):
        return default
    return y if math.isfinite(y) else default


def opt_float(x: Any) -> Optional[float]:
    y = finite(x)
    return y if math.isfinite(y) else None


def vec3_opt(x: Any) -> Optional[Vec3]:
    if not isinstance(x, Sequence) or isinstance(x, (str, bytes)) or len(x) < 3:
        return None
    out: List[float] = []
    for i in range(3):
        y = finite(x[i])
        if not math.isfinite(y):
            return None
        out.append(y)
    return (out[0], out[1], out[2])


def vec3_req(x: Any, name: str) -> Vec3:
    v = vec3_opt(x)
    if v is None:
        raise ValueError(f"missing/invalid 3-vector: {name}")
    return v


def vadd(a: Sequence[float], b: Sequence[float]) -> Vec3:
    return (float(a[0])+float(b[0]), float(a[1])+float(b[1]), float(a[2])+float(b[2]))


def vsub(a: Sequence[float], b: Sequence[float]) -> Vec3:
    return (float(a[0])-float(b[0]), float(a[1])-float(b[1]), float(a[2])-float(b[2]))


def vnorm(a: Sequence[float]) -> float:
    return math.sqrt(float(a[0])**2 + float(a[1])**2 + float(a[2])**2)


def json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, tuple):
        return list(obj)
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    return str(obj)


def stable_id(prefix: str, payload: Mapping[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=True, separators=(",", ":"), default=str)
    return f"{prefix}_{hashlib.sha1(raw.encode('utf-8')).hexdigest()[:16]}"


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return data


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if not isinstance(obj, Mapping):
                raise ValueError(f"Expected object at {path}:{i}")
            out.append(dict(obj))
    return out


def load_packets(path: Path) -> List[Dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        return read_jsonl(path)
    data = load_json(path)
    if "stitched_multiflyby_packet_id" in data:
        return [data]
    for key in ("packets", "results", "top_packets"):
        rows = data.get(key)
        if isinstance(rows, list):
            return [dict(r) for r in rows if isinstance(r, Mapping)]
    raise ValueError(f"Could not locate stitched multi-flyby packets in {path}")


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, allow_nan=False, default=json_default)
        f.write("\n")


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, allow_nan=False, separators=(",", ":"), default=json_default))
            f.write("\n")


@dataclass
class BodyMu:
    name: str
    mu_km3_s2: float


@dataclass
class SegmentCorrection:
    segment_index: int
    segment_kind: str
    origin_label: str
    target_label: str
    depart_et: float
    arrive_et: float
    tof_days: float
    miss_before_km: Optional[float]
    miss_after_km: Optional[float]
    velocity_miss_after_m_s: Optional[float]
    dvx_km_s: Optional[float]
    dvy_km_s: Optional[float]
    dvz_km_s: Optional[float]
    dv_norm_m_s: Optional[float]
    optimizer_success: bool
    optimizer_status: str
    nfev: Optional[int]
    pass_position: bool
    pass_velocity: bool
    pass_segment: bool
    notes: str


def _init_worker(cfg: Mapping[str, Any]) -> None:
    global _WORKER_CFG, _SPICE, _NP, _SCIPY_INTEGRATE, _SCIPY_OPT
    _WORKER_CFG = dict(cfg)
    _SPICE = importlib.import_module("spiceypy")
    _NP = importlib.import_module("numpy")
    _SCIPY_INTEGRATE = importlib.import_module("scipy.integrate")
    _SCIPY_OPT = importlib.import_module("scipy.optimize")
    _SPICE.kclear()
    for kernel in _WORKER_CFG.get("kernels", []):
        _SPICE.furnsh(str(kernel))


def _state_body(body: str, et: float, central: str, frame: str) -> Tuple[Vec3, Vec3]:
    st, _lt = _SPICE.spkezr(str(body), float(et), str(frame), "NONE", str(central))
    return (float(st[0]), float(st[1]), float(st[2])), (float(st[3]), float(st[4]), float(st[5]))


def _body_from_catalog(catalog: Mapping[str, Any], name: str) -> Optional[BodyMu]:
    bodies = catalog.get("bodies", {}) if isinstance(catalog, Mapping) else {}
    ent = bodies.get(name) if isinstance(bodies, Mapping) else None
    if not isinstance(ent, Mapping):
        for k, v in bodies.items() if isinstance(bodies, Mapping) else []:
            if str(k).lower() == str(name).lower() and isinstance(v, Mapping):
                ent = v
                break
    if not isinstance(ent, Mapping):
        return None
    mu = finite(ent.get("mu_km3_s2", ent.get("gm_km3_s2")))
    if not math.isfinite(mu) or mu <= 0:
        return None
    return BodyMu(name=str(name), mu_km3_s2=mu)


def _load_perturbers(catalog: Mapping[str, Any], names: Sequence[str]) -> List[BodyMu]:
    out: List[BodyMu] = []
    for n in names:
        b = _body_from_catalog(catalog, str(n))
        if b is not None:
            out.append(b)
    return out


def _accel(t: float, y: Sequence[float], central: str, frame: str, mu_c: float, perturbers: Sequence[BodyMu]) -> List[float]:
    np = _NP
    r = np.array([y[0], y[1], y[2]], dtype=float)
    v = np.array([y[3], y[4], y[5]], dtype=float)
    nr = float(np.linalg.norm(r))
    if nr <= 0:
        a = np.zeros(3)
    else:
        a = -mu_c * r / (nr**3)
    for b in perturbers:
        if b.name == central or b.mu_km3_s2 <= 0:
            continue
        try:
            rb, _ = _state_body(b.name, t, central, frame)
        except Exception:
            continue
        rbv = np.array(rb, dtype=float)
        dr = rbv - r
        ndr = float(np.linalg.norm(dr))
        nrb = float(np.linalg.norm(rbv))
        if ndr > 0:
            a += b.mu_km3_s2 * dr / (ndr**3)
        if nrb > 0:
            a -= b.mu_km3_s2 * rbv / (nrb**3)
    return [float(v[0]), float(v[1]), float(v[2]), float(a[0]), float(a[1]), float(a[2])]


def _propagate(r0: Vec3, v0: Vec3, t0: float, t1: float, perturbers: Sequence[BodyMu], cfg: Mapping[str, Any]) -> Tuple[Optional[Vec3], Optional[Vec3], bool, str]:
    y0 = [r0[0], r0[1], r0[2], v0[0], v0[1], v0[2]]
    max_step = max(1.0, finite(cfg.get("max_step_days"), 1.0) * 86400.0)
    try:
        sol = _SCIPY_INTEGRATE.solve_ivp(
            lambda t, y: _accel(t, y, str(cfg["central_body"]), str(cfg["frame"]), float(cfg["mu_central"]), perturbers),
            (float(t0), float(t1)), y0,
            method=str(cfg.get("integrator") or "DOP853"),
            rtol=float(cfg.get("rtol") or 1e-10),
            atol=float(cfg.get("atol") or 1e-12),
            max_step=max_step,
        )
    except Exception as exc:
        return None, None, False, f"integrator_exception:{exc!r}"
    if not sol.success or sol.y.shape[1] == 0:
        return None, None, False, f"integrator_failed:{sol.message}"
    yf = sol.y[:, -1]
    return (float(yf[0]), float(yf[1]), float(yf[2])), (float(yf[3]), float(yf[4]), float(yf[5])), True, str(sol.message)


def _lambert_v0(r0: Vec3, r1: Vec3, tof_s: float, mu: float) -> Optional[Vec3]:
    try:
        pk = importlib.import_module("pykep")
        lp = pk.lambert_problem(list(r0), list(r1), float(tof_s), float(mu), False, 0)
        v1s = lp.get_v1()
        if not v1s:
            return None
        v = v1s[0]
        return (float(v[0]), float(v[1]), float(v[2]))
    except Exception:
        return None


def _state_block_to_rv(block: Mapping[str, Any], name: str) -> Tuple[float, Vec3, Vec3]:
    et = finite(block.get("et"))
    sc = block.get("spacecraft_state_central") if isinstance(block.get("spacecraft_state_central"), Mapping) else {}
    r = vec3_req(sc.get("r_km"), f"{name}.r_km")
    v = vec3_req(sc.get("v_km_s"), f"{name}.v_km_s")
    if not math.isfinite(et):
        raise ValueError(f"missing et in {name}")
    return et, r, v


def _extract_sequence(packet: Mapping[str, Any]) -> List[str]:
    seq = packet.get("sequence")
    if isinstance(seq, str):
        if "->" in seq:
            return [s.strip() for s in seq.split("->") if s.strip()]
        if "," in seq:
            return [s.strip() for s in seq.split(",") if s.strip()]
        return [seq.strip()]
    if isinstance(seq, Sequence) and not isinstance(seq, (bytes, bytearray)):
        return [str(s) for s in seq]
    return []


def _find_tcm_plan(packet: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    # Local flyby target stores the route-level tcm_plan under flyby.target.tcm_plan.
    for f in packet.get("flybys") or []:
        if not isinstance(f, Mapping):
            continue
        t = f.get("target") if isinstance(f.get("target"), Mapping) else {}
        plan = t.get("tcm_plan")
        if isinstance(plan, list) and plan:
            return [p for p in plan if isinstance(p, Mapping)]
    return []


def _segment_definitions(packet: Mapping[str, Any], cfg: Mapping[str, Any]) -> Tuple[List[Dict[str, Any]], List[str]]:
    failures: List[str] = []
    seq = _extract_sequence(packet)
    flybys = [f for f in (packet.get("flybys") or []) if isinstance(f, Mapping) and f.get("ok")]
    flybys.sort(key=lambda f: finite(((f.get("patch_epochs") or {}).get("periapsis_et")), 1e99))
    if len(seq) < 2:
        failures.append("missing_sequence")
    if not flybys:
        failures.append("missing_flybys")
    plan = _find_tcm_plan(packet)
    if not plan:
        failures.append("missing_tcm_plan")

    segments: List[Dict[str, Any]] = []
    central = str(cfg["central_body"])
    frame = str(cfg["frame"])
    mu = float(cfg["mu_central"])

    try:
        # Segment 0: departure body center -> first flyby entry SOI.
        first = flybys[0]
        entry_block = ((first.get("patch_states") or {}).get("entry_soi") or {})
        entry_et, entry_r, entry_v = _state_block_to_rv(entry_block, "first.entry_soi")
        first_leg = plan[0] if plan else {}
        depart_et = finite(first_leg.get("depart_et"))
        if not math.isfinite(depart_et):
            # fall back to first leg t0 from source if available; otherwise fail
            failures.append("missing_departure_et")
            depart_et = entry_et - 100.0
        origin = seq[0]
        r0, v_body0 = _state_body(origin, depart_et, central, frame)
        lam_v = _lambert_v0(r0, entry_r, entry_et - depart_et, mu)
        if lam_v is None:
            failures.append("lambert_failed_for_departure_to_first_entry")
            lam_v = v_body0
        # Use available route-level leg correction as initial TCM guess if present, but still solve anew.
        segments.append({
            "segment_index": 0,
            "segment_kind": "departure_to_flyby_entry",
            "origin_label": f"{origin}_center",
            "target_label": f"{first.get('body')}_soi_in",
            "start_exclude_body": origin,
            "target_exclude_body": str(first.get("body")),
            "t0": depart_et,
            "t1": entry_et,
            "r0": r0,
            "v0": lam_v,
            "r_target": entry_r,
            "v_target": entry_v,
        })

        # Intermediate segments: flyby i exit SOI -> flyby i+1 entry SOI.
        for i in range(len(flybys)-1):
            fa = flybys[i]
            fb = flybys[i+1]
            exit_block = ((fa.get("patch_states") or {}).get("exit_soi") or {})
            next_entry_block = ((fb.get("patch_states") or {}).get("entry_soi") or {})
            t0, r0, v0 = _state_block_to_rv(exit_block, f"flyby[{i}].exit_soi")
            t1, r1, v1 = _state_block_to_rv(next_entry_block, f"flyby[{i+1}].entry_soi")
            segments.append({
                "segment_index": len(segments),
                "segment_kind": "flyby_exit_to_next_flyby_entry",
                "origin_label": f"{fa.get('body')}_soi_out",
                "target_label": f"{fb.get('body')}_soi_in",
                "start_exclude_body": str(fa.get("body")),
                "target_exclude_body": str(fb.get("body")),
                "t0": t0,
                "t1": t1,
                "r0": r0,
                "v0": v0,
                "r_target": r1,
                "v_target": v1,
            })

        # Final segment: last flyby exit SOI -> final target body center.
        last = flybys[-1]
        exit_block = ((last.get("patch_states") or {}).get("exit_soi") or {})
        t0, r0, v0 = _state_block_to_rv(exit_block, "last.exit_soi")
        final_body = seq[-1] if seq else str((plan[-1] or {}).get("target", ""))
        final_leg = plan[-1] if plan else {}
        arrive_et = finite(final_leg.get("arrive_et"))
        if not math.isfinite(arrive_et):
            failures.append("missing_final_arrive_et")
            arrive_et = t0 + 100.0
        r1, v1 = _state_body(final_body, arrive_et, central, frame)
        segments.append({
            "segment_index": len(segments),
            "segment_kind": "last_flyby_exit_to_final_target",
            "origin_label": f"{last.get('body')}_soi_out",
            "target_label": f"{final_body}_center",
            "start_exclude_body": str(last.get("body")),
            "target_exclude_body": final_body,
            "t0": t0,
            "t1": arrive_et,
            "r0": r0,
            "v0": v0,
            "r_target": r1,
            "v_target": v1,
        })
    except Exception as exc:
        failures.append(f"segment_build_exception:{exc!r}")
    return segments, failures


def _correct_segment(seg: Mapping[str, Any], all_perturbers: Sequence[BodyMu], cfg: Mapping[str, Any]) -> SegmentCorrection:
    max_corr = float(cfg.get("max_segment_correction_m_s", 50.0)) / 1000.0
    pos_threshold = float(cfg.get("target_position_miss_km", 10.0))
    vel_threshold_m_s = float(cfg.get("target_velocity_miss_m_s", 250.0))
    max_nfev = int(cfg.get("max_nfev", 30))

    r0 = vec3_req(seg.get("r0"), "segment.r0")
    v0 = vec3_req(seg.get("v0"), "segment.v0")
    r_target = vec3_req(seg.get("r_target"), "segment.r_target")
    v_target = vec3_req(seg.get("v_target"), "segment.v_target")
    t0 = finite(seg.get("t0"))
    t1 = finite(seg.get("t1"))
    if not (math.isfinite(t0) and math.isfinite(t1) and t1 > t0):
        return SegmentCorrection(int(seg.get("segment_index", -1)), str(seg.get("segment_kind")), str(seg.get("origin_label")), str(seg.get("target_label")), t0, t1, math.nan, None, None, None, None, None, None, None, False, "bad_time", None, False, False, False, "bad_time")

    excluded = {str(seg.get("start_exclude_body") or "").lower(), str(seg.get("target_exclude_body") or "").lower(), str(cfg.get("central_body")).lower()}
    pert = [b for b in all_perturbers if b.name.lower() not in excluded]

    def eval_miss(dv: Sequence[float]) -> Tuple[float, Optional[Vec3], Optional[Vec3], bool, str]:
        vstart = (v0[0]+float(dv[0]), v0[1]+float(dv[1]), v0[2]+float(dv[2]))
        rf, vf, ok, msg = _propagate(r0, vstart, t0, t1, pert, cfg)
        if not ok or rf is None or vf is None:
            return 1e99, None, None, False, msg
        miss = vnorm(vsub(rf, r_target))
        return miss, rf, vf, True, msg

    miss0, rf0, vf0, ok0, msg0 = eval_miss((0.0, 0.0, 0.0))

    def residual(dv_arr: Any) -> Any:
        miss, rf, _vf, ok, _msg = eval_miss((float(dv_arr[0]), float(dv_arr[1]), float(dv_arr[2])))
        if not ok or rf is None:
            return _NP.array([1e6, 1e6, 1e6], dtype=float)
        return _NP.array([rf[0]-r_target[0], rf[1]-r_target[1], rf[2]-r_target[2]], dtype=float) / max(1.0, pos_threshold)

    try:
        res = _SCIPY_OPT.least_squares(
            residual,
            _NP.zeros(3),
            bounds=([-max_corr, -max_corr, -max_corr], [max_corr, max_corr, max_corr]),
            max_nfev=max_nfev,
            xtol=1e-10,
            ftol=1e-10,
            gtol=1e-10,
        )
        dv = (float(res.x[0]), float(res.x[1]), float(res.x[2]))
        opt_success = bool(res.success)
        opt_status = str(res.message)
        nfev = int(res.nfev)
    except Exception as exc:
        dv = (0.0, 0.0, 0.0)
        opt_success = False
        opt_status = f"optimizer_exception:{exc!r}"
        nfev = None

    miss1, rf1, vf1, ok1, msg1 = eval_miss(dv)
    vel_mis = vnorm(vsub(vf1, v_target))*1000.0 if vf1 is not None else math.nan
    dv_m_s = vnorm(dv)*1000.0
    pass_pos = bool(ok1 and math.isfinite(miss1) and miss1 <= pos_threshold)
    pass_vel = bool(ok1 and math.isfinite(vel_mis) and vel_mis <= vel_threshold_m_s)
    return SegmentCorrection(
        segment_index=int(seg.get("segment_index", -1)),
        segment_kind=str(seg.get("segment_kind")),
        origin_label=str(seg.get("origin_label")),
        target_label=str(seg.get("target_label")),
        depart_et=t0,
        arrive_et=t1,
        tof_days=(t1-t0)/86400.0,
        miss_before_km=opt_float(miss0),
        miss_after_km=opt_float(miss1),
        velocity_miss_after_m_s=opt_float(vel_mis),
        dvx_km_s=opt_float(dv[0]),
        dvy_km_s=opt_float(dv[1]),
        dvz_km_s=opt_float(dv[2]),
        dv_norm_m_s=opt_float(dv_m_s),
        optimizer_success=opt_success,
        optimizer_status=opt_status,
        nfev=nfev,
        pass_position=pass_pos,
        pass_velocity=pass_vel,
        pass_segment=bool(pass_pos and pass_vel),
        notes=msg1 if ok1 else msg1,
    )


def _process_packet(packet: Mapping[str, Any]) -> Dict[str, Any]:
    cfg = _WORKER_CFG
    catalog = cfg.get("body_catalog") if isinstance(cfg.get("body_catalog"), Mapping) else {}
    grav_names = [str(x) for x in (cfg.get("gravitating_bodies") or [])]
    all_pert = _load_perturbers(catalog, grav_names)
    segments, failures = _segment_definitions(packet, cfg)
    seg_corrs: List[SegmentCorrection] = []
    if not failures:
        for seg in segments:
            try:
                seg_corrs.append(_correct_segment(seg, all_pert, cfg))
            except Exception as exc:
                seg_corrs.append(SegmentCorrection(int(seg.get("segment_index", -1)), str(seg.get("segment_kind")), str(seg.get("origin_label")), str(seg.get("target_label")), finite(seg.get("t0")), finite(seg.get("t1")), math.nan, None, None, None, None, None, None, None, False, f"exception:{exc!r}", None, False, False, False, f"exception:{exc!r}"))

    misses_before = [s.miss_before_km for s in seg_corrs if s.miss_before_km is not None and math.isfinite(s.miss_before_km)]
    misses_after = [s.miss_after_km for s in seg_corrs if s.miss_after_km is not None and math.isfinite(s.miss_after_km)]
    vels_after = [s.velocity_miss_after_m_s for s in seg_corrs if s.velocity_miss_after_m_s is not None and math.isfinite(s.velocity_miss_after_m_s)]
    dvs = [s.dv_norm_m_s for s in seg_corrs if s.dv_norm_m_s is not None and math.isfinite(s.dv_norm_m_s)]
    all_pass = bool(seg_corrs and all(s.pass_segment for s in seg_corrs) and not failures)
    corrected_id = stable_id("mfpatch", {
        "stitched": packet.get("stitched_multiflyby_packet_id"),
        "miss_after": max(misses_after) if misses_after else None,
        "dv": sum(dvs) if dvs else None,
    })
    return {
        "schema_version": SCHEMA_VERSION,
        "multiflyby_patch_correction_id": corrected_id,
        "stitched_multiflyby_packet_id": packet.get("stitched_multiflyby_packet_id"),
        "sequence": packet.get("sequence"),
        "flyby_bodies": packet.get("flyby_bodies"),
        "n_segments": len(seg_corrs),
        "all_segments_pass": all_pass,
        "max_miss_before_km": max(misses_before) if misses_before else None,
        "max_miss_after_km": max(misses_after) if misses_after else None,
        "max_velocity_miss_after_m_s": max(vels_after) if vels_after else None,
        "total_segment_correction_m_s": sum(dvs) if dvs else None,
        "max_segment_correction_m_s": max(dvs) if dvs else None,
        "segment_corrections": [asdict(s) for s in seg_corrs],
        "source_packet_metrics": packet.get("metrics"),
        "source_packet": packet if cfg.get("embed_source") else None,
        "status": {
            "pass_correction": all_pass,
            "failures": failures + [f"seg{s.segment_index}:{s.notes}" for s in seg_corrs if not s.pass_segment],
            "recommended_next_stage": "multiflyby_bplane_target_spec_or_final_packet",
        },
    }


def flatten(result: Mapping[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for s in result.get("segment_corrections") or []:
        if not isinstance(s, Mapping):
            continue
        rows.append({
            "correction_id": result.get("multiflyby_patch_correction_id"),
            "stitched_multiflyby_packet_id": result.get("stitched_multiflyby_packet_id"),
            "sequence": result.get("sequence"),
            "all_segments_pass": int(bool(result.get("all_segments_pass"))),
            "route_max_miss_before_km": result.get("max_miss_before_km"),
            "route_max_miss_after_km": result.get("max_miss_after_km"),
            "route_total_correction_m_s": result.get("total_segment_correction_m_s"),
            "segment_index": s.get("segment_index"),
            "segment_kind": s.get("segment_kind"),
            "origin_label": s.get("origin_label"),
            "target_label": s.get("target_label"),
            "tof_days": s.get("tof_days"),
            "miss_before_km": s.get("miss_before_km"),
            "miss_after_km": s.get("miss_after_km"),
            "velocity_miss_after_m_s": s.get("velocity_miss_after_m_s"),
            "dv_norm_m_s": s.get("dv_norm_m_s"),
            "pass_segment": int(bool(s.get("pass_segment"))),
            "pass_position": int(bool(s.get("pass_position"))),
            "pass_velocity": int(bool(s.get("pass_velocity"))),
            "optimizer_success": int(bool(s.get("optimizer_success"))),
            "nfev": s.get("nfev"),
            "notes": s.get("notes"),
        })
    return rows


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "correction_id", "stitched_multiflyby_packet_id", "sequence", "all_segments_pass",
        "route_max_miss_before_km", "route_max_miss_after_km", "route_total_correction_m_s",
        "segment_index", "segment_kind", "origin_label", "target_label", "tof_days",
        "miss_before_km", "miss_after_km", "velocity_miss_after_m_s", "dv_norm_m_s",
        "pass_segment", "pass_position", "pass_velocity", "optimizer_success", "nfev", "notes",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fields})


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Correct heliocentric patch arcs between stitched multi-flyby local segments.")
    p.add_argument("--bsp", required=True, type=Path)
    p.add_argument("--tpc", required=True, type=Path)
    p.add_argument("--body-catalog", required=True, type=Path)
    p.add_argument("--stitched-jsonl", required=True, type=Path)
    p.add_argument("--central-body", default="Sun")
    p.add_argument("--mu-central-km3-s2", required=True, type=float)
    p.add_argument("--frame", default="J2000")
    p.add_argument("--gravitating-bodies", nargs="+", required=True)
    p.add_argument("--workers", type=int, default=1, help="0=os.cpu_count(); 1=serial")
    p.add_argument("--multiprocessing-start-method", default="spawn", choices=["spawn", "fork", "forkserver"])
    p.add_argument("--integrator", default="DOP853")
    p.add_argument("--rtol", type=float, default=1e-10)
    p.add_argument("--atol", type=float, default=1e-12)
    p.add_argument("--max-step-days", type=float, default=1.0)
    p.add_argument("--max-segment-correction-m-s", type=float, default=50.0)
    p.add_argument("--target-position-miss-km", type=float, default=10.0)
    p.add_argument("--target-velocity-miss-m-s", type=float, default=250.0)
    p.add_argument("--max-nfev", type=int, default=30)
    p.add_argument("--top-n", type=int, default=0)
    p.add_argument("--embed-source", action="store_true")
    p.add_argument("--output-csv", required=True, type=Path)
    p.add_argument("--output-jsonl", required=True, type=Path)
    p.add_argument("--output-json", required=True, type=Path)
    p.add_argument("--output-best-json", required=True, type=Path)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    packets = [p for p in load_packets(args.stitched_jsonl) if bool(p.get("ok"))]
    if args.top_n > 0:
        packets = packets[:args.top_n]
    catalog = load_json(args.body_catalog)
    cfg = {
        "kernels": [str(args.tpc), str(args.bsp)],
        "body_catalog": catalog,
        "central_body": args.central_body,
        "mu_central": args.mu_central_km3_s2,
        "frame": args.frame,
        "gravitating_bodies": args.gravitating_bodies,
        "integrator": args.integrator,
        "rtol": args.rtol,
        "atol": args.atol,
        "max_step_days": args.max_step_days,
        "max_segment_correction_m_s": args.max_segment_correction_m_s,
        "target_position_miss_km": args.target_position_miss_km,
        "target_velocity_miss_m_s": args.target_velocity_miss_m_s,
        "max_nfev": args.max_nfev,
        "embed_source": args.embed_source,
    }
    workers = os.cpu_count() or 1 if args.workers == 0 else max(1, args.workers)
    results: List[Dict[str, Any]] = []
    if workers == 1 or len(packets) <= 1:
        _init_worker(cfg)
        for p in packets:
            results.append(_process_packet(p))
    else:
        ctx = mp.get_context(args.multiprocessing_start_method)
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx, initializer=_init_worker, initargs=(cfg,)) as ex:
            futs = [ex.submit(_process_packet, p) for p in packets]
            for fut in as_completed(futs):
                results.append(fut.result())

    results.sort(key=lambda r: (
        not bool(r.get("all_segments_pass")),
        finite(r.get("total_segment_correction_m_s"), 1e99),
        finite(r.get("max_miss_after_km"), 1e99),
        finite(r.get("max_velocity_miss_after_m_s"), 1e99),
    ))
    flat: List[Dict[str, Any]] = []
    for r in results:
        flat.extend(flatten(r))
    pass_count = sum(1 for r in results if r.get("all_segments_pass"))
    miss_before = [finite(r.get("max_miss_before_km")) for r in results if math.isfinite(finite(r.get("max_miss_before_km")))]
    miss_after = [finite(r.get("max_miss_after_km")) for r in results if math.isfinite(finite(r.get("max_miss_after_km")))]
    vel_after = [finite(r.get("max_velocity_miss_after_m_s")) for r in results if math.isfinite(finite(r.get("max_velocity_miss_after_m_s")))]
    dvs = [finite(r.get("total_segment_correction_m_s")) for r in results if math.isfinite(finite(r.get("total_segment_correction_m_s")))]
    def stats(xs: Sequence[float]) -> Dict[str, Optional[float]]:
        if not xs:
            return {"min": None, "median": None, "max": None}
        ys = sorted(xs)
        return {"min": ys[0], "median": ys[len(ys)//2], "max": ys[-1]}

    write_csv(args.output_csv, flat)
    write_jsonl(args.output_jsonl, results)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "input_packets": len(packets),
        "processed": len(results),
        "pass_correction": pass_count,
        "workers": workers,
        "max_segment_correction_m_s": args.max_segment_correction_m_s,
        "target_position_miss_km": args.target_position_miss_km,
        "target_velocity_miss_m_s": args.target_velocity_miss_m_s,
        "miss_before_km": stats(miss_before),
        "miss_after_km": stats(miss_after),
        "velocity_after_m_s": stats(vel_after),
        "total_segment_correction_m_s": stats(dvs),
        "top_results": results[:10],
    }
    write_json(args.output_json, summary)
    write_json(args.output_best_json, results[0] if results else {})

    print("="*80)
    print("MGA MULTI-FLYBY PATCH CORRECTOR V0.1")
    print("="*80)
    print(f"Packets processed: {len(results)}")
    print(f"Pass correction:   {pass_count}")
    print(f"Workers:           {workers}")
    print(f"Target pos miss:   {args.target_position_miss_km:g} km")
    print(f"Target vel miss:   {args.target_velocity_miss_m_s:g} m/s")
    print(f"Total correction:  min={stats(dvs)['min']} median={stats(dvs)['median']} max={stats(dvs)['max']} m/s")
    print(f"Max miss before:   min={stats(miss_before)['min']} median={stats(miss_before)['median']} max={stats(miss_before)['max']} km")
    print(f"Max miss after:    min={stats(miss_after)['min']} median={stats(miss_after)['median']} max={stats(miss_after)['max']} km")
    print(f"Max vel after:     min={stats(vel_after)['min']} median={stats(vel_after)['median']} max={stats(vel_after)['max']} m/s")
    print("\nTop corrected multi-flyby routes:")
    for i, r in enumerate(results[:10], 1):
        print(
            f" {i}. {r.get('sequence')} | pass={r.get('all_segments_pass')} | "
            f"miss {finite(r.get('max_miss_before_km')):.4g}->{finite(r.get('max_miss_after_km')):.4g} km | "
            f"vel={finite(r.get('max_velocity_miss_after_m_s')):.3g} m/s | "
            f"dv={finite(r.get('total_segment_correction_m_s')):.3f} m/s"
        )
    print("="*80)
    print(f"[OK] wrote CSV:       {args.output_csv}")
    print(f"[OK] wrote JSONL:     {args.output_jsonl}")
    print(f"[OK] wrote JSON:      {args.output_json}")
    print(f"[OK] wrote best JSON: {args.output_best_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
