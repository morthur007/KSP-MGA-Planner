#!/usr/bin/env python3
"""
mga_rebound_ias15_segment_validate_v0_1.py

Independent REBOUND/IAS15 validation for high-fidelity segment packets.

This validator is intentionally segment-based. It validates the exported states
without requiring a full continuous spacecraft model through planet centres.

Modes:
  patched (default):
    - heliocentric segments exclude local endpoint bodies from gravity, matching
      the patched heliocentric correction model.
    - endpoint target is the frozen packet target state.
  endpoint_bodies:
    - exclude only the body at which the spacecraft starts inside/near the local
      patch, include the target body and compare relative to its integrated state.
    - more aggressive; expect larger deviations.

Units: km, seconds, km/s. REBOUND uses sim.G=1 and masses=GM=mu.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

SECONDS_PER_DAY = 86400.0
SCHEMA_VERSION = "mga_rebound_ias15_segment_validation.v0.1"
Vec3 = Tuple[float, float, float]
_WORKER_CFG: Dict[str, Any] = {}
_WORKER_SPICE: Any = None
_WORKER_REBOUND: Any = None


def finite(x: Any, default: float = math.nan) -> float:
    try:
        y = float(x)
    except (TypeError, ValueError):
        return default
    return y if math.isfinite(y) else default


def vec3(x: Any) -> Optional[Vec3]:
    if not isinstance(x, Sequence) or isinstance(x, (str, bytes)) or len(x) < 3:
        return None
    out: List[float] = []
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


def vnorm(a: Sequence[float]) -> float:
    return math.sqrt(float(a[0])**2 + float(a[1])**2 + float(a[2])**2)


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


def load_packets(path: Path) -> List[Dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        return read_jsonl(path)
    data = load_json(path)
    if "segments" in data:
        return [data]
    for key in ("packets", "records", "results", "top_packets"):
        rows = data.get(key)
        if isinstance(rows, list):
            return [dict(r) for r in rows if isinstance(r, Mapping)]
    return [data]


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(sanitize(payload), f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write("\n")


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(sanitize(row), ensure_ascii=False, separators=(",", ":"), allow_nan=False))
            f.write("\n")


def load_body_catalog(path: Path) -> Dict[str, Any]:
    data = load_json(path)
    bodies = data.get("bodies")
    return bodies if isinstance(bodies, dict) else {}


def body_info(catalog: Mapping[str, Any], name: str) -> Dict[str, Any]:
    ent = catalog.get(name)
    if ent is None:
        for k, v in catalog.items():
            if str(k).lower() == name.lower():
                ent = v
                break
    return dict(ent) if isinstance(ent, Mapping) else {}


def _init_worker(cfg: Mapping[str, Any]) -> None:
    global _WORKER_CFG, _WORKER_SPICE, _WORKER_REBOUND
    _WORKER_CFG = dict(cfg)
    import spiceypy as spice  # type: ignore
    import rebound  # type: ignore
    _WORKER_SPICE = spice
    _WORKER_REBOUND = rebound
    spice.kclear()
    if cfg.get("tpc"):
        spice.furnsh(str(cfg["tpc"]))
    spice.furnsh(str(cfg["bsp"]))


def _spice_state(body: str, et: float, frame: str, central: str) -> Tuple[Vec3, Vec3]:
    st, _lt = _WORKER_SPICE.spkezr(str(body), float(et), str(frame), "NONE", str(central))
    return (float(st[0]), float(st[1]), float(st[2])), (float(st[3]), float(st[4]), float(st[5]))


def _sim_add_particle(sim: Any, m: float, r: Vec3, v: Vec3, hash_name: str) -> None:
    try:
        sim.add(m=float(m), x=r[0], y=r[1], z=r[2], vx=v[0], vy=v[1], vz=v[2], hash=hash_name)
    except TypeError:
        sim.add(m=float(m), x=r[0], y=r[1], z=r[2], vx=v[0], vy=v[1], vz=v[2])


def make_rebound_sim(t0_et: float, include_bodies: Sequence[str], catalog: Mapping[str, Any]) -> Tuple[Any, Dict[str, int]]:
    cfg = _WORKER_CFG
    rebound = _WORKER_REBOUND
    sim = rebound.Simulation()
    sim.G = 1.0
    sim.integrator = "ias15"
    try:
        sim.ri_ias15.epsilon = float(cfg.get("ias15_epsilon", 1e-10))
    except Exception:
        pass
    # Sun/central at origin in heliocentric coordinates.
    _sim_add_particle(sim, float(cfg["mu_central_km3_s2"]), (0.0,0.0,0.0), (0.0,0.0,0.0), str(cfg["central_body"]))
    index = {str(cfg["central_body"]): 0}
    for name in include_bodies:
        info = body_info(catalog, str(name))
        mu = finite(info.get("mu_km3_s2"))
        if not (mu > 0):
            continue
        r, v = _spice_state(str(name), t0_et, str(cfg["frame"]), str(cfg["central_body"]))
        _sim_add_particle(sim, mu, r, v, str(name))
        index[str(name)] = len(sim.particles) - 1
    return sim, index


def particle_state(p: Any) -> Tuple[Vec3, Vec3]:
    return (float(p.x), float(p.y), float(p.z)), (float(p.vx), float(p.vy), float(p.vz))


def include_bodies_for_segment(seg: Mapping[str, Any], mode: str, all_bodies: Sequence[str]) -> List[str]:
    stype = str(seg.get("segment_type"))
    if stype == "local_flyby_body_centered_two_body":
        return []
    names = [str(x) for x in all_bodies]
    if mode == "all_bodies":
        return names
    if mode == "endpoint_bodies":
        # Avoid starting at a centre/patch body, but allow the endpoint target to move dynamically.
        exclude = set()
        origin = str(seg.get("from", ""))
        if origin and not origin.endswith("SOI_out"):
            exclude.add(origin.lower())
        if str(seg.get("from", "")).endswith("SOI_out"):
            flyby = str(seg.get("from", "")).replace("_SOI_out", "")
            exclude.add(flyby.lower())
        return [n for n in names if n.lower() not in exclude]
    # patched default: reproduce patched heliocentric segment semantics.
    exclude = set()
    frm = str(seg.get("from", ""))
    to = str(seg.get("to", ""))
    target_body = str(seg.get("target_body", ""))
    if frm and not frm.endswith("SOI_out"):
        exclude.add(frm.lower())
    if frm.endswith("SOI_out"):
        exclude.add(frm.replace("_SOI_out", "").lower())
    if target_body:
        exclude.add(target_body.lower())
    if to and not to.endswith("SOI_in") and not to.endswith("SOI_out"):
        exclude.add(to.lower())
    return [n for n in names if n.lower() not in exclude]


def validate_heliocentric_segment(seg: Mapping[str, Any], catalog: Mapping[str, Any]) -> Dict[str, Any]:
    cfg = _WORKER_CFG
    mode = str(cfg.get("rebound_mode", "patched"))
    t0 = finite(seg.get("t0_et")); t1 = finite(seg.get("t1_et")); dt = t1 - t0
    r0 = vec3(seg.get("r0_km")); v0 = vec3(seg.get("v0_km_s"))
    if r0 is None or v0 is None or not (dt > 0):
        return {"ok": False, "message": "invalid segment state or time"}
    all_bodies = [str(x) for x in cfg.get("gravitating_bodies", [])]
    include = include_bodies_for_segment(seg, mode, all_bodies)
    sim, index = make_rebound_sim(t0, include, catalog)
    _sim_add_particle(sim, 0.0, r0, v0, "spacecraft")
    sc_index = len(sim.particles) - 1
    sim.t = 0.0
    sim.integrate(float(dt), exact_finish_time=1)
    sc_r, sc_v = particle_state(sim.particles[sc_index])

    target_body = str(seg.get("target_body") or seg.get("to") or "")
    if mode == "endpoint_bodies" and target_body in index:
        tb_r, tb_v = particle_state(sim.particles[index[target_body]])
        rel_r = vsub(sc_r, tb_r)
        rel_v = vsub(sc_v, tb_v)
        expected_r = vec3(seg.get("r1_target_body_centered_km")) or (0.0,0.0,0.0)
        expected_v = vec3(seg.get("v1_target_body_centered_km_s")) or (0.0,0.0,0.0)
        pos_miss = vnorm(vsub(rel_r, expected_r))
        vel_miss = vnorm(vsub(rel_v, expected_v))*1000.0
        compare_mode = "relative_to_integrated_target_body"
    else:
        expected_r = vec3(seg.get("r1_target_km"))
        expected_v = vec3(seg.get("v1_patch_km_s")) or vec3(seg.get("v1_target_km_s"))
        if expected_r is None:
            return {"ok": False, "message": "missing expected endpoint position"}
        pos_miss = vnorm(vsub(sc_r, expected_r))
        vel_miss = vnorm(vsub(sc_v, expected_v))*1000.0 if expected_v is not None else math.nan
        compare_mode = "absolute_packet_endpoint"
    return {
        "ok": True,
        "segment_index": seg.get("segment_index"),
        "segment_type": seg.get("segment_type"),
        "compare_mode": compare_mode,
        "included_bodies": include,
        "tof_days": dt/SECONDS_PER_DAY,
        "endpoint_position_miss_km": pos_miss,
        "endpoint_velocity_miss_m_s": vel_miss,
    }


def validate_local_segment(seg: Mapping[str, Any]) -> Dict[str, Any]:
    cfg = _WORKER_CFG
    rebound = _WORKER_REBOUND
    t0 = finite(seg.get("t0_et")); t1 = finite(seg.get("t1_et")); dt = t1 - t0
    mu = finite(seg.get("mu_body_km3_s2"))
    r0 = vec3(seg.get("r0_body_centered_km")); v0 = vec3(seg.get("v0_body_centered_km_s"))
    r1 = vec3(seg.get("r1_body_centered_km")); v1 = vec3(seg.get("v1_body_centered_km_s"))
    if r0 is None or v0 is None or r1 is None or not (mu > 0) or not (dt > 0):
        return {"ok": False, "message": "invalid local segment"}
    sim = rebound.Simulation(); sim.G = 1.0; sim.integrator = "ias15"
    try:
        sim.ri_ias15.epsilon = float(cfg.get("ias15_epsilon", 1e-10))
    except Exception:
        pass
    _sim_add_particle(sim, mu, (0.0,0.0,0.0), (0.0,0.0,0.0), str(seg.get("body", "flyby")))
    _sim_add_particle(sim, 0.0, r0, v0, "spacecraft")
    sim.t = 0.0
    sim.integrate(float(dt), exact_finish_time=1)
    sc_r, sc_v = particle_state(sim.particles[1])
    pos_miss = vnorm(vsub(sc_r, r1))
    vel_miss = vnorm(vsub(sc_v, v1))*1000.0 if v1 is not None else math.nan
    return {
        "ok": True,
        "segment_index": seg.get("segment_index"),
        "segment_type": seg.get("segment_type"),
        "compare_mode": "local_body_centered_two_body",
        "included_bodies": [str(seg.get("body", "flyby"))],
        "tof_days": dt/SECONDS_PER_DAY,
        "endpoint_position_miss_km": pos_miss,
        "endpoint_velocity_miss_m_s": vel_miss,
    }


def validate_packet(packet: Mapping[str, Any], catalog: Mapping[str, Any]) -> Dict[str, Any]:
    try:
        if not packet.get("ready_for_rebound_ias15", False):
            return {"ok": False, "pass_rebound_validation": False, "message": "packet_not_ready", "source_packet": packet}
        segments = packet.get("segments")
        if not isinstance(segments, list) or len(segments) != 3:
            return {"ok": False, "pass_rebound_validation": False, "message": "packet_requires_three_segments", "source_packet": packet}
        seg_results: List[Dict[str, Any]] = []
        for seg in segments:
            if not isinstance(seg, Mapping):
                seg_results.append({"ok": False, "message": "invalid_segment"}); continue
            if str(seg.get("segment_type")) == "local_flyby_body_centered_two_body":
                seg_results.append(validate_local_segment(seg))
            else:
                seg_results.append(validate_heliocentric_segment(seg, catalog))
        max_pos = max([finite(r.get("endpoint_position_miss_km"), math.inf) for r in seg_results], default=math.inf)
        max_vel = max([finite(r.get("endpoint_velocity_miss_m_s"), math.inf) for r in seg_results], default=math.inf)
        pass_val = all(r.get("ok") for r in seg_results) and max_pos <= float(_WORKER_CFG["max_position_miss_km"]) and max_vel <= float(_WORKER_CFG["max_velocity_miss_m_s"])
        return {
            "schema_version": SCHEMA_VERSION,
            "ok": True,
            "pass_rebound_validation": bool(pass_val),
            "class": "A" if pass_val else "diagnostic",
            "high_fidelity_packet_id": packet.get("high_fidelity_packet_id"),
            "rank": packet.get("rank"),
            "sequence": packet.get("sequence"),
            "flyby_body": packet.get("flyby_body"),
            "rebound_mode": _WORKER_CFG.get("rebound_mode"),
            "integrator": "ias15",
            "max_endpoint_position_miss_km": max_pos,
            "max_endpoint_velocity_miss_m_s": max_vel,
            "segment_results": seg_results,
            "source_quality": packet.get("quality"),
        }
    except Exception as exc:
        return {"schema_version": SCHEMA_VERSION, "ok": False, "pass_rebound_validation": False, "message": repr(exc), "high_fidelity_packet_id": packet.get("high_fidelity_packet_id"), "sequence": packet.get("sequence")}


def _worker(payload: Tuple[Mapping[str, Any], Mapping[str, Any]]) -> Dict[str, Any]:
    packet, catalog = payload
    return validate_packet(packet, catalog)


def stats(vals: Sequence[float]) -> Dict[str, Optional[float]]:
    xs = sorted([v for v in vals if math.isfinite(v)])
    if not xs:
        return {"min": None, "median": None, "max": None}
    return {"min": xs[0], "median": xs[len(xs)//2], "max": xs[-1]}


def flat_row(r: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "rank": r.get("rank"),
        "high_fidelity_packet_id": r.get("high_fidelity_packet_id"),
        "sequence": r.get("sequence"),
        "ok": int(bool(r.get("ok"))),
        "pass_rebound_validation": int(bool(r.get("pass_rebound_validation"))),
        "class": r.get("class"),
        "rebound_mode": r.get("rebound_mode"),
        "max_endpoint_position_miss_km": r.get("max_endpoint_position_miss_km"),
        "max_endpoint_velocity_miss_m_s": r.get("max_endpoint_velocity_miss_m_s"),
        "message": r.get("message"),
    }


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = list(flat_row({}).keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(flat_row(r))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate high-fidelity segment packets with REBOUND IAS15.")
    p.add_argument("--input-jsonl", required=True, type=Path)
    p.add_argument("--bsp", required=True, type=Path)
    p.add_argument("--tpc", type=Path, default=None)
    p.add_argument("--body-catalog", required=True, type=Path)
    p.add_argument("--central-body", default="Sun")
    p.add_argument("--mu-central-km3-s2", required=True, type=float)
    p.add_argument("--frame", default="J2000")
    p.add_argument("--gravitating-bodies", nargs="+", required=True)
    p.add_argument("--rebound-mode", choices=["patched", "endpoint_bodies", "all_bodies"], default="patched")
    p.add_argument("--ias15-epsilon", type=float, default=1e-10)
    p.add_argument("--workers", type=int, default=1, help="0=os.cpu_count(); 1=serial")
    p.add_argument("--multiprocessing-start-method", default="spawn", choices=["spawn", "fork", "forkserver"])
    p.add_argument("--top-n", type=int, default=0)
    p.add_argument("--max-position-miss-km", type=float, default=10000.0)
    p.add_argument("--max-velocity-miss-m-s", type=float, default=100.0)
    p.add_argument("--output-csv", required=True, type=Path)
    p.add_argument("--output-jsonl", required=True, type=Path)
    p.add_argument("--output-json", required=True, type=Path)
    p.add_argument("--output-best-json", required=True, type=Path)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    packets = [p for p in load_packets(args.input_jsonl) if isinstance(p, Mapping)]
    packets = [p for p in packets if bool(p.get("ready_for_rebound_ias15"))]
    packets.sort(key=lambda p: (finite((p.get("quality") or {}).get("known_total_corrections_m_s") if isinstance(p.get("quality"), Mapping) else None, math.inf)))
    if args.top_n > 0:
        packets = packets[:args.top_n]
    catalog = load_body_catalog(args.body_catalog)
    cfg = {
        "bsp": str(args.bsp), "tpc": str(args.tpc) if args.tpc else None,
        "central_body": args.central_body, "mu_central_km3_s2": args.mu_central_km3_s2,
        "frame": args.frame, "gravitating_bodies": list(args.gravitating_bodies),
        "rebound_mode": args.rebound_mode, "ias15_epsilon": args.ias15_epsilon,
        "max_position_miss_km": args.max_position_miss_km, "max_velocity_miss_m_s": args.max_velocity_miss_m_s,
    }
    workers = args.workers if args.workers != 0 else (mp.cpu_count() or 1)
    if workers <= 1:
        _init_worker(cfg)
        results = [validate_packet(p, catalog) for p in packets]
    else:
        ctx = mp.get_context(args.multiprocessing_start_method)
        results = []
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx, initializer=_init_worker, initargs=(cfg,)) as ex:
            futs = [ex.submit(_worker, (p, catalog)) for p in packets]
            for fut in as_completed(futs):
                results.append(fut.result())
        results.sort(key=lambda r: finite(r.get("rank"), math.inf))
    write_csv(args.output_csv, results)
    write_jsonl(args.output_jsonl, results)
    pos = [finite(r.get("max_endpoint_position_miss_km")) for r in results]
    vel = [finite(r.get("max_endpoint_velocity_miss_m_s")) for r in results]
    summary = {
        "schema_version": SCHEMA_VERSION + ".summary",
        "packets_input": len(packets),
        "validated": len(results),
        "pass_rebound_validation": sum(1 for r in results if r.get("pass_rebound_validation")),
        "rebound_mode": args.rebound_mode,
        "integrator": "ias15",
        "thresholds": {"max_position_miss_km": args.max_position_miss_km, "max_velocity_miss_m_s": args.max_velocity_miss_m_s, "ias15_epsilon": args.ias15_epsilon},
        "stats": {"max_endpoint_position_miss_km": stats(pos), "max_endpoint_velocity_miss_m_s": stats(vel)},
        "top_results": [flat_row(r) for r in results[:10]],
    }
    write_json(args.output_json, summary)
    write_json(args.output_best_json, results[0] if results else {"schema_version": SCHEMA_VERSION, "ok": False, "message": "no results"})
    print("="*80)
    print("MGA REBOUND IAS15 SEGMENT VALIDATION V0.1")
    print("="*80)
    print(f"Packets input:      {len(packets)}")
    print(f"Validated:          {len(results)}")
    print(f"Pass validation:    {summary['pass_rebound_validation']}")
    print(f"Workers:            {workers}")
    print(f"REBOUND mode:       {args.rebound_mode}")
    print(f"IAS15 epsilon:      {args.ias15_epsilon}")
    print(f"Endpoint pos miss:  min={summary['stats']['max_endpoint_position_miss_km']['min']} median={summary['stats']['max_endpoint_position_miss_km']['median']} max={summary['stats']['max_endpoint_position_miss_km']['max']} km")
    print(f"Endpoint vel miss:  min={summary['stats']['max_endpoint_velocity_miss_m_s']['min']} median={summary['stats']['max_endpoint_velocity_miss_m_s']['median']} max={summary['stats']['max_endpoint_velocity_miss_m_s']['max']} m/s")
    print("\nTop results:")
    for row in summary["top_results"]:
        print(f" {row['rank']}. {row['sequence']} | pass={bool(row['pass_rebound_validation'])} | pos={finite(row['max_endpoint_position_miss_km']):.6g} km | vel={finite(row['max_endpoint_velocity_miss_m_s']):.6g} m/s | class={row['class']}")
    print("="*80)
    print(f"[OK] wrote CSV:       {args.output_csv}")
    print(f"[OK] wrote JSONL:     {args.output_jsonl}")
    print(f"[OK] wrote JSON:      {args.output_json}")
    print(f"[OK] wrote best JSON: {args.output_best_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
