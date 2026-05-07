#!/usr/bin/env python3
"""
mga_stitch_multiflyby_packet_v0_1.py

Build route-level stitched global/local packets from local flyby validations.

This is the multi-flyby counterpart of mga_stitch_global_local_packet_v0_1.py.
It groups local flyby validation records by target_spec/packet/route, then stitches
all validated local hyperbolic flybys back into central-body coordinates:

    central state at SOI-in  = SPICE(body, et_in)  + local entry state
    central state at Pe      = SPICE(body, et_pe)  + local periapsis state
    central state at SOI-out = SPICE(body, et_out) + local exit state

Inputs:
  - JSONL/JSON from mga_local_flyby_validate_v0_1.py
  - SPICE BSP/TPC

Outputs:
  - route-level JSONL stitched packets
  - CSV summary
  - JSON summary
  - best packet JSON

Scope:
  This stage is a handoff/stitch artifact only. It does not solve a new
  trajectory and does not apply patch-point corrections. The next stage should
  validate/correct the stitched route against patched heliocentric dynamics.

Units:
  km, km/s, ET seconds.
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
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

SCHEMA_VERSION = "mga_stitched_multiflyby_packet.v0.1"
Vec3 = Tuple[float, float, float]
_WORKER_CFG: Dict[str, Any] = {}
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


def load_validation_records(path: Path) -> List[Dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        return read_jsonl(path)
    data = load_json(path)
    if "target" in data and "validation" in data:
        return [data]
    for key in ("records", "results", "validations", "top_results"):
        rows = data.get(key)
        if isinstance(rows, list):
            return [dict(r) for r in rows if isinstance(r, Mapping)]
    raise ValueError(f"Could not find local flyby validation records in {path}")


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


def _init_worker(cfg: Mapping[str, Any]) -> None:
    global _WORKER_CFG, _WORKER_SPICE
    _WORKER_CFG = dict(cfg)
    import spiceypy as spice  # type: ignore
    _WORKER_SPICE = spice
    spice.kclear()
    for kernel in (cfg.get("tpc"), cfg.get("bsp")):
        if kernel:
            spice.furnsh(str(kernel))


def _spice() -> Any:
    global _WORKER_SPICE
    if _WORKER_SPICE is None:
        _init_worker(_WORKER_CFG)
    return _WORKER_SPICE


def state_body_to_central(body: str, et: float, central: str, frame: str) -> Tuple[Vec3, Vec3]:
    spice = _spice()
    st, _lt = spice.spkezr(str(body), float(et), str(frame), "NONE", str(central))
    return (float(st[0]), float(st[1]), float(st[2])), (float(st[3]), float(st[4]), float(st[5]))


def heliocentric_patch_state(body: str, et: float, local_state: Mapping[str, Any], central: str, frame: str) -> Dict[str, Any]:
    r_loc = vec3(local_state.get("r_km"))
    v_loc = vec3(local_state.get("v_km_s"))
    if r_loc is None or v_loc is None:
        raise ValueError("missing local r/v state")
    r_body, v_body = state_body_to_central(body, et, central, frame)
    return {
        "et": et,
        "body": body,
        "body_state_central": {"r_km": r_body, "v_km_s": v_body},
        "local_body_centered": {"r_km": r_loc, "v_km_s": v_loc},
        "spacecraft_state_central": {"r_km": vadd(r_body, r_loc), "v_km_s": vadd(v_body, v_loc)},
        "local_radius_km": vnorm(r_loc),
        "local_speed_km_s": vnorm(v_loc),
    }


def group_key(record: Mapping[str, Any]) -> Tuple[str, str, str, str]:
    target = record.get("target") if isinstance(record.get("target"), Mapping) else {}
    return (
        str(target.get("target_spec_id") or ""),
        str(target.get("packet_id") or ""),
        str(target.get("route_id") or ""),
        str(target.get("sequence") or ""),
    )


def load_groups(records: Sequence[Mapping[str, Any]]) -> List[List[Dict[str, Any]]]:
    groups: Dict[Tuple[str, str, str, str], List[Dict[str, Any]]] = {}
    for rec in records:
        if not isinstance(rec.get("target"), Mapping):
            continue
        k = group_key(rec)
        groups.setdefault(k, []).append(dict(rec))
    out = list(groups.values())
    # Best groups first: all pass, low source score, low max local error, high margins.
    def key(g: List[Dict[str, Any]]) -> Tuple[Any, ...]:
        all_pass = all(bool((r.get("validation") or {}).get("pass_validation")) for r in g)
        source_score = min(finite(((r.get("target") or {}).get("quality") or {}).get("source_score"), 1e99) for r in g)
        max_pos = max(finite((r.get("validation") or {}).get("endpoint_position_miss_km"), 1e99) for r in g)
        min_margin = min(finite(((r.get("target") or {}).get("hyperbola") or {}).get("rp_margin_km"), -1e99) for r in g)
        return (not all_pass, source_score, max_pos, -min_margin)
    out.sort(key=key)
    return out


def stitch_flyby(record: Mapping[str, Any], cfg: Mapping[str, Any]) -> Dict[str, Any]:
    target = record.get("target") if isinstance(record.get("target"), Mapping) else None
    validation = record.get("validation") if isinstance(record.get("validation"), Mapping) else {}
    if target is None:
        return {"ok": False, "message": "missing target"}

    body = str(target.get("body"))
    central = str(cfg.get("central_body") or target.get("central_body") or "Sun")
    frame = str(cfg.get("frame") or target.get("frame") or "J2000")
    pe_et = finite(target.get("nominal_periapsis_et", target.get("encounter_et")))
    soi = target.get("soi_patch_estimate") if isinstance(target.get("soi_patch_estimate"), Mapping) else {}
    entry = soi.get("entry_state_body_centered") if isinstance(soi.get("entry_state_body_centered"), Mapping) else {}
    exit_ = soi.get("exit_state_body_centered") if isinstance(soi.get("exit_state_body_centered"), Mapping) else {}
    peri = target.get("periapsis_state_body_centered") if isinstance(target.get("periapsis_state_body_centered"), Mapping) else {}
    dt_in = finite(entry.get("dt_from_periapsis_s"))
    dt_out = finite(exit_.get("dt_from_periapsis_s"))

    failures: List[str] = []
    if not math.isfinite(pe_et): failures.append("missing_periapsis_et")
    if not math.isfinite(dt_in): failures.append("missing_entry_dt")
    if not math.isfinite(dt_out): failures.append("missing_exit_dt")
    if not entry: failures.append("missing_entry_state")
    if not exit_: failures.append("missing_exit_state")
    if not peri: failures.append("missing_periapsis_state")
    if failures:
        return {"ok": False, "body": body, "message": ";".join(failures), "target": target, "local_validation": validation}

    entry_et = pe_et + dt_in
    exit_et = pe_et + dt_out
    try:
        entry_patch = heliocentric_patch_state(body, entry_et, entry, central, frame)
        peri_patch = heliocentric_patch_state(body, pe_et, peri, central, frame)
        exit_patch = heliocentric_patch_state(body, exit_et, exit_, central, frame)
    except Exception as exc:
        return {"ok": False, "body": body, "message": repr(exc), "target": target, "local_validation": validation}

    hyp = target.get("hyperbola") if isinstance(target.get("hyperbola"), Mapping) else {}
    asym = target.get("asymptotes") if isinstance(target.get("asymptotes"), Mapping) else {}
    bp = target.get("b_plane") if isinstance(target.get("b_plane"), Mapping) else {}
    qual = target.get("quality") if isinstance(target.get("quality"), Mapping) else {}
    flyby_id = stable_id("stitchedflyby", {
        "local_target_id": target.get("local_target_id"),
        "body": body,
        "pe_et": round(pe_et, 6),
        "entry_et": round(entry_et, 6),
        "exit_et": round(exit_et, 6),
    })
    return {
        "ok": True,
        "stitched_flyby_id": flyby_id,
        "local_target_id": target.get("local_target_id"),
        "target_spec_id": target.get("target_spec_id"),
        "packet_id": target.get("packet_id"),
        "route_id": target.get("route_id"),
        "sequence": target.get("sequence"),
        "body": body,
        "central_body": central,
        "frame": frame,
        "local_validation": validation,
        "target": target,
        "patch_epochs": {
            "entry_et": entry_et,
            "periapsis_et": pe_et,
            "exit_et": exit_et,
            "entry_dt_from_periapsis_s": dt_in,
            "exit_dt_from_periapsis_s": dt_out,
            "soi_to_periapsis_s": abs(dt_in),
            "soi_to_periapsis_days": abs(dt_in)/86400.0,
        },
        "patch_states": {
            "entry_soi": entry_patch,
            "periapsis": peri_patch,
            "exit_soi": exit_patch,
        },
        "metrics": {
            "rp_margin_km": opt_float(hyp.get("rp_margin_km")),
            "periapsis_altitude_km": opt_float(hyp.get("periapsis_altitude_km")),
            "vinf_effective_km_s": opt_float(asym.get("vinf_effective_km_s")),
            "vinf_mismatch_m_s": opt_float(asym.get("vinf_mismatch_m_s")),
            "turn_angle_deg": opt_float(asym.get("turn_angle_deg")),
            "b_dot_t_km": opt_float(bp.get("b_dot_t_km")),
            "b_dot_r_km": opt_float(bp.get("b_dot_r_km")),
            "total_departure_correction_m_s": opt_float(qual.get("total_departure_correction_m_s")),
            "endpoint_position_miss_km": opt_float(validation.get("endpoint_position_miss_km")),
            "endpoint_velocity_miss_m_s": opt_float(validation.get("endpoint_velocity_miss_m_s")),
        },
    }


def stitch_group(records: Sequence[Mapping[str, Any]], cfg: Mapping[str, Any]) -> Dict[str, Any]:
    flybys = [stitch_flyby(r, cfg) for r in records]
    flybys.sort(key=lambda f: finite(((f.get("patch_epochs") or {}).get("periapsis_et")), 1e99))
    ok_flybys = [f for f in flybys if f.get("ok")]
    failures: List[str] = []
    min_flybys = int(cfg.get("min_flybys") or 1)
    if len(ok_flybys) < min_flybys:
        failures.append(f"insufficient_ok_flybys:{len(ok_flybys)}<{min_flybys}")
    for f in flybys:
        if not f.get("ok"):
            failures.append(f"{f.get('body','?')}:{f.get('message','failed')}")
    if cfg.get("require_all_local_pass"):
        for f in ok_flybys:
            if not bool((f.get("local_validation") or {}).get("pass_validation")):
                failures.append(f"{f.get('body')}:local_validation_failed")

    seq = None
    target_spec_id = None
    packet_id = None
    route_id = None
    for f in ok_flybys:
        seq = seq or f.get("sequence")
        target_spec_id = target_spec_id or f.get("target_spec_id")
        packet_id = packet_id or f.get("packet_id")
        route_id = route_id or f.get("route_id")

    min_margin = min([finite((f.get("metrics") or {}).get("rp_margin_km"), math.inf) for f in ok_flybys] or [math.nan])
    max_vinf_mis = max([finite((f.get("metrics") or {}).get("vinf_mismatch_m_s"), -math.inf) for f in ok_flybys] or [math.nan])
    max_turn = max([finite((f.get("metrics") or {}).get("turn_angle_deg"), -math.inf) for f in ok_flybys] or [math.nan])
    total_corr = max([finite((f.get("metrics") or {}).get("total_departure_correction_m_s"), math.nan) for f in ok_flybys] or [math.nan])
    max_local_pos = max([finite((f.get("metrics") or {}).get("endpoint_position_miss_km"), -math.inf) for f in ok_flybys] or [math.nan])
    max_local_vel = max([finite((f.get("metrics") or {}).get("endpoint_velocity_miss_m_s"), -math.inf) for f in ok_flybys] or [math.nan])

    timeline: List[Dict[str, Any]] = []
    for idx, f in enumerate(ok_flybys):
        epochs = f.get("patch_epochs") if isinstance(f.get("patch_epochs"), Mapping) else {}
        states = f.get("patch_states") if isinstance(f.get("patch_states"), Mapping) else {}
        body = f.get("body")
        for event_name, key in (("soi_in", "entry_soi"), ("periapsis", "periapsis"), ("soi_out", "exit_soi")):
            st = states.get(key) if isinstance(states.get(key), Mapping) else {}
            timeline.append({
                "event": f"{body}_{event_name}",
                "flyby_index": idx,
                "body": body,
                "et": st.get("et"),
                "state_kind": key,
                "spacecraft_state_central": st.get("spacecraft_state_central"),
                "body_state_central": st.get("body_state_central"),
                "local_body_centered": st.get("local_body_centered"),
            })
    timeline.sort(key=lambda x: finite(x.get("et"), 1e99))

    packet_id_out = stable_id("multistitch", {
        "target_spec_id": target_spec_id,
        "packet_id": packet_id,
        "route_id": route_id,
        "sequence": seq,
        "bodies": [f.get("body") for f in ok_flybys],
        "epochs": [round(finite(((f.get("patch_epochs") or {}).get("periapsis_et")), 0.0), 6) for f in ok_flybys],
    })
    return {
        "schema_version": SCHEMA_VERSION,
        "ok": len(failures) == 0,
        "stitched_multiflyby_packet_id": packet_id_out,
        "target_spec_id": target_spec_id,
        "packet_id": packet_id,
        "route_id": route_id,
        "sequence": seq,
        "flyby_bodies": [f.get("body") for f in ok_flybys],
        "num_flybys": len(ok_flybys),
        "flybys": ok_flybys,
        "timeline": timeline,
        "metrics": {
            "min_rp_margin_km": opt_float(min_margin),
            "max_vinf_mismatch_m_s": opt_float(max_vinf_mis),
            "max_turn_angle_deg": opt_float(max_turn),
            "total_departure_correction_m_s": opt_float(total_corr),
            "max_local_endpoint_position_miss_km": opt_float(max_local_pos),
            "max_local_endpoint_velocity_miss_m_s": opt_float(max_local_vel),
        },
        "handoff_status": {
            "ready_for_multiflyby_stitched_validation": len(failures) == 0,
            "recommended_next_stage": "stitched_multiflyby_validation_or_patch_correction_v0_1",
            "failures": failures,
            "notes": [
                "Each flyby patch state is SPICE body state + validated local two-body hyperbola state.",
                "This packet contains multiple local flybys for one route-level candidate.",
                "This is still a patched handoff, not a continuous Principia trajectory.",
            ],
        },
    }


def _worker_group(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    return stitch_group(records, _WORKER_CFG)


def flatten_packet(p: Mapping[str, Any]) -> Dict[str, Any]:
    metrics = p.get("metrics") if isinstance(p.get("metrics"), Mapping) else {}
    status = p.get("handoff_status") if isinstance(p.get("handoff_status"), Mapping) else {}
    return {
        "stitched_multiflyby_packet_id": p.get("stitched_multiflyby_packet_id"),
        "ok": p.get("ok"),
        "ready": status.get("ready_for_multiflyby_stitched_validation"),
        "sequence": p.get("sequence"),
        "flyby_bodies": "|".join(str(x) for x in (p.get("flyby_bodies") or [])),
        "num_flybys": p.get("num_flybys"),
        "min_rp_margin_km": metrics.get("min_rp_margin_km"),
        "max_vinf_mismatch_m_s": metrics.get("max_vinf_mismatch_m_s"),
        "max_turn_angle_deg": metrics.get("max_turn_angle_deg"),
        "total_departure_correction_m_s": metrics.get("total_departure_correction_m_s"),
        "max_local_endpoint_position_miss_km": metrics.get("max_local_endpoint_position_miss_km"),
        "max_local_endpoint_velocity_miss_m_s": metrics.get("max_local_endpoint_velocity_miss_m_s"),
        "failures": ";".join(str(x) for x in (status.get("failures") or [])),
    }


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "stitched_multiflyby_packet_id", "ok", "ready", "sequence", "flyby_bodies", "num_flybys",
        "min_rp_margin_km", "max_vinf_mismatch_m_s", "max_turn_angle_deg",
        "total_departure_correction_m_s", "max_local_endpoint_position_miss_km",
        "max_local_endpoint_velocity_miss_m_s", "failures",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fields})


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stitch validated local flybys into route-level multi-flyby packets.")
    p.add_argument("--bsp", required=True, type=Path)
    p.add_argument("--tpc", required=True, type=Path)
    p.add_argument("--input-validation", required=True, type=Path)
    p.add_argument("--central-body", default="Sun")
    p.add_argument("--frame", default="J2000")
    p.add_argument("--min-flybys", type=int, default=2)
    p.add_argument("--require-all-local-pass", action="store_true", default=True)
    p.add_argument("--top-n-groups", type=int, default=0)
    p.add_argument("--workers", type=int, default=1, help="0=os.cpu_count(); 1=serial")
    p.add_argument("--multiprocessing-start-method", default="spawn", choices=["spawn", "fork", "forkserver"])
    p.add_argument("--output-csv", required=True, type=Path)
    p.add_argument("--output-jsonl", required=True, type=Path)
    p.add_argument("--output-json", required=True, type=Path)
    p.add_argument("--output-best-json", required=True, type=Path)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    records = load_validation_records(args.input_validation)
    groups = load_groups(records)
    if args.top_n_groups > 0:
        groups = groups[:args.top_n_groups]

    cfg = {
        "bsp": str(args.bsp),
        "tpc": str(args.tpc),
        "central_body": args.central_body,
        "frame": args.frame,
        "min_flybys": args.min_flybys,
        "require_all_local_pass": bool(args.require_all_local_pass),
    }
    workers = os.cpu_count() or 1 if args.workers == 0 else max(1, args.workers)
    packets: List[Dict[str, Any]] = []
    if workers == 1 or len(groups) <= 1:
        _init_worker(cfg)
        for g in groups:
            packets.append(stitch_group(g, cfg))
    else:
        ctx = mp.get_context(args.multiprocessing_start_method)
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx, initializer=_init_worker, initargs=(cfg,)) as ex:
            futs = [ex.submit(_worker_group, g) for g in groups]
            for fut in as_completed(futs):
                packets.append(fut.result())

    packets.sort(key=lambda p: (
        not bool(p.get("ok")),
        finite(((p.get("metrics") or {}).get("total_departure_correction_m_s")), 1e99),
        finite(((p.get("metrics") or {}).get("max_vinf_mismatch_m_s")), 1e99),
        -finite(((p.get("metrics") or {}).get("min_rp_margin_km")), -1e99),
    ))
    flat = [flatten_packet(p) for p in packets]
    ok_count = sum(1 for p in packets if p.get("ok"))
    ready_count = sum(1 for p in packets if ((p.get("handoff_status") or {}).get("ready_for_multiflyby_stitched_validation")))
    bodies_counts: Dict[str, int] = {}
    for p in packets:
        for b in p.get("flyby_bodies") or []:
            bodies_counts[str(b)] = bodies_counts.get(str(b), 0) + 1

    write_csv(args.output_csv, flat)
    write_jsonl(args.output_jsonl, packets)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "input_validation": str(args.input_validation),
        "bsp": str(args.bsp),
        "tpc": str(args.tpc),
        "validation_records_input": len(records),
        "groups_input": len(groups),
        "packets_written": len(packets),
        "packets_ok": ok_count,
        "ready_for_multiflyby_stitched_validation": ready_count,
        "flyby_body_counts": bodies_counts,
        "workers": workers,
        "top_packets": flat[:20],
    }
    write_json(args.output_json, summary)
    write_json(args.output_best_json, packets[0] if packets else {})

    print("="*80)
    print("MGA STITCH MULTI-FLYBY PACKET V0.1")
    print("="*80)
    print(f"Validation records: {len(records)}")
    print(f"Route groups:        {len(groups)}")
    print(f"Packets written:     {len(packets)}")
    print(f"Packets OK:          {ok_count}")
    print(f"Ready next stage:    {ready_count}")
    print(f"Workers:             {workers}")
    print(f"Flyby body counts:   {bodies_counts}")
    print("\nTop stitched multi-flyby packets:")
    for i, r in enumerate(flat[:10], start=1):
        print(
            f" {i}. {r.get('sequence')} | bodies={r.get('flyby_bodies')} | "
            f"ok={r.get('ok')} | min rpM={finite(r.get('min_rp_margin_km')):.1f} km | "
            f"max v∞mis={finite(r.get('max_vinf_mismatch_m_s')):.2f} m/s | "
            f"corr={finite(r.get('total_departure_correction_m_s')):.3f} m/s | "
            f"failures={r.get('failures') or ''}"
        )
    print("="*80)
    print(f"[OK] wrote CSV:       {args.output_csv}")
    print(f"[OK] wrote JSONL:     {args.output_jsonl}")
    print(f"[OK] wrote JSON:      {args.output_json}")
    print(f"[OK] wrote best JSON: {args.output_best_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
