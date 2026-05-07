#!/usr/bin/env python3
"""
mga_stitch_global_local_packet_v0_1.py

Build a stitched global/local route packet from validated local flyby targets.

Input:
  - JSON/JSONL emitted by mga_local_flyby_validate_v0_1.py
  - SPICE BSP/TPC for central-body state lookup

Output:
  - CSV summary
  - JSONL stitched route packets
  - JSON summary
  - optional best packet JSON

Scope:
  This stage does not solve a new trajectory. It turns the abstract Duna flyby
  target into explicit patch states:

      heliocentric SOI-in state  = Duna state(et_in) + local entry state
      heliocentric periapsis     = Duna state(et_pe) + local periapsis state
      heliocentric SOI-out state = Duna state(et_out) + local exit state

  The local vectors are already expressed in the packet's J2000-like inertial
  basis, so the stitch is an additive translation by the body's SPICE state.

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
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

SCHEMA_VERSION = "mga_stitched_global_local_packet.v0.1"
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
    # A raw local target JSON can be accepted as already-unvalidated, but flag it.
    if data.get("schema_version") == "mga_local_flyby_target.v0.1":
        return [{"schema_version": "synthetic_unvalidated", "target": data, "validation": {"pass_validation": None}}]
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
        "body_state_central": {"r_km": r_body, "v_km_s": v_body},
        "local_body_centered": {"r_km": r_loc, "v_km_s": v_loc},
        "spacecraft_state_central": {"r_km": vadd(r_body, r_loc), "v_km_s": vadd(v_body, v_loc)},
        "local_radius_km": vnorm(r_loc),
        "local_speed_km_s": vnorm(v_loc),
    }


def flatten_packet(p: Mapping[str, Any]) -> Dict[str, Any]:
    tgt = p.get("target", {}) if isinstance(p.get("target"), Mapping) else {}
    val = p.get("local_validation", {}) if isinstance(p.get("local_validation"), Mapping) else {}
    hyp = tgt.get("hyperbola", {}) if isinstance(tgt.get("hyperbola"), Mapping) else {}
    bp = tgt.get("b_plane", {}) if isinstance(tgt.get("b_plane"), Mapping) else {}
    asym = tgt.get("asymptotes", {}) if isinstance(tgt.get("asymptotes"), Mapping) else {}
    patch = p.get("patch_epochs", {}) if isinstance(p.get("patch_epochs"), Mapping) else {}
    q = tgt.get("quality", {}) if isinstance(tgt.get("quality"), Mapping) else {}
    return {
        "stitched_packet_id": p.get("stitched_packet_id"),
        "sequence": tgt.get("sequence"),
        "flyby_body": tgt.get("body"),
        "pass_local_validation": val.get("pass_validation"),
        "class": val.get("class"),
        "entry_et": patch.get("entry_et"),
        "periapsis_et": patch.get("periapsis_et"),
        "exit_et": patch.get("exit_et"),
        "soi_to_periapsis_days": patch.get("soi_to_periapsis_days"),
        "periapsis_altitude_km": hyp.get("periapsis_altitude_km"),
        "rp_margin_km": hyp.get("rp_margin_km"),
        "vinf_effective_km_s": asym.get("vinf_effective_km_s"),
        "vinf_mismatch_m_s": asym.get("vinf_mismatch_m_s"),
        "turn_angle_deg": asym.get("turn_angle_deg"),
        "b_dot_t_km": bp.get("b_dot_t_km"),
        "b_dot_r_km": bp.get("b_dot_r_km"),
        "total_departure_correction_m_s": q.get("total_departure_correction_m_s"),
        "endpoint_position_miss_km": val.get("endpoint_position_miss_km"),
        "endpoint_velocity_miss_m_s": val.get("endpoint_velocity_miss_m_s"),
    }


def stitch_one(record: Mapping[str, Any], cfg: Mapping[str, Any]) -> Dict[str, Any]:
    target = record.get("target") if isinstance(record.get("target"), Mapping) else None
    validation = record.get("validation") if isinstance(record.get("validation"), Mapping) else {}
    if target is None:
        # Some summary top_results are already flattened and cannot be stitched.
        return {
            "schema_version": SCHEMA_VERSION,
            "ok": False,
            "message": "missing target object; provide JSONL output from mga_local_flyby_validate_v0_1.py",
            "source_record": dict(record),
        }

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
        return {
            "schema_version": SCHEMA_VERSION,
            "ok": False,
            "message": ";".join(failures),
            "target": target,
            "local_validation": validation,
        }

    entry_et = pe_et + dt_in
    exit_et = pe_et + dt_out
    try:
        entry_patch = heliocentric_patch_state(body, entry_et, entry, central, frame)
        peri_patch = heliocentric_patch_state(body, pe_et, peri, central, frame)
        exit_patch = heliocentric_patch_state(body, exit_et, exit_, central, frame)
    except Exception as exc:
        return {
            "schema_version": SCHEMA_VERSION,
            "ok": False,
            "message": repr(exc),
            "target": target,
            "local_validation": validation,
        }

    packet_id = stable_id("stitched", {
        "local_target_id": target.get("local_target_id"),
        "entry_et": round(entry_et, 6),
        "exit_et": round(exit_et, 6),
        "body": body,
    })
    return {
        "schema_version": SCHEMA_VERSION,
        "ok": True,
        "stitched_packet_id": packet_id,
        "source_schema_version": record.get("schema_version"),
        "target": target,
        "local_validation": validation,
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
        "handoff_status": {
            "ready_for_stitched_validation": bool(validation.get("pass_validation", True)),
            "recommended_next_stage": "stitched_global_local_validation_v0_1",
            "notes": [
                "Patch states are additive SPICE body state + local body-centered state.",
                "This is still a patched handoff, not a full Principia continuous trajectory.",
            ],
        },
    }


def _worker_stitch(record: Mapping[str, Any]) -> Dict[str, Any]:
    return stitch_one(record, _WORKER_CFG)


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "stitched_packet_id", "sequence", "flyby_body", "pass_local_validation", "class",
        "entry_et", "periapsis_et", "exit_et", "soi_to_periapsis_days",
        "periapsis_altitude_km", "rp_margin_km", "vinf_effective_km_s",
        "vinf_mismatch_m_s", "turn_angle_deg", "b_dot_t_km", "b_dot_r_km",
        "total_departure_correction_m_s", "endpoint_position_miss_km", "endpoint_velocity_miss_m_s",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fields})


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stitch validated local flyby target into global heliocentric patch states.")
    p.add_argument("--bsp", required=True, type=Path)
    p.add_argument("--tpc", required=True, type=Path)
    p.add_argument("--input-validation", required=True, type=Path, help="JSON/JSONL from mga_local_flyby_validate_v0_1.py")
    p.add_argument("--central-body", default="Sun")
    p.add_argument("--frame", default="J2000")
    p.add_argument("--top-n", type=int, default=0)
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
    # Keep best local validations first.
    records.sort(key=lambda r: (
        not bool(((r.get("validation") or {}).get("pass_validation"))),
        finite(((r.get("validation") or {}).get("endpoint_position_miss_km")), 1e99),
        finite((((r.get("target") or {}).get("quality") or {}).get("source_score")), 1e99),
    ))
    if args.top_n > 0:
        records = records[:args.top_n]

    cfg = {
        "bsp": str(args.bsp),
        "tpc": str(args.tpc),
        "central_body": args.central_body,
        "frame": args.frame,
    }
    workers = os.cpu_count() or 1 if args.workers == 0 else max(1, args.workers)

    packets: List[Dict[str, Any]] = []
    if workers == 1 or len(records) <= 1:
        _init_worker(cfg)
        for rec in records:
            packets.append(stitch_one(rec, cfg))
    else:
        ctx = mp.get_context(args.multiprocessing_start_method)
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx, initializer=_init_worker, initargs=(cfg,)) as ex:
            futs = [ex.submit(_worker_stitch, rec) for rec in records]
            for fut in as_completed(futs):
                packets.append(fut.result())

    packets.sort(key=lambda p: (
        not bool(p.get("ok")),
        not bool(((p.get("local_validation") or {}).get("pass_validation"))),
        finite(((p.get("local_validation") or {}).get("endpoint_position_miss_km")), 1e99),
        finite((((p.get("target") or {}).get("quality") or {}).get("source_score")), 1e99),
    ))
    flat = [flatten_packet(p) for p in packets if p.get("ok")]
    ok_count = sum(1 for p in packets if p.get("ok"))
    ready_count = sum(1 for p in packets if p.get("ok") and ((p.get("handoff_status") or {}).get("ready_for_stitched_validation")))

    write_csv(args.output_csv, flat)
    write_jsonl(args.output_jsonl, packets)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "input_validation": str(args.input_validation),
        "bsp": str(args.bsp),
        "tpc": str(args.tpc),
        "records_input": len(records),
        "packets_written": len(packets),
        "packets_ok": ok_count,
        "ready_for_stitched_validation": ready_count,
        "workers": workers,
        "top_packets": flat[:20],
    }
    write_json(args.output_json, summary)
    write_json(args.output_best_json, packets[0] if packets else {})

    print("="*80)
    print("MGA STITCH GLOBAL-LOCAL PACKET V0.1")
    print("="*80)
    print(f"Validation records: {len(records)}")
    print(f"Packets written:    {len(packets)}")
    print(f"Packets OK:         {ok_count}")
    print(f"Ready next stage:   {ready_count}")
    print(f"Workers:            {workers}")
    print("\nTop stitched packets:")
    for i, r in enumerate(flat[:10], start=1):
        print(
            f" {i}. {r.get('sequence')} @ {r.get('flyby_body')} | "
            f"alt={finite(r.get('periapsis_altitude_km')):.1f} km | "
            f"rp_margin={finite(r.get('rp_margin_km')):.1f} km | "
            f"SOI→Pe={finite(r.get('soi_to_periapsis_days')):.3f} d | "
            f"vinf_mis={finite(r.get('vinf_mismatch_m_s')):.3f} m/s | "
            f"corr={finite(r.get('total_departure_correction_m_s')):.3f} m/s"
        )
    print("="*80)
    print(f"[OK] wrote CSV:       {args.output_csv}")
    print(f"[OK] wrote JSONL:     {args.output_jsonl}")
    print(f"[OK] wrote JSON:      {args.output_json}")
    print(f"[OK] wrote best JSON: {args.output_best_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
