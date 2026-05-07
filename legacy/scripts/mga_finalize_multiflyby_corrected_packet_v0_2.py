#!/usr/bin/env python3
"""
MGA Finalize Multi-Flyby Corrected Route Packet V0.2

Consolidates the output of mga_multiflyby_patch_corrector_v0_2.py into a
route-level manifest. V0.2 fixes schema extraction for the V0.2 corrector
(segment_corrections, all_segments_pass, total_segment_correction_m_s, and
source_packet.flybys/source_packet.metrics). This stage intentionally distinguishes:
  - positional patch closure (the 3D corrector objective), and
  - velocity continuity diagnostics (not necessarily a pass/fail criterion yet).

The script is schema-tolerant because upstream records carry nested provenance.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def load_records(path: Path) -> List[Dict[str, Any]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text[0] == "[":
        data = json.loads(text)
        return data if isinstance(data, list) else [data]
    if text[0] == "{":
        # Could be a single JSON or JSONL beginning with object.
        lines = [ln for ln in text.splitlines() if ln.strip()]
        if len(lines) == 1:
            obj = json.loads(lines[0])
            if isinstance(obj, dict) and isinstance(obj.get("packets"), list):
                return obj["packets"]
            if isinstance(obj, dict) and isinstance(obj.get("records"), list):
                return obj["records"]
            return [obj]
        return [json.loads(ln) for ln in lines]
    return [json.loads(ln) for ln in text.splitlines() if ln.strip()]


def get_path(obj: Dict[str, Any], *paths: str, default: Any = None) -> Any:
    for path in paths:
        cur: Any = obj
        ok = True
        for part in path.split("."):
            if isinstance(cur, dict) and part in cur:
                cur = cur[part]
            else:
                ok = False
                break
        if ok:
            return cur
    return default


def as_float(x: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if x is None:
            return default
        v = float(x)
        if math.isnan(v):
            return default
        return v
    except Exception:
        return default


def as_bool(x: Any) -> bool:
    if isinstance(x, bool):
        return x
    if isinstance(x, str):
        return x.lower() in {"true", "1", "yes", "y", "pass", "passed", "ok"}
    return bool(x)


def seq_string(rec: Dict[str, Any]) -> str:
    seq = get_path(rec, "sequence", "route.sequence", "source.sequence", "source_stitched_packet.sequence", default=None)
    if isinstance(seq, list):
        return " -> ".join(str(x) for x in seq)
    if isinstance(seq, str):
        return seq.replace(",", " -> ") if "," in seq and "->" not in seq else seq
    return "?"


def extract_segments(rec: Dict[str, Any]) -> List[Dict[str, Any]]:
    # Canonical output from mga_multiflyby_patch_corrector_v0_2.py
    # is segment_corrections. Older/sibling tools used several aliases.
    segs = get_path(
        rec,
        "segment_corrections",
        "segments",
        "segment_results",
        "corrected_segments",
        "source_record.segment_corrections",
        default=None,
    )
    if isinstance(segs, list):
        return segs
    # Some records store under corrections/patches.
    for key in ("corrections", "patches", "legs"):
        val = rec.get(key)
        if isinstance(val, list):
            return val
    return []


def max_from_segments(segs: List[Dict[str, Any]], keys: Iterable[str]) -> Optional[float]:
    vals: List[float] = []
    for s in segs:
        for k in keys:
            v = get_path(s, k, default=None)
            fv = as_float(v)
            if fv is not None:
                vals.append(fv)
                break
    return max(vals) if vals else None


def sum_from_segments(segs: List[Dict[str, Any]], keys: Iterable[str]) -> Optional[float]:
    vals: List[float] = []
    for s in segs:
        for k in keys:
            v = get_path(s, k, default=None)
            fv = as_float(v)
            if fv is not None:
                vals.append(fv)
                break
    return sum(vals) if vals else None


def extract_flybys(rec: Dict[str, Any]) -> List[Dict[str, Any]]:
    for path in (
        "flybys",
        "local_flybys",
        "stitched_flybys",
        "source.flybys",
        "source_packet.flybys",
        "source_record.flybys",
        "source_stitched_packet.flybys",
        "stitched_packet.flybys",
    ):
        val = get_path(rec, path, default=None)
        if isinstance(val, list):
            return val
    # Try source embedded in corrected records.
    for key in ("source", "source_record", "source_packet", "source_stitched_packet", "stitched_packet"):
        src = rec.get(key)
        if isinstance(src, dict):
            val = extract_flybys(src)
            if val:
                return val
    return []


def flyby_summary(flybys: List[Dict[str, Any]]) -> Tuple[str, Optional[float], Optional[float], Optional[float]]:
    bodies = []
    rp_margins = []
    vinf_mis = []
    alts = []
    for f in flybys:
        body = get_path(f, "body", "flyby_body", "target_body", "name", default=None)
        if body:
            bodies.append(str(body))
        rp = as_float(get_path(f, "rp_margin_km", "metrics.rp_margin_km", "periapsis.rp_margin_km", "rp_margin", default=None))
        if rp is not None:
            rp_margins.append(rp)
        vm = as_float(get_path(f, "vinf_mismatch_m_s", "metrics.vinf_mismatch_m_s", "v_inf_mismatch_m_s", "vinf_mis_m_s", default=None))
        if vm is not None:
            vinf_mis.append(vm)
        alt = as_float(get_path(f, "periapsis_altitude_km", "metrics.periapsis_altitude_km", "altitude_km", "periapsis.altitude_km", default=None))
        if alt is not None:
            alts.append(alt)
    return "|".join(bodies), (min(rp_margins) if rp_margins else None), (max(vinf_mis) if vinf_mis else None), (min(alts) if alts else None)


def classify(pass_manifest: bool, patch_dv: float, pos_miss: float, min_rp: Optional[float]) -> str:
    if not pass_manifest:
        return "D"
    if patch_dv <= 30 and pos_miss <= 10 and (min_rp is None or min_rp >= 800):
        return "A"
    if patch_dv <= 75 and pos_miss <= 100 and (min_rp is None or min_rp >= 300):
        return "B"
    if patch_dv <= 150 and pos_miss <= 1000:
        return "C"
    return "D"


def build_manifest(rec: Dict[str, Any], args: argparse.Namespace, idx: int) -> Dict[str, Any]:
    segs = extract_segments(rec)
    flybys = extract_flybys(rec)
    bodies, min_rp, max_vinf_mis, min_alt = flyby_summary(flybys)

    patch_dv = as_float(get_path(
        rec,
        "total_segment_correction_m_s",
        "total_correction_m_s",
        "total_patch_correction_m_s",
        "known_correction_m_s",
        default=None,
    ))
    if patch_dv is None:
        patch_dv = sum_from_segments(segs, ["dv_norm_m_s", "correction_m_s", "dv_m_s", "delta_v_m_s", "correction_norm_m_s"]) or 0.0

    max_pos_after = as_float(get_path(
        rec,
        "max_miss_after_km",
        "max_position_miss_after_km",
        default=None,
    ))
    if max_pos_after is None:
        max_pos_after = max_from_segments(segs, ["miss_after_km", "position_miss_after_km", "final_position_miss_km"]) or float("inf")

    max_vel_after = as_float(get_path(rec, "max_velocity_miss_after_m_s", "max_vel_after_m_s", default=None))
    if max_vel_after is None:
        max_vel_after = max_from_segments(segs, ["velocity_miss_after_m_s", "vel_miss_after_m_s", "final_velocity_miss_m_s"])

    # V0.2 corrector stores pass in all_segments_pass and status.pass_correction.
    pass_source = as_bool(get_path(
        rec,
        "all_segments_pass",
        "status.pass_correction",
        "pass_correction",
        "pass",
        "pass_manifest",
        default=False,
    ))

    # If flyby extraction failed but the embedded source metrics exist, recover route-level margins.
    if min_rp is None:
        min_rp = as_float(get_path(rec, "source_packet.metrics.min_rp_margin_km", "source_record.metrics.min_rp_margin_km", "metrics.min_rp_margin_km", default=None))
    if max_vinf_mis is None:
        max_vinf_mis = as_float(get_path(rec, "source_packet.metrics.max_vinf_mismatch_m_s", "source_record.metrics.max_vinf_mismatch_m_s", "metrics.max_vinf_mismatch_m_s", default=None))
    if min_alt is None:
        min_alt = as_float(get_path(rec, "source_packet.metrics.min_periapsis_altitude_km", "source_record.metrics.min_periapsis_altitude_km", default=None))
    pass_manifest = bool(
        pass_source
        and patch_dv <= args.max_patch_dv_m_s
        and max_pos_after <= args.max_position_miss_km
        and (min_rp is None or min_rp >= args.min_rp_margin_km)
    )

    # Optional velocity gate. Default is diagnostic only.
    velocity_gate_pass = True
    if args.max_velocity_miss_m_s is not None:
        velocity_gate_pass = max_vel_after is not None and max_vel_after <= args.max_velocity_miss_m_s
        if args.velocity_required:
            pass_manifest = pass_manifest and velocity_gate_pass

    score = patch_dv
    if min_rp is not None:
        score += max(0.0, args.rp_soft_margin_km - min_rp) / max(args.rp_soft_margin_km, 1.0) * 5.0
    if max_vinf_mis is not None:
        score += max_vinf_mis / 25.0
    if max_pos_after != float("inf"):
        score += min(max_pos_after, 1000.0) / 1000.0

    cls = classify(pass_manifest, patch_dv, max_pos_after, min_rp)
    out = {
        "packet_id": rec.get("multiflyby_patch_correction_id") or rec.get("correction_id") or rec.get("id") or f"mf_final_{idx:04d}",
        "sequence": seq_string(rec),
        "pass_manifest": pass_manifest,
        "class": cls,
        "score": score,
        "velocity_gate_required": bool(args.velocity_required),
        "velocity_gate_pass": bool(velocity_gate_pass),
        "patch_correction_m_s": patch_dv,
        "max_position_miss_after_km": max_pos_after,
        "max_velocity_miss_after_m_s": max_vel_after,
        "flyby_bodies": bodies,
        "min_rp_margin_km": min_rp,
        "min_periapsis_altitude_km": min_alt,
        "max_vinf_mismatch_m_s": max_vinf_mis,
        "segments_count": len(segs),
        "flybys_count": len(flybys),
        "schema_extraction": {
            "segments_source": "segment_corrections" if isinstance(rec.get("segment_corrections"), list) else "fallback",
            "flybys_from_source_packet": isinstance((rec.get("source_packet") or {}).get("flybys") if isinstance(rec.get("source_packet"), dict) else None, list),
        },
        "notes": [
            "Position closure is the solved constraint for the V0.2 patch corrector.",
            "Velocity mismatch is diagnostic unless --velocity-required is set.",
        ],
    }
    if args.embed_source:
        out["source_record"] = rec
    return out


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    fields = [
        "packet_id", "sequence", "pass_manifest", "class", "score",
        "patch_correction_m_s", "max_position_miss_after_km", "max_velocity_miss_after_m_s",
        "flyby_bodies", "flybys_count", "segments_count", "min_rp_margin_km",
        "min_periapsis_altitude_km", "max_vinf_mismatch_m_s", "velocity_gate_required", "velocity_gate_pass",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fields})


def percentile(vals: List[float], p: float) -> Optional[float]:
    vals = sorted(v for v in vals if v is not None and not math.isnan(v))
    if not vals:
        return None
    if len(vals) == 1:
        return vals[0]
    pos = (len(vals) - 1) * p
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return vals[lo]
    return vals[lo] * (hi - pos) + vals[hi] * (pos - lo)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input-jsonl", required=True, type=Path)
    ap.add_argument("--top-n", type=int, default=20)
    ap.add_argument("--max-patch-dv-m-s", type=float, default=75.0)
    ap.add_argument("--max-position-miss-km", type=float, default=10.0)
    ap.add_argument("--max-velocity-miss-m-s", type=float, default=None)
    ap.add_argument("--velocity-required", action="store_true")
    ap.add_argument("--min-rp-margin-km", type=float, default=300.0)
    ap.add_argument("--rp-soft-margin-km", type=float, default=1500.0)
    ap.add_argument("--embed-source", action="store_true")
    ap.add_argument("--output-csv", required=True, type=Path)
    ap.add_argument("--output-jsonl", required=True, type=Path)
    ap.add_argument("--output-json", required=True, type=Path)
    ap.add_argument("--output-best-json", required=True, type=Path)
    args = ap.parse_args()

    records = load_records(args.input_jsonl)
    manifests = [build_manifest(r, args, i) for i, r in enumerate(records)]
    manifests.sort(key=lambda x: (not x["pass_manifest"], x["score"]))
    selected = manifests[: args.top_n]

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_csv, selected)
    write_jsonl(args.output_jsonl, selected)

    pass_rows = [r for r in selected if r["pass_manifest"]]
    classes: Dict[str, int] = {}
    for r in selected:
        classes[r["class"]] = classes.get(r["class"], 0) + 1
    patch_vals = [r["patch_correction_m_s"] for r in selected]
    pos_vals = [r["max_position_miss_after_km"] for r in selected]
    vel_vals = [r["max_velocity_miss_after_m_s"] for r in selected if r.get("max_velocity_miss_after_m_s") is not None]
    summary = {
        "input_records": len(records),
        "packets_written": len(selected),
        "pass_manifest": len(pass_rows),
        "classes": classes,
        "velocity_required": bool(args.velocity_required),
        "thresholds": {
            "max_patch_dv_m_s": args.max_patch_dv_m_s,
            "max_position_miss_km": args.max_position_miss_km,
            "max_velocity_miss_m_s": args.max_velocity_miss_m_s,
            "min_rp_margin_km": args.min_rp_margin_km,
        },
        "patch_dv_m_s": {"min": percentile(patch_vals, 0), "median": percentile(patch_vals, 0.5), "max": percentile(patch_vals, 1)},
        "position_miss_km": {"min": percentile(pos_vals, 0), "median": percentile(pos_vals, 0.5), "max": percentile(pos_vals, 1)},
        "velocity_miss_m_s": {"min": percentile(vel_vals, 0), "median": percentile(vel_vals, 0.5), "max": percentile(vel_vals, 1)},
        "best_packet": selected[0] if selected else None,
    }
    args.output_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    args.output_best_json.write_text(json.dumps(selected[0] if selected else {}, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")

    print("=" * 80)
    print("MGA FINALIZE MULTI-FLYBY CORRECTED PACKET V0.2")
    print("=" * 80)
    print(f"Input records:    {len(records)}")
    print(f"Packets written:  {len(selected)}")
    print(f"Pass manifest:    {len(pass_rows)}")
    print(f"Classes:          {classes}")
    print(f"Velocity required:{bool(args.velocity_required)}")
    print(f"Patch Δv m/s:     min={summary['patch_dv_m_s']['min']} median={summary['patch_dv_m_s']['median']} max={summary['patch_dv_m_s']['max']}")
    print(f"Pos miss km:      min={summary['position_miss_km']['min']} median={summary['position_miss_km']['median']} max={summary['position_miss_km']['max']}")
    print(f"Vel miss m/s:     min={summary['velocity_miss_m_s']['min']} median={summary['velocity_miss_m_s']['median']} max={summary['velocity_miss_m_s']['max']}")
    print("\nTop candidates:")
    for i, r in enumerate(selected[:10], 1):
        print(f" {i}. {r['sequence']} | pass={r['pass_manifest']} | class={r['class']} | score={r['score']:.3f} | "
              f"dv={r['patch_correction_m_s']:.3f} m/s | pos={r['max_position_miss_after_km']:.3g} km | "
              f"vel={r.get('max_velocity_miss_after_m_s')} m/s | rpM={r.get('min_rp_margin_km')} km")
    print("=" * 80)
    print(f"[OK] wrote CSV:       {args.output_csv}")
    print(f"[OK] wrote JSONL:     {args.output_jsonl}")
    print(f"[OK] wrote JSON:      {args.output_json}")
    print(f"[OK] wrote best JSON: {args.output_best_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
