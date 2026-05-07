#!/usr/bin/env python3
"""
MGA Export B6D Route Packet V0.1

Consolidates a selected 6D-viable multi-flyby route into a final handoff packet.
This script is intentionally schema-tolerant because earlier pipeline stages evolved
field names while preserving the physical content.

Inputs:
  - selected-jsonl: output of mga_select_6d_viable_multiflyby_v0_1.py
  - corrected-jsonl: output of mga_multiflyby_patch_corrector_v0_2.py
  - diagnostics-jsonl: output of mga_multiflyby_6d_patch_diagnostics_v0_3.py

Outputs:
  - CSV summary
  - JSONL final packets
  - JSON summary
  - best JSON packet
  - optional Markdown report

Important semantics:
  - B6D means intermediate patch-points are 6D-consistent within thresholds.
  - Final target arrival velocity mismatch is expected v_infinity, not failure.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def finite_or_none(x: Any) -> Any:
    if isinstance(x, float):
        return x if math.isfinite(x) else None
    if isinstance(x, list):
        return [finite_or_none(v) for v in x]
    if isinstance(x, tuple):
        return [finite_or_none(v) for v in x]
    if isinstance(x, dict):
        return {k: finite_or_none(v) for k, v in x.items()}
    return x


def read_json_or_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    p = Path(path)
    text = p.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if p.suffix.lower() == ".jsonl":
        out: List[Dict[str, Any]] = []
        for line in text.splitlines():
            line = line.strip()
            if line:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    out.append(obj)
        return out
    obj = json.loads(text)
    if isinstance(obj, list):
        return [x for x in obj if isinstance(x, dict)]
    if isinstance(obj, dict):
        # Common wrappers
        for key in ("packets", "records", "routes", "selected", "items", "results"):
            val = obj.get(key)
            if isinstance(val, list):
                return [x for x in val if isinstance(x, dict)]
        return [obj]
    return []


def write_json(path: str | Path, obj: Any) -> None:
    Path(path).write_text(json.dumps(finite_or_none(obj), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def write_jsonl(path: str | Path, rows: Iterable[Dict[str, Any]]) -> None:
    with Path(path).open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(finite_or_none(r), ensure_ascii=False, separators=(",", ":"), allow_nan=False))
            f.write("\n")


def deep_find(obj: Any, keys: Tuple[str, ...]) -> Optional[Any]:
    if isinstance(obj, dict):
        for k in keys:
            if k in obj and obj[k] not in (None, ""):
                return obj[k]
        for v in obj.values():
            found = deep_find(v, keys)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for v in obj:
            found = deep_find(v, keys)
            if found is not None:
                return found
    return None


def deep_collect(obj: Any, keys: Tuple[str, ...]) -> List[Any]:
    out: List[Any] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in keys and v not in (None, ""):
                out.append(v)
        for v in obj.values():
            out.extend(deep_collect(v, keys))
    elif isinstance(obj, list):
        for v in obj:
            out.extend(deep_collect(v, keys))
    return out


def as_float(x: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if x is None:
            return default
        f = float(x)
        return f if math.isfinite(f) else default
    except Exception:
        return default


def as_int(x: Any, default: Optional[int] = None) -> Optional[int]:
    try:
        if x is None:
            return default
        return int(float(x))
    except Exception:
        return default


def normalize_sequence(seq: Any) -> List[str]:
    if isinstance(seq, list):
        return [str(x) for x in seq]
    if isinstance(seq, str):
        if "->" in seq:
            return [x.strip() for x in seq.split("->") if x.strip()]
        return [x.strip() for x in seq.split(",") if x.strip()]
    return []


def route_number(obj: Dict[str, Any]) -> Optional[int]:
    # Prefer explicit route fields.
    for key in ("route_number", "route_index", "route_rank", "route", "rank", "source_route_index", "source_route_number"):
        val = deep_find(obj, (key,))
        n = as_int(val)
        if n is not None:
            return n
    return None


def route_id(obj: Dict[str, Any]) -> Optional[str]:
    val = deep_find(obj, ("route_id", "packet_id", "correction_id", "id"))
    return str(val) if val not in (None, "") else None


def sequence_of(obj: Dict[str, Any]) -> List[str]:
    val = deep_find(obj, ("sequence", "bodies", "route_sequence"))
    seq = normalize_sequence(val)
    if seq:
        return seq
    return []


def min_float(values: Iterable[Any]) -> Optional[float]:
    vals = [as_float(v) for v in values]
    vals = [v for v in vals if v is not None]
    return min(vals) if vals else None


def max_float(values: Iterable[Any]) -> Optional[float]:
    vals = [as_float(v) for v in values]
    vals = [v for v in vals if v is not None]
    return max(vals) if vals else None


def get_total_dv(obj: Dict[str, Any]) -> Optional[float]:
    # Try exact known fields first.
    val = deep_find(obj, (
        "total_segment_correction_m_s",
        "total_patch_correction_m_s",
        "total_correction_m_s",
        "known_correction_m_s",
        "patch_dv_m_s",
        "total_dv_m_s",
    ))
    f = as_float(val)
    if f is not None:
        return f
    # Fall back to summing segment corrections.
    vals = deep_collect(obj, ("correction_m_s", "delta_v_m_s", "dv_m_s", "segment_correction_m_s"))
    nums = [as_float(v) for v in vals]
    nums = [v for v in nums if v is not None]
    if nums:
        return sum(nums)
    return None


def get_max_position_miss(obj: Dict[str, Any]) -> Optional[float]:
    val = deep_find(obj, ("max_position_miss_km", "max_miss_after_km", "max_pos_miss_km", "position_miss_km"))
    f = as_float(val)
    if f is not None:
        return f
    vals = deep_collect(obj, ("miss_after_km", "position_miss_after_km", "position_miss_km", "pos_miss_km"))
    return max_float(vals)


def get_max_velocity_miss(obj: Dict[str, Any]) -> Optional[float]:
    val = deep_find(obj, ("max_velocity_miss_m_s", "max_vel_miss_m_s", "velocity_miss_m_s"))
    f = as_float(val)
    if f is not None:
        return f
    vals = deep_collect(obj, ("velocity_miss_m_s", "vel_miss_m_s", "endpoint_velocity_miss_m_s"))
    return max_float(vals)


def get_min_rp_margin(obj: Dict[str, Any]) -> Optional[float]:
    val = deep_find(obj, ("min_rp_margin_km", "rp_margin_min_km", "min_rpM_km", "rpM"))
    f = as_float(val)
    if f is not None:
        return f
    vals = deep_collect(obj, ("rp_margin_km", "rpM", "rp_margin"))
    return min_float(vals)


def get_segments(obj: Dict[str, Any]) -> List[Dict[str, Any]]:
    for key in ("segment_corrections", "segments", "corrected_segments"):
        val = deep_find(obj, (key,))
        if isinstance(val, list):
            return [x for x in val if isinstance(x, dict)]
    return []


def get_flybys(obj: Dict[str, Any]) -> List[Dict[str, Any]]:
    for key in ("flybys", "local_flybys", "flyby_targets"):
        val = deep_find(obj, (key,))
        if isinstance(val, list):
            return [x for x in val if isinstance(x, dict)]
    return []


def diag_for_route(diags: List[Dict[str, Any]], n: int) -> List[Dict[str, Any]]:
    rows = []
    for d in diags:
        dn = route_number(d)
        if dn == n:
            rows.append(d)
    return rows


def diagnosis_counts(rows: List[Dict[str, Any]]) -> Dict[str, int]:
    c: Dict[str, int] = {}
    for r in rows:
        d = deep_find(r, ("diagnosis", "class", "status"))
        if d is None:
            d = "unknown"
        d = str(d)
        c[d] = c.get(d, 0) + 1
    return c


def intermediate_velocity_max(rows: List[Dict[str, Any]]) -> Optional[float]:
    vals = []
    for r in rows:
        diagnosis = str(deep_find(r, ("diagnosis", "class", "status")) or "")
        seg_kind = str(deep_find(r, ("segment_kind", "kind", "segment_type")) or "")
        target = str(deep_find(r, ("target", "to", "end_label", "to_label")) or "")
        # Skip final arrival. Use diagnosis string when available.
        if "final_arrival" in diagnosis or "Jool_center" in target or seg_kind == "last_flyby_exit_to_final_target":
            continue
        v = as_float(deep_find(r, ("velocity_miss_central_m_s", "velocity_miss_m_s", "v_c_m_s", "v_c")))
        # Prefer local mismatch if present and finite.
        vl = as_float(deep_find(r, ("velocity_miss_local_m_s", "v_loc_m_s", "v_loc")))
        if vl is not None:
            v = vl
        if v is not None:
            vals.append(abs(v))
    return max(vals) if vals else None


def infer_class(total_dv: Optional[float], max_pos: Optional[float], min_rp: Optional[float], v_int: Optional[float], args: argparse.Namespace) -> str:
    if total_dv is None or max_pos is None or min_rp is None:
        return "D"
    if max_pos > args.max_position_miss_km or total_dv > args.max_patch_dv_m_s or min_rp < args.min_rp_margin_km:
        return "D"
    if v_int is not None and v_int > args.max_intermediate_velocity_m_s:
        return "C6D"
    if total_dv <= 30 and min_rp >= args.rp_a_km and (v_int is None or v_int <= 25):
        return "A6D"
    if total_dv <= 75 and min_rp >= args.min_rp_margin_km:
        return "B6D"
    return "C6D"


def score_packet(total_dv: Optional[float], min_rp: Optional[float], v_int: Optional[float], args: argparse.Namespace) -> float:
    s = float(total_dv or 0.0)
    if min_rp is not None:
        if min_rp < args.rp_soft_km:
            s += (args.rp_soft_km - min_rp) / max(args.rp_soft_km, 1.0) * 10.0
    else:
        s += 100.0
    if v_int is not None:
        s += min(v_int / 100.0, 500.0)
    return s


def write_csv(path: str | Path, rows: List[Dict[str, Any]]) -> None:
    fields = [
        "route_number", "route_id", "sequence", "route_class", "pass_manifest",
        "score", "total_patch_dv_m_s", "max_position_miss_km", "max_velocity_miss_m_s",
        "max_intermediate_velocity_m_s", "min_rp_margin_km", "segments", "flybys",
        "diagnosis_counts",
    ]
    with Path(path).open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fields})


def write_md(path: str | Path, rows: List[Dict[str, Any]], summary: Dict[str, Any]) -> None:
    lines = []
    lines.append("# MGA B6D Route Packet V0.1")
    lines.append("")
    lines.append(f"Packets exported: {summary.get('packets_written')}")
    lines.append(f"Pass manifest: {summary.get('pass_manifest')}")
    lines.append(f"Classes: {summary.get('classes')}")
    lines.append("")
    if rows:
        b = rows[0]
        lines.append("## Best packet")
        lines.append("")
        lines.append(f"- Sequence: `{b.get('sequence')}`")
        lines.append(f"- Class: `{b.get('route_class')}`")
        lines.append(f"- Score: `{b.get('score')}`")
        lines.append(f"- Patch Δv: `{b.get('total_patch_dv_m_s')}` m/s")
        lines.append(f"- Max position miss: `{b.get('max_position_miss_km')}` km")
        lines.append(f"- Intermediate velocity miss: `{b.get('max_intermediate_velocity_m_s')}` m/s")
        lines.append(f"- Final arrival v∞/velocity diagnostic: `{b.get('max_velocity_miss_m_s')}` m/s")
        lines.append(f"- Min rp margin: `{b.get('min_rp_margin_km')}` km")
        lines.append("")
        lines.append("## Interpretation")
        lines.append("")
        lines.append("This packet is a 6D-viable intermediate-patch candidate. Final-arrival velocity at Jool is treated as expected arrival v∞, not a rendezvous constraint.")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description="Export selected 6D-viable multi-flyby route packet.")
    ap.add_argument("--selected-jsonl", required=True)
    ap.add_argument("--corrected-jsonl", required=True)
    ap.add_argument("--diagnostics-jsonl", required=True)
    ap.add_argument("--route-number", type=int, default=None, help="1-based route number to export. Overrides selected-jsonl route choice.")
    ap.add_argument("--top-n", type=int, default=1)
    ap.add_argument("--max-patch-dv-m-s", type=float, default=75.0)
    ap.add_argument("--max-position-miss-km", type=float, default=10.0)
    ap.add_argument("--max-intermediate-velocity-m-s", type=float, default=100.0)
    ap.add_argument("--min-rp-margin-km", type=float, default=800.0)
    ap.add_argument("--rp-soft-km", type=float, default=1500.0)
    ap.add_argument("--rp-a-km", type=float, default=1500.0)
    ap.add_argument("--embed-source", action="store_true")
    ap.add_argument("--output-csv", required=True)
    ap.add_argument("--output-jsonl", required=True)
    ap.add_argument("--output-json", required=True)
    ap.add_argument("--output-best-json", required=True)
    ap.add_argument("--output-md", default=None)
    args = ap.parse_args()

    selected = read_json_or_jsonl(args.selected_jsonl)
    corrected = read_json_or_jsonl(args.corrected_jsonl)
    diagnostics = read_json_or_jsonl(args.diagnostics_jsonl)

    selected_numbers: List[int] = []
    if args.route_number is not None:
        selected_numbers = [args.route_number]
    else:
        for s in selected:
            # Prefer explicitly passing records.
            passed = deep_find(s, ("pass", "pass_manifest", "selected", "valid"))
            cls = str(deep_find(s, ("class", "route_class")) or "")
            if passed is False:
                continue
            n = route_number(s)
            if n is not None:
                selected_numbers.append(n)
        # Fallback: if selected file itself only contains a single best object and no route number.
        if not selected_numbers and selected:
            n = route_number(selected[0])
            if n is not None:
                selected_numbers.append(n)

    # De-duplicate, keep order.
    seen = set()
    selected_numbers = [n for n in selected_numbers if not (n in seen or seen.add(n))]

    packets: List[Dict[str, Any]] = []
    for n in selected_numbers:
        if n < 1 or n > len(corrected):
            continue
        src = corrected[n - 1]
        diag_rows = diag_for_route(diagnostics, n)
        seq = sequence_of(src) or sequence_of(selected[0] if selected else {})
        rid = route_id(src) or f"route_{n}"
        total_dv = get_total_dv(src)
        max_pos = get_max_position_miss(src)
        max_vel = get_max_velocity_miss(src)
        min_rp = get_min_rp_margin(src)
        segments = get_segments(src)
        flybys = get_flybys(src)
        dcounts = diagnosis_counts(diag_rows)
        v_int = intermediate_velocity_max(diag_rows)
        rclass = infer_class(total_dv, max_pos, min_rp, v_int, args)
        passed = rclass not in ("D", "C6D") and max_pos is not None and max_pos <= args.max_position_miss_km
        score = score_packet(total_dv, min_rp, v_int, args)
        packet = {
            "packet_type": "mga_b6d_route_packet_v0_1",
            "route_number": n,
            "route_id": rid,
            "sequence": " -> ".join(seq) if seq else None,
            "sequence_list": seq,
            "route_class": rclass,
            "pass_manifest": bool(passed),
            "score": score,
            "total_patch_dv_m_s": total_dv,
            "max_position_miss_km": max_pos,
            "max_velocity_miss_m_s": max_vel,
            "max_intermediate_velocity_m_s": v_int,
            "min_rp_margin_km": min_rp,
            "segments": len(segments),
            "flybys": len(flybys),
            "diagnosis_counts": dcounts,
            "semantics": {
                "intermediate_patch_points": "6D viability required/diagnosed",
                "final_arrival_velocity": "expected v_infinity at final target, not rendezvous constraint",
                "current_status": "ready for B-plane/multiple-shooting 6D or independent validation",
            },
            "diagnostics": diag_rows,
        }
        if args.embed_source:
            packet["source_corrected_record"] = src
        packets.append(packet)

    packets.sort(key=lambda p: (not p.get("pass_manifest", False), as_float(p.get("score"), 1e99) or 1e99))
    packets = packets[: max(args.top_n, 1)]

    classes: Dict[str, int] = {}
    for p in packets:
        classes[p["route_class"]] = classes.get(p["route_class"], 0) + 1

    summary = {
        "input_selected_records": len(selected),
        "input_corrected_records": len(corrected),
        "input_diagnostic_segments": len(diagnostics),
        "selected_route_numbers": selected_numbers,
        "packets_written": len(packets),
        "pass_manifest": sum(1 for p in packets if p.get("pass_manifest")),
        "classes": classes,
        "best": packets[0] if packets else None,
    }

    write_csv(args.output_csv, packets)
    write_jsonl(args.output_jsonl, packets)
    write_json(args.output_json, summary)
    if packets:
        write_json(args.output_best_json, packets[0])
    else:
        write_json(args.output_best_json, {})
    if args.output_md:
        write_md(args.output_md, packets, summary)

    print("=" * 80)
    print("MGA EXPORT B6D ROUTE PACKET V0.1")
    print("=" * 80)
    print(f"Selected records:  {len(selected)}")
    print(f"Corrected records: {len(corrected)}")
    print(f"Diag segments:     {len(diagnostics)}")
    print(f"Route numbers:     {selected_numbers}")
    print(f"Packets written:   {len(packets)}")
    print(f"Pass manifest:     {summary['pass_manifest']}")
    print(f"Classes:           {classes}")
    print("\nTop packets:")
    for i, p in enumerate(packets[:10], 1):
        print(
            f" {i}. route={p['route_number']} | {p.get('sequence')} | pass={p['pass_manifest']} | "
            f"class={p['route_class']} | score={p['score']:.3f} | dv={p.get('total_patch_dv_m_s')} m/s | "
            f"v_int={p.get('max_intermediate_velocity_m_s')} m/s | rpM={p.get('min_rp_margin_km')} km"
        )
    print("=" * 80)
    print(f"[OK] wrote CSV:       {args.output_csv}")
    print(f"[OK] wrote JSONL:     {args.output_jsonl}")
    print(f"[OK] wrote JSON:      {args.output_json}")
    print(f"[OK] wrote best JSON: {args.output_best_json}")
    if args.output_md:
        print(f"[OK] wrote Markdown:  {args.output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
