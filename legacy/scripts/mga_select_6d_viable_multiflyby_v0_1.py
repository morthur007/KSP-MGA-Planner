#!/usr/bin/env python3
"""
MGA Select 6D-Viable Multi-Flyby Packets V0.1

Reads segment diagnostics from mga_multiflyby_6d_patch_diagnostics_v0_3.py
and selects corrected multi-flyby packets whose intermediate patch segments are
6D-continuous enough for the next high-fidelity stage.

Final target arrivals are allowed to have nonzero v_inf by default.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


def finite(x: Any, default: float = math.nan) -> float:
    try:
        y = float(x)
    except (TypeError, ValueError):
        return default
    return y if math.isfinite(y) else default


def sanitize(obj: Any) -> Any:
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, (int, str, bool)) or obj is None:
        return obj
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, tuple):
        return [sanitize(x) for x in obj]
    if isinstance(obj, list):
        return [sanitize(x) for x in obj]
    if isinstance(obj, Mapping):
        return {str(k): sanitize(v) for k, v in obj.items()}
    return str(obj)


def read_json_records(path: Path) -> List[Dict[str, Any]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text[0] == "[":
        data = json.loads(text)
        return [dict(x) for x in data if isinstance(x, Mapping)]
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if len(lines) == 1:
        obj = json.loads(lines[0])
        if isinstance(obj, Mapping):
            for key in ("records", "results", "packets", "top_results", "top_packets"):
                val = obj.get(key)
                if isinstance(val, list):
                    return [dict(x) for x in val if isinstance(x, Mapping)]
            return [dict(obj)]
    return [dict(json.loads(ln)) for ln in lines]


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(sanitize(payload), f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write("\n")


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(sanitize(row), ensure_ascii=False, allow_nan=False, separators=(",", ":")))
            f.write("\n")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = [
        "route_index", "sequence", "pass_6d_selection", "class", "score", "total_patch_dv_m_s",
        "max_position_miss_km", "max_intermediate_velocity_m_s", "max_final_arrival_vinf_m_s",
        "min_rp_margin_km", "n_segments", "n_good_intermediate", "n_final_arrival", "n_rejected_segments",
        "rejected_diagnoses", "correction_id",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            out = dict(row)
            if isinstance(out.get("rejected_diagnoses"), (list, dict)):
                out["rejected_diagnoses"] = json.dumps(out["rejected_diagnoses"], ensure_ascii=False, separators=(",", ":"))
            w.writerow(out)


def get_path(obj: Any, path: str, default: Any = None) -> Any:
    cur = obj
    for part in path.split("."):
        if isinstance(cur, Mapping) and part in cur:
            cur = cur[part]
        else:
            return default
    return cur


def extract_sequence(rec: Mapping[str, Any]) -> str:
    seq = rec.get("sequence")
    if isinstance(seq, list):
        return " -> ".join(str(x) for x in seq)
    if isinstance(seq, str):
        return seq.replace(",", " -> ") if "," in seq and "->" not in seq else seq
    src = rec.get("source_packet")
    if isinstance(src, Mapping):
        return extract_sequence(src)
    return "?"


def extract_total_dv(rec: Mapping[str, Any]) -> float:
    candidates = [
        rec.get("total_segment_correction_m_s"),
        rec.get("total_patch_correction_m_s"),
        rec.get("total_correction_m_s"),
        get_path(rec, "metrics.total_segment_correction_m_s"),
        get_path(rec, "metrics.total_patch_correction_m_s"),
    ]
    for c in candidates:
        x = finite(c)
        if math.isfinite(x):
            return x
    segs = rec.get("segment_corrections") or rec.get("segments") or []
    total = 0.0
    found = False
    if isinstance(segs, list):
        for s in segs:
            if not isinstance(s, Mapping):
                continue
            dv = finite(s.get("dv_norm_m_s"), math.nan)
            if not math.isfinite(dv):
                dvx = finite(s.get("dvx_km_s"), 0.0)
                dvy = finite(s.get("dvy_km_s"), 0.0)
                dvz = finite(s.get("dvz_km_s"), 0.0)
                dv = 1000.0 * math.sqrt(dvx*dvx + dvy*dvy + dvz*dvz)
            total += dv
            found = True
    return total if found else math.nan


def extract_min_rp_margin(rec: Mapping[str, Any]) -> float:
    candidates = [
        rec.get("min_rp_margin_km"),
        rec.get("min_rp_margin"),
        get_path(rec, "metrics.min_rp_margin_km"),
        get_path(rec, "source_packet.metrics.min_rp_margin_km"),
    ]
    for c in candidates:
        x = finite(c)
        if math.isfinite(x):
            return x
    vals: List[float] = []
    for container in (rec.get("flybys"), get_path(rec, "source_packet.flybys")):
        if isinstance(container, list):
            for f in container:
                if isinstance(f, Mapping):
                    x = finite(f.get("rp_margin_km"), math.nan)
                    if math.isfinite(x):
                        vals.append(x)
    return min(vals) if vals else math.nan


def classify_packet(summary: Mapping[str, Any], args: argparse.Namespace) -> str:
    dv = finite(summary.get("total_patch_dv_m_s"), math.inf)
    vel = finite(summary.get("max_intermediate_velocity_m_s"), math.inf)
    rp = finite(summary.get("min_rp_margin_km"), -math.inf)
    pos = finite(summary.get("max_position_miss_km"), math.inf)
    if not summary.get("pass_6d_selection"):
        return "D"
    if dv <= args.class_a_dv_m_s and vel <= args.class_a_velocity_m_s and rp >= args.class_a_rp_margin_km and pos <= args.class_a_position_km:
        return "A6D"
    if dv <= args.class_b_dv_m_s and vel <= args.class_b_velocity_m_s and rp >= args.class_b_rp_margin_km and pos <= args.class_b_position_km:
        return "B6D"
    return "C6D"


def main() -> int:
    ap = argparse.ArgumentParser(description="Select multi-flyby routes with acceptable intermediate 6D patch continuity.")
    ap.add_argument("--diagnostics-jsonl", required=True, type=Path)
    ap.add_argument("--corrected-jsonl", required=True, type=Path)
    ap.add_argument("--top-n", type=int, default=10)
    ap.add_argument("--allow-final-arrival-vinf", action="store_true", default=True)
    ap.add_argument("--allowed-intermediate-diagnosis", action="append", default=["6d_continuity_good"])
    ap.add_argument("--max-intermediate-velocity-m-s", type=float, default=100.0)
    ap.add_argument("--max-position-miss-km", type=float, default=10.0)
    ap.add_argument("--max-patch-dv-m-s", type=float, default=75.0)
    ap.add_argument("--min-rp-margin-km", type=float, default=800.0)
    ap.add_argument("--class-a-dv-m-s", type=float, default=35.0)
    ap.add_argument("--class-b-dv-m-s", type=float, default=75.0)
    ap.add_argument("--class-a-velocity-m-s", type=float, default=25.0)
    ap.add_argument("--class-b-velocity-m-s", type=float, default=100.0)
    ap.add_argument("--class-a-rp-margin-km", type=float, default=1200.0)
    ap.add_argument("--class-b-rp-margin-km", type=float, default=800.0)
    ap.add_argument("--class-a-position-km", type=float, default=1e-3)
    ap.add_argument("--class-b-position-km", type=float, default=10.0)
    ap.add_argument("--output-csv", required=True, type=Path)
    ap.add_argument("--output-jsonl", required=True, type=Path)
    ap.add_argument("--output-json", required=True, type=Path)
    ap.add_argument("--output-best-json", required=True, type=Path)
    args = ap.parse_args()

    diag_rows = read_json_records(args.diagnostics_jsonl)
    packets = read_json_records(args.corrected_jsonl)
    by_route: Dict[int, List[Dict[str, Any]]] = {}
    for row in diag_rows:
        ridx = int(finite(row.get("route_index"), -1))
        if ridx > 0:
            by_route.setdefault(ridx, []).append(row)

    summaries: List[Dict[str, Any]] = []
    selected_packets: List[Dict[str, Any]] = []
    allowed = set(args.allowed_intermediate_diagnosis)
    for i, pkt in enumerate(packets, start=1):
        rows = by_route.get(i, [])
        rejected: List[Dict[str, Any]] = []
        n_good = 0
        n_final = 0
        max_pos = 0.0
        max_inter_vel = 0.0
        max_final_vinf = 0.0
        for r in rows:
            diag = str(r.get("diagnosis"))
            kind = str(r.get("segment_kind") or "")
            target = str(r.get("target_label") or "")
            pos = finite(r.get("position_miss_km"), math.inf)
            max_pos = max(max_pos, pos)
            vel = finite(r.get("local_velocity_delta_m_s"), finite(r.get("central_velocity_delta_m_s"), math.inf))
            is_final = diag == "final_arrival_vinf_expected_not_6d_match" or kind == "last_flyby_exit_to_final_target" or target.endswith("_center")
            if is_final and args.allow_final_arrival_vinf:
                n_final += 1
                max_final_vinf = max(max_final_vinf, vel if math.isfinite(vel) else 0.0)
                continue
            max_inter_vel = max(max_inter_vel, vel if math.isfinite(vel) else math.inf)
            if diag in allowed and pos <= args.max_position_miss_km and vel <= args.max_intermediate_velocity_m_s:
                n_good += 1
            else:
                rejected.append({
                    "segment_index": r.get("segment_index"),
                    "segment_kind": kind,
                    "origin_label": r.get("origin_label"),
                    "target_label": target,
                    "diagnosis": diag,
                    "position_miss_km": pos,
                    "local_velocity_delta_m_s": vel,
                })

        dv = extract_total_dv(pkt)
        rp = extract_min_rp_margin(pkt)
        pass_sel = (not rejected and max_pos <= args.max_position_miss_km and dv <= args.max_patch_dv_m_s and rp >= args.min_rp_margin_km)
        # Low score is better: primary dv, then intermediate velocity, then rp margin softness.
        rp_pen = max(0.0, args.class_a_rp_margin_km - rp) / max(1.0, args.class_a_rp_margin_km)
        score = (dv if math.isfinite(dv) else 1e9) + 0.02 * max_inter_vel + 5.0 * rp_pen
        summary = {
            "route_index": i,
            "correction_id": pkt.get("multiflyby_patch_correction_id") or pkt.get("correction_id"),
            "sequence": extract_sequence(pkt),
            "pass_6d_selection": pass_sel,
            "class": "",
            "score": score,
            "total_patch_dv_m_s": dv,
            "max_position_miss_km": max_pos,
            "max_intermediate_velocity_m_s": max_inter_vel,
            "max_final_arrival_vinf_m_s": max_final_vinf,
            "min_rp_margin_km": rp,
            "n_segments": len(rows),
            "n_good_intermediate": n_good,
            "n_final_arrival": n_final,
            "n_rejected_segments": len(rejected),
            "rejected_diagnoses": rejected,
        }
        summary["class"] = classify_packet(summary, args)
        summaries.append(summary)
        if pass_sel:
            out_pkt = dict(pkt)
            out_pkt["six_d_selection"] = summary
            selected_packets.append(out_pkt)

    selected_pairs = sorted(zip([s for s in summaries if s.get("pass_6d_selection")], selected_packets), key=lambda p: finite(p[0].get("score")))
    selected_summaries = [p[0] for p in selected_pairs[:args.top_n]]
    selected_packets_sorted = [p[1] for p in selected_pairs[:args.top_n]]
    all_sorted = sorted(summaries, key=lambda s: (not s.get("pass_6d_selection"), finite(s.get("score"))))

    write_csv(args.output_csv, all_sorted)
    write_jsonl(args.output_jsonl, selected_packets_sorted)
    best = selected_packets_sorted[0] if selected_packets_sorted else {}
    write_json(args.output_best_json, best)
    counts: Dict[str, int] = {}
    for s in summaries:
        counts[str(s.get("class"))] = counts.get(str(s.get("class")), 0) + 1
    payload = {
        "schema_version": "mga_select_6d_viable_multiflyby.v0.1",
        "diagnostics_jsonl": str(args.diagnostics_jsonl),
        "corrected_jsonl": str(args.corrected_jsonl),
        "routes_input": len(packets),
        "diagnostic_segments": len(diag_rows),
        "routes_selected": len(selected_packets_sorted),
        "class_counts": counts,
        "selected_summaries": selected_summaries,
        "all_summaries": all_sorted,
        "criteria": {
            "allowed_intermediate_diagnosis": sorted(allowed),
            "max_intermediate_velocity_m_s": args.max_intermediate_velocity_m_s,
            "max_position_miss_km": args.max_position_miss_km,
            "max_patch_dv_m_s": args.max_patch_dv_m_s,
            "min_rp_margin_km": args.min_rp_margin_km,
            "allow_final_arrival_vinf": args.allow_final_arrival_vinf,
        },
    }
    write_json(args.output_json, payload)

    print("="*80)
    print("MGA SELECT 6D-VIABLE MULTI-FLYBY PACKETS V0.1")
    print("="*80)
    print(f"Routes input:      {len(packets)}")
    print(f"Segments input:    {len(diag_rows)}")
    print(f"Selected routes:   {len(selected_packets_sorted)}")
    print(f"Classes:           {counts}")
    print("\nTop summaries:")
    for s in all_sorted[:10]:
        print(
            f"  route={s['route_index']} | pass={s['pass_6d_selection']} | class={s['class']} | "
            f"score={finite(s['score']):.3f} | dv={finite(s['total_patch_dv_m_s']):.3f} m/s | "
            f"v_int={finite(s['max_intermediate_velocity_m_s']):.3f} m/s | "
            f"v_final={finite(s['max_final_arrival_vinf_m_s']):.1f} m/s | "
            f"rpM={finite(s['min_rp_margin_km']):.1f} km | rej={s['n_rejected_segments']}"
        )
    print("="*80)
    print(f"[OK] wrote CSV:       {args.output_csv}")
    print(f"[OK] wrote JSONL:     {args.output_jsonl}")
    print(f"[OK] wrote JSON:      {args.output_json}")
    print(f"[OK] wrote best JSON: {args.output_best_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
