#!/usr/bin/env python3
"""
MGA SELECT 6D-VIABLE MULTI-FLYBY PACKETS V0.2

V0.1 was intentionally strict: any intermediate segment not classified as
6d_continuity_good could reject the route. V0.2 adds an operational mode where
moderate_6d_velocity_discontinuity is accepted if the numeric intermediate
velocity mismatch is below --max-intermediate-velocity-m-s.

Use this after mga_multiflyby_6d_patch_diagnostics_v0_3.py.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

BAD_DIAGNOSES_DEFAULT = {
    "energy_or_lambert_branch_mismatch",
    "likely_sign_convention_mismatch",
    "missing_target_local_velocity",
    "missing_patch_state",
    "integration_failed",
    "schema_error",
}
FINAL_EXPECTED = "final_arrival_vinf_expected_not_6d_match"
GOOD = "6d_continuity_good"
MODERATE = "moderate_6d_velocity_discontinuity"


def is_finite(x: Any) -> bool:
    try:
        return math.isfinite(float(x))
    except Exception:
        return False


def fnum(x: Any, default: float = math.nan) -> float:
    try:
        if x is None or x == "":
            return default
        y = float(x)
        return y if math.isfinite(y) else default
    except Exception:
        return default


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception as e:
                raise RuntimeError(f"Could not parse JSONL {path}:{line_no}: {e}") from e
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def sanitize(obj: Any) -> Any:
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {str(k): sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize(v) for v in obj]
    return obj


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sanitize(obj), ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(sanitize(r), ensure_ascii=False, separators=(",", ":")) + "\n")


def flatten_values(obj: Any, prefix: str = "") -> Iterable[Tuple[str, Any]]:
    if isinstance(obj, dict):
        for k, v in obj.items():
            p = f"{prefix}.{k}" if prefix else str(k)
            yield from flatten_values(v, p)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            p = f"{prefix}[{i}]"
            yield from flatten_values(v, p)
    else:
        yield prefix, obj


def first_value_by_keys(obj: Any, names: Iterable[str], default: Any = None) -> Any:
    names_l = {n.lower() for n in names}
    if isinstance(obj, dict):
        for k, v in obj.items():
            if str(k).lower() in names_l:
                return v
        for v in obj.values():
            got = first_value_by_keys(v, names_l, None)
            if got is not None:
                return got
    elif isinstance(obj, list):
        for v in obj:
            got = first_value_by_keys(v, names_l, None)
            if got is not None:
                return got
    return default


def max_numeric_by_key_contains(obj: Any, include: Iterable[str], exclude: Iterable[str] = ()) -> float:
    inc = [s.lower() for s in include]
    exc = [s.lower() for s in exclude]
    vals: List[float] = []
    for path, value in flatten_values(obj):
        pl = path.lower()
        if all(s in pl for s in inc) and not any(s in pl for s in exc):
            y = fnum(value)
            if math.isfinite(y):
                vals.append(abs(y))
    return max(vals) if vals else math.nan


def route_number_from_diag(row: Dict[str, Any], fallback: int) -> int:
    for key in (
        "route_number", "route", "route_index", "route_rank", "packet_index",
        "packet_number", "candidate_number", "rank", "source_index",
    ):
        if key in row:
            y = fnum(row[key])
            if math.isfinite(y):
                return int(y)
    # recursive fallback
    got = first_value_by_keys(row, ["route_number", "route", "route_index", "packet_index"], None)
    y = fnum(got)
    return int(y) if math.isfinite(y) else fallback


def segment_index_from_diag(row: Dict[str, Any]) -> Optional[int]:
    for key in ("segment_index", "seg", "segment", "segment_number"):
        if key in row:
            y = fnum(row[key])
            if math.isfinite(y):
                return int(y)
    return None


def diagnosis_from_row(row: Dict[str, Any]) -> str:
    for key in ("diagnosis", "class", "diagnostic_class", "status", "reason"):
        v = row.get(key)
        if isinstance(v, str) and v:
            return v
    got = first_value_by_keys(row, ["diagnosis", "diagnostic_class"], None)
    return str(got) if got is not None else "unknown"


def velocity_mismatch_from_diag(row: Dict[str, Any]) -> float:
    # Prefer explicit mismatch fields.
    preferred = [
        "velocity_mismatch_local_m_s",
        "local_velocity_mismatch_m_s",
        "v_local_mismatch_m_s",
        "v_loc_mismatch_m_s",
        "velocity_mismatch_m_s",
        "v_mismatch_m_s",
        "v_c_m_s",
        "v_loc_m_s",
        "speed_delta_m_s",
        "speed_delta_local_m_s",
        "speed_mismatch_m_s",
    ]
    for k in preferred:
        v = first_value_by_keys(row, [k], None)
        y = fnum(v)
        if math.isfinite(y):
            return abs(y)
    # Last resort: key contains patterns.
    candidates = []
    for path, value in flatten_values(row):
        pl = path.lower()
        if any(tok in pl for tok in ["mismatch", "speeddelta", "speed_delta", "v_c", "v_loc"]):
            if pl.endswith("_m_s") or "m_s" in pl or "velocity" in pl or "speed" in pl or "v_c" in pl or "v_loc" in pl:
                y = fnum(value)
                if math.isfinite(y):
                    candidates.append(abs(y))
    return max(candidates) if candidates else math.nan


def seq_from_record(record: Dict[str, Any]) -> str:
    # Common paths.
    for path in (
        ["sequence"],
        ["source_packet", "sequence"],
        ["source", "sequence"],
        ["route", "sequence"],
        ["corrected_route", "sequence"],
    ):
        obj: Any = record
        ok = True
        for key in path:
            if isinstance(obj, dict) and key in obj:
                obj = obj[key]
            else:
                ok = False
                break
        if ok:
            if isinstance(obj, list):
                return " -> ".join(str(x) for x in obj)
            if isinstance(obj, str):
                return obj.replace(",", " -> ")
    # Recursive fallback for a list containing Kerbin/Jool.
    def scan(o: Any, depth: int = 0) -> Optional[str]:
        if depth > 5:
            return None
        if isinstance(o, dict):
            for k, v in o.items():
                if str(k).lower() == "sequence":
                    if isinstance(v, list) and v:
                        return " -> ".join(str(x) for x in v)
                    if isinstance(v, str):
                        return v.replace(",", " -> ")
            for v in o.values():
                got = scan(v, depth + 1)
                if got:
                    return got
        elif isinstance(o, list):
            if o and all(isinstance(x, str) for x in o) and "Kerbin" in o:
                return " -> ".join(o)
            for v in o:
                got = scan(v, depth + 1)
                if got:
                    return got
        return None
    return scan(record) or "unknown"


def metric_from_record(record: Dict[str, Any], keys: Iterable[str], contains_fallback: Optional[List[str]] = None) -> float:
    v = first_value_by_keys(record, keys, None)
    y = fnum(v)
    if math.isfinite(y):
        return y
    if contains_fallback:
        return max_numeric_by_key_contains(record, contains_fallback)
    return math.nan


def corrected_metrics(record: Dict[str, Any]) -> Dict[str, float]:
    dv = metric_from_record(record, [
        "total_segment_correction_m_s", "total_correction_m_s", "total_patch_dv_m_s",
        "known_correction_m_s", "patch_dv_m_s",
    ], ["correction", "m_s"])
    pos = metric_from_record(record, [
        "max_position_miss_after_km", "max_miss_after_km", "max_pos_miss_after_km",
        "max_endpoint_miss_after_km", "position_miss_after_km",
    ], ["miss", "km"])
    rp = metric_from_record(record, [
        "min_rp_margin_km", "rp_margin_km", "minimum_rp_margin_km",
    ], ["rp", "margin"])
    return {"patch_dv_m_s": dv, "position_miss_km": pos, "min_rp_margin_km": rp}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--diagnostics-jsonl", required=True)
    ap.add_argument("--corrected-jsonl", required=True)
    ap.add_argument("--top-n", type=int, default=20)
    ap.add_argument("--max-intermediate-velocity-m-s", type=float, default=100.0)
    ap.add_argument("--max-position-miss-km", type=float, default=10.0)
    ap.add_argument("--max-patch-dv-m-s", type=float, default=75.0)
    ap.add_argument("--min-rp-margin-km", type=float, default=800.0)
    ap.add_argument("--accept-moderate", action="store_true", default=True)
    ap.add_argument("--strict-good-only", action="store_true", help="Require every intermediate segment to be 6d_continuity_good.")
    ap.add_argument("--max-moderate-segments", type=int, default=2)
    ap.add_argument("--embed-source", action="store_true")
    ap.add_argument("--output-csv", required=True)
    ap.add_argument("--output-jsonl", required=True)
    ap.add_argument("--output-json", required=True)
    ap.add_argument("--output-best-json", required=True)
    args = ap.parse_args()

    diag_rows = read_jsonl(Path(args.diagnostics_jsonl))
    corrected_rows = read_jsonl(Path(args.corrected_jsonl))

    diag_by_route: Dict[int, List[Dict[str, Any]]] = {}
    for i, row in enumerate(diag_rows, start=1):
        rn = route_number_from_diag(row, i)
        diag_by_route.setdefault(rn, []).append(row)

    summaries: List[Dict[str, Any]] = []
    selected_records: List[Dict[str, Any]] = []

    for rn, corr in enumerate(corrected_rows[: args.top_n], start=1):
        diags = diag_by_route.get(rn, [])
        metrics = corrected_metrics(corr)
        seq = seq_from_record(corr)

        inter = []
        final = []
        for d in diags:
            diag = diagnosis_from_row(d)
            if diag == FINAL_EXPECTED:
                final.append(d)
            else:
                inter.append(d)

        good_count = 0
        moderate_count = 0
        bad_count = 0
        unknown_count = 0
        max_inter_v = 0.0
        max_final_v = 0.0
        bad_diagnoses: List[str] = []

        for d in inter:
            diag = diagnosis_from_row(d)
            v = velocity_mismatch_from_diag(d)
            if math.isfinite(v):
                max_inter_v = max(max_inter_v, v)
            if diag == GOOD:
                good_count += 1
            elif diag == MODERATE:
                moderate_count += 1
            elif diag in BAD_DIAGNOSES_DEFAULT:
                bad_count += 1
                bad_diagnoses.append(diag)
            else:
                # Unknown but numerically small may be tolerated as moderate.
                unknown_count += 1
                bad_diagnoses.append(diag)

        for d in final:
            v = velocity_mismatch_from_diag(d)
            if math.isfinite(v):
                max_final_v = max(max_final_v, v)

        rejects: List[str] = []
        if not math.isfinite(metrics["patch_dv_m_s"]) or metrics["patch_dv_m_s"] > args.max_patch_dv_m_s:
            rejects.append("patch_dv")
        if not math.isfinite(metrics["position_miss_km"]) or metrics["position_miss_km"] > args.max_position_miss_km:
            rejects.append("position_miss")
        if not math.isfinite(metrics["min_rp_margin_km"]) or metrics["min_rp_margin_km"] < args.min_rp_margin_km:
            rejects.append("rp_margin")
        if bad_count > 0:
            rejects.append("bad_intermediate_diagnosis")
        if unknown_count > 0:
            rejects.append("unknown_intermediate_diagnosis")
        if args.strict_good_only and moderate_count > 0:
            rejects.append("moderate_not_allowed")
        if (not args.strict_good_only) and moderate_count > args.max_moderate_segments:
            rejects.append("too_many_moderate_segments")
        if max_inter_v > args.max_intermediate_velocity_m_s:
            rejects.append("intermediate_velocity")

        passed = len(rejects) == 0
        if passed:
            if moderate_count == 0 and metrics["patch_dv_m_s"] <= 30:
                cls = "A6D"
            elif moderate_count <= 1 and metrics["patch_dv_m_s"] <= 50:
                cls = "B6D"
            else:
                cls = "C6D"
        else:
            cls = "D"

        rp_penalty = max(0.0, (args.min_rp_margin_km + 500.0 - metrics["min_rp_margin_km"]) / 500.0) if math.isfinite(metrics["min_rp_margin_km"]) else 99.0
        score = (
            (metrics["patch_dv_m_s"] if math.isfinite(metrics["patch_dv_m_s"]) else 9999.0)
            + 0.03 * max_inter_v
            + 0.2 * max(0, moderate_count)
            + 100.0 * bad_count
            + 50.0 * unknown_count
            + 0.5 * rp_penalty
        )

        summary = {
            "route_number": rn,
            "sequence": seq,
            "pass": passed,
            "class": cls,
            "score": score,
            "patch_dv_m_s": metrics["patch_dv_m_s"],
            "max_position_miss_km": metrics["position_miss_km"],
            "max_intermediate_velocity_m_s": max_inter_v,
            "max_final_arrival_vinf_m_s": max_final_v,
            "min_rp_margin_km": metrics["min_rp_margin_km"],
            "intermediate_segments": len(inter),
            "final_segments": len(final),
            "good_segments": good_count,
            "moderate_segments": moderate_count,
            "bad_segments": bad_count,
            "unknown_segments": unknown_count,
            "reject_count": len(rejects),
            "reject_reasons": ";".join(rejects),
            "bad_diagnoses": ";".join(sorted(set(bad_diagnoses))),
        }
        summaries.append(summary)

        if passed:
            out = dict(corr) if args.embed_source else {"route_number": rn, **summary}
            if args.embed_source:
                out["b6d_selection"] = summary
            selected_records.append(out)

    summaries.sort(key=lambda r: (not r["pass"], r["score"], r["route_number"]))
    selected_records_ordered: List[Dict[str, Any]] = []
    selected_nums = [r["route_number"] for r in summaries if r["pass"]]
    for rn in selected_nums:
        corr = corrected_rows[rn - 1]
        summary = next(r for r in summaries if r["route_number"] == rn)
        out = dict(corr) if args.embed_source else {"route_number": rn, **summary}
        if args.embed_source:
            out["b6d_selection"] = summary
        selected_records_ordered.append(out)

    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = list(summaries[0].keys()) if summaries else ["route_number"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in summaries:
            w.writerow(sanitize(r))

    write_jsonl(Path(args.output_jsonl), selected_records_ordered)
    counts: Dict[str, int] = {}
    for r in summaries:
        counts[r["class"]] = counts.get(r["class"], 0) + 1
    payload = {
        "script": "mga_select_6d_viable_multiflyby_v0_2.py",
        "routes_input": len(corrected_rows),
        "segments_input": len(diag_rows),
        "selected_routes": len(selected_records_ordered),
        "classes": counts,
        "criteria": {
            "max_intermediate_velocity_m_s": args.max_intermediate_velocity_m_s,
            "max_position_miss_km": args.max_position_miss_km,
            "max_patch_dv_m_s": args.max_patch_dv_m_s,
            "min_rp_margin_km": args.min_rp_margin_km,
            "strict_good_only": args.strict_good_only,
            "max_moderate_segments": args.max_moderate_segments,
        },
        "top_summaries": summaries[: min(20, len(summaries))],
    }
    write_json(Path(args.output_json), payload)
    best = selected_records_ordered[0] if selected_records_ordered else (summaries[0] if summaries else {})
    write_json(Path(args.output_best_json), best)

    print("=" * 80)
    print("MGA SELECT 6D-VIABLE MULTI-FLYBY PACKETS V0.2")
    print("=" * 80)
    print(f"Routes input:      {len(corrected_rows)}")
    print(f"Segments input:    {len(diag_rows)}")
    print(f"Selected routes:   {len(selected_records_ordered)}")
    print(f"Classes:           {counts}")
    print(f"Strict good only:  {args.strict_good_only}")
    print(f"Max moderate segs: {args.max_moderate_segments}")
    print("\nTop summaries:")
    for r in summaries[:10]:
        print(
            f"  route={r['route_number']} | pass={r['pass']} | class={r['class']} | "
            f"score={r['score']:.3f} | dv={r['patch_dv_m_s']:.3f} m/s | "
            f"v_int={r['max_intermediate_velocity_m_s']:.3f} m/s | "
            f"v_final={r['max_final_arrival_vinf_m_s']:.1f} m/s | "
            f"rpM={r['min_rp_margin_km']:.1f} km | "
            f"good={r['good_segments']} moderate={r['moderate_segments']} bad={r['bad_segments']} | "
            f"rej={r['reject_count']} {r['reject_reasons']}"
        )
    print("=" * 80)
    print(f"[OK] wrote CSV:       {args.output_csv}")
    print(f"[OK] wrote JSONL:     {args.output_jsonl}")
    print(f"[OK] wrote JSON:      {args.output_json}")
    print(f"[OK] wrote best JSON: {args.output_best_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
