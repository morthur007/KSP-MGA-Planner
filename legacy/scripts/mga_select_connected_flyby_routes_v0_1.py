#!/usr/bin/env python3
"""
mga_select_connected_flyby_routes_v0_1.py

Select robust connected-flyby PyGMO routes before promotion into patched correction,
flyby closure, B-plane packet, and local targeting stages.

Consumes JSONL from mga_pygmo_refine_connected_flyby_v0_1.py.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

SCHEMA_VERSION = "mga_select_connected_flyby_routes.v0.1"


def opt_float(x: Any) -> Optional[float]:
    if x is None or x == "":
        return None
    try:
        y = float(x)
    except Exception:
        return None
    return y if math.isfinite(y) else None


def finite(x: Any, default: float = 0.0) -> float:
    y = opt_float(x)
    return default if y is None else y


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            obj = json.loads(s)
            if isinstance(obj, Mapping):
                out.append(dict(obj))
    return out


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n")


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")


def seq_text(r: Mapping[str, Any]) -> str:
    seq = r.get("sequence")
    if isinstance(seq, str):
        return seq.replace(",", "->")
    if isinstance(seq, Sequence):
        return "->".join(str(x) for x in seq)
    return ""


def seq_list(r: Mapping[str, Any]) -> List[str]:
    seq = r.get("sequence")
    if isinstance(seq, str):
        sep = "->" if "->" in seq else ","
        return [x.strip() for x in seq.split(sep) if x.strip()]
    if isinstance(seq, Sequence):
        return [str(x) for x in seq]
    return []


def flybys(r: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    fbs = r.get("flyby_evals") or r.get("flybys") or []
    if isinstance(fbs, list):
        return [x for x in fbs if isinstance(x, Mapping)]
    return []


def leg_tofs(r: Mapping[str, Any]) -> List[float]:
    legs = r.get("leg_evals") or []
    vals = []
    if isinstance(legs, list):
        for leg in legs:
            if isinstance(leg, Mapping):
                vals.append(finite(leg.get("tof_days"), math.nan))
    return vals


def route_key(r: Mapping[str, Any], depart_bin_days: float, tof_bin_days: float) -> Tuple[Any, ...]:
    seq = seq_text(r)
    dep = finite((r.get("decision_vector") or [None])[0] if isinstance(r.get("decision_vector"), list) and r.get("decision_vector") else r.get("depart_day"), math.nan)
    total = finite(r.get("total_tof_days"), math.nan)
    dep_bin = int(math.floor(dep / depart_bin_days)) if math.isfinite(dep) and depart_bin_days > 0 else 0
    tof_bin = int(math.floor(total / tof_bin_days)) if math.isfinite(total) and tof_bin_days > 0 else 0
    tof_tuple = tuple(int(math.floor(x / tof_bin_days)) for x in leg_tofs(r) if math.isfinite(x) and tof_bin_days > 0)
    return (seq, dep_bin, tof_bin, tof_tuple)


def robust_score(r: Mapping[str, Any], args: argparse.Namespace) -> float:
    obj = finite(r.get("objective"), 1e9)
    mismatch = finite(r.get("max_vinf_mismatch_km_s"), 1e9) * 1000.0
    turn = finite(r.get("max_turn_angle_deg"), 1e9)
    margin = finite(r.get("min_rp_margin_km"), -1e9)
    tof = finite(r.get("total_tof_days"), 1e9)
    # Lower is better. Reward margin up to rp_soft_margin, penalize mismatch/turn/TOF mildly.
    margin_deficit = max(0.0, args.rp_soft_margin_km - margin)
    return (
        obj
        + args.vinf_mismatch_weight * max(0.0, mismatch - args.vinf_soft_m_s) / max(args.vinf_soft_m_s, 1e-9)
        + args.rp_margin_weight * margin_deficit / max(args.rp_soft_margin_km, 1e-9)
        + args.turn_weight * turn / 180.0
        + args.tof_weight * tof / 1000.0
    )


def filter_reason(r: Mapping[str, Any], args: argparse.Namespace) -> Optional[str]:
    if args.valid_only and not bool(r.get("valid")):
        return "invalid"
    seq = seq_list(r)
    if args.require_sequence:
        req = [x.strip() for x in args.require_sequence.replace("->", ",").split(",") if x.strip()]
        if seq != req:
            return "sequence"
    mismatch = finite(r.get("max_vinf_mismatch_km_s"), 1e99) * 1000.0
    if mismatch > args.max_vinf_mismatch_m_s:
        return "vinf_mismatch"
    margin = finite(r.get("min_rp_margin_km"), -1e99)
    if margin < args.min_rp_margin_km:
        return "rp_margin"
    turn = finite(r.get("max_turn_angle_deg"), 1e99)
    if turn > args.max_turn_angle_deg:
        return "turn_angle"
    tof = finite(r.get("total_tof_days"), 1e99)
    if tof > args.max_tof_days:
        return "tof"
    fbs = flybys(r)
    if args.require_flyby_bodies:
        fb_names = [str(x.get("body")) for x in fbs]
        for name in args.require_flyby_bodies:
            if name not in fb_names:
                return f"missing_flyby_{name}"
    for fb in fbs:
        fb_margin = finite(fb.get("rp_margin_km"), -1e99)
        if fb_margin < args.min_per_flyby_rp_margin_km:
            return f"flyby_rp_margin_{fb.get('body')}"
        fb_mis = finite(fb.get("vinf_mag_mismatch_km_s"), 1e99) * 1000.0
        if fb_mis > args.max_per_flyby_vinf_mismatch_m_s:
            return f"flyby_vinf_{fb.get('body')}"
    return None


def enrich(r: Mapping[str, Any], rank: int, score: float) -> Dict[str, Any]:
    d = dict(r)
    d["selection_schema_version"] = SCHEMA_VERSION
    d["selection_rank"] = rank
    d["robust_score"] = score
    return d


def fmt(x: Any, nd: int = 6) -> str:
    y = opt_float(x)
    if y is None:
        return ""
    return f"{y:.{nd}g}"


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "selection_rank", "sequence", "valid", "status", "objective", "robust_score", "total_tof_days",
        "max_vinf_mismatch_m_s", "max_turn_angle_deg", "min_rp_margin_km", "flyby_bodies",
        "flyby_rp_margins_km", "flyby_vinf_mismatch_m_s", "decision_vector", "refined_id",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            fbs = flybys(r)
            w.writerow({
                "selection_rank": r.get("selection_rank"),
                "sequence": seq_text(r),
                "valid": int(bool(r.get("valid"))),
                "status": r.get("status"),
                "objective": fmt(r.get("objective"), 9),
                "robust_score": fmt(r.get("robust_score"), 9),
                "total_tof_days": fmt(r.get("total_tof_days"), 6),
                "max_vinf_mismatch_m_s": fmt(finite(r.get("max_vinf_mismatch_km_s"), math.nan) * 1000.0, 6),
                "max_turn_angle_deg": fmt(r.get("max_turn_angle_deg"), 6),
                "min_rp_margin_km": fmt(r.get("min_rp_margin_km"), 6),
                "flyby_bodies": ";".join(str(fb.get("body")) for fb in fbs),
                "flyby_rp_margins_km": ";".join(fmt(fb.get("rp_margin_km"), 6) for fb in fbs),
                "flyby_vinf_mismatch_m_s": ";".join(fmt(finite(fb.get("vinf_mag_mismatch_km_s"), math.nan)*1000.0, 6) for fb in fbs),
                "decision_vector": ";".join(fmt(x, 9) for x in (r.get("decision_vector") or [])),
                "refined_id": r.get("refined_id"),
            })


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Select robust connected-flyby routes from PyGMO output.")
    p.add_argument("--input-jsonl", required=True, type=Path)
    p.add_argument("--valid-only", action="store_true", default=True)
    p.add_argument("--require-sequence", default="")
    p.add_argument("--require-flyby-bodies", nargs="*", default=[])
    p.add_argument("--min-rp-margin-km", type=float, default=300.0)
    p.add_argument("--min-per-flyby-rp-margin-km", type=float, default=100.0)
    p.add_argument("--rp-soft-margin-km", type=float, default=1000.0)
    p.add_argument("--max-vinf-mismatch-m-s", type=float, default=25.0)
    p.add_argument("--max-per-flyby-vinf-mismatch-m-s", type=float, default=25.0)
    p.add_argument("--vinf-soft-m-s", type=float, default=5.0)
    p.add_argument("--max-turn-angle-deg", type=float, default=80.0)
    p.add_argument("--max-tof-days", type=float, default=5000.0)
    p.add_argument("--depart-bin-days", type=float, default=30.0)
    p.add_argument("--tof-bin-days", type=float, default=60.0)
    p.add_argument("--top-n-per-bin", type=int, default=1)
    p.add_argument("--top-n", type=int, default=20)
    p.add_argument("--vinf-mismatch-weight", type=float, default=1.0)
    p.add_argument("--rp-margin-weight", type=float, default=4.0)
    p.add_argument("--turn-weight", type=float, default=0.5)
    p.add_argument("--tof-weight", type=float, default=0.05)
    p.add_argument("--output-csv", required=True, type=Path)
    p.add_argument("--output-jsonl", required=True, type=Path)
    p.add_argument("--output-json", required=True, type=Path)
    p.add_argument("--output-best-json", required=True, type=Path)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    rows = read_jsonl(args.input_jsonl)
    reject: Dict[str, int] = {}
    eligible: List[Tuple[float, Dict[str, Any]]] = []
    for r in rows:
        reason = filter_reason(r, args)
        if reason:
            reject[reason] = reject.get(reason, 0) + 1
            continue
        sc = robust_score(r, args)
        eligible.append((sc, dict(r)))
    eligible.sort(key=lambda x: x[0])

    by_bin: Dict[Tuple[Any, ...], List[Tuple[float, Dict[str, Any]]]] = {}
    for sc, r in eligible:
        by_bin.setdefault(route_key(r, args.depart_bin_days, args.tof_bin_days), []).append((sc, r))

    diverse: List[Tuple[float, Dict[str, Any]]] = []
    for vals in by_bin.values():
        diverse.extend(vals[: max(1, args.top_n_per_bin)])
    diverse.sort(key=lambda x: x[0])
    selected_pairs = diverse[: args.top_n]
    selected = [enrich(r, i, sc) for i, (sc, r) in enumerate(selected_pairs, start=1)]

    write_csv(args.output_csv, selected)
    write_jsonl(args.output_jsonl, selected)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "input_jsonl": str(args.input_jsonl),
        "counts": {
            "input_routes": len(rows),
            "eligible_routes": len(eligible),
            "diverse_routes": len(diverse),
            "selected_routes": len(selected),
        },
        "filters": {
            "require_sequence": args.require_sequence,
            "require_flyby_bodies": args.require_flyby_bodies,
            "min_rp_margin_km": args.min_rp_margin_km,
            "min_per_flyby_rp_margin_km": args.min_per_flyby_rp_margin_km,
            "max_vinf_mismatch_m_s": args.max_vinf_mismatch_m_s,
            "max_per_flyby_vinf_mismatch_m_s": args.max_per_flyby_vinf_mismatch_m_s,
            "max_turn_angle_deg": args.max_turn_angle_deg,
            "max_tof_days": args.max_tof_days,
        },
        "reject_reasons": dict(sorted(reject.items(), key=lambda kv: (-kv[1], kv[0]))),
        "top_selected": [
            {
                "selection_rank": r.get("selection_rank"),
                "sequence": seq_text(r),
                "objective": r.get("objective"),
                "robust_score": r.get("robust_score"),
                "total_tof_days": r.get("total_tof_days"),
                "max_vinf_mismatch_m_s": finite(r.get("max_vinf_mismatch_km_s"), math.nan)*1000.0,
                "max_turn_angle_deg": r.get("max_turn_angle_deg"),
                "min_rp_margin_km": r.get("min_rp_margin_km"),
            }
            for r in selected[:20]
        ],
    }
    write_json(args.output_json, summary)
    if selected:
        write_json(args.output_best_json, selected[0])
    else:
        write_json(args.output_best_json, {"schema_version": SCHEMA_VERSION, "selected": None})

    print("=" * 80)
    print("MGA SELECT CONNECTED FLYBY ROUTES V0.1")
    print("=" * 80)
    print(f"Input routes:    {len(rows)}")
    print(f"Eligible routes: {len(eligible)}")
    print(f"Diverse routes:  {len(diverse)}")
    print(f"Selected routes: {len(selected)}")
    if reject:
        print("\nReject reasons:")
        for k, v in sorted(reject.items(), key=lambda kv: (-kv[1], kv[0]))[:10]:
            print(f"  - {k:<28} {v}")
    print("\nTop selected routes:")
    for r in selected[:10]:
        print(
            f" {int(r.get('selection_rank')):2d}. {seq_text(r)} | robust={finite(r.get('robust_score')):.3f} | "
            f"obj={finite(r.get('objective')):.3f} | TOF={finite(r.get('total_tof_days')):.1f} d | "
            f"v∞mis={finite(r.get('max_vinf_mismatch_km_s'))*1000:.2f} m/s | "
            f"turn={finite(r.get('max_turn_angle_deg')):.2f}° | rp_margin={finite(r.get('min_rp_margin_km')):.1f} km"
        )
    print("=" * 80)
    print(f"[OK] wrote CSV:       {args.output_csv}")
    print(f"[OK] wrote JSONL:     {args.output_jsonl}")
    print(f"[OK] wrote JSON:      {args.output_json}")
    print(f"[OK] wrote best JSON: {args.output_best_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
