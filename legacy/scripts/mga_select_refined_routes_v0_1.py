#!/usr/bin/env python3
"""
mga_select_refined_routes_v0_1.py

Select robust, diverse fixed-sequence PyGMO refined MGA candidates.

This is a planning-grade postprocessor between:
  mga_pygmo_refine_fixed_sequence_v0_3.py
and later B-plane / REBOUND / Tudat validation.

It intentionally does not solve dynamics. It filters and packages refined
Lambert-flyby candidates into a compact route packet for the next stage.

Example:
  python mga_select_refined_routes_v0_1.py \
    --input-jsonl data/mga_v0_1/mga_refined_kdj_flyby_v0_3.jsonl \
    --min-rp-margin-km 150 \
    --max-vinf-mag-jump 0.25 \
    --max-flyby-layover-days 3 \
    --depart-bin-days 30 \
    --tof-bin-days 30 \
    --top-n 20 \
    --output-csv data/mga_v0_1/mga_refined_kdj_selected_v0_1.csv \
    --output-jsonl data/mga_v0_1/mga_refined_kdj_selected_v0_1.jsonl \
    --output-json data/mga_v0_1/mga_refined_kdj_selected_v0_1.summary.json \
    --output-packet-json data/mga_v0_1/mga_refined_kdj_route_packet_v0_1.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

SCHEMA_VERSION = "mga_select_refined_routes.v0.1"


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_no}: {exc}") from exc
            if isinstance(row, Mapping):
                out.append(dict(row))
    return out


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=json_default)
        f.write("\n")


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":"), default=json_default))
            f.write("\n")


def json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, tuple):
        return list(obj)
    return str(obj)


def as_float(x: Any, default: float = math.nan) -> float:
    try:
        y = float(x)
    except (TypeError, ValueError):
        return default
    return y if math.isfinite(y) else default


def as_bool(x: Any, default: bool = False) -> bool:
    if isinstance(x, bool):
        return x
    if x is None:
        return default
    s = str(x).strip().lower()
    if s in {"1", "true", "yes", "y", "ok"}:
        return True
    if s in {"0", "false", "no", "n", "bad"}:
        return False
    return default


def seq_text(route: Mapping[str, Any]) -> str:
    seq = route.get("sequence", [])
    if isinstance(seq, str):
        if "->" in seq:
            return " -> ".join([x.strip() for x in seq.split("->") if x.strip()])
        if "," in seq:
            return " -> ".join([x.strip() for x in seq.split(",") if x.strip()])
        return seq
    if isinstance(seq, Sequence):
        return " -> ".join(str(x) for x in seq)
    return str(seq)


def depart_day(route: Mapping[str, Any]) -> float:
    labels = route.get("decision_labels", [])
    vector = route.get("decision_vector", [])
    if isinstance(labels, Sequence) and isinstance(vector, Sequence):
        for i, lab in enumerate(labels):
            if str(lab) == "depart_day" and i < len(vector):
                return as_float(vector[i])
    if isinstance(vector, Sequence) and vector:
        return as_float(vector[0])
    dep_et = as_float(route.get("depart_et"))
    return dep_et / 86400.0 if math.isfinite(dep_et) else math.nan


def layovers(route: Mapping[str, Any]) -> List[float]:
    out: List[float] = []
    for fb in route.get("flyby_evals", []) or []:
        if isinstance(fb, Mapping):
            x = as_float(fb.get("layover_days"))
            if math.isfinite(x):
                out.append(x)
    return out


def flyby_bodies(route: Mapping[str, Any]) -> List[str]:
    out: List[str] = []
    for fb in route.get("flyby_evals", []) or []:
        if isinstance(fb, Mapping):
            out.append(str(fb.get("body", "")))
    return out


def min_list(values: Iterable[float], default: Optional[float] = None) -> Optional[float]:
    vals = [v for v in values if math.isfinite(v)]
    return min(vals) if vals else default


def robust_score(route: Mapping[str, Any], args: argparse.Namespace) -> float:
    """Lower is better. Penalize thin margins and route operational complexity."""
    obj = as_float(route.get("objective"), 1.0e12)
    rp = as_float(route.get("min_rp_margin_km"), -1.0e9)
    turn_margin = as_float(route.get("min_turn_angle_margin_deg"), 0.0)
    vinf_jump = as_float(route.get("max_vinf_mag_jump_km_s"), 1.0e9)
    tof = as_float(route.get("total_tof_days"), 0.0)
    total_lay = as_float(route.get("total_layover_days"), 0.0)

    score = obj
    if math.isfinite(rp):
        if rp < args.rp_soft_margin_km:
            score += args.rp_penalty_weight * ((args.rp_soft_margin_km - rp) / max(args.rp_soft_margin_km, 1.0)) ** 2
        else:
            # Small reward for extra clearance, saturated so it does not dominate delta-v.
            score -= args.rp_clearance_reward * min(rp / max(args.rp_soft_margin_km, 1.0), 5.0)
    if math.isfinite(turn_margin) and turn_margin < args.turn_soft_margin_deg:
        score += args.turn_penalty_weight * ((args.turn_soft_margin_deg - turn_margin) / max(args.turn_soft_margin_deg, 1.0)) ** 2
    if math.isfinite(vinf_jump):
        score += args.vinf_jump_weight * vinf_jump
    if math.isfinite(tof):
        score += args.tof_weight * tof
    if math.isfinite(total_lay):
        score += args.layover_weight * total_lay
    return float(score)


def reject_reasons(route: Mapping[str, Any], args: argparse.Namespace) -> List[str]:
    reasons: List[str] = []
    if not as_bool(route.get("valid"), False):
        reasons.append("invalid")
    if str(route.get("status", "ok")) not in {"ok", "valid", ""}:
        reasons.append(f"status:{route.get('status')}")
    rp = as_float(route.get("min_rp_margin_km"))
    if not math.isfinite(rp) or rp < args.min_rp_margin_km:
        reasons.append("rp_margin")
    vinf_jump = as_float(route.get("max_vinf_mag_jump_km_s"))
    if math.isfinite(vinf_jump) and vinf_jump > args.max_vinf_mag_jump:
        reasons.append("vinf_mag_jump")
    turn = as_float(route.get("max_turn_angle_deg"))
    if math.isfinite(turn) and turn > args.max_turn_angle_deg:
        reasons.append("turn_angle")
    lays = layovers(route)
    if lays and max(lays) > args.max_flyby_layover_days:
        reasons.append("layover")
    total_tof = as_float(route.get("total_tof_days"))
    if math.isfinite(total_tof) and total_tof > args.max_total_tof_days:
        reasons.append("total_tof")
    return reasons


def bin_key(route: Mapping[str, Any], args: argparse.Namespace) -> Tuple[Any, ...]:
    d = depart_day(route)
    tof = as_float(route.get("total_tof_days"))
    seq = seq_text(route)
    d_bin = int(math.floor(d / args.depart_bin_days)) if math.isfinite(d) and args.depart_bin_days > 0 else 0
    t_bin = int(math.floor(tof / args.tof_bin_days)) if math.isfinite(tof) and args.tof_bin_days > 0 else 0
    fb = ",".join(flyby_bodies(route))
    return (seq, fb, d_bin, t_bin)


def flatten_route_for_csv(route: Mapping[str, Any]) -> Dict[str, Any]:
    labels = route.get("decision_labels", [])
    vector = route.get("decision_vector", [])
    decisions = {}
    if isinstance(labels, Sequence) and isinstance(vector, Sequence):
        decisions = {str(k): vector[i] for i, k in enumerate(labels) if i < len(vector)}
    fbs = route.get("flyby_evals", []) or []
    return {
        "selected_rank": route.get("selected_rank"),
        "refined_id": route.get("refined_id", ""),
        "source_route_id": route.get("source_route_id", ""),
        "sequence": seq_text(route),
        "valid": int(as_bool(route.get("valid"), False)),
        "status": route.get("status", ""),
        "objective": route.get("objective", ""),
        "robust_score": route.get("robust_score", ""),
        "source_nominal_score": route.get("source_nominal_score", ""),
        "improvement": route.get("improvement", ""),
        "depart_day": depart_day(route),
        "total_tof_days": route.get("total_tof_days", ""),
        "total_layover_days": route.get("total_layover_days", ""),
        "max_vinf_mag_jump_km_s": route.get("max_vinf_mag_jump_km_s", ""),
        "max_turn_angle_deg": route.get("max_turn_angle_deg", ""),
        "min_rp_margin_km": route.get("min_rp_margin_km", ""),
        "min_turn_angle_margin_deg": route.get("min_turn_angle_margin_deg", ""),
        "flyby_bodies": ";".join(flyby_bodies(route)),
        "flyby_layovers_days": ";".join(str(as_float(fb.get("layover_days"))) for fb in fbs if isinstance(fb, Mapping)),
        "flyby_rp_margins_km": ";".join(str(as_float(fb.get("rp_margin_km"))) for fb in fbs if isinstance(fb, Mapping)),
        "decision_vector": json.dumps(vector, separators=(",", ":"), ensure_ascii=True),
        "decision_labels": json.dumps(labels, separators=(",", ":"), ensure_ascii=True),
        **{f"x_{k}": v for k, v in decisions.items()},
    }


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    base_fields = [
        "selected_rank", "refined_id", "source_route_id", "sequence", "valid", "status",
        "objective", "robust_score", "source_nominal_score", "improvement", "depart_day",
        "total_tof_days", "total_layover_days", "max_vinf_mag_jump_km_s", "max_turn_angle_deg",
        "min_rp_margin_km", "min_turn_angle_margin_deg", "flyby_bodies", "flyby_layovers_days",
        "flyby_rp_margins_km", "decision_vector", "decision_labels",
    ]
    extra = sorted({k for r in rows for k in r.keys() if k.startswith("x_")})
    fieldnames = base_fields + extra
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


def make_route_packet(selected: Sequence[Mapping[str, Any]], args: argparse.Namespace) -> Dict[str, Any]:
    packet_routes: List[Dict[str, Any]] = []
    for r in selected:
        events: List[Dict[str, Any]] = []
        leg_evals = r.get("leg_evals", []) or []
        for i, leg in enumerate(leg_evals):
            if not isinstance(leg, Mapping):
                continue
            if i == 0:
                events.append({
                    "event_type": "departure",
                    "body": leg.get("origin"),
                    "et": leg.get("depart_et"),
                    "day": as_float(leg.get("depart_et")) / 86400.0,
                    "vinf_depart_km_s": leg.get("vinf_depart_km_s"),
                    "c3_km2_s2": leg.get("c3_km2_s2"),
                    "r_body_km": leg.get("origin_r_km"),
                    "v_body_km_s": leg.get("origin_v_km_s"),
                    "v_sc_depart_km_s": leg.get("sc_v_depart_km_s"),
                })
            events.append({
                "event_type": "arrival" if i == len(leg_evals) - 1 else "flyby_arrival",
                "body": leg.get("target"),
                "et": leg.get("arrive_et"),
                "day": as_float(leg.get("arrive_et")) / 86400.0,
                "tof_days": leg.get("tof_days"),
                "vinf_arrive_km_s": leg.get("vinf_arrive_km_s"),
                "r_body_km": leg.get("target_r_km"),
                "v_body_km_s": leg.get("target_v_km_s"),
                "v_sc_arrive_km_s": leg.get("sc_v_arrive_km_s"),
            })
        for fb in r.get("flyby_evals", []) or []:
            if not isinstance(fb, Mapping):
                continue
            events.append({
                "event_type": "flyby_gate",
                "body": fb.get("body"),
                "arrival_et": fb.get("arrival_et"),
                "depart_et": fb.get("depart_et"),
                "layover_days": fb.get("layover_days"),
                "turn_angle_deg": fb.get("turn_angle_deg"),
                "turn_angle_max_deg": fb.get("turn_angle_max_deg"),
                "rp_required_km": fb.get("rp_required_km"),
                "rp_margin_km": fb.get("rp_margin_km"),
                "vinf_in_km_s": fb.get("vinf_in_km_s"),
                "vinf_out_km_s": fb.get("vinf_out_km_s"),
                "status": fb.get("status"),
            })
        events.sort(key=lambda e: as_float(e.get("et", e.get("arrival_et", e.get("depart_et", 0.0))), 0.0))
        packet_routes.append({
            "refined_id": r.get("refined_id"),
            "selected_rank": r.get("selected_rank"),
            "sequence": r.get("sequence"),
            "objective": r.get("objective"),
            "robust_score": r.get("robust_score"),
            "valid": r.get("valid"),
            "status": r.get("status"),
            "depart_et": r.get("depart_et"),
            "arrive_et": r.get("arrive_et"),
            "total_tof_days": r.get("total_tof_days"),
            "total_layover_days": r.get("total_layover_days"),
            "decision_labels": r.get("decision_labels"),
            "decision_vector": r.get("decision_vector"),
            "metrics": {
                "max_vinf_depart_km_s": r.get("max_vinf_depart_km_s"),
                "max_vinf_arrive_km_s": r.get("max_vinf_arrive_km_s"),
                "max_vinf_mag_jump_km_s": r.get("max_vinf_mag_jump_km_s"),
                "max_turn_angle_deg": r.get("max_turn_angle_deg"),
                "min_rp_margin_km": r.get("min_rp_margin_km"),
                "min_turn_angle_margin_deg": r.get("min_turn_angle_margin_deg"),
            },
            "events": events,
        })
    return {
        "schema_version": SCHEMA_VERSION,
        "source": str(args.input_jsonl),
        "selection_policy": {
            "min_rp_margin_km": args.min_rp_margin_km,
            "rp_soft_margin_km": args.rp_soft_margin_km,
            "max_vinf_mag_jump": args.max_vinf_mag_jump,
            "max_flyby_layover_days": args.max_flyby_layover_days,
            "max_turn_angle_deg": args.max_turn_angle_deg,
            "depart_bin_days": args.depart_bin_days,
            "tof_bin_days": args.tof_bin_days,
            "top_n": args.top_n,
        },
        "routes": packet_routes,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input-jsonl", required=True, type=Path)
    p.add_argument("--min-rp-margin-km", type=float, default=100.0)
    p.add_argument("--rp-soft-margin-km", type=float, default=250.0)
    p.add_argument("--max-vinf-mag-jump", type=float, default=0.25)
    p.add_argument("--max-flyby-layover-days", type=float, default=3.0)
    p.add_argument("--max-turn-angle-deg", type=float, default=120.0)
    p.add_argument("--max-total-tof-days", type=float, default=5000.0)
    p.add_argument("--turn-soft-margin-deg", type=float, default=5.0)
    p.add_argument("--vinf-jump-weight", type=float, default=0.8)
    p.add_argument("--tof-weight", type=float, default=0.0005)
    p.add_argument("--layover-weight", type=float, default=0.02)
    p.add_argument("--rp-penalty-weight", type=float, default=2.0)
    p.add_argument("--rp-clearance-reward", type=float, default=0.05)
    p.add_argument("--turn-penalty-weight", type=float, default=1.0)
    p.add_argument("--depart-bin-days", type=float, default=30.0)
    p.add_argument("--tof-bin-days", type=float, default=30.0)
    p.add_argument("--top-n-per-bin", type=int, default=1)
    p.add_argument("--top-n", type=int, default=20)
    p.add_argument("--output-csv", required=True, type=Path)
    p.add_argument("--output-jsonl", required=True, type=Path)
    p.add_argument("--output-json", required=True, type=Path)
    p.add_argument("--output-packet-json", required=True, type=Path)
    args = p.parse_args(argv)

    routes = read_jsonl(args.input_jsonl)
    rejected: Dict[str, int] = {}
    eligible: List[Dict[str, Any]] = []
    for r0 in routes:
        r = dict(r0)
        reasons = reject_reasons(r, args)
        if reasons:
            for reason in reasons:
                rejected[reason] = rejected.get(reason, 0) + 1
            continue
        r["robust_score"] = robust_score(r, args)
        eligible.append(r)

    # Diversity: keep N per sequence/flyby/depart/tof bin, then global top N.
    buckets: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = {}
    for r in eligible:
        buckets.setdefault(bin_key(r, args), []).append(r)
    diverse: List[Dict[str, Any]] = []
    for key, rows in buckets.items():
        rows.sort(key=lambda x: as_float(x.get("robust_score"), 1.0e12))
        diverse.extend(rows[: max(args.top_n_per_bin, 1)])
    diverse.sort(key=lambda x: as_float(x.get("robust_score"), 1.0e12))
    selected = diverse[: max(args.top_n, 0)]
    for i, r in enumerate(selected, start=1):
        r["selected_rank"] = i

    csv_rows = [flatten_route_for_csv(r) for r in selected]
    write_csv(args.output_csv, csv_rows)
    write_jsonl(args.output_jsonl, selected)
    packet = make_route_packet(selected, args)
    write_json(args.output_packet_json, packet)

    summary = {
        "schema_version": SCHEMA_VERSION,
        "input_jsonl": str(args.input_jsonl),
        "input_routes": len(routes),
        "eligible_routes": len(eligible),
        "diverse_routes": len(diverse),
        "selected_routes": len(selected),
        "reject_reasons": dict(sorted(rejected.items(), key=lambda kv: (-kv[1], kv[0]))),
        "selection_policy": {
            "min_rp_margin_km": args.min_rp_margin_km,
            "rp_soft_margin_km": args.rp_soft_margin_km,
            "max_vinf_mag_jump": args.max_vinf_mag_jump,
            "max_flyby_layover_days": args.max_flyby_layover_days,
            "max_turn_angle_deg": args.max_turn_angle_deg,
            "depart_bin_days": args.depart_bin_days,
            "tof_bin_days": args.tof_bin_days,
            "top_n_per_bin": args.top_n_per_bin,
            "top_n": args.top_n,
        },
        "top_routes": [
            {
                "rank": r.get("selected_rank"),
                "refined_id": r.get("refined_id"),
                "sequence": seq_text(r),
                "objective": r.get("objective"),
                "robust_score": r.get("robust_score"),
                "depart_day": depart_day(r),
                "total_tof_days": r.get("total_tof_days"),
                "total_layover_days": r.get("total_layover_days"),
                "max_vinf_mag_jump_km_s": r.get("max_vinf_mag_jump_km_s"),
                "max_turn_angle_deg": r.get("max_turn_angle_deg"),
                "min_rp_margin_km": r.get("min_rp_margin_km"),
                "min_turn_angle_margin_deg": r.get("min_turn_angle_margin_deg"),
            }
            for r in selected[:10]
        ],
        "outputs": {
            "csv": str(args.output_csv),
            "jsonl": str(args.output_jsonl),
            "summary_json": str(args.output_json),
            "packet_json": str(args.output_packet_json),
        },
    }
    write_json(args.output_json, summary)

    print("=" * 80)
    print("MGA SELECT REFINED ROUTES V0.1")
    print("=" * 80)
    print(f"Input routes:      {len(routes)}")
    print(f"Eligible routes:   {len(eligible)}")
    print(f"Diverse routes:    {len(diverse)}")
    print(f"Selected routes:   {len(selected)}")
    if rejected:
        print("\nReject reasons:")
        for k, v in sorted(rejected.items(), key=lambda kv: (-kv[1], kv[0]))[:10]:
            print(f"  - {k:<20} {v}")
    print("\nTop selected routes:")
    for r in selected[:10]:
        print(
            f" {r.get('selected_rank'):>2}. {seq_text(r)} | robust={as_float(r.get('robust_score')):.4f} "
            f"| obj={as_float(r.get('objective')):.4f} | TOF={as_float(r.get('total_tof_days')):.1f} d "
            f"| lay={as_float(r.get('total_layover_days')):.2f} d "
            f"| Δv∞={as_float(r.get('max_vinf_mag_jump_km_s')):.3f} km/s "
            f"| turn={as_float(r.get('max_turn_angle_deg')):.1f}° "
            f"| rp_margin={as_float(r.get('min_rp_margin_km')):.1f} km"
        )
    print("=" * 80)
    print(f"[OK] wrote CSV:    {args.output_csv}")
    print(f"[OK] wrote JSONL:  {args.output_jsonl}")
    print(f"[OK] wrote JSON:   {args.output_json}")
    print(f"[OK] wrote packet: {args.output_packet_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
