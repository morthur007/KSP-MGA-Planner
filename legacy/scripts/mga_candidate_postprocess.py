#!/usr/bin/env python3
"""
mga_candidate_postprocess.py

V0.1 post-processor for spice_lambert_scout.py outputs.

Purpose
-------
The Lambert scout intentionally over-generates single-leg candidates. This tool
turns the raw top-N-per-target list into a diversity-preserving leg-seed archive
for the first MGA beam-search layer.

It performs three operations:
  1. cluster Lambert candidates by target, departure epoch and time-of-flight;
  2. keep the best candidate(s) per cluster;
  3. write a compact CSV plus JSONL leg-seed records suitable for route expansion.

Design notes
------------
This is still planning-grade. It does not validate flybys, does not propagate a
spacecraft in N-body, and does not claim targeting quality. It only ensures that
beam search is not fed 250 near-duplicates from the same Lambert valley.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

SECONDS_PER_DAY = 86_400.0
SCHEMA_VERSION = "mga_candidate_postprocess.v0.1"
LEG_SEED_SCHEMA_VERSION = "mga_leg_seed.v0.1"


@dataclass(frozen=True)
class LegSeed:
    schema_version: str
    seed_id: str
    source_leg_id: str
    origin: str
    target: str
    central_body: str
    ref_frame: str
    depart_et: float
    arrive_et: float
    tof_s: float
    depart_days_from_coverage_start: Optional[float]
    arrive_days_from_coverage_start: Optional[float]
    tof_days: float
    vinf_depart_km_s: float
    vinf_arrive_km_s: float
    c3_km2_s2: float
    score: float
    cw: bool
    solution_index: int
    max_revs: int
    policy_grade: str
    policy_requires_revalidation: bool
    policy_reason: str
    origin_r_km: Tuple[float, float, float]
    origin_v_km_s: Tuple[float, float, float]
    target_r_km: Tuple[float, float, float]
    target_v_km_s: Tuple[float, float, float]
    sc_v_depart_km_s: Tuple[float, float, float]
    sc_v_arrive_km_s: Tuple[float, float, float]
    cluster_id: str
    cluster_key: Mapping[str, Any]
    cluster_rank: int
    target_rank: int
    tags: Tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class RouteGenome:
    """Minimal route contract for the next beam-search milestone."""

    schema_version: str
    route_id: str
    sequence: Tuple[str, ...]
    leg_seed_ids: Tuple[str, ...]
    depart_et: float
    arrive_et: float
    tof_s_by_leg: Tuple[float, ...]
    total_tof_s: float
    nominal_score: float
    requires_local_revalidation: bool
    tags: Tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class BeamNode:
    """Serializable beam-search frontier node."""

    schema_version: str
    node_id: str
    route: RouteGenome
    last_body: str
    last_arrival_et: float
    depth: int
    parent_node_id: Optional[str]
    lower_bound_score: float
    dominance_bucket: str
    status: str = "open"


def parse_float(row: Mapping[str, str], key: str, default: Optional[float] = None) -> Optional[float]:
    value = row.get(key, "")
    if value is None or value == "":
        return default
    try:
        x = float(value)
    except ValueError:
        return default
    if not math.isfinite(x):
        return default
    return x


def parse_int(row: Mapping[str, str], key: str, default: int = 0) -> int:
    value = row.get(key, "")
    if value is None or value == "":
        return default
    try:
        return int(float(value))
    except ValueError:
        return default


def parse_bool(row: Mapping[str, str], key: str, default: bool = False) -> bool:
    value = str(row.get(key, "")).strip().lower()
    if value in {"1", "true", "t", "yes", "y"}:
        return True
    if value in {"0", "false", "f", "no", "n"}:
        return False
    return default


def parse_vec(row: Mapping[str, str], prefix: str) -> Tuple[float, float, float]:
    return (
        float(parse_float(row, f"{prefix}_x", 0.0) or 0.0),
        float(parse_float(row, f"{prefix}_y", 0.0) or 0.0),
        float(parse_float(row, f"{prefix}_z", 0.0) or 0.0),
    )


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return [dict(row) for row in csv.DictReader(f)]


def safe_bin(value: Optional[float], bin_size: float) -> int:
    if value is None:
        return 0
    if bin_size <= 0:
        return 0
    return int(math.floor(value / bin_size))


def cluster_key(row: Mapping[str, str], depart_bin_days: float, tof_bin_days: float, arrival_bin_days: float) -> Tuple[Any, ...]:
    target = row.get("target", "")
    dep = parse_float(row, "depart_days_from_coverage_start", parse_float(row, "depart_et", 0.0))
    tof = parse_float(row, "tof_days", 0.0)
    arr = parse_float(row, "arrive_days_from_coverage_start", None)
    key: List[Any] = [target, safe_bin(dep, depart_bin_days), safe_bin(tof, tof_bin_days)]
    if arrival_bin_days > 0:
        key.append(safe_bin(arr, arrival_bin_days))
    return tuple(key)


def make_cluster_id(key: Tuple[Any, ...]) -> str:
    return "cl_" + "_".join(str(x).replace(".", "p") for x in key)


def row_score(row: Mapping[str, str]) -> float:
    score = parse_float(row, "score", None)
    if score is not None:
        return score
    vinf_dep = parse_float(row, "vinf_depart_km_s", 1.0e9) or 1.0e9
    vinf_arr = parse_float(row, "vinf_arrive_km_s", 1.0e9) or 1.0e9
    tof_days = parse_float(row, "tof_days", 0.0) or 0.0
    return vinf_dep + 0.35 * vinf_arr + 0.05 * (tof_days / 365.25)


def enrich_rows(
    rows: Sequence[Dict[str, str]],
    *,
    depart_bin_days: float,
    tof_bin_days: float,
    arrival_bin_days: float,
    top_n_per_cluster: int,
    top_n_per_target: int,
) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[Any, ...], List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[cluster_key(row, depart_bin_days, tof_bin_days, arrival_bin_days)].append(row)

    clustered: List[Dict[str, Any]] = []
    for key, group in groups.items():
        group_sorted = sorted(group, key=lambda r: (row_score(r), parse_float(r, "depart_et", 0.0) or 0.0, parse_float(r, "tof_s", 0.0) or 0.0))
        keep = group_sorted if top_n_per_cluster <= 0 else group_sorted[:top_n_per_cluster]
        for rank, row in enumerate(keep, start=1):
            out: Dict[str, Any] = dict(row)
            out["cluster_id"] = make_cluster_id(key)
            out["cluster_rank"] = rank
            out["cluster_size"] = len(group)
            out["cluster_key_target"] = key[0]
            out["cluster_key_depart_bin"] = key[1]
            out["cluster_key_tof_bin"] = key[2]
            if len(key) > 3:
                out["cluster_key_arrival_bin"] = key[3]
            clustered.append(out)

    by_target: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in clustered:
        by_target[str(row.get("target", ""))].append(row)

    selected: List[Dict[str, Any]] = []
    for target, target_rows in by_target.items():
        target_sorted = sorted(target_rows, key=lambda r: (row_score(r), parse_float(r, "depart_et", 0.0) or 0.0, parse_float(r, "tof_s", 0.0) or 0.0))
        keep = target_sorted if top_n_per_target <= 0 else target_sorted[:top_n_per_target]
        for target_rank, row in enumerate(keep, start=1):
            row = dict(row)
            row["target_rank"] = target_rank
            selected.append(row)

    return sorted(selected, key=lambda r: (str(r.get("target", "")), int(r.get("target_rank", 0))))


def to_leg_seed(row: Mapping[str, Any], *, depart_bin_days: float, tof_bin_days: float, arrival_bin_days: float) -> LegSeed:
    source_leg_id = str(row.get("leg_id", ""))
    seed_id = f"seed_{source_leg_id}"
    dep_days = parse_float(row, "depart_days_from_coverage_start", None)
    arr_days = parse_float(row, "arrive_days_from_coverage_start", None)
    tof_days = float(parse_float(row, "tof_days", 0.0) or 0.0)
    key = {
        "target": row.get("cluster_key_target", row.get("target", "")),
        "depart_bin_days": depart_bin_days,
        "depart_bin_index": parse_int(row, "cluster_key_depart_bin", 0),
        "tof_bin_days": tof_bin_days,
        "tof_bin_index": parse_int(row, "cluster_key_tof_bin", 0),
    }
    if arrival_bin_days > 0:
        key["arrival_bin_days"] = arrival_bin_days
        key["arrival_bin_index"] = parse_int(row, "cluster_key_arrival_bin", 0)

    tags: List[str] = ["lambert", "single_leg", "global_seed"]
    if parse_bool(row, "policy_requires_revalidation", False):
        tags.append("requires_local_revalidation")

    return LegSeed(
        schema_version=LEG_SEED_SCHEMA_VERSION,
        seed_id=seed_id,
        source_leg_id=source_leg_id,
        origin=str(row.get("origin", "")),
        target=str(row.get("target", "")),
        central_body=str(row.get("central_body", "")),
        ref_frame=str(row.get("ref_frame", "J2000")),
        depart_et=float(parse_float(row, "depart_et", 0.0) or 0.0),
        arrive_et=float(parse_float(row, "arrive_et", 0.0) or 0.0),
        tof_s=float(parse_float(row, "tof_s", tof_days * SECONDS_PER_DAY) or tof_days * SECONDS_PER_DAY),
        depart_days_from_coverage_start=dep_days,
        arrive_days_from_coverage_start=arr_days,
        tof_days=tof_days,
        vinf_depart_km_s=float(parse_float(row, "vinf_depart_km_s", 0.0) or 0.0),
        vinf_arrive_km_s=float(parse_float(row, "vinf_arrive_km_s", 0.0) or 0.0),
        c3_km2_s2=float(parse_float(row, "c3_km2_s2", 0.0) or 0.0),
        score=float(parse_float(row, "score", 0.0) or 0.0),
        cw=parse_bool(row, "cw", False),
        solution_index=parse_int(row, "solution_index", 0),
        max_revs=parse_int(row, "max_revs", 0),
        policy_grade=str(row.get("policy_grade", "unknown")),
        policy_requires_revalidation=parse_bool(row, "policy_requires_revalidation", False),
        policy_reason=str(row.get("policy_reason", "")),
        origin_r_km=parse_vec(row, "origin_r_km"),
        origin_v_km_s=parse_vec(row, "origin_v_km_s"),
        target_r_km=parse_vec(row, "target_r_km"),
        target_v_km_s=parse_vec(row, "target_v_km_s"),
        sc_v_depart_km_s=parse_vec(row, "sc_v_depart_km_s"),
        sc_v_arrive_km_s=parse_vec(row, "sc_v_arrive_km_s"),
        cluster_id=str(row.get("cluster_id", "")),
        cluster_key=key,
        cluster_rank=parse_int(row, "cluster_rank", 0),
        target_rank=parse_int(row, "target_rank", 0),
        tags=tuple(tags),
    )


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with path.open("w", newline="", encoding="utf-8") as f:
            f.write("")
        return
    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")


def write_jsonl(path: Path, seeds: Sequence[LegSeed]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for seed in seeds:
            f.write(json.dumps(asdict(seed), ensure_ascii=False, separators=(",", ":")))
            f.write("\n")


def summarize(input_rows: Sequence[Mapping[str, str]], selected_rows: Sequence[Mapping[str, Any]], seeds: Sequence[LegSeed], args: argparse.Namespace) -> Dict[str, Any]:
    input_by_target: Dict[str, int] = defaultdict(int)
    selected_by_target: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in input_rows:
        input_by_target[str(row.get("target", ""))] += 1
    for row in selected_rows:
        selected_by_target[str(row.get("target", ""))].append(row)

    per_target: Dict[str, Any] = {}
    for target in sorted(set(input_by_target) | set(selected_by_target)):
        rows = sorted(selected_by_target.get(target, []), key=row_score)
        best = rows[0] if rows else None
        per_target[target] = {
            "input_count": input_by_target.get(target, 0),
            "selected_count": len(rows),
            "cluster_count_selected": len({str(r.get("cluster_id", "")) for r in rows}),
            "best_score": None if best is None else row_score(best),
            "best_depart_days_from_coverage_start": None if best is None else parse_float(best, "depart_days_from_coverage_start", None),
            "best_tof_days": None if best is None else parse_float(best, "tof_days", None),
            "best_vinf_depart_km_s": None if best is None else parse_float(best, "vinf_depart_km_s", None),
            "requires_local_revalidation_count": sum(1 for r in rows if parse_bool(r, "policy_requires_revalidation", False)),
        }

    return {
        "schema_version": SCHEMA_VERSION,
        "purpose": "diversity-preserving postprocess of Lambert scout candidates for MGA beam-search seeding",
        "inputs": {
            "input_csv": str(args.input_csv),
            "input_summary": str(args.input_summary) if args.input_summary else None,
        },
        "outputs": {
            "output_csv": str(args.output_csv),
            "output_json": str(args.output_json),
            "output_jsonl": str(args.output_jsonl),
        },
        "clustering": {
            "depart_bin_days": args.depart_bin_days,
            "tof_bin_days": args.tof_bin_days,
            "arrival_bin_days": args.arrival_bin_days,
            "top_n_per_cluster": args.top_n_per_cluster,
            "top_n_per_target": args.top_n_per_target,
        },
        "counts": {
            "input_rows": len(input_rows),
            "selected_rows": len(selected_rows),
            "leg_seeds": len(seeds),
        },
        "per_target": per_target,
        "next_contracts": {
            "leg_seed_schema_version": LEG_SEED_SCHEMA_VERSION,
            "route_genome_fields": list(RouteGenome.__dataclass_fields__.keys()),
            "beam_node_fields": list(BeamNode.__dataclass_fields__.keys()),
        },
    }


def load_input_summary(path: Optional[Path]) -> Dict[str, Any]:
    if path is None:
        return {}
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Cluster and serialize Lambert scout candidates for MGA beam-search seeding.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input-csv", required=True, type=Path, help="CSV from spice_lambert_scout.py")
    parser.add_argument("--input-summary", default=None, type=Path, help="Optional JSON summary from spice_lambert_scout.py")
    parser.add_argument("--output-csv", required=True, type=Path, help="Diversity-pruned candidate CSV")
    parser.add_argument("--output-json", required=True, type=Path, help="Postprocess summary JSON")
    parser.add_argument("--output-jsonl", required=True, type=Path, help="JSONL leg-seed archive for beam search")
    parser.add_argument("--depart-bin-days", type=float, default=40.0, help="Cluster bin size for departure epoch")
    parser.add_argument("--tof-bin-days", type=float, default=60.0, help="Cluster bin size for time of flight")
    parser.add_argument("--arrival-bin-days", type=float, default=0.0, help="Optional cluster bin size for arrival epoch; 0 disables it")
    parser.add_argument("--top-n-per-cluster", type=int, default=1, help="Candidates retained per cluster; 0 keeps all in each cluster")
    parser.add_argument("--top-n-per-target", type=int, default=80, help="Candidates retained per target after clustering; 0 keeps all")
    args = parser.parse_args(argv)

    if args.depart_bin_days <= 0:
        parser.error("--depart-bin-days must be positive")
    if args.tof_bin_days <= 0:
        parser.error("--tof-bin-days must be positive")
    if args.arrival_bin_days < 0:
        parser.error("--arrival-bin-days must be >= 0")

    input_rows = read_csv_rows(args.input_csv)
    _input_summary = load_input_summary(args.input_summary)
    selected_rows = enrich_rows(
        input_rows,
        depart_bin_days=args.depart_bin_days,
        tof_bin_days=args.tof_bin_days,
        arrival_bin_days=args.arrival_bin_days,
        top_n_per_cluster=args.top_n_per_cluster,
        top_n_per_target=args.top_n_per_target,
    )
    seeds = [
        to_leg_seed(
            row,
            depart_bin_days=args.depart_bin_days,
            tof_bin_days=args.tof_bin_days,
            arrival_bin_days=args.arrival_bin_days,
        )
        for row in selected_rows
    ]
    write_csv(args.output_csv, selected_rows)
    write_jsonl(args.output_jsonl, seeds)
    write_json(args.output_json, summarize(input_rows, selected_rows, seeds, args))

    print(f"[OK] input rows: {len(input_rows)}")
    print(f"[OK] selected rows: {len(selected_rows)}")
    print(f"[OK] wrote CSV: {args.output_csv}")
    print(f"[OK] wrote JSONL seeds: {args.output_jsonl}")
    print(f"[OK] wrote summary: {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
