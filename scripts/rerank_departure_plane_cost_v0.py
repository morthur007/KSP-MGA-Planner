#!/usr/bin/env python3
"""
rerank_departure_plane_cost_v0.py

Post-rank a snapshot executability JSON/CSV by explicitly penalizing departure
out-of-plane cost in the actual parking-orbit TNB burn.

This is meant to sit after rank_pykep_candidates_by_snapshot_executability_v0.py
and before Principia VBATCH/VBPLANE targeting.

It does not change the physics. It changes the ordering so candidates that need
large normal/binormal burn from the current vessel orbit do not dominate purely
because their patched-conics/PyKEP raw route score looked good.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


def as_float(x: Any, default: float | None = None) -> float | None:
    if x is None:
        return default
    try:
        return float(x)
    except Exception:
        return default


def get_tnb(row: dict[str, Any]) -> tuple[float | None, float | None, float | None]:
    # Common scalar names.
    t = as_float(row.get("dv_tangent_m_s", row.get("dvt_m_s")))
    n = as_float(row.get("dv_normal_m_s", row.get("dvn_m_s")))
    b = as_float(row.get("dv_binormal_m_s", row.get("dvb_m_s")))

    # Navigation vector fallback.
    nav = row.get("dv_navigation_m_s") or row.get("delta_v_navigation_m_s")
    if isinstance(nav, list) and len(nav) >= 3:
        t = t if t is not None else as_float(nav[0])
        n = n if n is not None else as_float(nav[1])
        b = b if b is not None else as_float(nav[2])

    return t, n, b


def score_row(row: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    t, n, b = get_tnb(row)
    out = dict(row)
    if t is None or n is None or b is None:
        out["plane_cost_status"] = "missing_tnb"
        out["plane_cost_score_m_s"] = float("inf")
        return out

    t_abs = abs(t)
    n_abs = abs(n)
    b_abs = abs(b)
    oop = math.hypot(n, b)
    dv_norm = as_float(row.get("dv_norm_m_s"))
    if dv_norm is None or not math.isfinite(dv_norm) or dv_norm <= 0:
        dv_norm = math.sqrt(t*t + n*n + b*b)

    plane_angle_deg = math.degrees(math.atan2(oop, max(t_abs, 1e-12)))
    oop_fraction = oop / max(dv_norm, 1e-12)

    # Base score is the actual burn magnitude. Extras intentionally penalize
    # off-plane components because they consume steering authority and usually
    # imply the patched-conics seed is poorly aligned with the live orbit.
    score = (
        args.dv_weight * dv_norm
        + args.normal_extra_weight * n_abs
        + args.binormal_extra_weight * b_abs
        + args.oop_extra_weight * oop
    )

    # Soft-limit penalties. Quadratic in metres per second, converted back to a
    # m/s-like score by dividing by the soft scale. This keeps the ordering
    # interpretable while strongly demoting candidates past the threshold.
    def soft_quad(value: float, soft: float, weight: float) -> float:
        if soft <= 0 or value <= soft:
            return 0.0
        over = value - soft
        return weight * (over * over) / max(soft, 1.0)

    score += soft_quad(n_abs, args.normal_soft_m_s, args.normal_over_weight)
    score += soft_quad(b_abs, args.binormal_soft_m_s, args.binormal_over_weight)
    score += soft_quad(oop, args.oop_soft_m_s, args.oop_over_weight)

    if args.max_oop_fraction > 0 and oop_fraction > args.max_oop_fraction:
        over = oop_fraction - args.max_oop_fraction
        score += args.oop_fraction_over_weight * over * over * 1000.0
    if args.max_plane_angle_deg > 0 and plane_angle_deg > args.max_plane_angle_deg:
        over = plane_angle_deg - args.max_plane_angle_deg
        score += args.plane_angle_over_weight * over * over

    # Optional route quality term if present. Keep this small; route search score
    # and live-orbit executability are not the same unit.
    for key in args.route_score_keys.split(","):
        key = key.strip()
        if not key:
            continue
        v = as_float(row.get(key))
        if v is not None and math.isfinite(v):
            score += args.route_score_weight * v
            out["plane_cost_route_score_key"] = key
            break

    out.update({
        "dv_tangent_m_s": t,
        "dv_normal_m_s": n,
        "dv_binormal_m_s": b,
        "dv_norm_m_s": dv_norm,
        "departure_oop_m_s": oop,
        "departure_oop_fraction": oop_fraction,
        "departure_plane_angle_deg": plane_angle_deg,
        "plane_cost_score_m_s": score,
        "plane_cost_status": "ok",
    })
    return out


def find_candidate_lists(x: Any) -> list[list[dict[str, Any]]]:
    lists: list[list[dict[str, Any]]] = []
    def walk(obj: Any):
        if isinstance(obj, list):
            if obj and all(isinstance(v, dict) for v in obj):
                # A candidate list if at least one dict has TNB or row_index0.
                if any(
                    ("row_index0" in v or get_tnb(v) != (None, None, None))
                    for v in obj
                ):
                    lists.append(obj)  # type: ignore[arg-type]
            for v in obj:
                walk(v)
        elif isinstance(obj, dict):
            for v in obj.values():
                walk(v)
    walk(x)
    return lists


def rerank_json(path: Path, out_json: Path, out_csv: Path | None, args: argparse.Namespace) -> None:
    data = json.loads(path.read_text())

    # Prefer common top-level lists; otherwise recursively rerank candidate lists.
    reranked_any = False
    for key in ("top", "candidates", "rows", "ranked", "results"):
        if isinstance(data, dict) and isinstance(data.get(key), list):
            rows = [score_row(r, args) if isinstance(r, dict) else r for r in data[key]]
            rows = [r for r in rows if isinstance(r, dict)]
            rows.sort(key=lambda r: as_float(r.get("plane_cost_score_m_s"), float("inf")))
            data[key] = rows[: args.top_n] if args.top_n > 0 else rows
            data["rerank_by_departure_plane_cost"] = vars(args)
            data["best_by_plane_cost"] = rows[0] if rows else None
            reranked_any = True
            csv_rows = rows
            break
    else:
        lists = find_candidate_lists(data)
        csv_rows = []
        for lst in lists:
            rows = [score_row(r, args) for r in lst]
            rows.sort(key=lambda r: as_float(r.get("plane_cost_score_m_s"), float("inf")))
            if args.top_n > 0:
                rows = rows[: args.top_n]
            lst[:] = rows
            csv_rows.extend(rows)
            reranked_any = True
        if isinstance(data, dict):
            data["rerank_by_departure_plane_cost"] = vars(args)
            csv_rows.sort(key=lambda r: as_float(r.get("plane_cost_score_m_s"), float("inf")))
            data["best_by_plane_cost"] = csv_rows[0] if csv_rows else None

    if not reranked_any:
        raise SystemExit("No candidate list found in JSON")

    out_json.write_text(json.dumps(data, indent=2, default=lambda o: str(o) if isinstance(o, Path) else o) + "\n")
    if out_csv is not None:
        write_csv(csv_rows, out_csv)


def rerank_csv(path: Path, out_csv: Path, args: argparse.Namespace) -> None:
    rows = []
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            rows.append(score_row(row, args))
    rows.sort(key=lambda r: as_float(r.get("plane_cost_score_m_s"), float("inf")))
    if args.top_n > 0:
        rows = rows[: args.top_n]
    write_csv(rows, out_csv)


def write_csv(rows: list[dict[str, Any]], out: Path) -> None:
    if not rows:
        out.write_text("")
        return
    keys = []
    seen = set()
    preferred = [
        "row_index0", "sequence", "plane_cost_score_m_s", "dv_norm_m_s",
        "dv_tangent_m_s", "dv_normal_m_s", "dv_binormal_m_s",
        "departure_oop_m_s", "departure_oop_fraction", "departure_plane_angle_deg",
        "ca_distance_km", "arrival_offset_days", "status",
    ]
    for k in preferred + [k for r in rows for k in r.keys()]:
        if k not in seen and any(k in r for r in rows):
            keys.append(k); seen.add(k)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output-json", type=Path)
    ap.add_argument("--output-csv", type=Path)
    ap.add_argument("--top-n", type=int, default=100)

    ap.add_argument("--dv-weight", type=float, default=1.0)
    ap.add_argument("--normal-extra-weight", type=float, default=1.0)
    ap.add_argument("--binormal-extra-weight", type=float, default=1.5)
    ap.add_argument("--oop-extra-weight", type=float, default=1.0)

    ap.add_argument("--normal-soft-m-s", type=float, default=150.0)
    ap.add_argument("--binormal-soft-m-s", type=float, default=250.0)
    ap.add_argument("--oop-soft-m-s", type=float, default=300.0)
    ap.add_argument("--normal-over-weight", type=float, default=2.0)
    ap.add_argument("--binormal-over-weight", type=float, default=3.0)
    ap.add_argument("--oop-over-weight", type=float, default=2.0)

    ap.add_argument("--max-oop-fraction", type=float, default=0.18)
    ap.add_argument("--oop-fraction-over-weight", type=float, default=3000.0)
    ap.add_argument("--max-plane-angle-deg", type=float, default=6.0)
    ap.add_argument("--plane-angle-over-weight", type=float, default=20.0)

    ap.add_argument("--route-score-keys", default="raw_sum,score,total_score,objective")
    ap.add_argument("--route-score-weight", type=float, default=0.0)
    args = ap.parse_args()

    if args.output_json is None and args.output_csv is None:
        suffix = args.input.suffix.lower()
        if suffix == ".json":
            args.output_json = args.input.with_name(args.input.stem + "_plane_reranked.json")
            args.output_csv = args.input.with_name(args.input.stem + "_plane_reranked.csv")
        else:
            args.output_csv = args.input.with_name(args.input.stem + "_plane_reranked.csv")

    if args.input.suffix.lower() == ".json":
        rerank_json(args.input, args.output_json, args.output_csv, args)
        print("wrote", args.output_json)
        if args.output_csv:
            print("wrote", args.output_csv)
    else:
        rerank_csv(args.input, args.output_csv, args)
        print("wrote", args.output_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
