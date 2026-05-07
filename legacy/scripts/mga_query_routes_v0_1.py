#!/usr/bin/env python3
"""
MGA Route Query V0.1

Fast user-facing query for the precomputed route atlas.
Filters by target, route class, TOF, C3, patch correction, flyby margin, and rocket limits.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional


def fnum(x: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if x is None or x == "":
            return default
        v = float(x)
        return v if math.isfinite(v) else default
    except Exception:
        return default


def read_rocket(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}
    return json.loads(Path(path).read_text(encoding="utf-8"))


def class_rank(cls: Optional[str]) -> int:
    order = {"A": 0, "B6D": 1, "B": 2, "C6D": 3, "C": 4, "D": 9}
    return order.get(str(cls), 5)


def feasibility(row: Dict[str, Any], rocket: Dict[str, Any]) -> Dict[str, Any]:
    flags: List[str] = []
    ok = True

    c3_max = fnum(rocket.get("max_c3_km2_s2"))
    if c3_max is not None and row.get("c3_km2_s2") is not None:
        if row["c3_km2_s2"] > c3_max:
            ok = False
            flags.append("c3_exceeds_rocket")

    patch_budget = fnum(rocket.get("correction_dv_budget_m_s"))
    if patch_budget is not None and row.get("patch_dv_m_s") is not None:
        if row["patch_dv_m_s"] > patch_budget:
            ok = False
            flags.append("patch_dv_exceeds_budget")

    final_vinf_max = fnum(rocket.get("max_arrival_vinf_m_s"))
    if final_vinf_max is not None and row.get("final_vinf_m_s") is not None:
        if row["final_vinf_m_s"] > final_vinf_max:
            flags.append("arrival_vinf_high")
            if rocket.get("target_type") in ("capture", "rendezvous"):
                ok = False

    min_margin = fnum(rocket.get("min_flyby_margin_km"))
    if min_margin is not None and row.get("min_rp_margin_km") is not None:
        if row["min_rp_margin_km"] < min_margin:
            ok = False
            flags.append("flyby_margin_below_vehicle_policy")

    return {"rocket_feasible": ok, "rocket_flags": ";".join(flags)}


def adjusted_score(row: Dict[str, Any], rocket: Dict[str, Any]) -> float:
    base = fnum(row.get("score"), 0.0) or 0.0
    score = base
    score += class_rank(row.get("class")) * 3.0
    if row.get("patch_dv_m_s") is not None:
        score += 0.05 * float(row["patch_dv_m_s"])
    if row.get("intermediate_velocity_m_s") is not None:
        score += 0.01 * float(row["intermediate_velocity_m_s"])
    if row.get("min_rp_margin_km") is not None:
        soft = fnum(rocket.get("preferred_flyby_margin_km"), 1500.0) or 1500.0
        if row["min_rp_margin_km"] < soft:
            score += (soft - row["min_rp_margin_km"]) / 200.0
    if row.get("tof_days") is not None:
        tof_weight = fnum(rocket.get("tof_score_weight"), 0.0) or 0.0
        score += tof_weight * float(row["tof_days"])
    return score


def rows_to_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        Path(path).write_text("", encoding="utf-8")
        return
    cols = [c for c in rows[0].keys() if c != "packet_json"]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c) for c in cols})


def main() -> int:
    ap = argparse.ArgumentParser(description="Query a precomputed MGA route database.")
    ap.add_argument("--db", required=True)
    ap.add_argument("--rocket-json")
    ap.add_argument("--origin")
    ap.add_argument("--target", required=True)
    ap.add_argument("--allowed-classes", nargs="*", default=["A", "B6D", "B", "C6D"])
    ap.add_argument("--max-tof-days", type=float)
    ap.add_argument("--max-c3", type=float)
    ap.add_argument("--max-patch-dv-m-s", type=float)
    ap.add_argument("--min-rp-margin-km", type=float)
    ap.add_argument("--sequence-contains", nargs="*")
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--output-csv")
    ap.add_argument("--output-json")
    args = ap.parse_args()

    rocket = read_rocket(args.rocket_json)
    params: List[Any] = [args.target]
    where = ["target = ?"]

    if args.origin:
        where.append("origin = ?")
        params.append(args.origin)
    if args.allowed_classes:
        q = ",".join("?" for _ in args.allowed_classes)
        where.append(f"class IN ({q})")
        params.extend(args.allowed_classes)
    if args.max_tof_days is not None:
        where.append("(tof_days IS NULL OR tof_days <= ?)")
        params.append(args.max_tof_days)
    if args.max_c3 is not None:
        where.append("(c3_km2_s2 IS NULL OR c3_km2_s2 <= ?)")
        params.append(args.max_c3)
    if args.max_patch_dv_m_s is not None:
        where.append("(patch_dv_m_s IS NULL OR patch_dv_m_s <= ?)")
        params.append(args.max_patch_dv_m_s)
    if args.min_rp_margin_km is not None:
        where.append("(min_rp_margin_km IS NULL OR min_rp_margin_km >= ?)")
        params.append(args.min_rp_margin_km)
    if args.sequence_contains:
        for token in args.sequence_contains:
            where.append("sequence LIKE ?")
            params.append(f"%{token}%")

    sql = "SELECT * FROM routes WHERE " + " AND ".join(where)
    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row
    rows = [dict(r) for r in conn.execute(sql, params).fetchall()]

    out: List[Dict[str, Any]] = []
    for r in rows:
        feas = feasibility(r, rocket)
        if not feas["rocket_feasible"]:
            continue
        r.update(feas)
        r["adjusted_score"] = adjusted_score(r, rocket)
        out.append(r)

    out.sort(key=lambda r: (r.get("adjusted_score", 1e99), r.get("score") or 1e99))
    out = out[: args.limit]

    print("=" * 80)
    print("MGA ROUTE QUERY V0.1")
    print("=" * 80)
    print(f"DB:       {args.db}")
    print(f"Target:   {args.target}")
    print(f"Matched:  {len(rows)} before rocket feasibility")
    print(f"Returned: {len(out)}")
    print("Top routes:")
    for i, r in enumerate(out, start=1):
        print(
            f" {i}. {r['sequence']} | class={r['class']} | adj={r['adjusted_score']:.3f} | "
            f"score={r.get('score')} | TOF={r.get('tof_days')} d | "
            f"patch={r.get('patch_dv_m_s')} m/s | rpM={r.get('min_rp_margin_km')} km | "
            f"flags={r.get('risk_flags') or '-'}"
        )
    print("=" * 80)

    if args.output_csv:
        rows_to_csv(args.output_csv, out)
    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_json).write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
