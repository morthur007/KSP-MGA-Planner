#!/usr/bin/env python3
"""
mga_finalize_corrected_route_packet_v0_1.py

Freeze corrected stitched MGA routes into a compact, auditable route-candidate
packet for the next stage: higher-fidelity validation / dense local SPK / KSP
execution planning.

Input:
  JSON/JSONL from mga_stitched_patch_corrector_v0_1.py

Output:
  - CSV ranking
  - JSONL selected candidate packets
  - JSON summary
  - JSON best candidate packet

This does not integrate dynamics. It is a contract/manifest builder that checks
that the route already passed the stitched patch corrector and exports the
mission event timeline and correction budget in a stable schema.

Units:
  km, km/s, m/s, ET seconds, days.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

SCHEMA_VERSION = "mga_corrected_route_packet.v0.1"
SECONDS_PER_DAY = 86400.0


def finite(x: Any, default: float = math.nan) -> float:
    try:
        y = float(x)
    except (TypeError, ValueError):
        return default
    return y if math.isfinite(y) else default


def get_path(obj: Mapping[str, Any], path: Sequence[str], default: Any = None) -> Any:
    cur: Any = obj
    for p in path:
        if not isinstance(cur, Mapping):
            return default
        cur = cur.get(p)
    return cur if cur is not None else default


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


def stable_id(prefix: str, payload: Mapping[str, Any]) -> str:
    raw = json.dumps(sanitize(payload), sort_keys=True, ensure_ascii=True, separators=(",", ":"), default=str)
    return f"{prefix}_{hashlib.sha1(raw.encode('utf-8')).hexdigest()[:16]}"


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


def load_corrected(path: Path) -> List[Dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        return read_jsonl(path)
    data = load_json(path)
    if "pre_arc" in data and "post_arc" in data:
        return [data]
    for key in ("corrected_routes", "routes", "results", "records", "top_results"):
        rows = data.get(key)
        if isinstance(rows, list):
            return [dict(r) for r in rows if isinstance(r, Mapping)]
    raise ValueError(f"Could not find corrected route records in {path}")


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


def stats(vals: Sequence[float]) -> Dict[str, Optional[float]]:
    xs = sorted([v for v in vals if math.isfinite(v)])
    if not xs:
        return {"min": None, "median": None, "max": None}
    return {"min": xs[0], "median": xs[len(xs)//2], "max": xs[-1]}


def classify_packet(total_dv_m_s: float, rp_margin_km: float, entry_vel_m_s: float, arrival_after_km: float) -> str:
    if not all(math.isfinite(x) for x in (total_dv_m_s, rp_margin_km, entry_vel_m_s, arrival_after_km)):
        return "invalid"
    if arrival_after_km <= 10 and total_dv_m_s <= 25 and rp_margin_km >= 800 and entry_vel_m_s <= 25:
        return "A"
    if arrival_after_km <= 100 and total_dv_m_s <= 50 and rp_margin_km >= 300 and entry_vel_m_s <= 100:
        return "B"
    if arrival_after_km <= 1000 and total_dv_m_s <= 100 and rp_margin_km >= 50:
        return "C"
    return "D"


def build_event_timeline(rec: Mapping[str, Any]) -> List[Dict[str, Any]]:
    epochs = rec.get("epochs") if isinstance(rec.get("epochs"), Mapping) else {}
    pre = rec.get("pre_arc") if isinstance(rec.get("pre_arc"), Mapping) else {}
    post = rec.get("post_arc") if isinstance(rec.get("post_arc"), Mapping) else {}
    seq = str(rec.get("sequence") or "")
    flyby_body = str(rec.get("flyby_body") or "flyby")
    source_stitched = rec.get("source_stitched_packet") if isinstance(rec.get("source_stitched_packet"), Mapping) else {}
    target = source_stitched.get("target") if isinstance(source_stitched.get("target"), Mapping) else {}
    hyp = target.get("hyperbola") if isinstance(target.get("hyperbola"), Mapping) else {}
    bp = target.get("b_plane") if isinstance(target.get("b_plane"), Mapping) else {}

    events = [
        {
            "event": "departure",
            "body": seq.split(" -> ")[0] if seq else None,
            "et": epochs.get("depart_et"),
            "notes": "patched heliocentric departure state",
        },
        {
            "event": "pre_flyby_tcm",
            "body": seq.split(" -> ")[0] if seq else None,
            "et": epochs.get("depart_et"),
            "dv_km_s": pre.get("dv_correction_km_s"),
            "dv_m_s": pre.get("dv_correction_m_s"),
            "purpose": f"hit {flyby_body} SOI-in patch point",
        },
        {
            "event": "flyby_soi_in",
            "body": flyby_body,
            "et": epochs.get("entry_et"),
            "entry_miss_after_km": pre.get("miss_after_km"),
            "entry_velocity_miss_after_m_s": pre.get("velocity_miss_after_m_s"),
        },
        {
            "event": "flyby_periapsis",
            "body": flyby_body,
            "et": epochs.get("periapsis_et"),
            "periapsis_altitude_km": hyp.get("periapsis_altitude_km"),
            "rp_margin_km": hyp.get("rp_margin_km"),
            "b_dot_t_km": bp.get("b_dot_t_km"),
            "b_dot_r_km": bp.get("b_dot_r_km"),
        },
        {
            "event": "flyby_soi_out",
            "body": flyby_body,
            "et": epochs.get("exit_et"),
            "notes": "stitched local hyperbola exit state",
        },
        {
            "event": "post_flyby_tcm",
            "body": flyby_body,
            "et": epochs.get("exit_et"),
            "dv_km_s": post.get("dv_correction_km_s"),
            "dv_m_s": post.get("dv_correction_m_s"),
            "purpose": "hit final target in patched heliocentric dynamics",
        },
        {
            "event": "arrival",
            "body": (seq.split(" -> ")[-1] if seq else post.get("target")),
            "et": epochs.get("arrival_et"),
            "arrival_miss_after_km": post.get("miss_after_km"),
            "arrival_vinf_after_km_s": post.get("arrival_vinf_after_km_s"),
        },
    ]
    return events


def corrected_to_packet(rec: Mapping[str, Any], rank: int, args: argparse.Namespace) -> Dict[str, Any]:
    pre = rec.get("pre_arc") if isinstance(rec.get("pre_arc"), Mapping) else {}
    post = rec.get("post_arc") if isinstance(rec.get("post_arc"), Mapping) else {}
    q = rec.get("quality") if isinstance(rec.get("quality"), Mapping) else {}
    source = rec.get("source_stitched_packet") if isinstance(rec.get("source_stitched_packet"), Mapping) else {}
    target = source.get("target") if isinstance(source.get("target"), Mapping) else {}
    hyp = target.get("hyperbola") if isinstance(target.get("hyperbola"), Mapping) else {}
    asym = target.get("asymptotes") if isinstance(target.get("asymptotes"), Mapping) else {}
    bp = target.get("b_plane") if isinstance(target.get("b_plane"), Mapping) else {}

    total_patch = finite(q.get("total_patch_correction_m_s"))
    source_departure = finite(q.get("source_total_departure_correction_m_s"), 0.0)
    total_known = source_departure + total_patch if math.isfinite(total_patch) else math.nan
    rp_margin = finite(q.get("rp_margin_km", hyp.get("rp_margin_km")))
    entry_vel = finite(pre.get("velocity_miss_after_m_s"))
    arrival_after = finite(post.get("miss_after_km"))
    route_class = classify_packet(total_patch, rp_margin, entry_vel, arrival_after)
    pass_manifest = bool(
        rec.get("pass_correction") and
        finite(pre.get("miss_after_km"), math.inf) <= args.max_entry_miss_km and
        entry_vel <= args.max_entry_velocity_miss_m_s and
        arrival_after <= args.max_arrival_miss_km and
        total_patch <= args.max_patch_dv_m_s and
        rp_margin >= args.min_rp_margin_km
    )

    # Lower score is better. Keep it simple and interpretable.
    margin_penalty = max(0.0, args.rp_soft_margin_km - rp_margin) / max(1.0, args.rp_soft_margin_km)
    vel_penalty = max(0.0, entry_vel - args.entry_velocity_soft_m_s) / max(1.0, args.entry_velocity_soft_m_s)
    score = total_patch + 2.0 * margin_penalty + 0.25 * vel_penalty

    packet = {
        "schema_version": SCHEMA_VERSION,
        "candidate_id": stable_id("mga_cand", {"corr": rec.get("correction_id"), "rank": rank}),
        "rank": rank,
        "pass_manifest": pass_manifest,
        "class": route_class,
        "score": score,
        "sequence": rec.get("sequence"),
        "flyby_body": rec.get("flyby_body"),
        "ids": {
            "correction_id": rec.get("correction_id"),
            "stitched_packet_id": rec.get("stitched_packet_id"),
            "packet_id": rec.get("packet_id"),
            "route_id": rec.get("route_id"),
        },
        "epochs": rec.get("epochs", {}),
        "event_timeline": build_event_timeline(rec),
        "corrections": {
            "pre_flyby_tcm_m_s": pre.get("dv_correction_m_s"),
            "pre_flyby_tcm_km_s": pre.get("dv_correction_km_s"),
            "post_flyby_tcm_m_s": post.get("dv_correction_m_s"),
            "post_flyby_tcm_km_s": post.get("dv_correction_km_s"),
            "stitched_patch_total_m_s": total_patch,
            "source_departure_corrections_m_s": source_departure,
            "known_total_corrections_m_s": total_known,
        },
        "quality": {
            "entry_miss_after_km": pre.get("miss_after_km"),
            "entry_velocity_miss_after_m_s": pre.get("velocity_miss_after_m_s"),
            "arrival_miss_after_km": post.get("miss_after_km"),
            "arrival_vinf_after_km_s": post.get("arrival_vinf_after_km_s"),
            "rp_margin_km": rp_margin,
            "periapsis_altitude_km": q.get("periapsis_altitude_km", hyp.get("periapsis_altitude_km")),
            "vinf_effective_km_s": asym.get("vinf_effective_km_s"),
            "vinf_mismatch_m_s": asym.get("vinf_mismatch_m_s"),
            "turn_angle_deg": asym.get("turn_angle_deg"),
            "b_dot_t_km": bp.get("b_dot_t_km"),
            "b_dot_r_km": bp.get("b_dot_r_km"),
            "pre_hit_component_bound": pre.get("hit_component_bound"),
            "post_hit_component_bound": post.get("hit_component_bound"),
        },
        "thresholds": {
            "max_entry_miss_km": args.max_entry_miss_km,
            "max_entry_velocity_miss_m_s": args.max_entry_velocity_miss_m_s,
            "max_arrival_miss_km": args.max_arrival_miss_km,
            "max_patch_dv_m_s": args.max_patch_dv_m_s,
            "min_rp_margin_km": args.min_rp_margin_km,
        },
        "source_corrected_route": rec if args.embed_source else None,
    }
    if not args.embed_source:
        packet.pop("source_corrected_route", None)
    return packet


def flat_row(p: Mapping[str, Any]) -> Dict[str, Any]:
    c = p.get("corrections") if isinstance(p.get("corrections"), Mapping) else {}
    q = p.get("quality") if isinstance(p.get("quality"), Mapping) else {}
    e = p.get("epochs") if isinstance(p.get("epochs"), Mapping) else {}
    return {
        "rank": p.get("rank"),
        "candidate_id": p.get("candidate_id"),
        "sequence": p.get("sequence"),
        "class": p.get("class"),
        "pass_manifest": int(bool(p.get("pass_manifest"))),
        "score": p.get("score"),
        "depart_et": e.get("depart_et"),
        "entry_et": e.get("entry_et"),
        "periapsis_et": e.get("periapsis_et"),
        "exit_et": e.get("exit_et"),
        "arrival_et": e.get("arrival_et"),
        "pre_tcm_m_s": c.get("pre_flyby_tcm_m_s"),
        "post_tcm_m_s": c.get("post_flyby_tcm_m_s"),
        "stitched_patch_total_m_s": c.get("stitched_patch_total_m_s"),
        "source_departure_corrections_m_s": c.get("source_departure_corrections_m_s"),
        "known_total_corrections_m_s": c.get("known_total_corrections_m_s"),
        "entry_miss_after_km": q.get("entry_miss_after_km"),
        "entry_velocity_miss_after_m_s": q.get("entry_velocity_miss_after_m_s"),
        "arrival_miss_after_km": q.get("arrival_miss_after_km"),
        "rp_margin_km": q.get("rp_margin_km"),
        "periapsis_altitude_km": q.get("periapsis_altitude_km"),
        "vinf_mismatch_m_s": q.get("vinf_mismatch_m_s"),
        "turn_angle_deg": q.get("turn_angle_deg"),
        "b_dot_t_km": q.get("b_dot_t_km"),
        "b_dot_r_km": q.get("b_dot_r_km"),
    }


def write_csv(path: Path, packets: Sequence[Mapping[str, Any]]) -> None:
    fields = list(flat_row({}).keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for p in packets:
            w.writerow(flat_row(p))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Finalize corrected stitched MGA route candidates into route packets.")
    p.add_argument("--input-jsonl", required=True, type=Path)
    p.add_argument("--top-n", type=int, default=0)
    p.add_argument("--max-entry-miss-km", type=float, default=10.0)
    p.add_argument("--max-entry-velocity-miss-m-s", type=float, default=100.0)
    p.add_argument("--max-arrival-miss-km", type=float, default=10.0)
    p.add_argument("--max-patch-dv-m-s", type=float, default=25.0)
    p.add_argument("--min-rp-margin-km", type=float, default=800.0)
    p.add_argument("--rp-soft-margin-km", type=float, default=1200.0)
    p.add_argument("--entry-velocity-soft-m-s", type=float, default=25.0)
    p.add_argument("--embed-source", action="store_true")
    p.add_argument("--output-csv", required=True, type=Path)
    p.add_argument("--output-jsonl", required=True, type=Path)
    p.add_argument("--output-json", required=True, type=Path)
    p.add_argument("--output-best-json", required=True, type=Path)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    records = [r for r in load_corrected(args.input_jsonl) if isinstance(r, Mapping)]
    records = [r for r in records if bool(r.get("ok"))]

    packets = [corrected_to_packet(r, i + 1, args) for i, r in enumerate(records)]
    packets.sort(key=lambda p: (0 if p.get("pass_manifest") else 1, finite(p.get("score"), math.inf)))
    if args.top_n > 0:
        packets = packets[:args.top_n]
    # Re-rank after sorting.
    for i, p in enumerate(packets, start=1):
        p["rank"] = i

    write_csv(args.output_csv, packets)
    write_jsonl(args.output_jsonl, packets)

    total_patch = [finite(get_path(p, ["corrections", "stitched_patch_total_m_s"])) for p in packets]
    entry_vel = [finite(get_path(p, ["quality", "entry_velocity_miss_after_m_s"])) for p in packets]
    rp_margin = [finite(get_path(p, ["quality", "rp_margin_km"])) for p in packets]
    arrival = [finite(get_path(p, ["quality", "arrival_miss_after_km"])) for p in packets]
    summary = {
        "schema_version": SCHEMA_VERSION + ".summary",
        "input_records": len(records),
        "packets_written": len(packets),
        "pass_manifest": sum(1 for p in packets if p.get("pass_manifest")),
        "class_counts": {c: sum(1 for p in packets if p.get("class") == c) for c in sorted(set(str(p.get("class")) for p in packets))},
        "thresholds": {
            "max_entry_miss_km": args.max_entry_miss_km,
            "max_entry_velocity_miss_m_s": args.max_entry_velocity_miss_m_s,
            "max_arrival_miss_km": args.max_arrival_miss_km,
            "max_patch_dv_m_s": args.max_patch_dv_m_s,
            "min_rp_margin_km": args.min_rp_margin_km,
        },
        "stats": {
            "total_patch_correction_m_s": stats(total_patch),
            "entry_velocity_miss_after_m_s": stats(entry_vel),
            "arrival_miss_after_km": stats(arrival),
            "rp_margin_km": stats(rp_margin),
        },
        "top_candidates": [flat_row(p) for p in packets[:10]],
    }
    write_json(args.output_json, summary)
    if packets:
        write_json(args.output_best_json, packets[0])
    else:
        write_json(args.output_best_json, {"schema_version": SCHEMA_VERSION, "ok": False, "message": "no packets"})

    print("=" * 80)
    print("MGA FINALIZE CORRECTED ROUTE PACKET V0.1")
    print("=" * 80)
    print(f"Input records:    {len(records)}")
    print(f"Packets written:  {len(packets)}")
    print(f"Pass manifest:    {summary['pass_manifest']}")
    print(f"Classes:          {summary['class_counts']}")
    print(f"Patch Δv m/s:     min={summary['stats']['total_patch_correction_m_s']['min']} median={summary['stats']['total_patch_correction_m_s']['median']} max={summary['stats']['total_patch_correction_m_s']['max']}")
    print(f"Entry vel m/s:    min={summary['stats']['entry_velocity_miss_after_m_s']['min']} median={summary['stats']['entry_velocity_miss_after_m_s']['median']} max={summary['stats']['entry_velocity_miss_after_m_s']['max']}")
    print(f"Arrival miss km:  min={summary['stats']['arrival_miss_after_km']['min']} median={summary['stats']['arrival_miss_after_km']['median']} max={summary['stats']['arrival_miss_after_km']['max']}")
    print("\nTop candidates:")
    for row in summary["top_candidates"]:
        print(f" {row['rank']}. {row['sequence']} | pass={bool(row['pass_manifest'])} | class={row['class']} | score={finite(row['score']):.3f} | patch_dv={finite(row['stitched_patch_total_m_s']):.3f} m/s | entry_v={finite(row['entry_velocity_miss_after_m_s']):.3f} m/s | rp_margin={finite(row['rp_margin_km']):.1f} km")
    print("=" * 80)
    print(f"[OK] wrote CSV:       {args.output_csv}")
    print(f"[OK] wrote JSONL:     {args.output_jsonl}")
    print(f"[OK] wrote JSON:      {args.output_json}")
    print(f"[OK] wrote best JSON: {args.output_best_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
