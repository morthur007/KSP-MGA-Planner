#!/usr/bin/env python3
"""
mga_export_route_timeline_v0_1.py

Export finalized corrected MGA route packets into an operational timeline/checklist.

Input:
  JSON/JSONL from mga_finalize_corrected_route_packet_v0_1.py, preferably with
  --embed-source so correction vectors and stitched patch states are retained.

Output:
  - CSV event timeline
  - Markdown mission checklist/report
  - JSON summary

Scope:
  This script does not integrate dynamics. It freezes the current Class-A
  patched/global-local result into a traceable handoff artifact for the next
  stages: high-fidelity validation, dense local SPK, REBOUND/Principia checks,
  and eventual execution planning.

Units:
  ET seconds, days, km, km/s, m/s.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

SECONDS_PER_DAY = 86400.0
SCHEMA_VERSION = "mga_route_timeline_export.v0.1"


def finite(x: Any, default: float = math.nan) -> float:
    try:
        y = float(x)
    except (TypeError, ValueError):
        return default
    return y if math.isfinite(y) else default


def fmt(x: Any, nd: int = 6, na: str = "") -> str:
    y = finite(x)
    if not math.isfinite(y):
        return na
    return f"{y:.{nd}f}"


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


def load_packets(path: Path) -> List[Dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        return read_jsonl(path)
    data = load_json(path)
    if "candidate_id" in data and "event_timeline" in data:
        return [data]
    for key in ("packets", "routes", "records", "results", "top_candidates"):
        rows = data.get(key)
        if isinstance(rows, list):
            return [dict(r) for r in rows if isinstance(r, Mapping)]
    raise ValueError(f"Could not find route packets in {path}")


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(sanitize(payload), f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write("\n")


def vec_norm_m_s(vec_km_s: Any) -> Optional[float]:
    if not isinstance(vec_km_s, Sequence) or isinstance(vec_km_s, (str, bytes)) or len(vec_km_s) < 3:
        return None
    xs: List[float] = []
    for i in range(3):
        y = finite(vec_km_s[i])
        if not math.isfinite(y):
            return None
        xs.append(y)
    return math.sqrt(sum(x*x for x in xs)) * 1000.0


def correction_vectors(packet: Mapping[str, Any]) -> Dict[str, Any]:
    src = packet.get("source_corrected_route") if isinstance(packet.get("source_corrected_route"), Mapping) else {}
    pre = src.get("pre_arc") if isinstance(src.get("pre_arc"), Mapping) else {}
    post = src.get("post_arc") if isinstance(src.get("post_arc"), Mapping) else {}
    return {
        "pre_flyby_tcm_km_s": pre.get("dv_correction_km_s"),
        "pre_flyby_tcm_m_s": pre.get("dv_correction_m_s"),
        "post_flyby_tcm_km_s": post.get("dv_correction_km_s"),
        "post_flyby_tcm_m_s": post.get("dv_correction_m_s"),
        "pre_arc_perturbers": pre.get("perturbers"),
        "post_arc_perturbers": post.get("perturbers"),
    }


def patch_state_summary(packet: Mapping[str, Any]) -> Dict[str, Any]:
    src = packet.get("source_corrected_route") if isinstance(packet.get("source_corrected_route"), Mapping) else {}
    stitched = src.get("source_stitched_packet") if isinstance(src.get("source_stitched_packet"), Mapping) else {}
    states = stitched.get("patch_states") if isinstance(stitched.get("patch_states"), Mapping) else {}
    out: Dict[str, Any] = {}
    for key in ("entry_soi", "periapsis", "exit_soi"):
        st = states.get(key) if isinstance(states.get(key), Mapping) else {}
        out[key] = {
            "et": st.get("et"),
            "local_radius_km": st.get("local_radius_km"),
            "local_speed_km_s": st.get("local_speed_km_s"),
            "r_central_km": get_path(st, ["spacecraft_state_central", "r_km"]),
            "v_central_km_s": get_path(st, ["spacecraft_state_central", "v_km_s"]),
            "r_local_km": get_path(st, ["local_body_centered", "r_km"]),
            "v_local_km_s": get_path(st, ["local_body_centered", "v_km_s"]),
        }
    return out


def packet_event_rows(packet: Mapping[str, Any]) -> List[Dict[str, Any]]:
    seq = packet.get("sequence")
    rank = packet.get("rank")
    cid = packet.get("candidate_id")
    klass = packet.get("class")
    epochs = packet.get("epochs") if isinstance(packet.get("epochs"), Mapping) else {}
    depart = finite(epochs.get("depart_et"))
    q = packet.get("quality") if isinstance(packet.get("quality"), Mapping) else {}
    c = packet.get("corrections") if isinstance(packet.get("corrections"), Mapping) else {}
    corr_vecs = correction_vectors(packet)
    patches = patch_state_summary(packet)

    rows: List[Dict[str, Any]] = []
    for ev in packet.get("event_timeline", []) or []:
        if not isinstance(ev, Mapping):
            continue
        et = finite(ev.get("et"))
        event = str(ev.get("event"))
        dv_vec = None
        dv_m_s = ev.get("dv_m_s")
        if event == "pre_flyby_tcm":
            dv_vec = corr_vecs.get("pre_flyby_tcm_km_s")
            dv_m_s = c.get("pre_flyby_tcm_m_s", dv_m_s)
        elif event == "post_flyby_tcm":
            dv_vec = corr_vecs.get("post_flyby_tcm_km_s")
            dv_m_s = c.get("post_flyby_tcm_m_s", dv_m_s)
        patch_key = {
            "flyby_soi_in": "entry_soi",
            "flyby_periapsis": "periapsis",
            "flyby_soi_out": "exit_soi",
        }.get(event)
        patch = patches.get(patch_key, {}) if patch_key else {}
        rows.append({
            "candidate_id": cid,
            "rank": rank,
            "class": klass,
            "sequence": seq,
            "event": event,
            "body": ev.get("body"),
            "et": et if math.isfinite(et) else None,
            "days_from_depart": ((et - depart)/SECONDS_PER_DAY if math.isfinite(et) and math.isfinite(depart) else None),
            "dv_m_s": dv_m_s,
            "dv_vector_km_s": dv_vec,
            "local_radius_km": patch.get("local_radius_km"),
            "local_speed_km_s": patch.get("local_speed_km_s"),
            "rp_margin_km": ev.get("rp_margin_km", q.get("rp_margin_km")),
            "periapsis_altitude_km": ev.get("periapsis_altitude_km", q.get("periapsis_altitude_km")),
            "b_dot_t_km": ev.get("b_dot_t_km", q.get("b_dot_t_km")),
            "b_dot_r_km": ev.get("b_dot_r_km", q.get("b_dot_r_km")),
            "notes": ev.get("notes", ev.get("purpose", "")),
        })
    return rows


def flat_packet_row(p: Mapping[str, Any]) -> Dict[str, Any]:
    q = p.get("quality") if isinstance(p.get("quality"), Mapping) else {}
    c = p.get("corrections") if isinstance(p.get("corrections"), Mapping) else {}
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
        "known_total_corrections_m_s": c.get("known_total_corrections_m_s"),
        "stitched_patch_total_m_s": c.get("stitched_patch_total_m_s"),
        "pre_flyby_tcm_m_s": c.get("pre_flyby_tcm_m_s"),
        "post_flyby_tcm_m_s": c.get("post_flyby_tcm_m_s"),
        "entry_velocity_miss_after_m_s": q.get("entry_velocity_miss_after_m_s"),
        "arrival_miss_after_km": q.get("arrival_miss_after_km"),
        "rp_margin_km": q.get("rp_margin_km"),
        "periapsis_altitude_km": q.get("periapsis_altitude_km"),
        "vinf_effective_km_s": q.get("vinf_effective_km_s"),
        "vinf_mismatch_m_s": q.get("vinf_mismatch_m_s"),
        "turn_angle_deg": q.get("turn_angle_deg"),
        "b_dot_t_km": q.get("b_dot_t_km"),
        "b_dot_r_km": q.get("b_dot_r_km"),
    }


def write_event_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "candidate_id", "rank", "class", "sequence", "event", "body", "et", "days_from_depart",
        "dv_m_s", "dv_vector_km_s", "local_radius_km", "local_speed_km_s", "rp_margin_km",
        "periapsis_altitude_km", "b_dot_t_km", "b_dot_r_km", "notes",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fields})


def write_packet_csv(path: Path, packets: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [flat_packet_row(p) for p in packets]
    fields = list(rows[0].keys()) if rows else list(flat_packet_row({}).keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def write_markdown(path: Path, packets: Sequence[Mapping[str, Any]], event_rows: Sequence[Mapping[str, Any]], args: argparse.Namespace) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: List[str] = []
    lines.append("# MGA Corrected Route Timeline V0.1")
    lines.append("")
    lines.append(f"Input: `{args.input_packet}`")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Packets exported: **{len(packets)}**")
    lines.append(f"- Classes: **{class_counts(packets)}**")
    lines.append(f"- This is a patched/global-local corrected handoff, not yet a Principia execution plan.")
    lines.append("")
    if packets:
        best = packets[0]
        r = flat_packet_row(best)
        lines.append("## Best candidate")
        lines.append("")
        lines.append(f"- Sequence: **{r.get('sequence')}**")
        lines.append(f"- Class: **{r.get('class')}**")
        lines.append(f"- Score: **{fmt(r.get('score'), 3)}**")
        lines.append(f"- Known total corrections: **{fmt(r.get('known_total_corrections_m_s'), 3)} m/s**")
        lines.append(f"- Stitched patch Δv: **{fmt(r.get('stitched_patch_total_m_s'), 3)} m/s**")
        lines.append(f"- Duna periapsis altitude: **{fmt(r.get('periapsis_altitude_km'), 3)} km**")
        lines.append(f"- Duna rp margin: **{fmt(r.get('rp_margin_km'), 3)} km**")
        lines.append(f"- Arrival miss after correction: **{fmt(r.get('arrival_miss_after_km'), 9)} km**")
        lines.append("")
    lines.append("## Event timeline")
    lines.append("")
    lines.append("| Rank | Event | Body | Days from departure | ET | Δv m/s | Notes |")
    lines.append("|---:|---|---|---:|---:|---:|---|")
    for ev in event_rows:
        lines.append(
            f"| {ev.get('rank') or ''} | {ev.get('event') or ''} | {ev.get('body') or ''} | "
            f"{fmt(ev.get('days_from_depart'), 6)} | {fmt(ev.get('et'), 6)} | {fmt(ev.get('dv_m_s'), 6)} | "
            f"{str(ev.get('notes') or '').replace('|', '/')} |"
        )
    lines.append("")
    lines.append("## Next gates")
    lines.append("")
    lines.append("1. Validate the corrected packet in a higher-fidelity independent model.")
    lines.append("2. Generate a dense local SPK around the Duna encounter if the validation remains stable.")
    lines.append("3. Compare the planned Duna B-plane and periapsis against KSP/Principia snapshots.")
    lines.append("4. Only then translate the burn/checkpoint list into an execution checklist.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def class_counts(packets: Sequence[Mapping[str, Any]]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for p in packets:
        c = str(p.get("class", "unknown"))
        out[c] = out.get(c, 0) + 1
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export corrected MGA route packets into operational timelines/checklists.")
    p.add_argument("--input-packet", required=True, type=Path, help="JSON/JSONL from mga_finalize_corrected_route_packet_v0_1.py")
    p.add_argument("--top-n", type=int, default=0)
    p.add_argument("--require-pass", action="store_true", default=True)
    p.add_argument("--class-filter", nargs="*", default=[])
    p.add_argument("--output-events-csv", required=True, type=Path)
    p.add_argument("--output-packets-csv", required=True, type=Path)
    p.add_argument("--output-json", required=True, type=Path)
    p.add_argument("--output-md", required=True, type=Path)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    packets = load_packets(args.input_packet)
    if args.require_pass:
        packets = [p for p in packets if bool(p.get("pass_manifest"))]
    if args.class_filter:
        allowed = {str(x) for x in args.class_filter}
        packets = [p for p in packets if str(p.get("class")) in allowed]
    packets.sort(key=lambda p: (0 if p.get("pass_manifest") else 1, finite(p.get("score"), 1e99)))
    if args.top_n > 0:
        packets = packets[: args.top_n]

    event_rows: List[Dict[str, Any]] = []
    for p in packets:
        event_rows.extend(packet_event_rows(p))

    write_event_csv(args.output_events_csv, event_rows)
    write_packet_csv(args.output_packets_csv, packets)

    summary = {
        "schema_version": SCHEMA_VERSION,
        "input_packet": str(args.input_packet),
        "packets_written": len(packets),
        "events_written": len(event_rows),
        "class_counts": class_counts(packets),
        "top_packets": [flat_packet_row(p) for p in packets[:10]],
        "outputs": {
            "events_csv": str(args.output_events_csv),
            "packets_csv": str(args.output_packets_csv),
            "markdown": str(args.output_md),
        },
    }
    write_json(args.output_json, summary)
    write_markdown(args.output_md, packets, event_rows, args)

    print("="*80)
    print("MGA ROUTE TIMELINE EXPORT V0.1")
    print("="*80)
    print(f"Packets exported: {len(packets)}")
    print(f"Events exported:  {len(event_rows)}")
    print(f"Classes:          {class_counts(packets)}")
    if packets:
        best = flat_packet_row(packets[0])
        print("\nBest packet:")
        print(f"  {best.get('sequence')} | class={best.get('class')} | score={finite(best.get('score')):.3f} | "
              f"known_corr={finite(best.get('known_total_corrections_m_s')):.3f} m/s | "
              f"rp_margin={finite(best.get('rp_margin_km')):.1f} km")
    print("="*80)
    print(f"[OK] wrote events CSV:  {args.output_events_csv}")
    print(f"[OK] wrote packets CSV: {args.output_packets_csv}")
    print(f"[OK] wrote JSON:        {args.output_json}")
    print(f"[OK] wrote Markdown:    {args.output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
