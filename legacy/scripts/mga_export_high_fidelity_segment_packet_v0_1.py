#!/usr/bin/env python3
"""
mga_export_high_fidelity_segment_packet_v0_1.py

Build an explicit high-fidelity handoff segment packet from a corrected
stitched MGA route candidate.

This script does not integrate. It freezes the exact segment boundary states
needed by the next independent validator (REBOUND/Tudat/local dense SPK):

  1) heliocentric pre-flyby arc:
       origin centre at departure, with velocity corrected by:
       - Lambert/PyGMO leg correction from B-plane packet
       - stitched pre-flyby TCM from patch corrector
       -> Duna SOI-in patch state

  2) Duna local flyby arc:
       Duna SOI-in -> periapsis -> Duna SOI-out

  3) heliocentric post-flyby arc:
       Duna SOI-out patch state + stitched post-flyby TCM
       -> final target centre at arrival

Inputs:
  - corrected route packet JSON/JSONL from mga_finalize_corrected_route_packet_v0_1.py
    preferably generated with --embed-source
  - B-plane packet JSON/JSONL from mga_make_bplane_packet_v0_1.py
  - SPICE BSP/TPC to recover origin/target body states at event epochs

Units: km, km/s, ET seconds, m/s.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

SCHEMA_VERSION = "mga_high_fidelity_segment_packet.v0.1"
SECONDS_PER_DAY = 86400.0
Vec3 = Tuple[float, float, float]


def finite(x: Any, default: float = math.nan) -> float:
    try:
        y = float(x)
    except (TypeError, ValueError):
        return default
    return y if math.isfinite(y) else default


def opt_float(x: Any) -> Optional[float]:
    y = finite(x)
    return y if math.isfinite(y) else None


def vec3(x: Any) -> Optional[Vec3]:
    if not isinstance(x, Sequence) or isinstance(x, (str, bytes)) or len(x) < 3:
        return None
    out: List[float] = []
    for i in range(3):
        y = finite(x[i])
        if not math.isfinite(y):
            return None
        out.append(y)
    return (out[0], out[1], out[2])


def vadd(a: Sequence[float], b: Sequence[float]) -> Vec3:
    return (float(a[0]) + float(b[0]), float(a[1]) + float(b[1]), float(a[2]) + float(b[2]))


def vsub(a: Sequence[float], b: Sequence[float]) -> Vec3:
    return (float(a[0]) - float(b[0]), float(a[1]) - float(b[1]), float(a[2]) - float(b[2]))


def vnorm(a: Sequence[float]) -> float:
    return math.sqrt(float(a[0])**2 + float(a[1])**2 + float(a[2])**2)


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


def load_rows(path: Path, single_keys: Sequence[str], collection_keys: Sequence[str]) -> List[Dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        return read_jsonl(path)
    data = load_json(path)
    if all(k in data for k in single_keys):
        return [data]
    for key in collection_keys:
        rows = data.get(key)
        if isinstance(rows, list):
            return [dict(r) for r in rows if isinstance(r, Mapping)]
    # Best JSON from earlier scripts may be a single object with only route metadata.
    if isinstance(data, Mapping):
        return [data]
    raise ValueError(f"Could not load rows from {path}")


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


def load_body_catalog(path: Optional[Path]) -> Dict[str, Any]:
    if not path:
        return {}
    data = load_json(path)
    bodies = data.get("bodies")
    return bodies if isinstance(bodies, dict) else {}


def body_info(catalog_bodies: Mapping[str, Any], name: str) -> Dict[str, Any]:
    ent = catalog_bodies.get(name)
    if ent is None:
        for k, v in catalog_bodies.items():
            if str(k).lower() == name.lower():
                ent = v
                break
    return dict(ent) if isinstance(ent, Mapping) else {}


def state_from_spice(spice: Any, body: str, et: float, frame: str, central: str) -> Tuple[Vec3, Vec3]:
    st, _lt = spice.spkezr(str(body), float(et), str(frame), "NONE", str(central))
    r = (float(st[0]), float(st[1]), float(st[2]))
    v = (float(st[3]), float(st[4]), float(st[5]))
    return r, v


def find_bplane_packet(packet: Mapping[str, Any], bplane_packets: Sequence[Mapping[str, Any]]) -> Optional[Mapping[str, Any]]:
    ids = packet.get("ids") if isinstance(packet.get("ids"), Mapping) else {}
    keys = [
        packet.get("packet_id"), packet.get("route_id"), packet.get("candidate_id"),
        ids.get("packet_id"), ids.get("route_id"), ids.get("correction_id"),
    ]
    for key in keys:
        if key is None:
            continue
        for bp in bplane_packets:
            for bp_key in (bp.get("packet_id"), bp.get("route_id"), bp.get("correction_id"), bp.get("closure_id")):
                if bp_key is not None and str(bp_key) == str(key):
                    return bp
    seq = packet.get("sequence")
    for bp in bplane_packets:
        if seq is not None and bp.get("sequence") == seq:
            return bp
    return None


def embedded_corrected_route(packet: Mapping[str, Any]) -> Optional[Mapping[str, Any]]:
    src = packet.get("source_corrected_route")
    if isinstance(src, Mapping):
        return src
    # Some callers may pass the corrector output directly.
    if "source_stitched_packet" in packet and "pre_arc" in packet and "post_arc" in packet:
        return packet
    return None


def extract_seq(packet: Mapping[str, Any]) -> List[str]:
    seq_raw = str(packet.get("sequence") or "")
    if "->" in seq_raw:
        return [s.strip() for s in seq_raw.split("->") if s.strip()]
    if "," in seq_raw:
        return [s.strip() for s in seq_raw.split(",") if s.strip()]
    return [seq_raw] if seq_raw else []


def patch_state(stitched: Mapping[str, Any], key: str) -> Tuple[Vec3, Vec3]:
    ps = stitched.get("patch_states") if isinstance(stitched.get("patch_states"), Mapping) else {}
    block = ps.get(key) if isinstance(ps.get(key), Mapping) else {}
    sc = block.get("spacecraft_state_central") if isinstance(block.get("spacecraft_state_central"), Mapping) else {}
    r = vec3(sc.get("r_km")); v = vec3(sc.get("v_km_s"))
    if r is None or v is None:
        raise ValueError(f"missing patch state {key}.spacecraft_state_central")
    return r, v


def patch_local_state(stitched: Mapping[str, Any], key: str) -> Tuple[Vec3, Vec3]:
    ps = stitched.get("patch_states") if isinstance(stitched.get("patch_states"), Mapping) else {}
    block = ps.get(key) if isinstance(ps.get(key), Mapping) else {}
    loc = block.get("local_body_centered") if isinstance(block.get("local_body_centered"), Mapping) else {}
    r = vec3(loc.get("r_km")); v = vec3(loc.get("v_km_s"))
    if r is None or v is None:
        raise ValueError(f"missing patch state {key}.local_body_centered")
    return r, v


def leg0_from_bplane(bp: Mapping[str, Any]) -> Optional[Mapping[str, Any]]:
    legs = bp.get("legs")
    if isinstance(legs, list) and legs:
        for leg in legs:
            if isinstance(leg, Mapping) and int(finite(leg.get("leg_index"), -1)) == 0:
                return leg
        return legs[0] if isinstance(legs[0], Mapping) else None
    return None


def make_segment_packet(packet: Mapping[str, Any], bplane_packets: Sequence[Mapping[str, Any]], spice: Any, bodies: Mapping[str, Any], args: argparse.Namespace, rank: int) -> Dict[str, Any]:
    failures: List[str] = []
    corr = embedded_corrected_route(packet)
    if corr is None:
        failures.append("missing_embedded_source_corrected_route; rerun finalize with --embed-source")
        corr = {}
    stitched = corr.get("source_stitched_packet") if isinstance(corr.get("source_stitched_packet"), Mapping) else {}
    if not stitched:
        failures.append("missing_source_stitched_packet")
    bp = find_bplane_packet(packet, bplane_packets)
    if bp is None:
        failures.append("missing_matching_bplane_packet")
        bp = {}
    seq = extract_seq(packet) or extract_seq(corr) or extract_seq(bp)
    if len(seq) < 3:
        failures.append("sequence_not_three_or_more_bodies")
    origin = seq[0] if seq else None
    flyby = str(packet.get("flyby_body") or corr.get("flyby_body") or (seq[1] if len(seq) > 1 else ""))
    final_target = seq[-1] if seq else None
    epochs = packet.get("epochs") if isinstance(packet.get("epochs"), Mapping) else corr.get("epochs") if isinstance(corr.get("epochs"), Mapping) else {}
    depart_et = finite(epochs.get("depart_et")); entry_et = finite(epochs.get("entry_et")); pe_et = finite(epochs.get("periapsis_et")); exit_et = finite(epochs.get("exit_et")); arrival_et = finite(epochs.get("arrival_et"))
    for name, val in (("depart_et", depart_et), ("entry_et", entry_et), ("periapsis_et", pe_et), ("exit_et", exit_et), ("arrival_et", arrival_et)):
        if not math.isfinite(val):
            failures.append(f"missing_{name}")
    frame = args.frame
    central = args.central_body
    central_mu = float(args.mu_central_km3_s2)

    # Body states and segment states.
    try:
        r_origin, v_origin = state_from_spice(spice, str(origin), depart_et, frame, central) if origin else (None, None)  # type: ignore
    except Exception as exc:
        failures.append(f"origin_spice_state_failed:{exc}"); r_origin = v_origin = None  # type: ignore
    try:
        r_target, v_target = state_from_spice(spice, str(final_target), arrival_et, frame, central) if final_target else (None, None)  # type: ignore
    except Exception as exc:
        failures.append(f"target_spice_state_failed:{exc}"); r_target = v_target = None  # type: ignore

    pre_arc = corr.get("pre_arc") if isinstance(corr.get("pre_arc"), Mapping) else {}
    post_arc = corr.get("post_arc") if isinstance(corr.get("post_arc"), Mapping) else {}
    pre_tcm = vec3(pre_arc.get("dv_correction_km_s")) or (0.0, 0.0, 0.0)
    post_tcm = vec3(post_arc.get("dv_correction_km_s")) or (0.0, 0.0, 0.0)
    bleg0 = leg0_from_bplane(bp)
    v_depart_base = vec3(bleg0.get("sc_v_depart_corrected_km_s")) if isinstance(bleg0, Mapping) else None
    if v_depart_base is None:
        failures.append("missing_pre_leg_departure_velocity_from_bplane_packet")
        v_depart_final = None
    else:
        v_depart_final = vadd(v_depart_base, pre_tcm)

    try:
        r_entry, v_entry = patch_state(stitched, "entry_soi")
        r_pe, v_pe = patch_state(stitched, "periapsis")
        r_exit, v_exit_base = patch_state(stitched, "exit_soi")
        v_exit_final = vadd(v_exit_base, post_tcm)
    except Exception as exc:
        failures.append(str(exc))
        r_entry = v_entry = r_pe = v_pe = r_exit = v_exit_base = v_exit_final = None  # type: ignore

    try:
        r_entry_loc, v_entry_loc = patch_local_state(stitched, "entry_soi")
        r_pe_loc, v_pe_loc = patch_local_state(stitched, "periapsis")
        r_exit_loc, v_exit_loc = patch_local_state(stitched, "exit_soi")
    except Exception as exc:
        failures.append(str(exc))
        r_entry_loc = v_entry_loc = r_pe_loc = v_pe_loc = r_exit_loc = v_exit_loc = None  # type: ignore

    q = packet.get("quality") if isinstance(packet.get("quality"), Mapping) else {}
    c = packet.get("corrections") if isinstance(packet.get("corrections"), Mapping) else {}
    flyby_info = body_info(bodies, flyby)
    segments: List[Dict[str, Any]] = []
    if r_origin is not None and v_depart_final is not None and r_entry is not None:
        segments.append({
            "segment_index": 0,
            "segment_type": "heliocentric_patched_pre_flyby",
            "from": origin,
            "to": f"{flyby}_SOI_in",
            "t0_et": depart_et,
            "t1_et": entry_et,
            "tof_days": (entry_et - depart_et)/SECONDS_PER_DAY,
            "central_body": central,
            "frame": frame,
            "dynamics_recommendation": "central Sun + third bodies, excluding origin and flyby body inside patched model",
            "r0_km": r_origin,
            "v0_km_s": v_depart_final,
            "r1_target_km": r_entry,
            "v1_patch_km_s": v_entry,
            "applied_tcm_km_s": pre_tcm,
            "applied_tcm_m_s": vnorm(pre_tcm)*1000.0,
        })
    if r_entry_loc is not None and r_exit_loc is not None:
        segments.append({
            "segment_index": 1,
            "segment_type": "local_flyby_body_centered_two_body",
            "body": flyby,
            "t0_et": entry_et,
            "periapsis_et": pe_et,
            "t1_et": exit_et,
            "tof_days": (exit_et - entry_et)/SECONDS_PER_DAY,
            "mu_body_km3_s2": opt_float(flyby_info.get("mu_km3_s2")),
            "r0_body_centered_km": r_entry_loc,
            "v0_body_centered_km_s": v_entry_loc,
            "r_periapsis_body_centered_km": r_pe_loc,
            "v_periapsis_body_centered_km_s": v_pe_loc,
            "r1_body_centered_km": r_exit_loc,
            "v1_body_centered_km_s": v_exit_loc,
            "periapsis_altitude_km": q.get("periapsis_altitude_km"),
            "rp_margin_km": q.get("rp_margin_km"),
            "b_dot_t_km": q.get("b_dot_t_km"),
            "b_dot_r_km": q.get("b_dot_r_km"),
        })
    if r_exit is not None and v_exit_final is not None and r_target is not None:
        segments.append({
            "segment_index": 2,
            "segment_type": "heliocentric_patched_post_flyby",
            "from": f"{flyby}_SOI_out",
            "to": final_target,
            "t0_et": exit_et,
            "t1_et": arrival_et,
            "tof_days": (arrival_et - exit_et)/SECONDS_PER_DAY,
            "central_body": central,
            "frame": frame,
            "dynamics_recommendation": "central Sun + third bodies, excluding flyby and final target inside patched model",
            "r0_km": r_exit,
            "v0_km_s": v_exit_final,
            "r1_target_km": r_target,
            "v1_target_km_s": v_target,
            "applied_tcm_km_s": post_tcm,
            "applied_tcm_m_s": vnorm(post_tcm)*1000.0,
        })

    ready = (
        bool(packet.get("pass_manifest", corr.get("pass_correction", False))) and
        len(segments) == 3 and
        finite(c.get("stitched_patch_total_m_s", q.get("total_patch_correction_m_s")), math.inf) <= args.max_patch_dv_m_s and
        finite(q.get("rp_margin_km"), -math.inf) >= args.min_rp_margin_km and
        finite(q.get("arrival_miss_after_km"), math.inf) <= args.max_arrival_miss_km and
        len(failures) == 0
    )
    packet_id = stable_id("hfpkt", {"candidate": packet.get("candidate_id"), "rank": rank, "epochs": epochs})
    out = {
        "schema_version": SCHEMA_VERSION,
        "high_fidelity_packet_id": packet_id,
        "source_candidate_id": packet.get("candidate_id"),
        "source_rank": packet.get("rank"),
        "rank": rank,
        "ready_for_rebound_or_tudat": bool(ready),
        "failures": failures,
        "sequence": " -> ".join(seq) if seq else packet.get("sequence"),
        "flyby_body": flyby,
        "class": packet.get("class"),
        "score": packet.get("score"),
        "central_body": central,
        "central_mu_km3_s2": central_mu,
        "frame": frame,
        "epochs": {"depart_et": depart_et, "entry_et": entry_et, "periapsis_et": pe_et, "exit_et": exit_et, "arrival_et": arrival_et},
        "quality": {
            "known_total_corrections_m_s": c.get("known_total_corrections_m_s"),
            "stitched_patch_total_m_s": c.get("stitched_patch_total_m_s"),
            "pre_flyby_tcm_m_s": c.get("pre_flyby_tcm_m_s"),
            "post_flyby_tcm_m_s": c.get("post_flyby_tcm_m_s"),
            "entry_miss_after_km": q.get("entry_miss_after_km"),
            "entry_velocity_miss_after_m_s": q.get("entry_velocity_miss_after_m_s"),
            "arrival_miss_after_km": q.get("arrival_miss_after_km"),
            "rp_margin_km": q.get("rp_margin_km"),
            "periapsis_altitude_km": q.get("periapsis_altitude_km"),
            "vinf_mismatch_m_s": q.get("vinf_mismatch_m_s"),
            "turn_angle_deg": q.get("turn_angle_deg"),
            "b_dot_t_km": q.get("b_dot_t_km"),
            "b_dot_r_km": q.get("b_dot_r_km"),
        },
        "segments": segments,
        "recommended_next_steps": [
            "Run independent segment propagation with REBOUND/IAS15 or Tudat using this packet.",
            "Generate dense local SPICE windows around Duna SOI/periapsis and Jool arrival if needed.",
            "Only after independent validation, compare event states against KSP/Principia via kRPC snapshots.",
        ],
    }
    return out


def flat_row(p: Mapping[str, Any]) -> Dict[str, Any]:
    q = p.get("quality") if isinstance(p.get("quality"), Mapping) else {}
    e = p.get("epochs") if isinstance(p.get("epochs"), Mapping) else {}
    segs = p.get("segments") if isinstance(p.get("segments"), list) else []
    return {
        "rank": p.get("rank"),
        "high_fidelity_packet_id": p.get("high_fidelity_packet_id"),
        "source_candidate_id": p.get("source_candidate_id"),
        "sequence": p.get("sequence"),
        "class": p.get("class"),
        "ready": int(bool(p.get("ready_for_rebound_or_tudat"))),
        "num_segments": len(segs),
        "depart_et": e.get("depart_et"),
        "entry_et": e.get("entry_et"),
        "periapsis_et": e.get("periapsis_et"),
        "exit_et": e.get("exit_et"),
        "arrival_et": e.get("arrival_et"),
        "known_total_corrections_m_s": q.get("known_total_corrections_m_s"),
        "stitched_patch_total_m_s": q.get("stitched_patch_total_m_s"),
        "pre_tcm_m_s": q.get("pre_flyby_tcm_m_s"),
        "post_tcm_m_s": q.get("post_flyby_tcm_m_s"),
        "rp_margin_km": q.get("rp_margin_km"),
        "periapsis_altitude_km": q.get("periapsis_altitude_km"),
        "arrival_miss_after_km": q.get("arrival_miss_after_km"),
        "failures": ";".join(str(x) for x in (p.get("failures") or [])),
    }


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = list(flat_row({}).keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(flat_row(r))


def stats(vals: Sequence[float]) -> Dict[str, Optional[float]]:
    xs = sorted(v for v in vals if math.isfinite(v))
    if not xs:
        return {"min": None, "median": None, "max": None}
    return {"min": xs[0], "median": xs[len(xs)//2], "max": xs[-1]}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export explicit high-fidelity segment packets from corrected MGA route candidates.")
    p.add_argument("--input-packet", required=True, type=Path)
    p.add_argument("--bplane-packet", required=True, type=Path)
    p.add_argument("--bsp", required=True, type=Path)
    p.add_argument("--tpc", required=True, type=Path)
    p.add_argument("--body-catalog", type=Path)
    p.add_argument("--central-body", default="Sun")
    p.add_argument("--mu-central-km3-s2", type=float, required=True)
    p.add_argument("--frame", default="J2000")
    p.add_argument("--top-n", type=int, default=0)
    p.add_argument("--max-patch-dv-m-s", type=float, default=25.0)
    p.add_argument("--min-rp-margin-km", type=float, default=800.0)
    p.add_argument("--max-arrival-miss-km", type=float, default=10.0)
    p.add_argument("--output-csv", required=True, type=Path)
    p.add_argument("--output-jsonl", required=True, type=Path)
    p.add_argument("--output-json", required=True, type=Path)
    p.add_argument("--output-best-json", required=True, type=Path)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    import spiceypy as spice  # type: ignore
    spice.kclear()
    spice.furnsh(str(args.tpc))
    spice.furnsh(str(args.bsp))
    try:
        packets = load_rows(args.input_packet, single_keys=("candidate_id",), collection_keys=("packets", "routes", "records", "top_candidates"))
        packets = [p for p in packets if bool(p.get("pass_manifest", True))]
        packets.sort(key=lambda p: (finite(p.get("score"), math.inf), int(finite(p.get("rank"), 999999))))
        if args.top_n > 0:
            packets = packets[:args.top_n]
        bplane_packets = load_rows(args.bplane_packet, single_keys=("packet_id", "legs"), collection_keys=("packets", "routes", "records", "top_packets"))
        bodies = load_body_catalog(args.body_catalog)
        out: List[Dict[str, Any]] = []
        for i, pkt in enumerate(packets, start=1):
            out.append(make_segment_packet(pkt, bplane_packets, spice, bodies, args, i))
        out.sort(key=lambda p: (0 if p.get("ready_for_rebound_or_tudat") else 1, finite((p.get("quality") or {}).get("known_total_corrections_m_s"), math.inf)))
        for i, pkt in enumerate(out, start=1):
            pkt["rank"] = i
        write_csv(args.output_csv, out)
        write_jsonl(args.output_jsonl, out)
        total_known = [finite((p.get("quality") or {}).get("known_total_corrections_m_s")) for p in out]
        rp = [finite((p.get("quality") or {}).get("rp_margin_km")) for p in out]
        summary = {
            "schema_version": SCHEMA_VERSION + ".summary",
            "input_packets": len(packets),
            "packets_written": len(out),
            "ready_for_rebound_or_tudat": sum(1 for p in out if p.get("ready_for_rebound_or_tudat")),
            "thresholds": {
                "max_patch_dv_m_s": args.max_patch_dv_m_s,
                "min_rp_margin_km": args.min_rp_margin_km,
                "max_arrival_miss_km": args.max_arrival_miss_km,
            },
            "stats": {
                "known_total_corrections_m_s": stats(total_known),
                "rp_margin_km": stats(rp),
            },
            "top_packets": [flat_row(p) for p in out[:10]],
        }
        write_json(args.output_json, summary)
        if out:
            write_json(args.output_best_json, out[0])
        else:
            write_json(args.output_best_json, {"schema_version": SCHEMA_VERSION, "ok": False, "message": "no packets"})
        print("=" * 80)
        print("MGA HIGH-FIDELITY SEGMENT PACKET EXPORT V0.1")
        print("=" * 80)
        print(f"Input packets:   {len(packets)}")
        print(f"Packets written: {len(out)}")
        print(f"Ready packets:   {summary['ready_for_rebound_or_tudat']}")
        print(f"Known corr m/s:  min={summary['stats']['known_total_corrections_m_s']['min']} median={summary['stats']['known_total_corrections_m_s']['median']} max={summary['stats']['known_total_corrections_m_s']['max']}")
        print(f"rp margin km:    min={summary['stats']['rp_margin_km']['min']} median={summary['stats']['rp_margin_km']['median']} max={summary['stats']['rp_margin_km']['max']}")
        print("\nTop packets:")
        for row in summary["top_packets"][:10]:
            print(f" {row['rank']}. {row['sequence']} | ready={bool(row['ready'])} | class={row['class']} | segs={row['num_segments']} | corr={finite(row['known_total_corrections_m_s']):.3f} m/s | rp_margin={finite(row['rp_margin_km']):.1f} km | failures={row['failures'] or '-'}")
        print("=" * 80)
        print(f"[OK] wrote CSV:       {args.output_csv}")
        print(f"[OK] wrote JSONL:     {args.output_jsonl}")
        print(f"[OK] wrote JSON:      {args.output_json}")
        print(f"[OK] wrote best JSON: {args.output_best_json}")
        return 0
    finally:
        try:
            spice.kclear()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
