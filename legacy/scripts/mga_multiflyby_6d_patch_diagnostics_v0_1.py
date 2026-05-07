#!/usr/bin/env python3
"""
MGA Multi-Flyby 6D Patch Diagnostics V0.1

Purpose
-------
Diagnose large velocity mismatches after the position-only multi-flyby patch
corrector. This script DOES NOT correct the trajectory. It replays each corrected
heliocentric segment using the same segment definitions and dynamics as
mga_multiflyby_patch_corrector_v0_2.py, then compares endpoint velocity in several
frames/conventions:

  1. central-frame velocity difference: vf - v_target_central
  2. body-relative/local difference: (vf - v_body_spice) - v_target_local
  3. sign-flip hypothesis: (vf - v_body_spice) + v_target_local
  4. speed-only mismatch and angle mismatch in the local frame

This separates:
  - true 6D discontinuity that needs multiple shooting / B-plane retargeting;
  - sign convention mismatch around incoming/outgoing asymptotes;
  - final target arrival v_inf being incorrectly treated as zero relative speed;
  - schema/frame issues.

Input should usually be the JSONL from:
  mga_multiflyby_patch_corrector_v0_2.py --embed-source

The script imports the V0.2 corrector module and reuses its internal segment
builder and propagator to avoid drifting from the already validated dynamics.
"""
from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

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
    vals: List[float] = []
    for i in range(3):
        y = finite(x[i])
        if not math.isfinite(y):
            return None
        vals.append(y)
    return (vals[0], vals[1], vals[2])


def vsub(a: Sequence[float], b: Sequence[float]) -> Vec3:
    return (float(a[0])-float(b[0]), float(a[1])-float(b[1]), float(a[2])-float(b[2]))


def vadd(a: Sequence[float], b: Sequence[float]) -> Vec3:
    return (float(a[0])+float(b[0]), float(a[1])+float(b[1]), float(a[2])+float(b[2]))


def vnorm(a: Sequence[float]) -> float:
    return math.sqrt(float(a[0])**2 + float(a[1])**2 + float(a[2])**2)


def vdot(a: Sequence[float], b: Sequence[float]) -> float:
    return float(a[0])*float(b[0]) + float(a[1])*float(b[1]) + float(a[2])*float(b[2])


def angle_deg(a: Sequence[float], b: Sequence[float]) -> Optional[float]:
    na = vnorm(a)
    nb = vnorm(b)
    if na <= 0 or nb <= 0 or not math.isfinite(na) or not math.isfinite(nb):
        return None
    c = max(-1.0, min(1.0, vdot(a, b)/(na*nb)))
    return math.degrees(math.acos(c))


def json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, tuple):
        return list(obj)
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
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


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, Mapping):
        raise ValueError(f"Expected JSON object in {path}")
    return dict(data)


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, allow_nan=False, default=json_default)
        f.write("\n")


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, allow_nan=False, separators=(",", ":"), default=json_default))
            f.write("\n")


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


def segment_dv(seg_corr: Mapping[str, Any]) -> Vec3:
    return (
        finite(seg_corr.get("dvx_km_s"), 0.0),
        finite(seg_corr.get("dvy_km_s"), 0.0),
        finite(seg_corr.get("dvz_km_s"), 0.0),
    )


def patch_state_for_label(source_packet: Mapping[str, Any], target_label: str) -> Tuple[Optional[str], Optional[str], Optional[Dict[str, Any]]]:
    """Return (body, patch_kind, patch_state) for labels like Eve_soi_in/Kerbin_soi_out.

    patch_kind is one of entry_soi, exit_soi, periapsis, or None.
    """
    label = str(target_label or "")
    if label.endswith("_soi_in"):
        body = label[:-7]
        kind = "entry_soi"
    elif label.endswith("_soi_out"):
        body = label[:-8]
        kind = "exit_soi"
    elif label.endswith("_periapsis"):
        body = label[:-10]
        kind = "periapsis"
    else:
        return None, None, None
    flybys = source_packet.get("flybys") if isinstance(source_packet.get("flybys"), list) else []
    for f in flybys:
        if not isinstance(f, Mapping):
            continue
        if str(f.get("body", "")).lower() != body.lower():
            continue
        states = f.get("patch_states") if isinstance(f.get("patch_states"), Mapping) else {}
        st = states.get(kind) if isinstance(states.get(kind), Mapping) else None
        if st:
            return str(f.get("body")), kind, dict(st)
    return body, kind, None


def classify_segment(row: Mapping[str, Any], args: argparse.Namespace) -> str:
    kind = str(row.get("segment_kind") or "")
    target_label = str(row.get("target_label") or "")
    local_delta = finite(row.get("local_velocity_delta_m_s"), math.inf)
    neg_delta = finite(row.get("local_velocity_delta_if_sign_flipped_m_s"), math.inf)
    speed_delta = finite(row.get("local_speed_delta_m_s"), math.inf)
    angle = finite(row.get("local_angle_deg"), math.inf)
    pos = finite(row.get("position_miss_km"), math.inf)
    v_c = finite(row.get("central_velocity_delta_m_s"), math.inf)

    if not math.isfinite(pos):
        return "propagation_or_schema_failure"
    if kind == "last_flyby_exit_to_final_target" or target_label.endswith("_center"):
        # Final arrival intentionally has nonzero v_inf; compare speed to planet only as diagnostic.
        return "final_arrival_vinf_expected_not_6d_match"
    if local_delta <= args.good_velocity_m_s:
        return "6d_continuity_good"
    if neg_delta < local_delta * args.sign_flip_ratio and neg_delta <= args.sign_flip_abs_m_s:
        return "likely_sign_convention_mismatch"
    if speed_delta <= args.speed_only_good_m_s and angle > args.large_angle_deg:
        return "same_energy_direction_mismatch_bplane_retarget"
    if speed_delta > args.large_speed_delta_m_s:
        return "energy_or_lambert_branch_mismatch"
    if v_c > args.large_velocity_m_s and local_delta > args.large_velocity_m_s:
        return "large_6d_velocity_discontinuity"
    return "moderate_6d_velocity_discontinuity"


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = [
        "route_index", "correction_id", "sequence", "segment_index", "segment_kind", "origin_label", "target_label",
        "target_patch_body", "target_patch_kind", "tof_days", "dv_norm_m_s", "position_miss_km",
        "central_velocity_delta_m_s", "local_velocity_delta_m_s", "local_velocity_delta_if_sign_flipped_m_s",
        "local_speed_delta_m_s", "local_angle_deg", "local_angle_if_sign_flipped_deg",
        "final_local_speed_km_s", "target_local_speed_km_s", "target_body_speed_km_s",
        "final_v_central_km_s", "target_v_central_km_s", "diagnosis",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fields})


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Diagnose 6D velocity mismatches after the multi-flyby position patch corrector.")
    p.add_argument("--input-jsonl", required=True, type=Path, help="JSONL from mga_multiflyby_patch_corrector_v0_2.py, preferably with --embed-source")
    p.add_argument("--bsp", required=True, type=Path)
    p.add_argument("--tpc", required=True, type=Path)
    p.add_argument("--body-catalog", required=True, type=Path)
    p.add_argument("--central-body", default="Sun")
    p.add_argument("--mu-central-km3-s2", required=True, type=float)
    p.add_argument("--frame", default="J2000")
    p.add_argument("--gravitating-bodies", nargs="+", required=True)
    p.add_argument("--top-n", type=int, default=0)
    p.add_argument("--integrator", default="DOP853")
    p.add_argument("--rtol", type=float, default=1e-10)
    p.add_argument("--atol", type=float, default=1e-12)
    p.add_argument("--max-step-days", type=float, default=1.0)
    p.add_argument("--good-velocity-m-s", type=float, default=25.0)
    p.add_argument("--large-velocity-m-s", type=float, default=1000.0)
    p.add_argument("--large-speed-delta-m-s", type=float, default=1000.0)
    p.add_argument("--speed-only-good-m-s", type=float, default=100.0)
    p.add_argument("--large-angle-deg", type=float, default=10.0)
    p.add_argument("--sign-flip-ratio", type=float, default=0.1)
    p.add_argument("--sign-flip-abs-m-s", type=float, default=250.0)
    p.add_argument("--output-csv", required=True, type=Path)
    p.add_argument("--output-jsonl", required=True, type=Path)
    p.add_argument("--output-json", required=True, type=Path)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    # Import companion module from the current working dir or from this script's directory.
    sys.path.insert(0, str(Path.cwd()))
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    corr = importlib.import_module("mga_multiflyby_patch_corrector_v0_2")

    records = read_json_records(args.input_jsonl)
    if args.top_n > 0:
        records = records[:args.top_n]
    catalog = load_json(args.body_catalog)
    cfg = {
        "kernels": [str(args.tpc), str(args.bsp)],
        "body_catalog": catalog,
        "central_body": args.central_body,
        "mu_central": args.mu_central_km3_s2,
        "frame": args.frame,
        "gravitating_bodies": args.gravitating_bodies,
        "integrator": args.integrator,
        "rtol": args.rtol,
        "atol": args.atol,
        "max_step_days": args.max_step_days,
        "max_segment_correction_m_s": 0.0,  # not used; we replay existing dv
        "target_position_miss_km": 10.0,
        "target_velocity_miss_m_s": args.good_velocity_m_s,
        "velocity_pass_mode": "none",
        "max_nfev": 1,
        "embed_source": False,
    }
    corr._init_worker(cfg)  # type: ignore[attr-defined]
    all_pert = corr._load_perturbers(catalog, [str(x) for x in args.gravitating_bodies])  # type: ignore[attr-defined]

    rows: List[Dict[str, Any]] = []
    route_summaries: List[Dict[str, Any]] = []
    diagnosis_counts: Dict[str, int] = {}
    missing_source = 0

    for ridx, rec in enumerate(records, start=1):
        source_packet = rec.get("source_packet") if isinstance(rec.get("source_packet"), Mapping) else None
        if not source_packet:
            missing_source += 1
            route_summaries.append({
                "route_index": ridx,
                "correction_id": rec.get("multiflyby_patch_correction_id"),
                "sequence": extract_sequence(rec),
                "status": "missing_source_packet; rerun corrector with --embed-source",
            })
            continue
        segments, failures = corr._segment_definitions(source_packet, cfg)  # type: ignore[attr-defined]
        seg_corrs_raw = rec.get("segment_corrections") if isinstance(rec.get("segment_corrections"), list) else []
        seg_corrs = {int(finite(s.get("segment_index"), -999)): s for s in seg_corrs_raw if isinstance(s, Mapping)}
        route_rows: List[Dict[str, Any]] = []
        for seg in segments:
            si = int(finite(seg.get("segment_index"), -1))
            sc = seg_corrs.get(si, {})
            dv = segment_dv(sc)
            r0 = corr.vec3_req(seg.get("r0"), "seg.r0")  # type: ignore[attr-defined]
            v0 = corr.vec3_req(seg.get("v0"), "seg.v0")  # type: ignore[attr-defined]
            r_target = corr.vec3_req(seg.get("r_target"), "seg.r_target")  # type: ignore[attr-defined]
            v_target = corr.vec3_req(seg.get("v_target"), "seg.v_target")  # type: ignore[attr-defined]
            t0 = finite(seg.get("t0"))
            t1 = finite(seg.get("t1"))
            excluded = {str(seg.get("start_exclude_body") or "").lower(), str(seg.get("target_exclude_body") or "").lower(), str(args.central_body).lower()}
            pert = [b for b in all_pert if b.name.lower() not in excluded]
            vstart = (v0[0]+dv[0], v0[1]+dv[1], v0[2]+dv[2])
            rf, vf, ok, msg = corr._propagate(r0, vstart, t0, t1, pert, cfg)  # type: ignore[attr-defined]

            target_label = str(seg.get("target_label") or "")
            patch_body, patch_kind, patch_state = patch_state_for_label(source_packet, target_label)
            body_v: Optional[Vec3] = None
            target_local_v: Optional[Vec3] = None
            if patch_state:
                bs = patch_state.get("body_state_central") if isinstance(patch_state.get("body_state_central"), Sequence) else None
                loc = patch_state.get("local_body_centered") if isinstance(patch_state.get("local_body_centered"), Sequence) else None
                bs_v = vec3(bs[3:6] if bs and len(bs) >= 6 else None)
                loc_v = vec3(loc[3:6] if loc and len(loc) >= 6 else None)
                body_v = bs_v
                target_local_v = loc_v
            elif patch_body and patch_kind is not None:
                # Patch label was parseable but not found in source_packet.
                pass
            elif target_label.endswith("_center"):
                body_name = target_label[:-7]
                try:
                    _r_b, body_v2 = corr._state_body(body_name, t1, args.central_body, args.frame)  # type: ignore[attr-defined]
                    body_v = body_v2
                except Exception:
                    body_v = None

            pos_miss = vnorm(vsub(rf, r_target)) if (ok and rf is not None) else math.nan
            central_delta = vnorm(vsub(vf, v_target))*1000.0 if (ok and vf is not None) else math.nan

            local_delta = math.nan
            local_delta_neg = math.nan
            speed_delta = math.nan
            local_ang = math.nan
            local_ang_neg = math.nan
            final_local_speed = math.nan
            target_local_speed = math.nan
            body_speed = vnorm(body_v) if body_v is not None else math.nan
            if ok and vf is not None and body_v is not None:
                final_local = vsub(vf, body_v)
                if target_local_v is None:
                    # For body center arrival, local target velocity is zero by definition,
                    # but that is not a continuity requirement for arrival; keep diagnostic.
                    target_local_v = (0.0, 0.0, 0.0)
                local_delta = vnorm(vsub(final_local, target_local_v))*1000.0
                local_delta_neg = vnorm(vadd(final_local, target_local_v))*1000.0
                final_local_speed = vnorm(final_local)
                target_local_speed = vnorm(target_local_v)
                speed_delta = abs(final_local_speed - target_local_speed)*1000.0
                a1 = angle_deg(final_local, target_local_v)
                a2 = angle_deg(final_local, (-target_local_v[0], -target_local_v[1], -target_local_v[2])) if target_local_speed > 0 else None
                local_ang = a1 if a1 is not None else math.nan
                local_ang_neg = a2 if a2 is not None else math.nan

            row: Dict[str, Any] = {
                "route_index": ridx,
                "correction_id": rec.get("multiflyby_patch_correction_id"),
                "sequence": extract_sequence(rec),
                "segment_index": si,
                "segment_kind": seg.get("segment_kind"),
                "origin_label": seg.get("origin_label"),
                "target_label": target_label,
                "target_patch_body": patch_body,
                "target_patch_kind": patch_kind,
                "tof_days": (t1-t0)/86400.0 if math.isfinite(t0) and math.isfinite(t1) else None,
                "dv_norm_m_s": vnorm(dv)*1000.0,
                "position_miss_km": pos_miss,
                "central_velocity_delta_m_s": central_delta,
                "local_velocity_delta_m_s": local_delta,
                "local_velocity_delta_if_sign_flipped_m_s": local_delta_neg,
                "local_speed_delta_m_s": speed_delta,
                "local_angle_deg": local_ang,
                "local_angle_if_sign_flipped_deg": local_ang_neg,
                "final_local_speed_km_s": final_local_speed,
                "target_local_speed_km_s": target_local_speed,
                "target_body_speed_km_s": body_speed,
                "final_v_central_km_s": vnorm(vf) if (ok and vf is not None) else None,
                "target_v_central_km_s": vnorm(v_target),
                "propagation_ok": ok,
                "propagation_message": msg,
                "segment_failures_from_build": failures,
            }
            row["diagnosis"] = classify_segment(row, args)
            diagnosis_counts[str(row["diagnosis"])] = diagnosis_counts.get(str(row["diagnosis"]), 0) + 1
            rows.append(row)
            route_rows.append(row)

        max_central = max([finite(r.get("central_velocity_delta_m_s"), -math.inf) for r in route_rows] or [math.nan])
        max_local = max([finite(r.get("local_velocity_delta_m_s"), -math.inf) for r in route_rows] or [math.nan])
        max_speed_delta = max([finite(r.get("local_speed_delta_m_s"), -math.inf) for r in route_rows] or [math.nan])
        route_summaries.append({
            "route_index": ridx,
            "correction_id": rec.get("multiflyby_patch_correction_id"),
            "sequence": extract_sequence(rec),
            "n_segments": len(route_rows),
            "max_position_miss_km": max([finite(r.get("position_miss_km"), -math.inf) for r in route_rows] or [math.nan]),
            "max_central_velocity_delta_m_s": max_central,
            "max_local_velocity_delta_m_s": max_local,
            "max_local_speed_delta_m_s": max_speed_delta,
            "diagnoses": {d: sum(1 for r in route_rows if r.get("diagnosis") == d) for d in sorted(set(str(r.get("diagnosis")) for r in route_rows))},
        })

    write_csv(args.output_csv, rows)
    write_jsonl(args.output_jsonl, rows)
    summary = {
        "schema_version": "mga_multiflyby_6d_patch_diagnostics.v0.1",
        "input_jsonl": str(args.input_jsonl),
        "routes_input": len(records),
        "routes_with_missing_source_packet": missing_source,
        "segments_diagnosed": len(rows),
        "diagnosis_counts": diagnosis_counts,
        "route_summaries": route_summaries,
        "notes": [
            "This script replays corrected segments using the same dynamics as mga_multiflyby_patch_corrector_v0_2.py.",
            "Velocity mismatch is diagnostic only. Position-only correction cannot enforce full 6D continuity.",
            "For final target arrival, nonzero relative velocity is expected and should not be treated as a 6D patch failure.",
        ],
    }
    write_json(args.output_json, summary)

    print("="*80)
    print("MGA MULTI-FLYBY 6D PATCH DIAGNOSTICS V0.1")
    print("="*80)
    print(f"Routes input:       {len(records)}")
    print(f"Missing source:     {missing_source}")
    print(f"Segments diagnosed: {len(rows)}")
    print("Diagnosis counts:")
    for k, v in sorted(diagnosis_counts.items(), key=lambda kv: (-kv[1], kv[0])):
        print(f"  - {k:<42} {v}")
    print("\nTop segment diagnostics:")
    for r in sorted(rows, key=lambda x: finite(x.get("local_velocity_delta_m_s"), finite(x.get("central_velocity_delta_m_s"), -1)), reverse=True)[:10]:
        print(
            f"  route={r.get('route_index')} seg={r.get('segment_index')} {r.get('origin_label')} -> {r.get('target_label')} | "
            f"v_c={finite(r.get('central_velocity_delta_m_s')):.1f} m/s | "
            f"v_loc={finite(r.get('local_velocity_delta_m_s')):.1f} m/s | "
            f"speedΔ={finite(r.get('local_speed_delta_m_s')):.1f} m/s | "
            f"ang={finite(r.get('local_angle_deg')):.1f}° | {r.get('diagnosis')}"
        )
    print("="*80)
    print(f"[OK] wrote CSV:   {args.output_csv}")
    print(f"[OK] wrote JSONL: {args.output_jsonl}")
    print(f"[OK] wrote JSON:  {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
