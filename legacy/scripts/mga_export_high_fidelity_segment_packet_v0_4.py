#!/usr/bin/env python3
"""
mga_export_high_fidelity_segment_packet_v0_4.py

Build an explicit high-fidelity handoff segment packet directly from the stitched
patch-corrector JSONL (mga_stitched_patch_corrector_v0_1.py).

V0.4 change: match B-plane packets using the same permissive strategy as
mga_stitched_patch_corrector_v0_1.py: explicit IDs first, then sequence and
periapsis epoch fallback. This avoids reconstructing the first-leg velocity via
Lambert when the original b-plane leg velocity is available but IDs were lost.

Input source of truth:
  --corrected-jsonl data/.../mga_stitched_corrected_*.jsonl

Output segments per route:
  0) heliocentric_pre_flyby: origin centre -> flyby SOI-in patch
  1) local_flyby_twobody: flyby SOI-in -> periapsis -> SOI-out
  2) heliocentric_post_flyby: flyby SOI-out -> final target centre

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

SCHEMA_VERSION = "mga_high_fidelity_segment_packet.v0.4"
SECONDS_PER_DAY = 86400.0
Vec3 = Tuple[float, float, float]


def finite(x: Any, default: float = math.nan) -> float:
    try:
        y = float(x)
    except (TypeError, ValueError):
        return default
    return y if math.isfinite(y) else default


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


def load_rows(path: Path) -> List[Dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        return read_jsonl(path)
    data = load_json(path)
    # Single corrected record
    if "pre_arc" in data and "post_arc" in data and "source_stitched_packet" in data:
        return [data]
    for key in ("records", "results", "corrected_routes", "packets", "routes", "top_results"):
        rows = data.get(key)
        if isinstance(rows, list):
            return [dict(r) for r in rows if isinstance(r, Mapping)]
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
        for row in rows:
            f.write(json.dumps(sanitize(row), ensure_ascii=False, separators=(",", ":"), allow_nan=False))
            f.write("\n")


def load_body_catalog(path: Optional[Path]) -> Dict[str, Any]:
    if not path:
        return {}
    data = load_json(path)
    bodies = data.get("bodies")
    return bodies if isinstance(bodies, dict) else {}


def body_info(catalog: Mapping[str, Any], name: str) -> Dict[str, Any]:
    ent = catalog.get(name)
    if ent is None:
        for k, v in catalog.items():
            if str(k).lower() == name.lower():
                ent = v
                break
    return dict(ent) if isinstance(ent, Mapping) else {}


def spice_state(spice: Any, body: str, et: float, frame: str, central: str) -> Tuple[Vec3, Vec3]:
    st, _lt = spice.spkezr(str(body), float(et), str(frame), "NONE", str(central))
    return (float(st[0]), float(st[1]), float(st[2])), (float(st[3]), float(st[4]), float(st[5]))


def lambert_departure_velocity(r0: Vec3, r1: Vec3, tof_s: float, mu_km3_s2: float) -> Vec3:
    """Return the zero-revolution Lambert departure velocity in km/s.

    PyKEP returns a list of possible velocities. We use the first zero-rev
    branch, consistent with the scout/refiner V0.1/V0.2 usage.
    """
    if not (tof_s > 0.0):
        raise ValueError("Lambert time of flight must be positive")
    try:
        import pykep as pk  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on user env
        raise RuntimeError(f"pykep import failed while reconstructing Lambert velocity: {exc}") from exc
    lp = pk.lambert_problem(list(map(float, r0)), list(map(float, r1)), float(tof_s), float(mu_km3_s2), max_revs=0)
    v1s = lp.get_v1()
    if not v1s:
        raise RuntimeError("PyKEP Lambert returned no departure velocity")
    return (float(v1s[0][0]), float(v1s[0][1]), float(v1s[0][2]))


def reconstruct_pre_leg_departure_velocity(corrected: Mapping[str, Any], origin: str, flyby: str, depart_et: float, pe_et: float, spice: Any, args: argparse.Namespace) -> Optional[Vec3]:
    """Fallback for legacy schemas missing bplane leg0 departure velocity.

    The connected-flyby optimizer defines the flyby encounter epoch at periapsis
    time. Its first Lambert leg is origin centre -> flyby body centre at that
    epoch. The stitched corrector then adds a small pre-flyby TCM to hit the
    SOI-in patch. Rebuilding that Lambert leg is therefore a stable fallback.
    """
    try:
        r0, _v0_body = spice_state(spice, origin, depart_et, args.frame, args.central_body)
        r1, _v1_body = spice_state(spice, flyby, pe_et, args.frame, args.central_body)
        return lambert_departure_velocity(r0, r1, pe_et - depart_et, float(args.mu_central_km3_s2))
    except Exception:
        return None


def load_bplane_packets(path: Optional[Path]) -> List[Dict[str, Any]]:
    if not path:
        return []
    if path.suffix.lower() == ".jsonl":
        return read_jsonl(path)
    data = load_json(path)
    for key in ("packets", "bplane_packets", "records", "results", "top_packets"):
        rows = data.get(key)
        if isinstance(rows, list):
            return [dict(r) for r in rows if isinstance(r, Mapping)]
    if "legs" in data:
        return [data]
    return []


def matching_bplane(corrected: Mapping[str, Any], bplanes: Sequence[Mapping[str, Any]]) -> Optional[Mapping[str, Any]]:
    """Match the corrected stitched record to its B-plane packet.

    V0.4 intentionally mirrors mga_stitched_patch_corrector_v0_1.py. The
    corrected record has already been generated by that corrector, so using the
    same permissive matching rule preserves the exact first-leg Lambert branch
    and avoids the unsafe V0.3 fallback reconstruction whenever possible.
    """
    stitched = corrected.get("source_stitched_packet") if isinstance(corrected.get("source_stitched_packet"), Mapping) else {}
    target = stitched.get("target") if isinstance(stitched.get("target"), Mapping) else {}

    # Strong IDs first.
    keys: List[str] = []
    for obj in (corrected, stitched, target):
        if isinstance(obj, Mapping):
            for k in ("packet_id", "route_id", "correction_id", "closure_id", "stitched_packet_id", "target_spec_id", "local_target_id"):
                val = obj.get(k)
                if val is not None:
                    keys.append(str(val))
    for bp in bplanes:
        for k in ("packet_id", "route_id", "correction_id", "closure_id", "target_spec_id", "local_target_id"):
            val = bp.get(k)
            if val is not None and str(val) in keys:
                return bp

    # Match by sequence, as the stitched corrector did.
    seq = str(corrected.get("sequence") or target.get("sequence") or "")
    if seq:
        seq_matches = [bp for bp in bplanes if str(bp.get("sequence") or "") == seq]
        if len(seq_matches) == 1:
            return seq_matches[0]
        if len(seq_matches) > 1:
            pe = finite((stitched.get("patch_epochs") or {}).get("periapsis_et")) if isinstance(stitched.get("patch_epochs"), Mapping) else math.nan
            if math.isfinite(pe):
                best = None
                best_dt = math.inf
                for bp in seq_matches:
                    flys = bp.get("flybys") if isinstance(bp.get("flybys"), list) else []
                    for fb in flys:
                        if isinstance(fb, Mapping):
                            fpe = finite(fb.get("encounter_et") or fb.get("periapsis_et"))
                            dt = abs(fpe - pe) if math.isfinite(fpe) else math.inf
                            if dt < best_dt:
                                best_dt = dt; best = bp
                if best is not None and best_dt < 10.0:
                    return best
            # Last-resort but deterministic: this is exactly what allowed the
            # stitched corrector to produce the current corrected record.
            return seq_matches[0]

    # Final fallback: approximate periapsis epoch even if sequence key was lost.
    pe = finite((stitched.get("patch_epochs") or {}).get("periapsis_et")) if isinstance(stitched.get("patch_epochs"), Mapping) else math.nan
    if math.isfinite(pe):
        for bp in bplanes:
            flys = bp.get("flybys") if isinstance(bp.get("flybys"), list) else []
            for fb in flys:
                if isinstance(fb, Mapping):
                    fpe = finite(fb.get("encounter_et") or fb.get("periapsis_et"))
                    if math.isfinite(fpe) and abs(fpe - pe) < 10.0:
                        return bp
    return None


def state_block(stitched: Mapping[str, Any], key: str, subkey: str) -> Tuple[Vec3, Vec3]:
    ps = stitched.get("patch_states") if isinstance(stitched.get("patch_states"), Mapping) else {}
    block = ps.get(key) if isinstance(ps.get(key), Mapping) else {}
    src = block.get(subkey) if isinstance(block.get(subkey), Mapping) else {}
    r = vec3(src.get("r_km")); v = vec3(src.get("v_km_s"))
    if r is None or v is None:
        raise ValueError(f"missing patch state {key}.{subkey}")
    return r, v


def leg0_departure_velocity(corrected: Mapping[str, Any], bp: Optional[Mapping[str, Any]]) -> Optional[Vec3]:
    if isinstance(bp, Mapping):
        legs = bp.get("legs")
        if isinstance(legs, list):
            for leg in legs:
                if isinstance(leg, Mapping) and int(finite(leg.get("leg_index"), -1)) == 0:
                    return vec3(leg.get("sc_v_depart_corrected_km_s")) or vec3(leg.get("sc_v_depart_km_s"))
            if legs and isinstance(legs[0], Mapping):
                return vec3(legs[0].get("sc_v_depart_corrected_km_s")) or vec3(legs[0].get("sc_v_depart_km_s"))
    # Some future corrected schemas may store it in pre_arc.
    pre = corrected.get("pre_arc") if isinstance(corrected.get("pre_arc"), Mapping) else {}
    return vec3(pre.get("v0_base_km_s")) or vec3(pre.get("v_depart_base_km_s"))


def sequence_list(x: Any) -> List[str]:
    if isinstance(x, list):
        return [str(s).strip() for s in x if str(s).strip()]
    s = str(x or "")
    if "->" in s:
        return [p.strip() for p in s.split("->") if p.strip()]
    if "," in s:
        return [p.strip() for p in s.split(",") if p.strip()]
    return [s.strip()] if s.strip() else []


def make_packet(corrected: Mapping[str, Any], bplanes: Sequence[Mapping[str, Any]], spice: Any, catalog: Mapping[str, Any], args: argparse.Namespace, rank: int) -> Dict[str, Any]:
    failures: List[str] = []
    if not corrected.get("ok", False):
        failures.append("corrected_record_not_ok")
    if not corrected.get("pass_correction", False):
        failures.append("corrected_record_did_not_pass_correction")
    stitched = corrected.get("source_stitched_packet") if isinstance(corrected.get("source_stitched_packet"), Mapping) else {}
    if not stitched:
        failures.append("missing_source_stitched_packet")
    target = stitched.get("target") if isinstance(stitched.get("target"), Mapping) else {}
    bp = matching_bplane(corrected, bplanes)
    bplane_match_note = "matched" if bp is not None else "not_matched_reconstructing_lambert"
    seq = sequence_list(corrected.get("sequence") or target.get("sequence") or (bp or {}).get("sequence"))
    if len(seq) < 3:
        failures.append("sequence_not_three_or_more_bodies")
    origin = seq[0] if seq else str(corrected.get("pre_arc", {}).get("origin", ""))
    flyby = str(corrected.get("flyby_body") or target.get("body") or (seq[1] if len(seq) > 1 else ""))
    final = seq[-1] if seq else str(corrected.get("post_arc", {}).get("target", ""))
    epochs = corrected.get("epochs") if isinstance(corrected.get("epochs"), Mapping) else {}
    depart_et = finite(epochs.get("depart_et")); entry_et = finite(epochs.get("entry_et")); pe_et = finite(epochs.get("periapsis_et")); exit_et = finite(epochs.get("exit_et")); arrival_et = finite(epochs.get("arrival_et"))
    for name, val in [("depart_et",depart_et),("entry_et",entry_et),("periapsis_et",pe_et),("exit_et",exit_et),("arrival_et",arrival_et)]:
        if not math.isfinite(val): failures.append(f"missing_{name}")

    pre = corrected.get("pre_arc") if isinstance(corrected.get("pre_arc"), Mapping) else {}
    post = corrected.get("post_arc") if isinstance(corrected.get("post_arc"), Mapping) else {}
    q = corrected.get("quality") if isinstance(corrected.get("quality"), Mapping) else {}
    pre_tcm = vec3(pre.get("dv_correction_km_s")) or (0.0,0.0,0.0)
    post_tcm = vec3(post.get("dv_correction_km_s")) or (0.0,0.0,0.0)
    vdep_base = leg0_departure_velocity(corrected, bp)

    try:
        r_origin, v_origin = spice_state(spice, origin, depart_et, args.frame, args.central_body)
    except Exception as exc:
        failures.append(f"origin_spice_state_failed:{exc}"); r_origin = v_origin = None  # type: ignore

    if vdep_base is None:
        vdep_base = reconstruct_pre_leg_departure_velocity(corrected, origin, flyby, depart_et, pe_et, spice, args)
        if vdep_base is not None:
            bplane_match_note = "fallback_lambert_reconstructed"
    if vdep_base is None:
        failures.append("missing_pre_leg_departure_velocity")
        vdep_final = None
    else:
        vdep_final = vadd(vdep_base, pre_tcm)
    try:
        r_target, v_target = spice_state(spice, final, arrival_et, args.frame, args.central_body)
    except Exception as exc:
        failures.append(f"target_spice_state_failed:{exc}"); r_target = v_target = None  # type: ignore
    try:
        r_entry, v_entry = state_block(stitched, "entry_soi", "spacecraft_state_central")
        r_pe, v_pe = state_block(stitched, "periapsis", "spacecraft_state_central")
        r_exit, v_exit_base = state_block(stitched, "exit_soi", "spacecraft_state_central")
        v_exit_final = vadd(v_exit_base, post_tcm)
        r_entry_loc, v_entry_loc = state_block(stitched, "entry_soi", "local_body_centered")
        r_pe_loc, v_pe_loc = state_block(stitched, "periapsis", "local_body_centered")
        r_exit_loc, v_exit_loc = state_block(stitched, "exit_soi", "local_body_centered")
    except Exception as exc:
        failures.append(str(exc))
        r_entry = v_entry = r_pe = v_pe = r_exit = v_exit_base = v_exit_final = None  # type: ignore
        r_entry_loc = v_entry_loc = r_pe_loc = v_pe_loc = r_exit_loc = v_exit_loc = None  # type: ignore

    fly_info = body_info(catalog, flyby)
    central_mu = float(args.mu_central_km3_s2)
    segments: List[Dict[str, Any]] = []
    if r_origin is not None and vdep_final is not None and r_entry is not None and r_entry_loc is not None:
        segments.append({
            "segment_index": 0,
            "segment_type": "heliocentric_patched_pre_flyby",
            "from": origin,
            "to": f"{flyby}_SOI_in",
            "target_body": flyby,
            "t0_et": depart_et,
            "t1_et": entry_et,
            "tof_days": (entry_et - depart_et)/SECONDS_PER_DAY,
            "central_body": args.central_body,
            "central_mu_km3_s2": central_mu,
            "frame": args.frame,
            "r0_km": r_origin,
            "v0_km_s": vdep_final,
            "r1_target_km": r_entry,
            "v1_patch_km_s": v_entry,
            "r1_target_body_centered_km": r_entry_loc,
            "v1_target_body_centered_km_s": v_entry_loc,
            "base_departure_velocity_source": bplane_match_note,
            "base_departure_velocity_km_s": vdep_base,
            "applied_tcm_km_s": pre_tcm,
            "applied_tcm_m_s": vnorm(pre_tcm)*1000.0,
            "recommended_rebound_compare": "spacecraft minus integrated target_body should match r1_target_body_centered_km",
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
            "mu_body_km3_s2": finite(fly_info.get("mu_km3_s2")),
            "r0_body_centered_km": r_entry_loc,
            "v0_body_centered_km_s": v_entry_loc,
            "r_periapsis_body_centered_km": r_pe_loc,
            "v_periapsis_body_centered_km_s": v_pe_loc,
            "r1_body_centered_km": r_exit_loc,
            "v1_body_centered_km_s": v_exit_loc,
            "periapsis_altitude_km": target.get("hyperbola", {}).get("periapsis_altitude_km") if isinstance(target.get("hyperbola"), Mapping) else q.get("periapsis_altitude_km"),
            "rp_margin_km": q.get("rp_margin_km"),
        })
    if r_exit is not None and v_exit_final is not None and r_target is not None:
        segments.append({
            "segment_index": 2,
            "segment_type": "heliocentric_patched_post_flyby",
            "from": f"{flyby}_SOI_out",
            "to": final,
            "target_body": final,
            "t0_et": exit_et,
            "t1_et": arrival_et,
            "tof_days": (arrival_et - exit_et)/SECONDS_PER_DAY,
            "central_body": args.central_body,
            "central_mu_km3_s2": central_mu,
            "frame": args.frame,
            "r0_km": r_exit,
            "v0_km_s": v_exit_final,
            "r1_target_km": r_target,
            "v1_target_km_s": v_target,
            "r1_target_body_centered_km": (0.0,0.0,0.0),
            "v1_target_body_centered_km_s": (0.0,0.0,0.0),
            "applied_tcm_km_s": post_tcm,
            "applied_tcm_m_s": vnorm(post_tcm)*1000.0,
            "recommended_rebound_compare": "spacecraft minus integrated target_body should match zero vector",
        })

    total_patch = finite(q.get("total_patch_correction_m_s"))
    source_dep = finite(q.get("source_total_departure_correction_m_s"), 0.0)
    known_total = source_dep + total_patch if math.isfinite(total_patch) else math.nan
    arrival_after = finite(post.get("miss_after_km"))
    rp_margin = finite(q.get("rp_margin_km"))
    ready = bool(corrected.get("pass_correction")) and len(segments) == 3 and total_patch <= args.max_patch_dv_m_s and rp_margin >= args.min_rp_margin_km and arrival_after <= args.max_arrival_miss_km and not failures
    out = {
        "schema_version": SCHEMA_VERSION,
        "high_fidelity_packet_id": stable_id("hfpkt", {"corr": corrected.get("correction_id"), "rank": rank, "epochs": epochs}),
        "rank": rank,
        "ready_for_rebound_ias15": bool(ready),
        "ready_for_tudat": bool(ready),
        "failures": failures,
        "sequence": " -> ".join(seq),
        "flyby_body": flyby,
        "class": "A" if ready else "unready",
        "central_body": args.central_body,
        "central_mu_km3_s2": central_mu,
        "frame": args.frame,
        "epochs": {"depart_et": depart_et, "entry_et": entry_et, "periapsis_et": pe_et, "exit_et": exit_et, "arrival_et": arrival_et},
        "quality": {
            "known_total_corrections_m_s": known_total,
            "stitched_patch_total_m_s": total_patch,
            "source_departure_corrections_m_s": source_dep,
            "pre_flyby_tcm_m_s": pre.get("dv_correction_m_s"),
            "post_flyby_tcm_m_s": post.get("dv_correction_m_s"),
            "entry_miss_after_km": pre.get("miss_after_km"),
            "entry_velocity_miss_after_m_s": pre.get("velocity_miss_after_m_s"),
            "arrival_miss_after_km": post.get("miss_after_km"),
            "arrival_vinf_after_km_s": post.get("arrival_vinf_after_km_s"),
            "rp_margin_km": rp_margin,
            "periapsis_altitude_km": q.get("periapsis_altitude_km"),
        },
        "segments": segments,
        "source_ids": {
            "correction_id": corrected.get("correction_id"),
            "stitched_packet_id": corrected.get("stitched_packet_id"),
            "bplane_packet_id": (bp or {}).get("packet_id") if isinstance(bp, Mapping) else None,
            "pre_departure_velocity_source": bplane_match_note,
            "route_id": corrected.get("route_id") or (bp or {}).get("route_id") if isinstance(bp, Mapping) else corrected.get("route_id"),
        },
    }
    if args.embed_source:
        out["source_corrected_record"] = corrected
        if bp is not None:
            out["source_bplane_packet"] = bp
    return out


def flat_row(p: Mapping[str, Any]) -> Dict[str, Any]:
    q = p.get("quality") if isinstance(p.get("quality"), Mapping) else {}
    return {
        "rank": p.get("rank"),
        "high_fidelity_packet_id": p.get("high_fidelity_packet_id"),
        "ready_for_rebound_ias15": int(bool(p.get("ready_for_rebound_ias15"))),
        "sequence": p.get("sequence"),
        "flyby_body": p.get("flyby_body"),
        "class": p.get("class"),
        "segments": len(p.get("segments") or []),
        "known_total_corrections_m_s": q.get("known_total_corrections_m_s"),
        "stitched_patch_total_m_s": q.get("stitched_patch_total_m_s"),
        "rp_margin_km": q.get("rp_margin_km"),
        "arrival_miss_after_km": q.get("arrival_miss_after_km"),
        "failures": ";".join(str(x) for x in (p.get("failures") or [])),
    }


def write_csv(path: Path, packets: Sequence[Mapping[str, Any]]) -> None:
    fields = list(flat_row({}).keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for p in packets:
            w.writerow(flat_row(p))


def stats(vals: Sequence[float]) -> Dict[str, Optional[float]]:
    xs = sorted([v for v in vals if math.isfinite(v)])
    if not xs:
        return {"min": None, "median": None, "max": None}
    return {"min": xs[0], "median": xs[len(xs)//2], "max": xs[-1]}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export high-fidelity segment packets directly from stitched patch-corrected routes.")
    p.add_argument("--corrected-jsonl", required=True, type=Path, help="JSON/JSONL from mga_stitched_patch_corrector_v0_1.py")
    p.add_argument("--bplane-packet", required=True, type=Path)
    p.add_argument("--bsp", required=True, type=Path)
    p.add_argument("--tpc", type=Path, default=None)
    p.add_argument("--body-catalog", required=True, type=Path)
    p.add_argument("--central-body", default="Sun")
    p.add_argument("--mu-central-km3-s2", required=True, type=float)
    p.add_argument("--frame", default="J2000")
    p.add_argument("--top-n", type=int, default=0)
    p.add_argument("--max-patch-dv-m-s", type=float, default=25.0)
    p.add_argument("--min-rp-margin-km", type=float, default=800.0)
    p.add_argument("--max-arrival-miss-km", type=float, default=10.0)
    p.add_argument("--embed-source", action="store_true")
    p.add_argument("--output-csv", required=True, type=Path)
    p.add_argument("--output-jsonl", required=True, type=Path)
    p.add_argument("--output-json", required=True, type=Path)
    p.add_argument("--output-best-json", required=True, type=Path)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    import spiceypy as spice  # type: ignore
    spice.kclear()
    if args.tpc:
        spice.furnsh(str(args.tpc))
    spice.furnsh(str(args.bsp))
    catalog = load_body_catalog(args.body_catalog)
    bplanes = load_bplane_packets(args.bplane_packet)
    records = [r for r in load_rows(args.corrected_jsonl) if isinstance(r, Mapping)]
    records = [r for r in records if bool(r.get("ok"))]
    # preserve corrector ranking: pass first, lowest total patch dv first
    records.sort(key=lambda r: (0 if r.get("pass_correction") else 1, finite((r.get("quality") or {}).get("total_patch_correction_m_s") if isinstance(r.get("quality"), Mapping) else None, math.inf)))
    if args.top_n > 0:
        records = records[:args.top_n]
    packets = [make_packet(r, bplanes, spice, catalog, args, i+1) for i, r in enumerate(records)]
    packets.sort(key=lambda p: (0 if p.get("ready_for_rebound_ias15") else 1, finite((p.get("quality") or {}).get("known_total_corrections_m_s") if isinstance(p.get("quality"), Mapping) else None, math.inf)))
    for i, p in enumerate(packets, start=1):
        p["rank"] = i
    write_csv(args.output_csv, packets)
    write_jsonl(args.output_jsonl, packets)
    corr = [finite((p.get("quality") or {}).get("known_total_corrections_m_s") if isinstance(p.get("quality"), Mapping) else None) for p in packets]
    rp = [finite((p.get("quality") or {}).get("rp_margin_km") if isinstance(p.get("quality"), Mapping) else None) for p in packets]
    summary = {
        "schema_version": SCHEMA_VERSION + ".summary",
        "input_records": len(records),
        "packets_written": len(packets),
        "ready_packets": sum(1 for p in packets if p.get("ready_for_rebound_ias15")),
        "thresholds": {"max_patch_dv_m_s": args.max_patch_dv_m_s, "min_rp_margin_km": args.min_rp_margin_km, "max_arrival_miss_km": args.max_arrival_miss_km},
        "stats": {"known_total_corrections_m_s": stats(corr), "rp_margin_km": stats(rp)},
        "top_packets": [flat_row(p) for p in packets[:10]],
    }
    write_json(args.output_json, summary)
    write_json(args.output_best_json, packets[0] if packets else {"schema_version": SCHEMA_VERSION, "ready_for_rebound_ias15": False, "message": "no packets"})
    print("="*80)
    print("MGA HIGH-FIDELITY SEGMENT PACKET EXPORT V0.4")
    print("="*80)
    print(f"Input corrected records: {len(records)}")
    print(f"Packets written:         {len(packets)}")
    print(f"Ready for REBOUND IAS15: {summary['ready_packets']}")
    print(f"Known corr m/s:          min={summary['stats']['known_total_corrections_m_s']['min']} median={summary['stats']['known_total_corrections_m_s']['median']} max={summary['stats']['known_total_corrections_m_s']['max']}")
    print(f"rp margin km:            min={summary['stats']['rp_margin_km']['min']} median={summary['stats']['rp_margin_km']['median']} max={summary['stats']['rp_margin_km']['max']}")
    print("\nTop packets:")
    for row in summary["top_packets"]:
        print(f" {row['rank']}. {row['sequence']} | ready={bool(row['ready_for_rebound_ias15'])} | segs={row['segments']} | corr={finite(row['known_total_corrections_m_s']):.3f} m/s | rp_margin={finite(row['rp_margin_km']):.1f} km | failures={row['failures']}")
    print("="*80)
    print(f"[OK] wrote CSV:       {args.output_csv}")
    print(f"[OK] wrote JSONL:     {args.output_jsonl}")
    print(f"[OK] wrote JSON:      {args.output_json}")
    print(f"[OK] wrote best JSON: {args.output_best_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
