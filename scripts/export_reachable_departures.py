#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import spiceypy as spice

from ksp_mga.native.impulse_server_client_v0_2 import PrincipiaImpulseServerV2


def norm(v: np.ndarray) -> float:
    return float(np.linalg.norm(v))


def unit(v: np.ndarray) -> np.ndarray:
    n = norm(v)
    if n <= 0:
        raise ValueError("zero vector")
    return v / n


def angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    na = norm(a); nb = norm(b)
    if na <= 0 or nb <= 0:
        return float("inf")
    c = float(np.dot(a, b) / (na * nb))
    c = max(-1.0, min(1.0, c))
    return math.degrees(math.acos(c))


def raw_to_levela(v: np.ndarray) -> np.ndarray:
    x, y, z = v
    return np.array([-y, z, x], dtype=float)


def levela_to_raw(v: np.ndarray) -> np.ndarray:
    x, y, z = v
    return np.array([z, -x, y], dtype=float)


def safe_float(x: Any, default: float = math.inf) -> float:
    try:
        if x is None or x == "":
            return default
        v = float(x)
        if math.isnan(v):
            return default
        return v
    except Exception:
        return default


def arr(row: dict[str, str], *names: str) -> np.ndarray:
    return np.array([float(row[n]) for n in names], dtype=float)


def read_leg_row(path: Path, leg: int) -> dict[str, str]:
    rows = list(csv.DictReader(path.open()))
    for row in rows:
        if int(float(row["leg"])) == leg:
            return row
    raise SystemExit(f"[FAIL] leg {leg} not found in {path}")


def body_state_raw(body: str, t_s: float, center: str, frame: str) -> tuple[np.ndarray, np.ndarray]:
    st, _ = spice.spkezr(body, t_s, frame, "NONE", center)
    r_levela = np.array([1000.0 * st[0], 1000.0 * st[1], 1000.0 * st[2]], dtype=float)
    v_levela = np.array([1000.0 * st[3], 1000.0 * st[4], 1000.0 * st[5]], dtype=float)
    return levela_to_raw(r_levela), levela_to_raw(v_levela)


def body_mu_m3_s2(body: str) -> float:
    _, vals = spice.bodvrd(body, "GM", 1)
    return float(vals[0]) * 1.0e9


def outgoing_osculating_vinf(r_rel: np.ndarray, v_rel: np.ndarray, mu: float) -> tuple[np.ndarray | None, float, float, float]:
    r = norm(r_rel)
    v = norm(v_rel)
    if r <= 0:
        return None, float("nan"), float("nan"), float("nan")
    eps = 0.5 * v * v - mu / r
    if eps <= 0:
        return None, eps, float("nan"), float("nan")
    h = np.cross(r_rel, v_rel)
    h_norm = norm(h)
    if h_norm <= 0:
        return None, eps, float("nan"), float("nan")
    e_vec = np.cross(v_rel, h) / mu - r_rel / r
    e = norm(e_vec)
    if e <= 1.0:
        return None, eps, e, math.sqrt(2.0 * eps)
    p_hat = e_vec / e
    h_hat = h / h_norm
    q_hat = unit(np.cross(h_hat, p_hat))
    f_inf = math.acos(max(-1.0, min(1.0, -1.0 / e)))
    direction = unit(math.cos(f_inf) * p_hat + math.sin(f_inf) * q_hat)
    vinf_mag = math.sqrt(2.0 * eps)
    return vinf_mag * direction, eps, e, vinf_mag


def load_rows(summary_path: Path | None, validated_top_path: Path | None) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    summary: dict[str, Any] = {}
    if summary_path:
        summary = json.loads(summary_path.read_text())
        if validated_top_path is None and summary.get("validated_top_json"):
            validated_top_path = Path(summary["validated_top_json"])
    if validated_top_path is None:
        raise SystemExit("[FAIL] pass --summary-json or --validated-top-json")
    rows = json.loads(validated_top_path.read_text())
    if not isinstance(rows, list):
        raise SystemExit(f"[FAIL] {validated_top_path} is not a list")
    return summary, rows


def row_key(row: dict[str, Any]) -> tuple:
    return (
        round(safe_float(row.get("tb0_s"), 0.0), 6),
        round(safe_float(row.get("dv0_raw_x_m_s"), 0.0), 6),
        round(safe_float(row.get("dv0_raw_y_m_s"), 0.0), 6),
        round(safe_float(row.get("dv0_raw_z_m_s"), 0.0), 6),
        str(row.get("validated_label", row.get("label", ""))),
    )


def pick_rows(rows: list[dict[str, Any]], per_criterion: int) -> list[dict[str, Any]]:
    criteria = [
        ("score", "validated_score"),
        ("vec", "validated_osc_vinf_vec_err_m_s"),
        ("mag", "validated_osc_vinf_mag_err_m_s"),
        ("angle", "validated_osc_vinf_angle_deg"),
        ("pos", "validated_rel_pos_err_km"),
        ("inst_vec", "validated_inst_vrel_vec_err_m_s"),
    ]
    picked: dict[tuple, dict[str, Any]] = {}
    for tag, field in criteria:
        ordered = sorted(rows, key=lambda r: safe_float(r.get(field), math.inf))[:per_criterion]
        for i, r in enumerate(ordered):
            rr = dict(r)
            rr.setdefault("selection_sources", [])
            srcs = list(rr.get("selection_sources", []))
            srcs.append(f"{tag}:{i+1}")
            rr["selection_sources"] = srcs
            k = row_key(rr)
            if k in picked:
                old = picked[k]
                old_srcs = list(old.get("selection_sources", []))
                old_srcs.extend(srcs)
                old["selection_sources"] = sorted(set(old_srcs))
            else:
                picked[k] = rr
    out = list(picked.values())
    out.sort(key=lambda r: safe_float(r.get("validated_osc_vinf_vec_err_m_s"), math.inf))
    return out


def enrich_departure(row: dict[str, Any], idx: int, args: argparse.Namespace, live: dict[str, Any], leg_row: dict[str, str], mu_dep: float, srv: PrincipiaImpulseServerV2) -> dict[str, Any] | None:
    t0 = float(live["ut_s"])
    r0 = np.array(live["r_raw_m"], dtype=float)
    v0 = np.array(live["v_raw_m_s"], dtype=float)
    tb0 = safe_float(row.get("tb0_s"))
    dv0 = np.array([
        safe_float(row.get("dv0_raw_x_m_s")),
        safe_float(row.get("dv0_raw_y_m_s")),
        safe_float(row.get("dv0_raw_z_m_s")),
    ], dtype=float)
    if not np.all(np.isfinite(dv0)) or not math.isfinite(tb0):
        return None

    t_patch = safe_float(row.get("validated_t_s"), args.patch_time_s)
    if args.patch_time_s is not None:
        t_patch = args.patch_time_s
    if not math.isfinite(t_patch):
        t_patch = float(leg_row["t_start_s"])
    if t_patch <= tb0:
        t_patch = tb0 + args.min_patch_after_burn_s

    res = srv.propagate_n(
        req_id=f"export_dep_{idx}",
        t0_s=t0,
        t1_s=t_patch,
        r0_m=r0,
        v0_m_s=v0,
        impulses=[(tb0, dv0)],
    )
    if res.status != "ok" or res.final_r_m is None or res.final_v_m_s is None:
        return {
            "departure_id": f"dep_{idx:05d}",
            "status": "PROPAGATION_FAILED",
            "server_status": res.status,
            "server_message": res.message,
            "source_row": row,
        }

    patch_r = np.array(res.final_r_m, dtype=float)
    patch_v = np.array(res.final_v_m_s, dtype=float)
    dep_body_r, dep_body_v = body_state_raw(args.dep_body, t_patch, args.center, args.frame)
    rel_r = patch_r - dep_body_r
    rel_v = patch_v - dep_body_v
    vinf_vec, eps, ecc, vinf_mag = outgoing_osculating_vinf(rel_r, rel_v, mu_dep)

    burn_info: dict[str, Any] = {}
    if res.burns:
        b = res.burns[0]
        burn_info = {
            "burn0_r_raw_m": list(map(float, b.r_m)),
            "burn0_v_before_raw_m_s": list(map(float, b.v_before_m_s)),
            "burn0_v_after_raw_m_s": list(map(float, b.v_after_m_s)),
        }

    # Original leg reference, useful for diagnostics only.
    leg_start_r = arr(leg_row, "start_x_raw_m", "start_y_raw_m", "start_z_raw_m")
    leg_start_v = arr(leg_row, "start_vx_raw_m_s", "start_vy_raw_m_s", "start_vz_raw_m_s")
    leg_dv = np.zeros(3)
    if "dvx_m_s" in leg_row:
        leg_dv = arr(leg_row, "dvx_m_s", "dvy_m_s", "dvz_m_s")
    leg_ref_v = leg_start_v + leg_dv if args.reference_mode == "post_correction" else leg_start_v
    leg_body_r, leg_body_v = body_state_raw(args.dep_body, float(leg_row["t_start_s"]), args.center, args.frame)
    leg_ref_vinf, leg_ref_eps, leg_ref_ecc, leg_ref_vinf_mag = outgoing_osculating_vinf(leg_start_r - leg_body_r, leg_ref_v - leg_body_v, mu_dep)

    if vinf_vec is not None and leg_ref_vinf is not None:
        vinf_vec_err = norm(vinf_vec - leg_ref_vinf)
        vinf_mag_err = abs(norm(vinf_vec) - norm(leg_ref_vinf))
        vinf_ang = angle_deg(vinf_vec, leg_ref_vinf)
    else:
        vinf_vec_err = vinf_mag_err = vinf_ang = float("inf")

    return {
        "departure_id": f"dep_{idx:05d}",
        "status": "OK",
        "selection_sources": row.get("selection_sources", []),
        "source_metrics": row,
        "t0_s": t0,
        "tb0_s": tb0,
        "patch_t_s": t_patch,
        "dv0_raw_m_s": dv0.tolist(),
        "dv0_levela_m_s": raw_to_levela(dv0).tolist(),
        "dv0_norm_m_s": norm(dv0),
        "patch_r_raw_m": patch_r.tolist(),
        "patch_v_raw_m_s": patch_v.tolist(),
        "patch_r_levela_m": raw_to_levela(patch_r).tolist(),
        "patch_v_levela_m_s": raw_to_levela(patch_v).tolist(),
        "dep_body_r_raw_m": dep_body_r.tolist(),
        "dep_body_v_raw_m_s": dep_body_v.tolist(),
        "rel_r_from_dep_body_raw_m": rel_r.tolist(),
        "rel_v_from_dep_body_raw_m_s": rel_v.tolist(),
        "rel_distance_from_dep_body_km": norm(rel_r) / 1000.0,
        "rel_speed_from_dep_body_m_s": norm(rel_v),
        "osculating_escape_energy_m2_s2": eps,
        "osculating_eccentricity": ecc,
        "osculating_vinf_m_s": vinf_mag,
        "osculating_vinf_raw_m_s": None if vinf_vec is None else vinf_vec.tolist(),
        "reference_leg_vinf_raw_m_s": None if leg_ref_vinf is None else leg_ref_vinf.tolist(),
        "reference_leg_vinf_m_s": leg_ref_vinf_mag,
        "reference_vinf_vec_err_m_s": vinf_vec_err,
        "reference_vinf_mag_err_m_s": vinf_mag_err,
        "reference_vinf_angle_deg": vinf_ang,
        **burn_info,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    flat_rows: list[dict[str, Any]] = []
    for r in rows:
        rr: dict[str, Any] = {}
        for k, v in r.items():
            if isinstance(v, (list, dict)):
                rr[k] = json.dumps(v, separators=(",", ":"))
            else:
                rr[k] = v
            if k not in fields:
                fields.append(k)
        flat_rows.append(rr)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(flat_rows)


def main() -> int:
    ap = argparse.ArgumentParser(description="Export a canonical Pareto set of reachable real LKO departures.")
    ap.add_argument("--summary-json", type=Path, default=None, help="burn0_vinf_fast_summary.json from v3 fast/pareto")
    ap.add_argument("--validated-top-json", type=Path, default=None, help="burn0_vinf_validated_top.json")
    ap.add_argument("--plugin-b64", required=True)
    ap.add_argument("--server", default="principia_impulsive_particle_server")
    ap.add_argument("--live-state-json", type=Path, required=True)
    ap.add_argument("--leg-optimizations", type=Path, required=True)
    ap.add_argument("--leg", type=int, default=1)
    ap.add_argument("--bsp", type=Path, required=True)
    ap.add_argument("--tpc", type=Path, required=True)
    ap.add_argument("--dep-body", default="KERBIN")
    ap.add_argument("--center", default="SUN")
    ap.add_argument("--frame", default="J2000")
    ap.add_argument("--reference-mode", choices=["post_correction", "pre_correction"], default="post_correction")
    ap.add_argument("--per-criterion", type=int, default=30)
    ap.add_argument("--max-departures", type=int, default=120)
    ap.add_argument("--patch-time-s", type=float, default=None, help="Default: each row's validated_t_s or leg t_start_s")
    ap.add_argument("--min-patch-after-burn-s", type=float, default=1800.0)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()

    spice.kclear(); spice.furnsh(str(args.tpc)); spice.furnsh(str(args.bsp))
    summary, rows = load_rows(args.summary_json, args.validated_top_json)
    picked = pick_rows(rows, args.per_criterion)[: args.max_departures]
    live = json.loads(args.live_state_json.read_text())
    leg_row = read_leg_row(args.leg_optimizations, args.leg)
    mu_dep = body_mu_m3_s2(args.dep_body)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print("=== EXPORT REACHABLE DEPARTURES ===")
    print(f"input rows     : {len(rows)}")
    print(f"picked rows    : {len(picked)}")
    print(f"output_dir     : {args.output_dir}")

    departures: list[dict[str, Any]] = []
    with PrincipiaImpulseServerV2(args.server, args.plugin_b64) as srv:
        print(f"ready          : {srv.ready_line}")
        if not srv.ping():
            raise SystemExit("[FAIL] server PING failed")
        for i, row in enumerate(picked, start=1):
            dep = enrich_departure(row, i, args, live, leg_row, mu_dep, srv)
            if dep is not None:
                departures.append(dep)
            if i <= 20 and dep is not None:
                print(
                    f"{dep['departure_id']} {dep.get('status'):<18} "
                    f"tb0={safe_float(dep.get('tb0_s')):.3f} "
                    f"dv0={safe_float(dep.get('dv0_norm_m_s')):8.2f} "
                    f"vinf={safe_float(dep.get('osculating_vinf_m_s')):8.2f} "
                    f"ref_vec={safe_float(dep.get('reference_vinf_vec_err_m_s')):8.2f} "
                    f"ref_ang={safe_float(dep.get('reference_vinf_angle_deg')):7.3f} "
                    f"sources={','.join(dep.get('selection_sources', []))}"
                )

    departures_ok = [d for d in departures if d.get("status") == "OK"]
    departures_ok.sort(key=lambda d: safe_float(d.get("reference_vinf_vec_err_m_s"), math.inf))
    out = {
        "schema": "reachable_departures.v1",
        "source_summary_json": None if args.summary_json is None else str(args.summary_json),
        "source_validated_top_json": summary.get("validated_top_json") or (None if args.validated_top_json is None else str(args.validated_top_json)),
        "leg": args.leg,
        "dep_body": args.dep_body,
        "reference_mode": args.reference_mode,
        "frame_contract": {
            "raw_to_levela": "(X,Y,Z)->(-Y,+Z,+X)",
            "levela_to_raw": "(X,Y,Z)->(+Z,-X,+Y)",
        },
        "n_departures": len(departures_ok),
        "departures": departures_ok,
    }
    (args.output_dir / "reachable_departures.json").write_text(json.dumps(out, indent=2) + "\n")
    write_csv(args.output_dir / "reachable_departures.csv", departures_ok)
    print(f"[OK] wrote {args.output_dir / 'reachable_departures.json'}")
    print(f"[OK] wrote {args.output_dir / 'reachable_departures.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
