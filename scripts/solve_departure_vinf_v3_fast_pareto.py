#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import heapq
import json
import math
import multiprocessing as mp
import os
import sys
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
    na = norm(a)
    nb = norm(b)
    if na <= 0 or nb <= 0:
        return float("inf")
    c = float(np.dot(a, b) / (na * nb))
    c = max(-1.0, min(1.0, c))
    return math.degrees(math.acos(c))


def levela_to_raw(v: np.ndarray) -> np.ndarray:
    # LevelA/SPICE -> Principia raw = [+Z, -X, +Y]
    x, y, z = v
    return np.array([z, -x, y], dtype=float)


def raw_to_levela(v: np.ndarray) -> np.ndarray:
    x, y, z = v
    return np.array([-y, z, x], dtype=float)


def arr(row: dict[str, str], *names: str) -> np.ndarray:
    return np.array([float(row[n]) for n in names], dtype=float)


def read_leg_row(path: Path, leg: int) -> dict[str, str]:
    rows = list(csv.DictReader(path.open()))
    for row in rows:
        if int(float(row["leg"])) == leg:
            return row
    raise SystemExit(f"[FAIL] leg {leg} not found in {path}")


def parse_float_list(s: str | None) -> list[float]:
    if s is None or str(s).strip() == "":
        return []
    return [float(x.strip()) for x in str(s).replace(";", ",").split(",") if x.strip()]


def float_range(min_v: float, max_v: float, step: float) -> list[float]:
    if step <= 0:
        raise ValueError("step must be positive")
    out: list[float] = []
    x = min_v
    while x <= max_v + 1e-9:
        out.append(float(x))
        x += step
    return out


def symmetric_grid(max_abs: float, step: float) -> list[float]:
    if max_abs < 0:
        raise ValueError("max_abs must be non-negative")
    if max_abs == 0:
        return [0.0]
    vals = float_range(-max_abs, max_abs, step)
    vals = [0.0 if abs(v) < 1e-12 else v for v in vals]
    if 0.0 not in vals:
        vals.append(0.0)
        vals.sort()
    return vals


def rtn_basis(r_rel: np.ndarray, v_rel: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    R = unit(r_rel)
    H = np.cross(r_rel, v_rel)
    N = unit(H)
    T = unit(np.cross(N, R))
    return R, T, N


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


def safe_float(x: Any, default: float = float("nan")) -> float:
    try:
        if x is None or x == "":
            return default
        return float(x)
    except Exception:
        return default


def build_tb0_grid(args: argparse.Namespace, live_t: float) -> list[float]:
    tb0_base = args.tb0_base_s if args.tb0_base_s is not None else live_t
    offsets = parse_float_list(args.tb0_offsets_s)
    if not offsets:
        offsets = float_range(args.tb0_offset_min_s, args.tb0_offset_max_s, args.tb0_offset_step_s)
    tb0s = []
    for off in offsets:
        tb0 = tb0_base + off
        if tb0 > live_t:
            tb0s.append(float(tb0))
    return sorted(set(tb0s))


def build_dv_grids(args: argparse.Namespace) -> tuple[list[float], list[float], list[float]]:
    dv0_t_grid = parse_float_list(args.dv0_t_grid_m_s)
    if not dv0_t_grid:
        dv0_t_grid = float_range(args.dv0_t_min, args.dv0_t_max, args.dv0_t_step)
    dv0_r_grid = parse_float_list(args.dv0_r_grid_m_s)
    if not dv0_r_grid:
        dv0_r_grid = symmetric_grid(args.dv0_r_max, args.dv0_r_step)
    dv0_n_grid = parse_float_list(args.dv0_n_grid_m_s)
    if not dv0_n_grid:
        dv0_n_grid = symmetric_grid(args.dv0_n_max, args.dv0_n_step)
    return dv0_t_grid, dv0_r_grid, dv0_n_grid


def make_reference(args: argparse.Namespace, row: dict[str, str]) -> dict[str, Any]:
    t_start = float(row["t_start_s"])
    ref_offsets = parse_float_list(args.diagnostic_offsets_s) or [0.0]
    diag_times = [t_start + off for off in ref_offsets]

    start_r = arr(row, "start_x_raw_m", "start_y_raw_m", "start_z_raw_m")
    start_v = arr(row, "start_vx_raw_m_s", "start_vy_raw_m_s", "start_vz_raw_m_s")
    dv_leg = np.zeros(3)
    if "dvx_m_s" in row:
        dv_leg = arr(row, "dvx_m_s", "dvy_m_s", "dvz_m_s")

    if args.reference_mode == "post_correction":
        ref_v0 = start_v + dv_leg
    elif args.reference_mode == "pre_correction":
        ref_v0 = start_v
    else:
        raise SystemExit(f"bad reference_mode: {args.reference_mode}")

    mu_dep = body_mu_m3_s2(args.dep_body)
    refs: list[dict[str, Any]] = []
    with PrincipiaImpulseServerV2(args.server, args.plugin_b64) as srv:
        if not srv.ping():
            raise SystemExit("[FAIL] server PING failed while building reference")
        for t in diag_times:
            if abs(t - t_start) < 1e-9:
                rr = start_r.copy()
                vv = ref_v0.copy()
            else:
                res = srv.propagate_n(
                    req_id=f"ref_{t:.3f}",
                    t0_s=t_start,
                    t1_s=t,
                    r0_m=start_r,
                    v0_m_s=ref_v0,
                    impulses=[],
                )
                if res.status != "ok" or res.final_r_m is None or res.final_v_m_s is None:
                    raise SystemExit(f"[FAIL] reference propagate failed at {t}: {res.status} {res.message}")
                rr = np.array(res.final_r_m, dtype=float)
                vv = np.array(res.final_v_m_s, dtype=float)

            body_r, body_v = body_state_raw(args.dep_body, t, args.center, args.frame)
            r_rel = rr - body_r
            v_rel = vv - body_v
            vinf_vec, eps, ecc, vinf_mag = outgoing_osculating_vinf(r_rel, v_rel, mu_dep)
            refs.append({
                "label": f"tstart+{t - t_start:.0f}s",
                "t_s": float(t),
                "rel_r_raw_m": r_rel.tolist(),
                "rel_v_raw_m_s": v_rel.tolist(),
                "rel_distance_km": norm(r_rel) / 1000.0,
                "rel_speed_m_s": norm(v_rel),
                "eps_m2_s2": eps,
                "eccentricity": ecc,
                "osc_vinf_m_s": vinf_mag,
                "osc_vinf_raw_m_s": None if vinf_vec is None else vinf_vec.tolist(),
            })
    return {
        "reference_mode": args.reference_mode,
        "t_start_s": t_start,
        "diagnostics": refs,
        "start_r_raw_m": start_r.tolist(),
        "start_v_raw_m_s": start_v.tolist(),
        "dv_leg_raw_m_s": dv_leg.tolist(),
        "reference_initial_v_raw_m_s": ref_v0.tolist(),
    }


def local_candidate_score(metrics: dict[str, Any], cfg: dict[str, Any]) -> float:
    vec = metrics["osc_vinf_vec_err_m_s"] / cfg["vinf_vec_scale_m_s"]
    mag = metrics["osc_vinf_mag_err_m_s"] / cfg["vinf_mag_scale_m_s"]
    ang = metrics["osc_vinf_angle_deg"] / cfg["vinf_angle_scale_deg"]
    dv0 = metrics["dv0_norm_m_s"] / cfg["dv0_scale_m_s"]
    nfrac = metrics["dv0_normal_fraction"] / max(cfg["normal_fraction_scale"], 1e-9)
    rfrac = metrics["dv0_radial_fraction"] / max(cfg["radial_fraction_scale"], 1e-9)
    return math.sqrt(
        vec * vec
        + cfg["mag_weight"] * mag * mag
        + cfg["angle_weight"] * ang * ang
        + cfg["dv0_weight"] * dv0 * dv0
        + cfg["normal_weight"] * nfrac * nfrac
        + cfg["radial_weight"] * rfrac * rfrac
    )


def heap_push_top(heap: list[tuple[float, int, dict[str, Any]]], row: dict[str, Any], limit: int, counter: int) -> None:
    score = safe_float(row.get("local_score"), float("inf"))
    if not math.isfinite(score):
        return
    item = (-score, counter, row)
    if len(heap) < limit:
        heapq.heappush(heap, item)
    else:
        if item[0] > heap[0][0]:  # less negative score = better than current worst
            heapq.heapreplace(heap, item)


def heap_push_top_key(
    heap: list[tuple[float, int, dict[str, Any]]],
    row: dict[str, Any],
    limit: int,
    counter: int,
    key: str,
) -> None:
    value = safe_float(row.get(key), float("inf"))
    if not math.isfinite(value):
        return
    item = (-value, counter, row)
    if len(heap) < limit:
        heapq.heappush(heap, item)
    else:
        if item[0] > heap[0][0]:
            heapq.heapreplace(heap, item)


def row_key(row: dict[str, Any]) -> tuple[Any, ...]:
    # Stable enough to merge rows selected by several local criteria.
    return (
        round(safe_float(row.get("tb0_s"), 0.0), 6),
        round(safe_float(row.get("dv0_t_m_s"), 0.0), 6),
        round(safe_float(row.get("dv0_r_m_s"), 0.0), 6),
        round(safe_float(row.get("dv0_n_m_s"), 0.0), 6),
        str(row.get("diag_label", "")),
    )


def union_unique(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in rows:
        k = row_key(row)
        if k not in seen:
            seen[k] = row
        else:
            # Preserve the lower local score if duplicate rows differ only by bookkeeping.
            if safe_float(row.get("local_score"), float("inf")) < safe_float(seen[k].get("local_score"), float("inf")):
                seen[k] = row
    return list(seen.values())


def top_by(rows: list[dict[str, Any]], key: str, n: int) -> list[dict[str, Any]]:
    rows2 = [r for r in rows if math.isfinite(safe_float(r.get(key), float("inf")))]
    return sorted(rows2, key=lambda r: safe_float(r.get(key), float("inf")))[:max(0, n)]


def local_screen_tb0_chunk(payload: tuple[list[float], dict[str, Any]]) -> list[dict[str, Any]]:
    tb0s, cfg = payload
    spice.kclear()
    spice.furnsh(str(cfg["tpc"]))
    spice.furnsh(str(cfg["bsp"]))
    mu_dep = body_mu_m3_s2(cfg["dep_body"])

    r0 = np.array(cfg["live_r"], dtype=float)
    v0 = np.array(cfg["live_v"], dtype=float)
    live_t = float(cfg["live_t"])
    dv_t_grid = cfg["dv0_t_grid"]
    dv_r_grid = cfg["dv0_r_grid"]
    dv_n_grid = cfg["dv0_n_grid"]
    refs = cfg["refs"]
    top_per_worker = int(cfg["local_top_per_worker"])

    # Keep independent Pareto-ish local fronts. A single scalar score was
    # discarding candidates that matched direction but not magnitude, or vice versa.
    heaps: dict[str, list[tuple[float, int, dict[str, Any]]]] = {
        "local_score": [],
        "osc_vinf_vec_err_m_s": [],
        "osc_vinf_mag_err_m_s": [],
        "osc_vinf_angle_deg": [],
    }
    counter = 0

    with PrincipiaImpulseServerV2(cfg["server"], cfg["plugin_b64"]) as srv:
        if not srv.ping():
            raise RuntimeError("server PING failed in local screen worker")

        for tb0 in tb0s:
            pre = srv.propagate_n(
                req_id=f"prelocal_{os.getpid()}_{tb0:.3f}",
                t0_s=live_t,
                t1_s=tb0,
                r0_m=r0,
                v0_m_s=v0,
                impulses=[(tb0, np.zeros(3))],
            )
            if pre.status != "ok" or not pre.burns:
                continue

            pre_r = np.array(pre.burns[0].r_m, dtype=float)
            pre_v = np.array(pre.burns[0].v_before_m_s, dtype=float)
            body_r, body_v = body_state_raw(cfg["dep_body"], tb0, cfg["center"], cfg["frame"])
            rel_r = pre_r - body_r
            rel_v = pre_v - body_v
            try:
                R, T, N = rtn_basis(rel_r, rel_v)
            except Exception:
                continue
            pre_dist_km = norm(rel_r) / 1000.0
            pre_speed = norm(rel_v)

            for dv_t in dv_t_grid:
                base_t = dv_t * T
                for dv_r in dv_r_grid:
                    base_tr = base_t + dv_r * R
                    for dv_n in dv_n_grid:
                        dv0_raw = base_tr + dv_n * N
                        dv0_norm = norm(dv0_raw)
                        v_after = pre_v + dv0_raw
                        v_rel_after = v_after - body_v
                        vinf_vec, eps, ecc, vinf_mag = outgoing_osculating_vinf(rel_r, v_rel_after, mu_dep)
                        if vinf_vec is None:
                            continue

                        nfrac = abs(dv_n) / max(dv0_norm, 1e-9)
                        rfrac = abs(dv_r) / max(dv0_norm, 1e-9)

                        best_row: dict[str, Any] | None = None
                        for ref in refs:
                            ref_vinf = None if ref["osc_vinf_raw_m_s"] is None else np.array(ref["osc_vinf_raw_m_s"], dtype=float)
                            if ref_vinf is None:
                                continue
                            row = {
                                "tb0_s": float(tb0),
                                "tb0_offset_s": float(tb0 - live_t),
                                "dv0_t_m_s": float(dv_t),
                                "dv0_r_m_s": float(dv_r),
                                "dv0_n_m_s": float(dv_n),
                                "dv0_raw_x_m_s": float(dv0_raw[0]),
                                "dv0_raw_y_m_s": float(dv0_raw[1]),
                                "dv0_raw_z_m_s": float(dv0_raw[2]),
                                "dv0_levela_x_m_s": float(raw_to_levela(dv0_raw)[0]),
                                "dv0_levela_y_m_s": float(raw_to_levela(dv0_raw)[1]),
                                "dv0_levela_z_m_s": float(raw_to_levela(dv0_raw)[2]),
                                "dv0_norm_m_s": dv0_norm,
                                "dv0_normal_fraction": nfrac,
                                "dv0_radial_fraction": rfrac,
                                "preburn_distance_from_dep_km": pre_dist_km,
                                "preburn_rel_speed_m_s": pre_speed,
                                "diag_label": ref["label"],
                                "diag_t_s": float(ref["t_s"]),
                                "candidate_eps_m2_s2": eps,
                                "candidate_eccentricity": ecc,
                                "candidate_osc_vinf_m_s": vinf_mag,
                                "reference_osc_vinf_m_s": float(ref["osc_vinf_m_s"]),
                                "osc_vinf_vec_err_m_s": norm(vinf_vec - ref_vinf),
                                "osc_vinf_mag_err_m_s": abs(norm(vinf_vec) - norm(ref_vinf)),
                                "osc_vinf_angle_deg": angle_deg(vinf_vec, ref_vinf),
                            }
                            row["local_score"] = local_candidate_score(row, cfg)
                            if best_row is None or row["local_score"] < best_row["local_score"]:
                                best_row = row
                        if best_row is not None:
                            counter += 1
                            for key, heap in heaps.items():
                                heap_push_top_key(heap, best_row, top_per_worker, counter, key)

    rows: list[dict[str, Any]] = []
    for heap in heaps.values():
        rows.extend(item[2] for item in heap)
    rows = union_unique(rows)
    rows.sort(key=lambda r: safe_float(r.get("local_score"), float("inf")))
    return rows


def validate_candidate_chunk(payload: tuple[list[dict[str, Any]], dict[str, Any]]) -> list[dict[str, Any]]:
    cands, cfg = payload
    spice.kclear()
    spice.furnsh(str(cfg["tpc"]))
    spice.furnsh(str(cfg["bsp"]))
    mu_dep = body_mu_m3_s2(cfg["dep_body"])
    r0 = np.array(cfg["live_r"], dtype=float)
    v0 = np.array(cfg["live_v"], dtype=float)
    live_t = float(cfg["live_t"])
    refs = cfg["refs"]

    out: list[dict[str, Any]] = []
    with PrincipiaImpulseServerV2(cfg["server"], cfg["plugin_b64"]) as srv:
        if not srv.ping():
            raise RuntimeError("server PING failed in validation worker")
        for cand in cands:
            tb0 = float(cand["tb0_s"])
            dv0_raw = np.array([cand["dv0_raw_x_m_s"], cand["dv0_raw_y_m_s"], cand["dv0_raw_z_m_s"]], dtype=float)
            best: dict[str, Any] | None = None
            for ref in refs:
                t_diag = float(ref["t_s"])
                if t_diag <= tb0:
                    continue
                prop = srv.propagate_n(
                    req_id=f"val_{os.getpid()}_{tb0:.3f}_{t_diag:.3f}",
                    t0_s=live_t,
                    t1_s=t_diag,
                    r0_m=r0,
                    v0_m_s=v0,
                    impulses=[(tb0, dv0_raw)],
                )
                if prop.status != "ok" or prop.final_r_m is None or prop.final_v_m_s is None:
                    continue
                cr = np.array(prop.final_r_m, dtype=float)
                cv = np.array(prop.final_v_m_s, dtype=float)
                body_r, body_v = body_state_raw(cfg["dep_body"], t_diag, cfg["center"], cfg["frame"])
                c_rel_r = cr - body_r
                c_rel_v = cv - body_v
                ref_rel_r = np.array(ref["rel_r_raw_m"], dtype=float)
                ref_rel_v = np.array(ref["rel_v_raw_m_s"], dtype=float)
                c_vinf, c_eps, c_ecc, c_vinf_mag = outgoing_osculating_vinf(c_rel_r, c_rel_v, mu_dep)
                ref_vinf = None if ref["osc_vinf_raw_m_s"] is None else np.array(ref["osc_vinf_raw_m_s"], dtype=float)

                if c_vinf is not None and ref_vinf is not None:
                    osc_vec_err = norm(c_vinf - ref_vinf)
                    osc_mag_err = abs(norm(c_vinf) - norm(ref_vinf))
                    osc_ang = angle_deg(c_vinf, ref_vinf)
                else:
                    osc_vec_err = float("inf")
                    osc_mag_err = float("inf")
                    osc_ang = float("inf")

                row = {
                    **cand,
                    "validated_label": ref["label"],
                    "validated_t_s": t_diag,
                    "validated_rel_pos_err_km": norm(c_rel_r - ref_rel_r) / 1000.0,
                    "validated_inst_vrel_vec_err_m_s": norm(c_rel_v - ref_rel_v),
                    "validated_inst_vrel_mag_err_m_s": abs(norm(c_rel_v) - norm(ref_rel_v)),
                    "validated_inst_vrel_angle_deg": angle_deg(c_rel_v, ref_rel_v),
                    "validated_eps_m2_s2": c_eps,
                    "validated_eccentricity": c_ecc,
                    "validated_osc_vinf_m_s": c_vinf_mag,
                    "validated_osc_vinf_vec_err_m_s": osc_vec_err,
                    "validated_osc_vinf_mag_err_m_s": osc_mag_err,
                    "validated_osc_vinf_angle_deg": osc_ang,
                }
                # Reuse the same scoring but with validated fields mapped back.
                score_metrics = {
                    "osc_vinf_vec_err_m_s": osc_vec_err,
                    "osc_vinf_mag_err_m_s": osc_mag_err,
                    "osc_vinf_angle_deg": osc_ang,
                    "dv0_norm_m_s": row["dv0_norm_m_s"],
                    "dv0_normal_fraction": row["dv0_normal_fraction"],
                    "dv0_radial_fraction": row["dv0_radial_fraction"],
                }
                row["validated_score"] = local_candidate_score(score_metrics, cfg)
                if best is None or row["validated_score"] < best["validated_score"]:
                    best = row
            if best is not None:
                out.append(best)
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def chunks(seq: list[Any], n: int) -> list[list[Any]]:
    return [seq[i:i + n] for i in range(0, len(seq), n)]


def main() -> int:
    p = argparse.ArgumentParser(description="Fast Pareto v∞ departure solver: local osculating screen by score/vector/magnitude/angle, then N-body validation of union.")
    p.add_argument("--plugin-b64", required=True)
    p.add_argument("--server", default="principia_impulsive_particle_server")
    p.add_argument("--live-state-json", type=Path, required=True)
    p.add_argument("--leg-optimizations", type=Path, required=True)
    p.add_argument("--leg", type=int, default=1)
    p.add_argument("--output-dir", type=Path, required=True)

    p.add_argument("--bsp", type=Path, required=True)
    p.add_argument("--tpc", type=Path, required=True)
    p.add_argument("--dep-body", default="KERBIN")
    p.add_argument("--center", default="SUN")
    p.add_argument("--frame", default="J2000")
    p.add_argument("--reference-mode", choices=["post_correction", "pre_correction"], default="post_correction")
    p.add_argument("--diagnostic-offsets-s", default="0,3600,7200")

    p.add_argument("--tb0-base-s", type=float, default=None)
    p.add_argument("--tb0-offsets-s", default=None)
    p.add_argument("--tb0-offset-min-s", type=float, default=300.0)
    p.add_argument("--tb0-offset-max-s", type=float, default=19000.0)
    p.add_argument("--tb0-offset-step-s", type=float, default=300.0)

    p.add_argument("--dv0-t-grid-m-s", default=None)
    p.add_argument("--dv0-r-grid-m-s", default=None)
    p.add_argument("--dv0-n-grid-m-s", default=None)
    p.add_argument("--dv0-t-min", type=float, default=800.0)
    p.add_argument("--dv0-t-max", type=float, default=4200.0)
    p.add_argument("--dv0-t-step", type=float, default=100.0)
    p.add_argument("--dv0-r-max", type=float, default=900.0)
    p.add_argument("--dv0-r-step", type=float, default=100.0)
    p.add_argument("--dv0-n-max", type=float, default=900.0)
    p.add_argument("--dv0-n-step", type=float, default=100.0)

    p.add_argument("--vinf-vec-scale-m-s", type=float, default=300.0)
    p.add_argument("--vinf-mag-scale-m-s", type=float, default=300.0)
    p.add_argument("--vinf-angle-scale-deg", type=float, default=5.0)
    p.add_argument("--dv0-scale-m-s", type=float, default=3000.0)
    p.add_argument("--normal-fraction-scale", type=float, default=0.35)
    p.add_argument("--radial-fraction-scale", type=float, default=0.35)
    p.add_argument("--mag-weight", type=float, default=0.25)
    p.add_argument("--angle-weight", type=float, default=1.0)
    p.add_argument("--dv0-weight", type=float, default=0.01)
    p.add_argument("--normal-weight", type=float, default=0.10)
    p.add_argument("--radial-weight", type=float, default=0.05)

    p.add_argument("--local-top-n", type=int, default=2000, help="Top local candidates to keep after the cheap osculating screen.")
    p.add_argument("--validate-top-n", type=int, default=200, help="Legacy alias: top candidates by scalar local score.")
    p.add_argument("--validate-top-n-per-criterion", type=int, default=None, help="Validate this many local candidates from each criterion: score, vector error, magnitude error, angle error. Default: --validate-top-n.")
    p.add_argument("--workers", type=int, default=max(1, min(4, os.cpu_count() or 1)))
    p.add_argument("--tb0-chunk-size", type=int, default=4)
    p.add_argument("--candidate-chunk-size", type=int, default=20)
    p.add_argument("--quiet-stderr", action="store_true")
    p.add_argument("--write-local-csv", action="store_true", help="Write local top CSV in addition to JSON.")
    args = p.parse_args()

    if args.quiet_stderr:
        sys.stderr = open(os.devnull, "w")

    spice.kclear()
    spice.furnsh(str(args.tpc))
    spice.furnsh(str(args.bsp))

    live = json.loads(args.live_state_json.read_text())
    live_t = float(live["ut_s"])
    row = read_leg_row(args.leg_optimizations, args.leg)
    tb0_grid = build_tb0_grid(args, live_t)
    dv0_t_grid, dv0_r_grid, dv0_n_grid = build_dv_grids(args)
    n_full = len(tb0_grid) * len(dv0_t_grid) * len(dv0_r_grid) * len(dv0_n_grid)
    reference = make_reference(args, row)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "reference_vinf.json").write_text(json.dumps(reference, indent=2) + "\n")

    print("=== DEPARTURE VINF SOLVER V3 FAST ===")
    print(f"live_t          : {live_t}")
    print(f"leg t_start/end : {row['t_start_s']} / {row['t_end_s']}")
    print(f"reference_mode  : {args.reference_mode}")
    print("diagnostic times: " + ", ".join(f"{r['t_s']:.3f}" for r in reference["diagnostics"]))
    print(f"tb0 count       : {len(tb0_grid)}")
    print(f"dv grid         : T={len(dv0_t_grid)} R={len(dv0_r_grid)} N={len(dv0_n_grid)} total={n_full}")
    print(f"workers         : {args.workers}")
    print(f"local_top_n     : {args.local_top_n}")
    print(f"validate_top_n  : {args.validate_top_n}")
    print(f"output_dir      : {args.output_dir}")
    print("")
    print("=== REFERENCE VINF ===")
    for r in reference["diagnostics"]:
        print(
            f"{r['label']:<14} dist={r['rel_distance_km']:10.3f} km "
            f"vrel={r['rel_speed_m_s']:9.3f} m/s "
            f"eps={r['eps_m2_s2']:.3e} osc_vinf={r['osc_vinf_m_s']}"
        )
    print("")

    cfg = {
        "plugin_b64": args.plugin_b64,
        "server": args.server,
        "bsp": str(args.bsp),
        "tpc": str(args.tpc),
        "dep_body": args.dep_body,
        "center": args.center,
        "frame": args.frame,
        "live_t": live_t,
        "live_r": live["r_raw_m"],
        "live_v": live["v_raw_m_s"],
        "refs": reference["diagnostics"],
        "dv0_t_grid": dv0_t_grid,
        "dv0_r_grid": dv0_r_grid,
        "dv0_n_grid": dv0_n_grid,
        "vinf_vec_scale_m_s": args.vinf_vec_scale_m_s,
        "vinf_mag_scale_m_s": args.vinf_mag_scale_m_s,
        "vinf_angle_scale_deg": args.vinf_angle_scale_deg,
        "dv0_scale_m_s": args.dv0_scale_m_s,
        "normal_fraction_scale": args.normal_fraction_scale,
        "radial_fraction_scale": args.radial_fraction_scale,
        "mag_weight": args.mag_weight,
        "angle_weight": args.angle_weight,
        "dv0_weight": args.dv0_weight,
        "normal_weight": args.normal_weight,
        "radial_weight": args.radial_weight,
        "local_top_per_worker": max(args.local_top_n, 100),
    }

    local_rows: list[dict[str, Any]] = []
    tb0_work = [(ch, cfg) for ch in chunks(tb0_grid, args.tb0_chunk_size)]
    if args.workers <= 1:
        for item in tb0_work:
            local_rows.extend(local_screen_tb0_chunk(item))
    else:
        with mp.Pool(processes=args.workers) as pool:
            for rows in pool.imap_unordered(local_screen_tb0_chunk, tb0_work):
                local_rows.extend(rows)

    local_rows = [r for r in union_unique(local_rows) if math.isfinite(safe_float(r.get("local_score"), float("inf")))]
    local_rows.sort(key=lambda r: safe_float(r.get("local_score"), float("inf")))
    # Keep a large union, not just the scalar-score top. This file is the local Pareto-ish candidate pool.
    local_top = union_unique(
        top_by(local_rows, "local_score", args.local_top_n)
        + top_by(local_rows, "osc_vinf_vec_err_m_s", args.local_top_n)
        + top_by(local_rows, "osc_vinf_mag_err_m_s", args.local_top_n)
        + top_by(local_rows, "osc_vinf_angle_deg", args.local_top_n)
    )
    local_top.sort(key=lambda r: safe_float(r.get("local_score"), float("inf")))
    (args.output_dir / "burn0_vinf_local_top.json").write_text(json.dumps(local_top, indent=2) + "\n")
    if args.write_local_csv:
        write_csv(args.output_dir / "burn0_vinf_local_top.csv", local_top)

    print("=== TOP LOCAL OSCULATING BURN0 ===")
    for i, r in enumerate(local_top[:20], start=1):
        print(
            f"{i:3d} lscore={safe_float(r.get('local_score')):9.4g} "
            f"{str(r.get('diag_label','')):<12} tb0={safe_float(r.get('tb0_s')):.3f} "
            f"dv0={safe_float(r.get('dv0_norm_m_s')):8.2f} "
            f"vosc={safe_float(r.get('osc_vinf_vec_err_m_s')):9.1f} m/s "
            f"omag={safe_float(r.get('osc_vinf_mag_err_m_s')):8.1f} "
            f"oang={safe_float(r.get('osc_vinf_angle_deg')):7.3f} deg "
            f"T={safe_float(r.get('dv0_t_m_s')):7.1f} R={safe_float(r.get('dv0_r_m_s')):7.1f} N={safe_float(r.get('dv0_n_m_s')):7.1f}"
        )

    validate_n = args.validate_top_n if args.validate_top_n_per_criterion is None else args.validate_top_n_per_criterion
    val_top = union_unique(
        top_by(local_top, "local_score", validate_n)
        + top_by(local_top, "osc_vinf_vec_err_m_s", validate_n)
        + top_by(local_top, "osc_vinf_mag_err_m_s", validate_n)
        + top_by(local_top, "osc_vinf_angle_deg", validate_n)
    )
    # Keep the legacy cap from exploding accidentally if users pass a huge per-criterion value.
    # A value of 0 disables validation; otherwise validate the union of criteria.
    validated_rows: list[dict[str, Any]] = []
    if val_top:
        val_work = [(ch, cfg) for ch in chunks(val_top, args.candidate_chunk_size)]
        if args.workers <= 1:
            for item in val_work:
                validated_rows.extend(validate_candidate_chunk(item))
        else:
            with mp.Pool(processes=args.workers) as pool:
                for rows in pool.imap_unordered(validate_candidate_chunk, val_work):
                    validated_rows.extend(rows)
        validated_rows = [r for r in validated_rows if math.isfinite(safe_float(r.get("validated_score"), float("inf")))]
        validated_rows.sort(key=lambda r: safe_float(r.get("validated_score"), float("inf")))
        (args.output_dir / "burn0_vinf_validated_top.json").write_text(json.dumps(validated_rows, indent=2) + "\n")
        write_csv(args.output_dir / "burn0_vinf_validated_top.csv", validated_rows)

        print("\n=== TOP N-BODY VALIDATED BURN0 ===")
        for i, r in enumerate(validated_rows[:20], start=1):
            print(
                f"{i:3d} vscore={safe_float(r.get('validated_score')):9.4g} "
                f"{str(r.get('validated_label','')):<12} tb0={safe_float(r.get('tb0_s')):.3f} "
                f"dv0={safe_float(r.get('dv0_norm_m_s')):8.2f} "
                f"vosc={safe_float(r.get('validated_osc_vinf_vec_err_m_s')):9.1f} m/s "
                f"omag={safe_float(r.get('validated_osc_vinf_mag_err_m_s')):8.1f} "
                f"oang={safe_float(r.get('validated_osc_vinf_angle_deg')):7.3f} deg "
                f"pos={safe_float(r.get('validated_rel_pos_err_km')):10.1f} km"
            )

    summary = {
        "n_full_grid_logical": n_full,
        "n_tb0": len(tb0_grid),
        "n_local_kept_before_trim": len(local_rows),
        "n_local_top": len(local_top),
        "n_validated": len(validated_rows),
        "reference_mode": args.reference_mode,
        "best_local": local_top[0] if local_top else None,
        "best_validated": validated_rows[0] if validated_rows else None,
        "best_validated_by_vec": top_by(validated_rows, "validated_osc_vinf_vec_err_m_s", 1)[0] if validated_rows else None,
        "best_validated_by_mag": top_by(validated_rows, "validated_osc_vinf_mag_err_m_s", 1)[0] if validated_rows else None,
        "best_validated_by_angle": top_by(validated_rows, "validated_osc_vinf_angle_deg", 1)[0] if validated_rows else None,
        "local_top_json": str(args.output_dir / "burn0_vinf_local_top.json"),
        "validated_top_json": str(args.output_dir / "burn0_vinf_validated_top.json"),
    }
    (args.output_dir / "burn0_vinf_fast_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    print("\n" + json.dumps({
        "best_local_score": None if not local_top else local_top[0].get("local_score"),
        "best_local_osc_vinf_vec_err_m_s": None if not local_top else local_top[0].get("osc_vinf_vec_err_m_s"),
        "best_local_osc_vinf_angle_deg": None if not local_top else local_top[0].get("osc_vinf_angle_deg"),
        "best_validated_score": None if not validated_rows else validated_rows[0].get("validated_score"),
        "best_validated_osc_vinf_vec_err_m_s": None if not validated_rows else validated_rows[0].get("validated_osc_vinf_vec_err_m_s"),
        "best_validated_osc_vinf_angle_deg": None if not validated_rows else validated_rows[0].get("validated_osc_vinf_angle_deg"),
    }, indent=2))
    print(f"[OK] wrote {args.output_dir / 'burn0_vinf_fast_summary.json'}")
    return 0 if local_top else 2


if __name__ == "__main__":
    raise SystemExit(main())
