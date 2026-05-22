#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import multiprocessing as mp
import os
import subprocess
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import spiceypy as spice


def norm(v: np.ndarray) -> float:
    return float(np.linalg.norm(v))


def unit(v: np.ndarray) -> np.ndarray:
    n = norm(v)
    if n <= 0:
        raise ValueError("zero vector")
    return v / n


def levela_to_raw(v: np.ndarray) -> np.ndarray:
    # LevelA/SPICE -> Principia raw = [+Z, -X, +Y]
    x, y, z = v
    return np.array([z, -x, y], dtype=float)


def raw_to_levela(v: np.ndarray) -> np.ndarray:
    x, y, z = v
    return np.array([-y, z, x], dtype=float)


def rtn_basis(r_rel: np.ndarray, v_rel: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    R = unit(r_rel)
    H = np.cross(r_rel, v_rel)
    N = unit(H)
    T = unit(np.cross(N, R))
    return R, T, N


def parse_csv_floats(s: str) -> list[float]:
    return [float(x.strip()) for x in (s or "").replace(";", ",").split(",") if x.strip()]


def parse_range_or_csv(spec: str | None, lo: float, hi: float, step: float) -> list[float]:
    if spec:
        vals = parse_csv_floats(spec)
        if not vals:
            raise ValueError(f"empty grid spec {spec!r}")
        return vals
    if step <= 0:
        raise ValueError("grid step must be > 0")
    vals = []
    x = lo
    # inclusive with small epsilon.
    while x <= hi + 1e-9:
        vals.append(float(x))
        x += step
    return vals


def safe_float(x: Any, default: float = math.nan) -> float:
    try:
        if x is None or x == "":
            return default
        return float(x)
    except Exception:
        return default


def read_leg_row(path: Path, leg: int) -> dict[str, str]:
    rows = list(csv.DictReader(path.open()))
    for r in rows:
        if int(float(r.get("leg", "nan"))) == leg:
            return r
    raise SystemExit(f"[FAIL] leg {leg} not found in {path}")


def arr(row: dict[str, str], *names: str) -> np.ndarray:
    missing = [n for n in names if n not in row or row[n] == ""]
    if missing:
        raise KeyError(f"missing columns: {missing}")
    return np.array([float(row[n]) for n in names], dtype=float)


def row_vec3(row: dict[str, str], prefix: str) -> np.ndarray:
    return arr(row, f"{prefix}_x_raw_m", f"{prefix}_y_raw_m", f"{prefix}_z_raw_m")


def row_vel3(row: dict[str, str], prefix: str) -> np.ndarray:
    return arr(row, f"{prefix}_vx_raw_m_s", f"{prefix}_vy_raw_m_s", f"{prefix}_vz_raw_m_s")


def leg_dv_raw(row: dict[str, str]) -> np.ndarray:
    if all(k in row and row[k] != "" for k in ["dvx_m_s", "dvy_m_s", "dvz_m_s"]):
        return arr(row, "dvx_m_s", "dvy_m_s", "dvz_m_s")
    return np.zeros(3)


def body_state_raw(body: str, t_s: float, center: str, frame: str) -> tuple[np.ndarray, np.ndarray]:
    st, _ = spice.spkezr(body, t_s, frame, "NONE", center)
    r_levela = np.array([1000.0 * st[0], 1000.0 * st[1], 1000.0 * st[2]], dtype=float)
    v_levela = np.array([1000.0 * st[3], 1000.0 * st[4], 1000.0 * st[5]], dtype=float)
    return levela_to_raw(r_levela), levela_to_raw(v_levela)


def body_mu_m3_s2(body: str) -> float:
    _, vals = spice.bodvrd(body, "GM", 1)
    return float(vals[0]) * 1.0e9


@dataclass
class BurnSnapshot:
    burn_t_s: float
    r_m: np.ndarray
    v_before_m_s: np.ndarray
    v_after_m_s: np.ndarray


@dataclass
class PropResult:
    status: str
    message: str
    id: str
    t0_s: float | None = None
    t1_s: float | None = None
    burns: list[BurnSnapshot] | None = None
    final_r_m: np.ndarray | None = None
    final_v_m_s: np.ndarray | None = None


class ServerSession:
    """Tiny standalone client for principia_impulsive_particle_server PROPN."""

    def __init__(self, server: str, plugin_b64: str, quiet_stderr: bool = False):
        stderr = subprocess.DEVNULL if quiet_stderr else None
        self.proc = subprocess.Popen(
            [server, plugin_b64],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=stderr,
            text=True,
            bufsize=1,
        )
        assert self.proc.stdout is not None
        ready = self.proc.stdout.readline().strip()
        if not ready.startswith("READY"):
            raise RuntimeError(f"server did not become ready: {ready!r}")
        self.ready_line = ready

    def close(self) -> None:
        if self.proc.poll() is None:
            try:
                self._write("QUIT")
                if self.proc.stdout is not None:
                    self.proc.stdout.readline()
            except Exception:
                pass
            try:
                self.proc.terminate()
            except Exception:
                pass

    def __enter__(self) -> "ServerSession":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _write(self, line: str) -> None:
        if self.proc.stdin is None:
            raise RuntimeError("server stdin closed")
        self.proc.stdin.write(line + "\n")
        self.proc.stdin.flush()

    def ping(self) -> bool:
        self._write("PING")
        assert self.proc.stdout is not None
        return self.proc.stdout.readline().strip() == "PONG"

    def propn(
        self,
        req_id: str,
        t0_s: float,
        t1_s: float,
        r0_m: np.ndarray,
        v0_m_s: np.ndarray,
        impulses: list[tuple[float, np.ndarray]],
    ) -> PropResult:
        fields: list[str] = [
            "PROPN", req_id,
            f"{float(t0_s):.17g}", f"{float(t1_s):.17g}", str(len(impulses)),
            *[f"{float(x):.17g}" for x in r0_m],
            *[f"{float(x):.17g}" for x in v0_m_s],
        ]
        for burn_t, dv in impulses:
            fields.append(f"{float(burn_t):.17g}")
            fields.extend(f"{float(x):.17g}" for x in dv)
        self._write("\t".join(fields))
        assert self.proc.stdout is not None
        line = self.proc.stdout.readline().strip()
        if not line:
            return PropResult(status="error", message="empty server response", id=req_id)
        parts = line.split("\t")
        if parts[0] == "ERR":
            return PropResult(status="error", message=parts[2] if len(parts) > 2 else "", id=parts[1] if len(parts) > 1 else req_id)
        if parts[0] != "OKN":
            return PropResult(status="error", message=f"unexpected response {parts[0]}: {line}", id=req_id)
        try:
            rid = parts[1]
            out_t0 = float(parts[2])
            out_t1 = float(parts[3])
            n = int(parts[4])
            idx = 5
            burns: list[BurnSnapshot] = []
            for _ in range(n):
                burn_t = float(parts[idx]); idx += 1
                r = np.array([float(parts[idx]), float(parts[idx + 1]), float(parts[idx + 2])], dtype=float); idx += 3
                vb = np.array([float(parts[idx]), float(parts[idx + 1]), float(parts[idx + 2])], dtype=float); idx += 3
                va = np.array([float(parts[idx]), float(parts[idx + 1]), float(parts[idx + 2])], dtype=float); idx += 3
                burns.append(BurnSnapshot(burn_t, r, vb, va))
            fr = np.array([float(parts[idx]), float(parts[idx + 1]), float(parts[idx + 2])], dtype=float); idx += 3
            fv = np.array([float(parts[idx]), float(parts[idx + 1]), float(parts[idx + 2])], dtype=float)
            return PropResult("ok", "", rid, out_t0, out_t1, burns, fr, fv)
        except Exception as e:
            return PropResult(status="error", message=f"parse OKN failed: {e}: {line}", id=req_id)


def load_common(config: dict[str, Any]) -> None:
    spice.kclear()
    spice.furnsh(str(config["tpc"]))
    spice.furnsh(str(config["bsp"]))


def compute_preburn_and_dv0(
    srv: ServerSession,
    config: dict[str, Any],
    live_r: np.ndarray,
    live_v: np.ndarray,
    live_t: float,
    tb0: float,
    dv0_t: float,
    dv0_r: float,
    dv0_n: float,
) -> tuple[np.ndarray, dict[str, float]]:
    zero = np.zeros(3)
    pre = srv.propn("preburn", live_t, tb0, live_r, live_v, [(tb0, zero)])
    if pre.status != "ok" or not pre.burns:
        raise RuntimeError(f"preburn failed: {pre.status} {pre.message}")
    burn_r = pre.burns[0].r_m
    burn_v_before = pre.burns[0].v_before_m_s
    dep_r, dep_v = body_state_raw(config["dep_body"], tb0, config["center"], config["frame"])
    rel_r = burn_r - dep_r
    rel_v = burn_v_before - dep_v
    R, T, N = rtn_basis(rel_r, rel_v)
    dv0_raw = dv0_t * T + dv0_r * R + dv0_n * N
    v_after = burn_v_before + dv0_raw
    mu = float(config["mu_dep"])
    eps = 0.5 * norm(v_after - dep_v) ** 2 - mu / norm(rel_r)
    diag = {
        "preburn_distance_km": norm(rel_r) / 1000.0,
        "preburn_rel_speed_m_s": norm(rel_v),
        "escape_energy_m2_s2": eps,
        "dv0_norm_m_s": norm(dv0_raw),
        "dv0_t_m_s": float(dv0_t),
        "dv0_r_m_s": float(dv0_r),
        "dv0_n_m_s": float(dv0_n),
        "dv0_radial_fraction": abs(dv0_r) / max(norm(dv0_raw), 1e-9),
        "dv0_normal_fraction": abs(dv0_n) / max(norm(dv0_raw), 1e-9),
    }
    return dv0_raw, diag


def residual_vec(pos: np.ndarray, vel: np.ndarray, ref_r: np.ndarray, ref_v: np.ndarray, pos_scale_m: float, vel_scale_m_s: float) -> np.ndarray:
    return np.concatenate([(pos - ref_r) / pos_scale_m, (vel - ref_v) / vel_scale_m_s])


def evaluate_burn0_chunk(payload: tuple[dict[str, Any], list[dict[str, float]]]) -> list[dict[str, Any]]:
    config, candidates = payload
    load_common(config)
    live_r = np.array(config["live_r_raw_m"], dtype=float)
    live_v = np.array(config["live_v_raw_m_s"], dtype=float)
    live_t = float(config["live_t_s"])
    match_t = float(config["match_time_s"])
    ref_r = np.array(config["match_ref_r_m"], dtype=float)
    ref_v = np.array(config["match_ref_v_m_s"], dtype=float)
    pos_scale_m = float(config["burn0_pos_scale_km"]) * 1000.0
    vel_scale = float(config["burn0_vel_scale_m_s"])
    rows: list[dict[str, Any]] = []
    with ServerSession(str(config["server"]), str(config["plugin_b64"]), quiet_stderr=bool(config.get("quiet_stderr", False))) as srv:
        if not srv.ping():
            raise RuntimeError("server ping failed")
        for c in candidates:
            tb0 = float(c["tb0_s"])
            try:
                dv0_raw, diag = compute_preburn_and_dv0(
                    srv, config, live_r, live_v, live_t, tb0,
                    float(c["dv0_t_m_s"]), float(c["dv0_r_m_s"]), float(c["dv0_n_m_s"]),
                )
                if diag["preburn_distance_km"] > float(config["burn0_max_kerbin_distance_km"]):
                    raise RuntimeError(f"preburn too far: {diag['preburn_distance_km']:.3f} km")
                res = srv.propn(
                    f"b0_{len(rows)}", live_t, match_t, live_r, live_v,
                    [(tb0, dv0_raw)],
                )
                if res.status != "ok" or res.final_r_m is None or res.final_v_m_s is None:
                    raise RuntimeError(f"prop failed: {res.status} {res.message}")
                pos_err_km = norm(res.final_r_m - ref_r) / 1000.0
                vel_err_m_s = norm(res.final_v_m_s - ref_v)
                score_vec = residual_vec(res.final_r_m, res.final_v_m_s, ref_r, ref_v, pos_scale_m, vel_scale)
                score = float(np.linalg.norm(score_vec))
                if diag["escape_energy_m2_s2"] <= 0:
                    score += 100.0 + abs(diag["escape_energy_m2_s2"]) / 1e6
                score += 0.05 * diag["dv0_norm_m_s"] / 3000.0
                score += 0.5 * abs(float(c["dv0_r_m_s"])) / max(abs(float(c["dv0_t_m_s"])), 1.0)
                score += 0.5 * abs(float(c["dv0_n_m_s"])) / max(abs(float(c["dv0_t_m_s"])), 1.0)
                rows.append({
                    "status": "ok",
                    "tb0_s": tb0,
                    "dv0_raw_m_s": dv0_raw.tolist(),
                    "dv0_levela_m_s": raw_to_levela(dv0_raw).tolist(),
                    "burn0_match_pos_err_km": pos_err_km,
                    "burn0_match_vel_err_m_s": vel_err_m_s,
                    "burn0_score": score,
                    **diag,
                })
            except Exception as e:
                rows.append({
                    "status": "error",
                    "tb0_s": tb0,
                    "dv0_t_m_s": c.get("dv0_t_m_s"),
                    "dv0_r_m_s": c.get("dv0_r_m_s"),
                    "dv0_n_m_s": c.get("dv0_n_m_s"),
                    "message": str(e),
                    "burn0_score": math.inf,
                })
    return rows


def solve_correction_chunk(payload: tuple[dict[str, Any], list[dict[str, Any]]]) -> list[dict[str, Any]]:
    config, burn0_candidates = payload
    load_common(config)
    live_r = np.array(config["live_r_raw_m"], dtype=float)
    live_v = np.array(config["live_v_raw_m_s"], dtype=float)
    live_t = float(config["live_t_s"])
    match_t = float(config["match_time_s"])
    final_t = float(config["final_time_s"])
    ref_r = np.array(config["match_ref_r_m"], dtype=float)
    ref_v = np.array(config["match_ref_v_m_s"], dtype=float)
    target_r = np.array(config["target_r_m"], dtype=float)
    target_v = np.array(config["target_v_m_s"], dtype=float)
    match_pos_scale_m = float(config["corr_match_pos_scale_km"]) * 1000.0
    match_vel_scale = float(config["corr_match_vel_scale_m_s"])
    final_pos_scale_m = float(config["final_pos_scale_km"]) * 1000.0
    final_vel_scale = float(config["final_vel_scale_m_s"])
    fd_step = float(config["fd_dv1_step_m_s"])
    dv1_hard = float(config["dv1_hard_max_m_s"])
    tb1_offsets = list(config["tb1_offsets_before_match_s"])
    rows: list[dict[str, Any]] = []
    with ServerSession(str(config["server"]), str(config["plugin_b64"]), quiet_stderr=bool(config.get("quiet_stderr", False))) as srv:
        if not srv.ping():
            raise RuntimeError("server ping failed")
        for c in burn0_candidates:
            if c.get("status") != "ok":
                continue
            tb0 = float(c["tb0_s"])
            dv0_raw = np.array(c["dv0_raw_m_s"], dtype=float)
            for offset in tb1_offsets:
                tb1 = match_t - float(offset)
                if not (tb0 < tb1 <= match_t):
                    continue
                try:
                    def eval_at_match(dv1: np.ndarray) -> tuple[np.ndarray, PropResult]:
                        res = srv.propn(
                            "corr_match", live_t, match_t, live_r, live_v,
                            [(tb0, dv0_raw), (tb1, dv1)],
                        )
                        if res.status != "ok" or res.final_r_m is None or res.final_v_m_s is None:
                            raise RuntimeError(f"match prop failed: {res.status} {res.message}")
                        return residual_vec(res.final_r_m, res.final_v_m_s, ref_r, ref_v, match_pos_scale_m, match_vel_scale), res

                    r0_vec, base_match = eval_at_match(np.zeros(3))
                    J = np.zeros((len(r0_vec), 3), dtype=float)
                    for j in range(3):
                        dv = np.zeros(3)
                        dv[j] = fd_step
                        rj, _ = eval_at_match(dv)
                        J[:, j] = (rj - r0_vec) / fd_step
                    dv1, *_ = np.linalg.lstsq(J, -r0_vec, rcond=None)
                    dv1_norm = norm(dv1)
                    clipped = False
                    if dv1_norm > dv1_hard:
                        dv1 = dv1 / dv1_norm * dv1_hard
                        dv1_norm = dv1_hard
                        clipped = True
                    r_after, match_after = eval_at_match(dv1)
                    match_pos_err_km = norm(match_after.final_r_m - ref_r) / 1000.0
                    match_vel_err_m_s = norm(match_after.final_v_m_s - ref_v)
                    final = srv.propn(
                        "corr_final", live_t, final_t, live_r, live_v,
                        [(tb0, dv0_raw), (tb1, dv1)],
                    )
                    if final.status != "ok" or final.final_r_m is None or final.final_v_m_s is None:
                        raise RuntimeError(f"final prop failed: {final.status} {final.message}")
                    final_pos_err_km = norm(final.final_r_m - target_r) / 1000.0
                    final_vel_err_m_s = norm(final.final_v_m_s - target_v)
                    score_vec_final = residual_vec(final.final_r_m, final.final_v_m_s, target_r, target_v, final_pos_scale_m, final_vel_scale)
                    score = float(np.linalg.norm(score_vec_final))
                    score += 0.15 * dv1_norm / max(float(config["dv1_soft_max_m_s"]), 1.0)
                    score += 0.05 * float(c.get("burn0_score", 0.0))
                    if clipped:
                        score += 10.0
                    rows.append({
                        "status": "ok",
                        "tb0_s": tb0,
                        "tb1_s": tb1,
                        "tb1_offset_before_match_s": float(offset),
                        "dv0_raw_m_s": dv0_raw.tolist(),
                        "dv0_levela_m_s": raw_to_levela(dv0_raw).tolist(),
                        "dv1_raw_m_s": dv1.tolist(),
                        "dv1_levela_m_s": raw_to_levela(dv1).tolist(),
                        "dv0_norm_m_s": norm(dv0_raw),
                        "dv1_norm_m_s": dv1_norm,
                        "dv1_clipped": clipped,
                        "match_pos_err_km": match_pos_err_km,
                        "match_vel_err_m_s": match_vel_err_m_s,
                        "final_pos_err_km": final_pos_err_km,
                        "final_vel_err_m_s": final_vel_err_m_s,
                        "score": score,
                        "burn0_score": float(c.get("burn0_score", math.inf)),
                        "burn0_match_pos_err_km": float(c.get("burn0_match_pos_err_km", math.nan)),
                        "burn0_match_vel_err_m_s": float(c.get("burn0_match_vel_err_m_s", math.nan)),
                        "escape_energy_m2_s2": float(c.get("escape_energy_m2_s2", math.nan)),
                    })
                except Exception as e:
                    rows.append({
                        "status": "error",
                        "tb0_s": tb0,
                        "tb1_s": tb1,
                        "message": str(e),
                        "score": math.inf,
                    })
    return rows


def chunks(seq: list[Any], n_chunks: int) -> list[list[Any]]:
    if n_chunks <= 1 or len(seq) == 0:
        return [seq]
    out = [[] for _ in range(min(n_chunks, len(seq)))]
    for i, item in enumerate(seq):
        out[i % len(out)].append(item)
    return out


def flatten(xs: Iterable[list[Any]]) -> list[Any]:
    out: list[Any] = []
    for x in xs:
        out.extend(x)
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    if not fields:
        fields = ["empty"]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            rr = {}
            for k in fields:
                v = r.get(k, "")
                if isinstance(v, (list, tuple, np.ndarray)):
                    rr[k] = ",".join(f"{float(x):.17g}" for x in v)
                else:
                    rr[k] = v
            w.writerow(rr)


def jsonable(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.float64, np.float32)):
        return float(obj)
    if isinstance(obj, (np.int64, np.int32)):
        return int(obj)
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [jsonable(v) for v in obj]
    return obj


def make_grid(args: argparse.Namespace, live_t: float) -> list[dict[str, float]]:
    tb0_offsets = parse_range_or_csv(args.tb0_offsets_s, args.tb0_offset_min_s, args.tb0_offset_max_s, args.tb0_offset_step_s)
    dv0_t_vals = parse_range_or_csv(args.dv0_t_grid_m_s, args.dv0_t_min, args.dv0_t_max, args.dv0_t_step)
    dv0_r_vals = parse_range_or_csv(args.dv0_r_grid_m_s, -args.dv0_r_max, args.dv0_r_max, args.dv0_r_step)
    dv0_n_vals = parse_range_or_csv(args.dv0_n_grid_m_s, -args.dv0_n_max, args.dv0_n_max, args.dv0_n_step)
    out: list[dict[str, float]] = []
    for off in tb0_offsets:
        tb0 = args.tb0_base_s + off if args.tb0_base_s is not None else live_t + off
        for t in dv0_t_vals:
            for r in dv0_r_vals:
                for n in dv0_n_vals:
                    out.append({"tb0_s": float(tb0), "dv0_t_m_s": float(t), "dv0_r_m_s": float(r), "dv0_n_m_s": float(n)})
    return out


def build_reference_state(args: argparse.Namespace, row: dict[str, str], live: dict[str, Any]) -> dict[str, Any]:
    t_start = float(row["t_start_s"])
    t_end = float(row["t_end_s"])
    match_t = args.match_time_s if args.match_time_s is not None else t_start + args.match_offset_s
    if match_t < t_start:
        raise SystemExit("[FAIL] match time must be >= leg t_start_s for reference propagation")

    r_start = row_vec3(row, "start")
    v_start = row_vel3(row, "start")
    dv_leg = leg_dv_raw(row)
    target_r = row_vec3(row, "target")
    target_v = row_vel3(row, "target")
    if args.reference_mode == "post_correction":
        ref_impulses = [(t_start, dv_leg)]
        ref_v0_for_match_at_start = v_start + dv_leg
    elif args.reference_mode == "pre_correction":
        ref_impulses = []
        ref_v0_for_match_at_start = v_start
    else:
        raise ValueError(args.reference_mode)

    with ServerSession(args.server, args.plugin_b64, quiet_stderr=args.quiet_stderr) as srv:
        if not srv.ping():
            raise SystemExit("[FAIL] server PING failed")
        if abs(match_t - t_start) < 1e-9:
            match_r = r_start
            match_v = ref_v0_for_match_at_start
        else:
            res = srv.propn("reference_match", t_start, match_t, r_start, v_start, ref_impulses)
            if res.status != "ok" or res.final_r_m is None or res.final_v_m_s is None:
                raise SystemExit(f"[FAIL] reference propagation failed: {res.status} {res.message}")
            match_r = res.final_r_m
            match_v = res.final_v_m_s

    return {
        "leg_t_start_s": t_start,
        "leg_t_end_s": t_end,
        "match_time_s": match_t,
        "start_r_m": r_start,
        "start_v_m_s": v_start,
        "leg_dv_raw_m_s": dv_leg,
        "match_ref_r_m": match_r,
        "match_ref_v_m_s": match_v,
        "target_r_m": target_r,
        "target_v_m_s": target_v,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="V2 departure corridor solver: burn0 first, dynamic correction second.")
    ap.add_argument("--plugin-b64", required=True)
    ap.add_argument("--server", default="principia_impulsive_particle_server")
    ap.add_argument("--live-state-json", type=Path, required=True)
    ap.add_argument("--leg-optimizations", type=Path, required=True)
    ap.add_argument("--leg", type=int, default=1)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--bsp", type=Path, required=True)
    ap.add_argument("--tpc", type=Path, required=True)
    ap.add_argument("--dep-body", default="KERBIN")
    ap.add_argument("--arr-body", default="EVE")
    ap.add_argument("--center", default="SUN")
    ap.add_argument("--frame", default="J2000")

    ap.add_argument("--reference-mode", choices=["post_correction", "pre_correction"], default="post_correction")
    ap.add_argument("--match-time-s", type=float, default=None)
    ap.add_argument("--match-offset-s", type=float, default=0.0, help="Default match time is leg t_start_s + this offset.")

    ap.add_argument("--tb0-base-s", type=float, default=None, help="If set, tb0 = base + offset. Else tb0 = live_t + offset.")
    ap.add_argument("--tb0-offsets-s", default=None, help="Comma-separated tb0 offsets. Overrides min/max/step.")
    ap.add_argument("--tb0-offset-min-s", type=float, default=300.0)
    ap.add_argument("--tb0-offset-max-s", type=float, default=3600.0)
    ap.add_argument("--tb0-offset-step-s", type=float, default=300.0)

    ap.add_argument("--dv0-t-grid-m-s", default=None)
    ap.add_argument("--dv0-t-min", type=float, default=1450.0)
    ap.add_argument("--dv0-t-max", type=float, default=3200.0)
    ap.add_argument("--dv0-t-step", type=float, default=100.0)
    ap.add_argument("--dv0-r-grid-m-s", default=None)
    ap.add_argument("--dv0-r-max", type=float, default=300.0)
    ap.add_argument("--dv0-r-step", type=float, default=100.0)
    ap.add_argument("--dv0-n-grid-m-s", default=None)
    ap.add_argument("--dv0-n-max", type=float, default=300.0)
    ap.add_argument("--dv0-n-step", type=float, default=100.0)

    ap.add_argument("--burn0-top-n", type=int, default=40)
    ap.add_argument("--burn0-pos-scale-km", type=float, default=100000.0)
    ap.add_argument("--burn0-vel-scale-m-s", type=float, default=800.0)
    ap.add_argument("--burn0-max-kerbin-distance-km", type=float, default=10000.0)

    ap.add_argument("--tb1-offsets-before-match-s", default="7200,14400,21600", help="Comma list: correction time = match_time - offset.")
    ap.add_argument("--corr-match-pos-scale-km", type=float, default=50000.0)
    ap.add_argument("--corr-match-vel-scale-m-s", type=float, default=500.0)
    ap.add_argument("--fd-dv1-step-m-s", type=float, default=10.0)
    ap.add_argument("--dv1-soft-max-m-s", type=float, default=800.0)
    ap.add_argument("--dv1-hard-max-m-s", type=float, default=1600.0)

    ap.add_argument("--final-pos-scale-km", type=float, default=100000.0)
    ap.add_argument("--final-vel-scale-m-s", type=float, default=1000.0)
    ap.add_argument("--accept-final-pos-km", type=float, default=100000.0)
    ap.add_argument("--accept-final-vel-m-s", type=float, default=1000.0)
    ap.add_argument("--accept-dv1-m-s", type=float, default=800.0)

    ap.add_argument("--workers", type=int, default=max(1, min(4, os.cpu_count() or 1)))
    ap.add_argument("--quiet-stderr", action="store_true")
    ap.add_argument("--max-burn0-candidates", type=int, default=0, help="Debug cap after grid generation; 0 means no cap.")
    args = ap.parse_args()

    spice.kclear()
    spice.furnsh(str(args.tpc))
    spice.furnsh(str(args.bsp))

    live = json.loads(args.live_state_json.read_text())
    row = read_leg_row(args.leg_optimizations, args.leg)
    live_t = float(live["ut_s"])
    live_r = np.array(live["r_raw_m"], dtype=float)
    live_v = np.array(live["v_raw_m_s"], dtype=float)

    ref = build_reference_state(args, row, live)
    mu_dep = body_mu_m3_s2(args.dep_body)

    config: dict[str, Any] = {
        "server": args.server,
        "plugin_b64": args.plugin_b64,
        "bsp": str(args.bsp),
        "tpc": str(args.tpc),
        "dep_body": args.dep_body,
        "arr_body": args.arr_body,
        "center": args.center,
        "frame": args.frame,
        "mu_dep": mu_dep,
        "live_t_s": live_t,
        "live_r_raw_m": live_r.tolist(),
        "live_v_raw_m_s": live_v.tolist(),
        "match_time_s": float(ref["match_time_s"]),
        "match_ref_r_m": ref["match_ref_r_m"].tolist(),
        "match_ref_v_m_s": ref["match_ref_v_m_s"].tolist(),
        "final_time_s": float(ref["leg_t_end_s"]),
        "target_r_m": ref["target_r_m"].tolist(),
        "target_v_m_s": ref["target_v_m_s"].tolist(),
        "burn0_pos_scale_km": args.burn0_pos_scale_km,
        "burn0_vel_scale_m_s": args.burn0_vel_scale_m_s,
        "burn0_max_kerbin_distance_km": args.burn0_max_kerbin_distance_km,
        "tb1_offsets_before_match_s": parse_csv_floats(args.tb1_offsets_before_match_s),
        "corr_match_pos_scale_km": args.corr_match_pos_scale_km,
        "corr_match_vel_scale_m_s": args.corr_match_vel_scale_m_s,
        "fd_dv1_step_m_s": args.fd_dv1_step_m_s,
        "dv1_soft_max_m_s": args.dv1_soft_max_m_s,
        "dv1_hard_max_m_s": args.dv1_hard_max_m_s,
        "final_pos_scale_km": args.final_pos_scale_km,
        "final_vel_scale_m_s": args.final_vel_scale_m_s,
        "quiet_stderr": args.quiet_stderr,
    }

    candidates = make_grid(args, live_t)
    if args.max_burn0_candidates and args.max_burn0_candidates > 0:
        candidates = candidates[: args.max_burn0_candidates]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "reference.json").write_text(json.dumps(jsonable({**ref, "reference_mode": args.reference_mode}), indent=2) + "\n")

    print("=== DEPARTURE CORRIDOR SOLVER V2 ===")
    print(f"live_t          : {live_t}")
    print(f"leg t_start/end : {ref['leg_t_start_s']} / {ref['leg_t_end_s']}")
    print(f"match_time      : {ref['match_time_s']}")
    print(f"burn0 grid      : {len(candidates)} candidates")
    print(f"workers         : {args.workers}")
    print(f"output_dir      : {args.output_dir}")
    print("")

    burn0_payloads = [(config, ch) for ch in chunks(candidates, args.workers)]
    if args.workers > 1 and len(burn0_payloads) > 1:
        with mp.Pool(processes=args.workers) as pool:
            burn0_rows = flatten(pool.map(evaluate_burn0_chunk, burn0_payloads))
    else:
        burn0_rows = flatten(evaluate_burn0_chunk(p) for p in burn0_payloads)

    burn0_rows_sorted = sorted(burn0_rows, key=lambda r: safe_float(r.get("burn0_score"), math.inf))
    write_csv(args.output_dir / "burn0_scan.csv", burn0_rows_sorted)
    burn0_good = [r for r in burn0_rows_sorted if r.get("status") == "ok"][: args.burn0_top_n]
    (args.output_dir / "burn0_top.json").write_text(json.dumps(jsonable(burn0_good), indent=2) + "\n")

    print("=== TOP BURN0 CANDIDATES ===")
    for i, r in enumerate(burn0_good[:10], start=1):
        print(
            f"{i:3d} score={safe_float(r.get('burn0_score')):10.4g} "
            f"tb0={safe_float(r.get('tb0_s')):.3f} "
            f"dv0={safe_float(r.get('dv0_norm_m_s')):8.2f} "
            f"match_pos={safe_float(r.get('burn0_match_pos_err_km')):12.3f} km "
            f"match_vel={safe_float(r.get('burn0_match_vel_err_m_s')):9.3f} m/s "
            f"eps={safe_float(r.get('escape_energy_m2_s2')):.3e}"
        )
    print("")

    corr_payloads = [(config, ch) for ch in chunks(burn0_good, args.workers)]
    if args.workers > 1 and len(corr_payloads) > 1:
        with mp.Pool(processes=args.workers) as pool:
            corr_rows = flatten(pool.map(solve_correction_chunk, corr_payloads))
    else:
        corr_rows = flatten(solve_correction_chunk(p) for p in corr_payloads)

    corr_rows_sorted = sorted(corr_rows, key=lambda r: safe_float(r.get("score"), math.inf))
    write_csv(args.output_dir / "correction_scan.csv", corr_rows_sorted)
    (args.output_dir / "correction_top.json").write_text(json.dumps(jsonable(corr_rows_sorted[:50]), indent=2) + "\n")

    print("=== TOP CORRECTION CANDIDATES ===")
    for i, r in enumerate([x for x in corr_rows_sorted if x.get("status") == "ok"][:10], start=1):
        print(
            f"{i:3d} score={safe_float(r.get('score')):10.4g} "
            f"tb0={safe_float(r.get('tb0_s')):.3f} tb1={safe_float(r.get('tb1_s')):.3f} "
            f"dv0={safe_float(r.get('dv0_norm_m_s')):8.2f} dv1={safe_float(r.get('dv1_norm_m_s')):8.2f} "
            f"match={safe_float(r.get('match_pos_err_km')):11.3f} km "
            f"final={safe_float(r.get('final_pos_err_km')):12.3f} km "
            f"fv={safe_float(r.get('final_vel_err_m_s')):9.3f}"
        )

    best = next((r for r in corr_rows_sorted if r.get("status") == "ok"), None)
    solution_valid = False
    invalid_reasons: list[str] = []
    if best is None:
        invalid_reasons.append("no_correction_candidate")
    else:
        if safe_float(best.get("final_pos_err_km"), math.inf) > args.accept_final_pos_km:
            invalid_reasons.append("final_pos_err_too_large")
        if safe_float(best.get("final_vel_err_m_s"), math.inf) > args.accept_final_vel_m_s:
            invalid_reasons.append("final_vel_err_too_large")
        if safe_float(best.get("dv1_norm_m_s"), math.inf) > args.accept_dv1_m_s:
            invalid_reasons.append("dv1_too_large")
        if best.get("dv1_clipped"):
            invalid_reasons.append("dv1_clipped")
        if safe_float(best.get("escape_energy_m2_s2"), -math.inf) <= 0:
            invalid_reasons.append("burn0_not_escape")
        solution_valid = len(invalid_reasons) == 0

    out = {
        "propagation_ok": best is not None,
        "solution_valid": solution_valid,
        "invalid_reasons": invalid_reasons,
        "best": best,
        "reference": jsonable(ref),
        "grid_counts": {
            "burn0_candidates": len(candidates),
            "burn0_ok": len([r for r in burn0_rows if r.get("status") == "ok"]),
            "correction_candidates": len(corr_rows),
            "correction_ok": len([r for r in corr_rows if r.get("status") == "ok"]),
        },
    }
    (args.output_dir / "result.json").write_text(json.dumps(jsonable(out), indent=2) + "\n")

    if best is not None:
        mission_event_preview = {
            "note": "Preview only. Use your existing exporter/pusher after validating solution_valid=true.",
            "events": [
                {
                    "event_key": "departure_burn0",
                    "initial_time": best["tb0_s"],
                    "delta_v_raw_m_s": best["dv0_raw_m_s"],
                    "delta_v_levela_m_s": best["dv0_levela_m_s"],
                    "dv_norm_m_s": best["dv0_norm_m_s"],
                },
                {
                    "event_key": "departure_cleanup_dv1",
                    "initial_time": best["tb1_s"],
                    "delta_v_raw_m_s": best["dv1_raw_m_s"],
                    "delta_v_levela_m_s": best["dv1_levela_m_s"],
                    "dv_norm_m_s": best["dv1_norm_m_s"],
                },
            ],
        }
        (args.output_dir / "mission_events_preview.json").write_text(json.dumps(jsonable(mission_event_preview), indent=2) + "\n")

    print("")
    print(json.dumps({
        "solution_valid": solution_valid,
        "invalid_reasons": invalid_reasons,
        "best_final_pos_err_km": None if best is None else best.get("final_pos_err_km"),
        "best_final_vel_err_m_s": None if best is None else best.get("final_vel_err_m_s"),
        "best_dv0_m_s": None if best is None else best.get("dv0_norm_m_s"),
        "best_dv1_m_s": None if best is None else best.get("dv1_norm_m_s"),
    }, indent=2))
    print(f"[OK] wrote {args.output_dir / 'result.json'}")
    return 0 if solution_valid else 2


if __name__ == "__main__":
    raise SystemExit(main())
