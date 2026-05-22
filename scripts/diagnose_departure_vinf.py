#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import multiprocessing as mp
import os
import subprocess
from dataclasses import dataclass
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


def angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    na = norm(a)
    nb = norm(b)
    if na <= 0 or nb <= 0:
        return math.nan
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


def parse_csv_floats(s: str | None) -> list[float]:
    if s is None:
        return []
    return [float(x.strip()) for x in str(s).replace(";", ",").split(",") if x.strip()]


def parse_vec3_any(x: Any) -> np.ndarray:
    if isinstance(x, np.ndarray):
        a = x.astype(float)
    elif isinstance(x, (list, tuple)):
        a = np.array([float(v) for v in x], dtype=float)
    else:
        vals = parse_csv_floats(str(x))
        a = np.array(vals, dtype=float)
    if a.shape != (3,):
        raise ValueError(f"expected vec3, got {x!r}")
    return a


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


def two_body_energy(r_rel: np.ndarray, v_rel: np.ndarray, mu: float) -> float:
    return 0.5 * norm(v_rel) ** 2 - mu / norm(r_rel)


def osculating_outgoing_vinf_vector(r_rel: np.ndarray, v_rel: np.ndarray, mu: float) -> np.ndarray | None:
    """Return two-body outgoing asymptotic v-infinity vector for a hyperbolic state.

    This is an osculating approximation around the departure body. It is useful as a
    diagnostic target near Kerbin, not as a replacement for full N-body propagation.
    """
    eps = two_body_energy(r_rel, v_rel, mu)
    if eps <= 0:
        return None
    h = np.cross(r_rel, v_rel)
    hn = norm(h)
    if hn <= 0:
        return None
    h_hat = h / hn
    e_vec = np.cross(v_rel, h) / mu - r_rel / norm(r_rel)
    e = norm(e_vec)
    if e <= 1.0:
        return None
    p_hat = e_vec / e
    q_hat = np.cross(h_hat, p_hat)
    cos_nu_inf = -1.0 / e
    sin_nu_inf = math.sqrt(max(0.0, 1.0 - cos_nu_inf * cos_nu_inf))
    u_out = cos_nu_inf * p_hat + sin_nu_inf * q_hat
    return math.sqrt(2.0 * eps) * unit(u_out)


def rtn_basis(r_rel: np.ndarray, v_rel: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    R = unit(r_rel)
    H = np.cross(r_rel, v_rel)
    N = unit(H)
    T = unit(np.cross(N, R))
    return R, T, N


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

    def propn(self, req_id: str, t0_s: float, t1_s: float, r0_m: np.ndarray, v0_m_s: np.ndarray,
              impulses: list[tuple[float, np.ndarray]]) -> PropResult:
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


def build_reference_state(config: dict[str, Any], row: dict[str, str], t_ref: float) -> tuple[np.ndarray, np.ndarray]:
    t_start = float(row["t_start_s"])
    r_start = row_vec3(row, "start")
    v_start = row_vel3(row, "start")
    dv_leg = leg_dv_raw(row)
    mode = config["reference_mode"]
    if t_ref < t_start - 1e-9:
        raise ValueError("reference diagnostic time must be >= leg t_start_s")
    if mode == "post_correction":
        impulses = [(t_start, dv_leg)]
        v_at_start = v_start + dv_leg
    elif mode == "pre_correction":
        impulses = []
        v_at_start = v_start
    else:
        raise ValueError(mode)
    if abs(t_ref - t_start) < 1e-9:
        return r_start, v_at_start
    with ServerSession(str(config["server"]), str(config["plugin_b64"]), quiet_stderr=bool(config.get("quiet_stderr", False))) as srv:
        if not srv.ping():
            raise RuntimeError("server ping failed")
        res = srv.propn("ref", t_start, t_ref, r_start, v_start, impulses)
        if res.status != "ok" or res.final_r_m is None or res.final_v_m_s is None:
            raise RuntimeError(f"reference propagation failed: {res.status} {res.message}")
        return res.final_r_m, res.final_v_m_s


def load_candidates(path: Path, top_n: int = 0) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".json":
        obj = json.loads(path.read_text())
        if isinstance(obj, list):
            rows = obj
        elif isinstance(obj, dict):
            if isinstance(obj.get("best"), dict):
                rows = [obj["best"]]
            elif isinstance(obj.get("events"), list):
                rows = obj["events"]
            else:
                # Common helper: allow {"candidates": [...]} or similar.
                for key in ["candidates", "rows", "burn0_top", "top"]:
                    if isinstance(obj.get(key), list):
                        rows = obj[key]
                        break
                else:
                    rows = [obj]
        else:
            raise ValueError(f"unsupported JSON candidate schema in {path}")
    else:
        with path.open(newline="") as f:
            rows = list(csv.DictReader(f))
    # Prefer rows that contain tb0 and a burn0 vector.
    rows = [dict(r) for r in rows if "tb0_s" in r or "initial_time" in r]
    def score_key(r: dict[str, Any]) -> float:
        for k in ["burn0_score", "score", "final_pos_err_km"]:
            if k in r:
                return safe_float(r.get(k), math.inf)
        return math.inf
    rows.sort(key=score_key)
    if top_n and top_n > 0:
        rows = rows[:top_n]
    return rows


def candidate_tb0(row: dict[str, Any]) -> float:
    if "tb0_s" in row:
        return float(row["tb0_s"])
    if "initial_time" in row:
        return float(row["initial_time"])
    raise KeyError("candidate has no tb0_s or initial_time")


def candidate_dv0_raw(row: dict[str, Any], srv: ServerSession, config: dict[str, Any], live_r: np.ndarray, live_v: np.ndarray, live_t: float, tb0: float) -> np.ndarray:
    for key in ["dv0_raw_m_s", "delta_v_raw_m_s"]:
        if key in row and row[key] not in (None, ""):
            return parse_vec3_any(row[key])
    if all(k in row and row[k] not in (None, "") for k in ["dv0_t_m_s", "dv0_r_m_s", "dv0_n_m_s"]):
        pre = srv.propn("preburn_for_rtn", live_t, tb0, live_r, live_v, [(tb0, np.zeros(3))])
        if pre.status != "ok" or not pre.burns:
            raise RuntimeError(f"preburn failed: {pre.status} {pre.message}")
        burn_r = pre.burns[0].r_m
        burn_v_before = pre.burns[0].v_before_m_s
        dep_r, dep_v = body_state_raw(config["dep_body"], tb0, config["center"], config["frame"])
        R, T, N = rtn_basis(burn_r - dep_r, burn_v_before - dep_v)
        return float(row["dv0_t_m_s"]) * T + float(row["dv0_r_m_s"]) * R + float(row["dv0_n_m_s"]) * N
    raise KeyError("candidate has no dv0_raw_m_s/delta_v_raw_m_s and no RTN components")


def compare_state(prefix: str, r: np.ndarray, v: np.ndarray, ref_r: np.ndarray, ref_v: np.ndarray, body_r: np.ndarray, body_v: np.ndarray, mu: float) -> dict[str, Any]:
    rel_r = r - body_r
    rel_v = v - body_v
    ref_rel_r = ref_r - body_r
    ref_rel_v = ref_v - body_v
    eps = two_body_energy(rel_r, rel_v, mu)
    ref_eps = two_body_energy(ref_rel_r, ref_rel_v, mu)
    vinf_inst_err = rel_v - ref_rel_v
    ref_vhat = unit(ref_rel_v) if norm(ref_rel_v) > 0 else np.zeros(3)
    inst_parallel_err = float(np.dot(vinf_inst_err, ref_vhat)) if norm(ref_vhat) > 0 else math.nan
    inst_cross_err = math.sqrt(max(0.0, norm(vinf_inst_err) ** 2 - inst_parallel_err ** 2)) if math.isfinite(inst_parallel_err) else math.nan

    vinf_osc = osculating_outgoing_vinf_vector(rel_r, rel_v, mu)
    ref_vinf_osc = osculating_outgoing_vinf_vector(ref_rel_r, ref_rel_v, mu)
    out: dict[str, Any] = {
        f"{prefix}_distance_from_body_km": norm(rel_r) / 1000.0,
        f"{prefix}_ref_distance_from_body_km": norm(ref_rel_r) / 1000.0,
        f"{prefix}_rel_pos_err_km": norm(rel_r - ref_rel_r) / 1000.0,
        f"{prefix}_rel_pos_angle_deg": angle_deg(rel_r, ref_rel_r),
        f"{prefix}_inst_vrel_m_s": norm(rel_v),
        f"{prefix}_ref_inst_vrel_m_s": norm(ref_rel_v),
        f"{prefix}_inst_vrel_mag_err_m_s": norm(rel_v) - norm(ref_rel_v),
        f"{prefix}_inst_vrel_vec_err_m_s": norm(vinf_inst_err),
        f"{prefix}_inst_vrel_parallel_err_m_s": inst_parallel_err,
        f"{prefix}_inst_vrel_cross_err_m_s": inst_cross_err,
        f"{prefix}_inst_vrel_angle_deg": angle_deg(rel_v, ref_rel_v),
        f"{prefix}_energy_m2_s2": eps,
        f"{prefix}_ref_energy_m2_s2": ref_eps,
        f"{prefix}_c3_m2_s2": 2.0 * eps,
        f"{prefix}_ref_c3_m2_s2": 2.0 * ref_eps,
    }
    if vinf_osc is not None:
        out[f"{prefix}_osc_vinf_m_s"] = norm(vinf_osc)
        out[f"{prefix}_osc_vinf_raw_m_s"] = vinf_osc.tolist()
    else:
        out[f"{prefix}_osc_vinf_m_s"] = math.nan
        out[f"{prefix}_osc_vinf_raw_m_s"] = ""
    if ref_vinf_osc is not None:
        out[f"{prefix}_ref_osc_vinf_m_s"] = norm(ref_vinf_osc)
        out[f"{prefix}_ref_osc_vinf_raw_m_s"] = ref_vinf_osc.tolist()
    else:
        out[f"{prefix}_ref_osc_vinf_m_s"] = math.nan
        out[f"{prefix}_ref_osc_vinf_raw_m_s"] = ""
    if vinf_osc is not None and ref_vinf_osc is not None:
        osc_err = vinf_osc - ref_vinf_osc
        ref_ohat = unit(ref_vinf_osc)
        osc_parallel = float(np.dot(osc_err, ref_ohat))
        osc_cross = math.sqrt(max(0.0, norm(osc_err) ** 2 - osc_parallel ** 2))
        out.update({
            f"{prefix}_osc_vinf_mag_err_m_s": norm(vinf_osc) - norm(ref_vinf_osc),
            f"{prefix}_osc_vinf_vec_err_m_s": norm(osc_err),
            f"{prefix}_osc_vinf_parallel_err_m_s": osc_parallel,
            f"{prefix}_osc_vinf_cross_err_m_s": osc_cross,
            f"{prefix}_osc_vinf_angle_deg": angle_deg(vinf_osc, ref_vinf_osc),
        })
    else:
        out.update({
            f"{prefix}_osc_vinf_mag_err_m_s": math.nan,
            f"{prefix}_osc_vinf_vec_err_m_s": math.nan,
            f"{prefix}_osc_vinf_parallel_err_m_s": math.nan,
            f"{prefix}_osc_vinf_cross_err_m_s": math.nan,
            f"{prefix}_osc_vinf_angle_deg": math.nan,
        })
    return out


def diagnose_chunk(payload: tuple[dict[str, Any], list[dict[str, Any]]]) -> list[dict[str, Any]]:
    config, rows = payload
    load_common(config)
    live_r = np.array(config["live_r_raw_m"], dtype=float)
    live_v = np.array(config["live_v_raw_m_s"], dtype=float)
    live_t = float(config["live_t_s"])
    mu = float(config["mu_dep"])
    ref_states = []
    for rs in config["reference_states"]:
        ref_states.append({
            "label": rs["label"],
            "t_s": float(rs["t_s"]),
            "r_m": np.array(rs["r_m"], dtype=float),
            "v_m_s": np.array(rs["v_m_s"], dtype=float),
        })
    out: list[dict[str, Any]] = []
    with ServerSession(str(config["server"]), str(config["plugin_b64"]), quiet_stderr=bool(config.get("quiet_stderr", False))) as srv:
        if not srv.ping():
            raise RuntimeError("server ping failed")
        for i, row in enumerate(rows):
            try:
                tb0 = candidate_tb0(row)
                if tb0 < live_t:
                    raise RuntimeError(f"tb0 before live_t: {tb0} < {live_t}")
                dv0 = candidate_dv0_raw(row, srv, config, live_r, live_v, live_t, tb0)
                max_t = max(rs["t_s"] for rs in ref_states)
                if max_t <= tb0:
                    raise RuntimeError(f"all diagnostic times <= tb0={tb0}")
                res = srv.propn(f"diag_{i}", live_t, max_t, live_r, live_v, [(tb0, dv0)])
                if res.status != "ok" or res.final_r_m is None or res.final_v_m_s is None or not res.burns:
                    raise RuntimeError(f"prop failed: {res.status} {res.message}")
                burn = res.burns[0]
                dep_r_burn, dep_v_burn = body_state_raw(config["dep_body"], tb0, config["center"], config["frame"])
                burn_rel_r = burn.r_m - dep_r_burn
                burn_rel_v_before = burn.v_before_m_s - dep_v_burn
                burn_rel_v_after = burn.v_after_m_s - dep_v_burn
                burn_eps = two_body_energy(burn_rel_r, burn_rel_v_after, mu)
                base = {
                    "candidate_index": i,
                    "status": "ok",
                    "tb0_s": tb0,
                    "dv0_raw_m_s": dv0.tolist(),
                    "dv0_levela_m_s": raw_to_levela(dv0).tolist(),
                    "dv0_norm_m_s": norm(dv0),
                    "source_burn0_score": safe_float(row.get("burn0_score"), math.nan),
                    "source_score": safe_float(row.get("score"), math.nan),
                    "source_burn0_match_pos_err_km": safe_float(row.get("burn0_match_pos_err_km"), math.nan),
                    "source_burn0_match_vel_err_m_s": safe_float(row.get("burn0_match_vel_err_m_s"), math.nan),
                    "burn0_distance_from_body_km": norm(burn_rel_r) / 1000.0,
                    "burn0_rel_speed_before_m_s": norm(burn_rel_v_before),
                    "burn0_rel_speed_after_m_s": norm(burn_rel_v_after),
                    "burn0_energy_after_m2_s2": burn_eps,
                    "burn0_c3_after_m2_s2": 2.0 * burn_eps,
                    "burn0_osculating_vinf_after_m_s": math.sqrt(2.0 * burn_eps) if burn_eps > 0 else math.nan,
                }
                # Propagate separately to each diagnostic time unless it is the max time already.
                for rs in ref_states:
                    t_diag = float(rs["t_s"])
                    if t_diag <= tb0:
                        rr = dict(base)
                        rr.update({"diag_label": rs["label"], "diag_t_s": t_diag, "status": "skipped", "message": "diag time <= tb0"})
                        out.append(rr)
                        continue
                    if abs(t_diag - max_t) < 1e-9:
                        ship_r, ship_v = res.final_r_m, res.final_v_m_s
                    else:
                        rd = srv.propn(f"diag_{i}_{rs['label']}", live_t, t_diag, live_r, live_v, [(tb0, dv0)])
                        if rd.status != "ok" or rd.final_r_m is None or rd.final_v_m_s is None:
                            raise RuntimeError(f"diag prop failed: {rd.status} {rd.message}")
                        ship_r, ship_v = rd.final_r_m, rd.final_v_m_s
                    body_r, body_v = body_state_raw(config["dep_body"], t_diag, config["center"], config["frame"])
                    metrics = compare_state("diag", ship_r, ship_v, rs["r_m"], rs["v_m_s"], body_r, body_v, mu)
                    rr = dict(base)
                    rr.update({
                        "diag_label": rs["label"],
                        "diag_t_s": t_diag,
                        "diag_offset_from_leg_start_s": t_diag - float(config["leg_t_start_s"]),
                    })
                    rr.update(metrics)
                    # Scores: one for instantaneous relative velocity, one for osculating asymptote.
                    rr["vinf_inst_score"] = (
                        safe_float(metrics.get("diag_inst_vrel_vec_err_m_s"), math.inf) / float(config["inst_v_score_scale_m_s"])
                        + safe_float(metrics.get("diag_rel_pos_err_km"), math.inf) / float(config["pos_score_scale_km"])
                    )
                    rr["vinf_osc_score"] = (
                        safe_float(metrics.get("diag_osc_vinf_vec_err_m_s"), math.inf) / float(config["osc_v_score_scale_m_s"])
                        + safe_float(metrics.get("diag_rel_pos_err_km"), math.inf) / float(config["pos_score_scale_km"])
                    )
                    out.append(rr)
            except Exception as e:
                out.append({
                    "candidate_index": i,
                    "status": "error",
                    "message": str(e),
                    "raw_candidate": json.dumps(row, default=str)[:1000],
                })
    return out


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
                    try:
                        rr[k] = ",".join(f"{float(x):.17g}" for x in v)
                    except Exception:
                        rr[k] = json.dumps(jsonable(v))
                else:
                    rr[k] = v
            w.writerow(rr)


def main() -> int:
    ap = argparse.ArgumentParser(description="Diagnose whether departure burn0 matches the leg's Kerbin-relative v-infinity corridor.")
    ap.add_argument("--plugin-b64", required=True)
    ap.add_argument("--server", default="principia_impulsive_particle_server")
    ap.add_argument("--live-state-json", type=Path, required=True)
    ap.add_argument("--leg-optimizations", type=Path, required=True)
    ap.add_argument("--leg", type=int, default=1)
    ap.add_argument("--candidate-file", type=Path, required=True, help="burn0_top.json, burn0_scan.csv, correction_top.json, or result.json")
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--bsp", type=Path, required=True)
    ap.add_argument("--tpc", type=Path, required=True)
    ap.add_argument("--dep-body", default="KERBIN")
    ap.add_argument("--center", default="SUN")
    ap.add_argument("--frame", default="J2000")
    ap.add_argument("--reference-mode", choices=["post_correction", "pre_correction"], default="post_correction")
    ap.add_argument("--diagnostic-offsets-s", default="0", help="Comma offsets from leg t_start_s. Example: 0,3600,7200")
    ap.add_argument("--diagnostic-times-s", default=None, help="Optional absolute diagnostic times; overrides offsets if provided.")
    ap.add_argument("--top-n", type=int, default=50)
    ap.add_argument("--workers", type=int, default=max(1, min(4, os.cpu_count() or 1)))
    ap.add_argument("--quiet-stderr", action="store_true")
    ap.add_argument("--pos-score-scale-km", type=float, default=100000.0)
    ap.add_argument("--inst-v-score-scale-m-s", type=float, default=1000.0)
    ap.add_argument("--osc-v-score-scale-m-s", type=float, default=1000.0)
    args = ap.parse_args()

    spice.kclear()
    spice.furnsh(str(args.tpc))
    spice.furnsh(str(args.bsp))

    live = json.loads(args.live_state_json.read_text())
    row = read_leg_row(args.leg_optimizations, args.leg)
    live_t = float(live["ut_s"])
    live_r = np.array(live["r_raw_m"], dtype=float)
    live_v = np.array(live["v_raw_m_s"], dtype=float)
    t_start = float(row["t_start_s"])
    mu_dep = body_mu_m3_s2(args.dep_body)

    if args.diagnostic_times_s:
        diag_times = parse_csv_floats(args.diagnostic_times_s)
    else:
        diag_offsets = parse_csv_floats(args.diagnostic_offsets_s)
        diag_times = [t_start + x for x in diag_offsets]
    if not diag_times:
        raise SystemExit("[FAIL] no diagnostic times")

    config: dict[str, Any] = {
        "server": args.server,
        "plugin_b64": args.plugin_b64,
        "bsp": str(args.bsp),
        "tpc": str(args.tpc),
        "dep_body": args.dep_body,
        "center": args.center,
        "frame": args.frame,
        "reference_mode": args.reference_mode,
        "mu_dep": mu_dep,
        "live_t_s": live_t,
        "live_r_raw_m": live_r.tolist(),
        "live_v_raw_m_s": live_v.tolist(),
        "leg_t_start_s": t_start,
        "quiet_stderr": args.quiet_stderr,
        "pos_score_scale_km": args.pos_score_scale_km,
        "inst_v_score_scale_m_s": args.inst_v_score_scale_m_s,
        "osc_v_score_scale_m_s": args.osc_v_score_scale_m_s,
    }

    # Build reference states once in parent to keep the output self-contained.
    ref_states = []
    for t in diag_times:
        r_ref, v_ref = build_reference_state(config, row, t)
        body_r, body_v = body_state_raw(args.dep_body, t, args.center, args.frame)
        rel_r = r_ref - body_r
        rel_v = v_ref - body_v
        eps = two_body_energy(rel_r, rel_v, mu_dep)
        osc = osculating_outgoing_vinf_vector(rel_r, rel_v, mu_dep)
        label = f"tstart{t - t_start:+.0f}s"
        ref_states.append({
            "label": label,
            "t_s": float(t),
            "r_m": r_ref.tolist(),
            "v_m_s": v_ref.tolist(),
            "distance_from_body_km": norm(rel_r) / 1000.0,
            "inst_vrel_m_s": norm(rel_v),
            "energy_m2_s2": eps,
            "c3_m2_s2": 2.0 * eps,
            "osc_vinf_m_s": None if osc is None else norm(osc),
            "osc_vinf_raw_m_s": None if osc is None else osc.tolist(),
        })
    config["reference_states"] = ref_states

    candidates = load_candidates(args.candidate_file, args.top_n)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "reference_vinf.json").write_text(json.dumps(jsonable({
        "reference_mode": args.reference_mode,
        "leg_t_start_s": t_start,
        "dep_body": args.dep_body,
        "states": ref_states,
    }), indent=2) + "\n")

    print("=== DEPARTURE VINF DIAGNOSTIC ===")
    print(f"candidate_file   : {args.candidate_file}")
    print(f"candidates       : {len(candidates)}")
    print(f"reference_mode   : {args.reference_mode}")
    print(f"diagnostic_times : {', '.join(f'{t:.3f}' for t in diag_times)}")
    print(f"workers          : {args.workers}")
    print(f"output_dir       : {args.output_dir}")
    print("")
    for rs in ref_states:
        print(
            f"REF {rs['label']:<14} "
            f"dist={rs['distance_from_body_km']:11.3f} km "
            f"vrel={rs['inst_vrel_m_s']:9.3f} m/s "
            f"eps={rs['energy_m2_s2']:.3e} "
            f"osc_vinf={rs['osc_vinf_m_s']}"
        )
    print("")

    payloads = [(config, ch) for ch in chunks(candidates, args.workers)]
    if args.workers > 1 and len(payloads) > 1:
        with mp.Pool(processes=args.workers) as pool:
            rows = flatten(pool.map(diagnose_chunk, payloads))
    else:
        rows = flatten(diagnose_chunk(p) for p in payloads)

    ok_rows = [r for r in rows if r.get("status") == "ok"]
    by_inst = sorted(ok_rows, key=lambda r: safe_float(r.get("vinf_inst_score"), math.inf))
    by_osc = sorted(ok_rows, key=lambda r: safe_float(r.get("vinf_osc_score"), math.inf))
    by_pos = sorted(ok_rows, key=lambda r: safe_float(r.get("diag_rel_pos_err_km"), math.inf))

    write_csv(args.output_dir / "departure_vinf_diagnostic.csv", rows)
    (args.output_dir / "departure_vinf_top_inst.json").write_text(json.dumps(jsonable(by_inst[:50]), indent=2) + "\n")
    (args.output_dir / "departure_vinf_top_osc.json").write_text(json.dumps(jsonable(by_osc[:50]), indent=2) + "\n")
    (args.output_dir / "departure_vinf_top_pos.json").write_text(json.dumps(jsonable(by_pos[:50]), indent=2) + "\n")

    print("=== TOP BY INSTANTANEOUS RELATIVE VELOCITY ===")
    for i, r in enumerate(by_inst[:10], start=1):
        print(
            f"{i:3d} {r.get('diag_label',''):<14} "
            f"score={safe_float(r.get('vinf_inst_score')):8.3f} "
            f"tb0={safe_float(r.get('tb0_s')):.3f} "
            f"dv0={safe_float(r.get('dv0_norm_m_s')):8.2f} "
            f"pos={safe_float(r.get('diag_rel_pos_err_km')):10.1f} km "
            f"vvec={safe_float(r.get('diag_inst_vrel_vec_err_m_s')):9.1f} m/s "
            f"vmag={safe_float(r.get('diag_inst_vrel_mag_err_m_s')):8.1f} "
            f"vang={safe_float(r.get('diag_inst_vrel_angle_deg')):7.3f} deg"
        )
    print("")

    print("=== TOP BY OSCULATING OUTGOING VINF ===")
    for i, r in enumerate(by_osc[:10], start=1):
        print(
            f"{i:3d} {r.get('diag_label',''):<14} "
            f"score={safe_float(r.get('vinf_osc_score')):8.3f} "
            f"tb0={safe_float(r.get('tb0_s')):.3f} "
            f"dv0={safe_float(r.get('dv0_norm_m_s')):8.2f} "
            f"pos={safe_float(r.get('diag_rel_pos_err_km')):10.1f} km "
            f"vosc={safe_float(r.get('diag_osc_vinf_vec_err_m_s')):9.1f} m/s "
            f"omag={safe_float(r.get('diag_osc_vinf_mag_err_m_s')):8.1f} "
            f"oang={safe_float(r.get('diag_osc_vinf_angle_deg')):7.3f} deg"
        )
    print("")

    summary = {
        "n_rows": len(rows),
        "n_ok": len(ok_rows),
        "best_inst": by_inst[0] if by_inst else None,
        "best_osc": by_osc[0] if by_osc else None,
        "best_pos": by_pos[0] if by_pos else None,
        "reference": ref_states,
    }
    (args.output_dir / "departure_vinf_summary.json").write_text(json.dumps(jsonable(summary), indent=2) + "\n")
    print(f"[OK] wrote {args.output_dir / 'departure_vinf_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
