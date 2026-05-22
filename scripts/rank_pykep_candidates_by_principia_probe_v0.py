#!/usr/bin/env python3
"""
rank_pykep_candidates_by_principia_probe_v2_mp_dedupe.py

Rerank PyKEP/snapshot-executable departure candidates using the real Principia
VBATCH_NAV2 backend.

Purpose
-------
The ordinary PyKEP family search does not know the current parking orbit. The
snapshot-executability rank adds a first-burn TNB decomposition, but a candidate
may still require a large plane/binormal correction once we try to target an
actual flyby in Principia.

This script probes the top candidates with VBATCH_NAV2, optionally performs one
finite-difference least-squares correction toward a target CA radius, and reranks
by operational cost:
  - corrected CA residual,
  - required TNB correction,
  - corrected out-of-plane cost sqrt(N^2+B^2),
  - soft-limit penalties,
  - edge/start/end penalties.

It does not try to compute the full B-plane for the next leg. It is a fast
"is this seed operationally sane in the real Principia backend?" filter.

Requires a server patched with VBATCH_NAV2.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

DAY_S = 86400.0


def norm(v: Sequence[float]) -> float:
    return float(np.linalg.norm(np.asarray(v, dtype=float)))


def unit(v: Sequence[float], fallback: Sequence[float] | None = None) -> np.ndarray:
    a = np.asarray(v, dtype=float)
    n = norm(a)
    if n <= 0 or not math.isfinite(n):
        if fallback is None:
            raise ValueError(f"cannot normalize {v!r}")
        return unit(fallback)
    return a / n


def clamp_norm(v: np.ndarray, max_norm: float) -> np.ndarray:
    n = norm(v)
    if max_norm <= 0 or n <= max_norm or n == 0:
        return v
    return v * (max_norm / n)


def load_json(path: Path) -> Any:
    with path.open() as f:
        return json.load(f)


def as_seq_string(seq: Any) -> str:
    if isinstance(seq, str):
        return seq
    if isinstance(seq, list):
        return " ".join(str(x) for x in seq)
    return str(seq)


def sequence_bodies(c: dict[str, Any]) -> list[str]:
    seq = c.get("sequence") or c.get("sequence_bodies") or c.get("source_sequence") or ""
    if isinstance(seq, list):
        return [str(x).upper() for x in seq if str(x).strip()]
    return [x.strip().upper() for x in str(seq).replace("-", " ").split() if x.strip()]


def candidate_leg1_dep_arr(c: dict[str, Any], args: argparse.Namespace) -> tuple[str, str, str]:
    bodies = sequence_bodies(c)

    dep_arg = str(args.dep_body).upper()
    arr_arg = str(args.arr_body).upper()
    nav_arg = str(args.nav_body).upper()

    dep = dep_arg
    if dep_arg in ("AUTO", "LEG1", ""):
        dep = str(c.get("dep_body") or c.get("leg1_dep") or (bodies[0] if len(bodies) >= 1 else "KERBIN")).upper()

    arr = arr_arg
    if arr_arg in ("AUTO", "LEG1", ""):
        arr = str(c.get("arr_body") or c.get("leg1_arr") or (bodies[1] if len(bodies) >= 2 else "EVE")).upper()

    nav = nav_arg
    if nav_arg in ("AUTO", "DEP", "LEG1", ""):
        nav = dep

    return dep, arr, nav


def find_candidates(data: Any) -> list[dict[str, Any]]:
    # Prefer top/candidates arrays when present; otherwise walk recursively.
    if isinstance(data, dict):
        for key in ("top", "candidates", "rows"):
            if isinstance(data.get(key), list):
                rows = [r for r in data[key] if isinstance(r, dict)]
                if rows:
                    return rows
    rows: list[dict[str, Any]] = []
    seen: set[int] = set()

    def walk(x: Any) -> None:
        if isinstance(x, dict):
            if ("row_index0" in x and
                ("burn_abs_s" in x or "burn_dt_s" in x) and
                ("dv_tangent_m_s" in x or "dvt_m_s" in x)):
                ident = id(x)
                if ident not in seen:
                    rows.append(x)
                    seen.add(ident)
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)

    walk(data)
    return rows


def get_float(d: dict[str, Any], *keys: str, default: float | None = None) -> float:
    for k in keys:
        if k in d and d[k] is not None:
            return float(d[k])
    if default is not None:
        return float(default)
    raise KeyError(keys)


def get_vec(d: dict[str, Any], key: str) -> np.ndarray:
    if key not in d:
        raise KeyError(key)
    v = d[key]
    if len(v) != 3:
        raise ValueError(f"{key} is not vec3: {v!r}")
    return np.asarray([float(x) for x in v], dtype=float)


def patch_candidate_tnb(candidate: dict[str, Any], tnb: np.ndarray) -> dict[str, Any]:
    c = json.loads(json.dumps(candidate))
    T, N, B = map(float, tnb)
    for k in ("dv_tangent_m_s", "dvt_m_s"):
        if k in c:
            c[k] = T
    for k in ("dv_normal_m_s", "dvn_m_s"):
        if k in c:
            c[k] = N
    for k in ("dv_binormal_m_s", "dvb_m_s"):
        if k in c:
            c[k] = B
    c["dv_norm_m_s"] = float(norm(tnb))
    return c


@dataclass
class VBatchCase:
    case_id: str
    dep_body: str
    arr_body: str
    nav_body: str
    state_abs_s: float
    scan_start_dt_s: float
    scan_end_dt_s: float
    samples: int
    rel_r_raw_m: np.ndarray
    rel_v_raw_m_s: np.ndarray
    tnb_m_s: np.ndarray

    def tokens(self) -> list[str]:
        T, N, B = map(float, self.tnb_m_s)
        return [
            self.case_id,
            self.dep_body,
            self.arr_body,
            self.nav_body,
            f"{self.state_abs_s:.17g}",
            f"{self.scan_start_dt_s:.17g}",
            f"{self.scan_end_dt_s:.17g}",
            str(int(self.samples)),
            *(f"{x:.17g}" for x in self.rel_r_raw_m),
            *(f"{x:.17g}" for x in self.rel_v_raw_m_s),
            "1",  # n_impulses; v0 supports one departure impulse only.
            "0",  # impulse dt_s relative to state_abs_s.
            f"{T:.17g}",
            f"{N:.17g}",
            f"{B:.17g}",
        ]


@dataclass
class VBatchResult:
    case_id: str
    ok: bool
    ca_distance_m: float
    ca_dt_s: float
    ca_t_game_s: float
    ca_speed_m_s: float
    ca_radial_velocity_m_s: float
    edge: str
    status: str
    ca_rel_r_raw_m: np.ndarray
    ca_rel_v_raw_m_s: np.ndarray

    @property
    def ca_distance_km(self) -> float:
        return self.ca_distance_m / 1000.0

    def serializable(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "ok": self.ok,
            "ca_distance_m": self.ca_distance_m,
            "ca_distance_km": self.ca_distance_km,
            "ca_dt_s": self.ca_dt_s,
            "ca_t_game_s": self.ca_t_game_s,
            "ca_speed_m_s": self.ca_speed_m_s,
            "ca_radial_velocity_m_s": self.ca_radial_velocity_m_s,
            "edge": self.edge,
            "status": self.status,
            "ca_rel_r_raw_m": [float(x) for x in self.ca_rel_r_raw_m],
            "ca_rel_v_raw_m_s": [float(x) for x in self.ca_rel_v_raw_m_s],
        }


class PrincipiaClient:
    def __init__(self, server: Path, plugin_b64: Path, plugin_on_argv: bool, timeout_s: float = 900.0, quiet_stderr: bool = False):
        self.timeout_s = timeout_s
        stderr = subprocess.DEVNULL if quiet_stderr else None
        argv = [str(server)]
        if plugin_on_argv:
            argv.append(str(plugin_b64))
        self.proc = subprocess.Popen(
            argv,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=stderr,
            text=True,
            bufsize=1,
        )
        if self.proc.stdin is None or self.proc.stdout is None:
            raise RuntimeError("failed to open server pipes")
        if not plugin_on_argv:
            out = self.command("\t".join(["LOADPLUGIN", "p0", str(plugin_b64)]))
            if not out.startswith("OKPLUGIN"):
                raise RuntimeError(f"LOADPLUGIN failed: {out}")

    def close(self) -> None:
        try:
            if self.proc.stdin:
                self.proc.stdin.write("QUIT\n")
                self.proc.stdin.flush()
        except Exception:
            pass
        try:
            self.proc.terminate()
        except Exception:
            pass

    def _read_protocol_line(self) -> str:
        assert self.proc.stdout is not None
        start = time.time()
        while True:
            if self.proc.poll() is not None:
                raise RuntimeError(f"server exited with code {self.proc.returncode}")
            line = self.proc.stdout.readline()
            if line == "":
                if time.time() - start > self.timeout_s:
                    raise TimeoutError("timeout waiting for server response")
                continue
            line = line.strip()
            if not line:
                continue
            # Ignore banners/version lines. Accept only protocol replies.
            if (line.startswith("OK") or line.startswith("ERR") or
                line.startswith("PONG")):
                return line
            # Example ignored lines: READY, principia_impulsive_particle_server_...

    def command(self, line: str) -> str:
        assert self.proc.stdin is not None
        self.proc.stdin.write(line.rstrip("\n") + "\n")
        self.proc.stdin.flush()
        return self._read_protocol_line()

    def vbatch_nav2(self, batch_id: str, cases: Sequence[VBatchCase]) -> list[VBatchResult]:
        toks = ["VBATCH_NAV2", batch_id, str(len(cases))]
        for c in cases:
            toks.extend(c.tokens())
        out = self.command("\t".join(toks))
        if not out.startswith("OKBATCH2"):
            raise RuntimeError(f"VBATCH_NAV2 failed: {out}")
        parts = out.split()
        if len(parts) < 3:
            raise RuntimeError(f"malformed OKBATCH2: {out}")
        n = int(parts[2])
        results: list[VBatchResult] = []
        i = 3
        for _ in range(n):
            case_id = parts[i]; i += 1
            ok = bool(int(parts[i])); i += 1
            ca_distance_m = float(parts[i]); i += 1
            ca_dt_s = float(parts[i]); i += 1
            ca_t_game_s = float(parts[i]); i += 1
            ca_speed_m_s = float(parts[i]); i += 1
            ca_radial_velocity_m_s = float(parts[i]); i += 1
            edge = parts[i]; i += 1
            status = parts[i]; i += 1
            rel_r = np.asarray([float(parts[i]), float(parts[i+1]), float(parts[i+2])], dtype=float); i += 3
            rel_v = np.asarray([float(parts[i]), float(parts[i+1]), float(parts[i+2])], dtype=float); i += 3
            results.append(VBatchResult(
                case_id=case_id,
                ok=ok,
                ca_distance_m=ca_distance_m,
                ca_dt_s=ca_dt_s,
                ca_t_game_s=ca_t_game_s,
                ca_speed_m_s=ca_speed_m_s,
                ca_radial_velocity_m_s=ca_radial_velocity_m_s,
                edge=edge,
                status=status,
                ca_rel_r_raw_m=rel_r,
                ca_rel_v_raw_m_s=rel_v,
            ))
        return results


def make_case_from_candidate(c: dict[str, Any], args: argparse.Namespace, tnb: np.ndarray, case_id: str) -> VBatchCase:
    state_abs_s = get_float(c, "burn_abs_s", "burn_abs", "t_burn_s")
    rel_r = get_vec(c, "burn_rel_r_raw_m")
    rel_v = get_vec(c, "burn_rel_v_raw_m_s")
    t_arr_s = get_float(c, "t_arr_s")
    tof = t_arr_s - state_abs_s
    scan_center = tof + args.scan_center_offset_days * DAY_S
    scan_start = scan_center - args.scan_half_width_days * DAY_S
    scan_end = scan_center + args.scan_half_width_days * DAY_S
    dep_body, arr_body, nav_body = candidate_leg1_dep_arr(c, args)
    return VBatchCase(
        case_id=case_id,
        dep_body=dep_body,
        arr_body=arr_body,
        nav_body=nav_body,
        state_abs_s=state_abs_s,
        scan_start_dt_s=scan_start,
        scan_end_dt_s=scan_end,
        samples=args.samples,
        rel_r_raw_m=rel_r,
        rel_v_raw_m_s=rel_v,
        tnb_m_s=tnb,
    )


def candidate_seed_tnb(c: dict[str, Any], flip_normal: bool, flip_binormal: bool) -> np.ndarray:
    T = get_float(c, "dv_tangent_m_s", "dvt_m_s")
    N = get_float(c, "dv_normal_m_s", "dvn_m_s", default=0.0)
    B = get_float(c, "dv_binormal_m_s", "dvb_m_s", default=0.0)
    if flip_normal:
        N = -N
    if flip_binormal:
        B = -B
    return np.asarray([T, N, B], dtype=float)


def residual_for_result(result: VBatchResult, target_ca_km: float, radial_velocity_scale_s: float) -> np.ndarray:
    # Aim for the same CA direction, but at a specified radius, and optionally
    # near-zero radial velocity. This is not the final B-plane target; it is a
    # first operational probe.
    r = result.ca_rel_r_raw_m
    rhat = unit(r, fallback=[1, 0, 0])
    target = rhat * (target_ca_km * 1000.0)
    pos_res = r - target
    radial_res_as_m = result.ca_radial_velocity_m_s * radial_velocity_scale_s
    return np.r_[pos_res, radial_res_as_m]


def probe_one_candidate(client: PrincipiaClient, candidate: dict[str, Any], args: argparse.Namespace, index: int) -> dict[str, Any]:
    seed = candidate_seed_tnb(candidate, args.flip_normal, args.flip_binormal)
    base_case = make_case_from_candidate(candidate, args, seed, f"c{index}_base")
    base = client.vbatch_nav2(f"base{index}", [base_case])[0]

    # If requested, do a single finite-difference least-squares correction.
    corrected = base
    corrected_tnb = seed.copy()
    dx = np.zeros(3)
    fd_rows: list[dict[str, Any]] = []
    accepted = False

    if args.probe_steps > 0 and base.ok and base.edge == "none":
        current_tnb = seed.copy()
        current = base
        for step_i in range(args.probe_steps):
            cases: list[VBatchCase] = []
            # central differences in T,N,B.
            axes = np.eye(3)
            for j, axis in enumerate(axes):
                cases.append(make_case_from_candidate(candidate, args, current_tnb + args.fd_step_m_s * axis, f"c{index}_s{step_i}_p{j}"))
                cases.append(make_case_from_candidate(candidate, args, current_tnb - args.fd_step_m_s * axis, f"c{index}_s{step_i}_m{j}"))
            res = client.vbatch_nav2(f"fd{index}_{step_i}", cases)
            if len(res) != 6:
                break
            base_resid = residual_for_result(current, args.target_ca_km, args.radial_velocity_scale_s)
            J = np.zeros((4, 3), dtype=float)
            for j in range(3):
                rp = residual_for_result(res[2*j], args.target_ca_km, args.radial_velocity_scale_s)
                rm = residual_for_result(res[2*j + 1], args.target_ca_km, args.radial_velocity_scale_s)
                J[:, j] = (rp - rm) / (2.0 * args.fd_step_m_s)
            scale = np.asarray([args.residual_scale_km * 1000.0] * 3 + [args.residual_scale_km * 1000.0], dtype=float)
            A = J / scale[:, None]
            b = base_resid / scale
            lhs = A.T @ A + args.lm * np.eye(3)
            rhs = -A.T @ b
            try:
                dx_step = np.linalg.solve(lhs, rhs)
            except np.linalg.LinAlgError:
                dx_step = np.linalg.lstsq(lhs, rhs, rcond=None)[0]
            dx_step = clamp_norm(dx_step, args.step_max_m_s)

            best_trial = current
            best_tnb = current_tnb
            best_score = score_result(candidate, current, current_tnb, np.zeros(3), args)
            best_alpha = 0.0
            for alpha in args.line_search:
                trial_tnb = current_tnb + alpha * dx_step
                trial = client.vbatch_nav2(
                    f"ls{index}_{step_i}_{alpha:g}",
                    [make_case_from_candidate(candidate, args, trial_tnb, f"c{index}_s{step_i}_a{alpha:g}")],
                )[0]
                trial_score = score_result(candidate, trial, trial_tnb, alpha * dx_step, args)
                if trial_score < best_score:
                    best_score = trial_score
                    best_trial = trial
                    best_tnb = trial_tnb
                    best_alpha = alpha
            fd_rows.append({
                "step": step_i,
                "base_ca_km": current.ca_distance_km,
                "chosen_ca_km": best_trial.ca_distance_km,
                "dx_step_m_s": [float(x) for x in dx_step],
                "chosen_alpha": float(best_alpha),
            })
            if best_alpha <= 0:
                break
            accepted = True
            current = best_trial
            current_tnb = best_tnb
        corrected = current
        corrected_tnb = current_tnb
        dx = corrected_tnb - seed

    score = score_result(candidate, corrected, corrected_tnb, dx, args)
    T, N, B = corrected_tnb
    oop = math.sqrt(N*N + B*B)
    plane_angle = math.degrees(math.atan2(oop, max(abs(T), 1e-12)))
    out = {
        "source_row_index0": candidate.get("row_index0"),
        "sequence": as_seq_string(candidate.get("sequence") or candidate.get("sequence_bodies")),
        "probe_dep_body": base_case.dep_body,
        "probe_arr_body": base_case.arr_body,
        "probe_nav_body": base_case.nav_body,
        "raw_sum": candidate.get("raw_sum"),
        "original_dv_norm_m_s": candidate.get("dv_norm_m_s"),
        "seed_tnb_m_s": [float(x) for x in seed],
        "seed_oop_m_s": float(math.sqrt(seed[1]**2 + seed[2]**2)),
        "seed_plane_angle_deg": float(math.degrees(math.atan2(math.sqrt(seed[1]**2 + seed[2]**2), max(abs(seed[0]), 1e-12)))),
        "base": base.serializable(),
        "probe": corrected.serializable(),
        "probe_steps": fd_rows,
        "probe_accepted": accepted,
        "probe_dx_tnb_m_s": [float(x) for x in dx],
        "probe_dx_norm_m_s": float(norm(dx)),
        "probe_tnb_m_s": [float(T), float(N), float(B)],
        "probe_dv_norm_m_s": float(norm(corrected_tnb)),
        "probe_oop_m_s": float(oop),
        "probe_plane_angle_deg": float(plane_angle),
        "principia_probe_score": float(score),
        "candidate": candidate,
    }
    return out


def score_result(candidate: dict[str, Any], result: VBatchResult, tnb: np.ndarray, dx: np.ndarray, args: argparse.Namespace) -> float:
    if not result.ok:
        return 1e12
    T, N, B = map(float, tnb)
    oop = math.sqrt(N*N + B*B)
    angle = math.degrees(math.atan2(oop, max(abs(T), 1e-12)))
    resid = residual_for_result(result, args.target_ca_km, args.radial_velocity_scale_s)
    pos_res_km = norm(resid[:3]) / 1000.0
    radial_res_km = abs(resid[3]) / 1000.0
    score = 0.0
    score += args.ca_residual_weight * pos_res_km
    score += args.radial_residual_weight * radial_res_km
    score += args.dv_weight * norm(tnb)
    score += args.oop_weight * oop
    score += args.dx_weight * norm(dx)
    # Soft penalties.
    if abs(N) > args.normal_soft_m_s:
        score += args.soft_penalty_weight * (abs(N) - args.normal_soft_m_s) ** 2
    if abs(B) > args.binormal_soft_m_s:
        score += args.soft_penalty_weight * (abs(B) - args.binormal_soft_m_s) ** 2
    if oop > args.oop_soft_m_s:
        score += args.soft_penalty_weight * (oop - args.oop_soft_m_s) ** 2
    if angle > args.plane_angle_soft_deg:
        score += args.angle_penalty_weight * (angle - args.plane_angle_soft_deg) ** 2
    if result.edge != "none":
        score += args.edge_penalty
    return float(score)


def candidate_probe_fingerprint(candidate: dict[str, Any], args: argparse.Namespace) -> str:
    """Fingerprint for first-leg probe equivalence.

    Candidates that differ only in later legs have identical first-leg Principia
    probe. We can evaluate one representative and clone the result, which saves
    a lot of time on repeated rows such as 38/52/72.
    """
    seed = candidate_seed_tnb(candidate, args.flip_normal, args.flip_binormal)
    dep_body, arr_body, nav_body = candidate_leg1_dep_arr(candidate, args)
    state_abs_s = get_float(candidate, "burn_abs_s", "burn_abs", "t_burn_s")
    t_arr_s = get_float(candidate, "t_arr_s")
    rel_r = get_vec(candidate, "burn_rel_r_raw_m")
    rel_v = get_vec(candidate, "burn_rel_v_raw_m_s")
    vals: list[Any] = [
        dep_body, arr_body, nav_body,
        round(state_abs_s, 6), round(t_arr_s, 6),
        args.scan_center_offset_days, args.scan_half_width_days, args.samples,
        args.target_ca_km, args.radial_velocity_scale_s,
        args.probe_steps, args.fd_step_m_s, args.step_max_m_s, args.lm,
        tuple(round(float(x), 6) for x in seed),
        tuple(round(float(x), 3) for x in rel_r),
        tuple(round(float(x), 6) for x in rel_v),
    ]
    return json.dumps(vals, sort_keys=True, separators=(",", ":"))


def clone_probe_row_for_candidate(row: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    cloned = json.loads(json.dumps(row))
    cloned["source_row_index0"] = candidate.get("row_index0")
    cloned["sequence"] = as_seq_string(candidate.get("sequence") or candidate.get("sequence_bodies"))
    cloned["raw_sum"] = candidate.get("raw_sum")
    cloned["original_dv_norm_m_s"] = candidate.get("dv_norm_m_s")
    cloned["candidate"] = candidate
    cloned["deduped_from_row_index0"] = row.get("source_row_index0")
    return cloned


def chunk_evenly(items: list[Any], n_chunks: int) -> list[list[Any]]:
    n_chunks = max(1, int(n_chunks))
    chunks = [[] for _ in range(n_chunks)]
    for i, item in enumerate(items):
        chunks[i % n_chunks].append(item)
    return [c for c in chunks if c]


def probe_worker(worker_id: int, indexed_candidates: list[tuple[int, dict[str, Any]]], args: argparse.Namespace) -> list[tuple[int, dict[str, Any] | None, str | None]]:
    client = PrincipiaClient(args.server, args.plugin_b64, args.plugin_on_argv, args.server_timeout_s, args.quiet_stderr)
    out: list[tuple[int, dict[str, Any] | None, str | None]] = []
    try:
        for unique_index, candidate in indexed_candidates:
            try:
                row = probe_one_candidate(client, candidate, args, unique_index)
                out.append((unique_index, row, None))
            except Exception as exc:
                out.append((unique_index, None, f"{type(exc).__name__}: {exc}"))
    finally:
        client.close()
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    flat_rows = []
    for r in rows:
        p = r["probe"]
        b = r["base"]
        flat_rows.append({
            "source_row_index0": r.get("source_row_index0"),
            "sequence": r.get("sequence"),
            "probe_dep_body": r.get("probe_dep_body"),
            "probe_arr_body": r.get("probe_arr_body"),
            "probe_nav_body": r.get("probe_nav_body"),
            "score": r.get("principia_probe_score"),
            "base_ca_km": b.get("ca_distance_km"),
            "base_edge": b.get("edge"),
            "base_status": b.get("status"),
            "probe_ca_km": p.get("ca_distance_km"),
            "probe_edge": p.get("edge"),
            "probe_status": p.get("status"),
            "probe_radial_m_s": p.get("ca_radial_velocity_m_s"),
            "probe_speed_m_s": p.get("ca_speed_m_s"),
            "T": r["probe_tnb_m_s"][0],
            "N": r["probe_tnb_m_s"][1],
            "B": r["probe_tnb_m_s"][2],
            "dv_norm_m_s": r.get("probe_dv_norm_m_s"),
            "oop_m_s": r.get("probe_oop_m_s"),
            "plane_angle_deg": r.get("probe_plane_angle_deg"),
            "dx_norm_m_s": r.get("probe_dx_norm_m_s"),
            "seed_T": r["seed_tnb_m_s"][0],
            "seed_N": r["seed_tnb_m_s"][1],
            "seed_B": r["seed_tnb_m_s"][2],
        })
    fields = list(flat_rows[0].keys()) if flat_rows else []
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(flat_rows)


def patch_rank_json(original_data: Any, top_rows: list[dict[str, Any]], out_path: Path) -> None:
    data = json.loads(json.dumps(original_data))
    patched_by_row: dict[int, dict[str, Any]] = {}
    for r in top_rows:
        row = r.get("source_row_index0")
        if row is not None:
            patched_by_row[int(row)] = r

    def walk(x: Any) -> None:
        if isinstance(x, dict):
            if "row_index0" in x and int(x["row_index0"]) in patched_by_row:
                pr = patched_by_row[int(x["row_index0"])]
                T, N, B = pr["probe_tnb_m_s"]
                x["principia_probe_score"] = pr["principia_probe_score"]
                x["principia_probe_ca_distance_km"] = pr["probe"]["ca_distance_km"]
                x["principia_probe_edge"] = pr["probe"]["edge"]
                x["principia_probe_status"] = pr["probe"]["status"]
                x["principia_probe_dx_tnb_m_s"] = pr["probe_dx_tnb_m_s"]
                x["principia_probe_oop_m_s"] = pr["probe_oop_m_s"]
                x["principia_probe_plane_angle_deg"] = pr["probe_plane_angle_deg"]
                x["dv_tangent_m_s"] = T
                x["dv_normal_m_s"] = N
                x["dv_binormal_m_s"] = B
                x["dv_norm_m_s"] = float(norm([T, N, B]))
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)
    walk(data)
    # Also replace/insert top ordered list for convenience.
    if isinstance(data, dict):
        data["principia_probe_top"] = top_rows
    out_path.write_text(json.dumps(data, indent=2) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--server", type=Path, required=True)
    ap.add_argument("--plugin-b64", type=Path, required=True)
    ap.add_argument("--plugin-on-argv", action="store_true")
    ap.add_argument("--rank-json", type=Path, required=True)
    ap.add_argument("--top-n", type=int, default=20)
    ap.add_argument("--jobs", type=int, default=1, help="Number of parallel Principia server processes. Each worker starts its own server.")
    ap.add_argument("--no-dedupe", action="store_true", help="Disable first-leg probe de-duplication.")
    ap.add_argument("--row-index0", type=int, action="append", default=[])
    ap.add_argument("--dep-body", default="AUTO", help="AUTO uses first body of the candidate sequence")
    ap.add_argument("--arr-body", default="AUTO", help="AUTO uses second body / leg1 arrival of each candidate")
    ap.add_argument("--nav-body", default="AUTO", help="AUTO uses the candidate departure body")
    ap.add_argument("--flip-normal", action="store_true")
    ap.add_argument("--flip-binormal", action="store_true")
    ap.add_argument("--scan-center-offset-days", type=float, default=0.0)
    ap.add_argument("--scan-half-width-days", type=float, default=30.0)
    ap.add_argument("--samples", type=int, default=121)
    ap.add_argument("--target-ca-km", type=float, default=200000.0)
    ap.add_argument("--radial-velocity-scale-s", type=float, default=1000.0)
    ap.add_argument("--probe-steps", type=int, default=1)
    ap.add_argument("--fd-step-m-s", type=float, default=1.0)
    ap.add_argument("--step-max-m-s", type=float, default=60.0)
    ap.add_argument("--lm", type=float, default=1e-4)
    ap.add_argument("--line-search", default="1,0.5,0.25,0.125")
    ap.add_argument("--residual-scale-km", type=float, default=100000.0)

    # Score weights.
    ap.add_argument("--ca-residual-weight", type=float, default=1.0)
    ap.add_argument("--radial-residual-weight", type=float, default=0.5)
    ap.add_argument("--dv-weight", type=float, default=0.02)
    ap.add_argument("--oop-weight", type=float, default=0.8)
    ap.add_argument("--dx-weight", type=float, default=2.0)
    ap.add_argument("--normal-soft-m-s", type=float, default=200.0)
    ap.add_argument("--binormal-soft-m-s", type=float, default=350.0)
    ap.add_argument("--oop-soft-m-s", type=float, default=400.0)
    ap.add_argument("--plane-angle-soft-deg", type=float, default=10.0)
    ap.add_argument("--soft-penalty-weight", type=float, default=0.05)
    ap.add_argument("--angle-penalty-weight", type=float, default=100.0)
    ap.add_argument("--edge-penalty", type=float, default=1e7)

    ap.add_argument("--quiet-stderr", action="store_true")
    ap.add_argument("--server-timeout-s", type=float, default=900.0)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()
    args.line_search = [float(x) for x in str(args.line_search).split(",") if x.strip()]

    data = load_json(args.rank_json)
    candidates = find_candidates(data)
    if args.row_index0:
        wanted = set(args.row_index0)
        candidates = [c for c in candidates if int(c.get("row_index0", -999999)) in wanted]
    else:
        candidates = candidates[:args.top_n]
    if not candidates:
        raise SystemExit("no candidates found")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # De-duplicate candidates that have the same first-leg probe. This is safe
    # for this script because it only evaluates leg 1 (departure -> first flyby),
    # not the downstream handoff yet.
    duplicate_groups: list[tuple[dict[str, Any], list[dict[str, Any]]]] = []
    if args.no_dedupe:
        duplicate_groups = [(c, [c]) for c in candidates]
    else:
        by_fp: dict[str, tuple[dict[str, Any], list[dict[str, Any]]]] = {}
        for c in candidates:
            try:
                fp = candidate_probe_fingerprint(c, args)
            except Exception:
                # If fingerprinting fails, still evaluate it uniquely so the
                # user sees the underlying failure rather than losing the row.
                fp = f"unique-error-{id(c)}"
            if fp not in by_fp:
                by_fp[fp] = (c, [])
            by_fp[fp][1].append(c)
        duplicate_groups = list(by_fp.values())

    unique_candidates = [rep for rep, _members in duplicate_groups]
    print(f"[INFO] selected={len(candidates)} unique_first_leg_probes={len(unique_candidates)} deduped={len(candidates) - len(unique_candidates)} jobs={args.jobs}")

    unique_rows: dict[int, dict[str, Any]] = {}
    failures: list[str] = []

    indexed = list(enumerate(unique_candidates))
    if args.jobs <= 1:
        client = PrincipiaClient(args.server, args.plugin_b64, args.plugin_on_argv, args.server_timeout_s, args.quiet_stderr)
        try:
            for unique_i, c in indexed:
                try:
                    row = probe_one_candidate(client, c, args, unique_i)
                    unique_rows[unique_i] = row
                    print(
                        f"[{unique_i+1:03d}/{len(unique_candidates):03d}] row={row.get('source_row_index0')} "
                        f"score={row['principia_probe_score']:12.3f} "
                        f"base_ca={row['base']['ca_distance_km']:12.3f} km "
                        f"probe_ca={row['probe']['ca_distance_km']:12.3f} km "
                        f"TNB=[{row['probe_tnb_m_s'][0]:.3f},{row['probe_tnb_m_s'][1]:.3f},{row['probe_tnb_m_s'][2]:.3f}] "
                        f"oop={row['probe_oop_m_s']:.3f} angle={row['probe_plane_angle_deg']:.2f} "
                        f"edge={row['probe']['edge']} status={row['probe']['status']}"
                    )
                except Exception as exc:
                    msg = f"FAILED unique={unique_i} row={c.get('row_index0')}: {exc}"
                    failures.append(msg)
                    print(msg, file=sys.stderr)
        finally:
            client.close()
    else:
        chunks = chunk_evenly(indexed, args.jobs)
        with ProcessPoolExecutor(max_workers=args.jobs) as executor:
            futs = [executor.submit(probe_worker, wid, chunk, args) for wid, chunk in enumerate(chunks)]
            done = 0
            for fut in as_completed(futs):
                for unique_i, row, err in fut.result():
                    done += 1
                    if err is not None or row is None:
                        msg = f"FAILED unique={unique_i} row={unique_candidates[unique_i].get('row_index0')}: {err}"
                        failures.append(msg)
                        print(msg, file=sys.stderr)
                        continue
                    unique_rows[unique_i] = row
                    print(
                        f"[{done:03d}/{len(unique_candidates):03d}] row={row.get('source_row_index0')} "
                        f"score={row['principia_probe_score']:12.3f} "
                        f"base_ca={row['base']['ca_distance_km']:12.3f} km "
                        f"probe_ca={row['probe']['ca_distance_km']:12.3f} km "
                        f"TNB=[{row['probe_tnb_m_s'][0]:.3f},{row['probe_tnb_m_s'][1]:.3f},{row['probe_tnb_m_s'][2]:.3f}] "
                        f"oop={row['probe_oop_m_s']:.3f} angle={row['probe_plane_angle_deg']:.2f} "
                        f"edge={row['probe']['edge']} status={row['probe']['status']}"
                    )

    rows: list[dict[str, Any]] = []
    for unique_i, (rep, members) in enumerate(duplicate_groups):
        if unique_i not in unique_rows:
            continue
        row = unique_rows[unique_i]
        for member in members:
            rows.append(clone_probe_row_for_candidate(row, member))

    rows.sort(key=lambda r: r["principia_probe_score"])
    result = {
        "schema": "rank_pykep_candidates_by_principia_probe_v2_mp_dedupe",
        "rank_json": str(args.rank_json),
        "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "n_rows": len(rows),
        "n_failures": len(failures),
        "failures": failures,
        "top": rows,
    }
    (args.output_dir / "candidate_rank_principia_probe.json").write_text(json.dumps(result, indent=2) + "\n")
    if rows:
        write_csv(args.output_dir / "candidate_rank_principia_probe.csv", rows)
        patch_rank_json(data, rows, args.output_dir / "candidate_departure_executability_rank_principia_probe.json")
    print(f"[OK] wrote {args.output_dir / 'candidate_rank_principia_probe.json'}")
    print(f"[OK] wrote {args.output_dir / 'candidate_rank_principia_probe.csv'}")
    print(f"[OK] wrote {args.output_dir / 'candidate_departure_executability_rank_principia_probe.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
