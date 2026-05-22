#!/usr/bin/env python3
"""
principia_vbatch_nav2_targeter_v0_3.py

Principia-native local targeter using the VBATCH_NAV2 protocol added to
principia_impulsive_particle_server.

Purpose
-------
Use the real Principia backend as the propagation/evaluation engine and solve
small local corrections in navigation TNB components using finite-difference
Jacobians computed in one VBATCH_NAV2 call per iteration.

This is not a Lambert/PyKEP solver. It assumes a candidate already exists and
refines its first impulsive burn against the Principia server.

Supported targets
-----------------
1) ca-radius-current-direction
   Keep the current closest-approach side/direction, but change the miss radius
   toward --target-ca-km. This is useful as a first flyby-safe target instead of
   targeting zero distance / impact.

2) rel-r
   Target an explicit raw body-relative position vector at closest approach.

3) bplane
   Target approximate B-plane coordinates in a frame built from the base
   relative velocity at closest approach and --bplane-reference-raw. This is
   intended for gravity-assist handoff once you have a desired B-plane aimpoint.

Protocol assumptions
--------------------
Server command:
  VBATCH_NAV2<TAB>batch_id<TAB>n_cases ...cases...

Per case:
  case_id dep_body arr_body nav_body state_abs_s scan_start_dt_s scan_end_dt_s samples
  rel_r_raw_m[3] rel_v_raw_m_s[3] n_impulses [dt_s dvt dvn dvb]...

Response:
  OKBATCH2 batch_id n_results
    case_id ok ca_distance_m ca_dt_s ca_t_game_s ca_speed_m_s ca_radial_velocity_m_s
    edge status ca_rel_r_raw_m[3] ca_rel_v_raw_m_s[3]

The current server patch may only physically support one impulse even if the
parser accepts n_impulses. This script defaults to one impulse.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shlex
import signal
import subprocess
import sys
import time
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
    if not math.isfinite(n) or n < 1e-30:
        if fallback is None:
            raise ValueError(f"cannot normalize {v!r}")
        return unit(fallback)
    return a / n


def parse_vec3_csv(s: str | None, *, default: Sequence[float] | None = None) -> np.ndarray | None:
    if s is None or str(s).strip() == "":
        return None if default is None else np.asarray(default, dtype=float)
    vals = [float(x.strip()) for x in s.split(",")]
    if len(vals) != 3:
        raise ValueError(f"expected 3 comma-separated values, got {s!r}")
    return np.asarray(vals, dtype=float)


def load_json(path: Path) -> Any:
    with path.open() as f:
        return json.load(f)


def find_candidate(data: Any, row_index0: int | None, top_index: int) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []

    def walk(x: Any) -> None:
        if isinstance(x, dict):
            if "row_index0" in x and (
                "burn_rel_r_raw_m" in x or "dv_tangent_m_s" in x or "t_arr_s" in x
            ):
                candidates.append(x)
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)

    walk(data)
    if row_index0 is not None:
        matches = [c for c in candidates if int(c.get("row_index0", -999999)) == int(row_index0)]
        if not matches:
            raise KeyError(f"row_index0={row_index0} not found")
        matches.sort(key=lambda c: sum(k in c for k in (
            "burn_abs_s", "burn_rel_r_raw_m", "burn_rel_v_raw_m_s", "t_arr_s",
            "dv_tangent_m_s", "dv_normal_m_s", "dv_binormal_m_s")), reverse=True)
        return matches[0]
    if isinstance(data, dict) and "top" in data and isinstance(data["top"], list):
        return data["top"][top_index]
    if candidates:
        return candidates[top_index]
    if isinstance(data, dict):
        return data
    raise ValueError("cannot find candidate")


def get_seed_tnb(c: dict[str, Any], flip_normal: bool, flip_binormal: bool) -> np.ndarray:
    t = float(c.get("dv_tangent_m_s", c.get("dvt_m_s")))
    n = float(c.get("dv_normal_m_s", c.get("dvn_m_s")))
    b = float(c.get("dv_binormal_m_s", c.get("dvb_m_s")))
    if flip_normal:
        n = -n
    if flip_binormal:
        b = -b
    return np.asarray([t, n, b], dtype=float)


def find_best_from_result(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    data = load_json(path)
    if isinstance(data, dict):
        b = data.get("best")
        if isinstance(b, dict):
            return b
        top = data.get("top") or data.get("top_validated")
        if isinstance(top, list) and top:
            return top[0]
    return None


@dataclass
class Impulse:
    dt_s: float
    dvt_m_s: float
    dvn_m_s: float
    dvb_m_s: float


@dataclass
class Case:
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
    impulses: list[Impulse]


@dataclass
class BatchResult:
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
    raw_tokens: list[str]

    @property
    def ca_distance_km(self) -> float:
        return self.ca_distance_m / 1000.0

    def to_json(self) -> dict[str, Any]:
        d = asdict(self)
        d["ca_rel_r_raw_m"] = self.ca_rel_r_raw_m.tolist()
        d["ca_rel_v_raw_m_s"] = self.ca_rel_v_raw_m_s.tolist()
        d["ca_distance_km"] = self.ca_distance_km
        return d


class PrincipiaBatchClient:
    def __init__(self, server: Path, plugin_b64: Path | None, *, plugin_on_argv: bool = False,
                 timeout_s: float = 900.0, quiet_stderr: bool = False):
        self.server = server
        self.plugin_b64 = plugin_b64
        self.timeout_s = timeout_s
        self.proc: subprocess.Popen[str] | None = None
        self.quiet_stderr = quiet_stderr
        self.plugin_on_argv = plugin_on_argv

    def __enter__(self) -> "PrincipiaBatchClient":
        cmd = [str(self.server)]
        if self.plugin_b64 is not None and self.plugin_on_argv:
            cmd.append(str(self.plugin_b64))
        stderr = subprocess.DEVNULL if self.quiet_stderr else None
        self.proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=stderr,
            text=True,
            bufsize=1,
        )
        if self.plugin_b64 is not None and not self.plugin_on_argv:
            out = self.command(f"LOADPLUGIN\tp0\t{self.plugin_b64}")
            if not out.startswith("OKPLUGIN"):
                self.close()
                raise RuntimeError(f"LOADPLUGIN failed: {out}")
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def close(self) -> None:
        if self.proc is None:
            return
        try:
            if self.proc.stdin:
                try:
                    self.proc.stdin.write("QUIT\n")
                    self.proc.stdin.flush()
                except Exception:
                    pass
            self.proc.terminate()
            try:
                self.proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        finally:
            self.proc = None

    def command(self, line: str) -> str:
        if self.proc is None or self.proc.stdin is None or self.proc.stdout is None:
            raise RuntimeError("server not started")
        self.proc.stdin.write(line.rstrip("\n") + "\n")
        self.proc.stdin.flush()
        # Blocking read; server commands are synchronous. Timeout is handled by polling loop.
        deadline = time.monotonic() + self.timeout_s
        while True:
            if time.monotonic() > deadline:
                raise TimeoutError(f"server timeout after {self.timeout_s}s for line: {line[:200]}")
            out = self.proc.stdout.readline()
            if out:
                out = out.rstrip("\n")
                stripped = out.strip()
                # Some patched server builds print startup banners before the
                # first command response, for example READY or a version string
                # such as pincipia_impulsive_particle_server_v0_7_....  Only
                # return actual protocol responses; ignore banners/noise.
                if stripped == "":
                    continue
                if stripped.startswith(("OK", "ERR", "PONG")):
                    return out
                # Startup/version banner or other informational line.
                continue
            if self.proc.poll() is not None:
                raise RuntimeError(f"server exited with code {self.proc.returncode}")
            time.sleep(0.01)

    @staticmethod
    def _format_case(c: Case) -> str:
        fields: list[str] = [
            c.case_id,
            c.dep_body,
            c.arr_body,
            c.nav_body,
            f"{c.state_abs_s:.17g}",
            f"{c.scan_start_dt_s:.17g}",
            f"{c.scan_end_dt_s:.17g}",
            str(int(c.samples)),
            *(f"{float(v):.17g}" for v in c.rel_r_raw_m),
            *(f"{float(v):.17g}" for v in c.rel_v_raw_m_s),
            str(len(c.impulses)),
        ]
        for imp in c.impulses:
            fields.extend([
                f"{imp.dt_s:.17g}",
                f"{imp.dvt_m_s:.17g}",
                f"{imp.dvn_m_s:.17g}",
                f"{imp.dvb_m_s:.17g}",
            ])
        return " ".join(fields)

    def vbatch_nav2(self, batch_id: str, cases: list[Case]) -> list[BatchResult]:
        body = " ".join(self._format_case(c) for c in cases)
        line = f"VBATCH_NAV2\t{batch_id}\t{len(cases)} {body}"
        out = self.command(line)
        if not out.startswith("OKBATCH2"):
            raise RuntimeError(f"VBATCH_NAV2 failed: {out}")
        return parse_okbatch2(out)


def parse_okbatch2(line: str) -> list[BatchResult]:
    toks = line.split()
    if len(toks) < 3 or toks[0] != "OKBATCH2":
        raise ValueError(f"not OKBATCH2: {line}")
    n = int(toks[2])
    i = 3
    rows: list[BatchResult] = []
    per = 15
    for _ in range(n):
        if i + per > len(toks):
            raise ValueError(f"truncated OKBATCH2 at token {i}: {line}")
        raw = toks[i:i+per]
        case_id = raw[0]
        ok = raw[1] in ("1", "true", "True")
        rows.append(BatchResult(
            case_id=case_id,
            ok=ok,
            ca_distance_m=float(raw[2]),
            ca_dt_s=float(raw[3]),
            ca_t_game_s=float(raw[4]),
            ca_speed_m_s=float(raw[5]),
            ca_radial_velocity_m_s=float(raw[6]),
            edge=raw[7],
            status=raw[8],
            ca_rel_r_raw_m=np.asarray([float(raw[9]), float(raw[10]), float(raw[11])], dtype=float),
            ca_rel_v_raw_m_s=np.asarray([float(raw[12]), float(raw[13]), float(raw[14])], dtype=float),
            raw_tokens=raw,
        ))
        i += per
    return rows


def bplane_frame(rel_v: np.ndarray, reference: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # S points along incoming relative velocity. The B-plane is perpendicular to S.
    S = unit(rel_v, fallback=[1, 0, 0])
    ref = unit(reference, fallback=[0, 0, 1])
    T = np.cross(ref, S)
    if norm(T) < 1e-12:
        ref = np.asarray([0.0, 1.0, 0.0])
        T = np.cross(ref, S)
    T = unit(T, fallback=[1, 0, 0])
    R = unit(np.cross(S, T), fallback=[0, 1, 0])
    return S, T, R


class TargetResidual:
    def __init__(self, mode: str, *, target_ca_m: float | None = None,
                 target_rel_r_raw_m: np.ndarray | None = None,
                 target_bplane_m: np.ndarray | None = None,
                 bplane_reference_raw: np.ndarray | None = None,
                 radial_velocity_scale_s: float = 0.0):
        self.mode = mode
        self.target_ca_m = target_ca_m
        self.target_rel_r_raw_m = target_rel_r_raw_m
        self.target_bplane_m = target_bplane_m
        self.bplane_reference_raw = np.asarray(bplane_reference_raw if bplane_reference_raw is not None else [0, 0, 1], dtype=float)
        self.radial_velocity_scale_s = float(radial_velocity_scale_s)
        self._fixed_direction: np.ndarray | None = None
        self._bplane_frame: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None

    def initialize_from_base(self, base: BatchResult) -> None:
        if self.mode == "ca-radius-current-direction":
            if self.target_ca_m is None:
                raise ValueError("--target-ca-km required for ca-radius-current-direction")
            self._fixed_direction = unit(base.ca_rel_r_raw_m, fallback=[1, 0, 0])
        elif self.mode == "bplane":
            if self.target_bplane_m is None:
                raise ValueError("--target-bplane-km required for bplane mode")
            self._bplane_frame = bplane_frame(base.ca_rel_v_raw_m_s, self.bplane_reference_raw)
        elif self.mode == "rel-r":
            if self.target_rel_r_raw_m is None:
                raise ValueError("--target-rel-r-raw-m required for rel-r mode")
        elif self.mode == "minimize-ca":
            pass
        else:
            raise ValueError(f"unknown target mode {self.mode!r}")

    def residual(self, r: BatchResult) -> np.ndarray:
        if self.mode == "minimize-ca":
            # Scalar residual; use the distance itself. This aims at impact and is mainly diagnostic.
            vals = [r.ca_distance_m]
        elif self.mode == "ca-radius-current-direction":
            assert self._fixed_direction is not None
            target = self._fixed_direction * float(self.target_ca_m)
            vals = list(r.ca_rel_r_raw_m - target)
        elif self.mode == "rel-r":
            assert self.target_rel_r_raw_m is not None
            vals = list(r.ca_rel_r_raw_m - self.target_rel_r_raw_m)
        elif self.mode == "bplane":
            assert self._bplane_frame is not None
            _, T, R = self._bplane_frame
            b_t = float(np.dot(r.ca_rel_r_raw_m, T))
            b_r = float(np.dot(r.ca_rel_r_raw_m, R))
            vals = [b_t - float(self.target_bplane_m[0]), b_r - float(self.target_bplane_m[1])]
        else:
            raise ValueError(self.mode)
        if self.radial_velocity_scale_s > 0:
            vals.append(r.ca_radial_velocity_m_s * self.radial_velocity_scale_s)
        return np.asarray(vals, dtype=float)


def make_case(template: Case, case_id: str, x_tnb: np.ndarray) -> Case:
    imp0 = template.impulses[0]
    return Case(
        case_id=case_id,
        dep_body=template.dep_body,
        arr_body=template.arr_body,
        nav_body=template.nav_body,
        state_abs_s=template.state_abs_s,
        scan_start_dt_s=template.scan_start_dt_s,
        scan_end_dt_s=template.scan_end_dt_s,
        samples=template.samples,
        rel_r_raw_m=template.rel_r_raw_m,
        rel_v_raw_m_s=template.rel_v_raw_m_s,
        impulses=[Impulse(imp0.dt_s, float(x_tnb[0]), float(x_tnb[1]), float(x_tnb[2]))],
    )


def solve_damped_ls(A: np.ndarray, b: np.ndarray, lm: float) -> np.ndarray:
    # Solve min ||A dx - b||^2 + lm ||dx||^2
    n = A.shape[1]
    lhs = A.T @ A + float(lm) * np.eye(n)
    rhs = A.T @ b
    try:
        return np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(lhs, rhs, rcond=None)[0]


def score_residual(res: np.ndarray, scale_m: float, dv_norm: float, dv_weight: float, edge_penalty: float, edge: str) -> float:
    s = norm(res) / max(scale_m, 1e-12) + dv_weight * dv_norm
    if edge != "none":
        s += edge_penalty
    return float(s)


def patch_rank_json(original: Any, row_index0: int, x_tnb: np.ndarray, out_path: Path, extra: dict[str, Any]) -> None:
    data = json.loads(json.dumps(original))
    patched = 0

    def walk(o: Any) -> None:
        nonlocal patched
        if isinstance(o, dict):
            if int(o.get("row_index0", -999999)) == int(row_index0):
                for k, v in (("dv_tangent_m_s", x_tnb[0]), ("dv_normal_m_s", x_tnb[1]), ("dv_binormal_m_s", x_tnb[2]),
                             ("dvt_m_s", x_tnb[0]), ("dvn_m_s", x_tnb[1]), ("dvb_m_s", x_tnb[2])):
                    if k in o:
                        o[k] = float(v)
                o["dv_norm_m_s"] = float(norm(x_tnb))
                o["principia_vbatch_nav2_targeter_v0_3"] = extra
                patched += 1
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)

    walk(data)
    if patched == 0:
        raise RuntimeError(f"row_index0={row_index0} not patched")
    out_path.write_text(json.dumps(data, indent=2) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--server", type=Path, required=True)
    ap.add_argument("--plugin-b64", type=Path, default=None)
    ap.add_argument("--plugin-on-argv", action="store_true", help="Start server with plugin path as argv[1] instead of LOADPLUGIN.")
    ap.add_argument("--rank-json", type=Path, required=True)
    ap.add_argument("--row-index0", type=int, required=True)
    ap.add_argument("--top-index", type=int, default=0)
    ap.add_argument("--seed-result-json", type=Path, default=None)
    ap.add_argument("--seed-tnb", default=None, help="Override seed T,N,B m/s as 'T,N,B'.")
    ap.add_argument("--flip-normal", action="store_true")
    ap.add_argument("--flip-binormal", action="store_true")

    ap.add_argument("--dep-body", default="KERBIN")
    ap.add_argument("--arr-body", default="EVE")
    ap.add_argument("--nav-body", default="KERBIN")
    ap.add_argument("--scan-center-offset-days", type=float, default=0.0)
    ap.add_argument("--scan-half-width-days", type=float, default=30.0)
    ap.add_argument("--samples", type=int, default=121)

    ap.add_argument("--target-mode", choices=["ca-radius-current-direction", "rel-r", "bplane", "minimize-ca"], default="ca-radius-current-direction")
    ap.add_argument("--target-ca-km", type=float, default=None)
    ap.add_argument("--target-rel-r-raw-m", default=None)
    ap.add_argument("--target-bplane-km", default=None, help="For bplane mode: 'BT,BR' in km.")
    ap.add_argument("--bplane-reference-raw", default="0,0,1")
    ap.add_argument("--radial-velocity-scale-s", type=float, default=0.0, help="Append radial_velocity*scale_s as residual component.")

    ap.add_argument("--iterations", type=int, default=8)
    ap.add_argument("--fd-step-m-s", type=float, default=1.0)
    ap.add_argument("--lm", type=float, default=1e-4)
    ap.add_argument("--step-max-m-s", type=float, default=80.0)
    ap.add_argument("--line-search", default="1,0.5,0.25,0.125,0.0625")
    ap.add_argument("--residual-scale-km", type=float, default=100000.0)
    ap.add_argument("--dv-weight", type=float, default=0.0)
    ap.add_argument("--edge-penalty", type=float, default=1000.0)
    ap.add_argument("--target-residual-km", type=float, default=50_000.0)
    ap.add_argument("--quiet-stderr", action="store_true")
    ap.add_argument("--server-timeout-s", type=float, default=900.0)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rank_data = load_json(args.rank_json)
    cand = find_candidate(rank_data, args.row_index0, args.top_index)

    burn_abs_s = float(cand["burn_abs_s"])
    t_arr_s = float(cand["t_arr_s"])
    tof_s = t_arr_s - burn_abs_s
    scan_center = tof_s + args.scan_center_offset_days * DAY_S
    scan_start = scan_center - args.scan_half_width_days * DAY_S
    scan_end = scan_center + args.scan_half_width_days * DAY_S

    rel_r = np.asarray(cand["burn_rel_r_raw_m"], dtype=float)
    rel_v = np.asarray(cand["burn_rel_v_raw_m_s"], dtype=float)

    x0 = get_seed_tnb(cand, args.flip_normal, args.flip_binormal)
    seed_result = find_best_from_result(args.seed_result_json)
    if seed_result is not None and all(k in seed_result for k in ("dvt_m_s", "dvn_m_s", "dvb_m_s")):
        x0 = np.asarray([seed_result["dvt_m_s"], seed_result["dvn_m_s"], seed_result["dvb_m_s"]], dtype=float)
    override = parse_vec3_csv(args.seed_tnb)
    if override is not None:
        x0 = override

    template = Case(
        case_id="template",
        dep_body=args.dep_body.upper(),
        arr_body=args.arr_body.upper(),
        nav_body=args.nav_body.upper(),
        state_abs_s=burn_abs_s,
        scan_start_dt_s=scan_start,
        scan_end_dt_s=scan_end,
        samples=args.samples,
        rel_r_raw_m=rel_r,
        rel_v_raw_m_s=rel_v,
        impulses=[Impulse(0.0, float(x0[0]), float(x0[1]), float(x0[2]))],
    )

    target_rel = parse_vec3_csv(args.target_rel_r_raw_m)
    target_bp = None
    if args.target_bplane_km is not None:
        vals = [float(x.strip()) for x in args.target_bplane_km.split(",")]
        if len(vals) != 2:
            raise ValueError("--target-bplane-km must be 'BT,BR'")
        target_bp = np.asarray(vals, dtype=float) * 1000.0
    target = TargetResidual(
        args.target_mode,
        target_ca_m=None if args.target_ca_km is None else args.target_ca_km * 1000.0,
        target_rel_r_raw_m=target_rel,
        target_bplane_m=target_bp,
        bplane_reference_raw=parse_vec3_csv(args.bplane_reference_raw, default=[0, 0, 1]),
        radial_velocity_scale_s=args.radial_velocity_scale_s,
    )

    alphas = [float(x.strip()) for x in args.line_search.split(",") if x.strip()]
    rows: list[dict[str, Any]] = []
    iterations: list[dict[str, Any]] = []
    best_result: BatchResult | None = None
    best_x = x0.copy()
    best_score = math.inf

    with PrincipiaBatchClient(args.server, args.plugin_b64, plugin_on_argv=args.plugin_on_argv,
                              timeout_s=args.server_timeout_s, quiet_stderr=args.quiet_stderr) as client:
        x = x0.copy()
        base = client.vbatch_nav2("base0", [make_case(template, "base", x)])[0]
        target.initialize_from_base(base)

        for it in range(args.iterations):
            base = client.vbatch_nav2(f"it{it}_base", [make_case(template, "base", x)])[0]
            r0 = target.residual(base)
            base_score = score_residual(r0, args.residual_scale_km * 1000.0, norm(x), args.dv_weight, args.edge_penalty, base.edge)
            rows.append({"iteration": it, "kind": "base", "x": x.tolist(), "result": base.to_json(), "residual": r0.tolist(), "score": base_score})
            if base_score < best_score:
                best_score = base_score
                best_result = base
                best_x = x.copy()

            print(
                f"[it {it:02d}] base ca={base.ca_distance_km:12.3f} km edge={base.edge:5s} "
                f"res={norm(r0)/1000:12.3f} km TNB=[{x[0]:.3f},{x[1]:.3f},{x[2]:.3f}]"
            )
            if norm(r0) / 1000.0 <= args.target_residual_km and base.edge == "none":
                iterations.append({"iteration": it, "stop": "target", "base_ca_km": base.ca_distance_km})
                break

            # Central differences in one batch.
            fd_cases: list[Case] = []
            for j, name in enumerate(("T", "N", "B")):
                xp = x.copy(); xp[j] += args.fd_step_m_s
                xm = x.copy(); xm[j] -= args.fd_step_m_s
                fd_cases.append(make_case(template, f"{name}p", xp))
                fd_cases.append(make_case(template, f"{name}m", xm))
            fd_results = {r.case_id: r for r in client.vbatch_nav2(f"it{it}_fd", fd_cases)}
            J_cols: list[np.ndarray] = []
            for name in ("T", "N", "B"):
                rp = target.residual(fd_results[f"{name}p"])
                rm = target.residual(fd_results[f"{name}m"])
                J_cols.append((rp - rm) / (2.0 * args.fd_step_m_s))
            J = np.column_stack(J_cols)
            dx = solve_damped_ls(J, -r0, args.lm)
            dx_norm = norm(dx)
            if dx_norm > args.step_max_m_s:
                dx *= args.step_max_m_s / dx_norm

            # Evaluate line-search candidates in one batch.
            ls_cases: list[Case] = []
            xs: list[np.ndarray] = []
            for alpha in alphas:
                xc = x + alpha * dx
                xs.append(xc)
                ls_cases.append(make_case(template, f"a{alpha:g}".replace(".", "p"), xc))
            ls_results = client.vbatch_nav2(f"it{it}_ls", ls_cases)
            scored = []
            for alpha, xc, rr in zip(alphas, xs, ls_results):
                res = target.residual(rr)
                score = score_residual(res, args.residual_scale_km * 1000.0, norm(xc), args.dv_weight, args.edge_penalty, rr.edge)
                rows.append({"iteration": it, "kind": "line", "alpha": alpha, "x": xc.tolist(), "result": rr.to_json(), "residual": res.tolist(), "score": score})
                scored.append((score, alpha, xc, rr, res))
            scored.sort(key=lambda z: z[0])
            chosen_score, chosen_alpha, chosen_x, chosen_result, chosen_res = scored[0]
            accepted = chosen_score < base_score
            if accepted:
                x = chosen_x.copy()
                if chosen_score < best_score:
                    best_score = chosen_score
                    best_result = chosen_result
                    best_x = chosen_x.copy()
            else:
                # Increase damping if no line-search candidate improves.
                args.lm *= 10.0

            print(
                f"          dx={dx.tolist()} |dx|={norm(dx):.3f} alpha={chosen_alpha:g} "
                f"chosen ca={chosen_result.ca_distance_km:12.3f} km edge={chosen_result.edge:5s} "
                f"res={norm(chosen_res)/1000:12.3f} km accepted={accepted} lm={args.lm:g}"
            )
            iterations.append({
                "iteration": it,
                "base_ca_km": base.ca_distance_km,
                "base_score": base_score,
                "base_residual_km": norm(r0) / 1000.0,
                "dx_m_s": dx.tolist(),
                "dx_norm_m_s": norm(dx),
                "chosen_alpha": chosen_alpha,
                "chosen_ca_km": chosen_result.ca_distance_km,
                "chosen_score": chosen_score,
                "chosen_residual_km": norm(chosen_res) / 1000.0,
                "accepted": accepted,
                "lm": args.lm,
            })

    if best_result is None:
        raise RuntimeError("no result")

    summary = {
        "schema": "principia_vbatch_nav2_targeter_v0_3",
        "rank_json": str(args.rank_json),
        "row_index0": args.row_index0,
        "server": str(args.server),
        "plugin_b64": None if args.plugin_b64 is None else str(args.plugin_b64),
        "target_mode": args.target_mode,
        "target_ca_km": args.target_ca_km,
        "target_rel_r_raw_m": None if target_rel is None else target_rel.tolist(),
        "target_bplane_km": None if target_bp is None else (target_bp / 1000.0).tolist(),
        "scan_start_dt_s": scan_start,
        "scan_end_dt_s": scan_end,
        "samples": args.samples,
        "initial_tnb_m_s": x0.tolist(),
        "best_tnb_m_s": best_x.tolist(),
        "best": best_result.to_json(),
        "best_score": best_score,
        "iterations": iterations,
        "rows": rows,
    }
    (args.output_dir / "principia_vbatch_nav2_targeter_result.json").write_text(json.dumps(summary, indent=2) + "\n")
    patch_rank_json(
        rank_data,
        args.row_index0,
        best_x,
        args.output_dir / "rank_row{}_principia_vbatch_nav2_targeted.json".format(args.row_index0),
        {
            "source": "principia_vbatch_nav2_targeter_v0_3",
            "best_ca_distance_km": best_result.ca_distance_km,
            "best_ca_t_game_s": best_result.ca_t_game_s,
            "best_edge": best_result.edge,
            "best_status": best_result.status,
            "target_mode": args.target_mode,
        },
    )

    print("\n=== BEST PRINCIPIA VBATCH_NAV2 TARGETER RESULT ===")
    print(json.dumps({
        "ca_distance_km": best_result.ca_distance_km,
        "ca_t_game_s": best_result.ca_t_game_s,
        "ca_speed_m_s": best_result.ca_speed_m_s,
        "ca_radial_velocity_m_s": best_result.ca_radial_velocity_m_s,
        "edge": best_result.edge,
        "status": best_result.status,
        "dvt_m_s": best_x[0],
        "dvn_m_s": best_x[1],
        "dvb_m_s": best_x[2],
        "dv_norm_m_s": norm(best_x),
        "ca_rel_r_raw_m": best_result.ca_rel_r_raw_m.tolist(),
        "ca_rel_v_raw_m_s": best_result.ca_rel_v_raw_m_s.tolist(),
    }, indent=2))
    print(f"[OK] wrote {args.output_dir / 'principia_vbatch_nav2_targeter_result.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
