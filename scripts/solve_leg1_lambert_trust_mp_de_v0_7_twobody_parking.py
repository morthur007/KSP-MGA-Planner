#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import spiceypy as spice


def norm(v: np.ndarray) -> float:
    return float(np.linalg.norm(v))


def unit(v: np.ndarray) -> np.ndarray:
    n = norm(v)
    if n <= 0 or not math.isfinite(n):
        raise ValueError("zero/non-finite vector")
    return v / n


def levela_to_raw(v: np.ndarray) -> np.ndarray:
    # LevelA/SPICE canonical -> Principia raw = [+Z, -X, +Y]
    x, y, z = v
    return np.array([z, -x, y], dtype=float)


def raw_to_levela(v: np.ndarray) -> np.ndarray:
    # Principia raw -> LevelA/SPICE canonical = [-Y, +Z, +X]
    x, y, z = v
    return np.array([-y, z, x], dtype=float)


def rtn_basis(r_rel: np.ndarray, v_rel: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    R = unit(r_rel)
    H = np.cross(r_rel, v_rel)
    N = unit(H)
    T = unit(np.cross(N, R))
    return R, T, N


def tangent_angle_basis(r_rel: np.ndarray, v_rel: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Returns R, T, N, with T nearly prograde. Good for departure burns in parking orbit.
    return rtn_basis(r_rel, v_rel)


def vector_from_tangent_angles(norm_m_s: float, yaw_rad: float, pitch_rad: float, R: np.ndarray, T: np.ndarray, N: np.ndarray) -> np.ndarray:
    # yaw rotates in T/R plane; pitch goes toward N.
    # yaw=0, pitch=0 => prograde/tangential.
    c = math.cos(pitch_rad)
    return float(norm_m_s) * (c * math.cos(yaw_rad) * T + c * math.sin(yaw_rad) * R + math.sin(pitch_rad) * N)

def orthonormal_about(axis: np.ndarray, hint1: np.ndarray, hint2: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return forward,u,v with u/v perpendicular to forward.

    `hint1` is projected into the transverse plane so yaw/pitch are stable and
    repeatable near the Lambert/hyperbolic departure direction.
    """
    f = unit(axis)
    h = np.array(hint1, dtype=float)
    u = h - float(np.dot(h, f)) * f
    if norm(u) < 1e-9 and hint2 is not None:
        h = np.array(hint2, dtype=float)
        u = h - float(np.dot(h, f)) * f
    if norm(u) < 1e-9:
        # Last-resort deterministic axis.
        h = np.array([1.0, 0.0, 0.0])
        if abs(float(np.dot(h, f))) > 0.9:
            h = np.array([0.0, 1.0, 0.0])
        u = h - float(np.dot(h, f)) * f
    u = unit(u)
    v = unit(np.cross(f, u))
    return f, u, v


def vector_from_axis_angles(norm_m_s: float, yaw_rad: float, pitch_rad: float, axis: np.ndarray, hint1: np.ndarray, hint2: np.ndarray | None = None) -> np.ndarray:
    """Small-angle parameterization around a nominal inertial axis.

    yaw=0,pitch=0 is exactly `axis`. This is the key difference from the older
    test, which centered the burn around local prograde and could miss the
    patched-conic/Lambert asymptote.
    """
    f, u, v = orthonormal_about(axis, hint1, hint2)
    c = math.cos(pitch_rad)
    return float(norm_m_s) * (c * math.cos(yaw_rad) * f + c * math.sin(yaw_rad) * v + math.sin(pitch_rad) * u)


def hyperbolic_rel_velocity_for_vinf(rel_r: np.ndarray, vinf_vec: np.ndarray, mu: float) -> np.ndarray:
    """Two-body velocity at rel_r whose outgoing asymptote is vinf_vec.

    This constructs a local osculating escape hyperbola around the departure
    body. It is only used to define the *nominal direction* of burn1. Principia
    still evaluates the candidate in full N-body.
    """
    r = norm(rel_r)
    vinf = norm(vinf_vec)
    if r <= 0 or vinf <= 0 or mu <= 0:
        raise ValueError("bad hyperbolic inputs")
    R = unit(rel_r)
    S = unit(vinf_vec)
    cpsi = max(-1.0, min(1.0, float(np.dot(R, S))))
    psi = math.acos(cpsi)
    # If the current radius is almost exactly on the asymptote direction, the
    # plane is ill-conditioned. Nudge via a deterministic transverse axis.
    if abs(math.sin(psi)) < 1e-8:
        raise ValueError("R and vinf asymptote nearly collinear")
    k = r * vinf * vinf / mu
    A = 1.0 - math.cos(psi)
    B = math.sin(psi)
    # u = sqrt(e^2 - 1)
    disc = k * k * B * B + 4.0 * k * A
    u = 0.5 * (k * B + math.sqrt(max(0.0, disc)))
    if u <= 0:
        raise ValueError("invalid hyperbola u")
    e = math.sqrt(1.0 + u * u)
    nu_inf = math.acos(-1.0 / e)
    nu = nu_inf - psi
    # Solve [R S] = [p q] [[cos nu, cos nu_inf], [sin nu, sin nu_inf]]
    M = np.array([[math.cos(nu), math.cos(nu_inf)], [math.sin(nu), math.sin(nu_inf)]], dtype=float)
    Minv = np.linalg.inv(M)
    P_Q = np.column_stack([R, S]) @ Minv
    p_hat = unit(P_Q[:, 0])
    q_hat = unit(P_Q[:, 1])
    # Re-orthogonalize q in case of small numerical drift.
    q_hat = unit(q_hat - float(np.dot(q_hat, p_hat)) * p_hat)
    h = mu / vinf * u
    vr = mu / h * e * math.sin(nu)
    vt = mu / h * (1.0 + e * math.cos(nu))
    transverse = -math.sin(nu) * p_hat + math.cos(nu) * q_hat
    return vr * R + vt * transverse


# ---- v0.7 parking-state provider -------------------------------------------------
# The earlier v0.5/v0.6 builds used the Principia server to coast the vessel from
# the captured live state to UT1 before the first burn. In low Kerbin orbit this
# produced non-parking preburn states (huge radial velocity after a few minutes),
# letting the optimizer exploit fake high-altitude/radial states. v0.7 separates
# the parking-orbit model from the N-body transfer: preburn can be generated by a
# local two-body propagator around the departure body, then the post-burn arc is
# handed to Principia.

def stumpff_c(z: float) -> float:
    if z > 1e-8:
        s = math.sqrt(z)
        return (1.0 - math.cos(s)) / z
    if z < -1e-8:
        s = math.sqrt(-z)
        return (math.cosh(s) - 1.0) / (-z)
    return 0.5 - z / 24.0 + z * z / 720.0


def stumpff_s(z: float) -> float:
    if z > 1e-8:
        s = math.sqrt(z)
        return (s - math.sin(s)) / (s * s * s)
    if z < -1e-8:
        s = math.sqrt(-z)
        return (math.sinh(s) - s) / (s * s * s)
    return 1.0 / 6.0 - z / 120.0 + z * z / 5040.0


def propagate_twobody_universal(r0: np.ndarray, v0: np.ndarray, dt: float, mu: float) -> tuple[np.ndarray, np.ndarray]:
    """Propagate r0/v0 by dt under a two-body point-mass model.

    Universal-variable solver, valid for bound and escape states. This is used
    only for the *parking coast before burn1*. Principia remains the truth model
    after burn1.
    """
    dt = float(dt)
    if abs(dt) < 1e-12:
        return np.array(r0, dtype=float).copy(), np.array(v0, dtype=float).copy()
    r0 = np.array(r0, dtype=float)
    v0 = np.array(v0, dtype=float)
    r0n = norm(r0)
    v0n = norm(v0)
    if r0n <= 0 or mu <= 0:
        raise ValueError("bad two-body initial state")
    sqrt_mu = math.sqrt(mu)
    vr0 = float(np.dot(r0, v0)) / r0n
    alpha = 2.0 / r0n - v0n * v0n / mu
    # Robust initial guess. For near-circular LKO alpha > 0.
    if abs(alpha) > 1e-12:
        chi = sqrt_mu * abs(alpha) * dt
    else:
        chi = sqrt_mu * dt / r0n
    if not math.isfinite(chi) or abs(chi) < 1e-12:
        chi = math.copysign(math.sqrt(mu) * abs(dt) / r0n, dt)

    for _ in range(50):
        z = alpha * chi * chi
        C = stumpff_c(z)
        S = stumpff_s(z)
        F = (r0n * vr0 / sqrt_mu) * chi * chi * C + (1.0 - alpha * r0n) * chi**3 * S + r0n * chi - sqrt_mu * dt
        dF = (r0n * vr0 / sqrt_mu) * chi * (1.0 - z * S) + (1.0 - alpha * r0n) * chi * chi * C + r0n
        if abs(dF) < 1e-12:
            break
        step = F / dF
        chi -= step
        if abs(step) < 1e-8:
            break
    z = alpha * chi * chi
    C = stumpff_c(z)
    S = stumpff_s(z)
    f = 1.0 - (chi * chi / r0n) * C
    g = dt - (chi**3 / sqrt_mu) * S
    r = f * r0 + g * v0
    rn = norm(r)
    if rn <= 0:
        raise ValueError("two-body propagation produced zero radius")
    fdot = (sqrt_mu / (rn * r0n)) * (alpha * chi**3 * S - chi)
    gdot = 1.0 - (chi * chi / rn) * C
    v = fdot * r0 + gdot * v0
    return r, v


def parking_preburn_state(cfg: dict[str, Any], tb1: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    """Return rel_r, rel_v, abs_r, abs_v for the pre-burn parking state."""
    live_t = float(cfg["live_t_s"])
    live_r = np.array(cfg["live_r_raw_m"], dtype=float)
    live_v = np.array(cfg["live_v_raw_m_s"], dtype=float)
    dep_body = str(cfg["dep_body"])
    center = str(cfg["center"])
    frame = str(cfg["frame"])
    dep_r_live, dep_v_live = body_state_raw(dep_body, live_t, center, frame)
    rel_r_live = live_r - dep_r_live
    rel_v_live = live_v - dep_v_live
    source = str(cfg.get("preburn_source", "twobody_parking"))
    if source == "twobody_parking":
        rel_r, rel_v = propagate_twobody_universal(rel_r_live, rel_v_live, float(tb1) - live_t, float(cfg["mu_dep"]))
        dep_r, dep_v = body_state_raw(dep_body, tb1, center, frame)
        return rel_r, rel_v, dep_r + rel_r, dep_v + rel_v, source
    raise ValueError(f"unsupported preburn source for parking_preburn_state: {source}")


def two_body_energy(rel_r: np.ndarray, rel_v: np.ndarray, mu: float) -> float:
    return 0.5 * norm(rel_v) ** 2 - mu / norm(rel_r)


def radial_velocity(rel_r: np.ndarray, rel_v: np.ndarray) -> float:
    return float(np.dot(rel_v, unit(rel_r)))


def angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = norm(a), norm(b)
    if na <= 0 or nb <= 0:
        return math.nan
    c = float(np.dot(a, b) / (na * nb))
    c = max(-1.0, min(1.0, c))
    return math.degrees(math.acos(c))


def max_turn_deg(vinf_m_s: float, rp_km: float, mu_m3_s2: float) -> float:
    rp_m = float(rp_km) * 1000.0
    if vinf_m_s <= 0 or rp_m <= 0 or mu_m3_s2 <= 0:
        return math.nan
    arg = 1.0 / (rp_m * vinf_m_s * vinf_m_s / mu_m3_s2 + 1.0)
    arg = max(-1.0, min(1.0, arg))
    return math.degrees(2.0 * math.asin(arg))


def maybe_get(d: dict[str, Any], keys: list[str]) -> Any:
    for k in keys:
        if k in d and d[k] not in (None, ''):
            return d[k]
    return None


def find_body_record(obj: Any, name: str) -> dict[str, Any] | None:
    lname = str(name).lower()
    if isinstance(obj, dict):
        for k, v in obj.items():
            if str(k).lower() == lname and isinstance(v, dict):
                return v
        nm = obj.get('name') or obj.get('body') or obj.get('id') or obj.get('spice_name')
        if nm is not None and str(nm).lower() == lname:
            return obj
        for key in ('bodies', 'items', 'catalog', 'body_catalog'):
            if key in obj:
                rec = find_body_record(obj[key], name)
                if rec is not None:
                    return rec
        for v in obj.values():
            rec = find_body_record(v, name)
            if rec is not None:
                return rec
    elif isinstance(obj, list):
        for v in obj:
            rec = find_body_record(v, name)
            if rec is not None:
                return rec
    return None


def load_radius_km(body: str, body_catalog: Path | None, override: float | None) -> float:
    if override is not None:
        return float(override)
    if body_catalog is None:
        return math.nan
    obj = json.loads(body_catalog.read_text())
    rec = find_body_record(obj, body)
    if rec is None:
        return math.nan
    km = maybe_get(rec, ['radius_km','mean_radius_km','equatorial_radius_km','body_radius_km','radius','mean_radius','equatorial_radius'])
    if km is not None:
        km = float(km)
        return km / 1000.0 if km > 1e5 else km
    m = maybe_get(rec, ['radius_m','mean_radius_m','equatorial_radius_m'])
    if m is not None:
        return float(m) / 1000.0
    return math.nan


def safe_float(x: Any, default: float = math.nan) -> float:
    try:
        if x is None or x == "":
            return default
        v = float(x)
        return v if math.isfinite(v) else default
    except Exception:
        return default


def parse_csv_floats(s: str | None) -> list[float]:
    if not s:
        return []
    return [float(x.strip()) for x in str(s).replace(";", ",").split(",") if x.strip()]


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def read_leg_row(path: Path, leg: int) -> dict[str, str]:
    for r in read_csv_rows(path):
        if int(float(r.get("leg", "nan"))) == int(leg):
            return r
    raise SystemExit(f"[FAIL] leg {leg} not found in {path}")


def arr(row: dict[str, str], *names: str) -> np.ndarray:
    return np.array([float(row[n]) for n in names], dtype=float)


def row_vec3(row: dict[str, str], prefix: str) -> np.ndarray:
    return arr(row, f"{prefix}_x_raw_m", f"{prefix}_y_raw_m", f"{prefix}_z_raw_m")


def row_vel3(row: dict[str, str], prefix: str) -> np.ndarray:
    return arr(row, f"{prefix}_vx_raw_m_s", f"{prefix}_vy_raw_m_s", f"{prefix}_vz_raw_m_s")


def leg_dv_raw(row: dict[str, str]) -> np.ndarray:
    if all(k in row and row[k] != "" for k in ["dvx_m_s", "dvy_m_s", "dvz_m_s"]):
        return arr(row, "dvx_m_s", "dvy_m_s", "dvz_m_s")
    return np.zeros(3)


def read_candidate_row(path: Path, rank: int) -> dict[str, str]:
    for r in read_csv_rows(path):
        if int(float(r.get("rank", "nan"))) == int(rank):
            return r
    raise SystemExit(f"[FAIL] rank {rank} not found in {path}")


def body_state_raw(body: str, t_s: float, center: str = "SUN", frame: str = "J2000") -> tuple[np.ndarray, np.ndarray]:
    st, _ = spice.spkezr(body, float(t_s), frame, "NONE", center)
    r_levela = np.array([1000.0 * st[0], 1000.0 * st[1], 1000.0 * st[2]], dtype=float)
    v_levela = np.array([1000.0 * st[3], 1000.0 * st[4], 1000.0 * st[5]], dtype=float)
    return levela_to_raw(r_levela), levela_to_raw(v_levela)


def body_mu_m3_s2(body: str) -> float:
    _, vals = spice.bodvrd(body, "GM", 1)
    return float(vals[0]) * 1.0e9


def load_radius_from_catalog(path: Path | None, body: str) -> float | None:
    if not path or not path.exists():
        return None
    obj = json.loads(path.read_text())
    bodies = obj.get("bodies", obj if isinstance(obj, list) else [])
    if isinstance(bodies, dict):
        iterable = bodies.values()
    else:
        iterable = bodies
    for b in iterable:
        name = str(b.get("name", b.get("body", ""))).upper()
        if name == body.upper():
            for key in ["radius_km", "mean_radius_km", "equatorial_radius_km"]:
                if key in b:
                    return float(b[key])
            for key in ["radius_m", "mean_radius_m", "equatorial_radius_m"]:
                if key in b:
                    return float(b[key]) / 1000.0
    return None


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

    def close(self):
        if self.proc.poll() is None:
            try:
                if self.proc.stdin:
                    self.proc.stdin.write("QUIT\n")
                    self.proc.stdin.flush()
                if self.proc.stdout:
                    self.proc.stdout.readline()
            except Exception:
                pass
            try:
                self.proc.terminate()
            except Exception:
                pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
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

    def propn(self, req_id: str, t0_s: float, t1_s: float, r0_m: np.ndarray, v0_m_s: np.ndarray, impulses: list[tuple[float, np.ndarray]]) -> PropResult:
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
                bt = float(parts[idx]); idx += 1
                r = np.array([float(parts[idx]), float(parts[idx + 1]), float(parts[idx + 2])], dtype=float); idx += 3
                vb = np.array([float(parts[idx]), float(parts[idx + 1]), float(parts[idx + 2])], dtype=float); idx += 3
                va = np.array([float(parts[idx]), float(parts[idx + 1]), float(parts[idx + 2])], dtype=float); idx += 3
                burns.append(BurnSnapshot(bt, r, vb, va))
            fr = np.array([float(parts[idx]), float(parts[idx + 1]), float(parts[idx + 2])], dtype=float); idx += 3
            fv = np.array([float(parts[idx]), float(parts[idx + 1]), float(parts[idx + 2])], dtype=float)
            return PropResult("ok", "", rid, out_t0, out_t1, burns, fr, fv)
        except Exception as e:
            return PropResult(status="error", message=f"parse OKN failed: {e}: {line}", id=req_id)


class Leg1TrustRegionUDP:
    def __init__(self, cfg: dict[str, Any]):
        self.cfg = cfg
        self._srv: ServerSession | None = None
        self.eval_count = 0
        self.best: dict[str, Any] | None = None

    def __getstate__(self):
        # PyGMO deep-copies UDPs. A live subprocess has locks/file handles and
        # must not be copied. Each copied UDP lazily opens its own server.
        return {"cfg": self.cfg, "eval_count": self.eval_count, "best": self.best}

    def __setstate__(self, state):
        self.cfg = state["cfg"]
        self.eval_count = state.get("eval_count", 0)
        self.best = state.get("best")
        self._srv = None

    def get_bounds(self):
        c = self.cfg
        deg = math.pi / 180.0
        lb = [
            -float(c["ut1_trust_s"]),
            -float(c["dv1_trust_m_s"]),  # delta from local Lambert/hyperbola nominal, not absolute dv
            -float(c["burn1_yaw_trust_deg"]) * deg,
            -float(c["burn1_pitch_trust_deg"]) * deg,
            -float(c["ut2_trust_s"]),
            0.0,
            -math.pi,
            -0.5 * math.pi,
            -float(c["arrival_trust_s"]),
        ]
        ub = [
            float(c["ut1_trust_s"]),
            float(c["dv1_trust_m_s"]),   # delta from local Lambert/hyperbola nominal, not absolute dv
            float(c["burn1_yaw_trust_deg"]) * deg,
            float(c["burn1_pitch_trust_deg"]) * deg,
            float(c["ut2_trust_s"]),
            float(c["dsm_max_m_s"]),
            math.pi,
            0.5 * math.pi,
            float(c["arrival_trust_s"]),
        ]
        return (lb, ub)

    def _server(self) -> ServerSession:
        if self._srv is None:
            self._srv = ServerSession(str(self.cfg["server"]), str(self.cfg["plugin_b64"]), bool(self.cfg.get("quiet_stderr", False)))
            if not self._srv.ping():
                raise RuntimeError("server ping failed")
        return self._srv

    def _penalty(self, amount: float, base: float = 1e8) -> list[float]:
        return [float(base + max(0.0, amount))]

    def _record_best(self, row: dict[str, Any], fitness: float) -> None:
        row = dict(row)
        row["fitness"] = float(fitness)
        row["eval_count"] = int(self.eval_count)
        if self.best is None or fitness < float(self.best.get("fitness", math.inf)):
            self.best = row

    def fitness(self, x: list[float]) -> list[float]:
        self.eval_count += 1
        c = self.cfg
        try:
            live_t = float(c["live_t_s"])
            live_r = np.array(c["live_r_raw_m"], dtype=float)
            live_v = np.array(c["live_v_raw_m_s"], dtype=float)
            tb1 = float(c["ut1_nominal_s"]) + float(x[0])
            dv1_delta = float(x[1])
            yaw1 = float(x[2]); pitch1 = float(x[3])
            t_dsm = float(c["ut2_nominal_s"]) + float(x[4])
            dsm_norm = float(x[5])
            yaw2 = float(x[6]); pitch2 = float(x[7])
            t_arr = float(c["arrival_nominal_s"]) + float(x[8])
            if tb1 <= live_t + float(c["min_burn_after_live_s"]):
                return self._penalty(1e7 + live_t - tb1)
            if t_dsm <= tb1 + float(c["min_dsm_after_burn_s"]):
                return self._penalty(2e7 + tb1 - t_dsm)
            if t_arr <= t_dsm + float(c["min_arrival_after_dsm_s"]):
                return self._penalty(3e7 + t_dsm - t_arr)

            srv = self._server()
            # v0.7: do NOT ask Principia to coast LKO to the first burn. The
            # previous versions showed that this produced non-parking states with
            # huge radial velocity. Instead, use a parking-state provider before
            # burn1 and hand the post-burn transfer to Principia.
            try:
                rel_r1, rel_v1, burn_abs_r, burn_abs_v, preburn_source = parking_preburn_state(c, tb1)
            except Exception:
                return self._penalty(4.05e7)
            dep_r1, dep_v1 = body_state_raw(str(c["dep_body"]), tb1, str(c["center"]), str(c["frame"]))
            burn_distance_km = norm(rel_r1) / 1000.0
            burn_radial_v_m_s = radial_velocity(rel_r1, rel_v1)
            preburn_energy = two_body_energy(rel_r1, rel_v1, float(c["mu_dep"]))
            if bool(c.get("require_preburn_bound", True)) and preburn_energy >= float(c.get("max_preburn_energy_m2_s2", 0.0)):
                return self._penalty(4.10e7 + max(0.0, preburn_energy - float(c.get("max_preburn_energy_m2_s2", 0.0))) / 10.0)
            if abs(burn_radial_v_m_s) > float(c.get("parking_max_abs_radial_v_m_s", 1.0e99)):
                return self._penalty(4.12e7 + (abs(burn_radial_v_m_s) - float(c.get("parking_max_abs_radial_v_m_s", 1.0e99))) * 1000.0)
            if burn_distance_km < float(c.get("burn_distance_min_from_dep_km", 0.0)):
                return self._penalty(4.15e7 + (float(c.get("burn_distance_min_from_dep_km", 0.0)) - burn_distance_km) * 1000.0)
            if burn_distance_km > float(c.get("burn_distance_max_from_dep_km", 1.0e99)):
                return self._penalty(4.16e7 + (burn_distance_km - float(c.get("burn_distance_max_from_dep_km", 1.0e99))) * 1000.0)
            R1, T1, N1 = tangent_angle_basis(rel_r1, rel_v1)
            # Center the departure burn around the patched-conic/Lambert v∞,
            # not around local prograde. This prevents the optimizer from
            # exploring nonsense transfers that satisfy only a loose escape gate.
            nominal_dv1 = None
            vinf_ref = c.get("candidate_vinf_ref_raw_m_s")
            if vinf_ref is not None:
                try:
                    vinf_ref = np.array(vinf_ref, dtype=float)
                    vrel_after_nom = hyperbolic_rel_velocity_for_vinf(rel_r1, vinf_ref, float(c["mu_dep"]))
                    nominal_dv1 = (dep_v1 + vrel_after_nom) - burn_abs_v
                except Exception:
                    nominal_dv1 = None
            if nominal_dv1 is None or norm(nominal_dv1) < 1e-6:
                nominal_dv1 = float(c["dv1_nominal_m_s"]) * T1
            # Optional hard anchoring: keep the *direction* from the local Lambert/hyperbola
            # construction, but force the magnitude to the patched-conic/global nominal.
            # This prevents the optimizer from exploiting high-altitude/coast epochs where
            # the local escape burn collapses to ~900 m/s even though the intended parking-
            # orbit transfer is ~2 km/s.
            if str(c.get("dv1_nominal_mode", "local")) == "global":
                nominal_dv1 = unit(nominal_dv1) * float(c["dv1_nominal_m_s"])
            nominal_dv1_norm = norm(nominal_dv1)
            if nominal_dv1_norm < float(c.get("local_nominal_dv_min_m_s", 0.0)):
                return self._penalty(4.5e7 + (float(c.get("local_nominal_dv_min_m_s", 0.0)) - nominal_dv1_norm) * 1000.0)
            if nominal_dv1_norm > float(c.get("local_nominal_dv_max_m_s", 1.0e9)):
                return self._penalty(4.6e7 + (nominal_dv1_norm - float(c.get("local_nominal_dv_max_m_s", 1.0e9))) * 1000.0)
            dv1_norm = max(1.0, nominal_dv1_norm + dv1_delta)
            if dv1_norm < float(c.get("min_burn_dv_m_s", 0.0)):
                return self._penalty(4.80e7 + (float(c.get("min_burn_dv_m_s", 0.0)) - dv1_norm) * 1000.0)
            if dv1_norm > float(c.get("max_burn_dv_m_s", 1.0e99)):
                return self._penalty(4.81e7 + (dv1_norm - float(c.get("max_burn_dv_m_s", 1.0e99))) * 1000.0)
            dv1 = vector_from_axis_angles(dv1_norm, yaw1, pitch1, unit(nominal_dv1), N1, R1)
            angle_from_nominal_deg_tmp = math.degrees(math.acos(max(-1.0, min(1.0, float(np.dot(unit(dv1), unit(nominal_dv1)))))))
            if angle_from_nominal_deg_tmp > float(c.get("burn1_max_angle_from_nominal_deg", 180.0)):
                return self._penalty(4.7e7 + (angle_from_nominal_deg_tmp - float(c.get("burn1_max_angle_from_nominal_deg", 180.0))) * 1e6)
            v_after1 = burn_abs_v + dv1
            eps1 = two_body_energy(rel_r1, v_after1 - dep_v1, float(c["mu_dep"]))
            if eps1 <= float(c["min_escape_energy_m2_s2"]):
                return self._penalty(5e7 + abs(eps1) / 10.0)

            # Propagate to DSM after burn1. Since preburn is supplied by
            # the parking provider, the Principia truth arc starts at tb1 with the
            # post-burn state.
            post_burn_v = burn_abs_v + dv1
            to_dsm = srv.propn(f"todsm_{os.getpid()}_{self.eval_count}", tb1, t_dsm, burn_abs_r, post_burn_v, [])
            if to_dsm.status != "ok" or to_dsm.final_r_m is None or to_dsm.final_v_m_s is None:
                return self._penalty(6e7)
            dep_rd, dep_vd = body_state_raw(str(c["dep_body"]), t_dsm, str(c["center"]), str(c["frame"]))
            rel_rd = to_dsm.final_r_m - dep_rd
            rel_vd = to_dsm.final_v_m_s - dep_vd
            d_dsm_km = norm(rel_rd) / 1000.0
            vr_dsm = radial_velocity(rel_rd, rel_vd)
            if d_dsm_km < float(c["min_dsm_distance_from_dep_km"]):
                return self._penalty(7e7 + (float(c["min_dsm_distance_from_dep_km"]) - d_dsm_km) * 1000.0)
            if vr_dsm < float(c["min_dsm_radial_v_m_s"]):
                return self._penalty(8e7 + abs(vr_dsm) * 1000.0)

            # DSM basis: local RTN around the departure body at DSM, so DSM cannot secretly be a near-Kerbin injection.
            try:
                Rd, Td, Nd = rtn_basis(rel_rd, rel_vd)
            except Exception:
                Rd, Td, Nd = rtn_basis(to_dsm.final_r_m, to_dsm.final_v_m_s)
            dsm = vector_from_tangent_angles(dsm_norm, yaw2, pitch2, Rd, Td, Nd)

            # Propagate to arrival.
            res = srv.propn(f"fit_{os.getpid()}_{self.eval_count}", tb1, t_arr, burn_abs_r, post_burn_v, [(t_dsm, dsm)])
            if res.status != "ok" or res.final_r_m is None or res.final_v_m_s is None:
                return self._penalty(9e7)
            target_r, target_v = body_state_raw(str(c["arr_body"]), t_arr, str(c["center"]), str(c["frame"]))
            pos_err_km = norm(res.final_r_m - target_r) / 1000.0
            vinf_in = res.final_v_m_s - target_v
            vinf_in_m_s = norm(vinf_in)

            score = pos_err_km / float(c["pos_scale_km"])
            score += dsm_norm / float(c["dsm_scale_m_s"]) * float(c["dsm_weight"])
            score += abs(float(x[8])) / 86400.0 * float(c["arrival_time_weight"])
            score += abs(float(x[0])) / max(float(c["ut1_trust_s"]), 1.0) * float(c["ut1_weight"])
            # Keep burn1 close to nominal and mostly prograde.
            score += abs(dv1_delta) / max(float(c["dv1_trust_m_s"]), 1.0) * float(c["dv1_anchor_weight"])
            score += abs(math.sin(yaw1)) * float(c["burn1_yaw_weight"])
            score += abs(math.sin(pitch1)) * float(c["burn1_pitch_weight"])

            # Optional flyby compatibility with next leg's outgoing v∞.
            # v0.5: penalize *turn deficit* instead of only raw angle. This
            # lets the optimizer keep any incoming vector that a physical flyby
            # can turn into the next leg.
            if c.get("vinf_out_ref_raw_m_s") is not None:
                vinf_out_ref = np.array(c["vinf_out_ref_raw_m_s"], dtype=float)
                vinf_out_mag = norm(vinf_out_ref)
                mag_mis = abs(vinf_in_m_s - vinf_out_mag)
                angle_req_deg = angle_deg(vinf_in, vinf_out_ref) if vinf_in_m_s > 1e-9 and vinf_out_mag > 1e-9 else 180.0
                vinf_turn_m_s = max(vinf_in_m_s, vinf_out_mag)
                rp_min_km = float(c.get("arr_body_radius_km", math.nan)) + float(c.get("flyby_safe_altitude_km", 0.0))
                max_turn = max_turn_deg(vinf_turn_m_s, rp_min_km, float(c.get("mu_arr", math.nan)))
                turn_deficit_deg = max(0.0, angle_req_deg - max_turn) if math.isfinite(max_turn) else 0.0
                score += mag_mis / float(c["vinf_mag_scale_m_s"]) * float(c["vinf_mag_weight"])
                score += angle_req_deg / float(c["vinf_angle_scale_deg"]) * float(c["vinf_angle_weight"])
                score += turn_deficit_deg / float(c["flyby_turn_deficit_scale_deg"]) * float(c["flyby_turn_weight"])
                score += max(0.0, mag_mis - float(c["flyby_mag_free_m_s"])) / float(c["flyby_powered_scale_m_s"]) * float(c["flyby_powered_weight"])
            else:
                mag_mis = math.nan
                angle_req_deg = math.nan
                max_turn = math.nan
                turn_deficit_deg = math.nan
                rp_min_km = math.nan

            row = {
                "tb1_s": tb1,
                "t_dsm_s": t_dsm,
                "t_arr_s": t_arr,
                "dt_burn1_s": float(x[0]),
                "dt_dsm_s": float(x[4]),
                "dt_arrival_s": float(x[8]),
                "dv1_norm_m_s": dv1_norm,
                "dv1_delta_m_s": dv1_delta,
                "dv1_yaw_deg": math.degrees(yaw1),
                "dv1_pitch_deg": math.degrees(pitch1),
                "dv1_raw_m_s": dv1.tolist(),
                "dv1_levela_m_s": raw_to_levela(dv1).tolist(),
                "nominal_dv1_raw_m_s": nominal_dv1.tolist(),
                "nominal_dv1_levela_m_s": raw_to_levela(nominal_dv1).tolist(),
                "nominal_dv1_norm_m_s": nominal_dv1_norm,
                "angle_from_nominal_dv1_deg": angle_from_nominal_deg_tmp,
                "dsm_norm_m_s": dsm_norm,
                "dsm_yaw_deg": math.degrees(yaw2),
                "dsm_pitch_deg": math.degrees(pitch2),
                "dsm_raw_m_s": dsm.tolist(),
                "dsm_levela_m_s": raw_to_levela(dsm).tolist(),
                "preburn_source": preburn_source,
                "burn_distance_from_dep_km": burn_distance_km,
                "burn_radial_v_m_s": burn_radial_v_m_s,
                "preburn_energy_m2_s2": preburn_energy,
                "preburn_r_raw_m": burn_abs_r.tolist(),
                "preburn_v_raw_m_s": burn_abs_v.tolist(),
                "postburn_v_raw_m_s": post_burn_v.tolist(),
                "escape_energy_m2_s2": eps1,
                "dsm_distance_from_dep_km": d_dsm_km,
                "dsm_radial_v_m_s": vr_dsm,
                "final_pos_err_km": pos_err_km,
                "arrival_vinf_in_m_s": vinf_in_m_s,
                "final_v_raw_m_s": res.final_v_m_s.tolist(),
                "target_v_raw_m_s": target_v.tolist(),
                "dv0_norm_m_s": dv1_norm,
                "vinf_mag_mismatch_m_s": mag_mis,
                "vinf_turn_required_deg": angle_req_deg,
                "vinf_max_turn_deg": max_turn,
                "vinf_turn_deficit_deg": turn_deficit_deg,
                "flyby_rp_min_km": rp_min_km,
                "fitness": float(score),
            }
            self._record_best(row, score)
            return [float(score)]
        except Exception as e:
            # Keep optimizer alive.
            return self._penalty(1e9)


def infer_nominals(args: argparse.Namespace, live: dict[str, Any], leg_row: dict[str, str], candidate_row: dict[str, str] | None) -> dict[str, Any]:
    live_t = float(live["ut_s"])
    out: dict[str, Any] = {}
    if args.ut1_nominal_s is not None:
        out["ut1_nominal_s"] = float(args.ut1_nominal_s)
    elif candidate_row is not None and candidate_row.get(f"event{args.leg-1}_et_s"):
        out["ut1_nominal_s"] = float(candidate_row[f"event{args.leg-1}_et_s"])
    else:
        out["ut1_nominal_s"] = max(live_t + 600.0, float(leg_row.get("t_start_s", live_t + 600.0)) - 3600.0)

    if args.arrival_nominal_s is not None:
        out["arrival_nominal_s"] = float(args.arrival_nominal_s)
    elif candidate_row is not None and candidate_row.get(f"event{args.leg}_et_s"):
        out["arrival_nominal_s"] = float(candidate_row[f"event{args.leg}_et_s"])
    else:
        out["arrival_nominal_s"] = float(leg_row["t_end_s"])

    if args.ut2_nominal_s is not None:
        out["ut2_nominal_s"] = float(args.ut2_nominal_s)
    else:
        out["ut2_nominal_s"] = out["ut1_nominal_s"] + float(args.dsm_nominal_fraction) * (out["arrival_nominal_s"] - out["ut1_nominal_s"])

    # Reference v∞ from candidate seed if available.
    vinf_ref_raw = None
    if candidate_row is not None:
        keys = [f"leg{args.leg}_vdep_x_km_s", f"leg{args.leg}_vdep_y_km_s", f"leg{args.leg}_vdep_z_km_s"]
        if all(candidate_row.get(k, "") != "" for k in keys):
            vdep_levela = np.array([float(candidate_row[k]) * 1000.0 for k in keys], dtype=float)
            _, dep_v_raw = body_state_raw(args.dep_body, out["ut1_nominal_s"], args.center, args.frame)
            dep_v_levela = raw_to_levela(dep_v_raw)
            vinf_ref_levela = vdep_levela - dep_v_levela
            vinf_ref_raw = levela_to_raw(vinf_ref_levela)
            out["candidate_vinf_ref_raw_m_s"] = vinf_ref_raw.tolist()
            out["candidate_vinf_ref_m_s"] = norm(vinf_ref_raw)

    if args.dv1_nominal_m_s is not None:
        out["dv1_nominal_m_s"] = float(args.dv1_nominal_m_s)
    else:
        # Estimate patched escape burn from v∞ magnitude and current parking orbit speed at nominal burn time.
        mu = body_mu_m3_s2(args.dep_body)
        with ServerSession(args.server, str(args.plugin_b64), args.quiet_stderr) as srv:
            pre = srv.propn("nominal_preburn", live_t, out["ut1_nominal_s"], np.array(live["r_raw_m"], dtype=float), np.array(live["v_raw_m_s"], dtype=float), [(out["ut1_nominal_s"], np.zeros(3))])
            if pre.status != "ok" or not pre.burns:
                raise SystemExit(f"[FAIL] could not propagate to nominal burn to infer dv1: {pre.status} {pre.message}")
            b = pre.burns[0]
            dep_r, dep_v = body_state_raw(args.dep_body, out["ut1_nominal_s"], args.center, args.frame)
            rel_r = b.r_m - dep_r
            rel_v = b.v_before_m_s - dep_v
            vinf_mag = norm(vinf_ref_raw) if vinf_ref_raw is not None else float(args.fallback_vinf_m_s)
            v_after = math.sqrt(max(0.0, vinf_mag * vinf_mag + 2.0 * mu / norm(rel_r)))
            out["dv1_nominal_m_s"] = max(0.0, v_after - norm(rel_v))
            out["inferred_preburn_distance_km"] = norm(rel_r) / 1000.0
            out["inferred_preburn_rel_speed_m_s"] = norm(rel_v)
            out["inferred_vinf_for_dv1_m_s"] = vinf_mag

    return out


def make_vinf_out_ref(args: argparse.Namespace, leg_out: int | None) -> list[float] | None:
    if not leg_out:
        return None
    row = read_leg_row(args.leg_optimizations, leg_out)
    t_ref = float(row["t_start_s"])
    start_v = row_vel3(row, "start")
    dv = leg_dv_raw(row) if args.next_leg_mode == "post_correction" else np.zeros(3)
    _, body_v = body_state_raw(args.arr_body, t_ref, args.center, args.frame)
    return (start_v + dv - body_v).tolist()


def jsonable(x: Any) -> Any:
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, dict):
        return {k: jsonable(v) for k, v in x.items()}
    if isinstance(x, list):
        return [jsonable(v) for v in x]
    return x


def build_event_preview(best: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": "mga_mission_event_v0_3_preview_only",
        "warning": "Do not push with insert_levela directly unless the FlightPlan adapter projects this inertial vector into Principia tangent/normal/binormal at burn time.",
        "event_type": "departure_burn_trust_region_result",
        "mode": "inertial_raw_delta_v",
        "initial_time": best["tb1_s"],
        "delta_v_raw_m_s": best["dv1_raw_m_s"],
        "delta_v_levela_m_s": best["dv1_levela_m_s"],
        "delta_v_norm_m_s": best["dv1_norm_m_s"],
        "is_inertially_fixed": True,
        "diagnostics": {
            "escape_energy_m2_s2": best.get("escape_energy_m2_s2"),
            "final_pos_err_km": best.get("final_pos_err_km"),
            "arrival_vinf_in_m_s": best.get("arrival_vinf_in_m_s"),
        },
        "dsm_preview": {
            "initial_time": best["t_dsm_s"],
            "delta_v_raw_m_s": best["dsm_raw_m_s"],
            "delta_v_levela_m_s": best["dsm_levela_m_s"],
            "delta_v_norm_m_s": best["dsm_norm_m_s"],
        },
    }


# ---- multiprocessing custom differential evolution driver v0.4 ----
import multiprocessing as mp
import random

_WORKER_UDP: Leg1TrustRegionUDP | None = None
_WORKER_CFG: dict[str, Any] | None = None


def _worker_init(cfg: dict[str, Any]) -> None:
    global _WORKER_UDP, _WORKER_CFG
    _WORKER_CFG = cfg
    # SPICE kernels must be loaded in every process.
    spice.kclear()
    spice.furnsh(str(cfg["tpc"]))
    spice.furnsh(str(cfg["bsp"]))
    _WORKER_UDP = Leg1TrustRegionUDP(cfg)


def _worker_eval(x: list[float]) -> tuple[float, list[float]]:
    global _WORKER_UDP
    if _WORKER_UDP is None:
        raise RuntimeError("worker UDP not initialized")
    try:
        f = float(_WORKER_UDP.fitness(list(map(float, x)))[0])
    except Exception:
        f = 1e12
    return f, list(map(float, x))


def _latin_random_population(lb: np.ndarray, ub: np.ndarray, size: int, rng: random.Random) -> np.ndarray:
    dim = len(lb)
    pop = np.empty((size, dim), dtype=float)
    for j in range(dim):
        vals = [(i + rng.random()) / size for i in range(size)]
        rng.shuffle(vals)
        pop[:, j] = lb[j] + np.array(vals) * (ub[j] - lb[j])
    return pop


def _load_warm_start(path: Path | None) -> list[float] | None:
    if not path:
        return None
    obj = json.loads(path.read_text())
    if isinstance(obj, dict):
        if "champion_x" in obj and isinstance(obj["champion_x"], list):
            return [float(x) for x in obj["champion_x"]]
        if "best" in obj and isinstance(obj["best"], dict):
            b = obj["best"]
            if "champion_x" in b and isinstance(b["champion_x"], list):
                return [float(x) for x in b["champion_x"]]
    raise SystemExit(f"[FAIL] could not find champion_x in warm-start json: {path}")


def _evaluate_many(xs: np.ndarray, cfg: dict[str, Any], pool, chunk_size: int) -> list[float]:
    items = [list(map(float, row)) for row in xs]
    if pool is None:
        global _WORKER_UDP
        if _WORKER_UDP is None:
            _worker_init(cfg)
        return [float(_worker_eval(x)[0]) for x in items]
    return [float(f) for f, _ in pool.imap(_worker_eval, items, chunksize=max(1, chunk_size))]


def _custom_de(cfg: dict[str, Any], bounds: tuple[list[float], list[float]], *,
               population: int, generations: int, seed: int, workers: int,
               chunk_size: int, F: float, CR: float,
               warm_start: list[float] | None = None,
               warm_start_jitter: float = 0.03) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    rng = random.Random(seed)
    lb = np.array(bounds[0], dtype=float)
    ub = np.array(bounds[1], dtype=float)
    dim = len(lb)
    pop_size = max(population, 4 * dim)
    pop = _latin_random_population(lb, ub, pop_size, rng)
    if warm_start is not None:
        w = np.clip(np.array(warm_start, dtype=float), lb, ub)
        pop[0, :] = w
        span = ub - lb
        for i in range(1, min(pop_size, 1 + dim * 2)):
            noise = np.array([rng.gauss(0.0, warm_start_jitter) for _ in range(dim)]) * span
            pop[i, :] = np.clip(w + noise, lb, ub)

    # v0.7: open the multiprocessing pool once for the whole DE run.
    # Earlier versions created a new process pool for every generation/evaluate
    # call, which dominated runtime and leaked server startup overhead.
    ctx = mp.get_context("spawn")
    pool = ctx.Pool(processes=workers, initializer=_worker_init, initargs=(cfg,)) if workers > 1 else None
    try:
        fitness = np.array(_evaluate_many(pop, cfg, pool, chunk_size), dtype=float)
        best_idx = int(np.argmin(fitness))
        print(f"   Gen:        Fevals:          Best:        Improved:      Accepted:")
        print(f"      0 {pop_size:14d} {fitness[best_idx]:14.6g} {0:14d} {0:14d}", flush=True)
        fevals = pop_size
        for gen in range(1, generations + 1):
            trials = np.empty_like(pop)
            for i in range(pop_size):
                choices = list(range(pop_size))
                choices.remove(i)
                a, b, c = rng.sample(choices, 3)
                mutant = pop[a] + F * (pop[b] - pop[c])
                mutant = np.clip(mutant, lb, ub)
                j_rand = rng.randrange(dim)
                trial = pop[i].copy()
                for j in range(dim):
                    if rng.random() < CR or j == j_rand:
                        trial[j] = mutant[j]
                trials[i] = np.clip(trial, lb, ub)
            trial_f = np.array(_evaluate_many(trials, cfg, pool, chunk_size), dtype=float)
            fevals += pop_size
            improved = trial_f < fitness
            accepted = int(np.sum(improved))
            pop[improved] = trials[improved]
            fitness[improved] = trial_f[improved]
            best_idx = int(np.argmin(fitness))
            if gen == 1 or gen % max(1, min(10, generations // 10 or 1)) == 0 or gen == generations:
                print(f"{gen:7d} {fevals:14d} {fitness[best_idx]:14.6g} {accepted:14d} {accepted:14d}", flush=True)
        best_idx = int(np.argmin(fitness))
        return pop[best_idx].copy(), float(fitness[best_idx]), pop, fitness
    finally:
        if pool is not None:
            pool.close()
            pool.join()

def main() -> int:
    ap = argparse.ArgumentParser(description="MP custom-DE leg solver with Lambert trust region, flyby gate, and parking-orbit burn gate.")
    ap.add_argument("--plugin-b64", type=Path, required=True)
    ap.add_argument("--server", default="principia_impulsive_particle_server")
    ap.add_argument("--live-state-json", type=Path, required=True)
    ap.add_argument("--leg-optimizations", type=Path, required=True)
    ap.add_argument("--candidate-seed", type=Path, default=None)
    ap.add_argument("--rank", type=int, default=12)
    ap.add_argument("--leg", type=int, default=1)
    ap.add_argument("--bsp", type=Path, required=True)
    ap.add_argument("--tpc", type=Path, required=True)
    ap.add_argument("--dep-body", default="KERBIN")
    ap.add_argument("--arr-body", default="EVE")
    ap.add_argument("--center", default="SUN")
    ap.add_argument("--frame", default="J2000")
    ap.add_argument("--preburn-source", choices=["twobody_parking"], default="twobody_parking",
                    help="Provider for the coast before burn1. v0.7 defaults to two-body parking instead of Principia LKO coast.")
    ap.add_argument("--parking-max-abs-radial-v-m-s", type=float, default=250.0,
                    help="Reject burn1 if the preburn parking provider has abs(radial velocity) above this.")
    ap.add_argument("--require-preburn-bound", action=argparse.BooleanOptionalAction, default=True,
                    help="Reject burn1 if preburn two-body energy around the departure body is not bound.")
    ap.add_argument("--max-preburn-energy-m2-s2", type=float, default=0.0)
    ap.add_argument("--body-catalog", type=Path, default=None)
    ap.add_argument("--arr-body-radius-km", type=float, default=None)
    ap.add_argument("--flyby-min-altitude-km", type=float, default=50.0)
    ap.add_argument("--flyby-atmosphere-margin-km", type=float, default=10.0)
    ap.add_argument("--flyby-next-leg", type=int, default=2)
    ap.add_argument("--next-leg-mode", choices=["pre_correction", "post_correction"], default="post_correction")

    ap.add_argument("--ut1-nominal-s", type=float, default=None)
    ap.add_argument("--dv1-nominal-m-s", type=float, default=None)
    ap.add_argument("--fallback-vinf-m-s", type=float, default=2500.0)
    ap.add_argument("--ut2-nominal-s", type=float, default=None)
    ap.add_argument("--arrival-nominal-s", type=float, default=None)
    ap.add_argument("--dsm-nominal-fraction", type=float, default=0.25)

    ap.add_argument("--ut1-trust-s", type=float, default=1800.0)
    ap.add_argument("--dv1-trust-m-s", type=float, default=150.0)
    ap.add_argument("--burn1-yaw-trust-deg", type=float, default=8.0)
    ap.add_argument("--burn1-pitch-trust-deg", type=float, default=8.0)
    ap.add_argument("--burn1-max-angle-from-nominal-deg", type=float, default=12.0)
    ap.add_argument("--local-nominal-dv-min-m-s", type=float, default=1200.0)
    ap.add_argument("--local-nominal-dv-max-m-s", type=float, default=2600.0)
    ap.add_argument("--dv1-nominal-mode", choices=["local", "global"], default="local",
                    help="local = use local Lambert/hyperbola burn magnitude; global = use --dv1-nominal-m-s as hard magnitude anchor while keeping local direction.")
    ap.add_argument("--burn-distance-min-from-dep-km", type=float, default=0.0,
                    help="Reject burn1 if preburn vessel distance from departure body is below this radius.")
    ap.add_argument("--burn-distance-max-from-dep-km", type=float, default=1.0e99,
                    help="Reject burn1 if preburn vessel distance from departure body is above this radius. Use this to force parking-orbit departures.")
    ap.add_argument("--min-burn-dv-m-s", type=float, default=0.0,
                    help="Reject burn1 magnitudes below this; useful to prevent sub-transfer burns.")
    ap.add_argument("--max-burn-dv-m-s", type=float, default=1.0e99,
                    help="Reject burn1 magnitudes above this.")
    ap.add_argument("--ut2-trust-days", type=float, default=5.0)
    ap.add_argument("--arrival-trust-days", type=float, default=7.0)
    ap.add_argument("--dsm-max-m-s", type=float, default=500.0)

    ap.add_argument("--min-burn-after-live-s", type=float, default=30.0)
    ap.add_argument("--min-dsm-after-burn-s", type=float, default=6*3600.0)
    ap.add_argument("--min-arrival-after-dsm-s", type=float, default=24*3600.0)
    ap.add_argument("--min-dsm-distance-from-dep-km", type=float, default=40000.0)
    ap.add_argument("--min-dsm-radial-v-m-s", type=float, default=0.0)
    ap.add_argument("--min-escape-energy-m2-s2", type=float, default=1.0)

    ap.add_argument("--pos-scale-km", type=float, default=100000.0)
    ap.add_argument("--dsm-scale-m-s", type=float, default=300.0)
    ap.add_argument("--dsm-weight", type=float, default=0.35)
    ap.add_argument("--arrival-time-weight", type=float, default=0.02)
    ap.add_argument("--ut1-weight", type=float, default=0.02)
    ap.add_argument("--dv1-anchor-weight", type=float, default=0.20)
    ap.add_argument("--burn1-yaw-weight", type=float, default=0.30)
    ap.add_argument("--burn1-pitch-weight", type=float, default=0.30)
    ap.add_argument("--vinf-mag-scale-m-s", type=float, default=500.0)
    ap.add_argument("--vinf-angle-scale-deg", type=float, default=20.0)
    ap.add_argument("--vinf-mag-weight", type=float, default=0.10)
    ap.add_argument("--vinf-angle-weight", type=float, default=0.10)
    ap.add_argument("--flyby-turn-deficit-scale-deg", type=float, default=1.0)
    ap.add_argument("--flyby-turn-weight", type=float, default=10.0)
    ap.add_argument("--flyby-mag-free-m-s", type=float, default=100.0)
    ap.add_argument("--flyby-powered-scale-m-s", type=float, default=100.0)
    ap.add_argument("--flyby-powered-weight", type=float, default=0.0)

    ap.add_argument("--population", type=int, default=64)
    ap.add_argument("--generations", type=int, default=80)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--chunk-size", type=int, default=1)
    ap.add_argument("--de-f", type=float, default=0.7)
    ap.add_argument("--de-cr", type=float, default=0.9)
    ap.add_argument("--warm-start-json", type=Path, default=None)
    ap.add_argument("--warm-start-jitter", type=float, default=0.03)
    ap.add_argument("--quiet-stderr", action="store_true")
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()

    spice.kclear()
    spice.furnsh(str(args.tpc))
    spice.furnsh(str(args.bsp))

    live = json.loads(args.live_state_json.read_text())
    live_t = float(live["ut_s"])
    leg_row = read_leg_row(args.leg_optimizations, args.leg)
    candidate_row = read_candidate_row(args.candidate_seed, args.rank) if args.candidate_seed else None
    nom = infer_nominals(args, live, leg_row, candidate_row)
    vinf_out = make_vinf_out_ref(args, args.flyby_next_leg if args.flyby_next_leg > 0 else None)

    cfg: dict[str, Any] = {
        "server": args.server,
        "plugin_b64": str(args.plugin_b64),
        "bsp": str(args.bsp),
        "tpc": str(args.tpc),
        "dep_body": args.dep_body,
        "arr_body": args.arr_body,
        "center": args.center,
        "frame": args.frame,
        "preburn_source": args.preburn_source,
        "parking_max_abs_radial_v_m_s": args.parking_max_abs_radial_v_m_s,
        "require_preburn_bound": args.require_preburn_bound,
        "max_preburn_energy_m2_s2": args.max_preburn_energy_m2_s2,
        "live_t_s": live_t,
        "live_r_raw_m": list(map(float, live["r_raw_m"])),
        "live_v_raw_m_s": list(map(float, live["v_raw_m_s"])),
        "mu_dep": body_mu_m3_s2(args.dep_body),
        "mu_arr": body_mu_m3_s2(args.arr_body),
        "arr_body_radius_km": load_radius_km(args.arr_body, args.body_catalog, args.arr_body_radius_km),
        "flyby_safe_altitude_km": args.flyby_min_altitude_km + args.flyby_atmosphere_margin_km,
        "quiet_stderr": args.quiet_stderr,
        **nom,
        "ut1_trust_s": args.ut1_trust_s,
        "dv1_trust_m_s": args.dv1_trust_m_s,
        "burn1_yaw_trust_deg": args.burn1_yaw_trust_deg,
        "burn1_pitch_trust_deg": args.burn1_pitch_trust_deg,
        "burn1_max_angle_from_nominal_deg": args.burn1_max_angle_from_nominal_deg,
        "local_nominal_dv_min_m_s": args.local_nominal_dv_min_m_s,
        "local_nominal_dv_max_m_s": args.local_nominal_dv_max_m_s,
        "dv1_nominal_mode": args.dv1_nominal_mode,
        "burn_distance_min_from_dep_km": args.burn_distance_min_from_dep_km,
        "burn_distance_max_from_dep_km": args.burn_distance_max_from_dep_km,
        "min_burn_dv_m_s": args.min_burn_dv_m_s,
        "max_burn_dv_m_s": args.max_burn_dv_m_s,
        "ut2_trust_s": args.ut2_trust_days * 86400.0,
        "arrival_trust_s": args.arrival_trust_days * 86400.0,
        "dsm_max_m_s": args.dsm_max_m_s,
        "min_burn_after_live_s": args.min_burn_after_live_s,
        "min_dsm_after_burn_s": args.min_dsm_after_burn_s,
        "min_arrival_after_dsm_s": args.min_arrival_after_dsm_s,
        "min_dsm_distance_from_dep_km": args.min_dsm_distance_from_dep_km,
        "min_dsm_radial_v_m_s": args.min_dsm_radial_v_m_s,
        "min_escape_energy_m2_s2": args.min_escape_energy_m2_s2,
        "pos_scale_km": args.pos_scale_km,
        "dsm_scale_m_s": args.dsm_scale_m_s,
        "dsm_weight": args.dsm_weight,
        "arrival_time_weight": args.arrival_time_weight,
        "ut1_weight": args.ut1_weight,
        "dv1_anchor_weight": args.dv1_anchor_weight,
        "burn1_yaw_weight": args.burn1_yaw_weight,
        "burn1_pitch_weight": args.burn1_pitch_weight,
        "vinf_mag_scale_m_s": args.vinf_mag_scale_m_s,
        "vinf_angle_scale_deg": args.vinf_angle_scale_deg,
        "vinf_mag_weight": args.vinf_mag_weight,
        "vinf_angle_weight": args.vinf_angle_weight,
        "flyby_turn_deficit_scale_deg": args.flyby_turn_deficit_scale_deg,
        "flyby_turn_weight": args.flyby_turn_weight,
        "flyby_mag_free_m_s": args.flyby_mag_free_m_s,
        "flyby_powered_scale_m_s": args.flyby_powered_scale_m_s,
        "flyby_powered_weight": args.flyby_powered_weight,
        "vinf_out_ref_raw_m_s": vinf_out,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    warm = _load_warm_start(args.warm_start_json)
    dummy_udp = Leg1TrustRegionUDP(cfg)
    bounds = dummy_udp.get_bounds()

    print("=== LEG1 LAMBERT TRUST REGION / MP CUSTOM DE ===")
    print(f"live_t        : {live_t}")
    print(f"ut1_nominal   : {cfg['ut1_nominal_s']}  trust=±{args.ut1_trust_s}s")
    print(f"dv1_nominal   : {cfg['dv1_nominal_m_s']:.3f} m/s  trust=±{args.dv1_trust_m_s} m/s")
    if cfg.get("candidate_vinf_ref_m_s") is not None:
        print(f"Lambert v∞ ref : {cfg['candidate_vinf_ref_m_s']:.3f} m/s")
    print(f"ut2_nominal   : {cfg['ut2_nominal_s']}")
    print(f"arrival_nom   : {cfg['arrival_nominal_s']}  trust=±{args.arrival_trust_days}d")
    print(f"dsm_max       : {args.dsm_max_m_s} m/s")
    print(f"preburn src   : {args.preburn_source} max_abs_vr={args.parking_max_abs_radial_v_m_s} m/s bound_required={args.require_preburn_bound}")
    print(f"burn gate     : r=[{args.burn_distance_min_from_dep_km}, {args.burn_distance_max_from_dep_km}] km dv=[{args.min_burn_dv_m_s}, {args.max_burn_dv_m_s}] m/s mode={args.dv1_nominal_mode}")
    print(f"flyby gate    : radius={cfg.get('arr_body_radius_km')} km safe_alt={cfg.get('flyby_safe_altitude_km')} km turn_w={args.flyby_turn_weight}")
    print(f"population    : {args.population} generations={args.generations} seed={args.seed}")
    print(f"workers       : {args.workers} chunk_size={args.chunk_size}")
    print(f"warm_start    : {args.warm_start_json if args.warm_start_json else 'none'}")
    print(f"output_dir    : {args.output_dir}")

    champ_x, champ_f, pop, fit = _custom_de(
        cfg, bounds,
        population=args.population,
        generations=args.generations,
        seed=args.seed,
        workers=args.workers,
        chunk_size=args.chunk_size,
        F=args.de_f,
        CR=args.de_cr,
        warm_start=warm,
        warm_start_jitter=args.warm_start_jitter,
    )

    # Re-evaluate champion in the parent with a fresh server to get diagnostics.
    parent_udp = Leg1TrustRegionUDP(cfg)
    parent_udp.fitness(list(map(float, champ_x)))
    best = parent_udp.best or {"fitness": champ_f}
    best["champion_f"] = champ_f
    best["champion_x"] = list(map(float, champ_x))

    result = {
        "schema": "leg1_lambert_trust_mp_de_result_v0_7_twobody_parking",
        "config": cfg,
        "champion_f": champ_f,
        "champion_x": list(map(float, champ_x)),
        "best": best,
        "nominals": nom,
        "optimizer": {
            "kind": "custom_de_multiprocessing",
            "population": args.population,
            "generations": args.generations,
            "seed": args.seed,
            "workers": args.workers,
            "de_f": args.de_f,
            "de_cr": args.de_cr,
        },
        "notes": [
            "Each worker keeps one principia_impulsive_particle_server process open; this avoids spawning a server per evaluation.",
            "v0.7 keeps the multiprocessing pool open for the whole DE run instead of recreating it every evaluation.",
            "v0.7 uses a two-body parking provider before burn1, then starts the Principia N-body arc at the post-burn state.",
            "SPICE kernels are loaded once per worker in the pool initializer.",
            "First burn is bounded around a local two-body escape hyperbola whose outgoing asymptote matches the patched-conic/Lambert v∞.",
            "FlightPlan export still requires projection into Principia tangent/normal/binormal; do not insert LevelA vector blindly.",
        ],
    }
    (args.output_dir / "leg1_mp_de_result.json").write_text(json.dumps(jsonable(result), indent=2) + "\n")
    if best and "dv1_raw_m_s" in best:
        (args.output_dir / "event_preview_first_burn.json").write_text(json.dumps(jsonable(build_event_preview(best, cfg)), indent=2) + "\n")
    # Save population summary for later warm starts / diagnostics.
    rows = []
    for i, (x, f) in enumerate(zip(pop, fit)):
        rows.append({"i": int(i), "fitness": float(f), "x": list(map(float, x))})
    rows.sort(key=lambda r: r["fitness"])
    (args.output_dir / "population_final_top.json").write_text(json.dumps(rows[: min(200, len(rows))], indent=2) + "\n")

    print("\n=== RESULT ===")
    print(json.dumps({
        "champion_f": champ_f,
        "best_fitness": best.get("fitness"),
        "tb1_s": best.get("tb1_s"),
        "dv1_norm_m_s": best.get("dv1_norm_m_s"),
        "dv1_delta_m_s": best.get("dv1_delta_m_s"),
        "dv1_yaw_deg": best.get("dv1_yaw_deg"),
        "dv1_pitch_deg": best.get("dv1_pitch_deg"),
        "nominal_dv1_norm_m_s": best.get("nominal_dv1_norm_m_s"),
        "angle_from_nominal_dv1_deg": best.get("angle_from_nominal_dv1_deg"),
        "t_dsm_s": best.get("t_dsm_s"),
        "dsm_norm_m_s": best.get("dsm_norm_m_s"),
        "t_arr_s": best.get("t_arr_s"),
        "final_pos_err_km": best.get("final_pos_err_km"),
        "arrival_vinf_in_m_s": best.get("arrival_vinf_in_m_s"),
        "preburn_source": best.get("preburn_source"),
        "burn_distance_from_dep_km": best.get("burn_distance_from_dep_km"),
        "burn_radial_v_m_s": best.get("burn_radial_v_m_s"),
        "preburn_energy_m2_s2": best.get("preburn_energy_m2_s2"),
        "escape_energy_m2_s2": best.get("escape_energy_m2_s2"),
        "dsm_distance_from_dep_km": best.get("dsm_distance_from_dep_km"),
    }, indent=2))
    print(f"[OK] wrote {args.output_dir / 'leg1_mp_de_result.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
