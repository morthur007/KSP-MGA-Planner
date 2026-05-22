#!/usr/bin/env python3
"""
spice_vcarelnav_targeter_v0_3.py

Backend SPICE para VCAREL_NAV sem plugin-b64/save. v0.3 aplica transform SPICE->pipeline raw e GM sem heurística por magnitude.

Mudança principal em relação ao v0:
  - integra a nave em coordenadas relativas ao nav_body, não em coordenadas
    heliocêntricas absolutas.
  - subtrai a aceleração do nav_body: a_rel = a_ship - a_nav.
  - isso reduz cancelamento numérico e evita o solve_ivp ficar tentando passos
    absurdamente pequenos perto de uma origem heliocêntrica grande.

API:
  SpiceVcarelNavTargeter(...).vcarel_nav_spice(...)

Retorno OKCARELNAV-like.
"""

from __future__ import annotations

from dataclasses import dataclass
import argparse
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np

try:
    import spiceypy as spice
except Exception as exc:
    spice = None
    _SPICE_IMPORT_ERROR = exc
else:
    _SPICE_IMPORT_ERROR = None

try:
    from scipy.integrate import solve_ivp
    from scipy.optimize import minimize_scalar
except Exception as exc:
    solve_ivp = None
    minimize_scalar = None
    _SCIPY_IMPORT_ERROR = exc
else:
    _SCIPY_IMPORT_ERROR = None


DAY_S = 86400.0


def norm(v: Sequence[float]) -> float:
    return float(np.linalg.norm(np.asarray(v, dtype=float)))


def unit(v: Sequence[float], *, name: str = "vector") -> np.ndarray:
    a = np.asarray(v, dtype=float)
    n = norm(a)
    if not math.isfinite(n) or n <= 0:
        raise ValueError(f"cannot normalize {name}: {v!r}")
    return a / n


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def body_name_upper(name: str) -> str:
    return str(name).strip().upper()


def spice_native_to_pipeline_raw(v: Sequence[float]) -> np.ndarray:
    """Transforma vetor do frame nativo do BSP para o raw/Principia usado no pipeline.

    Pelos testes de paridade:
      pipeline_raw = [+Z_spice, -X_spice, +Y_spice]

    O transform é ortonormal, então vale para posição, velocidade e aceleração.
    """
    a = np.asarray(v, dtype=float)
    return np.array([a[2], -a[0], a[1]], dtype=float)


def parse_body_list(s: str | None) -> list[str] | None:
    if not s:
        return None
    return [body_name_upper(x) for x in s.replace(";", ",").replace(" ", ",").split(",") if x.strip()]


def recursive_body_records(obj: Any) -> list[dict[str, Any]]:
    out = []
    if isinstance(obj, dict):
        has_name = any(k in obj for k in ("name", "body", "id", "spice_name"))
        has_mu = any(k in obj for k in (
            "mu_m3_s2", "gm_m3_s2", "gravitational_parameter_m3_s2",
            "mu", "gm", "gravitational_parameter",
        ))
        if has_name and has_mu:
            out.append(obj)
        for k, v in obj.items():
            if isinstance(v, dict):
                child = dict(v)
                if "name" not in child and str(k).isalpha():
                    child.setdefault("name", str(k))
                out.extend(recursive_body_records(child))
            elif isinstance(v, list):
                out.extend(recursive_body_records(v))
    elif isinstance(obj, list):
        for item in obj:
            out.extend(recursive_body_records(item))
    return out


def read_mu_from_record(rec: dict[str, Any]) -> float | None:
    # No body_catalog JNSQ/Principia usado aqui, gravitational_parameter já está
    # em m^3/s^2, inclusive para corpos pequenos. Portanto não aplicar heurística
    # por magnitude nessa chave.
    for k in ("mu_m3_s2", "gm_m3_s2", "gravitational_parameter_m3_s2", "gravitational_parameter", "mu", "gm"):
        if k in rec:
            return float(rec[k])

    # Só converte se houver chave explicitamente em km^3/s^2.
    for k in ("mu_km3_s2", "gm_km3_s2", "gravitational_parameter_km3_s2"):
        if k in rec:
            return float(rec[k]) * 1e9

    return None


def read_radius_km_from_record(rec: dict[str, Any]) -> float | None:
    for k in ("radius_km", "mean_radius_km", "equatorial_radius_km"):
        if k in rec:
            return float(rec[k])
    for k in ("radius_m", "mean_radius_m", "equatorial_radius_m", "equatorial_radius"):
        if k in rec:
            return float(rec[k]) / 1000.0
    if "radius" in rec:
        val = float(rec["radius"])
        return val / 1000.0 if abs(val) > 1e5 else val
    return None


def load_body_catalog(path: Path) -> dict[str, dict[str, Any]]:
    data = json.loads(path.read_text())
    bodies: dict[str, dict[str, Any]] = {}
    for rec in recursive_body_records(data):
        name = rec.get("name", rec.get("body", rec.get("id", rec.get("spice_name"))))
        if name is None:
            continue
        mu = read_mu_from_record(rec)
        if mu is None:
            continue
        name_u = body_name_upper(name)
        spice_name = body_name_upper(rec.get("spice_name", rec.get("spice", name_u)))
        bodies[name_u] = {
            "name": name_u,
            "spice_name": spice_name,
            "mu_m3_s2": float(mu),
            "radius_km": read_radius_km_from_record(rec),
            "raw_record": rec,
        }
    if not bodies:
        raise RuntimeError(f"no bodies with GM found in {path}")
    return bodies


@dataclass(frozen=True)
class NavImpulse:
    dt_s: float
    dvt_m_s: float
    dvn_m_s: float
    dvb_m_s: float


@dataclass
class PropagatedSegment:
    t0_rel_s: float
    t1_rel_s: float
    sol: Any


class SpiceVcarelNavTargeter:
    def __init__(
        self,
        *,
        bsp: Path,
        tpc: Path | None,
        body_catalog: Path,
        attractors: Sequence[str] | None = None,
        frame: str = "J2000",
        observer: str = "SUN",
        aberration: str = "NONE",
        spice_time_offset_s: float = 0.0,
        rtol: float = 1e-9,
        atol_pos_m: float = 1e-2,
        atol_vel_m_s: float = 1e-8,
        max_step_s: float = 7200.0,
    ):
        if spice is None:
            raise RuntimeError(f"spiceypy import failed: {_SPICE_IMPORT_ERROR}")
        if solve_ivp is None or minimize_scalar is None:
            raise RuntimeError(f"scipy import failed: {_SCIPY_IMPORT_ERROR}")

        self.bsp = Path(bsp)
        self.tpc = None if tpc is None else Path(tpc)
        self.body_catalog_path = Path(body_catalog)
        self.frame = frame
        self.observer = body_name_upper(observer)
        self.aberration = aberration
        self.spice_time_offset_s = float(spice_time_offset_s)
        self.rtol = float(rtol)
        self.atol = np.array([atol_pos_m, atol_pos_m, atol_pos_m, atol_vel_m_s, atol_vel_m_s, atol_vel_m_s], dtype=float)
        self.max_step_s = float(max_step_s)

        self.bodies = load_body_catalog(self.body_catalog_path)
        if attractors is None:
            self.attractor_names = sorted(self.bodies.keys())
        else:
            missing = [body_name_upper(b) for b in attractors if body_name_upper(b) not in self.bodies]
            if missing:
                raise KeyError(f"attractors missing from body catalog: {missing}")
            self.attractor_names = [body_name_upper(b) for b in attractors]

        self._loaded = False

    def __enter__(self):
        self.load_kernels()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.unload_kernels()

    def load_kernels(self) -> None:
        if self._loaded:
            return
        if self.tpc is not None:
            spice.furnsh(str(self.tpc))
        spice.furnsh(str(self.bsp))
        self._loaded = True

    def unload_kernels(self) -> None:
        if not self._loaded:
            return
        try:
            spice.kclear()
        finally:
            self._loaded = False

    def _et(self, t_game_s: float) -> float:
        return float(t_game_s) + self.spice_time_offset_s

    def body_state_abs_m(self, body: str, t_game_s: float) -> tuple[np.ndarray, np.ndarray]:
        name_u = body_name_upper(body)
        if name_u not in self.bodies:
            raise KeyError(f"body {body!r} not in catalog")
        spice_name = self.bodies[name_u]["spice_name"]

        # SPICE normalmente não permite/precisa computar target relativo a si mesmo.
        if spice_name == self.observer:
            return np.zeros(3), np.zeros(3)

        state_km, _lt = spice.spkezr(
            spice_name,
            self._et(t_game_s),
            self.frame,
            self.aberration,
            self.observer,
        )
        state_km = np.asarray(state_km, dtype=float)
        r_native_m = state_km[:3] * 1000.0
        v_native_m_s = state_km[3:] * 1000.0
        return spice_native_to_pipeline_raw(r_native_m), spice_native_to_pipeline_raw(v_native_m_s)

    def acceleration_abs_m_s2(self, r_abs_m: np.ndarray, t_game_s: float) -> np.ndarray:
        a = np.zeros(3)
        for name in self.attractor_names:
            rec = self.bodies[name]
            rb, _vb = self.body_state_abs_m(name, t_game_s)
            dr = rb - r_abs_m
            d = norm(dr)
            if d <= 0:
                continue
            a += rec["mu_m3_s2"] * dr / (d ** 3)
        return a

    def nav_acceleration_abs_m_s2(self, nav_body: str, t_game_s: float) -> np.ndarray:
        nav_u = body_name_upper(nav_body)
        r_nav, _v_nav = self.body_state_abs_m(nav_u, t_game_s)
        a = np.zeros(3)
        for name in self.attractor_names:
            if name == nav_u:
                continue
            rec = self.bodies[name]
            rb, _vb = self.body_state_abs_m(name, t_game_s)
            dr = rb - r_nav
            d = norm(dr)
            if d <= 0:
                continue
            a += rec["mu_m3_s2"] * dr / (d ** 3)
        return a

    def rhs_rel(self, t_rel_s: float, y_rel: np.ndarray, *, state_abs_s: float, nav_body: str) -> np.ndarray:
        t_game = state_abs_s + float(t_rel_s)
        r_nav, _v_nav = self.body_state_abs_m(nav_body, t_game)
        r_ship_abs = r_nav + y_rel[:3]

        a_ship = self.acceleration_abs_m_s2(r_ship_abs, t_game)
        a_nav = self.nav_acceleration_abs_m_s2(nav_body, t_game)

        dydt = np.empty(6)
        dydt[:3] = y_rel[3:]
        dydt[3:] = a_ship - a_nav
        return dydt

    @staticmethod
    def tnb_basis_from_rel(rel_r_raw_m: Sequence[float], rel_v_raw_m_s: Sequence[float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        r = np.asarray(rel_r_raw_m, dtype=float)
        v = np.asarray(rel_v_raw_m_s, dtype=float)
        T = unit(v, name="T/v_rel")
        h = np.cross(r, v)
        if norm(h) < 1e-12:
            radial = unit(r, name="radial")
            B = unit(np.cross(radial, T), name="fallback B")
        else:
            B = unit(h, name="B/h")
        N = unit(np.cross(B, T), name="N")
        return T, N, B

    def dv_nav_to_raw(self, rel_r: Sequence[float], rel_v: Sequence[float], dvt: float, dvn: float, dvb: float) -> tuple[np.ndarray, dict[str, Any]]:
        T, N, B = self.tnb_basis_from_rel(rel_r, rel_v)
        dv = float(dvt) * T + float(dvn) * N + float(dvb) * B
        return dv, {
            "tangent_raw": T.tolist(),
            "normal_raw": N.tolist(),
            "binormal_raw": B.tolist(),
            "dv_raw": dv.tolist(),
        }

    def propagate_with_impulses(
        self,
        *,
        nav_body: str,
        state_abs_s: float,
        rel_r_raw_m: Sequence[float],
        rel_v_raw_m_s: Sequence[float],
        impulses_nav: Sequence[NavImpulse],
        t_final_rel_s: float,
    ) -> tuple[list[PropagatedSegment], list[dict[str, Any]]]:
        y = np.empty(6)
        y[:3] = np.asarray(rel_r_raw_m, dtype=float)
        y[3:] = np.asarray(rel_v_raw_m_s, dtype=float)

        impulses = sorted([i for i in impulses_nav if 0.0 <= i.dt_s <= t_final_rel_s], key=lambda i: i.dt_s)
        times = [0.0] + [i.dt_s for i in impulses] + [float(t_final_rel_s)]

        segments: list[PropagatedSegment] = []
        burn_debug: list[dict[str, Any]] = []
        t_cur = 0.0
        i_imp = 0

        for t_next in times[1:]:
            if t_next > t_cur:
                sol = solve_ivp(
                    lambda t, yy: self.rhs_rel(t, yy, state_abs_s=state_abs_s, nav_body=nav_body),
                    (t_cur, t_next),
                    y,
                    method="DOP853",
                    rtol=self.rtol,
                    atol=self.atol,
                    dense_output=True,
                    max_step=self.max_step_s,
                )
                if not sol.success:
                    last_t = float(sol.t[-1]) if len(sol.t) else t_cur
                    last_y = sol.y[:, -1].copy() if sol.y.size else y.copy()
                    d_nav = norm(last_y[:3])
                    raise RuntimeError(
                        f"solve_ivp failed {t_cur}->{t_next}: {sol.message}; "
                        f"last_t_rel={last_t:.6f}; distance_to_{nav_body}={d_nav/1000:.6f} km; "
                        f"speed_rel={norm(last_y[3:]):.6f} m/s"
                    )
                segments.append(PropagatedSegment(t_cur, t_next, sol))
                y = sol.y[:, -1].copy()
                t_cur = float(t_next)

            while i_imp < len(impulses) and abs(impulses[i_imp].dt_s - t_cur) <= 1e-9:
                imp = impulses[i_imp]
                before_v_rel = y[3:].copy()
                r_nav, v_nav = self.body_state_abs_m(nav_body, state_abs_s + t_cur)
                dv_raw, dbg = self.dv_nav_to_raw(y[:3], y[3:], imp.dvt_m_s, imp.dvn_m_s, imp.dvb_m_s)
                y[3:] += dv_raw
                burn_debug.append({
                    "burn_dt_s": float(t_cur),
                    "burn_r_raw_m": (r_nav + y[:3]).tolist(),
                    "burn_rel_r_raw_m": y[:3].tolist(),
                    "burn_v_before_raw_m_s": (v_nav + before_v_rel).tolist(),
                    "burn_rel_v_before_raw_m_s": before_v_rel.tolist(),
                    "dv_navigation_m_s": [float(imp.dvt_m_s), float(imp.dvn_m_s), float(imp.dvb_m_s)],
                    **dbg,
                    "burn_v_after_raw_m_s": (v_nav + y[3:]).tolist(),
                    "burn_rel_v_after_raw_m_s": y[3:].tolist(),
                })
                i_imp += 1

        return segments, burn_debug

    def find_segment(self, segments: Sequence[PropagatedSegment], t_rel_s: float) -> PropagatedSegment:
        for seg in segments:
            if seg.t0_rel_s - 1e-9 <= t_rel_s <= seg.t1_rel_s + 1e-9:
                return seg
        return segments[-1]

    def rel_state_at(self, seg: PropagatedSegment, t_rel_s: float, *, state_abs_s: float, nav_body: str, arr_body: str):
        y = np.asarray(seg.sol.sol(float(t_rel_s)), dtype=float)
        t_game = state_abs_s + float(t_rel_s)

        r_nav, v_nav = self.body_state_abs_m(nav_body, t_game)
        r_arr, v_arr = self.body_state_abs_m(arr_body, t_game)

        ship_r = r_nav + y[:3]
        ship_v = v_nav + y[3:]

        rr = ship_r - r_arr
        vv = ship_v - v_arr
        return rr, vv, ship_r, ship_v, r_arr, v_arr, y

    def closest_approach(
        self,
        *,
        segments: Sequence[PropagatedSegment],
        nav_body: str,
        arr_body: str,
        state_abs_s: float,
        scan_start_rel_s: float,
        scan_end_rel_s: float,
        samples: int,
    ) -> dict[str, Any]:
        intervals = []
        for seg in segments:
            a = max(float(scan_start_rel_s), seg.t0_rel_s)
            b = min(float(scan_end_rel_s), seg.t1_rel_s)
            if b >= a:
                intervals.append((seg, a, b))
        if not intervals:
            raise RuntimeError("scan window does not overlap propagated segments")

        total = sum(max(1e-9, b - a) for _seg, a, b in intervals)
        best = None
        for seg, a, b in intervals:
            n = max(2, int(round(samples * max(1e-9, b - a) / total)))
            for t in np.linspace(a, b, n):
                rr, _vv, *_ = self.rel_state_at(seg, float(t), state_abs_s=state_abs_s, nav_body=nav_body, arr_body=arr_body)
                d = norm(rr)
                if best is None or d < best[0]:
                    best = (d, float(t), seg)

        if best is None:
            raise RuntimeError("no scan samples")
        _d, t0, seg0 = best
        opt_a = max(float(scan_start_rel_s), seg0.t0_rel_s)
        opt_b = min(float(scan_end_rel_s), seg0.t1_rel_s)

        def fdist(t: float) -> float:
            rr, _vv, *_ = self.rel_state_at(seg0, float(t), state_abs_s=state_abs_s, nav_body=nav_body, arr_body=arr_body)
            return norm(rr)

        if opt_b > opt_a:
            res = minimize_scalar(fdist, bounds=(opt_a, opt_b), method="bounded", options={"xatol": 1e-3})
            if res.success and math.isfinite(res.fun):
                ca_t = float(res.x)
                status = "refined"
            else:
                ca_t = t0
                status = "scan_best"
        else:
            ca_t = t0
            status = "scan_best"

        rr, vv, ship_r, ship_v, arr_r, arr_v, y_rel = self.rel_state_at(
            seg0, ca_t, state_abs_s=state_abs_s, nav_body=nav_body, arr_body=arr_body
        )
        d = norm(rr)
        speed = norm(vv)
        radial = float(np.dot(rr, vv) / d) if d > 0 else math.nan

        return {
            "ca_dt_s": ca_t,
            "ca_t_game_s": state_abs_s + ca_t,
            "ca_rel_r_raw_m": rr.tolist(),
            "ca_rel_v_raw_m_s": vv.tolist(),
            "ca_distance_m": d,
            "ca_speed_m_s": speed,
            "ca_radial_velocity_m_s": radial,
            "status": status,
            "ca_abs_debug_r_raw_m": ship_r.tolist(),
            "ca_abs_debug_v_raw_m_s": ship_v.tolist(),
            "arr_abs_debug_r_raw_m": arr_r.tolist(),
            "arr_abs_debug_v_raw_m_s": arr_v.tolist(),
            "ca_rel_to_nav_r_raw_m": y_rel[:3].tolist(),
            "ca_rel_to_nav_v_raw_m_s": y_rel[3:].tolist(),
        }

    def vcarel_nav_spice(
        self,
        *,
        rid: str,
        dep_body: str,
        arr_body: str,
        nav_body: str,
        state_abs_s: float,
        scan_start_rel_s: float,
        scan_end_rel_s: float,
        samples: int,
        rel_r_raw_m: Sequence[float],
        rel_v_raw_m_s: Sequence[float],
        impulses_nav: Sequence[NavImpulse | Sequence[float]],
    ) -> dict[str, Any]:
        impulses = []
        for imp in impulses_nav:
            if isinstance(imp, NavImpulse):
                impulses.append(imp)
            else:
                dt, dvt, dvn, dvb = imp
                impulses.append(NavImpulse(float(dt), float(dvt), float(dvn), float(dvb)))

        t_final = max(float(scan_end_rel_s), max([0.0] + [i.dt_s for i in impulses]))
        segments, burns = self.propagate_with_impulses(
            nav_body=nav_body,
            state_abs_s=float(state_abs_s),
            rel_r_raw_m=rel_r_raw_m,
            rel_v_raw_m_s=rel_v_raw_m_s,
            impulses_nav=impulses,
            t_final_rel_s=t_final,
        )
        ca = self.closest_approach(
            segments=segments,
            nav_body=nav_body,
            arr_body=arr_body,
            state_abs_s=float(state_abs_s),
            scan_start_rel_s=float(scan_start_rel_s),
            scan_end_rel_s=float(scan_end_rel_s),
            samples=int(samples),
        )
        return {
            "tag": "OKCARELNAV_SPICE",
            "id": rid,
            "dep_body": body_name_upper(dep_body),
            "arr_body": body_name_upper(arr_body),
            "nav_body": body_name_upper(nav_body),
            "state_dt_s": float(state_abs_s),
            "state_t_game_s": float(state_abs_s),
            **ca,
            "samples": int(samples),
            "n_burns": len(burns),
            "burns": burns,
        }


def _vec3_arg(s: str) -> list[float]:
    v = [float(x.strip()) for x in s.replace(";", ",").split(",") if x.strip()]
    if len(v) != 3:
        raise argparse.ArgumentTypeError(f"expected x,y,z, got {s!r}")
    return v


def _impulse_arg(s: str) -> NavImpulse:
    v = [float(x.strip()) for x in s.replace(";", ",").split(",") if x.strip()]
    if len(v) != 4:
        raise argparse.ArgumentTypeError(f"expected dt,dvt,dvn,dvb, got {s!r}")
    return NavImpulse(*v)


def main() -> int:
    ap = argparse.ArgumentParser(description="Probe direto do backend VCAREL_NAV_SPICE v0.3.")
    ap.add_argument("--bsp", type=Path, required=True)
    ap.add_argument("--tpc", type=Path, default=None)
    ap.add_argument("--body-catalog", type=Path, required=True)
    ap.add_argument("--attractors", default=None)
    ap.add_argument("--frame", default="J2000")
    ap.add_argument("--observer", default="SUN")
    ap.add_argument("--spice-time-offset-s", type=float, default=0.0)
    ap.add_argument("--dep-body", required=True)
    ap.add_argument("--arr-body", required=True)
    ap.add_argument("--nav-body", required=True)
    ap.add_argument("--state-abs-s", type=float, required=True)
    ap.add_argument("--scan-start-rel-s", type=float, required=True)
    ap.add_argument("--scan-end-rel-s", type=float, required=True)
    ap.add_argument("--samples", type=int, default=101)
    ap.add_argument("--rel-r-raw-m", type=_vec3_arg, required=True)
    ap.add_argument("--rel-v-raw-m-s", type=_vec3_arg, required=True)
    ap.add_argument("--impulse", type=_impulse_arg, action="append", default=[])
    ap.add_argument("--rtol", type=float, default=1e-9)
    ap.add_argument("--max-step-s", type=float, default=7200.0)
    args = ap.parse_args()

    with SpiceVcarelNavTargeter(
        bsp=args.bsp,
        tpc=args.tpc,
        body_catalog=args.body_catalog,
        attractors=parse_body_list(args.attractors),
        frame=args.frame,
        observer=args.observer,
        spice_time_offset_s=args.spice_time_offset_s,
        rtol=args.rtol,
        max_step_s=args.max_step_s,
    ) as targeter:
        out = targeter.vcarel_nav_spice(
            rid="spice_probe_0",
            dep_body=args.dep_body,
            arr_body=args.arr_body,
            nav_body=args.nav_body,
            state_abs_s=args.state_abs_s,
            scan_start_rel_s=args.scan_start_rel_s,
            scan_end_rel_s=args.scan_end_rel_s,
            samples=args.samples,
            rel_r_raw_m=args.rel_r_raw_m,
            rel_v_raw_m_s=args.rel_v_raw_m_s,
            impulses_nav=args.impulse,
        )

    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
