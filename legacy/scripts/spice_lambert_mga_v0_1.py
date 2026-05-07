#!/usr/bin/env python3
"""
spice_lambert_mga_v0_1.py

First real MGA fitness test for the KSP/Principia pipeline.

What this script does:
  - Loads the Principia-native SPICE kernels.
  - Uses PyGMO to optimize a fixed body sequence.
  - Solves each leg with a zero-revolution Lambert solver.
  - Scores departure v∞, arrival v∞, flyby v∞ mismatch, and flyby turn-angle feasibility.
  - Writes ranked candidates to CSV.

What this script intentionally does NOT do yet:
  - multi-revolution Lambert;
  - DSMs;
  - B-plane targeting;
  - finite burns;
  - Principia-native particle validation.

Units:
  - SPICE states: km and km/s.
  - epochs/TOFs: seconds.
  - μ: km^3/s^2.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import spiceypy as spice
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "spiceypy não está disponível. Ative o ambiente correto, ex.: conda activate space_working"
    ) from exc

try:
    import pygmo as pg
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "pygmo não está disponível. Instale/ative pygmo antes do teste Lambert."
    ) from exc

DAY_S = 86400.0
JULIAN_YEAR_S = 365.25 * DAY_S
BIG = 1.0e30


@dataclass(frozen=True)
class BodyInfo:
    mu_km3_s2: Optional[float] = None
    radius_km: Optional[float] = None


@dataclass(frozen=True)
class LambertSolution:
    v1_km_s: np.ndarray
    v2_km_s: np.ndarray
    path: str
    iterations: int


def norm_name(name: str) -> str:
    return str(name).strip().upper()


def as_float_or_none(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def extract_body_info_from_json(payload: Any) -> Dict[str, BodyInfo]:
    found: Dict[str, Dict[str, Optional[float]]] = {}

    name_keys = {"name", "body", "body_name", "naif_name"}
    mu_keys = {
        "mu",
        "gm",
        "GM",
        "gravitational_parameter",
        "gravitational_parameter_m3_s2",
        "mu_m3_s2",
        "gm_m3_s2",
    }
    radius_keys = {
        "radius",
        "mean_radius",
        "equatorial_radius",
        "radius_m",
        "mean_radius_m",
        "r",
    }

    def maybe_store(name: str, d: Dict[str, Any]) -> None:
        n = norm_name(name)
        if not n:
            return

        mu = None
        radius = None
        for k in mu_keys:
            if k in d:
                mu = as_float_or_none(d[k])
                break
        for k in radius_keys:
            if k in d:
                radius = as_float_or_none(d[k])
                break

        if mu is not None:
            if abs(mu) > 1.0e9:
                mu = mu / 1.0e9
        if radius is not None:
            if abs(radius) > 1.0e5:
                radius = radius / 1000.0

        if mu is not None or radius is not None:
            old = found.get(n, {})
            found[n] = {
                "mu_km3_s2": old.get("mu_km3_s2") if old.get("mu_km3_s2") is not None else mu,
                "radius_km": old.get("radius_km") if old.get("radius_km") is not None else radius,
            }

    def walk(obj: Any, hinted_name: Optional[str] = None) -> None:
        if isinstance(obj, dict):
            local_name = hinted_name
            for k in name_keys:
                if k in obj and isinstance(obj[k], str):
                    local_name = obj[k]
                    break

            if local_name is not None:
                maybe_store(local_name, obj)

            for k, v in obj.items():
                if isinstance(v, dict):
                    walk(v, hinted_name=str(k))
                elif isinstance(v, list):
                    walk(v, hinted_name=None)
        elif isinstance(obj, list):
            for item in obj:
                walk(item, hinted_name=hinted_name)

    walk(payload)
    return {k: BodyInfo(v.get("mu_km3_s2"), v.get("radius_km")) for k, v in found.items()}


def load_body_catalog(paths: Sequence[Path]) -> Dict[str, BodyInfo]:
    merged: Dict[str, BodyInfo] = {}
    for path in paths:
        if not path:
            continue
        try:
            payload = json.loads(path.read_text())
        except Exception as exc:
            print(f"[WARN] não consegui ler body catalog {path}: {exc}", file=sys.stderr)
            continue
        data = extract_body_info_from_json(payload)
        for name, info in data.items():
            old = merged.get(name, BodyInfo())
            merged[name] = BodyInfo(
                mu_km3_s2=old.mu_km3_s2 if old.mu_km3_s2 is not None else info.mu_km3_s2,
                radius_km=old.radius_km if old.radius_km is not None else info.radius_km,
            )
    return merged


def parse_body_value_pairs(items: Sequence[str], unit: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"esperado Body=value em {item!r} ({unit})")
        k, v = item.split("=", 1)
        out[norm_name(k)] = float(v)
    return out


def repeat_or_validate(values: Sequence[float], n: int, name: str) -> List[float]:
    if len(values) == 1:
        return [float(values[0])] * n
    if len(values) != n:
        raise ValueError(f"{name}: esperado 1 ou {n} valores, recebi {len(values)}")
    return [float(v) for v in values]


def safe_acos(x: float) -> float:
    return math.acos(max(-1.0, min(1.0, x)))


def turn_angle_max_rad(mu_km3_s2: float, rp_km: float, vinf_km_s: float) -> float:
    if mu_km3_s2 <= 0 or rp_km <= 0 or vinf_km_s <= 0:
        return 0.0
    denom = rp_km * vinf_km_s * vinf_km_s / mu_km3_s2 + 1.0
    return 2.0 * math.asin(max(0.0, min(1.0, 1.0 / denom)))


def stumpff_c(z: float) -> float:
    if z > 1.0e-8:
        sz = math.sqrt(z)
        return (1.0 - math.cos(sz)) / z
    if z < -1.0e-8:
        sz = math.sqrt(-z)
        return (math.cosh(sz) - 1.0) / (-z)
    # Series near zero.
    return 0.5 - z / 24.0 + z * z / 720.0 - z * z * z / 40320.0


def stumpff_s(z: float) -> float:
    if z > 1.0e-8:
        sz = math.sqrt(z)
        return (sz - math.sin(sz)) / (sz**3)
    if z < -1.0e-8:
        sz = math.sqrt(-z)
        return (math.sinh(sz) - sz) / (sz**3)
    # Series near zero.
    return 1.0 / 6.0 - z / 120.0 + z * z / 5040.0 - z * z * z / 362880.0


def lambert_universal_zero_rev(
    r1: np.ndarray,
    r2: np.ndarray,
    tof_s: float,
    mu_km3_s2: float,
    long_way: bool = False,
    max_iter: int = 80,
    tol_s: float = 1.0e-7,
) -> LambertSolution:
    """Solve the zero-revolution Lambert problem with universal variables.

    This is a robust smoke-test Lambert solver, not a production multi-rev solver.
    It tries short/long path via the sign of sin(dtheta).
    """
    r1 = np.asarray(r1, dtype=float)
    r2 = np.asarray(r2, dtype=float)
    r1n = float(np.linalg.norm(r1))
    r2n = float(np.linalg.norm(r2))

    if r1n <= 0 or r2n <= 0:
        raise ValueError("r1/r2 inválido")
    if tof_s <= 0:
        raise ValueError("TOF inválido")
    if mu_km3_s2 <= 0:
        raise ValueError("mu inválido")

    cos_dtheta = float(np.dot(r1, r2) / (r1n * r2n))
    cos_dtheta = max(-1.0, min(1.0, cos_dtheta))

    cross = np.cross(r1, r2)
    sin_abs = float(np.linalg.norm(cross) / (r1n * r2n))
    sin_dtheta = -sin_abs if long_way else sin_abs

    if abs(1.0 - cos_dtheta) < 1.0e-14:
        raise ValueError("Lambert degenerado: dtheta ~ 0")

    A = sin_dtheta * math.sqrt(r1n * r2n / (1.0 - cos_dtheta))
    if abs(A) < 1.0e-14:
        raise ValueError("Lambert degenerado: A ~ 0")

    sqrt_mu = math.sqrt(mu_km3_s2)

    def y_of_z(z: float) -> float:
        C = stumpff_c(z)
        S = stumpff_s(z)
        if C <= 0:
            return float("nan")
        return r1n + r2n + A * (z * S - 1.0) / math.sqrt(C)

    def tof_of_z(z: float) -> float:
        C = stumpff_c(z)
        S = stumpff_s(z)
        if C <= 0:
            return float("nan")
        y = y_of_z(z)
        if y < 0 or not math.isfinite(y):
            return float("nan")
        x = math.sqrt(y / C)
        return (x**3 * S + A * math.sqrt(y)) / sqrt_mu

    def f(z: float) -> float:
        t = tof_of_z(z)
        if not math.isfinite(t):
            return float("nan")
        return t - tof_s

    # Bracket a root. For elliptic zero-rev cases this usually lands quickly.
    z_low = -4.0 * math.pi * math.pi
    z_high = 4.0 * math.pi * math.pi

    f_low = f(z_low)
    f_high = f(z_high)

    # If lower side is invalid, move upward until valid.
    for _ in range(80):
        if math.isfinite(f_low):
            break
        z_low = 0.5 * (z_low + z_high)
        f_low = f(z_low)

    # Expand high side if needed.
    for _ in range(80):
        if math.isfinite(f_low) and math.isfinite(f_high) and f_low * f_high <= 0:
            break
        z_high *= 2.0
        if z_high > 1.0e6:
            break
        f_high = f(z_high)

    if not (math.isfinite(f_low) and math.isfinite(f_high) and f_low * f_high <= 0):
        raise ValueError(
            f"Lambert root não bracketado: z_low={z_low}, f_low={f_low}, z_high={z_high}, f_high={f_high}"
        )

    z_mid = 0.0
    iterations = 0
    for iterations in range(1, max_iter + 1):
        z_mid = 0.5 * (z_low + z_high)
        f_mid = f(z_mid)
        if not math.isfinite(f_mid):
            z_low = z_mid
            continue
        if abs(f_mid) < tol_s:
            break
        if f_low * f_mid <= 0:
            z_high = z_mid
            f_high = f_mid
        else:
            z_low = z_mid
            f_low = f_mid

    C = stumpff_c(z_mid)
    S = stumpff_s(z_mid)
    y = y_of_z(z_mid)
    if C <= 0 or y <= 0:
        raise ValueError("Lambert final inválido")

    f_l = 1.0 - y / r1n
    g_l = A * math.sqrt(y / mu_km3_s2)
    gdot_l = 1.0 - y / r2n

    if abs(g_l) < 1.0e-14:
        raise ValueError("Lambert g ~ 0")

    v1 = (r2 - f_l * r1) / g_l
    v2 = (gdot_l * r2 - r1) / g_l

    if not np.all(np.isfinite(v1)) or not np.all(np.isfinite(v2)):
        raise ValueError("Lambert produziu velocidade não-finita")

    return LambertSolution(v1, v2, "long" if long_way else "short", iterations)


class SpiceLambertMGA:
    def __init__(
        self,
        kernels: Sequence[str],
        sequence: Sequence[str],
        central_body: str,
        central_mu_km3_s2: float,
        bounds: Tuple[Sequence[float], Sequence[float]],
        body_info: Dict[str, BodyInfo],
        rp_min_km: Dict[str, float],
        default_rp_min_km: float,
        departure_weight: float,
        arrival_weight: float,
        powered_flyby_weight: float,
        turn_penalty_weight: float,
        tof_penalty_weight: float,
        try_long_way: bool,
    ):
        self.kernels = [str(k) for k in kernels]
        self.sequence = [norm_name(x) for x in sequence]
        self.central_body = norm_name(central_body)
        self.central_mu = float(central_mu_km3_s2)
        self.lb = list(map(float, bounds[0]))
        self.ub = list(map(float, bounds[1]))
        self.body_info = body_info
        self.rp_min_km = {norm_name(k): float(v) for k, v in rp_min_km.items()}
        self.default_rp_min_km = float(default_rp_min_km)
        self.departure_weight = float(departure_weight)
        self.arrival_weight = float(arrival_weight)
        self.powered_flyby_weight = float(powered_flyby_weight)
        self.turn_penalty_weight = float(turn_penalty_weight)
        self.tof_penalty_weight = float(tof_penalty_weight)
        self.try_long_way = bool(try_long_way)
        self._loaded = False

    def _load(self) -> None:
        if self._loaded:
            return
        for kernel in self.kernels:
            spice.furnsh(kernel)
        self._loaded = True

    def get_bounds(self) -> Tuple[List[float], List[float]]:
        return self.lb, self.ub

    def get_name(self) -> str:
        return "SPICE Lambert fixed-sequence MGA v0.1"

    def state(self, body: str, et: float) -> np.ndarray:
        self._load()
        st, _lt = spice.spkezr(norm_name(body), float(et), "J2000", "NONE", self.central_body)
        return np.asarray(st, dtype=float)

    def solve_leg_options(self, r1: np.ndarray, r2: np.ndarray, tof_s: float) -> List[LambertSolution]:
        options: List[LambertSolution] = []
        for long_way in ([False, True] if self.try_long_way else [False]):
            try:
                sol = lambert_universal_zero_rev(r1, r2, tof_s, self.central_mu, long_way=long_way)
                options.append(sol)
            except Exception:
                pass
        if not options:
            raise ValueError("nenhuma solução Lambert para a perna")
        return options

    def score_combo(
        self,
        epochs: List[float],
        tofs: List[float],
        states: List[np.ndarray],
        legs: List[LambertSolution],
    ) -> Dict[str, Any]:
        v_body = [s[3:] for s in states]

        vinf_dep_vec = legs[0].v1_km_s - v_body[0]
        vinf_arr_vec = legs[-1].v2_km_s - v_body[-1]
        vinf_dep = float(np.linalg.norm(vinf_dep_vec))
        vinf_arr = float(np.linalg.norm(vinf_arr_vec))

        powered_flyby_dv = 0.0
        turn_excess_rad = 0.0
        max_turn_required_deg = 0.0
        min_turn_margin_deg = None
        flyby_rows = []

        for j in range(1, len(self.sequence) - 1):
            body = self.sequence[j]
            vinf_in_vec = legs[j - 1].v2_km_s - v_body[j]
            vinf_out_vec = legs[j].v1_km_s - v_body[j]
            vinf_in = float(np.linalg.norm(vinf_in_vec))
            vinf_out = float(np.linalg.norm(vinf_out_vec))
            vinf_ref = 0.5 * (vinf_in + vinf_out)

            powered_flyby_dv += abs(vinf_out - vinf_in)

            if vinf_in > 0 and vinf_out > 0:
                req = safe_acos(float(np.dot(vinf_in_vec, vinf_out_vec) / (vinf_in * vinf_out)))
            else:
                req = math.pi

            info = self.body_info.get(body, BodyInfo())
            mu = info.mu_km3_s2
            rp = self.rp_min_km.get(body)
            if rp is None:
                if info.radius_km is not None:
                    rp = max(info.radius_km * 1.05, info.radius_km + 50.0)
                else:
                    rp = self.default_rp_min_km

            if mu is not None and mu > 0:
                max_turn = turn_angle_max_rad(mu, rp, vinf_ref)
                excess = max(0.0, req - max_turn)
                margin_deg = math.degrees(max_turn - req)
                turn_excess_rad += excess
            else:
                max_turn = float("nan")
                margin_deg = float("nan")

            max_turn_required_deg = max(max_turn_required_deg, math.degrees(req))
            if math.isfinite(margin_deg):
                min_turn_margin_deg = margin_deg if min_turn_margin_deg is None else min(min_turn_margin_deg, margin_deg)

            flyby_rows.append(
                {
                    "body": body,
                    "vinf_in_km_s": vinf_in,
                    "vinf_out_km_s": vinf_out,
                    "powered_flyby_dv_km_s": abs(vinf_out - vinf_in),
                    "turn_required_deg": math.degrees(req),
                    "turn_max_deg": math.degrees(max_turn) if math.isfinite(max_turn) else float("nan"),
                    "turn_margin_deg": margin_deg,
                    "rp_min_km": rp,
                    "mu_km3_s2": mu if mu is not None else float("nan"),
                }
            )

        tof_total_days = sum(tofs) / DAY_S
        cost = (
            self.departure_weight * vinf_dep
            + self.arrival_weight * vinf_arr
            + self.powered_flyby_weight * powered_flyby_dv
            + self.turn_penalty_weight * turn_excess_rad
            + self.tof_penalty_weight * tof_total_days
        )

        return {
            "status": "ok",
            "cost": float(cost),
            "departure_vinf_km_s": vinf_dep,
            "arrival_vinf_km_s": vinf_arr,
            "powered_flyby_dv_km_s": powered_flyby_dv,
            "turn_excess_deg": float(math.degrees(turn_excess_rad)),
            "max_turn_required_deg": float(max_turn_required_deg),
            "min_turn_margin_deg": float("nan") if min_turn_margin_deg is None else float(min_turn_margin_deg),
            "tof_total_days": float(tof_total_days),
            "epochs": epochs,
            "tofs_days": [dt / DAY_S for dt in tofs],
            "leg_paths": [leg.path for leg in legs],
            "leg_iterations": [leg.iterations for leg in legs],
            "flybys": flyby_rows,
        }

    def candidate_metrics(self, x: Sequence[float]) -> Dict[str, Any]:
        t0 = float(x[0])
        tofs = [float(v) for v in x[1:]]
        if len(tofs) != len(self.sequence) - 1:
            raise ValueError("número de TOFs incompatível com sequência")
        if any(dt <= 0 for dt in tofs):
            return {"status": "bad_tof", "cost": BIG}

        epochs = [t0]
        for dt in tofs:
            epochs.append(epochs[-1] + dt)

        states = [self.state(body, et) for body, et in zip(self.sequence, epochs)]
        r = [s[:3] for s in states]

        leg_options = [self.solve_leg_options(r[i], r[i + 1], tofs[i]) for i in range(len(tofs))]

        best: Optional[Dict[str, Any]] = None
        # For smoke tests with 2-4 legs this is fine. For large sequences we will prune.
        for combo in itertools.product(*leg_options):
            m = self.score_combo(epochs, tofs, states, list(combo))
            if best is None or float(m["cost"]) < float(best["cost"]):
                best = m

        if best is None:
            return {"status": "no_combo", "cost": BIG}
        return best

    def fitness(self, x: Sequence[float]) -> List[float]:
        try:
            m = self.candidate_metrics(x)
            c = float(m.get("cost", BIG))
            if not math.isfinite(c):
                c = BIG
            return [c]
        except Exception:
            return [BIG]


def write_rows(path: Path, rows: List[Dict[str, Any]], sequence: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sequence = [norm_name(x) for x in sequence]
    nlegs = len(sequence) - 1
    flyby_names = sequence[1:-1]

    fields = [
        "rank",
        "status",
        "cost",
        "departure_vinf_km_s",
        "arrival_vinf_km_s",
        "powered_flyby_dv_km_s",
        "tof_total_days",
        "turn_excess_deg",
        "max_turn_required_deg",
        "min_turn_margin_deg",
        "t0_et_s",
    ]
    fields += [f"tof{i+1}_days" for i in range(nlegs)]
    fields += [f"leg{i+1}_path" for i in range(nlegs)]
    fields += [f"event{i}_{body}_et_s" for i, body in enumerate(sequence)]

    for body in flyby_names:
        fields += [
            f"{body}_vinf_in_km_s",
            f"{body}_vinf_out_km_s",
            f"{body}_powered_flyby_dv_km_s",
            f"{body}_turn_required_deg",
            f"{body}_turn_max_deg",
            f"{body}_turn_margin_deg",
            f"{body}_rp_min_km",
        ]

    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for rank, row in enumerate(rows, start=1):
            out: Dict[str, Any] = {
                "rank": rank,
                "status": row.get("status", ""),
                "cost": row.get("cost", ""),
                "departure_vinf_km_s": row.get("departure_vinf_km_s", ""),
                "arrival_vinf_km_s": row.get("arrival_vinf_km_s", ""),
                "powered_flyby_dv_km_s": row.get("powered_flyby_dv_km_s", ""),
                "tof_total_days": row.get("tof_total_days", ""),
                "turn_excess_deg": row.get("turn_excess_deg", ""),
                "max_turn_required_deg": row.get("max_turn_required_deg", ""),
                "min_turn_margin_deg": row.get("min_turn_margin_deg", ""),
            }
            epochs = row.get("epochs", [])
            tofs = row.get("tofs_days", [])
            paths = row.get("leg_paths", [])

            out["t0_et_s"] = epochs[0] if epochs else ""
            for i, tof in enumerate(tofs):
                out[f"tof{i+1}_days"] = tof
            for i, path_name in enumerate(paths):
                out[f"leg{i+1}_path"] = path_name
            for i, et in enumerate(epochs):
                out[f"event{i}_{sequence[i]}_et_s"] = et

            flyby_map = {fb["body"]: fb for fb in row.get("flybys", [])}
            for body in flyby_names:
                fb = flyby_map.get(body, {})
                out[f"{body}_vinf_in_km_s"] = fb.get("vinf_in_km_s", "")
                out[f"{body}_vinf_out_km_s"] = fb.get("vinf_out_km_s", "")
                out[f"{body}_powered_flyby_dv_km_s"] = fb.get("powered_flyby_dv_km_s", "")
                out[f"{body}_turn_required_deg"] = fb.get("turn_required_deg", "")
                out[f"{body}_turn_max_deg"] = fb.get("turn_max_deg", "")
                out[f"{body}_turn_margin_deg"] = fb.get("turn_margin_deg", "")
                out[f"{body}_rp_min_km"] = fb.get("rp_min_km", "")

            w.writerow(out)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bsp", type=Path, required=True)
    p.add_argument("--tpc", type=Path, required=True)
    p.add_argument("--extra-kernel", type=Path, action="append", default=[])
    p.add_argument("--body-catalog", type=Path, action="append", default=[])
    p.add_argument("--metadata", type=Path, action="append", default=[])
    p.add_argument("--central-body", default="Sun")
    p.add_argument("--central-mu-km3-s2", type=float, default=None)
    p.add_argument("--sequence", nargs="+", required=True)
    p.add_argument("--start-et", type=float, required=True)
    p.add_argument("--search-years", type=float, default=20.0)
    p.add_argument("--tof-min-days", type=float, nargs="+", default=[20.0])
    p.add_argument("--tof-max-days", type=float, nargs="+", default=[800.0])
    p.add_argument("--rp-min-km", nargs="*", default=[])
    p.add_argument("--default-rp-min-km", type=float, default=1000.0)
    p.add_argument("--departure-weight", type=float, default=1.0)
    p.add_argument("--arrival-weight", type=float, default=1.0)
    p.add_argument("--powered-flyby-weight", type=float, default=1.0)
    p.add_argument("--turn-penalty-weight", type=float, default=1000.0, help="Peso por radiano de turn excess.")
    p.add_argument("--tof-penalty-weight", type=float, default=0.0)
    p.add_argument("--no-long-way", action="store_true", help="Desabilita tentativa Lambert long-way.")
    p.add_argument("--pop", type=int, default=128)
    p.add_argument("--gen", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--top", type=int, default=50)
    return p.parse_args()


def main() -> int:
    args = parse_args()

    sequence = [norm_name(x) for x in args.sequence]
    if len(sequence) < 2:
        raise SystemExit("--sequence precisa de pelo menos dois corpos")

    nlegs = len(sequence) - 1
    tof_min_days = repeat_or_validate(args.tof_min_days, nlegs, "--tof-min-days")
    tof_max_days = repeat_or_validate(args.tof_max_days, nlegs, "--tof-max-days")
    for a, b in zip(tof_min_days, tof_max_days):
        if b <= a:
            raise SystemExit(f"TOF max precisa ser > min; recebi {a}..{b} dias")

    body_info = load_body_catalog([*args.body_catalog, *args.metadata])
    rp_min_km = parse_body_value_pairs(args.rp_min_km, "km")

    central = norm_name(args.central_body)
    central_mu = args.central_mu_km3_s2
    if central_mu is None:
        central_info = body_info.get(central)
        if central_info and central_info.mu_km3_s2:
            central_mu = central_info.mu_km3_s2
    if central_mu is None:
        raise SystemExit(
            f"Não encontrei μ do corpo central {central}. Passe --central-mu-km3-s2 ou metadata/body-catalog com mu."
        )

    kernels = [args.tpc, *args.extra_kernel, args.bsp]
    lb = [args.start_et] + [d * DAY_S for d in tof_min_days]
    ub = [args.start_et + args.search_years * JULIAN_YEAR_S] + [d * DAY_S for d in tof_max_days]

    udp = SpiceLambertMGA(
        kernels=[str(k) for k in kernels],
        sequence=sequence,
        central_body=central,
        central_mu_km3_s2=central_mu,
        bounds=(lb, ub),
        body_info=body_info,
        rp_min_km=rp_min_km,
        default_rp_min_km=args.default_rp_min_km,
        departure_weight=args.departure_weight,
        arrival_weight=args.arrival_weight,
        powered_flyby_weight=args.powered_flyby_weight,
        turn_penalty_weight=args.turn_penalty_weight,
        tof_penalty_weight=args.tof_penalty_weight,
        try_long_way=not args.no_long_way,
    )

    print("[INFO] carregando kernels e testando corpos...")
    for k in kernels:
        if not Path(k).exists():
            raise SystemExit(f"kernel não encontrado: {k}")
    print(f"[INFO] central_mu({central}) = {central_mu:.17e} km^3/s^2")

    for body in sorted(set(sequence + [central])):
        if body == central:
            continue
        st = udp.state(body, args.start_et)
        print(f"  {body:<12} | r={np.linalg.norm(st[:3]):.6e} km | v={np.linalg.norm(st[3:]):.6e} km/s")

    print(f"[INFO] body metadata carregado para {len(body_info)} corpos")
    for body in sequence:
        info = body_info.get(body)
        if info:
            print(f"  meta {body:<12} mu={info.mu_km3_s2} km^3/s^2 radius={info.radius_km} km")

    # Probe one deterministic mid-bound point to catch Lambert failures early.
    probe = [0.5 * (a + b) for a, b in zip(lb, ub)]
    probe_metrics = udp.candidate_metrics(probe)
    print(
        "[INFO] probe cost={:.9g} dep_vinf={:.6g} arr_vinf={:.6g}".format(
            probe_metrics["cost"],
            probe_metrics["departure_vinf_km_s"],
            probe_metrics["arrival_vinf_km_s"],
        )
    )

    prob = pg.problem(udp)
    algo = pg.algorithm(pg.de(gen=args.gen, F=0.8, CR=0.9, seed=args.seed))
    pop = pg.population(prob, size=args.pop, seed=args.seed)

    print(f"[INFO] evoluindo PyGMO Lambert: pop={args.pop} gen={args.gen}")
    pop = algo.evolve(pop)

    xs = pop.get_x()
    rows: List[Dict[str, Any]] = []
    for x in xs:
        rows.append(udp.candidate_metrics(x))
    rows.sort(key=lambda r: float(r.get("cost", BIG)))
    rows = rows[: max(1, args.top)]

    write_rows(args.output, rows, sequence)

    champ = rows[0]
    print("\n=== CHAMPION LAMBERT ===")
    print(f"cost                   : {champ['cost']:.9g}")
    print(f"departure_vinf_km_s    : {champ['departure_vinf_km_s']:.9g}")
    print(f"arrival_vinf_km_s      : {champ['arrival_vinf_km_s']:.9g}")
    print(f"powered_flyby_dv_km_s  : {champ['powered_flyby_dv_km_s']:.9g}")
    print(f"tof_total_days         : {champ['tof_total_days']:.6f}")
    print(f"turn_excess_deg        : {champ['turn_excess_deg']:.9g}")
    print(f"min_turn_margin_deg    : {champ['min_turn_margin_deg']}")
    print("epochs_et_s            :", ", ".join(f"{v:.6f}" for v in champ["epochs"]))
    print("tofs_days              :", ", ".join(f"{v:.6f}" for v in champ["tofs_days"]))
    print("leg_paths              :", ", ".join(champ.get("leg_paths", [])))
    print(f"\n[OK] candidatos Lambert: {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
