#!/usr/bin/env python3
"""
spice_mga_smoke_v0_1.py

Smoke test for the KSP/Principia MGA pipeline.

Goal:
  - Load the synthetic Principia SPICE kernels.
  - Let PyGMO search epochs/TOFs for a fixed body sequence.
  - Use a cheap linear-transfer proxy, not a real Lambert solver yet.
  - Optionally apply simple flyby turn-angle penalties if body mu/radius are available.
  - Emit a ranked CSV of final population candidates.

This is intentionally a smoke test. It validates the optimization/data plumbing before
we introduce Lambert, B-plane targeting, DSMs, and Principia-native validation.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

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
        "pygmo não está disponível. Instale/ative pygmo antes do smoke test."
    ) from exc

DAY_S = 86400.0
JULIAN_YEAR_S = 365.25 * DAY_S
BIG = 1.0e30


@dataclass(frozen=True)
class BodyInfo:
    mu_km3_s2: Optional[float] = None
    radius_km: Optional[float] = None


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
    """Best-effort body metadata extractor.

    Handles many likely layouts:
      {"bodies": {"Kerbin": {"mu": ..., "radius": ...}}}
      {"Kerbin": {"gravitational_parameter": ..., "mean_radius": ...}}
      [{"name": "Kerbin", "mu": ..., "radius_m": ...}]
    Values may be in SI or km-ish. Heuristics convert SI to km units.
    """
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

        # Heuristics: KSP/Principia internals often store SI.
        # SPICE state vectors are km/km/s. Convert mu to km^3/s^2 and radius to km.
        if mu is not None:
            # Earth GM in m^3/s^2 ~ 3.986e14; in km^3/s^2 ~ 3.986e5.
            if abs(mu) > 1.0e9:
                mu = mu / 1.0e9
        if radius is not None:
            # Planetary radii in metres are usually > 1e5; in km usually < 1e5 here.
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
                    # If the key is a plausible body name, use it as a hint.
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
    """Parse Body=value pairs. Units are just for error messages."""
    out: Dict[str, float] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"esperado Body=value em {item!r} ({unit})")
        k, v = item.split("=", 1)
        out[norm_name(k)] = float(v)
    return out


def safe_acos(x: float) -> float:
    return math.acos(max(-1.0, min(1.0, x)))


def turn_angle_max_rad(mu_km3_s2: float, rp_km: float, vinf_km_s: float) -> float:
    if mu_km3_s2 <= 0 or rp_km <= 0 or vinf_km_s <= 0:
        return 0.0
    denom = rp_km * vinf_km_s * vinf_km_s / mu_km3_s2 + 1.0
    return 2.0 * math.asin(max(0.0, min(1.0, 1.0 / denom)))


class SpiceFixedSequenceSmoke:
    def __init__(
        self,
        kernels: Sequence[str],
        sequence: Sequence[str],
        central_body: str,
        bounds: Tuple[Sequence[float], Sequence[float]],
        body_info: Dict[str, BodyInfo],
        rp_min_km: Dict[str, float],
        default_rp_min_km: float,
        vinf_mismatch_weight: float,
        turn_penalty_weight: float,
        tof_penalty_weight: float,
    ):
        self.kernels = [str(k) for k in kernels]
        self.sequence = [norm_name(x) for x in sequence]
        self.central_body = norm_name(central_body)
        self.lb = list(map(float, bounds[0]))
        self.ub = list(map(float, bounds[1]))
        self.body_info = body_info
        self.rp_min_km = {norm_name(k): float(v) for k, v in rp_min_km.items()}
        self.default_rp_min_km = float(default_rp_min_km)
        self.vinf_mismatch_weight = float(vinf_mismatch_weight)
        self.turn_penalty_weight = float(turn_penalty_weight)
        self.tof_penalty_weight = float(tof_penalty_weight)
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
        return "SPICE fixed-sequence MGA smoke test"

    def state(self, body: str, et: float) -> np.ndarray:
        self._load()
        st, _lt = spice.spkezr(norm_name(body), float(et), "J2000", "NONE", self.central_body)
        return np.asarray(st, dtype=float)  # km, km/s

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
        v_body = [s[3:] for s in states]

        # Linear transfer proxy. Not Lambert. Good enough to test data plumbing.
        leg_v = []
        dv_proxy = 0.0
        for i, dt in enumerate(tofs):
            vt = (r[i + 1] - r[i]) / dt
            leg_v.append(vt)
            dv_proxy += float(np.linalg.norm(vt - v_body[i]))
            dv_proxy += float(np.linalg.norm(v_body[i + 1] - vt))

        vinf_mismatch = 0.0
        turn_excess_rad = 0.0
        max_turn_required_deg = 0.0
        min_turn_margin_deg = None
        flyby_rows = []

        for j in range(1, len(self.sequence) - 1):
            body = self.sequence[j]
            vinf_in_vec = leg_v[j - 1] - v_body[j]
            vinf_out_vec = leg_v[j] - v_body[j]
            vinf_in = float(np.linalg.norm(vinf_in_vec))
            vinf_out = float(np.linalg.norm(vinf_out_vec))
            vinf_ref = 0.5 * (vinf_in + vinf_out)

            vinf_mismatch += abs(vinf_out - vinf_in)

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
                    "turn_required_deg": math.degrees(req),
                    "turn_max_deg": math.degrees(max_turn) if math.isfinite(max_turn) else float("nan"),
                    "turn_margin_deg": margin_deg,
                    "rp_min_km": rp,
                    "mu_km3_s2": mu if mu is not None else float("nan"),
                }
            )

        tof_total = sum(tofs)
        cost = (
            dv_proxy
            + self.vinf_mismatch_weight * vinf_mismatch
            + self.turn_penalty_weight * turn_excess_rad
            + self.tof_penalty_weight * tof_total / DAY_S
        )

        return {
            "status": "ok",
            "cost": float(cost),
            "dv_proxy_km_s": float(dv_proxy),
            "tof_total_days": float(tof_total / DAY_S),
            "vinf_mismatch_km_s": float(vinf_mismatch),
            "turn_excess_deg": float(math.degrees(turn_excess_rad)),
            "max_turn_required_deg": float(max_turn_required_deg),
            "min_turn_margin_deg": float("nan") if min_turn_margin_deg is None else float(min_turn_margin_deg),
            "epochs": epochs,
            "tofs_days": [dt / DAY_S for dt in tofs],
            "flybys": flyby_rows,
        }

    def fitness(self, x: Sequence[float]) -> List[float]:
        try:
            m = self.candidate_metrics(x)
            c = float(m.get("cost", BIG))
            if not math.isfinite(c):
                c = BIG
            return [c]
        except Exception:
            # PyGMO needs a finite scalar even if SPICE rejects an epoch/body.
            return [BIG]


def repeat_or_validate(values: Sequence[float], n: int, name: str) -> List[float]:
    if len(values) == 1:
        return [float(values[0])] * n
    if len(values) != n:
        raise ValueError(f"{name}: esperado 1 ou {n} valores, recebi {len(values)}")
    return [float(v) for v in values]


def write_rows(path: Path, rows: List[Dict[str, Any]], sequence: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    nlegs = len(sequence) - 1
    flyby_names = [norm_name(x) for x in sequence[1:-1]]

    fields = [
        "rank",
        "status",
        "cost",
        "dv_proxy_km_s",
        "tof_total_days",
        "vinf_mismatch_km_s",
        "turn_excess_deg",
        "max_turn_required_deg",
        "min_turn_margin_deg",
        "t0_et_s",
    ]
    fields += [f"tof{i+1}_days" for i in range(nlegs)]
    fields += [f"event{i}_{body}_et_s" for i, body in enumerate(sequence)]

    for body in flyby_names:
        fields += [
            f"{body}_vinf_in_km_s",
            f"{body}_vinf_out_km_s",
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
                "dv_proxy_km_s": row.get("dv_proxy_km_s", ""),
                "tof_total_days": row.get("tof_total_days", ""),
                "vinf_mismatch_km_s": row.get("vinf_mismatch_km_s", ""),
                "turn_excess_deg": row.get("turn_excess_deg", ""),
                "max_turn_required_deg": row.get("max_turn_required_deg", ""),
                "min_turn_margin_deg": row.get("min_turn_margin_deg", ""),
            }
            epochs = row.get("epochs", [])
            tofs = row.get("tofs_days", [])
            out["t0_et_s"] = epochs[0] if epochs else ""
            for i, tof in enumerate(tofs):
                out[f"tof{i+1}_days"] = tof
            for i, et in enumerate(epochs):
                out[f"event{i}_{norm_name(sequence[i])}_et_s"] = et

            flyby_map = {fb["body"]: fb for fb in row.get("flybys", [])}
            for body in flyby_names:
                fb = flyby_map.get(body, {})
                out[f"{body}_vinf_in_km_s"] = fb.get("vinf_in_km_s", "")
                out[f"{body}_vinf_out_km_s"] = fb.get("vinf_out_km_s", "")
                out[f"{body}_turn_required_deg"] = fb.get("turn_required_deg", "")
                out[f"{body}_turn_max_deg"] = fb.get("turn_max_deg", "")
                out[f"{body}_turn_margin_deg"] = fb.get("turn_margin_deg", "")
                out[f"{body}_rp_min_km"] = fb.get("rp_min_km", "")

            w.writerow(out)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bsp", type=Path, required=True, help="SPK/BSP Type 13 Principia-native.")
    p.add_argument("--tpc", type=Path, required=True, help="Text kernel com IDs/names.")
    p.add_argument("--extra-kernel", type=Path, action="append", default=[], help="Kernel SPICE extra, repetível.")
    p.add_argument("--body-catalog", type=Path, action="append", default=[], help="JSON opcional com mu/radius dos corpos.")
    p.add_argument("--metadata", type=Path, action="append", default=[], help="JSON opcional do writer SPK; usado como body catalog se tiver mu/radius.")
    p.add_argument("--central-body", default="Sun")
    p.add_argument("--sequence", nargs="+", required=True, help="Ex.: Kerbin Eve Kerbin Jool")
    p.add_argument("--start-et", type=float, required=True, help="ET/TDB inicial em segundos J2000/fictício.")
    p.add_argument("--search-years", type=float, default=20.0)
    p.add_argument("--tof-min-days", type=float, nargs="+", default=[20.0], help="1 valor ou 1 por perna.")
    p.add_argument("--tof-max-days", type=float, nargs="+", default=[800.0], help="1 valor ou 1 por perna.")
    p.add_argument("--rp-min-km", nargs="*", default=[], help="Pares Body=rp_min_km. Ex.: Eve=800 Kerbin=700")
    p.add_argument("--default-rp-min-km", type=float, default=1000.0)
    p.add_argument("--vinf-mismatch-weight", type=float, default=10.0)
    p.add_argument("--turn-penalty-weight", type=float, default=1000.0, help="Peso por radiano de turn excess.")
    p.add_argument("--tof-penalty-weight", type=float, default=0.0, help="Peso leve por dia total, default 0.")
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

    kernels = [args.tpc, *args.extra_kernel, args.bsp]

    lb = [args.start_et] + [d * DAY_S for d in tof_min_days]
    ub = [args.start_et + args.search_years * JULIAN_YEAR_S] + [d * DAY_S for d in tof_max_days]

    udp = SpiceFixedSequenceSmoke(
        kernels=[str(k) for k in kernels],
        sequence=sequence,
        central_body=args.central_body,
        bounds=(lb, ub),
        body_info=body_info,
        rp_min_km=rp_min_km,
        default_rp_min_km=args.default_rp_min_km,
        vinf_mismatch_weight=args.vinf_mismatch_weight,
        turn_penalty_weight=args.turn_penalty_weight,
        tof_penalty_weight=args.tof_penalty_weight,
    )

    # Early kernel/body sanity check before PyGMO starts.
    print("[INFO] carregando kernels e testando corpos...")
    for k in kernels:
        if not Path(k).exists():
            raise SystemExit(f"kernel não encontrado: {k}")
    for body in sorted(set(sequence + [norm_name(args.central_body)])):
        if body == norm_name(args.central_body):
            continue
        st = udp.state(body, args.start_et)
        print(f"  {body:<12} | r={np.linalg.norm(st[:3]):.6e} km | v={np.linalg.norm(st[3:]):.6e} km/s")

    print(f"[INFO] body metadata carregado para {len(body_info)} corpos")
    for body in sequence:
        info = body_info.get(body)
        if info:
            print(f"  meta {body:<12} mu={info.mu_km3_s2} km^3/s^2 radius={info.radius_km} km")

    prob = pg.problem(udp)
    algo = pg.algorithm(pg.de(gen=args.gen, F=0.8, CR=0.9, seed=args.seed))
    pop = pg.population(prob, size=args.pop, seed=args.seed)

    print(f"[INFO] evoluindo PyGMO: pop={args.pop} gen={args.gen}")
    pop = algo.evolve(pop)

    xs = pop.get_x()
    rows: List[Dict[str, Any]] = []
    for x in xs:
        rows.append(udp.candidate_metrics(x))
    rows.sort(key=lambda r: float(r.get("cost", BIG)))
    rows = rows[: max(1, args.top)]

    write_rows(args.output, rows, sequence)

    champ = rows[0]
    print("\n=== CHAMPION ===")
    print(f"cost              : {champ['cost']:.9g}")
    print(f"dv_proxy_km_s     : {champ['dv_proxy_km_s']:.9g}")
    print(f"tof_total_days    : {champ['tof_total_days']:.6f}")
    print(f"vinf_mismatch     : {champ['vinf_mismatch_km_s']:.9g} km/s")
    print(f"turn_excess_deg   : {champ['turn_excess_deg']:.9g}")
    print(f"min_turn_margin   : {champ['min_turn_margin_deg']}")
    print("epochs_et_s       :", ", ".join(f"{v:.6f}" for v in champ["epochs"]))
    print("tofs_days         :", ", ".join(f"{v:.6f}" for v in champ["tofs_days"]))
    print(f"\n[OK] candidatos: {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
