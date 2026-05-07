#!/usr/bin/env python3
"""
spice_lambert_beam_search_v0_1.py

Fast deterministic Lambert beam search for MGA/flyby candidates on a SPICE kernel.

Why:
  PyGMO is too slow as the first-pass global search. This script does a cheap
  discrete search with beam pruning and ranks candidates by flyby quality before
  any expensive native Principia validation.

Core idea:
  Build the route leg by leg.
  For each partial route, expand only the next TOF/path grid.
  Keep only the best `beam_width` partial routes after each leg.

User-facing idea:
  This is not only "search this exact sequence"; it is the basis for a future
  UI where users ask for route families such as:
    KERBIN -> [EVE, KERBIN, KERBIN] -> JOOL
  and the tool returns ranked flyby candidates.

Output is intentionally compatible with downstream scripts where possible:
  event0_KERBIN_et_s,event1_EVE_et_s,...,leg1_path,...

Example fixed KEKKJ:
  python spice_lambert_beam_search_v0_1.py \
    --bsp data/spice_jnsq_v0_3/jnsq_principia_native_v0_3_1000y_1h_deg9.bsp \
    --tpc data/spice_jnsq_v0_3/jnsq_principia_native_v0_3_1000y_1h_deg9.ids.tpc \
    --metadata data/spice_jnsq_v0_3/jnsq_principia_native_v0_3_1000y_1h_deg9.metadata.json \
    --body-catalog data/jnsq_gate0/ksp_future_paused/body_catalog.json \
    --central-body Sun \
    --sequence Kerbin Eve Kerbin Kerbin Jool \
    --start-et 81.85168640136972 \
    --search-years 30 \
    --t0-step-days 20 \
    --tof-min-days 120 120 250 500 \
    --tof-max-days 450 700 1300 4500 \
    --tof-step-days 20 20 20 40 \
    --beam-width 2000 \
    --top-n 200 \
    --output data/mga_smoke/kekkj_beam_v0_1.csv

Example with explicit resonant Kerbin->Kerbin third leg values:
  --tof-values-days-3 292 365 438 584 730 876 1095
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
from pykep_gateway_v0_1 import solve_lambert_pykep

import numpy as np
import spiceypy as spice

from spice_lambert_mga_v0_1 import (
    DAY_S,
    BodyInfo,
    lambert_universal_zero_rev,
    load_body_catalog,
    norm_name,
)


@dataclass
class Leg:
    dep: str
    arr: str
    t_dep_s: float
    t_arr_s: float
    tof_days: float
    path: str
    v_dep_km_s: np.ndarray
    v_arr_km_s: np.ndarray
    dep_body_v_km_s: np.ndarray
    arr_body_v_km_s: np.ndarray


@dataclass
class FlybyAudit:
    body: str
    event_index: int
    vinf_in_km_s: float
    vinf_out_km_s: float
    vinf_mismatch_km_s: float
    turn_required_deg: float
    turn_max_deg: float
    turn_margin_deg: float
    turn_excess_deg: float
    rp_min_km: float


@dataclass
class Route:
    sequence: List[str]
    epochs_s: List[float]
    legs: List[Leg] = field(default_factory=list)
    flybys: List[FlybyAudit] = field(default_factory=list)
    dep_vinf_km_s: float = 0.0
    arr_vinf_km_s: float = 0.0
    powered_flyby_dv_km_s: float = 0.0
    turn_excess_deg: float = 0.0
    min_turn_margin_deg: float = float("inf")
    cost: float = 0.0

    @property
    def tof_total_days(self) -> float:
        return sum(l.tof_days for l in self.legs)

    @property
    def raw_sum_km_s(self) -> float:
        return self.dep_vinf_km_s + self.arr_vinf_km_s + self.powered_flyby_dv_km_s


def norm(v: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(v, dtype=float)))


def safe_acos(x: float) -> float:
    return math.acos(max(-1.0, min(1.0, x)))


def turn_angle_max_rad(mu_km3_s2: float, rp_km: float, vinf_km_s: float) -> float:
    if mu_km3_s2 <= 0.0 or rp_km <= 0.0 or vinf_km_s <= 0.0:
        return 0.0
    denom = rp_km * vinf_km_s * vinf_km_s / mu_km3_s2 + 1.0
    return 2.0 * math.asin(max(0.0, min(1.0, 1.0 / denom)))


def spk_state(body: str, et_s: float, central_body: str) -> np.ndarray:
    st, _ = spice.spkezr(norm_name(body), float(et_s), "J2000", "NONE", norm_name(central_body))
    return np.asarray(st, dtype=float)  # km, km/s


def make_tof_grid(min_days: float, max_days: float, step_days: float) -> List[float]:
    if step_days <= 0:
        raise ValueError("step_days must be positive")
    vals: List[float] = []
    x = float(min_days)
    # Small epsilon to include endpoint under floating error.
    while x <= float(max_days) + 1e-9:
        vals.append(round(x, 9))
        x += float(step_days)
    return vals


def parse_values_days(value: Optional[str]) -> Optional[List[float]]:
    if not value:
        return None
    out: List[float] = []
    for part in value.replace(",", " ").split():
        out.append(float(part))
    return out


def choose_rp_min_km(info: BodyInfo, altitude_km: float, scale: float) -> float:
    radius = info.radius_km
    if radius is None or radius <= 0:
        return float("nan")
    return max(radius + altitude_km, radius * scale)


def compute_flyby(
    incoming: Leg,
    outgoing: Leg,
    body: str,
    event_index: int,
    body_info: Dict[str, BodyInfo],
    rp_altitude_km: float,
    rp_scale: float,
) -> FlybyAudit:
    vin_vec = incoming.v_arr_km_s - incoming.arr_body_v_km_s
    vout_vec = outgoing.v_dep_km_s - outgoing.dep_body_v_km_s
    vin = norm(vin_vec)
    vout = norm(vout_vec)
    mismatch = abs(vout - vin)

    if vin > 0 and vout > 0:
        req = safe_acos(float(np.dot(vin_vec, vout_vec) / (vin * vout)))
    else:
        req = math.pi

    info = body_info.get(norm_name(body), BodyInfo())
    mu = info.mu_km3_s2
    rp = choose_rp_min_km(info, rp_altitude_km, rp_scale)
    if mu is None or not math.isfinite(rp):
        max_turn = 0.0
    else:
        max_turn = turn_angle_max_rad(mu, rp, 0.5 * (vin + vout))

    margin = math.degrees(max_turn - req)
    excess = max(0.0, math.degrees(req - max_turn))
    return FlybyAudit(
        body=norm_name(body),
        event_index=event_index,
        vinf_in_km_s=vin,
        vinf_out_km_s=vout,
        vinf_mismatch_km_s=mismatch,
        turn_required_deg=math.degrees(req),
        turn_max_deg=math.degrees(max_turn),
        turn_margin_deg=margin,
        turn_excess_deg=excess,
        rp_min_km=rp,
    )


def route_score(
    r: Route,
    dep_weight: float,
    arr_weight: float,
    powered_weight: float,
    turn_weight: float,
    tof_weight: float,
    margin_bonus_weight: float,
) -> float:
    # A deliberately transparent scalar cost. The output CSV also contains raw
    # metrics, so downstream ranking can ignore this if needed.
    margin_bonus = 0.0
    if math.isfinite(r.min_turn_margin_deg):
        margin_bonus = max(0.0, min(10.0, r.min_turn_margin_deg))
    return (
        dep_weight * r.dep_vinf_km_s
        + arr_weight * r.arr_vinf_km_s
        + powered_weight * r.powered_flyby_dv_km_s
        + turn_weight * r.turn_excess_deg
        + tof_weight * (r.tof_total_days / 365.25)
        - margin_bonus_weight * margin_bonus
    )


def solve_leg_pykep(
    dep: str,
    arr: str,
    t_dep_s: float,
    tof_days: float,
    central_body: str,
    central_mu_km3_s2: float,
    max_revs: int,
) -> list[Leg]:
    t_arr_s = t_dep_s + tof_days * DAY_S

    st_dep = spk_state(dep, t_dep_s, central_body)
    st_arr = spk_state(arr, t_arr_s, central_body)

    # O Segredo do Boost.Python: TUPLAS nativas, não listas!
    r1_tuple = tuple(float(x) for x in st_dep[:3])
    r2_tuple = tuple(float(x) for x in st_arr[:3])
    
    # Garantindo os tipos exatos do C++
    tof_c = float(tof_days * DAY_S)
    mu_c = float(central_mu_km3_s2)
    max_revs_c = int(max_revs)

    legs = []

    for cw in (False, True):
        # Aqui o PyKEP vai receber exatamente o que pediu
        sols = solve_lambert_pykep(
            r1_tuple,
            r2_tuple,
            tof_c,
            mu_c,
            cw=cw,
            max_revs=max_revs_c,
        )

        for sol in sols:
            legs.append(
                Leg(
                    dep=norm_name(dep),
                    arr=norm_name(arr),
                    t_dep_s=t_dep_s,
                    t_arr_s=t_arr_s,
                    tof_days=tof_days,
                    path=sol.path_label,
                    v_dep_km_s=np.asarray(sol.v0_km_s, dtype=float),
                    v_arr_km_s=np.asarray(sol.v1_km_s, dtype=float),
                    dep_body_v_km_s=np.asarray(st_dep[3:], dtype=float),
                    arr_body_v_km_s=np.asarray(st_arr[3:], dtype=float),
                )
            )

    return legs


def expand_route(
    route: Route,
    leg: Leg,
    body_info: Dict[str, BodyInfo],
    rp_altitude_km: float,
    rp_scale: float,
    central_body: str,
    final_leg: bool,
    scoring_args: argparse.Namespace,
) -> Route:
    new = Route(sequence=route.sequence, epochs_s=list(route.epochs_s), legs=list(route.legs), flybys=list(route.flybys))
    new.legs.append(leg)
    new.epochs_s.append(leg.t_arr_s)

    if len(new.legs) == 1:
        new.dep_vinf_km_s = norm(leg.v_dep_km_s - leg.dep_body_v_km_s)
    else:
        new.dep_vinf_km_s = route.dep_vinf_km_s
        prev = new.legs[-2]
        event_index = len(new.legs) - 1
        fb = compute_flyby(
            incoming=prev,
            outgoing=leg,
            body=leg.dep,
            event_index=event_index,
            body_info=body_info,
            rp_altitude_km=rp_altitude_km,
            rp_scale=rp_scale,
        )
        new.flybys.append(fb)

    if final_leg:
        new.arr_vinf_km_s = norm(leg.v_arr_km_s - leg.arr_body_v_km_s)
    else:
        new.arr_vinf_km_s = 0.0

    new.powered_flyby_dv_km_s = sum(f.vinf_mismatch_km_s for f in new.flybys)
    new.turn_excess_deg = sum(f.turn_excess_deg for f in new.flybys)
    if new.flybys:
        new.min_turn_margin_deg = min(f.turn_margin_deg for f in new.flybys)
    else:
        new.min_turn_margin_deg = float("inf")

    new.cost = route_score(
        new,
        dep_weight=scoring_args.dep_weight,
        arr_weight=scoring_args.arr_weight,
        powered_weight=scoring_args.powered_weight,
        turn_weight=scoring_args.turn_weight,
        tof_weight=scoring_args.tof_weight,
        margin_bonus_weight=scoring_args.margin_bonus_weight,
    )
    return new


def passes_filters(route: Route, args: argparse.Namespace, partial: bool) -> bool:
    if route.dep_vinf_km_s > args.max_departure_vinf_km_s:
        return False
    if route.powered_flyby_dv_km_s > args.max_powered_flyby_dv_km_s:
        return False
    if route.turn_excess_deg > args.max_turn_excess_deg:
        return False
    if route.flybys and route.min_turn_margin_deg < args.min_turn_margin_deg:
        return False
    if not partial and route.arr_vinf_km_s > args.max_arrival_vinf_km_s:
        return False
    return True


def row_for_route(route: Route) -> Dict[str, object]:
    row: Dict[str, object] = {
        "cost": route.cost,
        "raw_sum_km_s": route.raw_sum_km_s,
        "departure_vinf_km_s": route.dep_vinf_km_s,
        "arrival_vinf_km_s": route.arr_vinf_km_s,
        "powered_flyby_dv_km_s": route.powered_flyby_dv_km_s,
        "turn_excess_deg": route.turn_excess_deg,
        "min_turn_margin_deg": route.min_turn_margin_deg if math.isfinite(route.min_turn_margin_deg) else "",
        "tof_total_days": route.tof_total_days,
        "epochs_et_s": ",".join(f"{x:.15g}" for x in route.epochs_s),
        "tofs_days": ",".join(f"{l.tof_days:.9g}" for l in route.legs),
        "leg_paths": ",".join(l.path for l in route.legs),
    }
    for i, body in enumerate(route.sequence):
        row[f"event{i}_{norm_name(body)}_et_s"] = route.epochs_s[i]
        
    for i, leg in enumerate(route.legs, start=1):
        row[f"leg{i}_path"] = leg.path
        row[f"leg{i}_tof_days"] = leg.tof_days
        # === A SUA SUGESTÃO: SALVANDO O VETOR EXATO ===
        row[f"leg{i}_vdep_x_km_s"] = leg.v_dep_km_s[0]
        row[f"leg{i}_vdep_y_km_s"] = leg.v_dep_km_s[1]
        row[f"leg{i}_vdep_z_km_s"] = leg.v_dep_km_s[2]
        row[f"leg{i}_varr_x_km_s"] = leg.v_arr_km_s[0]
        row[f"leg{i}_varr_y_km_s"] = leg.v_arr_km_s[1]
        row[f"leg{i}_varr_z_km_s"] = leg.v_arr_km_s[2]
    for fb in route.flybys:
        idx = fb.event_index
        row[f"flyby{idx}_{fb.body}_vinf_in_km_s"] = fb.vinf_in_km_s
        row[f"flyby{idx}_{fb.body}_vinf_out_km_s"] = fb.vinf_out_km_s
        row[f"flyby{idx}_{fb.body}_vinf_mismatch_km_s"] = fb.vinf_mismatch_km_s
        row[f"flyby{idx}_{fb.body}_turn_required_deg"] = fb.turn_required_deg
        row[f"flyby{idx}_{fb.body}_turn_max_deg"] = fb.turn_max_deg
        row[f"flyby{idx}_{fb.body}_turn_margin_deg"] = fb.turn_margin_deg
        row[f"flyby{idx}_{fb.body}_turn_excess_deg"] = fb.turn_excess_deg
    return row


def write_routes(path: Path, routes: List[Route]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [row_for_route(r) for r in routes]
    fields: List[str] = []
    for row in rows:
        for k in row.keys():
            if k not in fields:
                fields.append(k)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bsp", type=Path, required=True)
    p.add_argument("--tpc", type=Path, required=True)
    p.add_argument("--max-revs", type=int, default=0, help="Número máximo de revoluções para o Lambert")
    p.add_argument("--metadata", type=Path, action="append", default=[])
    p.add_argument("--body-catalog", type=Path, action="append", default=[])
    p.add_argument("--central-body", default="Sun")
    p.add_argument("--central-mu-km3-s2", type=float, default=None)
    p.add_argument("--sequence", nargs="+", required=True)
    p.add_argument("--start-et", type=float, required=True)
    p.add_argument("--search-years", type=float, default=30.0)
    p.add_argument("--t0-step-days", type=float, default=20.0)
    p.add_argument("--tof-min-days", nargs="+", type=float, required=True)
    p.add_argument("--tof-max-days", nargs="+", type=float, required=True)
    p.add_argument("--tof-step-days", nargs="+", type=float, required=True)
    p.add_argument("--tof-values-days-1", default=None)
    p.add_argument("--tof-values-days-2", default=None)
    p.add_argument("--tof-values-days-3", default=None)
    p.add_argument("--tof-values-days-4", default=None)
    p.add_argument("--tof-values-days-5", default=None)
    p.add_argument("--paths", nargs="+", default=["short", "long"], choices=["short", "long"])
    p.add_argument("--beam-width", type=int, default=2000)
    p.add_argument("--top-n", type=int, default=200)
    p.add_argument("--rp-altitude-km", type=float, default=50.0)
    p.add_argument("--rp-scale", type=float, default=1.05)
    p.add_argument("--max-departure-vinf-km-s", type=float, default=10.0)
    p.add_argument("--max-arrival-vinf-km-s", type=float, default=20.0)
    p.add_argument("--max-powered-flyby-dv-km-s", type=float, default=5.0)
    p.add_argument("--max-turn-excess-deg", type=float, default=1e9)
    p.add_argument("--min-turn-margin-deg", type=float, default=-1e9)
    p.add_argument("--dep-weight", type=float, default=1.0)
    p.add_argument("--arr-weight", type=float, default=1.0)
    p.add_argument("--powered-weight", type=float, default=50.0)
    p.add_argument("--turn-weight", type=float, default=100.0)
    p.add_argument("--tof-weight", type=float, default=0.05)
    p.add_argument("--margin-bonus-weight", type=float, default=0.02)
    p.add_argument("--output", type=Path, required=True)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    seq = [norm_name(x) for x in args.sequence]
    nlegs = len(seq) - 1
    if len(args.tof_min_days) != nlegs or len(args.tof_max_days) != nlegs or len(args.tof_step_days) != nlegs:
        raise SystemExit(f"tof-min/max/step precisam ter {nlegs} valores")

    spice.furnsh(str(args.tpc))
    spice.furnsh(str(args.bsp))

    body_info = load_body_catalog([*args.body_catalog, *args.metadata])
    central = norm_name(args.central_body)
    central_mu = args.central_mu_km3_s2
    if central_mu is None:
        info = body_info.get(central)
        if info and info.mu_km3_s2:
            central_mu = info.mu_km3_s2
    if central_mu is None:
        raise SystemExit("Não encontrei μ central. Passe --central-mu-km3-s2 ou metadata/body-catalog.")

    explicit_values = [
        parse_values_days(args.tof_values_days_1),
        parse_values_days(args.tof_values_days_2),
        parse_values_days(args.tof_values_days_3),
        parse_values_days(args.tof_values_days_4),
        parse_values_days(args.tof_values_days_5),
    ]
    tof_grids: List[List[float]] = []
    for i in range(nlegs):
        if explicit_values[i] is not None:
            vals = [v for v in explicit_values[i] if args.tof_min_days[i] <= v <= args.tof_max_days[i]]
        else:
            vals = make_tof_grid(args.tof_min_days[i], args.tof_max_days[i], args.tof_step_days[i])
        if not vals:
            raise SystemExit(f"leg {i+1}: grid de TOF vazio")
        tof_grids.append(vals)

    t0_grid = make_tof_grid(0.0, args.search_years * 365.25, args.t0_step_days)
    t0_epochs = [args.start_et + d * DAY_S for d in t0_grid]

    print("=== SPICE LAMBERT BEAM SEARCH V0.1 ===")
    print(f"sequence: {' -> '.join(seq)}")
    print(f"central_mu={float(central_mu):.17g} km^3/s^2")
    print(f"t0 samples={len(t0_epochs)} step={args.t0_step_days} d search={args.search_years} y")
    for i, vals in enumerate(tof_grids, start=1):
        preview = ", ".join(str(v) for v in vals[:8])
        if len(vals) > 8:
            preview += ", ..."
        print(f"leg {i} {seq[i-1]}->{seq[i]} TOF n={len(vals)} [{preview}]")
    print(f"beam_width={args.beam_width} top_n={args.top_n}")

    # Initial partial routes have only the departure epoch.
    beam: List[Route] = [Route(sequence=seq, epochs_s=[t]) for t in t0_epochs]

    for leg_index in range(1, nlegs + 1):
        dep = seq[leg_index - 1]
        arr = seq[leg_index]
        final_leg = leg_index == nlegs
        expanded: List[Route] = []
        tries = 0
        solved = 0
        for route in beam:
            t_dep = route.epochs_s[-1]
            for tof_days in tof_grids[leg_index - 1]:
                tries += 1
                
                # Chama o PyKEP passando o args.max_revs
                legs = solve_leg_pykep(dep, arr, t_dep, tof_days, central, float(central_mu), args.max_revs)
                
                # Itera sobre todas as soluções encontradas (curtas, longas, multi-rev)
                for leg in legs:
                    solved += 1
                    new_route = expand_route(
                        route,
                        leg,
                        body_info=body_info,
                        rp_altitude_km=args.rp_altitude_km,
                        rp_scale=args.rp_scale,
                        central_body=central,
                        final_leg=final_leg,
                        scoring_args=args,
                    )
                    
                    if passes_filters(new_route, args, partial=not final_leg):
                        expanded.append(new_route)
        expanded.sort(key=lambda r: r.cost)
        beam = expanded[: args.beam_width]
        print(
            f"[LEG {leg_index}/{nlegs}] {dep}->{arr}: "
            f"tries={tries} solved={solved} kept={len(beam)} "
            f"best_cost={beam[0].cost if beam else float('nan'):.6g}"
        )
        if not beam:
            print("[FAIL] beam vazio; relaxe filtros ou aumente grade.")
            return 1

    final_routes = sorted(beam, key=lambda r: (r.raw_sum_km_s, r.powered_flyby_dv_km_s, r.cost))[: args.top_n]
    write_routes(args.output, final_routes)

    print("\n=== TOP CANDIDATES ===")
    print("rank raw_sum dep arr powered tof margin cost epochs tofs paths")
    for i, r in enumerate(final_routes[:20], start=1):
        print(
            f"{i:>3} "
            f"{r.raw_sum_km_s:8.4f} "
            f"{r.dep_vinf_km_s:7.3f} "
            f"{r.arr_vinf_km_s:7.3f} "
            f"{r.powered_flyby_dv_km_s:7.3f} "
            f"{r.tof_total_days:8.1f} "
            f"{r.min_turn_margin_deg:7.3f} "
            f"{r.cost:9.3f} "
            f"{','.join(f'{x:.0f}' for x in r.epochs_s)} "
            f"{','.join(f'{l.tof_days:.1f}' for l in r.legs)} "
            f"{','.join(l.path for l in r.legs)}"
        )
    print(f"\n[OK] wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
