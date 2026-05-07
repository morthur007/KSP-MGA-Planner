#!/usr/bin/env python3
"""
lambert_candidate_to_particle_leg_v0_1.py

Converts one Lambert MGA candidate row into a massless-particle propagation input
for principia_particle_validator.

Why we trim the leg endpoints:
  A patched-conic Lambert leg starts at the planet centre. A native N-body
  particle starting exactly at a massive body's centre is singular/unphysical.
  Therefore this script starts the particle after a configurable start buffer
  and compares it before a configurable end buffer.

Input:
  - candidate CSV from spice_lambert_mga_v0_1.py or mga_parallel_runner_v0_1.py
  - SPICE kernels
  - fixed sequence
  - leg index, 1-based

Output:
  1) particle input CSV for principia_particle_validator:
       id,t0_s,t1_s,x_m,y_m,z_m,vx_m_s,vy_m_s,vz_m_s
  2) expected reference CSV in the same frame/units:
       id,t0_s,t1_s,x_m,y_m,z_m,vx_m_s,vy_m_s,vz_m_s

The expected state is produced by two-body Kepler propagation under the central
body mu from the Lambert departure state, not by SPICE/Principia. This is a
reference for the patched-conic leg, useful for measuring N-body perturbation
and frame issues.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import spiceypy as spice

# Reuse the Lambert solver and metadata loader from the previous script.
from spice_lambert_mga_v0_1 import (
    DAY_S,
    BodyInfo,
    lambert_universal_zero_rev,
    load_body_catalog,
    norm_name,
    stumpff_c,
    stumpff_s,
)


def parse_transform(spec: str) -> Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]:
    """Parse transforms like '+X,+Y,+Z', '+Y,-Z,-X'.

    Returns three (sign, source_index) pairs.
    """
    parts = [p.strip().upper() for p in spec.split(",")]
    if len(parts) != 3:
        raise ValueError(f"transform precisa ter 3 componentes: {spec!r}")
    mapping = {"X": 0, "Y": 1, "Z": 2}
    out = []
    used = []
    for p in parts:
        if len(p) != 2 or p[0] not in "+-" or p[1] not in mapping:
            raise ValueError(f"componente inválida no transform: {p!r}")
        sign = 1 if p[0] == "+" else -1
        idx = mapping[p[1]]
        out.append((sign, idx))
        used.append(idx)
    if sorted(used) != [0, 1, 2]:
        raise ValueError(f"transform deve usar X,Y,Z uma vez cada: {spec!r}")
    return tuple(out)  # type: ignore[return-value]


def apply_transform(vec: Sequence[float], transform: Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]) -> np.ndarray:
    arr = np.asarray(vec, dtype=float)
    return np.asarray([sign * arr[idx] for sign, idx in transform], dtype=float)


def read_candidate(path: Path, rank: int) -> Dict[str, str]:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"CSV sem candidatos: {path}")
    if rank < 1 or rank > len(rows):
        raise ValueError(f"rank fora do intervalo 1..{len(rows)}: {rank}")
    return rows[rank - 1]


def get_float(row: Dict[str, str], key: str) -> float:
    if key not in row or row[key] == "":
        raise KeyError(f"coluna ausente/vazia: {key}")
    return float(row[key])


def state(body: str, et: float, central_body: str) -> np.ndarray:
    st, _ = spice.spkezr(norm_name(body), float(et), "J2000", "NONE", norm_name(central_body))
    return np.asarray(st, dtype=float)


def kepler_universal_propagate(r0: np.ndarray, v0: np.ndarray, dt_s: float, mu: float) -> Tuple[np.ndarray, np.ndarray]:
    """Two-body universal-variable Kepler propagation in km/km/s."""
    r0 = np.asarray(r0, dtype=float)
    v0 = np.asarray(v0, dtype=float)
    r0n = float(np.linalg.norm(r0))
    v0n2 = float(np.dot(v0, v0))
    vr0 = float(np.dot(r0, v0) / r0n)
    sqrt_mu = math.sqrt(mu)
    alpha = 2.0 / r0n - v0n2 / mu

    if abs(dt_s) < 1e-12:
        return r0.copy(), v0.copy()

    # Conservative initial guess.
    if abs(alpha) > 1e-12:
        chi = sqrt_mu * abs(alpha) * dt_s
    else:
        chi = sqrt_mu * dt_s / r0n

    def F(chi_val: float) -> float:
        z = alpha * chi_val * chi_val
        C = stumpff_c(z)
        S = stumpff_s(z)
        return (
            r0n * vr0 / sqrt_mu * chi_val * chi_val * C
            + (1.0 - alpha * r0n) * chi_val**3 * S
            + r0n * chi_val
            - sqrt_mu * dt_s
        )

    for _ in range(80):
        z = alpha * chi * chi
        C = stumpff_c(z)
        S = stumpff_s(z)
        f = F(chi)
        # dF/dchi is current radius.
        r = (
            chi * chi * C
            + r0n * vr0 / sqrt_mu * chi * (1.0 - z * S)
            + r0n * (1.0 - z * C)
        )
        if abs(r) < 1e-14:
            break
        dchi = f / r
        chi -= dchi
        if abs(dchi) < 1e-10:
            break

    z = alpha * chi * chi
    C = stumpff_c(z)
    S = stumpff_s(z)

    f_l = 1.0 - chi * chi / r0n * C
    g_l = dt_s - chi**3 / sqrt_mu * S
    r_vec = f_l * r0 + g_l * v0
    rn = float(np.linalg.norm(r_vec))

    fdot_l = sqrt_mu / (rn * r0n) * (z * S - 1.0) * chi
    gdot_l = 1.0 - chi * chi / rn * C
    v_vec = fdot_l * r0 + gdot_l * v0
    return r_vec, v_vec


def write_particle_csv(path: Path, row_id: str, t0: float, t1: float, r_m: np.ndarray, v_m_s: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f, lineterminator='\n')
        w.writerow(["id", "t0_s", "t1_s", "x_m", "y_m", "z_m", "vx_m_s", "vy_m_s", "vz_m_s"])
        w.writerow([
            row_id,
            f"{t0:.17g}",
            f"{t1:.17g}",
            f"{r_m[0]:.17g}",
            f"{r_m[1]:.17g}",
            f"{r_m[2]:.17g}",
            f"{v_m_s[0]:.17g}",
            f"{v_m_s[1]:.17g}",
            f"{v_m_s[2]:.17g}",
        ])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--candidate-csv", type=Path, required=True)
    p.add_argument("--rank", type=int, default=1)
    p.add_argument("--bsp", type=Path, required=True)
    p.add_argument("--tpc", type=Path, required=True)
    p.add_argument("--metadata", type=Path, action="append", default=[])
    p.add_argument("--body-catalog", type=Path, action="append", default=[])
    p.add_argument("--central-body", default="Sun")
    p.add_argument("--central-mu-km3-s2", type=float, default=None)
    p.add_argument("--sequence", nargs="+", required=True)
    p.add_argument("--leg", type=int, required=True, help="1-based Lambert leg index.")
    p.add_argument("--start-buffer-days", type=float, default=3.0)
    p.add_argument("--end-buffer-days", type=float, default=3.0)
    # Inverse of plugin.cpp transform_position:
    # native Principia -> exported/SPK = (-Y,+Z,+X)
    # exported/SPK -> native Principia = (+Z,-X,+Y)
    p.add_argument("--spice-to-principia-transform", default="+Z,-X,+Y")
    p.add_argument("--output-input-csv", type=Path, required=True)
    p.add_argument("--output-expected-csv", type=Path, required=True)
    p.add_argument("--row-id", default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    sequence = [norm_name(x) for x in args.sequence]
    nlegs = len(sequence) - 1
    if args.leg < 1 or args.leg > nlegs:
        raise SystemExit(f"--leg precisa estar em 1..{nlegs}")

    for k in [args.tpc, args.bsp]:
        spice.furnsh(str(k))

    body_info = load_body_catalog([*args.body_catalog, *args.metadata])
    central = norm_name(args.central_body)
    central_mu = args.central_mu_km3_s2
    if central_mu is None:
        info = body_info.get(central)
        if info and info.mu_km3_s2:
            central_mu = info.mu_km3_s2
    if central_mu is None:
        raise SystemExit("Não encontrei μ central. Passe --central-mu-km3-s2 ou metadata/body-catalog.")

    row = read_candidate(args.candidate_csv, args.rank)
    leg_idx = args.leg - 1
    dep_body = sequence[leg_idx]
    arr_body = sequence[leg_idx + 1]

    t_dep = get_float(row, f"event{leg_idx}_{dep_body}_et_s")
    t_arr = get_float(row, f"event{leg_idx + 1}_{arr_body}_et_s")
    tof = t_arr - t_dep
    if tof <= 0:
        raise SystemExit("TOF inválido no candidato")

    path_col = f"leg{args.leg}_path"
    long_way = row.get(path_col, "short").strip().lower() == "long"

    st0 = state(dep_body, t_dep, central)
    st1 = state(arr_body, t_arr, central)
    sol = lambert_universal_zero_rev(st0[:3], st1[:3], tof, central_mu, long_way=long_way)

    start_dt = args.start_buffer_days * DAY_S
    end_dt = args.end_buffer_days * DAY_S
    if start_dt + end_dt >= tof:
        raise SystemExit("buffers maiores que a perna")

    # Reference patched-conic state, trimmed away from planet centres.
    r_start_km, v_start_km_s = kepler_universal_propagate(st0[:3], sol.v1_km_s, start_dt, central_mu)
    r_end_km, v_end_km_s = kepler_universal_propagate(st0[:3], sol.v1_km_s, tof - end_dt, central_mu)

    t_start = t_dep + start_dt
    t_end = t_arr - end_dt

    transform = parse_transform(args.spice_to_principia_transform)
    r_start_m = apply_transform(r_start_km * 1000.0, transform)
    v_start_m_s = apply_transform(v_start_km_s * 1000.0, transform)
    r_end_m = apply_transform(r_end_km * 1000.0, transform)
    v_end_m_s = apply_transform(v_end_km_s * 1000.0, transform)

    row_id = args.row_id or f"rank{args.rank}_leg{args.leg}_{dep_body}_to_{arr_body}"
    write_particle_csv(args.output_input_csv, row_id, t_start, t_end, r_start_m, v_start_m_s)
    write_particle_csv(args.output_expected_csv, row_id, t_start, t_end, r_end_m, v_end_m_s)

    print(f"[OK] particle input:    {args.output_input_csv}")
    print(f"[OK] expected reference:{args.output_expected_csv}")
    print(f"[INFO] candidate rank={args.rank} leg={args.leg} {dep_body}->{arr_body}")
    print(f"[INFO] t_dep={t_dep:.6f} t_arr={t_arr:.6f} tof_days={tof / DAY_S:.6f}")
    print(f"[INFO] trimmed t_start={t_start:.6f} t_end={t_end:.6f}")
    print(f"[INFO] Lambert path={sol.path} iterations={sol.iterations}")
    print(f"[INFO] transform SPICE->Principia: {args.spice_to_principia_transform}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
