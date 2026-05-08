#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import itertools
import math
from pathlib import Path

import numpy as np
import spiceypy as spice

from ksp_mga.native.impulse_server_client import PrincipiaImpulseServer
from scripts.smoke_impulse_server import (
    kepler_universal_propagate,
    parse_transform,
    apply_transform,
    spk_state,
    norm,
    norm_name,
)


AXES = ["X", "Y", "Z"]


def all_transforms():
    out = []
    for perm in itertools.permutations(AXES):
        for signs in itertools.product([1, -1], repeat=3):
            tokens = []
            for sign, axis in zip(signs, perm):
                tokens.append(("+" if sign > 0 else "-") + axis)
            out.append(",".join(tokens))
    return out


def read_candidate(path: Path, rank: int):
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    return rows[rank - 1]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--candidate-seed", type=Path, required=True)
    p.add_argument("--rank", type=int, default=1)
    p.add_argument("--leg", type=int, default=1)
    p.add_argument("--bsp", type=Path, required=True)
    p.add_argument("--tpc", type=Path, required=True)
    p.add_argument("--plugin-b64", type=Path, required=True)
    p.add_argument("--server", default="principia_impulsive_particle_server")
    p.add_argument("--central-body", default="Sun")
    p.add_argument("--central-mu-km3-s2", type=float, required=True)
    p.add_argument("--dt-seconds", type=float, default=3600.0)
    p.add_argument("--buffer-days", type=float, default=0.235)
    p.add_argument("--top", type=int, default=20)
    args = p.parse_args()

    spice.furnsh(str(args.tpc))
    spice.furnsh(str(args.bsp))

    row = read_candidate(args.candidate_seed, args.rank)
    leg = args.leg

    dep = norm_name(row[f"leg{leg}_dep"])
    dep_i = leg - 1
    t_dep = float(row[f"event{dep_i}_et_s"])
    t0 = t_dep + args.buffer_days * 86400.0
    t1 = t0 + args.dt_seconds

    st_dep = spk_state(dep, t_dep, args.central_body)

    vdep = np.array([
        float(row[f"leg{leg}_vdep_x_km_s"]),
        float(row[f"leg{leg}_vdep_y_km_s"]),
        float(row[f"leg{leg}_vdep_z_km_s"]),
    ])

    r0_km, v0_km_s = kepler_universal_propagate(
        st_dep[:3],
        vdep,
        args.buffer_days * 86400.0,
        args.central_mu_km3_s2,
    )

    r1_kep_km, v1_kep_km_s = kepler_universal_propagate(
        r0_km,
        v0_km_s,
        args.dt_seconds,
        args.central_mu_km3_s2,
    )

    print("=== DIAGNOSE IMPULSE SERVER FRAME ===")
    print(f"candidate : {row.get('candidate_id')} rank={args.rank}")
    print(f"leg       : {leg} {row[f'leg{leg}_dep']}->{row[f'leg{leg}_arr']}")
    print(f"t0        : {t0:.9f}")
    print(f"dt        : {args.dt_seconds:.3f} s")
    print(f"transforms: 48")
    print("[INFO] loading server once...")

    results = []
    with PrincipiaImpulseServer(args.server, args.plugin_b64) as srv:
        for tr_spec in all_transforms():
            tr = parse_transform(tr_spec)

            r0_m = apply_transform(r0_km * 1000.0, tr)
            v0_m_s = apply_transform(v0_km_s * 1000.0, tr)

            expected_r_m = apply_transform(r1_kep_km * 1000.0, tr)
            expected_v_m_s = apply_transform(v1_kep_km_s * 1000.0, tr)

            res = srv.propagate(
                req_id=tr_spec,
                t0_s=t0,
                burn_t_s=t0,
                t1_s=t1,
                r0_m=r0_m,
                v0_m_s=v0_m_s,
                burn_dv_m_s=np.zeros(3),
            )

            if res.status != "ok":
                results.append((float("inf"), float("inf"), tr_spec, res.status))
                continue

            pos_m = norm(res.final_r_m - expected_r_m)
            vel_m_s = norm(res.final_v_m_s - expected_v_m_s)
            results.append((pos_m, vel_m_s, tr_spec, "ok"))

    results.sort(key=lambda x: x[0])

    print("\n=== BEST TRANSFORMS BY SHORT PROPAGATION ===")
    print("rank pos_m vel_m_s transform status")
    for i, (pos_m, vel_m_s, tr_spec, status) in enumerate(results[: args.top], start=1):
        print(f"{i:>3} {pos_m:14.6f} {vel_m_s:12.9f} {tr_spec:<12} {status}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
