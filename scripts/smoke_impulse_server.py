#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import spiceypy as spice

from ksp_mga.native.impulse_server_client import PrincipiaImpulseServer


DAY_S = 86400.0


def norm_name(s: str) -> str:
    return s.strip().upper()


def norm(v) -> float:
    return float(np.linalg.norm(np.asarray(v, dtype=float)))


def stumpff_c(z: float) -> float:
    if z > 1e-8:
        sz = math.sqrt(z)
        return (1.0 - math.cos(sz)) / z
    if z < -1e-8:
        sz = math.sqrt(-z)
        return (math.cosh(sz) - 1.0) / (-z)
    return 0.5 - z / 24.0 + z * z / 720.0


def stumpff_s(z: float) -> float:
    if z > 1e-8:
        sz = math.sqrt(z)
        return (sz - math.sin(sz)) / (sz ** 3)
    if z < -1e-8:
        sz = math.sqrt(-z)
        return (math.sinh(sz) - sz) / (sz ** 3)
    return 1.0 / 6.0 - z / 120.0 + z * z / 5040.0


def kepler_universal_propagate(r0_km: np.ndarray, v0_km_s: np.ndarray, dt_s: float, mu_km3_s2: float):
    r0_km = np.asarray(r0_km, dtype=float)
    v0_km_s = np.asarray(v0_km_s, dtype=float)

    r0n = norm(r0_km)
    v0n2 = float(np.dot(v0_km_s, v0_km_s))
    vr0 = float(np.dot(r0_km, v0_km_s)) / r0n

    alpha = 2.0 / r0n - v0n2 / mu_km3_s2
    sqrt_mu = math.sqrt(mu_km3_s2)

    if abs(alpha) > 1e-12:
        chi = sqrt_mu * abs(alpha) * dt_s
    else:
        chi = sqrt_mu * dt_s / r0n

    if chi == 0.0:
        return r0_km.copy(), v0_km_s.copy()

    for _ in range(80):
        z = alpha * chi * chi
        C = stumpff_c(z)
        S = stumpff_s(z)

        F = (
            r0n * vr0 / sqrt_mu * chi * chi * C
            + (1.0 - alpha * r0n) * chi ** 3 * S
            + r0n * chi
            - sqrt_mu * dt_s
        )

        dF = (
            r0n * vr0 / sqrt_mu * chi * (1.0 - z * S)
            + (1.0 - alpha * r0n) * chi * chi * C
            + r0n
        )

        step = F / dF
        chi -= step

        if abs(step) < 1e-10:
            break

    z = alpha * chi * chi
    C = stumpff_c(z)
    S = stumpff_s(z)

    f = 1.0 - chi * chi / r0n * C
    g = dt_s - chi ** 3 / sqrt_mu * S

    r_km = f * r0_km + g * v0_km_s
    rn = norm(r_km)

    fdot = sqrt_mu / (rn * r0n) * (alpha * chi ** 3 * S - chi)
    gdot = 1.0 - chi * chi / rn * C

    v_km_s = fdot * r0_km + gdot * v0_km_s

    return r_km, v_km_s


def parse_transform(spec: str):
    tokens = [t.strip().upper() for t in spec.split(",")]
    if len(tokens) != 3:
        raise ValueError(f"transform inválido: {spec}")

    parsed = []
    for token in tokens:
        sign = 1.0
        if token.startswith("+"):
            axis = token[1:]
        elif token.startswith("-"):
            sign = -1.0
            axis = token[1:]
        else:
            axis = token

        if axis not in ("X", "Y", "Z"):
            raise ValueError(f"token de transform inválido: {token}")

        idx = {"X": 0, "Y": 1, "Z": 2}[axis]
        parsed.append((sign, idx))

    return parsed


def apply_transform(v, transform):
    v = np.asarray(v, dtype=float)
    return np.array([sign * v[idx] for sign, idx in transform], dtype=float)


def spk_state(body: str, et_s: float, central_body: str) -> np.ndarray:
    st, _ = spice.spkezr(norm_name(body), float(et_s), "J2000", "NONE", norm_name(central_body))
    return np.asarray(st, dtype=float)


def read_candidate(path: Path, rank: int) -> dict[str, str]:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if rank < 1 or rank > len(rows):
        raise IndexError(f"rank {rank} fora do intervalo 1..{len(rows)}")
    return rows[rank - 1]


def try_load_mu_from_json(paths: list[Path], central_body: str) -> float | None:
    central = norm_name(central_body)

    def walk(obj):
        if isinstance(obj, dict):
            yield obj
            for v in obj.values():
                yield from walk(v)
        elif isinstance(obj, list):
            for v in obj:
                yield from walk(v)

    mu_keys = [
        "mu_km3_s2",
        "gm_km3_s2",
        "gravitational_parameter_km3_s2",
        "mu",
        "gm",
        "GM",
        "gravitational_parameter",
    ]

    name_keys = ["name", "body", "body_name", "id", "naif_name"]

    for path in paths:
        if not path or not path.exists():
            continue
        try:
            obj = json.loads(path.read_text())
        except Exception:
            continue

        for d in walk(obj):
            names = [str(d.get(k, "")).upper() for k in name_keys]
            if central not in names:
                continue

            for k in mu_keys:
                if k in d:
                    mu = float(d[k])
                    # Heurística: se estiver em m^3/s^2, converter para km^3/s^2.
                    if mu > 1e12:
                        mu /= 1e9
                    return mu

    return None


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--candidate-seed", type=Path, required=True)
    p.add_argument("--rank", type=int, default=1)
    p.add_argument("--leg", type=int, default=1)

    p.add_argument("--bsp", type=Path, required=True)
    p.add_argument("--tpc", type=Path, required=True)
    p.add_argument("--metadata", type=Path, default=None)
    p.add_argument("--body-catalog", type=Path, default=None)
    p.add_argument("--central-body", default="Sun")
    p.add_argument("--central-mu-km3-s2", type=float, default=None)

    p.add_argument("--plugin-b64", type=Path, required=True)
    p.add_argument("--server", default="principia_impulsive_particle_server")
    p.add_argument("--transform", default="+Z,-X,+Y")
    p.add_argument("--buffer-days", type=float, default=0.235)
    p.add_argument("--output-json", type=Path, default=None)
    p.add_argument("--server-time-offset-s", type=float, default=0.0)

    args = p.parse_args()

    spice.furnsh(str(args.tpc))
    spice.furnsh(str(args.bsp))

    mu = args.central_mu_km3_s2
    if mu is None:
        mu = try_load_mu_from_json(
            [x for x in [args.metadata, args.body_catalog] if x is not None],
            args.central_body,
        )

    if mu is None:
        raise SystemExit(
            "Não consegui descobrir mu central. Passe "
            "--central-mu-km3-s2 8390563181.8028126"
        )

    row = read_candidate(args.candidate_seed, args.rank)

    leg = args.leg
    dep = norm_name(row[f"leg{leg}_dep"])
    arr = norm_name(row[f"leg{leg}_arr"])

    dep_i = leg - 1
    arr_i = leg

    t_dep = float(row[f"event{dep_i}_et_s"])
    t_arr = float(row[f"event{arr_i}_et_s"])
    buffer_s = args.buffer_days * DAY_S

    t_start = t_dep + buffer_s
    t_end = t_arr - buffer_s

    if t_end <= t_start:
        raise SystemExit("buffer grande demais para esta perna")

    st_dep = spk_state(dep, t_dep, args.central_body)
    target_st = spk_state(arr, t_end, args.central_body)

    vdep_km_s = np.array([
        float(row[f"leg{leg}_vdep_x_km_s"]),
        float(row[f"leg{leg}_vdep_y_km_s"]),
        float(row[f"leg{leg}_vdep_z_km_s"]),
    ])

    r_start_km, v_start_km_s = kepler_universal_propagate(
        st_dep[:3],
        vdep_km_s,
        buffer_s,
        mu,
    )

    transform = parse_transform(args.transform)

    r0_raw_m = apply_transform(r_start_km * 1000.0, transform)
    v0_raw_m_s = apply_transform(v_start_km_s * 1000.0, transform)

    target_r_raw_m = apply_transform(target_st[:3] * 1000.0, transform)
    target_v_raw_m_s = apply_transform(target_st[3:] * 1000.0, transform)

    print("=== SMOKE IMPULSE SERVER ===")
    print(f"candidate : {row.get('candidate_id', '')} rank={args.rank}")
    print(f"sequence  : {row.get('sequence_bodies', '')}")
    print(f"leg       : {leg} {dep}->{arr}")
    print(f"t_start   : {t_start:.9f}")
    print(f"t_end     : {t_end:.9f}")
    print(f"duration  : {(t_end - t_start) / DAY_S:.6f} d")
    print(f"mu        : {mu:.17g} km^3/s^2")
    print(f"transform : {args.transform}")
    print("[INFO] launching Principia impulse server...")

    with PrincipiaImpulseServer(args.server, args.plugin_b64) as srv:
        res = srv.propagate(
            req_id=f"smoke_rank{args.rank}_leg{leg}",
            t0_s=t_start + args.server_time_offset_s,
            burn_t_s=t_start + args.server_time_offset_s,
            t1_s=t_end + args.server_time_offset_s,
            r0_m=r0_raw_m,
            v0_m_s=v0_raw_m_s,
            burn_dv_m_s=np.zeros(3),
        )

    if res.status != "ok":
        print(f"[FAIL] server status={res.status} message={res.message}")
        return 1

    miss_m = norm(res.final_r_m - target_r_raw_m)
    relv_m_s = norm(res.final_v_m_s - target_v_raw_m_s)

    payload = {
        "status": "ok",
        "candidate_id": row.get("candidate_id", ""),
        "rank": args.rank,
        "leg": leg,
        "dep": dep,
        "arr": arr,
        "t_start_s": t_start,
        "t_end_s": t_end,
        "duration_days": (t_end - t_start) / DAY_S,
        "miss_km": miss_m / 1000.0,
        "relv_m_s": relv_m_s,
        "final_r_m": res.final_r_m.tolist(),
        "final_v_m_s": res.final_v_m_s.tolist(),
        "target_r_m": target_r_raw_m.tolist(),
        "target_v_m_s": target_v_raw_m_s.tolist(),
    }

    print("[OK] server propagated")
    print(f"miss     : {miss_m / 1000.0:.6f} km")
    print(f"relv     : {relv_m_s:.6f} m/s")

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2))
        print(f"[OK] wrote {args.output_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
