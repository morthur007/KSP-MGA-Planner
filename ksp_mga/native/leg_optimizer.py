from __future__ import annotations

import argparse
import csv
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import spiceypy as spice
from scipy.optimize import least_squares

from ksp_mga.native.impulse_server_client import PrincipiaImpulseServer


DAY_S = 86400.0


def norm_name(s: str) -> str:
    return s.strip().upper()


def norm(v) -> float:
    return float(np.linalg.norm(np.asarray(v, dtype=float)))


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

        parsed.append((sign, {"X": 0, "Y": 1, "Z": 2}[axis]))

    return parsed


def apply_transform(v, transform) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    return np.array([sign * v[idx] for sign, idx in transform], dtype=float)


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


def kepler_universal_propagate(
    r0_km: np.ndarray,
    v0_km_s: np.ndarray,
    dt_s: float,
    mu_km3_s2: float,
) -> tuple[np.ndarray, np.ndarray]:
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


def spk_state(body: str, et_s: float, central_body: str) -> np.ndarray:
    st, _ = spice.spkezr(norm_name(body), float(et_s), "J2000", "NONE", norm_name(central_body))
    return np.asarray(st, dtype=float)


def sample_raw_body_state(
    *,
    sampler: str,
    plugin_b64: Path,
    target_body: str,
    sampler_central_body: str,
    et_s: float,
    plugin_base_et_s: float,
    work_dir: Path,
) -> tuple[np.ndarray, np.ndarray]:
    work_dir.mkdir(parents=True, exist_ok=True)

    safe_t = f"{et_s:.6f}".replace(".", "p").replace("-", "m")
    out_csv = work_dir / f"raw_{norm_name(target_body)}_{safe_t}.csv"

    if not out_csv.exists():
        offset_s = et_s - plugin_base_et_s

        cmd = [
            sampler,
            str(plugin_b64),
            str(out_csv),
            sampler_central_body,
            f"{offset_s:.17g}",
            "0",
            "21600",
        ]

        subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=True,
        )

    with out_csv.open(newline="") as f:
        rows = list(csv.DictReader(f))

    target = norm_name(target_body)
    for row in rows:
        if norm_name(row["body"]) == target:
            r = np.array([
                float(row["raw_x_m"]),
                float(row["raw_y_m"]),
                float(row["raw_z_m"]),
            ], dtype=float)
            v = np.array([
                float(row["raw_vx_m_s"]),
                float(row["raw_vy_m_s"]),
                float(row["raw_vz_m_s"]),
            ], dtype=float)
            return r, v

    raise RuntimeError(f"body {target_body!r} not found in {out_csv}")


@dataclass
class LegSetup:
    candidate_id: str
    rank: int
    leg: int
    dep_body: str
    arr_body: str
    path: str

    t_dep_s: float
    t_arr_s: float
    t_start_s: float
    t_end_s: float
    tof_days: float
    buffer_days: float

    r0_raw_m: np.ndarray
    v0_raw_m_s: np.ndarray
    target_r_raw_m: np.ndarray
    target_v_raw_m_s: np.ndarray

    initial_vdep_levela_km_s: np.ndarray
    transform: str


def read_candidate(path: Path, rank: int) -> dict[str, str]:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))

    if rank < 1 or rank > len(rows):
        raise IndexError(f"rank {rank} fora do intervalo 1..{len(rows)}")

    return rows[rank - 1]


def build_leg_setup(row: dict[str, str], leg: int, args: argparse.Namespace) -> LegSetup:
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
        raise ValueError(f"buffer grande demais na perna {leg}")

    st_dep = spk_state(dep, t_dep, args.central_body)

    vdep_levela_km_s = np.array([
        float(row[f"leg{leg}_vdep_x_km_s"]),
        float(row[f"leg{leg}_vdep_y_km_s"]),
        float(row[f"leg{leg}_vdep_z_km_s"]),
    ], dtype=float)

    # LevelA/Sun-centered spacecraft state after departure buffer.
    r_start_levela_km, v_start_levela_km_s = kepler_universal_propagate(
        st_dep[:3],
        vdep_levela_km_s,
        buffer_s,
        args.central_mu_km3_s2,
    )

    # IMPORTANT:
    # The leg optimizer must target the buffered Lambert/Kepler spacecraft
    # state, not the arrival planet center. Targeting the planet center at
    # t_arr-buffer drives the particle into a point-mass singularity and can
    # produce absurd relative velocities while still giving small position miss.
    r_target_levela_km, v_target_levela_km_s = kepler_universal_propagate(
        r_start_levela_km,
        v_start_levela_km_s,
        t_end - t_start,
        args.central_mu_km3_s2,
    )

    transform = parse_transform(args.transform)

    # Raw Barycentric absolute Sun state at start/end.
    origin_start_r_raw_m, origin_start_v_raw_m_s = sample_raw_body_state(
        sampler=args.sampler,
        plugin_b64=args.plugin_b64,
        target_body=args.raw_origin_body,
        sampler_central_body=args.raw_origin_body,
        et_s=t_start,
        plugin_base_et_s=args.plugin_base_et_s,
        work_dir=args.raw_cache_dir,
    )

    origin_end_r_raw_m, origin_end_v_raw_m_s = sample_raw_body_state(
        sampler=args.sampler,
        plugin_b64=args.plugin_b64,
        target_body=args.raw_origin_body,
        sampler_central_body=args.raw_origin_body,
        et_s=t_end,
        plugin_base_et_s=args.plugin_base_et_s,
        work_dir=args.raw_cache_dir,
    )

    # Proven frame contract:
    # raw_abs = Sun_raw_abs + LevelA_to_raw(LevelA_rel_Sun)
    r0_raw_m = origin_start_r_raw_m + apply_transform(r_start_levela_km * 1000.0, transform)
    v0_raw_m_s = origin_start_v_raw_m_s + apply_transform(v_start_levela_km_s * 1000.0, transform)

    target_r_raw_m = origin_end_r_raw_m + apply_transform(r_target_levela_km * 1000.0, transform)
    target_v_raw_m_s = origin_end_v_raw_m_s + apply_transform(v_target_levela_km_s * 1000.0, transform)

    return LegSetup(
        candidate_id=row.get("candidate_id", f"rank_{args.rank}"),
        rank=args.rank,
        leg=leg,
        dep_body=dep,
        arr_body=arr,
        path=row.get(f"leg{leg}_path", ""),
        t_dep_s=t_dep,
        t_arr_s=t_arr,
        t_start_s=t_start,
        t_end_s=t_end,
        tof_days=(t_arr - t_dep) / DAY_S,
        buffer_days=args.buffer_days,
        r0_raw_m=r0_raw_m,
        v0_raw_m_s=v0_raw_m_s,
        target_r_raw_m=target_r_raw_m,
        target_v_raw_m_s=target_v_raw_m_s,
        initial_vdep_levela_km_s=vdep_levela_km_s,
        transform=args.transform,
    )


def evaluate_leg(
    *,
    setup: LegSetup,
    server: PrincipiaImpulseServer,
    burn_dv_m_s: np.ndarray,
    req_id: str,
) -> dict[str, Any]:
    res = server.propagate(
        req_id=req_id,
        t0_s=setup.t_start_s,
        burn_t_s=setup.t_start_s,
        t1_s=setup.t_end_s,
        r0_m=setup.r0_raw_m,
        v0_m_s=setup.v0_raw_m_s,
        burn_dv_m_s=burn_dv_m_s,
    )

    if res.status != "ok":
        return {
            "status": res.status,
            "message": res.message,
            "miss_m": float("inf"),
            "relv_m_s": float("inf"),
            "final_r_m": np.full(3, np.nan),
            "final_v_m_s": np.full(3, np.nan),
        }

    miss_m = norm(res.final_r_m - setup.target_r_raw_m)
    relv_m_s = norm(res.final_v_m_s - setup.target_v_raw_m_s)

    return {
        "status": "ok",
        "message": "",
        "miss_m": miss_m,
        "relv_m_s": relv_m_s,
        "final_r_m": res.final_r_m,
        "final_v_m_s": res.final_v_m_s,
    }


def optimize_leg(
    *,
    setup: LegSetup,
    server: PrincipiaImpulseServer,
    work_dir: Path,
    max_nfev: int,
    pos_scale_km: float,
    dv_x_scale_m_s: float,
) -> dict[str, Any]:
    work_dir.mkdir(parents=True, exist_ok=True)

    history_path = work_dir / f"leg{setup.leg}_{setup.dep_body}_to_{setup.arr_body}_history.csv"
    pos_scale_m = pos_scale_km * 1000.0

    counter = {"n": 0}
    last_eval: dict[str, Any] | None = None

    with history_path.open("w", newline="") as f:
        fields = [
            "eval",
            "dvx_m_s",
            "dvy_m_s",
            "dvz_m_s",
            "dv_norm_m_s",
            "miss_km",
            "relv_m_s",
            "status",
            "message",
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()

        def residual(x: np.ndarray) -> np.ndarray:
            counter["n"] += 1
            dv = np.asarray(x, dtype=float)

            ev = evaluate_leg(
                setup=setup,
                server=server,
                burn_dv_m_s=dv,
                req_id=f"{setup.candidate_id}_leg{setup.leg}_eval{counter['n']}",
            )

            nonlocal last_eval
            last_eval = ev

            if ev["status"] != "ok":
                r = np.array([1e9, 1e9, 1e9], dtype=float)
                miss_km = float("inf")
                relv = float("inf")
            else:
                r = (ev["final_r_m"] - setup.target_r_raw_m) / pos_scale_m
                miss_km = ev["miss_m"] / 1000.0
                relv = ev["relv_m_s"]

            w.writerow({
                "eval": counter["n"],
                "dvx_m_s": dv[0],
                "dvy_m_s": dv[1],
                "dvz_m_s": dv[2],
                "dv_norm_m_s": norm(dv),
                "miss_km": miss_km,
                "relv_m_s": relv,
                "status": ev["status"],
                "message": ev["message"],
            })
            f.flush()

            print(
                f"[leg {setup.leg} eval {counter['n']:03d}] "
                f"miss={miss_km:14.3f} km "
                f"relv={relv:10.3f} m/s "
                f"dv={norm(dv):9.3f} m/s"
            )

            return r

        sol = least_squares(
            residual,
            x0=np.zeros(3),
            method="trf",
            max_nfev=max_nfev,
            x_scale=dv_x_scale_m_s,
            ftol=1e-10,
            xtol=1e-10,
            gtol=1e-10,
        )

    final_dv = np.asarray(sol.x, dtype=float)
    final_ev = evaluate_leg(
        setup=setup,
        server=server,
        burn_dv_m_s=final_dv,
        req_id=f"{setup.candidate_id}_leg{setup.leg}_final",
    )

    final_r = final_ev["final_r_m"]
    final_v = final_ev["final_v_m_s"]

    final_ok = (
        final_ev["status"] == "ok"
        and np.isfinite(final_ev["miss_m"])
        and np.isfinite(final_ev["relv_m_s"])
    )

    physical_success = bool(sol.success) and final_ok

    return {
        "candidate_id": setup.candidate_id,
        "rank": setup.rank,
        "leg": setup.leg,
        "dep_body": setup.dep_body,
        "arr_body": setup.arr_body,
        "path": setup.path,

        "t_dep_s": setup.t_dep_s,
        "t_arr_s": setup.t_arr_s,
        "t_start_s": setup.t_start_s,
        "t_end_s": setup.t_end_s,
        "tof_days": setup.tof_days,
        "buffer_days": setup.buffer_days,

        "transform": setup.transform,

        "solver_success": physical_success,
        "raw_solver_success": bool(sol.success),
        "solver_message": str(sol.message),
        "n_eval": counter["n"],

        "dvx_m_s": final_dv[0],
        "dvy_m_s": final_dv[1],
        "dvz_m_s": final_dv[2],
        "dv_norm_m_s": norm(final_dv),

        "initial_vx_raw_m_s": setup.v0_raw_m_s[0],
        "initial_vy_raw_m_s": setup.v0_raw_m_s[1],
        "initial_vz_raw_m_s": setup.v0_raw_m_s[2],

        "optimized_vx_raw_m_s": setup.v0_raw_m_s[0] + final_dv[0],
        "optimized_vy_raw_m_s": setup.v0_raw_m_s[1] + final_dv[1],
        "optimized_vz_raw_m_s": setup.v0_raw_m_s[2] + final_dv[2],

        "start_x_raw_m": setup.r0_raw_m[0],
        "start_y_raw_m": setup.r0_raw_m[1],
        "start_z_raw_m": setup.r0_raw_m[2],
        "start_vx_raw_m_s": setup.v0_raw_m_s[0],
        "start_vy_raw_m_s": setup.v0_raw_m_s[1],
        "start_vz_raw_m_s": setup.v0_raw_m_s[2],

        "target_x_raw_m": setup.target_r_raw_m[0],
        "target_y_raw_m": setup.target_r_raw_m[1],
        "target_z_raw_m": setup.target_r_raw_m[2],
        "target_vx_raw_m_s": setup.target_v_raw_m_s[0],
        "target_vy_raw_m_s": setup.target_v_raw_m_s[1],
        "target_vz_raw_m_s": setup.target_v_raw_m_s[2],

        "final_x_raw_m": final_r[0],
        "final_y_raw_m": final_r[1],
        "final_z_raw_m": final_r[2],
        "final_vx_raw_m_s": final_v[0],
        "final_vy_raw_m_s": final_v[1],
        "final_vz_raw_m_s": final_v[2],

        "final_miss_km": final_ev["miss_m"] / 1000.0,
        "final_relv_m_s": final_ev["relv_m_s"],
        "final_status": final_ev["status"],
        "final_message": final_ev["message"],

        "history_csv": str(history_path),
    }


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    fields: list[str] = []
    for row in rows:
        for k in row:
            if k not in fields:
                fields.append(k)

    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main_cli() -> int:
    p = argparse.ArgumentParser(description="Optimize candidate legs with Principia N-body impulse server.")
    p.add_argument("--candidate-seed", type=Path, required=True)
    p.add_argument("--rank", type=int, required=True)
    p.add_argument("--legs", nargs="*", type=int, default=None)

    p.add_argument("--bsp", type=Path, required=True)
    p.add_argument("--tpc", type=Path, required=True)
    p.add_argument("--plugin-b64", type=Path, required=True)
    p.add_argument("--server", default="principia_impulsive_particle_server")
    p.add_argument("--sampler", default="sample_principia_ephemeris")

    p.add_argument("--central-body", default="Sun")
    p.add_argument("--central-mu-km3-s2", type=float, required=True)

    p.add_argument("--plugin-base-et-s", type=float, default=81.65168640136972)
    p.add_argument("--raw-origin-body", default="Sun")
    p.add_argument("--raw-cache-dir", type=Path, default=Path("data/runs/frame_debug/raw_cache"))

    p.add_argument("--transform", default="+Z,-X,+Y")
    p.add_argument("--buffer-days", type=float, default=0.235)

    p.add_argument("--max-nfev", type=int, default=80)
    p.add_argument("--pos-scale-km", type=float, default=1000.0)
    p.add_argument("--dv-x-scale-m-s", type=float, default=100.0)

    p.add_argument("--work-dir", type=Path, required=True)
    p.add_argument("--output-csv", type=Path, required=True)

    args = p.parse_args()

    spice.furnsh(str(args.tpc))
    spice.furnsh(str(args.bsp))

    row = read_candidate(args.candidate_seed, args.rank)
    n_legs = int(row["n_legs"])

    legs = args.legs if args.legs else list(range(1, n_legs + 1))

    print("=== NATIVE LEG OPTIMIZER ===")
    print(f"candidate : {row.get('candidate_id')} rank={args.rank}")
    print(f"sequence  : {row.get('sequence_bodies')}")
    print(f"legs      : {legs}")
    print(f"transform : {args.transform}")
    print(f"buffer    : {args.buffer_days} d")
    print("")

    results = []

    with PrincipiaImpulseServer(args.server, args.plugin_b64) as server:
        for leg in legs:
            print(f"\n--- LEG {leg}/{n_legs} ---")
            setup = build_leg_setup(row, leg, args)
            result = optimize_leg(
                setup=setup,
                server=server,
                work_dir=args.work_dir,
                max_nfev=args.max_nfev,
                pos_scale_km=args.pos_scale_km,
                dv_x_scale_m_s=args.dv_x_scale_m_s,
            )
            results.append(result)

            print(
                f"[RESULT] leg {leg} {setup.dep_body}->{setup.arr_body}: "
                f"dv={result['dv_norm_m_s']:.3f} m/s "
                f"miss={result['final_miss_km']:.6f} km "
                f"relv={result['final_relv_m_s']:.3f} m/s "
                f"success={result['solver_success']}"
            )

    write_rows(args.output_csv, results)

    print(f"\n[OK] summary: {args.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main_cli())
