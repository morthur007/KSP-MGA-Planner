#!/usr/bin/env python3
"""
native_optimize_candidate_legs_v0_1.py

Optimize each Lambert leg of an MGA candidate against the native Principia C++
particle validator. 

UPDATED: Now uses PrincipiaImpulseServer (Daemon) instead of cold-booting 
the validator for every evaluation, increasing speed by ~1000x.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import spiceypy as spice
from scipy.optimize import least_squares, root

from spice_lambert_mga_v0_1 import (
    DAY_S,
    load_body_catalog,
    norm_name,
)
from lambert_candidate_to_particle_leg_v0_1 import (
    kepler_universal_propagate,
    parse_transform,
    apply_transform,
)

class PrincipiaImpulseServer:
    """Cliente para interagir com o daemon C++ via TSV."""
    def __init__(self, executable: str, plugin_b64: Path):
        cmd = executable.split() + [str(plugin_b64)]
        self.proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1
        )
        ready_line = self.proc.stdout.readline().strip()
        if not ready_line.startswith("READY"):
            raise RuntimeError(f"Falha ao iniciar o servidor: {ready_line}")

    def propagate(self, req_id: str, t0: float, burn_t: float, t1: float, r0: np.ndarray, v0: np.ndarray, burn_dv: np.ndarray) -> dict:
        req = (
            f"PROP\t{req_id}\t{t0:.17g}\t{burn_t:.17g}\t{t1:.17g}\t"
            f"{r0[0]:.17g}\t{r0[1]:.17g}\t{r0[2]:.17g}\t"
            f"{v0[0]:.17g}\t{v0[1]:.17g}\t{v0[2]:.17g}\t"
            f"{burn_dv[0]:.17g}\t{burn_dv[1]:.17g}\t{burn_dv[2]:.17g}\n"
        )
        self.proc.stdin.write(req)
        self.proc.stdin.flush()
        resp = self.proc.stdout.readline().strip()
        if not resp:
            return {"status": "crash"}
        parts = resp.split('\t')
        if parts[0] == "OK":
            return {
                "status": "ok",
                "final_x_m": float(parts[14]), "final_y_m": float(parts[15]), "final_z_m": float(parts[16]),
                "final_vx_m_s": float(parts[17]), "final_vy_m_s": float(parts[18]), "final_vz_m_s": float(parts[19]),
            }
        return {"status": "error", "message": parts[2] if len(parts) > 2 else "Erro desconhecido"}

    def close(self):
        if self.proc.poll() is None:
            self.proc.stdin.write("QUIT\n")
            self.proc.stdin.flush()
            self.proc.wait(timeout=2)


@dataclass
class LegSetup:
    leg: int
    dep_body: str
    arr_body: str
    t_dep_s: float
    t_arr_s: float
    t_start_s: float
    t_end_s: float
    tof_s: float
    path: str
    r0_raw_m: np.ndarray
    v0_raw_m_s: np.ndarray
    target_r_raw_m: np.ndarray
    target_v_raw_m_s: np.ndarray


@dataclass
class Evaluation:
    status: str
    miss_m: float
    relv_m_s: float
    final_r_raw_m: np.ndarray
    final_v_raw_m_s: np.ndarray
    message: str = ""


def read_candidate(path: Path, rank: int) -> Dict[str, str]:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    return rows[rank - 1]


def get_float(row: Dict[str, str], key: str) -> float:
    return float(row[key])


def spk_state(body: str, et_s: float, central_body: str) -> np.ndarray:
    st, _ = spice.spkezr(norm_name(body), float(et_s), "J2000", "NONE", norm_name(central_body))
    return np.asarray(st, dtype=float)


def norm(v: np.ndarray) -> float:
    return float(np.linalg.norm(v))


def build_leg_setup(
    row: Dict[str, str], sequence: Sequence[str], leg: int,
    central_body: str, central_mu_km3_s2: float, buffer_days: float, transform_spec: str
) -> LegSetup:
    dep_i = leg - 1
    arr_i = leg
    dep_body = norm_name(sequence[dep_i])
    arr_body = norm_name(sequence[arr_i])

    t_dep = get_float(row, f"event{dep_i}_{dep_body}_et_s")
    t_arr = get_float(row, f"event{arr_i}_{arr_body}_et_s")
    tof = t_arr - t_dep

    buffer_s = buffer_days * DAY_S
    path_col = f"leg{leg}_path"
    path = row.get(path_col, "short").strip().lower() or "short"

    st_dep = spk_state(dep_body, t_dep, central_body)
    st_arr = spk_state(arr_body, t_arr, central_body)

    # LÊ A VELOCIDADE EXATA DO CSV GERADO PELO PYKEP
    v_start_km_s = np.array([
        get_float(row, f"leg{leg}_vdep_x_km_s"),
        get_float(row, f"leg{leg}_vdep_y_km_s"),
        get_float(row, f"leg{leg}_vdep_z_km_s")
    ])

    r_start_km, v_start_km_s = kepler_universal_propagate(st_dep[:3], v_start_km_s, buffer_s, central_mu_km3_s2)

    t_start = t_dep + buffer_s
    t_end = t_arr - buffer_s
    target_st = spk_state(arr_body, t_end, central_body)

    tr = parse_transform(transform_spec)
    r0_raw_m = apply_transform(r_start_km * 1000.0, tr)
    v0_raw_m_s = apply_transform(v_start_km_s * 1000.0, tr)
    target_r_raw_m = apply_transform(target_st[:3] * 1000.0, tr)
    target_v_raw_m_s = apply_transform(target_st[3:] * 1000.0, tr)

    return LegSetup(
        leg=leg, dep_body=dep_body, arr_body=arr_body,
        t_dep_s=t_dep, t_arr_s=t_arr, t_start_s=t_start, t_end_s=t_end,
        tof_s=tof, path=path,
        r0_raw_m=r0_raw_m, v0_raw_m_s=v0_raw_m_s,
        target_r_raw_m=target_r_raw_m, target_v_raw_m_s=target_v_raw_m_s,
    )


def evaluate_dv_daemon(setup: LegSetup, dv_m_s: np.ndarray, server: PrincipiaImpulseServer, eval_index: int) -> Evaluation:
    v = setup.v0_raw_m_s + np.asarray(dv_m_s, dtype=float)
    row_id = f"leg{setup.leg}_eval{eval_index}"

    # Hack do Impulso Zero: burn_t = t1, burn_dv = 0
    resp = server.propagate(row_id, setup.t_start_s, setup.t_end_s, setup.t_end_s, setup.r0_raw_m, v, np.zeros(3))

    if resp.get("status") != "ok":
        return Evaluation("error", 1.0e18, 1.0e18, np.zeros(3), np.zeros(3), resp.get("message", "crash"))

    final_r = np.array([resp["final_x_m"], resp["final_y_m"], resp["final_z_m"]])
    final_v = np.array([resp["final_vx_m_s"], resp["final_vy_m_s"], resp["final_vz_m_s"]])

    miss = norm(final_r - setup.target_r_raw_m)
    relv = norm(final_v - setup.target_v_raw_m_s)
    
    return Evaluation("ok", miss, relv, final_r, final_v, "")


def optimize_leg(setup: LegSetup, server: PrincipiaImpulseServer, work_dir: Path, method: str, max_nfev: int, xtol: float, ftol: float, gtol: float, dv_bound_m_s: float) -> dict:
    work_dir.mkdir(parents=True, exist_ok=True)
    history_path = work_dir / "history.csv"
    eval_counter = {"n": 0}
    best = {"miss_m": float("inf"), "dv": np.zeros(3)}

    with history_path.open("w", newline="") as hist:
        hw = csv.writer(hist)
        hw.writerow(["eval", "dvx_m_s", "dvy_m_s", "dvz_m_s", "dv_norm_m_s", "miss_km", "relv_m_s", "status", "message"])

        def residual(dv: np.ndarray) -> np.ndarray:
            eval_counter["n"] += 1
            ev = evaluate_dv_daemon(setup, dv, server, eval_counter["n"])
            dvn = norm(dv)
            
            hw.writerow([eval_counter["n"], f"{dv[0]:.17g}", f"{dv[1]:.17g}", f"{dv[2]:.17g}", f"{dvn:.17g}", f"{ev.miss_m / 1000.0:.17g}", f"{ev.relv_m_s:.17g}", ev.status, ev.message])
            hist.flush()

            if ev.status == "ok" and ev.miss_m < best["miss_m"]:
                best["miss_m"] = ev.miss_m

            print(f"[leg {setup.leg} eval {eval_counter['n']:03d}] miss={ev.miss_m/1000.0:12.3f} km relv={ev.relv_m_s:10.3f} m/s dv={dvn:9.3f} m/s")

            if ev.status != "ok": return np.asarray([1.0e12, 1.0e12, 1.0e12], dtype=float)
            return (ev.final_r_raw_m - setup.target_r_raw_m) / 1000.0

        x0 = np.zeros(3, dtype=float)
        sol = root(residual, x0, method="hybr", options={"xtol": xtol, "maxfev": max_nfev})
        x = np.asarray(sol.x, dtype=float)

    final_ev = evaluate_dv_daemon(setup, x, server, eval_counter["n"] + 1)

    return {
        "leg": setup.leg,
        "dep_body": setup.dep_body,
        "arr_body": setup.arr_body,
        "path": setup.path,
        "t_dep_s": setup.t_dep_s,
        "t_arr_s": setup.t_arr_s,
        "t_start_s": setup.t_start_s,
        "t_end_s": setup.t_end_s,
        "tof_days": setup.tof_s / DAY_S,

        "solver_success": bool(sol.success),
        "solver_message": str(sol.message),
        "n_eval": eval_counter["n"],

        "dvx_m_s": x[0],
        "dvy_m_s": x[1],
        "dvz_m_s": x[2],
        "dv_norm_m_s": norm(x),

        "initial_vx_m_s": setup.v0_raw_m_s[0],
        "initial_vy_m_s": setup.v0_raw_m_s[1],
        "initial_vz_m_s": setup.v0_raw_m_s[2],

        "optimized_vx_m_s": setup.v0_raw_m_s[0] + x[0],
        "optimized_vy_m_s": setup.v0_raw_m_s[1] + x[1],
        "optimized_vz_m_s": setup.v0_raw_m_s[2] + x[2],

        # NOVO: estado final real propagado pelo Principia.
        "final_x_m": final_ev.final_r_raw_m[0],
        "final_y_m": final_ev.final_r_raw_m[1],
        "final_z_m": final_ev.final_r_raw_m[2],
        "final_vx_m_s": final_ev.final_v_raw_m_s[0],
        "final_vy_m_s": final_ev.final_v_raw_m_s[1],
        "final_vz_m_s": final_ev.final_v_raw_m_s[2],

        # Opcional, mas útil para debug/audit.
        "target_x_m": setup.target_r_raw_m[0],
        "target_y_m": setup.target_r_raw_m[1],
        "target_z_m": setup.target_r_raw_m[2],
        "target_vx_m_s": setup.target_v_raw_m_s[0],
        "target_vy_m_s": setup.target_v_raw_m_s[1],
        "target_vz_m_s": setup.target_v_raw_m_s[2],

        "final_miss_km": final_ev.miss_m / 1000.0,
        "final_relv_m_s": final_ev.relv_m_s,
        "final_status": final_ev.status,
        "final_message": final_ev.message,
        "history_csv": str(history_path),
    }

def write_summary(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows: return
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--candidate-csv", type=Path, required=True)
    p.add_argument("--rank", type=int, default=1)
    p.add_argument("--bsp", type=Path, required=True)
    p.add_argument("--tpc", type=Path, required=True)
    p.add_argument("--metadata", type=Path, action="append", default=[])
    p.add_argument("--body-catalog", type=Path, action="append", default=[])
    p.add_argument("--plugin-b64", type=Path, required=True)
    p.add_argument("--central-body", default="Sun")
    p.add_argument("--central-mu-km3-s2", type=float, default=None)
    p.add_argument("--sequence", nargs="+", required=True)
    p.add_argument("--legs", type=int, nargs="+", default=None)
    p.add_argument("--buffer-days", type=float, default=0.235)
    p.add_argument("--transform", default="+Z,-X,+Y")
    p.add_argument("--validator", default="principia_impulsive_particle_server") # ATENÇÃO: Agora usa o Daemon!
    p.add_argument("--work-dir", type=Path, required=True)
    p.add_argument("--output-csv", type=Path, required=True)
    p.add_argument("--method", choices=["root-hybr"], default="root-hybr")
    p.add_argument("--max-nfev", type=int, default=80)
    p.add_argument("--xtol", type=float, default=1e-8)
    p.add_argument("--ftol", type=float, default=1e-8)
    p.add_argument("--gtol", type=float, default=1e-8)
    p.add_argument("--dv-bound-m-s", type=float, default=2000.0)
    return p.parse_args()

def main() -> int:
    args = parse_args()
    for k in [args.tpc, args.bsp]: spice.furnsh(str(k))

    sequence = [norm_name(x) for x in args.sequence]
    legs = args.legs if args.legs else list(range(1, len(sequence)))
    
    body_info = load_body_catalog([*args.body_catalog, *args.metadata])
    central = norm_name(args.central_body)
    central_mu = args.central_mu_km3_s2 or body_info.get(central).mu_km3_s2

    row = read_candidate(args.candidate_csv, args.rank)

    print("=== NATIVE LEG OPTIMIZER V0.2 (DAEMON MODE) ===")
    
    # Inicia o Servidor C++ UMA ÚNICA VEZ para toda a sequência!
    server = PrincipiaImpulseServer(args.validator, args.plugin_b64)
    
    try:
        results = []
        for leg in legs:
            print(f"\nPreparing leg {leg}: {sequence[leg-1]} -> {sequence[leg]}")
            setup = build_leg_setup(row, sequence, leg, central, float(central_mu), args.buffer_days, args.transform)
            leg_dir = args.work_dir / f"leg{leg}_{setup.dep_body}_to_{setup.arr_body}"
            
            result = optimize_leg(setup, server, leg_dir, args.method, args.max_nfev, args.xtol, args.ftol, args.gtol, args.dv_bound_m_s)
            results.append(result)
            write_summary(args.output_csv, results)
    finally:
        # Garante que a memória RAM é limpa ao final
        server.close()

    print(f"\n[OK] summary: {args.output_csv}")
    return 0

if __name__ == "__main__":
    sys.exit(main())