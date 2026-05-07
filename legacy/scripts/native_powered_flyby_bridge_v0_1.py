#!/usr/bin/env python3
"""
native_powered_flyby_bridge_v0_1.py

Native powered-flyby bridge optimizer using principia_impulsive_particle_server.

Purpose:
  Replace a patched flyby mismatch at an intermediate body with a real native
  Principia arc containing one impulsive burn.

Variables optimized:
  x = [dv0_x, dv0_y, dv0_z, burn_offset_s, burn_dvx, burn_dvy, burn_dvz]
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import spiceypy as spice
from scipy.optimize import least_squares

from spice_lambert_mga_v0_1 import DAY_S, BodyInfo, load_body_catalog, norm_name
from native_corrected_flyby_audit_v0_1 import (
    get_float,
    leg_rows_by_index,
    norm,
    read_candidate,
    reconstruct_leg_start_position_raw,
    spk_body_state_raw,
    vector_from_summary,
)


class PrincipiaImpulseServer:
    """Cliente para interagir com o daemon C++ via TSV over stdin/stdout."""
    def __init__(self, executable: str, plugin_b64: Path):
        cmd = executable.split() + [str(plugin_b64)]
        self.proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            bufsize=1 # Line-buffered
        )
        
        # Aguarda a mensagem de READY
        ready_line = self.proc.stdout.readline().strip()
        if not ready_line.startswith("READY"):
            raise RuntimeError(f"Falha ao iniciar o servidor Principia: {ready_line}")

    def propagate(
        self, req_id: str, t0: float, burn_t: float, t1: float,
        r0: np.ndarray, v0: np.ndarray, burn_dv: np.ndarray
    ) -> Dict[str, str]:
        
        # Protocolo TSV: PROP -> t0 -> tb -> t1 -> r0 -> v0 -> burn_dv
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
            return {"status": "crash", "message": "Servidor C++ fechou inesperadamente"}
        
        parts = resp.split('\t')
        if parts[0] == "OK":
            return {
                "status": "ok",
                "id": parts[1],
                "t0_s": parts[2], "burn_t_s": parts[3], "t1_s": parts[4],
                "burn_x_m": parts[5], "burn_y_m": parts[6], "burn_z_m": parts[7],
                "burn_vx_before_m_s": parts[8], "burn_vy_before_m_s": parts[9], "burn_vz_before_m_s": parts[10],
                "burn_vx_after_m_s": parts[11], "burn_vy_after_m_s": parts[12], "burn_vz_after_m_s": parts[13],
                "final_x_m": parts[14], "final_y_m": parts[15], "final_z_m": parts[16],
                "final_vx_m_s": parts[17], "final_vy_m_s": parts[18], "final_vz_m_s": parts[19],
            }
        elif parts[0] == "ERR":
            return {"status": "error", "message": parts[2] if len(parts) > 2 else "Erro desconhecido"}
        else:
            return {"status": "unknown", "message": resp}

    def close(self):
        if self.proc.poll() is None:
            self.proc.stdin.write("QUIT\n")
            self.proc.stdin.flush()
            self.proc.wait(timeout=2)


def vec_from_output(row: Dict[str, str], prefix: str) -> Tuple[np.ndarray, np.ndarray]:
    if prefix == "burn":
        r = np.asarray([float(row["burn_x_m"]), float(row["burn_y_m"]), float(row["burn_z_m"])], dtype=float)
        v = np.asarray([
            float(row["burn_vx_after_m_s"]),
            float(row["burn_vy_after_m_s"]),
            float(row["burn_vz_after_m_s"]),
        ], dtype=float)
        return r, v
    if prefix == "burn_before":
        r = np.asarray([float(row["burn_x_m"]), float(row["burn_y_m"]), float(row["burn_z_m"])], dtype=float)
        v = np.asarray([
            float(row["burn_vx_before_m_s"]),
            float(row["burn_vy_before_m_s"]),
            float(row["burn_vz_before_m_s"]),
        ], dtype=float)
        return r, v
    if prefix == "final":
        r = np.asarray([float(row["final_x_m"]), float(row["final_y_m"]), float(row["final_z_m"])], dtype=float)
        v = np.asarray([
            float(row["final_vx_m_s"]),
            float(row["final_vy_m_s"]),
            float(row["final_vz_m_s"]),
        ], dtype=float)
        return r, v
    raise ValueError(prefix)


def choose_rp_km(info: BodyInfo, altitude_km: float | None, scale: float) -> float:
    if info.radius_km is None:
        raise ValueError("body radius missing; pass metadata/body-catalog with radius")
    if altitude_km is not None:
        return info.radius_km + altitude_km
    return info.radius_km * scale


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--candidate-csv", type=Path, required=True)
    p.add_argument("--rank", type=int, default=1)
    p.add_argument("--leg-summary-csv", type=Path, required=True)
    p.add_argument("--flyby-index", type=int, required=True, help="Sequence index of flyby body, e.g. 1 for KERBIN,EVE,KERBIN.")
    p.add_argument("--bsp", type=Path, required=True)
    p.add_argument("--tpc", type=Path, required=True)
    p.add_argument("--initial-dv0-m-s", nargs=3, type=float, default=None)
    p.add_argument("--initial-result-json", type=Path, default=None)
    p.add_argument("--metadata", type=Path, action="append", default=[])
    p.add_argument("--body-catalog", type=Path, action="append", default=[])
    p.add_argument("--plugin-b64", type=Path, required=True)
    p.add_argument("--central-body", default="Sun")
    p.add_argument("--central-mu-km3-s2", type=float, default=None)
    p.add_argument("--sequence", nargs="+", required=True)
    p.add_argument("--transform", default="+Z,-X,+Y")
    p.add_argument("--validator", default="principia_impulsive_particle_server")
    p.add_argument("--work-dir", type=Path, required=True)
    p.add_argument("--output-json", type=Path, required=True)
    p.add_argument("--output-history-csv", type=Path, required=True)
    p.add_argument("--periapsis-altitude-km", type=float, default=None)
    p.add_argument("--periapsis-radius-scale", type=float, default=1.05)
    p.add_argument("--max-nfev", type=int, default=80)
    p.add_argument("--dv0-bound-m-s", type=float, default=1000.0)
    p.add_argument("--burn-dv-bound-m-s", type=float, default=2000.0)
    p.add_argument("--burn-offset-bound-days", type=float, default=0.30)
    p.add_argument("--initial-burn-offset-s", type=float, default=0.0)
    p.add_argument("--initial-burn-dv-m-s", nargs=3, type=float, default=None)
    p.add_argument("--pos-scale-km", type=float, default=1000.0)
    p.add_argument("--vel-scale-km-s", type=float, default=0.1)
    p.add_argument("--rp-scale-km", type=float, default=100.0)
    p.add_argument("--radial-vel-scale-km-s", type=float, default=0.1)
    return p.parse_args()


def main() -> int:
    args = parse_args()

    spice.furnsh(str(args.tpc))
    spice.furnsh(str(args.bsp))

    sequence = [norm_name(x) for x in args.sequence]
    if args.flyby_index < 1 or args.flyby_index >= len(sequence) - 1:
        raise SystemExit("flyby-index precisa ter perna de entrada e saída")

    flyby_body = sequence[args.flyby_index]
    incoming_leg = args.flyby_index
    outgoing_leg = args.flyby_index + 1

    candidate = read_candidate(args.candidate_csv, args.rank)
    leg_rows = leg_rows_by_index(args.leg_summary_csv)

    body_info = load_body_catalog([*args.body_catalog, *args.metadata])
    central = norm_name(args.central_body)
    central_mu = args.central_mu_km3_s2
    if central_mu is None:
        info_c = body_info.get(central)
        if info_c and info_c.mu_km3_s2:
            central_mu = info_c.mu_km3_s2

    flyby_info = body_info.get(flyby_body, BodyInfo())
    rp_target_km = choose_rp_km(
        flyby_info,
        altitude_km=args.periapsis_altitude_km,
        scale=args.periapsis_radius_scale,
    )

    row_in = leg_rows[incoming_leg]
    row_out = leg_rows[outgoing_leg]

    t0 = get_float(row_in, "t_start_s")
    t1 = get_float(row_out, "t_start_s")
    t_event = get_float(candidate, f"event{args.flyby_index}_{flyby_body}_et_s")

    r0_raw = reconstruct_leg_start_position_raw(
        candidate=candidate, sequence=sequence, leg=incoming_leg,
        central_body=central, central_mu_km3_s2=float(central_mu),
        t_start_s=t0, transform_spec=args.transform,
    )
    v0_guess_raw = vector_from_summary(row_in, "optimized")

    target_r_raw = reconstruct_leg_start_position_raw(
        candidate=candidate, sequence=sequence, leg=outgoing_leg,
        central_body=central, central_mu_km3_s2=float(central_mu),
        t_start_s=t1, transform_spec=args.transform,
    )
    target_v_raw = vector_from_summary(row_out, "optimized")

    if args.initial_burn_dv_m_s is not None:
        burn_dv0 = np.asarray(args.initial_burn_dv_m_s, dtype=float)
    else:
        burn_dv0 = np.zeros(3, dtype=float)

    x0 = np.zeros(7, dtype=float)

    if args.initial_result_json:
        prev = json.load(open(args.initial_result_json))
        seed = prev["best"]["x"]
        x0[:] = np.asarray(seed, dtype=float)
    else:
        if args.initial_dv0_m_s is not None:
            x0[0:3] = np.asarray(args.initial_dv0_m_s, dtype=float)
        x0[3] = float(args.initial_burn_offset_s)
        if args.initial_burn_dv_m_s is not None:
            x0[4:7] = np.asarray(args.initial_burn_dv_m_s, dtype=float)

    lower = np.asarray([
        -args.dv0_bound_m_s, -args.dv0_bound_m_s, -args.dv0_bound_m_s,
        -args.burn_offset_bound_days * DAY_S,
        -args.burn_dv_bound_m_s, -args.burn_dv_bound_m_s, -args.burn_dv_bound_m_s,
    ], dtype=float)
    upper = -lower

    args.work_dir.mkdir(parents=True, exist_ok=True)
    args.output_history_csv.parent.mkdir(parents=True, exist_ok=True)
    eval_counter = {"n": 0}
    best: Dict[str, Any] = {"score": float("inf")}

    print("=== NATIVE POWERED FLYBY BRIDGE V0.1 ===")
    print(f"candidate: {args.candidate_csv} rank={args.rank}")
    print(f"flyby   : {flyby_body} index={args.flyby_index}")
    print(f"arc     : t0={t0:.6f} burn≈{t_event:.6f} t1={t1:.6f}")
    print(f"rp target: {rp_target_km:.6f} km")
    
    # Inicia o servidor daemon
    server = PrincipiaImpulseServer(args.validator, args.plugin_b64)

    try:
        with args.output_history_csv.open("w", newline="") as hist:
            hw = csv.writer(hist)
            hw.writerow([
                "eval", "dv0x_m_s", "dv0y_m_s", "dv0z_m_s", "burn_offset_s",
                "burn_dvx_m_s", "burn_dvy_m_s", "burn_dvz_m_s",
                "dv0_norm_m_s", "burn_dv_norm_m_s",
                "final_pos_err_km", "final_vel_err_m_s",
                "burn_radius_km", "burn_altitude_km", "burn_rp_err_km", "burn_radial_v_km_s",
                "status", "message",
            ])

            def residual(x: np.ndarray) -> np.ndarray:
                eval_counter["n"] += 1
                dv0 = np.asarray(x[0:3], dtype=float)
                burn_t = t_event + float(x[3])
                burn_dv = np.asarray(x[4:7], dtype=float)

                row_id = f"flyby_{flyby_body}_eval{eval_counter['n']}"
                
                # Chamada direta para a RAM via Pipes
                row = server.propagate(row_id, t0, burn_t, t1, r0_raw, v0_guess_raw + dv0, burn_dv)

                if row.get("status") != "ok":
                    status = row.get("status", "native_error")
                    msg = row.get("message", "")
                    res = np.ones(8) * 1e9
                    hw.writerow([eval_counter["n"], *x[:3], x[3], *x[4:7], norm(dv0), norm(burn_dv), "", "", "", "", "", "", status, msg])
                    hist.flush()
                    print(f"[eval {eval_counter['n']:03d}] ERROR native {status} {msg}")
                    return res

                final_r, final_v = vec_from_output(row, "final")
                burn_r, burn_v_after = vec_from_output(row, "burn")
                _burn_r2, burn_v_before = vec_from_output(row, "burn_before")

                body_r_burn, body_v_burn = spk_body_state_raw(flyby_body, burn_t, central, args.transform)
                rel_r = burn_r - body_r_burn
                rel_v_before = burn_v_before - body_v_burn
                rel_v_after = burn_v_after - body_v_burn

                radial_v_km_s = float(np.dot(rel_r, rel_v_before) / max(norm(rel_r), 1e-9)) / 1000.0
                burn_radius_km = norm(rel_r) / 1000.0
                burn_alt_km = burn_radius_km - float(flyby_info.radius_km or 0.0)
                rp_err_km = burn_radius_km - rp_target_km
                radial_v_km_s = float(np.dot(rel_r, rel_v_after) / max(norm(rel_r), 1e-9)) / 1000.0

                pos_err_km_vec = (final_r - target_r_raw) / 1000.0
                vel_err_km_s_vec = (final_v - target_v_raw) / 1000.0
                final_pos_err_km = norm(pos_err_km_vec)
                final_vel_err_m_s = norm(final_v - target_v_raw)

                res = np.concatenate([
                    pos_err_km_vec / args.pos_scale_km,
                    vel_err_km_s_vec / args.vel_scale_km_s,
                    np.asarray([rp_err_km / args.rp_scale_km], dtype=float),
                    np.asarray([radial_v_km_s / args.radial_vel_scale_km_s], dtype=float),
                ])
                score = float(np.dot(res, res))
                
                if score < best.get("score", float("inf")):
                    best.clear()
                    best.update({
                        "score": score,
                        "x": np.asarray(x, dtype=float).copy(),
                        "final_pos_err_km": final_pos_err_km,
                        "final_vel_err_m_s": final_vel_err_m_s,
                        "burn_radius_km": burn_radius_km,
                        "burn_altitude_km": burn_alt_km,
                        "rp_err_km": rp_err_km,
                        "radial_v_km_s": radial_v_km_s,
                        "burn_t_s": burn_t,
                        "dv0_norm_m_s": norm(dv0),
                        "burn_dv_norm_m_s": norm(burn_dv),
                    })

                hw.writerow([
                    eval_counter["n"],
                    *[f"{v:.17g}" for v in dv0],
                    f"{x[3]:.17g}",
                    *[f"{v:.17g}" for v in burn_dv],
                    f"{norm(dv0):.17g}",
                    f"{norm(burn_dv):.17g}",
                    f"{final_pos_err_km:.17g}",
                    f"{final_vel_err_m_s:.17g}",
                    f"{burn_radius_km:.17g}",
                    f"{burn_alt_km:.17g}",
                    f"{rp_err_km:.17g}",
                    f"{radial_v_km_s:.17g}",
                    "ok",
                    "",
                ])
                hist.flush()

                # Print clean, single line per eval (sem falhas de disco)
                print(
                    f"[eval {eval_counter['n']:03d}] "
                    f"pos={final_pos_err_km:10.3f} km "
                    f"vel={final_vel_err_m_s:9.3f} m/s "
                    f"rp={burn_radius_km:9.3f} km "
                    f"rperr={rp_err_km:9.3f} km "
                    f"vr={radial_v_km_s:8.4f} km/s "
                    f"dv0={norm(dv0):8.3f} m/s "
                    f"burn={norm(burn_dv):8.3f} m/s "
                    f"dt={x[3]:9.1f} s"
                )
                return res

            sol = least_squares(
                residual,
                x0,
                bounds=(lower, upper),
                max_nfev=args.max_nfev,
                x_scale="jac",
                ftol=1e-8,
                xtol=1e-8,
                gtol=1e-8,
                verbose=0,
            )

    finally:
        # Garante que o Daemon C++ seja desligado em caso de erro ou sucesso
        server.close()

    result = {
        "candidate_csv": str(args.candidate_csv),
        "rank": args.rank,
        "leg_summary_csv": str(args.leg_summary_csv),
        "sequence": sequence,
        "flyby_body": flyby_body,
        "flyby_index": args.flyby_index,
        "incoming_leg": incoming_leg,
        "outgoing_leg": outgoing_leg,
        "transform": args.transform,
        "rp_target_km": rp_target_km,
        "event_time_s": t_event,
        "t0_s": t0,
        "t1_s": t1,
        "solver_success": bool(sol.success),
        "solver_message": str(sol.message),
        "nfev": int(sol.nfev),
        "x": [float(v) for v in sol.x],
        "best": {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in best.items()},
        "history_csv": str(args.output_history_csv),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n=== POWERED FLYBY BRIDGE RESULT ===")
    print(f"success        : {sol.success} {sol.message}")
    print(f"nfev           : {sol.nfev}")
    if best:
        print(f"best pos err km: {best.get('final_pos_err_km')}")
        print(f"best vel err m/s: {best.get('final_vel_err_m_s')}")
        print(f"best burn dv m/s: {best.get('burn_dv_norm_m_s')}")
        print(f"best dv0 m/s    : {best.get('dv0_norm_m_s')}")
        print(f"best burn_t_s   : {best.get('burn_t_s')}")
        print(f"best burn alt km: {best.get('burn_altitude_km')}")
        print(f"best radial v km/s: {best.get('radial_v_km_s')}")
    print(f"[OK] result : {args.output_json}")
    print(f"[OK] history: {args.output_history_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())