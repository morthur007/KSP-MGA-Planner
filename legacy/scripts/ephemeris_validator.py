#!/usr/bin/env python3
"""
ephemeris_validator.py

Reconciliação de Modelos (KSP vs REBOUND).
Inclui modo de debug profundo para inspecionar os vetores de estado crus.
"""

import argparse
import csv
import math
from pathlib import Path
from collections import defaultdict

try:
    import spiceypy as spice
except ImportError:
    raise SystemExit("Instale o spiceypy: pip install spiceypy")

def norm3(v):
    return math.sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2])

def main():
    p = argparse.ArgumentParser(description="Validador de Efemérides KSP vs REBOUND")
    p.add_argument("--ksp-csv", type=Path, required=True, help="states.csv do kRPC")
    p.add_argument("--bsp", type=Path, required=True, help="Kernel .bsp do REBOUND")
    p.add_argument("--tpc", type=Path, required=True, help="Kernel .ids.tpc de mapeamento")
    p.add_argument("--central-body", default="SUN", help="Nome do corpo central (ex: SUN, KERBOL)")
    p.add_argument("--debug-body", type=str, default=None, help="Nome do corpo para imprimir log detalhado de coordenadas X,Y,Z")
    args = p.parse_args()

    spice.furnsh(str(args.tpc))
    spice.furnsh(str(args.bsp))

    errors = defaultdict(lambda: {"samples": 0, "max_pos_err_m": 0.0, "last_pos_err_m": 0.0, "max_vel_err_m_s": 0.0})
    debug_trace = []

    print(f"Comparando {args.ksp_csv.name} com {args.bsp.name}...\n")

    with args.ksp_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            body = row["body"]
            
            if not row["x_m"] or row["read_error"]:
                continue

            et_seconds = float(row["et_seconds"])
            ut_ksp = float(row["actual_ut_s"])
            
            # Posição (m) e Velocidade (m/s) cruas do KSP
            pos_ksp = (float(row["x_m"]), float(row["y_m"]), float(row["z_m"]))
            vel_ksp = (float(row["vx_m_s"]), float(row["vy_m_s"]), float(row["vz_m_s"]))

            try:
                spice_name = body.upper().replace(" ", "_").replace("-", "_")
                center_name = args.central_body.upper().replace(" ", "_")
                
                # SPICE retorna km e km/s
                state_spice, _ = spice.spkezr(spice_name, et_seconds, "J2000", "NONE", center_name)
                
                # Convertendo SPICE para metros e m/s
                pos_reb = (state_spice[0]*1000, state_spice[1]*1000, state_spice[2]*1000)
                vel_reb = (state_spice[3]*1000, state_spice[4]*1000, state_spice[5]*1000)

                err_pos = norm3((pos_ksp[0]-pos_reb[0], pos_ksp[1]-pos_reb[1], pos_ksp[2]-pos_reb[2]))
                err_vel = norm3((vel_ksp[0]-vel_reb[0], vel_ksp[1]-vel_reb[1], vel_ksp[2]-vel_reb[2]))

                # Gravar trace se for o corpo de debug escolhido
                if args.debug_body and body.upper() == args.debug_body.upper():
                    debug_trace.append({
                        "ut": ut_ksp, "et": et_seconds,
                        "pos_ksp": pos_ksp, "pos_reb": pos_reb, "err_m": err_pos
                    })

                stats = errors[body]
                stats["samples"] += 1
                stats["last_pos_err_m"] = err_pos
                if err_pos > stats["max_pos_err_m"]:
                    stats["max_pos_err_m"] = err_pos
                if err_vel > stats["max_vel_err_m_s"]:
                    stats["max_vel_err_m_s"] = err_vel

            except spice.stypes.SpiceyError as e:
                if body.upper() != args.central_body.upper():
                    pass 

    spice.kclear()

    # --- SESSÃO DE DEBUG VISUAL ---
    if args.debug_body and debug_trace:
        print(f"=== DEBUG CRU: {args.debug_body.upper()} ===")
        print("Mostrando os 3 primeiros dias e o último dia para verificação humana:\n")
        
        # Pega as 3 primeiras amostras e a última
        samples_to_show = debug_trace[:3]
        if len(debug_trace) > 3:
            samples_to_show.append(debug_trace[-1])

        for i, tr in enumerate(samples_to_show):
            if i == 3: print("... [Saltando para o final da simulação] ...")
            
            print(f"Tempo UT (Jogo) : {tr['ut']:.2f} s  | Tempo ET (SPICE) : {tr['et']:.2f} s")
            print(f"  KSP (X, Y, Z) : {tr['pos_ksp'][0]:+18.2f}, {tr['pos_ksp'][1]:+18.2f}, {tr['pos_ksp'][2]:+18.2f} (metros)")
            print(f"  REB (X, Y, Z) : {tr['pos_reb'][0]:+18.2f}, {tr['pos_reb'][1]:+18.2f}, {tr['pos_reb'][2]:+18.2f} (metros)")
            print(f"  -> Desvio 3D  : {tr['err_m']:.3f} metros")
            print("-" * 80)
        print("\n")

    # --- TABELA FINAL ---
    print(f"{'Corpo':<15} | {'Amostras':<8} | {'Erro Pos Máx (km)':<18} | {'Erro Vel Máx (m/s)':<18} | {'Erro Final (km)':<16} | {'Status'}")
    print("-" * 105)
    
    for body, stats in sorted(errors.items()):
        max_err_km = stats["max_pos_err_m"] / 1000.0
        last_err_km = stats["last_pos_err_m"] / 1000.0
        max_vel_err = stats["max_vel_err_m_s"]
        
        status = "[PERFEITO]"
        if max_err_km > 5000:
            status = "[DESVIO GRAVE]"
        elif max_err_km > 50:
            status = "[DESVIO LEVE]"

        print(f"{body:<15} | {stats['samples']:<8} | {max_err_km:<18.3f} | {max_vel_err:<18.3f} | {last_err_km:<16.3f} | {status}")

if __name__ == "__main__":
    main()