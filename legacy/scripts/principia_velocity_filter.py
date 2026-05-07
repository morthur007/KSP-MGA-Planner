#!/usr/bin/env python3
"""
principia_velocity_filter.py

Captura 7 amostras ao redor de um t0 e aplica um ajuste polinomial (Savitzky-Golay)
para extrair a velocidade N-body verdadeira, ignorando o jitter da engine da Unity.
"""

import krpc
import time
import json
import argparse
import numpy as np
from pathlib import Path

G_SI = 6.67430e-11

def main():
    p = argparse.ArgumentParser(description="Extrator de Velocidade N-Body via Polinômio")
    p.add_argument("--dt", type=float, required=True, help="Delta T entre as amostras (ex: 10, 30, 120)")
    p.add_argument("--output", type=Path, required=True, help="Caminho do JSON de saída")
    args = p.parse_args()

    conn = krpc.connect(name=f"Velocity_Filter_dt{args.dt}")
    sc = conn.space_center
    
    # Travar o jogo em velocidade 1x
    sc.rails_warp_factor = 0
    sc.physics_warp_factor = 0
    time.sleep(1) # Aguarda a física assentar

    bodies = sc.bodies
    ref_name = "Sun"
    ref_body = bodies[ref_name]
    frame = ref_body.non_rotating_reference_frame

    print(f"Iniciando captura de 7 amostras com dt = {args.dt}s...")
    
    samples = []
    start_ut = sc.ut

    # Coleta de 7 pontos: t0-3dt até t0+3dt
    for i in range(7):
        target_ut = start_ut + (i * args.dt)
        
        # Segura o código até o relógio do jogo atingir o target
        while sc.ut < target_ut:
            time.sleep(0.01)
            
        current_ut = sc.ut
        positions = {name: b.position(frame) for name, b in bodies.items()}
        samples.append((current_ut, positions))
        print(f"  [Amostra {i+1}/7] UT: {current_ut:.2f}")

    print("\nAplicando Filtro Polinomial...")
    
    # O momento central é o índice 3 (4ª amostra)
    t_mid = samples[3][0]
    t_array = np.array([s[0] for s in samples])
    
    # Centralizar o tempo em 0 melhora drasticamente a precisão do polyfit
    t_centered = t_array - t_mid

    export_bodies = {}
    catalog_bodies = {}

    for name, b in bodies.items():
        x_arr = np.array([s[1][name][0] for s in samples])
        y_arr = np.array([s[1][name][1] for s in samples])
        z_arr = np.array([s[1][name][2] for s in samples])

        # Ajuste polinomial de 3º grau (equivalente a Savitzky-Golay de 7 pontos)
        px = np.polyfit(t_centered, x_arr, 3)
        py = np.polyfit(t_centered, y_arr, 3)
        pz = np.polyfit(t_centered, z_arr, 3)

        # Derivada analítica
        vx_poly = np.polyder(px)
        vy_poly = np.polyder(py)
        vz_poly = np.polyder(pz)

        # Avaliar a derivada no tempo central (t_centered = 0)
        vx = np.polyval(vx_poly, 0.0)
        vy = np.polyval(vy_poly, 0.0)
        vz = np.polyval(vz_poly, 0.0)

        # Avaliar a posição alisada (filtra microsaltos espaciais também)
        sx = np.polyval(px, 0.0)
        sy = np.polyval(py, 0.0)
        sz = np.polyval(pz, 0.0)

        export_bodies[name] = {
            "states": [[t_mid, sx, sy, sz, vx, vy, vz]]
        }

        mu = b.gravitational_parameter
        catalog_bodies[name] = {
            "mu_m3_s2": mu,
            "mass_kg": mu / G_SI
        }

    # Montagem do JSON no formato canônico
    payload = {
        "reference_body": ref_name,
        "start_ut_seconds": t_mid,
        "et_offset_seconds": 0.0,
        "body_catalog": {"bodies": catalog_bodies},
        "ephemerides": export_bodies
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))
    print(f"[SUCESSO] Snapshot filtrado gravado em: {args.output}")

if __name__ == "__main__":
    main()