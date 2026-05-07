#!/usr/bin/env python3
"""
principia_true_snapshot.py

Calcula a velocidade N-body verdadeira usando Diferenças Finitas.
Resolve o bug da "Velocidade Envenenada" do KSP stock.
"""

import krpc
import time
import math
import json
from pathlib import Path

G_SI = 6.67430e-11

def main():
    conn = krpc.connect(name="True_Snapshot")
    sc = conn.space_center
    
    # GARANTIR QUE O WARP É 1x (Tempo Real)
    sc.rails_warp_factor = 0
    sc.physics_warp_factor = 0
    time.sleep(1) # Espera a física acalmar
    
    bodies = sc.bodies
    ref_name = "Sun"
    ref_body = bodies[ref_name]
    frame = ref_body.non_rotating_reference_frame

    print("Coletando Posição 1...")
    ut1 = sc.ut
    pos1 = {name: b.position(frame) for name, b in bodies.items()}
    
    # Esperar 10 segundos no jogo
    print("Aguardando 10 segundos in-game para Derivada Numérica...")
    while sc.ut < ut1 + 10.0:
        time.sleep(0.1)
        
    print("Coletando Posição 2...")
    ut2 = sc.ut
    pos2 = {name: b.position(frame) for name, b in bodies.items()}
    
    dt = ut2 - ut1
    print(f"Delta T Real: {dt:.4f} segundos")

    export_bodies = {}
    catalog_bodies = {}

    for name, b in bodies.items():
        p1 = pos1[name]
        p2 = pos2[name]
        
        # O Pulo do Gato: Velocidade verdadeira calculada manualmente!
        vx = (p2[0] - p1[0]) / dt
        vy = (p2[1] - p1[1]) / dt
        vz = (p2[2] - p1[2]) / dt
        
        # Usamos p1 como a posição oficial no tempo ut1
        export_bodies[name] = [
            ut1, p1[0], p1[1], p1[2], vx, vy, vz
        ]
        
        mu = b.gravitational_parameter
        catalog_bodies[name] = {
            "mu_m3_s2": mu,
            "mass_kg": mu / G_SI
        }

    # Montar JSON Legado para o REBOUND ler
    payload = {
        "reference_body": ref_name,
        "start_ut_seconds": ut1,
        "et_offset_seconds": 0.0,
        "body_catalog": {"bodies": catalog_bodies},
        "ephemerides": {name: {"states": [state]} for name, state in export_bodies.items()}
    }

    out_file = Path("data/true_snapshot.json")
    out_file.parent.mkdir(exist_ok=True)
    out_file.write_text(json.dumps(payload, indent=2))
    
    print(f"[SUCESSO] Snapshot Verdadeiro salvo em {out_file}")

if __name__ == "__main__":
    main()