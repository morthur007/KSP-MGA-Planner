#!/usr/bin/env python3
"""
krpc_brute_force_extractor.py

O Extrator Abusivo. Ignora a API onde ela falha. 
Mede a aceleração física real dos planetas para descobrir a massa secreta do Principia.
"""

import krpc
import math
import time
import numpy as np

# Valores Esperados (O que o Vanilla/Patch dizem)
EXPECTED_INC = {"Laythe": 3.05, "Vall": 6.47, "Tylo": 356.2}
VANILLA_MU_JOOL = 2.8252800e14
VANILLA_MU_VALL = 2.0748150e11

def norm3(v):
    return math.sqrt(v[0]**2 + v[1]**2 + v[2]**2)

def main():
    print("Iniciando Invasão kRPC (Modo Brute-Force)...")
    try:
        conn = krpc.connect(name="Brute Force")
        sc = conn.space_center
    except Exception as e:
        print(f"Erro fatal de conexão: {e}")
        return

    bodies = sc.bodies

    print("\n" + "="*85)
    print("FASE 1: VERIFICAÇÃO DE SERVIÇOS DO PRINCIPIA (CORRIGIDO)")
    print("="*85)
    try:
        # A forma correta e abusiva de puxar todos os serviços
        services = [s.name for s in conn.krpc.get_services().services]
        if 'Principia' in services:
            print("[SUCESSO] Principia detectado injetado na memória do kRPC!")
        else:
            print("[FALHA] Serviço Principia oculto ou não exposto via kRPC RPC.")
    except Exception as e:
        print(f"Erro ao listar serviços: {e}")

    print("\n" + "="*85)
    print("FASE 2: DETECTOR DE INCLINAÇÃO VETORIAL (BYPASS DE RAILS)")
    print("="*85)
    print("Calculando inclinação pela força bruta da matriz 3D (Momento Angular)...")
    
    jool = bodies.get("Jool")
    if not jool:
        print("Jool não encontrado.")
        return

    frame = jool.non_rotating_reference_frame

    for name, expected in EXPECTED_INC.items():
        if name not in bodies: continue
        moon = bodies[name]
        
        try:
            # Posição e velocidade relativas a Jool
            r = moon.position(frame)
            v = moon.velocity(frame)
            
            # Momento angular h = r x v
            hx = r[1]*v[2] - r[2]*v[1]
            hy = r[2]*v[0] - r[0]*v[2]
            hz = r[0]*v[1] - r[1]*v[0]
            
            h_mag = norm3((hx, hy, hz))
            
            # Inclinação i = arccos(hz / |h|)
            # Nota: O referencial do kRPC pode ter Y ou Z como UP, ajustamos pegando o vetor normal principal
            inc_rad = math.acos(hz / h_mag) if h_mag > 0 else 0
            inc_deg = math.degrees(inc_rad) % 360
            
            # Ler o que a API pensa que é
            api_inc = math.degrees(moon.orbit.inclination) % 360
            
            print(f"{name:<10} | Inc. API (Mentira?): {api_inc:<10.3f} | Inc. Vetorial (Real): {inc_deg:<10.3f} | Patch Esperado: {expected}")
        except Exception as e:
            print(f"Erro calculando {name}: {e}")

    print("\n" + "="*85)
    print("FASE 3: A BALANÇA DE NEWTON (DESCOBRINDO A MASSA SECRETA DO PRINCIPIA)")
    print("="*85)
    print("Isso vai travar o tempo em 1x e medir a aceleração de Vall caindo em direção a Jool.")
    
    try:
        sc.rails_warp_factor = 0
        sc.physics_warp_factor = 0
        time.sleep(1) # Deixa a engine respirar
        
        vall = bodies["Vall"]
        
        print("Coletando Amostra 1...")
        t1 = sc.ut
        r1 = np.array(vall.position(frame))
        v1 = np.array(vall.velocity(frame))
        
        print("Aguardando 5 segundos de gravidade N-body pura...")
        while sc.ut < t1 + 5.0:
            time.sleep(0.1)
            
        print("Coletando Amostra 2...")
        t2 = sc.ut
        r2 = np.array(vall.position(frame))
        v2 = np.array(vall.velocity(frame))
        
        dt = t2 - t1
        
        # Aceleração a = dv/dt
        a_vec = (v2 - v1) / dt
        a_mag = np.linalg.norm(a_vec)
        
        # Raio médio
        r_mag = np.linalg.norm((r1 + r2) / 2)
        
        # Leis de Newton: a = mu_sistema / r^2  => mu_sistema = a * r^2
        mu_sistema_medido = a_mag * (r_mag**2)
        
        # O mu medido é a soma da massa de Jool + Vall
        mu_api_soma = jool.gravitational_parameter + vall.gravitational_parameter
        mu_vanilla_soma = VANILLA_MU_JOOL + VANILLA_MU_VALL
        
        print(f"\nDelta T Físico real: {dt:.4f} s")
        print(f"Distância Vall-Jool  : {r_mag/1000:,.2f} km")
        print(f"Aceleração Medida    : {a_mag:.6f} m/s²")
        print("-" * 50)
        print(f"Mu do Sistema (Balança Newton): {mu_sistema_medido:,.4e}")
        print(f"Mu do Sistema (API kRPC)      : {mu_api_soma:,.4e}")
        print(f"Mu do Sistema (Vanilla)       : {mu_vanilla_soma:,.4e}")
        
        diff = abs(mu_sistema_medido - mu_api_soma) / mu_api_soma * 100
        print(f"\nDivergência entre o Físico e a API: {diff:.6f}%")
        
        if diff > 0.05:
            print("[ALERTA VERMELHO] O Principia ESTÁ USANDO UMA MASSA DIFERENTE. A API ESTÁ MENTINDO!")
        else:
            print("[STATUS] A massa medida bate com a API. A física gravitacional primária está idêntica ao Vanilla.")
            print("         Se o erro de Vall continua enorme, Jool tem J2 (Achatamento) massivo, ou a ressonância explodiu por erro de fase inicial.")

    except Exception as e:
        print(f"Erro na balança de Newton: {e}")

if __name__ == "__main__":
    main()