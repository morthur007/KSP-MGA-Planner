#!/usr/bin/env python3
"""
relative_residuals_analyzer.py

Isola o drift heliocêntrico do drift orbital interno.
Calcula o erro absoluto e o erro relativo a um planeta pai especificado.
"""

import argparse
import csv
import math
from pathlib import Path
from collections import defaultdict

def norm3(v):
    return math.sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2])

def sub3(a, b):
    return (a[0]-b[0], a[1]-b[1], a[2]-b[2])

def main():
    p = argparse.ArgumentParser(description="Analisa o erro relativo de luas vs planeta pai")
    p.add_argument("--ksp-csv", type=Path, required=True, help="states.csv do KSP/kRPC")
    p.add_argument("--reb-csv", type=Path, required=True, help="rebound_states.csv do Nível A")
    p.add_argument("--parent", type=str, required=True, help="Nome do planeta pai (ex: Jool, Sarnus)")
    p.add_argument("--moons", type=str, nargs="+", required=True, help="Luas a analisar separadas por espaço (ex: Laythe Vall Tylo Bop Pol)")
    args = p.parse_args()

    print(f"Carregando dados Nível A e isolando matriz de {args.parent}...")

    # Ler estados do REBOUND
    reb_states = defaultdict(dict)
    with args.reb_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            et = float(row["et_seconds"])
            body = row["body"]
            reb_states[et][body] = (float(row["x_m"]), float(row["y_m"]), float(row["z_m"]))

    # Ler estados do KSP
    ksp_states = defaultdict(dict)
    with args.ksp_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row["x_m"] or row.get("read_error", "False").lower() in ("true", "1"):
                continue
            et = float(row["et_seconds"])
            body = row["body"]
            ksp_states[et][body] = (float(row["x_m"]), float(row["y_m"]), float(row["z_m"]))

    # Calcular erros
    results = defaultdict(lambda: {"abs_max": 0.0, "rel_max": 0.0, "samples": 0})
    
    parent_name = args.parent
    target_moons = set(args.moons)

    for et, ksp_data in ksp_states.items():
        if et not in reb_states: continue
        reb_data = reb_states[et]

        if parent_name not in ksp_data or parent_name not in reb_data:
            continue

        p_ksp = ksp_data[parent_name]
        p_reb = reb_data[parent_name]

        for body in target_moons:
            if body not in ksp_data or body not in reb_data:
                continue
                
            c_ksp = ksp_data[body]
            c_reb = reb_data[body]

            # 1. Erro Absoluto (Heliocêntrico)
            err_abs = norm3(sub3(c_ksp, c_reb))

            # 2. Erro Relativo (Matemática sugerida na sua especificação)
            # Δr_rel = (r_KSP_body - r_KSP_parent) - (r_REB_body - r_REB_parent)
            rel_ksp = sub3(c_ksp, p_ksp)
            rel_reb = sub3(c_reb, p_reb)
            err_rel = norm3(sub3(rel_ksp, rel_reb))

            stats = results[body]
            stats["samples"] += 1
            if err_abs > stats["abs_max"]: stats["abs_max"] = err_abs
            if err_rel > stats["rel_max"]: stats["rel_max"] = err_rel

    # Exibir Tabela de Diagnóstico
    print(f"\n=== Análise de Erro Relativo: Sistema {parent_name.upper()} ===")
    print(f"{'Lua':<12} | {'Máx Erro Heliocêntrico':<25} | {'Máx Erro Relativo (Pai)':<25} | {'Diagnóstico'}")
    print("-" * 110)

    for body in target_moons:
        if body not in results:
            print(f"{body:<12} | DADOS AUSENTES NOS CSVS")
            continue
            
        stats = results[body]
        abs_km = stats["abs_max"] / 1000.0
        rel_km = stats["rel_max"] / 1000.0
        
        # Inteligência de Diagnóstico
        if rel_km < (abs_km * 0.05): # Erro relativo caiu 95% em relação ao absoluto
            diag = "Dinâmica OK (Problema é Jool)"
        elif rel_km > 10000:
            diag = "Divergência Interna Grave (J2/Massa ausente)"
        else:
            diag = "Divergência Interna Moderada"

        print(f"{body:<12} | {abs_km:<20.3f} km | {rel_km:<20.3f} km | {diag}")

if __name__ == "__main__":
    main()