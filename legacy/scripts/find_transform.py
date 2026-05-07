import subprocess
import itertools
import math
import csv

def main():
    axes = ['X', 'Y', 'Z']
    signs = ['+', '-']

    # Gera todas as 48 permutações de eixos e sinais (ex: +X,-Z,+Y)
    transforms = []
    for p in itertools.permutations(axes):
        for s in itertools.product(signs, repeat=3):
            transforms.append(f"{s[0]}{p[0]},{s[1]}{p[1]},{s[2]}{p[2]}")

    print(f"Iniciando teste de força-bruta para {len(transforms)} matrizes...\n")
    results = []

    for idx, tr in enumerate(transforms):
        # 1. Gera os CSVs de entrada e expected com o transform atual
        subprocess.run([
            "python", "lambert_candidate_to_particle_leg_v0_1.py",
            "--candidate-csv", "data/mga_smoke/kekj_lambert_w25_parallel_merged.csv",
            "--rank", "1",
            "--bsp", "data/spice_jnsq_v0_3/jnsq_principia_native_v0_3_1000y_1h_deg9.bsp",
            "--tpc", "data/spice_jnsq_v0_3/jnsq_principia_native_v0_3_1000y_1h_deg9.ids.tpc",
            "--metadata", "data/spice_jnsq_v0_3/jnsq_principia_native_v0_3_1000y_1h_deg9.metadata.json",
            "--body-catalog", "data/jnsq_gate0/ksp_future_paused/body_catalog.json",
            "--central-body", "Sun",
            "--sequence", "Kerbin", "Eve", "Kerbin", "Jool",
            "--leg", "1",
            "--start-buffer-days", "3",
            "--end-buffer-days", "3",
            "--output-input-csv", "data/mga_smoke/validator_inputs/rank1_leg1_input.csv",
            "--output-expected-csv", "data/mga_smoke/validator_inputs/rank1_leg1_expected.csv",
            "--spice-to-principia-transform", tr
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # 2. Roda a física N-Corpos no Principia C++
        subprocess.run([
            "principia_particle_validator",
            "data/jnsq_gate0/principia_serialized_plugin.b64",
            "data/mga_smoke/validator_inputs/rank1_leg1_input.csv",
            "data/mga_smoke/validator_inputs/rank1_leg1_output.csv"
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # 3. Lê o resultado
        with open("data/mga_smoke/validator_inputs/rank1_leg1_output.csv") as f:
            o = next(csv.DictReader(f))
        with open("data/mga_smoke/validator_inputs/rank1_leg1_expected.csv") as f:
            e = next(csv.DictReader(f))

        if o["status"] != "ok":
            print(f"[{idx+1}/48] {tr} -> ERRO no C++")
            continue

        pos = math.sqrt(sum((float(o[k])-float(e[k]))**2 for k in ["x_m","y_m","z_m"]))
        
        results.append((pos/1000, tr))
        # O end="\r" faz a linha se reescrever, criando uma barra de progresso no terminal
        print(f"Progresso: [{idx+1}/48] Analisando {tr}...", end="\r")

    print("\n\n=== RESULTADOS FINAIS (Ordenados por Erro) ===")
    results.sort(key=lambda x: x[0])
    
    for r in results:
        print(f"Transform: {r[1]:<10} | Desvio N-Corpos vs Kepler: {r[0]:10.2f} km")

if __name__ == "__main__":
    main()