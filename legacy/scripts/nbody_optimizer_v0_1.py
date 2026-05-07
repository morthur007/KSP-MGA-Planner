#!/usr/bin/env python3
import csv
import math
import subprocess
import numpy as np
from scipy.optimize import root
from pathlib import Path
import spiceypy as spice

# A sua Matriz Campeã da Verdade
TRANSFORM = "+Z,-X,+Y"
BUFFER_DAYS = 0.235

def parse_transform(spec):
    parts = [p.strip().upper() for p in spec.split(",")]
    mapping = {"X": 0, "Y": 1, "Z": 2}
    return [(1 if p[0] == "+" else -1, mapping[p[1]]) for p in parts]

def apply_transform(vec, transform):
    return [sign * float(vec[idx]) for sign, idx in transform]

def target_state_principia(body, et_s, central, tr_spec):
    """Pega a posição real do alvo no instante final usando SPICE e converte para os eixos do Principia."""
    st, _ = spice.spkezr(body.upper(), float(et_s), "J2000", "NONE", central.upper())
    tr = parse_transform(tr_spec)
    r_m = apply_transform([st[0]*1000.0, st[1]*1000.0, st[2]*1000.0], tr)
    return np.array(r_m)

class NBodyShooter:
    def __init__(self):
        self.input_csv = Path("data/mga_smoke/validator_inputs/rank1_leg1_input.csv")
        self.output_csv = Path("data/mga_smoke/validator_inputs/rank1_leg1_output.csv")
        self.plugin_b64 = "data/jnsq_gate0/principia_serialized_plugin.b64"
        self.iteration = 0
        
        # Lê o estado inicial que o Lambert chutou para usarmos de base
        with open(self.input_csv) as f:
            self.base_row = next(csv.DictReader(f))
            
        self.t1_s = float(self.base_row["t1_s"])
        
        # Posicao do alvo (Eve) no tempo final t1_s
        spice.furnsh("data/spice_jnsq_v0_3/jnsq_principia_native_v0_3_1000y_1h_deg9.ids.tpc")
        spice.furnsh("data/spice_jnsq_v0_3/jnsq_principia_native_v0_3_1000y_1h_deg9.bsp")
        self.target_r_m = target_state_principia("EVE", self.t1_s, "SUN", TRANSFORM)

    def evaluate_velocity(self, v_guess):
        """Dispara o validador C++ com o vetor de velocidade v_guess e retorna o erro (miss distance)."""
        # 1. Reescreve o input.csv com a nova velocidade e CORRIGE O \r
        with open(self.input_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=self.base_row.keys(), lineterminator='\n')
            w.writeheader()
            row = dict(self.base_row)
            row["vx_m_s"] = f"{v_guess[0]:.17g}"
            row["vy_m_s"] = f"{v_guess[1]:.17g}"
            row["vz_m_s"] = f"{v_guess[2]:.17g}"
            w.writerow(row)
            
        # Garante que o output fantasma da rodada anterior não exista
        self.output_csv.unlink(missing_ok=True)
            
        # 2. Roda o motor C++ (agora capturando erros reais)
        proc = subprocess.run([
            "principia_particle_validator", 
            self.plugin_b64, 
            str(self.input_csv), 
            str(self.output_csv)
        ], capture_output=True, text=True)
        
        # Se o C++ crashou (ex: colisão muito forte) ou não gerou output
        if proc.returncode != 0 or not self.output_csv.exists():
            return np.array([1e12, 1e12, 1e12])
        
        # 3. Lê o resultado final
        with open(self.output_csv) as f:
            out = next(csv.DictReader(f))
            
        if out["status"] != "ok":
            return np.array([1e12, 1e12, 1e12])
            
        final_r_m = np.array([float(out["x_m"]), float(out["y_m"]), float(out["z_m"])])
        
        # 4. O ERRO: Distância entre onde a nave parou e onde Eve está
        error_vector = final_r_m - self.target_r_m
        miss_km = np.linalg.norm(error_vector) / 1000.0
        
        self.iteration += 1
        print(f"[Iteração {self.iteration:03d}] Miss: {miss_km:12.3f} km | Vx:{v_guess[0]:8.3f} Vy:{v_guess[1]:8.3f} Vz:{v_guess[2]:8.3f}")
        
        return error_vector

def main():
    shooter = NBodyShooter()
    
    # O chute inicial do Lambert (com erro de 974 mil km)
    v0_guess = np.array([
        float(shooter.base_row["vx_m_s"]),
        float(shooter.base_row["vy_m_s"]),
        float(shooter.base_row["vz_m_s"])
    ])
    
    print("\n=== INICIANDO OTIMIZAÇÃO N-CORPOS ===")
    print("Alvo: Eve | Otimizador: Scipy 'hybr' (Matriz Jacobiana)")
    
    # A mágica acontece aqui: O Scipy vai alterar v0_guess até que evaluate_velocity retorne [0, 0, 0]
    result = root(shooter.evaluate_velocity, v0_guess, method='hybr', options={'xtol': 1e-8})
    
    print("\n=== RESULTADO FINAL ===")
    if result.success:
        dv_correction = np.linalg.norm(result.x - v0_guess)
        print("✅ TRAJETÓRIA N-CORPOS ENCONTRADA COM SUCESSO!")
        print(f"Velocidade Original Lambert : {v0_guess}")
        print(f"Velocidade Otimizada C++    : {result.x}")
        print(f"Custo da Correção (Delta-V) : {dv_correction:.3f} m/s")
    else:
        print("❌ O otimizador falhou em convergir.")
        print(result.message)

if __name__ == "__main__":
    main()