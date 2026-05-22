#!/usr/bin/env python3
import json
import argparse
import numpy as np
from scipy.spatial.transform import Rotation as R

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--live-state", required=True, help="O JSON atual")
    ap.add_argument("--vinf", nargs=3, type=float, required=True, help="Vetor V-infinito (X Y Z)")
    ap.add_argument("--out", required=True, help="JSON spoofed de saída")
    args = ap.parse_args()

    # 1. Carrega os dados reais de LKO (0 graus)
    with open(args.live_state, 'r') as f:
        state = json.load(f)
    
    r_old = np.array(state["relative_r_raw_m"])
    v_old = np.array(state["relative_v_raw_m_s"])
    v_inf = np.array(args.vinf)

    # 2. Geometria Orbital (Momento Angular)
    h_old = np.cross(r_old, v_old)
    h_hat = h_old / np.linalg.norm(h_old)
    v_inf_hat = v_inf / np.linalg.norm(v_inf)

    # 3. Novo plano orbital (perpendicular ao V-infinito)
    h_new = h_hat - np.dot(h_hat, v_inf_hat) * v_inf_hat
    h_new_hat = h_new / np.linalg.norm(h_new)

    # 4. Cálculo do eixo e ângulo de rotação 3D
    axis = np.cross(h_hat, h_new_hat)
    axis_norm = np.linalg.norm(axis)

    if axis_norm < 1e-8:
        print("[!] A órbita já está perfeitamente alinhada!")
        r_new, v_new = r_old, v_old
    else:
        axis_hat = axis / axis_norm
        angle = np.arccos(np.clip(np.dot(h_hat, h_new_hat), -1.0, 1.0))
        print(f"[+] Girando o plano orbital em {np.degrees(angle):.3f} graus...")

        # Aplica a Rotação de Rodrigues (Mágica da astrodinâmica)
        rot = R.from_rotvec(axis_hat * angle)
        r_new = rot.apply(r_old)
        v_new = rot.apply(v_old)

    # 5. Salva o JSON adulterado
    state["relative_r_raw_m"] = r_new.tolist()
    state["relative_v_raw_m_s"] = v_new.tolist()
    state["spoofed_note"] = "Alinhado perfeitamente com V-inf"

    with open(args.out, 'w') as f:
        json.dump(state, f, indent=2)
    
    print(f"[OK] State adulterado salvo em: {args.out}")

if __name__ == "__main__":
    main()