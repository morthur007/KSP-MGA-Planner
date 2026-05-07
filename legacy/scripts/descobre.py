import csv
import itertools
import math

def get_relative_states(file_path):
    states = {}
    sun_pos = None
    sun_vel = None
    
    with open(file_path) as f:
        reader = list(csv.DictReader(f))
        # 1. Localiza o Sol para referência
        for row in reader:
            if row["body"] in ["Sun", "Kerbol"]:
                sun_pos = [float(row["x_m"]), float(row["y_m"]), float(row["z_m"])]
                sun_vel = [float(row["vx_m_s"]), float(row["vy_m_s"]), float(row["vz_m_s"])]
                break
        
        if sun_pos is None:
            raise ValueError(f"Sol não encontrado em {file_path}")

        # 2. Calcula posição e velocidade relativas ao Sol
        for row in reader:
            name = row["body"]
            r_abs = [float(row["x_m"]), float(row["y_m"]), float(row["z_m"])]
            v_abs = [float(row["vx_m_s"]), float(row["vy_m_s"]), float(row["vz_m_s"])]
            states[name] = {
                "r": [r_abs[i] - sun_pos[i] for i in range(3)],
                "v": [v_abs[i] - sun_vel[i] for i in range(3)]
            }
    return states

def get_rms(p_rel_states, k_rel_states, transform, dt):
    p_idx, signs = transform
    errors = []
    for name in k_rel_states:
        if name not in p_rel_states or name in ["Sun", "Kerbol"]: 
            continue
        
        # kRPC (Referência no tempo t_final)
        kr = k_rel_states[name]["r"]
        
        # Principia (Dados no tempo t_inicial)
        pr_raw = p_rel_states[name]["r"]
        pv_raw = p_rel_states[name]["v"]
        
        # Aplica a rotação nos vetores do Principia
        pr_trans = [pr_raw[p_idx[i]] * signs[i] for i in range(3)]
        pv_trans = [pv_raw[p_idx[i]] * signs[i] for i in range(3)]
        
        # PROJEÇÃO: Onde o corpo do Principia estaria após 'dt' segundos?
        # P_final = P_inicial + (V_inicial * dt)
        pr_projected = [pr_trans[i] + (pv_trans[i] * dt) for i in range(3)]
        
        dist = math.sqrt(sum((kr[i] - pr_projected[i])**2 for i in range(3)))
        errors.append(dist)
        
    return math.sqrt(sum(e**2 for e in errors) / len(errors)) if errors else float('inf')

def main():
    p_file = "data/jnsq_gate0/principia_current_snapshot.csv"
    k_file = "data/jnsq_gate0/ksp_instant_sample/states.csv"

    # Tempos exatos dos seus logs anteriores
    t_principia = 81.65168640136972
    t_krpc = 93.0316864013686
    dt = t_krpc - t_principia

    print(f"Carregando estados relativos... (Compensando dt = {dt:.4f}s)")
    p_rel = get_relative_states(p_file)
    k_rel = get_relative_states(k_file)

    perms = list(itertools.permutations([0, 1, 2]))
    signs = list(itertools.product([1, -1], repeat=3))
    all_transforms = list(itertools.product(perms, signs))

    best_t = None
    min_rms = float('inf')

    for t in all_transforms:
        rms = get_rms(p_rel, k_rel, t, dt)
        if rms < min_rms:
            min_rms = rms
            best_t = t

    p_idx, s = best_t
    axes = ['X', 'Y', 'Z']
    res = [f"{'+' if s[i]>0 else '-'}{axes[p_idx[i]]}" for i in range(3)]
    
    print(f"\n--- RESULTADO SINCRONIZADO ---")
    print(f"RMS Mínimo: {min_rms:.4f} metros")
    print(f"Transformação ideal: (X, Y, Z) -> ({', '.join(res)})")

if __name__ == "__main__":
    main()