import csv, math

def get_principia_level_a(filepath):
    states = {}
    sun_pos = [0, 0, 0]
    with open(filepath) as f:
        reader = list(csv.DictReader(f))
        for row in reader:
            if row["body"] in ["Sun", "Kerbol"]:
                # Principia -> Level A: (-Y, +Z, +X)
                sun_pos = [-float(row["y_m"]), float(row["z_m"]), float(row["x_m"])]
                break
        
        for row in reader:
            name = row["body"]
            raw_pos = [-float(row["y_m"]), float(row["z_m"]), float(row["x_m"])]
            states[name] = [raw_pos[i] - sun_pos[i] for i in range(3)]
    return states

def get_rebound_states(filepath):
    states = {}
    with open(filepath) as f:
        for row in csv.DictReader(f):
            name = row["body"]
            states[name] = [float(row["x"]), float(row["y"]), float(row["z"])]
    return states

def main():
    p_states = get_principia_level_a("data/jnsq_gate0/principia_future_snapshot.csv")
    r_states = get_rebound_states("data/jnsq_gate0/saida_teste_futuro/rebound_states.csv")
    
    print(f"{'Corpo':<12} | {'Erro REBOUND vs PRINCIPIA (m)':<30}")
    print("-" * 50)
    
    errors = []
    for name in p_states:
        if name not in r_states or name in ["Sun", "Kerbol"]: continue
        
        p_pos = p_states[name]
        r_pos = r_states[name]
        
        dist = math.sqrt(sum((p_pos[i] - r_pos[i])**2 for i in range(3)))
        errors.append(dist)
        print(f"{name:<12} | {dist:.6f}")
        
    print("-" * 50)
    rms = math.sqrt(sum(e**2 for e in errors)/len(errors))
    print(f"RMS FINAL: {rms:.6f} metros")

if __name__ == "__main__":
    main()