import csv

def convert():
    in_file = "data/jnsq_gate0/principia_future_snapshot.csv"
    out_file = "data/jnsq_gate0/principia_target.csv"
    
    # 1. Encontrar o referencial do Sol
    sun_pos = [0, 0, 0]
    sun_vel = [0, 0, 0]
    with open(in_file, "r") as f:
        for row in csv.DictReader(f):
            if row["body"] in ["Sun", "Kerbol"]:
                sun_pos = [float(row["x_m"]), float(row["y_m"]), float(row["z_m"])]
                sun_vel = [float(row["vx_m_s"]), float(row["vy_m_s"]), float(row["vz_m_s"])]
                break

    # 2. Escrever o Falso CSV do kRPC
    with open(in_file, "r") as fin, open(out_file, "w", newline="") as fout:
        reader = csv.DictReader(fin)
        writer = csv.writer(fout)
        
        # O NOME EXATO EXIGIDO PELO REBOUND_LEVEL_A_CACHE.PY
        writer.writerow(["et_seconds", "body", "x_m", "y_m", "z_m", "vx_m_s", "vy_m_s", "vz_m_s"])
        
        for row in reader:
            name = row["body"]
            t = float(row["time_s"])
            
            # Posição relativa crua
            rx = float(row["x_m"]) - sun_pos[0]
            ry = float(row["y_m"]) - sun_pos[1]
            rz = float(row["z_m"]) - sun_pos[2]
            
            vx = float(row["vx_m_s"]) - sun_vel[0]
            vy = float(row["vy_m_s"]) - sun_vel[1]
            vz = float(row["vz_m_s"]) - sun_vel[2]
            
            # Matriz Level A: Principia Raw -> Level A (-Y, +Z, +X)
            levela_rx, levela_ry, levela_rz = -ry, rz, rx
            levela_vx, levela_vy, levela_vz = -vy, vz, vx
            
            writer.writerow([t, name, levela_rx, levela_ry, levela_rz, levela_vx, levela_vy, levela_vz])

if __name__ == "__main__":
    convert()