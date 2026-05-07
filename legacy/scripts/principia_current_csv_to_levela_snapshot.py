import csv
import json
import argparse
from pathlib import Path

def T(vec):
    """Transformação de referencial: Principia Barycentric -> KSP AliceSun"""
    x, y, z = vec
    return [-y, z, x]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--central-body", default="Sun")
    ap.add_argument("--template", default=None)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    rows = {}
    with open(args.csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows[row["body"]] = row

    if args.central_body not in rows:
        raise SystemExit(f"Corpo central {args.central_body!r} não encontrado no CSV.")

    # 1. Extração e Transformação do Corpo Central (Âncora)
    c = rows[args.central_body]

    
    cr_abs = [float(c["x_m"]), float(c["y_m"]), float(c["z_m"])]
    cv_abs = [float(c["vx_m_s"]), float(c["vy_m_s"]), float(c["vz_m_s"])]

    cr = T(cr_abs)
    cv = T(cv_abs)

    # 2. Preparação do Template
    if args.template:
        out = json.loads(Path(args.template).read_text())
    else:
        out = {"ephemerides": {}}

    c = rows[args.central_body]

    out["schema"] = "principia_native_snapshot.level_a_compatible.v2_no_time_fudge"
    epoch = float(c["time_s"])
    out["epoch_ut_s"] = epoch

    # Use epoch_sincronizado para preencher o JSON e os estados
    out["reference_body"] = args.central_body

    # 3. Processamento de todos os corpos
    for name, row in rows.items():
        # Vetores absolutos do dump C++
        r_abs_raw = [float(row["x_m"]), float(row["y_m"]), float(row["z_m"])]
        v_abs_raw = [float(row["vx_m_s"]), float(row["vy_m_s"]), float(row["vz_m_s"])]

        # Aplica a transformação T
        r_t = T(r_abs_raw)
        v_t = T(v_abs_raw)

        # Calcula coordenadas relativas ao corpo central transformado
        r_rel = [r_t[i] - cr[i] for i in range(3)]
        v_rel = [v_t[i] - cv[i] for i in range(3)]

        if name == args.central_body:
            r_rel = [0.0, 0.0, 0.0]
            v_rel = [0.0, 0.0, 0.0]

        # O REBOUND exige o formato de lista: [t, x, y, z, vx, vy, vz]
        
        state_list = [
            epoch,
            r_rel[0], r_rel[1], r_rel[2],
            v_rel[0], v_rel[1], v_rel[2],
        ]
        mu = float(row["mu_m3_s2"])
        out["ephemerides"][name] = {
            "mu": mu,
            "gravitational_parameter": mu,
            "states": [state_list]
        }

    Path(args.output).write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"[OK] {args.output}")
    print(f"[INFO] Corpos: {len(out['ephemerides'])} | Epoch: {epoch}")

if __name__ == "__main__":
    main()