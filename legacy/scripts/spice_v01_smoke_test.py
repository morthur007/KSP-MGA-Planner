#!/usr/bin/env python3
"""
spice_v01_smoke_test.py

Teste mínimo de leitura do BSP/TPC gerado: carrega kernels e extrai estados de corpos em alguns epochs.
Serve para validar a ponte SPICE -> planejador antes de integrar PyKEP.

Exemplo:
python spice_v01_smoke_test.py \
  --bsp data/spice_v0_1_33y/opm_mpe_principia_rebound_v0_1_33y.bsp \
  --tpc data/spice_v0_1_33y/opm_mpe_principia_rebound_v0_1_33y.ids.tpc \
  --metadata data/spice_v0_1_33y/opm_mpe_principia_rebound_v0_1_33y.metadata.json \
  --bodies Kerbin Duna Jool Sarnus Urlum Neidon Plock Soden \
  --central-body Sun \
  --years 0,1,5,10,20,30 \
  --output-csv data/spice_v0_1_33y/spice_smoke_states.csv
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Any

import spiceypy as spice

JULIAN_YEAR_S = 365.25 * 86400.0


def load_ids_from_tpc(path: Path) -> Dict[str, int]:
    # Fallback parser for lines like NAIF_BODY_CODE += ( 123 ) and NAIF_BODY_NAME += ( 'Kerbin' ).
    text = path.read_text(encoding="utf-8", errors="ignore")
    names = []
    codes = []
    import re
    for m in re.finditer(r"NAIF_BODY_NAME\s*\+?=\s*\(\s*'([^']+)'\s*\)", text):
        names.append(m.group(1))
    for m in re.finditer(r"NAIF_BODY_CODE\s*\+?=\s*\(\s*(-?\d+)\s*\)", text):
        codes.append(int(m.group(1)))
    return dict(zip(names, codes))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bsp", required=True)
    ap.add_argument("--tpc", required=True)
    ap.add_argument("--metadata")
    ap.add_argument("--bodies", nargs="+", required=True)
    ap.add_argument("--central-body", default="Sun")
    ap.add_argument("--years", default="0,1,5,10,20,30")
    ap.add_argument("--output-csv", required=True)
    args = ap.parse_args()

    spice.kclear()
    spice.furnsh(args.tpc)
    spice.furnsh(args.bsp)

    meta: Dict[str, Any] = {}
    if args.metadata and Path(args.metadata).exists():
        meta = json.loads(Path(args.metadata).read_text(encoding="utf-8"))

    years = [float(x.strip()) for x in args.years.split(",") if x.strip()]
    out = Path(args.output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    start_et = float(meta.get("start_et_s", meta.get("start_et", meta.get("coverage_start_et", 0.0))))

    # Validação Extra: Se o start_et for 0.0, pode ser que o kernel comece depois.
    # Vamos verificar a cobertura real do primeiro corpo da lista no kernel BSP.
    try:
        # Pega o ID numérico do primeiro corpo para checar a cobertura
        # Assumindo que Sun (10) ou o primeiro da lista tenha cobertura total
        body_id = spice.bodn2c(args.bodies[0])
        cover = spice.support_types.SPICEDOUBLE_CELL(10)
        spice.spkcov(args.bsp, body_id, cover)
        
        # O início real do arquivo BSP (primeiro intervalo, início)
        actual_bsp_start = cover[0] 
        
        if start_et < actual_bsp_start:
            print(f"[INFO] start_et {start_et} está fora do BSP. Ajustando para {actual_bsp_start}")
            start_et = actual_bsp_start
    except Exception as e:
        print(f"[WARN] Não foi possível validar cobertura do BSP: {e}")

    rows = []
    for yr in years:
        et = start_et + yr * JULIAN_YEAR_S
        for body in args.bodies:
            try:
                state, lt = spice.spkezr(body, et, "J2000", "NONE", args.central_body)
            except Exception as exc:
                rows.append({
                    "year": yr, "et_s": et, "body": body, "status": f"ERROR: {exc}",
                    "x_m": "", "y_m": "", "z_m": "", "vx_m_s": "", "vy_m_s": "", "vz_m_s": "",
                })
                continue
            rows.append({
                "year": yr, "et_s": f"{et:.16e}", "body": body, "status": "OK",
                "x_m": f"{state[0]:.16e}", "y_m": f"{state[1]:.16e}", "z_m": f"{state[2]:.16e}",
                "vx_m_s": f"{state[3]:.16e}", "vy_m_s": f"{state[4]:.16e}", "vz_m_s": f"{state[5]:.16e}",
            })

    with out.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["year", "et_s", "body", "status", "x_m", "y_m", "z_m", "vx_m_s", "vy_m_s", "vz_m_s"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    spice.kclear()
    print(f"[OK] smoke states: {out}")
    errors = [r for r in rows if r["status"] != "OK"]
    if errors:
        print(f"[WARN] {len(errors)} failed state queries")
        for r in errors[:10]:
            print(r["body"], r["year"], r["status"])
    else:
        print("[OK] all state queries succeeded")


if __name__ == "__main__":
    main()
