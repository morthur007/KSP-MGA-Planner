#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from collections import defaultdict

import numpy as np
import spiceypy as spice


DEFAULT_BASE_ID = -990000


def sanitize_segment_id(name: str) -> str:
    # SPICE segment id max tradicional: 40 chars.
    s = f"JNSQ_PRINCIPIA_{name}"
    return s[:40]


def load_principia_csv(path: Path):
    rows_by_body = defaultdict(list)

    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        required = {
            "et_seconds",
            "body",
            "x_m",
            "y_m",
            "z_m",
            "vx_m_s",
            "vy_m_s",
            "vz_m_s",
        }
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"CSV sem colunas necessárias: {sorted(missing)}")

        for row in reader:
            body = row["body"]
            rows_by_body[body].append(
                (
                    float(row["et_seconds"]),
                    [
                        float(row["x_m"]),
                        float(row["y_m"]),
                        float(row["z_m"]),
                        float(row["vx_m_s"]),
                        float(row["vy_m_s"]),
                        float(row["vz_m_s"]),
                    ],
                    float(row.get("mu_m3_s2", "nan")),
                )
            )

    for body, rows in rows_by_body.items():
        rows.sort(key=lambda x: x[0])

    return rows_by_body


def build_ids(bodies: list[str], central_body: str, base_id: int):
    ids = {}
    ids[central_body] = base_id

    n = 1
    for body in sorted(bodies):
        if body == central_body:
            continue
        ids[body] = base_id - n
        n += 1

    return ids


def write_tpc(path: Path, ids: dict[str, int], mus: dict[str, float]):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        f.write("\\begindata\n\n")

        for name, naif_id in sorted(ids.items(), key=lambda kv: kv[1], reverse=True):
            f.write(f"NAIF_BODY_NAME += ( '{name.upper()}' )\n")
            f.write(f"NAIF_BODY_CODE += ( {naif_id} )\n")

        f.write("\n")

        for name, naif_id in sorted(ids.items(), key=lambda kv: kv[1], reverse=True):
            mu = mus.get(name)
            if mu is not None and math.isfinite(mu):
                mu_km3_s2 = mu / 1.0e9
                f.write(f"BODY{naif_id}_GM = ( {mu_km3_s2:.17e} )\n")

        f.write("\n\\begintext\n")


def main():
    ap = argparse.ArgumentParser(
        description="Converte CSV Principia-native em SPK Type 13."
    )
    ap.add_argument("--csv", required=True, type=Path)
    ap.add_argument("--output-bsp", required=True, type=Path)
    ap.add_argument("--output-tpc", required=True, type=Path)
    ap.add_argument("--output-metadata", required=True, type=Path)
    ap.add_argument("--central-body", default="Sun")
    ap.add_argument("--frame", default="J2000")
    ap.add_argument("--base-id", type=int, default=DEFAULT_BASE_ID)
    ap.add_argument(
        "--degree",
        type=int,
        default=7,
        help="Grau Hermite do SPK Type 13. Use ímpar; 7 é conservador.",
    )
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    if args.degree % 2 == 0:
        raise ValueError("SPK Type 13 deve usar grau ímpar. Ex: 3, 5, 7, 9.")

    rows_by_body = load_principia_csv(args.csv)
    bodies = sorted(rows_by_body)

    if args.central_body not in rows_by_body:
        raise ValueError(f"Corpo central ausente no CSV: {args.central_body}")

    ids = build_ids(bodies, args.central_body, args.base_id)

    mus = {}
    for body, rows in rows_by_body.items():
        mu = rows[0][2]
        if math.isfinite(mu):
            mus[body] = mu

    first_et = min(rows[0][0] for rows in rows_by_body.values())
    last_et = max(rows[-1][0] for rows in rows_by_body.values())

    args.output_bsp.parent.mkdir(parents=True, exist_ok=True)
    args.output_tpc.parent.mkdir(parents=True, exist_ok=True)
    args.output_metadata.parent.mkdir(parents=True, exist_ok=True)

    if args.output_bsp.exists():
        if args.overwrite:
            args.output_bsp.unlink()
        else:
            raise FileExistsError(args.output_bsp)

    handle = spice.spkopn(str(args.output_bsp), "JNSQ PRINCIPIA NATIVE V0.2", 1024)

    try:
        center_id = ids[args.central_body]

        for body in bodies:
            if body == args.central_body:
                print(f"[SKIP] {body:<12} corpo central; segmento relativo a si mesmo não é gravado")
                continue

            body_id = ids[body]
            rows = rows_by_body[body]

            epochs = np.array([r[0] for r in rows], dtype=np.float64)
            states = np.array([r[1] for r in rows], dtype=np.float64)

            if len(epochs) <= args.degree:
                raise ValueError(
                    f"{body}: amostras insuficientes ({len(epochs)}) "
                    f"para degree={args.degree}"
                )

            if body == args.central_body:
                # Pode gravar o Sol como corpo central zero também. Isso ajuda
                # auditorias, mas não é necessário para navegação.
                pass

            segid = sanitize_segment_id(body)

            states_m = np.array([r[1] for r in rows], dtype=np.float64)

            states_spice = states_m.copy()
            states_spice[:, 0:3] /= 1000.0   # m → km
            states_spice[:, 3:6] /= 1000.0   # m/s → km/s

            spice.spkw13(
                handle,
                body_id,
                center_id,
                args.frame,
                float(epochs[0]),
                float(epochs[-1]),
                segid,
                args.degree,
                len(epochs),
                states_spice,
                epochs,
            )

            print(
                f"[OK] {body:<12} id={body_id} n={len(epochs)} "
                f"start={epochs[0]:.6f} end={epochs[-1]:.6f}"
            )

    finally:
        spice.spkcls(handle)

    write_tpc(args.output_tpc, ids, mus)

    metadata = {
        "schema": "spice_v0_2_principia_native.type13",
        "source_csv": str(args.csv),
        "output_bsp": str(args.output_bsp),
        "output_tpc": str(args.output_tpc),
        "central_body": args.central_body,
        "central_body_id": ids[args.central_body],
        "frame": args.frame,
        "spk_type": 13,
        "degree": args.degree,
        "first_et_seconds": first_et,
        "last_et_seconds": last_et,
        "body_count": len(bodies),
        "bodies": {
            body: {
                "naif_id": ids[body],
                "samples": len(rows_by_body[body]),
                "mu_m3_s2": mus.get(body),
            }
            for body in bodies
        },
    }

    args.output_metadata.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"[OK] BSP: {args.output_bsp}")
    print(f"[OK] TPC: {args.output_tpc}")
    print(f"[OK] metadata: {args.output_metadata}")


if __name__ == "__main__":
    main()