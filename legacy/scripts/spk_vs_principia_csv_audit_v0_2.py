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


def norm3(v):
    return math.sqrt(float(np.dot(v, v)))


def load_metadata(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_csv(path: Path):
    rows_by_body = defaultdict(list)

    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            body = row["body"]
            et = float(row["et_seconds"])
            state = np.array(
                [
                    float(row["x_m"]),
                    float(row["y_m"]),
                    float(row["z_m"]),
                    float(row["vx_m_s"]),
                    float(row["vy_m_s"]),
                    float(row["vz_m_s"]),
                ],
                dtype=np.float64,
            )
            rows_by_body[body].append((et, state))

    for body in rows_by_body:
        rows_by_body[body].sort(key=lambda x: x[0])

    return rows_by_body


def classify(max_pos_m: float, rms_pos_m: float):
    if max_pos_m < 0.01 and rms_pos_m < 0.005:
        return "EXCELENTE"
    if max_pos_m < 1.0 and rms_pos_m < 0.25:
        return "OK"
    if max_pos_m < 100.0 and rms_pos_m < 25.0:
        return "SUSPEITO"
    return "FALHA"


def main():
    ap = argparse.ArgumentParser(
        description="Audita BSP contra CSV Principia-native usado como fonte."
    )
    ap.add_argument("--csv", required=True, type=Path)
    ap.add_argument("--bsp", required=True, type=Path)
    ap.add_argument("--tpc", required=True, type=Path)
    ap.add_argument("--metadata", required=True, type=Path)
    ap.add_argument("--central-body", default="Sun")
    ap.add_argument("--output-csv", required=True, type=Path)
    ap.add_argument("--output-json", required=True, type=Path)
    ap.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Audita uma amostra a cada N linhas por corpo.",
    )
    args = ap.parse_args()

    metadata = load_metadata(args.metadata)
    rows_by_body = load_csv(args.csv)

    spice.furnsh(str(args.tpc))
    spice.furnsh(str(args.bsp))

    body_meta = metadata["bodies"]
    center_id = metadata["central_body_id"]
    frame = metadata["frame"]

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)

    detail_rows = []
    summary = []

    print("SPK vs CSV Principia-native")
    print(
        f"{'Corpo':<16} | {'N':>7} | {'Max pos (m)':>14} | "
        f"{'RMS pos (m)':>14} | {'Max vel (m/s)':>14} | Status"
    )
    print("-" * 88)

    for body in sorted(rows_by_body):
        if body not in body_meta:
            continue

        if body == args.central_body:
            continue

        body_id = body_meta[body]["naif_id"]

        pos_errs = []
        vel_errs = []

        rows = rows_by_body[body][:: args.stride]

        for et, csv_state in rows:
            spk_state, _lt = spice.spkezr(
                str(body_id),
                et,
                frame,
                "NONE",
                str(center_id),
            )

            spk_state = np.array(spk_state, dtype=np.float64)

            # SPICE km/km/s → CSV m/m/s
            spk_state_m = spk_state.copy()
            spk_state_m[0:3] *= 1000.0
            spk_state_m[3:6] *= 1000.0

            diff = spk_state_m - csv_state
            pos_err_m = norm3(diff[:3])
            vel_err_m_s = norm3(diff[3:])

            pos_errs.append(pos_err_m)
            vel_errs.append(vel_err_m_s)

            detail_rows.append(
                {
                    "body": body,
                    "et_seconds": et,
                    "pos_err_m": pos_err_m,
                    "vel_err_m_s": vel_err_m_s,
                    "csv_x_m": csv_state[0],
                    "csv_y_m": csv_state[1],
                    "csv_z_m": csv_state[2],
                    "spk_x_m": spk_state[0],
                    "spk_y_m": spk_state[1],
                    "spk_z_m": spk_state[2],
                }
            )

        if not pos_errs:
            continue

        max_pos = max(pos_errs)
        rms_pos = math.sqrt(sum(e * e for e in pos_errs) / len(pos_errs))
        max_vel = max(vel_errs)
        status = classify(max_pos, rms_pos)

        summary.append(
            {
                "body": body,
                "n": len(pos_errs),
                "max_pos_m": max_pos,
                "rms_pos_m": rms_pos,
                "max_vel_m_s": max_vel,
                "status": status,
            }
        )

        print(
            f"{body:<16} | {len(pos_errs):7d} | "
            f"{max_pos:14.6g} | {rms_pos:14.6g} | "
            f"{max_vel:14.6g} | {status}"
        )

    with args.output_csv.open("w", newline="") as f:
        fieldnames = [
            "body",
            "et_seconds",
            "pos_err_m",
            "vel_err_m_s",
            "csv_x_m",
            "csv_y_m",
            "csv_z_m",
            "spk_x_m",
            "spk_y_m",
            "spk_z_m",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(detail_rows)

    args.output_json.write_text(
        json.dumps(
            {
                "schema": "spk_vs_principia_csv_audit.v0_2",
                "csv": str(args.csv),
                "bsp": str(args.bsp),
                "tpc": str(args.tpc),
                "metadata": str(args.metadata),
                "stride": args.stride,
                "summary": summary,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(f"\n[OK] detail CSV: {args.output_csv}")
    print(f"[OK] summary JSON: {args.output_json}")


if __name__ == "__main__":
    main()