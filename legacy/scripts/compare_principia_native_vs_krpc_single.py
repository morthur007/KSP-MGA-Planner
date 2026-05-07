#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import itertools
import math
from pathlib import Path


def T_principia(vec):
    """
    Transformação descoberta pelo seu descobre.py:
      Principia (X,Y,Z) -> LevelA/KSP (-Y,+Z,+X)
    """
    x, y, z = vec
    return [-y, z, x]


def norm3(v):
    return math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)


def sub3(a, b):
    return [a[i] - b[i] for i in range(3)]


def read_principia_csv(path, central_body):
    rows = {}
    with Path(path).open() as f:
        r = csv.DictReader(f)
        for row in r:
            name = row["body"]
            rows[name] = row

    if central_body not in rows:
        raise SystemExit(f"central body {central_body!r} ausente no Principia CSV")

    c = rows[central_body]
    cr = T_principia([float(c["x_m"]), float(c["y_m"]), float(c["z_m"])])
    cv = T_principia([float(c["vx_m_s"]), float(c["vy_m_s"]), float(c["vz_m_s"])])

    out = {}
    for name, row in rows.items():
        r_abs = T_principia([float(row["x_m"]), float(row["y_m"]), float(row["z_m"])])
        v_abs = T_principia([float(row["vx_m_s"]), float(row["vy_m_s"]), float(row["vz_m_s"])])

        out[name] = {
            "time_s": float(row["time_s"]),
            "r": sub3(r_abs, cr),
            "v": sub3(v_abs, cv),
        }

    if central_body in out:
        out[central_body]["r"] = [0.0, 0.0, 0.0]
        out[central_body]["v"] = [0.0, 0.0, 0.0]

    return out


def read_krpc_csv(path, sample_index=None):
    rows_by_sample = {}

    with Path(path).open() as f:
        r = csv.DictReader(f)
        for row in r:
            idx = int(row["sample_index"])
            rows_by_sample.setdefault(idx, {})
            rows_by_sample[idx][row["body"]] = {
                "ut_mid_s": float(row["ut_mid_s"]),
                "read_duration_s": float(row["read_duration_s"]),
                "r": [float(row["x_m"]), float(row["y_m"]), float(row["z_m"])],
                "v": [float(row["vx_m_s"]), float(row["vy_m_s"]), float(row["vz_m_s"])],
            }

    if sample_index is not None:
        return {sample_index: rows_by_sample[sample_index]}

    return rows_by_sample


def score_sample(pr, kr, bodies):
    rows = []

    for body in bodies:
        if body not in pr or body not in kr:
            continue

        dr = sub3(kr[body]["r"], pr[body]["r"])
        dv = sub3(kr[body]["v"], pr[body]["v"])

        pr_vnorm = max(norm3(pr[body]["v"]), 1e-12)
        apparent_dt = sum(dr[i] * pr[body]["v"][i] for i in range(3)) / (pr_vnorm ** 2)

        rows.append({
            "body": body,
            "pos_err_m": norm3(dr),
            "vel_err_m_s": norm3(dv),
            "apparent_dt_s": apparent_dt,
            "krpc_ut_mid_s": kr[body]["ut_mid_s"],
            "principia_time_s": pr[body]["time_s"],
            "raw_epoch_delta_s": kr[body]["ut_mid_s"] - pr[body]["time_s"],
        })

    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--principia-csv", required=True)
    ap.add_argument("--krpc-csv", required=True)
    ap.add_argument("--central-body", default="Sun")
    ap.add_argument("--bodies", nargs="*", default=None)
    ap.add_argument("--output-csv", required=True)
    args = ap.parse_args()

    pr = read_principia_csv(args.principia_csv, args.central_body)
    kr_samples = read_krpc_csv(args.krpc_csv)

    if args.bodies:
        bodies = args.bodies
    else:
        bodies = sorted(set(pr.keys()) & set().union(*(set(s.keys()) for s in kr_samples.values())))

    all_rows = []
    summary = []

    for sample_index, kr in sorted(kr_samples.items()):
        rows = score_sample(pr, kr, bodies)

        for row in rows:
            row["sample_index"] = sample_index
            all_rows.append(row)

        if not rows:
            continue

        rms_pos = math.sqrt(sum(r["pos_err_m"] ** 2 for r in rows) / len(rows))
        max_pos = max(r["pos_err_m"] for r in rows)
        med_dt = sorted(r["apparent_dt_s"] for r in rows)[len(rows) // 2]

        summary.append((rms_pos, max_pos, med_dt, sample_index, rows))

    out = Path(args.output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "sample_index",
        "body",
        "pos_err_m",
        "vel_err_m_s",
        "apparent_dt_s",
        "krpc_ut_mid_s",
        "principia_time_s",
        "raw_epoch_delta_s",
    ]

    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(all_rows)

    print(f"[OK] detailed CSV: {out}")

    print("\n=== SAMPLE SUMMARY ===")
    for rms_pos, max_pos, med_dt, sample_index, rows in sorted(summary):
        print(
            f"sample={sample_index:<3} "
            f"rms={rms_pos/1000:10.6f} km "
            f"max={max_pos/1000:10.6f} km "
            f"med_dt={med_dt:10.6f} s"
        )

    if summary:
        best = sorted(summary)[0]
        rms_pos, max_pos, med_dt, sample_index, rows = best

        print("\n=== BEST SAMPLE BY RMS ===")
        print(
            f"sample={sample_index} "
            f"rms={rms_pos/1000:.6f} km "
            f"max={max_pos/1000:.6f} km "
            f"med_dt={med_dt:.6f} s"
        )

        rows_sorted = sorted(rows, key=lambda r: r["pos_err_m"], reverse=True)
        print("\nWorst bodies:")
        for r in rows_sorted[:20]:
            print(
                f'{r["body"]:<10} '
                f'pos={r["pos_err_m"]/1000:10.6f} km  '
                f'vel={r["vel_err_m_s"]:10.6f} m/s  '
                f'dt={r["apparent_dt_s"]:10.6f} s  '
                f'raw_epoch_delta={r["raw_epoch_delta_s"]:10.6f} s'
            )


if __name__ == "__main__":
    main()