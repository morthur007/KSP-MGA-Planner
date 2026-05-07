#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import itertools
import math
from pathlib import Path


def norm3(v):
    return math.sqrt(sum(x * x for x in v))


def sub3(a, b):
    return [a[i] - b[i] for i in range(3)]


def transforms():
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product([-1.0, 1.0], repeat=3):
            def make(perm=perm, signs=signs):
                return lambda v: [signs[i] * v[perm[i]] for i in range(3)]
            label = "(" + ",".join(
                f"{'+' if signs[i] > 0 else '-'}{'XYZ'[perm[i]]}"
                for i in range(3)
            ) + ")"
            yield label, make()


def read_principia(path, central="Sun"):
    rows = {}
    with Path(path).open() as f:
        for row in csv.DictReader(f):
            rows[row["body"]] = {
                "t": float(row["time_s"]),
                "r": [float(row["x_m"]), float(row["y_m"]), float(row["z_m"])],
                "v": [float(row["vx_m_s"]), float(row["vy_m_s"]), float(row["vz_m_s"])],
            }
    return rows


def read_krpc(path, sample_index=0, central="Sun"):
    rows = {}
    with Path(path).open() as f:
        for row in csv.DictReader(f):
            if int(row["sample_index"]) != sample_index:
                continue
            rows[row["body"]] = {
                "t": float(row["target_ut_s"]),
                "r": [float(row["x_m"]), float(row["y_m"]), float(row["z_m"])],
                "v": [float(row["vx_m_s"]), float(row["vy_m_s"]), float(row["vz_m_s"])],
            }
    return rows


def relative(rows, T, central="Sun"):
    cr = T(rows[central]["r"])
    cv = T(rows[central]["v"])
    out = {}
    for name, st in rows.items():
        r = sub3(T(st["r"]), cr)
        v = sub3(T(st["v"]), cv)
        if name == central:
            r = [0.0, 0.0, 0.0]
            v = [0.0, 0.0, 0.0]
        out[name] = {"t": st["t"], "r": r, "v": v}
    return out


def score(a, b, bodies):
    errs = []
    for body in bodies:
        if body not in a or body not in b:
            continue
        errs.append(norm3(sub3(a[body]["r"], b[body]["r"])))
    if not errs:
        return float("inf"), float("inf"), 0
    rms = math.sqrt(sum(e * e for e in errs) / len(errs))
    return rms, max(errs), len(errs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--principia-csv", required=True)
    ap.add_argument("--krpc-csv", required=True)
    ap.add_argument("--central-body", default="Sun")
    ap.add_argument("--sample-index", type=int, default=0)
    args = ap.parse_args()

    pr_raw = read_principia(args.principia_csv, args.central_body)
    kr_raw = read_krpc(args.krpc_csv, args.sample_index, args.central_body)

    common = sorted(set(pr_raw) & set(kr_raw))
    print("common bodies:", len(common))
    print("principia time:", pr_raw[common[0]]["t"])
    print("krpc time:", kr_raw[common[0]]["t"])
    print("raw delta:", kr_raw[common[0]]["t"] - pr_raw[common[0]]["t"])

    results = []

    for pr_label, Tpr in transforms():
        pr = relative(pr_raw, Tpr, args.central_body)

        for kr_label, Tkr in transforms():
            kr = relative(kr_raw, Tkr, args.central_body)
            rms, mx, n = score(pr, kr, common)
            results.append((rms, mx, n, pr_label, kr_label))

    results.sort(key=lambda x: x[0])

    print("\n=== BEST TRANSFORMS ===")
    for rms, mx, n, pr_label, kr_label in results[:20]:
        print(
            f"rms={rms/1000:12.6f} km "
            f"max={mx/1000:12.6f} km "
            f"n={n:2d} "
            f"Principia={pr_label:<18} kRPC={kr_label}"
        )


if __name__ == "__main__":
    main()