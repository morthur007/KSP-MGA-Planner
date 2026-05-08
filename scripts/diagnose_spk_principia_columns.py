#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import subprocess
from pathlib import Path

import numpy as np
import spiceypy as spice


def norm_name(s: str) -> str:
    return s.strip().upper()


def norm(v) -> float:
    return float(np.linalg.norm(np.asarray(v, dtype=float)))


def T_raw_to_levela(v: np.ndarray) -> np.ndarray:
    # raw -> LevelA/canonical from plugin.cpp comment:
    # (X,Y,Z) -> (-Y,+Z,+X)
    x, y, z = v
    return np.array([-y, z, x], dtype=float)


def T_levela_to_raw(v: np.ndarray) -> np.ndarray:
    # inverse:
    # LevelA -> raw = (+Z,-X,+Y)
    x, y, z = v
    return np.array([z, -x, y], dtype=float)


def read_rows(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    return {norm_name(r["body"]): r for r in rows}


def vec(row: dict[str, str], prefix: str) -> np.ndarray:
    if prefix == "raw_r":
        keys = ["raw_x_m", "raw_y_m", "raw_z_m"]
    elif prefix == "raw_v":
        keys = ["raw_vx_m_s", "raw_vy_m_s", "raw_vz_m_s"]
    elif prefix == "r":
        keys = ["x_m", "y_m", "z_m"]
    elif prefix == "v":
        keys = ["vx_m_s", "vy_m_s", "vz_m_s"]
    else:
        raise ValueError(prefix)
    return np.array([float(row[k]) for k in keys], dtype=float)


def spk_state_m(body: str, et_s: float, central_body: str) -> tuple[np.ndarray, np.ndarray]:
    st, _ = spice.spkezr(norm_name(body), float(et_s), "J2000", "NONE", norm_name(central_body))
    st = np.asarray(st, dtype=float)
    return st[:3] * 1000.0, st[3:] * 1000.0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--sampler", default="sample_principia_ephemeris")
    p.add_argument("--plugin-b64", type=Path, required=True)
    p.add_argument("--plugin-base-et-s", type=float, default=81.65168640136972)
    p.add_argument("--et-s", type=float, required=True)
    p.add_argument("--central-body", default="Sun")
    p.add_argument("--body", default="Kerbin")
    p.add_argument("--bsp", type=Path, required=True)
    p.add_argument("--tpc", type=Path, required=True)
    p.add_argument("--output-csv", type=Path, required=True)
    args = p.parse_args()

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)

    offset_s = args.et_s - args.plugin_base_et_s

    cmd = [
        args.sampler,
        str(args.plugin_b64),
        str(args.output_csv),
        args.central_body,
        f"{offset_s:.17g}",
        "0",
        "21600",
    ]

    subprocess.run(cmd, check=True)

    rows = read_rows(args.output_csv)

    central = norm_name(args.central_body)
    body = norm_name(args.body)

    if central not in rows:
        raise SystemExit(f"[FAIL] central body {central} not in CSV")
    if body not in rows:
        raise SystemExit(f"[FAIL] body {body} not in CSV")

    c = rows[central]
    b = rows[body]

    csv_r = vec(b, "r")
    csv_v = vec(b, "v")

    raw_r_c = vec(c, "raw_r")
    raw_v_c = vec(c, "raw_v")
    raw_r_b = vec(b, "raw_r")
    raw_v_b = vec(b, "raw_v")

    raw_rel_r = raw_r_b - raw_r_c
    raw_rel_v = raw_v_b - raw_v_c

    levela_from_raw_r = T_raw_to_levela(raw_rel_r)
    levela_from_raw_v = T_raw_to_levela(raw_rel_v)

    raw_from_levela_r = T_levela_to_raw(csv_r)
    raw_from_levela_v = T_levela_to_raw(csv_v)

    spice.furnsh(str(args.tpc))
    spice.furnsh(str(args.bsp))
    spk_r, spk_v = spk_state_m(body, args.et_s, central)

    print("=== SPK / PRINCIPIA COLUMN FRAME DIAGNOSIS ===")
    print(f"et_s       : {args.et_s:.12f}")
    print(f"central    : {central}")
    print(f"body       : {body}")
    print("")
    print("CSV transformed columns:")
    print(f"  csv_r x_m/y_m/z_m       = {csv_r}")
    print(f"  csv_v vx/vy/vz          = {csv_v}")
    print("")
    print("Raw barycentric:")
    print(f"  central raw_r           = {raw_r_c}")
    print(f"  body raw_r              = {raw_r_b}")
    print(f"  raw_rel_r body-central  = {raw_rel_r}")
    print("")
    print("Candidates:")
    print(f"  T(raw_rel_r)            = {levela_from_raw_r}")
    print(f"  T_inv(csv_r)            = {raw_from_levela_r}")
    print("")
    print("SPK state:")
    print(f"  spk_r body wrt central  = {spk_r}")
    print(f"  spk_v body wrt central  = {spk_v}")
    print("")
    print("Position residuals vs SPK:")
    print(f"  |SPK - csv_x_m|          = {norm(spk_r - csv_r):.6f} m")
    print(f"  |SPK - raw_rel|          = {norm(spk_r - raw_rel_r):.6f} m")
    print(f"  |SPK - T(raw_rel)|       = {norm(spk_r - levela_from_raw_r):.6f} m")
    print(f"  |SPK - T_inv(csv)|       = {norm(spk_r - raw_from_levela_r):.6f} m")
    print("")
    print("Velocity residuals vs SPK:")
    print(f"  |SPK - csv_v|            = {norm(spk_v - csv_v):.9f} m/s")
    print(f"  |SPK - raw_rel_v|        = {norm(spk_v - raw_rel_v):.9f} m/s")
    print(f"  |SPK - T(raw_rel_v)|     = {norm(spk_v - levela_from_raw_v):.9f} m/s")
    print(f"  |SPK - T_inv(csv_v)|     = {norm(spk_v - raw_from_levela_v):.9f} m/s")

    best = min(
        [
            ("csv_x_m", norm(spk_r - csv_r)),
            ("raw_rel", norm(spk_r - raw_rel_r)),
            ("T(raw_rel)", norm(spk_r - levela_from_raw_r)),
            ("T_inv(csv)", norm(spk_r - raw_from_levela_r)),
        ],
        key=lambda x: x[1],
    )

    print("")
    print(f"BEST POSITION MATCH: {best[0]} residual={best[1]:.6f} m")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
