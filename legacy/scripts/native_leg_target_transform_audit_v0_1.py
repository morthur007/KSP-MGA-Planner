#!/usr/bin/env python3
"""
native_leg_target_transform_audit_v0_1.py

Brute-force/audit transforms for one Lambert candidate leg using the native
Principia particle validator, but scoring against the target body, not against
Kepler.

This is the clean test we need now:
  Lambert candidate -> trimmed particle initial state -> Principia N-body flow
  then compare particle(t_end) against target_body(t_end) from the SPK kernel.

This separates:
  - frame/axis transform errors;
  - actual miss distance of the patched-conic Lambert leg under N-body;
  - Kepler-vs-N-body divergence, which is not itself a frame test.

Requires existing scripts/tools:
  - lambert_candidate_to_particle_leg_v0_1.py
  - principia_particle_validator in PATH or passed via --validator

Example:
  python native_leg_target_transform_audit_v0_1.py \
    --candidate-csv data/mga_smoke/kekj_lambert_w25_parallel_merged.csv \
    --rank 1 \
    --leg 1 \
    --bsp data/spice_jnsq_v0_3/jnsq_principia_native_v0_3_1000y_1h_deg9.bsp \
    --tpc data/spice_jnsq_v0_3/jnsq_principia_native_v0_3_1000y_1h_deg9.ids.tpc \
    --metadata data/spice_jnsq_v0_3/jnsq_principia_native_v0_3_1000y_1h_deg9.metadata.json \
    --body-catalog data/jnsq_gate0/ksp_future_paused/body_catalog.json \
    --plugin-b64 data/jnsq_gate0/principia_serialized_plugin.b64 \
    --central-body Sun \
    --sequence Kerbin Eve Kerbin Jool \
    --buffer-days 0.03 \
    --work-dir data/mga_smoke/native_target_audit/rank1_leg1 \
    --output-csv data/mga_smoke/native_target_audit/rank1_leg1_target_transform_audit.csv
"""

from __future__ import annotations

import argparse
import csv
import itertools
import math
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import spiceypy as spice

AXES = ["X", "Y", "Z"]
SIGNS = ["+", "-"]


def norm_name(name: str) -> str:
    return str(name).strip().upper()


def all_transforms() -> List[str]:
    out: List[str] = []
    for perm in itertools.permutations(AXES):
        for signs in itertools.product(SIGNS, repeat=3):
            out.append(",".join(s + a for s, a in zip(signs, perm)))
    return out


def parse_transform(spec: str) -> Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]:
    parts = [p.strip().upper() for p in spec.split(",")]
    if len(parts) != 3:
        raise ValueError(f"bad transform: {spec}")
    mapping = {"X": 0, "Y": 1, "Z": 2}
    out = []
    used = []
    for p in parts:
        if len(p) != 2 or p[0] not in "+-" or p[1] not in mapping:
            raise ValueError(f"bad transform component: {p}")
        sign = 1 if p[0] == "+" else -1
        idx = mapping[p[1]]
        out.append((sign, idx))
        used.append(idx)
    if sorted(used) != [0, 1, 2]:
        raise ValueError(f"transform must use X,Y,Z once: {spec}")
    return tuple(out)  # type: ignore[return-value]


def apply_transform(vec: Sequence[float], transform: Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]) -> List[float]:
    return [sign * float(vec[idx]) for sign, idx in transform]


def read_one_csv(path: Path) -> Dict[str, str]:
    with path.open(newline="") as f:
        return next(csv.DictReader(f))


def vector_from_row(row: Dict[str, str], keys: Sequence[str]) -> List[float]:
    return [float(row[k]) for k in keys]


def norm3(v: Sequence[float]) -> float:
    return math.sqrt(sum(float(x) * float(x) for x in v))


def sub3(a: Sequence[float], b: Sequence[float]) -> List[float]:
    return [float(a[i]) - float(b[i]) for i in range(3)]


def run_cmd(cmd: Sequence[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + " ".join(cmd) + "\n\n")
        log.flush()
        proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, text=True)
        return int(proc.returncode)


def target_state_principia_frame(
    body: str,
    et_s: float,
    central_body: str,
    transform_spec: str,
) -> Tuple[List[float], List[float]]:
    st, _ = spice.spkezr(norm_name(body), float(et_s), "J2000", "NONE", norm_name(central_body))
    tr = parse_transform(transform_spec)
    r_m = apply_transform([st[0] * 1000.0, st[1] * 1000.0, st[2] * 1000.0], tr)
    v_m_s = apply_transform([st[3] * 1000.0, st[4] * 1000.0, st[5] * 1000.0], tr)
    return r_m, v_m_s


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--candidate-csv", type=Path, required=True)
    p.add_argument("--rank", type=int, default=1)
    p.add_argument("--leg", type=int, required=True)
    p.add_argument("--bsp", type=Path, required=True)
    p.add_argument("--tpc", type=Path, required=True)
    p.add_argument("--metadata", type=Path, action="append", default=[])
    p.add_argument("--body-catalog", type=Path, action="append", default=[])
    p.add_argument("--plugin-b64", type=Path, required=True)
    p.add_argument("--central-body", default="Sun")
    p.add_argument("--sequence", nargs="+", required=True)
    p.add_argument("--buffer-days", type=float, default=0.03)
    p.add_argument("--work-dir", type=Path, required=True)
    p.add_argument("--output-csv", type=Path, required=True)
    p.add_argument("--generator", default="python lambert_candidate_to_particle_leg_v0_1.py")
    p.add_argument("--validator", default="principia_particle_validator")
    p.add_argument("--transforms", nargs="*", default=None, help="Optional subset, e.g. +Z,-Y,+X +X,+Y,+Z")
    p.add_argument("--keep-going", action="store_true", help="Continue after individual command failures.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    sequence = [norm_name(x) for x in args.sequence]
    if args.leg < 1 or args.leg >= len(sequence):
        raise SystemExit(f"--leg must be in 1..{len(sequence)-1}")
    target_body = sequence[args.leg]

    spice.furnsh(str(args.tpc))
    spice.furnsh(str(args.bsp))

    transforms = args.transforms if args.transforms else all_transforms()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)

    generator_prefix = args.generator.split()
    validator_prefix = args.validator.split()

    rows: List[Dict[str, object]] = []

    print(f"[INFO] transforms={len(transforms)} rank={args.rank} leg={args.leg} target={target_body} buffer={args.buffer_days} d")

    for idx, tr in enumerate(transforms, start=1):
        safe_tr = tr.replace(",", "_").replace("+", "p").replace("-", "m")
        run_dir = args.work_dir / safe_tr
        input_csv = run_dir / "particle_input.csv"
        expected_csv = run_dir / "kepler_expected.csv"
        output_csv = run_dir / "particle_output.csv"
        gen_log = run_dir / "generate.log"
        val_log = run_dir / "validator.log"

        gen_cmd = [
            *generator_prefix,
            "--candidate-csv", str(args.candidate_csv),
            "--rank", str(args.rank),
            "--bsp", str(args.bsp),
            "--tpc", str(args.tpc),
            "--central-body", str(args.central_body),
            "--sequence", *sequence,
            "--leg", str(args.leg),
            "--start-buffer-days", str(args.buffer_days),
            "--end-buffer-days", str(args.buffer_days),
            f"--spice-to-principia-transform={tr}",
            "--output-input-csv", str(input_csv),
            "--output-expected-csv", str(expected_csv),
        ]
        for m in args.metadata:
            gen_cmd.extend(["--metadata", str(m)])
        for b in args.body_catalog:
            gen_cmd.extend(["--body-catalog", str(b)])

        rc = run_cmd(gen_cmd, gen_log)
        if rc != 0:
            msg = f"generator failed rc={rc} log={gen_log}"
            print(f"[{idx:02d}/{len(transforms)}] {tr:<10} ERROR {msg}")
            if not args.keep_going:
                raise SystemExit(msg)
            rows.append({"transform": tr, "status": "generator_error", "message": msg})
            continue

        val_cmd = [
            *validator_prefix,
            str(args.plugin_b64),
            str(input_csv),
            str(output_csv),
        ]
        rc = run_cmd(val_cmd, val_log)
        if rc != 0:
            msg = f"validator failed rc={rc} log={val_log}"
            print(f"[{idx:02d}/{len(transforms)}] {tr:<10} ERROR {msg}")
            if not args.keep_going:
                raise SystemExit(msg)
            rows.append({"transform": tr, "status": "validator_error", "message": msg})
            continue

        o = read_one_csv(output_csv)
        e = read_one_csv(expected_csv)
        status = o.get("status", "")
        if status != "ok":
            msg = o.get("message", "")
            print(f"[{idx:02d}/{len(transforms)}] {tr:<10} ERROR native status={status} {msg}")
            rows.append({"transform": tr, "status": status, "message": msg})
            continue

        t_end = float(o["t1_s"])
        target_r, target_v = target_state_principia_frame(target_body, t_end, args.central_body, tr)
        part_r = vector_from_row(o, ["x_m", "y_m", "z_m"])
        part_v = vector_from_row(o, ["vx_m_s", "vy_m_s", "vz_m_s"])
        kep_r = vector_from_row(e, ["x_m", "y_m", "z_m"])
        kep_v = vector_from_row(e, ["vx_m_s", "vy_m_s", "vz_m_s"])

        miss_m = norm3(sub3(part_r, target_r))
        relv_m_s = norm3(sub3(part_v, target_v))
        nbody_vs_kepler_m = norm3(sub3(part_r, kep_r))
        nbody_vs_kepler_v_m_s = norm3(sub3(part_v, kep_v))
        kepler_target_m = norm3(sub3(kep_r, target_r))
        kepler_target_v_m_s = norm3(sub3(kep_v, target_v))

        row = {
            "transform": tr,
            "status": "ok",
            "target_body": target_body,
            "t_end_s": t_end,
            "miss_distance_km": miss_m / 1000.0,
            "relative_speed_m_s": relv_m_s,
            "nbody_vs_kepler_km": nbody_vs_kepler_m / 1000.0,
            "nbody_vs_kepler_v_m_s": nbody_vs_kepler_v_m_s,
            "kepler_target_km": kepler_target_m / 1000.0,
            "kepler_target_v_m_s": kepler_target_v_m_s,
            "input_csv": str(input_csv),
            "output_csv": str(output_csv),
            "expected_csv": str(expected_csv),
            "validator_log": str(val_log),
            "message": "",
        }
        rows.append(row)
        print(
            f"[{idx:02d}/{len(transforms)}] {tr:<10} miss={miss_m/1000.0:12.3f} km "
            f"relv={relv_m_s:10.3f} m/s nbody-kepler={nbody_vs_kepler_m/1000.0:10.3f} km"
        )

    rows.sort(key=lambda r: float(r.get("miss_distance_km", float("inf"))))

    fieldnames = [
        "transform",
        "status",
        "target_body",
        "t_end_s",
        "miss_distance_km",
        "relative_speed_m_s",
        "nbody_vs_kepler_km",
        "nbody_vs_kepler_v_m_s",
        "kepler_target_km",
        "kepler_target_v_m_s",
        "input_csv",
        "output_csv",
        "expected_csv",
        "validator_log",
        "message",
    ]
    with args.output_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})

    print("\n=== BEST BY TARGET MISS ===")
    for r in rows[:10]:
        if r.get("status") != "ok":
            continue
        print(
            f"{r['transform']:<10} miss={float(r['miss_distance_km']):12.3f} km "
            f"relv={float(r['relative_speed_m_s']):10.3f} m/s "
            f"nbody-kepler={float(r['nbody_vs_kepler_km']):10.3f} km"
        )
    print(f"\n[OK] wrote {args.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())