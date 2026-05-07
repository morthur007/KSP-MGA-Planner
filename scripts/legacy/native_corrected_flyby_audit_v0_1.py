#!/usr/bin/env python3
"""
native_corrected_flyby_audit_v0_1.py

Final audit for corrected N-body MGA legs.

Purpose:
  After native_optimize_candidate_legs_v0_1.py has optimized each leg
  independently, this script checks whether adjacent corrected legs are still
  connectable as physical gravity assists at the intermediate bodies.

For each flyby body B between legs:
  incoming leg ends at   t_event - buffer
  outgoing leg starts at t_event + buffer

We compute:
  v_inf_in  = spacecraft_velocity_incoming - body_velocity(t_event - buffer)
  v_inf_out = spacecraft_velocity_outgoing - body_velocity(t_event + buffer)

Then audit:
  - |v_inf_in|
  - |v_inf_out|
  - powered mismatch = ||v_inf_out| - |v_inf_in||
  - required turn angle between v_inf_in and v_inf_out
  - maximum turn angle from μ, rp_min, and average v∞
  - turn margin

Notes:
  This is still a patched flyby audit around the encounter, not a continuous
  close-approach integration through the sphere of influence. It is the right
  final check before moving to Vessel/FlightPlan/live bridge work.

Frame convention:
  plugin.cpp writer:
      Principia raw -> LevelA/SPK canonical = (-Y, +Z, +X)
  inverse used here:
      LevelA/SPK canonical -> Principia raw = (+Z, -X, +Y)

Inputs:
  - original candidate CSV and rank;
  - native leg optimization summary CSV;
  - SPICE kernels and body metadata;
  - Principia plugin.b64 and particle validator.

Output:
  - CSV with one row per flyby.
  - JSON summary with totals and pass/fail flags.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import spiceypy as spice

from spice_lambert_mga_v0_1 import (
    DAY_S,
    BodyInfo,
    lambert_universal_zero_rev,
    load_body_catalog,
    norm_name,
)
from lambert_candidate_to_particle_leg_v0_1 import (
    apply_transform,
    kepler_universal_propagate,
    parse_transform,
)


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def read_candidate(path: Path, rank: int) -> Dict[str, str]:
    rows = read_csv_rows(path)
    if not rows:
        raise ValueError(f"candidate CSV vazio: {path}")
    if rank < 1 or rank > len(rows):
        raise ValueError(f"rank fora de 1..{len(rows)}: {rank}")
    return rows[rank - 1]


def get_float(row: Dict[str, str], key: str) -> float:
    if key not in row or row[key] == "":
        raise KeyError(f"coluna ausente/vazia: {key}")
    return float(row[key])


def safe_float(row: Dict[str, str], key: str, default: float = float("nan")) -> float:
    try:
        return float(row.get(key, ""))
    except Exception:
        return default


def norm(v: Sequence[float]) -> float:
    return float(np.linalg.norm(np.asarray(v, dtype=float)))


def safe_acos(x: float) -> float:
    return math.acos(max(-1.0, min(1.0, x)))


def turn_angle_max_rad(mu_km3_s2: float, rp_km: float, vinf_km_s: float) -> float:
    if mu_km3_s2 <= 0 or rp_km <= 0 or vinf_km_s <= 0:
        return 0.0
    denom = rp_km * vinf_km_s * vinf_km_s / mu_km3_s2 + 1.0
    return 2.0 * math.asin(max(0.0, min(1.0, 1.0 / denom)))


def spk_state(body: str, et_s: float, central_body: str) -> np.ndarray:
    st, _ = spice.spkezr(norm_name(body), float(et_s), "J2000", "NONE", norm_name(central_body))
    return np.asarray(st, dtype=float)  # km, km/s in LevelA/SPK canonical frame


def spk_body_state_raw(
    body: str,
    et_s: float,
    central_body: str,
    transform_spec: str,
) -> Tuple[np.ndarray, np.ndarray]:
    st = spk_state(body, et_s, central_body)
    tr = parse_transform(transform_spec)
    r_raw_m = np.asarray(apply_transform(st[:3] * 1000.0, tr), dtype=float)
    v_raw_m_s = np.asarray(apply_transform(st[3:] * 1000.0, tr), dtype=float)
    return r_raw_m, v_raw_m_s


def write_particle_input(path: Path, row_id: str, t0_s: float, t1_s: float, r_m: np.ndarray, v_m_s: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["id", "t0_s", "t1_s", "x_m", "y_m", "z_m", "vx_m_s", "vy_m_s", "vz_m_s"])
        w.writerow([
            row_id,
            f"{t0_s:.17g}",
            f"{t1_s:.17g}",
            f"{r_m[0]:.17g}",
            f"{r_m[1]:.17g}",
            f"{r_m[2]:.17g}",
            f"{v_m_s[0]:.17g}",
            f"{v_m_s[1]:.17g}",
            f"{v_m_s[2]:.17g}",
        ])


def read_validator_output(path: Path) -> Dict[str, str]:
    with path.open(newline="") as f:
        return next(csv.DictReader(f))


def run_validator(
    validator: str,
    plugin_b64: Path,
    input_csv: Path,
    output_csv: Path,
    log_path: Path,
) -> int:
    cmd = validator.split() + [str(plugin_b64), str(input_csv), str(output_csv)]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + " ".join(cmd) + "\n\n")
        log.flush()
        proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, text=True)
    return int(proc.returncode)


def reconstruct_leg_start_position_raw(
    candidate: Dict[str, str], sequence: Sequence[str], leg: int,
    central_body: str, central_mu_km3_s2: float, t_start_s: float, transform_spec: str
) -> np.ndarray:
    dep_body = sequence[leg - 1]
    t_dep = get_float(candidate, f"event{leg-1}_{dep_body}_et_s")
    st_dep = spk_state(dep_body, t_dep, central_body)
    
    # Leitura direta da semente
    target_v1 = np.array([
        get_float(candidate, f"leg{leg}_vdep_x_km_s"),
        get_float(candidate, f"leg{leg}_vdep_y_km_s"),
        get_float(candidate, f"leg{leg}_vdep_z_km_s")
    ])
        
    from lambert_candidate_to_particle_leg_v0_1 import kepler_universal_propagate, parse_transform, apply_transform
    r_start_km, _ = kepler_universal_propagate(st_dep[:3], target_v1, t_start_s - t_dep, central_mu_km3_s2)
    
    tr = parse_transform(transform_spec)
    return apply_transform(r_start_km * 1000.0, tr)


def leg_rows_by_index(summary_csv: Path) -> Dict[int, Dict[str, str]]:
    rows = read_csv_rows(summary_csv)
    out: Dict[int, Dict[str, str]] = {}
    for row in rows:
        leg = int(float(row["leg"]))
        out[leg] = row
    return out


def vector_from_summary(row: Dict[str, str], prefix: str) -> np.ndarray:
    return np.asarray([
        get_float(row, f"{prefix}_vx_m_s"),
        get_float(row, f"{prefix}_vy_m_s"),
        get_float(row, f"{prefix}_vz_m_s"),
    ], dtype=float)


def propagate_corrected_leg(
    candidate: Dict[str, str],
    leg_row: Dict[str, str],
    sequence: Sequence[str],
    central_body: str,
    central_mu_km3_s2: float,
    transform_spec: str,
    validator: str,
    plugin_b64: Path,
    work_dir: Path,
) -> Dict[str, Any]:
    leg = int(float(leg_row["leg"]))
    dep_body = norm_name(leg_row["dep_body"])
    arr_body = norm_name(leg_row["arr_body"])
    t_start = get_float(leg_row, "t_start_s")
    t_end = get_float(leg_row, "t_end_s")
    v0_raw = vector_from_summary(leg_row, "optimized")
    r0_raw = reconstruct_leg_start_position_raw(
        candidate=candidate,
        sequence=sequence,
        leg=leg,
        central_body=central_body,
        central_mu_km3_s2=central_mu_km3_s2,
        t_start_s=t_start,
        transform_spec=transform_spec,
    )

    leg_dir = work_dir / f"leg{leg}_{dep_body}_to_{arr_body}"
    input_csv = leg_dir / "corrected_input.csv"
    output_csv = leg_dir / "corrected_output.csv"
    log_path = leg_dir / "validator.log"
    row_id = f"corrected_leg{leg}_{dep_body}_to_{arr_body}"

    write_particle_input(input_csv, row_id, t_start, t_end, r0_raw, v0_raw)
    rc = run_validator(validator, plugin_b64, input_csv, output_csv, log_path)
    if rc != 0:
        raise RuntimeError(f"validator failed for leg {leg}; rc={rc}; log={log_path}")

    out = read_validator_output(output_csv)
    if out.get("status") != "ok":
        raise RuntimeError(f"validator native status failed for leg {leg}: {out.get('status')} {out.get('message')}")

    final_r = np.asarray([float(out["x_m"]), float(out["y_m"]), float(out["z_m"])], dtype=float)
    final_v = np.asarray([float(out["vx_m_s"]), float(out["vy_m_s"]), float(out["vz_m_s"])], dtype=float)
    target_r, target_v = spk_body_state_raw(arr_body, t_end, central_body, transform_spec)

    return {
        "leg": leg,
        "dep_body": dep_body,
        "arr_body": arr_body,
        "t_start_s": t_start,
        "t_end_s": t_end,
        "start_r_raw_m": r0_raw,
        "start_v_raw_m_s": v0_raw,
        "final_r_raw_m": final_r,
        "final_v_raw_m_s": final_v,
        "target_r_raw_m": target_r,
        "target_v_raw_m_s": target_v,
        "final_miss_km": norm(final_r - target_r) / 1000.0,
        "final_relv_m_s": norm(final_v - target_v),
        "input_csv": str(input_csv),
        "output_csv": str(output_csv),
        "validator_log": str(log_path),
    }


def write_outputs(csv_path: Path, json_path: Path, flyby_rows: List[Dict[str, Any]], summary: Dict[str, Any]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "flyby_index",
        "event_index",
        "body",
        "event_time_s",
        "incoming_leg",
        "outgoing_leg",
        "t_in_s",
        "t_out_s",
        "vinf_in_km_s",
        "vinf_out_km_s",
        "vinf_mismatch_km_s",
        "turn_required_deg",
        "turn_max_deg",
        "turn_margin_deg",
        "rp_min_km",
        "mu_km3_s2",
        "incoming_miss_km",
        "outgoing_start_distance_from_body_km",
        "incoming_relv_m_s",
        "outgoing_relv_m_s",
        "status",
    ]
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in flyby_rows:
            w.writerow({k: row.get(k, "") for k in fields})

    json_path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(summary)
    payload["flybys"] = flyby_rows
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--candidate-csv", type=Path, required=True)
    p.add_argument("--rank", type=int, default=1)
    p.add_argument("--leg-summary-csv", type=Path, required=True)
    p.add_argument("--bsp", type=Path, required=True)
    p.add_argument("--tpc", type=Path, required=True)
    p.add_argument("--metadata", type=Path, action="append", default=[])
    p.add_argument("--body-catalog", type=Path, action="append", default=[])
    p.add_argument("--plugin-b64", type=Path, required=True)
    p.add_argument("--central-body", default="Sun")
    p.add_argument("--central-mu-km3-s2", type=float, default=None)
    p.add_argument("--sequence", nargs="+", required=True)
    p.add_argument("--transform", default="+Z,-X,+Y")
    p.add_argument("--validator", default="principia_particle_validator")
    p.add_argument("--work-dir", type=Path, required=True)
    p.add_argument("--output-csv", type=Path, required=True)
    p.add_argument("--output-json", type=Path, required=True)
    p.add_argument("--rp-altitude-km", type=float, default=50.0)
    p.add_argument("--rp-scale", type=float, default=1.05)
    p.add_argument("--max-vinf-mismatch-km-s", type=float, default=0.10)
    p.add_argument("--min-turn-margin-deg", type=float, default=0.0)
    return p.parse_args()


def main() -> int:
    args = parse_args()

    for k in [args.tpc, args.bsp]:
        spice.furnsh(str(k))

    sequence = [norm_name(x) for x in args.sequence]
    candidate = read_candidate(args.candidate_csv, args.rank)
    leg_rows = leg_rows_by_index(args.leg_summary_csv)
    nlegs = len(sequence) - 1
    missing = [leg for leg in range(1, nlegs + 1) if leg not in leg_rows]
    if missing:
        raise SystemExit(f"leg-summary não contém pernas: {missing}")

    body_info = load_body_catalog([*args.body_catalog, *args.metadata])
    central = norm_name(args.central_body)
    central_mu = args.central_mu_km3_s2
    if central_mu is None:
        info = body_info.get(central)
        if info and info.mu_km3_s2:
            central_mu = info.mu_km3_s2
    if central_mu is None:
        raise SystemExit("Não encontrei μ central. Passe --central-mu-km3-s2 ou metadata/body-catalog.")

    print("=== NATIVE CORRECTED FLYBY AUDIT V0.1 ===")
    print(f"candidate: {args.candidate_csv} rank={args.rank}")
    print(f"summary  : {args.leg_summary_csv}")
    print(f"sequence : {' -> '.join(sequence)}")
    print(f"transform: {args.transform}")

    propagated: Dict[int, Dict[str, Any]] = {}
    for leg in range(1, nlegs + 1):
        print(f"[INFO] propagating corrected leg {leg}/{nlegs}: {sequence[leg-1]}->{sequence[leg]}")
        propagated[leg] = propagate_corrected_leg(
            candidate=candidate,
            leg_row=leg_rows[leg],
            sequence=sequence,
            central_body=central,
            central_mu_km3_s2=float(central_mu),
            transform_spec=args.transform,
            validator=args.validator,
            plugin_b64=args.plugin_b64,
            work_dir=args.work_dir,
        )
        print(
            f"       miss={propagated[leg]['final_miss_km']:.9f} km "
            f"relv={propagated[leg]['final_relv_m_s']:.3f} m/s"
        )

    flyby_rows: List[Dict[str, Any]] = []
    for j in range(1, len(sequence) - 1):
        body = sequence[j]
        incoming = propagated[j]
        outgoing = propagated[j + 1]
        event_time = get_float(candidate, f"event{j}_{body}_et_s")

        # incoming is evaluated at t_event - buffer; outgoing starts at t_event + buffer.
        _body_r_in, body_v_in = spk_body_state_raw(body, incoming["t_end_s"], central, args.transform)
        body_r_out, body_v_out = spk_body_state_raw(body, outgoing["t_start_s"], central, args.transform)

        vinf_in_vec_m_s = incoming["final_v_raw_m_s"] - body_v_in
        vinf_out_vec_m_s = outgoing["start_v_raw_m_s"] - body_v_out
        vinf_in_km_s = norm(vinf_in_vec_m_s) / 1000.0
        vinf_out_km_s = norm(vinf_out_vec_m_s) / 1000.0
        vinf_mismatch_km_s = abs(vinf_out_km_s - vinf_in_km_s)

        if vinf_in_km_s > 0 and vinf_out_km_s > 0:
            turn_required_rad = safe_acos(
                float(np.dot(vinf_in_vec_m_s, vinf_out_vec_m_s) /
                      (norm(vinf_in_vec_m_s) * norm(vinf_out_vec_m_s)))
            )
        else:
            turn_required_rad = math.pi

        info = body_info.get(body, BodyInfo())
        mu = info.mu_km3_s2
        radius = info.radius_km
        if radius is None:
            rp_min = float("nan")
        else:
            rp_min = max(radius + args.rp_altitude_km, radius * args.rp_scale)

        vinf_avg = 0.5 * (vinf_in_km_s + vinf_out_km_s)
        if mu is not None and math.isfinite(rp_min):
            turn_max_rad = turn_angle_max_rad(mu, rp_min, vinf_avg)
            turn_margin_deg = math.degrees(turn_max_rad - turn_required_rad)
            turn_max_deg = math.degrees(turn_max_rad)
        else:
            turn_max_deg = float("nan")
            turn_margin_deg = float("nan")

        outgoing_start_distance_km = norm(outgoing["start_r_raw_m"] - body_r_out) / 1000.0

        ok = True
        if not math.isfinite(turn_margin_deg) or turn_margin_deg < args.min_turn_margin_deg:
            ok = False
        if vinf_mismatch_km_s > args.max_vinf_mismatch_km_s:
            ok = False

        row = {
            "flyby_index": j,
            "event_index": j,
            "body": body,
            "event_time_s": event_time,
            "incoming_leg": j,
            "outgoing_leg": j + 1,
            "t_in_s": incoming["t_end_s"],
            "t_out_s": outgoing["t_start_s"],
            "vinf_in_km_s": vinf_in_km_s,
            "vinf_out_km_s": vinf_out_km_s,
            "vinf_mismatch_km_s": vinf_mismatch_km_s,
            "turn_required_deg": math.degrees(turn_required_rad),
            "turn_max_deg": turn_max_deg,
            "turn_margin_deg": turn_margin_deg,
            "rp_min_km": rp_min,
            "mu_km3_s2": mu if mu is not None else float("nan"),
            "incoming_miss_km": incoming["final_miss_km"],
            "outgoing_start_distance_from_body_km": outgoing_start_distance_km,
            "incoming_relv_m_s": incoming["final_relv_m_s"],
            "outgoing_relv_m_s": norm(vinf_out_vec_m_s),
            "status": "PASS" if ok else "CHECK",
        }
        flyby_rows.append(row)

    total_leg_correction_m_s = sum(safe_float(leg_rows[i], "dv_norm_m_s", 0.0) for i in range(1, nlegs + 1))
    total_powered_flyby_mismatch_km_s = sum(float(r["vinf_mismatch_km_s"]) for r in flyby_rows)
    min_turn_margin_deg = min((float(r["turn_margin_deg"]) for r in flyby_rows), default=float("nan"))
    max_vinf_mismatch_km_s = max((float(r["vinf_mismatch_km_s"]) for r in flyby_rows), default=float("nan"))

    summary = {
        "candidate_csv": str(args.candidate_csv),
        "rank": args.rank,
        "leg_summary_csv": str(args.leg_summary_csv),
        "sequence": sequence,
        "transform": args.transform,
        "total_leg_correction_m_s": total_leg_correction_m_s,
        "total_powered_flyby_mismatch_km_s": total_powered_flyby_mismatch_km_s,
        "max_vinf_mismatch_km_s": max_vinf_mismatch_km_s,
        "min_turn_margin_deg": min_turn_margin_deg,
        "threshold_max_vinf_mismatch_km_s": args.max_vinf_mismatch_km_s,
        "threshold_min_turn_margin_deg": args.min_turn_margin_deg,
        "status": "PASS" if all(r["status"] == "PASS" for r in flyby_rows) else "CHECK",
    }

    write_outputs(args.output_csv, args.output_json, flyby_rows, summary)

    print("\n=== CORRECTED FLYBY AUDIT ===")
    for r in flyby_rows:
        print(
            f"{r['body']:<8} "
            f"vinf_in={float(r['vinf_in_km_s']):8.4f} km/s "
            f"vinf_out={float(r['vinf_out_km_s']):8.4f} km/s "
            f"mis={float(r['vinf_mismatch_km_s']):8.4f} km/s "
            f"turn={float(r['turn_required_deg']):8.3f}/{float(r['turn_max_deg']):8.3f} deg "
            f"margin={float(r['turn_margin_deg']):8.3f} deg "
            f"{r['status']}"
        )

    print("\n=== SUMMARY ===")
    print(f"total_leg_correction_m_s        : {total_leg_correction_m_s:.6f}")
    print(f"total_flyby_mismatch_km_s       : {total_powered_flyby_mismatch_km_s:.9f}")
    print(f"max_vinf_mismatch_km_s          : {max_vinf_mismatch_km_s:.9f}")
    print(f"min_turn_margin_deg             : {min_turn_margin_deg:.9f}")
    print(f"status                           : {summary['status']}")
    print(f"[OK] CSV : {args.output_csv}")
    print(f"[OK] JSON: {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
