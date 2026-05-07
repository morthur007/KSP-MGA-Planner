#!/usr/bin/env python3
"""
residual_correction_cross_validate.py

Out-of-sample validation for observation-derived residual corrections.

Purpose
-------
The residual_correction_builder can exactly remove residuals at the sampled
observation epochs when smoothing is disabled. That proves the correction algebra,
but not interpolation/generalization. This script performs a holdout test:

  1. Align KSP/Principia observed states and REBOUND base states by ET.
  2. Build an empirical correction delta_r(t) = r_ksp(t) - r_reb(t)
     using only training epochs.
  3. Interpolate delta_r(t) to withheld epochs.
  4. Score corrected REBOUND states against KSP at withheld epochs.

This answers: "Is our sampling cadence/residual interpolation sufficient?"

Example
-------
python residual_correction_cross_validate.py \
  --ksp-csv data/final_clean_120d/states.csv \
  --reb-csv data/final_clean_120d/level_a_fit_urlum_family/rebound_states.csv \
  --bodies Vant Zore Plock Thatmo Soden Crokslev Vall Tylo Pol Laythe Bop Hale Ovok Slate Tekto Eeloo Polta Priax Wal Tal \
  --method pchip \
  --holdout-stride 4 \
  --output-dir data/final_clean_120d/residual_correction_cv_pchip_s4
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

try:
    from scipy.interpolate import PchipInterpolator, CubicSpline, interp1d  # type: ignore
except Exception as exc:  # pragma: no cover
    PchipInterpolator = None
    CubicSpline = None
    interp1d = None

BODY_COLS = ["body", "name", "body_name", "Body", "Name"]
TIME_COLS = ["et_seconds", "et", "time", "t", "ET", "seconds"]
POS_COLS = [
    ("x_m", "y_m", "z_m"),
    ("x", "y", "z"),
    ("px", "py", "pz"),
    ("pos_x_m", "pos_y_m", "pos_z_m"),
    ("rx", "ry", "rz"),
]
VEL_COLS = [
    ("vx_m_s", "vy_m_s", "vz_m_s"),
    ("vx", "vy", "vz"),
    ("vel_x_m_s", "vel_y_m_s", "vel_z_m_s"),
    ("v_x", "v_y", "v_z"),
]

@dataclass
class StateSeries:
    body: str
    t: np.ndarray
    r: np.ndarray
    v: Optional[np.ndarray]


def find_col(fieldnames: Iterable[str], candidates: List[str]) -> Optional[str]:
    fset = set(fieldnames)
    lower_map = {f.lower(): f for f in fieldnames}
    for c in candidates:
        if c in fset:
            return c
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    return None


def find_triplet(fieldnames: Iterable[str], candidates: List[Tuple[str, str, str]]) -> Optional[Tuple[str, str, str]]:
    fset = set(fieldnames)
    lower_map = {f.lower(): f for f in fieldnames}
    for trip in candidates:
        if all(c in fset for c in trip):
            return trip
        if all(c.lower() in lower_map for c in trip):
            return tuple(lower_map[c.lower()] for c in trip)  # type: ignore
    return None


def read_states(path: Path) -> Dict[str, StateSeries]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"CSV sem header: {path}")
        body_col = find_col(reader.fieldnames, BODY_COLS)
        time_col = find_col(reader.fieldnames, TIME_COLS)
        pos_cols = find_triplet(reader.fieldnames, POS_COLS)
        vel_cols = find_triplet(reader.fieldnames, VEL_COLS)
        if body_col is None or time_col is None or pos_cols is None:
            raise ValueError(f"Não detectei colunas body/time/position em {path}: {reader.fieldnames}")
        tmp: Dict[str, List[Tuple[float, np.ndarray, Optional[np.ndarray]]]] = defaultdict(list)
        for row in reader:
            try:
                body = row[body_col].strip()
                t = float(row[time_col])
                r = np.array([float(row[pos_cols[0]]), float(row[pos_cols[1]]), float(row[pos_cols[2]])], dtype=float)
                v = None
                if vel_cols is not None and all(row.get(c, "") != "" for c in vel_cols):
                    v = np.array([float(row[vel_cols[0]]), float(row[vel_cols[1]]), float(row[vel_cols[2]])], dtype=float)
                tmp[body].append((t, r, v))
            except Exception:
                continue
    out: Dict[str, StateSeries] = {}
    for body, rows in tmp.items():
        rows.sort(key=lambda x: x[0])
        t = np.array([x[0] for x in rows], dtype=float)
        r = np.vstack([x[1] for x in rows])
        v = np.vstack([x[2] for x in rows]) if all(x[2] is not None for x in rows) else None  # type: ignore
        out[body] = StateSeries(body, t, r, v)
    return out


def make_index(series: StateSeries, round_digits: int) -> Dict[float, int]:
    return {round(float(t), round_digits): i for i, t in enumerate(series.t)}


def align(ksp: StateSeries, reb: StateSeries, round_digits: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    ki = make_index(ksp, round_digits)
    ri = make_index(reb, round_digits)
    keys = sorted(set(ki).intersection(ri))
    if len(keys) < 6:
        raise ValueError(f"Poucos epochs comuns para {ksp.body}: {len(keys)}")
    ik = [ki[k] for k in keys]
    ir = [ri[k] for k in keys]
    return ksp.t[ik], ksp.r[ik], reb.r[ir]


def make_interpolator(t_train: np.ndarray, y_train: np.ndarray, method: str):
    # Work in shifted time to improve conditioning.
    t0 = float(t_train[0])
    x = t_train - t0
    if method == "pchip":
        if PchipInterpolator is None:
            raise RuntimeError("scipy.interpolate.PchipInterpolator indisponível")
        fs = [PchipInterpolator(x, y_train[:, j], extrapolate=False) for j in range(3)]
    elif method == "cubic":
        if CubicSpline is None:
            raise RuntimeError("scipy.interpolate.CubicSpline indisponível")
        fs = [CubicSpline(x, y_train[:, j], bc_type="not-a-knot", extrapolate=False) for j in range(3)]
    elif method == "linear":
        if interp1d is None:
            raise RuntimeError("scipy.interpolate.interp1d indisponível")
        fs = [interp1d(x, y_train[:, j], kind="linear", bounds_error=False, fill_value=np.nan) for j in range(3)]
    else:
        raise ValueError(f"method desconhecido: {method}")

    def eval_at(t_eval: np.ndarray) -> np.ndarray:
        xe = t_eval - t0
        return np.vstack([f(xe) for f in fs]).T
    return eval_at


def metrics(err_m: np.ndarray) -> Dict[str, float]:
    norms = np.linalg.norm(err_m, axis=1)
    return {
        "max_km": float(np.max(norms) / 1000.0),
        "rms_km": float(math.sqrt(np.mean(norms**2)) / 1000.0),
        "median_km": float(np.median(norms) / 1000.0),
        "p95_km": float(np.percentile(norms, 95) / 1000.0),
        "n": int(norms.size),
    }


def classify(max_km: float, rms_km: float) -> str:
    if max_km <= 1.0 or rms_km <= 0.1:
        return "A"
    if max_km <= 10.0 or rms_km <= 1.0:
        return "A-"
    if max_km <= 100.0 or rms_km <= 10.0:
        return "B"
    return "C"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ksp-csv", required=True, type=Path)
    ap.add_argument("--reb-csv", required=True, type=Path)
    ap.add_argument("--bodies", nargs="+", required=True)
    ap.add_argument("--method", choices=["pchip", "cubic", "linear"], default="pchip")
    ap.add_argument("--holdout-stride", type=int, default=4, help="Withhold every Nth interior sample")
    ap.add_argument("--holdout-offset", type=int, default=0)
    ap.add_argument("--round-digits", type=int, default=6)
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--write-samples", action="store_true")
    args = ap.parse_args()

    if args.holdout_stride < 2:
        raise ValueError("holdout-stride deve ser >= 2")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    ksp_all = read_states(args.ksp_csv)
    reb_all = read_states(args.reb_csv)

    rows = []
    sample_rows = []
    for body in args.bodies:
        if body not in ksp_all or body not in reb_all:
            print(f"[WARN] pulando {body}: ausente em KSP ou REB")
            continue
        t, r_ksp, r_reb = align(ksp_all[body], reb_all[body], args.round_digits)
        n = len(t)
        idx = np.arange(n)
        # Hold out interior points only; endpoints always train so interpolation stays in-domain.
        holdout = (idx % args.holdout_stride == args.holdout_offset) & (idx > 0) & (idx < n - 1)
        train = ~holdout
        if np.sum(holdout) < 2 or np.sum(train) < 4:
            print(f"[WARN] pulando {body}: holdout/train insuficiente")
            continue

        delta = r_ksp - r_reb
        interp = make_interpolator(t[train], delta[train], args.method)
        delta_pred = interp(t[holdout])
        ok = np.all(np.isfinite(delta_pred), axis=1)
        if not np.all(ok):
            # This should not happen since endpoints are train, but be safe.
            hidx = np.where(holdout)[0][ok]
            delta_pred = delta_pred[ok]
        else:
            hidx = np.where(holdout)[0]

        before_err = r_reb[hidx] - r_ksp[hidx]
        after_err = (r_reb[hidx] + delta_pred) - r_ksp[hidx]
        bmet = metrics(before_err)
        amet = metrics(after_err)
        improvement = (bmet["rms_km"] / amet["rms_km"]) if amet["rms_km"] > 0 else float("inf")
        rows.append({
            "body": body,
            "method": args.method,
            "holdout_stride": args.holdout_stride,
            "holdout_offset": args.holdout_offset,
            "n_total": n,
            "n_train": int(np.sum(train)),
            "n_holdout": int(len(hidx)),
            "before_max_km": bmet["max_km"],
            "before_rms_km": bmet["rms_km"],
            "before_p95_km": bmet["p95_km"],
            "after_max_km": amet["max_km"],
            "after_rms_km": amet["rms_km"],
            "after_p95_km": amet["p95_km"],
            "improvement_rms": improvement,
            "class_after": classify(amet["max_km"], amet["rms_km"]),
        })
        if args.write_samples:
            for k, ii in enumerate(hidx):
                sample_rows.append({
                    "body": body,
                    "et_seconds": float(t[ii]),
                    "before_err_m": float(np.linalg.norm(before_err[k])),
                    "after_err_m": float(np.linalg.norm(after_err[k])),
                    "after_dx_m": float(after_err[k, 0]),
                    "after_dy_m": float(after_err[k, 1]),
                    "after_dz_m": float(after_err[k, 2]),
                })

    rows.sort(key=lambda r: float(r["after_rms_km"]), reverse=True)
    out_csv = args.output_dir / "cross_validation_summary.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        fieldnames = list(rows[0].keys()) if rows else []
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    out_json = args.output_dir / "cross_validation_summary.json"
    out_json.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    if args.write_samples:
        out_samp = args.output_dir / "cross_validation_samples.csv"
        with out_samp.open("w", encoding="utf-8", newline="") as f:
            fieldnames = list(sample_rows[0].keys()) if sample_rows else ["body", "et_seconds", "before_err_m", "after_err_m"]
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(sample_rows)

    print("\n=== RESIDUAL CORRECTION CROSS-VALIDATION ===")
    print(f"method={args.method} holdout_stride={args.holdout_stride} offset={args.holdout_offset}")
    print(f"{'Body':<10} {'Before RMS km':>14} {'After RMS km':>13} {'After Max km':>12} {'Imp':>8} {'Class':>6}")
    print("-" * 72)
    for r in rows[:30]:
        print(f"{r['body']:<10} {float(r['before_rms_km']):14.3f} {float(r['after_rms_km']):13.3f} {float(r['after_max_km']):12.3f} {float(r['improvement_rms']):8.2f} {r['class_after']:>6}")
    print(f"\n[OK] summary CSV: {out_csv}")
    print(f"[OK] summary JSON: {out_json}")
    if args.write_samples:
        print(f"[OK] samples CSV: {args.output_dir / 'cross_validation_samples.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
