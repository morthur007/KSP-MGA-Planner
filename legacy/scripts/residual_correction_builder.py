#!/usr/bin/env python3
"""
residual_correction_builder.py

Builds an observation-derived residual correction layer between a REBOUND base
trajectory and KSP/Principia observations.

Purpose
-------
This is not a dynamics model replacement. It treats the REBOUND trajectory as a
physical backbone and builds a time-bounded residual correction table from
observations:

    r_corrected(t) = r_REBOUND(t) + smooth( r_KSP(t) - r_REBOUND(t) )
    v_corrected(t) = v_REBOUND(t) + smooth( v_KSP(t) - v_REBOUND(t) )

For diagnostics it also decomposes position residuals into an RTN/RSW frame:
radial, transverse/prograde, normal.

The correction is valid only inside the observation interval. Do not extrapolate
it to long horizons without additional observations.

Outputs
-------
- corrected_states.csv
- residual_components.csv
- before_after_by_body.csv
- correction_manifest.json
- correction_model.json

Input CSV expectations
----------------------
The script is intentionally tolerant about column names.
Required logical fields:
    body, et_seconds, x, y, z
Optional:
    vx, vy, vz
Supported names include:
    body/name/body_name
    et_seconds/et/time/t
    x_m/x/px/pos_x_m, etc.
    vx_m_s/vx/vel_x_m_s, etc.

Example
-------
python residual_correction_builder.py \
  --ksp-csv data/final_clean_120d/states.csv \
  --reb-csv data/final_clean_120d/level_a_fit_urlum_family/rebound_states.csv \
  --family Jool:Laythe,Vall,Tylo,Bop,Pol \
  --family Sarnus:Hale,Ovok,Slate,Tekto,Eeloo \
  --family Urlum:Polta,Priax,Wal,Tal \
  --smooth savgol --window 9 --polyorder 3 \
  --output-dir data/final_clean_120d/residual_correction_v1
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
    from scipy.signal import savgol_filter  # type: ignore
except Exception:  # pragma: no cover
    savgol_filter = None


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
    for c in candidates:
        if c in fset:
            return c
    lower_map = {f.lower(): f for f in fieldnames}
    for c in candidates:
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
            raise ValueError(
                f"Não consegui detectar colunas body/time/position em {path}. "
                f"Header: {reader.fieldnames}"
            )

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
                # Skip malformed/blank lines rather than killing a long run.
                continue

    out: Dict[str, StateSeries] = {}
    for body, rows in tmp.items():
        rows.sort(key=lambda x: x[0])
        t = np.array([x[0] for x in rows], dtype=float)
        r = np.vstack([x[1] for x in rows])
        if all(x[2] is not None for x in rows):
            v = np.vstack([x[2] for x in rows])  # type: ignore
        else:
            v = None
        out[body] = StateSeries(body, t, r, v)
    return out


def make_time_index(series: StateSeries, round_digits: int) -> Dict[float, int]:
    return {round(float(t), round_digits): i for i, t in enumerate(series.t)}


def common_aligned(
    ksp: StateSeries,
    reb: StateSeries,
    round_digits: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray], np.ndarray, np.ndarray]:
    ki = make_time_index(ksp, round_digits)
    ri = make_time_index(reb, round_digits)
    keys = sorted(set(ki).intersection(ri))
    if not keys:
        raise ValueError(f"Sem epochs comuns para {ksp.body}")
    ik = np.array([ki[x] for x in keys], dtype=int)
    ir = np.array([ri[x] for x in keys], dtype=int)
    t = np.array([ksp.t[i] for i in ik], dtype=float)
    return t, ksp.r[ik], reb.r[ir], (ksp.v[ik] if ksp.v is not None else None), (reb.v[ir] if reb.v is not None else None), ik, ir


def finite_difference_velocity(t: np.ndarray, r: np.ndarray) -> np.ndarray:
    v = np.zeros_like(r)
    for j in range(3):
        v[:, j] = np.gradient(r[:, j], t, edge_order=2 if len(t) >= 3 else 1)
    return v


def unit(v: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if not np.isfinite(n) or n < 1e-30:
        return fallback.copy()
    return v / n


def rtn_basis(r: np.ndarray, v: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # R radial, N angular momentum normal, T completes right-handed frame.
    eR = unit(r, np.array([1.0, 0.0, 0.0]))
    h = np.cross(r, v)
    eN = unit(h, np.array([0.0, 0.0, 1.0]))
    eT = unit(np.cross(eN, eR), np.array([0.0, 1.0, 0.0]))
    return eR, eT, eN


def smooth_array(arr: np.ndarray, mode: str, window: int, polyorder: int) -> np.ndarray:
    if mode == "none" or arr.shape[0] < 5:
        return arr.copy()
    if mode != "savgol":
        raise ValueError(f"smooth mode desconhecido: {mode}")
    if savgol_filter is None:
        print("[WARN] scipy.signal.savgol_filter indisponível; usando sem smoothing.")
        return arr.copy()
    n = arr.shape[0]
    win = min(window, n if n % 2 == 1 else n - 1)
    if win < 5:
        return arr.copy()
    if win % 2 == 0:
        win -= 1
    po = min(polyorder, win - 2)
    out = np.zeros_like(arr)
    for j in range(arr.shape[1]):
        out[:, j] = savgol_filter(arr[:, j], window_length=win, polyorder=po, mode="interp")
    return out


def parse_parent_maps(families: List[str], pairs: List[str], json_path: Optional[Path]) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    if json_path:
        data = json.loads(json_path.read_text(encoding="utf-8"))
        # Accept child->parent or parent->[children].
        for k, v in data.items():
            if isinstance(v, str):
                mapping[k] = v
            elif isinstance(v, list):
                for child in v:
                    mapping[str(child)] = str(k)
    for fam in families:
        if ":" not in fam:
            raise ValueError(f"Formato --family inválido: {fam}; use Parent:child1,child2")
        parent, children_s = fam.split(":", 1)
        parent = parent.strip()
        for child in children_s.split(","):
            child = child.strip()
            if child:
                mapping[child] = parent
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"Formato --parent-pair inválido: {pair}; use child=parent")
        child, parent = pair.split("=", 1)
        mapping[child.strip()] = parent.strip()
    return mapping


def metrics(pos_err_m: np.ndarray, vel_err: Optional[np.ndarray]) -> Dict[str, float]:
    n = np.linalg.norm(pos_err_m, axis=1)
    d: Dict[str, float] = {
        "n": int(len(n)),
        "max_pos_km": float(np.max(n) / 1000.0),
        "rms_pos_km": float(np.sqrt(np.mean(n * n)) / 1000.0),
        "final_pos_km": float(n[-1] / 1000.0),
    }
    if vel_err is not None:
        vn = np.linalg.norm(vel_err, axis=1)
        d["max_vel_m_s"] = float(np.max(vn))
        d["rms_vel_m_s"] = float(np.sqrt(np.mean(vn * vn)))
    else:
        d["max_vel_m_s"] = float("nan")
        d["rms_vel_m_s"] = float("nan")
    return d


def classification(max_km: float, rms_km: float) -> str:
    if max_km <= 1.0 or rms_km <= 0.1:
        return "A"
    if max_km <= 10.0 or rms_km <= 1.0:
        return "A-"
    if max_km <= 100.0 or rms_km <= 10.0:
        return "B"
    if max_km <= 1000.0 or rms_km <= 100.0:
        return "C"
    return "D"


def write_csv(path: Path, fieldnames: List[str], rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def main() -> int:
    p = argparse.ArgumentParser(description="Build observation residual correction layer for REBOUND ephemerides.")
    p.add_argument("--ksp-csv", required=True, type=Path)
    p.add_argument("--reb-csv", required=True, type=Path)
    p.add_argument("--bodies", nargs="*", default=None, help="Bodies to correct. Default: all common bodies except Sun if present.")
    p.add_argument("--family", action="append", default=[], help="Parent:child1,child2 mapping. Can repeat.")
    p.add_argument("--parent-pair", action="append", default=[], help="child=parent mapping. Can repeat.")
    p.add_argument("--parent-map-json", type=Path, default=None)
    p.add_argument("--smooth", choices=["none", "savgol"], default="savgol")
    p.add_argument("--window", type=int, default=9, help="Savgol odd window length in samples.")
    p.add_argument("--polyorder", type=int, default=3)
    p.add_argument("--time-round-digits", type=int, default=6)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--include-sun", action="store_true")
    args = p.parse_args()

    print("Carregando estados...")
    ksp_all = read_states(args.ksp_csv)
    reb_all = read_states(args.reb_csv)
    parent_map = parse_parent_maps(args.family, args.parent_pair, args.parent_map_json)

    common_bodies = sorted(set(ksp_all).intersection(reb_all))
    if args.bodies:
        bodies = [b for b in args.bodies if b in ksp_all and b in reb_all]
    else:
        bodies = [b for b in common_bodies if args.include_sun or b.lower() != "sun"]
    if not bodies:
        raise SystemExit("Nenhum corpo comum para corrigir.")

    outdir: Path = args.output_dir
    outdir.mkdir(parents=True, exist_ok=True)

    corrected_rows: List[Dict[str, object]] = []
    comp_rows: List[Dict[str, object]] = []
    summary_rows: List[Dict[str, object]] = []
    model: Dict[str, object] = {"bodies": {}, "parent_map": parent_map, "validity": {}}

    global_t_min = float("inf")
    global_t_max = -float("inf")

    for body in bodies:
        print(f"Corrigindo {body}...")
        ksp = ksp_all[body]
        reb = reb_all[body]
        t, r_k, r_r, v_k, v_r, _, _ = common_aligned(ksp, reb, args.time_round_digits)
        if v_k is None:
            v_k = finite_difference_velocity(t, r_k)
        if v_r is None:
            v_r = finite_difference_velocity(t, r_r)
        dr = r_k - r_r
        dv = v_k - v_r

        parent = parent_map.get(body)
        if parent and parent in ksp_all and parent in reb_all:
            # Parent basis from REBOUND relative state, aligned to same times.
            pt, pk_r, pr_r, pk_v, pr_v, _, _ = common_aligned(ksp_all[parent], reb_all[parent], args.time_round_digits)
            # Align parent arrays to body t via exact rounded index.
            pidx = {round(float(x), args.time_round_digits): i for i, x in enumerate(pt)}
            parent_indices = [pidx[round(float(x), args.time_round_digits)] for x in t]
            pr_r_al = pr_r[parent_indices]
            if pr_v is None:
                pr_v_all = finite_difference_velocity(pt, pr_r)
                pr_v_al = pr_v_all[parent_indices]
            else:
                pr_v_al = pr_v[parent_indices]
            rel_r = r_r - pr_r_al
            rel_v = v_r - pr_v_al
        else:
            parent = ""
            rel_r = r_r
            rel_v = v_r

        comps = np.zeros_like(dr)
        bases = np.zeros((len(t), 3, 3), dtype=float)  # R,T,N as rows
        for i in range(len(t)):
            eR, eT, eN = rtn_basis(rel_r[i], rel_v[i])
            bases[i, 0, :] = eR
            bases[i, 1, :] = eT
            bases[i, 2, :] = eN
            comps[i, 0] = float(np.dot(dr[i], eR))
            comps[i, 1] = float(np.dot(dr[i], eT))
            comps[i, 2] = float(np.dot(dr[i], eN))

        comps_sm = smooth_array(comps, args.smooth, args.window, args.polyorder)
        dv_sm = smooth_array(dv, args.smooth, args.window, args.polyorder)

        corr_r = np.zeros_like(dr)
        for i in range(len(t)):
            corr_r[i] = comps_sm[i, 0] * bases[i, 0] + comps_sm[i, 1] * bases[i, 1] + comps_sm[i, 2] * bases[i, 2]
        r_c = r_r + corr_r
        v_c = v_r + dv_sm

        before = metrics(r_k - r_r, v_k - v_r)
        after = metrics(r_k - r_c, v_k - v_c)

        improvement_max = before["max_pos_km"] / after["max_pos_km"] if after["max_pos_km"] > 0 else float("inf")
        improvement_rms = before["rms_pos_km"] / after["rms_pos_km"] if after["rms_pos_km"] > 0 else float("inf")
        summary = {
            "body": body,
            "parent": parent,
            "n": int(len(t)),
            "t_start_et": float(t[0]),
            "t_end_et": float(t[-1]),
            "before_max_km": before["max_pos_km"],
            "before_rms_km": before["rms_pos_km"],
            "before_final_km": before["final_pos_km"],
            "after_max_km": after["max_pos_km"],
            "after_rms_km": after["rms_pos_km"],
            "after_final_km": after["final_pos_km"],
            "before_max_vel_m_s": before["max_vel_m_s"],
            "after_max_vel_m_s": after["max_vel_m_s"],
            "improvement_max": improvement_max,
            "improvement_rms": improvement_rms,
            "class_after": classification(after["max_pos_km"], after["rms_pos_km"]),
            "smooth": args.smooth,
        }
        summary_rows.append(summary)

        global_t_min = min(global_t_min, float(t[0]))
        global_t_max = max(global_t_max, float(t[-1]))

        model["bodies"][body] = {
            "parent": parent,
            "t_start_et": float(t[0]),
            "t_end_et": float(t[-1]),
            "smooth": args.smooth,
            "window": args.window,
            "polyorder": args.polyorder,
            "summary": summary,
            # Keep compact model: table file is authoritative; JSON stores summary only.
        }

        for i in range(len(t)):
            corrected_rows.append({
                "body": body,
                "et_seconds": f"{t[i]:.12f}",
                "x_m": f"{r_c[i,0]:.12f}",
                "y_m": f"{r_c[i,1]:.12f}",
                "z_m": f"{r_c[i,2]:.12f}",
                "vx_m_s": f"{v_c[i,0]:.15f}",
                "vy_m_s": f"{v_c[i,1]:.15f}",
                "vz_m_s": f"{v_c[i,2]:.15f}",
                "source": "rebound_plus_observed_residual",
            })
            comp_rows.append({
                "body": body,
                "parent": parent,
                "et_seconds": f"{t[i]:.12f}",
                "raw_R_m": f"{comps[i,0]:.12f}",
                "raw_T_m": f"{comps[i,1]:.12f}",
                "raw_N_m": f"{comps[i,2]:.12f}",
                "smooth_R_m": f"{comps_sm[i,0]:.12f}",
                "smooth_T_m": f"{comps_sm[i,1]:.12f}",
                "smooth_N_m": f"{comps_sm[i,2]:.12f}",
                "raw_norm_m": f"{np.linalg.norm(dr[i]):.12f}",
                "after_norm_m": f"{np.linalg.norm(r_k[i]-r_c[i]):.12f}",
            })

    corrected_rows.sort(key=lambda r: (float(r["et_seconds"]), str(r["body"])))
    comp_rows.sort(key=lambda r: (str(r["body"]), float(r["et_seconds"])))
    summary_rows.sort(key=lambda r: float(r["after_rms_km"]), reverse=True)

    write_csv(
        outdir / "corrected_states.csv",
        ["body", "et_seconds", "x_m", "y_m", "z_m", "vx_m_s", "vy_m_s", "vz_m_s", "source"],
        corrected_rows,
    )
    write_csv(
        outdir / "residual_components.csv",
        ["body", "parent", "et_seconds", "raw_R_m", "raw_T_m", "raw_N_m", "smooth_R_m", "smooth_T_m", "smooth_N_m", "raw_norm_m", "after_norm_m"],
        comp_rows,
    )
    write_csv(
        outdir / "before_after_by_body.csv",
        [
            "body", "parent", "n", "t_start_et", "t_end_et",
            "before_max_km", "before_rms_km", "before_final_km",
            "after_max_km", "after_rms_km", "after_final_km",
            "before_max_vel_m_s", "after_max_vel_m_s",
            "improvement_max", "improvement_rms", "class_after", "smooth",
        ],
        summary_rows,
    )

    model["validity"] = {"t_start_et": global_t_min, "t_end_et": global_t_max, "no_extrapolation": True}
    manifest = {
        "purpose": "Observation-derived residual correction layer for REBOUND ephemeris",
        "ksp_csv": str(args.ksp_csv),
        "reb_csv": str(args.reb_csv),
        "smooth": args.smooth,
        "window": args.window,
        "polyorder": args.polyorder,
        "time_round_digits": args.time_round_digits,
        "bodies": bodies,
        "parent_map": parent_map,
        "validity": model["validity"],
        "warning": "Correction is certified only inside the observation interval; do not extrapolate.",
    }
    (outdir / "correction_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (outdir / "correction_model.json").write_text(json.dumps(model, indent=2), encoding="utf-8")

    print("\n=== RESIDUAL CORRECTION SUMMARY ===")
    print(f"{'Body':<12} {'Parent':<10} {'Before max km':>14} {'After max km':>13} {'Before RMS':>12} {'After RMS':>11} {'Imp RMS':>9} {'Class':>6}")
    print("-" * 96)
    for row in summary_rows[:80]:
        print(
            f"{row['body']:<12} {row['parent']:<10} "
            f"{row['before_max_km']:14.3f} {row['after_max_km']:13.6f} "
            f"{row['before_rms_km']:12.3f} {row['after_rms_km']:11.6f} "
            f"{row['improvement_rms']:9.2f} {row['class_after']:>6}"
        )
    print(f"\n[OK] corrected states: {outdir / 'corrected_states.csv'}")
    print(f"[OK] residual components: {outdir / 'residual_components.csv'}")
    print(f"[OK] before/after: {outdir / 'before_after_by_body.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
