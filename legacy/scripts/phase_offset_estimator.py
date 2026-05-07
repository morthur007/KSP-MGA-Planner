#!/usr/bin/env python3
"""
phase_offset_estimator.py

Diagnóstico de erro de fase/mean-motion entre observações KSP/Principia e
propagação REBOUND, usando os CSVs já gerados pelo pipeline.

Objetivo:
- Converter o resíduo transversal RTN em um deslocamento temporal aparente:
      dt_phase ~= dot(delta_r, T_hat) / dot(v_rel_REB, T_hat)
- Ajustar dt_phase(t) por uma reta para distinguir:
      * offset temporal constante;
      * drift de frequência / mean motion;
      * erro oscilatório não secular.

Uso típico:
  python phase_offset_estimator.py \
    --ksp-csv data/final_clean_120d/states.csv \
    --reb-csv data/final_clean_120d/level_a_fit_urlum_family/rebound_states.csv \
    --parent Jool \
    --bodies Laythe Vall Tylo Bop Pol \
    --output-dir data/final_clean_120d/phase_offset_jool \
    --write-samples

Sem --parent, usa estado heliocêntrico/inercial como referência.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np


def norm(v: np.ndarray) -> float:
    return float(np.linalg.norm(v))


def unit(v: np.ndarray, fallback: Optional[np.ndarray] = None) -> np.ndarray:
    n = norm(v)
    if n > 0:
        return v / n
    if fallback is not None:
        return fallback.copy()
    return np.zeros(3)


def get_first(row: dict, names: Iterable[str]) -> Optional[str]:
    for n in names:
        if n in row and row[n] not in (None, ""):
            return row[n]
    return None


def parse_state_row(row: dict) -> Tuple[str, float, np.ndarray, np.ndarray]:
    body = get_first(row, ["body", "name", "body_name", "Body", "Corpo"])
    if body is None:
        raise KeyError(f"Não encontrei coluna de corpo em: {list(row.keys())}")

    et_s = get_first(row, ["et_seconds", "et_s", "ut_seconds", "ut_s", "time_s", "t"])
    if et_s is None:
        raise KeyError(f"Não encontrei coluna de tempo em: {list(row.keys())}")
    et = float(et_s)

    x = float(get_first(row, ["x_m", "x", "rx_m", "pos_x_m", "X"]))
    y = float(get_first(row, ["y_m", "y", "ry_m", "pos_y_m", "Y"]))
    z = float(get_first(row, ["z_m", "z", "rz_m", "pos_z_m", "Z"]))

    vx_s = get_first(row, ["vx_m_s", "vx", "vel_x_m_s", "VX"])
    vy_s = get_first(row, ["vy_m_s", "vy", "vel_y_m_s", "VY"])
    vz_s = get_first(row, ["vz_m_s", "vz", "vel_z_m_s", "VZ"])
    if vx_s is None or vy_s is None or vz_s is None:
        # Velocidade pode não existir em alguns CSVs; preenche com NaN.
        v = np.array([np.nan, np.nan, np.nan], dtype=float)
    else:
        v = np.array([float(vx_s), float(vy_s), float(vz_s)], dtype=float)

    r = np.array([x, y, z], dtype=float)
    return body, et, r, v


def load_csv(path: Path) -> Dict[str, Dict[float, Tuple[np.ndarray, np.ndarray]]]:
    data: Dict[str, Dict[float, Tuple[np.ndarray, np.ndarray]]] = defaultdict(dict)
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            body, et, r, v = parse_state_row(row)
            data[body][et] = (r, v)
    return data


def common_epochs(a: Dict[float, Tuple[np.ndarray, np.ndarray]], b: Dict[float, Tuple[np.ndarray, np.ndarray]]) -> List[float]:
    # Os CSVs do pipeline normalmente têm ETs idênticos. Se não tiverem,
    # faz correspondência por arredondamento a 1 ms.
    ka = set(a.keys())
    kb = set(b.keys())
    inter = sorted(ka & kb)
    if inter:
        return inter

    map_b = {round(k, 3): k for k in kb}
    out = []
    for k in sorted(ka):
        kk = round(k, 3)
        if kk in map_b:
            out.append(k)
    return out


def state_at(table: Dict[str, Dict[float, Tuple[np.ndarray, np.ndarray]]], body: str, et: float) -> Tuple[np.ndarray, np.ndarray]:
    if et in table[body]:
        return table[body][et]
    # fallback por arredondamento
    target = round(et, 3)
    for k, val in table[body].items():
        if round(k, 3) == target:
            return val
    raise KeyError((body, et))


def rtn_basis(r: np.ndarray, v: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    R = unit(r)
    h = np.cross(r, v)
    N = unit(h)
    T = unit(np.cross(N, R))
    v_t = float(np.dot(v, T))
    return R, T, N, v_t


def linear_fit(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
    """Retorna intercept, slope, r2 para y = intercept + slope*x."""
    if len(x) < 2:
        return float("nan"), float("nan"), float("nan")
    A = np.vstack([np.ones_like(x), x]).T
    coeff, *_ = np.linalg.lstsq(A, y, rcond=None)
    intercept, slope = float(coeff[0]), float(coeff[1])
    pred = intercept + slope * x
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - float(np.mean(y))) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return intercept, slope, r2


def zero_crossings(y: np.ndarray) -> int:
    if len(y) < 2:
        return 0
    s = np.sign(y)
    # ignora zeros exatos usando forward fill simples
    for i in range(1, len(s)):
        if s[i] == 0:
            s[i] = s[i-1]
    return int(np.sum(s[1:] * s[:-1] < 0))


def diagnose(row: dict) -> str:
    r2 = row["linear_r2"]
    final_over_max = row["final_over_max"]
    frac_t = row["frac_t_rms"]
    slope_s_per_day = row["slope_s_per_day"]
    zc = row["transverse_zero_crossings"]

    if frac_t >= 0.75 and final_over_max <= 0.05 and zc >= 1:
        return "FASE OSCILATÓRIA: erro transversal periódico; não corrigir com dV simples"
    if frac_t >= 0.75 and r2 >= 0.75:
        return "DRIFT DE FASE/MEAN-MOTION: testar μ/semieixo/frequência efetiva"
    if frac_t >= 0.75 and abs(slope_s_per_day) > 0.1:
        return "TRANSVERSAL COM TENDÊNCIA: provável diferença pequena de período"
    if row["frac_r_rms"] >= 0.55:
        return "RADIAL/MISTO: testar semieixo/excentricidade/estado relativo"
    if row["frac_n_rms"] >= 0.25:
        return "NORMAL: testar plano/nodo/precessão/eixo"
    return "MISTO/BAIXA EVIDÊNCIA: usar envelope de erro ou ajuste local"


def analyze_body(
    body: str,
    parent: Optional[str],
    ksp: Dict[str, Dict[float, Tuple[np.ndarray, np.ndarray]]],
    reb: Dict[str, Dict[float, Tuple[np.ndarray, np.ndarray]]],
) -> Tuple[dict, List[dict]]:
    epochs = common_epochs(ksp[body], reb[body])
    if parent:
        epochs = [e for e in epochs if e in ksp[parent] and e in reb[parent]]
    if not epochs:
        raise RuntimeError(f"Sem epochs comuns para {body}")

    t0 = epochs[0]
    samples: List[dict] = []
    totals = []
    rs = []
    ts = []
    ns = []
    dt_phases = []
    elapsed = []
    valid_phase = []

    for et in epochs:
        rk, vk = state_at(ksp, body, et)
        rr, vr = state_at(reb, body, et)
        if parent:
            rkp, vkp = state_at(ksp, parent, et)
            rrp, vrp = state_at(reb, parent, et)
            rk_rel = rk - rkp
            vk_rel = vk - vkp
            rr_rel = rr - rrp
            vr_rel = vr - vrp
        else:
            rk_rel = rk
            vk_rel = vk
            rr_rel = rr
            vr_rel = vr

        dr = rk_rel - rr_rel
        R, T, N, v_t = rtn_basis(rr_rel, vr_rel)
        comp_r = float(np.dot(dr, R))
        comp_t = float(np.dot(dr, T))
        comp_n = float(np.dot(dr, N))
        total = norm(dr)
        # dt aparente: se o resíduo transversal é equivalente a REB estar atrasado/adiantado no movimento.
        # Sinal: positivo significa residual na direção prograde do REB.
        dt_phase = comp_t / v_t if abs(v_t) > 1e-12 else float("nan")

        totals.append(total)
        rs.append(comp_r)
        ts.append(comp_t)
        ns.append(comp_n)
        elapsed.append(et - t0)
        if math.isfinite(dt_phase):
            dt_phases.append(dt_phase)
            valid_phase.append(et - t0)

        samples.append({
            "body": body,
            "parent": parent or "",
            "et_seconds": et,
            "elapsed_days": (et - t0) / 86400.0,
            "total_err_km": total / 1000.0,
            "r_err_km": comp_r / 1000.0,
            "t_err_km": comp_t / 1000.0,
            "n_err_km": comp_n / 1000.0,
            "v_t_m_s": v_t,
            "phase_dt_s": dt_phase,
        })

    totals_a = np.array(totals)
    r_a = np.array(rs)
    t_a = np.array(ts)
    n_a = np.array(ns)
    elapsed_a = np.array(elapsed)
    dt_a = np.array(dt_phases)
    valid_elapsed_a = np.array(valid_phase)

    rms_total = float(np.sqrt(np.mean(totals_a ** 2)))
    rms_r = float(np.sqrt(np.mean(r_a ** 2)))
    rms_t = float(np.sqrt(np.mean(t_a ** 2)))
    rms_n = float(np.sqrt(np.mean(n_a ** 2)))
    denom = rms_r + rms_t + rms_n if (rms_r + rms_t + rms_n) > 0 else 1.0

    intercept, slope, r2 = linear_fit(valid_elapsed_a, dt_a) if len(dt_a) >= 2 else (float("nan"), float("nan"), float("nan"))
    # slope: s/s. Multiplica por dia para leitura.
    slope_s_per_day = slope * 86400.0 if math.isfinite(slope) else float("nan")
    # aproximação: δn/n ≈ - d(dt_phase)/dt.
    frac_n_error_est = -slope if math.isfinite(slope) else float("nan")

    imax = int(np.argmax(totals_a))
    max_total = float(totals_a[imax])
    final_total = float(totals_a[-1])

    summary = {
        "body": body,
        "parent": parent or "",
        "n": len(epochs),
        "max_total_km": max_total / 1000.0,
        "rms_total_km": rms_total / 1000.0,
        "final_total_km": final_total / 1000.0,
        "final_over_max": final_total / max_total if max_total > 0 else 0.0,
        "max_epoch_et_s": epochs[imax],
        "max_epoch_day_from_start": (epochs[imax] - t0) / 86400.0,
        "rms_r_km": rms_r / 1000.0,
        "rms_t_km": rms_t / 1000.0,
        "rms_n_km": rms_n / 1000.0,
        "frac_r_rms": rms_r / denom,
        "frac_t_rms": rms_t / denom,
        "frac_n_rms": rms_n / denom,
        "phase_dt_median_s": float(np.nanmedian(dt_a)) if len(dt_a) else float("nan"),
        "phase_dt_rms_s": float(np.sqrt(np.nanmean(dt_a ** 2))) if len(dt_a) else float("nan"),
        "phase_dt_max_abs_s": float(np.nanmax(np.abs(dt_a))) if len(dt_a) else float("nan"),
        "phase_dt_initial_s": float(dt_a[0]) if len(dt_a) else float("nan"),
        "phase_dt_final_s": float(dt_a[-1]) if len(dt_a) else float("nan"),
        "linear_intercept_s": intercept,
        "linear_slope_s_per_s": slope,
        "slope_s_per_day": slope_s_per_day,
        "linear_r2": r2,
        "frac_mean_motion_error_est": frac_n_error_est,
        "transverse_zero_crossings": zero_crossings(t_a),
    }
    summary["diagnosis"] = diagnose(summary)
    return summary, samples


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ksp-csv", required=True)
    p.add_argument("--reb-csv", required=True)
    p.add_argument("--bodies", nargs="+", required=True)
    p.add_argument("--parent", default=None)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--write-samples", action="store_true")
    args = p.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("Carregando CSVs...")
    ksp = load_csv(Path(args.ksp_csv))
    reb = load_csv(Path(args.reb_csv))

    summaries = []
    all_samples = []
    for body in args.bodies:
        print(f"Analisando {body}...")
        s, smp = analyze_body(body, args.parent, ksp, reb)
        summaries.append(s)
        all_samples.extend(smp)

    summaries.sort(key=lambda r: r["max_total_km"], reverse=True)

    csv_path = out / "phase_offset_summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = list(summaries[0].keys()) if summaries else []
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(summaries)

    json_path = out / "phase_offset_summary.json"
    json_path.write_text(json.dumps(summaries, indent=2, ensure_ascii=False), encoding="utf-8")

    if args.write_samples:
        smp_path = out / "phase_offset_samples.csv"
        with smp_path.open("w", encoding="utf-8", newline="") as f:
            fieldnames = list(all_samples[0].keys()) if all_samples else []
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(all_samples)

    print("\n=== PHASE OFFSET / MEAN-MOTION DIAGNOSTICS ===")
    print(f"{'Body':<10} {'Parent':<8} {'Max km':>9} {'RMS km':>9} {'Final km':>9} {'dtRMS s':>10} {'s/day':>10} {'R2':>6} {'zc':>3}  Diagnóstico")
    print("-" * 150)
    for r in summaries:
        print(
            f"{r['body']:<10} {r['parent'] or '-':<8} "
            f"{r['max_total_km']:9.3f} {r['rms_total_km']:9.3f} {r['final_total_km']:9.3f} "
            f"{r['phase_dt_rms_s']:10.3f} {r['slope_s_per_day']:10.6f} {r['linear_r2']:6.3f} "
            f"{r['transverse_zero_crossings']:3d}  {r['diagnosis']}"
        )

    print(f"\n[OK] summary CSV: {csv_path}")
    print(f"[OK] summary JSON: {json_path}")
    if args.write_samples:
        print(f"[OK] samples CSV: {out / 'phase_offset_samples.csv'}")


if __name__ == "__main__":
    main()
