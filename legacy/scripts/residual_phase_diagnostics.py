#!/usr/bin/env python3
"""
residual_phase_diagnostics.py

Diagnóstico profissional de resíduos KSP/Principia vs REBOUND/SPK-cache.

Objetivo:
- Separar erro de fase/período de erro radial e erro de plano.
- Trabalhar diretamente com states.csv do KSP e rebound_states.csv do Nível A.
- Opcionalmente medir resíduos relativos ao corpo-pai, ex. Vall relativo a Jool.

Métrica RTN/RSW:
  R = direção radial do corpo em relação ao pai/centro
  T = direção transversal/prograde no plano orbital
  N = normal ao plano orbital

Interpretação rápida:
  |T| dominante  -> erro de fase / período / longitude média
  |R| dominante  -> semi-eixo, excentricidade, distância orbital
  |N| dominante  -> inclinação, nodo, plano orbital, frame local
  tendência final alta -> drift secular/modelo físico
  máximo alto e final baixo -> erro oscilatório/fase, não divergência monotônica

Exemplos:
  python residual_phase_diagnostics.py \
    --ksp-csv data/final_clean_120d/states.csv \
    --reb-csv data/final_clean_120d/level_a_fit_urlum_family/rebound_states.csv \
    --bodies Vall Tylo Pol Vant Zore \
    --parent Jool \
    --output-dir data/final_clean_120d/phase_diag_jool

  python residual_phase_diagnostics.py \
    --ksp-csv data/final_clean_120d/states.csv \
    --reb-csv data/final_clean_120d/level_a_fit_urlum_family/rebound_states.csv \
    --bodies Hale Eeloo Ovok Slate Tekto \
    --parent Sarnus \
    --output-dir data/final_clean_120d/phase_diag_sarnus

  python residual_phase_diagnostics.py \
    --ksp-csv data/final_clean_120d/states.csv \
    --reb-csv data/final_clean_120d/level_a_fit_urlum_family/rebound_states.csv \
    --bodies Polta Priax Wal Tal \
    --parent Urlum \
    --output-dir data/final_clean_120d/phase_diag_urlum
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


Vec = np.ndarray


@dataclass
class State:
    et: float
    r: Vec
    v: Vec


@dataclass
class BodyDiagnostics:
    body: str
    parent: Optional[str]
    n: int
    duration_days: float
    max_total_km: float
    rms_total_km: float
    final_total_km: float
    max_r_km: float
    max_t_km: float
    max_n_km: float
    rms_r_km: float
    rms_t_km: float
    rms_n_km: float
    final_r_km: float
    final_t_km: float
    final_n_km: float
    frac_r_rms: float
    frac_t_rms: float
    frac_n_rms: float
    max_epoch_et: float
    max_epoch_day_from_start: float
    final_over_max: float
    signed_t_zero_crossings: int
    phase_like_score: float
    drift_like_score: float
    dominant_component: str
    diagnosis: str


def pick_column(fieldnames: Sequence[str], candidates: Sequence[str], required: bool = True) -> Optional[str]:
    lower = {c.lower(): c for c in fieldnames}
    normalized = {c.lower().replace(" ", "").replace("-", "_"): c for c in fieldnames}
    for cand in candidates:
        key = cand.lower()
        if key in lower:
            return lower[key]
        norm = key.replace(" ", "").replace("-", "_")
        if norm in normalized:
            return normalized[norm]
    if required:
        raise KeyError(f"Não encontrei coluna entre candidatos: {candidates}; colunas disponíveis: {fieldnames}")
    return None


def detect_columns(path: Path) -> Dict[str, str]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"CSV sem cabeçalho: {path}")
        cols = reader.fieldnames

    body = pick_column(cols, ["body", "body_name", "name", "celestial_body"])
    et = pick_column(cols, ["et_seconds", "et", "time_et", "t_et", "time_seconds", "ut", "ut_seconds"])
    x = pick_column(cols, ["x_m", "x", "pos_x_m", "position_x_m", "rx", "r_x"])
    y = pick_column(cols, ["y_m", "y", "pos_y_m", "position_y_m", "ry", "r_y"])
    z = pick_column(cols, ["z_m", "z", "pos_z_m", "position_z_m", "rz", "r_z"])
    vx = pick_column(cols, ["vx_m_s", "vx", "vel_x_m_s", "velocity_x_m_s", "v_x"])
    vy = pick_column(cols, ["vy_m_s", "vy", "vel_y_m_s", "velocity_y_m_s", "v_y"])
    vz = pick_column(cols, ["vz_m_s", "vz", "vel_z_m_s", "velocity_z_m_s", "v_z"])
    return {"body": body, "et": et, "x": x, "y": y, "z": z, "vx": vx, "vy": vy, "vz": vz}


def load_states(path: Path, bodies: Optional[Iterable[str]] = None, parent: Optional[str] = None) -> Dict[str, Dict[float, State]]:
    wanted = set(bodies or [])
    if parent:
        wanted.add(parent)
    cols = detect_columns(path)
    out: Dict[str, Dict[float, State]] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            body = str(row[cols["body"]]).strip()
            if wanted and body not in wanted:
                continue
            try:
                et = float(row[cols["et"]])
                r = np.array([float(row[cols["x"]]), float(row[cols["y"]]), float(row[cols["z"]])], dtype=float)
                v = np.array([float(row[cols["vx"]]), float(row[cols["vy"]]), float(row[cols["vz"]])], dtype=float)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Linha inválida em {path}, body={body}: {row}") from exc
            out.setdefault(body, {})[et] = State(et=et, r=r, v=v)
    return out


def nearest_key(keys: Sequence[float], target: float, tolerance: float) -> Optional[float]:
    # keys sorted
    import bisect
    i = bisect.bisect_left(keys, target)
    best = None
    best_abs = float("inf")
    for j in (i - 1, i, i + 1):
        if 0 <= j < len(keys):
            d = abs(keys[j] - target)
            if d < best_abs:
                best_abs = d
                best = keys[j]
    if best is not None and best_abs <= tolerance:
        return best
    return None


def common_epochs(a: Dict[float, State], b: Dict[float, State], tolerance: float) -> List[Tuple[float, float, float]]:
    if tolerance <= 0:
        common = sorted(set(a.keys()) & set(b.keys()))
        return [(t, t, t) for t in common]
    bkeys = sorted(b.keys())
    pairs = []
    used = set()
    for ta in sorted(a.keys()):
        tb = nearest_key(bkeys, ta, tolerance)
        if tb is not None and tb not in used:
            used.add(tb)
            pairs.append((ta, ta, tb))
    return pairs


def safe_unit(x: Vec, fallback: Optional[Vec] = None) -> Vec:
    n = float(np.linalg.norm(x))
    if n > 0:
        return x / n
    if fallback is not None:
        return fallback
    return np.array([1.0, 0.0, 0.0], dtype=float)


def rtn_basis(r_ref: Vec, v_ref: Vec) -> Tuple[Vec, Vec, Vec]:
    r_hat = safe_unit(r_ref)
    h = np.cross(r_ref, v_ref)
    n_hat = safe_unit(h, fallback=np.array([0.0, 0.0, 1.0], dtype=float))
    t_hat = safe_unit(np.cross(n_hat, r_hat), fallback=np.array([0.0, 1.0, 0.0], dtype=float))
    return r_hat, t_hat, n_hat


def sign_crossings(values: Sequence[float]) -> int:
    prev = 0
    count = 0
    for v in values:
        if abs(v) < 1e-12:
            continue
        s = 1 if v > 0 else -1
        if prev != 0 and s != prev:
            count += 1
        prev = s
    return count


def diagnose(frac_r: float, frac_t: float, frac_n: float, final_over_max: float, zc_t: int) -> Tuple[str, float, float, str]:
    comps = {"radial": frac_r, "transversal": frac_t, "normal": frac_n}
    dominant = max(comps, key=comps.get)

    # Score heurístico, deliberadamente simples e auditável.
    phase_score = 0.0
    if dominant == "transversal":
        phase_score += 0.45
    phase_score += min(0.35, frac_t * 0.35)
    if final_over_max < 0.2:
        phase_score += 0.15
    if zc_t >= 1:
        phase_score += 0.05

    drift_score = 0.0
    if final_over_max > 0.6:
        drift_score += 0.45
    if dominant in ("radial", "normal"):
        drift_score += 0.20
    if zc_t == 0:
        drift_score += 0.15
    drift_score += min(0.20, max(frac_r, frac_n) * 0.20)

    if phase_score >= 0.65 and final_over_max < 0.35:
        diagnosis = "FASE/PERÍODO: resíduo predominantemente transversal e não secular"
    elif dominant == "radial" and final_over_max > 0.4:
        diagnosis = "RADIAL/ENERGIA: suspeitar semi-eixo, excentricidade ou μ efetivo"
    elif dominant == "normal":
        diagnosis = "PLANO/NODO: suspeitar inclinação, nodo, eixo/frame local ou precessão"
    elif final_over_max > 0.7:
        diagnosis = "DRIFT SECULAR: modelo/família/estado ainda não fecha"
    elif dominant == "transversal":
        diagnosis = "TRANSVERSAL: provável erro de fase, mas validar periodicidade"
    else:
        diagnosis = "MISTO: decomposição não aponta uma causa única"

    return dominant, phase_score, drift_score, diagnosis


def analyze_body(
    body: str,
    parent: Optional[str],
    ksp: Dict[str, Dict[float, State]],
    reb: Dict[str, Dict[float, State]],
    tolerance: float,
    samples_rows: List[Dict[str, object]],
) -> BodyDiagnostics:
    if body not in ksp:
        raise KeyError(f"{body} ausente no KSP CSV")
    if body not in reb:
        raise KeyError(f"{body} ausente no REBOUND CSV")
    if parent and (parent not in ksp or parent not in reb):
        raise KeyError(f"parent {parent} ausente em KSP ou REBOUND CSV")

    pairs = common_epochs(ksp[body], reb[body], tolerance=tolerance)
    if not pairs:
        raise ValueError(f"Sem epochs comuns para {body}")

    # Se há parent, precisamos garantir parent também presente em cada par. Para tolerância >0, parent usa nearest.
    parent_k_keys = sorted(ksp[parent].keys()) if parent else []
    parent_r_keys = sorted(reb[parent].keys()) if parent else []

    times = []
    total = []
    comp_r = []
    comp_t = []
    comp_n = []

    for _, tk, tr in pairs:
        ks = ksp[body][tk]
        rs = reb[body][tr]

        if parent:
            pk_t = tk if tk in ksp[parent] else nearest_key(parent_k_keys, tk, tolerance)
            pr_t = tr if tr in reb[parent] else nearest_key(parent_r_keys, tr, tolerance)
            if pk_t is None or pr_t is None:
                continue
            kp = ksp[parent][pk_t]
            rp = reb[parent][pr_t]
            k_rel_r = ks.r - kp.r
            k_rel_v = ks.v - kp.v
            r_rel_r = rs.r - rp.r
            delta = k_rel_r - r_rel_r
            ref_r = k_rel_r
            ref_v = k_rel_v
        else:
            delta = ks.r - rs.r
            ref_r = ks.r
            ref_v = ks.v

        rhat, that, nhat = rtn_basis(ref_r, ref_v)
        dr = float(np.dot(delta, rhat))
        dt = float(np.dot(delta, that))
        dn = float(np.dot(delta, nhat))
        dtotal = float(np.linalg.norm(delta))

        times.append(tk)
        total.append(dtotal)
        comp_r.append(dr)
        comp_t.append(dt)
        comp_n.append(dn)

        samples_rows.append({
            "body": body,
            "parent": parent or "",
            "et_seconds": tk,
            "day_from_start": 0.0,  # filled later
            "err_total_m": dtotal,
            "err_r_m": dr,
            "err_t_m": dt,
            "err_n_m": dn,
            "abs_r_m": abs(dr),
            "abs_t_m": abs(dt),
            "abs_n_m": abs(dn),
        })

    if not times:
        raise ValueError(f"Sem amostras utilizáveis para {body}")

    t0 = min(times)
    # fill day_from_start for rows of this body just appended
    for row in samples_rows[-len(times):]:
        row["day_from_start"] = (float(row["et_seconds"]) - t0) / 86400.0

    total_a = np.array(total)
    r_a = np.array(comp_r)
    t_a = np.array(comp_t)
    n_a = np.array(comp_n)

    rms_total = float(np.sqrt(np.mean(total_a ** 2)))
    rms_r = float(np.sqrt(np.mean(r_a ** 2)))
    rms_t = float(np.sqrt(np.mean(t_a ** 2)))
    rms_n = float(np.sqrt(np.mean(n_a ** 2)))
    denom = rms_r + rms_t + rms_n
    if denom == 0:
        frac_r = frac_t = frac_n = 0.0
    else:
        frac_r, frac_t, frac_n = rms_r / denom, rms_t / denom, rms_n / denom

    imax = int(np.argmax(total_a))
    max_total = float(total_a[imax])
    final_total = float(total_a[-1])
    final_over_max = final_total / max_total if max_total > 0 else 0.0
    zc_t = sign_crossings(t_a.tolist())
    dominant, phase_score, drift_score, diagnosis = diagnose(frac_r, frac_t, frac_n, final_over_max, zc_t)

    return BodyDiagnostics(
        body=body,
        parent=parent,
        n=len(total_a),
        duration_days=(max(times) - min(times)) / 86400.0,
        max_total_km=max_total / 1000.0,
        rms_total_km=rms_total / 1000.0,
        final_total_km=final_total / 1000.0,
        max_r_km=float(np.max(np.abs(r_a))) / 1000.0,
        max_t_km=float(np.max(np.abs(t_a))) / 1000.0,
        max_n_km=float(np.max(np.abs(n_a))) / 1000.0,
        rms_r_km=rms_r / 1000.0,
        rms_t_km=rms_t / 1000.0,
        rms_n_km=rms_n / 1000.0,
        final_r_km=float(r_a[-1]) / 1000.0,
        final_t_km=float(t_a[-1]) / 1000.0,
        final_n_km=float(n_a[-1]) / 1000.0,
        frac_r_rms=frac_r,
        frac_t_rms=frac_t,
        frac_n_rms=frac_n,
        max_epoch_et=float(times[imax]),
        max_epoch_day_from_start=(float(times[imax]) - min(times)) / 86400.0,
        final_over_max=final_over_max,
        signed_t_zero_crossings=zc_t,
        phase_like_score=phase_score,
        drift_like_score=drift_score,
        dominant_component=dominant,
        diagnosis=diagnosis,
    )


def write_summary(path: Path, rows: List[BodyDiagnostics]) -> None:
    fields = list(BodyDiagnostics.__dataclass_fields__.keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            d = r.__dict__.copy()
            w.writerow(d)


def write_samples(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    fields = ["body", "parent", "et_seconds", "day_from_start", "err_total_m", "err_r_m", "err_t_m", "err_n_m", "abs_r_m", "abs_t_m", "abs_n_m"]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def print_report(rows: List[BodyDiagnostics]) -> None:
    print("\n=== RESIDUAL PHASE DIAGNOSTICS / RTN ===")
    print("Body        | Parent  | Max km   | RMS km   | Final km | RMS R/T/N km                 | Dom        | fT   | Final/Max | Diagnóstico")
    print("-" * 160)
    for r in sorted(rows, key=lambda x: -x.max_total_km):
        print(
            f"{r.body:<11} | {(r.parent or '-'):7} | "
            f"{r.max_total_km:8.3f} | {r.rms_total_km:8.3f} | {r.final_total_km:8.3f} | "
            f"{r.rms_r_km:8.3f}/{r.rms_t_km:8.3f}/{r.rms_n_km:8.3f} | "
            f"{r.dominant_component:<10} | {r.frac_t_rms:4.2f} | {r.final_over_max:9.3f} | {r.diagnosis}"
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Diagnóstico RTN/RSW de resíduos KSP/Principia vs REBOUND.")
    p.add_argument("--ksp-csv", required=True, type=Path, help="states.csv observado do KSP/kRPC")
    p.add_argument("--reb-csv", required=True, type=Path, help="rebound_states.csv do Nível A")
    p.add_argument("--bodies", nargs="+", required=True, help="Corpos a diagnosticar")
    p.add_argument("--parent", default=None, help="Corpo pai para análise relativa. Ex.: Jool, Sarnus, Urlum")
    p.add_argument("--epoch-tolerance-s", type=float, default=1e-3, help="Tolerância para casar epochs entre CSVs")
    p.add_argument("--output-dir", required=True, type=Path, help="Diretório de saída")
    p.add_argument("--write-samples", action="store_true", help="Salvar série temporal RTN por amostra")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    wanted = list(args.bodies)
    if args.parent and args.parent not in wanted:
        load_bodies = wanted + [args.parent]
    else:
        load_bodies = wanted

    print("Carregando CSVs...")
    print(f"KSP: {args.ksp_csv}")
    print(f"REB: {args.reb_csv}")
    ksp = load_states(args.ksp_csv, load_bodies, parent=args.parent)
    reb = load_states(args.reb_csv, load_bodies, parent=args.parent)

    summaries: List[BodyDiagnostics] = []
    sample_rows: List[Dict[str, object]] = []
    for body in wanted:
        if body == args.parent:
            continue
        print(f"Analisando {body}...")
        summaries.append(analyze_body(body, args.parent, ksp, reb, args.epoch_tolerance_s, sample_rows))

    summary_csv = args.output_dir / "phase_diagnostics_summary.csv"
    samples_csv = args.output_dir / "phase_diagnostics_samples.csv"
    summary_json = args.output_dir / "phase_diagnostics_summary.json"

    write_summary(summary_csv, summaries)
    if args.write_samples:
        write_samples(samples_csv, sample_rows)
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump([s.__dict__ for s in summaries], f, indent=2, ensure_ascii=False)

    print_report(summaries)
    print("\n[OK] summary CSV:", summary_csv)
    print("[OK] summary JSON:", summary_json)
    if args.write_samples:
        print("[OK] samples CSV:", samples_csv)


if __name__ == "__main__":
    main()
