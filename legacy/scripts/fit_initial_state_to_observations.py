#!/usr/bin/env python3
"""
fit_initial_state_to_observations.py

Ajuste observacional de estado inicial para o pipeline KSP/Principia -> REBOUND.

Objetivo
--------
Dado um snapshot inicial (x0, v0) e um states.csv observado pelo KSP/Principia,
estimar pequenas correções em posição e/ou velocidade para corpos selecionados,
minimizando os resíduos de posição ao longo de um arco curto. Depois validar no
arco completo e escrever um novo snapshot JSON.

Isto é uma etapa de orbit determination local, não um hack físico: o script
mantém μ/massas e integra todos os corpos no REBOUND. O ajuste deve ser usado
para diagnosticar e reduzir erros de estado inicial antes de gerar SPK final.

Dependências
------------
- rebound
- scipy
- numpy
- rebound_level_a_cache.py no mesmo diretório ou no PYTHONPATH

Exemplo — ajustar só velocidades de corpos com erro oscilatório:
python fit_initial_state_to_observations.py \
  --snapshot data/final_clean_120d/snapshot.json \
  --observations data/final_clean_120d/states.csv \
  --central-body Sun \
  --bodies Vall Tylo Pol Vant Zore \
  --fit-days 20 \
  --validate-days 120 \
  --fit-mode velocity \
  --velocity-scale-m-s 1.0 \
  --position-residual-scale-m 1000 \
  --regularization-weight 0.05 \
  --output-snapshot data/final_clean_120d/snapshot_fit_phase_bodies.json \
  --output-dir data/final_clean_120d/fit_phase_bodies

Exemplo — ajuste posição+velocidade para uma família:
python fit_initial_state_to_observations.py \
  --snapshot data/final_clean_120d/snapshot.json \
  --observations data/final_clean_120d/states.csv \
  --central-body Sun \
  --bodies Hale Eeloo Sarnus Slate Tekto Ovok \
  --fit-days 30 \
  --validate-days 120 \
  --fit-mode position_velocity \
  --position-scale-m 1000 \
  --velocity-scale-m-s 0.1 \
  --position-residual-scale-m 1000 \
  --regularization-weight 0.05 \
  --output-snapshot data/final_clean_120d/snapshot_fit_sarnus_family.json \
  --output-dir data/final_clean_120d/fit_sarnus_family
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import least_squares

# Importa o núcleo já validado do Level A para garantir a mesma semântica.
try:
    from rebound_level_a_cache import (  # type: ignore
        BodyInitialState,
        InputSnapshot,
        RuntimeConfig,
        build_rebound_simulation,
        load_snapshot_json,
        read_ksp_csv,
        relative_state_m,
        safe_integrate,
        norm3,
        sub3,
        rtn_components,
        apparent_epoch_offset_s,
        rms,
    )
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "Não consegui importar rebound_level_a_cache.py. "
        "Coloque este script na raiz do projeto, ao lado de rebound_level_a_cache.py.\n"
        f"Erro original: {exc}"
    ) from exc

DAY_S = 86400.0


def load_payload(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_payload(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def clone_snapshot_with_deltas(
    snapshot: InputSnapshot,
    bodies: Sequence[str],
    param_vector: np.ndarray,
    fit_mode: str,
    position_scale_m: float,
    velocity_scale_m_s: float,
) -> InputSnapshot:
    """Return a new InputSnapshot with selected state corrections applied.

    Optimization variables are dimensionless. They are converted using the
    provided scales.
    """
    new_bodies = dict(snapshot.bodies)
    cursor = 0
    for name in bodies:
        b = new_bodies[name]
        dx = dy = dz = 0.0
        dvx = dvy = dvz = 0.0
        if fit_mode in {"position", "position_velocity"}:
            dx = float(param_vector[cursor]) * position_scale_m
            dy = float(param_vector[cursor + 1]) * position_scale_m
            dz = float(param_vector[cursor + 2]) * position_scale_m
            cursor += 3
        if fit_mode in {"velocity", "position_velocity"}:
            dvx = float(param_vector[cursor]) * velocity_scale_m_s
            dvy = float(param_vector[cursor + 1]) * velocity_scale_m_s
            dvz = float(param_vector[cursor + 2]) * velocity_scale_m_s
            cursor += 3
        new_bodies[name] = replace(
            b,
            x_m=b.x_m + dx,
            y_m=b.y_m + dy,
            z_m=b.z_m + dz,
            vx_m_s=b.vx_m_s + dvx,
            vy_m_s=b.vy_m_s + dvy,
            vz_m_s=b.vz_m_s + dvz,
        )
    return InputSnapshot(
        reference_body=snapshot.reference_body,
        start_ut_s=snapshot.start_ut_s,
        et_offset_seconds=snapshot.et_offset_seconds,
        frame_convention=snapshot.frame_convention,
        bodies=new_bodies,
    )


def param_count(n_bodies: int, fit_mode: str) -> int:
    if fit_mode in {"position", "velocity"}:
        return 3 * n_bodies
    if fit_mode == "position_velocity":
        return 6 * n_bodies
    raise ValueError(f"fit_mode inválido: {fit_mode}")


def decode_deltas(
    bodies: Sequence[str],
    param_vector: np.ndarray,
    fit_mode: str,
    position_scale_m: float,
    velocity_scale_m_s: float,
) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    cursor = 0
    for name in bodies:
        d = {
            "dx_m": 0.0,
            "dy_m": 0.0,
            "dz_m": 0.0,
            "dvx_m_s": 0.0,
            "dvy_m_s": 0.0,
            "dvz_m_s": 0.0,
        }
        if fit_mode in {"position", "position_velocity"}:
            d["dx_m"] = float(param_vector[cursor]) * position_scale_m
            d["dy_m"] = float(param_vector[cursor + 1]) * position_scale_m
            d["dz_m"] = float(param_vector[cursor + 2]) * position_scale_m
            cursor += 3
        if fit_mode in {"velocity", "position_velocity"}:
            d["dvx_m_s"] = float(param_vector[cursor]) * velocity_scale_m_s
            d["dvy_m_s"] = float(param_vector[cursor + 1]) * velocity_scale_m_s
            d["dvz_m_s"] = float(param_vector[cursor + 2]) * velocity_scale_m_s
            cursor += 3
        d["dpos_norm_m"] = math.sqrt(d["dx_m"]**2 + d["dy_m"]**2 + d["dz_m"]**2)
        d["dvel_norm_m_s"] = math.sqrt(d["dvx_m_s"]**2 + d["dvy_m_s"]**2 + d["dvz_m_s"]**2)
        out[name] = d
    return out


def selected_epochs(
    ksp_by_et: Dict[float, List[Dict[str, Any]]],
    start_et: float,
    days: float,
    max_epochs: Optional[int],
) -> List[float]:
    end_et = start_et + days * DAY_S
    epochs = [et for et in sorted(ksp_by_et) if et >= start_et - 1e-9 and et <= end_et + 1e-9]
    if max_epochs is not None and max_epochs > 0 and len(epochs) > max_epochs:
        idx = np.linspace(0, len(epochs) - 1, max_epochs).round().astype(int)
        epochs = [epochs[i] for i in sorted(set(int(i) for i in idx))]
    return epochs


def observations_by_epoch_and_body(
    ksp_by_et: Dict[float, List[Dict[str, Any]]],
    epochs: Sequence[float],
    bodies: Sequence[str],
) -> Dict[float, Dict[str, Tuple[float, float, float, float, float, float]]]:
    wanted = set(bodies)
    out: Dict[float, Dict[str, Tuple[float, float, float, float, float, float]]] = {}
    for et in epochs:
        d: Dict[str, Tuple[float, float, float, float, float, float]] = {}
        for row in ksp_by_et.get(et, []):
            b = row["body"]
            if b in wanted:
                d[b] = tuple(float(x) for x in row["state_m"])  # type: ignore[assignment]
        if d:
            out[et] = d
    return out


def propagate_states(
    snapshot: InputSnapshot,
    cfg: RuntimeConfig,
    epochs: Sequence[float],
    output_bodies: Sequence[str],
) -> Tuple[Dict[float, Dict[str, Tuple[float, float, float, float, float, float]]], Optional[float]]:
    sim, ordered = build_rebound_simulation(snapshot, cfg)
    name_to_index = {name: i for i, name in enumerate(ordered)}
    center_index = name_to_index[cfg.central_body]
    start_et = snapshot.start_ut_s + snapshot.et_offset_seconds
    states: Dict[float, Dict[str, Tuple[float, float, float, float, float, float]]] = {}
    initial_energy = None
    try:
        initial_energy = float(sim.energy())
    except Exception:
        pass
    for et in epochs:
        safe_integrate(sim, float(et - start_et))
        per: Dict[str, Tuple[float, float, float, float, float, float]] = {}
        for b in output_bodies:
            if b not in name_to_index:
                continue
            per[b] = relative_state_m(sim, name_to_index[b], center_index)
        states[et] = per
    edrift = None
    if initial_energy is not None and initial_energy != 0.0:
        try:
            edrift = (float(sim.energy()) - initial_energy) / abs(initial_energy)
        except Exception:
            edrift = None
    return states, edrift


def residual_vector_for_params(
    params: np.ndarray,
    base_snapshot: InputSnapshot,
    cfg: RuntimeConfig,
    fit_bodies: Sequence[str],
    observed: Dict[float, Dict[str, Tuple[float, float, float, float, float, float]]],
    epochs: Sequence[float],
    fit_mode: str,
    position_scale_m: float,
    velocity_scale_m_s: float,
    position_residual_scale_m: float,
    velocity_residual_scale_m_s: Optional[float],
    regularization_weight: float,
) -> np.ndarray:
    test_snapshot = clone_snapshot_with_deltas(
        base_snapshot,
        fit_bodies,
        params,
        fit_mode,
        position_scale_m,
        velocity_scale_m_s,
    )
    pred, _ = propagate_states(test_snapshot, cfg, epochs, fit_bodies)
    res: List[float] = []
    for et in epochs:
        obs_at = observed.get(et, {})
        pred_at = pred.get(et, {})
        for b in fit_bodies:
            if b not in obs_at or b not in pred_at:
                continue
            obs = obs_at[b]
            pr = pred_at[b]
            # REBOUND - KSP; sign doesn't matter for least squares.
            res.extend([
                (pr[0] - obs[0]) / position_residual_scale_m,
                (pr[1] - obs[1]) / position_residual_scale_m,
                (pr[2] - obs[2]) / position_residual_scale_m,
            ])
            if velocity_residual_scale_m_s is not None and velocity_residual_scale_m_s > 0:
                res.extend([
                    (pr[3] - obs[3]) / velocity_residual_scale_m_s,
                    (pr[4] - obs[4]) / velocity_residual_scale_m_s,
                    (pr[5] - obs[5]) / velocity_residual_scale_m_s,
                ])
    if regularization_weight > 0:
        res.extend((math.sqrt(regularization_weight) * params).tolist())
    return np.array(res, dtype=float)


def summarize_residuals(
    pred: Dict[float, Dict[str, Tuple[float, float, float, float, float, float]]],
    observed: Dict[float, Dict[str, Tuple[float, float, float, float, float, float]]],
    bodies: Sequence[str],
) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for b in bodies:
        pos_errs: List[float] = []
        vel_errs: List[float] = []
        radial: List[float] = []
        trans: List[float] = []
        normal: List[float] = []
        app_dt: List[float] = []
        max_et = None
        max_err = -1.0
        final_pos = None
        final_vel = None
        n = 0
        for et in sorted(observed):
            obs_at = observed.get(et, {})
            pr_at = pred.get(et, {})
            if b not in obs_at or b not in pr_at:
                continue
            obs = obs_at[b]
            pr = pr_at[b]
            dr = sub3(obs[0:3], pr[0:3])
            dv = sub3(obs[3:6], pr[3:6])
            pe = norm3(dr)
            ve = norm3(dv)
            pos_errs.append(pe)
            vel_errs.append(ve)
            rt = rtn_components(dr, pr[0:3], pr[3:6])
            if rt[0] is not None:
                radial.append(float(rt[0]))
            if rt[1] is not None:
                trans.append(float(rt[1]))
            if rt[2] is not None:
                normal.append(float(rt[2]))
            dt = apparent_epoch_offset_s(dr, pr[3:6])
            if dt is not None and math.isfinite(dt):
                app_dt.append(float(dt))
            if pe > max_err:
                max_err = pe
                max_et = et
            final_pos = pe
            final_vel = ve
            n += 1
        def med(vals: List[float]) -> Optional[float]:
            if not vals:
                return None
            return float(np.median(np.array(vals, dtype=float)))
        out[b] = {
            "samples": n,
            "max_pos_err_m": max(pos_errs) if pos_errs else None,
            "rms_pos_err_m": math.sqrt(sum(x*x for x in pos_errs)/len(pos_errs)) if pos_errs else None,
            "final_pos_err_m": final_pos,
            "max_vel_err_m_s": max(vel_errs) if vel_errs else None,
            "rms_vel_err_m_s": math.sqrt(sum(x*x for x in vel_errs)/len(vel_errs)) if vel_errs else None,
            "final_vel_err_m_s": final_vel,
            "max_error_et_s": max_et,
            "median_radial_m": med(radial),
            "median_transverse_m": med(trans),
            "median_normal_m": med(normal),
            "median_apparent_epoch_offset_s": med(app_dt),
        }
    return out


def write_summary_csv(path: Path, summary: Dict[str, Dict[str, Any]], deltas: Optional[Dict[str, Dict[str, float]]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "body", "samples", "max_pos_err_km", "rms_pos_err_km", "final_pos_err_km",
        "max_vel_err_m_s", "rms_vel_err_m_s", "final_vel_err_m_s", "max_error_et_s",
        "median_radial_km", "median_transverse_km", "median_normal_km", "median_apparent_epoch_offset_s",
        "dpos_norm_m", "dvel_norm_m_s", "dx_m", "dy_m", "dz_m", "dvx_m_s", "dvy_m_s", "dvz_m_s",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for body, s in sorted(summary.items()):
            d = deltas.get(body, {}) if deltas else {}
            row = {
                "body": body,
                "samples": s.get("samples"),
                "max_pos_err_km": None if s.get("max_pos_err_m") is None else s["max_pos_err_m"] / 1000.0,
                "rms_pos_err_km": None if s.get("rms_pos_err_m") is None else s["rms_pos_err_m"] / 1000.0,
                "final_pos_err_km": None if s.get("final_pos_err_m") is None else s["final_pos_err_m"] / 1000.0,
                "max_vel_err_m_s": s.get("max_vel_err_m_s"),
                "rms_vel_err_m_s": s.get("rms_vel_err_m_s"),
                "final_vel_err_m_s": s.get("final_vel_err_m_s"),
                "max_error_et_s": s.get("max_error_et_s"),
                "median_radial_km": None if s.get("median_radial_m") is None else s["median_radial_m"] / 1000.0,
                "median_transverse_km": None if s.get("median_transverse_m") is None else s["median_transverse_m"] / 1000.0,
                "median_normal_km": None if s.get("median_normal_m") is None else s["median_normal_m"] / 1000.0,
                "median_apparent_epoch_offset_s": s.get("median_apparent_epoch_offset_s"),
                "dpos_norm_m": d.get("dpos_norm_m", 0.0),
                "dvel_norm_m_s": d.get("dvel_norm_m_s", 0.0),
                "dx_m": d.get("dx_m", 0.0),
                "dy_m": d.get("dy_m", 0.0),
                "dz_m": d.get("dz_m", 0.0),
                "dvx_m_s": d.get("dvx_m_s", 0.0),
                "dvy_m_s": d.get("dvy_m_s", 0.0),
                "dvz_m_s": d.get("dvz_m_s", 0.0),
            }
            w.writerow(row)


def update_snapshot_payload(
    original_payload: Dict[str, Any],
    deltas: Dict[str, Dict[str, float]],
) -> Dict[str, Any]:
    payload = json.loads(json.dumps(original_payload))  # deep copy JSON-safe
    eph = payload.get("ephemerides", {})
    for body, d in deltas.items():
        if body not in eph:
            continue
        states = eph[body].get("states") if isinstance(eph[body], dict) else None
        if not states:
            continue
        row = states[0]
        # row = [ut,x,y,z,vx,vy,vz]
        row[1] = float(row[1]) + float(d.get("dx_m", 0.0))
        row[2] = float(row[2]) + float(d.get("dy_m", 0.0))
        row[3] = float(row[3]) + float(d.get("dz_m", 0.0))
        row[4] = float(row[4]) + float(d.get("dvx_m_s", 0.0))
        row[5] = float(row[5]) + float(d.get("dvy_m_s", 0.0))
        row[6] = float(row[6]) + float(d.get("dvz_m_s", 0.0))
    payload.setdefault("fit_history", [])
    return payload


def main() -> int:
    ap = argparse.ArgumentParser(description="Ajusta estado inicial REBOUND contra observações KSP/Principia.")
    ap.add_argument("--snapshot", type=Path, required=True)
    ap.add_argument("--observations", type=Path, required=True, help="states.csv recortado para começar após o snapshot")
    ap.add_argument("--central-body", default="Sun")
    ap.add_argument("--bodies", nargs="+", required=True)
    ap.add_argument("--fit-days", type=float, default=20.0)
    ap.add_argument("--validate-days", type=float, default=120.0)
    ap.add_argument("--max-fit-epochs", type=int, default=80, help="Subamostra epochs de ajuste para acelerar least_squares")
    ap.add_argument("--max-nfev", type=int, default=80)
    ap.add_argument("--fit-mode", choices=["position", "velocity", "position_velocity"], default="velocity")
    ap.add_argument("--position-scale-m", type=float, default=1000.0, help="1 unidade otimizada = N metros")
    ap.add_argument("--velocity-scale-m-s", type=float, default=1.0, help="1 unidade otimizada = N m/s")
    ap.add_argument("--position-residual-scale-m", type=float, default=1000.0, help="Escala dos resíduos de posição")
    ap.add_argument("--velocity-residual-scale-m-s", type=float, default=0.0, help="0 desativa resíduos de velocidade")
    ap.add_argument("--regularization-weight", type=float, default=0.05)
    ap.add_argument("--bounds-position-units", type=float, default=1000.0, help="Limite em unidades normalizadas para posição")
    ap.add_argument("--bounds-velocity-units", type=float, default=100.0, help="Limite em unidades normalizadas para velocidade")
    ap.add_argument("--integrator", default="ias15")
    ap.add_argument("--ias15-epsilon", type=float, default=1e-11)
    ap.add_argument("--no-move-to-com", action="store_true")
    ap.add_argument("--output-snapshot", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--flip-z-input", action="store_true")
    ap.add_argument("--flip-z-ksp-csv", action="store_true")
    ap.add_argument("--body-catalog", type=Path, default=None)
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    start_wall = time.time()

    base_snapshot = load_snapshot_json(
        input_json=args.snapshot,
        body_catalog_path=args.body_catalog,
        central_body=args.central_body,
        initial_sample_index=0,
        et_offset_seconds=None,
        flip_z_input=args.flip_z_input,
    )
    start_et = base_snapshot.start_ut_s + base_snapshot.et_offset_seconds

    for b in args.bodies:
        if b not in base_snapshot.bodies:
            raise SystemExit(f"Corpo {b!r} não existe no snapshot.")

    ksp_by_et, _, meta = read_ksp_csv(args.observations, flip_z=args.flip_z_ksp_csv)
    if meta.get("min_et") is not None and float(meta["min_et"]) < start_et - 1e-6:
        raise SystemExit(
            f"Observações começam antes do snapshot: min_et={meta['min_et']} start_et={start_et}. "
            "Recorte o CSV antes de ajustar."
        )

    fit_epochs = selected_epochs(ksp_by_et, start_et, args.fit_days, args.max_fit_epochs)
    val_epochs = selected_epochs(ksp_by_et, start_et, args.validate_days, None)
    if len(fit_epochs) < 3:
        raise SystemExit("Poucos epochs de ajuste; aumente --fit-days ou verifique CSV.")
    if not val_epochs:
        raise SystemExit("Nenhum epoch de validação selecionado.")

    fit_obs = observations_by_epoch_and_body(ksp_by_et, fit_epochs, args.bodies)
    val_obs = observations_by_epoch_and_body(ksp_by_et, val_epochs, args.bodies)

    cfg = RuntimeConfig(
        central_body=args.central_body,
        integrator=args.integrator,
        ias15_epsilon=args.ias15_epsilon,
        whfast_dt_seconds=None,
        move_to_com=not args.no_move_to_com,
        output_frame="central_relative",
    )

    npar = param_count(len(args.bodies), args.fit_mode)
    x0 = np.zeros(npar, dtype=float)
    lower: List[float] = []
    upper: List[float] = []
    for _ in args.bodies:
        if args.fit_mode in {"position", "position_velocity"}:
            lower.extend([-args.bounds_position_units] * 3)
            upper.extend([args.bounds_position_units] * 3)
        if args.fit_mode in {"velocity", "position_velocity"}:
            lower.extend([-args.bounds_velocity_units] * 3)
            upper.extend([args.bounds_velocity_units] * 3)

    print("=== FIT INITIAL STATE ===")
    print(f"snapshot: {args.snapshot}")
    print(f"observations: {args.observations}")
    print(f"start_et: {start_et:.9f}")
    print(f"bodies: {', '.join(args.bodies)}")
    print(f"fit_epochs: {len(fit_epochs)} over {args.fit_days} days; validate_epochs: {len(val_epochs)} over {args.validate_days} days")
    print(f"fit_mode={args.fit_mode}; n_params={npar}")

    # Baseline validation before fitting.
    baseline_pred, baseline_edrift = propagate_states(base_snapshot, cfg, val_epochs, args.bodies)
    baseline_summary = summarize_residuals(baseline_pred, val_obs, args.bodies)
    write_summary_csv(args.output_dir / "baseline_validation.csv", baseline_summary)

    def fun(p: np.ndarray) -> np.ndarray:
        return residual_vector_for_params(
            p,
            base_snapshot,
            cfg,
            args.bodies,
            fit_obs,
            fit_epochs,
            args.fit_mode,
            args.position_scale_m,
            args.velocity_scale_m_s,
            args.position_residual_scale_m,
            args.velocity_residual_scale_m_s if args.velocity_residual_scale_m_s > 0 else None,
            args.regularization_weight,
        )

    r0 = fun(x0)
    print(f"initial objective_norm={float(np.linalg.norm(r0)):.6g}; residual_len={len(r0)}")

    result = least_squares(
        fun,
        x0,
        bounds=(np.array(lower, dtype=float), np.array(upper, dtype=float)),
        method="trf",
        loss="soft_l1",
        f_scale=1.0,
        x_scale="jac",
        max_nfev=args.max_nfev,
        verbose=2,
    )

    deltas = decode_deltas(args.bodies, result.x, args.fit_mode, args.position_scale_m, args.velocity_scale_m_s)
    fitted_snapshot = clone_snapshot_with_deltas(
        base_snapshot,
        args.bodies,
        result.x,
        args.fit_mode,
        args.position_scale_m,
        args.velocity_scale_m_s,
    )

    fit_pred, fit_edrift = propagate_states(fitted_snapshot, cfg, val_epochs, args.bodies)
    fit_summary = summarize_residuals(fit_pred, val_obs, args.bodies)
    write_summary_csv(args.output_dir / "fitted_validation.csv", fit_summary, deltas)

    # Write before/after comparison.
    compare_rows = []
    for b in args.bodies:
        before = baseline_summary.get(b, {})
        after = fit_summary.get(b, {})
        def val(s: Dict[str, Any], key: str) -> Optional[float]:
            x = s.get(key)
            return None if x is None else float(x)
        before_rms = val(before, "rms_pos_err_m")
        after_rms = val(after, "rms_pos_err_m")
        before_max = val(before, "max_pos_err_m")
        after_max = val(after, "max_pos_err_m")
        compare_rows.append({
            "body": b,
            "before_max_km": None if before_max is None else before_max / 1000.0,
            "after_max_km": None if after_max is None else after_max / 1000.0,
            "max_improvement_factor": None if not before_max or after_max is None or after_max == 0 else before_max / after_max,
            "before_rms_km": None if before_rms is None else before_rms / 1000.0,
            "after_rms_km": None if after_rms is None else after_rms / 1000.0,
            "rms_improvement_factor": None if not before_rms or after_rms is None or after_rms == 0 else before_rms / after_rms,
            "after_final_km": None if after.get("final_pos_err_m") is None else float(after["final_pos_err_m"]) / 1000.0,
            **deltas[b],
        })

    with (args.output_dir / "before_after.csv").open("w", newline="", encoding="utf-8") as f:
        fields = list(compare_rows[0].keys()) if compare_rows else ["body"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in compare_rows:
            w.writerow(row)

    original_payload = load_payload(args.snapshot)
    fitted_payload = update_snapshot_payload(original_payload, deltas)
    fitted_payload.setdefault("fit_history", [])
    fitted_payload["fit_history"].append({
        "tool": "fit_initial_state_to_observations.py",
        "source_snapshot": str(args.snapshot),
        "observations": str(args.observations),
        "central_body": args.central_body,
        "bodies": list(args.bodies),
        "fit_mode": args.fit_mode,
        "fit_days": args.fit_days,
        "validate_days": args.validate_days,
        "max_fit_epochs": args.max_fit_epochs,
        "position_scale_m": args.position_scale_m,
        "velocity_scale_m_s": args.velocity_scale_m_s,
        "position_residual_scale_m": args.position_residual_scale_m,
        "velocity_residual_scale_m_s": args.velocity_residual_scale_m_s,
        "regularization_weight": args.regularization_weight,
        "optimizer_success": bool(result.success),
        "optimizer_status": int(result.status),
        "optimizer_message": str(result.message),
        "nfev": int(result.nfev),
        "cost": float(result.cost),
        "optimality": float(result.optimality),
        "baseline_edrift": baseline_edrift,
        "fitted_edrift": fit_edrift,
        "deltas": deltas,
    })
    write_payload(args.output_snapshot, fitted_payload)

    report = {
        "schema": "fit_initial_state_to_observations.v1",
        "snapshot": str(args.snapshot),
        "observations": str(args.observations),
        "output_snapshot": str(args.output_snapshot),
        "start_et_s": start_et,
        "fit_epochs": len(fit_epochs),
        "validate_epochs": len(val_epochs),
        "fit_mode": args.fit_mode,
        "optimizer": {
            "success": bool(result.success),
            "status": int(result.status),
            "message": str(result.message),
            "nfev": int(result.nfev),
            "cost": float(result.cost),
            "optimality": float(result.optimality),
        },
        "deltas": deltas,
        "baseline_edrift": baseline_edrift,
        "fitted_edrift": fit_edrift,
        "baseline_summary": baseline_summary,
        "fitted_summary": fit_summary,
        "wall_seconds": time.time() - start_wall,
    }
    write_payload(args.output_dir / "fit_report.json", report)

    print("\n=== BEFORE / AFTER ===")
    print(f"{'Body':<12} {'Max before km':>14} {'Max after km':>14} {'RMS before km':>14} {'RMS after km':>14} {'|dV| m/s':>12} {'|dR| m':>12}")
    for row in compare_rows:
        print(
            f"{row['body']:<12} "
            f"{row['before_max_km']:14.6f} {row['after_max_km']:14.6f} "
            f"{row['before_rms_km']:14.6f} {row['after_rms_km']:14.6f} "
            f"{row['dvel_norm_m_s']:12.6g} {row['dpos_norm_m']:12.6g}"
        )
    print(f"\n[OK] output snapshot: {args.output_snapshot}")
    print(f"[OK] report dir: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
