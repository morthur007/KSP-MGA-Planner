#!/usr/bin/env python3
"""
principia_live_force_audit.py

Auditoria viva kRPC/Principia por cinemática observada.

Objetivo:
- Não usar body.orbit como verdade dinâmica.
- Não usar body.velocity() como verdade primária.
- Amostrar posições dos corpos vivos no KSP/kRPC em um frame comum.
- Ajustar polinômios locais para extrair r, v, a em um epoch central.
- Comparar aceleração medida contra aceleração point-mass prevista pelos μ do jogo.
- Testar hipóteses do tipo "corpo X está omitido da gravidade do Principia" via:
    1) modelo full;
    2) modelo sem cada suspeito;
    3) multiplicador efetivo de μ para cada suspeito;
    4) multiplicador efetivo de μ do pai.

Interpretação importante:
- Este script NÃO lê a memória C++ interna do Principia.
- Ele mede a dinâmica viva observável via kRPC.
- Se remover um suspeito melhora muito o resíduo e o multiplicador efetivo fica perto de 0,
  isso é evidência operacional de "corpo se comporta como omitido" no modelo medido.
- Se a contribuição gravitacional do suspeito é muito menor que o resíduo, o teste é inconclusivo.

Exemplo:
python principia_live_force_audit.py \
  --central-body Sun \
  --parent Jool \
  --targets Laythe Vall Tylo Bop Pol \
  --suspects Vall Bop Pol Laythe Tylo Jool \
  --dt 120 \
  --points 7 \
  --poly-degree 4 \
  --sample-mode real \
  --output-dir data/live_force_audit_jool_dt120
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np

try:
    import krpc  # type: ignore
except ImportError as exc:
    raise SystemExit("O pacote 'krpc' não está instalado neste ambiente Python.") from exc


G_SI = 6.67430e-11
EPS = 1e-30
Vec = np.ndarray


def norm(v: Vec) -> float:
    return float(np.linalg.norm(v))


def unit(v: Vec) -> Vec:
    n = norm(v)
    if n <= 0:
        return np.zeros(3, dtype=float)
    return v / n


def angle_deg(a: Vec, b: Vec) -> float:
    na = norm(a)
    nb = norm(b)
    if na <= 0 or nb <= 0:
        return float("nan")
    c = float(np.dot(a, b) / (na * nb))
    c = max(-1.0, min(1.0, c))
    return math.degrees(math.acos(c))


def safe_float(x: Any, default: float = float("nan")) -> float:
    try:
        return float(x)
    except Exception:
        return default


def get_body_names(bodies: Any) -> List[str]:
    try:
        return list(bodies.keys())
    except Exception:
        return [getattr(b, "name", str(i)) for i, b in enumerate(bodies)]


def get_body(bodies: Any, name: str) -> Any:
    try:
        return bodies[name]
    except Exception:
        for b in bodies:
            if getattr(b, "name", None) == name:
                return b
    raise KeyError(name)


def maybe_get(obj: Any, attr: str, default: Any = None) -> Any:
    try:
        return getattr(obj, attr)
    except Exception:
        return default


@dataclass
class FittedState:
    body: str
    r: Vec
    v: Vec
    a: Vec
    fit_rms_m: float
    fit_max_m: float


def fit_state_from_samples(
    body: str,
    times: np.ndarray,
    positions: np.ndarray,
    t_mid: float,
    degree: int,
) -> FittedState:
    """Fit x/y/z(tau) and return r,v,a at tau=0."""
    t_centered = times - t_mid
    scale = float(np.max(np.abs(t_centered)))
    if scale <= 0:
        raise ValueError("Escala temporal inválida para polyfit.")

    tau = t_centered / scale
    deg = min(int(degree), len(times) - 1)

    r = np.zeros(3, dtype=float)
    v = np.zeros(3, dtype=float)
    a = np.zeros(3, dtype=float)
    residuals = []

    for j in range(3):
        coeff = np.polyfit(tau, positions[:, j], deg)
        p = np.poly1d(coeff)
        dp = np.polyder(p, 1)
        ddp = np.polyder(p, 2)

        r[j] = float(p(0.0))
        v[j] = float(dp(0.0) / scale)
        a[j] = float(ddp(0.0) / (scale * scale))

        pred = p(tau)
        residuals.append(pred - positions[:, j])

    res = np.vstack(residuals).T
    res_norms = np.linalg.norm(res, axis=1)
    return FittedState(
        body=body,
        r=r,
        v=v,
        a=a,
        fit_rms_m=float(np.sqrt(np.mean(res_norms * res_norms))),
        fit_max_m=float(np.max(res_norms)),
    )


def compute_accelerations(
    states: Dict[str, FittedState],
    mus: Dict[str, float],
    include: Iterable[str],
) -> Dict[str, Vec]:
    include_set = set(include)
    acc: Dict[str, Vec] = {}
    for i_name, i_state in states.items():
        ai = np.zeros(3, dtype=float)
        ri = i_state.r
        for k_name in include_set:
            if k_name == i_name:
                continue
            mu = mus.get(k_name, 0.0)
            if mu <= 0:
                continue
            rk = states[k_name].r
            dr = rk - ri
            d = norm(dr)
            if d <= 0:
                continue
            ai += mu * dr / (d ** 3)
        acc[i_name] = ai
    return acc


def radial_tangential_normal_basis(r_rel: Vec, v_rel: Vec) -> Tuple[Vec, Vec, Vec]:
    er = unit(r_rel)
    h = np.cross(r_rel, v_rel)
    en = unit(h)
    et = unit(np.cross(en, er))
    return er, et, en


def projection_components(vec: Vec, er: Vec, et: Vec, en: Vec) -> Tuple[float, float, float]:
    return float(np.dot(vec, er)), float(np.dot(vec, et)), float(np.dot(vec, en))


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def capture_samples(
    sc: Any,
    bodies: Any,
    body_names: List[str],
    frame: Any,
    points: int,
    dt: float,
    sample_mode: str,
    settle_s: float,
) -> Tuple[np.ndarray, List[Dict[str, Vec]]]:
    samples: List[Dict[str, Vec]] = []
    times: List[float] = []

    start_ut = float(sc.ut)
    print(f"[CAPTURE] start_ut={start_ut:.6f}; points={points}; dt={dt}; mode={sample_mode}")

    if sample_mode == "real":
        try:
            sc.rails_warp_factor = 0
        except Exception:
            pass
        try:
            sc.physics_warp_factor = 0
        except Exception:
            pass
        time.sleep(settle_s)

    for i in range(points):
        target_ut = start_ut + i * dt

        if sample_mode == "warp":
            try:
                sc.warp_to(target_ut)
            except Exception as exc:
                print(f"[WARN] warp_to falhou ({exc}); caindo para espera real.")
                while float(sc.ut) < target_ut:
                    time.sleep(0.01)
        else:
            while float(sc.ut) < target_ut:
                time.sleep(0.01)

        if settle_s > 0:
            time.sleep(settle_s)

        current_ut = float(sc.ut)
        pos: Dict[str, Vec] = {}
        for name in body_names:
            b = get_body(bodies, name)
            try:
                p = b.position(frame)
                pos[name] = np.array([float(p[0]), float(p[1]), float(p[2])], dtype=float)
            except Exception:
                pos[name] = np.array([float("nan"), float("nan"), float("nan")], dtype=float)

        times.append(current_ut)
        samples.append(pos)
        print(f"[CAPTURE] sample {i+1}/{points}: UT={current_ut:.6f} target={target_ut:.6f} err={current_ut-target_ut:+.6f}s")

    return np.array(times, dtype=float), samples


def main() -> None:
    ap = argparse.ArgumentParser(description="Auditoria viva kRPC/Principia por aceleração vetorial.")
    ap.add_argument("--central-body", default="Sun", help="Corpo cujo non_rotating_reference_frame será usado como frame global.")
    ap.add_argument("--parent", default="Jool", help="Pai para análise relativa, ex: Jool.")
    ap.add_argument("--targets", nargs="+", default=["Laythe", "Vall", "Tylo", "Bop", "Pol"], help="Corpos-alvo para auditoria relativa.")
    ap.add_argument("--suspects", nargs="+", default=None, help="Corpos a testar como omitidos. Default: parent + targets.")
    ap.add_argument("--dt", type=float, default=120.0, help="Espaçamento entre amostras em segundos de UT.")
    ap.add_argument("--points", type=int, default=7, help="Número de amostras, preferencialmente ímpar.")
    ap.add_argument("--poly-degree", type=int, default=4, help="Grau do polinômio local.")
    ap.add_argument("--sample-mode", choices=["real", "warp"], default="real", help="real=espera 1x; warp=usa warp_to.")
    ap.add_argument("--settle-s", type=float, default=0.05, help="Pausa real após cada amostra/warp.")
    ap.add_argument("--output-dir", type=Path, required=True, help="Diretório de saída.")
    ap.add_argument("--all-bodies", action="store_true", help="Também audita todos os corpos no force_residuals.csv.")
    args = ap.parse_args()

    if args.points < 5:
        raise SystemExit("--points deve ser >= 5.")
    if args.points % 2 == 0:
        raise SystemExit("--points deve ser ímpar para haver uma amostra central.")
    if args.poly_degree >= args.points:
        raise SystemExit("--poly-degree deve ser menor que --points.")

    conn = krpc.connect(name="Principia_Live_Force_Audit")
    try:
        sc = conn.space_center
        bodies = sc.bodies
        body_names = get_body_names(bodies)

        if args.central_body not in body_names:
            raise SystemExit(f"Corpo central não encontrado: {args.central_body}")
        if args.parent not in body_names:
            raise SystemExit(f"Pai não encontrado: {args.parent}")
        for t in args.targets:
            if t not in body_names:
                raise SystemExit(f"Target não encontrado: {t}")

        suspects = args.suspects or [args.parent] + args.targets
        for s in suspects:
            if s not in body_names:
                raise SystemExit(f"Suspeito não encontrado: {s}")

        print("=" * 88)
        print("SERVIÇOS kRPC")
        print("=" * 88)
        services = []
        try:
            services = [s.name for s in conn.krpc.get_services().services]
            print(", ".join(services))
            if "Principia" in services:
                print("[INFO] Serviço Principia exposto via kRPC.")
            else:
                print("[INFO] Nenhum serviço Principia exposto via kRPC. Isso não implica Principia ausente.")
        except Exception as exc:
            print(f"[WARN] Não foi possível listar serviços: {exc}")

        central = get_body(bodies, args.central_body)
        frame = central.non_rotating_reference_frame

        mus: Dict[str, float] = {}
        catalog: Dict[str, Any] = {}
        for name in body_names:
            b = get_body(bodies, name)
            mu = safe_float(maybe_get(b, "gravitational_parameter"))
            mus[name] = mu
            catalog[name] = {
                "mu_m3_s2": mu,
                "mass_kg_from_mu": mu / G_SI if math.isfinite(mu) else None,
                "equatorial_radius_m": safe_float(maybe_get(b, "equatorial_radius")),
                "sphere_of_influence_m": safe_float(maybe_get(b, "sphere_of_influence")),
                "rotational_period_s": safe_float(maybe_get(b, "rotational_period")),
                "has_atmosphere": bool(maybe_get(b, "has_atmosphere", False)),
                "atmosphere_depth_m": safe_float(maybe_get(b, "atmosphere_depth")),
            }

        times, samples = capture_samples(
            sc=sc,
            bodies=bodies,
            body_names=body_names,
            frame=frame,
            points=args.points,
            dt=args.dt,
            sample_mode=args.sample_mode,
            settle_s=args.settle_s,
        )

        mid_idx = args.points // 2
        t_mid = float(times[mid_idx])

        states: Dict[str, FittedState] = {}
        fit_rows: List[Dict[str, Any]] = []
        for name in body_names:
            pos_arr = np.vstack([s[name] for s in samples])
            if not np.isfinite(pos_arr).all():
                continue
            st = fit_state_from_samples(name, times, pos_arr, t_mid, args.poly_degree)
            states[name] = st
            fit_rows.append({
                "body": name,
                "fit_rms_m": st.fit_rms_m,
                "fit_max_m": st.fit_max_m,
                "x_m": st.r[0], "y_m": st.r[1], "z_m": st.r[2],
                "vx_m_s": st.v[0], "vy_m_s": st.v[1], "vz_m_s": st.v[2],
                "ax_m_s2": st.a[0], "ay_m_s2": st.a[1], "az_m_s2": st.a[2],
            })

        include_all = list(states.keys())
        pred_full = compute_accelerations(states, mus, include_all)

        audit_targets = list(states.keys()) if args.all_bodies else [args.parent] + args.targets
        force_rows: List[Dict[str, Any]] = []
        for target in audit_targets:
            if target not in states:
                continue
            a_meas = states[target].a
            a_pred = pred_full[target]
            res = a_meas - a_pred
            force_rows.append({
                "target": target,
                "mode": "absolute",
                "parent": "",
                "measured_acc_norm_m_s2": norm(a_meas),
                "predicted_acc_norm_m_s2": norm(a_pred),
                "residual_acc_norm_m_s2": norm(res),
                "relative_residual_pct": 100.0 * norm(res) / max(norm(a_meas), EPS),
                "angle_meas_pred_deg": angle_deg(a_meas, a_pred),
                "radial_residual_m_s2": "",
                "transverse_residual_m_s2": "",
                "normal_residual_m_s2": "",
            })

        parent = args.parent
        for target in args.targets:
            if target not in states or parent not in states:
                continue
            a_meas_rel = states[target].a - states[parent].a
            a_pred_rel = pred_full[target] - pred_full[parent]
            res_rel = a_meas_rel - a_pred_rel
            r_rel = states[target].r - states[parent].r
            v_rel = states[target].v - states[parent].v
            er, et, en = radial_tangential_normal_basis(r_rel, v_rel)
            rr, rt, rn = projection_components(res_rel, er, et, en)
            force_rows.append({
                "target": target,
                "mode": "relative_to_parent",
                "parent": parent,
                "measured_acc_norm_m_s2": norm(a_meas_rel),
                "predicted_acc_norm_m_s2": norm(a_pred_rel),
                "residual_acc_norm_m_s2": norm(res_rel),
                "relative_residual_pct": 100.0 * norm(res_rel) / max(norm(a_meas_rel), EPS),
                "angle_meas_pred_deg": angle_deg(a_meas_rel, a_pred_rel),
                "radial_residual_m_s2": rr,
                "transverse_residual_m_s2": rt,
                "normal_residual_m_s2": rn,
            })

        omission_rows: List[Dict[str, Any]] = []
        for suspect in suspects:
            include_without = [b for b in include_all if b != suspect]
            pred_without = compute_accelerations(states, mus, include_without)

            for target in args.targets:
                if target not in states or parent not in states:
                    continue

                meas_rel = states[target].a - states[parent].a
                full_rel = pred_full[target] - pred_full[parent]
                base_rel = pred_without[target] - pred_without[parent]
                contrib = full_rel - base_rel

                res_full = meas_rel - full_rel
                res_without = meas_rel - base_rel
                contrib_norm = norm(contrib)

                denom = float(np.dot(contrib, contrib))
                alpha = float("nan")
                if denom > 0:
                    alpha = float(np.dot(meas_rel - base_rel, contrib) / denom)

                improvement = norm(res_full) / max(norm(res_without), EPS)
                if contrib_norm < max(norm(res_full), EPS) * 1e-3:
                    verdict = "INCONCLUSIVO: contribuição pequena demais"
                elif improvement > 2.0 and abs(alpha) < 0.5:
                    verdict = "FORTE: remover suspeito melhora; alpha baixo"
                elif improvement > 1.1:
                    verdict = "FRACO: remover suspeito melhora"
                elif 0.5 <= alpha <= 1.5:
                    verdict = "compatível com μ nominal"
                else:
                    verdict = "sem evidência de omissão"

                omission_rows.append({
                    "target": target,
                    "parent": parent,
                    "suspect": suspect,
                    "suspect_mu_m3_s2": mus.get(suspect, float("nan")),
                    "contribution_norm_m_s2": contrib_norm,
                    "residual_full_norm_m_s2": norm(res_full),
                    "residual_without_suspect_norm_m_s2": norm(res_without),
                    "improvement_full_over_without": improvement,
                    "alpha_effective_mu_multiplier": alpha,
                    "effective_mu_m3_s2": alpha * mus.get(suspect, float("nan")) if math.isfinite(alpha) else float("nan"),
                    "verdict": verdict,
                })

        parent_rows: List[Dict[str, Any]] = []
        if parent in states:
            include_without_parent = [b for b in include_all if b != parent]
            pred_without_parent = compute_accelerations(states, mus, include_without_parent)
            for target in args.targets:
                if target not in states:
                    continue
                meas_rel = states[target].a - states[parent].a
                full_rel = pred_full[target] - pred_full[parent]
                base_rel = pred_without_parent[target] - pred_without_parent[parent]
                parent_contrib = full_rel - base_rel
                denom = float(np.dot(parent_contrib, parent_contrib))
                alpha = float("nan")
                if denom > 0:
                    alpha = float(np.dot(meas_rel - base_rel, parent_contrib) / denom)
                res_alpha = meas_rel - (base_rel + alpha * parent_contrib) if math.isfinite(alpha) else np.full(3, float("nan"))
                parent_rows.append({
                    "target": target,
                    "parent": parent,
                    "parent_mu_nominal_m3_s2": mus[parent],
                    "alpha_parent_mu_multiplier": alpha,
                    "effective_parent_mu_m3_s2": alpha * mus[parent] if math.isfinite(alpha) else float("nan"),
                    "nominal_residual_norm_m_s2": norm(meas_rel - full_rel),
                    "fitted_residual_norm_m_s2": norm(res_alpha) if np.isfinite(res_alpha).all() else float("nan"),
                    "parent_contribution_norm_m_s2": norm(parent_contrib),
                })

        plane_rows: List[Dict[str, Any]] = []
        x_axis = np.array([1.0, 0.0, 0.0])
        y_axis = np.array([0.0, 1.0, 0.0])
        z_axis = np.array([0.0, 0.0, 1.0])
        for target in args.targets:
            if target not in states or parent not in states:
                continue
            r_rel = states[target].r - states[parent].r
            v_rel = states[target].v - states[parent].v
            h = np.cross(r_rel, v_rel)
            hhat = unit(h)
            api_inc_deg = float("nan")
            try:
                api_inc_deg = math.degrees(get_body(bodies, target).orbit.inclination) % 360.0
            except Exception:
                pass
            plane_rows.append({
                "target": target,
                "parent": parent,
                "api_orbit_inclination_deg": api_inc_deg,
                "hhat_x": hhat[0], "hhat_y": hhat[1], "hhat_z": hhat[2],
                "angle_h_to_x_deg": angle_deg(hhat, x_axis),
                "angle_h_to_y_deg": angle_deg(hhat, y_axis),
                "angle_h_to_z_deg": angle_deg(hhat, z_axis),
                "note": "Inclinação vetorial depende do plano/eixo de referência; compare eixos, não acuse API diretamente.",
            })

        out = args.output_dir
        out.mkdir(parents=True, exist_ok=True)

        write_csv(out / "fit_states.csv", fit_rows, [
            "body", "fit_rms_m", "fit_max_m",
            "x_m", "y_m", "z_m", "vx_m_s", "vy_m_s", "vz_m_s", "ax_m_s2", "ay_m_s2", "az_m_s2",
        ])
        write_csv(out / "force_residuals.csv", force_rows, [
            "target", "mode", "parent",
            "measured_acc_norm_m_s2", "predicted_acc_norm_m_s2",
            "residual_acc_norm_m_s2", "relative_residual_pct", "angle_meas_pred_deg",
            "radial_residual_m_s2", "transverse_residual_m_s2", "normal_residual_m_s2",
        ])
        write_csv(out / "omission_audit.csv", omission_rows, [
            "target", "parent", "suspect", "suspect_mu_m3_s2",
            "contribution_norm_m_s2", "residual_full_norm_m_s2",
            "residual_without_suspect_norm_m_s2", "improvement_full_over_without",
            "alpha_effective_mu_multiplier", "effective_mu_m3_s2", "verdict",
        ])
        write_csv(out / "parent_mu_fit.csv", parent_rows, [
            "target", "parent", "parent_mu_nominal_m3_s2",
            "alpha_parent_mu_multiplier", "effective_parent_mu_m3_s2",
            "nominal_residual_norm_m_s2", "fitted_residual_norm_m_s2",
            "parent_contribution_norm_m_s2",
        ])
        write_csv(out / "plane_diagnostics.csv", plane_rows, [
            "target", "parent", "api_orbit_inclination_deg",
            "hhat_x", "hhat_y", "hhat_z",
            "angle_h_to_x_deg", "angle_h_to_y_deg", "angle_h_to_z_deg", "note",
        ])

        raw_names = sorted(set([parent] + args.targets + suspects + [args.central_body]))
        raw_payload = {
            "schema": "principia_live_force_audit.v1",
            "central_body": args.central_body,
            "parent": parent,
            "targets": args.targets,
            "suspects": suspects,
            "sample_mode": args.sample_mode,
            "dt_seconds": args.dt,
            "points": args.points,
            "poly_degree": args.poly_degree,
            "times_ut_seconds": [float(t) for t in times],
            "t_mid_ut_seconds": t_mid,
            "services": services,
            "catalog": catalog,
            "raw_positions_m": [
                {name: samples[i][name].tolist() for name in raw_names if name in samples[i]}
                for i in range(len(samples))
            ],
        }
        (out / "audit_payload.json").write_text(json.dumps(raw_payload, indent=2), encoding="utf-8")

        print("\n" + "=" * 88)
        print("RESUMO: força relativa ao pai")
        print("=" * 88)
        for row in force_rows:
            if row["mode"] != "relative_to_parent":
                continue
            print(
                f"{row['target']:<10} residual={row['residual_acc_norm_m_s2']:.6e} m/s² "
                f"({row['relative_residual_pct']:.3f}%) angle={row['angle_meas_pred_deg']:.3f}°"
            )

        print("\n" + "=" * 88)
        print("RESUMO: teste de omissão")
        print("=" * 88)
        for row in omission_rows:
            imp = row["improvement_full_over_without"]
            alpha = row["alpha_effective_mu_multiplier"]
            if imp > 1.1 or (math.isfinite(alpha) and (alpha < 0.5 or alpha > 1.5)):
                print(
                    f"target={row['target']:<8} suspect={row['suspect']:<8} "
                    f"improvement={imp:.3f} alpha={alpha:.3g} verdict={row['verdict']}"
                )

        print("\n[OK] Arquivos escritos em:", out)
        print(" - fit_states.csv")
        print(" - force_residuals.csv")
        print(" - omission_audit.csv")
        print(" - parent_mu_fit.csv")
        print(" - plane_diagnostics.csv")
        print(" - audit_payload.json")

    finally:
        try:
            conn.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
