#!/usr/bin/env python3
"""
reboundx_j2_sweep_level_a.py

A/B controlado para testar a hipótese J2/achatamento polar em um corpo
(e.g. Jool) usando os dados já coletados:

  KSP/Principia states.csv  vs  REBOUND/REBOUNDx direto

Sem SPK, sem Chebyshev, sem kRPC ao vivo.

O script faz uma varredura de J2 e eixo de spin (Omega) usando o efeito
REBOUNDx `gravitational_harmonics`, mede erro heliocêntrico e erro relativo
lua-pai, e escreve um ranking. Use isso como teste de hipótese, não como
calibração final.

Exemplo:
  python reboundx_j2_sweep_level_a.py \
    --input-json data/clean_snapshot_poly7_dt120.json \
    --ksp-csv data/fast_opm_mpe_120d/states_after_snapshot.csv \
    --central-body Sun \
    --parent Jool \
    --targets Laythe Vall Tylo Bop Pol \
    --j2-body Jool \
    --j2-radius-m 6000000 \
    --j2-values 0,0.0001,0.0003,0.001,0.003,0.01,0.03 \
    --axis-sweep x,y,z \
    --output-dir data/j2_sweep_jool_120d

Unidades: SI puro (m, kg, s).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

G_SI = 6.67430e-11

Vec3 = Tuple[float, float, float]
State6 = Tuple[float, float, float, float, float, float]


@dataclass(frozen=True)
class BodyState:
    name: str
    mu_m3_s2: float
    mass_kg: float
    state: State6
    radius_m: Optional[float] = None


@dataclass(frozen=True)
class Snapshot:
    reference_body: str
    start_ut_s: float
    et_offset_seconds: float
    bodies: Dict[str, BodyState]

    @property
    def start_et_s(self) -> float:
        return self.start_ut_s + self.et_offset_seconds


def norm3(v: Sequence[float]) -> float:
    return math.sqrt(float(v[0])**2 + float(v[1])**2 + float(v[2])**2)


def sub3(a: Sequence[float], b: Sequence[float]) -> Vec3:
    return (float(a[0])-float(b[0]), float(a[1])-float(b[1]), float(a[2])-float(b[2]))


def norm6_pos(a: State6, b: State6) -> float:
    return norm3((a[0]-b[0], a[1]-b[1], a[2]-b[2]))


def norm6_vel(a: State6, b: State6) -> float:
    return norm3((a[3]-b[3], a[4]-b[4], a[5]-b[5]))


def rel_state(child: State6, parent: State6) -> State6:
    return (
        child[0]-parent[0], child[1]-parent[1], child[2]-parent[2],
        child[3]-parent[3], child[4]-parent[4], child[5]-parent[5],
    )


def rms(xs: Sequence[float]) -> float:
    vals = [x for x in xs if math.isfinite(x)]
    return math.sqrt(sum(x*x for x in vals)/len(vals)) if vals else float("nan")


def median(xs: Sequence[float]) -> float:
    vals = [x for x in xs if math.isfinite(x)]
    return statistics.median(vals) if vals else float("nan")


def parse_csv_list(s: str) -> List[str]:
    return [x.strip() for x in s.replace(";", ",").split(",") if x.strip()]


def parse_float_list(s: str) -> List[float]:
    return [float(x.strip()) for x in s.replace(";", ",").split(",") if x.strip()]


def load_snapshot(path: Path, central_body: str, flip_z: bool = False) -> Snapshot:
    payload = json.loads(path.read_text(encoding="utf-8"))
    eph = payload.get("ephemerides")
    if not isinstance(eph, dict):
        raise ValueError("Snapshot precisa conter payload['ephemerides'][body]['states'].")

    bc = payload.get("body_catalog", {})
    catalog = bc.get("bodies", bc) if isinstance(bc, dict) else {}

    bodies: Dict[str, BodyState] = {}
    start_ut = float(payload.get("start_ut_seconds", 0.0))
    et_offset = float(payload.get("et_offset_seconds", 0.0))
    reference_body = str(payload.get("reference_body", central_body))

    for name, block in eph.items():
        if not isinstance(block, dict) or not block.get("states"):
            continue
        row = block["states"][0]
        if len(row) < 7:
            continue
        if start_ut == 0.0:
            start_ut = float(row[0])
        x, y, z, vx, vy, vz = map(float, row[1:7])
        if flip_z:
            z = -z
            vz = -vz
        entry = catalog.get(name, {}) if isinstance(catalog, dict) else {}
        mu = entry.get("mu_m3_s2", entry.get("gravitational_parameter"))
        mass = entry.get("mass_kg", entry.get("mass"))
        if mu is None and mass is None:
            raise ValueError(f"{name}: sem mu_m3_s2/mass_kg no body_catalog.")
        mu_f = float(mu) if mu is not None else float(mass) * G_SI
        mass_f = float(mass) if mass is not None else mu_f / G_SI
        radius = entry.get("equatorial_radius_m", entry.get("radius", entry.get("radius_m")))
        bodies[name] = BodyState(name, mu_f, mass_f, (x, y, z, vx, vy, vz), float(radius) if radius is not None else None)

    if central_body not in bodies:
        raise KeyError(f"central-body {central_body!r} ausente no snapshot.")
    return Snapshot(reference_body, start_ut, et_offset, bodies)


def read_ksp_csv(path: Path, flip_z: bool = False) -> Tuple[List[float], Dict[float, Dict[str, State6]]]:
    by_et: Dict[float, Dict[str, State6]] = defaultdict(dict)
    with path.open("r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            body = row.get("body")
            if not body:
                continue
            if row.get("read_error", "") not in ("", "0", "False", "false", "None"):
                continue
            try:
                et = float(row["et_seconds"])
                x, y, z = float(row["x_m"]), float(row["y_m"]), float(row["z_m"])
                vx, vy, vz = float(row["vx_m_s"]), float(row["vy_m_s"]), float(row["vz_m_s"])
                if flip_z:
                    z = -z
                    vz = -vz
            except Exception:
                continue
            by_et[et][body] = (x, y, z, vx, vy, vz)
    return sorted(by_et.keys()), by_et


def axis_vector(axis: str) -> Vec3:
    axis = axis.strip().lower()
    sign = -1.0 if axis.startswith("-") else 1.0
    a = axis[1:] if axis.startswith("-") else axis
    if a == "x":
        return (sign, 0.0, 0.0)
    if a == "y":
        return (0.0, sign, 0.0)
    if a == "z":
        return (0.0, 0.0, sign)
    if ":" in axis:
        vals = [float(v) for v in axis.split(":")]
    elif "/" in axis:
        vals = [float(v) for v in axis.split("/")]
    else:
        vals = [float(v) for v in axis.split(",")]
    if len(vals) != 3:
        raise ValueError(f"Eixo inválido: {axis!r}; use x/y/z/-x/-y/-z ou vx:vy:vz")
    n = norm3(vals)
    if n <= 0:
        raise ValueError("Eixo de spin com norma zero.")
    return (vals[0]/n, vals[1]/n, vals[2]/n)


def build_sim(snapshot: Snapshot, central_body: str, integrator: str, epsilon: float, move_to_com: bool) -> Tuple[Any, List[str], Dict[str, int]]:
    try:
        import rebound  # type: ignore
    except ImportError as exc:
        raise SystemExit("Instale REBOUND: pip install rebound") from exc
    sim = rebound.Simulation()
    sim.G = G_SI
    sim.integrator = integrator
    if integrator.lower() == "ias15":
        try:
            sim.ri_ias15.epsilon = epsilon
        except Exception:
            pass
    ordered = [central_body] + sorted(n for n in snapshot.bodies if n != central_body)
    for name in ordered:
        b = snapshot.bodies[name]
        sim.add(m=b.mass_kg, x=b.state[0], y=b.state[1], z=b.state[2], vx=b.state[3], vy=b.state[4], vz=b.state[5])
    if move_to_com:
        sim.move_to_com()
    return sim, ordered, {name: i for i, name in enumerate(ordered)}


def add_j2(sim: Any, body_index: int, j2: float, r_eq_m: float, omega_axis: Vec3) -> Optional[Any]:
    if j2 == 0.0:
        return None
    try:
        import reboundx  # type: ignore
    except ImportError as exc:
        raise SystemExit("Instale REBOUNDx: pip install reboundx") from exc
    rebx = reboundx.Extras(sim)
    gh = rebx.load_force("gravitational_harmonics")
    rebx.add_force(gh)
    p = sim.particles[body_index]
    p.params["J2"] = float(j2)
    p.params["R_eq"] = float(r_eq_m)
    # A magnitude não deve importar para gravitational_harmonics; usamos norma 1.
    try:
        p.params["Omega"] = tuple(float(x) for x in omega_axis)
    except Exception:
        # Algumas versões aceitam lista melhor que tuple.
        p.params["Omega"] = [float(x) for x in omega_axis]
    return rebx


def get_state_relative_to_central(sim: Any, idx: int, central_idx: int) -> State6:
    p = sim.particles[idx]
    c = sim.particles[central_idx]
    return (p.x-c.x, p.y-c.y, p.z-c.z, p.vx-c.vx, p.vy-c.vy, p.vz-c.vz)


def safe_integrate(sim: Any, t: float) -> None:
    try:
        sim.integrate(float(t), exact_finish_time=1)
    except TypeError:
        sim.integrate(float(t))


def evaluate_run(
    snapshot: Snapshot,
    epochs: List[float],
    ksp_by_et: Dict[float, Dict[str, State6]],
    central_body: str,
    parent: str,
    targets: List[str],
    integrator: str,
    epsilon: float,
    move_to_com: bool,
    j2_body: str,
    j2_value: float,
    j2_radius_m: float,
    omega_axis: Vec3,
) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
    sim, ordered, idx = build_sim(snapshot, central_body, integrator, epsilon, move_to_com)
    central_idx = idx[central_body]
    parent_idx = idx[parent]
    j2_idx = idx[j2_body]
    rebx = add_j2(sim, j2_idx, j2_value, j2_radius_m, omega_axis)
    e0 = None
    try:
        e0 = float(sim.energy())
    except Exception:
        pass

    per_target: Dict[str, Dict[str, List[float] | float | int]] = {
        t: {"abs_pos_m": [], "rel_pos_m": [], "abs_vel_m_s": [], "rel_vel_m_s": [], "n": 0}
        for t in targets
    }
    start_et = snapshot.start_et_s
    wall0 = time.time()

    for et in epochs:
        t_rel = et - start_et
        if t_rel < -1e-9:
            continue
        safe_integrate(sim, t_rel)
        ksp = ksp_by_et.get(et, {})
        if parent not in ksp:
            continue
        reb_parent = get_state_relative_to_central(sim, parent_idx, central_idx)
        ksp_parent = ksp[parent]
        for target in targets:
            if target not in ksp or target not in idx:
                continue
            reb_t = get_state_relative_to_central(sim, idx[target], central_idx)
            ksp_t = ksp[target]
            abs_pos = norm6_pos(ksp_t, reb_t)
            abs_vel = norm6_vel(ksp_t, reb_t)
            rel_ksp = rel_state(ksp_t, ksp_parent)
            rel_reb = rel_state(reb_t, reb_parent)
            rel_pos = norm6_pos(rel_ksp, rel_reb)
            rel_vel = norm6_vel(rel_ksp, rel_reb)
            d = per_target[target]
            d["abs_pos_m"].append(abs_pos)  # type: ignore[index]
            d["abs_vel_m_s"].append(abs_vel)  # type: ignore[index]
            d["rel_pos_m"].append(rel_pos)  # type: ignore[index]
            d["rel_vel_m_s"].append(rel_vel)  # type: ignore[index]
            d["n"] = int(d["n"]) + 1

    e1 = None
    edrift = None
    try:
        e1 = float(sim.energy())
        if e0 not in (None, 0.0):
            edrift = (e1 - e0) / abs(e0)
    except Exception:
        pass

    detail: Dict[str, Dict[str, Any]] = {}
    rel_rms_values = []
    rel_max_values = []
    for target, d in per_target.items():
        abs_pos = d["abs_pos_m"]  # type: ignore[assignment]
        rel_pos = d["rel_pos_m"]  # type: ignore[assignment]
        abs_vel = d["abs_vel_m_s"]  # type: ignore[assignment]
        rel_vel = d["rel_vel_m_s"]  # type: ignore[assignment]
        assert isinstance(abs_pos, list) and isinstance(rel_pos, list)
        assert isinstance(abs_vel, list) and isinstance(rel_vel, list)
        detail[target] = {
            "n": int(d["n"]),
            "abs_max_km": max(abs_pos)/1000.0 if abs_pos else float("nan"),
            "abs_rms_km": rms(abs_pos)/1000.0 if abs_pos else float("nan"),
            "abs_final_km": abs_pos[-1]/1000.0 if abs_pos else float("nan"),
            "rel_max_km": max(rel_pos)/1000.0 if rel_pos else float("nan"),
            "rel_rms_km": rms(rel_pos)/1000.0 if rel_pos else float("nan"),
            "rel_final_km": rel_pos[-1]/1000.0 if rel_pos else float("nan"),
            "rel_max_vel_m_s": max(rel_vel) if rel_vel else float("nan"),
            "abs_max_vel_m_s": max(abs_vel) if abs_vel else float("nan"),
        }
        if rel_pos:
            rel_rms_values.append(rms(rel_pos)/1000.0)
            rel_max_values.append(max(rel_pos)/1000.0)

    aggregate = {
        "j2": j2_value,
        "axis": omega_axis,
        "j2_body": j2_body,
        "j2_radius_m": j2_radius_m,
        "aggregate_rel_rms_km": rms(rel_rms_values) if rel_rms_values else float("nan"),
        "aggregate_rel_max_km": max(rel_max_values) if rel_max_values else float("nan"),
        "median_rel_rms_km": median(rel_rms_values),
        "energy_relative_drift": edrift,
        "wall_seconds": time.time() - wall0,
    }
    # Keep reference to REBOUNDx alive until function exits.
    if rebx is not None:
        aggregate["reboundx_enabled"] = True
    else:
        aggregate["reboundx_enabled"] = False
    return aggregate, detail


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Varredura REBOUNDx J2 contra KSP/Principia CSV já coletado.")
    p.add_argument("--input-json", type=Path, required=True)
    p.add_argument("--ksp-csv", type=Path, required=True)
    p.add_argument("--central-body", required=True)
    p.add_argument("--parent", required=True)
    p.add_argument("--targets", nargs="+", required=True)
    p.add_argument("--j2-body", default=None, help="Default: --parent")
    p.add_argument("--j2-radius-m", type=float, default=None, help="Raio equatorial do corpo oblato. Para Jool stock: 6000000.")
    p.add_argument("--j2-values", default="0,0.0001,0.0003,0.001,0.003,0.01,0.03")
    p.add_argument("--axis-sweep", default="z", help="Ex.: z ou x,y,z ou x,y,z,-x,-y,-z ou eixo custom 0:0:1")
    p.add_argument("--integrator", default="ias15")
    p.add_argument("--ias15-epsilon", type=float, default=1e-11)
    p.add_argument("--no-move-to-com", action="store_true")
    p.add_argument("--flip-z-input", action="store_true")
    p.add_argument("--flip-z-ksp-csv", action="store_true")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--max-epochs", type=int, default=0, help="Debug: limitar epochs; 0 = todos")
    args = p.parse_args(argv)

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    snapshot = load_snapshot(args.input_json, args.central_body, args.flip_z_input)
    epochs, ksp_by_et = read_ksp_csv(args.ksp_csv, args.flip_z_ksp_csv)
    # Safety: keep only forward epochs.
    epochs = [et for et in epochs if et >= snapshot.start_et_s - 1e-9]
    if args.max_epochs and args.max_epochs > 0:
        epochs = epochs[:args.max_epochs]
    if not epochs:
        raise ValueError("Nenhum epoch >= snapshot.start_et_s no KSP CSV.")

    j2_body = args.j2_body or args.parent
    if j2_body not in snapshot.bodies:
        raise KeyError(f"j2-body {j2_body!r} ausente no snapshot.")
    r_eq = args.j2_radius_m or snapshot.bodies[j2_body].radius_m
    if r_eq is None:
        raise ValueError(
            f"Não encontrei raio para {j2_body}. Passe --j2-radius-m explicitamente "
            "(Jool stock: 6000000)."
        )

    j2_values = parse_float_list(args.j2_values)
    axes = [a.strip() for a in args.axis_sweep.split(",") if a.strip()]

    summary_path = out / "j2_sweep_summary.csv"
    detail_path = out / "j2_sweep_detail.csv"
    manifest_path = out / "j2_sweep_manifest.json"

    summary_rows: List[Dict[str, Any]] = []
    detail_rows: List[Dict[str, Any]] = []

    print(f"[J2 SWEEP] epochs={len(epochs)} start_et={snapshot.start_et_s:.6f} first_csv={epochs[0]:.6f} last_csv={epochs[-1]:.6f}")
    print(f"[J2 SWEEP] body={j2_body} R_eq={r_eq} targets={args.targets}")

    for axis_label in axes:
        omega = axis_vector(axis_label)
        for j2 in j2_values:
            label = f"axis={axis_label} J2={j2:.8g}"
            print(f"[RUN] {label}", flush=True)
            agg, detail = evaluate_run(
                snapshot=snapshot,
                epochs=epochs,
                ksp_by_et=ksp_by_et,
                central_body=args.central_body,
                parent=args.parent,
                targets=args.targets,
                integrator=args.integrator,
                epsilon=args.ias15_epsilon,
                move_to_com=not args.no_move_to_com,
                j2_body=j2_body,
                j2_value=j2,
                j2_radius_m=r_eq,
                omega_axis=omega,
            )
            summary_rows.append({
                "axis_label": axis_label,
                "omega_x": omega[0], "omega_y": omega[1], "omega_z": omega[2],
                **agg,
            })
            for target, metrics in detail.items():
                detail_rows.append({
                    "axis_label": axis_label,
                    "j2": j2,
                    "target": target,
                    **metrics,
                })
            print(
                f"      aggregate_rel_rms={agg['aggregate_rel_rms_km']:.6g} km; "
                f"aggregate_rel_max={agg['aggregate_rel_max_km']:.6g} km; "
                f"edrift={agg['energy_relative_drift']}",
                flush=True,
            )

    # Write CSVs.
    if summary_rows:
        fields = list(summary_rows[0].keys())
        with summary_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader(); w.writerows(summary_rows)
    if detail_rows:
        fields = list(detail_rows[0].keys())
        with detail_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader(); w.writerows(detail_rows)

    best = min(summary_rows, key=lambda r: float(r["aggregate_rel_rms_km"])) if summary_rows else None
    baseline = None
    for r in summary_rows:
        if abs(float(r["j2"])) == 0.0:
            baseline = r
            break
    manifest = {
        "schema": "reboundx_j2_sweep_level_a.v1",
        "input_json": str(args.input_json),
        "ksp_csv": str(args.ksp_csv),
        "central_body": args.central_body,
        "parent": args.parent,
        "targets": args.targets,
        "j2_body": j2_body,
        "j2_radius_m": r_eq,
        "epochs": {"n": len(epochs), "first_et_s": epochs[0], "last_et_s": epochs[-1], "snapshot_start_et_s": snapshot.start_et_s},
        "integrator": args.integrator,
        "ias15_epsilon": args.ias15_epsilon,
        "move_to_com": not args.no_move_to_com,
        "best_by_aggregate_rel_rms": best,
        "baseline_first_j2_zero": baseline,
        "outputs": {"summary_csv": str(summary_path), "detail_csv": str(detail_path)},
        "interpretation_note": (
            "This is a hypothesis test. A lower aggregate relative RMS for nonzero J2 indicates "
            "that an oblate-body term may explain part of the residual, but it does not by itself "
            "prove the physical J2 or spin axis used by Principia."
        ),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n=== RESULTADO ===")
    if baseline:
        print(f"Baseline J2=0: aggregate_rel_rms={float(baseline['aggregate_rel_rms_km']):.6g} km")
    if best:
        print(
            f"Melhor: axis={best['axis_label']} J2={float(best['j2']):.8g} "
            f"aggregate_rel_rms={float(best['aggregate_rel_rms_km']):.6g} km "
            f"aggregate_rel_max={float(best['aggregate_rel_max_km']):.6g} km"
        )
    print(f"Summary: {summary_path}")
    print(f"Detail : {detail_path}")
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
