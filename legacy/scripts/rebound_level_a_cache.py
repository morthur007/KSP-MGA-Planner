#!/usr/bin/env python3
"""
rebound_level_a_cache.py

NÍVEL A — REBOUND como fonte canônica dinâmica, sem SPK.

Objetivo
--------
1) Ler um snapshot inicial KSP/Principia, normalmente principia_true_snapshot_v2.json.
2) Construir uma simulação N-body REBOUND em SI (m, kg, s).
3) Propagar uma única vez e amostrar estados diretamente do REBOUND.
4) Gravar um cache canônico de estados em CSV/JSON.
5) Se fornecido states.csv do kRPC/Principia, comparar KSP vs REBOUND direto,
   sem SpiceyPy, sem SPK, sem Chebyshev.

Este script é deliberadamente "pré-SPK". Ele existe para separar erros de:
- dinâmica/modelo físico REBOUND vs Principia;
- frame/epoch/μ/corpos faltantes;
- erro de exportação SPK/interpolação Chebyshev.

Unidades
--------
Entrada e integração: SI puro.
  posição: m
  velocidade: m/s
  massa: kg
  tempo: s
  G = 6.67430e-11 m^3 kg^-1 s^-2

Frame de saída
--------------
Por padrão, todos os estados exportados são relativos ao corpo central:
  r_rel = r_body - r_central
  v_rel = v_body - v_central
Isso casa com a forma como coletamos no kRPC usando
central_body.non_rotating_reference_frame.

Exemplo — comparar direto contra states.csv de 360 dias
-------------------------------------------------------
python rebound_level_a_cache.py \
  --input-json data/true_snapshot_v2.json \
  --central-body Sun \
  --ksp-csv data/opm_mpe_360d/states.csv \
  --integrator ias15 \
  --ias15-epsilon 1e-11 \
  --output-dir data/level_a_rebound_vs_ksp_1y \
  --write-residual-samples

Exemplo — gerar cache uniforme de 105 anos, sem KSP CSV
-------------------------------------------------------
python rebound_level_a_cache.py \
  --input-json data/true_snapshot_v2.json \
  --central-body Sun \
  --duration-years 105 \
  --sample-step-days 1 \
  --output-dir data/level_a_rebound_105y
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, DefaultDict, Dict, Iterable, List, Optional, Sequence, Tuple

G_SI = 6.67430e-11
DAY_S = 86400.0
JULIAN_YEAR_S = 365.25 * DAY_S


@dataclass(frozen=True)
class BodyInitialState:
    name: str
    mu_m3_s2: float
    mass_kg: float
    x_m: float
    y_m: float
    z_m: float
    vx_m_s: float
    vy_m_s: float
    vz_m_s: float


@dataclass(frozen=True)
class InputSnapshot:
    reference_body: str
    start_ut_s: float
    et_offset_seconds: float
    frame_convention: str
    bodies: Dict[str, BodyInitialState]


@dataclass(frozen=True)
class RuntimeConfig:
    central_body: str
    integrator: str
    ias15_epsilon: float
    whfast_dt_seconds: Optional[float]
    move_to_com: bool
    output_frame: str


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def norm3(v: Sequence[float]) -> float:
    return math.sqrt(float(v[0]) ** 2 + float(v[1]) ** 2 + float(v[2]) ** 2)


def dot3(a: Sequence[float], b: Sequence[float]) -> float:
    return float(a[0]) * float(b[0]) + float(a[1]) * float(b[1]) + float(a[2]) * float(b[2])


def sub3(a: Sequence[float], b: Sequence[float]) -> Tuple[float, float, float]:
    return float(a[0]) - float(b[0]), float(a[1]) - float(b[1]), float(a[2]) - float(b[2])


def cross3(a: Sequence[float], b: Sequence[float]) -> Tuple[float, float, float]:
    return (
        float(a[1]) * float(b[2]) - float(a[2]) * float(b[1]),
        float(a[2]) * float(b[0]) - float(a[0]) * float(b[2]),
        float(a[0]) * float(b[1]) - float(a[1]) * float(b[0]),
    )


def mul3(a: Sequence[float], s: float) -> Tuple[float, float, float]:
    return float(a[0]) * s, float(a[1]) * s, float(a[2]) * s


def unit3(a: Sequence[float]) -> Optional[Tuple[float, float, float]]:
    n = norm3(a)
    if not math.isfinite(n) or n <= 0.0:
        return None
    return float(a[0]) / n, float(a[1]) / n, float(a[2]) / n


def rtn_components(
    residual_r_m: Sequence[float],
    r_ref_m: Sequence[float],
    v_ref_m_s: Sequence[float],
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Return radial, transverse, normal components of residual position.

    Radial uses r direction. Normal uses angular momentum r × v. Transverse is
    n × r, prograde if r/v define a normal frame. Returns None components when
    the frame is degenerate.
    """
    er = unit3(r_ref_m)
    h = cross3(r_ref_m, v_ref_m_s)
    en = unit3(h)
    if er is None or en is None:
        return None, None, None
    et = cross3(en, er)
    return dot3(residual_r_m, er), dot3(residual_r_m, et), dot3(residual_r_m, en)


def apparent_epoch_offset_s(residual_r_m: Sequence[float], v_ref_m_s: Sequence[float]) -> Optional[float]:
    vv = dot3(v_ref_m_s, v_ref_m_s)
    if vv <= 0.0 or not math.isfinite(vv):
        return None
    # If residual = KSP - REBOUND ≈ v * dt, positive dt means KSP appears ahead
    # of the REBOUND state along the local velocity direction.
    return dot3(residual_r_m, v_ref_m_s) / vv


def median_or_none(values: Sequence[float]) -> Optional[float]:
    vals = [v for v in values if math.isfinite(v)]
    if not vals:
        return None
    return float(statistics.median(vals))


def rms(values: Sequence[float]) -> Optional[float]:
    vals = [v for v in values if math.isfinite(v)]
    if not vals:
        return None
    return math.sqrt(sum(v * v for v in vals) / len(vals))


def find_catalog_bodies(catalog_obj: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    if "bodies" in catalog_obj and isinstance(catalog_obj["bodies"], dict):
        return catalog_obj["bodies"]
    if "body_catalog" in catalog_obj:
        bc = catalog_obj["body_catalog"]
        if isinstance(bc, dict) and "bodies" in bc and isinstance(bc["bodies"], dict):
            return bc["bodies"]
    if all(isinstance(v, dict) for v in catalog_obj.values()):
        return catalog_obj  # type: ignore[return-value]
    raise ValueError("Não consegui localizar body_catalog.bodies no JSON/catálogo físico.")


def mu_and_mass(name: str, catalog_bodies: Dict[str, Dict[str, Any]]) -> Tuple[float, float]:
    if name not in catalog_bodies:
        raise KeyError(f"Corpo {name!r} ausente no catálogo físico.")
    entry = catalog_bodies[name]
    mu = entry.get("mu_m3_s2", entry.get("gravitational_parameter"))
    mass = entry.get("mass_kg", entry.get("mass"))
    if mu is None and mass is None:
        raise ValueError(f"Corpo {name!r} sem mu_m3_s2 e sem mass_kg.")
    if mu is None:
        mu = float(mass) * G_SI
    if mass is None:
        mass = float(mu) / G_SI
    return float(mu), float(mass)


def maybe_flip_z(state6: Sequence[Any], flip_z: bool) -> Tuple[float, float, float, float, float, float]:
    x, y, z, vx, vy, vz = map(float, state6)
    if flip_z:
        return x, y, -z, vx, vy, -vz
    return x, y, z, vx, vy, vz


def load_snapshot_json(
    input_json: Path,
    body_catalog_path: Optional[Path],
    central_body: Optional[str],
    initial_sample_index: int,
    et_offset_seconds: Optional[float],
    flip_z_input: bool,
) -> InputSnapshot:
    payload = load_json(input_json)
    if body_catalog_path is not None:
        catalog_obj = load_json(body_catalog_path)
    elif "body_catalog" in payload:
        catalog_obj = payload["body_catalog"]
    else:
        raise ValueError("Passe --body-catalog ou inclua body_catalog no JSON.")

    catalog_bodies = find_catalog_bodies(catalog_obj)
    reference_body = central_body or payload.get("reference_body") or payload.get("reference_body_name")
    if not reference_body:
        raise ValueError("Informe --central-body ou inclua reference_body no JSON.")

    eph = payload.get("ephemerides")
    if not isinstance(eph, dict):
        raise ValueError("Formato esperado: payload['ephemerides'][body]['states'].")

    bodies: Dict[str, BodyInitialState] = {}
    start_ut_s: Optional[float] = payload.get("start_ut_seconds")

    for name, block in eph.items():
        states = block.get("states") if isinstance(block, dict) else None
        if not states:
            continue
        if initial_sample_index >= len(states):
            raise IndexError(f"{name}: initial_sample_index={initial_sample_index} fora do vetor de estados.")
        row = states[initial_sample_index]
        if len(row) < 7:
            raise ValueError(f"{name}: estado precisa ser [ut,x,y,z,vx,vy,vz].")
        if start_ut_s is None:
            start_ut_s = float(row[0])
        x, y, z, vx, vy, vz = maybe_flip_z(row[1:7], flip_z_input)
        mu, mass = mu_and_mass(name, catalog_bodies)
        bodies[name] = BodyInitialState(name, mu, mass, x, y, z, vx, vy, vz)

    if reference_body not in bodies:
        mu, mass = mu_and_mass(reference_body, catalog_bodies)
        bodies[reference_body] = BodyInitialState(reference_body, mu, mass, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    if start_ut_s is None:
        raise ValueError("Nenhum estado inicial encontrado no JSON.")

    offset = float(et_offset_seconds if et_offset_seconds is not None else payload.get("et_offset_seconds", 0.0))
    if flip_z_input:
        frame_convention = "input converted by z -> -z; output right-handed relative central frame"
    else:
        frame_convention = "as stored in input JSON; output relative central frame"

    return InputSnapshot(
        reference_body=str(reference_body),
        start_ut_s=float(start_ut_s),
        et_offset_seconds=offset,
        frame_convention=frame_convention,
        bodies=bodies,
    )


def build_rebound_simulation(snapshot: InputSnapshot, cfg: RuntimeConfig):
    try:
        import rebound  # type: ignore
    except ImportError as exc:
        raise SystemExit("Instale REBOUND no ambiente: pip install rebound") from exc

    if cfg.central_body not in snapshot.bodies:
        raise KeyError(f"central_body {cfg.central_body!r} não existe no snapshot.")

    sim = rebound.Simulation()
    sim.G = G_SI
    sim.integrator = cfg.integrator

    if cfg.integrator.lower() == "ias15":
        try:
            sim.ri_ias15.epsilon = cfg.ias15_epsilon
        except Exception:
            pass
    elif cfg.integrator.lower() == "whfast":
        if cfg.whfast_dt_seconds is None:
            raise ValueError("WHFast exige --whfast-dt-seconds.")
        sim.dt = float(cfg.whfast_dt_seconds)
    elif cfg.whfast_dt_seconds is not None:
        sim.dt = float(cfg.whfast_dt_seconds)

    ordered_names = [cfg.central_body] + sorted(n for n in snapshot.bodies if n != cfg.central_body)
    for name in ordered_names:
        b = snapshot.bodies[name]
        sim.add(m=b.mass_kg, x=b.x_m, y=b.y_m, z=b.z_m, vx=b.vx_m_s, vy=b.vy_m_s, vz=b.vz_m_s)

    if cfg.move_to_com:
        sim.move_to_com()

    return sim, ordered_names


def relative_state_m(sim: Any, body_index: int, center_index: int) -> Tuple[float, float, float, float, float, float]:
    p = sim.particles[body_index]
    c = sim.particles[center_index]
    return (
        p.x - c.x,
        p.y - c.y,
        p.z - c.z,
        p.vx - c.vx,
        p.vy - c.vy,
        p.vz - c.vz,
    )


def absolute_state_m(sim: Any, body_index: int) -> Tuple[float, float, float, float, float, float]:
    p = sim.particles[body_index]
    return (p.x, p.y, p.z, p.vx, p.vy, p.vz)


def read_ksp_csv(ksp_csv: Path, flip_z: bool = False) -> Tuple[Dict[float, List[Dict[str, Any]]], List[str], Dict[str, Any]]:
    """Load KSP states grouped by et_seconds.

    We intentionally keep the original et_seconds values. They are the exact
    epochs at which REBOUND will be sampled.
    """
    by_et: DefaultDict[float, List[Dict[str, Any]]] = defaultdict(list)
    body_names = set()
    rows_total = 0
    rows_used = 0
    min_et = None
    max_et = None

    with ksp_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows_total += 1
            body = row.get("body", "")
            if not body:
                continue
            read_error = row.get("read_error", "")
            if read_error and read_error not in {"0", "False", "false", "None", ""}:
                continue
            try:
                et = float(row["et_seconds"])
                x, y, z, vx, vy, vz = maybe_flip_z(
                    [row["x_m"], row["y_m"], row["z_m"], row["vx_m_s"], row["vy_m_s"], row["vz_m_s"]],
                    flip_z,
                )
            except Exception:
                continue
            row2 = dict(row)
            row2["body"] = body
            row2["et_seconds_float"] = et
            row2["state_m"] = (x, y, z, vx, vy, vz)
            by_et[et].append(row2)
            body_names.add(body)
            rows_used += 1
            min_et = et if min_et is None else min(min_et, et)
            max_et = et if max_et is None else max(max_et, et)

    meta = {
        "rows_total": rows_total,
        "rows_used": rows_used,
        "unique_epochs": len(by_et),
        "min_et": min_et,
        "max_et": max_et,
    }
    return dict(by_et), sorted(body_names), meta


def uniform_epochs(start_et: float, duration_years: float, sample_step_days: float) -> List[float]:
    if duration_years <= 0:
        raise ValueError("--duration-years precisa ser positivo.")
    if sample_step_days <= 0:
        raise ValueError("--sample-step-days precisa ser positivo.")
    duration_s = duration_years * JULIAN_YEAR_S
    step_s = sample_step_days * DAY_S
    n = int(math.floor(duration_s / step_s))
    epochs = [start_et + i * step_s for i in range(n + 1)]
    if epochs[-1] < start_et + duration_s:
        epochs.append(start_et + duration_s)
    return epochs


class ResidualAccumulator:
    def __init__(self) -> None:
        self.n = 0
        self.pos_err_m: List[float] = []
        self.vel_err_m_s: List[float] = []
        self.radial_m: List[float] = []
        self.transverse_m: List[float] = []
        self.normal_m: List[float] = []
        self.apparent_dt_s: List[float] = []
        self.first_et: Optional[float] = None
        self.last_et: Optional[float] = None
        self.final_pos_err_m: Optional[float] = None
        self.final_vel_err_m_s: Optional[float] = None

    def add(
        self,
        et: float,
        pos_err_m: float,
        vel_err_m_s: float,
        rtn: Tuple[Optional[float], Optional[float], Optional[float]],
        apparent_dt: Optional[float],
    ) -> None:
        self.n += 1
        self.pos_err_m.append(pos_err_m)
        self.vel_err_m_s.append(vel_err_m_s)
        r, t, n = rtn
        if r is not None:
            self.radial_m.append(r)
        if t is not None:
            self.transverse_m.append(t)
        if n is not None:
            self.normal_m.append(n)
        if apparent_dt is not None and math.isfinite(apparent_dt):
            self.apparent_dt_s.append(apparent_dt)
        if self.first_et is None:
            self.first_et = et
        self.last_et = et
        self.final_pos_err_m = pos_err_m
        self.final_vel_err_m_s = vel_err_m_s

    def summary(self) -> Dict[str, Any]:
        pos_max = max(self.pos_err_m) if self.pos_err_m else None
        vel_max = max(self.vel_err_m_s) if self.vel_err_m_s else None
        return {
            "samples": self.n,
            "first_et_s": self.first_et,
            "last_et_s": self.last_et,
            "max_pos_err_m": pos_max,
            "rms_pos_err_m": rms(self.pos_err_m),
            "final_pos_err_m": self.final_pos_err_m,
            "max_vel_err_m_s": vel_max,
            "rms_vel_err_m_s": rms(self.vel_err_m_s),
            "final_vel_err_m_s": self.final_vel_err_m_s,
            "median_radial_m": median_or_none(self.radial_m),
            "median_transverse_m": median_or_none(self.transverse_m),
            "median_normal_m": median_or_none(self.normal_m),
            "median_apparent_epoch_offset_s": median_or_none(self.apparent_dt_s),
        }


def write_cache_header(writer: csv.writer) -> None:
    writer.writerow([
        "et_seconds",
        "t_rebound_seconds",
        "body",
        "x_m",
        "y_m",
        "z_m",
        "vx_m_s",
        "vy_m_s",
        "vz_m_s",
    ])


def write_residual_header(writer: csv.writer) -> None:
    writer.writerow([
        "et_seconds",
        "body",
        "pos_err_m",
        "vel_err_m_s",
        "radial_m",
        "transverse_m",
        "normal_m",
        "apparent_epoch_offset_s",
        "ksp_x_m",
        "ksp_y_m",
        "ksp_z_m",
        "reb_x_m",
        "reb_y_m",
        "reb_z_m",
        "ksp_vx_m_s",
        "ksp_vy_m_s",
        "ksp_vz_m_s",
        "reb_vx_m_s",
        "reb_vy_m_s",
        "reb_vz_m_s",
    ])


def safe_integrate(sim: Any, t_rel_s: float) -> None:
    try:
        sim.integrate(float(t_rel_s), exact_finish_time=1)
    except TypeError:
        sim.integrate(float(t_rel_s))


def run_level_a(args: argparse.Namespace) -> Dict[str, Any]:
    snapshot = load_snapshot_json(
        input_json=args.input_json,
        body_catalog_path=args.body_catalog,
        central_body=args.central_body,
        initial_sample_index=args.initial_sample_index,
        et_offset_seconds=args.et_offset_seconds,
        flip_z_input=args.flip_z_input,
    )
    cfg = RuntimeConfig(
        central_body=args.central_body,
        integrator=args.integrator,
        ias15_epsilon=args.ias15_epsilon,
        whfast_dt_seconds=args.whfast_dt_seconds,
        move_to_com=not args.no_move_to_com,
        output_frame="central_relative" if not args.absolute_output else "absolute_rebound_frame",
    )

    sim, ordered_names = build_rebound_simulation(snapshot, cfg)
    name_to_index = {name: i for i, name in enumerate(ordered_names)}
    center_index = name_to_index[cfg.central_body]
    start_et = snapshot.start_ut_s + snapshot.et_offset_seconds

    ksp_by_et: Dict[float, List[Dict[str, Any]]] = {}
    ksp_meta: Dict[str, Any] = {}
    if args.ksp_csv:
        ksp_by_et, _, ksp_meta = read_ksp_csv(args.ksp_csv, flip_z=args.flip_z_ksp_csv)
        epochs = sorted(ksp_by_et.keys())
    else:
        epochs = uniform_epochs(start_et, args.duration_years, args.sample_step_days)

    if not epochs:
        raise ValueError("Nenhum epoch de amostragem foi gerado.")



    outdir = args.output_dir
    outdir.mkdir(parents=True, exist_ok=True)
    cache_csv = outdir / "rebound_states.csv"
    residual_samples_csv = outdir / "residual_samples.csv"
    residual_summary_csv = outdir / "residuals_by_body.csv"
    residual_summary_json = outdir / "residuals_by_body.json"
    energy_csv = outdir / "energy.csv"
    manifest_json = outdir / "manifest.json"

    initial_energy = None
    try:
        initial_energy = float(sim.energy())
    except Exception:
        pass

    acc: DefaultDict[str, ResidualAccumulator] = defaultdict(ResidualAccumulator)
    missing_bodies = set()
    wall0 = time.time()

    with cache_csv.open("w", newline="", encoding="utf-8") as f_cache, \
         energy_csv.open("w", newline="", encoding="utf-8") as f_energy:
        cache_writer = csv.writer(f_cache)
        energy_writer = csv.writer(f_energy)
        write_cache_header(cache_writer)
        energy_writer.writerow(["et_seconds", "t_rebound_seconds", "energy"])

        residual_writer = None
        f_res = None
        if args.write_residual_samples and args.ksp_csv:
            f_res = residual_samples_csv.open("w", newline="", encoding="utf-8")
            residual_writer = csv.writer(f_res)
            write_residual_header(residual_writer)

        try:
            for idx, et in enumerate(epochs):
                t_rel = float(et - start_et)
                safe_integrate(sim, t_rel)

                # Cache all REBOUND states at this epoch.
                states_at_epoch: Dict[str, Tuple[float, float, float, float, float, float]] = {}
                for name in ordered_names:
                    if cfg.output_frame == "central_relative":
                        st = relative_state_m(sim, name_to_index[name], center_index)
                    else:
                        st = absolute_state_m(sim, name_to_index[name])
                    states_at_epoch[name] = st
                    cache_writer.writerow([et, t_rel, name, *st])

                if idx % max(1, args.energy_every_epochs) == 0 or idx == len(epochs) - 1:
                    try:
                        energy_writer.writerow([et, t_rel, float(sim.energy())])
                    except Exception:
                        energy_writer.writerow([et, t_rel, ""])

                if args.ksp_csv:
                    for row in ksp_by_et.get(et, []):
                        body = row["body"]
                        if body not in states_at_epoch:
                            missing_bodies.add(body)
                            continue
                        ksp_state = row["state_m"]
                        reb_state = states_at_epoch[body]
                        dr = sub3(ksp_state[0:3], reb_state[0:3])
                        dv = sub3(ksp_state[3:6], reb_state[3:6])
                        pos_err = norm3(dr)
                        vel_err = norm3(dv)
                        rtn = rtn_components(dr, reb_state[0:3], reb_state[3:6])
                        dt_app = apparent_epoch_offset_s(dr, reb_state[3:6])
                        acc[body].add(et, pos_err, vel_err, rtn, dt_app)

                        if residual_writer is not None:
                            r, tr, n = rtn
                            residual_writer.writerow([
                                et,
                                body,
                                pos_err,
                                vel_err,
                                "" if r is None else r,
                                "" if tr is None else tr,
                                "" if n is None else n,
                                "" if dt_app is None else dt_app,
                                *ksp_state[0:3],
                                *reb_state[0:3],
                                *ksp_state[3:6],
                                *reb_state[3:6],
                            ])

                if args.report_every_epochs > 0 and ((idx + 1) % args.report_every_epochs == 0 or idx == len(epochs) - 1):
                    elapsed = time.time() - wall0
                    print(
                        f"[LEVEL A] epoch {idx+1}/{len(epochs)}; "
                        f"t={t_rel / JULIAN_YEAR_S:.6f} yr; wall={elapsed:.1f}s",
                        flush=True,
                    )
        finally:
            if f_res is not None:
                f_res.close()

    final_energy = None
    rel_energy_drift = None
    try:
        final_energy = float(sim.energy())
        if initial_energy not in (None, 0.0):
            rel_energy_drift = (final_energy - initial_energy) / abs(initial_energy)
    except Exception:
        pass

    residual_summary = {body: accumulator.summary() for body, accumulator in sorted(acc.items())}

    if args.ksp_csv:
        with residual_summary_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                "body",
                "samples",
                "max_pos_err_km",
                "rms_pos_err_km",
                "final_pos_err_km",
                "max_vel_err_m_s",
                "rms_vel_err_m_s",
                "final_vel_err_m_s",
                "median_radial_km",
                "median_transverse_km",
                "median_normal_km",
                "median_apparent_epoch_offset_s",
            ])
            for body, s in sorted(residual_summary.items(), key=lambda kv: (-(kv[1].get("max_pos_err_m") or 0.0), kv[0])):
                def km(x: Optional[float]) -> str:
                    return "" if x is None else f"{x / 1000.0:.12g}"
                w.writerow([
                    body,
                    s["samples"],
                    km(s["max_pos_err_m"]),
                    km(s["rms_pos_err_m"]),
                    km(s["final_pos_err_m"]),
                    "" if s["max_vel_err_m_s"] is None else f"{s['max_vel_err_m_s']:.12g}",
                    "" if s["rms_vel_err_m_s"] is None else f"{s['rms_vel_err_m_s']:.12g}",
                    "" if s["final_vel_err_m_s"] is None else f"{s['final_vel_err_m_s']:.12g}",
                    km(s["median_radial_m"]),
                    km(s["median_transverse_m"]),
                    km(s["median_normal_m"]),
                    "" if s["median_apparent_epoch_offset_s"] is None else f"{s['median_apparent_epoch_offset_s']:.12g}",
                ])
        residual_summary_json.write_text(json.dumps(residual_summary, indent=2, ensure_ascii=False), encoding="utf-8")

    manifest = {
        "schema": "rebound_level_a_cache.v1",
        "purpose": "canonical REBOUND direct propagation, no SPK, no Chebyshev",
        "input_json": str(args.input_json),
        "body_catalog": str(args.body_catalog) if args.body_catalog else None,
        "ksp_csv": str(args.ksp_csv) if args.ksp_csv else None,
        "start_ut_s": snapshot.start_ut_s,
        "et_offset_seconds": snapshot.et_offset_seconds,
        "start_et_s": start_et,
        "reference_body_from_snapshot": snapshot.reference_body,
        "central_body": cfg.central_body,
        "frame_convention": snapshot.frame_convention,
        "output_frame": cfg.output_frame,
        "flip_z_input": bool(args.flip_z_input),
        "flip_z_ksp_csv": bool(args.flip_z_ksp_csv),
        "integrator": cfg.integrator,
        "ias15_epsilon": cfg.ias15_epsilon,
        "whfast_dt_seconds": cfg.whfast_dt_seconds,
        "move_to_com": cfg.move_to_com,
        "n_bodies": len(ordered_names),
        "ordered_names": ordered_names,
        "epochs": {
            "n": len(epochs),
            "first_et_s": epochs[0],
            "last_et_s": epochs[-1],
            "source": "ksp_csv" if args.ksp_csv else "uniform",
        },
        "ksp_csv_meta": ksp_meta,
        "outputs": {
            "rebound_states_csv": str(cache_csv),
            "energy_csv": str(energy_csv),
            "residual_samples_csv": str(residual_samples_csv) if args.write_residual_samples and args.ksp_csv else None,
            "residuals_by_body_csv": str(residual_summary_csv) if args.ksp_csv else None,
            "residuals_by_body_json": str(residual_summary_json) if args.ksp_csv else None,
        },
        "energy": {
            "initial": initial_energy,
            "final": final_energy,
            "relative_drift": rel_energy_drift,
        },
        "missing_bodies_in_rebound": sorted(missing_bodies),
        "wall_seconds": time.time() - wall0,
    }
    manifest_json.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    # Console summary.
    print(f"[OK] Cache REBOUND direto: {cache_csv}")
    print(f"[OK] Manifest: {manifest_json}")
    if rel_energy_drift is not None:
        print(f"[CHECK] relative energy drift: {rel_energy_drift:.3e}")

    if args.ksp_csv:
        print("\nResiduals diretos: KSP/Principia CSV vs REBOUND, sem SPK")
        print(f"{'Corpo':<16} | {'N':>5} | {'Max km':>12} | {'RMS km':>12} | {'Final km':>12} | {'Max m/s':>10} | {'med dt s':>10}")
        print("-" * 93)
        for body, s in sorted(residual_summary.items(), key=lambda kv: (-(kv[1].get("max_pos_err_m") or 0.0), kv[0])):
            max_km = (s["max_pos_err_m"] or 0.0) / 1000.0
            rms_km = (s["rms_pos_err_m"] or 0.0) / 1000.0
            final_km = (s["final_pos_err_m"] or 0.0) / 1000.0
            max_vel = s["max_vel_err_m_s"] or 0.0
            med_dt = s["median_apparent_epoch_offset_s"]
            med_dt_str = "" if med_dt is None else f"{med_dt:10.3f}"
            print(f"{body:<16} | {s['samples']:5d} | {max_km:12.3f} | {rms_km:12.3f} | {final_km:12.3f} | {max_vel:10.3f} | {med_dt_str}")
        if missing_bodies:
            print(f"\n[WARN] Corpos no KSP CSV ausentes no REBOUND: {sorted(missing_bodies)}")
        print(f"\n[OK] Residual summary CSV: {residual_summary_csv}")
        if args.write_residual_samples:
            print(f"[OK] Residual samples CSV: {residual_samples_csv}")

    return manifest


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Nível A: cache canônico REBOUND direto, sem SPK.")
    p.add_argument("--input-json", type=Path, required=True, help="Snapshot JSON, ex: data/true_snapshot_v2.json")
    p.add_argument("--body-catalog", type=Path, default=None, help="Catálogo físico externo, se não embutido no JSON.")
    p.add_argument("--central-body", required=True, help="Corpo central do frame de comparação, ex: Sun.")
    p.add_argument("--initial-sample-index", type=int, default=0)
    p.add_argument("--et-offset-seconds", type=float, default=None)
    p.add_argument("--flip-z-input", action="store_true", help="Aplicar z->-z no snapshot de entrada.")
    p.add_argument("--flip-z-ksp-csv", action="store_true", help="Aplicar z->-z também no states.csv de comparação.")

    p.add_argument("--ksp-csv", type=Path, default=None, help="states.csv do kRPC/Principia. Se fornecido, epochs vêm dele.")
    p.add_argument("--duration-years", type=float, default=1.0, help="Usado apenas se --ksp-csv não for fornecido.")
    p.add_argument("--sample-step-days", type=float, default=1.0, help="Usado apenas se --ksp-csv não for fornecido.")

    p.add_argument("--integrator", default="ias15", choices=["ias15", "whfast", "mercurius", "trace", "leapfrog", "sei", "saba"])
    p.add_argument("--ias15-epsilon", type=float, default=1e-11)
    p.add_argument("--whfast-dt-seconds", type=float, default=None)
    p.add_argument("--no-move-to-com", action="store_true", help="Não chamar sim.move_to_com(). Para debug de frame apenas.")
    p.add_argument("--absolute-output", action="store_true", help="Exportar estado absoluto REBOUND em vez de relativo ao central.")

    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--write-residual-samples", action="store_true", help="Grava uma linha de residual por corpo/epoch.")
    p.add_argument("--energy-every-epochs", type=int, default=50)
    p.add_argument("--report-every-epochs", type=int, default=100)
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    run_level_a(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
