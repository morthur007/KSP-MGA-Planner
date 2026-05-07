#!/usr/bin/env python3
"""
rebound_ephemeris_to_spk_type3.py

Stage after kRPC/Principia acquisition:
1) Read a KSP/kRPC ephemeris snapshot JSON or an acquisition directory.
2) Build a SI-unit REBOUND N-body simulation from the observed initial state.
3) Propagate offline for a requested horizon, e.g. 105 years.
4) Fit piecewise Chebyshev position+velocity records.
5) Write a SPICE SPK Type 3 kernel with one segment per body.

Design intent
-------------
- kRPC/Principia is used only to calibrate the initial state and physical catalog.
- REBOUND is the offline N-body "iron judge".
- SPK Type 3 is used as the optimizer-facing ephemeris product.

Input requirements
------------------
A physically valid N-body integration requires, for every massive body:
- position [m]
- velocity [m/s]
- gravitational parameter mu [m^3/s^2] or mass [kg]

The original ksp_ephemerides.json contains states only; therefore pass a
body_catalog.json produced by principia_ephemeris_acquirer.py, or embed body
metadata in the JSON under {"body_catalog": {"bodies": ...}}.

Examples
--------
From a directory produced by principia_ephemeris_acquirer.py:

  python rebound_ephemeris_to_spk_type3.py \
    --acquisition-dir data/opm_mpe_principia_run01 \
    --central-body Kerbol \
    --duration-years 105 \
    --record-span-days 32 \
    --cheby-degree 15 \
    --output-spk data/opm_mpe_105y.bsp

From the older JSON plus a body catalog:

  python rebound_ephemeris_to_spk_type3.py \
    --input-json ksp_ephemerides.json \
    --body-catalog body_catalog.json \
    --central-body Kerbol \
    --flip-z-input \
    --duration-years 105 \
    --output-spk opm_mpe_105y.bsp

Important SPICE convention
--------------------------
SPICE SPK states are conventionally kilometers and kilometers/second. This
script integrates in SI units inside REBOUND, then writes km and km/s into the
SPK Type 3 Chebyshev records.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from numpy.polynomial.chebyshev import chebfit, chebval

G_SI = 6.67430e-11
DAY_S = 86400.0
JULIAN_YEAR_S = 365.25 * DAY_S
M_TO_KM = 1.0e-3


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
class InputEphemeris:
    reference_body: str
    start_ut_s: float
    et_offset_seconds: float
    frame_convention: str
    bodies: Dict[str, BodyInitialState]


@dataclass(frozen=True)
class ExportConfig:
    duration_years: float
    record_span_days: float
    cheby_degree: int
    samples_per_record: int
    integrator: str
    whfast_dt_seconds: Optional[float]
    ias15_epsilon: float
    central_body: str
    frame_name: str
    center_naif_code: int
    target_naif_code_base: int
    archive_path: Optional[str]
    archive_every_records: int


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def find_catalog_bodies(catalog_obj: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Return {body_name: metadata} from supported catalog shapes."""
    if "bodies" in catalog_obj and isinstance(catalog_obj["bodies"], dict):
        return catalog_obj["bodies"]
    if "body_catalog" in catalog_obj:
        bc = catalog_obj["body_catalog"]
        if isinstance(bc, dict) and "bodies" in bc and isinstance(bc["bodies"], dict):
            return bc["bodies"]
    # Last resort: assume the object itself is {name: metadata}
    if all(isinstance(v, dict) for v in catalog_obj.values()):
        return catalog_obj  # type: ignore[return-value]
    raise ValueError("Não consegui localizar um dicionário de corpos no catálogo físico.")


def mu_and_mass(name: str, catalog_bodies: Dict[str, Dict[str, Any]]) -> Tuple[float, float]:
    if name not in catalog_bodies:
        raise KeyError(
            f"Corpo {name!r} não está no body_catalog. "
            "REBOUND precisa de mu_m3_s2 ou mass_kg para integrar N-body."
        )
    entry = catalog_bodies[name]
    mu = entry.get("mu_m3_s2")
    if mu is None:
        mu = entry.get("gravitational_parameter")
    mass = entry.get("mass_kg")
    if mass is None:
        mass = entry.get("mass")

    if mu is None and mass is None:
        raise ValueError(f"Corpo {name!r} sem mu_m3_s2 e sem mass_kg no catálogo.")
    if mu is None:
        mu = float(mass) * G_SI
    if mass is None:
        mass = float(mu) / G_SI
    return float(mu), float(mass)


def maybe_flip_z(state: Sequence[float], flip_z: bool) -> Tuple[float, float, float, float, float, float]:
    x, y, z, vx, vy, vz = map(float, state)
    if flip_z:
        return x, y, -z, vx, vy, -vz
    return x, y, z, vx, vy, vz


def load_legacy_json(
    input_json: Path,
    body_catalog_path: Optional[Path],
    central_body: Optional[str],
    initial_sample_index: int,
    et_offset_seconds: Optional[float],
    flip_z_input: bool,
) -> InputEphemeris:
    payload = load_json(input_json)
    catalog_obj: Dict[str, Any]
    if body_catalog_path is not None:
        catalog_obj = load_json(body_catalog_path)
    elif "body_catalog" in payload:
        catalog_obj = payload["body_catalog"]
    else:
        raise ValueError(
            "O JSON legado contém estados, mas não contém catálogo físico. "
            "Passe --body-catalog body_catalog.json."
        )

    catalog_bodies = find_catalog_bodies(catalog_obj)
    reference_body = central_body or payload.get("reference_body") or payload.get("reference_body_name")
    if not reference_body:
        raise ValueError("Informe --central-body ou inclua reference_body no JSON.")

    eph = payload.get("ephemerides")
    if not isinstance(eph, dict):
        raise ValueError("Formato legado esperado: payload['ephemerides'][body]['states'].")

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
        ut = float(row[0])
        if start_ut_s is None:
            start_ut_s = ut
        x, y, z, vx, vy, vz = maybe_flip_z(row[1:7], flip_z_input)
        mu, mass = mu_and_mass(name, catalog_bodies)
        bodies[name] = BodyInitialState(name, mu, mass, x, y, z, vx, vy, vz)

    if reference_body not in bodies:
        # The acquisition may have excluded the reference body. Add it at the
        # origin of its own non-rotating frame, with zero relative velocity.
        mu, mass = mu_and_mass(reference_body, catalog_bodies)
        bodies[reference_body] = BodyInitialState(reference_body, mu, mass, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    if start_ut_s is None:
        raise ValueError("Não encontrei nenhum estado inicial no JSON.")

    offset = float(et_offset_seconds if et_offset_seconds is not None else payload.get("et_offset_seconds", 0.0))
    frame_convention = "right_handed_after_input_z_flip" if flip_z_input else "as_stored_in_input_json"
    return InputEphemeris(reference_body, float(start_ut_s), offset, frame_convention, bodies)


def load_acquisition_dir(
    acquisition_dir: Path,
    central_body: Optional[str],
    initial_sample_index: int,
    et_offset_seconds: Optional[float],
    flip_z_input: bool,
) -> InputEphemeris:
    manifest = load_json(acquisition_dir / "manifest.json")
    catalog = load_json(acquisition_dir / "body_catalog.json")
    catalog_bodies = find_catalog_bodies(catalog)
    states_csv = acquisition_dir / "states.csv"
    if not states_csv.exists():
        raise FileNotFoundError(states_csv)

    reference_body = central_body or manifest.get("reference_frame", {}).get("reference_body") or manifest.get("reference_body")
    if not reference_body:
        raise ValueError("Não consegui inferir o corpo central; use --central-body.")

    bodies: Dict[str, BodyInitialState] = {}
    start_ut_s: Optional[float] = None

    with states_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if int(row["sample_index"]) != initial_sample_index:
                continue
            name = row["body"]
            target_ut = float(row.get("target_ut_s") or row.get("actual_ut_s"))
            if start_ut_s is None:
                start_ut_s = target_ut
            x, y, z, vx, vy, vz = maybe_flip_z(
                [row["x_m"], row["y_m"], row["z_m"], row["vx_m_s"], row["vy_m_s"], row["vz_m_s"]],
                flip_z_input,
            )
            mu, mass = mu_and_mass(name, catalog_bodies)
            bodies[name] = BodyInitialState(name, mu, mass, x, y, z, vx, vy, vz)

    if reference_body not in bodies:
        mu, mass = mu_and_mass(reference_body, catalog_bodies)
        bodies[reference_body] = BodyInitialState(reference_body, mu, mass, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    if start_ut_s is None:
        raise ValueError(f"Nenhuma linha com sample_index={initial_sample_index} em {states_csv}.")

    manifest_offset = manifest.get("time_mapping", {}).get("et_offset_seconds", 0.0)
    offset = float(et_offset_seconds if et_offset_seconds is not None else manifest_offset)
    stored_frame = manifest.get("reference_frame", {}).get("frame_convention", "unknown")
    frame_convention = f"{stored_frame}; additional_flip_z_input={flip_z_input}"
    return InputEphemeris(reference_body, float(start_ut_s), offset, frame_convention, bodies)


def make_rebound_simulation(eph: InputEphemeris, config: ExportConfig):
    try:
        import rebound  # type: ignore
    except ImportError as exc:
        raise SystemExit("Instale REBOUND no ambiente: pip install rebound") from exc

    sim = rebound.Simulation()
    sim.G = G_SI
    sim.integrator = config.integrator

    if config.integrator.lower() == "ias15":
        try:
            sim.ri_ias15.epsilon = config.ias15_epsilon
        except Exception:
            pass
    elif config.integrator.lower() == "whfast":
        if config.whfast_dt_seconds is None:
            raise ValueError("WHFast exige --whfast-dt-seconds.")
        sim.dt = float(config.whfast_dt_seconds)
    elif config.integrator.lower() in {"mercurius", "trace", "leapfrog", "sei", "saba"}:
        if config.whfast_dt_seconds is not None:
            sim.dt = float(config.whfast_dt_seconds)

    # Add central body first for stable indexing and later relative-state export.
    if config.central_body not in eph.bodies:
        raise KeyError(f"central_body {config.central_body!r} não encontrado nos estados iniciais.")

    ordered_names = [config.central_body] + sorted(n for n in eph.bodies if n != config.central_body)
    for name in ordered_names:
        b = eph.bodies[name]
        sim.add(m=b.mass_kg, x=b.x_m, y=b.y_m, z=b.z_m, vx=b.vx_m_s, vy=b.vy_m_s, vz=b.vz_m_s)

    sim.move_to_com()
    return sim, ordered_names


def particle_state_relative_km(sim: Any, body_index: int, center_index: int) -> Tuple[float, float, float, float, float, float]:
    p = sim.particles[body_index]
    c = sim.particles[center_index]
    return (
        (p.x - c.x) * M_TO_KM,
        (p.y - c.y) * M_TO_KM,
        (p.z - c.z) * M_TO_KM,
        (p.vx - c.vx) * M_TO_KM,
        (p.vy - c.vy) * M_TO_KM,
        (p.vz - c.vz) * M_TO_KM,
    )


def chebyshev_lobatto_nodes(n: int) -> np.ndarray:
    if n < 2:
        raise ValueError("São necessários pelo menos dois nós de ajuste por registro.")
    k = np.arange(n, dtype=float)
    tau = np.cos(math.pi * k / (n - 1))
    return np.sort(tau)


def fit_type3_record(
    taus: np.ndarray,
    values: np.ndarray,
    degree: int,
) -> Tuple[List[float], float]:
    """
    Fit one SPK Type 3 record for spice.spkw03.

    spice.spkw03/cspice_spkw03 wants CDATA containing only Chebyshev
    coefficients, not the internal MID/RADIUS fields described in low-level
    SPK record layouts. For each record the API expects:

        X coeffs, Y coeffs, Z coeffs, VX coeffs, VY coeffs, VZ coeffs

    i.e. exactly 6 * (degree + 1) doubles per record. The segment start
    time (btime) and fixed interval length (step/intlen) are passed as
    separate arguments to spice.spkw03.

    values shape: (n_samples, 6), units km and km/s.
    """
    record: List[float] = []
    max_resid = 0.0
    for component in range(6):
        coeff = chebfit(taus, values[:, component], degree)
        reconstructed = chebval(taus, coeff)
        resid = float(np.max(np.abs(reconstructed - values[:, component])))
        max_resid = max(max_resid, resid)
        record.extend(float(c) for c in coeff)
    expected = 6 * (degree + 1)
    if len(record) != expected:
        raise AssertionError(f"SPK Type 3 record length mismatch: got {len(record)}, expected {expected}")
    return record, max_resid


def safe_save_archive(sim: Any, path: str, delete_file: bool = False) -> None:
    """Support both current and older REBOUND Python APIs."""
    try:
        sim.save_to_file(path, delete_file=delete_file)
    except TypeError:
        # Older API used deletefile.
        try:
            sim.save_to_file(path, deletefile=delete_file)
        except Exception:
            sim.simulationarchive_snapshot(path, deletefile=delete_file)
    except AttributeError:
        sim.simulationarchive_snapshot(path, deletefile=delete_file)


def build_chebyshev_records(
    sim: Any,
    ordered_names: List[str],
    config: ExportConfig,
    start_et: float,
    output_report_every: int = 20,
) -> Tuple[Dict[str, List[float]], Dict[str, Any]]:
    if not (0 <= config.cheby_degree <= 27):
        raise ValueError("SPK Type 3 impõe grau Chebyshev entre 0 e 27; use 12-17 como ponto de partida.")
    if config.record_span_days <= 0 or config.duration_years <= 0:
        raise ValueError("duration_years e record_span_days precisam ser positivos.")

    record_span_s = config.record_span_days * DAY_S
    duration_s = config.duration_years * JULIAN_YEAR_S
    n_records = int(math.ceil(duration_s / record_span_s))
    fit_samples = max(config.samples_per_record, config.cheby_degree + 1)
    taus = chebyshev_lobatto_nodes(fit_samples)

    name_to_index = {name: i for i, name in enumerate(ordered_names)}
    center_index = name_to_index[config.central_body]
    target_names = [name for name in ordered_names if name != config.central_body]

    cdata_by_body: Dict[str, List[float]] = {name: [] for name in target_names}
    residuals: Dict[str, List[float]] = {name: [] for name in target_names}
    energy_samples: List[Tuple[float, float]] = []
    wall_start = time.time()

    initial_energy = None
    try:
        initial_energy = float(sim.energy())
    except Exception:
        pass

    if config.archive_path:
        archive = Path(config.archive_path)
        archive.parent.mkdir(parents=True, exist_ok=True)
        safe_save_archive(sim, str(archive), delete_file=True)

    for rec in range(n_records):
        # SPK Type 3 uses one fixed interval length (step/intlen) for all
        # records. Even if the segment END time falls inside the last
        # interval, that last record must be fitted on the full interval
        # implied by btime + rec * intlen.
        rec_start_rel = rec * record_span_s
        rec_end_rel = (rec + 1) * record_span_s
        mid_rel = 0.5 * (rec_start_rel + rec_end_rel)
        radius_s = 0.5 * record_span_s

        samples_by_body: Dict[str, List[Tuple[float, float, float, float, float, float]]] = {
            name: [] for name in target_names
        }

        for tau in taus:
            t_rel = mid_rel + radius_s * float(tau)
            # exact_finish_time=1 makes output epochs exactly equal to requested times in many REBOUND integrators.
            try:
                sim.integrate(t_rel, exact_finish_time=1)
            except TypeError:
                sim.integrate(t_rel)
            for name in target_names:
                samples_by_body[name].append(particle_state_relative_km(sim, name_to_index[name], center_index))

        for name in target_names:
            values = np.asarray(samples_by_body[name], dtype=float)
            record, max_resid = fit_type3_record(taus, values, config.cheby_degree)
            cdata_by_body[name].extend(record)
            residuals[name].append(max_resid)

        if config.archive_path and config.archive_every_records > 0 and ((rec + 1) % config.archive_every_records == 0):
            safe_save_archive(sim, config.archive_path, delete_file=False)

        try:
            energy_samples.append((float(sim.t), float(sim.energy())))
        except Exception:
            pass

        if output_report_every > 0 and ((rec + 1) % output_report_every == 0 or rec == n_records - 1):
            elapsed = time.time() - wall_start
            print(f"[REBOUND] record {rec+1}/{n_records}; sim_t={sim.t / JULIAN_YEAR_S:.3f} yr; wall={elapsed:.1f}s", flush=True)

    if config.archive_path:
        safe_save_archive(sim, config.archive_path, delete_file=False)

    residual_summary: Dict[str, Any] = {}
    for name, vals in residuals.items():
        residual_summary[name] = {
            "n_records": len(vals),
            "max_abs_fit_residual_component_units_km_or_km_s": max(vals) if vals else None,
            "median_abs_fit_residual_component_units_km_or_km_s": statistics.median(vals) if vals else None,
        }

    final_energy = None
    rel_energy_drift = None
    try:
        final_energy = float(sim.energy())
        if initial_energy not in (None, 0.0):
            rel_energy_drift = (final_energy - initial_energy) / abs(initial_energy)
    except Exception:
        pass

    report = {
        "record_span_s": record_span_s,
        "requested_duration_s": duration_s,
        "fit_coverage_s": n_records * record_span_s,
        "n_records": n_records,
        "cheby_degree": config.cheby_degree,
        "samples_per_record": fit_samples,
        "target_names": target_names,
        "fit_residual_summary": residual_summary,
        "energy": {
            "initial": initial_energy,
            "final": final_energy,
            "relative_drift": rel_energy_drift,
            "samples": energy_samples[:10] + ([("...", "...")] if len(energy_samples) > 20 else []) + energy_samples[-10:],
        },
    }
    return cdata_by_body, report


def make_naif_code_map(ordered_names: List[str], config: ExportConfig) -> Dict[str, int]:
    code_map: Dict[str, int] = {config.central_body: config.center_naif_code}
    code = config.target_naif_code_base
    for name in ordered_names:
        if name == config.central_body:
            continue
        while code == config.center_naif_code:
            code += 1
        code_map[name] = code
        code += 1
    return code_map


def write_naif_ids_kernel(path: Path, code_map: Dict[str, int]) -> None:
    lines = [
        "KPL/IK",
        "",
        "Custom NAIF ID mapping for KSP/Principia-generated SPK.",
        "Load this text kernel before resolving body names with SpiceyPy/CSPICE.",
        "",
        "\\begindata",
    ]
    for name, code in code_map.items():
        # SPICE names are case-insensitive but text kernels are easier to inspect in uppercase.
        spice_name = name.upper().replace(" ", "_").replace("-", "_")
        lines.append(f"   NAIF_BODY_NAME += ( '{spice_name}' )")
        lines.append(f"   NAIF_BODY_CODE += ( {code} )")
    lines.extend(["", "\\begintext", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def write_spk_type3(
    output_spk: Path,
    cdata_by_body: Dict[str, List[float]],
    code_map: Dict[str, int],
    config: ExportConfig,
    start_et: float,
    first_et: float,
    last_et: float,
    n_records: int,
) -> None:
    try:
        import spiceypy as spice  # type: ignore
    except ImportError as exc:
        raise SystemExit("Instale SpiceyPy para escrever SPK: pip install spiceypy") from exc

    if output_spk.exists():
        output_spk.unlink()
    output_spk.parent.mkdir(parents=True, exist_ok=True)

    handle = spice.spkopn(str(output_spk), "KSP_REBOUND_TYPE3", 0)
    try:
        expected_cdata_len = n_records * 6 * (config.cheby_degree + 1)
        for name, cdata in cdata_by_body.items():
            if len(cdata) != expected_cdata_len:
                raise ValueError(
                    f"{name}: invalid SPK Type 3 CDATA length {len(cdata)}; "
                    f"expected {expected_cdata_len} = n_records*6*(degree+1)."
                )
            segid = f"KSP_{name[:32]}"[:40]
            spice.spkw03(
                handle,
                int(code_map[name]),
                int(config.center_naif_code),
                config.frame_name,
                float(first_et),
                float(last_et),
                segid,
                float(config.record_span_days * DAY_S),
                int(n_records),
                int(config.cheby_degree),
                np.asarray(cdata, dtype=float),
                float(start_et),
            )
    finally:
        spice.spkcls(handle)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Propaga estados KSP/Principia com REBOUND e escreve SPK Type 3.")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--input-json", type=Path, help="JSON legado ksp_ephemerides.json.")
    src.add_argument("--acquisition-dir", type=Path, help="Diretório com manifest.json, body_catalog.json e states.csv.")
    p.add_argument("--body-catalog", type=Path, help="Catálogo físico para JSON legado.")
    p.add_argument("--central-body", required=True, help="Corpo central: Kerbol, Sun, etc.")
    p.add_argument("--initial-sample-index", type=int, default=0)
    p.add_argument("--et-offset-seconds", type=float, default=None)
    p.add_argument("--flip-z-input", action="store_true", help="Converter entrada raw kRPC left-handed para right-handed por z -> -z.")

    p.add_argument("--duration-years", type=float, default=105.0)
    p.add_argument("--record-span-days", type=float, default=32.0)
    p.add_argument("--cheby-degree", type=int, default=15)
    p.add_argument("--samples-per-record", type=int, default=24)
    p.add_argument("--integrator", default="ias15", choices=["ias15", "whfast", "mercurius", "trace", "leapfrog", "sei", "saba"])
    p.add_argument("--ias15-epsilon", type=float, default=1e-11)
    p.add_argument("--whfast-dt-seconds", type=float, default=None)

    p.add_argument("--frame-name", default="J2000", help="Frame SPICE reconhecido. Para KSP fictício, J2000 é rótulo de eixos fixos.")
    p.add_argument("--center-naif-code", type=int, default=990000)
    p.add_argument("--target-naif-code-base", type=int, default=990001)
    p.add_argument("--output-spk", type=Path, required=True)
    p.add_argument("--metadata-json", type=Path, default=None)
    p.add_argument("--naif-ids-kernel", type=Path, default=None)
    p.add_argument("--archive-path", type=str, default=None, help="REBOUND Simulationarchive para restart/debug.")
    p.add_argument("--archive-every-records", type=int, default=50)
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    if args.input_json:
        eph = load_legacy_json(
            input_json=args.input_json,
            body_catalog_path=args.body_catalog,
            central_body=args.central_body,
            initial_sample_index=args.initial_sample_index,
            et_offset_seconds=args.et_offset_seconds,
            flip_z_input=args.flip_z_input,
        )
    else:
        eph = load_acquisition_dir(
            acquisition_dir=args.acquisition_dir,
            central_body=args.central_body,
            initial_sample_index=args.initial_sample_index,
            et_offset_seconds=args.et_offset_seconds,
            flip_z_input=args.flip_z_input,
        )

    config = ExportConfig(
        duration_years=args.duration_years,
        record_span_days=args.record_span_days,
        cheby_degree=args.cheby_degree,
        samples_per_record=args.samples_per_record,
        integrator=args.integrator,
        whfast_dt_seconds=args.whfast_dt_seconds,
        ias15_epsilon=args.ias15_epsilon,
        central_body=args.central_body,
        frame_name=args.frame_name,
        center_naif_code=args.center_naif_code,
        target_naif_code_base=args.target_naif_code_base,
        archive_path=args.archive_path,
        archive_every_records=args.archive_every_records,
    )

    start_et = eph.start_ut_s + eph.et_offset_seconds
    first_et = start_et
    last_et = start_et + config.duration_years * JULIAN_YEAR_S

    print("[LOAD] corpos:", len(eph.bodies), "referência:", eph.reference_body, "central:", config.central_body)
    print("[TIME] start_ut_s:", eph.start_ut_s, "start_et_s:", start_et, "duration_years:", config.duration_years)
    print("[REBOUND] integrator:", config.integrator, "record_span_days:", config.record_span_days, "degree:", config.cheby_degree)

    sim, ordered_names = make_rebound_simulation(eph, config)
    code_map = make_naif_code_map(ordered_names, config)

    cdata_by_body, report = build_chebyshev_records(sim, ordered_names, config, start_et)
    write_spk_type3(
        output_spk=args.output_spk,
        cdata_by_body=cdata_by_body,
        code_map=code_map,
        config=config,
        start_et=start_et,
        first_et=first_et,
        last_et=last_et,
        n_records=report["n_records"],
    )

    ids_path = args.naif_ids_kernel or args.output_spk.with_suffix(".ids.tpc")
    write_naif_ids_kernel(ids_path, code_map)

    metadata_path = args.metadata_json or args.output_spk.with_suffix(".metadata.json")
    metadata = {
        "schema": "ksp_rebound_spk_type3_export.v1",
        "input": {
            "reference_body": eph.reference_body,
            "central_body": config.central_body,
            "start_ut_s": eph.start_ut_s,
            "et_offset_seconds": eph.et_offset_seconds,
            "start_et_seconds_past_j2000_tdb_convention": start_et,
            "frame_convention": eph.frame_convention,
        },
        "rebound": {
            "units": "SI internally: meter, kilogram, second",
            "G_SI": G_SI,
            "integrator": config.integrator,
            "ias15_epsilon": config.ias15_epsilon,
            "whfast_dt_seconds": config.whfast_dt_seconds,
            "move_to_com_applied": True,
        },
        "spk": {
            "path": str(args.output_spk),
            "type": 3,
            "units_written": "km and km/s, SPICE convention",
            "frame_name": config.frame_name,
            "first_et": first_et,
            "last_et": last_et,
            "record_span_days": config.record_span_days,
            "cheby_degree": config.cheby_degree,
            "center_naif_code": config.center_naif_code,
            "naif_ids_kernel": str(ids_path),
            "code_map": code_map,
            "warning": "For fictional KSP systems, frame_name=J2000 is a computational inertial-axis label unless you provide a proper SPICE frame kernel.",
        },
        "bodies": {name: asdict(body) for name, body in eph.bodies.items()},
        "fit_and_integration_report": report,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    print("[OK] SPK escrito:", args.output_spk)
    print("[OK] NAIF IDs:", ids_path)
    print("[OK] metadata:", metadata_path)
    drift = report.get("energy", {}).get("relative_drift")
    if drift is not None:
        print("[CHECK] relative energy drift:", f"{drift:.3e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
