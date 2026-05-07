"""Canonical CSV/JSON schemas for the MGA pipeline.

This module is intentionally conservative: it validates required fields and
provides typed accessors, but does not encode project-specific mission logic.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np


def norm_body(name: str) -> str:
    return name.strip().upper()


def f(row: Mapping[str, Any], key: str, default: float | None = None) -> float:
    value = row.get(key, None)
    if value in (None, ""):
        if default is None:
            raise KeyError(f"missing required numeric field: {key}")
        return float(default)
    return float(value)


def s(row: Mapping[str, Any], key: str, default: str | None = None) -> str:
    value = row.get(key, None)
    if value in (None, ""):
        if default is None:
            raise KeyError(f"missing required string field: {key}")
        return default
    return str(value)


def vec3(row: Mapping[str, Any], prefix: str, unit_suffix: str) -> np.ndarray:
    return np.asarray([
        f(row, f"{prefix}_x_{unit_suffix}"),
        f(row, f"{prefix}_y_{unit_suffix}"),
        f(row, f"{prefix}_z_{unit_suffix}"),
    ], dtype=float)


@dataclass(frozen=True)
class LegSeed:
    leg: int
    dep_body: str
    arr_body: str
    tof_days: float
    path: str
    vdep_km_s: np.ndarray
    varr_km_s: np.ndarray


@dataclass(frozen=True)
class CandidateSeed:
    candidate_id: str
    rank: int
    sequence: list[str]
    epochs_et_s: list[float]
    legs: list[LegSeed]
    raw_sum_km_s: float
    departure_vinf_km_s: float
    arrival_vinf_km_s: float
    powered_flyby_dv_km_s: float
    turn_excess_deg: float
    min_turn_margin_deg: float
    tof_total_days: float
    source_row: dict[str, Any]


@dataclass(frozen=True)
class LegOptimizationResult:
    candidate_id: str
    leg: int
    dep_body: str
    arr_body: str
    t_dep_s: float
    t_arr_s: float
    t_start_s: float
    t_end_s: float
    buffer_days: float
    frame: str
    transform: str
    initial_v_m_s: np.ndarray
    correction_dv_m_s: np.ndarray
    optimized_v_m_s: np.ndarray
    start_r_m: np.ndarray
    start_v_m_s: np.ndarray
    target_r_m: np.ndarray
    target_v_m_s: np.ndarray
    final_r_m: np.ndarray
    final_v_m_s: np.ndarray
    final_miss_km: float
    final_relv_m_s: float
    solver_success: bool
    solver_message: str

    def to_row(self) -> dict[str, Any]:
        row = {
            "candidate_id": self.candidate_id,
            "leg": self.leg,
            "dep_body": self.dep_body,
            "arr_body": self.arr_body,
            "t_dep_s": self.t_dep_s,
            "t_arr_s": self.t_arr_s,
            "t_start_s": self.t_start_s,
            "t_end_s": self.t_end_s,
            "buffer_days": self.buffer_days,
            "frame": self.frame,
            "transform": self.transform,
            "final_miss_km": self.final_miss_km,
            "final_relv_m_s": self.final_relv_m_s,
            "solver_success": self.solver_success,
            "solver_message": self.solver_message,
        }
        add_vec(row, "initial_v", "m_s", self.initial_v_m_s)
        add_vec(row, "dv", "m_s", self.correction_dv_m_s)
        row["dv_norm_m_s"] = float(np.linalg.norm(self.correction_dv_m_s))
        add_vec(row, "optimized_v", "m_s", self.optimized_v_m_s)
        add_vec(row, "start_r", "m", self.start_r_m)
        add_vec(row, "start_v", "m_s", self.start_v_m_s)
        add_vec(row, "target_r", "m", self.target_r_m)
        add_vec(row, "target_v", "m_s", self.target_v_m_s)
        add_vec(row, "final_r", "m", self.final_r_m)
        add_vec(row, "final_v", "m_s", self.final_v_m_s)
        return row


@dataclass(frozen=True)
class FlybyAuditEntry:
    event_index: int
    body: str
    vinf_in_km_s: float
    vinf_out_km_s: float
    vinf_mismatch_km_s: float
    turn_required_deg: float
    turn_max_deg: float
    turn_margin_deg: float
    status: str


@dataclass(frozen=True)
class FlybyAuditResult:
    candidate_id: str
    sequence: list[str]
    status: str
    total_leg_correction_m_s: float
    max_vinf_mismatch_km_s: float
    min_turn_margin_deg: float
    flybys: list[FlybyAuditEntry]

    def to_json_obj(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "sequence": self.sequence,
            "status": self.status,
            "total_leg_correction_m_s": self.total_leg_correction_m_s,
            "max_vinf_mismatch_km_s": self.max_vinf_mismatch_km_s,
            "min_turn_margin_deg": self.min_turn_margin_deg,
            "flybys": [asdict(x) for x in self.flybys],
        }


def add_vec(row: dict[str, Any], prefix: str, unit_suffix: str, v: Sequence[float]) -> None:
    row[f"{prefix}_x_{unit_suffix}"] = float(v[0])
    row[f"{prefix}_y_{unit_suffix}"] = float(v[1])
    row[f"{prefix}_z_{unit_suffix}"] = float(v[2])


def read_csv_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="") as fh:
        return list(csv.DictReader(fh))


def write_csv_rows(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fields:
                fields.append(key)
    with p.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow(dict(row))


def parse_sequence(row: Mapping[str, Any]) -> list[str]:
    if row.get("sequence_bodies"):
        return [norm_body(x) for x in str(row["sequence_bodies"]).replace(",", " ").split()]
    if row.get("sequence"):
        return [norm_body(x) for x in str(row["sequence"]).replace("-", " ").replace(",", " ").split()]
    raise KeyError("candidate row needs sequence_bodies or sequence")


def parse_epochs(row: Mapping[str, Any], sequence: Sequence[str]) -> list[float]:
    epochs: list[float] = []
    for i, body in enumerate(sequence):
        key_specific = f"event{i}_{norm_body(body)}_et_s"
        key_generic = f"event{i}_et_s"
        if key_specific in row and row[key_specific] not in (None, ""):
            epochs.append(float(row[key_specific]))
        elif key_generic in row and row[key_generic] not in (None, ""):
            epochs.append(float(row[key_generic]))
        elif row.get("epochs_et_s"):
            parts = str(row["epochs_et_s"]).split(",")
            return [float(x) for x in parts if x.strip()]
        else:
            raise KeyError(f"missing epoch for event {i} {body}")
    return epochs


def parse_candidate_seed_row(row: Mapping[str, Any], rank: int | None = None) -> CandidateSeed:
    sequence = parse_sequence(row)
    epochs = parse_epochs(row, sequence)
    n_legs = len(sequence) - 1
    legs: list[LegSeed] = []
    tof_parts = str(row.get("tofs_days", "")).split(",") if row.get("tofs_days") else []
    path_parts = str(row.get("leg_paths", "")).split(",") if row.get("leg_paths") else []

    for leg in range(1, n_legs + 1):
        dep = norm_body(row.get(f"leg{leg}_dep", sequence[leg - 1]))
        arr = norm_body(row.get(f"leg{leg}_arr", sequence[leg]))
        tof = float(row.get(f"leg{leg}_tof_days") or (tof_parts[leg - 1] if len(tof_parts) >= leg else (epochs[leg] - epochs[leg - 1]) / 86400.0))
        path = str(row.get(f"leg{leg}_path") or (path_parts[leg - 1] if len(path_parts) >= leg else ""))

        # Prefer explicit velocities. If absent, fail loudly; the refactored
        # pipeline should not re-solve Lambert in the N-body stage.
        vdep = np.asarray([
            f(row, f"leg{leg}_vdep_x_km_s"),
            f(row, f"leg{leg}_vdep_y_km_s"),
            f(row, f"leg{leg}_vdep_z_km_s"),
        ], dtype=float)
        varr = np.asarray([
            f(row, f"leg{leg}_varr_x_km_s"),
            f(row, f"leg{leg}_varr_y_km_s"),
            f(row, f"leg{leg}_varr_z_km_s"),
        ], dtype=float)
        legs.append(LegSeed(leg, dep, arr, tof, path, vdep, varr))

    inferred_rank = int(row.get("rank") or rank or 0)
    return CandidateSeed(
        candidate_id=str(row.get("candidate_id") or f"rank{inferred_rank}"),
        rank=inferred_rank,
        sequence=sequence,
        epochs_et_s=epochs,
        legs=legs,
        raw_sum_km_s=f(row, "raw_sum_km_s", 0.0),
        departure_vinf_km_s=f(row, "departure_vinf_km_s", 0.0),
        arrival_vinf_km_s=f(row, "arrival_vinf_km_s", 0.0),
        powered_flyby_dv_km_s=f(row, "powered_flyby_dv_km_s", 0.0),
        turn_excess_deg=f(row, "turn_excess_deg", 0.0),
        min_turn_margin_deg=f(row, "min_turn_margin_deg", 0.0),
        tof_total_days=f(row, "tof_total_days", 0.0),
        source_row=dict(row),
    )


def read_candidate_seed(path: str | Path, rank: int) -> CandidateSeed:
    rows = read_csv_rows(path)
    if rank < 1 or rank > len(rows):
        raise IndexError(f"rank {rank} out of range 1..{len(rows)}")
    return parse_candidate_seed_row(rows[rank - 1], rank=rank)


def write_leg_optimizations(path: str | Path, results: Sequence[LegOptimizationResult]) -> None:
    write_csv_rows(path, [r.to_row() for r in results])


def read_leg_optimizations(path: str | Path) -> list[dict[str, str]]:
    rows = read_csv_rows(path)
    required = [
        "leg", "dep_body", "arr_body",
        "final_r_x_m", "final_r_y_m", "final_r_z_m",
        "final_v_x_m_s", "final_v_y_m_s", "final_v_z_m_s",
        "optimized_v_x_m_s", "optimized_v_y_m_s", "optimized_v_z_m_s",
    ]
    missing = [k for k in required if rows and k not in rows[0]]
    if missing:
        raise KeyError(f"leg optimization file missing columns: {missing}")
    return rows


def write_flyby_audit(path: str | Path, audit: FlybyAuditResult) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(audit.to_json_obj(), indent=2, ensure_ascii=False), encoding="utf-8")


def read_flyby_audit(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))
