#!/usr/bin/env python3
"""
spice_lambert_scout.py

V0.1 Lambert scout for an offline-first KSP + Principia MGA pipeline.

Purpose
-------
Read a synthetic SPICE/SPK ephemeris, solve many single-leg Lambert arcs from
one origin body to one or more target bodies, and write ranked candidate windows
for the next global MGA beam-search layer.

This script is intentionally planning-grade:
  * SPICE is the ephemeris contract.
  * Lambert is a seed/ranking model, not the truth model.
  * kRPC is not used here.
  * Final targeting remains the responsibility of local dense kernels,
    B-plane/flyby targeters, Tudat/REBOUND validation, and Principia regression.

Suggested dependency install
----------------------------
  conda install -c conda-forge spiceypy numpy pykep

Minimal example
---------------
  python spice_lambert_scout.py \
    --bsp data/spice_v0_1_33y/opm_mpe_principia_rebound_v0_1_33y.bsp \
    --tpc data/spice_v0_1_33y/opm_mpe_principia_rebound_v0_1_33y.ids.tpc \
    --metadata data/spice_v0_1_33y/opm_mpe_principia_rebound_v0_1_33y.metadata.json \
    --policy data/spice_v0_1_33y/target_policy_v0_1.json \
    --origin Kerbin \
    --targets Duna Jool Sarnus Urlum Neidon Plock Soden \
    --central-body Sun \
    --depart-start-days 0 \
    --depart-stop-days 3650 \
    --depart-step-days 5 \
    --tof-min-days 60 \
    --tof-max-days 4000 \
    --tof-step-days 10 \
    --max-c3 500 \
    --top-n-per-target 250 \
    --output-csv data/mga_v0_1/kerbin_lambert_scout.csv \
    --output-json data/mga_v0_1/kerbin_lambert_scout.summary.json
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import heapq
import importlib
import json
import logging
import math
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy as np

SECONDS_PER_DAY = 86_400.0
DAYS_PER_YEAR = 365.25
SCHEMA_VERSION = "lambert_scout.v0.1"


# ---------------------------------------------------------------------------
# Data contracts for the V0.1 scout and the future MGA planner.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class KernelBundle:
    bsp: Path
    tpc: Optional[Path] = None
    metadata: Optional[Path] = None
    policy: Optional[Path] = None
    lsk: Optional[Path] = None
    frame: str = "J2000"


@dataclass(frozen=True)
class BodyPolicy:
    name: str
    allowed_for_global: bool = True
    allowed_for_fine_targeting: bool = False
    requires_local_revalidation: bool = False
    target_grade: str = "unknown"
    reason: str = ""
    raw: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BodyModel:
    name: str
    naif_id: Optional[int]
    mu_km3_s2: Optional[float]
    radius_km: Optional[float]
    policy: BodyPolicy


@dataclass(frozen=True)
class MissionSpec:
    origin: str
    targets: Tuple[str, ...]
    central_body: str
    ref_frame: str
    depart_start_et: float
    depart_stop_et: float
    depart_step_s: float
    tof_min_s: float
    tof_max_s: float
    tof_step_s: float
    max_revs: int
    cw_values: Tuple[bool, ...]


@dataclass(frozen=True)
class StateVector:
    body: str
    observer: str
    et: float
    r_km: np.ndarray
    v_km_s: np.ndarray


@dataclass(frozen=True)
class LambertCandidate:
    leg_id: str
    origin: str
    target: str
    central_body: str
    ref_frame: str
    depart_et: float
    arrive_et: float
    tof_s: float
    cw: bool
    solution_index: int
    max_revs: int
    c3_km2_s2: float
    vinf_depart_km_s: float
    vinf_arrive_km_s: float
    score: float
    origin_r_km: Tuple[float, float, float]
    origin_v_km_s: Tuple[float, float, float]
    target_r_km: Tuple[float, float, float]
    target_v_km_s: Tuple[float, float, float]
    sc_v_depart_km_s: Tuple[float, float, float]
    sc_v_arrive_km_s: Tuple[float, float, float]
    policy_grade: str
    policy_requires_revalidation: bool
    policy_reason: str

    @property
    def tof_days(self) -> float:
        return self.tof_s / SECONDS_PER_DAY

    def as_csv_row(self, coverage_start_et: Optional[float]) -> Dict[str, Any]:
        dep_days = ""
        arr_days = ""
        if coverage_start_et is not None:
            dep_days = (self.depart_et - coverage_start_et) / SECONDS_PER_DAY
            arr_days = (self.arrive_et - coverage_start_et) / SECONDS_PER_DAY

        row: Dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "leg_id": self.leg_id,
            "origin": self.origin,
            "target": self.target,
            "central_body": self.central_body,
            "ref_frame": self.ref_frame,
            "depart_et": self.depart_et,
            "arrive_et": self.arrive_et,
            "depart_days_from_coverage_start": dep_days,
            "arrive_days_from_coverage_start": arr_days,
            "tof_s": self.tof_s,
            "tof_days": self.tof_days,
            "cw": int(self.cw),
            "solution_index": self.solution_index,
            "max_revs": self.max_revs,
            "c3_km2_s2": self.c3_km2_s2,
            "vinf_depart_km_s": self.vinf_depart_km_s,
            "vinf_arrive_km_s": self.vinf_arrive_km_s,
            "score": self.score,
            "policy_grade": self.policy_grade,
            "policy_requires_revalidation": int(self.policy_requires_revalidation),
            "policy_reason": self.policy_reason,
        }
        add_vec(row, "origin_r_km", self.origin_r_km)
        add_vec(row, "origin_v_km_s", self.origin_v_km_s)
        add_vec(row, "target_r_km", self.target_r_km)
        add_vec(row, "target_v_km_s", self.target_v_km_s)
        add_vec(row, "sc_v_depart_km_s", self.sc_v_depart_km_s)
        add_vec(row, "sc_v_arrive_km_s", self.sc_v_arrive_km_s)
        return row

    def as_json_dict(self) -> Dict[str, Any]:
        d = self.as_csv_row(coverage_start_et=None)
        d.pop("depart_days_from_coverage_start", None)
        d.pop("arrive_days_from_coverage_start", None)
        return d


# Future MGA-layer contracts. The scout does not yet expand multi-leg routes,
# but its CSV rows map directly into these objects.

@dataclass(frozen=True)
class DSMSeed:
    enabled: bool
    eta: float = 0.5
    dv_km_s: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    frame: str = "RTN"


@dataclass(frozen=True)
class LegSeedRef:
    leg_id: str
    origin: str
    target: str
    depart_et: float
    arrive_et: float
    vinf_depart_km_s: float
    vinf_arrive_km_s: float
    c3_km2_s2: float
    score: float
    policy_requires_revalidation: bool


@dataclass(frozen=True)
class FlybyEnvelope:
    body: str
    v_inf_in_km_s: float
    v_inf_out_km_s: float
    rp_min_km: float
    turn_angle_required_rad: Optional[float] = None
    turn_angle_max_rad: Optional[float] = None
    bplane_bt_km: Optional[float] = None
    bplane_br_km: Optional[float] = None
    status: str = "uncomputed"


@dataclass(frozen=True)
class RouteGenome:
    route_id: str
    sequence: Tuple[str, ...]
    leg_ids: Tuple[str, ...]
    depart_et: float
    tof_s: Tuple[float, ...]
    dsm_seeds: Tuple[DSMSeed, ...] = field(default_factory=tuple)
    flyby_envelopes: Tuple[FlybyEnvelope, ...] = field(default_factory=tuple)
    tags: Tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class RouteMetrics:
    total_score: float
    total_tof_s: float
    total_c3_km2_s2: float
    sum_vinf_arrive_km_s: float
    min_policy_grade: str = "unknown"
    requires_local_revalidation: bool = False
    robustness_status: str = "not_evaluated"


@dataclass(frozen=True)
class BeamNode:
    genome: RouteGenome
    metrics: RouteMetrics
    last_body: str
    last_arrival_et: float
    pareto_bucket: str = "default"


# ---------------------------------------------------------------------------
# Generic utilities.
# ---------------------------------------------------------------------------

def add_vec(row: Dict[str, Any], prefix: str, vec: Sequence[float]) -> None:
    row[f"{prefix}_x"] = float(vec[0])
    row[f"{prefix}_y"] = float(vec[1])
    row[f"{prefix}_z"] = float(vec[2])


def stable_leg_id(parts: Sequence[Any]) -> str:
    payload = "|".join(str(p) for p in parts)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:20]


def require_module(module_name: str, install_hint: str):
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        raise SystemExit(
            f"[FATAL] Missing dependency '{module_name}'. Install it with: {install_hint}"
        ) from exc


def load_json(path: Optional[Path]) -> Dict[str, Any]:
    if path is None:
        return {}
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(float(x))


def norm(vec: np.ndarray) -> float:
    return float(np.linalg.norm(vec))


def vec3_tuple(vec: np.ndarray) -> Tuple[float, float, float]:
    return (float(vec[0]), float(vec[1]), float(vec[2]))


def make_grid(start: float, stop: float, step: float) -> List[float]:
    if step <= 0:
        raise ValueError("grid step must be positive")
    if stop < start:
        raise ValueError(f"grid stop {stop} is before start {start}")
    values: List[float] = []
    n = int(math.floor((stop - start) / step + 1e-12))
    for i in range(n + 1):
        values.append(start + i * step)
    if not values or values[-1] < stop - 1e-9:
        values.append(stop)
    return values


def parse_bool_values(values: Sequence[str]) -> Tuple[bool, ...]:
    out: List[bool] = []
    for value in values:
        v = str(value).strip().lower()
        if v in {"0", "false", "f", "no", "n", "ccw", "direct", "short"}:
            out.append(False)
        elif v in {"1", "true", "t", "yes", "y", "cw", "clockwise", "long"}:
            out.append(True)
        else:
            raise argparse.ArgumentTypeError(f"invalid boolean/cw value: {value}")
    return tuple(dict.fromkeys(out))  # preserve order, remove duplicates


def iter_dicts(obj: Any) -> Iterator[Mapping[str, Any]]:
    if isinstance(obj, Mapping):
        yield obj
        for value in obj.values():
            yield from iter_dicts(value)
    elif isinstance(obj, list):
        for item in obj:
            yield from iter_dicts(item)


def dict_contains_body(d: Mapping[str, Any], body: str) -> bool:
    body_l = body.lower()
    for k, v in d.items():
        if str(k).lower() in {body_l, "name", "body", "target", "naif_name"}:
            if str(v).lower() == body_l or str(k).lower() == body_l:
                return True
    return False


def scalar_from_any(value: Any) -> Optional[float]:
    if is_number(value):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and value:
        return scalar_from_any(value[0])
    return None


def convert_mu_to_km3_s2(value: float, key_hint: str = "") -> float:
    key_l = key_hint.lower()
    if "m3" in key_l or "m^3" in key_l:
        return value / 1.0e9
    if "km3" in key_l or "km^3" in key_l:
        return value
    # Heuristic: real-Sun GM in km^3/s^2 is ~1.327e11; in m^3/s^2 it is
    # ~1.327e20. KSP-scale stars/planets may vary, but values above 1e15 are
    # almost certainly SI m^3/s^2 in this pipeline.
    if abs(value) > 1.0e15:
        return value / 1.0e9
    return value


def extract_named_body_dict(root: Mapping[str, Any], body: str) -> Dict[str, Any]:
    body_l = body.lower()

    # Direct maps: {"Sun": {...}}, {"bodies": {"Sun": {...}}}, etc.
    for container_key in ("bodies", "body_catalog", "catalog", "targets", "target_policy", "policies"):
        container = root.get(container_key)
        if isinstance(container, Mapping):
            for k, v in container.items():
                if str(k).lower() == body_l and isinstance(v, Mapping):
                    return dict(v)
        elif isinstance(container, list):
            for item in container:
                if isinstance(item, Mapping) and dict_contains_body(item, body):
                    return dict(item)

    for d in iter_dicts(root):
        if dict_contains_body(d, body):
            return dict(d)
    return {}


def find_mu_in_metadata(metadata: Mapping[str, Any], body: str) -> Optional[float]:
    body_d = extract_named_body_dict(metadata, body)
    keys = (
        "mu_km3_s2", "gm_km3_s2", "GM_km3_s2", "gravitational_parameter_km3_s2",
        "mu_m3_s2", "gm_m3_s2", "GM_m3_s2", "gravitational_parameter_m3_s2",
        "mu", "gm", "GM", "gravitational_parameter",
    )
    for key in keys:
        if key in body_d:
            value = scalar_from_any(body_d[key])
            if value is not None:
                return convert_mu_to_km3_s2(value, key)
    return None


def find_radius_in_metadata(metadata: Mapping[str, Any], body: str) -> Optional[float]:
    body_d = extract_named_body_dict(metadata, body)
    keys = (
        "radius_km", "mean_radius_km", "equatorial_radius_km",
        "radius_m", "mean_radius_m", "equatorial_radius_m",
        "radius", "mean_radius", "equatorial_radius",
    )
    for key in keys:
        if key in body_d:
            value = scalar_from_any(body_d[key])
            if value is None:
                continue
            key_l = key.lower()
            if key_l.endswith("_m") or "radius_m" in key_l:
                return value / 1000.0
            # Heuristic: radii above 1e5 are likely metres; radii in km for KSP
            # bodies are normally much smaller than that unless scaled 1:1.
            if key in {"radius", "mean_radius", "equatorial_radius"} and value > 1.0e5:
                return value / 1000.0
            return value
    return None


def infer_coverage_from_metadata(metadata: Mapping[str, Any]) -> Tuple[Optional[float], Optional[float]]:
    if not metadata:
        return None, None

    start_keys = [
        "coverage_start_et", "start_et", "kernel_start_et", "bsp_start_et",
        "et_start", "t_start_et", "minimum_et", "min_et",
    ]
    end_keys = [
        "coverage_end_et", "end_et", "kernel_end_et", "bsp_end_et",
        "et_end", "t_end_et", "maximum_et", "max_et",
    ]

    def find_first(keys: Sequence[str]) -> Optional[float]:
        for d in iter_dicts(metadata):
            for key in keys:
                if key in d:
                    value = scalar_from_any(d[key])
                    if value is not None:
                        return value
        return None

    return find_first(start_keys), find_first(end_keys)


# ---------------------------------------------------------------------------
# Target policy handling.
# ---------------------------------------------------------------------------

def infer_body_policy(policy_root: Mapping[str, Any], body: str) -> BodyPolicy:
    if not policy_root:
        return BodyPolicy(name=body, reason="no policy file supplied")

    body_d = extract_named_body_dict(policy_root, body)
    if not body_d:
        return BodyPolicy(name=body, reason="body not present in policy file")

    text_blob = json.dumps(body_d, ensure_ascii=False).lower()

    def truthy_any(keys: Sequence[str]) -> Optional[bool]:
        for key in keys:
            if key in body_d:
                value = body_d[key]
                if isinstance(value, bool):
                    return value
                if isinstance(value, str):
                    v = value.strip().lower()
                    if v in {"allow", "allowed", "yes", "true", "coarse", "planning", "global"}:
                        return True
                    if v in {"deny", "blocked", "false", "no", "forbidden", "disabled"}:
                        return False
        return None

    explicit_global = truthy_any([
        "allowed_for_global", "global_allowed", "use_for_global", "allow_global",
        "planning_allowed", "allowed_for_planning", "global_planning",
    ])
    explicit_fine = truthy_any([
        "allowed_for_fine_targeting", "fine_targeting_allowed", "fine_allowed",
        "targeting_allowed", "allowed_for_targeting",
    ])

    blocked_words = ["do_not_use", "blocked", "forbidden", "disabled", "blacklist"]
    reval_words = ["revalidate", "revalidation", "local_kernel", "local revalidation"]
    coarse_words = ["coarse", "planning-grade", "planning_grade", "system-arrival", "system_arrival"]

    allowed_for_global = True if explicit_global is None else explicit_global

    # Important: V0.1 policy files often block *fine targeting* for some moons
    # while still allowing system-arrival/coarse global planning. Do not convert
    # a generic occurrence of the word "blocked" into a global ban. Only a
    # global/planning field, a top-level status field, or an explicit boolean
    # should disable the body for this scout.
    if explicit_global is False:
        allowed_for_global = False
    elif explicit_global is None:
        for key, value in body_d.items():
            key_l = str(key).lower()
            value_l = str(value).lower()
            if any(word in value_l for word in blocked_words):
                if (
                    "global" in key_l
                    or "planning" in key_l
                    or key_l in {"status", "state", "use", "allowed", "availability"}
                ):
                    allowed_for_global = False
                    break

    requires_revalidation = any(word in text_blob for word in reval_words)
    allowed_for_fine = bool(explicit_fine) if explicit_fine is not None else not requires_revalidation

    grade = "unknown"
    for key in ("target_grade", "grade", "class", "validation_class", "policy_class", "status"):
        if key in body_d and body_d[key] is not None:
            grade = str(body_d[key])
            break
    if grade == "unknown" and any(word in text_blob for word in coarse_words):
        grade = "coarse"
    if grade == "unknown" and requires_revalidation:
        grade = "revalidate"

    reason = ""
    for key in ("reason", "notes", "note", "comment", "description"):
        if key in body_d and body_d[key] is not None:
            reason = str(body_d[key])
            break

    return BodyPolicy(
        name=body,
        allowed_for_global=allowed_for_global,
        allowed_for_fine_targeting=allowed_for_fine,
        requires_local_revalidation=requires_revalidation,
        target_grade=grade,
        reason=reason,
        raw=body_d,
    )


# ---------------------------------------------------------------------------
# SPICE provider.
# ---------------------------------------------------------------------------

class SpiceEphemeris:
    def __init__(self, bundle: KernelBundle, central_body: str):
        self.bundle = bundle
        self.central_body = central_body
        self.sp = None
        self._state_cache: Dict[Tuple[str, float, str], StateVector] = {}

    def __enter__(self) -> "SpiceEphemeris":
        self.sp = require_module("spiceypy", "conda install -c conda-forge spiceypy")
        if self.bundle.lsk:
            self.sp.furnsh(str(self.bundle.lsk))
        if self.bundle.tpc:
            self.sp.furnsh(str(self.bundle.tpc))
        self.sp.furnsh(str(self.bundle.bsp))
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.sp is not None:
            self.sp.kclear()

    def body_code(self, body: str) -> Optional[int]:
        assert self.sp is not None
        try:
            return int(self.sp.bodn2c(body))
        except Exception:
            try:
                return int(body)
            except Exception:
                return None

    def state(self, body: str, et: float, observer: Optional[str] = None) -> StateVector:
        assert self.sp is not None
        obs = observer or self.central_body
        key = (body, float(et), obs)
        if key in self._state_cache:
            return self._state_cache[key]
        state, _lt = self.sp.spkezr(body, float(et), self.bundle.frame, "NONE", obs)
        arr = np.asarray(state, dtype=float)
        sv = StateVector(
            body=body,
            observer=obs,
            et=float(et),
            r_km=arr[:3].copy(),
            v_km_s=arr[3:].copy(),
        )
        self._state_cache[key] = sv
        return sv

    def gm_km3_s2(self, body: str, metadata: Mapping[str, Any]) -> Optional[float]:
        assert self.sp is not None
        try:
            _dim, values = self.sp.bodvrd(body, "GM", 1)
            if len(values) > 0:
                return convert_mu_to_km3_s2(float(values[0]), "GM")
        except Exception:
            pass
        return find_mu_in_metadata(metadata, body)

    def radius_km(self, body: str, metadata: Mapping[str, Any]) -> Optional[float]:
        assert self.sp is not None
        try:
            _dim, values = self.sp.bodvrd(body, "RADII", 3)
            if len(values) > 0:
                return float(max(values))
        except Exception:
            pass
        return find_radius_in_metadata(metadata, body)

    def coverage_for_body(self, body: str) -> List[Tuple[float, float]]:
        assert self.sp is not None
        code = self.body_code(body)
        if code is None:
            return []
        try:
            cell = self.sp.spkcov(str(self.bundle.bsp), code)
            n = int(self.sp.wncard(cell))
            return [tuple(map(float, self.sp.wnfetd(cell, i))) for i in range(n)]
        except Exception as exc:
            logging.debug("Could not read SPK coverage for %s: %s", body, exc)
            return []


# ---------------------------------------------------------------------------
# Lambert solver wrapper.
# ---------------------------------------------------------------------------

def solve_lambert_pykep(
    r1_km: np.ndarray,
    r2_km: np.ndarray,
    tof_s: float,
    mu_km3_s2: float,
    *,
    cw: bool,
    max_revs: int,
) -> List[Tuple[np.ndarray, np.ndarray, int]]:
    pk = require_module("pykep", "conda install -c conda-forge pykep")
    try:
        lp = pk.lambert_problem(
            r1_km.astype(float).tolist(),
            r2_km.astype(float).tolist(),
            float(tof_s),
            float(mu_km3_s2),
            bool(cw),
            int(max_revs),
        )
    except TypeError:
        # Some builds expose keyword names; keep a fallback for PyKEP variants.
        lp = pk.lambert_problem(
            r0=r1_km.astype(float).tolist(),
            r1=r2_km.astype(float).tolist(),
            tof=float(tof_s),
            mu=float(mu_km3_s2),
            cw=bool(cw),
            max_revs=int(max_revs),
        )

    v1s = lp.get_v1()
    v2s = lp.get_v2()
    solutions: List[Tuple[np.ndarray, np.ndarray, int]] = []
    for i, (v1, v2) in enumerate(zip(v1s, v2s)):
        v1_arr = np.asarray(v1, dtype=float)
        v2_arr = np.asarray(v2, dtype=float)
        if np.all(np.isfinite(v1_arr)) and np.all(np.isfinite(v2_arr)):
            solutions.append((v1_arr, v2_arr, i))
    return solutions


# ---------------------------------------------------------------------------
# Scoring, filtering, and bounded result collection.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ScoreWeights:
    vinf_depart: float
    vinf_arrive: float
    tof_years: float
    c3_sqrt: float
    revalidation_penalty: float


def score_candidate(
    *,
    vinf_depart_km_s: float,
    vinf_arrive_km_s: float,
    tof_s: float,
    c3_km2_s2: float,
    policy: BodyPolicy,
    weights: ScoreWeights,
) -> float:
    tof_years = tof_s / SECONDS_PER_DAY / DAYS_PER_YEAR
    score = (
        weights.vinf_depart * vinf_depart_km_s
        + weights.vinf_arrive * vinf_arrive_km_s
        + weights.tof_years * tof_years
        + weights.c3_sqrt * math.sqrt(max(c3_km2_s2, 0.0))
    )
    if policy.requires_local_revalidation:
        score += weights.revalidation_penalty
    return float(score)


class CandidateCollector:
    def __init__(self, top_n_per_target: int):
        self.top_n_per_target = int(top_n_per_target)
        self._heaps: Dict[str, List[Tuple[float, int, LambertCandidate]]] = {}
        self._all: List[LambertCandidate] = []
        self._counter = 0

    def add(self, candidate: LambertCandidate) -> None:
        self._counter += 1
        if self.top_n_per_target <= 0:
            self._all.append(candidate)
            return
        heap = self._heaps.setdefault(candidate.target, [])
        entry = (-candidate.score, self._counter, candidate)
        if len(heap) < self.top_n_per_target:
            heapq.heappush(heap, entry)
        elif candidate.score < -heap[0][0]:
            heapq.heapreplace(heap, entry)

    def candidates(self) -> List[LambertCandidate]:
        if self.top_n_per_target <= 0:
            return sorted(self._all, key=lambda c: (c.target, c.score, c.depart_et, c.tof_s))
        out: List[LambertCandidate] = []
        for heap in self._heaps.values():
            out.extend(entry[2] for entry in heap)
        return sorted(out, key=lambda c: (c.target, c.score, c.depart_et, c.tof_s))


# ---------------------------------------------------------------------------
# Main scout logic.
# ---------------------------------------------------------------------------

def infer_common_coverage(
    ephem: SpiceEphemeris,
    metadata: Mapping[str, Any],
    bodies: Sequence[str],
) -> Tuple[Optional[float], Optional[float], Dict[str, List[Tuple[float, float]]]]:
    per_body: Dict[str, List[Tuple[float, float]]] = {}
    starts: List[float] = []
    ends: List[float] = []
    for body in bodies:
        intervals = ephem.coverage_for_body(body)
        per_body[body] = intervals
        if intervals:
            starts.append(min(i[0] for i in intervals))
            ends.append(max(i[1] for i in intervals))

    if starts and ends:
        # Conservative common interval across the queried bodies.
        return max(starts), min(ends), per_body

    meta_start, meta_end = infer_coverage_from_metadata(metadata)
    return meta_start, meta_end, per_body


def resolve_departure_bounds(args: argparse.Namespace, coverage_start: Optional[float], coverage_end: Optional[float]) -> Tuple[float, float]:
    if args.depart_start_et is not None:
        depart_start = float(args.depart_start_et)
    else:
        if coverage_start is None:
            raise ValueError("coverage start is unknown; supply --depart-start-et")
        depart_start = coverage_start + args.depart_start_days * SECONDS_PER_DAY

    if args.depart_stop_et is not None:
        depart_stop = float(args.depart_stop_et)
    else:
        if coverage_start is None:
            raise ValueError("coverage start is unknown; supply --depart-stop-et")
        depart_stop = coverage_start + args.depart_stop_days * SECONDS_PER_DAY

    if coverage_start is not None and depart_start < coverage_start:
        logging.info("depart_start_et %.6f is before coverage start %.6f; clamping", depart_start, coverage_start)
        depart_start = coverage_start
    if coverage_end is not None and depart_stop > coverage_end:
        logging.info("depart_stop_et %.6f is after coverage end %.6f; clamping", depart_stop, coverage_end)
        depart_stop = coverage_end
    if depart_stop < depart_start:
        raise ValueError("departure window is empty after coverage clamping")
    return depart_start, depart_stop


def build_body_models(
    ephem: SpiceEphemeris,
    metadata: Mapping[str, Any],
    policy_root: Mapping[str, Any],
    bodies: Sequence[str],
) -> Dict[str, BodyModel]:
    out: Dict[str, BodyModel] = {}
    for body in bodies:
        policy = infer_body_policy(policy_root, body)
        out[body] = BodyModel(
            name=body,
            naif_id=ephem.body_code(body),
            mu_km3_s2=ephem.gm_km3_s2(body, metadata),
            radius_km=ephem.radius_km(body, metadata),
            policy=policy,
        )
    return out


def run_scout(args: argparse.Namespace) -> Dict[str, Any]:
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="[%(levelname)s] %(message)s",
    )

    bundle = KernelBundle(
        bsp=Path(args.bsp),
        tpc=Path(args.tpc) if args.tpc else None,
        metadata=Path(args.metadata) if args.metadata else None,
        policy=Path(args.policy) if args.policy else None,
        lsk=Path(args.lsk) if args.lsk else None,
        frame=args.frame,
    )
    metadata = load_json(bundle.metadata)
    policy_root = load_json(bundle.policy)

    with SpiceEphemeris(bundle, args.central_body) as ephem:
        all_bodies = tuple(dict.fromkeys([args.origin, *args.targets]))
        coverage_start, coverage_end, per_body_coverage = infer_common_coverage(ephem, metadata, all_bodies)
        if coverage_start is not None and coverage_end is not None:
            logging.info(
                "Common SPK coverage for scout bodies: [%.6f, %.6f] ET, %.3f days",
                coverage_start,
                coverage_end,
                (coverage_end - coverage_start) / SECONDS_PER_DAY,
            )
        else:
            logging.warning("Could not infer complete coverage; relying on SPICE exceptions during queries")

        depart_start, depart_stop = resolve_departure_bounds(args, coverage_start, coverage_end)
        depart_grid = make_grid(depart_start, depart_stop, args.depart_step_days * SECONDS_PER_DAY)
        tof_grid = make_grid(args.tof_min_days * SECONDS_PER_DAY, args.tof_max_days * SECONDS_PER_DAY, args.tof_step_days * SECONDS_PER_DAY)
        cw_values = parse_bool_values(args.cw_values)

        body_models = build_body_models(ephem, metadata, policy_root, all_bodies + (args.central_body,))
        central_mu = args.mu_central_km3_s2 or body_models.get(args.central_body, BodyModel(args.central_body, None, None, None, BodyPolicy(args.central_body))).mu_km3_s2
        if central_mu is None:
            central_mu = find_mu_in_metadata(metadata, args.central_body)
        if central_mu is None:
            raise SystemExit(
                "[FATAL] Could not resolve central-body GM in km^3/s^2. "
                "Supply --mu-central-km3-s2 explicitly."
            )
        logging.info("Using mu(%s) = %.15g km^3/s^2", args.central_body, central_mu)

        spec = MissionSpec(
            origin=args.origin,
            targets=tuple(args.targets),
            central_body=args.central_body,
            ref_frame=args.frame,
            depart_start_et=depart_start,
            depart_stop_et=depart_stop,
            depart_step_s=args.depart_step_days * SECONDS_PER_DAY,
            tof_min_s=args.tof_min_days * SECONDS_PER_DAY,
            tof_max_s=args.tof_max_days * SECONDS_PER_DAY,
            tof_step_s=args.tof_step_days * SECONDS_PER_DAY,
            max_revs=args.max_revs,
            cw_values=cw_values,
        )

        if args.dry_run:
            return {
                "schema_version": SCHEMA_VERSION,
                "dry_run": True,
                "coverage_start_et": coverage_start,
                "coverage_end_et": coverage_end,
                "mission_spec": spec.__dict__,
                "central_mu_km3_s2": central_mu,
                "body_models": summarize_body_models(body_models),
            }

        weights = ScoreWeights(
            vinf_depart=args.weight_vinf_depart,
            vinf_arrive=args.weight_vinf_arrive,
            tof_years=args.weight_tof_years,
            c3_sqrt=args.weight_c3_sqrt,
            revalidation_penalty=args.revalidation_penalty,
        )

        collector = CandidateCollector(args.top_n_per_target)
        counters: Dict[str, Dict[str, int]] = {
            target: {
                "attempted": 0,
                "lambert_fail": 0,
                "policy_skipped": 0,
                "coverage_skipped": 0,
                "filtered": 0,
                "accepted": 0,
            }
            for target in args.targets
        }

        origin_states: Dict[float, StateVector] = {}
        for dep_et in depart_grid:
            try:
                origin_states[dep_et] = ephem.state(args.origin, dep_et)
            except Exception as exc:
                logging.debug("Origin state query failed at ET %.6f: %s", dep_et, exc)

        for target in args.targets:
            policy = body_models[target].policy
            if not policy.allowed_for_global:
                logging.warning("Skipping %s: target policy does not allow global planning", target)
                counters[target]["policy_skipped"] += len(depart_grid) * len(tof_grid)
                continue
            if policy.requires_local_revalidation:
                logging.info("%s requires local revalidation; keeping as coarse/global candidate only", target)

            for dep_et in depart_grid:
                origin_state = origin_states.get(dep_et)
                if origin_state is None:
                    counters[target]["coverage_skipped"] += len(tof_grid)
                    continue

                for tof_s in tof_grid:
                    arr_et = dep_et + tof_s
                    counters[target]["attempted"] += 1
                    if coverage_end is not None and arr_et > coverage_end:
                        counters[target]["coverage_skipped"] += 1
                        continue
                    if coverage_start is not None and arr_et < coverage_start:
                        counters[target]["coverage_skipped"] += 1
                        continue

                    try:
                        target_state = ephem.state(target, arr_et)
                    except Exception as exc:
                        logging.debug("Target state query failed: %s ET %.6f: %s", target, arr_et, exc)
                        counters[target]["coverage_skipped"] += 1
                        continue

                    for cw in cw_values:
                        try:
                            solutions = solve_lambert_pykep(
                                origin_state.r_km,
                                target_state.r_km,
                                tof_s,
                                central_mu,
                                cw=cw,
                                max_revs=args.max_revs,
                            )
                        except Exception as exc:
                            logging.debug(
                                "Lambert failed for %s->%s dep %.6f tof %.3fd cw=%s: %s",
                                args.origin,
                                target,
                                dep_et,
                                tof_s / SECONDS_PER_DAY,
                                cw,
                                exc,
                            )
                            counters[target]["lambert_fail"] += 1
                            continue

                        for sc_v1, sc_v2, sol_idx in solutions:
                            vinf_dep = norm(sc_v1 - origin_state.v_km_s)
                            vinf_arr = norm(sc_v2 - target_state.v_km_s)
                            c3 = vinf_dep * vinf_dep

                            if args.max_c3 is not None and c3 > args.max_c3:
                                counters[target]["filtered"] += 1
                                continue
                            if args.max_vinf_depart is not None and vinf_dep > args.max_vinf_depart:
                                counters[target]["filtered"] += 1
                                continue
                            if args.max_vinf_arrive is not None and vinf_arr > args.max_vinf_arrive:
                                counters[target]["filtered"] += 1
                                continue
                            if args.min_tof_days is not None and tof_s / SECONDS_PER_DAY < args.min_tof_days:
                                counters[target]["filtered"] += 1
                                continue

                            score = score_candidate(
                                vinf_depart_km_s=vinf_dep,
                                vinf_arrive_km_s=vinf_arr,
                                tof_s=tof_s,
                                c3_km2_s2=c3,
                                policy=policy,
                                weights=weights,
                            )
                            leg_id = stable_leg_id([
                                SCHEMA_VERSION,
                                args.origin,
                                target,
                                args.central_body,
                                args.frame,
                                f"{dep_et:.9f}",
                                f"{arr_et:.9f}",
                                int(cw),
                                sol_idx,
                                args.max_revs,
                            ])
                            candidate = LambertCandidate(
                                leg_id=leg_id,
                                origin=args.origin,
                                target=target,
                                central_body=args.central_body,
                                ref_frame=args.frame,
                                depart_et=float(dep_et),
                                arrive_et=float(arr_et),
                                tof_s=float(tof_s),
                                cw=bool(cw),
                                solution_index=int(sol_idx),
                                max_revs=int(args.max_revs),
                                c3_km2_s2=float(c3),
                                vinf_depart_km_s=float(vinf_dep),
                                vinf_arrive_km_s=float(vinf_arr),
                                score=float(score),
                                origin_r_km=vec3_tuple(origin_state.r_km),
                                origin_v_km_s=vec3_tuple(origin_state.v_km_s),
                                target_r_km=vec3_tuple(target_state.r_km),
                                target_v_km_s=vec3_tuple(target_state.v_km_s),
                                sc_v_depart_km_s=vec3_tuple(sc_v1),
                                sc_v_arrive_km_s=vec3_tuple(sc_v2),
                                policy_grade=policy.target_grade,
                                policy_requires_revalidation=policy.requires_local_revalidation,
                                policy_reason=policy.reason,
                            )
                            collector.add(candidate)
                            counters[target]["accepted"] += 1

            logging.info(
                "%s: accepted=%d filtered=%d lambert_fail=%d coverage_skipped=%d attempted=%d",
                target,
                counters[target]["accepted"],
                counters[target]["filtered"],
                counters[target]["lambert_fail"],
                counters[target]["coverage_skipped"],
                counters[target]["attempted"],
            )

        candidates = collector.candidates()
        write_csv(Path(args.output_csv), candidates, coverage_start)
        summary = build_summary(
            args=args,
            spec=spec,
            candidates=candidates,
            counters=counters,
            coverage_start=coverage_start,
            coverage_end=coverage_end,
            per_body_coverage=per_body_coverage,
            body_models=body_models,
            central_mu=central_mu,
        )
        write_json(Path(args.output_json), summary)
        logging.info("Wrote %d candidates to %s", len(candidates), args.output_csv)
        logging.info("Wrote summary to %s", args.output_json)
        return summary


def summarize_body_models(body_models: Mapping[str, BodyModel]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for name, bm in body_models.items():
        out[name] = {
            "naif_id": bm.naif_id,
            "mu_km3_s2": bm.mu_km3_s2,
            "radius_km": bm.radius_km,
            "policy": {
                "allowed_for_global": bm.policy.allowed_for_global,
                "allowed_for_fine_targeting": bm.policy.allowed_for_fine_targeting,
                "requires_local_revalidation": bm.policy.requires_local_revalidation,
                "target_grade": bm.policy.target_grade,
                "reason": bm.policy.reason,
            },
        }
    return out


def write_csv(path: Path, candidates: Sequence[LambertCandidate], coverage_start: Optional[float]) -> None:
    ensure_parent(path)
    rows = [c.as_csv_row(coverage_start) for c in candidates]
    if rows:
        fieldnames = list(rows[0].keys())
    else:
        fieldnames = [
            "schema_version", "leg_id", "origin", "target", "central_body", "ref_frame",
            "depart_et", "arrive_et", "tof_days", "cw", "solution_index", "max_revs",
            "c3_km2_s2", "vinf_depart_km_s", "vinf_arrive_km_s", "score",
            "policy_grade", "policy_requires_revalidation", "policy_reason",
        ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    ensure_parent(path)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")


def build_summary(
    *,
    args: argparse.Namespace,
    spec: MissionSpec,
    candidates: Sequence[LambertCandidate],
    counters: Mapping[str, Mapping[str, int]],
    coverage_start: Optional[float],
    coverage_end: Optional[float],
    per_body_coverage: Mapping[str, Sequence[Tuple[float, float]]],
    body_models: Mapping[str, BodyModel],
    central_mu: float,
) -> Dict[str, Any]:
    best_by_target: Dict[str, Any] = {}
    for target in args.targets:
        target_candidates = [c for c in candidates if c.target == target]
        if target_candidates:
            best_by_target[target] = target_candidates[0].as_json_dict()
        else:
            best_by_target[target] = None

    return {
        "schema_version": SCHEMA_VERSION,
        "purpose": "single-leg SPICE/PyKEP Lambert scout for MGA global candidate generation",
        "kernel": {
            "bsp": str(args.bsp),
            "tpc": str(args.tpc) if args.tpc else None,
            "metadata": str(args.metadata) if args.metadata else None,
            "policy": str(args.policy) if args.policy else None,
            "lsk": str(args.lsk) if args.lsk else None,
            "frame": args.frame,
        },
        "coverage": {
            "common_start_et": coverage_start,
            "common_end_et": coverage_end,
            "common_span_days": None if coverage_start is None or coverage_end is None else (coverage_end - coverage_start) / SECONDS_PER_DAY,
            "per_body": {
                body: [{"start_et": a, "end_et": b, "span_days": (b - a) / SECONDS_PER_DAY} for a, b in intervals]
                for body, intervals in per_body_coverage.items()
            },
        },
        "mission_spec": {
            "origin": spec.origin,
            "targets": list(spec.targets),
            "central_body": spec.central_body,
            "ref_frame": spec.ref_frame,
            "depart_start_et": spec.depart_start_et,
            "depart_stop_et": spec.depart_stop_et,
            "depart_step_days": spec.depart_step_s / SECONDS_PER_DAY,
            "tof_min_days": spec.tof_min_s / SECONDS_PER_DAY,
            "tof_max_days": spec.tof_max_s / SECONDS_PER_DAY,
            "tof_step_days": spec.tof_step_s / SECONDS_PER_DAY,
            "max_revs": spec.max_revs,
            "cw_values": [int(x) for x in spec.cw_values],
        },
        "central_mu_km3_s2": central_mu,
        "filters": {
            "max_c3": args.max_c3,
            "max_vinf_depart": args.max_vinf_depart,
            "max_vinf_arrive": args.max_vinf_arrive,
            "top_n_per_target": args.top_n_per_target,
        },
        "score_weights": {
            "vinf_depart": args.weight_vinf_depart,
            "vinf_arrive": args.weight_vinf_arrive,
            "tof_years": args.weight_tof_years,
            "c3_sqrt": args.weight_c3_sqrt,
            "revalidation_penalty": args.revalidation_penalty,
        },
        "body_models": summarize_body_models(body_models),
        "counters": counters,
        "candidate_count_written": len(candidates),
        "best_by_target": best_by_target,
        "next_schema_hint": {
            "leg_seed_ref": list(LegSeedRef.__dataclass_fields__.keys()),
            "route_genome": list(RouteGenome.__dataclass_fields__.keys()),
            "beam_node": list(BeamNode.__dataclass_fields__.keys()),
        },
    }


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Scan single-leg Lambert windows using a synthetic SPICE BSP and write planning-grade MGA seeds.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--bsp", required=True, help="Input SPK/BSP kernel.")
    p.add_argument("--tpc", default=None, help="Text kernel with names, IDs, GM/radii if available.")
    p.add_argument("--lsk", default=None, help="Optional leap-seconds kernel. Not needed for numeric ET-only operation.")
    p.add_argument("--metadata", default=None, help="Metadata JSON produced with the synthetic SPK.")
    p.add_argument("--policy", default=None, help="Target policy JSON, e.g. target_policy_v0_1.json.")
    p.add_argument("--origin", required=True, help="Origin body name known to SPICE, e.g. Kerbin.")
    p.add_argument("--targets", nargs="+", required=True, help="Destination bodies to scan.")
    p.add_argument("--central-body", default="Sun", help="Observer/central body for heliocentric Lambert states.")
    p.add_argument("--frame", default="J2000", help="SPICE frame for state queries.")
    p.add_argument("--mu-central-km3-s2", type=float, default=None, help="Override central-body GM in km^3/s^2.")

    time = p.add_argument_group("Departure and time-of-flight grid")
    time.add_argument("--depart-start-et", type=float, default=None, help="Absolute departure start ET seconds past J2000/TDB-like epoch.")
    time.add_argument("--depart-stop-et", type=float, default=None, help="Absolute departure stop ET seconds past J2000/TDB-like epoch.")
    time.add_argument("--depart-start-days", type=float, default=0.0, help="Departure start in days from common coverage start if ET not supplied.")
    time.add_argument("--depart-stop-days", type=float, default=3650.0, help="Departure stop in days from common coverage start if ET not supplied.")
    time.add_argument("--depart-step-days", type=float, default=5.0, help="Departure grid step.")
    time.add_argument("--tof-min-days", type=float, default=60.0, help="Minimum time of flight.")
    time.add_argument("--tof-max-days", type=float, default=4000.0, help="Maximum time of flight.")
    time.add_argument("--tof-step-days", type=float, default=10.0, help="Time-of-flight grid step.")
    # Backward-compatible alias if the operator wants to use a second minimum gate.
    time.add_argument("--min-tof-days", type=float, default=None, help=argparse.SUPPRESS)

    lambert = p.add_argument_group("Lambert options")
    lambert.add_argument("--max-revs", type=int, default=0, help="Maximum Lambert revolutions passed to PyKEP.")
    lambert.add_argument(
        "--cw-values",
        nargs="+",
        default=["0"],
        help="PyKEP cw values to evaluate. Use '0 1' to scan both orientations.",
    )

    filters = p.add_argument_group("Filtering and ranking")
    filters.add_argument("--max-c3", type=float, default=None, help="Maximum C3 in km^2/s^2.")
    filters.add_argument("--max-vinf-depart", type=float, default=None, help="Maximum departure v_inf in km/s.")
    filters.add_argument("--max-vinf-arrive", type=float, default=None, help="Maximum arrival v_inf in km/s.")
    filters.add_argument("--top-n-per-target", type=int, default=250, help="Keep only the top N candidates per target; 0 keeps all accepted candidates.")
    filters.add_argument("--weight-vinf-depart", type=float, default=1.0, help="Score weight for departure v_inf.")
    filters.add_argument("--weight-vinf-arrive", type=float, default=0.35, help="Score weight for arrival v_inf.")
    filters.add_argument("--weight-tof-years", type=float, default=0.05, help="Score penalty per year of TOF.")
    filters.add_argument("--weight-c3-sqrt", type=float, default=0.0, help="Additional score weight for sqrt(C3). Usually redundant with v_inf_depart.")
    filters.add_argument("--revalidation-penalty", type=float, default=0.25, help="Score penalty for targets marked as requiring local revalidation.")

    output = p.add_argument_group("Output")
    output.add_argument("--output-csv", required=True, help="Output CSV path.")
    output.add_argument("--output-json", required=True, help="Output JSON summary path.")
    output.add_argument("--dry-run", action="store_true", help="Validate kernels/config and write only summary JSON.")
    output.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Logging level.")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.depart_step_days <= 0 or args.tof_step_days <= 0:
        parser.error("grid steps must be positive")
    if args.tof_max_days < args.tof_min_days:
        parser.error("--tof-max-days must be >= --tof-min-days")
    summary = run_scout(args)
    if args.dry_run:
        write_json(Path(args.output_json), summary)
        logging.info("Dry-run summary written to %s", args.output_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
