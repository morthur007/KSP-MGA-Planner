#!/usr/bin/env python3
"""
mga_pygmo_refine_connected_flyby_v0_1.py

Fixed-sequence connected-flyby Lambert refiner for the offline-first
KSP + Principia MGA pipeline.

Why this exists
---------------
The previous fixed-sequence refiner allowed an intermediate layover variable:

    x = [depart_day, tof_1, layover_1, tof_2, ...]

That is useful for stopover / rendezvous / parking-orbit designs, but it is not
an unpowered gravity assist. A ballistic flyby has one encounter epoch; the
incoming Lambert leg arrives at body B at the same epoch at which the outgoing
Lambert leg departs from B.

This script optimizes the connected-flyby version:

    x = [depart_day, tof_1, tof_2, ..., tof_N]

For each intermediate body, it enforces/checks:
  * same encounter epoch for incoming/outgoing legs;
  * |v_inf_in| ~= |v_inf_out|;
  * turn angle fits the physical rp_min envelope;
  * rp_required - rp_min >= margin.

It is still low-fidelity / planning-grade. It does not do B-plane targeting, it
just finds connected Lambert flyby seeds that are suitable for the next local
corrector stage.

Dependencies
------------
  conda install -c conda-forge spiceypy pykep pygmo numpy
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import math
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

SECONDS_PER_DAY = 86400.0
DAYS_PER_YEAR = 365.25
SCHEMA_VERSION = "mga_pygmo_refine_connected_flyby.v0.1"
Vec3 = Tuple[float, float, float]


@dataclass(frozen=True)
class BodyEntry:
    name: str
    mu_km3_s2: Optional[float]
    radius_km: Optional[float]
    rp_min_km: Optional[float]
    allow_flyby: bool
    raw: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LegEval:
    leg_index: int
    origin: str
    target: str
    depart_et: float
    arrive_et: float
    tof_days: float
    c3_km2_s2: float
    vinf_depart_km_s: float
    vinf_arrive_km_s: float
    leg_score: float
    origin_r_km: Vec3
    origin_v_km_s: Vec3
    target_r_km: Vec3
    target_v_km_s: Vec3
    sc_v_depart_km_s: Vec3
    sc_v_arrive_km_s: Vec3
    cw: bool
    solution_index: int

    def vinf_depart_vec(self) -> Vec3:
        return vec_sub(self.sc_v_depart_km_s, self.origin_v_km_s)

    def vinf_arrive_vec(self) -> Vec3:
        return vec_sub(self.sc_v_arrive_km_s, self.target_v_km_s)


@dataclass(frozen=True)
class FlybyEval:
    flyby_index: int
    body: str
    encounter_et: float
    vinf_in_km_s: float
    vinf_out_km_s: float
    vinf_mag_mismatch_km_s: float
    turn_angle_deg: float
    mu_km3_s2: Optional[float]
    rp_min_km: Optional[float]
    vinf_eff_km_s: Optional[float]
    turn_angle_max_deg: Optional[float]
    turn_angle_margin_deg: Optional[float]
    rp_required_km: Optional[float]
    rp_margin_km: Optional[float]
    ok: bool
    status: str
    transition_score: float


@dataclass(frozen=True)
class ConnectedFlybyRoute:
    schema_version: str
    refined_id: str
    source_route_id: str
    sequence: Tuple[str, ...]
    central_body: str
    ref_frame: str
    objective: float
    source_nominal_score: Optional[float]
    improvement: Optional[float]
    depart_et: float
    arrive_et: float
    total_tof_days: float
    decision_vector: Tuple[float, ...]
    decision_labels: Tuple[str, ...]
    leg_evals: Tuple[LegEval, ...]
    flyby_evals: Tuple[FlybyEval, ...]
    max_vinf_depart_km_s: float
    max_vinf_arrive_km_s: float
    max_vinf_mismatch_km_s: float
    max_turn_angle_deg: float
    min_rp_margin_km: Optional[float]
    min_turn_angle_margin_deg: Optional[float]
    valid: bool
    status: str
    optimizer: Mapping[str, Any]


# ---------------------------------------------------------------------------
# Math / JSON helpers
# ---------------------------------------------------------------------------


def vec_sub(a: Sequence[float], b: Sequence[float]) -> Vec3:
    return (float(a[0]) - float(b[0]), float(a[1]) - float(b[1]), float(a[2]) - float(b[2]))


def vec_norm(a: Sequence[float]) -> float:
    return math.sqrt(float(a[0]) ** 2 + float(a[1]) ** 2 + float(a[2]) ** 2)


def vec_dot(a: Sequence[float], b: Sequence[float]) -> float:
    return float(a[0]) * float(b[0]) + float(a[1]) * float(b[1]) + float(a[2]) * float(b[2])


def angle_deg(a: Sequence[float], b: Sequence[float]) -> float:
    na = vec_norm(a)
    nb = vec_norm(b)
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    c = max(-1.0, min(1.0, vec_dot(a, b) / (na * nb)))
    return math.degrees(math.acos(c))


def max_turn_angle_deg(mu_km3_s2: float, rp_km: float, vinf_km_s: float) -> float:
    if mu_km3_s2 <= 0.0 or rp_km <= 0.0 or vinf_km_s <= 0.0:
        return 0.0
    arg = 1.0 / (rp_km * vinf_km_s * vinf_km_s / mu_km3_s2 + 1.0)
    arg = max(-1.0, min(1.0, arg))
    return math.degrees(2.0 * math.asin(arg))


def required_rp_km(mu_km3_s2: float, vinf_km_s: float, turn_angle_deg: float) -> Optional[float]:
    if mu_km3_s2 <= 0.0 or vinf_km_s <= 0.0:
        return None
    if turn_angle_deg <= 0.0:
        return math.inf
    if turn_angle_deg >= 180.0:
        return 0.0
    s = math.sin(math.radians(turn_angle_deg) / 2.0)
    if s <= 0.0:
        return math.inf
    return mu_km3_s2 / (vinf_km_s * vinf_km_s) * (1.0 / s - 1.0)


def stable_id(prefix: str, payload: Mapping[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=True, separators=(",", ":"), default=str)
    return f"{prefix}_{hashlib.sha1(raw.encode('utf-8')).hexdigest()[:16]}"


def opt_float(x: Any) -> Optional[float]:
    if x is None or x == "":
        return None
    try:
        y = float(x)
    except (TypeError, ValueError):
        return None
    return y if math.isfinite(y) else None


def finite(x: Any, default: float = 0.0) -> float:
    y = opt_float(x)
    return default if y is None else y


def opt_bool(x: Any, default: bool = False) -> bool:
    if isinstance(x, bool):
        return x
    if x is None:
        return default
    s = str(x).strip().lower()
    if s in {"1", "true", "yes", "y", "ok"}:
        return True
    if s in {"0", "false", "no", "n", "bad"}:
        return False
    return default


def json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, tuple):
        return list(obj)
    return str(obj)


# ---------------------------------------------------------------------------
# Imports / SPICE / PyKEP
# ---------------------------------------------------------------------------


def import_spiceypy():
    try:
        return importlib.import_module("spiceypy")
    except ImportError as exc:
        raise RuntimeError("spiceypy is required: conda install -c conda-forge spiceypy") from exc


def import_pykep():
    for name in ("pykep", "kep3"):
        try:
            return importlib.import_module(name)
        except ImportError:
            continue
    raise RuntimeError("pykep/kep3 is required: conda install -c conda-forge pykep")


def import_pygmo():
    try:
        return importlib.import_module("pygmo")
    except ImportError as exc:
        raise RuntimeError("pygmo is required: conda install -c conda-forge pygmo") from exc


def furnish_kernels(spice: Any, paths: Sequence[Path]) -> None:
    for p in paths:
        if p is None:
            continue
        if not Path(p).exists():
            raise FileNotFoundError(p)
        spice.furnsh(str(p))


def parse_spice_state(spice: Any, target: str, et: float, ref_frame: str, central_body: str) -> Tuple[Vec3, Vec3]:
    state, _lt = spice.spkezr(target, float(et), ref_frame, "NONE", central_body)
    return (float(state[0]), float(state[1]), float(state[2])), (float(state[3]), float(state[4]), float(state[5]))


def pykep_lambert_solutions(pk: Any, r1: Vec3, r2: Vec3, tof_s: float, mu: float, max_revs: int) -> List[Tuple[Vec3, Vec3, bool, int]]:
    out: List[Tuple[Vec3, Vec3, bool, int]] = []
    for cw in (False, True):
        try:
            lp = pk.lambert_problem(list(r1), list(r2), float(tof_s), float(mu), cw, int(max_revs))
        except TypeError:
            try:
                lp = pk.lambert_problem(r1=list(r1), r2=list(r2), tof=float(tof_s), mu=float(mu), cw=cw, max_revs=int(max_revs))
            except Exception:
                continue
        except Exception:
            continue
        try:
            v1s = lp.get_v1(); v2s = lp.get_v2()
        except Exception:
            continue
        for idx, (v1, v2) in enumerate(zip(v1s, v2s)):
            if len(v1) == 3 and len(v2) == 3:
                out.append(((float(v1[0]), float(v1[1]), float(v1[2])), (float(v2[0]), float(v2[1]), float(v2[2])), cw, idx))
    return out


# ---------------------------------------------------------------------------
# Input readers
# ---------------------------------------------------------------------------


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_no}: {exc}") from exc
            if isinstance(obj, Mapping):
                rows.append(dict(obj))
    return rows


def load_body_catalog(path: Path) -> Dict[str, BodyEntry]:
    root = read_json(path)
    raw_bodies = root.get("bodies", root)
    if not isinstance(raw_bodies, Mapping):
        raise ValueError(f"{path} does not contain a body mapping")
    out: Dict[str, BodyEntry] = {}
    for name, raw in raw_bodies.items():
        if not isinstance(raw, Mapping):
            continue
        out[str(name)] = BodyEntry(
            name=str(raw.get("name", name)),
            mu_km3_s2=opt_float(raw.get("mu_km3_s2", raw.get("gm_km3_s2"))),
            radius_km=opt_float(raw.get("radius_km")),
            rp_min_km=opt_float(raw.get("rp_min_km")),
            allow_flyby=opt_bool(raw.get("allow_flyby", True), True),
            raw=raw,
        )
    return out


def route_sequence(record: Mapping[str, Any]) -> Tuple[str, ...]:
    seq = record.get("sequence")
    if isinstance(seq, str):
        return tuple(x.strip() for x in seq.replace("->", ",").split(",") if x.strip())
    if isinstance(seq, Sequence) and not isinstance(seq, (str, bytes)):
        return tuple(str(x) for x in seq)
    route = record.get("route", {}) if isinstance(record.get("route", {}), Mapping) else {}
    seq = route.get("sequence")
    if isinstance(seq, str):
        return tuple(x.strip() for x in seq.replace("->", ",").split(",") if x.strip())
    if isinstance(seq, Sequence) and not isinstance(seq, (str, bytes)):
        return tuple(str(x) for x in seq)
    legs = record.get("legs", record.get("leg_evals", []))
    if isinstance(legs, list) and legs:
        bodies = [str(legs[0].get("origin", ""))]
        for leg in legs:
            if isinstance(leg, Mapping):
                bodies.append(str(leg.get("target", "")))
        return tuple(x for x in bodies if x)
    return ()


def source_score(record: Mapping[str, Any]) -> Optional[float]:
    for key in ("objective", "robust_score", "source_nominal_score"):
        x = opt_float(record.get(key))
        if x is not None:
            return x
    route = record.get("route", {}) if isinstance(record.get("route", {}), Mapping) else {}
    return opt_float(route.get("nominal_score", route.get("score")))


def source_route_id(record: Mapping[str, Any]) -> str:
    for key in ("refined_id", "route_id", "source_route_id"):
        if record.get(key):
            return str(record.get(key))
    route = record.get("route", {}) if isinstance(record.get("route", {}), Mapping) else {}
    return str(route.get("route_id", ""))


def infer_coverage_start(record: Mapping[str, Any]) -> Optional[float]:
    # Best case: refined record has depart_et and decision_vector[0] = depart day from coverage start.
    labels = record.get("decision_labels", [])
    vector = record.get("decision_vector", [])
    dep_et = opt_float(record.get("depart_et"))
    if dep_et is not None and isinstance(vector, Sequence) and vector:
        dep_day = None
        if isinstance(labels, Sequence):
            for i, lab in enumerate(labels):
                if str(lab) == "depart_day" and i < len(vector):
                    dep_day = opt_float(vector[i])
                    break
        if dep_day is None:
            dep_day = opt_float(vector[0])
        if dep_day is not None:
            return dep_et - dep_day * SECONDS_PER_DAY

    legs = record.get("legs", [])
    if isinstance(legs, list) and legs:
        dep_et = opt_float(legs[0].get("depart_et"))
        dep_days = opt_float(legs[0].get("depart_days_from_coverage_start"))
        if dep_et is not None and dep_days is not None:
            return dep_et - dep_days * SECONDS_PER_DAY
    leg_evals = record.get("leg_evals", [])
    if isinstance(leg_evals, list) and leg_evals:
        dep_et = opt_float(leg_evals[0].get("depart_et"))
        if dep_et is not None and isinstance(vector, Sequence) and vector:
            dep_day = opt_float(vector[0])
            if dep_day is not None:
                return dep_et - dep_day * SECONDS_PER_DAY
    return None


def connected_seed(record: Mapping[str, Any], coverage_start: float) -> Tuple[List[float], List[str]]:
    # x = [depart_day, tof_1, tof_2, ...]. Layovers are intentionally ignored.
    labels: List[str] = ["depart_day"]
    x: List[float] = []
    dep_et = opt_float(record.get("depart_et"))
    if dep_et is None:
        legs0 = record.get("legs", record.get("leg_evals", []))
        if isinstance(legs0, list) and legs0:
            dep_et = opt_float(legs0[0].get("depart_et"))
    if dep_et is None:
        labels_raw = record.get("decision_labels", [])
        vec_raw = record.get("decision_vector", [])
        if isinstance(labels_raw, Sequence) and isinstance(vec_raw, Sequence):
            for i, lab in enumerate(labels_raw):
                if str(lab) == "depart_day" and i < len(vec_raw):
                    x.append(finite(vec_raw[i])); break
        if not x:
            raise ValueError("Could not infer depart_day/depart_et from record")
    else:
        x.append((dep_et - coverage_start) / SECONDS_PER_DAY)

    legs = record.get("leg_evals", record.get("legs", []))
    if isinstance(legs, list) and legs:
        for i, leg in enumerate(legs):
            tof = opt_float(leg.get("tof_days"))
            if tof is None:
                dep = opt_float(leg.get("depart_et")); arr = opt_float(leg.get("arrive_et"))
                if dep is not None and arr is not None:
                    tof = (arr - dep) / SECONDS_PER_DAY
            if tof is None:
                tof = opt_float(leg.get("tof_s"))
                if tof is not None:
                    tof /= SECONDS_PER_DAY
            if tof is None:
                raise ValueError(f"Could not infer tof for leg {i}")
            x.append(float(tof)); labels.append(f"tof_{i+1}_days")
        return x, labels

    # Fallback: old vector [depart, tof1, lay1, tof2,...]
    vector = record.get("decision_vector", [])
    dlabels = record.get("decision_labels", [])
    if isinstance(vector, Sequence) and isinstance(dlabels, Sequence):
        for i, lab in enumerate(dlabels):
            if str(lab).startswith("tof_") and i < len(vector):
                x.append(finite(vector[i])); labels.append(str(lab))
    if len(x) < 2:
        raise ValueError("Could not infer connected seed vector")
    return x, labels


def select_records(records: Sequence[Mapping[str, Any]], args: argparse.Namespace) -> List[Mapping[str, Any]]:
    wanted = tuple(x.strip() for x in args.sequence.replace("->", ",").split(",") if x.strip()) if args.sequence else None
    out: List[Mapping[str, Any]] = []
    for r in records:
        seq = route_sequence(r)
        if not seq:
            continue
        if wanted and seq != wanted:
            continue
        if args.valid_only and not opt_bool(r.get("valid", True), True):
            continue
        out.append(r)
    out.sort(key=lambda r: source_score(r) if source_score(r) is not None else 1e99)
    if args.route_index is not None:
        if args.route_index < 0 or args.route_index >= len(out):
            raise IndexError(f"--route-index {args.route_index} outside selected count {len(out)}")
        return [out[args.route_index]]
    return out[: args.max_routes]


# ---------------------------------------------------------------------------
# Central mu loader
# ---------------------------------------------------------------------------


def find_mu_in_metadata(obj: Any, central_body: str) -> Optional[float]:
    keys = {
        "central_mu_km3_s2", "mu_central_km3_s2", "central_gm_km3_s2",
        "gm_km3_s2", "mu_km3_s2",
    }
    if isinstance(obj, Mapping):
        for k, v in obj.items():
            kl = str(k).lower()
            if kl in keys or (central_body.lower() in kl and ("mu" in kl or "gm" in kl)):
                val = opt_float(v)
                if val and val > 0:
                    return val
        for k, v in obj.items():
            if str(k).lower() in {central_body.lower(), "sun", "kerbol"}:
                found = find_mu_in_metadata(v, central_body)
                if found:
                    return found
        for v in obj.values():
            found = find_mu_in_metadata(v, central_body)
            if found:
                return found
    elif isinstance(obj, list):
        for v in obj:
            found = find_mu_in_metadata(v, central_body)
            if found:
                return found
    return None


def load_central_mu(spice: Any, metadata: Optional[Path], central_body: str, explicit: Optional[float]) -> float:
    if explicit is not None and explicit > 0:
        return float(explicit)
    for name in [central_body, "Kerbol" if central_body.lower() == "sun" else "Sun"]:
        try:
            _dim, values = spice.bodvrd(name, "GM", 1)
            mu = float(values[0])
            if mu > 0:
                return mu
        except Exception:
            pass
    if metadata and metadata.exists():
        found = find_mu_in_metadata(read_json(metadata), central_body)
        if found and found > 0:
            return found
    raise RuntimeError("Could not determine central mu. Use --mu-central-km3-s2.")


# ---------------------------------------------------------------------------
# Evaluator / PyGMO UDP
# ---------------------------------------------------------------------------


class ConnectedFlybyEvaluator:
    def __init__(self, *, spice: Any, pykep: Any, sequence: Sequence[str], central_body: str,
                 ref_frame: str, central_mu: float, coverage_start_et: float,
                 body_catalog: Mapping[str, BodyEntry], args: argparse.Namespace) -> None:
        self.spice = spice
        self.pk = pykep
        self.sequence = tuple(sequence)
        self.central_body = central_body
        self.ref_frame = ref_frame
        self.central_mu = central_mu
        self.coverage_start_et = coverage_start_et
        self.body_catalog = body_catalog
        self.args = args

    def __getstate__(self) -> Dict[str, Any]:
        state = dict(self.__dict__)
        state["spice"] = None
        state["pk"] = None
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self.__dict__.update(state)
        self.spice = import_spiceypy()
        self.pk = import_pykep()

    def decode_times(self, x: Sequence[float]) -> Tuple[List[float], List[float]]:
        n_legs = len(self.sequence) - 1
        if len(x) != 1 + n_legs:
            raise ValueError(f"Expected {1+n_legs} variables [depart_day,tof...], got {len(x)}")
        depart_ets: List[float] = []
        arrive_ets: List[float] = []
        current = self.coverage_start_et + float(x[0]) * SECONDS_PER_DAY
        idx = 1
        for _i in range(n_legs):
            tof = float(x[idx]); idx += 1
            depart_ets.append(current)
            arr = current + tof * SECONDS_PER_DAY
            arrive_ets.append(arr)
            current = arr  # connected flyby: next leg departs at same epoch
        return depart_ets, arrive_ets

    def effective_vinf(self, vinf_in: float, vinf_out: float) -> float:
        mode = self.args.flyby_vinf_mode
        if mode == "incoming":
            return vinf_in
        if mode == "outgoing":
            return vinf_out
        if mode in {"max", "conservative"}:
            return max(vinf_in, vinf_out)
        if mode == "rms":
            return math.sqrt(0.5 * (vinf_in * vinf_in + vinf_out * vinf_out))
        return 0.5 * (vinf_in + vinf_out)

    def evaluate(self, x: Sequence[float], full: bool = False) -> Tuple[float, Optional[ConnectedFlybyRoute], str]:
        penalty = 0.0
        status_parts: List[str] = []
        try:
            depart_ets, arrive_ets = self.decode_times(x)
        except Exception as exc:
            return 1e99, None, f"decode_error:{exc}"
        if any(arr <= dep for dep, arr in zip(depart_ets, arrive_ets)):
            return 1e99, None, "nonpositive_tof"

        leg_evals: List[LegEval] = []
        for i, (origin, target) in enumerate(zip(self.sequence[:-1], self.sequence[1:])):
            dep, arr = depart_ets[i], arrive_ets[i]
            tof_s = arr - dep
            tof_days = tof_s / SECONDS_PER_DAY
            try:
                r1, v_origin = parse_spice_state(self.spice, origin, dep, self.ref_frame, self.central_body)
                r2, v_target = parse_spice_state(self.spice, target, arr, self.ref_frame, self.central_body)
            except Exception as exc:
                return 1e98, None, f"spice_state_error:{origin}->{target}:{exc}"
            sols = pykep_lambert_solutions(self.pk, r1, r2, tof_s, self.central_mu, self.args.max_revs)
            if not sols:
                return 1e97, None, f"lambert_fail:{origin}->{target}"
            best: Optional[LegEval] = None
            best_score = math.inf
            for v_dep, v_arr, cw, sol_idx in sols:
                vinf_dep = vec_norm(vec_sub(v_dep, v_origin))
                vinf_arr = vec_norm(vec_sub(v_arr, v_target))
                c3 = vinf_dep * vinf_dep
                leg_score = (self.args.vinf_depart_weight * vinf_dep +
                             self.args.vinf_arrive_weight * vinf_arr +
                             self.args.c3_weight * c3 +
                             self.args.tof_weight * (tof_days / DAYS_PER_YEAR))
                if leg_score < best_score:
                    best_score = leg_score
                    best = LegEval(i, origin, target, dep, arr, tof_days, c3, vinf_dep, vinf_arr, leg_score,
                                   r1, v_origin, r2, v_target, v_dep, v_arr, cw, sol_idx)
            assert best is not None
            leg_evals.append(best)

        flybys: List[FlybyEval] = []
        for i in range(len(leg_evals) - 1):
            prev = leg_evals[i]
            nxt = leg_evals[i + 1]
            body = prev.target
            vin_vec_in = prev.vinf_arrive_vec()
            vin_vec_out = nxt.vinf_depart_vec()
            vin_in = vec_norm(vin_vec_in)
            vin_out = vec_norm(vin_vec_out)
            mismatch = abs(vin_out - vin_in)
            turn = angle_deg(vin_vec_in, vin_vec_out)
            entry = self.body_catalog.get(body)
            ok = True
            status = "ok"
            mu = rp_min = vin_eff = turn_max = turn_margin = rp_req = rp_margin = None
            if entry is None:
                ok = False; status = "missing_body_catalog"
            elif not entry.allow_flyby:
                ok = False; status = "flyby_not_allowed"; mu = entry.mu_km3_s2; rp_min = entry.rp_min_km
            elif entry.mu_km3_s2 is None or entry.rp_min_km is None:
                ok = False; status = "missing_mu_or_rp_min"; mu = entry.mu_km3_s2; rp_min = entry.rp_min_km
            else:
                mu = entry.mu_km3_s2
                rp_min = entry.rp_min_km
                vin_eff = self.effective_vinf(vin_in, vin_out)
                turn_max = max_turn_angle_deg(mu, rp_min, vin_eff)
                turn_margin = turn_max - turn
                rp_req = required_rp_km(mu, vin_eff, turn)
                # Positive means altitude margin above minimum; required rp must be >= rp_min.
                # If required_rp is inf for zero turn, treat it as a very safe distant flyby.
                if rp_req is None:
                    rp_margin = None
                elif math.isinf(rp_req):
                    rp_margin = math.inf
                else:
                    rp_margin = rp_req - rp_min
                if turn_margin < -1e-12:
                    ok = False; status = "turn_angle_exceeds_physical_envelope"
                if rp_margin is not None and rp_margin < self.args.min_rp_margin_km:
                    ok = False; status = "rp_margin"
            if mismatch > self.args.max_vinf_mismatch_km_s:
                ok = False
                status = "vinf_mismatch" if status == "ok" else status + ";vinf_mismatch"

            if not ok:
                penalty += self.args.hard_constraint_penalty
                status_parts.append(status)
            # Smooth penalties steer the optimizer away from cliff edges.
            penalty += self.args.vinf_mismatch_penalty_weight * max(0.0, mismatch - self.args.vinf_mismatch_soft_km_s) ** 2
            if rp_margin is not None and math.isfinite(rp_margin):
                penalty += self.args.rp_margin_penalty_weight * max(0.0, self.args.rp_soft_margin_km - rp_margin) ** 2 / max(self.args.rp_soft_margin_km, 1.0) ** 2
            if turn_margin is not None and math.isfinite(turn_margin):
                penalty += self.args.turn_margin_penalty_weight * max(0.0, self.args.turn_soft_margin_deg - turn_margin) ** 2 / max(self.args.turn_soft_margin_deg, 1.0) ** 2

            transition_score = (self.args.vinf_mismatch_weight * mismatch +
                                self.args.turn_angle_weight * (turn / 180.0))
            if rp_margin is not None and math.isfinite(rp_margin) and self.args.flyby_margin_weight > 0:
                transition_score += self.args.flyby_margin_weight / (1.0 + max(0.0, rp_margin) / max(float(rp_min or 1.0), 1.0))
            flybys.append(FlybyEval(i, body, prev.arrive_et, vin_in, vin_out, mismatch, turn,
                                    mu, rp_min, vin_eff, turn_max, turn_margin, rp_req, rp_margin,
                                    ok, status, transition_score))

        objective = sum(l.leg_score for l in leg_evals) + sum(f.transition_score for f in flybys) + penalty
        if not full:
            return objective, None, ";".join(status_parts) if status_parts else "ok"

        rp_margins = [f.rp_margin_km for f in flybys if f.rp_margin_km is not None and math.isfinite(f.rp_margin_km)]
        turn_margins = [f.turn_angle_margin_deg for f in flybys if f.turn_angle_margin_deg is not None and math.isfinite(f.turn_angle_margin_deg)]
        valid = penalty < self.args.hard_constraint_penalty * 0.5 and all(f.ok for f in flybys)
        status = "ok" if valid else (";".join(status_parts) if status_parts else "penalized")
        rid = stable_id("connected", {"seq": self.sequence, "x": [round(float(v), 9) for v in x], "objective": round(objective, 9)})
        route = ConnectedFlybyRoute(
            schema_version=SCHEMA_VERSION,
            refined_id=rid,
            source_route_id="",
            sequence=self.sequence,
            central_body=self.central_body,
            ref_frame=self.ref_frame,
            objective=objective,
            source_nominal_score=None,
            improvement=None,
            depart_et=leg_evals[0].depart_et,
            arrive_et=leg_evals[-1].arrive_et,
            total_tof_days=(leg_evals[-1].arrive_et - leg_evals[0].depart_et) / SECONDS_PER_DAY,
            decision_vector=tuple(float(v) for v in x),
            decision_labels=tuple(),
            leg_evals=tuple(leg_evals),
            flyby_evals=tuple(flybys),
            max_vinf_depart_km_s=max((l.vinf_depart_km_s for l in leg_evals), default=0.0),
            max_vinf_arrive_km_s=max((l.vinf_arrive_km_s for l in leg_evals), default=0.0),
            max_vinf_mismatch_km_s=max((f.vinf_mag_mismatch_km_s for f in flybys), default=0.0),
            max_turn_angle_deg=max((f.turn_angle_deg for f in flybys), default=0.0),
            min_rp_margin_km=min(rp_margins) if rp_margins else None,
            min_turn_angle_margin_deg=min(turn_margins) if turn_margins else None,
            valid=valid,
            status=status,
            optimizer={},
        )
        return objective, route, status


class PygmoUDP:
    def __init__(self, evaluator: ConnectedFlybyEvaluator, lb: Sequence[float], ub: Sequence[float]) -> None:
        self.evaluator = evaluator
        self.lb = list(lb)
        self.ub = list(ub)

    def fitness(self, x: Sequence[float]) -> List[float]:
        f, _r, _s = self.evaluator.evaluate(x, full=False)
        return [float(f if math.isfinite(f) else 1e99)]

    def get_bounds(self) -> Tuple[List[float], List[float]]:
        return self.lb, self.ub

    def get_name(self) -> str:
        return "KSP-Principia connected-flyby fixed-sequence Lambert refiner"


# ---------------------------------------------------------------------------
# Optimization
# ---------------------------------------------------------------------------


def make_bounds(seed_x: Sequence[float], labels: Sequence[str], args: argparse.Namespace) -> Tuple[List[float], List[float]]:
    lb: List[float] = []
    ub: List[float] = []
    for val, lab in zip(seed_x, labels):
        if lab == "depart_day":
            lb.append(max(args.depart_min_day, val - args.depart_window_days))
            ub.append(min(args.depart_max_day, val + args.depart_window_days))
        elif lab.startswith("tof_"):
            lb.append(max(args.tof_min_days, val - args.tof_window_days))
            ub.append(min(args.tof_max_days, val + args.tof_window_days))
        else:
            lb.append(val); ub.append(val)
    for i, (a, b) in enumerate(zip(lb, ub)):
        if b < a:
            raise ValueError(f"Invalid bounds for {labels[i]}: [{a}, {b}]")
        if abs(b - a) < 1e-9:
            ub[i] = a + 1e-9
    return lb, ub


def build_algorithm(pg: Any, args: argparse.Namespace):
    name = args.algorithm.lower()
    if name == "de":
        return pg.algorithm(pg.de(gen=args.generations, F=args.de_f, CR=args.de_cr, variant=args.de_variant))
    if name == "sade":
        return pg.algorithm(pg.sade(gen=args.generations))
    if name == "pso":
        return pg.algorithm(pg.pso(gen=args.generations))
    if name == "bee_colony":
        return pg.algorithm(pg.bee_colony(gen=args.generations))
    raise ValueError(f"Unsupported algorithm: {args.algorithm}")


def optimize_one(pg: Any, evaluator: ConnectedFlybyEvaluator, source: Mapping[str, Any], seed_x: Sequence[float], labels: Sequence[str], args: argparse.Namespace) -> ConnectedFlybyRoute:
    lb, ub = make_bounds(seed_x, labels, args)
    prob = pg.problem(PygmoUDP(evaluator, lb, ub))
    algo = build_algorithm(pg, args)
    best_x = [float(v) for v in seed_x]
    best_f, _r, seed_status = evaluator.evaluate(best_x, full=False)
    runs = [{"run": "seed", "objective": float(best_f), "status": seed_status}]
    for run in range(args.runs):
        seed = None if args.seed is None else args.seed + run
        try:
            pop = pg.population(prob, size=args.population, seed=seed) if seed is not None else pg.population(prob, size=args.population)
        except TypeError:
            pop = pg.population(prob, args.population)
        try:
            pop.set_x(0, list(seed_x))
        except Exception:
            pass
        pop = algo.evolve(pop)
        f = float(pop.champion_f[0])
        x = [float(v) for v in pop.champion_x]
        status = evaluator.evaluate(x, full=False)[2]
        runs.append({"run": run, "objective": f, "status": status})
        if f < best_f:
            best_f = f; best_x = x
    obj, route, status = evaluator.evaluate(best_x, full=True)
    if route is None:
        raise RuntimeError(f"Full evaluation failed: {status}")
    src_score = source_score(source)
    payload = dict(route.__dict__)
    payload.update({
        "source_route_id": source_route_id(source),
        "source_nominal_score": src_score,
        "improvement": None if src_score is None else src_score - obj,
        "decision_labels": tuple(labels),
        "optimizer": {
            "library": "pygmo",
            "algorithm": args.algorithm,
            "generations": args.generations,
            "population": args.population,
            "runs": args.runs,
            "seed": args.seed,
            "connected_flyby": True,
            "bounds": {lab: [a, b] for lab, a, b in zip(labels, lb, ub)},
            "source_seed_objective": float(best_f),
            "run_summaries": runs,
        },
    })
    return ConnectedFlybyRoute(**payload)


def _worker(payload: Mapping[str, Any]) -> Tuple[int, ConnectedFlybyRoute]:
    idx = int(payload["idx"])
    args = argparse.Namespace(**payload["args"])
    record = payload["record"]
    catalog = payload["catalog"]
    central_mu = float(payload["central_mu"])
    spice = import_spiceypy(); pk = import_pykep(); pg = import_pygmo()
    furnish_kernels(spice, [args.tpc, args.bsp])
    try:
        seq = route_sequence(record)
        coverage_start = infer_coverage_start(record)
        if coverage_start is None:
            raise RuntimeError("Could not infer coverage_start")
        seed_x, labels = connected_seed(record, coverage_start)
        evaluator = ConnectedFlybyEvaluator(spice=spice, pykep=pk, sequence=seq, central_body=args.central_body,
                                            ref_frame=args.ref_frame, central_mu=central_mu,
                                            coverage_start_et=coverage_start, body_catalog=catalog, args=args)
        route = optimize_one(pg, evaluator, record, seed_x, labels, args)
        return idx, route
    finally:
        try:
            spice.kclear()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Output / CLI
# ---------------------------------------------------------------------------


def fmt(x: Any, digits: int = 9) -> str:
    y = opt_float(x)
    if y is None:
        return ""
    if math.isinf(y):
        return "inf" if y > 0 else "-inf"
    return f"{y:.{digits}f}"


def write_jsonl(path: Path, rows: Sequence[ConnectedFlybyRoute]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(asdict(r), ensure_ascii=False, separators=(",", ":"), default=json_default) + "\n")


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=json_default)
        f.write("\n")


def write_csv(path: Path, routes: Sequence[ConnectedFlybyRoute]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "rank", "refined_id", "source_route_id", "sequence", "valid", "status", "objective",
        "source_nominal_score", "improvement", "depart_day", "total_tof_days",
        "max_vinf_mismatch_km_s", "max_turn_angle_deg", "min_rp_margin_km", "min_turn_angle_margin_deg",
        "leg_tof_days", "vinf_depart_by_leg_km_s", "vinf_arrive_by_leg_km_s",
        "flyby_bodies", "flyby_statuses", "rp_margins_km", "turn_margins_deg", "decision_vector",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for i, r in enumerate(routes, start=1):
            w.writerow({
                "rank": i,
                "refined_id": r.refined_id,
                "source_route_id": r.source_route_id,
                "sequence": "->".join(r.sequence),
                "valid": int(r.valid),
                "status": r.status,
                "objective": fmt(r.objective, 9),
                "source_nominal_score": fmt(r.source_nominal_score, 9),
                "improvement": fmt(r.improvement, 9),
                "depart_day": fmt(r.decision_vector[0] if r.decision_vector else None, 6),
                "total_tof_days": fmt(r.total_tof_days, 6),
                "max_vinf_mismatch_km_s": fmt(r.max_vinf_mismatch_km_s, 9),
                "max_turn_angle_deg": fmt(r.max_turn_angle_deg, 6),
                "min_rp_margin_km": fmt(r.min_rp_margin_km, 6),
                "min_turn_angle_margin_deg": fmt(r.min_turn_angle_margin_deg, 6),
                "leg_tof_days": ";".join(fmt(l.tof_days, 6) for l in r.leg_evals),
                "vinf_depart_by_leg_km_s": ";".join(fmt(l.vinf_depart_km_s, 6) for l in r.leg_evals),
                "vinf_arrive_by_leg_km_s": ";".join(fmt(l.vinf_arrive_km_s, 6) for l in r.leg_evals),
                "flyby_bodies": ";".join(fb.body for fb in r.flyby_evals),
                "flyby_statuses": ";".join(fb.status for fb in r.flyby_evals),
                "rp_margins_km": ";".join(fmt(fb.rp_margin_km, 3) for fb in r.flyby_evals),
                "turn_margins_deg": ";".join(fmt(fb.turn_angle_margin_deg, 3) for fb in r.flyby_evals),
                "decision_vector": ";".join(fmt(x, 9) for x in r.decision_vector),
            })


def short_route(r: ConnectedFlybyRoute) -> Dict[str, Any]:
    return {
        "refined_id": r.refined_id,
        "sequence": "->".join(r.sequence),
        "valid": r.valid,
        "status": r.status,
        "objective": r.objective,
        "source_nominal_score": r.source_nominal_score,
        "improvement": r.improvement,
        "total_tof_days": r.total_tof_days,
        "max_vinf_mismatch_km_s": r.max_vinf_mismatch_km_s,
        "max_turn_angle_deg": r.max_turn_angle_deg,
        "min_rp_margin_km": r.min_rp_margin_km,
        "decision": dict(zip(r.decision_labels, r.decision_vector)),
    }


def print_report(routes: Sequence[ConnectedFlybyRoute]) -> None:
    print("=" * 80)
    print("MGA PYGMO CONNECTED-FLYBY REFINER V0.1")
    print("=" * 80)
    print(f"Refined routes: {len(routes)}")
    print(f"Valid routes:   {sum(1 for r in routes if r.valid)}")
    print("\nTop connected-flyby routes:")
    for i, r in enumerate(routes[:10], start=1):
        imp = "n/a" if r.improvement is None else f"{r.improvement:+.4f}"
        margin = "n/a" if r.min_rp_margin_km is None else f"{r.min_rp_margin_km:.1f} km"
        print(
            f"{i:2d}. {' -> '.join(r.sequence)} | valid={r.valid} | obj={r.objective:.4f} | "
            f"impr={imp} | TOF={r.total_tof_days:.1f} d | "
            f"v∞ mismatch={r.max_vinf_mismatch_km_s*1000:.2f} m/s | "
            f"turn={r.max_turn_angle_deg:.2f}° | rp_margin={margin} | status={r.status}"
        )
    print("=" * 80)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Refine fixed sequence as a true connected unpowered flyby chain.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--bsp", required=True, type=Path)
    p.add_argument("--tpc", required=True, type=Path)
    p.add_argument("--metadata", type=Path, default=None)
    p.add_argument("--body-catalog", required=True, type=Path)
    p.add_argument("--routes-jsonl", required=True, type=Path, help="Selected/refined/beam routes JSONL")
    p.add_argument("--sequence", required=True, help="Fixed sequence, e.g. Kerbin,Duna,Jool")
    p.add_argument("--central-body", default="Sun")
    p.add_argument("--mu-central-km3-s2", type=float, default=None)
    p.add_argument("--ref-frame", default="J2000")
    p.add_argument("--max-revs", type=int, default=0)
    p.add_argument("--valid-only", action="store_true")
    p.add_argument("--max-routes", type=int, default=20)
    p.add_argument("--route-index", type=int, default=None)

    p.add_argument("--depart-window-days", type=float, default=240.0)
    p.add_argument("--tof-window-days", type=float, default=360.0)
    p.add_argument("--depart-min-day", type=float, default=0.0)
    p.add_argument("--depart-max-day", type=float, default=3650.0)
    p.add_argument("--tof-min-days", type=float, default=30.0)
    p.add_argument("--tof-max-days", type=float, default=5000.0)

    p.add_argument("--max-vinf-mismatch-m-s", type=float, default=25.0, help="Hard |v_inf_out|-|v_inf_in| limit")
    p.add_argument("--vinf-mismatch-soft-m-s", type=float, default=5.0, help="Soft mismatch before quadratic penalty")
    p.add_argument("--min-rp-margin-km", type=float, default=50.0)
    p.add_argument("--rp-soft-margin-km", type=float, default=150.0)
    p.add_argument("--turn-soft-margin-deg", type=float, default=5.0)
    p.add_argument("--flyby-vinf-mode", choices=["conservative", "average", "incoming", "outgoing", "max", "rms"], default="conservative")

    p.add_argument("--vinf-depart-weight", type=float, default=1.0)
    p.add_argument("--vinf-arrive-weight", type=float, default=0.35)
    p.add_argument("--c3-weight", type=float, default=0.0)
    p.add_argument("--tof-weight", type=float, default=0.05)
    p.add_argument("--vinf-mismatch-weight", type=float, default=5.0)
    p.add_argument("--turn-angle-weight", type=float, default=0.6)
    p.add_argument("--flyby-margin-weight", type=float, default=0.1)
    p.add_argument("--vinf-mismatch-penalty-weight", type=float, default=200.0)
    p.add_argument("--rp-margin-penalty-weight", type=float, default=2.0)
    p.add_argument("--turn-margin-penalty-weight", type=float, default=2.0)
    p.add_argument("--hard-constraint-penalty", type=float, default=1e5)

    p.add_argument("--algorithm", choices=["de", "sade", "pso", "bee_colony"], default="de")
    p.add_argument("--generations", type=int, default=240)
    p.add_argument("--population", type=int, default=96)
    p.add_argument("--runs", type=int, default=6)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--de-f", type=float, default=0.8)
    p.add_argument("--de-cr", type=float, default=0.9)
    p.add_argument("--de-variant", type=int, default=2)
    p.add_argument("--workers", type=int, default=1, help="0=os.cpu_count, 1=serial, >1 multiprocessing per source route")
    p.add_argument("--multiprocessing-start-method", choices=["spawn", "fork", "forkserver"], default="spawn")

    p.add_argument("--output-csv", required=True, type=Path)
    p.add_argument("--output-jsonl", required=True, type=Path)
    p.add_argument("--output-json", required=True, type=Path)
    args = p.parse_args(argv)

    args.max_vinf_mismatch_km_s = args.max_vinf_mismatch_m_s / 1000.0
    args.vinf_mismatch_soft_km_s = args.vinf_mismatch_soft_m_s / 1000.0
    if args.max_routes <= 0 and args.route_index is None:
        p.error("--max-routes must be positive unless --route-index is used")
    if args.population < 5:
        p.error("--population should be >=5")
    if args.runs < 1:
        p.error("--runs must be >=1")
    return args


def worker_count(n: int) -> int:
    return max(1, os.cpu_count() or 1) if n == 0 else max(1, n)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    spice = import_spiceypy(); pk = import_pykep(); pg = import_pygmo()
    furnish_kernels(spice, [args.tpc, args.bsp])
    central_mu = load_central_mu(spice, args.metadata, args.central_body, args.mu_central_km3_s2)
    catalog = load_body_catalog(args.body_catalog)
    records = select_records(read_jsonl(args.routes_jsonl), args)
    if not records:
        raise RuntimeError("No routes selected")

    wc = worker_count(args.workers)
    routes: List[ConnectedFlybyRoute] = []
    if wc == 1 or len(records) == 1:
        for i, rec in enumerate(records, start=1):
            seq = route_sequence(rec)
            coverage = infer_coverage_start(rec)
            if coverage is None:
                raise RuntimeError("Could not infer coverage_start")
            seed_x, labels = connected_seed(rec, coverage)
            print(f"[INFO] Refining {i}/{len(records)} connected {' -> '.join(seq)} seed={dict(zip(labels,[round(v,3) for v in seed_x]))}")
            ev = ConnectedFlybyEvaluator(spice=spice, pykep=pk, sequence=seq, central_body=args.central_body,
                                         ref_frame=args.ref_frame, central_mu=central_mu, coverage_start_et=coverage,
                                         body_catalog=catalog, args=args)
            r = optimize_one(pg, ev, rec, seed_x, labels, args)
            routes.append(r)
            print(f"[INFO]   obj={r.objective:.6f} valid={r.valid} status={r.status} mismatch={r.max_vinf_mismatch_km_s*1000:.3f} m/s rp={r.min_rp_margin_km}")
    else:
        ctx = mp.get_context(args.multiprocessing_start_method)
        arg_payload = vars(args).copy()
        arg_payload["bsp"] = str(args.bsp); arg_payload["tpc"] = str(args.tpc); arg_payload["metadata"] = str(args.metadata) if args.metadata else None
        # Convert back to Path in worker via argparse namespace? furnish accepts Path-like; string works with Path(p).
        payloads = [{"idx": i, "record": rec, "args": arg_payload, "catalog": catalog, "central_mu": central_mu} for i, rec in enumerate(records)]
        with ProcessPoolExecutor(max_workers=wc, mp_context=ctx) as ex:
            futs = [ex.submit(_worker, p) for p in payloads]
            for fut in as_completed(futs):
                idx, route = fut.result()
                routes.append(route)
                print(f"[INFO] worker route {idx+1}/{len(records)} obj={route.objective:.6f} valid={route.valid} status={route.status}")

    routes.sort(key=lambda r: (not r.valid, r.objective))
    write_csv(args.output_csv, routes)
    write_jsonl(args.output_jsonl, routes)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "counts": {"refined_routes": len(routes), "valid_routes": sum(1 for r in routes if r.valid)},
        "search_spec": {
            "sequence": args.sequence,
            "central_body": args.central_body,
            "central_mu_km3_s2": central_mu,
            "max_vinf_mismatch_m_s": args.max_vinf_mismatch_m_s,
            "min_rp_margin_km": args.min_rp_margin_km,
            "connected_flyby": True,
            "workers": wc,
            "algorithm": args.algorithm,
            "generations": args.generations,
            "population": args.population,
            "runs": args.runs,
        },
        "best": short_route(routes[0]) if routes else None,
        "top_routes": [short_route(r) for r in routes[:20]],
        "caveats": [
            "Connected flyby means zero explicit layover; same encounter epoch is used for incoming and outgoing Lambert legs.",
            "This is still Lambert + first-order flyby envelope, not B-plane targeting or full N-body closure.",
        ],
    }
    write_json(args.output_json, summary)
    print_report(routes)
    print(f"[OK] wrote CSV:   {args.output_csv}")
    print(f"[OK] wrote JSONL: {args.output_jsonl}")
    print(f"[OK] wrote JSON:  {args.output_json}")
    try:
        spice.kclear()
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
