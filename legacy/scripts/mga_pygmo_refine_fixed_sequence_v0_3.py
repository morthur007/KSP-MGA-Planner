#!/usr/bin/env python3
"""
mga_pygmo_refine_fixed_sequence_v0_3.py

Continuous fixed-sequence Lambert refiner for the offline-first KSP + Principia
MGA pipeline.

Purpose
-------
Consume coarse route genomes produced by mga_beam_search_v0_2.py and refine the
continuous variables of one fixed sequence:

    x = [depart_day, tof_1, layover_1, tof_2, layover_2, ..., tof_N]

The sequence of bodies is NOT optimized here. Beam search chooses families;
this script improves epochs, time-of-flight values, and intermediate layovers
inside a selected family using PyGMO/Pagmo algorithms.

This is still planning-grade:
  * Lambert arcs are two-body seeds under the selected central body.
  * Flybys are checked with v_infinity continuity and a physical turn-angle /
    periapsis envelope from BodyCatalog.
  * No B-plane targeting, no powered-flyby closure, no N-body validation.
  * Output routes are candidates for local targeters / PyKEP-MGA / Tudat / REBOUND.

Dependencies
------------
  conda install -c conda-forge spiceypy pykep pygmo numpy

Example
-------
  python mga_pygmo_refine_fixed_sequence_v0_3.py \
    --bsp data/spice_v0_1_33y/opm_mpe_principia_rebound_v0_1_33y.bsp \
    --tpc data/spice_v0_1_33y/opm_mpe_principia_rebound_v0_1_33y.ids.tpc \
    --metadata data/spice_v0_1_33y/opm_mpe_principia_rebound_v0_1_33y.metadata.json \
    --body-catalog data/mga_v0_1/body_catalog_v0_1.krpc.json \
    --routes-jsonl data/mga_v0_1/mga_routes_v0_2_krpc.jsonl \
    --sequence Kerbin,Duna,Jool \
    --max-routes 10 \
    --generations 160 \
    --population 64 \
    --runs 4 \
    --output-csv data/mga_v0_1/mga_refined_kdj_v0_1.csv \
    --output-jsonl data/mga_v0_1/mga_refined_kdj_v0_1.jsonl \
    --output-json data/mga_v0_1/mga_refined_kdj_v0_1.summary.json
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import math
import os
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

SECONDS_PER_DAY = 86_400.0
DAYS_PER_YEAR = 365.25
SCHEMA_VERSION = "mga_pygmo_refine_fixed_sequence.v0.3"

Vec3 = Tuple[float, float, float]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


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
    arrival_et: float
    depart_et: float
    layover_days: float
    vinf_in_km_s: float
    vinf_out_km_s: float
    vinf_mag_jump_km_s: float
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
class RefinedRoute:
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
    total_layover_days: float
    decision_vector: Tuple[float, ...]
    decision_labels: Tuple[str, ...]
    leg_evals: Tuple[LegEval, ...]
    flyby_evals: Tuple[FlybyEval, ...]
    max_vinf_depart_km_s: float
    max_vinf_arrive_km_s: float
    max_vinf_mag_jump_km_s: float
    max_turn_angle_deg: float
    min_rp_margin_km: Optional[float]
    min_turn_angle_margin_deg: Optional[float]
    valid: bool
    status: str
    optimizer: Mapping[str, Any]


# ---------------------------------------------------------------------------
# Math helpers
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


def stable_id(prefix: str, payload: Mapping[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=True, separators=(",", ":"), default=str)
    return f"{prefix}_{hashlib.sha1(raw.encode('utf-8')).hexdigest()[:16]}"


def finite(x: Any, default: float = 0.0) -> float:
    try:
        y = float(x)
    except (TypeError, ValueError):
        return default
    return y if math.isfinite(y) else default


def opt_float(x: Any) -> Optional[float]:
    if x is None or x == "":
        return None
    try:
        y = float(x)
    except (TypeError, ValueError):
        return None
    return y if math.isfinite(y) else None


def opt_bool(x: Any, default: bool = False) -> bool:
    if isinstance(x, bool):
        return x
    if x is None:
        return default
    s = str(x).strip().lower()
    if s in {"1", "true", "t", "yes", "y", "allow", "allowed"}:
        return True
    if s in {"0", "false", "f", "no", "n", "deny", "blocked"}:
        return False
    return default


def max_turn_angle_deg(mu_km3_s2: float, rp_min_km: float, vinf_km_s: float) -> float:
    if mu_km3_s2 <= 0.0 or rp_min_km <= 0.0 or vinf_km_s <= 0.0:
        return 0.0
    arg = 1.0 / (rp_min_km * vinf_km_s * vinf_km_s / mu_km3_s2 + 1.0)
    arg = max(0.0, min(1.0, arg))
    return math.degrees(2.0 * math.asin(arg))


def required_rp_km(mu_km3_s2: float, vinf_km_s: float, turn_angle_deg: float) -> Optional[float]:
    if mu_km3_s2 <= 0.0 or vinf_km_s <= 0.0:
        return None
    if turn_angle_deg <= 1.0e-12:
        return math.inf
    half = math.radians(turn_angle_deg) / 2.0
    s = math.sin(half)
    if s <= 0.0:
        return math.inf
    return mu_km3_s2 / (vinf_km_s * vinf_km_s) * (1.0 / s - 1.0)


# ---------------------------------------------------------------------------
# Robust imports / SPICE / Lambert
# ---------------------------------------------------------------------------


def import_spiceypy():
    try:
        return importlib.import_module("spiceypy")
    except ImportError as exc:
        raise RuntimeError("spiceypy is required. Install with: conda install -c conda-forge spiceypy") from exc


def import_pykep():
    for name in ("pykep", "kep3"):
        try:
            return importlib.import_module(name)
        except ImportError:
            continue
    raise RuntimeError("pykep/kep3 is required. Install with: conda install -c conda-forge pykep")


def import_pygmo():
    try:
        return importlib.import_module("pygmo")
    except ImportError as exc:
        raise RuntimeError("pygmo is required for this V0.1 refiner. Install with: conda install -c conda-forge pygmo") from exc


def furnish_kernels(spice: Any, paths: Sequence[Path]) -> None:
    for path in paths:
        if path is None:
            continue
        if not path.exists():
            raise FileNotFoundError(path)
        spice.furnsh(str(path))


def parse_spice_state(spice: Any, target: str, et: float, ref_frame: str, central_body: str) -> Tuple[Vec3, Vec3]:
    state, _lt = spice.spkezr(target, float(et), ref_frame, "NONE", central_body)
    return (float(state[0]), float(state[1]), float(state[2])), (float(state[3]), float(state[4]), float(state[5]))


def pykep_lambert_solutions(pk: Any, r1: Vec3, r2: Vec3, tof_s: float, mu: float, max_revs: int) -> List[Tuple[Vec3, Vec3, bool, int]]:
    out: List[Tuple[Vec3, Vec3, bool, int]] = []
    # Try both senses. In PyKEP, cw is a boolean in the constructor.
    for cw in (False, True):
        try:
            lp = pk.lambert_problem(list(r1), list(r2), float(tof_s), float(mu), cw, int(max_revs))
        except TypeError:
            # Some versions expose keyword names differently, but the positional
            # signature above is the common PyKEP path.
            try:
                lp = pk.lambert_problem(r1=list(r1), r2=list(r2), tof=float(tof_s), mu=float(mu), cw=cw, max_revs=int(max_revs))
            except Exception:
                continue
        except Exception:
            continue
        try:
            v1s = lp.get_v1()
            v2s = lp.get_v2()
        except Exception:
            continue
        for idx, (v1, v2) in enumerate(zip(v1s, v2s)):
            if len(v1) != 3 or len(v2) != 3:
                continue
            out.append(((float(v1[0]), float(v1[1]), float(v1[2])), (float(v2[0]), float(v2[1]), float(v2[2])), cw, idx))
    return out


# ---------------------------------------------------------------------------
# Input readers
# ---------------------------------------------------------------------------


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_routes_jsonl(path: Path) -> List[Dict[str, Any]]:
    routes: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                routes.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_no}: {exc}") from exc
    return routes


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


def scalar_from_any(value: Any) -> Optional[float]:
    """Return a scalar float from numbers, numeric strings or 1-element arrays."""
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
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
    """Normalize GM/mu values to km^3/s^2.

    Project metadata has appeared in both km^3/s^2 and SI m^3/s^2 forms.
    The key name is authoritative when available; otherwise very large values
    are treated as SI.
    """
    key_l = key_hint.lower()
    if "m3" in key_l or "m^3" in key_l or "m_3" in key_l:
        return value / 1.0e9
    if "km3" in key_l or "km^3" in key_l or "km_3" in key_l:
        return value
    if abs(value) > 1.0e15:
        return value / 1.0e9
    return value


def dict_contains_body(d: Mapping[str, Any], body: str) -> bool:
    body_l = body.lower()
    aliases = {body_l}
    if body_l == "sun":
        aliases.add("kerbol")
    if body_l == "kerbol":
        aliases.add("sun")
    for k, v in d.items():
        k_l = str(k).lower()
        if k_l in aliases:
            return True
        if k_l in {"name", "body", "target", "naif_name", "central_body", "body_name"}:
            if str(v).lower() in aliases:
                return True
    return False


def iter_dicts(obj: Any) -> Iterable[Mapping[str, Any]]:
    if isinstance(obj, Mapping):
        yield obj
        for val in obj.values():
            yield from iter_dicts(val)
    elif isinstance(obj, list):
        for val in obj:
            yield from iter_dicts(val)


def extract_named_body_dict(root: Mapping[str, Any], body: str) -> Dict[str, Any]:
    body_l = body.lower()
    aliases = {body_l}
    if body_l == "sun":
        aliases.add("kerbol")
    if body_l == "kerbol":
        aliases.add("sun")

    # Direct maps: {"Sun": {...}}, {"bodies": {"Sun": {...}}}, etc.
    for k, v in root.items():
        if str(k).lower() in aliases and isinstance(v, Mapping):
            return dict(v)

    for container_key in (
        "bodies", "body_catalog", "catalog", "targets", "target_policy",
        "policies", "body_models", "physical_parameters", "gm", "mu",
    ):
        container = root.get(container_key)
        if isinstance(container, Mapping):
            for k, v in container.items():
                if str(k).lower() in aliases and isinstance(v, Mapping):
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
    """Find central-body GM/mu in flexible project metadata formats."""
    body_l = body.lower()
    aliases = [body]
    if body_l == "sun":
        aliases.append("Kerbol")
    elif body_l == "kerbol":
        aliases.append("Sun")

    body_keys = (
        "mu_km3_s2", "gm_km3_s2", "GM_km3_s2", "gravitational_parameter_km3_s2",
        "mu_m3_s2", "gm_m3_s2", "GM_m3_s2", "gravitational_parameter_m3_s2",
        "mu", "gm", "GM", "gravitational_parameter", "standard_gravitational_parameter",
    )
    for alias in aliases:
        body_d = extract_named_body_dict(metadata, alias)
        for key in body_keys:
            if key in body_d:
                value = scalar_from_any(body_d[key])
                if value is not None:
                    mu = convert_mu_to_km3_s2(value, key)
                    if mu > 0:
                        return mu

    # Project-level central-body keys. These are safe because they explicitly
    # identify the central body or central GM.
    preferred_keys = [
        "central_mu_km3_s2", "central_gm_km3_s2", "central_body_mu_km3_s2",
        "central_body_gm_km3_s2", "mu_central_km3_s2", "gm_central_km3_s2",
        "central_mu_m3_s2", "central_gm_m3_s2", "central_body_mu_m3_s2",
        "central_body_gm_m3_s2",
    ]
    for alias in aliases:
        safe = alias.replace(" ", "_")
        preferred_keys.extend([
            f"mu_{safe}_km3_s2", f"gm_{safe}_km3_s2",
            f"{safe}_mu_km3_s2", f"{safe}_gm_km3_s2",
            f"mu_{safe}_m3_s2", f"gm_{safe}_m3_s2",
            f"{safe}_mu_m3_s2", f"{safe}_gm_m3_s2",
        ])

    def find_preferred(obj: Any) -> Optional[Tuple[float, str]]:
        if isinstance(obj, Mapping):
            lower_map = {str(k).lower(): (k, v) for k, v in obj.items()}
            for key in preferred_keys:
                hit = lower_map.get(key.lower())
                if hit is None:
                    continue
                raw_key, raw_val = hit
                val = scalar_from_any(raw_val)
                if val is not None:
                    return val, str(raw_key)
            for val in obj.values():
                found = find_preferred(val)
                if found is not None:
                    return found
        elif isinstance(obj, list):
            for val in obj:
                found = find_preferred(val)
                if found is not None:
                    return found
        return None

    found = find_preferred(metadata)
    if found is not None:
        value, key = found
        mu = convert_mu_to_km3_s2(value, key)
        if mu > 0:
            return mu
    return None


def load_central_mu(spice: Any, metadata_path: Optional[Path], central_body: str, explicit_mu_km3_s2: Optional[float] = None) -> float:
    if explicit_mu_km3_s2 is not None and explicit_mu_km3_s2 > 0:
        return float(explicit_mu_km3_s2)

    # SPICE/TPC first: this is the strongest source when BODY*_GM is present.
    for name in [central_body, "Kerbol" if central_body.lower() == "sun" else "Sun" if central_body.lower() == "kerbol" else central_body]:
        try:
            _dim, values = spice.bodvrd(name, "GM", 1)
            mu = float(values[0])
            if mu > 0:
                return mu
        except Exception:
            pass

    if metadata_path and metadata_path.exists():
        meta = read_json(metadata_path)
        found = find_mu_in_metadata(meta, central_body)
        if found is not None and found > 0:
            return found

    raise RuntimeError(
        f"Could not determine GM/mu for central body {central_body!r}. "
        "Use --mu-central-km3-s2 explicitly, or provide metadata/TPC with a central-body GM. "
        "For your current kernel, the smoke scout reported mu(Sun) ≈ 1172332794.83249 km^3/s^2."
    )


def infer_coverage_start_from_route(route_record: Mapping[str, Any]) -> Optional[float]:
    legs = route_record.get("legs", [])
    if not isinstance(legs, list) or not legs:
        return None
    leg0 = legs[0]
    if not isinstance(leg0, Mapping):
        return None
    dep_et = opt_float(leg0.get("depart_et"))
    dep_days = opt_float(leg0.get("depart_days_from_coverage_start"))
    if dep_et is None or dep_days is None:
        return None
    return dep_et - dep_days * SECONDS_PER_DAY


def route_sequence(route_record: Mapping[str, Any]) -> Tuple[str, ...]:
    route = route_record.get("route", {}) if isinstance(route_record.get("route", {}), Mapping) else {}
    seq = route.get("sequence", None)
    if isinstance(seq, str):
        return tuple(x.strip() for x in seq.replace("->", ",").split(",") if x.strip())
    if isinstance(seq, list):
        return tuple(str(x) for x in seq)
    legs = route_record.get("legs", [])
    if isinstance(legs, list) and legs:
        bodies = [str(legs[0].get("origin", ""))]
        for leg in legs:
            if isinstance(leg, Mapping):
                bodies.append(str(leg.get("target", "")))
        return tuple(x for x in bodies if x)
    return ()


def source_score(route_record: Mapping[str, Any]) -> Optional[float]:
    route = route_record.get("route", {}) if isinstance(route_record.get("route", {}), Mapping) else {}
    return opt_float(route.get("nominal_score"))


def seed_decision_vector(route_record: Mapping[str, Any], coverage_start_et: float) -> Tuple[List[float], List[str]]:
    legs = route_record.get("legs", [])
    if not isinstance(legs, list) or not legs:
        raise ValueError("Route record has no legs")
    x: List[float] = []
    labels: List[str] = []
    first = legs[0]
    dep_et = finite(first.get("depart_et"))
    x.append((dep_et - coverage_start_et) / SECONDS_PER_DAY)
    labels.append("depart_day")
    for i, leg in enumerate(legs):
        tof_days = finite(leg.get("tof_days"), finite(leg.get("tof_s")) / SECONDS_PER_DAY)
        x.append(tof_days)
        labels.append(f"tof_{i+1}_days")
        if i < len(legs) - 1:
            arr = finite(leg.get("arrive_et"))
            nxt_dep = finite(legs[i + 1].get("depart_et"))
            lay = max(0.0, (nxt_dep - arr) / SECONDS_PER_DAY)
            x.append(lay)
            labels.append(f"layover_{i+1}_days")
    return x, labels


def make_bounds(
    seed_x: Sequence[float],
    labels: Sequence[str],
    args: argparse.Namespace,
) -> Tuple[List[float], List[float]]:
    lb: List[float] = []
    ub: List[float] = []
    for value, label in zip(seed_x, labels):
        if label == "depart_day":
            lb.append(max(args.depart_min_day, value - args.depart_window_days))
            ub.append(min(args.depart_max_day, value + args.depart_window_days))
        elif label.startswith("tof_"):
            lb.append(max(args.tof_min_days, value - args.tof_window_days))
            ub.append(min(args.tof_max_days, value + args.tof_window_days))
        elif label.startswith("layover_"):
            if getattr(args, "encounter_mode", "flyby") == "flyby":
                # A gravity assist is an encounter, not a parking-orbit wait. Ignore the
                # source route's coarse layover and force a short encounter window.
                lb.append(max(args.layover_min_days, 0.0))
                ub.append(min(args.layover_max_days, args.max_flyby_layover_days))
            else:
                lb.append(max(args.layover_min_days, value - args.layover_window_days))
                ub.append(min(args.layover_max_days, value + args.layover_window_days))
        else:
            lb.append(value)
            ub.append(value)
    for i, (a, b) in enumerate(zip(lb, ub)):
        if b < a:
            raise ValueError(f"Invalid bounds for {labels[i]}: [{a}, {b}]")
        if abs(b - a) < 1.0e-9:
            ub[i] = a + 1.0e-9
    return lb, ub


# ---------------------------------------------------------------------------
# Objective evaluator
# ---------------------------------------------------------------------------


class FixedSequenceLambertEvaluator:
    def __init__(
        self,
        *,
        spice: Any,
        pykep: Any,
        sequence: Sequence[str],
        central_body: str,
        ref_frame: str,
        central_mu: float,
        coverage_start_et: float,
        body_catalog: Mapping[str, BodyEntry],
        args: argparse.Namespace,
    ) -> None:
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
        # Python modules are not picklable. PyGMO may copy the UDP/problem internally,
        # so drop module objects and re-import them in __setstate__.
        state["spice"] = None
        state["pk"] = None
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self.__dict__.update(state)
        self.spice = import_spiceypy()
        self.pk = import_pykep()

    def decode_times(self, x: Sequence[float]) -> Tuple[List[float], List[float], List[float]]:
        # x = [depart_day, tof1, layover1, tof2, layover2, ..., tofN]
        n_legs = len(self.sequence) - 1
        if len(x) != 1 + n_legs + (n_legs - 1):
            raise ValueError(f"Expected {1 + n_legs + (n_legs - 1)} variables, got {len(x)}")
        depart_ets: List[float] = []
        arrive_ets: List[float] = []
        layover_days: List[float] = []
        current_depart = self.coverage_start_et + float(x[0]) * SECONDS_PER_DAY
        idx = 1
        for leg_i in range(n_legs):
            tof_days = float(x[idx]); idx += 1
            depart_ets.append(current_depart)
            arrive = current_depart + tof_days * SECONDS_PER_DAY
            arrive_ets.append(arrive)
            if leg_i < n_legs - 1:
                lay = float(x[idx]); idx += 1
                layover_days.append(lay)
                current_depart = arrive + lay * SECONDS_PER_DAY
        return depart_ets, arrive_ets, layover_days

    def evaluate(self, x: Sequence[float], full: bool = False) -> Tuple[float, Optional[RefinedRoute], str]:
        penalty = 0.0
        status_parts: List[str] = []
        try:
            depart_ets, arrive_ets, layover_days = self.decode_times(x)
        except Exception as exc:
            return 1.0e99, None, f"decode_error:{exc}"

        if self.args.encounter_mode == "flyby":
            for lay in layover_days:
                if lay > self.args.max_flyby_layover_days + 1.0e-12:
                    penalty += self.args.hard_constraint_penalty * (1.0 + lay - self.args.max_flyby_layover_days)
                    status_parts.append("flyby_layover_exceeds_limit")

        leg_evals: List[LegEval] = []
        # Bounds should handle these, but keep objective robust for islands/random values.
        if any(arr <= dep for dep, arr in zip(depart_ets, arrive_ets)):
            return 1.0e99, None, "nonpositive_tof"

        for i, (origin, target) in enumerate(zip(self.sequence[:-1], self.sequence[1:])):
            dep = depart_ets[i]
            arr = arrive_ets[i]
            tof_s = arr - dep
            tof_days = tof_s / SECONDS_PER_DAY
            try:
                r1, v_origin = parse_spice_state(self.spice, origin, dep, self.ref_frame, self.central_body)
                r2, v_target = parse_spice_state(self.spice, target, arr, self.ref_frame, self.central_body)
            except Exception as exc:
                return 1.0e98, None, f"spice_state_error:{origin}->{target}:{exc}"

            sols = pykep_lambert_solutions(self.pk, r1, r2, tof_s, self.central_mu, self.args.max_revs)
            if not sols:
                return 1.0e97, None, f"lambert_fail:{origin}->{target}"

            best: Optional[LegEval] = None
            best_score = math.inf
            for v_sc_dep, v_sc_arr, cw, sol_idx in sols:
                vinf_dep_vec = vec_sub(v_sc_dep, v_origin)
                vinf_arr_vec = vec_sub(v_sc_arr, v_target)
                vinf_dep = vec_norm(vinf_dep_vec)
                vinf_arr = vec_norm(vinf_arr_vec)
                c3 = vinf_dep * vinf_dep
                leg_score = (
                    self.args.vinf_depart_weight * vinf_dep
                    + self.args.vinf_arrive_weight * vinf_arr
                    + self.args.c3_weight * c3
                    + self.args.tof_weight * (tof_days / DAYS_PER_YEAR)
                )
                if leg_score < best_score:
                    best_score = leg_score
                    best = LegEval(
                        leg_index=i,
                        origin=origin,
                        target=target,
                        depart_et=dep,
                        arrive_et=arr,
                        tof_days=tof_days,
                        c3_km2_s2=c3,
                        vinf_depart_km_s=vinf_dep,
                        vinf_arrive_km_s=vinf_arr,
                        leg_score=leg_score,
                        origin_r_km=r1,
                        origin_v_km_s=v_origin,
                        target_r_km=r2,
                        target_v_km_s=v_target,
                        sc_v_depart_km_s=v_sc_dep,
                        sc_v_arrive_km_s=v_sc_arr,
                        cw=cw,
                        solution_index=sol_idx,
                    )
            assert best is not None
            leg_evals.append(best)

        flyby_evals: List[FlybyEval] = []
        for i in range(len(leg_evals) - 1):
            prev = leg_evals[i]
            nxt = leg_evals[i + 1]
            body = prev.target
            vinf_in_vec = prev.vinf_arrive_vec()
            vinf_out_vec = nxt.vinf_depart_vec()
            vinf_in = vec_norm(vinf_in_vec)
            vinf_out = vec_norm(vinf_out_vec)
            mag_jump = abs(vinf_out - vinf_in)
            turn = angle_deg(vinf_in_vec, vinf_out_vec)
            entry = self.body_catalog.get(body)
            ok = True
            fly_status = "ok"
            mu = None
            rp_min = None
            vinf_eff = None
            turn_max = None
            turn_margin = None
            rp_req = None
            rp_margin = None
            if entry is None:
                ok = self.args.missing_body_action != "reject"
                fly_status = "missing_body_catalog"
            elif not entry.allow_flyby:
                ok = False
                fly_status = "flyby_not_allowed"
                mu = entry.mu_km3_s2
                rp_min = entry.rp_min_km
            elif entry.mu_km3_s2 is None or entry.rp_min_km is None:
                ok = self.args.missing_body_action != "reject"
                fly_status = "missing_mu_or_rp_min"
                mu = entry.mu_km3_s2
                rp_min = entry.rp_min_km
            else:
                mu = entry.mu_km3_s2
                rp_min = entry.rp_min_km
                vinf_eff = self.effective_vinf(vinf_in, vinf_out)
                turn_max = max_turn_angle_deg(mu, rp_min, vinf_eff)
                turn_margin = turn_max - turn
                rp_req = required_rp_km(mu, vinf_eff, turn)
                rp_margin = None if rp_req is None else rp_req - rp_min
                ok = turn_margin >= -1.0e-12
                fly_status = "ok" if ok else "turn_angle_exceeds_physical_envelope"
                if rp_margin is not None and rp_margin < self.args.min_rp_margin_km:
                    ok = False
                    fly_status = "rp_margin"

            if not ok:
                penalty += self.args.hard_constraint_penalty
                status_parts.append(fly_status)
            if mag_jump > self.args.max_vinf_mag_jump:
                penalty += self.args.hard_constraint_penalty * (1.0 + mag_jump - self.args.max_vinf_mag_jump)
                status_parts.append("vinf_mag_jump")
            if turn > self.args.max_turn_angle_deg > 0:
                penalty += self.args.hard_constraint_penalty * (1.0 + (turn - self.args.max_turn_angle_deg) / 180.0)
                status_parts.append("turn_angle_proxy")

            transition_score = (
                self.args.vinf_jump_weight * mag_jump
                + self.args.turn_angle_weight * (turn / 180.0)
                + self.args.layover_weight * (layover_days[i] / DAYS_PER_YEAR)
            )
            if self.args.encounter_mode == "flyby" and self.args.flyby_layover_weight > 0:
                transition_score += self.args.flyby_layover_weight * (layover_days[i] / max(self.args.max_flyby_layover_days, 1.0e-9))
            if rp_margin not in (None, math.inf, -math.inf) and self.args.flyby_margin_weight > 0:
                normalized = max(0.0, rp_margin) / max(float(rp_min or 1.0), 1.0)
                transition_score += self.args.flyby_margin_weight / (1.0 + normalized)

            flyby_evals.append(
                FlybyEval(
                    flyby_index=i,
                    body=body,
                    arrival_et=prev.arrive_et,
                    depart_et=nxt.depart_et,
                    layover_days=layover_days[i],
                    vinf_in_km_s=vinf_in,
                    vinf_out_km_s=vinf_out,
                    vinf_mag_jump_km_s=mag_jump,
                    turn_angle_deg=turn,
                    mu_km3_s2=mu,
                    rp_min_km=rp_min,
                    vinf_eff_km_s=vinf_eff,
                    turn_angle_max_deg=turn_max,
                    turn_angle_margin_deg=turn_margin,
                    rp_required_km=rp_req,
                    rp_margin_km=rp_margin,
                    ok=ok and mag_jump <= self.args.max_vinf_mag_jump,
                    status=fly_status,
                    transition_score=transition_score,
                )
            )

        leg_score_sum = sum(l.leg_score for l in leg_evals)
        transition_score_sum = sum(f.transition_score for f in flyby_evals)
        objective = leg_score_sum + transition_score_sum + penalty

        if not full:
            return objective, None, ";".join(status_parts) if status_parts else "ok"

        max_vinf_dep = max((l.vinf_depart_km_s for l in leg_evals), default=0.0)
        max_vinf_arr = max((l.vinf_arrive_km_s for l in leg_evals), default=0.0)
        max_mag_jump = max((f.vinf_mag_jump_km_s for f in flyby_evals), default=0.0)
        max_turn = max((f.turn_angle_deg for f in flyby_evals), default=0.0)
        rp_margins = [f.rp_margin_km for f in flyby_evals if f.rp_margin_km not in (None, math.inf, -math.inf)]
        turn_margins = [f.turn_angle_margin_deg for f in flyby_evals if f.turn_angle_margin_deg is not None and math.isfinite(f.turn_angle_margin_deg)]
        valid = penalty <= 0.0 and all(f.ok for f in flyby_evals)
        status = "ok" if valid else (";".join(status_parts) if status_parts else "penalized")
        rid = stable_id(
            "refined",
            {
                "sequence": self.sequence,
                "x": [round(float(v), 9) for v in x],
                "objective": round(objective, 9),
            },
        )
        route = RefinedRoute(
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
            total_layover_days=sum(layover_days),
            decision_vector=tuple(float(v) for v in x),
            decision_labels=tuple(),
            leg_evals=tuple(leg_evals),
            flyby_evals=tuple(flyby_evals),
            max_vinf_depart_km_s=max_vinf_dep,
            max_vinf_arrive_km_s=max_vinf_arr,
            max_vinf_mag_jump_km_s=max_mag_jump,
            max_turn_angle_deg=max_turn,
            min_rp_margin_km=min(rp_margins) if rp_margins else None,
            min_turn_angle_margin_deg=min(turn_margins) if turn_margins else None,
            valid=valid,
            status=status,
            optimizer={},
        )
        return objective, route, status

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


class PygmoUDP:
    def __init__(self, evaluator: FixedSequenceLambertEvaluator, lb: Sequence[float], ub: Sequence[float]) -> None:
        self.evaluator = evaluator
        self.lb = list(lb)
        self.ub = list(ub)

    def fitness(self, x: Sequence[float]) -> List[float]:
        f, _route, _status = self.evaluator.evaluate(x, full=False)
        if not math.isfinite(f):
            f = 1.0e99
        return [float(f)]

    def get_bounds(self) -> Tuple[List[float], List[float]]:
        return self.lb, self.ub

    def get_name(self) -> str:
        return "KSP-Principia fixed-sequence Lambert MGA refiner"


# ---------------------------------------------------------------------------
# Optimization
# ---------------------------------------------------------------------------


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
    raise ValueError(f"Unsupported --algorithm {args.algorithm!r}")


def optimize_one_route(
    *,
    pg: Any,
    evaluator: FixedSequenceLambertEvaluator,
    source_record: Mapping[str, Any],
    seed_x: Sequence[float],
    labels: Sequence[str],
    args: argparse.Namespace,
) -> RefinedRoute:
    lb, ub = make_bounds(seed_x, labels, args)
    prob = pg.problem(PygmoUDP(evaluator, lb, ub))
    algo = build_algorithm(pg, args)

    best_x: Optional[List[float]] = None
    best_f = math.inf
    run_summaries: List[Dict[str, Any]] = []

    # Evaluate seed explicitly; keep it as fallback and baseline.
    seed_f, _seed_route, seed_status = evaluator.evaluate(seed_x, full=False)
    best_x = [float(v) for v in seed_x]
    best_f = float(seed_f)
    run_summaries.append({"run": "seed", "objective": best_f, "status": seed_status})

    for run in range(args.runs):
        seed = args.seed + run if args.seed is not None else None
        try:
            pop = pg.population(prob, size=args.population, seed=seed) if seed is not None else pg.population(prob, size=args.population)
        except TypeError:
            pop = pg.population(prob, args.population)

        # Inject the beam-search seed if supported by this PyGMO version.
        try:
            pop.set_x(0, list(seed_x))
        except Exception:
            try:
                pop.push_back(list(seed_x))
            except Exception:
                pass

        pop = algo.evolve(pop)
        f = float(pop.champion_f[0])
        x = [float(v) for v in pop.champion_x]
        status = evaluator.evaluate(x, full=False)[2]
        run_summaries.append({"run": run, "objective": f, "status": status})
        if f < best_f:
            best_f = f
            best_x = x

    assert best_x is not None
    objective, route, status = evaluator.evaluate(best_x, full=True)
    if route is None:
        raise RuntimeError(f"Best solution could not be fully evaluated: {status}")

    src_route = source_record.get("route", {}) if isinstance(source_record.get("route", {}), Mapping) else {}
    src_id = str(src_route.get("route_id", ""))
    src_score = source_score(source_record)
    improvement = None if src_score is None else src_score - objective
    # Keep this shallow: dataclasses.asdict() recursively deep-copies nested objects and
    # can trip over non-copyable state in scientific backends on some Python/PyGMO builds.
    route_payload = dict(route.__dict__)
    route_payload.update({
        "source_route_id": src_id,
        "source_nominal_score": src_score,
        "improvement": improvement,
        "decision_labels": tuple(labels),
        "optimizer": {
            "library": "pygmo",
            "algorithm": args.algorithm,
            "generations": args.generations,
            "population": args.population,
            "runs": args.runs,
            "seed": args.seed,
            "bounds": {label: [lb_i, ub_i] for label, lb_i, ub_i in zip(labels, lb, ub)},
            "source_seed_objective": seed_f,
            "run_summaries": run_summaries,
        },
    })
    route = RefinedRoute(**route_payload)
    return route


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=json_default)
        f.write("\n")


def write_jsonl(path: Path, routes: Sequence[RefinedRoute]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for route in routes:
            f.write(json.dumps(asdict(route), ensure_ascii=False, separators=(",", ":"), default=json_default))
            f.write("\n")


def json_default(obj: Any) -> Any:
    if isinstance(obj, tuple):
        return list(obj)
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def fmt(x: Optional[float], digits: int = 9) -> str:
    if x is None:
        return ""
    if x == math.inf:
        return "inf"
    if x == -math.inf:
        return "-inf"
    return f"{float(x):.{digits}f}"


def write_csv(path: Path, routes: Sequence[RefinedRoute]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "refined_id",
        "source_route_id",
        "sequence",
        "valid",
        "status",
        "objective",
        "source_nominal_score",
        "improvement",
        "depart_et",
        "arrive_et",
        "depart_day",
        "total_tof_days",
        "total_layover_days",
        "max_vinf_depart_km_s",
        "max_vinf_arrive_km_s",
        "max_vinf_mag_jump_km_s",
        "max_turn_angle_deg",
        "min_rp_margin_km",
        "min_turn_angle_margin_deg",
        "leg_tof_days",
        "layover_days",
        "vinf_depart_by_leg_km_s",
        "vinf_arrive_by_leg_km_s",
        "flyby_bodies",
        "flyby_statuses",
        "rp_margins_km",
        "turn_margins_deg",
        "decision_vector",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in routes:
            dep_day = r.decision_vector[0] if r.decision_vector else None
            leg_tofs = [l.tof_days for l in r.leg_evals]
            layovers = [fb.layover_days for fb in r.flyby_evals]
            writer.writerow(
                {
                    "refined_id": r.refined_id,
                    "source_route_id": r.source_route_id,
                    "sequence": "->".join(r.sequence),
                    "valid": int(r.valid),
                    "status": r.status,
                    "objective": fmt(r.objective),
                    "source_nominal_score": fmt(r.source_nominal_score),
                    "improvement": fmt(r.improvement),
                    "depart_et": fmt(r.depart_et),
                    "arrive_et": fmt(r.arrive_et),
                    "depart_day": fmt(dep_day),
                    "total_tof_days": fmt(r.total_tof_days),
                    "total_layover_days": fmt(r.total_layover_days),
                    "max_vinf_depart_km_s": fmt(r.max_vinf_depart_km_s),
                    "max_vinf_arrive_km_s": fmt(r.max_vinf_arrive_km_s),
                    "max_vinf_mag_jump_km_s": fmt(r.max_vinf_mag_jump_km_s),
                    "max_turn_angle_deg": fmt(r.max_turn_angle_deg),
                    "min_rp_margin_km": fmt(r.min_rp_margin_km),
                    "min_turn_angle_margin_deg": fmt(r.min_turn_angle_margin_deg),
                    "leg_tof_days": ";".join(fmt(x, 6) for x in leg_tofs),
                    "layover_days": ";".join(fmt(x, 6) for x in layovers),
                    "vinf_depart_by_leg_km_s": ";".join(fmt(l.vinf_depart_km_s, 6) for l in r.leg_evals),
                    "vinf_arrive_by_leg_km_s": ";".join(fmt(l.vinf_arrive_km_s, 6) for l in r.leg_evals),
                    "flyby_bodies": ";".join(fb.body for fb in r.flyby_evals),
                    "flyby_statuses": ";".join(fb.status for fb in r.flyby_evals),
                    "rp_margins_km": ";".join(fmt(fb.rp_margin_km, 3) for fb in r.flyby_evals),
                    "turn_margins_deg": ";".join(fmt(fb.turn_angle_margin_deg, 3) for fb in r.flyby_evals),
                    "decision_vector": ";".join(fmt(x, 9) for x in r.decision_vector),
                }
            )


# ---------------------------------------------------------------------------
# Reporting / CLI
# ---------------------------------------------------------------------------


def select_routes(records: Sequence[Mapping[str, Any]], args: argparse.Namespace) -> List[Mapping[str, Any]]:
    wanted_seq: Optional[Tuple[str, ...]] = None
    if args.sequence:
        wanted_seq = tuple(x.strip() for x in args.sequence.replace("->", ",").split(",") if x.strip())
    candidates: List[Mapping[str, Any]] = []
    for record in records:
        seq = route_sequence(record)
        if not seq:
            continue
        if wanted_seq and seq != wanted_seq:
            continue
        if args.valid_only:
            phys = record.get("physical_flyby_summary", {}) if isinstance(record.get("physical_flyby_summary", {}), Mapping) else {}
            statuses = phys.get("statuses", [])
            if isinstance(statuses, list) and any(str(s) != "ok" for s in statuses):
                continue
        candidates.append(record)
    candidates.sort(key=lambda r: (source_score(r) if source_score(r) is not None else 1.0e99))
    if args.route_index is not None:
        if args.route_index < 0 or args.route_index >= len(candidates):
            raise IndexError(f"--route-index {args.route_index} outside selected route count {len(candidates)}")
        return [candidates[args.route_index]]
    return candidates[: args.max_routes]


def make_summary(routes: Sequence[RefinedRoute], args: argparse.Namespace, central_mu: float) -> Dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "purpose": "PyGMO continuous refinement of fixed MGA body sequences using SPICE + Lambert + physical flyby gates",
        "inputs": {
            "routes_jsonl": str(args.routes_jsonl),
            "body_catalog": str(args.body_catalog),
            "bsp": str(args.bsp),
            "tpc": str(args.tpc),
            "metadata": str(args.metadata) if args.metadata else None,
        },
        "outputs": {
            "output_csv": str(args.output_csv),
            "output_jsonl": str(args.output_jsonl),
            "output_json": str(args.output_json),
        },
        "search_spec": {
            "sequence": args.sequence,
            "max_routes": args.max_routes,
            "route_index": args.route_index,
            "central_body": args.central_body,
            "ref_frame": args.ref_frame,
            "central_mu_km3_s2": central_mu,
            "depart_window_days": args.depart_window_days,
            "tof_window_days": args.tof_window_days,
            "layover_window_days": args.layover_window_days,
            "encounter_mode": args.encounter_mode,
            "max_flyby_layover_days": args.max_flyby_layover_days,
            "tof_min_days": args.tof_min_days,
            "tof_max_days": args.tof_max_days,
            "layover_min_days": args.layover_min_days,
            "layover_max_days": args.layover_max_days,
            "max_vinf_mag_jump": args.max_vinf_mag_jump,
            "min_rp_margin_km": args.min_rp_margin_km,
            "algorithm": args.algorithm,
            "generations": args.generations,
            "population": args.population,
            "runs": args.runs,
            "workers": args.workers,
            "multiprocessing_start_method": args.multiprocessing_start_method,
            "parallelism_granularity": "source_route",
        },
        "counts": {
            "refined_routes": len(routes),
            "valid_routes": sum(1 for r in routes if r.valid),
        },
        "best": route_short(routes[0]) if routes else None,
        "top_routes": [route_short(r) for r in routes[: min(20, len(routes))]],
        "caveats": [
            "This is Lambert-only continuous refinement, not N-body validation.",
            "The sequence is fixed; beam search remains responsible for discrete route-family discovery.",
            "Flyby feasibility is an unpowered first-order envelope, not B-plane targeting.",
            "In encounter-mode=flyby, intermediate layovers are forcibly short; use encounter-mode=stopover for parking/rendezvous waits.",
        ],
    }


def route_short(r: RefinedRoute) -> Dict[str, Any]:
    return {
        "refined_id": r.refined_id,
        "source_route_id": r.source_route_id,
        "sequence": "->".join(r.sequence),
        "valid": r.valid,
        "status": r.status,
        "objective": r.objective,
        "source_nominal_score": r.source_nominal_score,
        "improvement": r.improvement,
        "total_tof_days": r.total_tof_days,
        "total_layover_days": r.total_layover_days,
        "max_vinf_mag_jump_km_s": r.max_vinf_mag_jump_km_s,
        "max_turn_angle_deg": r.max_turn_angle_deg,
        "min_rp_margin_km": r.min_rp_margin_km,
        "min_turn_angle_margin_deg": r.min_turn_angle_margin_deg,
        "decision": dict(zip(r.decision_labels, r.decision_vector)),
    }


def print_report(routes: Sequence[RefinedRoute]) -> None:
    print("=" * 80)
    print("MGA PYGMO FIXED-SEQUENCE REFINER V0.2")
    print("=" * 80)
    print(f"Refined routes: {len(routes)}")
    print(f"Valid routes:   {sum(1 for r in routes if r.valid)}")
    print("\nTop refined routes:")
    for i, r in enumerate(routes[:10], start=1):
        imp = "n/a" if r.improvement is None else f"{r.improvement:+.4f}"
        margin = "n/a" if r.min_rp_margin_km is None else f"{r.min_rp_margin_km:.1f} km"
        print(
            f"{i:2d}. {' -> '.join(r.sequence)} | valid={r.valid} | obj={r.objective:.4f} | "
            f"src={r.source_nominal_score if r.source_nominal_score is not None else float('nan'):.4f} | "
            f"impr={imp} | TOF={r.total_tof_days:.1f} d | layover={r.total_layover_days:.1f} d | "
            f"max Δv∞={r.max_vinf_mag_jump_km_s:.3f} km/s | max turn={r.max_turn_angle_deg:.1f}° | "
            f"min rp margin={margin} | status={r.status}"
        )
    print("=" * 80)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Refine fixed body-sequence MGA routes with PyGMO, SPICE states and PyKEP Lambert arcs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--bsp", required=True, type=Path, help="SPK/BSP ephemeris")
    parser.add_argument("--tpc", required=True, type=Path, help="Text kernel with IDs/names")
    parser.add_argument("--metadata", type=Path, default=None, help="Metadata JSON containing central_mu_km3_s2 if available")
    parser.add_argument("--body-catalog", required=True, type=Path, help="BodyCatalog JSON from kRPC/KSP configs")
    parser.add_argument("--routes-jsonl", required=True, type=Path, help="Routes JSONL from mga_beam_search_v0_2.py")
    parser.add_argument("--sequence", default="", help="Exact sequence to refine, e.g. Kerbin,Duna,Jool. Empty means top routes regardless of sequence")
    parser.add_argument("--route-index", type=int, default=None, help="Refine only selected index after filtering/sorting")
    parser.add_argument("--max-routes", type=int, default=10, help="Max source routes to refine when --route-index is not used")
    parser.add_argument("--valid-only", action="store_true", help="Only use source routes whose physical_flyby_summary statuses are ok")
    parser.add_argument("--central-body", default="Sun", help="Central body / observer for SPICE states")
    parser.add_argument("--mu-central-km3-s2", type=float, default=None, help="Override central-body GM in km^3/s^2")
    parser.add_argument("--ref-frame", default="J2000", help="SPICE reference frame")
    parser.add_argument("--max-revs", type=int, default=0, help="Max Lambert revolutions")

    parser.add_argument("--depart-window-days", type=float, default=120.0, help="± search window around source departure day")
    parser.add_argument("--tof-window-days", type=float, default=180.0, help="± search window around each source leg TOF")
    parser.add_argument("--layover-window-days", type=float, default=240.0, help="± search window around each source layover")
    parser.add_argument("--depart-min-day", type=float, default=0.0, help="Absolute lower bound for departure day from coverage start")
    parser.add_argument("--depart-max-day", type=float, default=3650.0, help="Absolute upper bound for departure day from coverage start")
    parser.add_argument("--tof-min-days", type=float, default=30.0, help="Absolute lower bound per leg TOF")
    parser.add_argument("--tof-max-days", type=float, default=5000.0, help="Absolute upper bound per leg TOF")
    parser.add_argument("--layover-min-days", type=float, default=0.0, help="Absolute lower bound per flyby layover")
    parser.add_argument("--layover-max-days", type=float, default=1000.0, help="Absolute upper bound per encounter layover")
    parser.add_argument("--encounter-mode", choices=["flyby", "stopover"], default="flyby", help="flyby forces short layovers; stopover allows parking/rendezvous-like waits")
    parser.add_argument("--max-flyby-layover-days", type=float, default=3.0, help="Maximum allowed layover at intermediate bodies in --encounter-mode flyby")

    parser.add_argument("--max-vinf-mag-jump", type=float, default=4.0, help="Hard maximum |vinf_out|-|vinf_in| mismatch at flyby")
    parser.add_argument("--max-turn-angle-deg", type=float, default=170.0, help="Hard proxy turn angle cap; <=0 disables")
    parser.add_argument("--min-rp-margin-km", type=float, default=0.0, help="Require rp_required-rp_min above this")
    parser.add_argument("--missing-body-action", choices=["reject", "allow", "warn"], default="reject", help="Action when flyby body missing in catalog")
    parser.add_argument("--flyby-vinf-mode", choices=["conservative", "average", "incoming", "outgoing", "max", "rms"], default="conservative", help="Effective v∞ for turn-angle envelope")

    parser.add_argument("--vinf-depart-weight", type=float, default=1.0, help="Leg score weight on departure v∞")
    parser.add_argument("--vinf-arrive-weight", type=float, default=0.35, help="Leg score weight on arrival v∞")
    parser.add_argument("--c3-weight", type=float, default=0.0, help="Leg score weight on C3")
    parser.add_argument("--tof-weight", type=float, default=0.05, help="Leg score weight on TOF years")
    parser.add_argument("--vinf-jump-weight", type=float, default=1.5, help="Transition score weight on v∞ magnitude mismatch")
    parser.add_argument("--turn-angle-weight", type=float, default=0.6, help="Transition score weight on turn angle / 180")
    parser.add_argument("--layover-weight", type=float, default=0.05, help="Transition score weight on layover years")
    parser.add_argument("--flyby-layover-weight", type=float, default=0.25, help="Extra normalized penalty for nonzero layover in --encounter-mode flyby")
    parser.add_argument("--flyby-margin-weight", type=float, default=0.1, help="Small penalty for low rp margin")
    parser.add_argument("--hard-constraint-penalty", type=float, default=1.0e5, help="Penalty for violated hard constraints")

    parser.add_argument("--algorithm", choices=["de", "sade", "pso", "bee_colony"], default="de", help="PyGMO algorithm")
    parser.add_argument("--generations", type=int, default=160, help="Algorithm generations")
    parser.add_argument("--population", type=int, default=64, help="Population size per run")
    parser.add_argument("--runs", type=int, default=4, help="Independent PyGMO runs per source route")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    parser.add_argument("--de-f", type=float, default=0.8, help="Differential evolution F")
    parser.add_argument("--de-cr", type=float, default=0.9, help="Differential evolution CR")
    parser.add_argument("--de-variant", type=int, default=2, help="Differential evolution variant")
    parser.add_argument("--workers", type=int, default=1, help="Parallel worker processes. Use 1 for serial, 0 for os.cpu_count(). Parallelism is per source route.")
    parser.add_argument("--multiprocessing-start-method", choices=["spawn", "fork", "forkserver"], default="spawn", help="Python multiprocessing start method. spawn is safest with SPICE/PyGMO because each child imports and furnishes kernels independently.")

    parser.add_argument("--output-csv", required=True, type=Path, help="Output CSV")
    parser.add_argument("--output-jsonl", required=True, type=Path, help="Output JSONL")
    parser.add_argument("--output-json", required=True, type=Path, help="Output summary JSON")
    args = parser.parse_args(argv)

    if args.max_routes <= 0 and args.route_index is None:
        parser.error("--max-routes must be positive unless --route-index is used")
    if args.population < 5:
        parser.error("--population should be at least 5")
    if args.runs < 1:
        parser.error("--runs must be >= 1")
    if args.depart_max_day <= args.depart_min_day:
        parser.error("--depart-max-day must be > --depart-min-day")
    if args.tof_max_days <= args.tof_min_days:
        parser.error("--tof-max-days must be > --tof-min-days")
    if args.layover_max_days < args.layover_min_days:
        parser.error("--layover-max-days must be >= --layover-min-days")
    if args.max_flyby_layover_days < 0:
        parser.error("--max-flyby-layover-days must be >= 0")
    if args.encounter_mode == "flyby" and args.max_flyby_layover_days < args.layover_min_days:
        parser.error("In flyby mode, --max-flyby-layover-days must be >= --layover-min-days")
    if args.workers < 0:
        parser.error("--workers must be >= 0")
    return args


def _worker_refine_one_route(payload: Mapping[str, Any]) -> Tuple[int, RefinedRoute, Dict[str, Any]]:
    """Worker entry point for multiprocessing.

    Each process imports scientific modules and furnishes SPICE kernels locally.
    This avoids pickling module objects and avoids relying on inherited CSPICE
    kernel-pool state. Parallelism is intentionally coarse: one source route per
    worker task, with all PyGMO runs for that route executed inside the task.
    """
    idx = int(payload["idx"])
    total = int(payload["total"])
    args = argparse.Namespace(**payload["args"])
    record = payload["record"]
    central_mu = float(payload["central_mu"])
    catalog = payload["catalog"]

    spice = import_spiceypy()
    pk = import_pykep()
    pg = import_pygmo()
    furnish_kernels(spice, [args.tpc, args.bsp])

    try:
        seq = route_sequence(record)
        coverage_start = infer_coverage_start_from_route(record)
        if coverage_start is None:
            raise RuntimeError("Could not infer coverage_start_et from route legs. Ensure routes JSONL contains depart_days_from_coverage_start.")
        seed_x, labels = seed_decision_vector(record, coverage_start)
        evaluator = FixedSequenceLambertEvaluator(
            spice=spice,
            pykep=pk,
            sequence=seq,
            central_body=args.central_body,
            ref_frame=args.ref_frame,
            central_mu=central_mu,
            coverage_start_et=coverage_start,
            body_catalog=catalog,
            args=args,
        )
        route = optimize_one_route(pg=pg, evaluator=evaluator, source_record=record, seed_x=seed_x, labels=labels, args=args)
        info = {
            "idx": idx,
            "total": total,
            "sequence": " -> ".join(seq),
            "seed": dict(zip(labels, [round(float(v), 3) for v in seed_x])),
            "objective": route.objective,
            "valid": route.valid,
            "status": route.status,
            "total_tof_days": route.total_tof_days,
        }
        return idx, route, info
    finally:
        try:
            spice.kclear()
        except Exception:
            pass


def resolve_worker_count(requested: int) -> int:
    if requested == 0:
        return max(1, os.cpu_count() or 1)
    return max(1, requested)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    spice = import_spiceypy()
    pk = import_pykep()
    pg = import_pygmo()

    furnish_kernels(spice, [args.tpc, args.bsp])
    central_mu = load_central_mu(spice, args.metadata, args.central_body, args.mu_central_km3_s2)
    catalog = load_body_catalog(args.body_catalog)
    records = read_routes_jsonl(args.routes_jsonl)
    selected = select_routes(records, args)
    if not selected:
        raise RuntimeError("No source routes selected. Check --sequence, --route-index, --valid-only and input JSONL.")

    refined: List[RefinedRoute] = []
    worker_count = resolve_worker_count(args.workers)

    if worker_count == 1 or len(selected) == 1:
        for idx, record in enumerate(selected, start=1):
            seq = route_sequence(record)
            coverage_start = infer_coverage_start_from_route(record)
            if coverage_start is None:
                raise RuntimeError("Could not infer coverage_start_et from route legs. Ensure routes JSONL contains depart_days_from_coverage_start.")
            seed_x, labels = seed_decision_vector(record, coverage_start)
            evaluator = FixedSequenceLambertEvaluator(
                spice=spice,
                pykep=pk,
                sequence=seq,
                central_body=args.central_body,
                ref_frame=args.ref_frame,
                central_mu=central_mu,
                coverage_start_et=coverage_start,
                body_catalog=catalog,
                args=args,
            )
            print(f"[INFO] Refining {idx}/{len(selected)}: {' -> '.join(seq)} seed_x={dict(zip(labels, [round(v, 3) for v in seed_x]))}")
            route = optimize_one_route(pg=pg, evaluator=evaluator, source_record=record, seed_x=seed_x, labels=labels, args=args)
            refined.append(route)
            print(f"[INFO]   obj={route.objective:.6f} valid={route.valid} status={route.status} TOF={route.total_tof_days:.2f} d")
    else:
        # Parent only uses SPICE for central-mu discovery. Children import modules and
        # furnish kernels independently; do not pass module objects across processes.
        try:
            spice.kclear()
        except Exception:
            pass

        jobs = []
        args_payload = dict(vars(args))
        for idx, record in enumerate(selected, start=1):
            seq = route_sequence(record)
            seed_preview = {}
            coverage_start = infer_coverage_start_from_route(record)
            if coverage_start is not None:
                try:
                    sx, labels = seed_decision_vector(record, coverage_start)
                    seed_preview = dict(zip(labels, [round(float(v), 3) for v in sx]))
                except Exception:
                    seed_preview = {}
            print(f"[INFO] Dispatching {idx}/{len(selected)}: {' -> '.join(seq)} seed_x={seed_preview}")
            jobs.append({
                "idx": idx,
                "total": len(selected),
                "args": args_payload,
                "record": record,
                "central_mu": central_mu,
                "catalog": catalog,
            })

        print(f"[INFO] Multiprocessing enabled: workers={worker_count}, start_method={args.multiprocessing_start_method}, tasks={len(jobs)}")
        ctx = mp.get_context(args.multiprocessing_start_method)
        with ProcessPoolExecutor(max_workers=worker_count, mp_context=ctx) as pool:
            future_map = {pool.submit(_worker_refine_one_route, job): job["idx"] for job in jobs}
            for future in as_completed(future_map):
                idx, route, info = future.result()
                refined.append(route)
                print(
                    f"[INFO] Done {idx}/{len(selected)}: {info['sequence']} "
                    f"obj={route.objective:.6f} valid={route.valid} status={route.status} "
                    f"TOF={route.total_tof_days:.2f} d"
                )

    refined.sort(key=lambda r: (not r.valid, r.objective, r.total_tof_days, r.refined_id))
    write_csv(args.output_csv, refined)
    write_jsonl(args.output_jsonl, refined)
    summary = make_summary(refined, args, central_mu)
    write_json(args.output_json, summary)
    print_report(refined)
    print(f"[OK] wrote CSV:   {args.output_csv}")
    print(f"[OK] wrote JSONL: {args.output_jsonl}")
    print(f"[OK] wrote JSON:  {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
