#!/usr/bin/env python3
"""
mga_beam_search_v0_3.py

V0.2 beam-search expander for an offline-first KSP + Principia MGA pipeline.

Purpose
-------
Consume one or more JSONL leg-seed archives produced by mga_candidate_postprocess.py
and assemble coarse multi-leg route genomes.

This is intentionally still planning-grade:
  * Leg seeds are Lambert arcs, not truth-model trajectories.
  * Flybys are checked with v-infinity continuity plus a physical turn-angle envelope.
  * No B-plane targeting, no powered-flyby closure, no N-body closure.
  * The output is a ranked route-frontier for the next local-targeting layer.

Recommended use
---------------
First build a leg library, not only Kerbin -> targets. For example, run the
Lambert scout/postprocess once per useful origin body, then pass all JSONL files:

  python mga_beam_search_v0_3.py \
    --input-jsonl data/mga_v0_1/legs/*_lambert_leg_seeds.jsonl \
    --body-catalog data/mga_v0_1/body_catalog_v0_1.json \
    --start-body Kerbin \
    --final-targets Jool Sarnus Urlum Neidon Plock Soden \
    --require-final-target \
    --min-depth 2 \
    --max-depth 3 \
    --beam-width 500 \
    --output-csv data/mga_v0_1/mga_routes_v0_2.csv \
    --output-jsonl data/mga_v0_1/mga_routes_v0_2.jsonl \
    --output-json data/mga_v0_1/mga_routes_v0_2.summary.json
"""

from __future__ import annotations

import argparse
import bisect
import csv
import hashlib
import glob
import json
import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, DefaultDict, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

SECONDS_PER_DAY = 86_400.0
DAYS_PER_YEAR = 365.25
SCHEMA_VERSION = "mga_beam_search.v0.3"
ROUTE_SCHEMA_VERSION = "mga_route_genome.v0.3"
NODE_SCHEMA_VERSION = "mga_beam_node.v0.3"

Vec3 = Tuple[float, float, float]


# ---------------------------------------------------------------------------
# Data contracts
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LegSeed:
    schema_version: str
    seed_id: str
    source_leg_id: str
    origin: str
    target: str
    central_body: str
    ref_frame: str
    depart_et: float
    arrive_et: float
    tof_s: float
    depart_days_from_coverage_start: Optional[float]
    arrive_days_from_coverage_start: Optional[float]
    tof_days: float
    vinf_depart_km_s: float
    vinf_arrive_km_s: float
    c3_km2_s2: float
    score: float
    cw: bool
    solution_index: int
    max_revs: int
    policy_grade: str
    policy_requires_revalidation: bool
    policy_reason: str
    origin_r_km: Vec3
    origin_v_km_s: Vec3
    target_r_km: Vec3
    target_v_km_s: Vec3
    sc_v_depart_km_s: Vec3
    sc_v_arrive_km_s: Vec3
    cluster_id: str
    cluster_key: Mapping[str, Any]
    cluster_rank: int
    target_rank: int
    tags: Tuple[str, ...] = field(default_factory=tuple)
    source_file: str = ""

    @staticmethod
    def from_record(record: Mapping[str, Any], source_file: str = "") -> "LegSeed":
        def f(key: str, default: float = 0.0) -> float:
            value = record.get(key, default)
            if value is None or value == "":
                return default
            try:
                x = float(value)
            except (TypeError, ValueError):
                return default
            return x if math.isfinite(x) else default

        def i(key: str, default: int = 0) -> int:
            value = record.get(key, default)
            try:
                return int(value)
            except (TypeError, ValueError):
                try:
                    return int(float(value))
                except (TypeError, ValueError):
                    return default

        def b(key: str, default: bool = False) -> bool:
            value = record.get(key, default)
            if isinstance(value, bool):
                return value
            value_s = str(value).strip().lower()
            if value_s in {"1", "true", "t", "yes", "y"}:
                return True
            if value_s in {"0", "false", "f", "no", "n"}:
                return False
            return default

        def optf(key: str) -> Optional[float]:
            value = record.get(key, None)
            if value is None or value == "":
                return None
            try:
                x = float(value)
            except (TypeError, ValueError):
                return None
            return x if math.isfinite(x) else None

        def vec(key: str) -> Vec3:
            raw = record.get(key, (0.0, 0.0, 0.0))
            if isinstance(raw, str):
                try:
                    raw = json.loads(raw)
                except json.JSONDecodeError:
                    raw = (0.0, 0.0, 0.0)
            if not isinstance(raw, (list, tuple)) or len(raw) != 3:
                return (0.0, 0.0, 0.0)
            return (float(raw[0]), float(raw[1]), float(raw[2]))

        return LegSeed(
            schema_version=str(record.get("schema_version", "mga_leg_seed.v0.1")),
            seed_id=str(record.get("seed_id", record.get("leg_id", ""))),
            source_leg_id=str(record.get("source_leg_id", record.get("leg_id", ""))),
            origin=str(record.get("origin", "")),
            target=str(record.get("target", "")),
            central_body=str(record.get("central_body", "")),
            ref_frame=str(record.get("ref_frame", "J2000")),
            depart_et=f("depart_et"),
            arrive_et=f("arrive_et"),
            tof_s=f("tof_s"),
            depart_days_from_coverage_start=optf("depart_days_from_coverage_start"),
            arrive_days_from_coverage_start=optf("arrive_days_from_coverage_start"),
            tof_days=f("tof_days"),
            vinf_depart_km_s=f("vinf_depart_km_s"),
            vinf_arrive_km_s=f("vinf_arrive_km_s"),
            c3_km2_s2=f("c3_km2_s2"),
            score=f("score"),
            cw=b("cw"),
            solution_index=i("solution_index"),
            max_revs=i("max_revs"),
            policy_grade=str(record.get("policy_grade", "unknown")),
            policy_requires_revalidation=b("policy_requires_revalidation"),
            policy_reason=str(record.get("policy_reason", "")),
            origin_r_km=vec("origin_r_km"),
            origin_v_km_s=vec("origin_v_km_s"),
            target_r_km=vec("target_r_km"),
            target_v_km_s=vec("target_v_km_s"),
            sc_v_depart_km_s=vec("sc_v_depart_km_s"),
            sc_v_arrive_km_s=vec("sc_v_arrive_km_s"),
            cluster_id=str(record.get("cluster_id", "")),
            cluster_key=record.get("cluster_key", {}) if isinstance(record.get("cluster_key", {}), Mapping) else {},
            cluster_rank=i("cluster_rank"),
            target_rank=i("target_rank"),
            tags=tuple(record.get("tags", ()) or ()),
            source_file=source_file,
        )

    def vinf_depart_vec(self) -> Vec3:
        return vec_sub(self.sc_v_depart_km_s, self.origin_v_km_s)

    def vinf_arrive_vec(self) -> Vec3:
        return vec_sub(self.sc_v_arrive_km_s, self.target_v_km_s)


@dataclass(frozen=True)
class TransitionMetrics:
    from_body: str
    from_seed_id: str
    to_seed_id: str
    layover_s: float
    layover_days: float
    vinf_in_km_s: float
    vinf_out_km_s: float
    vinf_mag_jump_km_s: float
    turn_angle_deg: float
    # Physical unpowered-flyby envelope at from_body. These values are
    # planning-grade and assume a single hyperbolic passage with effective
    # v_infinity selected by --flyby-vinf-mode.
    flyby_body: str
    flyby_mu_km3_s2: Optional[float]
    flyby_rp_min_km: Optional[float]
    flyby_vinf_eff_km_s: Optional[float]
    turn_angle_max_deg: Optional[float]
    turn_angle_margin_deg: Optional[float]
    rp_required_km: Optional[float]
    rp_margin_km: Optional[float]
    physical_flyby_ok: bool
    physical_flyby_status: str
    transition_score: float


@dataclass(frozen=True)
class BodyCatalogEntry:
    name: str
    mu_km3_s2: Optional[float]
    radius_km: Optional[float]
    rp_min_km: Optional[float]
    allow_flyby: bool
    target_grade: str = "unknown"
    requires_local_revalidation: bool = False
    raw: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RouteGenome:
    schema_version: str
    route_id: str
    sequence: Tuple[str, ...]
    leg_seed_ids: Tuple[str, ...]
    depart_et: float
    arrive_et: float
    total_tof_s: float
    total_layover_s: float
    depth: int
    central_body: str
    ref_frame: str
    sum_leg_score: float
    sum_transition_score: float
    nominal_score: float
    max_vinf_depart_km_s: float
    max_vinf_arrive_km_s: float
    max_vinf_mag_jump_km_s: float
    max_turn_angle_deg: float
    requires_local_revalidation_count: int
    policy_grades: Tuple[str, ...]
    terminal: bool
    tags: Tuple[str, ...] = field(default_factory=tuple)

    @property
    def total_tof_days(self) -> float:
        return self.total_tof_s / SECONDS_PER_DAY

    @property
    def total_layover_days(self) -> float:
        return self.total_layover_s / SECONDS_PER_DAY


@dataclass(frozen=True)
class BeamNode:
    schema_version: str
    node_id: str
    route: RouteGenome
    last_body: str
    last_arrival_et: float
    depth: int
    parent_node_id: Optional[str]
    dominance_bucket: str
    status: str = "open"


@dataclass(frozen=True)
class RouteState:
    route: RouteGenome
    legs: Tuple[LegSeed, ...]
    transitions: Tuple[TransitionMetrics, ...]
    parent_node_id: Optional[str]


# ---------------------------------------------------------------------------
# Vector/math helpers
# ---------------------------------------------------------------------------


def vec_sub(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def vec_norm(a: Vec3) -> float:
    return math.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2])


def vec_dot(a: Vec3, b: Vec3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def angle_deg(a: Vec3, b: Vec3) -> float:
    na = vec_norm(a)
    nb = vec_norm(b)
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    c = max(-1.0, min(1.0, vec_dot(a, b) / (na * nb)))
    return math.degrees(math.acos(c))


def stable_id(prefix: str, payload: Mapping[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    return f"{prefix}_{hashlib.sha1(raw.encode('utf-8')).hexdigest()[:16]}"


def finite_float(x: float, default: float = 0.0) -> float:
    return x if math.isfinite(x) else default


def parse_optional_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def parse_optional_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    v = str(value).strip().lower()
    if v in {"1", "true", "t", "yes", "y", "allow", "allowed"}:
        return True
    if v in {"0", "false", "f", "no", "n", "deny", "blocked"}:
        return False
    return default


def load_body_catalog(path: Optional[Path]) -> Dict[str, BodyCatalogEntry]:
    if path is None:
        return {}
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as f:
        root = json.load(f)
    bodies = root.get("bodies", root)
    if not isinstance(bodies, Mapping):
        raise ValueError(f"Body catalog {path} does not contain a 'bodies' mapping")

    out: Dict[str, BodyCatalogEntry] = {}
    for name, raw in bodies.items():
        if not isinstance(raw, Mapping):
            continue
        policy = raw.get("policy", {}) if isinstance(raw.get("policy", {}), Mapping) else {}
        target_grade = str(policy.get("target_grade", raw.get("target_grade", "unknown")))
        requires_revalidation = parse_optional_bool(
            policy.get("requires_local_revalidation", raw.get("requires_local_revalidation", False)),
            False,
        )
        out[str(name)] = BodyCatalogEntry(
            name=str(raw.get("name", name)),
            mu_km3_s2=parse_optional_float(raw.get("mu_km3_s2", raw.get("gm_km3_s2"))),
            radius_km=parse_optional_float(raw.get("radius_km")),
            rp_min_km=parse_optional_float(raw.get("rp_min_km")),
            allow_flyby=parse_optional_bool(raw.get("allow_flyby", True), True),
            target_grade=target_grade,
            requires_local_revalidation=requires_revalidation,
            raw=raw,
        )
    return out


def flyby_vinf_eff(vinf_in: float, vinf_out: float, mode: str) -> float:
    mode = mode.lower().strip()
    if mode == "incoming":
        return vinf_in
    if mode == "outgoing":
        return vinf_out
    if mode == "max":
        return max(vinf_in, vinf_out)
    if mode == "rms":
        return math.sqrt(0.5 * (vinf_in * vinf_in + vinf_out * vinf_out))
    # Conservative default for turn angle: use the larger v∞, because larger
    # v∞ yields a smaller maximum bend angle for the same rp_min.
    if mode == "conservative":
        return max(vinf_in, vinf_out)
    return 0.5 * (vinf_in + vinf_out)


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


def flyby_check(
    body: str,
    vinf_in: float,
    vinf_out: float,
    turn_angle: float,
    catalog: Mapping[str, BodyCatalogEntry],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    if args.disable_physical_flyby_check:
        return {
            "ok": True,
            "status": "disabled",
            "mu": None,
            "rp_min": None,
            "vinf_eff": None,
            "turn_max": None,
            "turn_margin": None,
            "rp_required": None,
            "rp_margin": None,
            "score_penalty": 0.0,
        }

    entry = catalog.get(body)
    if entry is None:
        action = args.missing_body_action
        ok = action != "reject"
        return {
            "ok": ok,
            "status": "missing_body_catalog",
            "mu": None,
            "rp_min": None,
            "vinf_eff": None,
            "turn_max": None,
            "turn_margin": None,
            "rp_required": None,
            "rp_margin": None,
            "score_penalty": 0.0 if ok else 1.0e99,
        }
    if not entry.allow_flyby:
        return {
            "ok": False,
            "status": "flyby_not_allowed",
            "mu": entry.mu_km3_s2,
            "rp_min": entry.rp_min_km,
            "vinf_eff": None,
            "turn_max": None,
            "turn_margin": None,
            "rp_required": None,
            "rp_margin": None,
            "score_penalty": 1.0e99,
        }
    if entry.mu_km3_s2 is None or entry.rp_min_km is None:
        action = args.missing_body_action
        ok = action != "reject"
        return {
            "ok": ok,
            "status": "missing_mu_or_rp_min",
            "mu": entry.mu_km3_s2,
            "rp_min": entry.rp_min_km,
            "vinf_eff": None,
            "turn_max": None,
            "turn_margin": None,
            "rp_required": None,
            "rp_margin": None,
            "score_penalty": 0.0 if ok else 1.0e99,
        }

    vinf_eff = flyby_vinf_eff(vinf_in, vinf_out, args.flyby_vinf_mode)
    turn_max = max_turn_angle_deg(entry.mu_km3_s2, entry.rp_min_km, vinf_eff)
    rp_req = required_rp_km(entry.mu_km3_s2, vinf_eff, turn_angle)
    rp_margin = None if rp_req is None else rp_req - entry.rp_min_km
    turn_margin = turn_max - turn_angle

    ok = turn_margin >= -1.0e-12
    status = "ok" if ok else "turn_angle_exceeds_physical_envelope"
    if rp_margin is not None and rp_margin < args.min_rp_margin_km:
        ok = False
        status = "rp_margin"
    if args.max_rp_required_ratio > 0.0 and rp_req not in (None, math.inf):
        if rp_req > args.max_rp_required_ratio * entry.rp_min_km:
            # Not physically impossible, but too distant/weak for this coarse
            # patched flyby gate. Keep this optional and disabled by default.
            ok = False
            status = "rp_required_ratio"

    penalty = 0.0
    if ok and args.flyby_margin_weight > 0.0 and rp_margin not in (None, math.inf):
        # Prefer routes with some breathing room without letting this dominate
        # leg scores. The normalization by rp_min keeps small moons sane.
        normalized = max(0.0, rp_margin) / max(entry.rp_min_km, 1.0)
        penalty = args.flyby_margin_weight / (1.0 + normalized)

    return {
        "ok": ok,
        "status": status,
        "mu": entry.mu_km3_s2,
        "rp_min": entry.rp_min_km,
        "vinf_eff": vinf_eff,
        "turn_max": turn_max,
        "turn_margin": turn_margin,
        "rp_required": rp_req,
        "rp_margin": rp_margin,
        "score_penalty": penalty,
    }


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------


def read_jsonl(path: Path) -> List[LegSeed]:
    seeds: List[LegSeed] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_no}: {exc}") from exc
            seed = LegSeed.from_record(record, source_file=str(path))
            if not seed.origin or not seed.target:
                continue
            if seed.arrive_et <= seed.depart_et:
                continue
            seeds.append(seed)
    return seeds


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")


def write_jsonl(path: Path, routes: Sequence[RouteState]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for state in routes:
            payload = {
                "schema_version": SCHEMA_VERSION,
                "route": asdict(state.route),
                "legs": [asdict(leg) for leg in state.legs],
                "transitions": [asdict(t) for t in state.transitions],
                "physical_flyby_summary": physical_summary(state),
            }
            f.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
            f.write("\n")


def fmt_optional(x: Optional[float], digits: int = 9) -> str:
    if x is None:
        return ""
    if x == math.inf:
        return "inf"
    if x == -math.inf:
        return "-inf"
    return f"{x:.{digits}f}"


def physical_summary(state: RouteState) -> Dict[str, Any]:
    transitions = state.transitions
    if not transitions:
        return {
            "flyby_bodies": [],
            "statuses": [],
            "min_rp_margin_km": None,
            "min_turn_angle_margin_deg": None,
            "max_turn_angle_max_deg": None,
            "max_rp_required_km": None,
        }

    rp_margins = [t.rp_margin_km for t in transitions if t.rp_margin_km not in (None, math.inf, -math.inf)]
    turn_margins = [t.turn_angle_margin_deg for t in transitions if t.turn_angle_margin_deg is not None and math.isfinite(t.turn_angle_margin_deg)]
    turn_maxes = [t.turn_angle_max_deg for t in transitions if t.turn_angle_max_deg is not None and math.isfinite(t.turn_angle_max_deg)]
    rp_required = [t.rp_required_km for t in transitions if t.rp_required_km not in (None, math.inf, -math.inf)]
    return {
        "flyby_bodies": [t.flyby_body for t in transitions],
        "statuses": [t.physical_flyby_status for t in transitions],
        "min_rp_margin_km": min(rp_margins) if rp_margins else None,
        "min_turn_angle_margin_deg": min(turn_margins) if turn_margins else None,
        "max_turn_angle_max_deg": max(turn_maxes) if turn_maxes else None,
        "max_rp_required_km": max(rp_required) if rp_required else None,
    }


def write_csv(path: Path, routes: Sequence[RouteState]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "route_id",
        "sequence",
        "depth",
        "terminal",
        "depart_et",
        "arrive_et",
        "depart_days_from_coverage_start",
        "arrive_days_from_coverage_start",
        "total_tof_days",
        "total_layover_days",
        "nominal_score",
        "sum_leg_score",
        "sum_transition_score",
        "max_vinf_depart_km_s",
        "max_vinf_arrive_km_s",
        "max_vinf_mag_jump_km_s",
        "max_turn_angle_deg",
        "min_rp_margin_km",
        "min_turn_angle_margin_deg",
        "max_turn_angle_max_deg",
        "max_rp_required_km",
        "flyby_bodies",
        "physical_flyby_statuses",
        "requires_local_revalidation_count",
        "policy_grades",
        "leg_seed_ids",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for state in routes:
            route = state.route
            first_leg = state.legs[0]
            last_leg = state.legs[-1]
            phys = physical_summary(state)
            writer.writerow(
                {
                    "route_id": route.route_id,
                    "sequence": "->".join(route.sequence),
                    "depth": route.depth,
                    "terminal": int(route.terminal),
                    "depart_et": f"{route.depart_et:.9f}",
                    "arrive_et": f"{route.arrive_et:.9f}",
                    "depart_days_from_coverage_start": "" if first_leg.depart_days_from_coverage_start is None else f"{first_leg.depart_days_from_coverage_start:.9f}",
                    "arrive_days_from_coverage_start": "" if last_leg.arrive_days_from_coverage_start is None else f"{last_leg.arrive_days_from_coverage_start:.9f}",
                    "total_tof_days": f"{route.total_tof_days:.9f}",
                    "total_layover_days": f"{route.total_layover_days:.9f}",
                    "nominal_score": f"{route.nominal_score:.9f}",
                    "sum_leg_score": f"{route.sum_leg_score:.9f}",
                    "sum_transition_score": f"{route.sum_transition_score:.9f}",
                    "max_vinf_depart_km_s": f"{route.max_vinf_depart_km_s:.9f}",
                    "max_vinf_arrive_km_s": f"{route.max_vinf_arrive_km_s:.9f}",
                    "max_vinf_mag_jump_km_s": f"{route.max_vinf_mag_jump_km_s:.9f}",
                    "max_turn_angle_deg": f"{route.max_turn_angle_deg:.9f}",
                    "min_rp_margin_km": fmt_optional(phys.get("min_rp_margin_km")),
                    "min_turn_angle_margin_deg": fmt_optional(phys.get("min_turn_angle_margin_deg")),
                    "max_turn_angle_max_deg": fmt_optional(phys.get("max_turn_angle_max_deg")),
                    "max_rp_required_km": fmt_optional(phys.get("max_rp_required_km")),
                    "flyby_bodies": ";".join(phys.get("flyby_bodies", [])),
                    "physical_flyby_statuses": ";".join(phys.get("statuses", [])),
                    "requires_local_revalidation_count": route.requires_local_revalidation_count,
                    "policy_grades": ";".join(route.policy_grades),
                    "leg_seed_ids": ";".join(route.leg_seed_ids),
                }
            )


# ---------------------------------------------------------------------------
# Beam-search core
# ---------------------------------------------------------------------------


class LegIndex:
    def __init__(self, legs: Sequence[LegSeed]) -> None:
        self.by_origin: Dict[str, List[LegSeed]] = defaultdict(list)
        for leg in legs:
            self.by_origin[leg.origin].append(leg)
        for origin in self.by_origin:
            self.by_origin[origin].sort(key=lambda x: (x.depart_et, x.score, x.arrive_et))
        self.depart_ets: Dict[str, List[float]] = {origin: [leg.depart_et for leg in group] for origin, group in self.by_origin.items()}

    def successors(self, origin: str, min_depart_et: float, max_depart_et: float) -> List[LegSeed]:
        group = self.by_origin.get(origin, [])
        if not group:
            return []
        ets = self.depart_ets[origin]
        i0 = bisect.bisect_left(ets, min_depart_et)
        i1 = bisect.bisect_right(ets, max_depart_et)
        return group[i0:i1]


def transition_metrics(
    prev: LegSeed,
    nxt: LegSeed,
    args: argparse.Namespace,
    body_catalog: Mapping[str, BodyCatalogEntry],
) -> TransitionMetrics:
    layover_s = nxt.depart_et - prev.arrive_et
    vinf_in_vec = prev.vinf_arrive_vec()
    vinf_out_vec = nxt.vinf_depart_vec()
    vinf_in = vec_norm(vinf_in_vec)
    vinf_out = vec_norm(vinf_out_vec)
    mag_jump = abs(vinf_out - vinf_in)
    turn = angle_deg(vinf_in_vec, vinf_out_vec)
    layover_days = layover_s / SECONDS_PER_DAY

    check = flyby_check(prev.target, vinf_in, vinf_out, turn, body_catalog, args)
    transition_score = (
        args.vinf_jump_weight * mag_jump
        + args.turn_angle_weight * (turn / 180.0)
        + args.layover_weight * (layover_days / DAYS_PER_YEAR)
        + float(check.get("score_penalty", 0.0) or 0.0)
    )
    return TransitionMetrics(
        from_body=prev.target,
        from_seed_id=prev.seed_id,
        to_seed_id=nxt.seed_id,
        layover_s=layover_s,
        layover_days=layover_days,
        vinf_in_km_s=vinf_in,
        vinf_out_km_s=vinf_out,
        vinf_mag_jump_km_s=mag_jump,
        turn_angle_deg=turn,
        flyby_body=prev.target,
        flyby_mu_km3_s2=check.get("mu"),
        flyby_rp_min_km=check.get("rp_min"),
        flyby_vinf_eff_km_s=check.get("vinf_eff"),
        turn_angle_max_deg=check.get("turn_max"),
        turn_angle_margin_deg=check.get("turn_margin"),
        rp_required_km=check.get("rp_required"),
        rp_margin_km=check.get("rp_margin"),
        physical_flyby_ok=bool(check.get("ok", False)),
        physical_flyby_status=str(check.get("status", "unknown")),
        transition_score=finite_float(transition_score, 1.0e99),
    )

def transition_allowed(tm: TransitionMetrics, args: argparse.Namespace) -> Tuple[bool, str]:
    if tm.layover_days < args.min_layover_days:
        return False, "min_layover"
    if tm.layover_days > args.max_layover_days:
        return False, "max_layover"
    if args.max_vinf_mag_jump >= 0.0 and tm.vinf_mag_jump_km_s > args.max_vinf_mag_jump:
        return False, "vinf_mag_jump"
    if args.max_turn_angle_deg > 0.0 and tm.turn_angle_deg > args.max_turn_angle_deg:
        return False, "turn_angle_proxy"
    if not tm.physical_flyby_ok:
        return False, tm.physical_flyby_status
    return True, "ok"


def route_terminal(route: RouteGenome, final_targets: Sequence[str], require_final_target: bool) -> bool:
    if not final_targets:
        return not require_final_target
    return route.sequence[-1] in set(final_targets)


def build_initial_route(leg: LegSeed, args: argparse.Namespace) -> RouteState:
    sequence = (leg.origin, leg.target)
    leg_ids = (leg.seed_id,)
    terminal = route_terminal_placeholder(sequence[-1], args)
    payload = {
        "sequence": sequence,
        "leg_seed_ids": leg_ids,
        "depart_et": round(leg.depart_et, 6),
        "arrive_et": round(leg.arrive_et, 6),
    }
    route_id = stable_id("route", payload)
    route = RouteGenome(
        schema_version=ROUTE_SCHEMA_VERSION,
        route_id=route_id,
        sequence=sequence,
        leg_seed_ids=leg_ids,
        depart_et=leg.depart_et,
        arrive_et=leg.arrive_et,
        total_tof_s=leg.tof_s,
        total_layover_s=0.0,
        depth=1,
        central_body=leg.central_body,
        ref_frame=leg.ref_frame,
        sum_leg_score=leg.score,
        sum_transition_score=0.0,
        nominal_score=leg.score,
        max_vinf_depart_km_s=leg.vinf_depart_km_s,
        max_vinf_arrive_km_s=leg.vinf_arrive_km_s,
        max_vinf_mag_jump_km_s=0.0,
        max_turn_angle_deg=0.0,
        requires_local_revalidation_count=1 if leg.policy_requires_revalidation else 0,
        policy_grades=(leg.policy_grade,),
        terminal=terminal,
        tags=tuple(sorted(set(("lambert_seed_route",) + tuple(leg.tags)))),
    )
    return RouteState(route=route, legs=(leg,), transitions=(), parent_node_id=None)


def route_terminal_placeholder(last_body: str, args: argparse.Namespace) -> bool:
    if not args.final_targets:
        return not args.require_final_target
    return last_body in set(args.final_targets)


def extend_route(state: RouteState, nxt: LegSeed, tm: TransitionMetrics, args: argparse.Namespace) -> RouteState:
    prev_route = state.route
    sequence = prev_route.sequence + (nxt.target,)
    leg_ids = prev_route.leg_seed_ids + (nxt.seed_id,)
    transitions = state.transitions + (tm,)
    legs = state.legs + (nxt,)
    sum_leg_score = prev_route.sum_leg_score + nxt.score
    sum_transition_score = prev_route.sum_transition_score + tm.transition_score
    nominal_score = sum_leg_score + sum_transition_score
    payload = {
        "sequence": sequence,
        "leg_seed_ids": leg_ids,
        "depart_et": round(prev_route.depart_et, 6),
        "arrive_et": round(nxt.arrive_et, 6),
    }
    route_id = stable_id("route", payload)
    tags = set(prev_route.tags) | set(nxt.tags)
    if transitions:
        tags.add("coarse_vinf_continuity_checked")
    terminal = route_terminal_placeholder(sequence[-1], args)
    route = RouteGenome(
        schema_version=ROUTE_SCHEMA_VERSION,
        route_id=route_id,
        sequence=sequence,
        leg_seed_ids=leg_ids,
        depart_et=prev_route.depart_et,
        arrive_et=nxt.arrive_et,
        total_tof_s=prev_route.total_tof_s + nxt.tof_s + tm.layover_s,
        total_layover_s=prev_route.total_layover_s + tm.layover_s,
        depth=prev_route.depth + 1,
        central_body=prev_route.central_body,
        ref_frame=prev_route.ref_frame,
        sum_leg_score=sum_leg_score,
        sum_transition_score=sum_transition_score,
        nominal_score=finite_float(nominal_score, 1.0e99),
        max_vinf_depart_km_s=max(prev_route.max_vinf_depart_km_s, nxt.vinf_depart_km_s),
        max_vinf_arrive_km_s=max(prev_route.max_vinf_arrive_km_s, nxt.vinf_arrive_km_s),
        max_vinf_mag_jump_km_s=max(prev_route.max_vinf_mag_jump_km_s, tm.vinf_mag_jump_km_s),
        max_turn_angle_deg=max(prev_route.max_turn_angle_deg, tm.turn_angle_deg),
        requires_local_revalidation_count=prev_route.requires_local_revalidation_count + (1 if nxt.policy_requires_revalidation else 0),
        policy_grades=prev_route.policy_grades + (nxt.policy_grade,),
        terminal=terminal,
        tags=tuple(sorted(tags)),
    )
    parent_node_id = stable_id("node", {"route_id": prev_route.route_id, "depth": prev_route.depth})
    return RouteState(route=route, legs=legs, transitions=transitions, parent_node_id=parent_node_id)


def dominance_bucket(route: RouteGenome, arrival_bin_days: float) -> str:
    if arrival_bin_days <= 0.0:
        arr_bin = 0
    else:
        arr_bin = int(math.floor((route.arrive_et / SECONDS_PER_DAY) / arrival_bin_days))
    # Include the complete sequence to avoid collapsing different gravity-assist topologies too early.
    return f"d{route.depth}|last={route.sequence[-1]}|arr={arr_bin}|seq={'-'.join(route.sequence)}"


def prune_frontier(routes: Sequence[RouteState], args: argparse.Namespace) -> List[RouteState]:
    sorted_routes = sorted(routes, key=lambda s: (s.route.nominal_score, s.route.total_tof_s, s.route.route_id))
    if args.max_per_bucket <= 0:
        return sorted_routes[: args.beam_width]
    counts: Counter[str] = Counter()
    kept: List[RouteState] = []
    for state in sorted_routes:
        b = dominance_bucket(state.route, args.arrival_bin_days)
        if counts[b] >= args.max_per_bucket:
            continue
        counts[b] += 1
        kept.append(state)
        if len(kept) >= args.beam_width:
            break
    return kept


def route_output_allowed(route: RouteGenome, args: argparse.Namespace) -> bool:
    if route.depth < args.min_depth:
        return False
    if args.require_final_target and not route.terminal:
        return False
    if args.max_total_score >= 0.0 and route.nominal_score > args.max_total_score:
        return False
    return True



def body_repeat_allowed(target: str, sequence: Sequence[str], args: argparse.Namespace) -> Tuple[bool, str]:
    """Return whether a candidate target may be visited again.

    V0.3 adds a middle ground between "no repeats" and "repeat anything":
    repeatable bodies such as Kerbin/Earth may be revisited for Galileo-like
    sequences (Eve/Earth/Earth analogues) while still suppressing unhelpful
    loops through every body.
    """
    visits = sum(1 for b in sequence if b == target)
    if visits == 0:
        return True, "ok"

    repeatable = set(getattr(args, "repeatable_bodies", []) or [])
    max_visits = max(1, int(getattr(args, "max_visits_per_repeatable_body", 2)))

    if getattr(args, "allow_repeat_bodies", False):
        if getattr(args, "max_visits_per_body", 0) > 0 and visits >= args.max_visits_per_body:
            return False, "max_visits_per_body"
        return True, "ok"

    if target in repeatable:
        if visits >= max_visits:
            return False, "max_visits_repeatable"
        return True, "ok"

    return False, "repeat_body"


def flyby_body_allowed_by_mission(body: str, args: argparse.Namespace) -> Tuple[bool, str]:
    """Return whether *body* may be used as an intermediate gravity-assist body.

    This is a mission-level gate, independent of physical feasibility. It is used
    to keep searches for e.g. Jool from wasting branches through outer bodies
    such as Neidon/Plock/Soden unless the mission explicitly allows them.
    """
    allowed = set(getattr(args, "allowed_flyby_bodies", []) or [])
    disallowed = set(getattr(args, "disallowed_flyby_bodies", []) or [])
    if body in disallowed:
        return False, "flyby_body_disallowed"
    if allowed and body not in allowed:
        return False, "flyby_body_not_allowed"
    return True, "ok"


def successor_target_allowed_by_mission(target: str, args: argparse.Namespace) -> Tuple[bool, str]:
    """Fast pruning for successor targets.

    A successor target can be:
      * a final target,
      * a permitted flyby/intermediate body,
      * a repeatable body such as Kerbin.

    This avoids carrying branches through bodies that can neither be terminal nor
    legal flybys.
    """
    finals = set(getattr(args, "final_targets", []) or [])
    repeatable = set(getattr(args, "repeatable_bodies", []) or [])
    allowed = set(getattr(args, "allowed_flyby_bodies", []) or [])
    disallowed = set(getattr(args, "disallowed_flyby_bodies", []) or [])
    if target in disallowed and target not in finals:
        return False, "target_body_disallowed"
    if target in finals or target in repeatable:
        return True, "ok"
    if allowed and target not in allowed:
        return False, "target_body_not_allowed"
    return True, "ok"

def search_routes(legs: Sequence[LegSeed], args: argparse.Namespace, body_catalog: Mapping[str, BodyCatalogEntry]) -> Tuple[List[RouteState], Dict[str, Any]]:
    index = LegIndex(legs)
    reject: Counter[str] = Counter()
    level_counts: Dict[str, Any] = {}

    initial_legs = [leg for leg in index.by_origin.get(args.start_body, [])]
    if args.max_leg_score >= 0.0:
        initial_legs = [leg for leg in initial_legs if leg.score <= args.max_leg_score]
    if args.initial_top_n > 0:
        initial_legs = sorted(initial_legs, key=lambda x: (x.score, x.depart_et, x.arrive_et))[: args.initial_top_n]

    frontier = [build_initial_route(leg, args) for leg in initial_legs]
    frontier = prune_frontier(frontier, args)
    all_routes: List[RouteState] = [state for state in frontier if route_output_allowed(state.route, args)]
    level_counts["1"] = {"frontier": len(frontier), "output": len(all_routes), "expanded": 0, "raw_children": 0}

    for depth in range(1, args.max_depth):
        raw_children: List[RouteState] = []
        expanded = 0
        for state in frontier:
            last_leg = state.legs[-1]
            last_body = state.route.sequence[-1]
            min_depart = state.route.arrive_et + args.min_layover_days * SECONDS_PER_DAY
            max_depart = state.route.arrive_et + args.max_layover_days * SECONDS_PER_DAY
            candidates = index.successors(last_body, min_depart, max_depart)
            if not candidates:
                reject["no_successor_leg"] += 1
                continue
            expanded += 1
            local_children: List[Tuple[float, RouteState]] = []
            # If we are extending from last_body, that body becomes an actual
            # flyby/intermediate encounter. Apply the mission-level flyby list here.
            flyby_ok, flyby_reason = flyby_body_allowed_by_mission(last_body, args)
            if not flyby_ok:
                reject[flyby_reason] += len(candidates)
                continue

            for nxt in candidates:
                target_ok, target_reason = successor_target_allowed_by_mission(nxt.target, args)
                if not target_ok:
                    reject[target_reason] += 1
                    continue
                if args.max_leg_score >= 0.0 and nxt.score > args.max_leg_score:
                    reject["leg_score"] += 1
                    continue
                repeat_ok, repeat_reason = body_repeat_allowed(nxt.target, state.route.sequence, args)
                if not repeat_ok:
                    reject[repeat_reason] += 1
                    continue
                if (not args.allow_return_to_start
                        and nxt.target == args.start_body
                        and nxt.target not in set(getattr(args, "repeatable_bodies", []) or [])):
                    reject["return_to_start"] += 1
                    continue
                tm = transition_metrics(last_leg, nxt, args, body_catalog)
                ok, reason = transition_allowed(tm, args)
                if not ok:
                    reject[reason] += 1
                    continue
                child = extend_route(state, nxt, tm, args)
                if args.max_total_tof_days >= 0.0 and child.route.total_tof_days > args.max_total_tof_days:
                    reject["max_total_tof"] += 1
                    continue
                if args.max_total_score >= 0.0 and child.route.nominal_score > args.max_total_score:
                    reject["max_total_score"] += 1
                    continue
                local_children.append((child.route.nominal_score, child))
            local_children.sort(key=lambda x: (x[0], x[1].route.total_tof_s, x[1].route.route_id))
            if args.branch_factor_per_node > 0:
                local_children = local_children[: args.branch_factor_per_node]
            raw_children.extend(child for _, child in local_children)

        frontier = prune_frontier(raw_children, args)
        outputs_this_level = [state for state in frontier if route_output_allowed(state.route, args)]
        all_routes.extend(outputs_this_level)
        level_counts[str(depth + 1)] = {
            "frontier": len(frontier),
            "output": len(outputs_this_level),
            "expanded": expanded,
            "raw_children": len(raw_children),
        }
        if not frontier:
            break

    # Global output pruning keeps the artifact usable while preserving best routes.
    all_routes = sorted(all_routes, key=lambda s: (not s.route.terminal, s.route.nominal_score, s.route.total_tof_s, s.route.route_id))
    if args.output_top_n > 0:
        all_routes = all_routes[: args.output_top_n]

    stats = {
        "levels": level_counts,
        "reject_reasons": dict(reject),
        "input_leg_count": len(legs),
        "origins_available": {k: len(v) for k, v in sorted(index.by_origin.items())},
    }
    return all_routes, stats


# ---------------------------------------------------------------------------
# Summary/reporting
# ---------------------------------------------------------------------------


def summarize(legs: Sequence[LegSeed], routes: Sequence[RouteState], stats: Mapping[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    legs_by_pair: Counter[str] = Counter(f"{leg.origin}->{leg.target}" for leg in legs)
    routes_by_depth: Counter[str] = Counter(str(state.route.depth) for state in routes)
    routes_by_final: Counter[str] = Counter(state.route.sequence[-1] for state in routes)
    best_by_final: Dict[str, Any] = {}
    for state in routes:
        final = state.route.sequence[-1]
        current = best_by_final.get(final)
        if current is None or state.route.nominal_score < current["nominal_score"]:
            best_by_final[final] = route_summary_row(state)

    return {
        "schema_version": SCHEMA_VERSION,
        "purpose": "coarse MGA beam search over Lambert leg seeds with physical flyby gates",
        "inputs": {
            "input_jsonl": [str(p) for p in args.input_jsonl],
            "body_catalog": str(args.body_catalog) if args.body_catalog else None,
        },
        "outputs": {
            "output_csv": str(args.output_csv),
            "output_jsonl": str(args.output_jsonl),
            "output_json": str(args.output_json),
        },
        "search_spec": {
            "start_body": args.start_body,
            "final_targets": args.final_targets,
            "allowed_flyby_bodies": args.allowed_flyby_bodies,
            "disallowed_flyby_bodies": args.disallowed_flyby_bodies,
            "require_final_target": args.require_final_target,
            "min_depth": args.min_depth,
            "max_depth": args.max_depth,
            "min_layover_days": args.min_layover_days,
            "max_layover_days": args.max_layover_days,
            "max_vinf_mag_jump": args.max_vinf_mag_jump,
            "max_turn_angle_deg": args.max_turn_angle_deg,
            "disable_physical_flyby_check": args.disable_physical_flyby_check,
            "missing_body_action": args.missing_body_action,
            "flyby_vinf_mode": args.flyby_vinf_mode,
            "min_rp_margin_km": args.min_rp_margin_km,
            "max_rp_required_ratio": args.max_rp_required_ratio,
            "flyby_margin_weight": args.flyby_margin_weight,
            "beam_width": args.beam_width,
            "branch_factor_per_node": args.branch_factor_per_node,
            "arrival_bin_days": args.arrival_bin_days,
            "max_per_bucket": args.max_per_bucket,
        },
        "counts": {
            "input_legs": len(legs),
            "output_routes": len(routes),
            "terminal_routes": sum(1 for s in routes if s.route.terminal),
            "routes_by_depth": dict(routes_by_depth),
            "routes_by_final_body": dict(routes_by_final),
        },
        "leg_library": {
            "origins_available": stats.get("origins_available", {}),
            "legs_by_pair_top": dict(legs_by_pair.most_common(50)),
        },
        "beam_stats": stats,
        "best_by_final_body": best_by_final,
        "top_routes": [route_summary_row(state) for state in routes[: min(20, len(routes))]],
        "caveats": [
            "Lambert legs are planning seeds only; no N-body closure has been performed.",
            "Transition checks include a planning-grade unpowered flyby turn-angle/periapsis envelope, but they are still not B-plane targeting or N-body closure.",
            "If only start-body legs are provided, the search cannot produce depth > 1 routes.",
        ],
    }


def route_summary_row(state: RouteState) -> Dict[str, Any]:
    r = state.route
    phys = physical_summary(state)
    row = {
        "route_id": r.route_id,
        "sequence": "->".join(r.sequence),
        "depth": r.depth,
        "terminal": r.terminal,
        "nominal_score": r.nominal_score,
        "sum_leg_score": r.sum_leg_score,
        "sum_transition_score": r.sum_transition_score,
        "total_tof_days": r.total_tof_days,
        "total_layover_days": r.total_layover_days,
        "max_vinf_mag_jump_km_s": r.max_vinf_mag_jump_km_s,
        "max_turn_angle_deg": r.max_turn_angle_deg,
        "leg_seed_ids": list(r.leg_seed_ids),
    }
    row.update(phys)
    return row


def print_report(routes: Sequence[RouteState], summary: Mapping[str, Any]) -> None:
    print("=" * 80)
    print("MGA BEAM SEARCH V0.2")
    print("=" * 80)
    print(f"Input legs:      {summary['counts']['input_legs']}")
    print(f"Output routes:   {summary['counts']['output_routes']}")
    print(f"Terminal routes: {summary['counts']['terminal_routes']}")
    print("\nOrigins available:")
    for origin, count in summary["leg_library"]["origins_available"].items():
        print(f"  - {origin:<12} {count}")
    print("\nRoutes by depth:")
    for depth, count in sorted(summary["counts"]["routes_by_depth"].items(), key=lambda kv: int(kv[0])):
        print(f"  - depth {depth}: {count}")
    if summary["beam_stats"].get("reject_reasons"):
        print("\nTop reject reasons:")
        for reason, count in sorted(summary["beam_stats"]["reject_reasons"].items(), key=lambda kv: kv[1], reverse=True)[:8]:
            print(f"  - {reason:<20} {count}")
    print("\nTop routes:")
    for idx, state in enumerate(routes[:10], start=1):
        r = state.route
        phys = physical_summary(state)
        margin = phys.get("min_rp_margin_km")
        margin_s = "n/a" if margin is None else f"{margin:.1f} km"
        print(
            f"{idx:2d}. {' -> '.join(r.sequence)} | depth={r.depth} | "
            f"score={r.nominal_score:.4f} | TOF={r.total_tof_days:.1f} d | "
            f"layover={r.total_layover_days:.1f} d | max Δv∞={r.max_vinf_mag_jump_km_s:.3f} km/s | "
            f"max turn={r.max_turn_angle_deg:.1f}° | min rp margin={margin_s}"
        )
    print("=" * 80)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Assemble coarse MGA route genomes from Lambert leg-seed JSONL archives.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input-jsonl", required=True, nargs="+", type=Path, help="One or more JSONL leg-seed archives")
    parser.add_argument("--body-catalog", type=Path, default=None, help="Body catalog JSON from mga_build_body_catalog_v0_1.py")
    parser.add_argument("--start-body", default="Kerbin", help="Route start body")
    parser.add_argument("--final-targets", nargs="*", default=[], help="Optional final bodies of interest")
    parser.add_argument("--allowed-flyby-bodies", nargs="*", default=[], help="Mission-level whitelist for intermediate/flyby bodies. Empty disables the whitelist.")
    parser.add_argument("--disallowed-flyby-bodies", nargs="*", default=[], help="Mission-level blacklist for intermediate/flyby bodies.")
    parser.add_argument("--require-final-target", action="store_true", help="Only output routes ending at one of --final-targets")
    parser.add_argument("--min-depth", type=int, default=1, help="Minimum number of legs to output")
    parser.add_argument("--max-depth", type=int, default=3, help="Maximum number of legs to expand")
    parser.add_argument("--min-layover-days", type=float, default=0.0, help="Minimum time between incoming and outgoing leg at a flyby body")
    parser.add_argument("--max-layover-days", type=float, default=800.0, help="Maximum time between incoming and outgoing leg at a flyby body")
    parser.add_argument("--max-total-tof-days", type=float, default=-1.0, help="Maximum route duration; negative disables")
    parser.add_argument("--max-leg-score", type=float, default=-1.0, help="Filter individual legs by score; negative disables")
    parser.add_argument("--max-total-score", type=float, default=-1.0, help="Filter route total score; negative disables")
    parser.add_argument("--max-vinf-mag-jump", type=float, default=4.0, help="Max |vinf_out|-|vinf_in| mismatch at transition; negative disables")
    parser.add_argument("--max-turn-angle-deg", type=float, default=170.0, help="Max coarse/proxy turn angle at transition; <=0 disables")
    parser.add_argument("--disable-physical-flyby-check", action="store_true", help="Disable BodyCatalog unpowered-flyby envelope checks")
    parser.add_argument("--missing-body-action", choices=["reject", "warn", "allow"], default="reject", help="What to do when a flyby body is missing from the BodyCatalog")
    parser.add_argument("--flyby-vinf-mode", choices=["conservative", "average", "incoming", "outgoing", "max", "rms"], default="conservative", help="Effective v-infinity used for turn-angle envelope")
    parser.add_argument("--min-rp-margin-km", type=float, default=0.0, help="Require rp_required - rp_min to be at least this margin")
    parser.add_argument("--max-rp-required-ratio", type=float, default=-1.0, help="Optional upper bound on rp_required/rp_min; negative disables")
    parser.add_argument("--flyby-margin-weight", type=float, default=0.0, help="Small score penalty for low physical flyby margin; 0 disables")
    parser.add_argument("--vinf-jump-weight", type=float, default=1.5, help="Transition score weight for v-infinity magnitude mismatch")
    parser.add_argument("--turn-angle-weight", type=float, default=0.6, help="Transition score weight for turn-angle proxy normalized by 180 deg")
    parser.add_argument("--layover-weight", type=float, default=0.05, help="Transition score weight for layover years")
    parser.add_argument("--beam-width", type=int, default=400, help="Routes kept at each depth")
    parser.add_argument("--branch-factor-per-node", type=int, default=40, help="Successors kept per expanded node; 0 keeps all")
    parser.add_argument("--initial-top-n", type=int, default=0, help="Initial start-body legs to keep before beam pruning; 0 keeps all")
    parser.add_argument("--arrival-bin-days", type=float, default=80.0, help="Arrival-time bin for dominance buckets")
    parser.add_argument("--max-per-bucket", type=int, default=8, help="Max routes per dominance bucket; 0 disables bucket cap")
    parser.add_argument("--allow-repeat-bodies", action="store_true", help="Allow a route to visit the same target body more than once")
    parser.add_argument("--repeatable-bodies", nargs="*", default=[], help="Bodies allowed to repeat even when --allow-repeat-bodies is not set, e.g. Kerbin for Galileo-like returns")
    parser.add_argument("--max-visits-per-repeatable-body", type=int, default=2, help="Maximum visits for each --repeatable-bodies entry")
    parser.add_argument("--max-visits-per-body", type=int, default=0, help="When --allow-repeat-bodies is set, optional max visits per body; 0 disables")
    parser.add_argument("--allow-return-to-start", action="store_true", help="Allow route to return to start body")
    parser.add_argument("--output-csv", required=True, type=Path, help="Output route CSV")
    parser.add_argument("--output-jsonl", required=True, type=Path, help="Output route JSONL with legs and transitions")
    parser.add_argument("--output-json", required=True, type=Path, help="Output summary JSON")
    parser.add_argument("--output-top-n", type=int, default=500, help="Max routes written; 0 writes all")
    args = parser.parse_args(argv)

    if args.min_depth < 1:
        parser.error("--min-depth must be >= 1")
    if args.max_depth < args.min_depth:
        parser.error("--max-depth must be >= --min-depth")
    if args.beam_width <= 0:
        parser.error("--beam-width must be positive")
    if args.min_layover_days < 0.0:
        parser.error("--min-layover-days must be non-negative")
    if args.max_layover_days < args.min_layover_days:
        parser.error("--max-layover-days must be >= --min-layover-days")
    if args.min_rp_margin_km < 0.0:
        parser.error("--min-rp-margin-km must be non-negative")
    if not args.disable_physical_flyby_check and args.body_catalog is None:
        parser.error("--body-catalog is required unless --disable-physical-flyby-check is set")
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    legs: List[LegSeed] = []
    expanded_paths: List[Path] = []
    for path in args.input_jsonl:
        text = str(path)
        if any(ch in text for ch in "*?["):
            matches = sorted(Path(x) for x in glob.glob(text))
            if not matches:
                raise FileNotFoundError(path)
            expanded_paths.extend(matches)
        else:
            expanded_paths.append(path)
    args.input_jsonl = expanded_paths
    for path in expanded_paths:
        if not path.exists():
            raise FileNotFoundError(path)
        legs.extend(read_jsonl(path))
    # De-duplicate by seed id, keeping the lowest-score copy if the same file is passed twice.
    dedup: Dict[str, LegSeed] = {}
    for leg in legs:
        old = dedup.get(leg.seed_id)
        if old is None or leg.score < old.score:
            dedup[leg.seed_id] = leg
    legs = list(dedup.values())

    body_catalog = load_body_catalog(args.body_catalog)
    routes, stats = search_routes(legs, args, body_catalog)
    summary = summarize(legs, routes, stats, args)
    write_csv(args.output_csv, routes)
    write_jsonl(args.output_jsonl, routes)
    write_json(args.output_json, summary)
    print_report(routes, summary)
    print(f"[OK] wrote CSV:   {args.output_csv}")
    print(f"[OK] wrote JSONL: {args.output_jsonl}")
    print(f"[OK] wrote JSON:  {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
