#!/usr/bin/env python3
"""
mga_build_body_catalog_v0_1.py

Build a versioned physical body catalog for the KSP + Principia offline-first
MGA planner.

Purpose
-------
The Lambert scout and V0.1 beam search only know about states and v-infinity
vectors. V0.2 needs physical flyby gates:

  * gravitational parameter mu for the flyby body;
  * body radius;
  * minimum safe periapsis radius rp_min;
  * operational policy flags and revalidation notes.

This tool queries SPICE where possible, falls back to the SPICE metadata JSON,
merges target policy information, applies optional user overrides, and writes a
small JSON catalog consumed by mga_beam_search_v0_2.py.

Recommended example
-------------------
  python mga_build_body_catalog_v0_1.py \
    --bsp data/spice_v0_1_33y/opm_mpe_principia_rebound_v0_1_33y.bsp \
    --tpc data/spice_v0_1_33y/opm_mpe_principia_rebound_v0_1_33y.ids.tpc \
    --metadata data/spice_v0_1_33y/opm_mpe_principia_rebound_v0_1_33y.metadata.json \
    --policy data/spice_v0_1_33y/target_policy_v0_1.json \
    --input-jsonl data/mga_v0_1/legs/*_lambert_leg_seeds.jsonl \
    --default-min-altitude-km 50 \
    --default-atmosphere-margin-km 20 \
    --output-json data/mga_v0_1/body_catalog_v0_1.json

Override file format
--------------------
{
  "bodies": {
    "Duna": {
      "min_flyby_altitude_km": 80,
      "atmosphere_top_km": 50,
      "safety_margin_km": 20,
      "allow_flyby": true,
      "notes": "custom Duna flyby margin"
    }
  }
}

All distances are km and mu is km^3/s^2.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Set, Tuple

SCHEMA_VERSION = "mga_body_catalog.v0.1"


@dataclass(frozen=True)
class BodyPolicy:
    allowed_for_global: bool = True
    allowed_for_fine_targeting: bool = False
    requires_local_revalidation: bool = False
    target_grade: str = "unknown"
    reason: str = ""


@dataclass(frozen=True)
class BodyCatalogEntry:
    name: str
    naif_id: Optional[int]
    mu_km3_s2: Optional[float]
    radius_km: Optional[float]
    atmosphere_top_km: float
    min_flyby_altitude_km: float
    safety_margin_km: float
    rp_min_km: Optional[float]
    allow_flyby: bool
    policy: BodyPolicy
    source_notes: Tuple[str, ...] = field(default_factory=tuple)
    raw_overrides: Mapping[str, Any] = field(default_factory=dict)


def is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(float(x))


def scalar_from_any(value: Any) -> Optional[float]:
    if is_number(value):
        return float(value)
    if isinstance(value, str):
        try:
            x = float(value)
        except ValueError:
            return None
        return x if math.isfinite(x) else None
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and value:
        return scalar_from_any(value[0])
    return None


def bool_from_any(value: Any, default: Optional[bool] = None) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return bool(value)
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"1", "true", "t", "yes", "y", "allow", "allowed", "enabled"}:
            return True
        if v in {"0", "false", "f", "no", "n", "deny", "blocked", "disabled", "forbidden"}:
            return False
    return default


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
        k_l = str(k).lower()
        v_l = str(v).lower()
        if k_l == body_l:
            return True
        if k_l in {"name", "body", "target", "naif_name", "id"} and v_l == body_l:
            return True
    return False


def extract_named_body_dict(root: Mapping[str, Any], body: str) -> Dict[str, Any]:
    body_l = body.lower()
    for container_key in ("bodies", "body_catalog", "catalog", "targets", "target_policy", "policies", "physical_parameters"):
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


def load_json(path: Optional[Path]) -> Dict[str, Any]:
    if path is None:
        return {}
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_bodies_from_jsonl(paths: Sequence[Path]) -> Set[str]:
    bodies: Set[str] = set()
    for path in paths:
        with path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
                for key in ("origin", "target"):
                    value = rec.get(key)
                    if value:
                        bodies.add(str(value))
    return bodies


def convert_mu_to_km3_s2(value: float, key_hint: str = "") -> float:
    key_l = key_hint.lower()
    if "m3" in key_l or "m^3" in key_l:
        return value / 1.0e9
    if "km3" in key_l or "km^3" in key_l:
        return value
    if abs(value) > 1.0e15:
        return value / 1.0e9
    return value


def convert_radius_to_km(value: float, key_hint: str = "") -> float:
    key_l = key_hint.lower()
    if key_l.endswith("_m") or "radius_m" in key_l or "radii_m" in key_l or "altitude_m" in key_l:
        return value / 1000.0
    if key_l.endswith("_km") or "radius_km" in key_l or "altitude_km" in key_l:
        return value
    # KSP stock radii can be hundreds of km; real-scale radii thousands of km.
    # A bare radius/altitude above 100000 is almost certainly metres.
    if abs(value) > 1.0e5:
        return value / 1000.0
    return value


def find_mu_in_metadata(metadata: Mapping[str, Any], body: str) -> Tuple[Optional[float], str]:
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
                return convert_mu_to_km3_s2(value, key), f"metadata.{key}"
    return None, "not_found"


def find_radius_in_metadata(metadata: Mapping[str, Any], body: str) -> Tuple[Optional[float], str]:
    body_d = extract_named_body_dict(metadata, body)
    keys = (
        "radius_km", "mean_radius_km", "equatorial_radius_km", "radii_km",
        "radius_m", "mean_radius_m", "equatorial_radius_m", "radii_m",
        "radius", "mean_radius", "equatorial_radius", "radii",
    )
    for key in keys:
        if key in body_d:
            value = scalar_from_any(body_d[key])
            if value is not None:
                return convert_radius_to_km(value, key), f"metadata.{key}"
    return None, "not_found"


def find_atmosphere_top_km(metadata: Mapping[str, Any], body: str) -> Tuple[float, str]:
    body_d = extract_named_body_dict(metadata, body)
    keys = (
        "atmosphere_top_km", "atmosphere_height_km", "atmosphere_depth_km", "atmosphere_altitude_km",
        "atmosphere_top_m", "atmosphere_height_m", "atmosphere_depth_m", "atmosphere_altitude_m",
        "atmosphere_top", "atmosphere_height", "atmosphere_depth", "atmosphere_altitude",
    )
    for key in keys:
        if key in body_d:
            value = scalar_from_any(body_d[key])
            if value is not None:
                return max(0.0, convert_radius_to_km(value, key)), f"metadata.{key}"
    # Some KSP configs use "Atmosphere" booleans and max altitude elsewhere; do
    # not guess a height from boolean alone.
    return 0.0, "default_zero"


def infer_body_policy(policy_root: Mapping[str, Any], body: str) -> BodyPolicy:
    if not policy_root:
        return BodyPolicy(reason="no policy file supplied")
    body_d = extract_named_body_dict(policy_root, body)
    if not body_d:
        return BodyPolicy(reason="body not present in policy file")

    text_blob = json.dumps(body_d, ensure_ascii=False).lower()

    def truthy_any(keys: Sequence[str]) -> Optional[bool]:
        for key in keys:
            if key in body_d:
                return bool_from_any(body_d[key], None)
        return None

    explicit_global = truthy_any([
        "allowed_for_global", "global_allowed", "use_for_global", "allow_global",
        "planning_allowed", "allowed_for_planning", "global_planning",
    ])
    explicit_fine = truthy_any([
        "allowed_for_fine_targeting", "fine_targeting_allowed", "fine_allowed",
        "targeting_allowed", "allowed_for_targeting",
    ])

    allowed_for_global = True if explicit_global is None else bool(explicit_global)
    requires_revalidation = any(w in text_blob for w in ["revalidate", "revalidation", "local_kernel", "local revalidation"])
    allowed_for_fine = bool(explicit_fine) if explicit_fine is not None else not requires_revalidation

    grade = "unknown"
    for key in ("target_grade", "grade", "class", "validation_class", "policy_class", "status"):
        if key in body_d and body_d[key] is not None:
            grade = str(body_d[key])
            break
    if grade == "unknown" and requires_revalidation:
        grade = "revalidate"

    reason = ""
    for key in ("reason", "notes", "note", "comment", "description"):
        if key in body_d and body_d[key] is not None:
            reason = str(body_d[key])
            break

    return BodyPolicy(
        allowed_for_global=allowed_for_global,
        allowed_for_fine_targeting=allowed_for_fine,
        requires_local_revalidation=requires_revalidation,
        target_grade=grade,
        reason=reason,
    )


class SpiceReader:
    def __init__(self, bsp: Optional[Path], tpc: Optional[Path], lsk: Optional[Path]) -> None:
        self.bsp = bsp
        self.tpc = tpc
        self.lsk = lsk
        self.sp = None

    def __enter__(self) -> "SpiceReader":
        import spiceypy as spice
        self.sp = spice
        if self.lsk:
            self.sp.furnsh(str(self.lsk))
        if self.tpc:
            self.sp.furnsh(str(self.tpc))
        if self.bsp:
            self.sp.furnsh(str(self.bsp))
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.sp is not None:
            self.sp.kclear()

    def body_code(self, body: str) -> Optional[int]:
        if self.sp is None:
            return None
        try:
            return int(self.sp.bodn2c(body))
        except Exception:
            try:
                return int(body)
            except Exception:
                return None

    def gm(self, body: str) -> Tuple[Optional[float], str]:
        if self.sp is None:
            return None, "spice_not_loaded"
        try:
            _dim, values = self.sp.bodvrd(body, "GM", 1)
            if len(values):
                return convert_mu_to_km3_s2(float(values[0]), "GM"), "spice.GM"
        except Exception:
            pass
        return None, "not_found"

    def radius(self, body: str) -> Tuple[Optional[float], str]:
        if self.sp is None:
            return None, "spice_not_loaded"
        try:
            _dim, values = self.sp.bodvrd(body, "RADII", 3)
            if len(values):
                return float(max(values)), "spice.RADII"
        except Exception:
            pass
        return None, "not_found"


def body_overrides(overrides: Mapping[str, Any], body: str) -> Dict[str, Any]:
    if not overrides:
        return {}
    body_l = body.lower()
    containers = [overrides]
    for key in ("bodies", "body_catalog", "overrides"):
        c = overrides.get(key) if isinstance(overrides, Mapping) else None
        if isinstance(c, Mapping):
            containers.append(c)
    for c in containers:
        for k, v in c.items():
            if str(k).lower() == body_l and isinstance(v, Mapping):
                return dict(v)
    return {}


def override_float(ov: Mapping[str, Any], keys: Sequence[str]) -> Optional[float]:
    for key in keys:
        if key in ov:
            value = scalar_from_any(ov[key])
            if value is not None:
                return convert_radius_to_km(value, key)
    return None


def build_entry(
    body: str,
    spice: SpiceReader,
    metadata: Mapping[str, Any],
    policy_root: Mapping[str, Any],
    overrides_root: Mapping[str, Any],
    args: argparse.Namespace,
) -> BodyCatalogEntry:
    notes: List[str] = []
    ov = body_overrides(overrides_root, body)

    mu, mu_src = spice.gm(body)
    if mu is None:
        mu, mu_src = find_mu_in_metadata(metadata, body)
    if mu is not None:
        notes.append(f"mu={mu_src}")
    else:
        notes.append("mu=missing")

    radius, radius_src = spice.radius(body)
    if radius is None:
        radius, radius_src = find_radius_in_metadata(metadata, body)
    if radius is not None:
        notes.append(f"radius={radius_src}")
    else:
        notes.append("radius=missing")

    atmosphere_top, atm_src = find_atmosphere_top_km(metadata, body)
    notes.append(f"atmosphere_top={atm_src}")

    ov_atm = override_float(ov, ["atmosphere_top_km", "atmosphere_height_km", "atmosphere_altitude_km", "atmosphere_top_m", "atmosphere_height_m", "atmosphere_altitude_m"])
    if ov_atm is not None:
        atmosphere_top = max(0.0, ov_atm)
        notes.append("atmosphere_top=override")

    safety_margin = float(args.default_atmosphere_margin_km if atmosphere_top > 0.0 else args.default_vacuum_margin_km)
    ov_safety = override_float(ov, ["safety_margin_km", "safety_altitude_km", "margin_km", "safety_margin_m"])
    if ov_safety is not None:
        safety_margin = max(0.0, ov_safety)
        notes.append("safety_margin=override")

    min_flyby_altitude = max(float(args.default_min_altitude_km), atmosphere_top + safety_margin)
    ov_min_alt = override_float(ov, ["min_flyby_altitude_km", "minimum_flyby_altitude_km", "flyby_altitude_min_km", "min_altitude_km", "min_flyby_altitude_m"])
    if ov_min_alt is not None:
        min_flyby_altitude = max(0.0, ov_min_alt)
        notes.append("min_flyby_altitude=override")

    rp_min = None if radius is None else radius + min_flyby_altitude

    policy = infer_body_policy(policy_root, body)
    allow_flyby = bool(policy.allowed_for_global and mu is not None and radius is not None)
    ov_allow = bool_from_any(ov.get("allow_flyby"), None) if ov else None
    if ov_allow is not None:
        allow_flyby = ov_allow
        notes.append("allow_flyby=override")

    if ov:
        notes.append("has_overrides")

    return BodyCatalogEntry(
        name=body,
        naif_id=spice.body_code(body),
        mu_km3_s2=mu,
        radius_km=radius,
        atmosphere_top_km=atmosphere_top,
        min_flyby_altitude_km=min_flyby_altitude,
        safety_margin_km=safety_margin,
        rp_min_km=rp_min,
        allow_flyby=allow_flyby,
        policy=policy,
        source_notes=tuple(notes),
        raw_overrides=ov,
    )


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a physical body catalog for MGA flyby feasibility checks.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--bsp", type=Path, default=None, help="Optional BSP to load for name/code coverage")
    parser.add_argument("--tpc", type=Path, default=None, help="TPC/text kernel with IDs, GM, radii, etc.")
    parser.add_argument("--lsk", type=Path, default=None, help="Optional leap-second kernel")
    parser.add_argument("--metadata", type=Path, default=None, help="SPICE V0.1 metadata JSON")
    parser.add_argument("--policy", type=Path, default=None, help="target_policy_v0_1.json")
    parser.add_argument("--overrides", type=Path, default=None, help="Optional body override JSON")
    parser.add_argument("--input-jsonl", nargs="*", type=Path, default=[], help="Leg-seed JSONL files used to infer body names")
    parser.add_argument("--bodies", nargs="*", default=[], help="Explicit body names to include")
    parser.add_argument("--default-min-altitude-km", type=float, default=50.0, help="Default minimum flyby altitude above radius for vacuum bodies")
    parser.add_argument("--default-vacuum-margin-km", type=float, default=0.0, help="Extra margin used for vacuum bodies unless overridden")
    parser.add_argument("--default-atmosphere-margin-km", type=float, default=20.0, help="Altitude margin above atmosphere top")
    parser.add_argument("--output-json", required=True, type=Path, help="Output catalog JSON")
    args = parser.parse_args(argv)
    if args.default_min_altitude_km < 0.0:
        parser.error("--default-min-altitude-km must be non-negative")
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    metadata = load_json(args.metadata)
    policy = load_json(args.policy)
    overrides = load_json(args.overrides)

    bodies: Set[str] = set(str(b) for b in args.bodies if str(b).strip())
    if args.input_jsonl:
        bodies |= read_bodies_from_jsonl(args.input_jsonl)
    if not bodies:
        raise SystemExit("[FATAL] No bodies supplied. Use --bodies or --input-jsonl.")

    entries: Dict[str, BodyCatalogEntry] = {}
    with SpiceReader(args.bsp, args.tpc, args.lsk) as spice:
        for body in sorted(bodies):
            entries[body] = build_entry(body, spice, metadata, policy, overrides, args)

    counts = Counter()
    for e in entries.values():
        if e.mu_km3_s2 is None:
            counts["missing_mu"] += 1
        if e.radius_km is None:
            counts["missing_radius"] += 1
        if e.rp_min_km is None:
            counts["missing_rp_min"] += 1
        if e.allow_flyby:
            counts["allow_flyby"] += 1
        if e.policy.requires_local_revalidation:
            counts["requires_revalidation"] += 1

    payload: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "units": {
            "distance": "km",
            "velocity": "km/s",
            "mu": "km^3/s^2",
        },
        "defaults": {
            "default_min_altitude_km": args.default_min_altitude_km,
            "default_vacuum_margin_km": args.default_vacuum_margin_km,
            "default_atmosphere_margin_km": args.default_atmosphere_margin_km,
        },
        "inputs": {
            "bsp": str(args.bsp) if args.bsp else None,
            "tpc": str(args.tpc) if args.tpc else None,
            "metadata": str(args.metadata) if args.metadata else None,
            "policy": str(args.policy) if args.policy else None,
            "overrides": str(args.overrides) if args.overrides else None,
            "input_jsonl": [str(p) for p in args.input_jsonl],
        },
        "bodies": {name: asdict(entry) for name, entry in entries.items()},
        "counts": dict(counts),
        "caveats": [
            "This catalog defines planning-grade safety radii, not final targeting constraints.",
            "Atmosphere and safety altitudes are conservative defaults unless supplied by metadata or overrides.",
            "Bodies with missing mu or radius are not allowed as physical flyby bodies by default.",
        ],
    }
    write_json(args.output_json, payload)

    print("=" * 80)
    print("MGA BODY CATALOG V0.1")
    print("=" * 80)
    print(f"Bodies:          {len(entries)}")
    print(f"Flyby allowed:   {counts.get('allow_flyby', 0)}")
    print(f"Missing mu:      {counts.get('missing_mu', 0)}")
    print(f"Missing radius:  {counts.get('missing_radius', 0)}")
    print(f"Revalidate:      {counts.get('requires_revalidation', 0)}")
    print("\nBody summary:")
    print(f"{'Body':<12} {'mu km^3/s^2':>16} {'R km':>12} {'rp_min km':>12} {'flyby':>7} {'grade':>12}")
    print("-" * 80)
    for name, entry in entries.items():
        mu_s = "" if entry.mu_km3_s2 is None else f"{entry.mu_km3_s2:.6g}"
        r_s = "" if entry.radius_km is None else f"{entry.radius_km:.6g}"
        rp_s = "" if entry.rp_min_km is None else f"{entry.rp_min_km:.6g}"
        print(f"{name:<12} {mu_s:>16} {r_s:>12} {rp_s:>12} {str(entry.allow_flyby):>7} {entry.policy.target_grade:>12}")
    print("=" * 80)
    print(f"[OK] wrote JSON: {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
