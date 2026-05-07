#!/usr/bin/env python3
"""
mga_extract_body_catalog_krpc_v0_1.py

Extract or patch an MGA BodyCatalog directly from a live KSP/kRPC session.

Purpose
-------
This utility is meant to close the "missing radii / atmosphere" gap in the
offline-first MGA pipeline:

    KSP/kRPC -> physical body catalog JSON -> MGA beam search V0.2+

It does NOT use kRPC as a solver. It only reads immutable-ish body constants
and a small amount of topology/rotation metadata from the running game/modded
system.

Primary outputs
---------------
- A BodyCatalog JSON compatible with mga_beam_search_v0_2.py.
- Optional CSV report for quick audit.

Units
-----
kRPC returns SI units:
- mu in m^3/s^2
- radii, atmosphere depth and SOI in m

This script writes:
- mu_km3_s2 in km^3/s^2
- radius_km, atmosphere_top_km, sphere_of_influence_km, rp_min_km in km

Typical usage
-------------
Patch an existing catalog, keeping only bodies already used by leg seeds:

    python mga_extract_body_catalog_krpc_v0_1.py \\
      --input-catalog data/mga_v0_1/body_catalog_v0_1.json \\
      --policy data/spice_v0_1_33y/target_policy_v0_1.json \\
      --default-min-altitude-km 50 \\
      --default-atmosphere-margin-km 20 \\
      --output-json data/mga_v0_1/body_catalog_v0_1.krpc.json \\
      --report-csv data/mga_v0_1/body_catalog_v0_1.krpc.csv

Extract selected bodies without an input catalog:

    python mga_extract_body_catalog_krpc_v0_1.py \\
      --bodies Kerbin Duna Jool Sarnus Urlum Neidon Plock Soden \\
      --output-json data/mga_v0_1/body_catalog_v0_1.krpc.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import socket
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


KM_PER_M = 1.0e-3
KM3_PER_M3 = 1.0e-9


def log(msg: str) -> None:
    print(msg, flush=True)


def warn(msg: str) -> None:
    print(f"[WARN] {msg}", file=sys.stderr, flush=True)


def load_json(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"JSON not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, data: Dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, sort_keys=True)


def try_get(obj: Any, attr: str, default: Any = None) -> Any:
    try:
        return getattr(obj, attr)
    except Exception:
        return default


def finite_or_none(x: Any) -> Optional[float]:
    try:
        y = float(x)
    except Exception:
        return None
    if not math.isfinite(y):
        return None
    return y


def normalize_name(name: str) -> str:
    return str(name).strip()


def collect_requested_body_names(
    args: argparse.Namespace,
    input_catalog: Dict[str, Any],
) -> Optional[List[str]]:
    names: List[str] = []

    if args.bodies:
        names.extend(args.bodies)

    if args.input_catalog:
        bodies = input_catalog.get("bodies", {})
        if isinstance(bodies, dict):
            names.extend(bodies.keys())

    if not names and not args.include_all_game_bodies:
        return None

    if args.include_all_game_bodies and not names:
        return None

    dedup: Dict[str, None] = {}
    for name in names:
        n = normalize_name(name)
        if n:
            dedup[n] = None
    return list(dedup.keys())


def policy_lookup(policy: Dict[str, Any], body_name: str) -> Dict[str, Any]:
    """
    Tolerant lookup for target_policy_v0_1.json variants.

    Supports common layouts:
      {"bodies": {"Duna": {...}}}
      {"targets": {"Duna": {...}}}
      {"Duna": {...}}
      {"body_policy": {"Duna": {...}}}

    Unknown policy fields are preserved in "policy".
    """
    for key in ("bodies", "targets", "body_policy", "target_policy"):
        block = policy.get(key)
        if isinstance(block, dict) and body_name in block and isinstance(block[body_name], dict):
            return dict(block[body_name])
    if body_name in policy and isinstance(policy[body_name], dict):
        return dict(policy[body_name])
    return {}


def policy_allows_flyby(policy_entry: Dict[str, Any]) -> Optional[bool]:
    """
    Return explicit policy decision when present; otherwise None.

    We intentionally accept many possible field names because the target policy
    has evolved during the project.
    """
    explicit_false_values = {"blocked", "deny", "disabled", "false", "no", "none"}
    explicit_true_values = {"allowed", "allow", "enabled", "true", "yes", "coarse", "planning"}

    for key in (
        "allow_flyby",
        "flyby_allowed",
        "allow_as_flyby",
        "use_as_flyby",
        "enabled_for_flyby",
    ):
        if key in policy_entry:
            return bool(policy_entry[key])

    for key in ("status", "policy", "targeting", "flyby_policy", "planning_status"):
        val = policy_entry.get(key)
        if isinstance(val, str):
            s = val.strip().lower()
            if s in explicit_false_values:
                return False
            if s in explicit_true_values:
                return True

    # Some earlier policies use warning classes rather than hard block.
    # "revalidate" should NOT block coarse planning by default.
    return None


def policy_revalidate(policy_entry: Dict[str, Any]) -> bool:
    for key in ("requires_revalidation", "revalidate", "needs_local_kernel", "local_revalidation"):
        if key in policy_entry:
            return bool(policy_entry[key])
    for key in ("status", "policy", "targeting", "target_class", "grade"):
        val = policy_entry.get(key)
        if isinstance(val, str) and "revalid" in val.lower():
            return True
    return False


@dataclass
class KRPCBodyPhysical:
    name: str
    source: str = "krpc"
    mu_km3_s2: Optional[float] = None
    mass_kg: Optional[float] = None
    radius_km: Optional[float] = None
    atmosphere_top_km: float = 0.0
    has_atmosphere: bool = False
    has_solid_surface: Optional[bool] = None
    is_star: Optional[bool] = None
    sphere_of_influence_km: Optional[float] = None
    rotational_period_s: Optional[float] = None
    rotational_speed_rad_s: Optional[float] = None
    parent: Optional[str] = None
    satellites: List[str] = None  # type: ignore[assignment]
    min_flyby_altitude_km: Optional[float] = None
    rp_min_km: Optional[float] = None
    allow_flyby: bool = False
    revalidate: bool = False
    policy: Dict[str, Any] = None  # type: ignore[assignment]
    extraction_warnings: List[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.satellites is None:
            self.satellites = []
        if self.policy is None:
            self.policy = {}
        if self.extraction_warnings is None:
            self.extraction_warnings = []


def extract_body_from_krpc(body: Any, args: argparse.Namespace, policy: Dict[str, Any]) -> KRPCBodyPhysical:
    name = str(body.name)
    entry = policy_lookup(policy, name)

    mu_m3_s2 = finite_or_none(try_get(body, "gravitational_parameter"))
    mass_kg = finite_or_none(try_get(body, "mass"))
    radius_m = finite_or_none(try_get(body, "equatorial_radius"))
    soi_m = finite_or_none(try_get(body, "sphere_of_influence"))

    has_atm = bool(try_get(body, "has_atmosphere", False))
    atm_m = finite_or_none(try_get(body, "atmosphere_depth")) if has_atm else 0.0
    if atm_m is None:
        atm_m = 0.0

    is_star = try_get(body, "is_star", None)
    has_solid_surface = try_get(body, "has_solid_surface", None)

    parent_name = None
    try:
        orbit = body.orbit
        parent = orbit.body
        if parent is not None and getattr(parent, "name", None) != name:
            parent_name = str(parent.name)
    except Exception:
        parent_name = None

    satellites: List[str] = []
    try:
        satellites = sorted(str(s.name) for s in body.satellites)
    except Exception:
        satellites = []

    rotational_period_s = finite_or_none(try_get(body, "rotational_period"))
    rotational_speed_rad_s = finite_or_none(try_get(body, "rotational_speed"))

    radius_km = radius_m * KM_PER_M if radius_m is not None else None
    mu_km3_s2 = mu_m3_s2 * KM3_PER_M3 if mu_m3_s2 is not None else None
    atmosphere_top_km = float(atm_m) * KM_PER_M
    soi_km = soi_m * KM_PER_M if soi_m is not None else None

    # Determine operational minimum altitude.
    requested_min_alt = args.default_min_altitude_km
    if args.per_body_min_altitude_json:
        alt_map = load_json(args.per_body_min_altitude_json)
        # Supports {"Duna": 80} or {"bodies":{"Duna":80}}
        if isinstance(alt_map.get("bodies"), dict) and name in alt_map["bodies"]:
            requested_min_alt = float(alt_map["bodies"][name])
        elif name in alt_map:
            requested_min_alt = float(alt_map[name])

    if has_atm:
        if args.atmosphere_margin_mode == "none":
            protected_altitude = atmosphere_top_km
        elif args.atmosphere_margin_mode == "add":
            protected_altitude = atmosphere_top_km + args.default_atmosphere_margin_km
        else:
            raise ValueError(f"unknown atmosphere_margin_mode: {args.atmosphere_margin_mode}")
        min_flyby_alt_km = max(float(requested_min_alt), protected_altitude)
    else:
        min_flyby_alt_km = float(requested_min_alt)

    rp_min_km = None
    if radius_km is not None:
        rp_min_km = radius_km + min_flyby_alt_km

    warnings: List[str] = []
    if mu_km3_s2 is None:
        warnings.append("missing_mu")
    if radius_km is None:
        warnings.append("missing_radius")
    if rp_min_km is None:
        warnings.append("missing_rp_min")

    explicit_policy = policy_allows_flyby(entry)
    revalidate = policy_revalidate(entry)

    allow_flyby = bool(mu_km3_s2 is not None and rp_min_km is not None)
    if args.disallow_stars and bool(is_star):
        allow_flyby = False
        warnings.append("star_flyby_disabled")
    if args.disallow_no_solid_surface and has_solid_surface is False:
        allow_flyby = False
        warnings.append("no_solid_surface_disabled")
    if explicit_policy is not None:
        allow_flyby = allow_flyby and explicit_policy
        if not explicit_policy:
            warnings.append("policy_disallows_flyby")

    return KRPCBodyPhysical(
        name=name,
        mu_km3_s2=mu_km3_s2,
        mass_kg=mass_kg,
        radius_km=radius_km,
        atmosphere_top_km=atmosphere_top_km,
        has_atmosphere=has_atm,
        has_solid_surface=bool(has_solid_surface) if has_solid_surface is not None else None,
        is_star=bool(is_star) if is_star is not None else None,
        sphere_of_influence_km=soi_km,
        rotational_period_s=rotational_period_s,
        rotational_speed_rad_s=rotational_speed_rad_s,
        parent=parent_name,
        satellites=satellites,
        min_flyby_altitude_km=min_flyby_alt_km,
        rp_min_km=rp_min_km,
        allow_flyby=allow_flyby,
        revalidate=revalidate,
        policy=entry,
        extraction_warnings=warnings,
    )


def merge_with_input_catalog(
    input_catalog: Dict[str, Any],
    extracted: Dict[str, KRPCBodyPhysical],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    if input_catalog and not args.include_all_game_bodies:
        out = dict(input_catalog)
        out_bodies = dict(out.get("bodies", {}))
        body_names = list(out_bodies.keys())
    else:
        out = {}
        out_bodies = {}
        body_names = list(extracted.keys())

    for name in body_names:
        old = dict(out_bodies.get(name, {}))
        x = extracted.get(name)
        if x is None:
            old.setdefault("extraction_warnings", [])
            old["extraction_warnings"] = list(old["extraction_warnings"]) + ["not_found_in_krpc"]
            out_bodies[name] = old
            continue

        new = asdict(x)

        # Preserve existing non-physical fields that downstream scripts may use,
        # but replace direct physical fields with live kRPC values.
        merged = dict(old)
        merged.update(new)

        # Keep stable spelling aliases for older scripts.
        merged["radius_km"] = new["radius_km"]
        merged["equatorial_radius_km"] = new["radius_km"]
        merged["atmosphere_top_km"] = new["atmosphere_top_km"]
        merged["rp_min_km"] = new["rp_min_km"]
        merged["mu_km3_s2"] = new["mu_km3_s2"]
        merged["allow_flyby"] = new["allow_flyby"]

        out_bodies[name] = merged

    out["schema"] = "mga.body_catalog.v0_1.krpc"
    out["generated_by"] = "mga_extract_body_catalog_krpc_v0_1.py"
    out["generated_utc"] = now
    out["units"] = {
        "mu_km3_s2": "km^3/s^2",
        "radius_km": "km",
        "equatorial_radius_km": "km",
        "atmosphere_top_km": "km",
        "sphere_of_influence_km": "km",
        "rp_min_km": "km",
        "rotational_period_s": "s",
        "rotational_speed_rad_s": "rad/s",
    }
    out["flyby_policy_notes"] = {
        "rp_min_rule": (
            "radius_km + max(default_min_altitude_km, atmosphere_top_km + margin) "
            "for atmospheric bodies; radius_km + default_min_altitude_km otherwise"
            if args.atmosphere_margin_mode == "add"
            else
            "radius_km + max(default_min_altitude_km, atmosphere_top_km) for atmospheric bodies; "
            "radius_km + default_min_altitude_km otherwise"
        ),
        "default_min_altitude_km": args.default_min_altitude_km,
        "default_atmosphere_margin_km": args.default_atmosphere_margin_km,
        "atmosphere_margin_mode": args.atmosphere_margin_mode,
        "disallow_stars": args.disallow_stars,
        "disallow_no_solid_surface": args.disallow_no_solid_surface,
    }
    out["bodies"] = out_bodies
    return out


def write_csv_report(path: str, catalog: Dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    fields = [
        "name",
        "mu_km3_s2",
        "radius_km",
        "atmosphere_top_km",
        "min_flyby_altitude_km",
        "rp_min_km",
        "allow_flyby",
        "revalidate",
        "has_atmosphere",
        "has_solid_surface",
        "is_star",
        "sphere_of_influence_km",
        "parent",
        "satellites",
        "extraction_warnings",
    ]
    with p.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for name, b in sorted(catalog.get("bodies", {}).items()):
            row = {k: b.get(k) for k in fields}
            row["name"] = name
            row["satellites"] = ";".join(b.get("satellites", []) or [])
            row["extraction_warnings"] = ";".join(b.get("extraction_warnings", []) or [])
            w.writerow(row)


def print_summary(catalog: Dict[str, Any], requested: Optional[List[str]]) -> None:
    bodies = catalog.get("bodies", {})
    flyby_allowed = sum(1 for b in bodies.values() if b.get("allow_flyby"))
    missing_mu = sum(1 for b in bodies.values() if b.get("mu_km3_s2") is None)
    missing_radius = sum(1 for b in bodies.values() if b.get("radius_km") is None)
    revalidate = sum(1 for b in bodies.values() if b.get("revalidate"))

    print("=" * 80)
    print("MGA BODY CATALOG FROM KRPC V0.1")
    print("=" * 80)
    print(f"Bodies:          {len(bodies)}")
    print(f"Requested only:  {'yes' if requested else 'no'}")
    print(f"Flyby allowed:   {flyby_allowed}")
    print(f"Missing mu:      {missing_mu}")
    print(f"Missing radius:  {missing_radius}")
    print(f"Revalidate:      {revalidate}")
    print()
    print("Body summary:")
    print(f"{'Body':<14} {'mu km^3/s^2':>16} {'R km':>10} {'atm km':>9} {'rp_min km':>12} {'SOI km':>12} {'flyby':>8} {'parent':>12}")
    print("-" * 100)
    for name, b in sorted(bodies.items()):
        def fmt(x: Any, width: int, prec: int = 6) -> str:
            if x is None:
                return " " * width
            try:
                return f"{float(x):{width}.{prec}g}"
            except Exception:
                return str(x)[:width].rjust(width)
        print(
            f"{name:<14} "
            f"{fmt(b.get('mu_km3_s2'), 16)} "
            f"{fmt(b.get('radius_km'), 10)} "
            f"{fmt(b.get('atmosphere_top_km'), 9)} "
            f"{fmt(b.get('rp_min_km'), 12)} "
            f"{fmt(b.get('sphere_of_influence_km'), 12)} "
            f"{str(bool(b.get('allow_flyby'))):>8} "
            f"{str(b.get('parent') or ''):>12}"
        )
    print("=" * 80)


def connect_krpc(args: argparse.Namespace) -> Any:
    try:
        import krpc  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "Could not import krpc. Install the Python client in this environment "
            "and make sure the kRPC mod/server is running in KSP."
        ) from e

    kwargs: Dict[str, Any] = {"name": args.connection_name}
    if args.address:
        kwargs["address"] = args.address
    if args.rpc_port is not None:
        kwargs["rpc_port"] = args.rpc_port
    if args.stream_port is not None:
        kwargs["stream_port"] = args.stream_port

    try:
        return krpc.connect(**kwargs)
    except (OSError, socket.error, ConnectionError) as e:
        raise RuntimeError(
            "Could not connect to kRPC. Check that KSP is running, the save is loaded, "
            "and the kRPC server is enabled."
        ) from e


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Extract/patch MGA body catalog constants directly from a live KSP/kRPC session."
    )
    ap.add_argument("--input-catalog", help="Existing BodyCatalog JSON to patch.")
    ap.add_argument("--policy", help="Optional target_policy_v0_1.json.")
    ap.add_argument("--bodies", nargs="*", help="Body names to extract when no input catalog is supplied.")
    ap.add_argument("--include-all-game-bodies", action="store_true", help="Extract all kRPC bodies, not just requested/input-catalog bodies.")

    ap.add_argument("--default-min-altitude-km", type=float, default=50.0)
    ap.add_argument("--default-atmosphere-margin-km", type=float, default=20.0)
    ap.add_argument("--atmosphere-margin-mode", choices=["add", "none"], default="add")
    ap.add_argument("--per-body-min-altitude-json", help="Optional JSON mapping body name -> min flyby altitude km.")
    ap.add_argument("--disallow-stars", action="store_true", default=True)
    ap.add_argument("--allow-stars", action="store_false", dest="disallow_stars")
    ap.add_argument("--disallow-no-solid-surface", action="store_true", help="Usually leave false: gas giant flybys above atmosphere are valid.")

    ap.add_argument("--connection-name", default="MGA Body Catalog Extractor")
    ap.add_argument("--address", help="kRPC server address. Omit for client default.")
    ap.add_argument("--rpc-port", type=int, help="kRPC RPC port. Omit for client default.")
    ap.add_argument("--stream-port", type=int, help="kRPC stream port. Omit for client default.")

    ap.add_argument("--output-json", required=True)
    ap.add_argument("--report-csv")
    return ap.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    input_catalog = load_json(args.input_catalog)
    policy = load_json(args.policy)

    requested = collect_requested_body_names(args, input_catalog)

    conn = connect_krpc(args)
    sc = conn.space_center

    try:
        ut = float(sc.ut)
        log(f"[INFO] Connected to kRPC. Game UT = {ut:.3f} s")
    except Exception:
        log("[INFO] Connected to kRPC.")

    game_bodies = sc.bodies
    # kRPC returns dict name -> body in modern clients.
    if isinstance(game_bodies, dict):
        game_body_map = {str(k): v for k, v in game_bodies.items()}
    else:
        game_body_map = {str(b.name): b for b in game_bodies}

    if requested is None or args.include_all_game_bodies:
        selected_names = sorted(game_body_map.keys())
    else:
        selected_names = requested

    extracted: Dict[str, KRPCBodyPhysical] = {}
    missing: List[str] = []
    for name in selected_names:
        body = game_body_map.get(name)
        if body is None:
            missing.append(name)
            continue
        extracted[name] = extract_body_from_krpc(body, args, policy)

    if missing:
        warn(f"Bodies requested but not found in kRPC: {', '.join(missing)}")

    catalog = merge_with_input_catalog(input_catalog, extracted, args)

    # Add missing requested names if no input catalog exists.
    if not input_catalog:
        for name in missing:
            catalog.setdefault("bodies", {})[name] = {
                "name": name,
                "source": "krpc",
                "allow_flyby": False,
                "extraction_warnings": ["not_found_in_krpc"],
            }

    save_json(args.output_json, catalog)
    if args.report_csv:
        write_csv_report(args.report_csv, catalog)

    print_summary(catalog, requested)
    log(f"[OK] wrote JSON: {args.output_json}")
    if args.report_csv:
        log(f"[OK] wrote CSV:  {args.report_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
