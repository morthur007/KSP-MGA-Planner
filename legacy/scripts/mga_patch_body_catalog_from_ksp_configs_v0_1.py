#!/usr/bin/env python3
"""
mga_patch_body_catalog_from_ksp_configs_v0_1.py

Patch/fill a planning-grade MGA BodyCatalog with physical radii and atmosphere
limits parsed from KSP/Kopernicus/ModuleManager config files.

Why this exists
---------------
SPICE text kernels in the current synthetic pipeline may define NAIF names and
GM values but not BODY*_RADII variables. The V0.2 beam search needs radius and
rp_min to evaluate physical unpowered-flyby turn-angle envelopes. This script
keeps the existing catalog shape intact and fills missing values from the KSP
config cache or planet-pack cfg/proto files.

Expected inputs
---------------
  * --input-catalog: body_catalog_v0_1.json from mga_build_body_catalog_v0_1.py
  * --config-cache: ModuleManager.ConfigCache, preferably the final resolved one
  * --ksp-config: optional extra Kopernicus cfg/txt files, used as fallback
  * --principia-gravity-model: optional Principia gravity model proto text,
    used as an additional GM fallback

All output distances are km and mu is km^3/s^2.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

NUMBER_RE = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")
ASSIGN_RE = re.compile(r"^\s*([A-Za-z0-9_@%+!.,\-\[\]/]+)\s*=\s*(.*?)\s*$")
BODY_START_RE = re.compile(r"^\s*[@%+!\-]*Body(?:\s*\[([^\]]+)\])?\s*$", re.IGNORECASE)
NODE_START_RE = re.compile(r"^\s*([A-Za-z0-9_@%+!.,\-\[\]/]+)(?:\s*\[[^\]]+\])?\s*$")


@dataclass
class KspBodyPhysicals:
    name: str
    radius_km: Optional[float] = None
    atmosphere_top_km: Optional[float] = None
    mu_km3_s2: Optional[float] = None
    gee_asl: Optional[float] = None
    source: str = ""
    source_notes: Tuple[str, ...] = ()


def strip_line_comment(line: str) -> str:
    """Remove // comments using a simple quote-aware scan."""
    in_quote = False
    i = 0
    while i < len(line) - 1:
        c = line[i]
        if c == '"':
            in_quote = not in_quote
        if not in_quote and line[i:i+2] == "//":
            return line[:i]
        i += 1
    return line


def parse_number(value: Any) -> Optional[float]:
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    if value is None:
        return None
    text = str(value).strip().strip('"').strip("'")
    m = NUMBER_RE.search(text)
    if not m:
        return None
    try:
        x = float(m.group(0))
    except ValueError:
        return None
    return x if math.isfinite(x) else None


def boolish(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    v = str(value).strip().strip('"').strip("'").lower()
    if v in {"true", "yes", "1", "enabled"}:
        return True
    if v in {"false", "no", "0", "disabled"}:
        return False
    return None


def metres_to_km_if_needed(value: float, kind: str) -> float:
    """KSP configs normally store radius/altitude in metres."""
    k = kind.lower()
    if k.endswith("_km") or "km" in k:
        return value
    if k.endswith("_m") or "meter" in k or "metre" in k:
        return value / 1000.0
    # Bare KSP radii and atmosphere altitudes are almost always metres.
    # Keep small values untouched to avoid corrupting already-km overrides.
    if abs(value) > 1.0e5:
        return value / 1000.0
    return value


def mu_to_km3_s2_if_needed(value: float, kind: str = "") -> float:
    k = kind.lower()
    if "km" in k:
        return value
    if "m3" in k or "m^3" in k:
        return value / 1.0e9
    # KSP/Principia GM values in m^3/s^2 are usually >= 1e9 even for small bodies.
    if abs(value) > 1.0e7:
        return value / 1.0e9
    return value


def find_matching_brace(lines: Sequence[str], start_idx: int) -> Optional[int]:
    depth = 0
    seen_open = False
    for i in range(start_idx, len(lines)):
        line = strip_line_comment(lines[i])
        for ch in line:
            if ch == "{":
                depth += 1
                seen_open = True
            elif ch == "}":
                depth -= 1
                if seen_open and depth <= 0:
                    return i
    return None


def extract_named_blocks(text: str, wanted_node: str) -> Iterator[Tuple[Optional[str], str]]:
    """Yield blocks such as Body { ... } or Atmosphere { ... } with raw text."""
    lines = text.splitlines()
    i = 0
    wanted_l = wanted_node.lower()
    while i < len(lines):
        clean = strip_line_comment(lines[i]).strip()
        named_hint: Optional[str] = None
        is_match = False

        if wanted_l == "body":
            m = BODY_START_RE.match(clean)
            if m:
                is_match = True
                named_hint = m.group(1)
            # Also tolerate one-line beginnings like "Body {".
            elif re.match(r"^\s*[@%+!\-]*Body(?:\s*\[[^\]]+\])?\s*\{", clean, re.IGNORECASE):
                is_match = True
                mm = re.search(r"Body\s*\[([^\]]+)\]", clean, re.IGNORECASE)
                named_hint = mm.group(1) if mm else None
        else:
            m = NODE_START_RE.match(clean)
            if m and m.group(1).lstrip("@%+!-").lower() == wanted_l:
                is_match = True
            elif re.match(rf"^\s*[@%+!\-]*{re.escape(wanted_node)}\s*\{{", clean, re.IGNORECASE):
                is_match = True

        if not is_match:
            i += 1
            continue

        # Find first opening brace at or after this line.
        open_idx = i
        while open_idx < len(lines) and "{" not in strip_line_comment(lines[open_idx]):
            open_idx += 1
        if open_idx >= len(lines):
            i += 1
            continue
        close_idx = find_matching_brace(lines, open_idx)
        if close_idx is None:
            i += 1
            continue
        yield named_hint, "\n".join(lines[i:close_idx + 1])
        i = close_idx + 1


def top_level_assignments(block: str) -> Dict[str, str]:
    """Collect simple key=value assignments anywhere in a block.

    For our use case this intentionally sees nested values too. We choose keys
    narrowly (radius, gravParameter, geeASL, name), so this remains robust for
    final ModuleManager caches.
    """
    out: Dict[str, str] = {}
    for raw in block.splitlines():
        line = strip_line_comment(raw).strip()
        m = ASSIGN_RE.match(line)
        if not m:
            continue
        key = m.group(1).strip().lstrip("@%!")
        value = m.group(2).strip()
        # Preserve first assignment by default. For final cache there should be
        # one relevant value; for patches, first inside the block is usually the
        # actual key rather than a later patch operator.
        out.setdefault(key, value)
    return out


def first_assignment(block: str, keys: Sequence[str]) -> Tuple[Optional[float], Optional[str]]:
    key_lut = {k.lower(): k for k in keys}
    for raw in block.splitlines():
        line = strip_line_comment(raw).strip()
        m = ASSIGN_RE.match(line)
        if not m:
            continue
        key_raw = m.group(1).strip().lstrip("@%!")
        if key_raw.lower() not in key_lut:
            continue
        x = parse_number(m.group(2))
        if x is not None:
            return x, key_raw
    return None, None


def parse_ksp_body_block(block: str, source: str, body_name_hint: Optional[str] = None) -> Optional[KspBodyPhysicals]:
    assigns = top_level_assignments(block)
    name = body_name_hint or assigns.get("name") or assigns.get("cbNameLater")
    if not name:
        return None
    name = str(name).strip().strip('"').strip("'")

    notes: List[str] = []

    radius_raw, radius_key = first_assignment(block, ["radius", "Radius", "meanRadius", "equatorialRadius"])
    radius_km = None
    if radius_raw is not None:
        radius_km = metres_to_km_if_needed(radius_raw, radius_key or "radius")
        notes.append(f"radius={source}:{radius_key}")

    mu_raw, mu_key = first_assignment(block, ["gravParameter", "gravitationalParameter", "mu", "GM", "geeASL"])
    mu_km3_s2 = None
    gee_asl = None
    if mu_raw is not None and mu_key:
        if mu_key.lower() == "geeasl":
            gee_asl = mu_raw
            notes.append(f"geeASL={source}:{mu_key}")
        else:
            mu_km3_s2 = mu_to_km3_s2_if_needed(mu_raw, mu_key)
            notes.append(f"mu={source}:{mu_key}")

    atmosphere_top_km = None
    for _hint, atm_block in extract_named_blocks(block, "Atmosphere"):
        atm_assigns = top_level_assignments(atm_block)
        enabled = boolish(atm_assigns.get("enabled"))
        if enabled is False:
            continue
        alt_raw, alt_key = first_assignment(atm_block, ["altitude", "maxAltitude", "atmosphereDepth", "AtmosphereDepth"])
        if alt_raw is not None:
            atmosphere_top_km = max(0.0, metres_to_km_if_needed(alt_raw, alt_key or "altitude"))
            notes.append(f"atmosphere={source}:{alt_key}")
            break

    return KspBodyPhysicals(
        name=name,
        radius_km=radius_km,
        atmosphere_top_km=atmosphere_top_km,
        mu_km3_s2=mu_km3_s2,
        gee_asl=gee_asl,
        source=source,
        source_notes=tuple(notes),
    )


def merge_physical(old: Optional[KspBodyPhysicals], new: KspBodyPhysicals) -> KspBodyPhysicals:
    if old is None:
        return new
    notes = list(old.source_notes) + list(new.source_notes)
    return KspBodyPhysicals(
        name=old.name,
        radius_km=new.radius_km if new.radius_km is not None else old.radius_km,
        atmosphere_top_km=new.atmosphere_top_km if new.atmosphere_top_km is not None else old.atmosphere_top_km,
        mu_km3_s2=new.mu_km3_s2 if new.mu_km3_s2 is not None else old.mu_km3_s2,
        gee_asl=new.gee_asl if new.gee_asl is not None else old.gee_asl,
        source=new.source or old.source,
        source_notes=tuple(notes),
    )


def parse_ksp_configs(paths: Sequence[Path]) -> Dict[str, KspBodyPhysicals]:
    result: Dict[str, KspBodyPhysicals] = {}
    for path in paths:
        if not path:
            continue
        if not path.exists():
            raise FileNotFoundError(path)
        text = path.read_text(encoding="utf-8", errors="replace")
        for name_hint, block in extract_named_blocks(text, "Body"):
            rec = parse_ksp_body_block(block, str(path), name_hint)
            if rec is None:
                continue
            key = rec.name.lower()
            result[key] = merge_physical(result.get(key), rec)
    return result


def parse_principia_gravity_model(path: Optional[Path]) -> Dict[str, float]:
    """Best-effort parser for Principia text-proto gravity models.

    The exact text format has varied. This function looks for quoted/unquoted
    body names near gravitational_parameter/standard_gravitational_parameter
    values and returns mu in km^3/s^2.
    """
    if path is None:
        return {}
    if not path.exists():
        raise FileNotFoundError(path)
    text = path.read_text(encoding="utf-8", errors="replace")
    out: Dict[str, float] = {}

    # Split into rough message blocks and inspect each one.
    blocks: List[str] = []
    for token in ("body", "massive_body", "celestial"):
        for _hint, block in extract_named_blocks(text, token):
            blocks.append(block)
    if not blocks:
        blocks = [text]

    for block in blocks:
        name: Optional[str] = None
        for pat in [
            r"\bname\s*[:=]\s*\"([^\"]+)\"",
            r"\bname\s*[:=]\s*([A-Za-z0-9_\-]+)",
            r"\bbody\s*[:=]\s*\"([^\"]+)\"",
            r"\bcelestial_name\s*[:=]\s*\"([^\"]+)\"",
        ]:
            m = re.search(pat, block)
            if m:
                name = m.group(1).strip()
                break
        if not name:
            continue
        mu_value: Optional[float] = None
        for pat in [
            r"\bgravitational_parameter\b[^\n{}]*(?:[:=]|value\s*[:=])\s*(%s)" % NUMBER_RE.pattern,
            r"\bstandard_gravitational_parameter\b[^\n{}]*(?:[:=]|value\s*[:=])\s*(%s)" % NUMBER_RE.pattern,
            r"\bmu\b\s*[:=]\s*(%s)" % NUMBER_RE.pattern,
            r"\bGM\b\s*[:=]\s*(%s)" % NUMBER_RE.pattern,
        ]:
            m = re.search(pat, block)
            if m:
                mu_value = parse_number(m.group(1))
                break
        if mu_value is not None:
            out[name.lower()] = mu_to_km3_s2_if_needed(mu_value, "gravitational_parameter_m3_s2")
    return out


def policy_allows_global(entry: Mapping[str, Any]) -> bool:
    pol = entry.get("policy", {})
    if isinstance(pol, Mapping):
        v = pol.get("allowed_for_global")
        if isinstance(v, bool):
            return v
    return True


def patch_catalog(
    catalog: Dict[str, Any],
    ksp: Mapping[str, KspBodyPhysicals],
    proto_mu: Mapping[str, float],
    args: argparse.Namespace,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    bodies = catalog.setdefault("bodies", {})
    report: List[Dict[str, Any]] = []

    for body, entry in bodies.items():
        if not isinstance(entry, dict):
            continue
        key = str(body).lower()
        rec = ksp.get(key)
        changed: List[str] = []

        # Keep the existing SPICE/metadata mu unless missing or explicitly asked.
        existing_mu = entry.get("mu_km3_s2")
        if args.prefer_ksp_mu or existing_mu in (None, ""):
            mu = None
            if rec and rec.mu_km3_s2 is not None:
                mu = rec.mu_km3_s2
            elif key in proto_mu:
                mu = proto_mu[key]
            if mu is not None:
                entry["mu_km3_s2"] = mu
                changed.append("mu")

        if entry.get("radius_km") in (None, "") and rec and rec.radius_km is not None:
            entry["radius_km"] = rec.radius_km
            changed.append("radius")
        elif args.prefer_ksp_radius and rec and rec.radius_km is not None:
            entry["radius_km"] = rec.radius_km
            changed.append("radius")

        if rec and rec.atmosphere_top_km is not None:
            old_atm = entry.get("atmosphere_top_km")
            if args.prefer_ksp_atmosphere or old_atm in (None, "", 0, 0.0):
                entry["atmosphere_top_km"] = rec.atmosphere_top_km
                changed.append("atmosphere_top")

        # Normalize numeric fields.
        mu_now = parse_number(entry.get("mu_km3_s2"))
        radius_now = parse_number(entry.get("radius_km"))
        atm_now = parse_number(entry.get("atmosphere_top_km")) or 0.0
        old_min_alt = parse_number(entry.get("min_flyby_altitude_km"))
        safety = parse_number(entry.get("safety_margin_km"))
        if safety is None:
            safety = args.default_atmosphere_margin_km if atm_now > 0 else args.default_vacuum_margin_km
            entry["safety_margin_km"] = safety
            changed.append("safety_margin")

        min_alt = max(args.default_min_altitude_km, atm_now + safety)
        if old_min_alt is not None and not args.recompute_min_altitude:
            min_alt = max(old_min_alt, atm_now + safety, args.default_min_altitude_km)
        entry["min_flyby_altitude_km"] = min_alt

        if radius_now is not None:
            entry["rp_min_km"] = radius_now + min_alt
            if "rp_min" not in changed:
                changed.append("rp_min")
        else:
            entry["rp_min_km"] = None

        allow = bool(mu_now is not None and radius_now is not None and policy_allows_global(entry))
        if args.force_disallow_start_body and str(body).lower() == str(args.start_body).lower():
            allow = False
        entry["allow_flyby"] = allow
        changed.append("allow_flyby")

        notes = list(entry.get("source_notes", []) or [])
        if rec and rec.source_notes:
            notes.extend([f"ksp_config:{n}" for n in rec.source_notes])
        if key in proto_mu:
            notes.append(f"principia_gravity_model:{args.principia_gravity_model}")
        # Stable unique-ish notes.
        seen = set()
        clean_notes = []
        for n in notes:
            if n not in seen:
                seen.add(n)
                clean_notes.append(n)
        entry["source_notes"] = clean_notes

        report.append({
            "body": body,
            "mu_km3_s2": entry.get("mu_km3_s2"),
            "radius_km": entry.get("radius_km"),
            "atmosphere_top_km": entry.get("atmosphere_top_km"),
            "min_flyby_altitude_km": entry.get("min_flyby_altitude_km"),
            "rp_min_km": entry.get("rp_min_km"),
            "allow_flyby": entry.get("allow_flyby"),
            "patched": bool(changed),
            "changed": ",".join(sorted(set(changed))),
            "ksp_source": rec.source if rec else "",
        })

    catalog["schema_version"] = str(catalog.get("schema_version", "mga_body_catalog.v0.1")) + "+ksp_patch.v0.1"
    caveats = list(catalog.get("caveats", []) or [])
    caveats.append("Radii/atmospheres may be patched from KSP ModuleManager/Kopernicus configs; verify with save/mod version before fine targeting.")
    catalog["caveats"] = caveats
    return catalog, report


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_report_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "body", "mu_km3_s2", "radius_km", "atmosphere_top_km",
        "min_flyby_altitude_km", "rp_min_km", "allow_flyby",
        "patched", "changed", "ksp_source",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fields})


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Patch BodyCatalog radii/atmospheres from KSP/ModuleManager configs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input-catalog", required=True, type=Path)
    p.add_argument("--config-cache", type=Path, default=None, help="ModuleManager.ConfigCache")
    p.add_argument("--ksp-config", nargs="*", type=Path, default=[], help="Extra KSP/Kopernicus cfg/txt files, fallback order")
    p.add_argument("--principia-gravity-model", type=Path, default=None, help="Optional kerbol_gravity_model.proto.txt")
    p.add_argument("--default-min-altitude-km", type=float, default=50.0)
    p.add_argument("--default-vacuum-margin-km", type=float, default=0.0)
    p.add_argument("--default-atmosphere-margin-km", type=float, default=20.0)
    p.add_argument("--prefer-ksp-mu", action="store_true", help="Overwrite existing mu with KSP/Principia parsed mu when available")
    p.add_argument("--prefer-ksp-radius", action="store_true", help="Overwrite existing radius with KSP config radius when available")
    p.add_argument("--prefer-ksp-atmosphere", action="store_true", default=True, help="Overwrite atmosphere_top_km with KSP config when available")
    p.add_argument("--recompute-min-altitude", action="store_true", help="Ignore old min altitude and recompute from defaults/atmosphere")
    p.add_argument("--start-body", default="Kerbin")
    p.add_argument("--force-disallow-start-body", action="store_true", help="Mark start body as not usable as flyby")
    p.add_argument("--output-json", required=True, type=Path)
    p.add_argument("--report-csv", type=Path, default=None)
    args = p.parse_args(argv)
    if args.default_min_altitude_km < 0:
        p.error("--default-min-altitude-km must be non-negative")
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    catalog = json.loads(args.input_catalog.read_text(encoding="utf-8"))

    config_paths: List[Path] = []
    if args.config_cache:
        config_paths.append(args.config_cache)
    config_paths.extend(args.ksp_config)
    if not config_paths:
        raise SystemExit("[FATAL] supply --config-cache and/or --ksp-config")

    ksp = parse_ksp_configs(config_paths)
    proto_mu = parse_principia_gravity_model(args.principia_gravity_model)
    patched, rows = patch_catalog(catalog, ksp, proto_mu, args)

    write_json(args.output_json, patched)
    if args.report_csv:
        write_report_csv(args.report_csv, rows)

    allowed = sum(1 for r in rows if r.get("allow_flyby"))
    missing_radius = sum(1 for r in rows if r.get("radius_km") in (None, ""))
    missing_mu = sum(1 for r in rows if r.get("mu_km3_s2") in (None, ""))

    print("=" * 80)
    print("MGA BODY CATALOG KSP PATCH V0.1")
    print("=" * 80)
    print(f"Catalog bodies:   {len(rows)}")
    print(f"Parsed KSP bodies:{len(ksp)}")
    print(f"Parsed proto mu:  {len(proto_mu)}")
    print(f"Flyby allowed:    {allowed}")
    print(f"Missing mu:       {missing_mu}")
    print(f"Missing radius:   {missing_radius}")
    print("\nBody summary:")
    print(f"{'Body':<12} {'mu km^3/s^2':>16} {'R km':>12} {'atm km':>10} {'rp_min km':>12} {'flyby':>7}")
    print("-" * 80)
    for r in rows:
        mu_s = "" if r.get("mu_km3_s2") in (None, "") else f"{float(r['mu_km3_s2']):.6g}"
        rad_s = "" if r.get("radius_km") in (None, "") else f"{float(r['radius_km']):.6g}"
        atm_s = "" if r.get("atmosphere_top_km") in (None, "") else f"{float(r['atmosphere_top_km']):.6g}"
        rp_s = "" if r.get("rp_min_km") in (None, "") else f"{float(r['rp_min_km']):.6g}"
        print(f"{r['body']:<12} {mu_s:>16} {rad_s:>12} {atm_s:>10} {rp_s:>12} {str(r.get('allow_flyby')):>7}")
    print("=" * 80)
    print(f"[OK] wrote JSON: {args.output_json}")
    if args.report_csv:
        print(f"[OK] wrote CSV:  {args.report_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
