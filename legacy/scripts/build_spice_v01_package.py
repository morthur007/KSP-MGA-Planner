#!/usr/bin/env python3
"""
build_spice_v01_package.py

Congela um pacote SPICE V0.1 de planejamento a partir dos relatórios já gerados.

Ele não recalcula dinâmica. Ele lê:
- body_absolute_errors.csv
- family_relative_errors.csv
- metadata do BSP, se existir

E escreve:
- README_v0_1.md
- validation_manifest_v0_1.json
- target_policy_v0_1.json

Uso típico:
python build_spice_v01_package.py \
  --package-dir data/spice_v0_1_33y \
  --validation-dir data/spice_v0_1_33y/future_validation_year32 \
  --bsp data/spice_v0_1_33y/opm_mpe_principia_rebound_v0_1_33y.bsp \
  --tpc data/spice_v0_1_33y/opm_mpe_principia_rebound_v0_1_33y.ids.tpc \
  --label SPICE_V0.1_HELIO_33Y
"""
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def as_float(row: Dict[str, str], *names: str, default: float = 0.0) -> float:
    for name in names:
        if name in row and row[name] not in (None, ""):
            return float(row[name])
    return default


def classify_abs(abs_km: float) -> str:
    # Planejamento global, não flyby fino.
    if abs_km <= 1.0:
        return "A"
    if abs_km <= 10.0:
        return "A-"
    if abs_km <= 100.0:
        return "B"
    if abs_km <= 1000.0:
        return "C"
    return "D"


def operational_use_from_abs(abs_km: float) -> str:
    if abs_km <= 10.0:
        return "global_and_local_seed"
    if abs_km <= 100.0:
        return "global_search"
    if abs_km <= 1000.0:
        return "system_arrival_only"
    return "do_not_target_without_revalidation"


def classify_rel(rel_km: float) -> str:
    # Erro relativo de lua contra pai.
    if rel_km <= 1.0:
        return "A"
    if rel_km <= 10.0:
        return "A-"
    if rel_km <= 100.0:
        return "B"
    if rel_km <= 1000.0:
        return "C"
    return "D"


def rel_use(rel_km: float) -> str:
    if rel_km <= 10.0:
        return "flyby_seed_allowed"
    if rel_km <= 100.0:
        return "coarse_flyby_seed_only"
    if rel_km <= 1000.0:
        return "system_context_only"
    return "do_not_target_without_local_kernel"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--package-dir", required=True)
    ap.add_argument("--validation-dir", required=True)
    ap.add_argument("--bsp", required=True)
    ap.add_argument("--tpc", required=True)
    ap.add_argument("--label", default="SPICE_V0.1_HELIO")
    ap.add_argument("--planning-years", type=float, default=33.0)
    args = ap.parse_args()

    package_dir = Path(args.package_dir)
    validation_dir = Path(args.validation_dir)
    package_dir.mkdir(parents=True, exist_ok=True)

    abs_csv = validation_dir / "body_absolute_errors.csv"
    rel_csv = validation_dir / "family_relative_errors.csv"
    if not abs_csv.exists():
        raise SystemExit(f"missing {abs_csv}")
    if not rel_csv.exists():
        raise SystemExit(f"missing {rel_csv}")

    abs_rows = read_csv(abs_csv)
    rel_rows = read_csv(rel_csv)

    body_policy: Dict[str, Dict[str, Any]] = {}
    for row in abs_rows:
        body = row.get("body") or row.get("Body") or row.get("name")
        if not body:
            continue
        abs_km = as_float(row, "abs_pos_err_km", "Abs km", "abs_km", "pos_err_km")
        vel_ms = as_float(row, "vel_err_m_s", "Vel m/s", "vel_m_s", default=0.0)
        cls = classify_abs(abs_km)
        body_policy[body] = {
            "absolute_error_km": abs_km,
            "velocity_error_m_s": vel_ms,
            "absolute_class": cls,
            "operational_use": operational_use_from_abs(abs_km),
        }

    family_policy: List[Dict[str, Any]] = []
    for row in rel_rows:
        parent = row.get("parent") or row.get("Parent")
        child = row.get("child") or row.get("Child")
        if not parent or not child:
            continue
        rel_km = as_float(row, "relative_pos_err_km", "Rel km", "rel_km")
        parent_abs = as_float(row, "parent_abs_err_km", "Parent km", default=0.0)
        child_abs = as_float(row, "child_abs_err_km", "Child km", default=0.0)
        frac = as_float(row, "relative_error_fraction", "Rel frac", default=0.0)
        family_policy.append({
            "parent": parent,
            "child": child,
            "parent_absolute_error_km": parent_abs,
            "child_absolute_error_km": child_abs,
            "relative_error_km": rel_km,
            "relative_error_fraction": frac,
            "relative_class": classify_rel(rel_km),
            "operational_use": rel_use(rel_km),
        })

    # Derived target sets.
    global_targets = sorted([
        b for b, p in body_policy.items()
        if p["operational_use"] in {"global_and_local_seed", "global_search"}
    ])
    system_arrival_targets = sorted([
        b for b, p in body_policy.items()
        if p["operational_use"] == "system_arrival_only"
    ])
    blocked_targets = sorted([
        b for b, p in body_policy.items()
        if p["operational_use"] == "do_not_target_without_revalidation"
    ])

    flyby_seed_allowed = sorted([
        f"{r['parent']}->{r['child']}" for r in family_policy
        if r["operational_use"] == "flyby_seed_allowed"
    ])
    coarse_flyby_seed_only = sorted([
        f"{r['parent']}->{r['child']}" for r in family_policy
        if r["operational_use"] == "coarse_flyby_seed_only"
    ])
    local_kernel_required = sorted([
        f"{r['parent']}->{r['child']}" for r in family_policy
        if r["operational_use"] in {"system_context_only", "do_not_target_without_local_kernel"}
    ])

    manifest = {
        "label": args.label,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "bsp": str(Path(args.bsp)),
        "tpc": str(Path(args.tpc)),
        "planning_years": args.planning_years,
        "status": "planning_grade_heliocentric_v0_1",
        "validation_snapshot": str(validation_dir),
        "validation_interpretation": {
            "absolute_errors": "KSP/Principia future snapshot versus SPK/REBOUND prediction at the same ET.",
            "relative_family_errors": "Child-parent state error; preferred metric for moon system coherence.",
            "do_not_extrapolate_residual_corrections": True,
        },
        "body_policy": body_policy,
        "family_policy": family_policy,
        "target_sets": {
            "global_targets": global_targets,
            "system_arrival_targets": system_arrival_targets,
            "blocked_targets": blocked_targets,
            "flyby_seed_allowed": flyby_seed_allowed,
            "coarse_flyby_seed_only": coarse_flyby_seed_only,
            "local_kernel_required": local_kernel_required,
        },
    }

    target_policy = {
        "label": args.label,
        "allowed_for_global_mga": global_targets,
        "allowed_for_system_arrival_only": system_arrival_targets,
        "requires_local_revalidation_or_dense_family_kernel": blocked_targets,
        "family_flyby_policy": {
            "flyby_seed_allowed": flyby_seed_allowed,
            "coarse_flyby_seed_only": coarse_flyby_seed_only,
            "local_kernel_required": local_kernel_required,
        },
        "notes": [
            "Use this policy for sequence generation and pruning, not for final targeting.",
            "Use dense local/family kernels for moon flybys and final approach.",
            "Residual corrections from 120d validation are not extrapolated to 33y.",
        ],
    }

    (package_dir / "validation_manifest_v0_1.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (package_dir / "target_policy_v0_1.json").write_text(
        json.dumps(target_policy, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # README concise and operational.
    top_abs = sorted(body_policy.items(), key=lambda kv: kv[1]["absolute_error_km"], reverse=True)[:12]
    top_rel = sorted(family_policy, key=lambda r: r["relative_error_km"], reverse=True)[:12]

    readme = []
    readme.append(f"# {args.label}\n")
    readme.append("## Status\n")
    readme.append("Planning-grade heliocentric SPICE kernel for MGA search and PyKEP integration.\n")
    readme.append("This is not a final operational ephemeris for moon flybys or execution without local revalidation.\n")
    readme.append("\n## Kernel files\n")
    readme.append(f"- BSP: `{args.bsp}`\n")
    readme.append(f"- TPC: `{args.tpc}`\n")
    readme.append("\n## Operational interpretation\n")
    readme.append("- Use for global MGA search, Lambert grids, pruning, and candidate ranking.\n")
    readme.append("- Use parent-system arrival for Jool, Sarnus, Urlum, Neidon, Plock as appropriate.\n")
    readme.append("- Use dense local/family kernels for moon flybys and final targeting.\n")
    readme.append("- Do not extrapolate 120-day residual corrections to the multi-decade horizon.\n")
    readme.append("\n## Main accepted global targets\n")
    readme.append(", ".join(global_targets[:80]) + "\n")
    readme.append("\n## System-arrival-only / caution targets\n")
    readme.append(", ".join(system_arrival_targets[:80]) + "\n")
    readme.append("\n## Requires local revalidation or dense family kernel\n")
    readme.append(", ".join(blocked_targets[:80]) + "\n")
    readme.append("\n## Family flyby policy\n")
    readme.append("### Flyby seed allowed\n")
    readme.append(", ".join(flyby_seed_allowed) + "\n")
    readme.append("\n### Coarse flyby seed only\n")
    readme.append(", ".join(coarse_flyby_seed_only) + "\n")
    readme.append("\n### Local kernel required\n")
    readme.append(", ".join(local_kernel_required) + "\n")
    readme.append("\n## Largest absolute validation errors at year ~32\n")
    for body, pol in top_abs:
        readme.append(f"- {body}: {pol['absolute_error_km']:.3f} km, class {pol['absolute_class']}\n")
    readme.append("\n## Largest relative family errors at year ~32\n")
    for r in top_rel:
        readme.append(
            f"- {r['parent']} -> {r['child']}: {r['relative_error_km']:.3f} km, "
            f"class {r['relative_class']}, use={r['operational_use']}\n"
        )

    (package_dir / "README_v0_1.md").write_text("".join(readme), encoding="utf-8")

    print(f"[OK] {package_dir / 'README_v0_1.md'}")
    print(f"[OK] {package_dir / 'validation_manifest_v0_1.json'}")
    print(f"[OK] {package_dir / 'target_policy_v0_1.json'}")
    print("\nGlobal targets:", ", ".join(global_targets[:30]))
    print("System-arrival-only:", ", ".join(system_arrival_targets[:30]))
    print("Blocked/revalidate:", ", ".join(blocked_targets[:30]))


if __name__ == "__main__":
    main()
