"""Flyby audit module skeleton.

This module should consume LegOptimizationResult rows only. It should not
recompute Lambert or infer missing states.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ksp_mga.core.schemas import FlybyAuditResult, read_leg_optimizations, write_flyby_audit


@dataclass(frozen=True)
class FlybyAuditConfig:
    bsp: Path
    tpc: Path
    central_body: str
    transform: str
    rp_altitude_km: float = 50.0
    rp_scale: float = 1.05
    pass_vinf_mismatch_km_s: float = 0.1
    check_vinf_mismatch_km_s: float = 0.3


def audit_corrected_flybys(
    leg_optimization_csv: Path,
    sequence: list[str],
    candidate_id: str,
    config: FlybyAuditConfig,
) -> FlybyAuditResult:
    """Audit v-infinity continuity and flyby turn feasibility.

    Migration target:
      Move logic from native_corrected_flyby_audit_v0_1.py here, using the
      canonical leg_optimization schema.
    """
    _rows = read_leg_optimizations(leg_optimization_csv)
    raise NotImplementedError(
        "Move native_corrected_flyby_audit_v0_1.py logic into audit_corrected_flybys()."
    )


def main_cli() -> int:
    import argparse

    p = argparse.ArgumentParser(description="Audit corrected native flybys.")
    p.add_argument("--leg-optimization-csv", type=Path, required=True)
    p.add_argument("--sequence", nargs="+", required=True)
    p.add_argument("--candidate-id", required=True)
    p.add_argument("--bsp", type=Path, required=True)
    p.add_argument("--tpc", type=Path, required=True)
    p.add_argument("--central-body", default="Sun")
    p.add_argument("--transform", default="+Z,-X,+Y")
    p.add_argument("--output-json", type=Path, required=True)
    args = p.parse_args()

    result = audit_corrected_flybys(
        args.leg_optimization_csv,
        [x.upper() for x in args.sequence],
        args.candidate_id,
        FlybyAuditConfig(args.bsp, args.tpc, args.central_body, args.transform),
    )
    write_flyby_audit(args.output_json, result)
    print(f"[OK] wrote {args.output_json}")
    return 0
