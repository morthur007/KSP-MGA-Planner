"""N-body leg optimization module skeleton.

This file defines the public API the rest of the pipeline should call. Move the
working implementation from `native_optimize_candidate_legs_v0_1.py` into
`optimize_candidate_legs()` in small steps.

Important rule:
  This module consumes complete CandidateSeed leg velocities. It must not
  re-solve Lambert.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from ksp_mga.core.schemas import CandidateSeed, LegOptimizationResult, read_candidate_seed, write_leg_optimizations


@dataclass(frozen=True)
class LegOptimizerConfig:
    bsp: Path
    tpc: Path
    plugin_b64: Path
    central_body: str
    transform: str
    buffer_days: float = 0.235
    impulse_server_executable: str = "principia_impulsive_particle_server"


def optimize_candidate_legs(
    candidate: CandidateSeed,
    config: LegOptimizerConfig,
    work_dir: Path,
) -> list[LegOptimizationResult]:
    """Optimize all legs of a candidate with native Principia propagation.

    Migration target:
      Paste/refactor the logic from native_optimize_candidate_legs_v0_1.py here.

    Contract:
      - input: CandidateSeed with explicit legN_vdep/varr velocities.
      - output: LegOptimizationResult list containing final state columns.
    """
    raise NotImplementedError(
        "Move the working implementation from native_optimize_candidate_legs_v0_1.py "
        "into ksp_mga.native.leg_optimizer.optimize_candidate_legs()."
    )


def main_cli() -> int:
    import argparse

    p = argparse.ArgumentParser(description="Optimize candidate legs with native Principia propagation.")
    p.add_argument("--candidate-csv", type=Path, required=True)
    p.add_argument("--rank", type=int, required=True)
    p.add_argument("--bsp", type=Path, required=True)
    p.add_argument("--tpc", type=Path, required=True)
    p.add_argument("--plugin-b64", type=Path, required=True)
    p.add_argument("--central-body", default="Sun")
    p.add_argument("--transform", default="+Z,-X,+Y")
    p.add_argument("--buffer-days", type=float, default=0.235)
    p.add_argument("--impulse-server", default="principia_impulsive_particle_server")
    p.add_argument("--work-dir", type=Path, required=True)
    p.add_argument("--output-csv", type=Path, required=True)
    args = p.parse_args()

    candidate = read_candidate_seed(args.candidate_csv, args.rank)
    cfg = LegOptimizerConfig(
        bsp=args.bsp,
        tpc=args.tpc,
        plugin_b64=args.plugin_b64,
        central_body=args.central_body,
        transform=args.transform,
        buffer_days=args.buffer_days,
        impulse_server_executable=args.impulse_server,
    )
    results = optimize_candidate_legs(candidate, cfg, args.work_dir)
    write_leg_optimizations(args.output_csv, results)
    print(f"[OK] wrote {args.output_csv}")
    return 0
