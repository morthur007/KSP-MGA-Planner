"""Pipeline orchestration skeleton.

The final triage should call Python functions, not Python scripts via subprocess.
The only subprocess should be the native Principia impulse server.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ksp_mga.core.schemas import read_candidate_seed
from ksp_mga.native.leg_optimizer import LegOptimizerConfig, optimize_candidate_legs


@dataclass(frozen=True)
class TriageConfig:
    candidates_csv: Path
    ranks: list[int]
    work_dir: Path
    leg_optimizer: LegOptimizerConfig
    jobs: int = 1


def run_triage(config: TriageConfig) -> list[dict]:
    """Run triage over candidate ranks.

    Migration target:
      Move nbody_pipeline_orchestrator_v0_1.py here after leg_optimizer and
      flyby_audit become callable modules.
    """
    results: list[dict] = []
    for rank in config.ranks:
        candidate = read_candidate_seed(config.candidates_csv, rank)
        rank_dir = config.work_dir / f"rank_{rank:03d}"
        try:
            leg_results = optimize_candidate_legs(candidate, config.leg_optimizer, rank_dir)
            results.append({"rank": rank, "candidate_id": candidate.candidate_id, "status": "LEGS_OK", "legs": leg_results})
        except NotImplementedError:
            raise
        except Exception as e:
            results.append({"rank": rank, "candidate_id": candidate.candidate_id, "status": "FAIL", "message": str(e)})
    return results
