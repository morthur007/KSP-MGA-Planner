#!/usr/bin/env python3
"""Placeholder CLI for the refactored triage pipeline.

Use after moving the working leg optimizer/audit implementations into modules.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ksp_mga.core.config import load_config


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("config", type=Path)
    args = p.parse_args()
    cfg = load_config(args.config)
    print(f"[OK] loaded config: {cfg.name}")
    print("This CLI is a scaffold. Move working orchestrator logic into ksp_mga.pipeline.triage first.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
