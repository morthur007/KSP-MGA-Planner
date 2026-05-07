"""Powered flyby bridge module skeleton."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PoweredBridgeConfig:
    bsp: Path
    tpc: Path
    plugin_b64: Path
    central_body: str
    transform: str
    impulse_server_executable: str = "principia_impulsive_particle_server"


def solve_powered_bridge(*args, **kwargs):
    """Migration target for native_powered_flyby_bridge_v0_1.py."""
    raise NotImplementedError("Move native_powered_flyby_bridge_v0_1.py logic here.")
