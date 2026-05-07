"""YAML config loader with minimal validation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml


@dataclass(frozen=True)
class KernelConfig:
    bsp: Path
    tpc: Path
    metadata: Path | None = None
    body_catalog: Path | None = None


@dataclass(frozen=True)
class PrincipiaConfig:
    plugin_b64: Path
    impulse_server: str = "principia_impulsive_particle_server"


@dataclass(frozen=True)
class MissionConfig:
    name: str
    kernels: KernelConfig
    principia: PrincipiaConfig
    transform: str
    raw: dict[str, Any]


def require(mapping: Mapping[str, Any], key: str) -> Any:
    if key not in mapping:
        raise KeyError(f"missing required config key: {key}")
    return mapping[key]


def load_config(path: str | Path) -> MissionConfig:
    p = Path(path)
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"config must be a mapping: {p}")

    kernels = require(data, "kernels")
    principia = require(data, "principia")
    frames = data.get("frames", {})

    return MissionConfig(
        name=str(data.get("mission", p.stem)),
        kernels=KernelConfig(
            bsp=Path(require(kernels, "bsp")),
            tpc=Path(require(kernels, "tpc")),
            metadata=Path(kernels["metadata"]) if kernels.get("metadata") else None,
            body_catalog=Path(kernels["body_catalog"]) if kernels.get("body_catalog") else None,
        ),
        principia=PrincipiaConfig(
            plugin_b64=Path(require(principia, "plugin_b64")),
            impulse_server=str(principia.get("impulse_server", "principia_impulsive_particle_server")),
        ),
        transform=str(frames.get("canonical_to_principia_raw", "+Z,-X,+Y")),
        raw=data,
    )
