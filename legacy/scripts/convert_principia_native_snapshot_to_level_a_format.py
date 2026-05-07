#!/usr/bin/env python3
"""
convert_principia_native_snapshot_to_level_a_format.py

Converte snapshot_principia_native_v2.json:
  bodies[name].r / bodies[name].v

para o formato aceito pelo rebound_level_a_cache.py:
  ephemerides[name].states[0] = [ut, x, y, z, vx, vy, vz]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--template", default=None)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    src = json.loads(Path(args.input).read_text())

    epoch = float(src.get("epoch_ut_s", src.get("principia_current_time_s", 0.0)))
    bodies = src["bodies"]

    # Se houver template aceito pelo rebound_level_a_cache, preserva schema extra.
    if args.template:
        out = json.loads(Path(args.template).read_text())
    else:
        out = {
            "schema": "principia_native_snapshot.level_a_compatible.v0",
            "epoch_ut_s": epoch,
            "reference_body": src.get("reference_body", "Sun"),
            "ephemerides": {},
        }

    out["epoch_ut_s"] = epoch
    out["reference_body"] = src.get("reference_body", out.get("reference_body", "Sun"))
    out.setdefault("ephemerides", {})

    for name, b in bodies.items():
        r = b["r"]
        v = b["v"]
        mu = b.get("mu", b.get("gravitational_parameter"))

        # O validador Level A exige estritamente a matriz [ut, x, y, z, vx, vy, vz]
        state_list = [
            epoch,
            float(r[0]), float(r[1]), float(r[2]),
            float(v[0]), float(v[1]), float(v[2])
        ]

        # Injeta diretamente, ignorando a estrutura de states do template antigo
        out["ephemerides"][name] = {
            **out["ephemerides"].get(name, {}),
            "gravitational_parameter": mu,
            "mu": mu,
            "states": [state_list],
        }

    Path(args.output).write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print("[OK]", args.output)
    print("[INFO] bodies:", len(out["ephemerides"]))
    print("[INFO] epoch:", epoch)

if __name__ == "__main__":
    main()