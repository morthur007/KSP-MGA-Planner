#!/usr/bin/env python3
"""
fix_event_principia_binormal_sign_v0.py

Converte um evento insert_navigation gerado pelos scripts SPICE antigos
(v0.3/v0/polish01/02) para a convenção real da UI/adapter do Principia.

Diagnóstico:
  - O backend Python antigo usava B = +unit(r x v).
  - A UI/adapter do Principia usa binormal com sinal oposto neste caso:
        B_principia = -B_python
  - Logo, para preservar o mesmo dv_raw planejado, basta inverter o
    componente Binormal do evento: [T,N,B] -> [T,N,-B].

Este script NÃO muda T nem N.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def norm3(v):
    return math.sqrt(sum(float(x) * float(x) for x in v))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--tag", default="principia_binormal_sign_fix_v0")
    args = ap.parse_args()

    d = json.loads(args.input.read_text())
    dv = d.get("delta_v_navigation_m_s")
    if not isinstance(dv, list) or len(dv) != 3:
        raise SystemExit("input does not contain delta_v_navigation_m_s[3]")

    old = [float(dv[0]), float(dv[1]), float(dv[2])]
    new = [old[0], old[1], -old[2]]

    d["delta_v_navigation_m_s_original_python_convention"] = old
    d["delta_v_navigation_m_s"] = new
    d["binormal_sign_fix"] = {
        "schema": args.tag,
        "reason": "Python backend used B=+h; Principia insert_navigation/UI uses opposite binormal sign for the exported navigation frame. Flipped B to preserve planned raw delta-v.",
        "old_delta_v_navigation_m_s": old,
        "new_delta_v_navigation_m_s": new,
        "old_norm_m_s": norm3(old),
        "new_norm_m_s": norm3(new),
    }

    pf = d.setdefault("planned_from_state", {})
    pf["binormal_sign_convention"] = "principia_ui"
    pf["binormal_sign_fix_applied"] = True

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(d, indent=2) + "\n")

    print("=== FIX EVENT PRINCIPIA BINORMAL SIGN ===")
    print(f"input  : {args.input}")
    print(f"output : {args.output}")
    print(f"old    : {old} |dv|={norm3(old):.9f}")
    print(f"new    : {new} |dv|={norm3(new):.9f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
