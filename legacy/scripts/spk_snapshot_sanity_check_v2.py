#!/usr/bin/env python3
"""
spk_snapshot_sanity_check.py

Teste de sanidade zero-epoch para a cadeia:
  true_snapshot.json -> REBOUND -> SPK Type 3 -> SpiceyPy

Objetivo:
  1) Confirmar que o SPK, no epoch inicial, reproduz o snapshot de entrada.
  2) Separar erro de empacotamento/SPICE/frame de erro físico-dinâmico de propagação.

Interpretação:
  - Erro inicial ~ metros ou menor: o SPK está lendo a condição inicial corretamente.
  - Erro inicial grande: problema no writer SPK, mapeamento NAIF, frame, sinal de eixo, centro ou epoch.
  - Erro inicial baixo mas erro cresce com o tempo: problema de modelo físico/estado inicial/calibração, não do SPK.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

try:
    import spiceypy as spice
except ImportError as exc:
    raise SystemExit("Instale spiceypy: pip install spiceypy") from exc


def norm3(v: Iterable[float]) -> float:
    a, b, c = v
    return math.sqrt(a*a + b*b + c*c)


def spice_name(name: str) -> str:
    return name.upper().replace(" ", "_").replace("-", "_")


def maybe_flip_z(state6, flip_z: bool):
    x, y, z, vx, vy, vz = map(float, state6)
    if flip_z:
        return x, y, -z, vx, vy, -vz
    return x, y, z, vx, vy, vz


def load_snapshot(path: Path) -> Tuple[float, Dict[str, Tuple[float, float, float, float, float, float]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    start_ut = float(payload.get("start_ut_seconds", 0.0))
    et_offset = float(payload.get("et_offset_seconds", 0.0))
    et = start_ut + et_offset
    eph = payload["ephemerides"]
    states = {}
    for body, block in eph.items():
        rows = block.get("states", [])
        if not rows:
            continue
        row = rows[0]
        states[body] = tuple(map(float, row[1:7]))
    return et, states


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot-json", type=Path, required=True)
    ap.add_argument("--bsp", type=Path, required=True)
    ap.add_argument("--tpc", type=Path, required=True)
    ap.add_argument("--central-body", required=True)
    ap.add_argument("--frame", default="J2000")
    ap.add_argument("--flip-z-snapshot", action="store_true", help="Aplique z -> -z no snapshot antes de comparar.")
    ap.add_argument("--max-print", type=int, default=1000)
    args = ap.parse_args()

    et0, snapshot = load_snapshot(args.snapshot_json)
    center = spice_name(args.central_body)

    spice.furnsh(str(args.tpc))
    spice.furnsh(str(args.bsp))

    rows = []
    try:
        for body, s in snapshot.items():
            if body.upper() == args.central_body.upper():
                continue
            sx, sy, sz, svx, svy, svz = maybe_flip_z(s, args.flip_z_snapshot)
            try:
                st, _ = spice.spkezr(spice_name(body), et0, args.frame, "NONE", center)
            except spice.stypes.SpiceyError as e:
                rows.append((body, None, None, f"SPICE_ERROR: {str(e).splitlines()[0]}"))
                continue
            rx, ry, rz = st[0]*1000.0, st[1]*1000.0, st[2]*1000.0
            rvx, rvy, rvz = st[3]*1000.0, st[4]*1000.0, st[5]*1000.0
            pos_err = norm3((sx-rx, sy-ry, sz-rz))
            vel_err = norm3((svx-rvx, svy-rvy, svz-rvz))
            if pos_err <= 1.0 and vel_err <= 1e-6:
                status = "EXCELENTE"
            elif pos_err <= 100.0 and vel_err <= 1e-3:
                status = "OK"
            elif pos_err <= 10_000.0 and vel_err <= 0.1:
                status = "SUSPEITO"
            else:
                status = "FALHA"
            rows.append((body, pos_err, vel_err, status))
    finally:
        spice.kclear()

    numeric = [r for r in rows if r[1] is not None]
    max_pos = max((r[1] for r in numeric), default=float("nan"))
    max_vel = max((r[2] for r in numeric), default=float("nan"))

    print(f"Epoch inicial ET: {et0:.9f} s")
    print(f"Corpo central: {args.central_body}")
    print(f"Max erro inicial posição: {max_pos:.6f} m")
    print(f"Max erro inicial velocidade: {max_vel:.9f} m/s")
    print()
    print(f"{'Corpo':<18} | {'Erro pos inicial (m)':>22} | {'Erro vel inicial (m/s)':>24} | Status")
    print("-"*92)
    for body, pe, ve, status in sorted(rows)[:args.max_print]:
        if pe is None:
            print(f"{body:<18} | {'-':>22} | {'-':>24} | {status}")
        else:
            print(f"{body:<18} | {pe:22.6f} | {ve:24.9f} | {status}")

    if max_pos > 100.0 or max_vel > 1e-3:
        print("\n[DIAGNÓSTICO] Erro no epoch inicial é grande: investigue writer SPK, CDATA, frame/eixo, centro, NAIF IDs ou epoch.")
    else:
        print("\n[DIAGNÓSTICO] Epoch inicial fecha bem: divergência posterior é dinâmica/modelo físico/calibração.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
