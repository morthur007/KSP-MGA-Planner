#!/usr/bin/env python3
"""
principia_absolute_zero_snapshot_v2.py

Snapshot kRPC com o jogo pausado:
- pausa pelo serviço KRPC, não pelo SpaceCenter;
- espera buffers assentarem;
- lê todos os corpos no mesmo estado congelado;
- faz dupla leitura enquanto pausado para medir jitter residual;
- exporta RAW por padrão.

Objetivo:
testar se pausar o jogo elimina o ruído de snapshot/FixedUpdate/kRPC.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, Tuple

import krpc  # type: ignore


Vec3 = Tuple[float, float, float]


def vec3(v: Any) -> Vec3:
    return (float(v[0]), float(v[1]), float(v[2]))


def sub(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def norm(v: Vec3) -> float:
    return math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])


def transform(v: Vec3, mode: str) -> Vec3:
    if mode == "raw":
        return v
    if mode == "swap_yz":
        return (v[0], v[2], v[1])
    if mode == "negate_z":
        return (v[0], v[1], -v[2])
    raise ValueError(mode)


def get_body_names(bodies: Any) -> list[str]:
    try:
        return sorted(str(k) for k in bodies.keys())
    except Exception:
        return sorted(str(getattr(b, "name")) for b in bodies)


def get_body(bodies: Any, name: str) -> Any:
    try:
        return bodies[name]
    except Exception:
        for b in bodies:
            if getattr(b, "name", None) == name:
                return b
        raise KeyError(name)


def set_paused(conn: Any, value: bool) -> None:
    """
    kRPC pause lives on conn.krpc in normal Python stubs.
    Keep fallbacks because generated bindings can vary.
    """
    



def is_paused(conn: Any) -> bool | None:
    try:
        return bool(conn.krpc.paused)
    except Exception:
        return None


def read_states(sc: Any, bodies: Any, names: list[str], frame: Any, transform_mode: str) -> tuple[float, Dict[str, Dict[str, Any]]]:
    ut = float(sc.ut)
    out: Dict[str, Dict[str, Any]] = {}

    for name in names:
        body = get_body(bodies, name)
        p = transform(vec3(body.position(frame)), transform_mode)
        v = transform(vec3(body.velocity(frame)), transform_mode)

        entry: Dict[str, Any] = {
            "r": [p[0], p[1], p[2]],
            "v": [v[0], v[1], v[2]],
        }

        for attr in ("gravitational_parameter", "mass", "equatorial_radius", "radius", "sphere_of_influence"):
            try:
                entry[attr] = float(getattr(body, attr))
            except Exception:
                pass

        out[name] = entry

    return ut, out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--central-body", default="Sun")
    ap.add_argument("--output", default="data/jnsq_gate0/snapshot_absolute_zero.json")
    ap.add_argument("--transform-mode", choices=["raw", "swap_yz", "negate_z"], default="raw")
    ap.add_argument("--settle-seconds", type=float, default=1.0)
    ap.add_argument("--double-read-delay-seconds", type=float, default=0.5)
    ap.add_argument("--leave-paused", action="store_true")
    args = ap.parse_args()

    print("Conectando ao kRPC...")
    conn = krpc.connect(name="Absolute_Zero_Snapshot_V2")

    try:
        sc = conn.space_center
        bodies = sc.bodies
        names = get_body_names(bodies)

        central = get_body(bodies, args.central_body)
        frame = central.non_rotating_reference_frame

        original_paused = is_paused(conn)
        print(f"Estado paused original: {original_paused}")

        print("Pausando pelo serviço kRPC...")
        set_paused(conn, True)
        time.sleep(args.settle_seconds)

        paused_now = is_paused(conn)
        ut1, states1 = read_states(sc, bodies, names, frame, args.transform_mode)

        time.sleep(args.double_read_delay_seconds)

        ut2, states2 = read_states(sc, bodies, names, frame, args.transform_mode)

        # Auditoria: em pause, UT idealmente não deve avançar e r/v devem ser idênticos.
        max_dr = 0.0
        max_dv = 0.0
        max_dr_body = ""
        max_dv_body = ""

        for name in names:
            r1 = tuple(states1[name]["r"])
            r2 = tuple(states2[name]["r"])
            v1 = tuple(states1[name]["v"])
            v2 = tuple(states2[name]["v"])

            dr = norm(sub(r2, r1))
            dv = norm(sub(v2, v1))

            if dr > max_dr:
                max_dr = dr
                max_dr_body = name
            if dv > max_dv:
                max_dv = dv
                max_dv_body = name

        snapshot = {
            "schema": "principia_snapshot.v4_absolute_zero_pause",
            "epoch_ut_s": ut2,
            "start_ut_s_first_read": ut1,
            "reference_body": args.central_body,
            "reference_frame": "central.non_rotating_reference_frame",
            "transform_mode": args.transform_mode,
            "paused_confirmed": paused_now,
            "double_read_delay_seconds": args.double_read_delay_seconds,
            "pause_audit": {
                "ut_delta_s": ut2 - ut1,
                "max_position_delta_m": max_dr,
                "max_position_delta_body": max_dr_body,
                "max_velocity_delta_m_s": max_dv,
                "max_velocity_delta_body": max_dv_body,
            },
            "bodies": states2,
        }

        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(snapshot, indent=2, ensure_ascii=False), encoding="utf-8")

        print("[OK] snapshot salvo:", out)
        print(f"UT1={ut1:.9f} UT2={ut2:.9f} ΔUT={ut2-ut1:.9e} s")
        print(f"max Δr paused = {max_dr:.9e} m em {max_dr_body}")
        print(f"max Δv paused = {max_dv:.9e} m/s em {max_dv_body}")

    finally:
            
        conn.close()


if __name__ == "__main__":
    main()