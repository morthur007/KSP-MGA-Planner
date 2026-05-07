#!/usr/bin/env python3
"""
principia_true_snapshot_v2.py

Snapshot inicial para REBOUND usando posições kRPC e velocidade por diferença finita central.

Por que v2:
  - A versão de dois pontos calcula uma velocidade forward-difference associada ao meio do intervalo,
    mas grava a posição no começo do intervalo.
  - Esta versão usa três passagens: p_minus, p0, p_plus, e grava a posição em p0 com velocidade
    central (p_plus - p_minus)/(t_plus - t_minus).
  - Também registra duração/skew de cada passagem de RPC, para detectar se a coleta sequencial
    está introduzindo erro por não simultaneidade.

Contrato de saída:
  - JSON legado compatível com rebound_ephemeris_to_spk_type3.py
  - unidades SI: m, m/s, kg, m^3/s^2
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, Tuple

try:
    import krpc  # type: ignore
except ImportError as exc:
    raise SystemExit("Instale krpc: pip install krpc") from exc

G_SI = 6.67430e-11


def vec(v: Any) -> Tuple[float, float, float]:
    return float(v[0]), float(v[1]), float(v[2])


def sub(a, b):
    return a[0]-b[0], a[1]-b[1], a[2]-b[2]


def div(a, s: float):
    return a[0]/s, a[1]/s, a[2]/s


def norm(a):
    return math.sqrt(a[0]*a[0] + a[1]*a[1] + a[2]*a[2])


def wait_until_ut(sc, target_ut: float, poll_s: float = 0.02):
    while float(sc.ut) < target_ut:
        time.sleep(poll_s)


def sample_positions(sc, bodies, frame):
    """One sequential pass. Returns pass midpoint UT and per-body positions."""
    t_start = float(sc.ut)
    positions = {}
    body_times = {}
    for name, b in bodies.items():
        tb0 = float(sc.ut)
        positions[name] = vec(b.position(frame))
        tb1 = float(sc.ut)
        body_times[name] = 0.5 * (tb0 + tb1)
    t_end = float(sc.ut)
    return {
        "pass_start_ut": t_start,
        "pass_end_ut": t_end,
        "pass_mid_ut": 0.5 * (t_start + t_end),
        "pass_duration_s": t_end - t_start,
        "positions": positions,
        "body_times": body_times,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, default=Path("data/true_snapshot_v2.json"))
    ap.add_argument("--central-body", default="Sun")
    ap.add_argument("--dt-seconds", type=float, default=120.0, help="Half interval for central difference. Total baseline is 2*dt.")
    ap.add_argument("--settle-seconds", type=float, default=1.0)
    ap.add_argument("--poll-seconds", type=float, default=0.02)
    ap.add_argument("--include-body-velocity-diagnostic", action="store_true")
    args = ap.parse_args()

    if args.dt_seconds <= 0:
        raise SystemExit("--dt-seconds precisa ser positivo")

    conn = krpc.connect(name="Principia_True_Snapshot_v2")
    try:
        sc = conn.space_center
        sc.rails_warp_factor = 0
        sc.physics_warp_factor = 0
        time.sleep(args.settle_seconds)

        bodies = sc.bodies
        if args.central_body not in bodies:
            raise SystemExit(f"Corpo central {args.central_body!r} não encontrado. Disponíveis: {sorted(bodies.keys())}")
        frame = bodies[args.central_body].non_rotating_reference_frame

        print("[1/3] Coletando p_minus...")
        minus = sample_positions(sc, bodies, frame)

        target0 = minus["pass_mid_ut"] + args.dt_seconds
        print(f"[2/3] Aguardando até UT ~ {target0:.6f} para p0...")
        wait_until_ut(sc, target0, args.poll_seconds)
        zero = sample_positions(sc, bodies, frame)

        target_plus = zero["pass_mid_ut"] + args.dt_seconds
        print(f"[3/3] Aguardando até UT ~ {target_plus:.6f} para p_plus...")
        wait_until_ut(sc, target_plus, args.poll_seconds)
        plus = sample_positions(sc, bodies, frame)

        t_minus = minus["pass_mid_ut"]
        t0 = zero["pass_mid_ut"]
        t_plus = plus["pass_mid_ut"]
        baseline = t_plus - t_minus

        if baseline <= 0:
            raise SystemExit("Baseline temporal inválida")

        export_bodies = {}
        catalog_bodies = {}
        diagnostics = {
            "method": "central_difference_three_passes",
            "central_body": args.central_body,
            "dt_seconds_requested": args.dt_seconds,
            "t_minus_ut": t_minus,
            "t0_ut": t0,
            "t_plus_ut": t_plus,
            "baseline_seconds": baseline,
            "passes": {
                "minus": {k: minus[k] for k in ("pass_start_ut", "pass_end_ut", "pass_mid_ut", "pass_duration_s")},
                "zero": {k: zero[k] for k in ("pass_start_ut", "pass_end_ut", "pass_mid_ut", "pass_duration_s")},
                "plus": {k: plus[k] for k in ("pass_start_ut", "pass_end_ut", "pass_mid_ut", "pass_duration_s")},
            },
            "body_velocity_diagnostic_m_s": {},
        }

        for name, b in bodies.items():
            p_minus = minus["positions"][name]
            p0 = zero["positions"][name]
            p_plus = plus["positions"][name]
            v = div(sub(p_plus, p_minus), baseline)
            export_bodies[name] = [t0, p0[0], p0[1], p0[2], v[0], v[1], v[2]]

            mu = float(b.gravitational_parameter)
            catalog_bodies[name] = {
                "mu_m3_s2": mu,
                "mass_kg": mu / G_SI,
                "radius_m": float(getattr(b, "equatorial_radius", 0.0) or 0.0),
            }

            if args.include_body_velocity_diagnostic:
                try:
                    kv = vec(b.velocity(frame))
                    diagnostics["body_velocity_diagnostic_m_s"][name] = {
                        "krpc_velocity_now": list(kv),
                        "central_difference_velocity": list(v),
                        "norm_delta_m_s": norm(sub(kv, v)),
                    }
                except Exception as exc:
                    diagnostics["body_velocity_diagnostic_m_s"][name] = {"error": repr(exc)}

        payload = {
            "schema": "principia_true_snapshot.v2",
            "reference_body": args.central_body,
            "start_ut_seconds": t0,
            "et_offset_seconds": 0.0,
            "time_scale": "KSP_UT_seconds_used_as_ET_like_argument_for_custom_SPICE_kernel",
            "frame": {
                "source": "kRPC CelestialBody.non_rotating_reference_frame",
                "center": args.central_body,
                "handedness": "KSP/kRPC native; left-handed unless converted downstream",
            },
            "sampling_diagnostics": diagnostics,
            "body_catalog": {"bodies": catalog_bodies},
            "ephemerides": {name: {"states": [state]} for name, state in export_bodies.items()},
        }

        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[OK] Snapshot v2 salvo em {args.output}")
        print(f"Epoch t0: {t0:.9f}; baseline central: {baseline:.6f} s")
        print(f"Duração passagens RPC: minus={minus['pass_duration_s']:.6f}s zero={zero['pass_duration_s']:.6f}s plus={plus['pass_duration_s']:.6f}s")
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
