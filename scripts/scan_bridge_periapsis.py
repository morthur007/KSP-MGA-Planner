#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np

from ksp_mga.native.powered_flyby_bridge import build_setup, ExtendedImpulseServer
from ksp_mga.native.leg_optimizer import norm, sample_raw_body_state

class LinearEphemerisCache:
    """Interpola o estado do planeta a partir de um único ponto para evitar I/O no disco."""
    def __init__(self, t_ref: float, r_ref: np.ndarray, v_ref: np.ndarray):
        self.t_ref = t_ref
        self.r_ref = r_ref
        self.v_ref = v_ref

    def get_state(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        # S = S_0 + V * dt
        dt = t - self.t_ref
        r = self.r_ref + self.v_ref * dt
        v = self.v_ref # A aceleração gravitacional em 60s não muda a velocidade de forma detectável.
        return r, v

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--leg-optimizations", type=Path, required=True)
    p.add_argument("--flyby-audit", type=Path, required=True)
    p.add_argument("--flyby-index", type=int, required=True)
    p.add_argument("--body-catalog", type=Path, required=True)
    p.add_argument("--plugin-b64", type=Path, required=True)
    p.add_argument("--server", default="principia_impulsive_particle_server")
    p.add_argument("--sampler", default="sample_principia_ephemeris")
    p.add_argument("--plugin-base-et-s", type=float, default=81.65168640136972)
    p.add_argument("--raw-origin-body", default="Sun")
    p.add_argument("--raw-cache-dir", type=Path, default=Path("data/runs/frame_debug/raw_cache"))

    p.add_argument("--min-altitude-km", type=float, default=50.0)
    p.add_argument("--atmosphere-margin-km", type=float, default=10.0)
    p.add_argument("--samples", type=int, default=81)
    p.add_argument("--time-margin-s", type=float, default=60.0)
    p.add_argument("--target-altitude-km", type=float, default=None)
    p.add_argument("--output-csv", type=Path, required=True)

    args = p.parse_args()
    setup = build_setup(args)

    lower_dt = (setup.t0_s + args.time_margin_s) - setup.t_event_s
    upper_dt = (setup.t1_s - args.time_margin_s) - setup.t_event_s

    if args.target_altitude_km is None:
        target_alt = setup.alt_required_km
    else:
        target_alt = args.target_altitude_km

    print("=== BRIDGE PERIAPSIS SCAN ===")
    print(f"body       : {setup.body}")
    print(f"flyby_index: {setup.flyby_index}")
    print(f"dt range   : {lower_dt:.1f} .. {upper_dt:.1f} s")
    print(f"target alt : {target_alt:.3f} km")
    print("")

    # =========================================================================
    # PRE-CACHE DO PLANETA: Lê do disco UMA ÚNICA VEZ no centro da janela
    # =========================================================================
    print(f"Buscando posição âncora de {setup.body} em t={setup.t_event_s:.1f}...")
    body_r_ref, body_v_ref = sample_raw_body_state(
        sampler=args.sampler,
        plugin_b64=args.plugin_b64,
        target_body=setup.body,
        sampler_central_body=args.raw_origin_body,
        et_s=setup.t_event_s,
        plugin_base_et_s=args.plugin_base_et_s,
        work_dir=args.raw_cache_dir,
    )
    
    ephem_cache = LinearEphemerisCache(setup.t_event_s, body_r_ref, body_v_ref)
    print("Cache criado. Iniciando varredura rápida...\n")
    # =========================================================================

    rows = []

    with ExtendedImpulseServer(args.server, args.plugin_b64) as server:
        for i, dt in enumerate(np.linspace(lower_dt, upper_dt, args.samples), start=1):
            burn_t = setup.t_event_s + float(dt)

            resp = server.propagate(
                req_id=f"scan_{i}",
                t0_s=setup.t0_s,
                burn_t_s=burn_t,
                t1_s=setup.t1_s,
                r0_m=setup.r0_m,
                v0_m_s=setup.v0_m_s,
                burn_dv_m_s=np.zeros(3),
            )

            row = {
                "i": i,
                "dt_s": float(dt),
                "burn_t_s": burn_t,
                "status": resp.status,
                "message": resp.message,
            }

            if resp.status == "ok":
                # OBTÉM DADOS DO CACHE LINEAR EM MICROSSEGUNDOS
                body_r, body_v = ephem_cache.get_state(burn_t)

                rel_r = resp.burn_r_m - body_r
                rel_v = resp.burn_v_before_m_s - body_v

                radius_km = norm(rel_r) / 1000.0
                alt_km = radius_km - setup.radius_km
                vr_km_s = float(np.dot(rel_r, rel_v) / max(norm(rel_r), 1.0)) / 1000.0
                speed_km_s = norm(rel_v) / 1000.0

                score = abs(vr_km_s) + 0.002 * abs(alt_km - target_alt)

                row.update({
                    "radius_km": radius_km,
                    "altitude_km": alt_km,
                    "radial_v_km_s": vr_km_s,
                    "speed_km_s": speed_km_s,
                    "score": score,
                })

                print(
                    f"[{i:03d}] dt={dt:9.1f} s "
                    f"alt={alt_km:10.3f} km "
                    f"vr={vr_km_s:9.5f} km/s "
                    f"speed={speed_km_s:9.5f} km/s "
                    f"score={score:10.6f}"
                )
            else:
                row.update({
                    "radius_km": math.inf,
                    "altitude_km": math.inf,
                    "radial_v_km_s": math.inf,
                    "speed_km_s": math.inf,
                    "score": math.inf,
                })

            rows.append(row)

    rows.sort(key=lambda r: float(r["score"]))

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with args.output_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    best = rows[0]
    print("")
    print("=== BEST SCAN POINT ===")
    print(f"dt_s        : {best['dt_s']}")
    print(f"burn_t_s    : {best['burn_t_s']}")
    print(f"altitude_km : {best['altitude_km']}")
    print(f"radial_v    : {best['radial_v_km_s']}")
    print(f"score       : {best['score']}")
    print(f"[OK] wrote {args.output_csv}")

    return 0

if __name__ == "__main__":
    raise SystemExit(main())