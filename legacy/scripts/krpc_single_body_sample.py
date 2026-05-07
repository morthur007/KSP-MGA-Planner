#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import krpc


def T_krpc(vec):
    """
    Conversão usada historicamente no seu pipeline kRPC right-handed:
    KSP body.position(frame) -> LevelA/export astrodinâmico.

    Se sua coleta de 5d já usa exatamente outra conversão, ajuste aqui.
    Pelo histórico, esta é a convenção que você vinha usando:
      r = [x, z, y]
    """
    x, y, z = vec
    return [-y, z, x]

def stop_warp(sc):
    try:
        sc.rails_warp_factor = 0
    except Exception:
        pass
    try:
        sc.physics_warp_factor = 0
    except Exception:
        pass
    time.sleep(0.25)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-csv", required=True)
    ap.add_argument("--output-json", default=None)
    ap.add_argument("--reference-body", default="Sun")
    ap.add_argument("--pause", action="store_true")
    ap.add_argument("--settle-seconds", type=float, default=2.0)
    ap.add_argument("--samples", type=int, default=1)
    ap.add_argument("--sample-spacing-seconds", type=float, default=0.25)
    args = ap.parse_args()

    conn = krpc.connect(name="single_body_sample")
    sc = conn.space_center

    stop_warp(sc)

    

    time.sleep(args.settle_seconds)

    bodies = sc.bodies
    if args.reference_body not in bodies:
        raise SystemExit(f"Reference body {args.reference_body!r} não encontrado.")

    frame = bodies[args.reference_body].non_rotating_reference_frame

    all_samples = []

    for sample_index in range(args.samples):
        ut_before = float(sc.ut)

        raw = {}
        for name, body in bodies.items():
            try:
                pos = body.position(frame)
                vel = body.velocity(frame)

                r = T_krpc(pos)
                v = T_krpc(vel)

                raw[name] = {
                    "r": r,
                    "v": v,
                }
            except Exception as e:
                print(f"[WARN] falha lendo {name}: {e}")

        ut_after = float(sc.ut)
        ut_mid = 0.5 * (ut_before + ut_after)

        if args.reference_body in raw:
            cr = raw[args.reference_body]["r"]
            cv = raw[args.reference_body]["v"]
        else:
            cr = [0.0, 0.0, 0.0]
            cv = [0.0, 0.0, 0.0]

        for name, st in raw.items():
            r_rel = [st["r"][i] - cr[i] for i in range(3)]
            v_rel = [st["v"][i] - cv[i] for i in range(3)]

            if name == args.reference_body:
                r_rel = [0.0, 0.0, 0.0]
                v_rel = [0.0, 0.0, 0.0]

            all_samples.append({
                "sample_index": sample_index,
                "body": name,
                "ut_before_s": ut_before,
                "ut_after_s": ut_after,
                "ut_mid_s": ut_mid,
                "read_duration_s": ut_after - ut_before,
                "x_m": r_rel[0],
                "y_m": r_rel[1],
                "z_m": r_rel[2],
                "vx_m_s": v_rel[0],
                "vy_m_s": v_rel[1],
                "vz_m_s": v_rel[2],
            })

        if sample_index + 1 < args.samples:
            time.sleep(args.sample_spacing_seconds)

    

    out = Path(args.output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "sample_index",
        "body",
        "ut_before_s",
        "ut_after_s",
        "ut_mid_s",
        "read_duration_s",
        "x_m",
        "y_m",
        "z_m",
        "vx_m_s",
        "vy_m_s",
        "vz_m_s",
    ]

    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(all_samples)

    print(f"[OK] CSV: {out}")
    print(f"[INFO] rows: {len(all_samples)}")
    if all_samples:
        print(f"[INFO] first ut_mid: {all_samples[0]['ut_mid_s']:.9f}")
        print(f"[INFO] last  ut_mid: {all_samples[-1]['ut_mid_s']:.9f}")

    if args.output_json:
        Path(args.output_json).write_text(
            json.dumps({
                "reference_body": args.reference_body,
                "samples": all_samples,
            }, indent=2),
            encoding="utf-8",
        )
        print(f"[OK] JSON: {args.output_json}")


if __name__ == "__main__":
    main()