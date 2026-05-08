#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import itertools
from pathlib import Path

import numpy as np
import spiceypy as spice

from scripts.smoke_impulse_server import (
    DAY_S,
    apply_transform,
    kepler_universal_propagate,
    norm,
    norm_name,
    parse_transform,
    sample_raw_body_state,
    spk_state,
)


AXES = ["X", "Y", "Z"]


def all_transforms() -> list[str]:
    out = []
    for perm in itertools.permutations(AXES):
        for signs in itertools.product([1, -1], repeat=3):
            out.append(",".join(("+" if s > 0 else "-") + a for s, a in zip(signs, perm)))
    return out


def read_candidate(path: Path, rank: int) -> dict[str, str]:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    return rows[rank - 1]


def body_mapping_error(
    *,
    body: str,
    et_s: float,
    central_body: str,
    transform_spec: str,
    sampler: str,
    plugin_b64: Path,
    plugin_base_et_s: float,
    raw_origin_body: str,
    raw_cache_dir: Path,
) -> tuple[float, np.ndarray]:
    tr = parse_transform(transform_spec)

    # SPK body state: LevelA / central-body relative.
    st_body = spk_state(body, et_s, central_body)
    body_levela_rel_m = st_body[:3] * 1000.0

    # Raw origin and raw body from Principia.
    origin_raw_r_m, _ = sample_raw_body_state(
        sampler=sampler,
        plugin_b64=plugin_b64,
        target_body=raw_origin_body,
        sampler_central_body=raw_origin_body,
        et_s=et_s,
        plugin_base_et_s=plugin_base_et_s,
        work_dir=raw_cache_dir,
    )

    body_raw_r_m, _ = sample_raw_body_state(
        sampler=sampler,
        plugin_b64=plugin_b64,
        target_body=body,
        sampler_central_body=raw_origin_body,
        et_s=et_s,
        plugin_base_et_s=plugin_base_et_s,
        work_dir=raw_cache_dir,
    )

    predicted_body_raw_m = origin_raw_r_m + apply_transform(body_levela_rel_m, tr)
    residual_vec_m = predicted_body_raw_m - body_raw_r_m
    return norm(residual_vec_m), residual_vec_m


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--candidate-seed", type=Path, required=True)
    p.add_argument("--rank", type=int, default=1)
    p.add_argument("--leg", type=int, default=1)

    p.add_argument("--bsp", type=Path, required=True)
    p.add_argument("--tpc", type=Path, required=True)
    p.add_argument("--plugin-b64", type=Path, required=True)
    p.add_argument("--sampler", default="sample_principia_ephemeris")

    p.add_argument("--central-body", default="Sun")
    p.add_argument("--central-mu-km3-s2", type=float, required=True)
    p.add_argument("--plugin-base-et-s", type=float, default=81.65168640136972)
    p.add_argument("--raw-origin-body", default="Sun")
    p.add_argument("--raw-cache-dir", type=Path, default=Path("data/runs/frame_debug/raw_cache"))

    p.add_argument("--buffer-days", type=float, default=0.235)
    p.add_argument("--top", type=int, default=20)
    args = p.parse_args()

    spice.furnsh(str(args.tpc))
    spice.furnsh(str(args.bsp))

    row = read_candidate(args.candidate_seed, args.rank)
    leg = args.leg

    dep = norm_name(row[f"leg{leg}_dep"])
    arr = norm_name(row[f"leg{leg}_arr"])

    dep_i = leg - 1
    arr_i = leg

    t_dep = float(row[f"event{dep_i}_et_s"])
    t_arr = float(row[f"event{arr_i}_et_s"])
    buffer_s = args.buffer_days * DAY_S

    t_start = t_dep + buffer_s
    t_end = t_arr - buffer_s

    st_dep_epoch = spk_state(dep, t_dep, args.central_body)
    vdep_km_s = np.array([
        float(row[f"leg{leg}_vdep_x_km_s"]),
        float(row[f"leg{leg}_vdep_y_km_s"]),
        float(row[f"leg{leg}_vdep_z_km_s"]),
    ])

    # Spacecraft LevelA state
    r_sc_start_levela_km, v_sc_start_levela_km_s = kepler_universal_propagate(
        st_dep_epoch[:3], vdep_km_s, buffer_s, args.central_mu_km3_s2,
    )
    r_sc_end_levela_km, v_sc_end_levela_km_s = kepler_universal_propagate(
        r_sc_start_levela_km, v_sc_start_levela_km_s, t_end - t_start, args.central_mu_km3_s2,
    )

    st_dep_start = spk_state(dep, t_start, args.central_body)
    st_arr_end = spk_state(arr, t_end, args.central_body)

    expected_dep_sep_km = norm(r_sc_start_levela_km - st_dep_start[:3])
    expected_arr_sep_km = norm(r_sc_end_levela_km - st_arr_end[:3])

    print("=== FRAME GEOMETRY PROOF ===")
    print(f"candidate : {row.get('candidate_id')} rank={args.rank}")
    print(f"leg       : {leg} {dep}->{arr}")
    print(f"t_start   : {t_start:.9f}")
    print(f"t_end     : {t_end:.9f}")
    print("")
    print("LevelA expected geometry:")
    print(f"  spacecraft distance from {dep} at start: {expected_dep_sep_km:.6f} km")
    print(f"  spacecraft distance from {arr} at end  : {expected_arr_sep_km:.6f} km")
    print("")
    
    # ---------------------------------------------------------
    # OTIMIZAÇÃO: Chamar o C++ apenas 4 vezes FORA do loop!
    # ---------------------------------------------------------
    print("Iniciando extração do Principia C++ (isso levará ~2 segundos)...")
    
    origin_start_raw_r_m, _ = sample_raw_body_state(
        sampler=args.sampler, plugin_b64=args.plugin_b64, target_body=args.raw_origin_body,
        sampler_central_body=args.raw_origin_body, et_s=t_start,
        plugin_base_et_s=args.plugin_base_et_s, work_dir=args.raw_cache_dir,
    )
    origin_end_raw_r_m, _ = sample_raw_body_state(
        sampler=args.sampler, plugin_b64=args.plugin_b64, target_body=args.raw_origin_body,
        sampler_central_body=args.raw_origin_body, et_s=t_end,
        plugin_base_et_s=args.plugin_base_et_s, work_dir=args.raw_cache_dir,
    )
    dep_raw_r_m, _ = sample_raw_body_state(
        sampler=args.sampler, plugin_b64=args.plugin_b64, target_body=dep,
        sampler_central_body=args.raw_origin_body, et_s=t_start,
        plugin_base_et_s=args.plugin_base_et_s, work_dir=args.raw_cache_dir,
    )
    arr_raw_r_m, _ = sample_raw_body_state(
        sampler=args.sampler, plugin_b64=args.plugin_b64, target_body=arr,
        sampler_central_body=args.raw_origin_body, et_s=t_end,
        plugin_base_et_s=args.plugin_base_et_s, work_dir=args.raw_cache_dir,
    )
    print("Extração concluída. Calculando as 48 matrizes de transformação...\n")
    # ---------------------------------------------------------

    dep_levela_rel_m = st_dep_start[:3] * 1000.0
    arr_levela_rel_m = st_arr_end[:3] * 1000.0

    rows = []

    # O loop agora é puramente matemático e roda na velocidade da luz
    for tr_spec in all_transforms():
        tr = parse_transform(tr_spec)

        # Start Body Mapping Error
        predicted_dep_raw_m = origin_start_raw_r_m + apply_transform(dep_levela_rel_m, tr)
        start_body_map_m = norm(predicted_dep_raw_m - dep_raw_r_m)

        # End Body Mapping Error
        predicted_arr_raw_m = origin_end_raw_r_m + apply_transform(arr_levela_rel_m, tr)
        end_body_map_m = norm(predicted_arr_raw_m - arr_raw_r_m)

        # Verify spacecraft geometry in raw space
        sc_start_raw_m = origin_start_raw_r_m + apply_transform(r_sc_start_levela_km * 1000.0, tr)
        sc_end_raw_m = origin_end_raw_r_m + apply_transform(r_sc_end_levela_km * 1000.0, tr)

        dep_sep_raw_km = norm(sc_start_raw_m - dep_raw_r_m) / 1000.0
        arr_sep_raw_km = norm(sc_end_raw_m - arr_raw_r_m) / 1000.0

        dep_sep_err_km = abs(dep_sep_raw_km - expected_dep_sep_km)
        arr_sep_err_km = abs(arr_sep_raw_km - expected_arr_sep_km)

        score_m = start_body_map_m + end_body_map_m

        rows.append({
            "transform": tr_spec,
            "start_body_map_m": start_body_map_m,
            "end_body_map_m": end_body_map_m,
            "score_m": score_m,
            "dep_sep_raw_km": dep_sep_raw_km,
            "arr_sep_raw_km": arr_sep_raw_km,
            "dep_sep_err_km": dep_sep_err_km,
            "arr_sep_err_km": arr_sep_err_km,
        })

    rows.sort(key=lambda r: r["score_m"])

    print("=== BEST TRANSFORMS BY BODY MAPPING ERROR ===")
    print("rank transform     start_body_m    end_body_m      score_m      dep_sep_km     arr_sep_km")
    for i, r in enumerate(rows[: args.top], start=1):
        print(
            f"{i:>3} {r['transform']:<12} "
            f"{r['start_body_map_m']:14.6f} "
            f"{r['end_body_map_m']:14.6f} "
            f"{r['score_m']:14.6f} "
            f"{r['dep_sep_raw_km']:14.6f} "
            f"{r['arr_sep_raw_km']:14.6f}"
        )

    print("")
    for tr_name in ["+Z,-X,+Y", "-Y,+Z,+X", "+X,+Y,+Z"]:
        try:
            found = next(r for r in rows if r["transform"] == tr_name)
            print(f"--- {tr_name} ---")
            print(f"start body mapping error: {found['start_body_map_m']:.6f} m")
            print(f"end body mapping error  : {found['end_body_map_m']:.6f} m")
            print(f"dep sep raw             : {found['dep_sep_raw_km']:.6f} km")
            print(f"arr sep raw             : {found['arr_sep_raw_km']:.6f} km")
        except StopIteration:
            pass

    return 0

if __name__ == "__main__":
    raise SystemExit(main())