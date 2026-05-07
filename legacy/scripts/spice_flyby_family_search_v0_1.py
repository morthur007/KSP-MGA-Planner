#!/usr/bin/env python3
"""
spice_flyby_family_search_v0_1_mp.py

Sequence-family search wrapper for fast in-game MGA candidate discovery.
NOW WITH MULTIPROCESSING: Runs multiple sequence families in parallel using ProcessPoolExecutor.

Example:

python spice_flyby_family_search_v0_1.py \
  --beam-script spice_lambert_beam_search_v0_2.py \
  --bsp data/spice_jnsq_v0_3/jnsq_principia_native_v0_3_1000y_1h_deg9.bsp \
  --tpc data/spice_jnsq_v0_3/jnsq_principia_native_v0_3_1000y_1h_deg9.ids.tpc \
  --metadata data/spice_jnsq_v0_3/jnsq_principia_native_v0_3_1000y_1h_deg9.metadata.json \
  --body-catalog data/jnsq_gate0/ksp_future_paused/body_catalog.json \
  --central-body Sun \
  --origin Kerbin \
  --target Jool \
  --allowed-flybys Kerbin Eve \
  --max-flybys 3 \
  --start-et 81.85168640136972 \
  --search-years 30 \
  --t0-step-days 20 \
  --beam-width 3000 \
  --per-sequence-top-n 80 \
  --top-n 200 \
  --max-revs 1 \
  --max-departure-vinf-km-s 8 \
  --max-arrival-vinf-km-s 10 \
  --max-powered-flyby-dv-km-s 1.0 \
  --max-turn-excess-deg 0 \
  --min-turn-margin-deg 1 \
  --powered-weight 150 \
  --turn-weight 300 \
  --work-dir data/mga_smoke/family_search_kj_v0_1 \
  --output data/mga_smoke/family_search_kj_v0_1/merged_candidates.csv \
  --jobs 8
"""

from __future__ import annotations

import argparse
import csv
import itertools
import math
import subprocess
import sys
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


def norm_name(s: str) -> str:
    return s.strip().upper()


def sequence_label(seq: Sequence[str]) -> str:
    return "-".join(norm_name(x) for x in seq)


def enumerate_sequences(origin: str, target: str, allowed: Sequence[str], max_flybys: int) -> List[List[str]]:
    origin = norm_name(origin)
    target = norm_name(target)
    allowed = [norm_name(x) for x in allowed]

    sequences: List[List[str]] = []
    for n in range(max_flybys + 1):
        for flybys in itertools.product(allowed, repeat=n):
            seq = [origin, *flybys, target]

            # Avoid completely degenerate immediate final target repeats
            bad = False
            for a, b in zip(seq[:-1], seq[1:]):
                if a == b and a != origin:
                    bad = True
                    break
            if bad:
                continue

            sequences.append(seq)

    # Stable order: shorter first, then lexical.
    sequences.sort(key=lambda s: (len(s), sequence_label(s)))
    return sequences


def leg_grid(dep: str, arr: str) -> Tuple[float, float, float, str | None]:
    dep = norm_name(dep)
    arr = norm_name(arr)

    if dep == arr:
        return 250.0, 1300.0, 20.0, "292 365 438 584 730 876 1095"
    if {dep, arr} == {"KERBIN", "EVE"}:
        return 90.0, 800.0, 20.0, None
    if arr == "JOOL":
        return 500.0, 5000.0, 40.0, None
    if dep == "JOOL":
        return 500.0, 5000.0, 40.0, None
    return 120.0, 2500.0, 40.0, None


def build_beam_command(args: argparse.Namespace, seq: Sequence[str], output_csv: Path) -> List[str]:
    mins: List[str] = []
    maxs: List[str] = []
    steps: List[str] = []
    explicit_by_leg: Dict[int, str] = {}

    for i, (dep, arr) in enumerate(zip(seq[:-1], seq[1:]), start=1):
        mn, mx, st, explicit = leg_grid(dep, arr)
        mins.append(str(mn))
        maxs.append(str(mx))
        steps.append(str(st))
        if explicit:
            explicit_by_leg[i] = explicit

    cmd = [
        sys.executable,
        str(args.beam_script),
        "--bsp", str(args.bsp),
        "--tpc", str(args.tpc),
        "--central-body", args.central_body,
        "--sequence", *seq,
        "--start-et", str(args.start_et),
        "--search-years", str(args.search_years),
        "--t0-step-days", str(args.t0_step_days),
        "--tof-min-days", *mins,
        "--tof-max-days", *maxs,
        "--tof-step-days", *steps,
        "--beam-width", str(args.beam_width),
        "--top-n", str(args.per_sequence_top_n),
        "--max-departure-vinf-km-s", str(args.max_departure_vinf_km_s),
        "--max-arrival-vinf-km-s", str(args.max_arrival_vinf_km_s),
        "--max-powered-flyby-dv-km-s", str(args.max_powered_flyby_dv_km_s),
        "--max-turn-excess-deg", str(args.max_turn_excess_deg),
        "--min-turn-margin-deg", str(args.min_turn_margin_deg),
        "--powered-weight", str(args.powered_weight),
        "--turn-weight", str(args.turn_weight),
        "--max-revs", str(args.max_revs),
        "--output", str(output_csv),
    ]

    for p in args.metadata:
        cmd.extend(["--metadata", str(p)])
    for p in args.body_catalog:
        cmd.extend(["--body-catalog", str(p)])
    for leg_idx, values in explicit_by_leg.items():
        cmd.extend([f"--tof-values-days-{leg_idx}", values])

    return cmd


def read_rows(path: Path, seq: Sequence[str]) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    label = sequence_label(seq)
    for r in rows:
        r["sequence"] = label
        r["sequence_bodies"] = " ".join(seq)
        r["sequence_len"] = str(len(seq))
    return rows


def as_float(row: Dict[str, str], key: str, default: float = float("inf")) -> float:
    try:
        v = row.get(key, "")
        if v == "" or v is None:
            return default
        return float(v)
    except Exception:
        return default


def sort_key(row: Dict[str, str], mode: str) -> Tuple[float, ...]:
    raw = as_float(row, "raw_sum_km_s")
    dep = as_float(row, "departure_vinf_km_s")
    arr = as_float(row, "arrival_vinf_km_s")
    powered = as_float(row, "powered_flyby_dv_km_s")
    margin = as_float(row, "min_turn_margin_deg", default=-float("inf"))
    tof = as_float(row, "tof_total_days")
    cost = as_float(row, "cost")

    if mode == "raw_sum":
        return (raw, powered, dep, arr, tof)
    if mode == "departure":
        return (dep, powered, arr, raw, tof)
    if mode == "arrival":
        return (arr, powered, dep, raw, tof)
    if mode == "powered":
        return (powered, raw, dep, arr, tof)
    if mode == "duration":
        return (tof, raw, powered, dep, arr)
    if mode == "margin":
        return (-margin, powered, raw, dep, arr)
    return (cost, raw, powered, dep, arr, tof)


def write_merged(path: Path, rows: List[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: List[str] = []
    preferred = [
        "sequence", "sequence_bodies", "cost", "raw_sum_km_s",
        "departure_vinf_km_s", "arrival_vinf_km_s", "powered_flyby_dv_km_s",
        "turn_excess_deg", "min_turn_margin_deg", "tof_total_days",
        "epochs_et_s", "tofs_days", "leg_paths",
    ]
    for k in preferred:
        if any(k in r for r in rows) and k not in fields:
            fields.append(k)
    for r in rows:
        for k in r.keys():
            if k not in fields:
                fields.append(k)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


# --- Worker Process Function ---
def evaluate_sequence(args_tuple):
    idx, total_seqs, seq, args = args_tuple
    label = sequence_label(seq)
    out_csv = args.work_dir / label / "candidates.csv"
    log_txt = args.work_dir / label / "run.log"
    cmd = build_beam_command(args, seq, out_csv)

    if args.dry_run:
        return idx, total_seqs, label, "dry-run", []

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with log_txt.open("w", encoding="utf-8") as log:
        log.write("$ " + " ".join(cmd) + "\n\n")
        log.flush()
        proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, text=True)

    if proc.returncode != 0:
        return idx, total_seqs, label, f"failed rc={proc.returncode}", []

    rows = read_rows(out_csv, seq)
    return idx, total_seqs, label, "ok", rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--beam-script", type=Path, default=Path("spice_lambert_beam_search_v0_2.py"))
    p.add_argument("--bsp", type=Path, required=True)
    p.add_argument("--tpc", type=Path, required=True)
    p.add_argument("--metadata", type=Path, action="append", default=[])
    p.add_argument("--body-catalog", type=Path, action="append", default=[])
    p.add_argument("--central-body", default="Sun")
    p.add_argument("--origin", required=True)
    p.add_argument("--target", required=True)
    p.add_argument("--allowed-flybys", nargs="+", required=True)
    p.add_argument("--max-flybys", type=int, default=3)
    p.add_argument("--start-et", type=float, required=True)
    p.add_argument("--search-years", type=float, default=30.0)
    p.add_argument("--t0-step-days", type=float, default=20.0)
    p.add_argument("--beam-width", type=int, default=3000)
    p.add_argument("--per-sequence-top-n", type=int, default=80)
    p.add_argument("--top-n", type=int, default=200)
    p.add_argument("--max-revs", type=int, default=1)
    p.add_argument("--max-departure-vinf-km-s", type=float, default=8.0)
    p.add_argument("--max-arrival-vinf-km-s", type=float, default=10.0)
    p.add_argument("--max-powered-flyby-dv-km-s", type=float, default=1.0)
    p.add_argument("--max-turn-excess-deg", type=float, default=0.0)
    p.add_argument("--min-turn-margin-deg", type=float, default=1.0)
    p.add_argument("--powered-weight", type=float, default=150.0)
    p.add_argument("--turn-weight", type=float, default=300.0)
    p.add_argument("--sort-by", choices=["cost", "raw_sum", "departure", "arrival", "powered", "duration", "margin"], default="raw_sum")
    p.add_argument("--work-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--jobs", type=int, default=multiprocessing.cpu_count(), help="Número de subprocessos em paralelo")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    seqs = enumerate_sequences(args.origin, args.target, args.allowed_flybys, args.max_flybys)
    args.work_dir.mkdir(parents=True, exist_ok=True)

    print("=== SPICE FLYBY FAMILY SEARCH V0.1 (MULTIPROCESSING) ===")
    print(f"origin={norm_name(args.origin)} target={norm_name(args.target)} allowed={','.join(norm_name(x) for x in args.allowed_flybys)} max_flybys={args.max_flybys}")
    print(f"sequences={len(seqs)} sort_by={args.sort_by} workers={args.jobs}")

    all_rows: List[Dict[str, str]] = []
    
    # Prepara os argumentos para o Worker
    tasks = [(idx, len(seqs), seq, args) for idx, seq in enumerate(seqs, start=1)]

    # Pool de processos paralelos
    with ProcessPoolExecutor(max_workers=args.jobs) as executor:
        futures = [executor.submit(evaluate_sequence, t) for t in tasks]
        
        for future in as_completed(futures):
            idx, total_seqs, label, status, rows = future.result()
            if status == "ok":
                print(f"[{idx:02d}/{total_seqs:02d}] {label} | kept={len(rows)}")
                all_rows.extend(rows)
            elif status == "dry-run":
                print(f"[{idx:02d}/{total_seqs:02d}] {label} | dry-run")
            else:
                print(f"[{idx:02d}/{total_seqs:02d}] {label} | [WARN] {status}")

    if args.dry_run:
        return 0

    all_rows.sort(key=lambda r: sort_key(r, args.sort_by))
    final_rows = all_rows[: args.top_n]
    write_merged(args.output, final_rows)

    print("\n=== TOP FAMILY CANDIDATES ===")
    print("rank seq raw_sum dep arr powered tof margin paths")
    for i, r in enumerate(final_rows[:30], start=1):
        print(
            f"{i:>3} "
            f"{r.get('sequence',''):<30} "
            f"{as_float(r,'raw_sum_km_s'):8.4f} "
            f"{as_float(r,'departure_vinf_km_s'):7.3f} "
            f"{as_float(r,'arrival_vinf_km_s'):7.3f} "
            f"{as_float(r,'powered_flyby_dv_km_s'):7.3f} "
            f"{as_float(r,'tof_total_days'):8.1f} "
            f"{as_float(r,'min_turn_margin_deg', -999):7.3f} "
            f"{r.get('leg_paths','')}"
        )
    print(f"\n[OK] wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())