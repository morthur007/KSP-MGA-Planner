#!/usr/bin/env python3
"""
mga_parallel_runner_v0_1.py

Process-level parallel runner for spice_lambert_mga_v0_1.py.

Why this exists:
  - SPICE has process-global kernel state.
  - PyGMO populations are stochastic and benefit from many independent seeds.
  - Running one OS process per seed is safer than multiprocessing inside fitness().

This wrapper:
  - launches N independent spice_lambert_mga_v0_1.py runs concurrently;
  - assigns different seeds;
  - writes one CSV/log per run;
  - merges all candidates into one ranked CSV by `cost`;
  - prints the global best candidate.

Example:
  python mga_parallel_runner_v0_1.py \
    --script spice_lambert_mga_v0_1.py \
    --workers 8 \
    --runs 32 \
    --base-seed 1000 \
    --output-dir data/mga_smoke/parallel_kekj_w25 \
    --merged-output data/mga_smoke/kekj_lambert_w25_parallel_merged.csv \
    -- \
    --bsp data/spice_jnsq_v0_3/jnsq_principia_native_v0_3_1000y_1h_deg9.bsp \
    --tpc data/spice_jnsq_v0_3/jnsq_principia_native_v0_3_1000y_1h_deg9.ids.tpc \
    --metadata data/spice_jnsq_v0_3/jnsq_principia_native_v0_3_1000y_1h_deg9.metadata.json \
    --body-catalog data/jnsq_gate0/ksp_future_paused/body_catalog.json \
    --central-body Sun \
    --sequence Kerbin Eve Kerbin Jool \
    --start-et 81.85168640136972 \
    --search-years 20 \
    --tof-min-days 30 30 300 \
    --tof-max-days 500 800 4000 \
    --powered-flyby-weight 25 \
    --pop 256 \
    --gen 700
"""

from __future__ import annotations

import argparse
import csv
import os
import queue
import shlex
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class RunSpec:
    run_index: int
    seed: int
    csv_path: Path
    log_path: Path
    cmd: List[str]


@dataclass
class RunResult:
    spec: RunSpec
    returncode: int
    error: Optional[str] = None


def split_wrapper_args(argv: Sequence[str]) -> Tuple[List[str], List[str]]:
    if "--" not in argv:
        return list(argv), []
    idx = list(argv).index("--")
    return list(argv[:idx]), list(argv[idx + 1 :])


def parse_args(argv: Optional[Sequence[str]] = None) -> Tuple[argparse.Namespace, List[str]]:
    wrapper_argv, passthrough = split_wrapper_args(sys.argv[1:] if argv is None else argv)

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--script", type=Path, default=Path("spice_lambert_mga_v0_1.py"))
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--workers", type=int, default=max(1, os.cpu_count() or 1))
    p.add_argument("--runs", type=int, default=8, help="Número total de runs independentes/seeds.")
    p.add_argument("--base-seed", type=int, default=1000)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--merged-output", type=Path, required=True)
    p.add_argument("--top", type=int, default=200, help="Top N linhas mantidas no CSV merged.")
    p.add_argument("--fail-fast", action="store_true")
    p.add_argument("--dry-run", action="store_true")

    args = p.parse_args(wrapper_argv)
    if args.workers < 1:
        raise SystemExit("--workers precisa ser >= 1")
    if args.runs < 1:
        raise SystemExit("--runs precisa ser >= 1")
    if not passthrough:
        raise SystemExit("Passe os argumentos do spice_lambert_mga_v0_1.py após um separador --")

    return args, passthrough


def strip_output_and_seed(passthrough: Sequence[str]) -> List[str]:
    """Remove --output/--seed from passthrough so each worker can override them."""
    out: List[str] = []
    i = 0
    while i < len(passthrough):
        tok = passthrough[i]
        if tok in {"--output", "--seed"}:
            i += 2
            continue
        if tok.startswith("--output=") or tok.startswith("--seed="):
            i += 1
            continue
        out.append(tok)
        i += 1
    return out


def build_specs(args: argparse.Namespace, passthrough: Sequence[str]) -> List[RunSpec]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    clean = strip_output_and_seed(passthrough)
    specs: List[RunSpec] = []

    for i in range(args.runs):
        seed = args.base_seed + i
        csv_path = args.output_dir / f"run_{i:03d}_seed_{seed}.csv"
        log_path = args.output_dir / f"run_{i:03d}_seed_{seed}.log"
        cmd = [
            str(args.python),
            str(args.script),
            *clean,
            "--seed",
            str(seed),
            "--output",
            str(csv_path),
        ]
        specs.append(RunSpec(i, seed, csv_path, log_path, cmd))

    return specs


def run_one(spec: RunSpec) -> RunResult:
    with spec.log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + " ".join(shlex.quote(x) for x in spec.cmd) + "\n\n")
        log.flush()
        try:
            proc = subprocess.run(
                spec.cmd,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
            return RunResult(spec=spec, returncode=proc.returncode)
        except Exception as exc:
            log.write(f"\n[WRAPPER ERROR] {exc}\n")
            return RunResult(spec=spec, returncode=999, error=str(exc))


def worker_loop(
    name: str,
    q: "queue.Queue[RunSpec]",
    results: List[RunResult],
    lock: threading.Lock,
    stop_event: threading.Event,
) -> None:
    while not stop_event.is_set():
        try:
            spec = q.get_nowait()
        except queue.Empty:
            return

        print(f"[{name}] start run={spec.run_index} seed={spec.seed}")
        result = run_one(spec)
        with lock:
            results.append(result)
        if result.returncode == 0:
            print(f"[{name}] ok    run={spec.run_index} seed={spec.seed} -> {spec.csv_path}")
        else:
            print(f"[{name}] FAIL  run={spec.run_index} seed={spec.seed} rc={result.returncode} log={spec.log_path}")
        q.task_done()


def read_candidate_csv(path: Path, source_run: int, source_seed: int) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        r = csv.DictReader(f)
        rows: List[Dict[str, str]] = []
        for row in r:
            row = dict(row)
            row["source_run"] = str(source_run)
            row["source_seed"] = str(source_seed)
            rows.append(row)
        return rows


def safe_float(row: Dict[str, str], key: str, default: float = float("inf")) -> float:
    try:
        return float(row.get(key, ""))
    except Exception:
        return default


def merge_results(results: Sequence[RunResult], merged_output: Path, top: int) -> List[Dict[str, str]]:
    all_rows: List[Dict[str, str]] = []
    for result in results:
        if result.returncode != 0:
            continue
        all_rows.extend(
            read_candidate_csv(
                result.spec.csv_path,
                source_run=result.spec.run_index,
                source_seed=result.spec.seed,
            )
        )

    all_rows.sort(key=lambda row: safe_float(row, "cost"))
    if top > 0:
        all_rows = all_rows[:top]

    merged_output.parent.mkdir(parents=True, exist_ok=True)
    if not all_rows:
        merged_output.write_text("", encoding="utf-8")
        return []

    # Preserve common columns first; append any extras.
    preferred = [
        "global_rank",
        "source_run",
        "source_seed",
        "rank",
        "status",
        "cost",
        "departure_vinf_km_s",
        "arrival_vinf_km_s",
        "powered_flyby_dv_km_s",
        "tof_total_days",
        "turn_excess_deg",
        "max_turn_required_deg",
        "min_turn_margin_deg",
        "t0_et_s",
    ]

    keys = set()
    for row in all_rows:
        keys.update(row.keys())
    fieldnames = [k for k in preferred if k == "global_rank" or k in keys]
    fieldnames += sorted(k for k in keys if k not in set(fieldnames))

    with merged_output.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for i, row in enumerate(all_rows, start=1):
            out = dict(row)
            out["global_rank"] = i
            w.writerow(out)

    return all_rows


def main() -> int:
    args, passthrough = parse_args()
    specs = build_specs(args, passthrough)

    if args.dry_run:
        for spec in specs:
            print(" ".join(shlex.quote(x) for x in spec.cmd))
        return 0

    print(f"[INFO] workers={args.workers} runs={args.runs} output_dir={args.output_dir}")

    q: "queue.Queue[RunSpec]" = queue.Queue()
    for spec in specs:
        q.put(spec)

    results: List[RunResult] = []
    lock = threading.Lock()
    stop_event = threading.Event()

    threads = []
    for i in range(min(args.workers, args.runs)):
        t = threading.Thread(
            target=worker_loop,
            args=(f"w{i:02d}", q, results, lock, stop_event),
            daemon=True,
        )
        threads.append(t)
        t.start()

    try:
        for t in threads:
            t.join()
    except KeyboardInterrupt:
        stop_event.set()
        raise

    failed = [r for r in results if r.returncode != 0]
    if failed:
        print(f"[WARN] failed runs: {len(failed)}/{len(results)}")
        for r in failed[:10]:
            print(f"  run={r.spec.run_index} seed={r.spec.seed} rc={r.returncode} log={r.spec.log_path}")
        if args.fail_fast:
            return 2

    merged = merge_results(results, args.merged_output, args.top)
    print(f"[OK] merged: {args.merged_output} rows={len(merged)}")

    if merged:
        champ = merged[0]
        print("\n=== GLOBAL BEST ===")
        for key in [
            "cost",
            "departure_vinf_km_s",
            "arrival_vinf_km_s",
            "powered_flyby_dv_km_s",
            "tof_total_days",
            "turn_excess_deg",
            "min_turn_margin_deg",
            "source_seed",
            "source_run",
        ]:
            if key in champ:
                print(f"{key:<24}: {champ[key]}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
