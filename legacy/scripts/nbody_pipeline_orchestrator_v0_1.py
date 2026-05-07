#!/usr/bin/env python3
"""
nbody_pipeline_orchestrator_v0_3_mp.py

Automated N-Body Triage Pipeline (MULTIPROCESSING).
Evaluates multiple candidate ranks concurrently using ProcessPoolExecutor.
Now correctly propagates ALL required arguments (including plugin-b64) to the audit script.
"""

import argparse
import csv
import json
import subprocess
import sys
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

def get_row(csv_path: Path, rank: int) -> dict:
    with csv_path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    return rows[rank - 1]

def run_cmd_silent(cmd: list, log_path: Path) -> bool:
    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + " ".join(str(x) for x in cmd) + "\n\n")
        proc = subprocess.run([str(x) for x in cmd], stdout=log, stderr=subprocess.STDOUT, text=True)

    if proc.returncode != 0:
        print(f"\n[ERROR] rc={proc.returncode} log={log_path}")
        try:
            lines = log_path.read_text(errors="replace").splitlines()
            for line in lines[-50:]:
                print("    " + line)
        except Exception as e:
            print(f"    could not read log: {e}")
        return False

    return True

def triage_candidate(rank: int, args: argparse.Namespace, catalog_args: list) -> tuple:
    """Função Worker executada em paralelo para um Rank específico."""
    try:
        row = get_row(args.candidates, rank)
    except IndexError:
        return rank, "UNKNOWN", False, "NOT FOUND"
        
    seq_bodies = row["sequence_bodies"].split()
    label = row["sequence"]
    
    rank_dir = args.work_dir / f"rank{rank}_{label}"
    rank_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. OPTIMIZE LEGS
    legs_csv = rank_dir / "leg_optimizations.csv"
    cmd_legs = [
        sys.executable, "native_optimize_candidate_legs_v0_1.py",
        "--candidate-csv", args.candidates,
        "--rank", rank,
        "--bsp", args.bsp, "--tpc", args.tpc,
        "--plugin-b64", args.plugin_b64,
        "--central-body", args.central_body,
        *catalog_args,
        "--sequence", *seq_bodies,
        "--transform", args.transform,
        "--work-dir", rank_dir / "legs",
        "--output-csv", legs_csv
    ]
    if not run_cmd_silent(cmd_legs, rank_dir / "1_optimize_legs.log"):
        return rank, label, False, "Leg optimization failed"
        
    # 2. FLYBY AUDIT
    audit_json = rank_dir / "flyby_audit.json"
    audit_csv = rank_dir / "flyby_audit.csv"
    cmd_audit = [
        sys.executable, "native_corrected_flyby_audit_v0_1.py",
        "--candidate-csv", args.candidates,
        "--rank", rank,
        "--leg-summary-csv", legs_csv,
        "--bsp", args.bsp, "--tpc", args.tpc,
        "--central-body", args.central_body,
        *catalog_args,
        "--sequence", *seq_bodies,
        "--transform", args.transform,
        "--plugin-b64", args.plugin_b64,              # CORREÇÃO: Argumento que estava faltando!
        "--work-dir", rank_dir / "audit_work",        # CORREÇÃO: Argumento que estava faltando!
        "--output-csv", audit_csv,                    # CORREÇÃO: Argumento que estava faltando!
        "--output-json", audit_json
    ]
    if not run_cmd_silent(cmd_audit, rank_dir / "2_flyby_audit.log"):
        return rank, label, False, "Flyby audit failed"
        
    with audit_json.open() as f:
        audit_data = json.load(f)
        
    max_mismatch = audit_data.get("max_vinf_mismatch_km_s", 999.0)
    
    if max_mismatch > 0.8: # > 800 m/s
        return rank, label, False, f"Mismatch too high ({max_mismatch * 1000:.1f} m/s)"
        
    # 3. POWERED BRIDGE
    if max_mismatch >= 0.01: # > 10 m/s exige correção ativa
        for i, flyby in enumerate(audit_data.get("flybys", [])):
            if flyby["vinf_mismatch_km_s"] >= 0.01:
                # CORREÇÃO: Busca por event_index ou index, ou usa o contador como fallback
                fb_index = flyby.get("event_index", flyby.get("index", i + 1))
                fb_body = flyby["body"]
                
                cmd_bridge = [
                    sys.executable, "native_powered_flyby_bridge_v0_1.py",
                    "--candidate-csv", args.candidates,
                    "--rank", rank,
                    "--leg-summary-csv", legs_csv,
                    "--flyby-index", fb_index,
                    "--bsp", args.bsp, "--tpc", args.tpc,
                    "--plugin-b64", args.plugin_b64,
                    "--central-body", args.central_body,
                    *catalog_args,
                    "--sequence", *seq_bodies,
                    "--transform", args.transform,
                    "--validator", "principia_impulsive_particle_server",
                    "--work-dir", rank_dir / f"bridge_fb{fb_index}",
                    "--output-json", rank_dir / f"bridge_fb{fb_index}_result.json",
                    "--output-history-csv", rank_dir / f"bridge_fb{fb_index}_history.csv"
                ]
                if not run_cmd_silent(cmd_bridge, rank_dir / f"3_bridge_fb{fb_index}.log"):
                    return rank, label, False, f"Bridge failed on {fb_body}"
    
    return rank, label, True, f"N-Body Validated! (Max mismatch: {max_mismatch * 1000:.1f} m/s)"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--candidates", type=Path, required=True)
    p.add_argument("--ranks", type=int, nargs="+", required=True)
    p.add_argument("--bsp", type=Path, required=True)
    p.add_argument("--tpc", type=Path, required=True)
    p.add_argument("--plugin-b64", type=Path, required=True)
    p.add_argument("--metadata", type=Path, action="append", default=[])
    p.add_argument("--body-catalog", type=Path, action="append", default=[])
    p.add_argument("--central-body", default="Sun")
    p.add_argument("--transform", default="+Z,-X,+Y")
    p.add_argument("--work-dir", type=Path, required=True)
    p.add_argument("--jobs", type=int, default=multiprocessing.cpu_count())
    return p.parse_args()

def main():
    args = parse_args()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    
    print("=== N-BODY PIPELINE ORCHESTRATOR V0.3 (MULTIPROCESSING) ===")
    print(f"Candidates: {args.candidates}")
    print(f"Testing Ranks: {args.ranks}")
    print(f"Workers: {args.jobs}")
    print(f"Transform: {args.transform}")
    print("-" * 60)
    
    catalog_args = []
    for cat in args.metadata:
        catalog_args.extend(["--metadata", str(cat)])
    for cat in args.body_catalog:
        catalog_args.extend(["--body-catalog", str(cat)])
        
    success_count = 0

    # Lança os trabalhadores em paralelo
    with ProcessPoolExecutor(max_workers=args.jobs) as executor:
        futures = {executor.submit(triage_candidate, rank, args, catalog_args): rank for rank in args.ranks}
        
        for future in as_completed(futures):
            rank, label, success, message = future.result()
            
            if success:
                print(f"[RANK {rank:03d}] {label:<30} | [SUCCESS] {message}")
                success_count += 1
            else:
                print(f"[RANK {rank:03d}] {label:<30} | [FAIL]    {message}")

    print("-" * 60)
    print(f"=== TRIAGE COMPLETE | Validated {success_count}/{len(args.ranks)} candidates ===")

if __name__ == "__main__":
    main()