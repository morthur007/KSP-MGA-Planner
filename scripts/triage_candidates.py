#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures as cf
import csv
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


def read_candidates(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def safe_float(x: Any, default: float = math.inf) -> float:
    try:
        if x in ("", None):
            return default
        v = float(x)
        if math.isnan(v):
            return default
        return v
    except Exception:
        return default


def status_score(status: str) -> int:
    s = (status or "").upper()
    if s == "PASS":
        return 0
    if s == "POWERED":
        return 1
    if s in ("CHECK", "CHECK_POWERED"):
        return 2
    if s.startswith("FAIL"):
        return 4
    return 3


def run_cmd(cmd: list[str], log_path: Path) -> tuple[int, str]:
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with log_path.open("w") as log:
        log.write("$ " + " ".join(cmd) + "\n\n")
        log.flush()

        proc = subprocess.run(
            cmd,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )

    return proc.returncode, str(log_path)


def summarize_leg_optimizations(path: Path) -> dict[str, Any]:
    rows = list(csv.DictReader(path.open()))

    total_leg_dv_m_s = 0.0
    max_leg_miss_km = 0.0
    max_leg_relv_m_s = 0.0
    all_success = True

    for r in rows:
        total_leg_dv_m_s += safe_float(r.get("dv_norm_m_s"), 0.0)
        max_leg_miss_km = max(max_leg_miss_km, safe_float(r.get("final_miss_km"), math.inf))
        max_leg_relv_m_s = max(max_leg_relv_m_s, safe_float(r.get("final_relv_m_s"), math.inf))

        if str(r.get("solver_success", "")).lower() != "true":
            all_success = False

    return {
        "n_legs_optimized": len(rows),
        "all_legs_success": all_success,
        "total_leg_correction_m_s": total_leg_dv_m_s,
        "max_leg_miss_km": max_leg_miss_km,
        "max_leg_relv_m_s": max_leg_relv_m_s,
    }


def summarize_audit(path: Path) -> dict[str, Any]:
    obj = json.loads(path.read_text())
    summary = obj.get("summary", {})
    flybys = obj.get("flybys", [])

    min_alt_req = math.inf
    min_turn_margin = safe_float(summary.get("min_turn_margin_deg"), math.inf)
    max_vinf_mismatch = safe_float(summary.get("max_vinf_mismatch_km_s"), math.inf)
    total_powered = safe_float(summary.get("total_powered_dv_lower_bound_km_s"), math.inf)
    overall_status = str(summary.get("overall_status", "UNKNOWN"))

    for f in flybys:
        min_alt_req = min(min_alt_req, safe_float(f.get("alt_required_km"), math.inf))

    return {
        "overall_status": overall_status,
        "n_flybys": int(summary.get("n_flybys", len(flybys))),
        "total_powered_dv_lower_bound_m_s": total_powered * 1000.0,
        "max_vinf_mismatch_m_s": max_vinf_mismatch * 1000.0,
        "min_turn_margin_deg": min_turn_margin,
        "min_alt_required_km": min_alt_req,
    }


def run_one_rank(rank: int, args: argparse.Namespace, candidate_row: dict[str, str]) -> dict[str, Any]:
    rank_dir = args.work_dir / f"rank_{rank:05d}"
    opt_dir = rank_dir / "leg_opt"
    audit_dir = rank_dir / "flyby_audit"
    logs_dir = rank_dir / "logs"

    opt_csv = opt_dir / "leg_optimizations.csv"
    audit_csv = audit_dir / "flyby_audit.csv"
    audit_json = audit_dir / "flyby_audit.json"

    result: dict[str, Any] = {
        "rank": rank,
        "candidate_id": candidate_row.get("candidate_id", f"rank_{rank}"),
        "sequence_bodies": candidate_row.get("sequence_bodies", ""),
        "n_legs": candidate_row.get("n_legs", ""),
        "status": "UNKNOWN",
        "opt_log": str(logs_dir / "optimize.log"),
        "audit_log": str(logs_dir / "audit.log"),
        "opt_csv": str(opt_csv),
        "audit_csv": str(audit_csv),
        "audit_json": str(audit_json),
    }

    if not args.reuse_existing or not opt_csv.exists():
        opt_cmd = [
            sys.executable,
            "scripts/optimize_candidate_legs.py",
            "--candidate-seed", str(args.candidate_seed),
            "--rank", str(rank),
            "--bsp", str(args.bsp),
            "--tpc", str(args.tpc),
            "--plugin-b64", str(args.plugin_b64),
            "--central-body", args.central_body,
            "--central-mu-km3-s2", str(args.central_mu_km3_s2),
            "--transform", args.transform,
            "--buffer-days", str(args.buffer_days),
            "--max-nfev", str(args.max_nfev),
            "--pos-scale-km", str(args.pos_scale_km),
            "--dv-x-scale-m-s", str(args.dv_x_scale_m_s),
            "--work-dir", str(opt_dir),
            "--output-csv", str(opt_csv),
        ]

        if args.raw_cache_dir:
            opt_cmd += ["--raw-cache-dir", str(args.raw_cache_dir)]

        rc, log = run_cmd(opt_cmd, logs_dir / "optimize.log")
        if rc != 0 or not opt_csv.exists():
            result.update({
                "status": "FAIL_OPT",
                "error": f"optimize rc={rc}",
            })
            return result

    leg_summary = summarize_leg_optimizations(opt_csv)
    result.update(leg_summary)

    if not leg_summary["all_legs_success"]:
        result.update({
            "status": "FAIL_LEG_OPT",
            "error": "one or more legs failed",
        })
        return result

    if not args.reuse_existing or not audit_json.exists():
        audit_cmd = [
            sys.executable,
            "scripts/audit_corrected_flybys.py",
            "--leg-optimizations", str(opt_csv),
            "--body-catalog", str(args.body_catalog),
            "--plugin-b64", str(args.plugin_b64),
            "--min-altitude-km", str(args.min_altitude_km),
            "--atmosphere-margin-km", str(args.atmosphere_margin_km),
            "--vinf-mismatch-pass-km-s", str(args.vinf_mismatch_pass_km_s),
            "--vinf-mismatch-powered-km-s", str(args.vinf_mismatch_powered_km_s),
            "--output-csv", str(audit_csv),
            "--output-json", str(audit_json),
        ]

        if args.raw_cache_dir:
            audit_cmd += ["--raw-cache-dir", str(args.raw_cache_dir)]

        rc, log = run_cmd(audit_cmd, logs_dir / "audit.log")
        if rc != 0 or not audit_json.exists():
            result.update({
                "status": "FAIL_AUDIT",
                "error": f"audit rc={rc}",
            })
            return result

    audit_summary = summarize_audit(audit_json)
    result.update(audit_summary)

    result["status"] = audit_summary["overall_status"]

    # Ranking helper: lower is better.
    result["sort_status_score"] = status_score(result["status"])
    result["sort_total_dv_m_s"] = (
        safe_float(result.get("total_leg_correction_m_s"), math.inf)
        + safe_float(result.get("total_powered_dv_lower_bound_m_s"), math.inf)
    )

    return result


def write_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    fields: list[str] = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)

    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main() -> int:
    p = argparse.ArgumentParser(description="Run N-body leg optimization + corrected flyby audit for multiple candidates.")
    p.add_argument("--candidate-seed", type=Path, required=True)
    p.add_argument("--ranks", nargs="*", type=int, default=None)
    p.add_argument("--top-n", type=int, default=None)

    p.add_argument("--bsp", type=Path, required=True)
    p.add_argument("--tpc", type=Path, required=True)
    p.add_argument("--body-catalog", type=Path, required=True)
    p.add_argument("--plugin-b64", type=Path, required=True)

    p.add_argument("--central-body", default="Sun")
    p.add_argument("--central-mu-km3-s2", type=float, required=True)
    p.add_argument("--transform", default="+Z,-X,+Y")
    p.add_argument("--buffer-days", type=float, default=0.235)

    p.add_argument("--max-nfev", type=int, default=120)
    p.add_argument("--pos-scale-km", type=float, default=1000.0)
    p.add_argument("--dv-x-scale-m-s", type=float, default=100.0)

    p.add_argument("--min-altitude-km", type=float, default=50.0)
    p.add_argument("--atmosphere-margin-km", type=float, default=10.0)
    p.add_argument("--vinf-mismatch-pass-km-s", type=float, default=0.05)
    p.add_argument("--vinf-mismatch-powered-km-s", type=float, default=2.0)

    p.add_argument("--workers", type=int, default=max(1, min(4, os.cpu_count() or 1)))
    p.add_argument("--work-dir", type=Path, required=True)
    p.add_argument("--output-csv", type=Path, required=True)
    p.add_argument("--raw-cache-dir", type=Path, default=Path("data/runs/frame_debug/raw_cache"))
    p.add_argument("--reuse-existing", action="store_true")

    args = p.parse_args()

    candidates = read_candidates(args.candidate_seed)

    if args.ranks:
        ranks = args.ranks
    elif args.top_n:
        ranks = list(range(1, min(args.top_n, len(candidates)) + 1))
    else:
        raise SystemExit("Use --ranks ... or --top-n N")

    print("=== N-BODY CANDIDATE TRIAGE ===")
    print(f"candidate seed : {args.candidate_seed}")
    print(f"ranks          : {ranks}")
    print(f"workers        : {args.workers}")
    print(f"work dir       : {args.work_dir}")
    print("")

    rows: list[dict[str, Any]] = []

    with cf.ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {
            ex.submit(run_one_rank, rank, args, candidates[rank - 1]): rank
            for rank in ranks
        }

        for fut in cf.as_completed(futs):
            rank = futs[fut]
            try:
                r = fut.result()
            except Exception as e:
                r = {
                    "rank": rank,
                    "candidate_id": candidates[rank - 1].get("candidate_id", f"rank_{rank}"),
                    "sequence_bodies": candidates[rank - 1].get("sequence_bodies", ""),
                    "status": "FAIL_EXCEPTION",
                    "error": repr(e),
                }

            rows.append(r)

            err = r.get("error", "")
            print(
                f"[RANK {rank:05d}] {r.get('sequence_bodies',''):<35} "
                f"{r.get('status','UNKNOWN'):<14} "
                f"leg_dv={safe_float(r.get('total_leg_correction_m_s'), math.inf):9.2f} m/s "
                f"powered={safe_float(r.get('total_powered_dv_lower_bound_m_s'), math.inf):9.2f} m/s "
                f"margin={safe_float(r.get('min_turn_margin_deg'), math.inf):8.3f} deg "
                f"{err}"
            )

    rows.sort(key=lambda r: (
        status_score(str(r.get("status", ""))),
        safe_float(r.get("total_powered_dv_lower_bound_m_s"), math.inf),
        -safe_float(r.get("min_turn_margin_deg"), -math.inf),
        safe_float(r.get("total_leg_correction_m_s"), math.inf),
    ))

    for i, r in enumerate(rows, start=1):
        r["triage_rank"] = i

    write_summary(args.output_csv, rows)

    print("")
    print("=== TOP TRIAGE RESULTS ===")
    print("triage rank status seq powered_m_s leg_dv_m_s margin_deg")
    for r in rows[:20]:
        print(
            f"{r['triage_rank']:>3} "
            f"{r.get('rank'):>5} "
            f"{r.get('status','UNKNOWN'):<14} "
            f"{r.get('sequence_bodies',''):<35} "
            f"{safe_float(r.get('total_powered_dv_lower_bound_m_s'), math.inf):10.2f} "
            f"{safe_float(r.get('total_leg_correction_m_s'), math.inf):10.2f} "
            f"{safe_float(r.get('min_turn_margin_deg'), math.inf):10.3f}"
        )

    print(f"\n[OK] summary: {args.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
