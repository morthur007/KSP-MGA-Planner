#!/usr/bin/env python3
"""
find_pykep_anchor_row.py

Find the candidate CSV row that actually corresponds to a promoted/finalist
leg_optimizations.csv, instead of guessing from rank numbers.

Why:
  triage rank, candidate_seed rank, merged row index, and promoted folder names
  are easy to mix up. This script matches by:
    - sequence_bodies, when available;
    - first event/departure epoch near leg_optimizations t_dep_s;
    - optional TOF similarity.

It prints the best rows and the exact --row-index0 to pass into
extract_pykep_anchor_packet.py.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


DAY_S = 86400.0


def f(row: dict[str, str], key: str, default: float = math.nan) -> float:
    try:
        v = row.get(key, "")
        if v in ("", None):
            return default
        return float(v)
    except Exception:
        return default


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as fp:
        return list(csv.DictReader(fp))


def normseq(s: str) -> str:
    return " ".join(str(s).replace("-", " ").upper().split())


def candidate_first_epoch(row: dict[str, str]) -> float:
    if row.get("epochs_et_s"):
        try:
            return float(str(row["epochs_et_s"]).replace(";", ",").split(",")[0])
        except Exception:
            pass
    matches = []
    for k, v in row.items():
        if k.startswith("event0_") and k.endswith("_et_s") and v not in ("", None):
            matches.append(v)
    if matches:
        return float(matches[0])
    return math.nan


def candidate_tof1_days(row: dict[str, str]) -> float:
    if row.get("tofs_days"):
        try:
            return float(str(row["tofs_days"]).replace(";", ",").split(",")[0])
        except Exception:
            pass
    return f(row, "leg1_tof_days")


def candidate_seq(row: dict[str, str]) -> str:
    if row.get("sequence_bodies"):
        return normseq(row["sequence_bodies"])
    if row.get("sequence"):
        return normseq(row["sequence"])
    # Infer from event fields.
    events = []
    for k in row:
        if k.startswith("event") and k.endswith("_et_s"):
            try:
                mid = k[len("event"):]
                idx_s, rest = mid.split("_", 1)
                body = rest[:-len("_et_s")]
                events.append((int(idx_s), body.upper()))
            except Exception:
                pass
    if events:
        return " ".join(body for _, body in sorted(events))
    return ""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--candidate-csv", type=Path, required=True)
    ap.add_argument("--leg-optimizations", type=Path, required=True)
    ap.add_argument("--sequence", default=None, help='Optional exact sequence, e.g. "KERBIN EVE KERBIN JOOL"')
    ap.add_argument("--leg", type=int, default=1)
    ap.add_argument("--top", type=int, default=20)
    args = ap.parse_args()

    legs = read_rows(args.leg_optimizations)
    if not legs:
        raise SystemExit(f"No rows in {args.leg_optimizations}")

    # Use first row with matching leg if present, otherwise first row.
    legrow = None
    for r in legs:
        try:
            if int(float(r.get("leg", "nan"))) in (args.leg, args.leg - 1):
                legrow = r
                break
        except Exception:
            pass
    if legrow is None:
        legrow = legs[0]

    target_tdep = f(legrow, "t_dep_s", f(legrow, "t_start_s"))
    target_tarr = f(legrow, "t_arr_s", f(legrow, "t_end_s"))
    target_tof_days = (target_tarr - target_tdep) / DAY_S if math.isfinite(target_tdep) and math.isfinite(target_tarr) else math.nan
    target_seq = normseq(args.sequence) if args.sequence else None

    rows = read_rows(args.candidate_csv)
    scored = []
    for i, r in enumerate(rows):
        seq = candidate_seq(r)
        t0 = candidate_first_epoch(r)
        tof1 = candidate_tof1_days(r)

        seq_pen = 0.0
        if target_seq and seq != target_seq:
            # Do not discard; show it, but make it rank lower.
            seq_pen = 1e9

        dt_days = abs(t0 - target_tdep) / DAY_S if math.isfinite(t0) and math.isfinite(target_tdep) else 1e8
        dtof_days = abs(tof1 - target_tof_days) if math.isfinite(tof1) and math.isfinite(target_tof_days) else 1e6
        score = seq_pen + dt_days + 0.1 * dtof_days
        scored.append((score, i, seq, t0, tof1, dt_days, dtof_days, r))

    scored.sort(key=lambda x: x[0])

    print("=== FIND PYKEP ANCHOR ROW ===")
    print(f"candidate_csv      : {args.candidate_csv}")
    print(f"leg_optimizations  : {args.leg_optimizations}")
    print(f"target_tdep_s      : {target_tdep}")
    print(f"target_tarr_s      : {target_tarr}")
    print(f"target_tof_days    : {target_tof_days}")
    print(f"target_sequence    : {target_seq or '(not constrained)'}")
    print("")
    print("rank row_index0 dt_days dtof_days sequence t0 tof1 raw_sum dep_vinf arr_vinf paths")
    for rank, (score, i, seq, t0, tof1, dt_days, dtof_days, r) in enumerate(scored[: args.top], start=1):
        print(
            f"{rank:>3} {i:>10} "
            f"{dt_days:9.3f} {dtof_days:9.3f} "
            f"{seq:<38} "
            f"{t0:14.6f} {tof1:8.3f} "
            f"{r.get('raw_sum_km_s',''):<10} "
            f"{r.get('departure_vinf_km_s',''):<10} "
            f"{r.get('arrival_vinf_km_s',''):<10} "
            f"{r.get('leg_paths','')}"
        )
    print("")
    best = scored[0]
    print(f"[BEST] row_index0={best[1]} sequence='{best[2]}' dt_days={best[5]:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
