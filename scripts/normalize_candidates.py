#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def split_csv_field(value: str) -> list[str]:
    return [x.strip() for x in (value or "").split(",") if x.strip()]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--top-n", type=int, default=100)
    args = p.parse_args()

    with args.input.open(newline="") as f:
        rows = list(csv.DictReader(f))

    out_rows = []

    for rank, row in enumerate(rows[: args.top_n], start=1):
        sequence = row.get("sequence_bodies") or row.get("sequence") or ""
        bodies = sequence.replace("-", " ").split()
        epochs = split_csv_field(row.get("epochs_et_s", ""))
        tofs = split_csv_field(row.get("tofs_days", ""))
        paths = split_csv_field(row.get("leg_paths", ""))

        if len(bodies) < 2:
            raise ValueError(f"rank {rank}: invalid sequence: {sequence}")
        if len(epochs) != len(bodies):
            raise ValueError(f"rank {rank}: epochs/body mismatch: {len(epochs)} vs {len(bodies)}")
        if len(tofs) != len(bodies) - 1:
            raise ValueError(f"rank {rank}: tof/body mismatch")
        if len(paths) != len(bodies) - 1:
            raise ValueError(f"rank {rank}: paths/body mismatch")

        out = {
            "candidate_id": f"cand_{rank:05d}",
            "rank": rank,
            "sequence_bodies": " ".join(bodies),
            "n_events": len(bodies),
            "n_legs": len(bodies) - 1,
            "source_sequence": row.get("sequence", ""),
            "cost": row.get("cost", ""),
            "raw_sum_km_s": row.get("raw_sum_km_s", ""),
            "departure_vinf_km_s": row.get("departure_vinf_km_s", ""),
            "arrival_vinf_km_s": row.get("arrival_vinf_km_s", ""),
            "powered_flyby_dv_km_s": row.get("powered_flyby_dv_km_s", ""),
            "turn_excess_deg": row.get("turn_excess_deg", ""),
            "min_turn_margin_deg": row.get("min_turn_margin_deg", ""),
            "tof_total_days": row.get("tof_total_days", ""),
        }

        for i, body in enumerate(bodies):
            out[f"event{i}_body"] = body
            out[f"event{i}_et_s"] = epochs[i]

        for i in range(1, len(bodies)):
            out[f"leg{i}_dep"] = bodies[i - 1]
            out[f"leg{i}_arr"] = bodies[i]
            out[f"leg{i}_tof_days"] = tofs[i - 1]
            out[f"leg{i}_path"] = paths[i - 1]

        out_rows.append(out)

    fields = []
    preferred = [
        "candidate_id", "rank", "sequence_bodies", "n_events", "n_legs",
        "source_sequence", "cost", "raw_sum_km_s",
        "departure_vinf_km_s", "arrival_vinf_km_s",
        "powered_flyby_dv_km_s", "turn_excess_deg",
        "min_turn_margin_deg", "tof_total_days",
    ]
    for k in preferred:
        if k not in fields:
            fields.append(k)
    for r in out_rows:
        for k in r:
            if k not in fields:
                fields.append(k)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(out_rows)

    print(f"[OK] wrote {args.output}")
    print(f"[INFO] rows={len(out_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
