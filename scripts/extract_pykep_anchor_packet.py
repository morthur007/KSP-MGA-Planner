#!/usr/bin/env python3
"""
extract_pykep_anchor_packet.py

Extract a clean PyKEP/Lambert anchor packet directly from the beam/family-search
candidate CSV, bypassing downstream rerender/bridge/IPOPT artifacts.

Purpose:
  - Recover the original patched-conics/PyKEP leg geometry.
  - Preserve exact Lambert heliocentric velocities per leg.
  - Recompute body states from the SPICE kernel at the candidate epochs.
  - Emit v-infinity vectors in both SPICE/LevelA and Principia raw frames.
  - Provide a clean seed for later N-body refinement.

Frame convention used by this project:
  Principia raw -> LevelA/SPICE:
    (X, Y, Z) -> (-Y, +Z, +X)

  LevelA/SPICE -> Principia raw:
    (X, Y, Z) -> (+Z, -X, +Y)

Units in output:
  *_km, *_km_s are SPICE/LevelA/J2000 units from the kernel / PyKEP.
  *_m, *_m_s are SI.
  *_raw_* are Principia raw axes.

Typical use:

python scripts/extract_pykep_anchor_packet.py \
  --candidate-csv data/runs/family_search_smoke/merged_candidates.csv \
  --row-index0 12 \
  --bsp data/kernels/jnsq/v0_3/jnsq_principia_native_v0_3_1000y_1h_deg9.bsp \
  --tpc data/kernels/jnsq/v0_3/jnsq_principia_native_v0_3_1000y_1h_deg9.ids.tpc \
  --central-body Sun \
  --output-json data/runs/game_export/rank12_real/pykep_anchor_rank12/anchor_packet.json

If the CSV lacks leg{i}_vdep/varr fields, pass --central-mu-km3-s2 and the
script will try to re-solve the Lambert leg through ksp_mga.lambert.pykep_gateway.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

try:
    import spiceypy as spice
except Exception as exc:  # pragma: no cover
    spice = None
    _SPICE_IMPORT_ERROR = exc
else:
    _SPICE_IMPORT_ERROR = None


DAY_S = 86400.0


def norm_name(s: str) -> str:
    return str(s).strip().upper()


def norm(v: Sequence[float]) -> float:
    return float(np.linalg.norm(np.asarray(v, dtype=float)))


def levela_to_raw_vec(v_xyz: Sequence[float]) -> list[float]:
    """LevelA/SPICE canonical -> Principia raw."""
    x, y, z = map(float, v_xyz)
    return [z, -x, y]


def raw_to_levela_vec(v_xyz: Sequence[float]) -> list[float]:
    """Principia raw -> LevelA/SPICE canonical."""
    x, y, z = map(float, v_xyz)
    return [-y, z, x]


def km_to_m_vec(v: Sequence[float]) -> list[float]:
    return [1000.0 * float(x) for x in v]


def km_s_to_m_s_vec(v: Sequence[float]) -> list[float]:
    return [1000.0 * float(x) for x in v]


def parse_csv_float(row: dict[str, str], key: str, default: float | None = None) -> float:
    val = row.get(key, "")
    if val not in ("", None):
        return float(val)
    if default is not None:
        return float(default)
    raise KeyError(f"Missing numeric CSV field {key!r}")


def parse_float_list(s: str) -> list[float]:
    return [float(x.strip()) for x in str(s).replace(";", ",").split(",") if x.strip()]


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def get_sequence(row: dict[str, str], explicit: Sequence[str] | None) -> list[str]:
    if explicit:
        return [norm_name(x) for x in explicit]
    if row.get("sequence_bodies"):
        return [norm_name(x) for x in row["sequence_bodies"].split()]
    if row.get("sequence"):
        return [norm_name(x) for x in row["sequence"].replace("-", " ").split()]

    # Last resort: event0_KERBIN_et_s, event1_EVE_et_s, ...
    events: list[tuple[int, str]] = []
    for k in row:
        if not k.startswith("event") or not k.endswith("_et_s"):
            continue
        try:
            idx_s, body = k[len("event"):].split("_", 1)
            idx = int(idx_s)
            body = body[:-len("_et_s")]
            events.append((idx, norm_name(body)))
        except Exception:
            pass
    if events:
        return [b for _, b in sorted(events)]

    raise RuntimeError("Could not infer sequence. Pass --sequence KERBIN EVE ...")


def get_epochs(row: dict[str, str], seq: Sequence[str]) -> list[float]:
    if row.get("epochs_et_s"):
        vals = parse_float_list(row["epochs_et_s"])
        if len(vals) == len(seq):
            return vals

    vals: list[float] = []
    for i, body in enumerate(seq):
        key = f"event{i}_{norm_name(body)}_et_s"
        if key not in row:
            # Search any event{i}_*_et_s
            prefix = f"event{i}_"
            matches = [k for k in row if k.startswith(prefix) and k.endswith("_et_s")]
            if not matches:
                raise RuntimeError(f"Missing epoch for event {i} / {body}")
            key = matches[0]
        vals.append(float(row[key]))
    return vals


def spk_state(body: str, et_s: float, central_body: str) -> np.ndarray:
    if spice is None:
        raise RuntimeError(f"spiceypy is not importable: {_SPICE_IMPORT_ERROR}")
    st, _ = spice.spkezr(norm_name(body), float(et_s), "J2000", "NONE", norm_name(central_body))
    return np.asarray(st, dtype=float)  # km, km/s


def solve_lambert_if_needed(
    r0_km: Sequence[float],
    r1_km: Sequence[float],
    tof_s: float,
    mu_km3_s2: float,
    path_hint: str | None,
    max_revs: int,
) -> tuple[np.ndarray, np.ndarray, str]:
    try:
        from ksp_mga.lambert.pykep_gateway import solve_lambert_pykep
    except Exception as exc:
        raise RuntimeError(
            "CSV does not contain Lambert vdep/varr fields and "
            "ksp_mga.lambert.pykep_gateway is not importable."
        ) from exc

    candidates = []
    for cw in (False, True):
        for sol in solve_lambert_pykep(
            tuple(float(x) for x in r0_km),
            tuple(float(x) for x in r1_km),
            float(tof_s),
            float(mu_km3_s2),
            cw=cw,
            max_revs=int(max_revs),
        ):
            candidates.append(sol)

    if not candidates:
        raise RuntimeError("Lambert re-solve returned no candidates.")

    if path_hint:
        for sol in candidates:
            if str(sol.path_label) == str(path_hint):
                return np.asarray(sol.v0_km_s, dtype=float), np.asarray(sol.v1_km_s, dtype=float), str(sol.path_label)

    # Fallback: minimum departure speed candidate.
    best = min(candidates, key=lambda s: norm(s.v0_km_s))
    return np.asarray(best.v0_km_s, dtype=float), np.asarray(best.v1_km_s, dtype=float), str(best.path_label)


def candidate_leg_velocities(
    row: dict[str, str],
    leg_i: int,
    dep_state: np.ndarray,
    arr_state: np.ndarray,
    tof_s: float,
    central_mu_km3_s2: float | None,
    max_revs: int,
) -> tuple[np.ndarray, np.ndarray, str, str]:
    prefix = f"leg{leg_i}"
    vx_key = f"{prefix}_vdep_x_km_s"
    if vx_key in row and row.get(vx_key, "") != "":
        vdep = np.array([
            float(row[f"{prefix}_vdep_x_km_s"]),
            float(row[f"{prefix}_vdep_y_km_s"]),
            float(row[f"{prefix}_vdep_z_km_s"]),
        ], dtype=float)
        varr = np.array([
            float(row[f"{prefix}_varr_x_km_s"]),
            float(row[f"{prefix}_varr_y_km_s"]),
            float(row[f"{prefix}_varr_z_km_s"]),
        ], dtype=float)
        return vdep, varr, row.get(f"{prefix}_path", ""), "csv_exact"

    if central_mu_km3_s2 is None:
        raise RuntimeError(
            f"CSV lacks {vx_key}; pass --central-mu-km3-s2 to re-solve Lambert."
        )

    path_hint = row.get(f"{prefix}_path", "")
    vdep, varr, path = solve_lambert_if_needed(
        dep_state[:3],
        arr_state[:3],
        tof_s,
        float(central_mu_km3_s2),
        path_hint,
        max_revs,
    )
    return vdep, varr, path, "recomputed_lambert"


def leg_packet(
    row: dict[str, str],
    seq: Sequence[str],
    epochs: Sequence[float],
    leg_i: int,
    central_body: str,
    central_mu_km3_s2: float | None,
    max_revs: int,
) -> dict[str, Any]:
    dep = norm_name(seq[leg_i - 1])
    arr = norm_name(seq[leg_i])
    t_dep = float(epochs[leg_i - 1])
    t_arr = float(epochs[leg_i])
    tof_s = t_arr - t_dep

    dep_st = spk_state(dep, t_dep, central_body)
    arr_st = spk_state(arr, t_arr, central_body)

    vdep_lambert, varr_lambert, path, source = candidate_leg_velocities(
        row,
        leg_i,
        dep_st,
        arr_st,
        tof_s,
        central_mu_km3_s2,
        max_revs,
    )

    dep_body_v = dep_st[3:]
    arr_body_v = arr_st[3:]
    vinf_dep = vdep_lambert - dep_body_v
    vinf_arr = varr_lambert - arr_body_v

    return {
        "leg_index": leg_i,
        "dep_body": dep,
        "arr_body": arr,
        "path": path,
        "source": source,
        "t_dep_s": t_dep,
        "t_arr_s": t_arr,
        "tof_s": tof_s,
        "tof_days": tof_s / DAY_S,

        "dep_body_r_levela_km": dep_st[:3].tolist(),
        "dep_body_v_levela_km_s": dep_body_v.tolist(),
        "arr_body_r_levela_km": arr_st[:3].tolist(),
        "arr_body_v_levela_km_s": arr_body_v.tolist(),

        "dep_body_r_raw_m": km_to_m_vec(levela_to_raw_vec(dep_st[:3])),
        "dep_body_v_raw_m_s": km_s_to_m_s_vec(levela_to_raw_vec(dep_body_v)),
        "arr_body_r_raw_m": km_to_m_vec(levela_to_raw_vec(arr_st[:3])),
        "arr_body_v_raw_m_s": km_s_to_m_s_vec(levela_to_raw_vec(arr_body_v)),

        "vdep_lambert_levela_km_s": vdep_lambert.tolist(),
        "varr_lambert_levela_km_s": varr_lambert.tolist(),
        "vdep_lambert_raw_m_s": km_s_to_m_s_vec(levela_to_raw_vec(vdep_lambert)),
        "varr_lambert_raw_m_s": km_s_to_m_s_vec(levela_to_raw_vec(varr_lambert)),

        "vinf_dep_levela_km_s": vinf_dep.tolist(),
        "vinf_arr_levela_km_s": vinf_arr.tolist(),
        "vinf_dep_raw_m_s": km_s_to_m_s_vec(levela_to_raw_vec(vinf_dep)),
        "vinf_arr_raw_m_s": km_s_to_m_s_vec(levela_to_raw_vec(vinf_arr)),
        "vinf_dep_norm_km_s": norm(vinf_dep),
        "vinf_arr_norm_km_s": norm(vinf_arr),
    }


def summarize_anchor(packet: dict[str, Any]) -> str:
    lines = []
    lines.append("=== PYKEP ANCHOR PACKET ===")
    lines.append(f"row_index0 : {packet['row_index0']}")
    lines.append(f"sequence   : {' -> '.join(packet['sequence'])}")
    lines.append(f"epochs     : {', '.join(f'{x:.6f}' for x in packet['epochs_s'])}")
    for leg in packet["legs"]:
        lines.append(
            f"leg {leg['leg_index']} {leg['dep_body']}->{leg['arr_body']} "
            f"tof={leg['tof_days']:.3f} d path={leg['path']} source={leg['source']} "
            f"vinf_dep={leg['vinf_dep_norm_km_s']:.6f} km/s "
            f"vinf_arr={leg['vinf_arr_norm_km_s']:.6f} km/s"
        )
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--candidate-csv", type=Path, required=True)
    ap.add_argument("--row-index0", type=int, required=True, help="Zero-based row index in the CSV after its current sorting.")
    ap.add_argument("--sequence", nargs="+", default=None)
    ap.add_argument("--bsp", type=Path, required=True)
    ap.add_argument("--tpc", type=Path, required=True)
    ap.add_argument("--central-body", default="Sun")
    ap.add_argument("--central-mu-km3-s2", type=float, default=None)
    ap.add_argument("--max-revs", type=int, default=1)
    ap.add_argument("--output-json", type=Path, required=True)
    ap.add_argument("--output-summary", type=Path, default=None)
    args = ap.parse_args()

    if spice is None:
        raise SystemExit(f"spiceypy is not importable: {_SPICE_IMPORT_ERROR}")

    spice.furnsh(str(args.tpc))
    spice.furnsh(str(args.bsp))

    rows = load_rows(args.candidate_csv)
    if args.row_index0 < 0 or args.row_index0 >= len(rows):
        raise SystemExit(f"--row-index0 out of range: {args.row_index0}; rows={len(rows)}")
    row = rows[args.row_index0]

    seq = get_sequence(row, args.sequence)
    epochs = get_epochs(row, seq)
    if len(epochs) != len(seq):
        raise SystemExit(f"len(epochs)={len(epochs)} != len(sequence)={len(seq)}")

    legs = [
        leg_packet(
            row=row,
            seq=seq,
            epochs=epochs,
            leg_i=i,
            central_body=args.central_body,
            central_mu_km3_s2=args.central_mu_km3_s2,
            max_revs=args.max_revs,
        )
        for i in range(1, len(seq))
    ]

    packet = {
        "schema": "pykep_anchor_packet_v0_1",
        "candidate_csv": str(args.candidate_csv),
        "row_index0": args.row_index0,
        "central_body": norm_name(args.central_body),
        "sequence": seq,
        "epochs_s": epochs,
        "candidate_metrics": {
            k: row[k]
            for k in [
                "sequence", "sequence_bodies", "cost", "raw_sum_km_s",
                "departure_vinf_km_s", "arrival_vinf_km_s",
                "powered_flyby_dv_km_s", "turn_excess_deg",
                "min_turn_margin_deg", "tof_total_days", "leg_paths",
            ]
            if k in row
        },
        "legs": legs,
        "frame_contract": {
            "levela_to_raw": "(X,Y,Z)->(+Z,-X,+Y)",
            "raw_to_levela": "(X,Y,Z)->(-Y,+Z,+X)",
        },
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(packet, indent=2) + "\n")

    summary = summarize_anchor(packet)
    print(summary)

    if args.output_summary:
        args.output_summary.parent.mkdir(parents=True, exist_ok=True)
        args.output_summary.write_text(summary + "\n")
        print(f"[OK] wrote {args.output_summary}")

    print(f"[OK] wrote {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
