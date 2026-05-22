#!/usr/bin/env python3
"""
audit_vinf_frame_transforms_v0.py

Test whether the PyKEP/Lambert v∞ vector from anchor_packet.json is being
interpreted in the correct coordinate frame.

It queries the vessel relative state once per burn offset, then evaluates all
signed axis permutations of the anchor v∞ vector. For each transform it reports:
  - best phase error
  - best out-of-plane/binormal component
  - best combined "executable" score

This distinguishes:
  A) real parking-plane mismatch
  B) wrong LevelA/SPICE -> Principia raw transform for candidate CSV velocities
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from principia_targeter_client import PrincipiaTargeterClient


AXIS_NAMES = ("X", "Y", "Z")


def norm(v: Sequence[float]) -> float:
    return float(np.linalg.norm(np.asarray(v, dtype=float)))


def unit(v: Sequence[float]) -> np.ndarray:
    a = np.asarray(v, dtype=float)
    n = np.linalg.norm(a)
    if not np.isfinite(n) or n <= 0:
        raise ValueError(f"cannot normalize vector {v}")
    return a / n


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def read_live_t(path: Path) -> float:
    r = json.loads(path.read_text())
    for k in ("ut_s", "t_s", "spice_t_s", "et_s"):
        if k in r:
            return float(r[k])
    raise KeyError(f"could not find time in {path}")


def find_leg(anchor: dict[str, Any], leg_index: int) -> dict[str, Any]:
    for leg in anchor["legs"]:
        if int(leg["leg_index"]) == int(leg_index):
            return leg
    raise KeyError(f"leg_index {leg_index} not found")


def find_mu_from_catalog(path: Path, body: str) -> float:
    data = json.loads(path.read_text())
    body_l = body.lower()

    def walk(obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                if str(k).lower() == body_l and isinstance(v, dict):
                    yield v
            name = str(obj.get("name", obj.get("body", obj.get("id", "")))).lower()
            if name == body_l:
                yield obj
            for v in obj.values():
                yield from walk(v)
        elif isinstance(obj, list):
            for x in obj:
                yield from walk(x)

    for obj in walk(data):
        for key in ("mu_m3_s2", "gm_m3_s2", "gravitational_parameter_m3_s2", "mu", "gm", "gravitational_parameter"):
            if key in obj:
                val = float(obj[key])
                if val < 1e12:
                    val *= 1e9
                return val
    raise RuntimeError(f"mu not found for {body}")


def signed_permutations():
    # Maps source vector [x,y,z] to candidate raw [s0*src[p0], s1*src[p1], s2*src[p2]].
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((-1.0, 1.0), repeat=3):
            name = ",".join(("-" if s < 0 else "+") + AXIS_NAMES[p] for p, s in zip(perm, signs))
            yield name, perm, signs


def apply_transform(v: np.ndarray, perm, signs) -> np.ndarray:
    return np.array([signs[i] * v[perm[i]] for i in range(3)], dtype=float)


def compute_ejection(r_raw_m, v_raw_m_s, vinf_raw_m_s, mu_m3_s2):
    r = np.asarray(r_raw_m, dtype=float)
    v = np.asarray(v_raw_m_s, dtype=float)
    s = unit(vinf_raw_m_s)
    vinf_mag = norm(vinf_raw_m_s)

    rmag = norm(r)
    rhat = r / rmag
    vp_mag = math.sqrt(vinf_mag * vinf_mag + 2.0 * mu_m3_s2 / rmag)
    ecc = 1.0 + rmag * vinf_mag * vinf_mag / mu_m3_s2
    nu_inf = math.acos(clamp(-1.0 / ecc, -1.0, 1.0))
    theta = math.acos(clamp(float(np.dot(rhat, s)), -1.0, 1.0))
    phase_error_deg = math.degrees(theta - nu_inf)

    proj = s - float(np.dot(s, rhat)) * rhat
    if norm(proj) < 1e-12:
        proj = v - float(np.dot(v, rhat)) * rhat
    t_hat = unit(proj)
    v_after = vp_mag * t_hat
    dv = v_after - v

    transverse_current = v - float(np.dot(v, rhat)) * rhat
    T = unit(transverse_current) if norm(transverse_current) > 1e-9 else unit(v)
    B = unit(np.cross(r, v))
    N = unit(np.cross(B, T))
    dv_tnb = np.array([np.dot(dv, T), np.dot(dv, N), np.dot(dv, B)], dtype=float)

    h_hat = unit(np.cross(r, v))
    plane_angle_deg = math.degrees(math.asin(clamp(float(np.dot(h_hat, s)), -1.0, 1.0)))

    return {
        "phase_error_deg": phase_error_deg,
        "phase_abs_deg": abs(phase_error_deg),
        "plane_angle_deg": plane_angle_deg,
        "plane_abs_deg": abs(plane_angle_deg),
        "dv_norm_m_s": norm(dv),
        "dv_tangent_m_s": float(dv_tnb[0]),
        "dv_normal_m_s": float(dv_tnb[1]),
        "dv_binormal_m_s": float(dv_tnb[2]),
        "out_of_plane_abs_m_s": abs(float(dv_tnb[2])),
        "out_of_plane_fraction": abs(float(dv_tnb[2])) / max(1e-9, norm(dv)),
    }


def score_exec(r):
    # Low phase + low out-of-plane + plausible dv. Units balanced roughly.
    return r["phase_abs_deg"] + 0.01 * r["out_of_plane_abs_m_s"] + 0.0002 * r["dv_norm_m_s"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--server", required=True)
    ap.add_argument("--plugin-b64", type=Path, required=True)
    ap.add_argument("--plugin-arg-mode", choices=["option", "positional"], default="positional")
    ap.add_argument("--vessel-guid", required=True)
    ap.add_argument("--live-state-json", type=Path, required=True)
    ap.add_argument("--anchor-json", type=Path, required=True)
    ap.add_argument("--leg", type=int, default=1)
    ap.add_argument("--dep-body", default=None)
    ap.add_argument("--body-catalog", type=Path, required=True)
    ap.add_argument("--burn-offset-min-s", type=float, default=-7200.0)
    ap.add_argument("--burn-offset-max-s", type=float, default=7200.0)
    ap.add_argument("--burn-offset-step-s", type=float, default=60.0)
    ap.add_argument("--server-timeout-s", type=float, default=600.0)
    ap.add_argument("--quiet-stderr", action="store_true")
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    anchor = json.loads(args.anchor_json.read_text())
    leg = find_leg(anchor, args.leg)
    dep_body = (args.dep_body or leg["dep_body"]).upper()
    live_t = read_live_t(args.live_state_json)
    mu = find_mu_from_catalog(args.body_catalog, dep_body)

    # Source vector as stored in anchor before any raw conversion assumption.
    # We use levela km/s because that is the most natural PyKEP/SPICE output.
    vinf_src = np.asarray(leg["vinf_dep_levela_km_s"], dtype=float) * 1000.0
    t_dep_abs = float(leg["t_dep_s"])
    nominal_burn_dt = t_dep_abs - live_t

    offsets = []
    x = args.burn_offset_min_s
    while x <= args.burn_offset_max_s + 1e-9:
        offsets.append(float(x))
        x += args.burn_offset_step_s

    states = []
    with PrincipiaTargeterClient(
        args.server,
        args.plugin_b64,
        plugin_arg_mode=args.plugin_arg_mode,
        response_timeout_s=args.server_timeout_s,
        quiet_stderr=args.quiet_stderr,
    ) as client:
        for i, off in enumerate(offsets):
            burn_dt = nominal_burn_dt + off

            if burn_dt < 0.0:
                continue

            st = client.vrel(
                f"frameaudit_{os.getpid()}_{i}",
                args.vessel_guid,
                dep_body,
                burn_dt,
                [],
                timeout_s=args.server_timeout_s,
            )
            states.append({
                "burn_offset_s": off,
                "burn_dt_s": burn_dt,
                "r": st["final_rel_r_raw_m"],
                "v": st["final_rel_v_raw_m_s"],
            })

    summaries = []
    all_rows = []
    for name, perm, signs in signed_permutations():
        vinf_raw = apply_transform(vinf_src, perm, signs)
        rows = []
        for st in states:
            try:
                r = compute_ejection(st["r"], st["v"], vinf_raw, mu)
                r.update({
                    "transform": name,
                    "burn_offset_s": st["burn_offset_s"],
                    "burn_dt_s": st["burn_dt_s"],
                    "vinf_raw_0": float(vinf_raw[0]),
                    "vinf_raw_1": float(vinf_raw[1]),
                    "vinf_raw_2": float(vinf_raw[2]),
                    "score_exec": score_exec(r),
                })
                rows.append(r)
                all_rows.append(r)
            except Exception:
                pass
        if not rows:
            continue
        by_exec = min(rows, key=lambda r: r["score_exec"])
        by_phase = min(rows, key=lambda r: (r["phase_abs_deg"], r["out_of_plane_abs_m_s"]))
        by_oop = min(rows, key=lambda r: (r["out_of_plane_abs_m_s"], r["phase_abs_deg"]))
        summaries.append({
            "transform": name,
            "vinf_raw_m_s": [float(x) for x in vinf_raw],
            "best_exec": by_exec,
            "best_phase": by_phase,
            "best_out_of_plane": by_oop,
            "min_phase_abs_deg": by_phase["phase_abs_deg"],
            "min_out_of_plane_abs_m_s": by_oop["out_of_plane_abs_m_s"],
            "min_exec_score": by_exec["score_exec"],
        })

    summaries.sort(key=lambda s: (
        s["best_exec"]["score_exec"],
        s["best_exec"]["out_of_plane_abs_m_s"],
        s["best_exec"]["phase_abs_deg"],
    ))

    out = {
        "schema": "vinf_frame_transform_audit_v0",
        "leg": {
            "leg_index": args.leg,
            "dep_body": dep_body,
            "arr_body": leg["arr_body"],
            "vinf_dep_levela_km_s": leg["vinf_dep_levela_km_s"],
            "vinf_norm_m_s": norm(vinf_src),
            "t_dep_s": t_dep_abs,
            "nominal_burn_dt_s": nominal_burn_dt,
        },
        "n_offsets": len(offsets),
        "n_transforms": len(summaries),
        "top_transforms": summaries[:20],
    }

    json_path = args.output_dir / "vinf_frame_transform_audit.json"
    csv_path = args.output_dir / "vinf_frame_transform_audit_rows.csv"
    json_path.write_text(json.dumps(out, indent=2) + "\n")

    if all_rows:
        fields = sorted(all_rows[0].keys())
        with csv_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(all_rows)

    print("=== VINF FRAME TRANSFORM AUDIT ===")
    print(f"vinf source levela m/s: {vinf_src.tolist()} |v|={norm(vinf_src):.6f}")
    print("rank transform best_exec_score phase_deg oop_m_s dv_m_s T N B burn_off")
    for i, s in enumerate(summaries[:20], 1):
        r = s["best_exec"]
        print(
            f"{i:3d} {s['transform']:<12} "
            f"{r['score_exec']:10.3f} "
            f"{r['phase_error_deg']:10.3f} "
            f"{r['out_of_plane_abs_m_s']:10.2f} "
            f"{r['dv_norm_m_s']:8.2f} "
            f"{r['dv_tangent_m_s']:8.2f} "
            f"{r['dv_normal_m_s']:8.2f} "
            f"{r['dv_binormal_m_s']:8.2f} "
            f"{r['burn_offset_s']:8.1f}"
        )
    print(f"[OK] wrote {json_path}")
    print(f"[OK] wrote {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
