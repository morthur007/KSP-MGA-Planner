#!/usr/bin/env python3
"""
audit_pykep_parking_plane_v0.py

Audit whether the current parking orbit can realize the PyKEP departure v-infinity
with a mostly prograde burn.

Core diagnostic:
  plane_angle_deg = asin(dot(h_hat, vinf_hat))

If |plane_angle_deg| is large, the desired PyKEP v∞ is far out of the parking
orbit plane. No choice of true anomaly will make a single impulsive LKO burn
mostly tangent/prograde; a large out-of-plane component is geometrically required.

This script also reconstructs the ideal periapsis ejection burn at each sampled
burn time and reports its local T/N/B components.
"""

from __future__ import annotations

import argparse
import csv
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
    raise KeyError(f"leg_index {leg_index} not found in anchor")


def find_mu_from_catalog(path: Path, body: str) -> float | None:
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
        for key in (
            "mu_m3_s2", "gm_m3_s2", "gravitational_parameter_m3_s2",
            "mu", "gm", "gravitational_parameter",
        ):
            if key in obj:
                val = float(obj[key])
                if val < 1e12:
                    val *= 1e9
                return val
    return None


def compute_ejection_dv(r_raw_m, v_raw_m_s, vinf_raw_m_s, mu_m3_s2):
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
    phase_error = theta - nu_inf

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
    plane_angle = math.degrees(math.asin(clamp(float(np.dot(h_hat, s)), -1.0, 1.0)))

    return {
        "rmag_km": rmag / 1000.0,
        "vmag_m_s": norm(v),
        "vinf_mag_m_s": vinf_mag,
        "vp_mag_m_s": vp_mag,
        "ecc": ecc,
        "nu_inf_deg": math.degrees(nu_inf),
        "theta_to_vinf_deg": math.degrees(theta),
        "phase_error_deg": math.degrees(phase_error),
        "plane_angle_deg": plane_angle,
        "abs_plane_angle_deg": abs(plane_angle),
        "dv_norm_m_s": norm(dv),
        "dv_tangent_m_s": float(dv_tnb[0]),
        "dv_normal_m_s": float(dv_tnb[1]),
        "dv_binormal_m_s": float(dv_tnb[2]),
        "out_of_plane_abs_m_s": abs(float(dv_tnb[2])),
        "out_of_plane_fraction": abs(float(dv_tnb[2])) / max(1e-9, norm(dv)),
        "dv_raw_m_s": dv.tolist(),
    }


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
    ap.add_argument("--burn-offset-step-s", type=float, default=30.0)
    ap.add_argument("--server-timeout-s", type=float, default=600.0)
    ap.add_argument("--quiet-stderr", action="store_true")
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    anchor = json.loads(args.anchor_json.read_text())
    leg = find_leg(anchor, args.leg)
    live_t = read_live_t(args.live_state_json)

    dep_body = (args.dep_body or leg["dep_body"]).upper()
    mu = find_mu_from_catalog(args.body_catalog, dep_body)
    if mu is None:
        raise SystemExit(f"Could not find mu for {dep_body} in {args.body_catalog}")

    t_dep_abs = float(leg["t_dep_s"])
    nominal_burn_dt = t_dep_abs - live_t
    vinf_raw = np.asarray(leg["vinf_dep_raw_m_s"], dtype=float)

    offsets = []
    x = args.burn_offset_min_s
    while x <= args.burn_offset_max_s + 1e-9:
        offsets.append(float(x))
        x += args.burn_offset_step_s

    rows = []
    with PrincipiaTargeterClient(
        args.server,
        args.plugin_b64,
        plugin_arg_mode=args.plugin_arg_mode,
        response_timeout_s=args.server_timeout_s,
        quiet_stderr=args.quiet_stderr,
    ) as client:
        for i, off in enumerate(offsets):
            burn_dt = nominal_burn_dt + off
            try:
                st = client.vrel(
                    f"plane_{os.getpid()}_{i}",
                    args.vessel_guid,
                    dep_body,
                    burn_dt,
                    [],
                    timeout_s=args.server_timeout_s,
                )
                d = compute_ejection_dv(
                    st["final_rel_r_raw_m"],
                    st["final_rel_v_raw_m_s"],
                    vinf_raw,
                    float(mu),
                )
                d.update({
                    "ok": True,
                    "error": "",
                    "burn_offset_s": off,
                    "burn_dt_s": burn_dt,
                    "burn_abs_s": live_t + burn_dt,
                })
            except Exception as exc:
                d = {
                    "ok": False,
                    "error": str(exc),
                    "burn_offset_s": off,
                    "burn_dt_s": burn_dt,
                    "burn_abs_s": live_t + burn_dt,
                }
            rows.append(d)

    ok = [r for r in rows if r.get("ok")]
    by_phase = sorted(ok, key=lambda r: (abs(r["phase_error_deg"]), r["dv_norm_m_s"]))
    by_plane = sorted(ok, key=lambda r: (r["abs_plane_angle_deg"], r["out_of_plane_abs_m_s"]))
    by_out = sorted(ok, key=lambda r: (r["out_of_plane_abs_m_s"], abs(r["phase_error_deg"])))

    summary = {
        "schema": "pykep_parking_plane_audit_v0",
        "leg": {
            "leg_index": args.leg,
            "dep_body": dep_body,
            "arr_body": leg["arr_body"],
            "t_dep_s": t_dep_abs,
            "vinf_dep_raw_m_s": vinf_raw.tolist(),
            "vinf_dep_norm_m_s": norm(vinf_raw),
        },
        "live_t_s": live_t,
        "nominal_burn_dt_s": nominal_burn_dt,
        "mu_m3_s2": mu,
        "n_rows": len(rows),
        "n_ok": len(ok),
        "best_phase": by_phase[0] if by_phase else None,
        "best_plane": by_plane[0] if by_plane else None,
        "best_out_of_plane": by_out[0] if by_out else None,
        "top_phase": by_phase[:20],
        "top_plane": by_plane[:20],
        "top_out_of_plane": by_out[:20],
    }

    json_path = args.output_dir / "parking_plane_audit.json"
    csv_path = args.output_dir / "parking_plane_audit.csv"

    fields = sorted({k for r in rows for k in r.keys() if k != "dv_raw_m_s"} | {"dv_raw_m_s_0", "dv_raw_m_s_1", "dv_raw_m_s_2"})
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            rr = dict(r)
            dv = rr.pop("dv_raw_m_s", [math.nan, math.nan, math.nan])
            rr["dv_raw_m_s_0"] = dv[0]
            rr["dv_raw_m_s_1"] = dv[1]
            rr["dv_raw_m_s_2"] = dv[2]
            w.writerow(rr)

    json_path.write_text(json.dumps(summary, indent=2) + "\n")

    print("=== PARKING PLANE / PYKEP VINF AUDIT ===")
    print(f"dep_body       : {dep_body}")
    print(f"vinf_norm_m_s  : {norm(vinf_raw):.6f}")
    print(f"rows/ok        : {len(rows)} / {len(ok)}")
    if by_phase:
        r = by_phase[0]
        print(
            "best_phase     : "
            f"off={r['burn_offset_s']:.1f}s phase={r['phase_error_deg']:.4f}deg "
            f"plane={r['plane_angle_deg']:.4f}deg "
            f"dv={r['dv_norm_m_s']:.2f} "
            f"T={r['dv_tangent_m_s']:.2f} N={r['dv_normal_m_s']:.2f} B={r['dv_binormal_m_s']:.2f}"
        )
    if by_plane:
        r = by_plane[0]
        print(
            "best_plane     : "
            f"off={r['burn_offset_s']:.1f}s phase={r['phase_error_deg']:.4f}deg "
            f"plane={r['plane_angle_deg']:.4f}deg "
            f"dv={r['dv_norm_m_s']:.2f} "
            f"T={r['dv_tangent_m_s']:.2f} N={r['dv_normal_m_s']:.2f} B={r['dv_binormal_m_s']:.2f}"
        )
    if by_out:
        r = by_out[0]
        print(
            "best_out_plane : "
            f"off={r['burn_offset_s']:.1f}s phase={r['phase_error_deg']:.4f}deg "
            f"plane={r['plane_angle_deg']:.4f}deg "
            f"dv={r['dv_norm_m_s']:.2f} "
            f"T={r['dv_tangent_m_s']:.2f} N={r['dv_normal_m_s']:.2f} B={r['dv_binormal_m_s']:.2f}"
        )
    print(f"[OK] wrote {json_path}")
    print(f"[OK] wrote {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
