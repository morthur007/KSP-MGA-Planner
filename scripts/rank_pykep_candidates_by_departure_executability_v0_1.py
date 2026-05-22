#!/usr/bin/env python3
"""
rank_pykep_candidates_by_departure_executability_v0_1.py

Universal candidate hunter for KSP + Principia MGA.

Goal:
  Find patched-conics/PyKEP/Lambert candidates whose FIRST departure v∞ is
  actually executable from the user's current parking orbit, instead of only
  being good heliocentrically.

It ranks candidate rows by:
  - sequence/body filters;
  - PyKEP/Lambert first-leg v∞;
  - estimated physical LKO ejection burn from current parking orbit;
  - out-of-plane/binormal requirement;
  - phase error at best burn phase;
  - total Lambert score/raw_sum when available.

Default preburn model:
  --preburn-source twobody

This intentionally avoids trusting the current Principia targeter server to
coast the low orbit before burn0. It queries the live parking state at dt=0,
then propagates that parking orbit with a simple two-body Kepler propagator
around the departure body to test burn phases.

Frame convention:
  LevelA/SPICE canonical -> Principia raw:
    (X,Y,Z) -> (+Z,-X,+Y)

The CSV is expected to contain beam_search/PyKEP fields such as:
  sequence_bodies
  event0_KERBIN_et_s
  leg1_vdep_x_km_s, leg1_vdep_y_km_s, leg1_vdep_z_km_s
  event0_KERBIN_vx_km_s, event0_KERBIN_vy_km_s, event0_KERBIN_vz_km_s

If body velocity is missing, pass --bsp/--tpc/--central-body and spiceypy will
be used to reconstruct the body velocity.

Example:

python scripts/rank_pykep_candidates_by_departure_executability_v0_1.py \
  --candidate-csv data/runs/family_search_smoke/candidate_seed.csv \
  --server /home/matheus/Principia/bin/x64/principia_impulsive_particle_server \
  --plugin-b64 data/principia/live_probe/principia_serialized_plugin_rocket.b64 \
  --plugin-arg-mode positional \
  --vessel-guid 60735c81-7e29-4c06-9551-9e5283e37586 \
  --live-state-json data/runs/game_export/rank12_real/live_state_raw_near_tdep.json \
  --body-catalog data/catalogs/jnsq/body_catalog.json \
  --sequence "KERBIN EVE KERBIN JOOL" \
  --preburn-source twobody \
  --phase-offset-min-s -7200 \
  --phase-offset-max-s 7200 \
  --phase-offset-step-s 60 \
  --max-binormal-m-s 500 \
  --max-phase-abs-deg 5 \
  --max-dv-m-s 3200 \
  --top-n 50 \
  --output-dir data/runs/game_export/rank12_real/candidate_hunt_departure_exec01
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

try:
    from principia_targeter_client import PrincipiaTargeterClient
except Exception as exc:  # pragma: no cover
    PrincipiaTargeterClient = None
    _CLIENT_IMPORT_ERROR = exc
else:
    _CLIENT_IMPORT_ERROR = None

try:
    import spiceypy as spice
except Exception as exc:  # pragma: no cover
    spice = None
    _SPICE_IMPORT_ERROR = exc
else:
    _SPICE_IMPORT_ERROR = None


DAY_S = 86400.0
AXIS_EPS = 1e-12


# -----------------------------------------------------------------------------
# Generic math helpers
# -----------------------------------------------------------------------------

def norm(v: Sequence[float]) -> float:
    return float(np.linalg.norm(np.asarray(v, dtype=float)))


def unit(v: Sequence[float]) -> np.ndarray:
    a = np.asarray(v, dtype=float)
    n = np.linalg.norm(a)
    if not np.isfinite(n) or n <= 0:
        raise ValueError(f"cannot normalize vector: {v}")
    return a / n


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def safe_float(x: Any, default: float = math.nan) -> float:
    try:
        if x in ("", None):
            return default
        return float(x)
    except Exception:
        return default


def levela_to_raw_vec(v_xyz: Sequence[float]) -> list[float]:
    x, y, z = map(float, v_xyz)
    return [z, -x, y]


def raw_to_levela_vec(v_xyz: Sequence[float]) -> list[float]:
    x, y, z = map(float, v_xyz)
    return [-y, z, x]


def parse_float_list(s: str) -> list[float]:
    return [float(x.strip()) for x in str(s).replace(";", ",").split(",") if x.strip()]


def normseq(s: str | Sequence[str]) -> str:
    if isinstance(s, str):
        return " ".join(str(s).replace("-", " ").upper().split())
    return " ".join(str(x).upper() for x in s)


def read_live_t(path: Path) -> float:
    r = json.loads(path.read_text())
    for k in ("ut_s", "t_s", "spice_t_s", "et_s"):
        if k in r:
            return float(r[k])
    raise KeyError(f"could not find time in {path}")


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


# -----------------------------------------------------------------------------
# Body catalog / SPICE helpers
# -----------------------------------------------------------------------------

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
        for key in (
            "mu_m3_s2",
            "gm_m3_s2",
            "gravitational_parameter_m3_s2",
            "mu",
            "gm",
            "gravitational_parameter",
        ):
            if key in obj:
                val = float(obj[key])
                if val < 1e12:  # likely km^3/s^2
                    val *= 1e9
                return val
    raise RuntimeError(f"could not find mu/gm for {body} in {path}")


def load_spice_if_needed(bsp: Path | None, tpc: Path | None):
    if bsp is None and tpc is None:
        return
    if spice is None:
        raise RuntimeError(f"spiceypy not importable: {_SPICE_IMPORT_ERROR}")
    if tpc:
        spice.furnsh(str(tpc))
    if bsp:
        spice.furnsh(str(bsp))


def spice_body_velocity_levela_km_s(body: str, et_s: float, central_body: str) -> np.ndarray:
    if spice is None:
        raise RuntimeError("Need spiceypy + --bsp/--tpc when body velocity is absent from CSV.")
    st, _ = spice.spkezr(body.upper(), float(et_s), "J2000", "NONE", central_body.upper())
    return np.asarray(st[3:6], dtype=float)


# -----------------------------------------------------------------------------
# CSV candidate extraction
# -----------------------------------------------------------------------------

def candidate_seq(row: dict[str, str]) -> list[str]:
    if row.get("sequence_bodies"):
        return normseq(row["sequence_bodies"]).split()
    if row.get("sequence"):
        return normseq(row["sequence"]).split()

    events: list[tuple[int, str]] = []
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
        return [b for _, b in sorted(events)]
    return []


def candidate_epochs(row: dict[str, str], seq: Sequence[str]) -> list[float]:
    if row.get("epochs_et_s"):
        vals = parse_float_list(row["epochs_et_s"])
        if len(vals) >= len(seq):
            return vals[: len(seq)]

    out = []
    for i, body in enumerate(seq):
        exact = f"event{i}_{body.upper()}_et_s"
        if exact in row and row[exact] not in ("", None):
            out.append(float(row[exact]))
            continue
        # fallback any event{i}_*_et_s
        prefix = f"event{i}_"
        matches = [k for k in row if k.startswith(prefix) and k.endswith("_et_s") and row[k] not in ("", None)]
        if not matches:
            return []
        out.append(float(row[matches[0]]))
    return out


def vector_from_keys(row: dict[str, str], keys: Sequence[str]) -> np.ndarray | None:
    vals = []
    for k in keys:
        if k not in row or row[k] in ("", None):
            return None
        vals.append(float(row[k]))
    return np.asarray(vals, dtype=float)


def find_event_body_velocity_csv(row: dict[str, str], event_idx: int, body: str) -> np.ndarray | None:
    body = body.upper()
    candidates = [
        [f"event{event_idx}_{body}_vx_km_s", f"event{event_idx}_{body}_vy_km_s", f"event{event_idx}_{body}_vz_km_s"],
        [f"event{event_idx}_{body}_v_x_km_s", f"event{event_idx}_{body}_v_y_km_s", f"event{event_idx}_{body}_v_z_km_s"],
        [f"event{event_idx}_{body}_vel_x_km_s", f"event{event_idx}_{body}_vel_y_km_s", f"event{event_idx}_{body}_vel_z_km_s"],
        [f"event{event_idx}_vx_km_s", f"event{event_idx}_vy_km_s", f"event{event_idx}_vz_km_s"],
    ]
    for keys in candidates:
        v = vector_from_keys(row, keys)
        if v is not None:
            return v

    # Generic fuzzy fallback.
    lower_body = body.lower()
    vx_keys = []
    for k in row:
        kl = k.lower()
        if (
            f"event{event_idx}" in kl
            and lower_body in kl
            and ("vx" in kl or "v_x" in kl)
            and "km_s" in kl
        ):
            vx_keys.append(k)
    for vx in vx_keys:
        vy = vx.replace("vx", "vy").replace("v_x", "v_y")
        vz = vx.replace("vx", "vz").replace("v_x", "v_z")
        v = vector_from_keys(row, [vx, vy, vz])
        if v is not None:
            return v

    return None


def get_first_leg_vinf_levela_m_s(
    row: dict[str, str],
    seq: Sequence[str],
    epochs: Sequence[float],
    central_body: str,
) -> tuple[np.ndarray, str]:
    # Direct v∞ columns if present.
    direct_key_sets = [
        ["leg1_vinf_dep_x_km_s", "leg1_vinf_dep_y_km_s", "leg1_vinf_dep_z_km_s"],
        ["leg1_vinf_x_km_s", "leg1_vinf_y_km_s", "leg1_vinf_z_km_s"],
        ["departure_vinf_x_km_s", "departure_vinf_y_km_s", "departure_vinf_z_km_s"],
    ]
    for keys in direct_key_sets:
        v = vector_from_keys(row, keys)
        if v is not None:
            return 1000.0 * v, "csv_vinf"

    vdep = vector_from_keys(row, ["leg1_vdep_x_km_s", "leg1_vdep_y_km_s", "leg1_vdep_z_km_s"])
    if vdep is None:
        vdep = vector_from_keys(row, ["leg0_vdep_x_km_s", "leg0_vdep_y_km_s", "leg0_vdep_z_km_s"])
    if vdep is None:
        raise RuntimeError("missing leg1_vdep_* km/s columns")

    dep_body = seq[0].upper()
    t_dep = float(epochs[0])

    body_v = find_event_body_velocity_csv(row, 0, dep_body)
    source = "csv_vdep_minus_csv_body_v"
    if body_v is None:
        body_v = spice_body_velocity_levela_km_s(dep_body, t_dep, central_body)
        source = "csv_vdep_minus_spice_body_v"

    return 1000.0 * (vdep - body_v), source


# -----------------------------------------------------------------------------
# Kepler propagation for parking orbit
# -----------------------------------------------------------------------------

def stumpff_C(z: float) -> float:
    if z > 1e-8:
        sz = math.sqrt(z)
        return (1.0 - math.cos(sz)) / z
    if z < -1e-8:
        sz = math.sqrt(-z)
        return (math.cosh(sz) - 1.0) / (-z)
    # Series near zero.
    return 0.5 - z / 24.0 + z * z / 720.0 - z * z * z / 40320.0


def stumpff_S(z: float) -> float:
    if z > 1e-8:
        sz = math.sqrt(z)
        return (sz - math.sin(sz)) / (sz ** 3)
    if z < -1e-8:
        sz = math.sqrt(-z)
        return (math.sinh(sz) - sz) / (sz ** 3)
    return 1.0 / 6.0 - z / 120.0 + z * z / 5040.0 - z * z * z / 362880.0


def kepler_period_if_bound(r0: np.ndarray, v0: np.ndarray, mu: float) -> float | None:
    r = norm(r0)
    v2 = float(np.dot(v0, v0))
    energy = 0.5 * v2 - mu / r
    if energy >= 0:
        return None
    a = -mu / (2.0 * energy)
    return 2.0 * math.pi * math.sqrt(a ** 3 / mu)


def propagate_kepler_universal(r0: Sequence[float], v0: Sequence[float], dt_s: float, mu: float) -> tuple[np.ndarray, np.ndarray]:
    r0 = np.asarray(r0, dtype=float)
    v0 = np.asarray(v0, dtype=float)
    r0n = norm(r0)
    vr0 = float(np.dot(r0, v0)) / r0n
    alpha = 2.0 / r0n - float(np.dot(v0, v0)) / mu

    sqrt_mu = math.sqrt(mu)
    # Good first guess for elliptical orbits.
    if alpha > 1e-12:
        x = sqrt_mu * dt_s * alpha
    else:
        x = math.copysign(math.sqrt(abs(1.0 / max(abs(alpha), 1e-12))) * math.log(1 + abs(dt_s)), dt_s)

    # Newton solve universal anomaly.
    for _ in range(80):
        z = alpha * x * x
        C = stumpff_C(z)
        S = stumpff_S(z)
        F = (
            r0n * vr0 / sqrt_mu * x * x * C
            + (1.0 - alpha * r0n) * x ** 3 * S
            + r0n * x
            - sqrt_mu * dt_s
        )
        dF = (
            r0n * vr0 / sqrt_mu * x * (1.0 - alpha * x * x * S)
            + (1.0 - alpha * r0n) * x * x * C
            + r0n
        )
        if abs(dF) < 1e-12:
            break
        dx = F / dF
        x -= dx
        if abs(dx) < 1e-8:
            break

    z = alpha * x * x
    C = stumpff_C(z)
    S = stumpff_S(z)
    f = 1.0 - x * x / r0n * C
    g = dt_s - x ** 3 / sqrt_mu * S
    r = f * r0 + g * v0
    rn = norm(r)
    fdot = sqrt_mu / (rn * r0n) * (alpha * x ** 3 * S - x)
    gdot = 1.0 - x * x / rn * C
    v = fdot * r0 + gdot * v0
    return r, v


# -----------------------------------------------------------------------------
# Ejection reconstruction / ranking
# -----------------------------------------------------------------------------

def compute_ejection_dv(
    r_raw_m: Sequence[float],
    v_raw_m_s: Sequence[float],
    vinf_raw_m_s: Sequence[float],
    mu_m3_s2: float,
) -> dict[str, Any]:
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
    if norm(proj) < AXIS_EPS:
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

    eps_after = 0.5 * norm(v_after) ** 2 - mu_m3_s2 / rmag
    return {
        "rmag_km": rmag / 1000.0,
        "vmag_m_s": norm(v),
        "vinf_mag_m_s": vinf_mag,
        "vp_mag_m_s": vp_mag,
        "ecc": ecc,
        "nu_inf_deg": math.degrees(nu_inf),
        "theta_to_vinf_deg": math.degrees(theta),
        "phase_error_deg": math.degrees(phase_error),
        "phase_abs_deg": abs(math.degrees(phase_error)),
        "plane_angle_deg": plane_angle,
        "plane_abs_deg": abs(plane_angle),
        "dv_norm_m_s": norm(dv),
        "dv_tangent_m_s": float(dv_tnb[0]),
        "dv_normal_m_s": float(dv_tnb[1]),
        "dv_binormal_m_s": float(dv_tnb[2]),
        "out_of_plane_abs_m_s": abs(float(dv_tnb[2])),
        "out_of_plane_fraction": abs(float(dv_tnb[2])) / max(1e-9, norm(dv)),
        "post_burn_specific_energy_m2_s2": eps_after,
        "dv_raw_m_s": dv.tolist(),
        "dv_levela_m_s": raw_to_levela_vec(dv),
    }


def row_score(
    r: dict[str, Any],
    raw_sum_km_s: float,
    args: argparse.Namespace,
) -> float:
    # Lower is better. We emphasize operational executability first.
    over_binormal = max(0.0, r["out_of_plane_abs_m_s"] - args.max_binormal_m_s)
    over_phase = max(0.0, r["phase_abs_deg"] - args.max_phase_abs_deg)
    over_dv = max(0.0, r["dv_norm_m_s"] - args.max_dv_m_s)
    retro_penalty = max(0.0, -r["dv_tangent_m_s"])

    return (
        args.weight_binormal * r["out_of_plane_abs_m_s"]
        + args.weight_phase * r["phase_abs_deg"]
        + args.weight_dv * r["dv_norm_m_s"]
        + args.weight_raw_sum * (raw_sum_km_s * 1000.0 if math.isfinite(raw_sum_km_s) else 0.0)
        + args.penalty_binormal * over_binormal
        + args.penalty_phase * over_phase
        + args.penalty_dv * over_dv
        + args.penalty_retrograde * retro_penalty
    )


def candidate_metric_float(row: dict[str, str], keys: Sequence[str]) -> float:
    for k in keys:
        if k in row and row[k] not in ("", None):
            return safe_float(row[k])
    return math.nan

def json_sanitize(obj: Any) -> Any:
    """Convert NaN/Inf floats into None so jq and strict JSON consumers behave."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, np.floating):
        x = float(obj)
        return x if math.isfinite(x) else None
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, dict):
        return {str(k): json_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_sanitize(v) for v in obj]
    return obj


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)

    ap.add_argument("--candidate-csv", type=Path, required=True)
    ap.add_argument("--sequence", default=None, help='Optional exact sequence filter, e.g. "KERBIN EVE KERBIN JOOL"')
    ap.add_argument("--dep-body", default=None)
    ap.add_argument("--arr-body", default=None)
    ap.add_argument("--max-candidates", type=int, default=0, help="0 means all rows after filters.")

    ap.add_argument("--server", required=True)
    ap.add_argument("--plugin-b64", type=Path, required=True)
    ap.add_argument("--plugin-arg-mode", choices=["option", "positional"], default="positional")
    ap.add_argument("--vessel-guid", required=True)
    ap.add_argument("--live-state-json", type=Path, required=True)
    ap.add_argument("--body-catalog", type=Path, required=True)
    ap.add_argument("--server-timeout-s", type=float, default=600.0)
    ap.add_argument("--quiet-stderr", action="store_true")

    ap.add_argument("--bsp", type=Path, default=None)
    ap.add_argument("--tpc", type=Path, default=None)
    ap.add_argument("--central-body", default="SUN")

    ap.add_argument("--preburn-source", choices=["twobody", "live_only", "vrel"], default="twobody")
    ap.add_argument("--phase-offset-min-s", type=float, default=-7200.0)
    ap.add_argument("--phase-offset-max-s", type=float, default=7200.0)
    ap.add_argument("--phase-offset-step-s", type=float, default=120.0)
    ap.add_argument("--mod-period", action="store_true", default=True)
    ap.add_argument("--no-mod-period", action="store_false", dest="mod_period")

    ap.add_argument("--max-binormal-m-s", type=float, default=500.0)
    ap.add_argument("--max-phase-abs-deg", type=float, default=5.0)
    ap.add_argument("--max-dv-m-s", type=float, default=3200.0)
    ap.add_argument("--max-plane-angle-deg", type=float, default=5.0)

    ap.add_argument("--weight-binormal", type=float, default=1.0)
    ap.add_argument("--weight-phase", type=float, default=50.0)
    ap.add_argument("--weight-dv", type=float, default=0.05)
    ap.add_argument("--weight-raw-sum", type=float, default=0.02)
    ap.add_argument("--penalty-binormal", type=float, default=10.0)
    ap.add_argument("--penalty-phase", type=float, default=500.0)
    ap.add_argument("--penalty-dv", type=float, default=5.0)
    ap.add_argument("--penalty-retrograde", type=float, default=100.0)

    ap.add_argument("--top-n", type=int, default=50)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()

    if PrincipiaTargeterClient is None:
        raise SystemExit(f"Could not import principia_targeter_client: {_CLIENT_IMPORT_ERROR}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    load_spice_if_needed(args.bsp, args.tpc)

    target_sequence = normseq(args.sequence) if args.sequence else None
    dep_filter = args.dep_body.upper() if args.dep_body else None
    arr_filter = args.arr_body.upper() if args.arr_body else None

    live_t = read_live_t(args.live_state_json)
    rows = load_rows(args.candidate_csv)

    # Phase offsets.
    phase_offsets = []
    x = float(args.phase_offset_min_s)
    while x <= float(args.phase_offset_max_s) + 1e-9:
        phase_offsets.append(x)
        x += float(args.phase_offset_step_s)

    print("=== RANK PYKEP CANDIDATES BY DEPARTURE EXECUTABILITY V0 ===")
    print(f"candidate_csv     : {args.candidate_csv}")
    print(f"rows              : {len(rows)}")
    print(f"sequence filter   : {target_sequence or '(none)'}")
    print(f"dep/arr filter    : {dep_filter or '*'} -> {arr_filter or '*'}")
    print(f"live_t            : {live_t}")
    print(f"preburn_source    : {args.preburn_source}")
    print(f"phase offsets     : {len(phase_offsets)} ({args.phase_offset_min_s}..{args.phase_offset_max_s} step {args.phase_offset_step_s})")
    print(f"output_dir        : {args.output_dir}")

    results: list[dict[str, Any]] = []
    skipped: dict[str, int] = {}

    def skip(reason: str):
        skipped[reason] = skipped.get(reason, 0) + 1

    # Query live parking state once when using twobody/live_only.
    live_states_by_body: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    with PrincipiaTargeterClient(
        args.server,
        args.plugin_b64,
        plugin_arg_mode=args.plugin_arg_mode,
        response_timeout_s=args.server_timeout_s,
        quiet_stderr=args.quiet_stderr,
    ) as client:
        candidate_count = 0

        for row_index0, row in enumerate(rows):
            seq = candidate_seq(row)
            if len(seq) < 2:
                skip("no_sequence")
                continue

            seq_norm = normseq(seq)
            if target_sequence and seq_norm != target_sequence:
                skip("sequence_mismatch")
                continue

            dep = seq[0].upper()
            arr = seq[1].upper()
            if dep_filter and dep != dep_filter:
                skip("dep_mismatch")
                continue
            if arr_filter and arr != arr_filter:
                skip("arr_mismatch")
                continue

            epochs = candidate_epochs(row, seq)
            if len(epochs) < 2:
                skip("missing_epochs")
                continue

            t_dep = float(epochs[0])
            t_arr = float(epochs[1])
            burn_dt_nom = t_dep - live_t
            if burn_dt_nom + max(phase_offsets) < 0:
                skip("departure_in_past")
                continue

            try:
                mu = find_mu_from_catalog(args.body_catalog, dep)
                vinf_levela_m_s, vinf_source = get_first_leg_vinf_levela_m_s(row, seq, epochs, args.central_body)
                vinf_raw_m_s = np.asarray(levela_to_raw_vec(vinf_levela_m_s), dtype=float)
            except Exception as exc:
                skip("vinf_extract_failed")
                results.append({
                    "row_index0": row_index0,
                    "ok": False,
                    "skip_reason": f"vinf_extract_failed: {exc}",
                    "sequence": seq_norm,
                })
                continue

            raw_sum = candidate_metric_float(row, ["raw_sum_km_s", "raw_sum", "cost_km_s"])
            departure_vinf_csv = candidate_metric_float(row, ["departure_vinf_km_s", "dep_vinf_km_s"])
            cost = candidate_metric_float(row, ["cost", "score"])

            # Prepare state source.
            if args.preburn_source in ("twobody", "live_only"):
                if dep not in live_states_by_body:
                    try:
                        st0 = client.vrel(
                            f"live_{os.getpid()}_{dep}",
                            args.vessel_guid,
                            dep,
                            0.0,
                            [],
                            timeout_s=args.server_timeout_s,
                        )
                        live_states_by_body[dep] = (
                            np.asarray(st0["final_rel_r_raw_m"], dtype=float),
                            np.asarray(st0["final_rel_v_raw_m_s"], dtype=float),
                        )
                    except Exception as exc:
                        skip("live_state_failed")
                        results.append({
                            "row_index0": row_index0,
                            "ok": False,
                            "skip_reason": f"live_state_failed: {exc}",
                            "sequence": seq_norm,
                        })
                        continue
                r0, v0 = live_states_by_body[dep]
                period = kepler_period_if_bound(r0, v0, mu)
            else:
                period = None

            best: dict[str, Any] | None = None
            eval_count = 0
            errors = 0

            for off in phase_offsets:
                burn_dt = burn_dt_nom + off
                if burn_dt < 0:
                    continue

                try:
                    if args.preburn_source == "twobody":
                        r0, v0 = live_states_by_body[dep]
                        dt_prop = burn_dt
                        if args.mod_period and period and period > 0:
                            dt_prop = math.fmod(dt_prop, period)
                            if dt_prop < 0:
                                dt_prop += period
                        r, v = propagate_kepler_universal(r0, v0, dt_prop, mu)
                    elif args.preburn_source == "live_only":
                        r, v = live_states_by_body[dep]
                    else:
                        st = client.vrel(
                            f"cand_{os.getpid()}_{row_index0}_{eval_count}",
                            args.vessel_guid,
                            dep,
                            burn_dt,
                            [],
                            timeout_s=args.server_timeout_s,
                        )
                        r = np.asarray(st["final_rel_r_raw_m"], dtype=float)
                        v = np.asarray(st["final_rel_v_raw_m_s"], dtype=float)

                    e = compute_ejection_dv(r, v, vinf_raw_m_s, mu)

                    # Synthetic parking-state at burn epoch. This is the state
                    # the new VCAREL server command needs; it avoids long-coasting
                    # the serialized vessel for hundreds of days before burn0.
                    r_arr = np.asarray(r, dtype=float)
                    v_arr = np.asarray(v, dtype=float)
                    dv_arr = np.asarray(e["dv_raw_m_s"], dtype=float)
                    e["burn_rel_r_raw_m"] = r_arr.tolist()
                    e["burn_rel_v_raw_m_s"] = v_arr.tolist()
                    e["post_burn_rel_v_raw_m_s"] = (v_arr + dv_arr).tolist()

                    e["score_exec"] = row_score(e, raw_sum, args)
                    e["phase_offset_s"] = off
                    e["burn_dt_s"] = burn_dt
                    e["burn_abs_s"] = live_t + burn_dt
                    e["period_s"] = period
                    eval_count += 1

                    if best is None or e["score_exec"] < best["score_exec"]:
                        best = e
                except Exception:
                    errors += 1
                    continue

            if best is None:
                skip("no_phase_solution")
                results.append({
                    "row_index0": row_index0,
                    "ok": False,
                    "skip_reason": "no_phase_solution",
                    "sequence": seq_norm,
                    "t_dep_s": t_dep,
                    "burn_dt_nominal_s": burn_dt_nom,
                    "errors": errors,
                })
                continue

            pass_gates = (
                best["out_of_plane_abs_m_s"] <= args.max_binormal_m_s
                and best["phase_abs_deg"] <= args.max_phase_abs_deg
                and best["dv_norm_m_s"] <= args.max_dv_m_s
                and best["plane_abs_deg"] <= args.max_plane_angle_deg
                and best["dv_tangent_m_s"] > 0
            )

            result = {
                "ok": True,
                "pass_gates": pass_gates,
                "row_index0": row_index0,
                "sequence": seq_norm,
                "dep_body": dep,
                "arr_body": arr,
                "t_dep_s": t_dep,
                "t_arr_s": t_arr,
                "tof1_days": (t_arr - t_dep) / DAY_S,
                "burn_dt_nominal_s": burn_dt_nom,
                "vinf_source": vinf_source,
                "vinf_dep_levela_m_s": vinf_levela_m_s.tolist(),
                "vinf_dep_raw_m_s": vinf_raw_m_s.tolist(),
                "vinf_dep_norm_m_s": norm(vinf_raw_m_s),
                "raw_sum_km_s": raw_sum,
                "departure_vinf_km_s_csv": departure_vinf_csv,
                "cost": cost,
                "leg_paths": row.get("leg_paths", ""),
                "candidate_id": row.get("candidate_id", ""),
                "rank_field": row.get("rank", ""),
                "phase_evals": eval_count,
                "phase_errors": errors,
                **best,
            }
            results.append(result)
            candidate_count += 1
            if args.max_candidates and candidate_count >= args.max_candidates:
                break

    ok_results = [r for r in results if r.get("ok")]
    ok_results.sort(key=lambda r: (not r.get("pass_gates", False), r["score_exec"], r["out_of_plane_abs_m_s"], r["phase_abs_deg"]))

    print("")
    print("=== TOP EXECUTABLE CANDIDATES ===")
    print("rank pass row seq raw_sum vinf dv T B phase plane burn_off score")
    for i, r in enumerate(ok_results[: args.top_n], start=1):
        print(
            f"{i:3d} {str(r['pass_gates']):5s} "
            f"{r['row_index0']:5d} "
            f"{r['sequence']:<28} "
            f"{r['raw_sum_km_s'] if math.isfinite(r['raw_sum_km_s']) else float('nan'):7.3f} "
            f"{r['vinf_dep_norm_m_s']/1000:6.3f} "
            f"{r['dv_norm_m_s']:7.1f} "
            f"{r['dv_tangent_m_s']:7.1f} "
            f"{r['dv_binormal_m_s']:8.1f} "
            f"{r['phase_error_deg']:7.3f} "
            f"{r['plane_angle_deg']:7.3f} "
            f"{r['phase_offset_s']:8.1f} "
            f"{r['score_exec']:9.1f}"
        )

    out_json = {
        "schema": "pykep_candidate_departure_executability_rank_v0_1",
        "config": {
            "candidate_csv": str(args.candidate_csv),
            "sequence_filter": target_sequence,
            "dep_filter": dep_filter,
            "arr_filter": arr_filter,
            "live_t_s": live_t,
            "preburn_source": args.preburn_source,
            "phase_offset_min_s": args.phase_offset_min_s,
            "phase_offset_max_s": args.phase_offset_max_s,
            "phase_offset_step_s": args.phase_offset_step_s,
            "max_binormal_m_s": args.max_binormal_m_s,
            "max_phase_abs_deg": args.max_phase_abs_deg,
            "max_dv_m_s": args.max_dv_m_s,
            "max_plane_angle_deg": args.max_plane_angle_deg,
        },
        "skipped_counts": skipped,
        "n_results": len(results),
        "n_ok": len(ok_results),
        "n_pass_gates": sum(1 for r in ok_results if r.get("pass_gates")),
        "best": ok_results[0] if ok_results else None,
        "top": ok_results[: args.top_n],
    }

    json_path = args.output_dir / "candidate_departure_executability_rank.json"
    csv_path = args.output_dir / "candidate_departure_executability_rank.csv"

    # Flatten for CSV.
    flat_rows = []
    for r in ok_results:
        rr: dict[str, Any] = {}
        for k, v in r.items():
            if isinstance(v, list):
                for j, x in enumerate(v):
                    rr[f"{k}_{j}"] = x
            else:
                rr[k] = v
        flat_rows.append(rr)

    if flat_rows:
        fieldnames = sorted({k for r in flat_rows for k in r.keys()})
        with csv_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(flat_rows)

    json_path.write_text(json.dumps(json_sanitize(out_json), indent=2) + "\n")
    print(f"[OK] wrote {json_path}")
    print(f"[OK] wrote {csv_path}")

    if not ok_results:
        print("[WARN] no ok results; inspect skipped_counts in JSON.")
    elif out_json["n_pass_gates"] == 0:
        print("[WARN] no candidate passed gates. Loosen gates or run broader candidate CSV / sequences.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
