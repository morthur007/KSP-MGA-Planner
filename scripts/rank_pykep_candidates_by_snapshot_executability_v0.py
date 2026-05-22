#!/usr/bin/env python3
"""
rank_pykep_candidates_by_snapshot_executability_v0.py

Snapshot-aware replacement for rank_pykep_candidates_by_departure_executability_v0_1.py.

Purpose:
  Rank PyKEP/Lambert candidate rows by whether the FIRST departure v∞ is
  executable from the current live Principia navigation snapshot.

Important changes vs v0_1:
  - Does NOT query the old serialized vessel from .b64.
  - Does NOT use VREL to obtain the live parking state.
  - Reads the current vessel state directly from
      principia_live_navigation_snapshot_v0_1.json
  - Uses the snapshot's corrected raw pipeline frame:
      vessel.rel_r_raw_m
      vessel.rel_v_raw_m_s
  - Scores candidates by two-body parking-orbit ejection feasibility.
  - Uses the Principia/insert_navigation TNB convention validated by SNAPVCA_NAV:
      T = unit(v)
      H = unit(r x v)
      N = unit(H x T)
      B = unit(N x T) = -H
    This matters because the old ranker used B=+H, while the UI/SNAPVCA_NAV
    diagnostics showed the operational binormal has the opposite sign.

This script is deliberately fast and safe. It does not validate every candidate
with Principia N-body. It finds plausible rows first. After this ranker, use a
SNAPVCA_NAV validator/refiner only on the top rows that pass gates.

Typical use:

python scripts/rank_pykep_candidates_by_snapshot_executability_v0.py \
  --candidate-csv data/runs/family_search_smoke/candidate_seed.csv \
  --snapshot-json "/home/matheus/.steam/steam/steamapps/common/Kerbal Space Program/GameData/MGAPlanner/principia_live_navigation_snapshot_v0_1.json" \
  --body-catalog data/catalogs/jnsq/body_catalog.json \
  --bsp data/kernels/jnsq/v0_3/jnsq_principia_native_v0_3_1000y_1h_deg9.bsp \
  --tpc data/kernels/jnsq/v0_3/jnsq_principia_native_v0_3_1000y_1h_deg9.ids.tpc \
  --central-body Sun \
  --dep-body KERBIN \
  --phase-offset-min-s -21600 \
  --phase-offset-max-s 21600 \
  --phase-offset-step-s 60 \
  --max-normal-m-s 350 \
  --max-binormal-m-s 500 \
  --max-out-of-plane-fraction 0.30 \
  --max-phase-abs-deg 8 \
  --max-dv-m-s 4200 \
  --max-plane-angle-deg 8 \
  --top-n 100 \
  --output-dir data/runs/game_export/current_orbit/snap_candidate_hunt01
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
        raise ValueError(f"cannot normalize vector: {v!r}")
    return a / n


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def safe_float(x: Any, default: float = math.nan) -> float:
    try:
        if x in ("", None):
            return default
        y = float(x)
        return y if math.isfinite(y) else default
    except Exception:
        return default


def parse_float_list(s: str) -> list[float]:
    return [float(x.strip()) for x in str(s).replace(";", ",").split(",") if x.strip()]


def normseq(s: str | Sequence[str]) -> str:
    if isinstance(s, str):
        return " ".join(str(s).replace("-", " ").upper().split())
    return " ".join(str(x).upper() for x in s)


def levela_to_raw_vec(v_xyz: Sequence[float]) -> list[float]:
    # LevelA/SPICE canonical -> Principia raw pipeline:
    # (X,Y,Z) -> (+Z,-X,+Y)
    x, y, z = map(float, v_xyz)
    return [z, -x, y]


def raw_to_levela_vec(v_xyz: Sequence[float]) -> list[float]:
    x, y, z = map(float, v_xyz)
    return [-y, z, x]


def json_sanitize(obj: Any) -> Any:
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, np.floating):
        x = float(obj)
        return x if math.isfinite(x) else None
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return [json_sanitize(x) for x in obj.tolist()]
    if isinstance(obj, dict):
        return {str(k): json_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_sanitize(v) for v in obj]
    return obj


# -----------------------------------------------------------------------------
# Snapshot helpers
# -----------------------------------------------------------------------------

def load_snapshot(path: Path) -> dict[str, Any]:
    d = json.loads(path.read_text())
    vessel = d.get("vessel", d)

    for k in ("t_game_s", "rel_r_raw_m", "rel_v_raw_m_s"):
        if k not in vessel:
            raise RuntimeError(f"snapshot vessel missing {k!r}: {path}")

    nav_body = str(vessel.get("nav_body") or d.get("active_body", {}).get("name") or "").upper()
    if not nav_body:
        raise RuntimeError(f"snapshot missing vessel.nav_body/active_body.name: {path}")

    return {
        "schema": d.get("schema"),
        "path": str(path),
        "source": d.get("source"),
        "t_game_s": float(vessel["t_game_s"]),
        "t_spice_s": float(vessel.get("t_spice_s", vessel["t_game_s"])),
        "vessel_guid": vessel.get("vessel_guid", ""),
        "vessel_name": vessel.get("vessel_name", ""),
        "nav_body": nav_body,
        "rel_r_raw_m": [float(x) for x in vessel["rel_r_raw_m"]],
        "rel_v_raw_m_s": [float(x) for x in vessel["rel_v_raw_m_s"]],
        "mass_tonnes": safe_float(vessel.get("mass_tonnes"), math.nan),
        "available_thrust_kN": safe_float(vessel.get("available_thrust_kN"), math.nan),
        "specific_impulse_s_g0": safe_float(vessel.get("specific_impulse_s_g0"), math.nan),
        "state_source": vessel.get("state_source"),
        "frame_fix": vessel.get("frame_fix"),
        "principia_basis": vessel.get("principia_basis"),
    }


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
                # If it is small, assume km^3/s^2 and convert.
                if val < 1e12:
                    val *= 1e9
                return val

    raise RuntimeError(f"could not find mu/gm for {body} in {path}")


def load_spice_if_needed(bsp: Path | None, tpc: Path | None) -> None:
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

def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


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

    out: list[float] = []
    for i, body in enumerate(seq):
        exact = f"event{i}_{body.upper()}_et_s"
        if exact in row and row[exact] not in ("", None):
            out.append(float(row[exact]))
            continue

        prefix = f"event{i}_"
        matches = [
            k for k in row
            if k.startswith(prefix) and k.endswith("_et_s") and row[k] not in ("", None)
        ]
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


def candidate_metric_float(row: dict[str, str], keys: Sequence[str]) -> float:
    for k in keys:
        if k in row and row[k] not in ("", None):
            return safe_float(row[k])
    return math.nan


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
    return 0.5 - z / 24.0 + z * z / 720.0 - z * z * z / 40320.0


def stumpff_S(z: float) -> float:
    if z > 1e-8:
        sz = math.sqrt(z)
        return (sz - math.sin(sz)) / (sz**3)
    if z < -1e-8:
        sz = math.sqrt(-z)
        return (math.sinh(sz) - sz) / (sz**3)
    return 1.0 / 6.0 - z / 120.0 + z * z / 5040.0 - z * z * z / 362880.0


def kepler_period_if_bound(r0: np.ndarray, v0: np.ndarray, mu: float) -> float | None:
    r = norm(r0)
    v2 = float(np.dot(v0, v0))
    energy = 0.5 * v2 - mu / r
    if energy >= 0:
        return None
    a = -mu / (2.0 * energy)
    return 2.0 * math.pi * math.sqrt(a**3 / mu)


def propagate_kepler_universal(
    r0: Sequence[float],
    v0: Sequence[float],
    dt_s: float,
    mu: float,
) -> tuple[np.ndarray, np.ndarray]:
    r0 = np.asarray(r0, dtype=float)
    v0 = np.asarray(v0, dtype=float)
    r0n = norm(r0)
    vr0 = float(np.dot(r0, v0)) / r0n
    alpha = 2.0 / r0n - float(np.dot(v0, v0)) / mu

    sqrt_mu = math.sqrt(mu)
    if alpha > 1e-12:
        x = sqrt_mu * dt_s * alpha
    else:
        x = math.copysign(
            math.sqrt(abs(1.0 / max(abs(alpha), 1e-12))) * math.log(1 + abs(dt_s)),
            dt_s,
        )

    for _ in range(80):
        z = alpha * x * x
        C = stumpff_C(z)
        S = stumpff_S(z)
        F = (
            r0n * vr0 / sqrt_mu * x * x * C
            + (1.0 - alpha * r0n) * x**3 * S
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
    g = dt_s - x**3 / sqrt_mu * S
    r = f * r0 + g * v0
    rn = norm(r)
    fdot = sqrt_mu / (rn * r0n) * (alpha * x**3 * S - x)
    gdot = 1.0 - x * x / rn * C
    v = fdot * r0 + gdot * v0
    return r, v


# -----------------------------------------------------------------------------
# Ejection reconstruction / ranking
# -----------------------------------------------------------------------------

def tnb_principia_navigation_basis(r_raw_m: Sequence[float], v_raw_m_s: Sequence[float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """TNB convention matching SNAPVCA_NAV/insert_navigation diagnostics."""
    r = np.asarray(r_raw_m, dtype=float)
    v = np.asarray(v_raw_m_s, dtype=float)

    T = unit(v)
    H = unit(np.cross(r, v))
    N = unit(np.cross(H, T))
    B = unit(np.cross(N, T))  # equals -H
    return T, N, B


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

    # Periapsis tangential direction that yields the requested asymptote.
    proj = s - float(np.dot(s, rhat)) * rhat
    if norm(proj) < AXIS_EPS:
        proj = v - float(np.dot(v, rhat)) * rhat
    t_hat = unit(proj)

    v_after = vp_mag * t_hat
    dv = v_after - v

    T, N, B = tnb_principia_navigation_basis(r, v)
    dv_tnb = np.array([np.dot(dv, T), np.dot(dv, N), np.dot(dv, B)], dtype=float)

    h_hat = unit(np.cross(r, v))
    plane_angle = math.degrees(math.asin(clamp(float(np.dot(h_hat, s)), -1.0, 1.0)))
    eps_after = 0.5 * norm(v_after) ** 2 - mu_m3_s2 / rmag

    out_of_plane = math.sqrt(float(dv_tnb[1]) ** 2 + float(dv_tnb[2]) ** 2)

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
        "normal_abs_m_s": abs(float(dv_tnb[1])),
        "binormal_abs_m_s": abs(float(dv_tnb[2])),
        "out_of_plane_abs_m_s": out_of_plane,
        "out_of_plane_fraction": out_of_plane / max(1e-9, norm(dv)),

        "post_burn_specific_energy_m2_s2": eps_after,
        "dv_raw_m_s": dv.tolist(),
        "dv_levela_m_s": raw_to_levela_vec(dv),
        "tangent_raw": T.tolist(),
        "normal_raw": N.tolist(),
        "binormal_raw": B.tolist(),
    }


def row_score(r: dict[str, Any], raw_sum_km_s: float, args: argparse.Namespace) -> float:
    over_normal = max(0.0, r["normal_abs_m_s"] - args.max_normal_m_s)
    over_binormal = max(0.0, r["binormal_abs_m_s"] - args.max_binormal_m_s)
    over_oop_frac = max(0.0, r["out_of_plane_fraction"] - args.max_out_of_plane_fraction)
    over_phase = max(0.0, r["phase_abs_deg"] - args.max_phase_abs_deg)
    over_dv = max(0.0, r["dv_norm_m_s"] - args.max_dv_m_s)
    over_plane = max(0.0, r["plane_abs_deg"] - args.max_plane_angle_deg)
    retro_penalty = max(0.0, -r["dv_tangent_m_s"])
    wait_days = max(0.0, r["burn_dt_s"] / DAY_S)

    return (
        args.weight_out_of_plane * r["out_of_plane_abs_m_s"]
        + args.weight_binormal * r["binormal_abs_m_s"]
        + args.weight_normal * r["normal_abs_m_s"]
        + args.weight_phase * r["phase_abs_deg"]
        + args.weight_dv * r["dv_norm_m_s"]
        + args.weight_raw_sum * (raw_sum_km_s * 1000.0 if math.isfinite(raw_sum_km_s) else 0.0)
        + args.weight_wait_days * wait_days
        + args.penalty_normal * over_normal
        + args.penalty_binormal * over_binormal
        + args.penalty_oop_fraction * over_oop_frac
        + args.penalty_phase * over_phase
        + args.penalty_dv * over_dv
        + args.penalty_plane * over_plane
        + args.penalty_retrograde * retro_penalty
    )


def flatten_for_csv(row: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in row.items():
        if isinstance(v, np.ndarray):
            v = v.tolist()
        if isinstance(v, list):
            if all(not isinstance(x, (dict, list, tuple)) for x in v):
                for i, x in enumerate(v):
                    out[f"{k}_{i}"] = x
            else:
                out[k] = json.dumps(json_sanitize(v), ensure_ascii=False)
        elif isinstance(v, dict):
            out[k] = json.dumps(json_sanitize(v), ensure_ascii=False)
        else:
            out[k] = json_sanitize(v)
    return out


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)

    ap.add_argument("--candidate-csv", type=Path, required=True)
    ap.add_argument("--snapshot-json", type=Path, required=True)
    ap.add_argument("--body-catalog", type=Path, required=True)

    ap.add_argument("--sequence", default=None, help='Optional exact sequence filter, e.g. "KERBIN DUNA KERBIN JOOL"')
    ap.add_argument("--dep-body", default=None)
    ap.add_argument("--arr-body", default=None)
    ap.add_argument("--max-candidates", type=int, default=0, help="0 means all rows after filters.")

    ap.add_argument("--bsp", type=Path, default=None)
    ap.add_argument("--tpc", type=Path, default=None)
    ap.add_argument("--central-body", default="SUN")

    ap.add_argument("--preburn-source", choices=["twobody", "live_only"], default="twobody")
    ap.add_argument("--phase-offset-min-s", type=float, default=-7200.0)
    ap.add_argument("--phase-offset-max-s", type=float, default=7200.0)
    ap.add_argument("--phase-offset-step-s", type=float, default=120.0)
    ap.add_argument("--mod-period", action="store_true", default=True)
    ap.add_argument("--no-mod-period", action="store_false", dest="mod_period")

    ap.add_argument("--max-normal-m-s", type=float, default=350.0)
    ap.add_argument("--max-binormal-m-s", type=float, default=500.0)
    ap.add_argument("--max-out-of-plane-fraction", type=float, default=0.30)
    ap.add_argument("--max-phase-abs-deg", type=float, default=5.0)
    ap.add_argument("--max-dv-m-s", type=float, default=3200.0)
    ap.add_argument("--max-plane-angle-deg", type=float, default=5.0)
    ap.add_argument("--max-wait-days", type=float, default=0.0, help="0 disables wait filter.")

    ap.add_argument("--weight-out-of-plane", type=float, default=1.0)
    ap.add_argument("--weight-binormal", type=float, default=0.5)
    ap.add_argument("--weight-normal", type=float, default=0.25)
    ap.add_argument("--weight-phase", type=float, default=50.0)
    ap.add_argument("--weight-dv", type=float, default=0.05)
    ap.add_argument("--weight-raw-sum", type=float, default=0.02)
    ap.add_argument("--weight-wait-days", type=float, default=0.0)

    ap.add_argument("--penalty-normal", type=float, default=10.0)
    ap.add_argument("--penalty-binormal", type=float, default=10.0)
    ap.add_argument("--penalty-oop-fraction", type=float, default=20000.0)
    ap.add_argument("--penalty-phase", type=float, default=500.0)
    ap.add_argument("--penalty-dv", type=float, default=5.0)
    ap.add_argument("--penalty-plane", type=float, default=200.0)
    ap.add_argument("--penalty-retrograde", type=float, default=100.0)

    ap.add_argument("--top-n", type=int, default=50)
    ap.add_argument("--output-dir", type=Path, required=True)

    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    load_spice_if_needed(args.bsp, args.tpc)

    snapshot = load_snapshot(args.snapshot_json)
    live_t = float(snapshot["t_game_s"])

    target_sequence = normseq(args.sequence) if args.sequence else None
    dep_filter = args.dep_body.upper() if args.dep_body else None
    arr_filter = args.arr_body.upper() if args.arr_body else None

    rows = load_rows(args.candidate_csv)

    phase_offsets: list[float] = []
    x = float(args.phase_offset_min_s)
    while x <= float(args.phase_offset_max_s) + 1e-9:
        phase_offsets.append(x)
        x += float(args.phase_offset_step_s)

    print("=== RANK PYKEP CANDIDATES BY SNAPSHOT EXECUTABILITY V0 ===")
    print(f"candidate_csv     : {args.candidate_csv}")
    print(f"rows              : {len(rows)}")
    print(f"snapshot_json     : {args.snapshot_json}")
    print(f"snapshot_t        : {live_t}")
    print(f"snapshot_nav_body : {snapshot['nav_body']}")
    print(f"snapshot_r_km     : {norm(snapshot['rel_r_raw_m'])/1000.0:.6f}")
    print(f"snapshot_v_m_s    : {norm(snapshot['rel_v_raw_m_s']):.6f}")
    print(f"sequence filter   : {target_sequence or '(none)'}")
    print(f"dep/arr filter    : {dep_filter or '*'} -> {arr_filter or '*'}")
    print(f"preburn_source    : {args.preburn_source}")
    print(f"phase offsets     : {len(phase_offsets)} ({args.phase_offset_min_s}..{args.phase_offset_max_s} step {args.phase_offset_step_s})")
    print(f"output_dir        : {args.output_dir}")

    skipped: dict[str, int] = {}
    results: list[dict[str, Any]] = []

    def skip(reason: str) -> None:
        skipped[reason] = skipped.get(reason, 0) + 1

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

        if dep != snapshot["nav_body"]:
            skip("dep_not_snapshot_nav_body")
            results.append({
                "row_index0": row_index0,
                "ok": False,
                "skip_reason": f"dep_not_snapshot_nav_body:{dep}!={snapshot['nav_body']}",
                "sequence": seq_norm,
            })
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

        if args.max_wait_days and burn_dt_nom - max(abs(args.phase_offset_min_s), abs(args.phase_offset_max_s)) > args.max_wait_days * DAY_S:
            skip("departure_too_far_future")
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

        r0 = np.asarray(snapshot["rel_r_raw_m"], dtype=float)
        v0 = np.asarray(snapshot["rel_v_raw_m_s"], dtype=float)
        period = kepler_period_if_bound(r0, v0, mu)

        best: dict[str, Any] | None = None
        eval_count = 0
        errors = 0

        for off in phase_offsets:
            burn_dt = burn_dt_nom + off
            if burn_dt < 0:
                continue

            try:
                if args.preburn_source == "twobody":
                    dt_prop = burn_dt
                    if args.mod_period and period and period > 0:
                        dt_prop = math.fmod(dt_prop, period)
                        if dt_prop < 0:
                            dt_prop += period
                    r, v = propagate_kepler_universal(r0, v0, dt_prop, mu)
                else:
                    r, v = r0, v0

                e = compute_ejection_dv(r, v, vinf_raw_m_s, mu)

                r_arr = np.asarray(r, dtype=float)
                v_arr = np.asarray(v, dtype=float)
                dv_arr = np.asarray(e["dv_raw_m_s"], dtype=float)

                e["burn_rel_r_raw_m"] = r_arr.tolist()
                e["burn_rel_v_raw_m_s"] = v_arr.tolist()
                e["post_burn_rel_v_raw_m_s"] = (v_arr + dv_arr).tolist()
                e["phase_offset_s"] = off
                e["burn_dt_s"] = burn_dt
                e["burn_abs_s"] = live_t + burn_dt
                e["period_s"] = period
                e["score_exec"] = row_score(e, raw_sum, args)
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

        wait_days = max(0.0, best["burn_dt_s"] / DAY_S)
        pass_gates = (
            best["normal_abs_m_s"] <= args.max_normal_m_s
            and best["binormal_abs_m_s"] <= args.max_binormal_m_s
            and best["out_of_plane_fraction"] <= args.max_out_of_plane_fraction
            and best["phase_abs_deg"] <= args.max_phase_abs_deg
            and best["dv_norm_m_s"] <= args.max_dv_m_s
            and best["plane_abs_deg"] <= args.max_plane_angle_deg
            and best["dv_tangent_m_s"] > 0
            and (not args.max_wait_days or wait_days <= args.max_wait_days)
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
            "wait_days_best": wait_days,
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
    ok_results.sort(
        key=lambda r: (
            not r.get("pass_gates", False),
            r["score_exec"],
            r["out_of_plane_abs_m_s"],
            r["phase_abs_deg"],
            r["wait_days_best"],
        )
    )

    print("")
    print("=== TOP SNAPSHOT-EXECUTABLE CANDIDATES ===")
    print("rank pass row seq raw_sum vinf dv T N B oop phase plane wait_d burn_off score")
    for i, r in enumerate(ok_results[: args.top_n], start=1):
        print(
            f"{i:3d} {str(r['pass_gates']):5s} "
            f"{r['row_index0']:5d} "
            f"{r['sequence']:<32} "
            f"{r['raw_sum_km_s'] if math.isfinite(r['raw_sum_km_s']) else float('nan'):7.3f} "
            f"{r['vinf_dep_norm_m_s']/1000:6.3f} "
            f"{r['dv_norm_m_s']:7.1f} "
            f"{r['dv_tangent_m_s']:7.1f} "
            f"{r['dv_normal_m_s']:7.1f} "
            f"{r['dv_binormal_m_s']:8.1f} "
            f"{r['out_of_plane_fraction']:5.3f} "
            f"{r['phase_error_deg']:7.3f} "
            f"{r['plane_angle_deg']:7.3f} "
            f"{r['wait_days_best']:7.2f} "
            f"{r['phase_offset_s']:8.1f} "
            f"{r['score_exec']:9.1f}"
        )

    out_json = {
        "schema": "pykep_candidate_snapshot_departure_executability_rank_v0",
        "snapshot": snapshot,
        "config": {
            "candidate_csv": str(args.candidate_csv),
            "snapshot_json": str(args.snapshot_json),
            "sequence_filter": target_sequence,
            "dep_filter": dep_filter,
            "arr_filter": arr_filter,
            "live_t_s": live_t,
            "preburn_source": args.preburn_source,
            "phase_offset_min_s": args.phase_offset_min_s,
            "phase_offset_max_s": args.phase_offset_max_s,
            "phase_offset_step_s": args.phase_offset_step_s,
            "max_normal_m_s": args.max_normal_m_s,
            "max_binormal_m_s": args.max_binormal_m_s,
            "max_out_of_plane_fraction": args.max_out_of_plane_fraction,
            "max_phase_abs_deg": args.max_phase_abs_deg,
            "max_dv_m_s": args.max_dv_m_s,
            "max_plane_angle_deg": args.max_plane_angle_deg,
            "max_wait_days": args.max_wait_days,
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

    flat_rows = [flatten_for_csv(r) for r in ok_results]
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
        print("[WARN] no candidate passed gates. Loosen gates, broaden candidate CSV, or change parking orbit.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
