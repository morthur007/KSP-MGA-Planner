#!/usr/bin/env python3
"""
future_snapshot_family_validator.py

Compare a future KSP/Principia snapshot against a synthetic SPK and separate:
  1) absolute heliocentric error of each body;
  2) relative child-parent error inside dynamical families.

This is intended for long-horizon validation of the REBOUND/SPICE V0.1 kernel.
It does NOT require KSP to be running. It compares a snapshot JSON already captured
from the game to a BSP/TPC pair.

Outputs:
  body_absolute_errors.csv/.json
  family_relative_errors.csv/.json
  validation_manifest.json

Assumptions:
  - Snapshot states are in SI units: meters and meters/second.
  - SPICE returns km and km/s; this script converts SPICE to SI.
  - Snapshot body states are expressed relative to the same central/reference body
    used by the SPK validation path, typically Sun/Kerbol.

The JSON parser is intentionally permissive because our snapshot files evolved
through several pipeline versions.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

try:
    import spiceypy as spice
except Exception as exc:  # pragma: no cover
    raise SystemExit("spiceypy is required: pip install spiceypy") from exc


Vec6 = np.ndarray


@dataclass
class Snapshot:
    path: Path
    epoch_et_s: float
    epoch_ut_s: Optional[float]
    reference_body: Optional[str]
    states: Dict[str, Vec6]
    raw: Dict[str, Any]


def norm3(v: np.ndarray) -> float:
    return float(np.linalg.norm(v[:3]))


def normv(v: np.ndarray) -> float:
    return float(np.linalg.norm(v[3:6]))


def maybe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        y = float(x)
        if not math.isfinite(y):
            return None
        return y
    except Exception:
        return None


def find_epoch(raw: Dict[str, Any]) -> Tuple[float, Optional[float]]:
    """Return (ET seconds, UT seconds if known)."""
    # Direct ET-like keys first.
    for key in (
        "epoch_et_seconds",
        "epoch_et_s",
        "start_et_seconds",
        "start_et_s",
        "et_seconds",
        "et_s",
        "epoch_et",
        "start_et",
    ):
        val = maybe_float(raw.get(key))
        if val is not None:
            ut = None
            for uk in ("epoch_ut_seconds", "start_ut_seconds", "ut_seconds", "start_ut_s", "epoch_ut_s"):
                uval = maybe_float(raw.get(uk))
                if uval is not None:
                    ut = uval
                    break
            return val, ut

    # UT + offset.
    ut = None
    for uk in ("epoch_ut_seconds", "start_ut_seconds", "ut_seconds", "start_ut_s", "epoch_ut_s"):
        uval = maybe_float(raw.get(uk))
        if uval is not None:
            ut = uval
            break
    if ut is not None:
        off = 0.0
        for ok in ("et_offset_seconds", "ut_to_et_offset_seconds", "tdb_minus_ut_seconds"):
            oval = maybe_float(raw.get(ok))
            if oval is not None:
                off = oval
                break
        return ut + off, ut

    # Nested metadata fallback.
    for container_key in ("metadata", "manifest", "time", "epoch"):
        sub = raw.get(container_key)
        if isinstance(sub, dict):
            try:
                et, ut2 = find_epoch(sub)
                return et, ut2
            except Exception:
                pass

    raise ValueError("Could not find epoch ET/UT in snapshot JSON")


def as_vec3(x: Any) -> Optional[np.ndarray]:
    if isinstance(x, (list, tuple)) and len(x) >= 3:
        vals = [maybe_float(x[i]) for i in range(3)]
        if all(v is not None for v in vals):
            return np.array(vals, dtype=float)
    return None


def as_vec6(x: Any) -> Optional[Vec6]:
    if isinstance(x, (list, tuple)) and len(x) >= 6:
        vals = [maybe_float(x[i]) for i in range(6)]
        if all(v is not None for v in vals):
            return np.array(vals, dtype=float)
    return None


def parse_state_from_obj(obj: Any) -> Optional[Vec6]:
    """Attempt to parse one body's 6-vector from a dict/list."""
    v6 = as_vec6(obj)
    if v6 is not None:
        return v6
    if not isinstance(obj, dict):
        return None

    # Direct state keys.
    for key in ("state", "state_m", "state_si", "state_mks", "cartesian_state", "cartesian_state_m"):
        v6 = as_vec6(obj.get(key))
        if v6 is not None:
            return v6

    # Position/velocity arrays.
    pos = None
    vel = None
    for pk in ("position", "position_m", "r", "r_m", "pos", "pos_m", "xyz_m"):
        pos = as_vec3(obj.get(pk))
        if pos is not None:
            break
    for vk in ("velocity", "velocity_m_s", "v", "v_m_s", "vel", "vel_m_s", "vxyz_m_s"):
        vel = as_vec3(obj.get(vk))
        if vel is not None:
            break
    if pos is not None and vel is not None:
        return np.concatenate([pos, vel]).astype(float)

    # Scalar fields.
    scalar_sets = [
        ("x_m", "y_m", "z_m", "vx_m_s", "vy_m_s", "vz_m_s"),
        ("x", "y", "z", "vx", "vy", "vz"),
        ("px", "py", "pz", "vx", "vy", "vz"),
    ]
    for keys in scalar_sets:
        vals = [maybe_float(obj.get(k)) for k in keys]
        if all(v is not None for v in vals):
            return np.array(vals, dtype=float)

    return None


def normalize_name(name: str) -> str:
    return str(name).strip().replace("^N", "")


def collect_states(raw: Dict[str, Any]) -> Dict[str, Vec6]:
    """Collect states using several known schema variants."""
    states: Dict[str, Vec6] = {}

    def add(name: Any, obj: Any) -> None:
        if name is None:
            return
        n = normalize_name(str(name))
        if not n:
            return
        st = parse_state_from_obj(obj)
        if st is not None and st.shape == (6,):
            states[n] = st.astype(float)

    # Common dict containers.
    for key in ("states", "body_states", "bodies", "snapshot", "ephemerides"):
        cont = raw.get(key)
        if isinstance(cont, dict):
            for name, obj in cont.items():
                if isinstance(obj, dict) and "states" in obj and isinstance(obj["states"], list):
                    # Ephemerides-like list rows [t,x,y,z,vx,vy,vz]. Take first row.
                    rows = obj["states"]
                    if rows and isinstance(rows[0], (list, tuple)) and len(rows[0]) >= 7:
                        row = rows[0]
                        add(name, row[1:7])
                    else:
                        add(name, obj)
                else:
                    add(name, obj)

    # Common list containers.
    for key in ("bodies", "states", "body_states"):
        cont = raw.get(key)
        if isinstance(cont, list):
            for obj in cont:
                if isinstance(obj, dict):
                    name = None
                    for nk in ("name", "body", "body_name", "id"):
                        if nk in obj:
                            name = obj[nk]
                            break
                    add(name, obj)

    # If top-level dict has body names as keys.
    for name, obj in raw.items():
        if isinstance(obj, (dict, list, tuple)):
            add(name, obj)

    if not states:
        raise ValueError("Could not parse any body states from snapshot JSON")
    return states


def read_snapshot(path: str | Path) -> Snapshot:
    p = Path(path)
    raw = json.loads(p.read_text(encoding="utf-8"))
    epoch_et_s, epoch_ut_s = find_epoch(raw)
    ref = raw.get("reference_body") or raw.get("central_body") or raw.get("frame_center")
    states = collect_states(raw)
    return Snapshot(path=p, epoch_et_s=epoch_et_s, epoch_ut_s=epoch_ut_s, reference_body=ref, states=states, raw=raw)


def spice_state(body: str, et_s: float, central_body: str, frame: str = "J2000") -> Vec6:
    st_km, _lt = spice.spkezr(body, et_s, frame, "NONE", central_body)
    return np.asarray(st_km, dtype=float) * 1000.0


def parse_family(s: str) -> Tuple[str, List[str]]:
    if ":" not in s:
        raise argparse.ArgumentTypeError("family must look like Parent:Child1,Child2")
    parent, children = s.split(":", 1)
    kids = [normalize_name(x) for x in children.split(",") if x.strip()]
    return normalize_name(parent), kids


def classify_abs(err_km: float) -> str:
    if err_km <= 1.0:
        return "A"
    if err_km <= 10.0:
        return "A-"
    if err_km <= 100.0:
        return "B"
    if err_km <= 1000.0:
        return "C"
    return "D"


def classify_rel(err_km: float, frac: Optional[float]) -> str:
    # Absolute threshold plus relative-orbit threshold.
    if err_km <= 1.0 or (frac is not None and frac <= 1e-6):
        return "A"
    if err_km <= 10.0 or (frac is not None and frac <= 1e-5):
        return "A-"
    if err_km <= 100.0 or (frac is not None and frac <= 1e-4):
        return "B"
    if err_km <= 1000.0 or (frac is not None and frac <= 1e-3):
        return "C"
    return "D"


def write_csv(path: Path, rows: List[Dict[str, Any]], fields: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--snapshot-json", required=True)
    ap.add_argument("--bsp", required=True)
    ap.add_argument("--tpc", required=True)
    ap.add_argument("--central-body", default="Sun")
    ap.add_argument("--frame", default="J2000")
    ap.add_argument("--family", action="append", default=[], type=parse_family,
                    help="Parent:Child1,Child2. May be repeated.")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--include-all-bodies", action="store_true",
                    help="Compare all bodies parsed from snapshot, not only family bodies.")
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    snap = read_snapshot(args.snapshot_json)

    spice.kclear()
    spice.furnsh(args.tpc)
    spice.furnsh(args.bsp)

    wanted = set()
    for p, kids in args.family:
        wanted.add(p)
        wanted.update(kids)
    if args.include_all_bodies or not wanted:
        wanted.update(snap.states.keys())

    abs_rows: List[Dict[str, Any]] = []
    spk_cache: Dict[str, Vec6] = {}

    for body in sorted(wanted):
        if body not in snap.states:
            continue
        try:
            spk_st = spice_state(body, snap.epoch_et_s, args.central_body, args.frame)
        except Exception as exc:
            abs_rows.append({
                "body": body, "status": "SPICE_ERROR", "error": str(exc),
                "pos_err_m": "", "pos_err_km": "", "vel_err_m_s": "",
                "class_abs": "NA",
            })
            continue
        spk_cache[body] = spk_st
        err = snap.states[body] - spk_st
        pos_m = norm3(err)
        vel_m_s = normv(err)
        abs_rows.append({
            "body": body,
            "status": "OK",
            "pos_err_m": f"{pos_m:.12g}",
            "pos_err_km": f"{pos_m/1000.0:.12g}",
            "vel_err_m_s": f"{vel_m_s:.12g}",
            "class_abs": classify_abs(pos_m/1000.0),
        })

    rel_rows: List[Dict[str, Any]] = []
    for parent, kids in args.family:
        if parent not in snap.states:
            continue
        if parent not in spk_cache:
            try:
                spk_cache[parent] = spice_state(parent, snap.epoch_et_s, args.central_body, args.frame)
            except Exception:
                continue
        p_snap = snap.states[parent]
        p_spk = spk_cache[parent]
        parent_abs_km = norm3(p_snap - p_spk) / 1000.0
        for child in kids:
            if child not in snap.states:
                continue
            if child not in spk_cache:
                try:
                    spk_cache[child] = spice_state(child, snap.epoch_et_s, args.central_body, args.frame)
                except Exception:
                    continue
            c_snap = snap.states[child]
            c_spk = spk_cache[child]
            rel_snap = c_snap - p_snap
            rel_spk = c_spk - p_spk
            rel_err = rel_snap - rel_spk
            rel_pos_m = norm3(rel_err)
            rel_vel_m_s = normv(rel_err)
            orbit_radius_m = norm3(rel_snap)
            frac = rel_pos_m / orbit_radius_m if orbit_radius_m > 0 else None
            child_abs_km = norm3(c_snap - c_spk) / 1000.0
            common_mode_ratio = rel_pos_m / max(norm3(c_snap - c_spk), 1e-30)
            rel_rows.append({
                "parent": parent,
                "child": child,
                "parent_abs_err_km": f"{parent_abs_km:.12g}",
                "child_abs_err_km": f"{child_abs_km:.12g}",
                "relative_pos_err_m": f"{rel_pos_m:.12g}",
                "relative_pos_err_km": f"{rel_pos_m/1000.0:.12g}",
                "relative_vel_err_m_s": f"{rel_vel_m_s:.12g}",
                "orbit_radius_km": f"{orbit_radius_m/1000.0:.12g}",
                "relative_error_fraction": f"{frac:.12g}" if frac is not None else "",
                "common_mode_ratio": f"{common_mode_ratio:.12g}",
                "class_relative": classify_rel(rel_pos_m/1000.0, frac),
            })

    abs_fields = ["body", "status", "pos_err_m", "pos_err_km", "vel_err_m_s", "class_abs", "error"]
    rel_fields = [
        "parent", "child", "parent_abs_err_km", "child_abs_err_km",
        "relative_pos_err_m", "relative_pos_err_km", "relative_vel_err_m_s",
        "orbit_radius_km", "relative_error_fraction", "common_mode_ratio", "class_relative",
    ]
    # Ensure missing keys exist.
    for r in abs_rows:
        for f in abs_fields:
            r.setdefault(f, "")
    for r in rel_rows:
        for f in rel_fields:
            r.setdefault(f, "")

    write_csv(out / "body_absolute_errors.csv", abs_rows, abs_fields)
    write_csv(out / "family_relative_errors.csv", rel_rows, rel_fields)
    (out / "body_absolute_errors.json").write_text(json.dumps(abs_rows, indent=2), encoding="utf-8")
    (out / "family_relative_errors.json").write_text(json.dumps(rel_rows, indent=2), encoding="utf-8")
    manifest = {
        "snapshot_json": str(snap.path),
        "bsp": args.bsp,
        "tpc": args.tpc,
        "central_body": args.central_body,
        "frame": args.frame,
        "epoch_et_s": snap.epoch_et_s,
        "epoch_ut_s": snap.epoch_ut_s,
        "snapshot_reference_body": snap.reference_body,
        "families": [{"parent": p, "children": kids} for p, kids in args.family],
        "n_abs_rows": len(abs_rows),
        "n_rel_rows": len(rel_rows),
    }
    (out / "validation_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("\n=== FUTURE SNAPSHOT VALIDATION — ABSOLUTE ===")
    rows_ok = [r for r in abs_rows if r.get("status") == "OK"]
    rows_ok.sort(key=lambda r: float(r["pos_err_km"]), reverse=True)
    print(f"Epoch ET: {snap.epoch_et_s:.9f} s")
    print(f"{'Body':<14} {'Abs km':>14} {'Vel m/s':>12} {'Class':>7}")
    print("-" * 56)
    for r in rows_ok[:30]:
        print(f"{r['body']:<14} {float(r['pos_err_km']):14.6f} {float(r['vel_err_m_s']):12.6f} {r['class_abs']:>7}")

    print("\n=== FUTURE SNAPSHOT VALIDATION — RELATIVE FAMILY ===")
    rel_rows.sort(key=lambda r: float(r["relative_pos_err_km"]), reverse=True)
    print(f"{'Parent':<10} {'Child':<12} {'Parent km':>11} {'Child km':>11} {'Rel km':>11} {'Rel frac':>11} {'Class':>7}")
    print("-" * 82)
    for r in rel_rows[:40]:
        print(
            f"{r['parent']:<10} {r['child']:<12} "
            f"{float(r['parent_abs_err_km']):11.3f} {float(r['child_abs_err_km']):11.3f} "
            f"{float(r['relative_pos_err_km']):11.3f} "
            f"{float(r['relative_error_fraction'] or 0.0):11.3e} {r['class_relative']:>7}"
        )

    print(f"\n[OK] {out / 'body_absolute_errors.csv'}")
    print(f"[OK] {out / 'family_relative_errors.csv'}")


if __name__ == "__main__":
    main()
