#!/usr/bin/env python3
"""
fast_principia_observer.py

Fast kRPC observer for KSP + Principia celestial-body ephemerides.

Why this exists
---------------
The conservative acquirer calls SpaceCenter.warp_to(target_ut) for every sample
and waits for warp to settle before reading. That is safe for spacecraft ops, but
very slow when the only goal is observing celestial bodies.

This script keeps rails warp running continuously and samples bodies whenever the
current KSP UT passes the next requested epoch. It stores the *actual* UT of each
sample, so downstream validation should use et_seconds from the CSV, not assume a
perfectly uniform target grid.

Intended uses
-------------
- quick stability/ejection checks for systems such as Jool + moons;
- building a witness CSV for REBOUND Level-A comparison;
- comparing mod patches without waiting hours for kRPC's auto warp.

It is still an observer, not a solver. It does not access Principia internals.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import os
import platform
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import krpc  # type: ignore
except ImportError as exc:
    raise SystemExit("Instale o pacote 'krpc' no Python usado para conectar ao KSP.") from exc

Vec3 = Tuple[float, float, float]
G_SI = 6.67430e-11


def utc_now_iso() -> str:
    return dt.datetime.now(tz=dt.timezone.utc).isoformat()


def safe_get(obj: Any, attr: str, default: Any = None) -> Any:
    try:
        return getattr(obj, attr)
    except Exception:
        return default


def safe_call(fn: Any, *args: Any, default: Any = None) -> Any:
    try:
        return fn(*args)
    except Exception:
        return default


def norm3(v: Vec3) -> float:
    return math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])


def sub3(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def vec3(x: Any) -> Vec3:
    return (float(x[0]), float(x[1]), float(x[2]))


def export_vec3(v: Vec3, right_handed: bool) -> Vec3:
    return (v[0], v[1], -v[2]) if right_handed else v


def get_body_names(bodies: Any) -> List[str]:
    try:
        return sorted(str(k) for k in bodies.keys())
    except Exception:
        return sorted(str(getattr(b, "name", "")) for b in bodies if getattr(b, "name", None))


def get_body(bodies: Any, name: str) -> Any:
    try:
        return bodies[name]
    except Exception:
        for b in bodies:
            if getattr(b, "name", None) == name:
                return b
        raise KeyError(name)


def parse_csv_list(text: Optional[str]) -> Optional[List[str]]:
    if text is None or not text.strip():
        return None
    return [x.strip() for x in text.split(",") if x.strip()]


def pick_reference_body(bodies: Any, preferred: Optional[str]) -> Any:
    names = get_body_names(bodies)
    if preferred and preferred in names:
        return get_body(bodies, preferred)
    for name in ("Sun", "Kerbol"):
        if name in names:
            return get_body(bodies, name)
    if not names:
        raise RuntimeError("Nenhum corpo encontrado em SpaceCenter.bodies.")
    return get_body(bodies, names[0])


def body_catalog_entry(body: Any) -> Dict[str, Any]:
    orbit = safe_get(body, "orbit")
    parent = None
    if orbit is not None:
        parent_body = safe_get(orbit, "body")
        parent = safe_get(parent_body, "name")
    mu = safe_get(body, "gravitational_parameter")
    return {
        "name": safe_get(body, "name"),
        "mu_m3_s2": mu,
        "mass_kg_from_mu": (float(mu) / G_SI) if mu is not None else None,
        "mass_kg_api": safe_get(body, "mass"),
        "equatorial_radius_m": safe_get(body, "equatorial_radius"),
        "sphere_of_influence_m": safe_get(body, "sphere_of_influence"),
        "rotational_period_s": safe_get(body, "rotational_period"),
        "has_atmosphere": safe_get(body, "has_atmosphere"),
        "atmosphere_depth_m": safe_get(body, "atmosphere_depth"),
        "orbit_parent": parent,
        "stock_orbit_snapshot": {
            "semi_major_axis_m": safe_get(orbit, "semi_major_axis") if orbit else None,
            "eccentricity": safe_get(orbit, "eccentricity") if orbit else None,
            "inclination_rad": safe_get(orbit, "inclination") if orbit else None,
            "longitude_of_ascending_node_rad": safe_get(orbit, "longitude_of_ascending_node") if orbit else None,
            "argument_of_periapsis_rad": safe_get(orbit, "argument_of_periapsis") if orbit else None,
            "mean_anomaly_at_epoch_rad": safe_get(orbit, "mean_anomaly_at_epoch") if orbit else None,
            "epoch_ut_s": safe_get(orbit, "epoch") if orbit else None,
            "period_s": safe_get(orbit, "period") if orbit else None,
        },
    }


def write_json(path: Path, obj: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def sha256_file(path: Path) -> Optional[str]:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def make_streams(conn: Any, bodies: Dict[str, Any], frame: Any) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    pos_streams: Dict[str, Any] = {}
    vel_streams: Dict[str, Any] = {}
    for name, body in bodies.items():
        try:
            pos_streams[name] = conn.add_stream(body.position, frame)
        except Exception:
            pass
        try:
            vel_streams[name] = conn.add_stream(body.velocity, frame)
        except Exception:
            pass
    return pos_streams, vel_streams


def read_state(body: Any, frame: Any, pos_stream: Any, vel_stream: Any, right_handed: bool) -> Tuple[Optional[Vec3], Optional[Vec3], str]:
    try:
        p_raw = pos_stream() if pos_stream is not None else body.position(frame)
        v_raw = vel_stream() if vel_stream is not None else body.velocity(frame)
        p = export_vec3(vec3(p_raw), right_handed)
        v = export_vec3(vec3(v_raw), right_handed)
        return p, v, ""
    except Exception as exc:
        return None, None, f"{type(exc).__name__}: {exc}"


def set_fast_warp(sc: Any, rails_factor: int, physics_factor: int = 0) -> Dict[str, Any]:
    before = {
        "rails_warp_factor": safe_get(sc, "rails_warp_factor"),
        "physics_warp_factor": safe_get(sc, "physics_warp_factor"),
        "warp_rate": safe_get(sc, "warp_rate"),
    }
    # Try to force rails warp. Some KSP states may refuse a factor; record actual after.
    try:
        sc.physics_warp_factor = physics_factor
    except Exception:
        pass
    try:
        sc.rails_warp_factor = rails_factor
    except Exception as exc:
        before["rails_set_error"] = f"{type(exc).__name__}: {exc}"
    time.sleep(0.2)
    after = {
        "rails_warp_factor": safe_get(sc, "rails_warp_factor"),
        "physics_warp_factor": safe_get(sc, "physics_warp_factor"),
        "warp_rate": safe_get(sc, "warp_rate"),
    }
    return {"before": before, "after": after}


def stop_warp(sc: Any) -> Dict[str, Any]:
    errors: List[str] = []
    try:
        sc.rails_warp_factor = 0
    except Exception as exc:
        errors.append(f"rails: {type(exc).__name__}: {exc}")
    try:
        sc.physics_warp_factor = 0
    except Exception as exc:
        errors.append(f"physics: {type(exc).__name__}: {exc}")
    time.sleep(0.2)
    return {
        "errors": errors,
        "rails_warp_factor": safe_get(sc, "rails_warp_factor"),
        "physics_warp_factor": safe_get(sc, "physics_warp_factor"),
        "warp_rate": safe_get(sc, "warp_rate"),
    }


def sample_once(
    *,
    writer: csv.DictWriter,
    run_id: str,
    sample_index: int,
    target_ut: float,
    actual_ut: float,
    et_offset: float,
    wall_utc: str,
    bodies: Dict[str, Any],
    frame: Any,
    pos_streams: Dict[str, Any],
    vel_streams: Dict[str, Any],
    right_handed: bool,
) -> Tuple[int, int, Dict[str, Vec3]]:
    ok = 0
    failed = 0
    positions: Dict[str, Vec3] = {}
    et = actual_ut + et_offset
    for name, body in bodies.items():
        p, v, err = read_state(body, frame, pos_streams.get(name), vel_streams.get(name), right_handed)
        row: Dict[str, Any] = {
            "run_id": run_id,
            "sample_index": sample_index,
            "body": name,
            "target_ut_s": f"{target_ut:.9f}",
            "actual_ut_s": f"{actual_ut:.9f}",
            "et_seconds": f"{et:.9f}",
            "wall_utc": wall_utc,
            "x_m": "",
            "y_m": "",
            "z_m": "",
            "vx_m_s": "",
            "vy_m_s": "",
            "vz_m_s": "",
            "read_error": err,
        }
        if p is not None and v is not None:
            ok += 1
            positions[name] = p
            row.update({
                "x_m": f"{p[0]:.16e}",
                "y_m": f"{p[1]:.16e}",
                "z_m": f"{p[2]:.16e}",
                "vx_m_s": f"{v[0]:.16e}",
                "vy_m_s": f"{v[1]:.16e}",
                "vz_m_s": f"{v[2]:.16e}",
            })
        else:
            failed += 1
        writer.writerow(row)
    return ok, failed, positions


def main() -> int:
    ap = argparse.ArgumentParser(description="Fast continuous-warp observer for KSP/Principia bodies.")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--duration-days", type=float, default=90.0)
    ap.add_argument("--step-seconds", type=float, default=21600.0, help="Desired game-time spacing between samples. Actual UT is recorded.")
    ap.add_argument("--reference-body", default="Sun")
    ap.add_argument("--bodies", default=None, help="CSV body list. For Jool test: Sun,Jool,Laythe,Vall,Tylo,Bop,Pol")
    ap.add_argument("--exclude-bodies", default="")
    ap.add_argument("--rails-warp-factor", type=int, default=7, help="KSP rails warp index, not rate. Try 6/7/8 depending on install.")
    ap.add_argument("--physics-warp-factor", type=int, default=0)
    ap.add_argument("--poll-real-seconds", type=float, default=0.02, help="How often to poll UT while warping.")
    ap.add_argument("--max-real-seconds", type=float, default=0.0, help="Abort after this real time; 0 disables.")
    ap.add_argument("--et-offset-seconds", type=float, default=0.0)
    ap.add_argument("--right-handed-export", action="store_true")
    ap.add_argument("--monitor-parent", default=None, help="Parent body for relative-distance monitor, e.g. Jool.")
    ap.add_argument("--monitor-moons", default=None, help="CSV moons for relative monitor, e.g. Laythe,Vall,Tylo,Bop,Pol.")
    ap.add_argument("--ejection-multiplier", type=float, default=0.0, help="If >0, stop if distance to parent exceeds multiplier times initial distance.")
    ap.add_argument("--ejection-absolute-m", type=float, default=0.0, help="If >0, stop if distance to parent exceeds this absolute distance.")
    ap.add_argument("--flush-every", type=int, default=1)
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    states_csv = out / "states.csv"
    sample_log_csv = out / "sample_log.csv"
    catalog_json = out / "body_catalog.json"
    manifest_json = out / "manifest.json"
    monitor_csv = out / "relative_monitor.csv"

    if states_csv.exists():
        raise FileExistsError(f"{states_csv} já existe. Escolha outro --output-dir para evitar append acidental.")

    run_id = str(uuid.uuid4())
    conn = krpc.connect(name="Fast_Principia_Observer")

    try:
        sc = conn.space_center
        bodies_obj = sc.bodies
        ref_body = pick_reference_body(bodies_obj, args.reference_body)
        ref_name = str(ref_body.name)
        frame = ref_body.non_rotating_reference_frame

        all_names = get_body_names(bodies_obj)
        allow = parse_csv_list(args.bodies) or all_names
        block = set(parse_csv_list(args.exclude_bodies) or [])
        selected = [n for n in allow if n in all_names and n not in block]
        if ref_name not in selected:
            # Keep reference body in CSV so relative/absolute checks have an origin.
            selected = [ref_name] + selected
        selected = list(dict.fromkeys(selected))
        bodies = {n: get_body(bodies_obj, n) for n in selected}

        monitor_parent = args.monitor_parent
        monitor_moons = parse_csv_list(args.monitor_moons) or []
        monitor_enabled = bool(monitor_parent and monitor_moons)
        if monitor_enabled:
            for n in [monitor_parent] + monitor_moons:
                if n not in selected:
                    raise ValueError(f"Monitor body {n!r} precisa estar em --bodies.")

        start_ut = float(sc.ut)
        end_ut = start_ut + args.duration_days * 86400.0
        target_ut = start_ut
        next_sample_index = 0
        expected_samples = int(math.floor((end_ut - start_ut) / args.step_seconds)) + 1

        catalog = {
            "schema": "fast_ksp_body_catalog.v1",
            "run_id": run_id,
            "created_utc": utc_now_iso(),
            "reference_body": ref_name,
            "bodies": {n: body_catalog_entry(b) for n, b in bodies.items()},
        }
        write_json(catalog_json, catalog)

        pos_streams, vel_streams = make_streams(conn, bodies, frame)
        try:
            ut_stream = conn.add_stream(getattr, sc, "ut")
        except Exception:
            ut_stream = None

        state_fields = [
            "run_id", "sample_index", "body", "target_ut_s", "actual_ut_s", "et_seconds",
            "wall_utc", "x_m", "y_m", "z_m", "vx_m_s", "vy_m_s", "vz_m_s", "read_error",
        ]
        log_fields = [
            "run_id", "sample_index", "target_ut_s", "actual_ut_s", "ut_lag_s", "wall_utc",
            "warp_rate", "rails_warp_factor", "bodies_ok", "bodies_failed", "samples_missed_estimate",
        ]
        mon_fields = [
            "run_id", "sample_index", "actual_ut_s", "et_seconds", "parent", "moon",
            "distance_m", "distance_ratio_to_initial", "ejection_flag",
        ]

        initial_dist: Dict[str, float] = {}
        stop_reason = "completed"
        warp_info = set_fast_warp(sc, args.rails_warp_factor, args.physics_warp_factor)
        wall_start = time.time()

        manifest = {
            "schema": "fast_principia_observer.v1",
            "run_id": run_id,
            "created_utc": utc_now_iso(),
            "purpose": "continuous rails-warp kRPC observation of celestial bodies; actual UT is authoritative",
            "host": {"platform": platform.platform(), "python": sys.version, "cwd": os.getcwd()},
            "reference_body": ref_name,
            "frame": "reference_body.non_rotating_reference_frame",
            "time_mapping": {
                "et_seconds": "ksp_ut_seconds + et_offset_seconds",
                "et_offset_seconds": args.et_offset_seconds,
            },
            "sampling": {
                "start_ut_s": start_ut,
                "end_ut_requested_s": end_ut,
                "duration_days": args.duration_days,
                "step_seconds": args.step_seconds,
                "expected_samples": expected_samples,
                "rails_warp_factor_requested": args.rails_warp_factor,
                "physics_warp_factor_requested": args.physics_warp_factor,
                "poll_real_seconds": args.poll_real_seconds,
                "right_handed_export": args.right_handed_export,
                "selected_bodies": selected,
                "monitor_parent": monitor_parent,
                "monitor_moons": monitor_moons,
                "ejection_multiplier": args.ejection_multiplier,
                "ejection_absolute_m": args.ejection_absolute_m,
            },
            "warp_start": warp_info,
        }
        write_json(manifest_json, manifest)

        with states_csv.open("w", newline="", encoding="utf-8") as sf, \
             sample_log_csv.open("w", newline="", encoding="utf-8") as lf, \
             monitor_csv.open("w", newline="", encoding="utf-8") as mf:
            sw = csv.DictWriter(sf, fieldnames=state_fields)
            lw = csv.DictWriter(lf, fieldnames=log_fields)
            mw = csv.DictWriter(mf, fieldnames=mon_fields)
            sw.writeheader(); lw.writeheader(); mw.writeheader()

            last_print = 0.0
            while True:
                now_real = time.time()
                actual_ut = float(ut_stream() if ut_stream is not None else sc.ut)

                if args.max_real_seconds > 0 and now_real - wall_start > args.max_real_seconds:
                    stop_reason = "max_real_seconds"
                    break
                if actual_ut >= end_ut:
                    # final sample at/after end_ut if not already sampled too recently
                    if next_sample_index == 0 or actual_ut >= target_ut:
                        pass
                    else:
                        break

                if actual_ut >= target_ut:
                    wall_utc = utc_now_iso()
                    ok, failed, positions = sample_once(
                        writer=sw,
                        run_id=run_id,
                        sample_index=next_sample_index,
                        target_ut=target_ut,
                        actual_ut=actual_ut,
                        et_offset=args.et_offset_seconds,
                        wall_utc=wall_utc,
                        bodies=bodies,
                        frame=frame,
                        pos_streams=pos_streams,
                        vel_streams=vel_streams,
                        right_handed=args.right_handed_export,
                    )
                    missed = max(0, int(math.floor((actual_ut - target_ut) / args.step_seconds)))
                    lw.writerow({
                        "run_id": run_id,
                        "sample_index": next_sample_index,
                        "target_ut_s": f"{target_ut:.9f}",
                        "actual_ut_s": f"{actual_ut:.9f}",
                        "ut_lag_s": f"{actual_ut - target_ut:.9f}",
                        "wall_utc": wall_utc,
                        "warp_rate": safe_get(sc, "warp_rate", ""),
                        "rails_warp_factor": safe_get(sc, "rails_warp_factor", ""),
                        "bodies_ok": ok,
                        "bodies_failed": failed,
                        "samples_missed_estimate": missed,
                    })

                    ejection = False
                    if monitor_enabled and monitor_parent in positions:
                        parent_pos = positions[monitor_parent]
                        for moon in monitor_moons:
                            if moon not in positions:
                                continue
                            d = norm3(sub3(positions[moon], parent_pos))
                            if moon not in initial_dist:
                                initial_dist[moon] = d if d > 0 else math.nan
                            ratio = d / initial_dist[moon] if initial_dist.get(moon) and math.isfinite(initial_dist[moon]) else math.nan
                            flag = False
                            if args.ejection_multiplier > 0 and math.isfinite(ratio) and ratio >= args.ejection_multiplier:
                                flag = True
                            if args.ejection_absolute_m > 0 and d >= args.ejection_absolute_m:
                                flag = True
                            if flag:
                                ejection = True
                            mw.writerow({
                                "run_id": run_id,
                                "sample_index": next_sample_index,
                                "actual_ut_s": f"{actual_ut:.9f}",
                                "et_seconds": f"{actual_ut + args.et_offset_seconds:.9f}",
                                "parent": monitor_parent,
                                "moon": moon,
                                "distance_m": f"{d:.16e}",
                                "distance_ratio_to_initial": f"{ratio:.16e}" if math.isfinite(ratio) else "",
                                "ejection_flag": "1" if flag else "0",
                            })

                    if next_sample_index % max(1, args.flush_every) == 0:
                        sf.flush(); lf.flush(); mf.flush()

                    if now_real - last_print > 2.0:
                        frac = (actual_ut - start_ut) / max(1.0, end_ut - start_ut)
                        print(
                            f"[{next_sample_index+1}] UT={actual_ut:.2f} day={(actual_ut-start_ut)/86400:.2f}/"
                            f"{args.duration_days:.2f} lag={actual_ut-target_ut:.2f}s ok={ok} failed={failed} "
                            f"warp={safe_get(sc, 'warp_rate', '')} progress={100*frac:.1f}%",
                            flush=True,
                        )
                        last_print = now_real

                    next_sample_index += 1
                    # Do not try to synthesize missed samples. Advance the target grid beyond actual_ut.
                    while target_ut <= actual_ut:
                        target_ut += args.step_seconds

                    if ejection:
                        stop_reason = "ejection_detected"
                        break
                    if actual_ut >= end_ut:
                        break

                time.sleep(max(0.0, args.poll_real_seconds))

        stop_info = stop_warp(sc)
        manifest["completed_utc"] = utc_now_iso()
        manifest["stop_reason"] = stop_reason
        manifest["wall_elapsed_s"] = time.time() - wall_start
        manifest["samples_written"] = next_sample_index
        manifest["initial_monitor_distances_m"] = initial_dist
        manifest["warp_stop"] = stop_info
        manifest["files_sha256"] = {
            "states_csv": sha256_file(states_csv),
            "sample_log_csv": sha256_file(sample_log_csv),
            "relative_monitor_csv": sha256_file(monitor_csv),
            "body_catalog_json": sha256_file(catalog_json),
        }
        write_json(manifest_json, manifest)

        print(json.dumps({
            "output_dir": str(out),
            "stop_reason": stop_reason,
            "samples_written": next_sample_index,
            "wall_elapsed_s": manifest["wall_elapsed_s"],
            "states_csv": str(states_csv),
            "relative_monitor_csv": str(monitor_csv),
        }, indent=2, ensure_ascii=False))
        return 0

    finally:
        try:
            stop_warp(sc)  # type: ignore[name-defined]
        except Exception:
            pass
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
