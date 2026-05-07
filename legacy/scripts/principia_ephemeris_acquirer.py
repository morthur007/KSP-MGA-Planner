#!/usr/bin/env python3
"""
principia_ephemeris_acquirer.py

Offline-first ephemeris acquisition for KSP + kRPC, designed for Principia
systems where there is no stable public Principia ephemeris API.

Purpose
-------
1) Treat KSP/kRPC as a synthetic observatory:
   - warp to explicitly scheduled KSP UT epochs;
   - read current CelestialBody center position/velocity in a declared frame;
   - emit tabulated state vectors and body metadata for later fitting to
     Chebyshev/SPK-type ephemerides.

2) Do NOT treat kRPC as the solver:
   - this script only acquires and validates observed states;
   - downstream tools should fit piecewise Chebyshev/SPK segments and feed an
     offline propagator/optimizer.

3) Empirically test whether the stream has a "Principia-like" signature:
   - compare observed future body positions against one-step Kepler predictions
     from kRPC Orbit.position_at(...);
   - compare finite-difference velocity from sampled positions against reported
     body.velocity(...);
   - report "detected", "not_detected", or "inconclusive".

Important limitations
---------------------
- kRPC does not expose the internal Principia Ephemeris object.
- The Principia signal test is empirical. It cannot prove that data came from
  principia::physics::Ephemeris; it can only show that observed states do not
  behave like the stock kRPC Kepler orbit prediction over the chosen sampling
  interval.
- KSP/kRPC use a left-handed coordinate system. This script can either export
  raw kRPC axes or flip Z to produce a right-handed translational state for
  downstream SPICE-like tooling. The chosen convention is recorded in metadata.
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import hashlib
import json
import math
import os
import platform
import statistics
import sys
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    import krpc  # type: ignore
except ImportError as exc:
    raise SystemExit(
        "O pacote 'krpc' não está instalado neste ambiente Python. "
        "Instale-o no ambiente que se conecta ao KSP/kRPC."
    ) from exc


Vec3 = Tuple[float, float, float]


@dataclass(frozen=True)
class AcquisitionConfig:
    output_dir: str
    duration_days: float
    step_seconds: float
    reference_body_name: Optional[str]
    max_rails_rate: float
    max_physics_rate: float
    settle_timeout_s: float
    settle_poll_s: float
    post_warp_pause_s: float
    ut_tolerance_s: float
    et_offset_seconds: float
    right_handed_export: bool
    include_attitude: bool
    include_reference_body: bool
    body_allowlist: Optional[List[str]]
    body_blocklist: List[str]
    resume: bool


@dataclass
class OnlineStats:
    n: int = 0
    min: float = math.inf
    max: float = -math.inf
    sum: float = 0.0
    sum_sq: float = 0.0

    def add(self, x: float) -> None:
        if not math.isfinite(x):
            return
        self.n += 1
        self.min = min(self.min, x)
        self.max = max(self.max, x)
        self.sum += x
        self.sum_sq += x * x

    def as_dict(self) -> Dict[str, Optional[float]]:
        if self.n == 0:
            return {"n": 0, "min": None, "max": None, "mean": None, "rms": None}
        mean = self.sum / self.n
        rms = math.sqrt(self.sum_sq / self.n)
        return {"n": self.n, "min": self.min, "max": self.max, "mean": mean, "rms": rms}


def utc_now_iso() -> str:
    return _dt.datetime.now(tz=_dt.timezone.utc).isoformat()


def safe_get(obj: Any, attr: str, default: Any = None) -> Any:
    try:
        value = getattr(obj, attr)
        return value() if callable(value) and attr.startswith("get_") else value
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


def mul3(s: float, a: Vec3) -> Vec3:
    return (s * a[0], s * a[1], s * a[2])


def avg3(a: Vec3, b: Vec3) -> Vec3:
    return (0.5 * (a[0] + b[0]), 0.5 * (a[1] + b[1]), 0.5 * (a[2] + b[2]))


def vec3(x: Any) -> Vec3:
    return (float(x[0]), float(x[1]), float(x[2]))


def maybe_vec3(x: Any) -> Optional[Vec3]:
    try:
        return vec3(x)
    except Exception:
        return None


def export_vec3(v: Vec3, right_handed: bool) -> Vec3:
    # kRPC/KSP coordinates are left-handed. For a SPICE-like right-handed
    # translational state, flip a single axis. We choose Z -> -Z and record it.
    if right_handed:
        return (v[0], v[1], -v[2])
    return v


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


def pick_reference_body(bodies: Any, preferred: Optional[str]) -> Any:
    names = get_body_names(bodies)
    if preferred and preferred in names:
        return get_body(bodies, preferred)
    for candidate in ("Sun", "Kerbol"):
        if candidate in names:
            return get_body(bodies, candidate)
    if not names:
        raise RuntimeError("Nenhum corpo celeste encontrado via kRPC SpaceCenter.bodies.")
    return get_body(bodies, names[0])


def parse_body_list(text: Optional[str]) -> Optional[List[str]]:
    if text is None or not text.strip():
        return None
    return [x.strip() for x in text.split(",") if x.strip()]


def body_catalog_entry(body: Any) -> Dict[str, Any]:
    orbit = safe_get(body, "orbit")
    parent = None
    if orbit is not None:
        parent_body = safe_get(orbit, "body")
        parent = safe_get(parent_body, "name")

    return {
        "name": safe_get(body, "name"),
        "mass_kg": safe_get(body, "mass"),
        "mu_m3_s2": safe_get(body, "gravitational_parameter"),
        "equatorial_radius_m": safe_get(body, "equatorial_radius"),
        "sphere_of_influence_m": safe_get(body, "sphere_of_influence"),
        "rotational_period_s": safe_get(body, "rotational_period"),
        "rotational_speed_rad_s": safe_get(body, "rotational_speed"),
        "has_solid_surface": safe_get(body, "has_solid_surface"),
        "has_atmosphere": safe_get(body, "has_atmosphere"),
        "atmosphere_depth_m": safe_get(body, "atmosphere_depth"),
        "has_atmospheric_oxygen": safe_get(body, "has_atmospheric_oxygen"),
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


def file_sha256(path: Path) -> Optional[str]:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def detect_principia_install_hint(ksp_root: Optional[Path]) -> Dict[str, Any]:
    """
    Filesystem hint only. It says "Principia files seem installed", not "active".
    """
    if ksp_root is None:
        return {"ksp_root": None, "principia_filesystem_hint": "not_checked"}

    gd = ksp_root / "GameData"
    candidates = [
        gd / "Principia",
        gd / "Principia" / "principia.dll",
        gd / "Principia" / "GameData" / "Principia",
    ]
    existing = [str(p) for p in candidates if p.exists()]
    return {
        "ksp_root": str(ksp_root),
        "principia_filesystem_hint": "present" if existing else "not_found",
        "existing_paths": existing,
    }


def load_existing_sample_indices(states_csv: Path) -> set[int]:
    if not states_csv.exists():
        return set()
    seen: set[int] = set()
    try:
        with states_csv.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("sample_index") not in (None, ""):
                    seen.add(int(row["sample_index"]))
    except Exception:
        # If resume cannot safely parse, force caller to avoid accidental append.
        raise RuntimeError(
            f"Não consegui ler {states_csv} para retomar. "
            "Remova --resume ou mova o diretório de saída."
        )
    return seen


def wait_for_warp_settle(
    sc: Any,
    target_ut: float,
    tolerance_s: float,
    timeout_s: float,
    poll_s: float,
    post_pause_s: float,
) -> Dict[str, Any]:
    deadline = time.time() + timeout_s
    last_ut = float(sc.ut)
    last_warp_rate = float(safe_get(sc, "warp_rate", 1.0) or 1.0)

    while time.time() < deadline:
        last_ut = float(sc.ut)
        last_warp_rate = float(safe_get(sc, "warp_rate", 1.0) or 1.0)
        if abs(last_ut - target_ut) <= tolerance_s and last_warp_rate <= 1.01:
            break
        time.sleep(poll_s)

    if post_pause_s > 0:
        time.sleep(post_pause_s)

    final_ut = float(sc.ut)
    return {
        "target_ut_s": target_ut,
        "actual_ut_s": final_ut,
        "ut_error_s": final_ut - target_ut,
        "warp_rate_after_settle": float(safe_get(sc, "warp_rate", 1.0) or 1.0),
    }


def make_state_streams(conn: Any, bodies: Dict[str, Any], frame: Any) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    pos_streams: Dict[str, Any] = {}
    vel_streams: Dict[str, Any] = {}
    for name, body in bodies.items():
        try:
            pos_streams[name] = conn.add_stream(body.position, frame)
            vel_streams[name] = conn.add_stream(body.velocity, frame)
        except Exception:
            # Direct call fallback is handled at read time.
            pass
    return pos_streams, vel_streams


def read_state(
    body: Any,
    frame: Any,
    pos_stream: Any,
    vel_stream: Any,
    right_handed: bool,
) -> Tuple[Optional[Vec3], Optional[Vec3], Optional[str]]:
    try:
        raw_pos = maybe_vec3(pos_stream() if pos_stream is not None else body.position(frame))
        raw_vel = maybe_vec3(vel_stream() if vel_stream is not None else body.velocity(frame))
        if raw_pos is None or raw_vel is None:
            return None, None, "state_vector_unavailable"
        return export_vec3(raw_pos, right_handed), export_vec3(raw_vel, right_handed), None
    except Exception as exc:
        return None, None, f"{type(exc).__name__}: {exc}"


def read_attitude(body: Any, frame: Any) -> Dict[str, Any]:
    # Rotation quaternions are kept in raw kRPC convention. Do not use them
    # directly in a right-handed SPICE frame without a separate frame model.
    return {
        "rotation_xyzw_raw": safe_call(body.rotation, frame, default=None),
        "north_direction_raw": safe_call(body.direction, frame, default=None),
        "angular_velocity_raw": safe_call(body.angular_velocity, frame, default=None),
    }


def write_json(path: Path, obj: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def classify_principia_signal(validation: Dict[str, Any]) -> str:
    """
    Conservative classification:
    - detected: multiple bodies show large one-step conic divergence;
    - not_detected: enough comparisons and no divergence;
    - inconclusive: too few data or only tiny/ambiguous residuals.
    """
    body_results = validation.get("body_results", {})
    compared = 0
    detected = 0

    for _name, data in body_results.items():
        stats = data.get("one_step_orbit_prediction_error_m", {})
        n = stats.get("n") or 0
        max_err = stats.get("max")
        scale = data.get("typical_radius_m") or 1.0
        # Tolerance: max(1 km, 1e-8 of typical radius). Deliberately loose to
        # avoid false positives from frame/roundoff.
        threshold = max(1_000.0, 1e-8 * scale)

        if n >= 2:
            compared += 1
            if max_err is not None and max_err > threshold:
                detected += 1
                data["principia_like_non_keplerian_signal"] = True
                data["principia_signal_threshold_m"] = threshold
            else:
                data["principia_like_non_keplerian_signal"] = False
                data["principia_signal_threshold_m"] = threshold

    validation["principia_signal_compared_bodies"] = compared
    validation["principia_signal_detected_bodies"] = detected

    if compared < 2:
        return "inconclusive"
    if detected >= max(2, math.ceil(0.2 * compared)):
        return "detected"
    return "not_detected_or_too_short_baseline"


def acquire_ephemerides(config: AcquisitionConfig, ksp_root: Optional[Path] = None) -> Dict[str, Any]:
    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    states_csv = out_dir / "states.csv"
    sample_log_csv = out_dir / "sample_log.csv"
    manifest_json = out_dir / "manifest.json"
    catalog_json = out_dir / "body_catalog.json"
    validation_json = out_dir / "validation.json"

    if states_csv.exists() and not config.resume:
        raise FileExistsError(
            f"{states_csv} já existe. Use --resume para continuar ou escolha outro --output-dir."
        )

    run_id = str(uuid.uuid4())
    conn = krpc.connect(name="Principia_Ephemeris_Acquirer")

    try:
        sc = conn.space_center
        bodies_obj = sc.bodies
        reference_body = pick_reference_body(bodies_obj, config.reference_body_name)
        reference_name = str(reference_body.name)
        frame = reference_body.non_rotating_reference_frame

        all_names = get_body_names(bodies_obj)
        allow = set(config.body_allowlist) if config.body_allowlist else set(all_names)
        block = set(config.body_blocklist)
        selected_names = [
            n for n in all_names
            if n in allow and n not in block and (config.include_reference_body or n != reference_name)
        ]

        bodies: Dict[str, Any] = {name: get_body(bodies_obj, name) for name in selected_names}

        start_ut = float(sc.ut)
        if config.duration_days <= 0 or config.step_seconds <= 0:
            raise ValueError("duration_days e step_seconds precisam ser positivos.")

        n_steps = int(math.floor(config.duration_days * 86400.0 / config.step_seconds))
        target_uts = [start_ut + i * config.step_seconds for i in range(n_steps + 1)]

        skipped_indices = load_existing_sample_indices(states_csv) if config.resume else set()

        frame_convention = (
            "right_handed_export_from_krpc_left_handed_by_z_flip"
            if config.right_handed_export
            else "raw_krpc_left_handed"
        )

        manifest = {
            "schema": "principia_ephemeris_acquisition.v1",
            "run_id": run_id,
            "created_utc": utc_now_iso(),
            "host": {
                "platform": platform.platform(),
                "python": sys.version,
                "cwd": os.getcwd(),
            },
            "ksp": {
                "ut_start_s": start_ut,
                "ut_end_requested_s": target_uts[-1],
                "ut_scale": "KSP_UniversalTime_seconds",
                "principia_reading_claim": (
                    "kRPC reads current KSP object states. If Principia is installed and actively "
                    "driving celestial-body dynamics, these observed states should reflect the "
                    "current game state after Principia updates. This is not direct access to "
                    "principia::physics::Ephemeris."
                ),
                **detect_principia_install_hint(ksp_root),
            },
            "time_mapping": {
                "internal_time_scale": "ET_TDB_seconds_past_J2000_affine",
                "formula": "et_seconds = ksp_ut_seconds + et_offset_seconds",
                "et_offset_seconds": config.et_offset_seconds,
                "note": (
                    "For fictitious systems this is a software convention, not a physical relativistic "
                    "time-scale conversion. Store it permanently with any generated SPK/kernel."
                ),
            },
            "reference_frame": {
                "reference_body": reference_name,
                "krpc_frame": "reference_body.non_rotating_reference_frame",
                "frame_convention": frame_convention,
                "warning": (
                    "kRPC/KSP uses a left-handed coordinate system. If right_handed_export is true, "
                    "positions and velocities use (x, y, -z). Rotation quaternions, if exported, "
                    "remain raw kRPC and need a dedicated frame model."
                ),
            },
            "sampling": asdict(config),
            "body_names": selected_names,
            "files": {
                "states_csv": states_csv.name,
                "sample_log_csv": sample_log_csv.name,
                "body_catalog_json": catalog_json.name,
                "validation_json": validation_json.name,
            },
        }

        catalog = {
            "schema": "ksp_body_catalog.v1",
            "run_id": run_id,
            "created_utc": utc_now_iso(),
            "reference_body": reference_name,
            "bodies": {name: body_catalog_entry(body) for name, body in bodies.items()},
        }

        write_json(manifest_json, manifest)
        write_json(catalog_json, catalog)

        pos_streams, vel_streams = make_state_streams(conn, bodies, frame)

        state_fieldnames = [
            "run_id", "sample_index", "body", "target_ut_s", "actual_ut_s",
            "et_seconds", "wall_utc", "x_m", "y_m", "z_m", "vx_m_s", "vy_m_s",
            "vz_m_s", "read_error"
        ]
        if config.include_attitude:
            state_fieldnames += [
                "rotation_xyzw_raw_json",
                "north_direction_raw_json",
                "angular_velocity_raw_json",
            ]

        sample_fieldnames = [
            "run_id", "sample_index", "target_ut_s", "actual_ut_s", "ut_error_s",
            "warp_rate_after_settle", "wall_utc", "bodies_attempted", "bodies_ok",
            "bodies_failed"
        ]

        states_new = not states_csv.exists()
        sample_log_new = not sample_log_csv.exists()

        previous_state: Dict[str, Tuple[float, Vec3, Vec3]] = {}
        next_orbit_prediction: Dict[str, Tuple[float, Vec3]] = {}

        validation_work: Dict[str, Dict[str, Any]] = {
            name: {
                "finite_difference_velocity_error_m_s": OnlineStats(),
                "one_step_orbit_prediction_error_m": OnlineStats(),
                "typical_radius_samples": [],
                "read_errors": 0,
            }
            for name in selected_names
        }

        with states_csv.open("a", newline="", encoding="utf-8") as sf, sample_log_csv.open("a", newline="", encoding="utf-8") as lf:
            state_writer = csv.DictWriter(sf, fieldnames=state_fieldnames)
            sample_writer = csv.DictWriter(lf, fieldnames=sample_fieldnames)
            if states_new:
                state_writer.writeheader()
            if sample_log_new:
                sample_writer.writeheader()

            for i, target_ut in enumerate(target_uts):
                if i in skipped_indices:
                    continue

                if target_ut < float(sc.ut) - config.ut_tolerance_s:
                    raise RuntimeError(
                        f"Target UT {target_ut} ficou atrás do UT atual {float(sc.ut)}; "
                        "kRPC warp_to só avança no tempo. Reinicie de um save anterior."
                    )

                sc.warp_to(
                    target_ut,
                    max_rails_rate=config.max_rails_rate,
                    max_physics_rate=config.max_physics_rate,
                )
                settle = wait_for_warp_settle(
                    sc,
                    target_ut=target_ut,
                    tolerance_s=config.ut_tolerance_s,
                    timeout_s=config.settle_timeout_s,
                    poll_s=config.settle_poll_s,
                    post_pause_s=config.post_warp_pause_s,
                )
                actual_ut = float(settle["actual_ut_s"])
                et_seconds = actual_ut + config.et_offset_seconds
                wall_utc = utc_now_iso()

                ok = 0
                failed = 0

                for name, body in bodies.items():
                    pos, vel, err = read_state(
                        body=body,
                        frame=frame,
                        pos_stream=pos_streams.get(name),
                        vel_stream=vel_streams.get(name),
                        right_handed=config.right_handed_export,
                    )

                    row: Dict[str, Any] = {
                        "run_id": run_id,
                        "sample_index": i,
                        "body": name,
                        "target_ut_s": f"{target_ut:.9f}",
                        "actual_ut_s": f"{actual_ut:.9f}",
                        "et_seconds": f"{et_seconds:.9f}",
                        "wall_utc": wall_utc,
                        "x_m": "",
                        "y_m": "",
                        "z_m": "",
                        "vx_m_s": "",
                        "vy_m_s": "",
                        "vz_m_s": "",
                        "read_error": err or "",
                    }

                    if config.include_attitude:
                        att = read_attitude(body, frame)
                        row.update({
                            "rotation_xyzw_raw_json": json.dumps(att["rotation_xyzw_raw"]),
                            "north_direction_raw_json": json.dumps(att["north_direction_raw"]),
                            "angular_velocity_raw_json": json.dumps(att["angular_velocity_raw"]),
                        })

                    if pos is None or vel is None:
                        failed += 1
                        validation_work[name]["read_errors"] += 1
                        state_writer.writerow(row)
                        continue

                    ok += 1
                    row.update({
                        "x_m": f"{pos[0]:.16e}",
                        "y_m": f"{pos[1]:.16e}",
                        "z_m": f"{pos[2]:.16e}",
                        "vx_m_s": f"{vel[0]:.16e}",
                        "vy_m_s": f"{vel[1]:.16e}",
                        "vz_m_s": f"{vel[2]:.16e}",
                    })
                    state_writer.writerow(row)

                    rnorm = norm3(pos)
                    if math.isfinite(rnorm):
                        validation_work[name]["typical_radius_samples"].append(rnorm)

                    # Kinematic consistency: finite difference velocity vs reported velocity.
                    if name in previous_state:
                        prev_ut, prev_pos, prev_vel = previous_state[name]
                        dt = actual_ut - prev_ut
                        if dt > 0:
                            v_fd = mul3(1.0 / dt, sub3(pos, prev_pos))
                            v_mid = avg3(vel, prev_vel)
                            fd_err = norm3(sub3(v_fd, v_mid))
                            validation_work[name]["finite_difference_velocity_error_m_s"].add(fd_err)

                    previous_state[name] = (actual_ut, pos, vel)

                    # Compare actual position at this sample against previous sample's
                    # one-step stock/KSP Orbit.position_at prediction.
                    if name in next_orbit_prediction:
                        pred_ut, pred_pos = next_orbit_prediction[name]
                        if abs(pred_ut - actual_ut) <= max(config.ut_tolerance_s, 0.5 * config.step_seconds):
                            pred_err = norm3(sub3(pos, pred_pos))
                            validation_work[name]["one_step_orbit_prediction_error_m"].add(pred_err)

                    # Build prediction for next sample from the current orbit object.
                    if i + 1 < len(target_uts):
                        next_ut = target_uts[i + 1]
                        orbit = safe_get(body, "orbit")
                        if orbit is not None:
                            pred_raw = safe_call(orbit.position_at, next_ut, frame, default=None)
                            pred = maybe_vec3(pred_raw)
                            if pred is not None:
                                next_orbit_prediction[name] = (
                                    next_ut,
                                    export_vec3(pred, config.right_handed_export),
                                )

                sample_writer.writerow({
                    "run_id": run_id,
                    "sample_index": i,
                    "target_ut_s": f"{target_ut:.9f}",
                    "actual_ut_s": f"{actual_ut:.9f}",
                    "ut_error_s": f"{float(settle['ut_error_s']):.9f}",
                    "warp_rate_after_settle": f"{float(settle['warp_rate_after_settle']):.9f}",
                    "wall_utc": wall_utc,
                    "bodies_attempted": len(selected_names),
                    "bodies_ok": ok,
                    "bodies_failed": failed,
                })
                sf.flush()
                lf.flush()

                print(
                    f"[{i + 1}/{len(target_uts)}] UT={actual_ut:.3f} "
                    f"ok={ok} failed={failed}",
                    flush=True,
                )

        validation: Dict[str, Any] = {
            "schema": "principia_ephemeris_validation.v1",
            "run_id": run_id,
            "created_utc": utc_now_iso(),
            "interpretation": {
                "finite_difference_velocity_error_m_s": (
                    "Checks local consistency between sampled positions and reported velocities. "
                    "Large values can indicate too-large sampling interval, frame issues, insufficient "
                    "settle time, or kRPC/Principia state latency."
                ),
                "one_step_orbit_prediction_error_m": (
                    "Compares observed state after warp against the previous sample's kRPC "
                    "Orbit.position_at prediction. Large systematic divergence is a Principia-like "
                    "non-Keplerian signal, but absence of divergence is inconclusive for short arcs."
                ),
            },
            "body_results": {},
        }

        for name, work in validation_work.items():
            radii = work["typical_radius_samples"]
            typical_radius = statistics.median(radii) if radii else None
            validation["body_results"][name] = {
                "read_errors": work["read_errors"],
                "typical_radius_m": typical_radius,
                "finite_difference_velocity_error_m_s": work["finite_difference_velocity_error_m_s"].as_dict(),
                "one_step_orbit_prediction_error_m": work["one_step_orbit_prediction_error_m"].as_dict(),
            }

        validation["principia_signal_classification"] = classify_principia_signal(validation)
        write_json(validation_json, validation)

        manifest["completed_utc"] = utc_now_iso()
        manifest["files_sha256"] = {
            "states_csv": file_sha256(states_csv),
            "sample_log_csv": file_sha256(sample_log_csv),
            "body_catalog_json": file_sha256(catalog_json),
            "validation_json": file_sha256(validation_json),
        }
        write_json(manifest_json, manifest)

        return {
            "manifest": manifest,
            "validation": validation,
            "output_dir": str(out_dir),
        }

    finally:
        conn.close()


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Amostra estados de corpos celestes via kRPC para criar uma base de "
            "efemérides offline compatível com fitting Chebyshev/SPK."
        )
    )
    p.add_argument("--output-dir", default="ksp_ephemeris_acquisition")
    p.add_argument("--duration-days", type=float, default=2.0)
    p.add_argument("--step-seconds", type=float, default=3600.0)
    p.add_argument("--reference-body", default=None, help="Ex.: Sun ou Kerbol. Default: Sun/Kerbol/primeiro corpo.")
    p.add_argument("--max-rails-rate", type=float, default=100000.0)
    p.add_argument("--max-physics-rate", type=float, default=2.0)
    p.add_argument("--settle-timeout-s", type=float, default=20.0)
    p.add_argument("--settle-poll-s", type=float, default=0.10)
    p.add_argument("--post-warp-pause-s", type=float, default=0.25)
    p.add_argument("--ut-tolerance-s", type=float, default=0.25)
    p.add_argument(
        "--et-offset-seconds",
        type=float,
        default=0.0,
        help="Offset afim: et_seconds = ksp_ut_seconds + offset. Guarde isso para SPK.",
    )
    p.add_argument(
        "--right-handed-export",
        action="store_true",
        help="Exporta translational states como (x,y,-z), útil para frame SPICE-like.",
    )
    p.add_argument("--include-attitude", action="store_true", help="Inclui rotação/direção/omega raw kRPC.")
    p.add_argument("--include-reference-body", action="store_true")
    p.add_argument("--bodies", default=None, help="Lista CSV de corpos a incluir. Ex.: Kerbin,Duna,Jool")
    p.add_argument("--exclude-bodies", default="", help="Lista CSV de corpos a excluir.")
    p.add_argument("--resume", action="store_true")
    p.add_argument(
        "--ksp-root",
        default=None,
        help="Opcional: caminho para raiz do KSP; usado só para hint de arquivos Principia em GameData.",
    )
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    cfg = AcquisitionConfig(
        output_dir=args.output_dir,
        duration_days=args.duration_days,
        step_seconds=args.step_seconds,
        reference_body_name=args.reference_body,
        max_rails_rate=args.max_rails_rate,
        max_physics_rate=args.max_physics_rate,
        settle_timeout_s=args.settle_timeout_s,
        settle_poll_s=args.settle_poll_s,
        post_warp_pause_s=args.post_warp_pause_s,
        ut_tolerance_s=args.ut_tolerance_s,
        et_offset_seconds=args.et_offset_seconds,
        right_handed_export=args.right_handed_export,
        include_attitude=args.include_attitude,
        include_reference_body=args.include_reference_body,
        body_allowlist=parse_body_list(args.bodies),
        body_blocklist=parse_body_list(args.exclude_bodies) or [],
        resume=args.resume,
    )
    result = acquire_ephemerides(cfg, ksp_root=Path(args.ksp_root) if args.ksp_root else None)
    print(json.dumps({
        "output_dir": result["output_dir"],
        "principia_signal_classification": result["validation"].get("principia_signal_classification"),
        "body_count": len(result["manifest"].get("body_names", [])),
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
