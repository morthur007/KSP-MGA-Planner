#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
try:
    import spiceypy as spice
except Exception as e:  # pragma: no cover
    raise SystemExit(f"[FAIL] spiceypy is required: {e}")


def norm(v: np.ndarray) -> float:
    return float(np.linalg.norm(v))


def unit(v: np.ndarray, name: str = "vector") -> np.ndarray:
    n = norm(v)
    if n <= 0 or not np.isfinite(n):
        raise ValueError(f"cannot normalize {name}: norm={n}")
    return v / n


def raw_to_levela(v: np.ndarray) -> np.ndarray:
    # Principia raw -> LevelA/SPICE canonical: (X,Y,Z) -> (-Y,+Z,+X)
    return np.array([-v[1], v[2], v[0]], dtype=float)


def levela_to_raw(v: np.ndarray) -> np.ndarray:
    # LevelA/SPICE canonical -> Principia raw: (X,Y,Z) -> (+Z,-X,+Y)
    return np.array([v[2], -v[0], v[1]], dtype=float)


def arr3(obj: dict[str, Any], key: str) -> np.ndarray:
    if key not in obj:
        raise KeyError(f"missing {key}")
    a = np.array([float(x) for x in obj[key]], dtype=float)
    if a.shape != (3,):
        raise ValueError(f"{key} is not length-3")
    return a


def load_live_state(path: Path) -> tuple[float, np.ndarray, np.ndarray]:
    obj = json.loads(path.read_text())
    t = float(obj.get("t_s", obj.get("ut", obj.get("ut_s"))))
    # common variants from capture_live_state_raw.py / project artifacts
    for rk, vk in [
        ("r_raw_m", "v_raw_m_s"),
        ("position_raw_m", "velocity_raw_m_s"),
        ("r_m", "v_m_s"),
    ]:
        if rk in obj and vk in obj:
            return t, arr3(obj, rk), arr3(obj, vk)
    # flat fields fallback
    r_keys = ["x_raw_m", "y_raw_m", "z_raw_m"]
    v_keys = ["vx_raw_m_s", "vy_raw_m_s", "vz_raw_m_s"]
    if all(k in obj for k in r_keys + v_keys):
        return t, np.array([float(obj[k]) for k in r_keys]), np.array([float(obj[k]) for k in v_keys])
    raise KeyError(f"could not find raw state vectors in {path}")


@dataclass
class BurnSnapshot:
    burn_t_s: float
    r_m: np.ndarray
    v_before_m_s: np.ndarray
    v_after_m_s: np.ndarray


@dataclass
class PropResult:
    status: str
    message: str
    id: str
    burns: list[BurnSnapshot]
    final_r_m: np.ndarray | None = None
    final_v_m_s: np.ndarray | None = None


class ServerSession:
    def __init__(self, server: str, plugin_b64: str, quiet_stderr: bool = False):
        stderr = subprocess.DEVNULL if quiet_stderr else None
        self.proc = subprocess.Popen(
            [server, plugin_b64],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=stderr,
            text=True,
            bufsize=1,
        )
        assert self.proc.stdout is not None
        ready = self.proc.stdout.readline().strip()
        if not ready.startswith("READY"):
            raise RuntimeError(f"server did not become ready: {ready!r}")

    def close(self) -> None:
        if self.proc.poll() is None:
            try:
                if self.proc.stdin:
                    self.proc.stdin.write("QUIT\n")
                    self.proc.stdin.flush()
                if self.proc.stdout:
                    self.proc.stdout.readline()
            except Exception:
                pass
            try:
                self.proc.terminate()
            except Exception:
                pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def propn(self, req_id: str, t0_s: float, t1_s: float, r0_m: np.ndarray, v0_m_s: np.ndarray, impulses: list[tuple[float, np.ndarray]]) -> PropResult:
        fields: list[str] = [
            "PROPN", req_id,
            f"{float(t0_s):.17g}", f"{float(t1_s):.17g}", str(len(impulses)),
            *[f"{float(x):.17g}" for x in r0_m],
            *[f"{float(x):.17g}" for x in v0_m_s],
        ]
        for burn_t, dv in impulses:
            fields.append(f"{float(burn_t):.17g}")
            fields.extend(f"{float(x):.17g}" for x in dv)
        assert self.proc.stdin is not None and self.proc.stdout is not None
        self.proc.stdin.write("\t".join(fields) + "\n")
        self.proc.stdin.flush()
        line = self.proc.stdout.readline().strip()
        if not line:
            return PropResult("error", "empty server response", req_id, [])
        parts = line.split("\t")
        if parts[0] == "ERR":
            return PropResult("error", parts[2] if len(parts) > 2 else "", parts[1] if len(parts) > 1 else req_id, [])
        if parts[0] != "OKN":
            return PropResult("error", f"unexpected response {parts[0]}: {line}", req_id, [])
        try:
            rid = parts[1]
            n = int(parts[4])
            idx = 5
            burns: list[BurnSnapshot] = []
            for _ in range(n):
                bt = float(parts[idx]); idx += 1
                r = np.array([float(parts[idx]), float(parts[idx + 1]), float(parts[idx + 2])], dtype=float); idx += 3
                vb = np.array([float(parts[idx]), float(parts[idx + 1]), float(parts[idx + 2])], dtype=float); idx += 3
                va = np.array([float(parts[idx]), float(parts[idx + 1]), float(parts[idx + 2])], dtype=float); idx += 3
                burns.append(BurnSnapshot(bt, r, vb, va))
            fr = np.array([float(parts[idx]), float(parts[idx + 1]), float(parts[idx + 2])], dtype=float); idx += 3
            fv = np.array([float(parts[idx]), float(parts[idx + 1]), float(parts[idx + 2])], dtype=float)
            return PropResult("ok", "", rid, burns, fr, fv)
        except Exception as e:
            return PropResult("error", f"parse OKN failed: {e}: {line}", req_id, [])


def body_state_raw(body: str, t_s: float, center: str, frame: str) -> tuple[np.ndarray, np.ndarray]:
    st, _ = spice.spkezr(body, float(t_s), frame, "NONE", center)
    r_levela = np.array(st[:3], dtype=float) * 1000.0
    v_levela = np.array(st[3:], dtype=float) * 1000.0
    return levela_to_raw(r_levela), levela_to_raw(v_levela)


def tnb_basis_from_state(r_ship_raw: np.ndarray, v_ship_raw: np.ndarray, t_s: float, basis_body: str, center: str, frame: str, mode: str) -> dict[str, np.ndarray]:
    body_r, body_v = body_state_raw(basis_body, t_s, center, frame)
    rel_r = r_ship_raw - body_r
    rel_v = v_ship_raw - body_v

    if mode == "frenet_body":
        # Tangent along body-relative velocity. Binormal from orbital angular momentum.
        T = unit(rel_v, "relative velocity")
        B = unit(np.cross(rel_r, rel_v), "relative angular momentum")
        N = unit(np.cross(B, T), "frenet normal")
    elif mode == "rtn_body":
        # Useful for debugging only: R, T, N orbital basis. We still return keys T,N,B-like.
        R = unit(rel_r, "relative radius")
        H = unit(np.cross(rel_r, rel_v), "relative angular momentum")
        T = unit(np.cross(H, R), "transverse")
        N = R
        B = H
    elif mode == "velocity_barycentric":
        # Fallback when no central body is meaningful. Not ideal for LKO departure.
        T = unit(v_ship_raw, "barycentric velocity")
        B = unit(np.cross(r_ship_raw, v_ship_raw), "barycentric angular momentum")
        N = unit(np.cross(B, T), "barycentric frenet normal")
    else:
        raise ValueError(f"unknown basis mode {mode!r}")

    return {
        "T_raw": T,
        "N_raw": N,
        "B_raw": B,
        "T_levela": raw_to_levela(T),
        "N_levela": raw_to_levela(N),
        "B_levela": raw_to_levela(B),
        "rel_r_raw_m": rel_r,
        "rel_v_raw_m_s": rel_v,
    }


def project_to_tnb(dv_raw: np.ndarray, basis: dict[str, np.ndarray]) -> np.ndarray:
    T = basis["T_raw"]
    N = basis["N_raw"]
    B = basis["B_raw"]
    return np.array([float(np.dot(dv_raw, T)), float(np.dot(dv_raw, N)), float(np.dot(dv_raw, B))], dtype=float)


def make_event(base: dict[str, Any], *, name: str, burn_t: float, dv_raw: np.ndarray, tnb: np.ndarray, basis: dict[str, np.ndarray], plan_duration_s: float, compat_insert_levela: bool) -> dict[str, Any]:
    dv_levela = raw_to_levela(dv_raw)
    out = {
        "enabled": True,
        "request_id": f"{name}_attempt0",
        "dedupe_tag": name,
        "event_key": name,
        "attempt": 0,
        "ensure_flight_plan": True,
        "extend_existing_flight_plan": True,
        "insert_index": -1,
        "burn_template": base.get("burn_template", "json"),
        "thrust_kN": base.get("thrust_kN", 1000.0),
        "specific_impulse_s_g0": base.get("specific_impulse_s_g0", base.get("isp_s", 315.0)),
        "initial_time": float(burn_t),
        "plan_final_time": float(burn_t + plan_duration_s),
        "is_inertially_fixed": bool(base.get("is_inertially_fixed", True)),
        "delta_v_raw_m_s": dv_raw.tolist(),
        "delta_v_levela_inertial_m_s": dv_levela.tolist(),
        "delta_v_tnb_m_s": tnb.tolist(),
        "delta_v_tangent_normal_binormal_m_s": tnb.tolist(),
        "delta_v_norm_m_s": norm(dv_raw),
        "basis": {
            "mode": base.get("basis_mode"),
            "T_raw": basis["T_raw"].tolist(),
            "N_raw": basis["N_raw"].tolist(),
            "B_raw": basis["B_raw"].tolist(),
            "T_levela": basis["T_levela"].tolist(),
            "N_levela": basis["N_levela"].tolist(),
            "B_levela": basis["B_levela"].tolist(),
            "distance_from_basis_body_km": norm(basis["rel_r_raw_m"]) / 1000.0,
            "speed_relative_to_basis_body_m_s": norm(basis["rel_v_raw_m_s"]),
        },
        "diagnostics": {
            "tnb_norm_m_s": norm(tnb),
            "projection_error_m_s": abs(norm(tnb) - norm(dv_raw)),
            "dominant_component": ["tangent", "normal", "binormal"][int(np.argmax(np.abs(tnb)))],
        },
    }
    # Preferred if the bridge supports it.
    out["mode"] = "insert_tnb"
    out["delta_v_navigation_m_s"] = tnb.tolist()

    if compat_insert_levela:
        # Compatibility hack for bridges that treat delta_v_levela_m_s as Burn.delta_v components.
        # This is intentionally labelled so it is not mistaken for an inertial LevelA vector.
        out["mode"] = "insert_levela"
        out["delta_v_levela_m_s"] = tnb.tolist()
        out["compatibility_warning"] = (
            "delta_v_levela_m_s contains TNB components for the current bridge. "
            "The true inertial LevelA vector is delta_v_levela_inertial_m_s."
        )
    else:
        out["delta_v_levela_m_s"] = dv_levela.tolist()
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Project inertial N-body burn vectors to Principia FlightPlan Tangent/Normal/Binormal events.")
    ap.add_argument("--event-preview", required=True, type=Path)
    ap.add_argument("--live-state-json", required=True, type=Path)
    ap.add_argument("--plugin-b64", required=True)
    ap.add_argument("--server", default="principia_impulsive_particle_server")
    ap.add_argument("--bsp", required=True, type=Path)
    ap.add_argument("--tpc", required=True, type=Path)
    ap.add_argument("--basis-body", default="KERBIN")
    ap.add_argument("--center", default="SUN")
    ap.add_argument("--frame", default="J2000")
    ap.add_argument("--basis-mode", choices=["frenet_body", "rtn_body", "velocity_barycentric"], default="frenet_body")
    ap.add_argument("--include-dsm", action="store_true")
    ap.add_argument("--plan-duration-s", type=float, default=30.0)
    ap.add_argument("--quiet-stderr", action="store_true")
    ap.add_argument("--compat-insert-levela", action="store_true", help="Write mode=insert_levela and put TNB components in delta_v_levela_m_s for the current bridge.")
    ap.add_argument("--output-dir", required=True, type=Path)
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    spice.kclear()
    spice.furnsh(str(args.tpc))
    spice.furnsh(str(args.bsp))

    evp = json.loads(args.event_preview.read_text())
    live_t, live_r, live_v = load_live_state(args.live_state_json)
    base = {
        "burn_template": "json",
        "thrust_kN": evp.get("thrust_kN", 1000.0),
        "specific_impulse_s_g0": evp.get("specific_impulse_s_g0", 315.0),
        "is_inertially_fixed": True,
        "basis_mode": args.basis_mode,
    }

    events: list[dict[str, Any]] = []
    report: dict[str, Any] = {
        "source_event_preview": str(args.event_preview),
        "basis_body": args.basis_body,
        "basis_mode": args.basis_mode,
        "compat_insert_levela": args.compat_insert_levela,
        "events": [],
    }

    burn1_t = float(evp["initial_time"])
    burn1_dv_raw = arr3(evp, "delta_v_raw_m_s")

    impulses_for_state: list[tuple[float, np.ndarray]] = [(burn1_t, np.zeros(3))]
    with ServerSession(args.server, args.plugin_b64, args.quiet_stderr) as srv:
        # Burn 1 pre-state.
        pre1 = srv.propn("tnb_preburn1", live_t, burn1_t, live_r, live_v, impulses_for_state)
        if pre1.status != "ok" or not pre1.burns:
            raise SystemExit(f"[FAIL] preburn1 propagation failed: {pre1.status} {pre1.message}")
        b1 = pre1.burns[0]
        basis1 = tnb_basis_from_state(b1.r_m, b1.v_before_m_s, burn1_t, args.basis_body, args.center, args.frame, args.basis_mode)
        tnb1 = project_to_tnb(burn1_dv_raw, basis1)
        event1 = make_event(base, name="event1_departure_tnb", burn_t=burn1_t, dv_raw=burn1_dv_raw, tnb=tnb1, basis=basis1, plan_duration_s=args.plan_duration_s, compat_insert_levela=args.compat_insert_levela)
        events.append(event1)
        report["events"].append({"name": "event1_departure_tnb", "tnb": tnb1.tolist(), "dv_norm": norm(burn1_dv_raw), "dominant": event1["diagnostics"]["dominant_component"]})

        if args.include_dsm and "dsm_preview" in evp and evp["dsm_preview"]:
            dsm = evp["dsm_preview"]
            dsm_t = float(dsm["initial_time"])
            dsm_dv_raw = arr3(dsm, "delta_v_raw_m_s")
            # Propagate to DSM with real burn1 applied, then a zero impulse at DSM to get pre-state.
            pre2 = srv.propn("tnb_predsm", live_t, dsm_t, live_r, live_v, [(burn1_t, burn1_dv_raw), (dsm_t, np.zeros(3))])
            if pre2.status != "ok" or not pre2.burns:
                raise SystemExit(f"[FAIL] predsm propagation failed: {pre2.status} {pre2.message}")
            b2 = pre2.burns[-1]
            basis2 = tnb_basis_from_state(b2.r_m, b2.v_before_m_s, dsm_t, args.basis_body, args.center, args.frame, args.basis_mode)
            tnb2 = project_to_tnb(dsm_dv_raw, basis2)
            event2 = make_event(base, name="event2_dsm_tnb", burn_t=dsm_t, dv_raw=dsm_dv_raw, tnb=tnb2, basis=basis2, plan_duration_s=args.plan_duration_s, compat_insert_levela=args.compat_insert_levela)
            events.append(event2)
            report["events"].append({"name": "event2_dsm_tnb", "tnb": tnb2.tolist(), "dv_norm": norm(dsm_dv_raw), "dominant": event2["diagnostics"]["dominant_component"]})

    # Write individual events and jsonl.
    for i, ev in enumerate(events, start=1):
        (args.output_dir / f"event{i}_{'departure' if i == 1 else 'dsm'}_tnb.json").write_text(json.dumps(ev, indent=2) + "\n")
    with (args.output_dir / "mission_events_tnb.jsonl").open("w") as f:
        for ev in events:
            f.write(json.dumps(ev, separators=(",", ":")) + "\n")
    (args.output_dir / "tnb_projection_report.json").write_text(json.dumps(report, indent=2) + "\n")

    print("=== TNB FLIGHTPLAN EVENT PREP ===")
    print(f"basis_body : {args.basis_body}")
    print(f"basis_mode : {args.basis_mode}")
    print(f"compat     : {args.compat_insert_levela}")
    for ev in events:
        tnb = ev["delta_v_tnb_m_s"]
        print(f"{ev['event_key']:24s} t={ev['initial_time']:.6f} |dv|={ev['delta_v_norm_m_s']:.3f}  T={tnb[0]: .3f} N={tnb[1]: .3f} B={tnb[2]: .3f} dominant={ev['diagnostics']['dominant_component']}")
    print(f"[OK] wrote {args.output_dir / 'mission_events_tnb.jsonl'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
