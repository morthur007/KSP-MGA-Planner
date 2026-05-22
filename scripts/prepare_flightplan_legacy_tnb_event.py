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
except Exception as e:
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
    for rk, vk in [
        ("r_raw_m", "v_raw_m_s"),
        ("position_raw_m", "velocity_raw_m_s"),
        ("r_m", "v_m_s"),
    ]:
        if rk in obj and vk in obj:
            return t, arr3(obj, rk), arr3(obj, vk)
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

    def propn(
        self,
        req_id: str,
        t0_s: float,
        t1_s: float,
        r0_m: np.ndarray,
        v0_m_s: np.ndarray,
        impulses: list[tuple[float, np.ndarray]],
    ) -> PropResult:
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


def tnb_basis_from_state(
    r_ship_raw: np.ndarray,
    v_ship_raw: np.ndarray,
    t_s: float,
    basis_body: str,
    center: str,
    frame: str,
    mode: str,
) -> dict[str, np.ndarray]:
    body_r, body_v = body_state_raw(basis_body, t_s, center, frame)
    rel_r = r_ship_raw - body_r
    rel_v = v_ship_raw - body_v

    if mode == "frenet_body":
        # Intended to match vessel Tangent/Normal/Binormal in a body-centred navigation frame.
        T = unit(rel_v, "relative velocity")
        B = unit(np.cross(rel_r, rel_v), "relative angular momentum/binormal")
        N = unit(np.cross(B, T), "normal")
    elif mode == "rtn_body":
        R = unit(rel_r, "relative radius")
        B = unit(np.cross(rel_r, rel_v), "relative angular momentum/binormal")
        T = unit(np.cross(B, R), "transverse")
        N = R
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
    return np.array([
        float(np.dot(dv_raw, basis["T_raw"])),
        float(np.dot(dv_raw, basis["N_raw"])),
        float(np.dot(dv_raw, basis["B_raw"])),
    ], dtype=float)


LEGACY_DEFAULTS: dict[str, Any] = {
    "enabled": True,
    "vessel_guid": "",
    "ensure_flight_plan": True,
    "extend_existing_flight_plan": True,
    "mass_tonnes": 2.6,
    "insert_index": 0,
    "burn_template": "json",
    "thrust_kN": 2686.87701225281,
    "specific_impulse_s_g0": 1000.0,
    "is_inertially_fixed": False,
    "frame_extension": 6000,
    "frame_centre_from_active_body": True,
    "frame_centre_index": -1,
    "frame_primary_index": -1,
    "frame_secondary_index": -1,
    "placeholder_dv_m_s": 0.001,
    "require_status_ok": True,
    "cleanup_on_error": True,
    "tolerance_time_s": 0.01,
    "tolerance_dv_m_s": 1e-6,
    "one_shot": True,
    "disable_after_success": True,
    "mode": "insert_levela",
    "require_warp_close": True,
    "max_lead_before_insert_s": 600.0,
    "reject_long_plan_before_first_burn": True,
    "max_first_plan_duration_s": 900.0,
    "rollback_on_status_error": True,
    "auto_sort_insert_index": True,
}


def make_legacy_event(
    template: dict[str, Any],
    *,
    request_prefix: str,
    event_key: str,
    burn_t: float,
    tnb: np.ndarray,
    plan_duration_s: float,
    insert_index: int | None,
    inertially_fixed: str,
) -> dict[str, Any]:
    # Strict legacy schema: copy template/defaults, update only fields the DLL expects.
    ev = dict(LEGACY_DEFAULTS)
    ev.update(template)
    ev["enabled"] = True
    ev["ensure_flight_plan"] = True
    ev["extend_existing_flight_plan"] = True
    ev["request_id"] = f"{request_prefix}_attempt0"
    ev["dedupe_tag"] = request_prefix
    ev["event_key"] = event_key
    ev["attempt"] = 0
    ev["mode"] = "insert_levela"
    ev["initial_time"] = float(burn_t)
    ev["plan_final_time"] = float(burn_t + plan_duration_s)
    ev["delta_v_levela_m_s"] = [float(x) for x in tnb]
    if insert_index is not None:
        ev["insert_index"] = int(insert_index)
    if inertially_fixed == "true":
        ev["is_inertially_fixed"] = True
    elif inertially_fixed == "false":
        ev["is_inertially_fixed"] = False
    # else preserve template/default.
    return ev


def minimal_report_event(
    name: str,
    burn_t: float,
    dv_raw: np.ndarray,
    basis: dict[str, np.ndarray],
    tnb: np.ndarray,
    legacy_event_path: str,
) -> dict[str, Any]:
    return {
        "name": name,
        "legacy_event_path": legacy_event_path,
        "initial_time": float(burn_t),
        "dv_raw_m_s": dv_raw.tolist(),
        "dv_levela_inertial_m_s": raw_to_levela(dv_raw).tolist(),
        "dv_tnb_m_s": tnb.tolist(),
        "dv_norm_m_s": norm(dv_raw),
        "tnb_norm_m_s": norm(tnb),
        "projection_error_m_s": abs(norm(tnb) - norm(dv_raw)),
        "dominant_component": ["tangent", "normal", "binormal"][int(np.argmax(np.abs(tnb)))],
        "basis": {
            "T_raw": basis["T_raw"].tolist(),
            "N_raw": basis["N_raw"].tolist(),
            "B_raw": basis["B_raw"].tolist(),
            "distance_from_basis_body_km": norm(basis["rel_r_raw_m"]) / 1000.0,
            "speed_relative_to_basis_body_m_s": norm(basis["rel_v_raw_m_s"]),
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Prepare DLL-compatible legacy JSON events, projecting inertial raw/LevelA burns into Tangent/Normal/Binormal components.")
    ap.add_argument("--event-preview", required=True, type=Path)
    ap.add_argument("--template-json", type=Path, help="Existing legacy event JSON; its schema/fields are preserved.")
    ap.add_argument("--live-state-json", required=True, type=Path)
    ap.add_argument("--plugin-b64", required=True)
    ap.add_argument("--server", default="principia_impulsive_particle_server")
    ap.add_argument("--bsp", required=True, type=Path)
    ap.add_argument("--tpc", required=True, type=Path)
    ap.add_argument("--basis-body", default="KERBIN")
    ap.add_argument("--center", default="SUN")
    ap.add_argument("--frame", default="J2000")
    ap.add_argument("--basis-mode", choices=["frenet_body", "rtn_body"], default="frenet_body")
    ap.add_argument("--include-dsm", action="store_true")
    ap.add_argument("--plan-duration-s", type=float, default=30.0)
    ap.add_argument("--request-prefix", default="rank12_leg1_tnb_departure")
    ap.add_argument("--event-key", default="rank12_leg1_departure_burn0_tnb")
    ap.add_argument("--dsm-request-prefix", default="rank12_leg1_tnb_dsm")
    ap.add_argument("--dsm-event-key", default="rank12_leg1_dsm_tnb")
    ap.add_argument("--insert-index", type=int, default=None, help="Override insert_index for departure event. If omitted, template/default is preserved.")
    ap.add_argument("--dsm-insert-index", type=int, default=None)
    ap.add_argument("--inertially-fixed", choices=["preserve", "true", "false"], default="false", help="Legacy UI T/N/B burns generally use false. Default false.")
    ap.add_argument("--quiet-stderr", action="store_true")
    ap.add_argument("--output-dir", required=True, type=Path)
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    spice.kclear()
    spice.furnsh(str(args.tpc))
    spice.furnsh(str(args.bsp))

    template: dict[str, Any] = {}
    if args.template_json:
        template = json.loads(args.template_json.read_text())

    evp = json.loads(args.event_preview.read_text())
    live_t, live_r, live_v = load_live_state(args.live_state_json)

    events: list[dict[str, Any]] = []
    report: dict[str, Any] = {
        "source_event_preview": str(args.event_preview),
        "template_json": str(args.template_json) if args.template_json else None,
        "basis_body": args.basis_body,
        "basis_mode": args.basis_mode,
        "note": "Legacy event delta_v_levela_m_s contains T/N/B components because this DLL mode reads that field as NavigationFrame burn.delta_v. True inertial vectors are only in this report.",
        "events": [],
    }

    burn1_t = float(evp["initial_time"])
    burn1_dv_raw = arr3(evp, "delta_v_raw_m_s")

    with ServerSession(args.server, args.plugin_b64, args.quiet_stderr) as srv:
        pre1 = srv.propn("legacy_tnb_preburn1", live_t, burn1_t, live_r, live_v, [(burn1_t, np.zeros(3))])
        if pre1.status != "ok" or not pre1.burns:
            raise SystemExit(f"[FAIL] preburn1 propagation failed: {pre1.status} {pre1.message}")
        b1 = pre1.burns[0]
        basis1 = tnb_basis_from_state(b1.r_m, b1.v_before_m_s, burn1_t, args.basis_body, args.center, args.frame, args.basis_mode)
        tnb1 = project_to_tnb(burn1_dv_raw, basis1)
        ev1 = make_legacy_event(
            template,
            request_prefix=args.request_prefix,
            event_key=args.event_key,
            burn_t=burn1_t,
            tnb=tnb1,
            plan_duration_s=args.plan_duration_s,
            insert_index=args.insert_index,
            inertially_fixed=args.inertially_fixed,
        )
        path1 = args.output_dir / "event1_departure_legacy_tnb.json"
        events.append(ev1)
        report["events"].append(minimal_report_event("event1_departure", burn1_t, burn1_dv_raw, basis1, tnb1, str(path1)))

        if args.include_dsm and evp.get("dsm_preview"):
            dsm = evp["dsm_preview"]
            dsm_t = float(dsm["initial_time"])
            dsm_dv_raw = arr3(dsm, "delta_v_raw_m_s")
            pre2 = srv.propn("legacy_tnb_predsm", live_t, dsm_t, live_r, live_v, [(burn1_t, burn1_dv_raw), (dsm_t, np.zeros(3))])
            if pre2.status != "ok" or not pre2.burns:
                raise SystemExit(f"[FAIL] predsm propagation failed: {pre2.status} {pre2.message}")
            b2 = pre2.burns[-1]
            basis2 = tnb_basis_from_state(b2.r_m, b2.v_before_m_s, dsm_t, args.basis_body, args.center, args.frame, args.basis_mode)
            tnb2 = project_to_tnb(dsm_dv_raw, basis2)
            ev2 = make_legacy_event(
                template,
                request_prefix=args.dsm_request_prefix,
                event_key=args.dsm_event_key,
                burn_t=dsm_t,
                tnb=tnb2,
                plan_duration_s=args.plan_duration_s,
                insert_index=args.dsm_insert_index,
                inertially_fixed=args.inertially_fixed,
            )
            path2 = args.output_dir / "event2_dsm_legacy_tnb.json"
            events.append(ev2)
            report["events"].append(minimal_report_event("event2_dsm", dsm_t, dsm_dv_raw, basis2, tnb2, str(path2)))

    # Write only DLL-compatible legacy JSON files. Diagnostics go to separate report.
    if events:
        (args.output_dir / "event1_departure_legacy_tnb.json").write_text(json.dumps(events[0], indent=2) + "\n")
    if len(events) > 1:
        (args.output_dir / "event2_dsm_legacy_tnb.json").write_text(json.dumps(events[1], indent=2) + "\n")
    with (args.output_dir / "mission_events_legacy_tnb.jsonl").open("w") as f:
        for ev in events:
            f.write(json.dumps(ev, separators=(",", ":")) + "\n")
    (args.output_dir / "legacy_tnb_projection_report.json").write_text(json.dumps(report, indent=2) + "\n")

    print("=== LEGACY TNB EVENT PREP ===")
    print(f"basis_body : {args.basis_body}")
    print(f"basis_mode : {args.basis_mode}")
    print(f"template   : {args.template_json if args.template_json else '[built-in defaults]'}")
    for r in report["events"]:
        tnb = r["dv_tnb_m_s"]
        print(f"{r['name']:18s} t={r['initial_time']:.6f} |dv|={r['dv_norm_m_s']:.3f}  T={tnb[0]: .3f} N={tnb[1]: .3f} B={tnb[2]: .3f} dominant={r['dominant_component']}")
    print(f"[OK] wrote {args.output_dir / 'event1_departure_legacy_tnb.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
