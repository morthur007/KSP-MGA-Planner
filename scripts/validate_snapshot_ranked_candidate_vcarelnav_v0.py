#!/usr/bin/env python3
"""
validate_snapshot_ranked_candidate_vcarelnav_v0.py

Valida um candidato já ranqueado por:

  rank_pykep_candidates_by_snapshot_executability_v0.py

usando o protocolo novo:

  LOADSNAP
  SNAPVCA_NAV

Ponto crítico:
  Este script usa o burn_dt_s ABSOLUTO do ranker. Para uma row futura,
  por exemplo partida daqui a ~860 dias, o impulso é enviado como:

    impulses = [(burn_dt_s, dvt, dvn, dvb)]

  e NÃO como uma queima nos próximos 0..21600 segundos.

Também escreve um evento insert_navigation no formato da DLL, com
initial_time = burn_abs_s e delta_v_navigation_m_s = [T,N,B].
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import queue
import subprocess
import sys
import threading
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Sequence


DAY_S = 86400.0


def norm3(v: Sequence[float]) -> float:
    return math.sqrt(sum(float(x) * float(x) for x in v))


def vec3(fields: Sequence[str], i: int) -> list[float]:
    return [float(fields[i]), float(fields[i + 1]), float(fields[i + 2])]


def safe_float(x: Any, default: float) -> float:
    try:
        if x is None or x == "":
            return float(default)
        y = float(x)
        return y if math.isfinite(y) else float(default)
    except Exception:
        return float(default)


def split_offsets(s: str) -> list[float]:
    if not s:
        return [0.0]
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def sanitize_body(s: Any) -> str:
    return str(s or "").strip().upper()


def json_sanitize(obj: Any) -> Any:
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {str(k): json_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_sanitize(v) for v in obj]
    return obj


@dataclass
class OkSnap:
    id: str
    schema: str
    t_game_s: float
    vessel_guid: str
    dep_body: str
    nav_body: str
    rel_r_raw_m: list[float]
    rel_v_raw_m_s: list[float]
    mass_tonnes: float
    available_thrust_kN: float
    specific_impulse_s_g0: float


@dataclass
class BurnDiag:
    burn_dt_s: float
    burn_r_raw_m: list[float] | None = None
    burn_v_before_raw_m_s: list[float] | None = None
    dv_tnb_cmd_m_s: list[float] | None = None
    tangent_raw: list[float] | None = None
    normal_raw: list[float] | None = None
    binormal_raw: list[float] | None = None
    dv_raw_m_s: list[float] | None = None
    burn_v_after_raw_m_s: list[float] | None = None


@dataclass
class OkCarelNav:
    id: str
    dep_body: str
    arr_body: str
    nav_body: str
    state_dt_s: float
    state_t_game_s: float
    ca_dt_s: float
    ca_t_game_s: float
    ca_rel_r_raw_m: list[float]
    ca_rel_v_raw_m_s: list[float]
    ca_distance_m: float
    ca_speed_m_s: float
    ca_radial_velocity_m_s: float
    samples: int
    status: str
    ca_abs_debug_r_raw_m: list[float] | None = None
    ca_abs_debug_v_raw_m_s: list[float] | None = None
    arr_abs_debug_r_raw_m: list[float] | None = None
    arr_abs_debug_v_raw_m_s: list[float] | None = None
    n_burns: int = 0
    burns: list[BurnDiag] | None = None


class ProtocolError(RuntimeError):
    pass


class SnapshotNavClient:
    def __init__(
        self,
        server: Path,
        plugin_b64: Path,
        plugin_arg_mode: str = "positional",
        startup_timeout_s: float = 90.0,
        command_timeout_s: float = 900.0,
        stderr_log: Path | None = None,
        quiet_stderr: bool = False,
    ):
        self.server = Path(server)
        self.plugin_b64 = Path(plugin_b64)
        self.plugin_arg_mode = plugin_arg_mode
        self.startup_timeout_s = float(startup_timeout_s)
        self.command_timeout_s = float(command_timeout_s)
        self.stderr_log = Path(stderr_log) if stderr_log else None
        self.quiet_stderr = quiet_stderr
        self.proc: subprocess.Popen[str] | None = None
        self.stdout_q: queue.Queue[str] = queue.Queue()
        self.stderr_q: queue.Queue[str] = queue.Queue()
        self.stderr_file = None
        self.ready_line = ""

    def _argv(self) -> list[str]:
        argv = [str(self.server)]
        if self.plugin_arg_mode == "positional":
            argv.append(str(self.plugin_b64))
        elif self.plugin_arg_mode == "flag":
            argv.extend(["--plugin-b64", str(self.plugin_b64)])
        elif self.plugin_arg_mode == "none":
            pass
        else:
            raise ValueError(f"bad plugin_arg_mode={self.plugin_arg_mode!r}")
        return argv

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def start(self) -> None:
        if self.stderr_log:
            self.stderr_log.parent.mkdir(parents=True, exist_ok=True)
            self.stderr_file = self.stderr_log.open("w", encoding="utf-8")

        self.proc = subprocess.Popen(
            self._argv(),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            universal_newlines=True,
        )
        assert self.proc.stdout is not None
        assert self.proc.stderr is not None

        def pump_stdout():
            try:
                for line in self.proc.stdout:
                    self.stdout_q.put(line.rstrip("\n"))
            except Exception as e:
                self.stdout_q.put(f"__STDOUT_READER_ERROR__ {e!r}")

        def pump_stderr():
            try:
                for line in self.proc.stderr:
                    line = line.rstrip("\n")
                    self.stderr_q.put(line)
                    if self.stderr_file:
                        self.stderr_file.write(line + "\n")
                        self.stderr_file.flush()
                    if not self.quiet_stderr:
                        print(f"[server-stderr] {line}", file=sys.stderr)
            except Exception as e:
                self.stderr_q.put(f"__STDERR_READER_ERROR__ {e!r}")

        threading.Thread(target=pump_stdout, daemon=True).start()
        threading.Thread(target=pump_stderr, daemon=True).start()

        first = self._read_line(self.startup_timeout_s)
        if not first.startswith("READY"):
            raise ProtocolError(f"expected READY, got {first!r}")

        banners = [first]
        while True:
            try:
                extra = self.stdout_q.get(timeout=0.15)
            except queue.Empty:
                break
            if extra.startswith(("OK", "ERR", "PONG")):
                raise ProtocolError(f"unexpected protocol line during startup banner drain: {extra!r}")
            if extra:
                banners.append(extra)

        self.ready_line = "\t".join(banners)

    def close(self) -> None:
        if self.proc is not None:
            try:
                if self.proc.stdin:
                    self.proc.stdin.close()
            except Exception:
                pass
            try:
                self.proc.terminate()
                self.proc.wait(timeout=3)
            except Exception:
                try:
                    self.proc.kill()
                except Exception:
                    pass
            self.proc = None

        if self.stderr_file:
            try:
                self.stderr_file.close()
            except Exception:
                pass
            self.stderr_file = None

    def _read_line(self, timeout_s: float) -> str:
        try:
            line = self.stdout_q.get(timeout=timeout_s)
        except queue.Empty:
            rc = None if self.proc is None else self.proc.poll()
            raise TimeoutError(f"timeout waiting for server response; returncode={rc}")
        if line.startswith("__STDOUT_READER_ERROR__"):
            raise ProtocolError(line)
        return line

    def command_fields(self, fields: Sequence[Any], timeout_s: float | None = None) -> list[str]:
        if self.proc is None or self.proc.stdin is None:
            raise RuntimeError("server is not running")
        line = "\t".join(str(x) for x in fields)
        self.proc.stdin.write(line + "\n")
        self.proc.stdin.flush()

        out = self._read_line(timeout_s or self.command_timeout_s)
        parts = out.split("\t")
        if not parts:
            raise ProtocolError("empty response")
        if parts[0] == "ERR" or parts[0].startswith("ERR"):
            raise ProtocolError(out)
        return parts

    def loadsnap(self, snapshot_path: Path, rid: str = "snap0") -> OkSnap:
        p = self.command_fields(["LOADSNAP", rid, str(snapshot_path)], timeout_s=self.command_timeout_s)
        if p[0] != "OKSNAP":
            raise ProtocolError(f"expected OKSNAP, got {p[:8]}")
        if len(p) < 16:
            raise ProtocolError(f"OKSNAP too short ({len(p)} fields): {p}")

        return OkSnap(
            id=p[1],
            schema=p[2],
            t_game_s=float(p[3]),
            vessel_guid=p[4],
            dep_body=sanitize_body(p[5]),
            nav_body=sanitize_body(p[6]),
            rel_r_raw_m=vec3(p, 7),
            rel_v_raw_m_s=vec3(p, 10),
            mass_tonnes=float(p[13]),
            available_thrust_kN=float(p[14]),
            specific_impulse_s_g0=float(p[15]),
        )

    def snapvca_nav(
        self,
        rid: str,
        arr_body: str,
        nav_body: str,
        scan_start_dt_s: float,
        scan_end_dt_s: float,
        samples: int,
        impulses: Sequence[tuple[float, float, float, float]],
        timeout_s: float | None = None,
    ) -> OkCarelNav:
        fields: list[Any] = [
            "SNAPVCA_NAV",
            rid,
            sanitize_body(arr_body),
            nav_body if nav_body else "AUTO",
            float(scan_start_dt_s),
            float(scan_end_dt_s),
            int(samples),
            int(len(impulses)),
        ]
        for dt, dvt, dvn, dvb in impulses:
            fields.extend([float(dt), float(dvt), float(dvn), float(dvb)])
        parts = self.command_fields(fields, timeout_s=timeout_s)
        return parse_okcarelnav(parts)


def parse_okcarelnav(p: Sequence[str]) -> OkCarelNav:
    if not p or p[0] != "OKCARELNAV":
        raise ProtocolError(f"expected OKCARELNAV, got {p[:8]}")
    if len(p) < 33:
        raise ProtocolError(f"OKCARELNAV too short ({len(p)} fields): {p}")

    n_burns = int(p[32])
    burns: list[BurnDiag] = []
    i = 33
    for _ in range(n_burns):
        if i + 25 <= len(p):
            burns.append(
                BurnDiag(
                    burn_dt_s=float(p[i]),
                    burn_r_raw_m=vec3(p, i + 1),
                    burn_v_before_raw_m_s=vec3(p, i + 4),
                    dv_tnb_cmd_m_s=vec3(p, i + 7),
                    tangent_raw=vec3(p, i + 10),
                    normal_raw=vec3(p, i + 13),
                    binormal_raw=vec3(p, i + 16),
                    dv_raw_m_s=vec3(p, i + 19),
                    burn_v_after_raw_m_s=vec3(p, i + 22),
                )
            )
            i += 25
        else:
            burns.append(BurnDiag(burn_dt_s=float(p[i])))
            break

    return OkCarelNav(
        id=p[1],
        dep_body=sanitize_body(p[2]),
        arr_body=sanitize_body(p[3]),
        nav_body=sanitize_body(p[4]),
        state_dt_s=float(p[5]),
        state_t_game_s=float(p[6]),
        ca_dt_s=float(p[7]),
        ca_t_game_s=float(p[8]),
        ca_rel_r_raw_m=vec3(p, 9),
        ca_rel_v_raw_m_s=vec3(p, 12),
        ca_distance_m=float(p[15]),
        ca_speed_m_s=float(p[16]),
        ca_radial_velocity_m_s=float(p[17]),
        samples=int(p[18]),
        status=p[19],
        ca_abs_debug_r_raw_m=vec3(p, 20),
        ca_abs_debug_v_raw_m_s=vec3(p, 23),
        arr_abs_debug_r_raw_m=vec3(p, 26),
        arr_abs_debug_v_raw_m_s=vec3(p, 29),
        n_burns=n_burns,
        burns=burns,
    )


def select_candidate(rank: dict[str, Any], top_index: int | None, row_index0: int | None) -> dict[str, Any]:
    top = rank.get("top") or []
    if row_index0 is not None:
        for r in top:
            if int(r.get("row_index0", -999999)) == int(row_index0):
                return r
        raise SystemExit(f"row_index0={row_index0} not found in rank JSON top[]")
    idx = int(top_index or 0)
    if not (0 <= idx < len(top)):
        raise SystemExit(f"top_index={idx} outside top[0..{len(top)-1}]")
    return top[idx]


def write_event(output_dir: Path, snapshot_data: dict[str, Any], snap: OkSnap, cand: dict[str, Any], arr_body: str, nav_body: str) -> Path:
    vessel = snapshot_data.get("vessel", {}) if isinstance(snapshot_data, dict) else {}

    burn_abs = float(cand["burn_abs_s"])
    dvt = float(cand["dv_tangent_m_s"])
    dvn = float(cand["dv_normal_m_s"])
    dvb = float(cand["dv_binormal_m_s"])

    event = {
        "enabled": True,
        "vessel_guid": snap.vessel_guid or vessel.get("vessel_guid", ""),
        "ensure_flight_plan": True,
        "extend_existing_flight_plan": True,
        "mass_tonnes": safe_float(snap.mass_tonnes, safe_float(vessel.get("mass_tonnes"), 2.6)),
        "insert_index": 0,
        "burn_template": "json",
        "thrust_kN": safe_float(snap.available_thrust_kN, safe_float(vessel.get("available_thrust_kN"), 90.0)),
        "specific_impulse_s_g0": safe_float(snap.specific_impulse_s_g0, safe_float(vessel.get("specific_impulse_s_g0"), 345.0)),
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
        "request_id": f"snap_ranked_row{cand.get('row_index0')}_burn0_attempt0",
        "dedupe_tag": f"snap_ranked_row{cand.get('row_index0')}_burn0",
        "event_key": f"snap_ranked_row{cand.get('row_index0')}_burn0",
        "attempt": 0,
        "mode": "insert_navigation",
        "initial_time": burn_abs,
        "plan_final_time": burn_abs + 600.0,
        "delta_v_navigation_m_s": [dvt, dvn, dvb],
        "planned_from_state": {
            "schema": "planned_from_snapshot_ranked_candidate_vcarelnav_v0",
            "row_index0": cand.get("row_index0"),
            "sequence": cand.get("sequence"),
            "snapshot_t_game_s": snap.t_game_s,
            "burn_dt_s": cand.get("burn_dt_s"),
            "burn_abs_s": cand.get("burn_abs_s"),
            "t_arr_s": cand.get("t_arr_s"),
            "arr_body": arr_body,
            "nav_body": nav_body,
            "dv_norm_m_s": cand.get("dv_norm_m_s"),
            "score_exec": cand.get("score_exec"),
            "pass_gates": cand.get("pass_gates"),
            "rank_fields": {
                "phase_error_deg": cand.get("phase_error_deg"),
                "plane_angle_deg": cand.get("plane_angle_deg"),
                "out_of_plane_fraction": cand.get("out_of_plane_fraction"),
                "raw_sum_km_s": cand.get("raw_sum_km_s"),
            },
        },
        "require_warp_close": True,
        "max_lead_before_insert_s": 600.0,
        "reject_long_plan_before_first_burn": True,
        "max_first_plan_duration_s": 900.0,
        "rollback_on_status_error": True,
        "auto_sort_insert_index": True,
    }

    p = output_dir / f"event1_snapshot_ranked_row{cand.get('row_index0')}_burn0_navigation.json"
    p.write_text(json.dumps(json_sanitize(event), indent=2) + "\n")
    return p


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rank-json", type=Path, required=True)
    ap.add_argument("--top-index", type=int, default=0)
    ap.add_argument("--row-index0", type=int, default=None)

    ap.add_argument("--server", type=Path, required=True)
    ap.add_argument("--plugin-b64", type=Path, required=True)
    ap.add_argument("--plugin-arg-mode", choices=["positional", "flag", "none"], default="positional")
    ap.add_argument("--snapshot-json", type=Path, required=True)

    ap.add_argument("--arr-body", default=None)
    ap.add_argument("--nav-body", default="AUTO")
    ap.add_argument("--arrival-offsets-days", default="0")
    ap.add_argument("--scan-half-width-days", type=float, default=45.0)
    ap.add_argument("--samples", type=int, default=31)

    ap.add_argument("--startup-timeout-s", type=float, default=90.0)
    ap.add_argument("--command-timeout-s", type=float, default=1800.0)
    ap.add_argument("--quiet-stderr", action="store_true")
    ap.add_argument("--write-event", action="store_true")
    ap.add_argument("--write-event-even-if-no-vca", action="store_true")
    ap.add_argument("--skip-vca", action="store_true")
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    rank = json.loads(args.rank_json.read_text())
    snapshot_data = json.loads(args.snapshot_json.read_text())
    cand = select_candidate(rank, args.top_index, args.row_index0)

    row_index0 = int(cand["row_index0"])
    sequence = cand.get("sequence", "")
    arr_body = sanitize_body(args.arr_body or cand.get("arr_body") or (sequence.split()[1] if sequence else ""))
    nav_body = args.nav_body or "AUTO"

    burn_dt_s = float(cand["burn_dt_s"])
    burn_abs_s = float(cand["burn_abs_s"])
    t_arr_s = float(cand["t_arr_s"])
    dvt = float(cand["dv_tangent_m_s"])
    dvn = float(cand["dv_normal_m_s"])
    dvb = float(cand["dv_binormal_m_s"])
    dv_norm = norm3([dvt, dvn, dvb])

    stderr_log = args.output_dir / "principia_server_stderr.log"
    rows: list[dict[str, Any]] = []

    print("=== VALIDATE SNAPSHOT RANKED CANDIDATE VCAREL_NAV V0 ===")
    print(f"rank_json          : {args.rank_json}")
    print(f"row_index0         : {row_index0}")
    print(f"sequence           : {sequence}")
    print(f"pass_gates         : {cand.get('pass_gates')}")
    print(f"snapshot_json      : {args.snapshot_json}")
    print(f"arr/nav            : {arr_body} / {nav_body}")
    print(f"burn_dt_s          : {burn_dt_s}")
    print(f"burn_abs_s         : {burn_abs_s}")
    print(f"t_arr_s            : {t_arr_s}")
    print(f"wait_days          : {burn_dt_s / DAY_S:.6f}")
    print(f"arrival_from_snap_d: {(t_arr_s - float((snapshot_data.get('vessel') or snapshot_data).get('t_game_s', 0.0))) / DAY_S:.6f}")
    print(f"dv_nav             : [{dvt}, {dvn}, {dvb}] |v|={dv_norm:.6f}")
    print(f"rank phase/plane   : phase={cand.get('phase_error_deg')} plane={cand.get('plane_angle_deg')} oop={cand.get('out_of_plane_fraction')}")
    print(f"output_dir         : {args.output_dir}")

    with SnapshotNavClient(
        server=args.server,
        plugin_b64=args.plugin_b64,
        plugin_arg_mode=args.plugin_arg_mode,
        startup_timeout_s=args.startup_timeout_s,
        command_timeout_s=args.command_timeout_s,
        stderr_log=stderr_log,
        quiet_stderr=args.quiet_stderr,
    ) as client:
        print(f"ready_line         : {client.ready_line}")
        snap = client.loadsnap(args.snapshot_json, rid="snap0")
        print(f"loaded_snapshot    : t={snap.t_game_s} r={norm3(snap.rel_r_raw_m)/1000:.3f} km v={norm3(snap.rel_v_raw_m_s):.3f} m/s")

        if abs((snap.t_game_s + burn_dt_s) - burn_abs_s) > 1.0:
            print("[WARN] burn_abs_s differs from snap.t_game_s + burn_dt_s by >1s")

        if args.skip_vca:
            print("[SKIP] VCA validation disabled by --skip-vca")
        else:
            for j, off_days in enumerate(split_offsets(args.arrival_offsets_days), 1):
                scan_center = (t_arr_s - snap.t_game_s) + off_days * DAY_S
                scan_start = scan_center - args.scan_half_width_days * DAY_S
                scan_end = scan_center + args.scan_half_width_days * DAY_S
                rid = f"rankrow{row_index0}_off{j:03d}"

                row = {
                    "ok": False,
                    "row_index0": row_index0,
                    "sequence": sequence,
                    "arrival_offset_days": off_days,
                    "scan_start_dt_s": scan_start,
                    "scan_end_dt_s": scan_end,
                    "burn_dt_s": burn_dt_s,
                    "burn_abs_s": burn_abs_s,
                    "dvt_m_s": dvt,
                    "dvn_m_s": dvn,
                    "dvb_m_s": dvb,
                    "dv_norm_m_s": dv_norm,
                }

                try:
                    res = client.snapvca_nav(
                        rid=rid,
                        arr_body=arr_body,
                        nav_body=nav_body,
                        scan_start_dt_s=scan_start,
                        scan_end_dt_s=scan_end,
                        samples=args.samples,
                        impulses=[(burn_dt_s, dvt, dvn, dvb)],
                        timeout_s=args.command_timeout_s,
                    )
                    row.update(
                        {
                            "ok": True,
                            "ca_distance_km": res.ca_distance_m / 1000.0,
                            "ca_distance_m": res.ca_distance_m,
                            "ca_dt_s": res.ca_dt_s,
                            "ca_t_game_s": res.ca_t_game_s,
                            "ca_speed_m_s": res.ca_speed_m_s,
                            "ca_radial_velocity_m_s": res.ca_radial_velocity_m_s,
                            "status": res.status,
                            "ca_rel_r_raw_m": res.ca_rel_r_raw_m,
                            "ca_rel_v_raw_m_s": res.ca_rel_v_raw_m_s,
                            "burns": [asdict(b) for b in (res.burns or [])],
                        }
                    )
                    print(
                        f"[VCA {j:2d}] off={off_days:7.2f}d "
                        f"ca={row['ca_distance_km']:12.3f} km "
                        f"ca_t={row['ca_t_game_s']:14.3f} "
                        f"speed={row['ca_speed_m_s']:9.3f} "
                        f"radial={row['ca_radial_velocity_m_s']:9.3f} "
                        f"status={row['status']}"
                    )
                except Exception as exc:
                    row.update({"ok": False, "error": str(exc)})
                    print(f"[VCA {j:2d}] off={off_days:7.2f}d ERR {exc}")
                    if isinstance(exc, TimeoutError) or "timeout" in str(exc).lower():
                        print("[ABORT] timeout; protocol stream may be desynchronized.")
                        rows.append(row)
                        break

                rows.append(row)

        if args.write_event or args.write_event_even_if_no_vca:
            if args.write_event_even_if_no_vca or any(r.get("ok") for r in rows):
                p = write_event(args.output_dir, snapshot_data, snap, cand, arr_body, nav_body)
                print(f"[OK] wrote {p}")

    good = [r for r in rows if r.get("ok")]
    good.sort(key=lambda r: float(r.get("ca_distance_km", math.inf)))
    best = good[0] if good else None

    result = {
        "schema": "snapshot_ranked_candidate_vcarelnav_validation_v0",
        "rank_json": str(args.rank_json),
        "snapshot_json": str(args.snapshot_json),
        "candidate": cand,
        "arr_body": arr_body,
        "nav_body": nav_body,
        "n_rows": len(rows),
        "n_ok": len(good),
        "best": best,
        "rows": rows,
    }

    json_path = args.output_dir / "snapshot_ranked_candidate_vcarelnav_validation.json"
    csv_path = args.output_dir / "snapshot_ranked_candidate_vcarelnav_validation.csv"

    json_path.write_text(json.dumps(json_sanitize(result), indent=2) + "\n")

    if rows:
        flat_rows = []
        for r in rows:
            flat = {}
            for k, v in r.items():
                if isinstance(v, list):
                    flat[k] = json.dumps(json_sanitize(v))
                elif isinstance(v, dict):
                    flat[k] = json.dumps(json_sanitize(v))
                else:
                    flat[k] = json_sanitize(v)
            flat_rows.append(flat)
        fields = sorted({k for r in flat_rows for k in r})
        with csv_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(flat_rows)

    print(f"[OK] wrote {json_path}")
    print(f"[OK] wrote {csv_path}")
    print(f"[OK] wrote {stderr_log}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
