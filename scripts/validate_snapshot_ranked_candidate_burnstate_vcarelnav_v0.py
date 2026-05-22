#!/usr/bin/env python3
"""
validate_snapshot_ranked_candidate_burnstate_vcarelnav_v0.py

Valida um candidato do ranker snapshot-aware SEM propagar a nave desde o snapshot
por centenas de dias.

Em vez de:
  SNAPVCA_NAV(snapshot_atual, impulse_dt=burn_dt_s)

usa:
  VCAREL_NAV(state_at_burn, impulse_dt=0)

onde state_at_burn = burn_rel_r_raw_m / burn_rel_v_raw_m_s gravados pelo
rank_pykep_candidates_by_snapshot_executability_v0.py.

Isto é uma validação de transfer pós-queima e não valida a deriva N-body da
órbita de estacionamento até a queima. Para queima muito futura, a validação
final deve ser refeita perto da data da manobra com novo snapshot vivo.
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
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

DAY_S = 86400.0


def sanitize_body(x: Any) -> str:
    return str(x or "").strip().upper()


def vec3(fields: Sequence[str], i: int) -> list[float]:
    return [float(fields[i]), float(fields[i + 1]), float(fields[i + 2])]


def norm3(v: Sequence[float]) -> float:
    return math.sqrt(sum(float(x) * float(x) for x in v))


def split_offsets(s: str) -> list[float]:
    return [float(x.strip()) for x in str(s).split(",") if x.strip()]


def json_sanitize(obj: Any) -> Any:
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {str(k): json_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_sanitize(v) for v in obj]
    return obj


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


class VCarelNavClient:
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

    def vcarel_nav(
        self,
        rid: str,
        dep_body: str,
        arr_body: str,
        nav_body: str,
        state_dt_s: float,
        scan_start_dt_s: float,
        scan_end_dt_s: float,
        samples: int,
        rel_r_raw_m: Sequence[float],
        rel_v_raw_m_s: Sequence[float],
        impulses: Sequence[tuple[float, float, float, float]],
        timeout_s: float | None = None,
    ) -> OkCarelNav:
        fields: list[Any] = [
            "VCAREL_NAV",
            rid,
            sanitize_body(dep_body),
            sanitize_body(arr_body),
            nav_body if nav_body else "AUTO",
            float(state_dt_s),
            float(scan_start_dt_s),
            float(scan_end_dt_s),
            int(samples),
            float(rel_r_raw_m[0]),
            float(rel_r_raw_m[1]),
            float(rel_r_raw_m[2]),
            float(rel_v_raw_m_s[0]),
            float(rel_v_raw_m_s[1]),
            float(rel_v_raw_m_s[2]),
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


def select_candidate(rank: dict[str, Any], row_index0: int | None, top_index: int | None) -> dict[str, Any]:
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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rank-json", type=Path, required=True)
    ap.add_argument("--row-index0", type=int, default=None)
    ap.add_argument("--top-index", type=int, default=0)

    ap.add_argument("--server", type=Path, required=True)
    ap.add_argument("--plugin-b64", type=Path, required=True)
    ap.add_argument("--plugin-arg-mode", choices=["positional", "flag", "none"], default="positional")

    ap.add_argument("--dep-body", default=None)
    ap.add_argument("--arr-body", default=None)
    ap.add_argument("--nav-body", default="AUTO")

    ap.add_argument("--arrival-offsets-days", default="-30,-15,0,15,30")
    ap.add_argument("--scan-half-width-days", type=float, default=45.0)
    ap.add_argument("--samples", type=int, default=61)

    ap.add_argument("--startup-timeout-s", type=float, default=90.0)
    ap.add_argument("--command-timeout-s", type=float, default=900.0)
    ap.add_argument("--quiet-stderr", action="store_true")
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    rank = json.loads(args.rank_json.read_text())
    cand = select_candidate(rank, args.row_index0, args.top_index)

    seq = str(cand.get("sequence") or "")
    seq_parts = seq.split()

    dep_body = sanitize_body(args.dep_body or cand.get("dep_body") or (seq_parts[0] if len(seq_parts) >= 1 else ""))
    arr_body = sanitize_body(args.arr_body or cand.get("arr_body") or (seq_parts[1] if len(seq_parts) >= 2 else ""))

    if not dep_body or not arr_body:
        raise SystemExit("Could not infer dep/arr; pass --dep-body and --arr-body.")

    burn_abs_s = float(cand["burn_abs_s"])
    t_arr_s = float(cand["t_arr_s"])
    dvt = float(cand["dv_tangent_m_s"])
    dvn = float(cand["dv_normal_m_s"])
    dvb = float(cand["dv_binormal_m_s"])
    rel_r = [float(x) for x in cand["burn_rel_r_raw_m"]]
    rel_v = [float(x) for x in cand["burn_rel_v_raw_m_s"]]

    stderr_log = args.output_dir / "principia_server_stderr.log"
    rows: list[dict[str, Any]] = []

    print("=== VALIDATE SNAPSHOT RANKED CANDIDATE FROM BURN STATE VCAREL_NAV V0 ===")
    print(f"rank_json       : {args.rank_json}")
    print(f"row_index0      : {cand.get('row_index0')}")
    print(f"sequence        : {seq}")
    print(f"pass_gates      : {cand.get('pass_gates')}")
    print(f"dep -> arr/nav  : {dep_body} -> {arr_body} / {args.nav_body}")
    print(f"state_abs_s     : {burn_abs_s}")
    print(f"t_arr_s         : {t_arr_s}")
    print(f"tof_after_burn_d: {(t_arr_s - burn_abs_s) / DAY_S:.6f}")
    print(f"rel_r_norm_km   : {norm3(rel_r)/1000:.6f}")
    print(f"rel_v_norm_m_s  : {norm3(rel_v):.6f}")
    print(f"dv_nav          : [{dvt}, {dvn}, {dvb}] |v|={norm3([dvt,dvn,dvb]):.6f}")
    print(f"rank score      : {cand.get('score_exec')}")
    print(f"output_dir      : {args.output_dir}")

    with VCarelNavClient(
        server=args.server,
        plugin_b64=args.plugin_b64,
        plugin_arg_mode=args.plugin_arg_mode,
        startup_timeout_s=args.startup_timeout_s,
        command_timeout_s=args.command_timeout_s,
        stderr_log=stderr_log,
        quiet_stderr=args.quiet_stderr,
    ) as client:
        print(f"ready_line      : {client.ready_line}")

        for i, off_days in enumerate(split_offsets(args.arrival_offsets_days), 1):
            center = (t_arr_s - burn_abs_s) + off_days * DAY_S
            scan_start = center - args.scan_half_width_days * DAY_S
            scan_end = center + args.scan_half_width_days * DAY_S

            row = {
                "ok": False,
                "row_index0": cand.get("row_index0"),
                "sequence": seq,
                "arrival_offset_days": off_days,
                "state_abs_s": burn_abs_s,
                "scan_start_dt_s": scan_start,
                "scan_end_dt_s": scan_end,
                "dvt_m_s": dvt,
                "dvn_m_s": dvn,
                "dvb_m_s": dvb,
                "dv_norm_m_s": norm3([dvt, dvn, dvb]),
            }

            try:
                res = client.vcarel_nav(
                    rid=f"burnstate_row{cand.get('row_index0')}_off{i:03d}",
                    dep_body=dep_body,
                    arr_body=arr_body,
                    nav_body=args.nav_body,
                    state_dt_s=burn_abs_s,
                    scan_start_dt_s=scan_start,
                    scan_end_dt_s=scan_end,
                    samples=args.samples,
                    rel_r_raw_m=rel_r,
                    rel_v_raw_m_s=rel_v,
                    impulses=[(0.0, dvt, dvn, dvb)],
                    timeout_s=args.command_timeout_s,
                )
                row.update({
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
                })
                print(
                    f"[VCA {i:2d}] off={off_days:7.2f}d "
                    f"ca={row['ca_distance_km']:12.3f} km "
                    f"ca_t={row['ca_t_game_s']:14.3f} "
                    f"speed={row['ca_speed_m_s']:9.3f} "
                    f"radial={row['ca_radial_velocity_m_s']:9.3f} "
                    f"status={row['status']}"
                )
            except Exception as exc:
                row.update({"ok": False, "error": str(exc)})
                print(f"[VCA {i:2d}] off={off_days:7.2f}d ERR {exc}")
                if isinstance(exc, TimeoutError) or "timeout" in str(exc).lower():
                    print("[ABORT] timeout; protocol stream may be desynchronized.")
                    rows.append(row)
                    break

            rows.append(row)

    good = [r for r in rows if r.get("ok")]
    good.sort(key=lambda r: float(r.get("ca_distance_km", math.inf)))
    best = good[0] if good else None

    out = {
        "schema": "snapshot_ranked_candidate_burnstate_vcarelnav_validation_v0",
        "rank_json": str(args.rank_json),
        "candidate": cand,
        "dep_body": dep_body,
        "arr_body": arr_body,
        "nav_body": args.nav_body,
        "state_abs_s": burn_abs_s,
        "n_rows": len(rows),
        "n_ok": len(good),
        "best": best,
        "rows": rows,
    }

    out_json = args.output_dir / "burnstate_vcarelnav_validation.json"
    out_csv = args.output_dir / "burnstate_vcarelnav_validation.csv"
    out_json.write_text(json.dumps(json_sanitize(out), indent=2) + "\n")

    if rows:
        flat_rows = []
        for r in rows:
            flat = {}
            for k, v in r.items():
                if isinstance(v, (list, dict)):
                    flat[k] = json.dumps(json_sanitize(v))
                else:
                    flat[k] = json_sanitize(v)
            flat_rows.append(flat)
        fields = sorted({k for r in flat_rows for k in r.keys()})
        with out_csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(flat_rows)

    print(f"[OK] wrote {out_json}")
    print(f"[OK] wrote {out_csv}")
    print(f"[OK] wrote {stderr_log}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
