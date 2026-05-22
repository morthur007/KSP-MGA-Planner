#!/usr/bin/env python3
"""
Client for principia_impulsive_particle_server_v0_4_vessel.

Protocol expected by this client
--------------------------------
READY principia_impulsive_particle_server_v0_4_vessel
PING
PONG

VPROPN <id> <vessel_guid> <final_dt_s> <n_burns>
       [<burn_dt_s> <dvx_raw_m_s> <dvy_raw_m_s> <dvz_raw_m_s>] * n_burns

Response:
OKVN <id> <vessel_guid> <t0_game_s> <t1_game_s> <n_burns>
     [burn_dt, burn_r_raw(3), burn_v_before_raw(3), burn_v_after_raw(3)] * n_burns
     initial_r_raw(3)
     initial_v_raw(3)
     initial_parent_r(3)
     initial_parent_v(3)
     initial_parent_distance
     initial_parent_speed
     initial_parent_radial_velocity
     final_r_raw(3)
     final_v_raw(3)
     final_parent_r(3)
     final_parent_v(3)
     final_parent_distance
     final_parent_speed
     final_parent_radial_velocity

All vectors are Principia raw/Barycentric in metres and metres/second.
All burn times and final_dt_s are relative to the vessel canonical t0
(vessel->psychohistory()->back().time on the server side).
"""

from __future__ import annotations

import argparse
import json
import select
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


class VesselServerError(RuntimeError):
    pass


def _as_float(token: str, name: str) -> float:
    try:
        return float(token)
    except Exception as exc:
        raise VesselServerError(f"Cannot parse {name} as float: {token!r}") from exc


def _as_int(token: str, name: str) -> int:
    try:
        return int(token)
    except Exception as exc:
        raise VesselServerError(f"Cannot parse {name} as int: {token!r}") from exc


class TokenReader:
    def __init__(self, tokens: Sequence[str]):
        self.tokens = list(tokens)
        self.i = 0

    def remaining(self) -> int:
        return len(self.tokens) - self.i

    def pop(self, name: str) -> str:
        if self.i >= len(self.tokens):
            raise VesselServerError(f"Missing token for {name}; consumed {self.i}/{len(self.tokens)}")
        v = self.tokens[self.i]
        self.i += 1
        return v

    def pop_float(self, name: str) -> float:
        return _as_float(self.pop(name), name)

    def pop_int(self, name: str) -> int:
        return _as_int(self.pop(name), name)

    def pop_vec3(self, name: str) -> list[float]:
        return [
            self.pop_float(f"{name}_x"),
            self.pop_float(f"{name}_y"),
            self.pop_float(f"{name}_z"),
        ]


def parse_okvn_line(line: str) -> dict[str, Any]:
    parts = line.strip().split()
    if not parts:
        raise VesselServerError("Empty server response")
    if parts[0] != "OKVN":
        raise VesselServerError(f"Expected OKVN, got: {line.strip()}")

    tr = TokenReader(parts[1:])
    request_id = tr.pop("id")
    vessel_guid = tr.pop("vessel_guid")
    t0_game_s = tr.pop_float("t0_game_s")
    t1_game_s = tr.pop_float("t1_game_s")
    n_burns = tr.pop_int("n_burns")

    burns: list[dict[str, Any]] = []
    for i in range(n_burns):
        burn_dt_s = tr.pop_float(f"burn[{i}].dt_s")
        burns.append({
            "burn_dt_s": burn_dt_s,
            "burn_r_raw_m": tr.pop_vec3(f"burn[{i}].r_raw_m"),
            "burn_v_before_raw_m_s": tr.pop_vec3(f"burn[{i}].v_before_raw_m_s"),
            "burn_v_after_raw_m_s": tr.pop_vec3(f"burn[{i}].v_after_raw_m_s"),
        })

    out = {
        "status": "ok",
        "request_id": request_id,
        "vessel_guid": vessel_guid,
        "t0_game_s": t0_game_s,
        "t1_game_s": t1_game_s,
        "final_dt_s": t1_game_s - t0_game_s,
        "n_burns": n_burns,
        "burns": burns,
        "initial_r_raw_m": tr.pop_vec3("initial_r_raw_m"),
        "initial_v_raw_m_s": tr.pop_vec3("initial_v_raw_m_s"),
        "initial_parent_r_m": tr.pop_vec3("initial_parent_r_m"),
        "initial_parent_v_m_s": tr.pop_vec3("initial_parent_v_m_s"),
        "initial_parent_distance_m": tr.pop_float("initial_parent_distance_m"),
        "initial_parent_speed_m_s": tr.pop_float("initial_parent_speed_m_s"),
        "initial_parent_radial_velocity_m_s": tr.pop_float("initial_parent_radial_velocity_m_s"),
        "final_r_raw_m": tr.pop_vec3("final_r_raw_m"),
        "final_v_raw_m_s": tr.pop_vec3("final_v_raw_m_s"),
        "final_parent_r_m": tr.pop_vec3("final_parent_r_m"),
        "final_parent_v_m_s": tr.pop_vec3("final_parent_v_m_s"),
        "final_parent_distance_m": tr.pop_float("final_parent_distance_m"),
        "final_parent_speed_m_s": tr.pop_float("final_parent_speed_m_s"),
        "final_parent_radial_velocity_m_s": tr.pop_float("final_parent_radial_velocity_m_s"),
        "raw_line": line.strip(),
    }

    if tr.remaining() != 0:
        out["extra_tokens"] = tr.tokens[tr.i:]
    return out


@dataclass
class Burn:
    burn_dt_s: float
    dv_raw_m_s: Sequence[float]

    def to_command_tokens(self) -> list[str]:
        if len(self.dv_raw_m_s) != 3:
            raise ValueError(f"dv_raw_m_s must have length 3, got {self.dv_raw_m_s!r}")
        return [
            f"{float(self.burn_dt_s):.17g}",
            f"{float(self.dv_raw_m_s[0]):.17g}",
            f"{float(self.dv_raw_m_s[1]):.17g}",
            f"{float(self.dv_raw_m_s[2]):.17g}",
        ]


class VesselPropnClient:
    def __init__(
        self,
        server: str | Sequence[str],
        plugin_b64: str | Path | None = None,
        *,
        startup_timeout_s: float = 30.0,
        response_timeout_s: float = 300.0,
        quiet_stderr: bool = False,
        cwd: str | Path | None = None,
        extra_args: Sequence[str] | None = None,
        plugin_arg_mode: str = "option",
    ):
        self.server = server
        self.plugin_b64 = Path(plugin_b64) if plugin_b64 is not None else None
        self.startup_timeout_s = float(startup_timeout_s)
        self.response_timeout_s = float(response_timeout_s)
        self.quiet_stderr = bool(quiet_stderr)
        self.cwd = Path(cwd) if cwd is not None else None
        self.extra_args = list(extra_args or [])
        self.plugin_arg_mode = plugin_arg_mode
        self.proc: subprocess.Popen[str] | None = None

    def _build_cmd(self) -> list[str]:
        if isinstance(self.server, str):
            cmd = shlex.split(self.server)
        else:
            cmd = [str(x) for x in self.server]

        if self.plugin_b64 is not None:
            if self.plugin_arg_mode == "option":
                cmd += ["--plugin-b64", str(self.plugin_b64)]
            elif self.plugin_arg_mode == "positional":
                cmd += [str(self.plugin_b64)]
            elif self.plugin_arg_mode == "none":
                pass
            else:
                raise ValueError("plugin_arg_mode must be option, positional, or none")
        cmd += self.extra_args
        return cmd

    def start(self) -> "VesselPropnClient":
        if self.proc is not None:
            return self

        stderr = subprocess.DEVNULL if self.quiet_stderr else None
        self.proc = subprocess.Popen(
            self._build_cmd(),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=stderr,
            text=True,
            bufsize=1,
            cwd=str(self.cwd) if self.cwd is not None else None,
        )

        ready = self._readline(timeout_s=self.startup_timeout_s)
        if not ready.startswith("READY"):
            self.close()
            raise VesselServerError(f"Expected READY from server, got: {ready!r}")

        pong = self.command("PING", timeout_s=self.startup_timeout_s)
        if pong.strip() != "PONG":
            self.close()
            raise VesselServerError(f"Expected PONG from server, got: {pong!r}")
        return self

    def close(self) -> None:
        proc = self.proc
        self.proc = None
        if proc is None:
            return
        try:
            if proc.stdin:
                try:
                    proc.stdin.write("QUIT\n")
                    proc.stdin.flush()
                except Exception:
                    pass
            proc.terminate()
            try:
                proc.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                proc.kill()
        finally:
            pass

    def __enter__(self) -> "VesselPropnClient":
        return self.start()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _readline(self, timeout_s: float | None = None) -> str:
        if self.proc is None or self.proc.stdout is None:
            raise VesselServerError("Server process is not running")
        timeout_s = self.response_timeout_s if timeout_s is None else float(timeout_s)
        fd = self.proc.stdout.fileno()
        ready, _, _ = select.select([fd], [], [], timeout_s)
        if not ready:
            raise VesselServerError(f"Timed out waiting for server response after {timeout_s:.1f}s")
        line = self.proc.stdout.readline()
        if line == "":
            rc = self.proc.poll()
            raise VesselServerError(f"Server closed stdout unexpectedly; returncode={rc}")
        return line.rstrip("\n")

    def command(self, line: str, *, timeout_s: float | None = None) -> str:
        if self.proc is None:
            self.start()
        assert self.proc is not None and self.proc.stdin is not None
        self.proc.stdin.write(line.rstrip("\n") + "\n")
        self.proc.stdin.flush()
        return self._readline(timeout_s=timeout_s)

    def vpropn(
        self,
        request_id: str,
        vessel_guid: str,
        final_dt_s: float,
        burns: Iterable[Burn | tuple[float, Sequence[float]]],
        *,
        timeout_s: float | None = None,
    ) -> dict[str, Any]:
        burn_objs: list[Burn] = []
        for b in burns:
            if isinstance(b, Burn):
                burn_objs.append(b)
            else:
                burn_objs.append(Burn(float(b[0]), b[1]))

        tokens = [
            "VPROPN",
            str(request_id),
            str(vessel_guid),
            f"{float(final_dt_s):.17g}",
            str(len(burn_objs)),
        ]
        for burn in burn_objs:
            tokens.extend(burn.to_command_tokens())

        line = self.command("\t".join(tokens), timeout_s=timeout_s)
        return parse_okvn_line(line)


def main() -> int:
    ap = argparse.ArgumentParser(description="Smoke-test client for principia_impulsive_particle_server_v0_4_vessel.")
    ap.add_argument("--server", required=True)
    ap.add_argument("--plugin-b64", required=True)
    ap.add_argument("--plugin-arg-mode", choices=["option", "positional", "none"], default="option")
    ap.add_argument("--vessel-guid", required=True)
    ap.add_argument("--final-dt-s", type=float, default=600.0)
    ap.add_argument("--burn", action="append", default=[], help="Burn as 'dt,dvx,dvy,dvz' in raw m/s. Can be repeated.")
    ap.add_argument("--quiet-stderr", action="store_true")
    ap.add_argument("--timeout-s", type=float, default=120.0)
    ap.add_argument("--output-json", type=Path, default=None)
    args = ap.parse_args()

    burns: list[Burn] = []
    for spec in args.burn:
        vals = [float(x.strip()) for x in spec.replace(";", ",").split(",") if x.strip()]
        if len(vals) != 4:
            raise SystemExit(f"--burn must have 4 comma-separated values, got {spec!r}")
        burns.append(Burn(vals[0], vals[1:]))

    with VesselPropnClient(
        args.server,
        args.plugin_b64,
        quiet_stderr=args.quiet_stderr,
        response_timeout_s=args.timeout_s,
        plugin_arg_mode=args.plugin_arg_mode,
    ) as client:
        result = client.vpropn("client_smoke", args.vessel_guid, args.final_dt_s, burns, timeout_s=args.timeout_s)

    text = json.dumps(result, indent=2)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text + "\n")
        print(f"[OK] wrote {args.output_json}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
