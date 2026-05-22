#!/usr/bin/env python3
"""
Persistent Python client for principia_impulsive_particle_server_v0_5_targeter.

Protocol notes:
- stdout is reserved for protocol lines.
- commands are sent as TAB-separated fields.
- impulse vectors are Principia raw / barycentric m/s.
- vessel-based times are seconds relative to the vessel canonical t0 inside Principia.

Supported:
  PING
  VPROPN
  VREL
  VCA
  VCAREL_NAV

The parser is intentionally strict for protocol tags, but tolerant of either
TAB-separated output or legacy whitespace-separated output.
"""

from __future__ import annotations

import argparse
import json
import os
import select
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


class PrincipiaServerError(RuntimeError):
    pass


@dataclass(frozen=True)
class Burn:
    dt_s: float
    dvx_raw_m_s: float
    dvy_raw_m_s: float
    dvz_raw_m_s: float

    def fields(self) -> list[str]:
        return [
            _fmt(self.dt_s),
            _fmt(self.dvx_raw_m_s),
            _fmt(self.dvy_raw_m_s),
            _fmt(self.dvz_raw_m_s),
        ]



@dataclass(frozen=True)
class NavBurn:
    """Impulse expressed in Principia FlightPlan navigation/Frenet components.

    dvt/dvn/dvb are Tangent, Normal and Binormal components in m/s. This is the
    operational language used by Principia manoeuvres and by insert_navigation.
    """

    dt_s: float
    dvt_m_s: float
    dvn_m_s: float
    dvb_m_s: float

    def fields(self) -> list[str]:
        return [
            _fmt(self.dt_s),
            _fmt(self.dvt_m_s),
            _fmt(self.dvn_m_s),
            _fmt(self.dvb_m_s),
        ]


def _fmt(x: float) -> str:
    return f"{float(x):.17g}"


def _split_line(line: str) -> list[str]:
    line = line.rstrip("\n")
    if "\t" in line:
        return line.split("\t")
    return line.split()


class _Tok:
    def __init__(self, fields: Sequence[str], start: int = 0):
        self.fields = list(fields)
        self.i = start

    def remaining(self) -> int:
        return len(self.fields) - self.i

    def pop(self, name: str) -> str:
        if self.i >= len(self.fields):
            raise PrincipiaServerError(f"missing field {name!r}; consumed {self.i}/{len(self.fields)} fields")
        v = self.fields[self.i]
        self.i += 1
        return v

    def pop_float(self, name: str) -> float:
        s = self.pop(name)
        try:
            return float(s)
        except Exception as exc:
            raise PrincipiaServerError(f"field {name!r} is not float: {s!r}") from exc

    def pop_int(self, name: str) -> int:
        s = self.pop(name)
        try:
            return int(float(s))
        except Exception as exc:
            raise PrincipiaServerError(f"field {name!r} is not int: {s!r}") from exc

    def pop_vec3(self, name: str) -> list[float]:
        return [
            self.pop_float(f"{name}_x"),
            self.pop_float(f"{name}_y"),
            self.pop_float(f"{name}_z"),
        ]


def _parse_burn_diag(t: _Tok, i: int) -> dict[str, Any]:
    return {
        "burn_dt_s": t.pop_float(f"burn[{i}].dt_s"),
        "burn_r_raw_m": t.pop_vec3(f"burn[{i}].r_raw_m"),
        "burn_v_before_raw_m_s": t.pop_vec3(f"burn[{i}].v_before_raw_m_s"),
        "burn_v_after_raw_m_s": t.pop_vec3(f"burn[{i}].v_after_raw_m_s"),
    }

def parse_okcarel_nav(line: str) -> dict[str, Any]:
    fields = _split_line(line)

    if not fields or fields[0] != "OKCARELNAV":
        raise PrincipiaServerError(f"expected OKCARELNAV, got: {line[:500]!r}")

    if len(fields) < 20:
        raise PrincipiaServerError(f"OKCARELNAV too short: {len(fields)} fields: {line[:500]!r}")

    t = _Tok(fields, 1)
    out: dict[str, Any] = {
        "tag": "OKCARELNAV",
        "id": t.pop("id"),
        "dep_body": t.pop("dep_body"),
        "arr_body": t.pop("arr_body"),
        "nav_body": t.pop("nav_body"),
        "state_dt_s": t.pop_float("state_dt_s"),
        "state_t_game_s": t.pop_float("state_t_game_s"),
        "ca_dt_s": t.pop_float("ca_dt_s"),
        "ca_t_game_s": t.pop_float("ca_t_game_s"),
        "ca_rel_r_raw_m": t.pop_vec3("ca_rel_r_raw_m"),
        "ca_rel_v_raw_m_s": t.pop_vec3("ca_rel_v_raw_m_s"),
        "ca_distance_m": t.pop_float("ca_distance_m"),
        "ca_speed_m_s": t.pop_float("ca_speed_m_s"),
        "ca_radial_velocity_m_s": t.pop_float("ca_radial_velocity_m_s"),
        "samples": t.pop_int("samples"),
        "status": t.pop("status"),
        "raw_line": line,
        "raw_field_count": len(fields),
    }

    if t.remaining() >= 13:
        save_i = t.i
        try:
            out["ca_abs_debug_r_raw_m"] = t.pop_vec3("ca_abs_debug_r_raw_m")
            out["ca_abs_debug_v_raw_m_s"] = t.pop_vec3("ca_abs_debug_v_raw_m_s")
            out["arr_abs_debug_r_raw_m"] = t.pop_vec3("arr_abs_debug_r_raw_m")
            out["arr_abs_debug_v_raw_m_s"] = t.pop_vec3("arr_abs_debug_v_raw_m_s")
            n_burns = t.pop_int("n_burns")
        except Exception:
            t.i = save_i
            n_burns = 0
    else:
        n_burns = 0

    out["n_burns"] = n_burns
    burns: list[dict[str, Any]] = []
    for i in range(n_burns):
        if t.remaining() < 1:
            break
        burn: dict[str, Any] = {
            "burn_dt_s": t.pop_float(f"burn[{i}].dt_s"),
        }

        # Current VCAREL_NAV burn debug layout:
        # burn_dt_s
        # burn_r_raw_m[3]
        # burn_v_before_raw_m_s[3]
        # dv_navigation_m_s[3]
        # tangent_unit_raw[3]
        # normal_unit_raw[3]
        # binormal_unit_raw[3]
        # dv_raw_m_s[3]
        # burn_v_after_raw_m_s[3]
        #
        # Older drafts only returned tangent/normal/binormal/dv_raw. Parse the
        # rich layout first, then fall back to the short layout.
        if t.remaining() >= 24:
            burn["burn_r_raw_m"] = t.pop_vec3(f"burn[{i}].r_raw_m")
            burn["burn_v_before_raw_m_s"] = t.pop_vec3(f"burn[{i}].v_before_raw_m_s")
            burn["dv_navigation_m_s"] = t.pop_vec3(f"burn[{i}].dv_navigation_m_s")
            burn["tangent_raw"] = t.pop_vec3(f"burn[{i}].tangent_unit_raw")
            burn["normal_raw"] = t.pop_vec3(f"burn[{i}].normal_unit_raw")
            burn["binormal_raw"] = t.pop_vec3(f"burn[{i}].binormal_unit_raw")
            burn["dv_raw"] = t.pop_vec3(f"burn[{i}].dv_raw_m_s")
            burn["burn_v_after_raw_m_s"] = t.pop_vec3(f"burn[{i}].v_after_raw_m_s")
        elif t.remaining() >= 12:
            burn["tangent_raw"] = t.pop_vec3(f"burn[{i}].tangent_unit_raw")
            burn["normal_raw"] = t.pop_vec3(f"burn[{i}].normal_unit_raw")
            burn["binormal_raw"] = t.pop_vec3(f"burn[{i}].binormal_unit_raw")
            burn["dv_raw"] = t.pop_vec3(f"burn[{i}].dv_raw_m_s")
        burns.append(burn)

    out["burns"] = burns
    if t.remaining():
        out["extra_fields"] = t.fields[t.i:]
    return out


def vcarel_nav(
    client,
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
    impulses_nav: Iterable[NavBurn | Sequence[float]],
    timeout_s: float = 900.0,
) -> dict[str, Any]:
    """Module-level helper for VCAREL_NAV.

    impulses_nav:
        [(impulse_dt_s, dvt_m_s, dvn_m_s, dvb_m_s), ...]
    """

    burns_list = list(impulses_nav)
    fields: list[Any] = [
        "VCAREL_NAV",
        rid,
        dep_body.upper(),
        arr_body.upper(),
        nav_body.upper(),
        _fmt(state_dt_s),
        _fmt(scan_start_dt_s),
        _fmt(scan_end_dt_s),
        str(int(samples)),
        _fmt(rel_r_raw_m[0]),
        _fmt(rel_r_raw_m[1]),
        _fmt(rel_r_raw_m[2]),
        _fmt(rel_v_raw_m_s[0]),
        _fmt(rel_v_raw_m_s[1]),
        _fmt(rel_v_raw_m_s[2]),
        str(len(burns_list)),
    ]

    for b in burns_list:
        if isinstance(b, NavBurn):
            burn = b
        else:
            if len(b) != 4:
                raise ValueError(f"nav burn must have 4 values (dt,dvt,dvn,dvb), got {b!r}")
            burn = NavBurn(float(b[0]), float(b[1]), float(b[2]), float(b[3]))
        fields += burn.fields()

    line = client.command_fields(fields, timeout_s=timeout_s)
    return parse_okcarel_nav(line)


def parse_okvn(fields: Sequence[str], raw_line: str = "") -> dict[str, Any]:
    if not fields or fields[0] != "OKVN":
        raise PrincipiaServerError(f"expected OKVN, got: {raw_line or fields!r}")
    t = _Tok(fields, 1)
    rid = t.pop("id")
    vessel_guid = t.pop("vessel_guid")
    t0 = t.pop_float("t0_game_s")
    t1 = t.pop_float("t1_game_s")
    n_burns = t.pop_int("n_burns")
    burns = [_parse_burn_diag(t, i) for i in range(n_burns)]

    out = {
        "tag": "OKVN",
        "id": rid,
        "vessel_guid": vessel_guid,
        "t0_game_s": t0,
        "t1_game_s": t1,
        "final_dt_s": t1 - t0,
        "n_burns": n_burns,
        "burns": burns,
        "initial_r_raw_m": t.pop_vec3("initial_r_raw_m"),
        "initial_v_raw_m_s": t.pop_vec3("initial_v_raw_m_s"),
        "initial_parent_r_raw_m": t.pop_vec3("initial_parent_r_raw_m"),
        "initial_parent_v_raw_m_s": t.pop_vec3("initial_parent_v_raw_m_s"),
        "initial_parent_distance_m": t.pop_float("initial_parent_distance_m"),
        "initial_parent_speed_m_s": t.pop_float("initial_parent_speed_m_s"),
        "initial_parent_radial_velocity_m_s": t.pop_float("initial_parent_radial_velocity_m_s"),
        "final_r_raw_m": t.pop_vec3("final_r_raw_m"),
        "final_v_raw_m_s": t.pop_vec3("final_v_raw_m_s"),
        "final_parent_r_raw_m": t.pop_vec3("final_parent_r_raw_m"),
        "final_parent_v_raw_m_s": t.pop_vec3("final_parent_v_raw_m_s"),
        "final_parent_distance_m": t.pop_float("final_parent_distance_m"),
        "final_parent_speed_m_s": t.pop_float("final_parent_speed_m_s"),
        "final_parent_radial_velocity_m_s": t.pop_float("final_parent_radial_velocity_m_s"),
        "raw_line": raw_line,
    }
    if t.remaining():
        out["extra_fields"] = t.fields[t.i:]
    return out


def parse_okvr(fields: Sequence[str], raw_line: str = "") -> dict[str, Any]:
    if not fields or fields[0] != "OKVR":
        raise PrincipiaServerError(f"expected OKVR, got: {raw_line or fields!r}")
    t = _Tok(fields, 1)
    rid = t.pop("id")
    vessel_guid = t.pop("vessel_guid")
    reference_body = t.pop("reference_body")
    t0 = t.pop_float("t0_game_s")
    t1 = t.pop_float("t1_game_s")
    n_burns = t.pop_int("n_burns")

    out = {
        "tag": "OKVR",
        "id": rid,
        "vessel_guid": vessel_guid,
        "reference_body": reference_body,
        "t0_game_s": t0,
        "t1_game_s": t1,
        "final_dt_s": t1 - t0,
        "n_burns": n_burns,
        "final_rel_r_raw_m": t.pop_vec3("final_rel_r_raw_m"),
        "final_rel_v_raw_m_s": t.pop_vec3("final_rel_v_raw_m_s"),
        "distance_m": t.pop_float("distance_m"),
        "speed_m_s": t.pop_float("speed_m_s"),
        "radial_v_m_s": t.pop_float("radial_v_m_s"),
        "final_abs_debug_r_raw_m": t.pop_vec3("final_abs_debug_r_raw_m"),
        "final_abs_debug_v_raw_m_s": t.pop_vec3("final_abs_debug_v_raw_m_s"),
        "reference_abs_debug_r_raw_m": t.pop_vec3("reference_abs_debug_r_raw_m"),
        "reference_abs_debug_v_raw_m_s": t.pop_vec3("reference_abs_debug_v_raw_m_s"),
        "raw_line": raw_line,
    }

    burns: list[dict[str, Any]] = []
    for i in range(n_burns):
        if t.remaining() >= 10:
            burns.append(_parse_burn_diag(t, i))
    out["burns"] = burns
    if t.remaining():
        out["extra_fields"] = t.fields[t.i:]
    return out


def parse_okca(fields: Sequence[str], raw_line: str = "") -> dict[str, Any]:
    if not fields or fields[0] != "OKCA":
        raise PrincipiaServerError(f"expected OKCA, got: {raw_line or fields!r}")

    # Two supported layouts:
    # without t0: tag id guid body ca_dt ca_t rel_r3 rel_v3 dist speed radial samples status  -> len 17
    # with t0:    tag id guid body t0 ca_dt ca_t rel_r3 rel_v3 dist speed radial samples status -> len 18
    if len(fields) not in (17, 18):
        raise PrincipiaServerError(f"unexpected OKCA field count {len(fields)}: {raw_line!r}")

    t = _Tok(fields, 1)
    rid = t.pop("id")
    vessel_guid = t.pop("vessel_guid")
    target_body = t.pop("target_body")
    out: dict[str, Any] = {
        "tag": "OKCA",
        "id": rid,
        "vessel_guid": vessel_guid,
        "target_body": target_body,
        "raw_line": raw_line,
    }

    if len(fields) == 18:
        out["t0_game_s"] = t.pop_float("t0_game_s")
    else:
        out["t0_game_s"] = None

    out.update({
        "ca_dt_s": t.pop_float("ca_dt_s"),
        "ca_t_game_s": t.pop_float("ca_t_game_s"),
        "ca_rel_r_raw_m": t.pop_vec3("ca_rel_r_raw_m"),
        "ca_rel_v_raw_m_s": t.pop_vec3("ca_rel_v_raw_m_s"),
        "ca_distance_m": t.pop_float("ca_distance_m"),
        "ca_speed_m_s": t.pop_float("ca_speed_m_s"),
        "ca_radial_v_m_s": t.pop_float("ca_radial_v_m_s"),
        "samples": t.pop_int("samples"),
        "status": t.pop("status"),
    })

    if t.remaining():
        out["extra_fields"] = t.fields[t.i:]
    return out


class PrincipiaTargeterClient:
    def __init__(
        self,
        server: str | Sequence[str],
        plugin_b64: str | Path | None = None,
        *,
        plugin_arg_mode: str = "positional",
        startup_timeout_s: float = 300.0,
        response_timeout_s: float = 300.0,
        quiet_stderr: bool = False,
        cwd: str | Path | None = None,
        extra_args: Sequence[str] | None = None,
    ):
        self.server = server
        self.plugin_b64 = Path(plugin_b64) if plugin_b64 is not None else None
        self.plugin_arg_mode = plugin_arg_mode
        self.startup_timeout_s = float(startup_timeout_s)
        self.response_timeout_s = float(response_timeout_s)
        self.quiet_stderr = bool(quiet_stderr)
        self.cwd = Path(cwd) if cwd is not None else None
        self.extra_args = list(extra_args or [])
        self.proc: subprocess.Popen[str] | None = None
        self.ready_line: str | None = None

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
            else:
                raise ValueError(f"unknown plugin_arg_mode: {self.plugin_arg_mode!r}")
        cmd += self.extra_args
        return cmd

    def start(self) -> "PrincipiaTargeterClient":
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
        ready = self._readline(self.startup_timeout_s)
        self.ready_line = ready
        if not ready.startswith("READY"):
            self.close()
            raise PrincipiaServerError(f"expected READY, got: {ready!r}")

        pong = self.ping(timeout_s=self.startup_timeout_s)
        if pong != "PONG":
            self.close()
            raise PrincipiaServerError(f"expected PONG, got: {pong!r}")
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

    def __enter__(self) -> "PrincipiaTargeterClient":
        return self.start()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _readline(self, timeout_s: float | None = None) -> str:
        if self.proc is None or self.proc.stdout is None:
            raise PrincipiaServerError("server is not running")
        timeout_s = self.response_timeout_s if timeout_s is None else float(timeout_s)
        ready, _, _ = select.select([self.proc.stdout.fileno()], [], [], timeout_s)
        if not ready:
            raise PrincipiaServerError(f"timeout waiting for server response after {timeout_s:.1f}s")
        line = self.proc.stdout.readline()
        if line == "":
            rc = self.proc.poll()
            raise PrincipiaServerError(f"server stdout closed; returncode={rc}")
        return line.rstrip("\n")

    def command_fields(self, fields: Sequence[Any], *, timeout_s: float | None = None) -> str:
        if self.proc is None:
            self.start()
        assert self.proc is not None and self.proc.stdin is not None
        line = "\t".join(str(x) for x in fields)
        self.proc.stdin.write(line + "\n")
        self.proc.stdin.flush()
        resp = self._readline(timeout_s)
        parts = _split_line(resp)
        if parts and parts[0] == "ERR":
            rid = parts[1] if len(parts) > 1 else "<unknown>"
            msg = " ".join(parts[2:]) if len(parts) > 2 else resp
            raise PrincipiaServerError(f"ERR {rid}: {msg}")
        return resp

    def ping(self, *, timeout_s: float | None = None) -> str:
        return self.command_fields(["PING"], timeout_s=timeout_s)

    @staticmethod
    def _burns_fields(burns: Iterable[Burn | Sequence[float]]) -> list[str]:
        out: list[str] = []
        for b in burns:
            if isinstance(b, Burn):
                burn = b
            else:
                if len(b) != 4:
                    raise ValueError(f"burn must have 4 values (dt,dvx,dvy,dvz), got {b!r}")
                burn = Burn(float(b[0]), float(b[1]), float(b[2]), float(b[3]))
            out.extend(burn.fields())
        return out

    @staticmethod
    def _nav_burns_fields(burns: Iterable[NavBurn | Sequence[float]]) -> list[str]:
        out: list[str] = []
        for b in burns:
            if isinstance(b, NavBurn):
                burn = b
            else:
                if len(b) != 4:
                    raise ValueError(f"nav burn must have 4 values (dt,dvt,dvn,dvb), got {b!r}")
                burn = NavBurn(float(b[0]), float(b[1]), float(b[2]), float(b[3]))
            out.extend(burn.fields())
        return out

    def vpropn(self, rid: str, vessel_guid: str, final_dt_s: float, burns: Iterable[Burn | Sequence[float]], *, timeout_s: float | None = None) -> dict[str, Any]:
        burns_list = list(burns)
        fields = ["VPROPN", rid, vessel_guid, _fmt(final_dt_s), str(len(burns_list))]
        fields += self._burns_fields(burns_list)
        line = self.command_fields(fields, timeout_s=timeout_s)
        return parse_okvn(_split_line(line), line)

    def vrel(self, rid: str, vessel_guid: str, reference_body: str, final_dt_s: float, burns: Iterable[Burn | Sequence[float]], *, timeout_s: float | None = None) -> dict[str, Any]:
        burns_list = list(burns)
        fields = ["VREL", rid, vessel_guid, reference_body.upper(), _fmt(final_dt_s), str(len(burns_list))]
        fields += self._burns_fields(burns_list)
        line = self.command_fields(fields, timeout_s=timeout_s)
        return parse_okvr(_split_line(line), line)

    def vca(
        self,
        rid: str,
        vessel_guid: str,
        target_body: str,
        scan_start_dt_s: float,
        scan_end_dt_s: float,
        samples: int,
        burns: Iterable[Burn | Sequence[float]],
        *,
        timeout_s: float | None = None,
    ) -> dict[str, Any]:
        burns_list = list(burns)
        fields = [
            "VCA", rid, vessel_guid, target_body.upper(),
            _fmt(scan_start_dt_s), _fmt(scan_end_dt_s), str(int(samples)),
            str(len(burns_list)),
        ]
        fields += self._burns_fields(burns_list)
        line = self.command_fields(fields, timeout_s=timeout_s)
        return parse_okca(_split_line(line), line)


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
        burns: Iterable[NavBurn | Sequence[float]],
        *,
        timeout_s: float | None = None,
    ) -> dict[str, Any]:
        """Closest-approach propagation from a synthetic relative state.

        Impulses are navigation/Frenet components (Tangent, Normal, Binormal),
        not raw XYZ. The binary converts them to raw using its own Principia
        basis and returns tangent_raw/normal_raw/binormal_raw/dv_raw debug data
        when available.
        """
        burns_list = list(burns)
        fields: list[Any] = [
            "VCAREL_NAV",
            rid,
            dep_body.upper(),
            arr_body.upper(),
            nav_body.upper(),
            _fmt(state_dt_s),
            _fmt(scan_start_dt_s),
            _fmt(scan_end_dt_s),
            str(int(samples)),
            _fmt(rel_r_raw_m[0]),
            _fmt(rel_r_raw_m[1]),
            _fmt(rel_r_raw_m[2]),
            _fmt(rel_v_raw_m_s[0]),
            _fmt(rel_v_raw_m_s[1]),
            _fmt(rel_v_raw_m_s[2]),
            str(len(burns_list)),
        ]
        fields += self._nav_burns_fields(burns_list)
        line = self.command_fields(fields, timeout_s=timeout_s)
        return parse_okcarel_nav(line)


def _main() -> int:
    ap = argparse.ArgumentParser(description="Smoke client for principia_impulsive_particle_server_v0_5_targeter.")
    ap.add_argument("--server", required=True)
    ap.add_argument("--plugin-b64", type=Path, required=True)
    ap.add_argument("--plugin-arg-mode", choices=["option", "positional"], default="positional")
    ap.add_argument("--vessel-guid", required=True)
    ap.add_argument("--quiet-stderr", action="store_true")
    ap.add_argument("--timeout-s", type=float, default=180.0)
    ap.add_argument("--target-body", default="EVE")
    ap.add_argument("--output-json", type=Path, required=True)
    args = ap.parse_args()

    with PrincipiaTargeterClient(
        args.server,
        args.plugin_b64,
        plugin_arg_mode=args.plugin_arg_mode,
        quiet_stderr=args.quiet_stderr,
        response_timeout_s=args.timeout_s,
    ) as client:
        out = {
            "ready_line": client.ready_line,
            "ping": client.ping(timeout_s=args.timeout_s),
            "vrel_kerbin_600": client.vrel("smoke_vrel_kerbin_600", args.vessel_guid, "KERBIN", 600.0, [], timeout_s=args.timeout_s),
            "vca_kerbin_1h": client.vca("smoke_vca_kerbin_1h", args.vessel_guid, "KERBIN", 0.0, 3600.0, 25, [], timeout_s=args.timeout_s),
            "vca_target_1d": client.vca("smoke_vca_target_1d", args.vessel_guid, args.target_body, 0.0, 86400.0, 25, [], timeout_s=args.timeout_s),
        }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(out, indent=2) + "\n")
    print("=== TARGETER CLIENT SMOKE ===")
    print(f"READY: {out['ready_line']}")
    print(f"VREL Kerbin 600 distance_m: {out['vrel_kerbin_600']['distance_m']}")
    print(f"VCA Kerbin 1h ca_distance_m: {out['vca_kerbin_1h']['ca_distance_m']} status={out['vca_kerbin_1h']['status']}")
    print(f"VCA {args.target_body} 1d ca_distance_m: {out['vca_target_1d']['ca_distance_m']} status={out['vca_target_1d']['status']}")
    print(f"[OK] wrote {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
