"""Client for principia_impulsive_particle_server.

The C++ daemon is the only long-lived subprocess in the refactored pipeline.
All Python stages should share this client instead of spawning validators per
evaluation.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class ImpulseRequest:
    request_id: str
    t0_s: float
    burn_t_s: float
    t1_s: float
    r0_m: np.ndarray
    v0_m_s: np.ndarray
    burn_dv_m_s: np.ndarray


@dataclass(frozen=True)
class ImpulseResult:
    request_id: str
    t0_s: float
    burn_t_s: float
    t1_s: float
    burn_r_m: np.ndarray
    burn_v_before_m_s: np.ndarray
    burn_v_after_m_s: np.ndarray
    final_r_m: np.ndarray
    final_v_m_s: np.ndarray


class PrincipiaImpulseServer:
    def __init__(self, executable: str, plugin_b64: str | Path, *, stderr_log: Path | None = None) -> None:
        self.executable = executable
        self.plugin_b64 = Path(plugin_b64)
        self.stderr_log = stderr_log
        self._proc: subprocess.Popen[str] | None = None
        self._stderr_fh = None

    def start(self) -> None:
        if self._proc is not None:
            return
        stderr = subprocess.PIPE
        if self.stderr_log is not None:
            self.stderr_log.parent.mkdir(parents=True, exist_ok=True)
            self._stderr_fh = self.stderr_log.open("w", encoding="utf-8")
            stderr = self._stderr_fh
        self._proc = subprocess.Popen(
            [self.executable, str(self.plugin_b64)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=stderr,
            text=True,
            bufsize=1,
        )
        ready = self._readline()
        if not ready.startswith("READY"):
            raise RuntimeError(f"server did not become ready: {ready!r}")

    def close(self) -> None:
        if self._proc is None:
            return
        try:
            self._write("QUIT\n")
            _ = self._readline()
        except Exception:
            pass
        try:
            self._proc.terminate()
            self._proc.wait(timeout=2)
        except Exception:
            try:
                self._proc.kill()
            except Exception:
                pass
        self._proc = None
        if self._stderr_fh is not None:
            self._stderr_fh.close()
            self._stderr_fh = None

    def __enter__(self) -> "PrincipiaImpulseServer":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def ping(self) -> bool:
        self.start()
        self._write("PING\n")
        return self._readline().strip() == "PONG"

    def propagate(self, req: ImpulseRequest) -> ImpulseResult:
        self.start()
        r = np.asarray(req.r0_m, dtype=float)
        v = np.asarray(req.v0_m_s, dtype=float)
        dv = np.asarray(req.burn_dv_m_s, dtype=float)
        fields = [
            "PROP", req.request_id,
            f"{req.t0_s:.17g}", f"{req.burn_t_s:.17g}", f"{req.t1_s:.17g}",
            *(f"{x:.17g}" for x in r),
            *(f"{x:.17g}" for x in v),
            *(f"{x:.17g}" for x in dv),
        ]
        self._write("\t".join(fields) + "\n")
        line = self._readline()
        parts = line.rstrip("\n").split("\t")
        if not parts:
            raise RuntimeError("empty response from impulse server")
        if parts[0] == "ERR":
            msg = parts[2] if len(parts) > 2 else line
            raise RuntimeError(f"Principia impulse server error for {req.request_id}: {msg}")
        if parts[0] != "OK" or len(parts) != 20:
            raise RuntimeError(f"unexpected response from impulse server: {line!r}")
        vals = [float(x) for x in parts[2:]]
        return ImpulseResult(
            request_id=parts[1],
            t0_s=vals[0], burn_t_s=vals[1], t1_s=vals[2],
            burn_r_m=np.asarray(vals[3:6], dtype=float),
            burn_v_before_m_s=np.asarray(vals[6:9], dtype=float),
            burn_v_after_m_s=np.asarray(vals[9:12], dtype=float),
            final_r_m=np.asarray(vals[12:15], dtype=float),
            final_v_m_s=np.asarray(vals[15:18], dtype=float),
        )

    def _write(self, text: str) -> None:
        if self._proc is None or self._proc.stdin is None:
            raise RuntimeError("server is not started")
        self._proc.stdin.write(text)
        self._proc.stdin.flush()

    def _readline(self) -> str:
        if self._proc is None or self._proc.stdout is None:
            raise RuntimeError("server is not started")
        line = self._proc.stdout.readline()
        if line == "":
            raise RuntimeError("server terminated or closed stdout")
        return line
