from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class PropagationResult:
    status: str
    final_r_m: np.ndarray | None = None
    final_v_m_s: np.ndarray | None = None
    message: str = ""


class PrincipiaImpulseServer:
    """
    Thin client for principia_impulsive_particle_server.

    Protocol:
      PROP id t0 burn_t t1 x y z vx vy vz dvx dvy dvz
      QUIT

    Units:
      times: seconds
      position: m, Principia raw frame
      velocity: m/s, Principia raw frame
      burn_dv: m/s, Principia raw frame
    """

    def __init__(self, executable: str, plugin_b64: str | Path):
        cmd = executable.split() + [str(plugin_b64)]
        self.proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

        if self.proc.stdout is None:
            raise RuntimeError("Principia server stdout is unavailable")

        ready_line = self.proc.stdout.readline().strip()
        if not ready_line.startswith("READY"):
            err = ""
            if self.proc.stderr is not None:
                try:
                    err = self.proc.stderr.read()
                except Exception:
                    pass
            raise RuntimeError(f"Failed to start Principia server: {ready_line}\n{err}")

    def propagate(
        self,
        req_id: str,
        t0_s: float,
        burn_t_s: float,
        t1_s: float,
        r0_m: np.ndarray,
        v0_m_s: np.ndarray,
        burn_dv_m_s: np.ndarray,
    ) -> PropagationResult:
        if self.proc.stdin is None or self.proc.stdout is None:
            return PropagationResult(status="crash", message="server pipe unavailable")

        r0_m = np.asarray(r0_m, dtype=float)
        v0_m_s = np.asarray(v0_m_s, dtype=float)
        burn_dv_m_s = np.asarray(burn_dv_m_s, dtype=float)

        req = (
            f"PROP\t{req_id}\t{t0_s:.17g}\t{burn_t_s:.17g}\t{t1_s:.17g}\t"
            f"{r0_m[0]:.17g}\t{r0_m[1]:.17g}\t{r0_m[2]:.17g}\t"
            f"{v0_m_s[0]:.17g}\t{v0_m_s[1]:.17g}\t{v0_m_s[2]:.17g}\t"
            f"{burn_dv_m_s[0]:.17g}\t{burn_dv_m_s[1]:.17g}\t{burn_dv_m_s[2]:.17g}\n"
        )

        try:
            self.proc.stdin.write(req)
            self.proc.stdin.flush()
            resp = self.proc.stdout.readline().strip()
        except Exception as e:
            return PropagationResult(status="crash", message=repr(e))

        if not resp:
            return PropagationResult(status="crash", message="empty response")

        parts = resp.split("\t")

        if parts[0] != "OK":
            return PropagationResult(
                status="error",
                message=parts[2] if len(parts) > 2 else resp,
            )

        try:
            # Server output format from the existing C++ tool:
            # OK id t0 burn_t t1 burn_x burn_y burn_z burn_v_before...
            # final position/velocity are parts[14:20].
            final_r = np.array(
                [float(parts[14]), float(parts[15]), float(parts[16])],
                dtype=float,
            )
            final_v = np.array(
                [float(parts[17]), float(parts[18]), float(parts[19])],
                dtype=float,
            )
        except Exception as e:
            return PropagationResult(status="parse_error", message=f"{repr(e)} resp={resp}")

        return PropagationResult(status="ok", final_r_m=final_r, final_v_m_s=final_v)

    def close(self) -> None:
        if self.proc.poll() is None:
            try:
                if self.proc.stdin is not None:
                    self.proc.stdin.write("QUIT\n")
                    self.proc.stdin.flush()
                self.proc.wait(timeout=2)
            except Exception:
                self.proc.kill()

    def __enter__(self) -> "PrincipiaImpulseServer":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
