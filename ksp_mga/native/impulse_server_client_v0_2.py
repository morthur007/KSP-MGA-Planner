from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np


@dataclass
class BurnSnapshot:
    burn_t_s: float
    r_m: np.ndarray
    v_before_m_s: np.ndarray
    v_after_m_s: np.ndarray


@dataclass
class PropagationResult:
    status: str
    final_r_m: np.ndarray | None = None
    final_v_m_s: np.ndarray | None = None
    message: str = ""
    burns: list[BurnSnapshot] = field(default_factory=list)


class PrincipiaImpulseServerV2:
    """
    Client for principia_impulsive_particle_server v0.2.

    Supported protocol:
      legacy: PROP  id t0 burn_t t1 x y z vx vy vz dvx dvy dvz
      new:    PROP2 id t0 tb0 tb1 t1 x y z vx vy vz dv0x dv0y dv0z dv1x dv1y dv1z
      new:    PROPN id t0 t1 n x y z vx vy vz [burn_t dvx dvy dvz] * n

    Units:
      time: seconds from Principia/KSP epoch
      r:    m, Principia raw barycentric frame
      v:    m/s, Principia raw barycentric frame
      dv:   m/s, Principia raw barycentric frame
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
        ready = self.proc.stdout.readline().strip()
        if not ready.startswith("READY"):
            err = ""
            if self.proc.stderr is not None:
                try:
                    err = self.proc.stderr.read()
                except Exception:
                    pass
            raise RuntimeError(f"Failed to start Principia server: {ready}\n{err}")
        self.ready_line = ready

    @staticmethod
    def _arr3(parts: Sequence[str], offset: int) -> np.ndarray:
        return np.array(
            [float(parts[offset]), float(parts[offset + 1]), float(parts[offset + 2])],
            dtype=float,
        )

    def _request(self, line: str) -> str:
        if self.proc.stdin is None or self.proc.stdout is None:
            raise RuntimeError("server pipe unavailable")
        self.proc.stdin.write(line)
        self.proc.stdin.flush()
        resp = self.proc.stdout.readline().strip()
        if not resp:
            raise RuntimeError("empty response from server")
        return resp

    def ping(self) -> bool:
        return self._request("PING\n") == "PONG"

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
        # Backward-compatible wrapper using legacy PROP and parsing legacy OK.
        r0_m = np.asarray(r0_m, dtype=float)
        v0_m_s = np.asarray(v0_m_s, dtype=float)
        burn_dv_m_s = np.asarray(burn_dv_m_s, dtype=float)
        line = (
            f"PROP\t{req_id}\t{t0_s:.17g}\t{burn_t_s:.17g}\t{t1_s:.17g}\t"
            f"{r0_m[0]:.17g}\t{r0_m[1]:.17g}\t{r0_m[2]:.17g}\t"
            f"{v0_m_s[0]:.17g}\t{v0_m_s[1]:.17g}\t{v0_m_s[2]:.17g}\t"
            f"{burn_dv_m_s[0]:.17g}\t{burn_dv_m_s[1]:.17g}\t{burn_dv_m_s[2]:.17g}\n"
        )
        try:
            resp = self._request(line)
        except Exception as e:
            return PropagationResult(status="crash", message=repr(e))
        parts = resp.split("\t")
        if parts[0] != "OK":
            return PropagationResult(status="error", message=parts[2] if len(parts) > 2 else resp)
        try:
            burn = BurnSnapshot(
                burn_t_s=float(parts[3]),
                r_m=self._arr3(parts, 5),
                v_before_m_s=self._arr3(parts, 8),
                v_after_m_s=self._arr3(parts, 11),
            )
            return PropagationResult(
                status="ok",
                final_r_m=self._arr3(parts, 14),
                final_v_m_s=self._arr3(parts, 17),
                burns=[burn],
            )
        except Exception as e:
            return PropagationResult(status="parse_error", message=f"{repr(e)} resp={resp}")

    def propagate_n(
        self,
        req_id: str,
        t0_s: float,
        t1_s: float,
        r0_m: np.ndarray,
        v0_m_s: np.ndarray,
        impulses: Sequence[tuple[float, np.ndarray]],
    ) -> PropagationResult:
        r0_m = np.asarray(r0_m, dtype=float)
        v0_m_s = np.asarray(v0_m_s, dtype=float)
        fields = [
            "PROPN",
            req_id,
            f"{t0_s:.17g}",
            f"{t1_s:.17g}",
            str(len(impulses)),
            f"{r0_m[0]:.17g}", f"{r0_m[1]:.17g}", f"{r0_m[2]:.17g}",
            f"{v0_m_s[0]:.17g}", f"{v0_m_s[1]:.17g}", f"{v0_m_s[2]:.17g}",
        ]
        for burn_t_s, dv in impulses:
            dv = np.asarray(dv, dtype=float)
            fields += [
                f"{burn_t_s:.17g}",
                f"{dv[0]:.17g}", f"{dv[1]:.17g}", f"{dv[2]:.17g}",
            ]
        line = "\t".join(fields) + "\n"
        try:
            resp = self._request(line)
        except Exception as e:
            return PropagationResult(status="crash", message=repr(e))
        return self._parse_okn(resp)

    def propagate2(
        self,
        req_id: str,
        t0_s: float,
        tb0_s: float,
        tb1_s: float,
        t1_s: float,
        r0_m: np.ndarray,
        v0_m_s: np.ndarray,
        dv0_m_s: np.ndarray,
        dv1_m_s: np.ndarray,
    ) -> PropagationResult:
        return self.propagate_n(
            req_id=req_id,
            t0_s=t0_s,
            t1_s=t1_s,
            r0_m=r0_m,
            v0_m_s=v0_m_s,
            impulses=[(tb0_s, dv0_m_s), (tb1_s, dv1_m_s)],
        )

    def _parse_okn(self, resp: str) -> PropagationResult:
        parts = resp.split("\t")
        if parts[0] not in {"OKN", "OK2"}:
            return PropagationResult(status="error", message=parts[2] if len(parts) > 2 else resp)
        try:
            n = int(parts[4])
            expected = 5 + 10 * n + 6
            if len(parts) != expected:
                return PropagationResult(
                    status="parse_error",
                    message=f"expected {expected} fields for OKN n={n}, got {len(parts)} resp={resp}",
                )
            burns: list[BurnSnapshot] = []
            off = 5
            for _ in range(n):
                burns.append(
                    BurnSnapshot(
                        burn_t_s=float(parts[off + 0]),
                        r_m=self._arr3(parts, off + 1),
                        v_before_m_s=self._arr3(parts, off + 4),
                        v_after_m_s=self._arr3(parts, off + 7),
                    )
                )
                off += 10
            final_r = self._arr3(parts, off)
            final_v = self._arr3(parts, off + 3)
            return PropagationResult(status="ok", final_r_m=final_r, final_v_m_s=final_v, burns=burns)
        except Exception as e:
            return PropagationResult(status="parse_error", message=f"{repr(e)} resp={resp}")

    def close(self) -> None:
        if self.proc.poll() is None:
            try:
                if self.proc.stdin is not None:
                    self.proc.stdin.write("QUIT\n")
                    self.proc.stdin.flush()
                self.proc.wait(timeout=2)
            except Exception:
                self.proc.kill()

    def __enter__(self) -> "PrincipiaImpulseServerV2":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
