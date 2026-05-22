#!/usr/bin/env python3
"""
target_snap_departure_vcarelnav_v0.py

Refina a primeira perna de uma rota PyKEP usando o servidor Principia com:
  LOADSNAP
  SNAPVCA_NAV

- Estado inicial da nave: snapshot vivo da DLL.
- Corpos/efeméride: .b64 carregado pelo servidor.
- Manobras: TNB/Frenet dvt,dvn,dvb.
- Saída: JSON/CSV + evento insert_navigation opcional.
"""
from __future__ import annotations

import argparse, csv, json, math, queue, subprocess, sys, threading, time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np

try:
    from scipy.optimize import minimize
except Exception:
    minimize = None
try:
    from scipy.stats import qmc
except Exception:
    qmc = None

DAY_S = 86400.0


def norm(v: Sequence[float]) -> float:
    return float(np.linalg.norm(np.asarray(v, dtype=float)))


def sanitize_body(s: str | None) -> str:
    return (s or "").strip().upper()


def vec3(fields: Sequence[str], i: int) -> list[float]:
    return [float(fields[i]), float(fields[i + 1]), float(fields[i + 2])]


def safe_float(x: Any, default: float) -> float:
    try:
        if x is None:
            return float(default)
        y = float(x)
        return y if math.isfinite(y) else float(default)
    except Exception:
        return float(default)


def flatten(row: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in row.items():
        if isinstance(v, np.ndarray):
            v = v.tolist()
        if isinstance(v, (list, tuple)):
            if all(not isinstance(x, (list, tuple, dict)) for x in v):
                for i, x in enumerate(v):
                    out[f"{k}_{i}"] = x
            else:
                out[k] = json.dumps(v, ensure_ascii=False)
        elif isinstance(v, dict):
            out[k] = json.dumps(v, ensure_ascii=False)
        else:
            out[k] = v
    return out


def load_anchor_leg(path: Path, leg_index: int) -> dict[str, Any]:
    d = json.loads(path.read_text())
    if isinstance(d.get("legs"), list):
        return d["legs"][leg_index - 1]
    key = f"leg{leg_index}"
    if key in d:
        return d[key]
    if leg_index == 1:
        return d
    raise ValueError(f"cannot find leg{leg_index} in {path}")


def leg_get(leg: dict[str, Any] | None, *keys: str, default=None):
    if not leg:
        return default
    for k in keys:
        if k in leg:
            return leg[k]
    return default


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
    ca_abs_debug_r_raw_m: list[float]
    ca_abs_debug_v_raw_m_s: list[float]
    arr_abs_debug_r_raw_m: list[float]
    arr_abs_debug_v_raw_m_s: list[float]
    n_burns: int
    burns: list[BurnDiag]


class ProtocolError(RuntimeError):
    pass


def parse_okcarelnav(parts: Sequence[str]) -> OkCarelNav:
    if not parts or parts[0] != "OKCARELNAV":
        raise ProtocolError(f"expected OKCARELNAV, got {parts[:4]}")
    if len(parts) < 33:
        raise ProtocolError(f"OKCARELNAV too short: {len(parts)} fields")
    n_burns = int(parts[32])
    burns: list[BurnDiag] = []
    i = 33
    for _ in range(n_burns):
        if i + 25 <= len(parts):
            burns.append(BurnDiag(
                burn_dt_s=float(parts[i]),
                burn_r_raw_m=vec3(parts, i + 1),
                burn_v_before_raw_m_s=vec3(parts, i + 4),
                dv_tnb_cmd_m_s=vec3(parts, i + 7),
                tangent_raw=vec3(parts, i + 10),
                normal_raw=vec3(parts, i + 13),
                binormal_raw=vec3(parts, i + 16),
                dv_raw_m_s=vec3(parts, i + 19),
                burn_v_after_raw_m_s=vec3(parts, i + 22),
            ))
            i += 25
        else:
            break
    return OkCarelNav(
        id=parts[1], dep_body=sanitize_body(parts[2]), arr_body=sanitize_body(parts[3]), nav_body=sanitize_body(parts[4]),
        state_dt_s=float(parts[5]), state_t_game_s=float(parts[6]), ca_dt_s=float(parts[7]), ca_t_game_s=float(parts[8]),
        ca_rel_r_raw_m=vec3(parts, 9), ca_rel_v_raw_m_s=vec3(parts, 12), ca_distance_m=float(parts[15]),
        ca_speed_m_s=float(parts[16]), ca_radial_velocity_m_s=float(parts[17]), samples=int(parts[18]), status=parts[19],
        ca_abs_debug_r_raw_m=vec3(parts, 20), ca_abs_debug_v_raw_m_s=vec3(parts, 23),
        arr_abs_debug_r_raw_m=vec3(parts, 26), arr_abs_debug_v_raw_m_s=vec3(parts, 29),
        n_burns=n_burns, burns=burns,
    )


class SnapshotNavClient:
    def __init__(self, server: Path, plugin_b64: Path, plugin_arg_mode: str, timeout_s: float, quiet_stderr: bool, stderr_log: Path):
        self.server = Path(server)
        self.plugin_b64 = Path(plugin_b64)
        self.plugin_arg_mode = plugin_arg_mode
        self.timeout_s = timeout_s
        self.quiet_stderr = quiet_stderr
        self.stderr_log = stderr_log
        self.proc: subprocess.Popen[str] | None = None
        self.out_q: queue.Queue[str] = queue.Queue()
        self.ready_line: str | None = None
        self._stderr_f = None

    def argv(self) -> list[str]:
        if self.plugin_arg_mode == "positional":
            return [str(self.server), str(self.plugin_b64)]
        if self.plugin_arg_mode == "flag":
            return [str(self.server), "--plugin-b64", str(self.plugin_b64)]
        return [str(self.server)]

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.close()

    def start(self):
        self.stderr_log.parent.mkdir(parents=True, exist_ok=True)
        self._stderr_f = self.stderr_log.open("w", encoding="utf-8")
        self.proc = subprocess.Popen(
            self.argv(), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, bufsize=1, universal_newlines=True,
        )
        assert self.proc.stdout is not None and self.proc.stderr is not None
        def pump_out():
            try:
                for line in self.proc.stdout:
                    self.out_q.put(line.rstrip("\n"))
            except Exception as e:
                self.out_q.put(f"__STDOUT_ERROR__ {e!r}")
        def pump_err():
            try:
                for line in self.proc.stderr:
                    s = line.rstrip("\n")
                    self._stderr_f.write(s + "\n"); self._stderr_f.flush()
                    if not self.quiet_stderr:
                        print(f"[server] {s}", file=sys.stderr)
            except Exception:
                pass
        threading.Thread(target=pump_out, daemon=True).start()
        threading.Thread(target=pump_err, daemon=True).start()
        self.ready_line = self.read_line(timeout_s=self.timeout_s)
        if not self.ready_line.startswith("READY"):
            raise ProtocolError(f"expected READY, got {self.ready_line!r}")

        # Some server builds print the startup banner in two stdout lines:
        #   READY
        #   principia_impulsive_particle_server_v...
        # The protocol client must consume that banner line before sending PING,
        # otherwise PING reads the banner and LOADSNAP reads the delayed PONG.
        try:
            extra = self.out_q.get(timeout=0.10)
            if extra.startswith(("OK", "ERR", "PONG")):
                # This should not happen before any command, but keep it visible
                # instead of silently corrupting the stream.
                raise ProtocolError(
                    f"unexpected protocol response during startup banner drain: {extra!r}"
                )
            if extra:
                self.ready_line = self.ready_line + "\\t" + extra
        except queue.Empty:
            pass

    def close(self):
        if self.proc:
            try:
                if self.proc.stdin:
                    self.proc.stdin.close()
            except Exception:
                pass
            try:
                self.proc.terminate(); self.proc.wait(timeout=3)
            except Exception:
                try: self.proc.kill()
                except Exception: pass
            self.proc = None
        if self._stderr_f:
            self._stderr_f.close(); self._stderr_f = None

    def read_line(self, timeout_s: float) -> str:
        try:
            line = self.out_q.get(timeout=timeout_s)
        except queue.Empty:
            rc = None if self.proc is None else self.proc.poll()
            raise TimeoutError(f"timeout waiting for server stdout; returncode={rc}")
        if line.startswith("__STDOUT_ERROR__"):
            raise ProtocolError(line)
        return line

    def command(self, fields: Sequence[Any], timeout_s: float | None = None) -> list[str]:
        if self.proc is None or self.proc.stdin is None:
            raise RuntimeError("server not started")
        self.proc.stdin.write("\t".join(map(str, fields)) + "\n")
        self.proc.stdin.flush()
        line = self.read_line(timeout_s or self.timeout_s)
        parts = line.split("\t")
        if parts[0].startswith("ERR"):
            raise ProtocolError(line)
        return parts

    def ping(self) -> str:
        return "\t".join(self.command(["PING", "ping"]))

    def loadsnap(self, snapshot_json: Path) -> OkSnap:
        p = self.command(["LOADSNAP", "snap0", str(snapshot_json)])
        if p[0] != "OKSNAP":
            raise ProtocolError(f"expected OKSNAP, got {p[:4]}")
        return OkSnap(
            id=p[1], schema=p[2], t_game_s=float(p[3]), vessel_guid=p[4], dep_body=sanitize_body(p[5]), nav_body=sanitize_body(p[6]),
            rel_r_raw_m=vec3(p, 7), rel_v_raw_m_s=vec3(p, 10), mass_tonnes=float(p[13]),
            available_thrust_kN=float(p[14]), specific_impulse_s_g0=float(p[15]),
        )

    def snapvca_nav(self, rid: str, arr_body: str, nav_body: str, scan_start: float, scan_end: float, samples: int, impulses: list[tuple[float, float, float, float]]) -> OkCarelNav:
        fields: list[Any] = ["SNAPVCA_NAV", rid, sanitize_body(arr_body), nav_body or "AUTO", float(scan_start), float(scan_end), int(samples), len(impulses)]
        for dt, t, n, b in impulses:
            fields.extend([float(dt), float(t), float(n), float(b)])
        return parse_okcarelnav(self.command(fields))


@dataclass
class Config:
    dep_body: str
    arr_body: str
    nav_body: str
    state_t_game_s: float
    nominal_arrival_t_game_s: float
    scan_half_width_days: float
    samples: int
    burn_dt_min_s: float
    burn_dt_max_s: float
    dvt_min_m_s: float
    dvt_max_m_s: float
    dvn_max_abs_m_s: float
    dvb_max_abs_m_s: float
    arrival_offset_min_days: float
    arrival_offset_max_days: float
    enable_dsm: bool
    dsm_frac_min: float
    dsm_frac_max: float
    dsm_max_abs_m_s: float
    ca_scale_km: float
    dv_scale_m_s: float
    dv_weight: float
    dsm_weight: float
    out_of_plane_weight: float
    binormal_weight: float
    normal_weight: float
    burn_time_weight: float
    require_positive_tangent: bool
    max_total_dv_m_s: float | None
    max_out_of_plane_fraction: float | None


class Targeter:
    def __init__(self, client: SnapshotNavClient, cfg: Config):
        self.client = client; self.cfg = cfg
        self.rows: list[dict[str, Any]] = []
        self.best: dict[str, Any] | None = None
        self.count = 0
        self.cache: dict[tuple[float, ...], dict[str, Any]] = {}

    def bounds(self) -> list[tuple[float, float]]:
        c = self.cfg
        b = [(c.burn_dt_min_s, c.burn_dt_max_s), (c.dvt_min_m_s, c.dvt_max_m_s), (-c.dvn_max_abs_m_s, c.dvn_max_abs_m_s), (-c.dvb_max_abs_m_s, c.dvb_max_abs_m_s)]
        if c.enable_dsm:
            b += [(c.dsm_frac_min, c.dsm_frac_max), (-c.dsm_max_abs_m_s, c.dsm_max_abs_m_s), (-c.dsm_max_abs_m_s, c.dsm_max_abs_m_s), (-c.dsm_max_abs_m_s, c.dsm_max_abs_m_s)]
        b.append((c.arrival_offset_min_days, c.arrival_offset_max_days))
        return b

    def impulses_from_x(self, x: Sequence[float]) -> tuple[list[tuple[float, float, float, float]], float, float, float, float, float, float]:
        c = self.cfg; x = list(map(float, x))
        burn_dt, t, n, b = x[0], x[1], x[2], x[3]
        arr_off = x[-1]
        impulses = [(burn_dt, t, n, b)]
        dsm_norm = 0.0
        if c.enable_dsm:
            frac, dt, dn, db = x[4], x[5], x[6], x[7]
            center = (c.nominal_arrival_t_game_s - c.state_t_game_s) + arr_off * DAY_S
            dsm_dt = burn_dt + frac * max(1.0, center - burn_dt)
            impulses.append((dsm_dt, dt, dn, db))
            dsm_norm = norm([dt, dn, db])
        return impulses, burn_dt, t, n, b, arr_off, dsm_norm

    def eval(self, x0: Sequence[float], kind: str = "eval") -> dict[str, Any]:
        bounds = self.bounds()
        x = np.asarray(x0, dtype=float)
        for i, (lo, hi) in enumerate(bounds):
            x[i] = min(max(float(x[i]), lo), hi)
        key = tuple(round(float(v), 5) for v in x)
        if key in self.cache:
            row = dict(self.cache[key]); row["kind"] = kind + "_cached"; self.rows.append(row); return row

        c = self.cfg
        impulses, burn_dt, dvt, dvn, dvb, arr_off, dsm_norm = self.impulses_from_x(x)
        dv0 = norm([dvt, dvn, dvb]); total_dv = dv0 + dsm_norm
        oop = norm([dvn, dvb]); oop_frac = oop / max(dv0, 1.0)
        center = (c.nominal_arrival_t_game_s - c.state_t_game_s) + arr_off * DAY_S
        scan_start = center - c.scan_half_width_days * DAY_S
        scan_end = center + c.scan_half_width_days * DAY_S
        penalty = 0.0
        if c.require_positive_tangent and dvt <= 0: penalty += 1e6 + abs(dvt)
        if c.max_total_dv_m_s and total_dv > c.max_total_dv_m_s: penalty += 1000 * ((total_dv - c.max_total_dv_m_s) / c.dv_scale_m_s) ** 2
        if c.max_out_of_plane_fraction and oop_frac > c.max_out_of_plane_fraction: penalty += 1000 * ((oop_frac - c.max_out_of_plane_fraction) / max(c.max_out_of_plane_fraction, 1e-3)) ** 2
        if scan_end <= scan_start or scan_end <= 0: penalty += 1e9

        base = dict(kind=kind, ok=False, error="", eval_index=self.count, x=[float(v) for v in x], burn_dt_s=burn_dt, burn_abs_s=c.state_t_game_s + burn_dt,
                    dvt_m_s=dvt, dvn_m_s=dvn, dvb_m_s=dvb, dv0_norm_m_s=dv0, dsm_norm_m_s=dsm_norm, total_dv_m_s=total_dv,
                    out_of_plane_abs_m_s=oop, out_of_plane_fraction=oop_frac, arrival_offset_days=arr_off,
                    scan_start_dt_s=scan_start, scan_end_dt_s=scan_end, scan_center_dt_s=center)
        try:
            rid = f"snapopt_{self.count:06d}"; self.count += 1
            res = self.client.snapvca_nav(rid, c.arr_body, c.nav_body, scan_start, scan_end, c.samples, impulses)
            ca_km = res.ca_distance_m / 1000.0
            score = (ca_km / c.ca_scale_km + c.dv_weight * total_dv / c.dv_scale_m_s + c.dsm_weight * dsm_norm / c.dv_scale_m_s +
                     c.out_of_plane_weight * oop_frac + c.normal_weight * abs(dvn) / c.dv_scale_m_s + c.binormal_weight * abs(dvb) / c.dv_scale_m_s +
                     c.burn_time_weight * burn_dt / max(c.burn_dt_max_s, 1.0) + penalty)
            row = dict(base, ok=True, score=score, ca_distance_km=ca_km, ca_distance_m=res.ca_distance_m, ca_dt_s=res.ca_dt_s, ca_t_game_s=res.ca_t_game_s,
                       ca_speed_m_s=res.ca_speed_m_s, ca_radial_velocity_m_s=res.ca_radial_velocity_m_s, status=res.status,
                       dep_body=res.dep_body, arr_body=res.arr_body, nav_body=res.nav_body, ca_rel_r_raw_m=res.ca_rel_r_raw_m,
                       ca_rel_v_raw_m_s=res.ca_rel_v_raw_m_s, ca_abs_debug_r_raw_m=res.ca_abs_debug_r_raw_m,
                       ca_abs_debug_v_raw_m_s=res.ca_abs_debug_v_raw_m_s, arr_abs_debug_r_raw_m=res.arr_abs_debug_r_raw_m,
                       arr_abs_debug_v_raw_m_s=res.arr_abs_debug_v_raw_m_s, n_burns=res.n_burns, burns=[asdict(b) for b in res.burns])
            if res.burns:
                b0 = res.burns[0]
                row.update(burn0_dv_raw_m_s=b0.dv_raw_m_s, burn0_tangent_raw=b0.tangent_raw, burn0_normal_raw=b0.normal_raw, burn0_binormal_raw=b0.binormal_raw, burn0_v_after_raw_m_s=b0.burn_v_after_raw_m_s)
        except Exception as e:
            self.count += 1
            row = dict(base, ok=False, error=str(e), score=1e30 + penalty, ca_distance_km=math.inf)
        self.cache[key] = dict(row); self.rows.append(row)
        if row.get("ok") and (self.best is None or float(row["score"]) < float(self.best["score"])):
            self.best = row
        return row

    def objective(self, x: Sequence[float]) -> float:
        s = float(self.eval(x, kind="objective").get("score", 1e30))
        return s if math.isfinite(s) else 1e30


def sample(bounds: list[tuple[float, float]], n: int, seed: int) -> np.ndarray:
    dim = len(bounds)
    lo = np.array([b[0] for b in bounds]); hi = np.array([b[1] for b in bounds])
    if qmc is not None:
        sampler = qmc.Sobol(d=dim, scramble=True, seed=seed)
        u = sampler.random_base2(m=int(math.ceil(math.log2(max(1, n)))))[:n]
    else:
        u = np.random.default_rng(seed).random((n, dim))
    return lo + u * (hi - lo)


def parse_manual_seeds(s: str, dim: int) -> list[list[float]]:
    out = []
    for part in (s or "").split(";"):
        part = part.strip()
        if not part: continue
        vals = [float(x.strip()) for x in part.split(",") if x.strip()]
        if len(vals) != dim: raise ValueError(f"manual seed {part!r} has {len(vals)} values, expected {dim}")
        out.append(vals)
    return out


def write_event(path: Path, snap_data: dict[str, Any], snap: OkSnap, best: dict[str, Any], cfg: Config, suffix: str = "burn0"):
    v = snap_data.get("vessel", {}) if isinstance(snap_data.get("vessel"), dict) else {}
    event = {
        "enabled": True, "vessel_guid": snap.vessel_guid or v.get("vessel_guid", ""), "ensure_flight_plan": True, "extend_existing_flight_plan": True,
        "mass_tonnes": safe_float(snap.mass_tonnes, safe_float(v.get("mass_tonnes"), 2.6)), "insert_index": 0, "burn_template": "json",
        "thrust_kN": safe_float(snap.available_thrust_kN, safe_float(v.get("available_thrust_kN"), 90.0)),
        "specific_impulse_s_g0": safe_float(snap.specific_impulse_s_g0, safe_float(v.get("specific_impulse_s_g0"), 345.0)),
        "is_inertially_fixed": False, "frame_extension": 6000, "frame_centre_from_active_body": True,
        "frame_centre_index": -1, "frame_primary_index": -1, "frame_secondary_index": -1, "placeholder_dv_m_s": 0.001,
        "require_status_ok": True, "cleanup_on_error": True, "tolerance_time_s": 0.01, "tolerance_dv_m_s": 1e-6,
        "one_shot": True, "disable_after_success": True, "request_id": f"snap_target_{suffix}_attempt0",
        "dedupe_tag": f"snap_target_{suffix}", "event_key": f"snap_target_{suffix}", "attempt": 0, "mode": "insert_navigation",
        "initial_time": float(best["burn_abs_s"]), "plan_final_time": float(best["burn_abs_s"]) + 600.0,
        "delta_v_navigation_m_s": [float(best["dvt_m_s"]), float(best["dvn_m_s"]), float(best["dvb_m_s"])],
        "planned_from_state": {"schema": "planned_from_snap_departure_vcarelnav_v0", "snapshot_schema": snap_data.get("schema"), "state_t_game_s": snap.t_game_s,
                               "vessel_guid": snap.vessel_guid, "dep_body": cfg.dep_body, "arr_body": cfg.arr_body, "nav_body": cfg.nav_body,
                               "rel_r_raw_m": snap.rel_r_raw_m, "rel_v_raw_m_s": snap.rel_v_raw_m_s,
                               "score": best.get("score"), "ca_distance_km": best.get("ca_distance_km"), "ca_t_game_s": best.get("ca_t_game_s"), "status": best.get("status")},
        "require_warp_close": True, "max_lead_before_insert_s": 600.0, "reject_long_plan_before_first_burn": True,
        "max_first_plan_duration_s": 900.0, "rollback_on_status_error": True, "auto_sort_insert_index": True,
    }
    path.write_text(json.dumps(event, indent=2, ensure_ascii=False) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--server", type=Path, required=True)
    ap.add_argument("--plugin-b64", type=Path, required=True)
    ap.add_argument("--plugin-arg-mode", choices=["positional", "flag", "none"], default="positional")
    ap.add_argument("--snapshot-json", type=Path, required=True)
    ap.add_argument("--anchor-json", type=Path)
    ap.add_argument("--leg", type=int, default=1)
    ap.add_argument("--dep-body")
    ap.add_argument("--arr-body")
    ap.add_argument("--nav-body", default="AUTO")
    ap.add_argument("--arrival-t-game-s", type=float)
    ap.add_argument("--burn-dt-min-s", type=float, default=0.0)
    ap.add_argument("--burn-dt-max-s", type=float, default=21600.0)
    ap.add_argument("--dvt-min-m-s", type=float, default=0.0)
    ap.add_argument("--dvt-max-m-s", type=float, default=4500.0)
    ap.add_argument("--dvn-max-abs-m-s", type=float, default=500.0)
    ap.add_argument("--dvb-max-abs-m-s", type=float, default=500.0)
    ap.add_argument("--arrival-offset-min-days", type=float, default=-30.0)
    ap.add_argument("--arrival-offset-max-days", type=float, default=30.0)
    ap.add_argument("--enable-dsm", action="store_true")
    ap.add_argument("--dsm-frac-min", type=float, default=0.08)
    ap.add_argument("--dsm-frac-max", type=float, default=0.85)
    ap.add_argument("--dsm-max-abs-m-s", type=float, default=250.0)
    ap.add_argument("--scan-half-width-days", type=float, default=45.0)
    ap.add_argument("--samples", type=int, default=41)
    ap.add_argument("--coarse-samples", type=int, default=192)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--manual-seeds", default="")
    ap.add_argument("--refine-top-n", type=int, default=6)
    ap.add_argument("--powell-maxiter", type=int, default=3)
    ap.add_argument("--powell-maxfev", type=int, default=80)
    ap.add_argument("--ca-scale-km", type=float, default=100000.0)
    ap.add_argument("--dv-scale-m-s", type=float, default=1000.0)
    ap.add_argument("--dv-weight", type=float, default=0.02)
    ap.add_argument("--dsm-weight", type=float, default=0.05)
    ap.add_argument("--out-of-plane-weight", type=float, default=3.0)
    ap.add_argument("--binormal-weight", type=float, default=0.2)
    ap.add_argument("--normal-weight", type=float, default=0.05)
    ap.add_argument("--burn-time-weight", type=float, default=0.0)
    ap.add_argument("--allow-negative-tangent", action="store_true")
    ap.add_argument("--max-total-dv-m-s", type=float, default=5200.0)
    ap.add_argument("--max-out-of-plane-fraction", type=float, default=0.45)
    ap.add_argument("--timeout-s", type=float, default=600.0)
    ap.add_argument("--quiet-stderr", action="store_true")
    ap.add_argument("--no-warmup", action="store_true")
    ap.add_argument("--write-events", action="store_true")
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    snap_data = json.loads(args.snapshot_json.read_text())
    leg = load_anchor_leg(args.anchor_json, args.leg) if args.anchor_json else None
    dep_body = sanitize_body(args.dep_body or leg_get(leg, "dep", "dep_body", default=None) or (snap_data.get("vessel", {}) or {}).get("nav_body") or "KERBIN")
    arr_body = sanitize_body(args.arr_body or leg_get(leg, "arr", "arr_body", default=None))
    if not arr_body: raise SystemExit("--arr-body or anchor leg arr/arr_body is required")
    if args.arrival_t_game_s is not None:
        arrival = args.arrival_t_game_s
    else:
        arrival = leg_get(leg, "t_arr_s", "arrival_t_game_s", default=None)
        if arrival is None: raise SystemExit("--arrival-t-game-s or anchor leg t_arr_s is required")
        arrival = float(arrival)

    with SnapshotNavClient(args.server, args.plugin_b64, args.plugin_arg_mode, args.timeout_s, args.quiet_stderr, args.output_dir / "principia_server_stderr.log") as client:
        print("=== SNAP DEPARTURE VCAREL_NAV TARGETER V0 ===")
        print(f"ready_line       : {client.ready_line}")
        print(f"ping             : {client.ping()}")
        snap = client.loadsnap(args.snapshot_json)
        cfg = Config(dep_body=dep_body, arr_body=arr_body, nav_body=args.nav_body, state_t_game_s=snap.t_game_s, nominal_arrival_t_game_s=arrival,
                     scan_half_width_days=args.scan_half_width_days, samples=args.samples, burn_dt_min_s=args.burn_dt_min_s, burn_dt_max_s=args.burn_dt_max_s,
                     dvt_min_m_s=args.dvt_min_m_s, dvt_max_m_s=args.dvt_max_m_s, dvn_max_abs_m_s=args.dvn_max_abs_m_s, dvb_max_abs_m_s=args.dvb_max_abs_m_s,
                     arrival_offset_min_days=args.arrival_offset_min_days, arrival_offset_max_days=args.arrival_offset_max_days,
                     enable_dsm=args.enable_dsm, dsm_frac_min=args.dsm_frac_min, dsm_frac_max=args.dsm_frac_max, dsm_max_abs_m_s=args.dsm_max_abs_m_s,
                     ca_scale_km=args.ca_scale_km, dv_scale_m_s=args.dv_scale_m_s, dv_weight=args.dv_weight, dsm_weight=args.dsm_weight,
                     out_of_plane_weight=args.out_of_plane_weight, binormal_weight=args.binormal_weight, normal_weight=args.normal_weight,
                     burn_time_weight=args.burn_time_weight, require_positive_tangent=not args.allow_negative_tangent,
                     max_total_dv_m_s=args.max_total_dv_m_s if args.max_total_dv_m_s > 0 else None,
                     max_out_of_plane_fraction=args.max_out_of_plane_fraction if args.max_out_of_plane_fraction > 0 else None)
        print(f"snapshot         : {snap}")
        print(f"dep -> arr/nav   : {cfg.dep_body} -> {cfg.arr_body} / {cfg.nav_body}")
        print(f"state_t_game_s   : {cfg.state_t_game_s}")
        print(f"arrival_t_game_s : {cfg.nominal_arrival_t_game_s}")
        print(f"tof_days         : {(cfg.nominal_arrival_t_game_s - cfg.state_t_game_s)/DAY_S:.6f}")
        if not args.no_warmup:
            try:
                w = client.snapvca_nav("warmup_kerbin_1h", snap.nav_body or cfg.dep_body, "AUTO", 0, 3600, 15, [])
                print(f"warmup           : ca={w.ca_distance_m/1000:.3f} km status={w.status}")
            except Exception as e:
                print(f"warmup WARN      : {e}")

        targeter = Targeter(client, cfg)
        bounds = targeter.bounds(); dim = len(bounds)
        pts = sample(bounds, args.coarse_samples, args.seed)
        manual = parse_manual_seeds(args.manual_seeds, dim)
        if manual: pts = np.vstack([np.asarray(manual, dtype=float), pts])
        print(f"coarse_points    : {len(pts)} dim={dim}")

        for i, x in enumerate(pts, 1):
            targeter.eval(x, kind="coarse")
            if i == 1 or i % max(1, min(25, len(pts)//8 or 1)) == 0:
                b = targeter.best
                if b:
                    print(f"[coarse {i:4d}/{len(pts):4d}] best score={b['score']:.6g} ca={b['ca_distance_km']:.3f} km dv={b['total_dv_m_s']:.2f} burn={b['burn_dt_s']:.1f}s TNB={[b['dvt_m_s'], b['dvn_m_s'], b['dvb_m_s']]} arr_off={b['arrival_offset_days']:.3f}d")

        ok = [r for r in targeter.rows if r.get("ok") and math.isfinite(float(r.get("score", math.inf)))]
        ok.sort(key=lambda r: r["score"])
        print("\n=== TOP COARSE ===")
        for i, r in enumerate(ok[:20], 1):
            print(f"{i:2d} score={r['score']:12.6g} ca={r['ca_distance_km']:12.3f} km dv={r['total_dv_m_s']:8.2f} oop={r['out_of_plane_fraction']:6.3f} burn={r['burn_dt_s']:8.1f}s T={r['dvt_m_s']:8.1f} N={r['dvn_m_s']:8.1f} B={r['dvb_m_s']:8.1f} arr_off={r['arrival_offset_days']:7.2f}d status={r.get('status')}")

        if minimize is not None and args.refine_top_n > 0 and ok:
            print("\n=== POWELL REFINEMENT ===")
            for j, r in enumerate(ok[:args.refine_top_n], 1):
                x0 = np.asarray(r["x"], dtype=float)
                print(f"[powell {j}] x0={x0.tolist()} ca={r['ca_distance_km']:.3f} km score={r['score']:.6g}")
                res = minimize(targeter.objective, x0, method="Powell", bounds=bounds,
                               options={"maxiter": args.powell_maxiter, "maxfev": args.powell_maxfev, "xtol": 1e-3, "ftol": 1e-4, "disp": False})
                rr = targeter.eval(res.x, kind="powell_final")
                rr["powell_success"] = bool(res.success); rr["powell_message"] = str(res.message); rr["powell_fun"] = float(res.fun)
                print(f"[powell {j}] success={res.success} ca={rr.get('ca_distance_km')} score={rr.get('score')} x={res.x.tolist()}")

        ok = [r for r in targeter.rows if r.get("ok") and math.isfinite(float(r.get("score", math.inf)))]
        ok.sort(key=lambda r: r["score"])
        best = ok[0] if ok else None
        print("\n=== TOP FINAL ===")
        for i, r in enumerate(ok[:30], 1):
            print(f"{i:2d} score={r['score']:12.6g} ca={r['ca_distance_km']:12.3f} km ca_t={r['ca_t_game_s']:.3f} dv={r['total_dv_m_s']:8.2f} oop={r['out_of_plane_fraction']:6.3f} burn={r['burn_dt_s']:8.1f}s T={r['dvt_m_s']:8.1f} N={r['dvn_m_s']:8.1f} B={r['dvb_m_s']:8.1f} arr_off={r['arrival_offset_days']:7.2f}d status={r.get('status')}")

        result = {"schema": "snap_departure_vcarelnav_targeter_v0", "ready_line": client.ready_line, "server": str(args.server), "plugin_b64": str(args.plugin_b64),
                  "snapshot_json": str(args.snapshot_json), "anchor_json": str(args.anchor_json) if args.anchor_json else None, "leg": args.leg,
                  "snapshot": asdict(snap), "config": asdict(cfg), "n_evaluations": len(targeter.rows), "n_ok": len(ok), "best": best, "top": ok[:50]}
        (args.output_dir / "snap_departure_vcarelnav_targeter_result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
        flat = [flatten(r) for r in targeter.rows]
        if flat:
            fields = sorted({k for r in flat for k in r.keys()})
            with (args.output_dir / "snap_departure_vcarelnav_targeter_rows.csv").open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(flat)
        print(f"[OK] wrote {args.output_dir / 'snap_departure_vcarelnav_targeter_result.json'}")
        print(f"[OK] wrote {args.output_dir / 'snap_departure_vcarelnav_targeter_rows.csv'}")

        if args.write_events and best:
            write_event(args.output_dir / "event1_snap_target_burn0_navigation.json", snap_data, snap, best, cfg, "burn0")
            print(f"[OK] wrote {args.output_dir / 'event1_snap_target_burn0_navigation.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
