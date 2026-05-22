#!/usr/bin/env python3
"""
Standalone safe SNAPVCA_NAV departure scanner.

No imports from project scripts. Uses:
  LOADSNAP
  SNAPVCA_NAV

It avoids random delta-v coarse search. It first generates physically-plausible
local two-body ejection seeds from the PyKEP departure v_inf, then validates only
those seeds with Principia.
"""
from __future__ import annotations

import argparse, csv, json, math, queue, subprocess, sys, threading
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy.integrate import solve_ivp

DAY_S = 86400.0


def norm(v: Sequence[float]) -> float:
    return float(np.linalg.norm(np.asarray(v, dtype=float)))


def unit(v: Sequence[float], name: str = "vector") -> np.ndarray:
    a = np.asarray(v, dtype=float)
    n = norm(a)
    if n <= 0 or not math.isfinite(n):
        raise ValueError(f"cannot normalize {name}: {v!r}")
    return a / n


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def angle_deg(a: Sequence[float], b: Sequence[float]) -> float:
    na, nb = norm(a), norm(b)
    if na <= 0 or nb <= 0:
        return math.nan
    return math.degrees(math.acos(clamp(float(np.dot(a, b)) / (na * nb), -1, 1)))


def sanitize_body(x: Any) -> str:
    return "" if x is None else str(x).strip().upper()


def vec3(fields: Sequence[str], i: int) -> list[float]:
    return [float(fields[i]), float(fields[i + 1]), float(fields[i + 2])]


def safe_float(x: Any, default: float) -> float:
    try:
        y = float(x)
        return y if math.isfinite(y) else default
    except Exception:
        return default


def flatten(row: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for k, v in row.items():
        if isinstance(v, np.ndarray):
            v = v.tolist()
        if isinstance(v, (list, tuple)):
            if all(not isinstance(x, (dict, list, tuple)) for x in v):
                for i, x in enumerate(v):
                    out[f"{k}_{i}"] = x
            else:
                out[k] = json.dumps(v, ensure_ascii=False)
        elif isinstance(v, dict):
            out[k] = json.dumps(v, ensure_ascii=False)
        else:
            out[k] = v
    return out


def infer(leg: dict[str, Any], *names: str, default=None):
    for n in names:
        if n in leg and leg[n] is not None:
            return leg[n]
    return default


def load_leg(path: Path, leg_index: int) -> dict[str, Any]:
    d = json.loads(path.read_text())
    if isinstance(d.get("legs"), list):
        return d["legs"][leg_index - 1]
    if f"leg{leg_index}" in d:
        return d[f"leg{leg_index}"]
    if leg_index == 1:
        return d
    raise RuntimeError(f"could not find leg {leg_index} in {path}")


def levela_to_raw(v: Sequence[float]) -> list[float]:
    x, y, z = map(float, v)
    return [z, -x, y]


def get_vinf_raw_m_s(leg: dict[str, Any], kind: str) -> np.ndarray:
    for k in (f"vinf_{kind}_raw_m_s", f"vinf_{kind}_m_s_raw", f"vinf_{kind}_raw"):
        if k in leg:
            return np.asarray(leg[k], dtype=float)
    if f"vinf_{kind}_levela_km_s" in leg:
        return np.asarray(levela_to_raw([1000 * float(x) for x in leg[f"vinf_{kind}_levela_km_s"]]), dtype=float)
    if f"vinf_{kind}_levela_m_s" in leg:
        return np.asarray(levela_to_raw(leg[f"vinf_{kind}_levela_m_s"]), dtype=float)
    raise RuntimeError(f"cannot find vinf_{kind}; leg keys={sorted(leg.keys())}")


def load_body_catalog(path: Path) -> dict[str, dict[str, float | None]]:
    data = json.loads(path.read_text())
    out = {}
    def visit(o: Any):
        if isinstance(o, dict):
            name = o.get("name") or o.get("body") or o.get("id")
            mu = o.get("gravitational_parameter") or o.get("mu") or o.get("gm_m3_s2") or o.get("gm")
            radius = o.get("radius") or o.get("radius_m") or o.get("radius_km")
            if name and mu is not None:
                rkm = None
                if radius is not None:
                    r = float(radius)
                    rkm = r / 1000 if r > 1e5 else r
                out[sanitize_body(name)] = {"mu_m3_s2": float(mu), "radius_km": rkm}
            for v in o.values():
                visit(v)
        elif isinstance(o, list):
            for x in o: visit(x)
    visit(data)
    return out


class ProtocolError(RuntimeError):
    pass


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
    n_burns: int
    raw_field_count: int


class SnapshotClient:
    def __init__(self, server: Path, plugin_b64: Path, plugin_arg_mode="positional", startup_timeout_s=90, command_timeout_s=75, stderr_log: Path | None = None, quiet_stderr=False):
        self.server = Path(server)
        self.plugin_b64 = Path(plugin_b64)
        self.plugin_arg_mode = plugin_arg_mode
        self.startup_timeout_s = startup_timeout_s
        self.command_timeout_s = command_timeout_s
        self.stderr_log = stderr_log
        self.quiet_stderr = quiet_stderr
        self.proc = None
        self.stdout_q = queue.Queue()
        self.stderr_f = None
        self.ready_line = ""

    def argv(self):
        a = [str(self.server)]
        if self.plugin_arg_mode == "positional": a.append(str(self.plugin_b64))
        elif self.plugin_arg_mode == "flag": a += ["--plugin-b64", str(self.plugin_b64)]
        elif self.plugin_arg_mode == "none": pass
        else: raise ValueError(self.plugin_arg_mode)
        return a

    def __enter__(self):
        self.start(); return self
    def __exit__(self, *args):
        self.close()

    def start(self):
        if self.stderr_log:
            self.stderr_log.parent.mkdir(parents=True, exist_ok=True)
            self.stderr_f = self.stderr_log.open("w", encoding="utf-8")
        self.proc = subprocess.Popen(self.argv(), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
        assert self.proc.stdout and self.proc.stderr
        def outpump():
            for line in self.proc.stdout:
                self.stdout_q.put(line.rstrip("\n"))
        def errpump():
            for line in self.proc.stderr:
                line = line.rstrip("\n")
                if self.stderr_f:
                    self.stderr_f.write(line + "\n"); self.stderr_f.flush()
                if not self.quiet_stderr:
                    print(f"[server-stderr] {line}", file=sys.stderr)
        threading.Thread(target=outpump, daemon=True).start()
        threading.Thread(target=errpump, daemon=True).start()
        first = self.read(self.startup_timeout_s)
        if not first.startswith("READY"):
            raise ProtocolError(f"expected READY, got {first!r}")
        banners = [first]
        while True:
            try:
                extra = self.stdout_q.get(timeout=0.15)
            except queue.Empty:
                break
            if extra.startswith(("OK", "ERR", "PONG")):
                raise ProtocolError(f"unexpected protocol line during startup drain: {extra!r}")
            if extra: banners.append(extra)
        self.ready_line = "\t".join(banners)

    def close(self):
        if self.proc:
            try:
                if self.proc.stdin: self.proc.stdin.close()
            except Exception: pass
            try:
                self.proc.terminate(); self.proc.wait(timeout=3)
            except Exception:
                try: self.proc.kill()
                except Exception: pass
            self.proc = None
        if self.stderr_f:
            self.stderr_f.close(); self.stderr_f = None

    def read(self, timeout_s: float) -> str:
        try:
            return self.stdout_q.get(timeout=timeout_s)
        except queue.Empty:
            rc = None if self.proc is None else self.proc.poll()
            raise TimeoutError(f"timeout waiting for server response; returncode={rc}")

    def command(self, fields: Sequence[Any], timeout_s: float | None = None) -> list[str]:
        if not self.proc or not self.proc.stdin:
            raise RuntimeError("server not started")
        self.proc.stdin.write("\t".join(str(x) for x in fields) + "\n")
        self.proc.stdin.flush()
        line = self.read(timeout_s or self.command_timeout_s)
        p = line.split("\t")
        if p[0].startswith("ERR"):
            raise ProtocolError(line)
        return p

    def loadsnap(self, path: Path, rid="snap0") -> OkSnap:
        p = self.command(["LOADSNAP", rid, str(path)], timeout_s=self.command_timeout_s)
        if p[0] != "OKSNAP":
            raise ProtocolError(f"expected OKSNAP, got {p[:8]}")
        if len(p) < 16:
            raise ProtocolError(f"OKSNAP too short: {p}")
        return OkSnap(p[1], p[2], float(p[3]), p[4], sanitize_body(p[5]), sanitize_body(p[6]), vec3(p,7), vec3(p,10), float(p[13]), float(p[14]), float(p[15]))

    def snapvca_nav(self, rid: str, arr_body: str, nav_body: str, scan_start: float, scan_end: float, samples: int, impulses: Sequence[tuple[float,float,float,float]], timeout_s: float | None = None) -> OkCarelNav:
        f = ["SNAPVCA_NAV", rid, sanitize_body(arr_body), nav_body or "AUTO", float(scan_start), float(scan_end), int(samples), len(impulses)]
        for dt,t,n,b in impulses:
            f += [float(dt), float(t), float(n), float(b)]
        p = self.command(f, timeout_s=timeout_s)
        return parse_okcarelnav(p)


def parse_okcarelnav(p: Sequence[str]) -> OkCarelNav:
    if p[0] != "OKCARELNAV":
        raise ProtocolError(f"expected OKCARELNAV, got {p[:8]}")
    if len(p) < 33:
        raise ProtocolError(f"OKCARELNAV too short ({len(p)}): {p}")
    return OkCarelNav(
        id=p[1], dep_body=sanitize_body(p[2]), arr_body=sanitize_body(p[3]), nav_body=sanitize_body(p[4]),
        state_dt_s=float(p[5]), state_t_game_s=float(p[6]), ca_dt_s=float(p[7]), ca_t_game_s=float(p[8]),
        ca_rel_r_raw_m=vec3(p,9), ca_rel_v_raw_m_s=vec3(p,12), ca_distance_m=float(p[15]),
        ca_speed_m_s=float(p[16]), ca_radial_velocity_m_s=float(p[17]), samples=int(p[18]), status=p[19],
        n_burns=int(p[32]), raw_field_count=len(p))


def period_s(r, v, mu):
    rn, vn = norm(r), norm(v)
    eps = 0.5*vn*vn - mu/rn
    if eps >= 0: return None
    a = -mu/(2*eps)
    if a <= 0 or not math.isfinite(a): return None
    return 2*math.pi*math.sqrt(a**3/mu)


def propagate_dense(r0, v0, mu, tmax):
    def rhs(_t, y):
        r, v = y[:3], y[3:]
        return np.r_[v, -mu*r/norm(r)**3]
    sol = solve_ivp(rhs, (0,float(tmax)), np.r_[r0,v0], method="DOP853", dense_output=True, rtol=1e-11, atol=[1e-3]*3+[1e-9]*3, max_step=120)
    if not sol.success: raise RuntimeError(sol.message)
    return sol


def tnb_principia(r, v):
    T = unit(v, "T")
    H = unit(np.cross(r,v), "H")
    N = unit(np.cross(H,T), "N")
    B = unit(np.cross(N,T), "B")
    return T,N,B


def ejection_at(r, v, mu, vinf):
    rp = norm(r); vinfm = norm(vinf); vinfh = unit(vinf, "vinf"); rh = unit(r, "r")
    e = 1 + rp*vinfm*vinfm/mu
    c = 1/e; s = math.sqrt(max(0,1-c*c))
    vp = math.sqrt(vinfm*vinfm + 2*mu/rp)
    raw_that = vinfh + c*rh
    tang = raw_that - float(np.dot(raw_that,rh))*rh
    if norm(tang) <= 1e-12: raise RuntimeError("degenerate tangent")
    that = unit(tang, "that")
    pred = unit(-c*rh + s*that, "pred")
    dv_raw = vp*that - v
    T,N,B = tnb_principia(r,v)
    dvt,dvn,dvb = float(np.dot(dv_raw,T)), float(np.dot(dv_raw,N)), float(np.dot(dv_raw,B))
    dvnrm = norm([dvt,dvn,dvb]); oop = norm([dvn,dvb])
    return dict(phase_angle_deg=angle_deg(pred, vinfh), phase_scalar_error=float(np.dot(rh,vinfh)+c), rp_km=rp/1000, v_pre_norm_m_s=norm(v), v_post_norm_m_s=vp, dvt_m_s=dvt, dvn_m_s=dvn, dvb_m_s=dvb, dv_norm_m_s=dvnrm, out_of_plane_abs_m_s=oop, out_of_plane_fraction=oop/max(dvnrm,1), dv_raw_m_s=dv_raw.tolist(), predicted_vinf_hat=pred.tolist())


def write_event(outdir: Path, snapshot_json: Path, snap: OkSnap, best: dict[str,Any], dep: str, arr: str, nav: str):
    try: sd = json.loads(snapshot_json.read_text())
    except Exception: sd = {}
    vessel = sd.get("vessel", {}) if isinstance(sd, dict) else {}
    t0 = snap.t_game_s + float(best["burn_dt_s"])
    ev = {
        "enabled": True, "vessel_guid": snap.vessel_guid or vessel.get("vessel_guid", ""),
        "ensure_flight_plan": True, "extend_existing_flight_plan": True,
        "mass_tonnes": safe_float(snap.mass_tonnes, safe_float(vessel.get("mass_tonnes"), 2.6)),
        "insert_index": 0, "burn_template": "json",
        "thrust_kN": safe_float(snap.available_thrust_kN, safe_float(vessel.get("available_thrust_kN"), 90.0)),
        "specific_impulse_s_g0": safe_float(snap.specific_impulse_s_g0, safe_float(vessel.get("specific_impulse_s_g0"), 345.0)),
        "is_inertially_fixed": False, "frame_extension": 6000, "frame_centre_from_active_body": True,
        "frame_centre_index": -1, "frame_primary_index": -1, "frame_secondary_index": -1,
        "placeholder_dv_m_s": 0.001, "require_status_ok": True, "cleanup_on_error": True,
        "tolerance_time_s": 0.01, "tolerance_dv_m_s": 1e-6,
        "one_shot": True, "disable_after_success": True,
        "request_id": "snap_vinf_ejection_burn0_attempt0", "dedupe_tag": "snap_vinf_ejection_burn0", "event_key": "snap_vinf_ejection_burn0", "attempt": 0,
        "mode": "insert_navigation", "initial_time": t0, "plan_final_time": t0 + 600,
        "delta_v_navigation_m_s": [float(best["dvt_m_s"]), float(best["dvn_m_s"]), float(best["dvb_m_s"])],
        "planned_from_state": {"schema":"planned_from_snap_vinf_ejection_v0_1", "state_t_game_s":snap.t_game_s, "vessel_guid":snap.vessel_guid, "dep_body":dep, "arr_body":arr, "nav_body":nav, "rel_r_raw_m":snap.rel_r_raw_m, "rel_v_raw_m_s":snap.rel_v_raw_m_s, "score_ca_distance_km":best.get("ca_distance_km"), "phase_angle_deg":best.get("phase_angle_deg"), "status":best.get("status")},
        "require_warp_close": True, "max_lead_before_insert_s": 600, "reject_long_plan_before_first_burn": True, "max_first_plan_duration_s": 900, "rollback_on_status_error": True, "auto_sort_insert_index": True,
    }
    p = outdir / "event1_snap_vinf_ejection_burn0_navigation.json"
    p.write_text(json.dumps(ev, indent=2, ensure_ascii=False) + "\n")
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", type=Path, required=True); ap.add_argument("--plugin-b64", type=Path, required=True); ap.add_argument("--plugin-arg-mode", default="positional", choices=["positional","flag","none"])
    ap.add_argument("--snapshot-json", type=Path, required=True); ap.add_argument("--anchor-json", type=Path, required=True); ap.add_argument("--body-catalog", type=Path, required=True); ap.add_argument("--leg", type=int, default=1)
    ap.add_argument("--dep-body"); ap.add_argument("--arr-body"); ap.add_argument("--nav-body", default="AUTO"); ap.add_argument("--arrival-t-game-s", type=float)
    ap.add_argument("--burn-dt-min-s", type=float, default=0); ap.add_argument("--burn-dt-max-s", type=float); ap.add_argument("--burn-grid", type=int, default=721); ap.add_argument("--phase-top-n", type=int, default=80); ap.add_argument("--validate-top-n", type=int, default=25)
    ap.add_argument("--dv-min-m-s", type=float, default=500); ap.add_argument("--dv-max-m-s", type=float, default=4200); ap.add_argument("--max-normal-abs-m-s", type=float, default=350); ap.add_argument("--max-binormal-abs-m-s", type=float, default=350); ap.add_argument("--max-out-of-plane-fraction", type=float, default=0.30); ap.add_argument("--max-phase-deg", type=float, default=25)
    ap.add_argument("--arrival-offset-days", type=float, default=0); ap.add_argument("--scan-half-width-days", type=float, default=45); ap.add_argument("--samples", type=int, default=31)
    ap.add_argument("--startup-timeout-s", type=float, default=90); ap.add_argument("--command-timeout-s", type=float, default=75); ap.add_argument("--quiet-stderr", action="store_true"); ap.add_argument("--write-event", action="store_true"); ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True)

    leg = load_leg(args.anchor_json, args.leg)
    dep = sanitize_body(args.dep_body or infer(leg,"dep","dep_body", default="KERBIN")); arr = sanitize_body(args.arr_body or infer(leg,"arr","arr_body"))
    if not arr: raise SystemExit("arr_body missing")
    t_arr = float(args.arrival_t_game_s if args.arrival_t_game_s is not None else infer(leg,"t_arr_s","arrival_t_game_s"))
    bodies = load_body_catalog(args.body_catalog)
    if dep not in bodies: raise SystemExit(f"{dep} missing in body catalog")
    mu = float(bodies[dep]["mu_m3_s2"])
    vinf = get_vinf_raw_m_s(leg, "dep")
    phase_rows=[]; val_rows=[]
    stderr_log = args.output_dir / "principia_server_stderr.log"

    with SnapshotClient(args.server,args.plugin_b64,args.plugin_arg_mode,args.startup_timeout_s,args.command_timeout_s,stderr_log,args.quiet_stderr) as client:
        print("=== SCAN SNAP VINF EJECTION VCAREL_NAV V0.1 STANDALONE ===")
        print(f"ready_line       : {client.ready_line}")
        snap = client.loadsnap(args.snapshot_json)
        print(f"snapshot         : t={snap.t_game_s:.6f} r={norm(snap.rel_r_raw_m)/1000:.3f} km v={norm(snap.rel_v_raw_m_s):.3f} m/s")
        print(f"dep -> arr/nav   : {dep} -> {arr} / {args.nav_body}")
        print(f"t_arr            : {t_arr:.6f}")
        print(f"tof_remaining_d  : {(t_arr-snap.t_game_s)/DAY_S:.6f}")
        print(f"vinf_dep_raw     : {vinf.tolist()} |v|={norm(vinf):.6f}")
        per = period_s(snap.rel_r_raw_m, snap.rel_v_raw_m_s, mu)
        tmax = args.burn_dt_max_s if args.burn_dt_max_s is not None else min(21600.0, max(7200.0, 3*per if per else 21600.0))
        print(f"period_s         : {per}"); print(f"burn_dt range    : {args.burn_dt_min_s} .. {tmax}")
        sol = propagate_dense(np.asarray(snap.rel_r_raw_m,float), np.asarray(snap.rel_v_raw_m_s,float), mu, tmax)
        for burn_dt in np.linspace(args.burn_dt_min_s, float(tmax), args.burn_grid):
            y=np.asarray(sol.sol(float(burn_dt)),float); r=y[:3]; v=y[3:]
            try:
                ej=ejection_at(r,v,mu,vinf)
                gate=(abs(ej["phase_angle_deg"])<=args.max_phase_deg and args.dv_min_m_s<=ej["dv_norm_m_s"]<=args.dv_max_m_s and abs(ej["dvn_m_s"])<=args.max_normal_abs_m_s and abs(ej["dvb_m_s"])<=args.max_binormal_abs_m_s and ej["out_of_plane_fraction"]<=args.max_out_of_plane_fraction)
                row={"ok":True,"gate_ok":gate,"burn_dt_s":float(burn_dt),"burn_abs_s":snap.t_game_s+float(burn_dt),"rmag_km":norm(r)/1000,"vmag_m_s":norm(v),"phase_score":abs(ej["phase_angle_deg"])+0.0001*ej["dv_norm_m_s"]+0.01*ej["out_of_plane_abs_m_s"], **ej}
            except Exception as e:
                row={"ok":False,"gate_ok":False,"burn_dt_s":float(burn_dt),"error":str(e)}
            phase_rows.append(row)
        phase_ok=[r for r in phase_rows if r.get("ok") and r.get("gate_ok")]
        if not phase_ok:
            print("[WARN] no candidates pass strict gates; validating best phase candidates without gate")
            phase_ok=[r for r in phase_rows if r.get("ok")]
        phase_ok.sort(key=lambda r:float(r["phase_score"])); top=phase_ok[:args.phase_top_n]
        print("\n=== TOP TWO-BODY PHASE SEEDS ===")
        for i,r in enumerate(top[:20],1):
            print(f"{i:2d} burn={r['burn_dt_s']:9.1f}s phase={r['phase_angle_deg']:8.3f}deg dv={r['dv_norm_m_s']:8.2f} T={r['dvt_m_s']:8.1f} N={r['dvn_m_s']:8.1f} B={r['dvb_m_s']:8.1f} oop={r['out_of_plane_fraction']:6.3f}")
        center=(t_arr-snap.t_game_s)+args.arrival_offset_days*DAY_S; start=center-args.scan_half_width_days*DAY_S; end=center+args.scan_half_width_days*DAY_S
        print("\n=== SNAPVCA_NAV VALIDATION ===")
        for i,seed in enumerate(top[:args.validate_top_n],1):
            try:
                res=client.snapvca_nav(f"vinfseed_{i:04d}", arr, args.nav_body, start, end, args.samples, [(seed["burn_dt_s"],seed["dvt_m_s"],seed["dvn_m_s"],seed["dvb_m_s"])], timeout_s=args.command_timeout_s)
                row=dict(seed); row.update({"validation_ok":True,"ca_distance_km":res.ca_distance_m/1000,"ca_distance_m":res.ca_distance_m,"ca_t_game_s":res.ca_t_game_s,"ca_dt_s":res.ca_dt_s,"ca_speed_m_s":res.ca_speed_m_s,"ca_radial_velocity_m_s":res.ca_radial_velocity_m_s,"status":res.status,"ca_rel_r_raw_m":res.ca_rel_r_raw_m,"ca_rel_v_raw_m_s":res.ca_rel_v_raw_m_s,"raw_field_count":res.raw_field_count,"score":res.ca_distance_m/1000+0.001*seed["dv_norm_m_s"]+10*seed["out_of_plane_fraction"]})
                print(f"[val {i:3d}/{min(args.validate_top_n,len(top)):3d}] ca={row['ca_distance_km']:12.3f}km burn={row['burn_dt_s']:9.1f}s dv={row['dv_norm_m_s']:8.2f} T={row['dvt_m_s']:8.1f} N={row['dvn_m_s']:8.1f} B={row['dvb_m_s']:8.1f} status={row['status']}")
            except Exception as e:
                row=dict(seed); row.update({"validation_ok":False,"error":str(e),"score":math.inf}); print(f"[val {i:3d}] ERR {e}")
                if isinstance(e, TimeoutError) or "timeout" in str(e).lower():
                    val_rows.append(row); print("[ABORT] server timed out; stopping validation to avoid protocol desync"); break
            val_rows.append(row)

    valid=[r for r in val_rows if r.get("validation_ok") and math.isfinite(float(r.get("score",math.inf)))]
    valid.sort(key=lambda r:float(r["score"])); best=valid[0] if valid else None
    print("\n=== TOP VALIDATED ===")
    for i,r in enumerate(valid[:30],1):
        print(f"{i:2d} score={r['score']:12.3f} ca={r['ca_distance_km']:12.3f}km burn={r['burn_dt_s']:9.1f}s dv={r['dv_norm_m_s']:8.2f} T={r['dvt_m_s']:8.1f} N={r['dvn_m_s']:8.1f} B={r['dvb_m_s']:8.1f} oop={r['out_of_plane_fraction']:6.3f} phase={r['phase_angle_deg']:8.3f}")
    result={"schema":"scan_snap_vinf_ejection_vcarelnav_v0_1_standalone","snapshot_json":str(args.snapshot_json),"anchor_json":str(args.anchor_json),"leg":args.leg,"dep_body":dep,"arr_body":arr,"nav_body":args.nav_body,"vinf_dep_raw_m_s":vinf.tolist(),"vinf_dep_norm_m_s":norm(vinf),"n_phase_rows":len(phase_rows),"n_phase_ok":len([r for r in phase_rows if r.get("ok")]),"n_phase_gate_ok":len([r for r in phase_rows if r.get("ok") and r.get("gate_ok")]),"n_validated":len(val_rows),"n_valid":len(valid),"best":best,"top_phase":top[:50],"top_validated":valid[:50],"config":{k:(str(v) if isinstance(v,Path) else v) for k,v in vars(args).items()}}
    result_json=args.output_dir/"scan_snap_vinf_ejection_result.json"; phase_csv=args.output_dir/"scan_snap_vinf_ejection_phase_rows.csv"; valid_csv=args.output_dir/"scan_snap_vinf_ejection_validated_rows.csv"
    result_json.write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n")
    for p,rows in [(phase_csv,phase_rows),(valid_csv,val_rows)]:
        flat=[flatten(r) for r in rows]
        if flat:
            fields=sorted({k for r in flat for k in r})
            with p.open("w",newline="",encoding="utf-8") as f:
                w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(flat)
    print(f"[OK] wrote {result_json}"); print(f"[OK] wrote {phase_csv}"); print(f"[OK] wrote {valid_csv}"); print(f"[OK] wrote {stderr_log}")
    if args.write_event and best:
        # Reloading snapshot data is enough for event metadata.
        # snap is still in local scope if LOADSNAP succeeded.
        event_path=write_event(args.output_dir,args.snapshot_json,snap,best,dep,arr,args.nav_body); print(f"[OK] wrote {event_path}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
