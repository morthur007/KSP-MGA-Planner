#!/usr/bin/env python3
"""
rank_snapshot_anchor_executability_v0.py

Ranking rápido offline de anchors PyKEP contra o snapshot vivo atual.

Não chama o Principia server. Serve para decidir quais rotas/anchors valem a
validação cara com SNAPVCA_NAV.

Para cada anchor:
  - lê a primeira perna;
  - pega vinf_dep_raw_m_s;
  - propaga a órbita atual da nave em two-body local;
  - calcula a queima de ejeção que produz esse v_inf;
  - aplica gates de executabilidade: |N|, |B|, fração fora do plano, dv, T>0;
  - ranqueia anchors com candidato plausível.

Uso típico:
  python scripts/rank_snapshot_anchor_executability_v0.py \
    --snapshot-json GameData/MGAPlanner/principia_live_navigation_snapshot_v0_1.json \
    --body-catalog data/catalogs/jnsq/body_catalog.json \
    --anchor-glob 'data/runs/game_export/rank12_real/**/anchor_packet.json' \
    --output-dir data/runs/game_export/rank12_real/snap_global_anchor_rank01
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
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
    if n <= 0.0 or not math.isfinite(n):
        raise ValueError(f"cannot normalize {name}: {v!r}")
    return a / n


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def angle_deg(a: Sequence[float], b: Sequence[float]) -> float:
    na, nb = norm(a), norm(b)
    if na <= 0 or nb <= 0:
        return math.nan
    return math.degrees(math.acos(clamp(float(np.dot(a, b)) / (na * nb), -1.0, 1.0)))


def sanitize_body(name: Any) -> str:
    return "" if name is None else str(name).strip().upper()


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


def infer_leg_field(leg: dict[str, Any] | None, *names: str, default=None):
    if not isinstance(leg, dict):
        return default
    for name in names:
        if name in leg and leg[name] is not None:
            return leg[name]
    return default


def levela_to_raw(v: Sequence[float]) -> list[float]:
    x, y, z = map(float, v)
    return [z, -x, y]


def get_vinf_raw_m_s(leg: dict[str, Any], kind: str) -> np.ndarray:
    for k in (f"vinf_{kind}_raw_m_s", f"vinf_{kind}_m_s_raw", f"vinf_{kind}_raw"):
        if k in leg:
            return np.asarray(leg[k], dtype=float)
    if f"vinf_{kind}_levela_km_s" in leg:
        return np.asarray(levela_to_raw([1000.0 * float(x) for x in leg[f"vinf_{kind}_levela_km_s"]]), dtype=float)
    if f"vinf_{kind}_levela_m_s" in leg:
        return np.asarray(levela_to_raw(leg[f"vinf_{kind}_levela_m_s"]), dtype=float)
    raise RuntimeError(f"cannot find vinf_{kind}; keys={sorted(leg.keys())}")


def load_anchor(path: Path, leg_index: int = 1) -> tuple[dict[str, Any], dict[str, Any]]:
    data = json.loads(path.read_text())
    if isinstance(data.get("legs"), list):
        leg = data["legs"][leg_index - 1]
    elif f"leg{leg_index}" in data:
        leg = data[f"leg{leg_index}"]
    elif leg_index == 1:
        leg = data.get("leg1", data)
    else:
        raise ValueError(f"no leg{leg_index}")

    return data, leg


def sequence_string(anchor: dict[str, Any], leg: dict[str, Any]) -> str:
    seq = anchor.get("sequence") or anchor.get("sequence_bodies")
    if isinstance(seq, list):
        return " ".join(map(str, seq))
    if isinstance(seq, str):
        return seq
    cm = anchor.get("candidate_metrics") or {}
    if cm.get("sequence_bodies"):
        return str(cm["sequence_bodies"])
    dep = infer_leg_field(leg, "dep", "dep_body", default="?")
    arr = infer_leg_field(leg, "arr", "arr_body", default="?")
    return f"{dep} {arr}"


def load_body_catalog(path: Path) -> dict[str, dict[str, float | None]]:
    data = json.loads(path.read_text())
    out: dict[str, dict[str, float | None]] = {}

    def visit(obj: Any) -> None:
        if isinstance(obj, dict):
            name = obj.get("name") or obj.get("body") or obj.get("id")
            gm = obj.get("gravitational_parameter") or obj.get("mu") or obj.get("gm") or obj.get("gm_m3_s2")
            radius = obj.get("radius") or obj.get("radius_m") or obj.get("radius_km")
            if name and gm is not None:
                body = sanitize_body(name)
                r_km = None
                if radius is not None:
                    r = float(radius)
                    r_km = r / 1000.0 if r > 1e5 else r
                out[body] = {"mu_m3_s2": float(gm), "radius_km": r_km}
            for v in obj.values():
                visit(v)
        elif isinstance(obj, list):
            for x in obj:
                visit(x)

    visit(data)
    return out


def load_snapshot_vessel(path: Path) -> dict[str, Any]:
    d = json.loads(path.read_text())
    if "vessel" in d and isinstance(d["vessel"], dict):
        return d["vessel"]
    return d


def tnb_basis_principia_like(r: Sequence[float], v: Sequence[float]):
    r = np.asarray(r, dtype=float)
    v = np.asarray(v, dtype=float)
    T = unit(v, "T")
    H = unit(np.cross(r, v), "H")
    N = unit(np.cross(H, T), "N")
    B = unit(np.cross(N, T), "B")
    return T, N, B


def two_body_period_s(r: Sequence[float], v: Sequence[float], mu: float) -> float | None:
    rn = norm(r)
    vn = norm(v)
    eps = 0.5 * vn * vn - mu / rn
    if eps >= 0.0:
        return None
    a = -mu / (2.0 * eps)
    if a <= 0.0 or not math.isfinite(a):
        return None
    return 2.0 * math.pi * math.sqrt(a ** 3 / mu)


def propagate_twobody_dense(r0: np.ndarray, v0: np.ndarray, mu: float, t_final_s: float):
    def rhs(_t, y):
        r = y[:3]
        v = y[3:]
        rn = norm(r)
        return np.r_[v, -mu * r / rn**3]

    sol = solve_ivp(
        rhs,
        (0.0, float(t_final_s)),
        np.r_[r0, v0],
        method="DOP853",
        dense_output=True,
        rtol=1e-11,
        atol=[1e-3, 1e-3, 1e-3, 1e-9, 1e-9, 1e-9],
        max_step=120.0,
    )
    if not sol.success:
        raise RuntimeError(sol.message)
    return sol


def ejection_solution_at_state(r: np.ndarray, v: np.ndarray, mu: float, vinf_raw: np.ndarray) -> dict[str, Any]:
    rp = norm(r)
    vinf_mag = norm(vinf_raw)
    vinf_hat = unit(vinf_raw, "vinf")
    rhat = unit(r, "r")

    e = 1.0 + rp * vinf_mag * vinf_mag / mu
    c = 1.0 / e
    s = math.sqrt(max(0.0, 1.0 - c * c))
    vp = math.sqrt(vinf_mag * vinf_mag + 2.0 * mu / rp)

    raw_that = vinf_hat + c * rhat
    tangential = raw_that - float(np.dot(raw_that, rhat)) * rhat
    if norm(tangential) <= 1e-12:
        raise RuntimeError("degenerate tangential direction")

    that = unit(tangential, "that")
    predicted_vinf_hat = unit(-c * rhat + s * that, "predicted_vinf_hat")
    v_post = vp * that
    dv_raw = v_post - v

    T, N, B = tnb_basis_principia_like(r, v)
    dvt = float(np.dot(dv_raw, T))
    dvn = float(np.dot(dv_raw, N))
    dvb = float(np.dot(dv_raw, B))
    dvnrm = norm([dvt, dvn, dvb])
    oop = norm([dvn, dvb])

    return {
        "phase_angle_deg": angle_deg(predicted_vinf_hat, vinf_hat),
        "dvt_m_s": dvt,
        "dvn_m_s": dvn,
        "dvb_m_s": dvb,
        "dv_norm_m_s": dvnrm,
        "out_of_plane_abs_m_s": oop,
        "out_of_plane_fraction": oop / max(dvnrm, 1.0),
        "rmag_km": rp / 1000.0,
        "v_pre_norm_m_s": norm(v),
        "v_post_norm_m_s": norm(v_post),
        "dv_raw_m_s": dv_raw.tolist(),
    }


def score_seed(s: dict[str, Any]) -> float:
    return (
        abs(float(s["phase_angle_deg"]))
        + 0.0001 * float(s["dv_norm_m_s"])
        + 0.01 * float(s["out_of_plane_abs_m_s"])
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--snapshot-json", type=Path, required=True)
    ap.add_argument("--body-catalog", type=Path, required=True)
    ap.add_argument("--anchor-glob", action="append", required=True)
    ap.add_argument("--leg", type=int, default=1)
    ap.add_argument("--dep-body", default=None)
    ap.add_argument("--arr-body", default=None)

    ap.add_argument("--burn-dt-min-s", type=float, default=0.0)
    ap.add_argument("--burn-dt-max-s", type=float, default=None)
    ap.add_argument("--burn-grid", type=int, default=721)

    ap.add_argument("--dv-min-m-s", type=float, default=500.0)
    ap.add_argument("--dv-max-m-s", type=float, default=4200.0)
    ap.add_argument("--max-normal-abs-m-s", type=float, default=350.0)
    ap.add_argument("--max-binormal-abs-m-s", type=float, default=350.0)
    ap.add_argument("--max-out-of-plane-fraction", type=float, default=0.30)
    ap.add_argument("--max-phase-deg", type=float, default=25.0)
    ap.add_argument("--require-positive-tangent", action="store_true", default=True)
    ap.add_argument("--allow-negative-tangent", dest="require_positive_tangent", action="store_false")

    ap.add_argument("--top-n", type=int, default=100)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    vessel = load_snapshot_vessel(args.snapshot_json)
    state_t = float(vessel.get("t_spice_s", vessel.get("t_game_s")))
    rel_r = np.asarray(vessel["rel_r_raw_m"], dtype=float)
    rel_v = np.asarray(vessel["rel_v_raw_m_s"], dtype=float)
    dep_body_state = sanitize_body(args.dep_body or vessel.get("nav_body") or vessel.get("reference_body") or "KERBIN")

    bodies = load_body_catalog(args.body_catalog)
    if dep_body_state not in bodies:
        raise SystemExit(f"{dep_body_state} not found in body catalog")
    mu = float(bodies[dep_body_state]["mu_m3_s2"])

    period = two_body_period_s(rel_r, rel_v, mu)
    burn_dt_max = args.burn_dt_max_s
    if burn_dt_max is None:
        burn_dt_max = min(21600.0, max(7200.0, 3.0 * period if period else 21600.0))

    sol = propagate_twobody_dense(rel_r, rel_v, mu, burn_dt_max)

    anchor_paths: list[Path] = []
    for g in args.anchor_glob:
        anchor_paths.extend(Path(p) for p in glob.glob(g, recursive=True))
    anchor_paths = sorted(set(anchor_paths))

    rows: list[dict[str, Any]] = []

    print("=== RANK SNAPSHOT ANCHOR EXECUTABILITY V0 ===")
    print(f"snapshot       : {args.snapshot_json}")
    print(f"state_t        : {state_t}")
    print(f"state_r_km     : {norm(rel_r)/1000:.3f}")
    print(f"state_v_m_s    : {norm(rel_v):.3f}")
    print(f"dep_body_state : {dep_body_state}")
    print(f"period_s       : {period}")
    print(f"burn_dt_max    : {burn_dt_max}")
    print(f"anchors        : {len(anchor_paths)}")

    for idx, path in enumerate(anchor_paths, 1):
        try:
            anchor, leg = load_anchor(path, args.leg)
            dep = sanitize_body(args.dep_body or infer_leg_field(leg, "dep", "dep_body", default=dep_body_state))
            arr = sanitize_body(args.arr_body or infer_leg_field(leg, "arr", "arr_body", default=""))
            if dep != dep_body_state:
                # Route does not depart from current body.
                continue
            if args.arr_body and arr != sanitize_body(args.arr_body):
                continue

            t_arr = float(infer_leg_field(leg, "t_arr_s", "arrival_t_game_s"))
            vinf = get_vinf_raw_m_s(leg, "dep")

            best_any = None
            best_gate = None
            n_ok = 0
            n_gate = 0

            for burn_dt in np.linspace(args.burn_dt_min_s, burn_dt_max, args.burn_grid):
                y = np.asarray(sol.sol(float(burn_dt)), dtype=float)
                try:
                    e = ejection_solution_at_state(y[:3], y[3:], mu, vinf)
                except Exception:
                    continue

                n_ok += 1
                gate = (
                    abs(e["phase_angle_deg"]) <= args.max_phase_deg
                    and args.dv_min_m_s <= e["dv_norm_m_s"] <= args.dv_max_m_s
                    and abs(e["dvn_m_s"]) <= args.max_normal_abs_m_s
                    and abs(e["dvb_m_s"]) <= args.max_binormal_abs_m_s
                    and e["out_of_plane_fraction"] <= args.max_out_of_plane_fraction
                    and (not args.require_positive_tangent or e["dvt_m_s"] > 0)
                )
                e = dict(e)
                e.update({
                    "burn_dt_s": float(burn_dt),
                    "burn_abs_s": state_t + float(burn_dt),
                    "phase_score": score_seed(e),
                })

                if best_any is None or e["phase_score"] < best_any["phase_score"]:
                    best_any = e

                if gate:
                    n_gate += 1
                    if best_gate is None or e["phase_score"] < best_gate["phase_score"]:
                        best_gate = e

            cm = anchor.get("candidate_metrics") or {}
            row = {
                "anchor_path": str(path),
                "anchor_parent": str(path.parent),
                "row_index0": anchor.get("row_index0", cm.get("row_index0")),
                "sequence": sequence_string(anchor, leg),
                "dep_body": dep,
                "arr_body": arr,
                "t_arr_s": t_arr,
                "tof_remaining_days": (t_arr - state_t) / DAY_S,
                "vinf_dep_norm_m_s": norm(vinf),
                "raw_sum_km_s": safe_num(cm.get("raw_sum_km_s")),
                "cost": safe_num(cm.get("cost")),
                "n_phase_ok": n_ok,
                "n_phase_gate_ok": n_gate,
                "pass_gate": best_gate is not None,
                "reject_reason": "" if best_gate is not None else "no_seed_passed_gates",
            }

            src = best_gate or best_any
            if src:
                prefix = "best_gate" if best_gate else "best_any"
                for k, v in src.items():
                    row[f"{prefix}_{k}"] = v
                # unified sortable fields
                row.update({
                    "rank_score": (
                        0.0 if best_gate else 1e9
                    ) + float(src["phase_score"]) + 0.001 * float(row.get("raw_sum_km_s") or 0.0),
                    "burn_dt_s": src["burn_dt_s"],
                    "dvt_m_s": src["dvt_m_s"],
                    "dvn_m_s": src["dvn_m_s"],
                    "dvb_m_s": src["dvb_m_s"],
                    "dv_norm_m_s": src["dv_norm_m_s"],
                    "out_of_plane_fraction": src["out_of_plane_fraction"],
                    "phase_angle_deg": src["phase_angle_deg"],
                    "phase_score": src["phase_score"],
                })
            else:
                row.update({
                    "rank_score": math.inf,
                    "phase_score": math.inf,
                })

            rows.append(row)

        except Exception as exc:
            rows.append({
                "anchor_path": str(path),
                "error": str(exc),
                "pass_gate": False,
                "rank_score": math.inf,
                "reject_reason": "exception",
            })

        if idx % 25 == 0:
            print(f"[{idx}/{len(anchor_paths)}] processed")

    rows_ok = [r for r in rows if math.isfinite(float(r.get("rank_score", math.inf)))]
    rows_ok.sort(key=lambda r: float(r["rank_score"]))

    print("\n=== TOP EXECUTABLE ANCHORS ===")
    top = rows_ok[:args.top_n]
    for i, r in enumerate(top[:30], 1):
        print(
            f"{i:2d} pass={str(r.get('pass_gate')):5s} "
            f"score={float(r.get('rank_score', math.inf)):10.3f} "
            f"row={r.get('row_index0')} "
            f"seq={str(r.get('sequence'))[:36]:36s} "
            f"arr={r.get('arr_body'):6s} "
            f"tof={float(r.get('tof_remaining_days', math.nan)):8.2f}d "
            f"dv={float(r.get('dv_norm_m_s', math.nan)):8.1f} "
            f"T={float(r.get('dvt_m_s', math.nan)):8.1f} "
            f"N={float(r.get('dvn_m_s', math.nan)):8.1f} "
            f"B={float(r.get('dvb_m_s', math.nan)):8.1f} "
            f"oop={float(r.get('out_of_plane_fraction', math.nan)):6.3f} "
            f"phase={float(r.get('phase_angle_deg', math.nan)):7.3f} "
            f"anchor={r.get('anchor_parent')}"
        )

    result = {
        "schema": "rank_snapshot_anchor_executability_v0",
        "snapshot_json": str(args.snapshot_json),
        "body_catalog": str(args.body_catalog),
        "state_t_game_s": state_t,
        "state_rel_r_raw_m": rel_r.tolist(),
        "state_rel_v_raw_m_s": rel_v.tolist(),
        "dep_body_state": dep_body_state,
        "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "n_anchors": len(anchor_paths),
        "n_rows": len(rows),
        "n_pass_gate": len([r for r in rows if r.get("pass_gate")]),
        "top": top,
    }

    result_json = args.output_dir / "snapshot_anchor_executability_rank.json"
    rows_csv = args.output_dir / "snapshot_anchor_executability_rank.csv"
    result_json.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")

    flat = [flatten(r) for r in rows]
    if flat:
        fields = sorted({k for r in flat for k in r.keys()})
        with rows_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(flat)

    print(f"[OK] wrote {result_json}")
    print(f"[OK] wrote {rows_csv}")
    return 0


def safe_num(x):
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


if __name__ == "__main__":
    raise SystemExit(main())
