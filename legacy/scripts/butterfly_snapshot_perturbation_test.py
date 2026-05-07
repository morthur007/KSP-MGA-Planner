#!/usr/bin/env python3
"""
butterfly_snapshot_perturbation_test.py

Teste de sensibilidade / efeito borboleta para o pipeline Principia -> REBOUND.

O script NÃO reimplementa a física. Ele usa o seu snapshot JSON e, opcionalmente,
chama o seu rebound_level_a_cache.py para propagar cada snapshot perturbado.
Depois compara o REBOUND perturbado contra um CSV REBOUND nominal.

Fluxo recomendado:
  1. Identifica estados x,y,z,vx,vy,vz no snapshot.
  2. Aplica perturbações controladas no estado inicial de corpos alvo.
  3. Salva snapshots perturbados.
  4. Opcionalmente roda rebound_level_a_cache.py para cada snapshot.
  5. Compara perturbed/rebound_states.csv vs nominal/rebound_states.csv.

Exemplo:
python butterfly_snapshot_perturbation_test.py \
  --snapshot data/final_clean_120d/snapshot_fit_urlum_family.json \
  --nominal-reb-csv data/final_clean_120d/level_a_fit_urlum_family/rebound_states.csv \
  --ksp-csv data/final_clean_120d/states.csv \
  --central-body Sun \
  --targets Vall:Jool Tylo:Jool Pol:Jool Laythe:Jool Crokslev:Sun \
  --position-perturb-m 0.001,0.01,1.0 \
  --velocity-perturb-m-s 0.000001,0.001 \
  --directions radial,tangential,normal \
  --output-dir data/final_clean_120d/butterfly_test \
  --run-level-a
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np


STATE_KEYS = ["x_m", "y_m", "z_m", "vx_m_s", "vy_m_s", "vz_m_s"]
ALT_STATE_KEYS = ["x", "y", "z", "vx", "vy", "vz"]
POS_KEYS = ["x_m", "y_m", "z_m"]
VEL_KEYS = ["vx_m_s", "vy_m_s", "vz_m_s"]


@dataclass
class StateRef:
    body: str
    container: Any
    kind: str
    path: str

    def get_state(self) -> np.ndarray:
        c = self.container
        if self.kind == "keys_m":
            return np.array([float(c[k]) for k in STATE_KEYS], dtype=float)
        if self.kind == "keys_plain":
            return np.array([float(c[k]) for k in ALT_STATE_KEYS], dtype=float)
        if self.kind == "state_list":
            return np.array([float(x) for x in c["state"][:6]], dtype=float)
        if self.kind == "state_m_list":
            return np.array([float(x) for x in c["state_m"][:6]], dtype=float)
        if self.kind == "state_vector":
            return np.array([float(x) for x in c["state_vector"][:6]], dtype=float)
        if self.kind == "position_velocity":
            p = c.get("position") or c.get("position_m") or c.get("r") or c.get("r_m")
            v = c.get("velocity") or c.get("velocity_m_s") or c.get("v") or c.get("v_m_s")
            return np.array([float(*[])], dtype=float)  # unreachable placeholder
        if self.kind == "ksp_states_list":
            # Pega o primeiro snapshot [0] e os índices [1:7] (x, y, z, vx, vy, vz)
            return np.array([float(x) for x in c["states"][0][1:7]], dtype=float)
        raise RuntimeError(f"Unknown StateRef kind {self.kind}")

    def set_state(self, state: np.ndarray) -> None:
        c = self.container
        vals = [float(x) for x in state]
        if self.kind == "keys_m":
            for k, v in zip(STATE_KEYS, vals):
                c[k] = v
            return
        if self.kind == "keys_plain":
            for k, v in zip(ALT_STATE_KEYS, vals):
                c[k] = v
            return
        if self.kind == "state_list":
            c["state"][:6] = vals
            return
        if self.kind == "state_m_list":
            c["state_m"][:6] = vals
            return
        if self.kind == "state_vector":
            c["state_vector"][:6] = vals
            return
        if self.kind == "ksp_states_list":
            # Sobrescreve apenas os valores de posição e velocidade no primeiro snapshot
            c["states"][0][1:7] = vals
            return
        raise RuntimeError(f"Unknown StateRef kind {self.kind}")


def looks_like_state_dict(d: Dict[str, Any]) -> Optional[str]:
    if all(k in d for k in STATE_KEYS):
        return "keys_m"
    if all(k in d for k in ALT_STATE_KEYS):
        return "keys_plain"
    for key in ("state", "state_m", "state_vector", "states"): # Adicione "states" aqui
        if key in d and isinstance(d[key], list) and len(d[key]) >= 1: # Verifica se há pelo menos um snapshot
             # Se for a chave "states", precisamos verificar se o conteúdo é uma lista de listas
             if key == "states" and isinstance(d[key][0], list) and len(d[key][0]) >= 7:
                return "ksp_states_list"
    # We intentionally do not support position+velocity separate here because schemas vary.
    return None


def discover_states(obj: Any, path: str = "root", parent_key: Optional[str] = None, out: Optional[Dict[str, StateRef]] = None) -> Dict[str, StateRef]:
    if out is None:
        out = {}
    if isinstance(obj, dict):
        body_name = None
        for nk in ("name", "body", "body_name"):
            if nk in obj and isinstance(obj[nk], str):
                body_name = obj[nk]
                break
        if body_name is None and parent_key is not None:
            body_name = parent_key

        kind = looks_like_state_dict(obj)
        if kind is not None and body_name:
            out[body_name] = StateRef(body=body_name, container=obj, kind=kind, path=path)

        for k, v in obj.items():
            discover_states(v, f"{path}.{k}", str(k), out)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            discover_states(v, f"{path}[{i}]", parent_key, out)
    return out


def unit(v: np.ndarray, fallback: Optional[np.ndarray] = None) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n > 0:
        return v / n
    if fallback is not None:
        return unit(fallback)
    return np.array([1.0, 0.0, 0.0])


def rtn_basis(child_state: np.ndarray, parent_state: Optional[np.ndarray]) -> Dict[str, np.ndarray]:
    if parent_state is None:
        rel_r = child_state[:3]
        rel_v = child_state[3:]
    else:
        rel_r = child_state[:3] - parent_state[:3]
        rel_v = child_state[3:] - parent_state[3:]

    rhat = unit(rel_r)
    h = np.cross(rel_r, rel_v)
    nhat = unit(h, fallback=np.array([0.0, 0.0, 1.0]))
    that = unit(np.cross(nhat, rhat), fallback=rel_v)
    # Re-orthogonalize normal in case fallback was used.
    nhat = unit(np.cross(rhat, that), fallback=nhat)
    return {
        "radial": rhat,
        "tangential": that,
        "transversal": that,
        "normal": nhat,
        "x": np.array([1.0, 0.0, 0.0]),
        "y": np.array([0.0, 1.0, 0.0]),
        "z": np.array([0.0, 0.0, 1.0]),
    }


def parse_csv_list_floats(s: str) -> List[float]:
    if not s:
        return []
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def parse_csv_list_str(s: str) -> List[str]:
    if not s:
        return []
    return [x.strip() for x in s.split(",") if x.strip()]


def read_states_csv(path: Path) -> Dict[str, List[Tuple[float, np.ndarray]]]:
    data: Dict[str, List[Tuple[float, np.ndarray]]] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        fields = set(r.fieldnames or [])
        body_key = "body" if "body" in fields else "name"
        et_key = "et_seconds" if "et_seconds" in fields else "t"
        for row in r:
            b = row[body_key]
            et = float(row[et_key])
            # support both x_m and x style
            if all(k in row for k in STATE_KEYS):
                st = np.array([float(row[k]) for k in STATE_KEYS], dtype=float)
            elif all(k in row for k in ALT_STATE_KEYS):
                st = np.array([float(row[k]) for k in ALT_STATE_KEYS], dtype=float)
            else:
                raise ValueError(f"CSV {path} lacks state columns")
            data.setdefault(b, []).append((et, st))
    for b in data:
        data[b].sort(key=lambda x: x[0])
    return data


def compare_body_series(nominal: List[Tuple[float, np.ndarray]], perturbed: List[Tuple[float, np.ndarray]]) -> Dict[str, float]:
    # assume same epochs/order; use min length and warn by metric count only
    n = min(len(nominal), len(perturbed))
    if n == 0:
        return {"n": 0, "max_m": math.nan, "rms_m": math.nan, "final_m": math.nan}
    errs = []
    for i in range(n):
        e = float(np.linalg.norm(perturbed[i][1][:3] - nominal[i][1][:3]))
        errs.append(e)
    arr = np.array(errs, dtype=float)
    return {
        "n": n,
        "max_m": float(np.max(arr)),
        "rms_m": float(math.sqrt(np.mean(arr * arr))),
        "final_m": float(arr[-1]),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="REBOUND butterfly/sensitivity test via perturbed snapshots")
    ap.add_argument("--snapshot", required=True, help="Base snapshot JSON")
    ap.add_argument("--nominal-reb-csv", required=True, help="Nominal REBOUND states CSV for comparison")
    ap.add_argument("--ksp-csv", help="KSP states CSV passed to rebound_level_a_cache.py if --run-level-a")
    ap.add_argument("--central-body", default="Sun")
    ap.add_argument("--targets", nargs="+", required=True, help="Targets as Body:Parent or Body. Example: Vall:Jool Crokslev:Sun")
    ap.add_argument("--position-perturb-m", default="0.001,0.01,1.0", help="Comma-list of position perturbations in m")
    ap.add_argument("--velocity-perturb-m-s", default="", help="Comma-list of velocity perturbations in m/s")
    ap.add_argument("--directions", default="radial,tangential,normal", help="Comma-list: radial,tangential,normal,x,y,z")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--run-level-a", action="store_true", help="Run rebound_level_a_cache.py for each perturbed snapshot")
    ap.add_argument("--level-a-script", default="rebound_level_a_cache.py")
    ap.add_argument("--integrator", default="ias15")
    ap.add_argument("--ias15-epsilon", default=None)
    ap.add_argument("--max-cases", type=int, default=0, help="Limit number of generated cases for quick tests")
    args = ap.parse_args()

    snapshot_path = Path(args.snapshot)
    outdir = Path(args.output_dir)
    snaps_dir = outdir / "snapshots"
    runs_dir = outdir / "runs"
    outdir.mkdir(parents=True, exist_ok=True)
    snaps_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)

    base = json.loads(snapshot_path.read_text(encoding="utf-8"))
    refs = discover_states(base)
    if not refs:
        print("[ERROR] Could not discover any state vectors in snapshot.", file=sys.stderr)
        print("Supported schemas: dict with x_m/y_m/z_m/vx_m_s/vy_m_s/vz_m_s, or state/state_m/state_vector length>=6.", file=sys.stderr)
        return 2

    print(f"[DISCOVER] Found {len(refs)} state vectors")
    print("[DISCOVER] Example bodies:", ", ".join(sorted(refs)[:20]))

    target_pairs: List[Tuple[str, str]] = []
    for t in args.targets:
        if ":" in t:
            body, parent = t.split(":", 1)
        else:
            body, parent = t, args.central_body
        body = body.strip(); parent = parent.strip()
        if body not in refs:
            raise SystemExit(f"Target body {body!r} not found in snapshot. Available examples: {sorted(refs)[:30]}")
        if parent and parent not in refs:
            print(f"[WARN] Parent {parent!r} for {body!r} not found; using central/inertial RTN", file=sys.stderr)
        target_pairs.append((body, parent))

    pos_deltas = parse_csv_list_floats(args.position_perturb_m)
    vel_deltas = parse_csv_list_floats(args.velocity_perturb_m_s)
    directions = parse_csv_list_str(args.directions)

    cases = []
    for body, parent in target_pairs:
        c_state = refs[body].get_state()
        p_state = refs[parent].get_state() if parent in refs else None
        basis = rtn_basis(c_state, p_state)
        for direction in directions:
            if direction not in basis:
                raise SystemExit(f"Unknown direction {direction!r}")
            vec = basis[direction]
            for delta in pos_deltas:
                cases.append((body, parent, "pos", direction, delta, vec))
            for delta in vel_deltas:
                cases.append((body, parent, "vel", direction, delta, vec))

    if args.max_cases and len(cases) > args.max_cases:
        cases = cases[: args.max_cases]

    nominal = read_states_csv(Path(args.nominal_reb_csv))
    summary_rows = []

    for idx, (body, parent, kind, direction, delta, vec) in enumerate(cases, start=1):
        case_name = f"{idx:04d}_{body}_{kind}_{direction}_{delta:g}".replace("+", "p").replace("-", "m").replace(".", "p")
        print(f"\n[CASE {idx}/{len(cases)}] {body} {kind} {direction} {delta:g}")

        pert = copy.deepcopy(base)
        pert_refs = discover_states(pert)
        st = pert_refs[body].get_state()
        if kind == "pos":
            st[:3] += float(delta) * vec
            perturb_norm_m = abs(float(delta))
            perturb_norm_v = 0.0
        else:
            st[3:] += float(delta) * vec
            perturb_norm_m = 0.0
            perturb_norm_v = abs(float(delta))
        pert_refs[body].set_state(st)

        pert_path = snaps_dir / f"{case_name}.json"
        pert_path.write_text(json.dumps(pert, indent=2, ensure_ascii=False), encoding="utf-8")
        run_dir = runs_dir / case_name

        if args.run_level_a:
            if not args.ksp_csv:
                raise SystemExit("--ksp-csv is required with --run-level-a")
            run_dir.mkdir(parents=True, exist_ok=True)
            cmd = [
                sys.executable, args.level_a_script,
                "--input-json", str(pert_path),
                "--central-body", args.central_body,
                "--ksp-csv", args.ksp_csv,
                "--integrator", args.integrator,
                "--output-dir", str(run_dir),
            ]
            if args.ias15_epsilon is not None:
                cmd += ["--ias15-epsilon", str(args.ias15_epsilon)]
            print("[RUN]", " ".join(cmd))
            subprocess.run(cmd, check=True)

        pert_csv = run_dir / "rebound_states.csv"
        if pert_csv.exists():
            perturbed = read_states_csv(pert_csv)
            if body not in nominal or body not in perturbed:
                print(f"[WARN] body {body} missing from nominal or perturbed CSV")
                continue
            metrics = compare_body_series(nominal[body], perturbed[body])
            amp_final_pos = metrics["final_m"] / perturb_norm_m if perturb_norm_m > 0 else math.nan
            amp_max_pos = metrics["max_m"] / perturb_norm_m if perturb_norm_m > 0 else math.nan
            # For velocity perturbations, report equivalent seconds amplification separately not defined.
            row = {
                "case": case_name,
                "body": body,
                "parent": parent,
                "kind": kind,
                "direction": direction,
                "delta": delta,
                "n": metrics["n"],
                "max_sep_m": metrics["max_m"],
                "rms_sep_m": metrics["rms_m"],
                "final_sep_m": metrics["final_m"],
                "amp_max_pos": amp_max_pos,
                "amp_final_pos": amp_final_pos,
                "snapshot": str(pert_path),
                "run_dir": str(run_dir),
            }
            summary_rows.append(row)
            print(f"[RESULT] max={metrics['max_m']/1000:.6g} km rms={metrics['rms_m']/1000:.6g} km final={metrics['final_m']/1000:.6g} km amp_final={amp_final_pos:.3g}")
        else:
            summary_rows.append({
                "case": case_name,
                "body": body,
                "parent": parent,
                "kind": kind,
                "direction": direction,
                "delta": delta,
                "n": "",
                "max_sep_m": "",
                "rms_sep_m": "",
                "final_sep_m": "",
                "amp_max_pos": "",
                "amp_final_pos": "",
                "snapshot": str(pert_path),
                "run_dir": str(run_dir),
            })

    summary_path = outdir / "butterfly_summary.csv"
    fieldnames = ["case", "body", "parent", "kind", "direction", "delta", "n", "max_sep_m", "rms_sep_m", "final_sep_m", "amp_max_pos", "amp_final_pos", "snapshot", "run_dir"]
    with summary_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in summary_rows:
            w.writerow(r)

    manifest = {
        "snapshot": str(snapshot_path),
        "nominal_reb_csv": args.nominal_reb_csv,
        "ksp_csv": args.ksp_csv,
        "central_body": args.central_body,
        "targets": args.targets,
        "position_perturb_m": pos_deltas,
        "velocity_perturb_m_s": vel_deltas,
        "directions": directions,
        "run_level_a": args.run_level_a,
        "summary_csv": str(summary_path),
    }
    (outdir / "butterfly_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[OK] summary: {summary_path}")
    print(f"[OK] manifest: {outdir / 'butterfly_manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
