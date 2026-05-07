#!/usr/bin/env python3
"""
select_best_snapshot_by_body.py

Constrói um snapshot composto escolhendo, corpo a corpo, o estado inicial vindo
do snapshot_dt* que teve melhor desempenho no Level A.

Uso típico
----------
python select_best_snapshot_by_body.py \
  --snapshot-glob "data/jnsq_gate0/snapshot_dt*.json" \
  --residual-dir-glob "data/jnsq_gate0/level_a_5d_v3_snapshot_dt*" \
  --score max \
  --output-snapshot data/jnsq_gate0/snapshot_composite_best_5d.json \
  --output-csv data/jnsq_gate0/best_dt_by_body.csv

Depois valide:
python rebound_level_a_cache.py \
  --input-json data/jnsq_gate0/snapshot_composite_best_5d.json \
  --central-body Sun \
  --ksp-csv data/jnsq_gate0/ksp_5d_v3_raw/states.csv \
  --integrator ias15 \
  --ias15-epsilon 1e-11 \
  --output-dir data/jnsq_gate0/level_a_5d_composite_best \
  --write-residual-samples

Notas
-----
- O script tenta preservar o formato original do JSON.
- Suporta snapshots com estados em: bodies, body_states, states, ephemerides.
- Se o formato for desconhecido, falha explicitamente.
"""

from __future__ import annotations

import argparse
import copy
import csv
import glob
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


DT_RE = re.compile(r"(?:^|[_\-/])dt(?P<dt>\d+(?:\.\d+)?)(?:[_\-.\/]|$)", re.IGNORECASE)

BODY_KEY_CANDIDATES = ["body", "Body", "name", "Name", "corpo", "Corpo"]

MAX_KM_CANDIDATES = [
    "max_km", "Max km", "max_pos_km", "max_position_km", "max_pos_err_km",
    "position_max_km", "err_pos_max_km", "Erro Pos Máx (km)", "Erro Pos Max (km)",
    "before_max_km", "Before max km", "max_position_error_km",
]
RMS_KM_CANDIDATES = [
    "rms_km", "RMS km", "rms_pos_km", "rms_position_km", "rms_pos_err_km",
    "position_rms_km", "err_pos_rms_km", "RMS pos km", "before_rms_km",
    "Before RMS km", "rms_position_error_km",
]
FINAL_KM_CANDIDATES = [
    "final_km", "Final km", "final_pos_km", "final_position_km", "final_pos_err_km",
    "Erro Final (km)", "before_final_km", "final_position_error_km",
]
MAX_M_CANDIDATES = ["max_m", "max_pos_m", "max_position_m", "max_pos_err_m", "position_max_m"]
RMS_M_CANDIDATES = ["rms_m", "rms_pos_m", "rms_position_m", "rms_pos_err_m", "position_rms_m"]
FINAL_M_CANDIDATES = ["final_m", "final_pos_m", "final_position_m", "final_pos_err_m"]


def parse_dt_from_path(path: str | Path) -> Optional[float]:
    text = str(path)
    matches = list(DT_RE.finditer(text))
    if matches:
        return float(matches[-1].group("dt"))
    m = re.search(r"dt(?P<dt>\d+(?:\.\d+)?)", text, re.IGNORECASE)
    return float(m.group("dt")) if m else None


def first_existing_key(row: Dict[str, Any], candidates: Iterable[str]) -> Optional[str]:
    for c in candidates:
        if c in row:
            return c

    def norm(s: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", s.lower())

    nmap = {norm(k): k for k in row.keys()}
    for c in candidates:
        nc = norm(c)
        if nc in nmap:
            return nmap[nc]

    for k in row.keys():
        nk = norm(k)
        for c in candidates:
            nc = norm(c)
            if nc and (nc in nk or nk in nc):
                return k
    return None


def parse_float(value: Any) -> float:
    if value is None:
        return float("nan")
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip()
    if not s:
        return float("nan")
    return float(s.replace(",", "."))


def detect_metric_keys(row: Dict[str, Any]) -> Dict[str, Optional[Tuple[str, float]]]:
    body_key = first_existing_key(row, BODY_KEY_CANDIDATES)

    max_key = first_existing_key(row, MAX_KM_CANDIDATES)
    rms_key = first_existing_key(row, RMS_KM_CANDIDATES)
    final_key = first_existing_key(row, FINAL_KM_CANDIDATES)

    max_scale = rms_scale = final_scale = 1.0

    if max_key is None:
        max_key = first_existing_key(row, MAX_M_CANDIDATES)
        max_scale = 1.0 / 1000.0
    if rms_key is None:
        rms_key = first_existing_key(row, RMS_M_CANDIDATES)
        rms_scale = 1.0 / 1000.0
    if final_key is None:
        final_key = first_existing_key(row, FINAL_M_CANDIDATES)
        final_scale = 1.0 / 1000.0

    return {
        "body": (body_key, 1.0) if body_key else None,
        "max": (max_key, max_scale) if max_key else None,
        "rms": (rms_key, rms_scale) if rms_key else None,
        "final": (final_key, final_scale) if final_key else None,
    }


def read_residuals_csv(path: Path) -> Dict[str, Dict[str, float]]:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames

    if not rows:
        raise ValueError(f"CSV sem linhas: {path}")

    keys = detect_metric_keys(rows[0])
    if keys["body"] is None:
        raise ValueError(f"Não consegui detectar coluna de corpo em: {path}; colunas={fieldnames}")

    body_key = keys["body"][0]  # type: ignore[index]
    out: Dict[str, Dict[str, float]] = {}

    for row in rows:
        body = str(row.get(body_key, "")).strip()
        if not body:
            continue
        metrics: Dict[str, float] = {}
        for metric in ("max", "rms", "final"):
            spec = keys[metric]
            if spec is None:
                metrics[metric] = float("nan")
                continue
            key, scale = spec
            try:
                metrics[metric] = parse_float(row.get(key)) * scale
            except Exception:
                metrics[metric] = float("nan")
        out[body] = metrics
    return out


def find_residual_files(pattern: str) -> List[Path]:
    paths: List[Path] = []
    for hit in glob.glob(pattern):
        p = Path(hit)
        if p.is_dir():
            cand = p / "residuals_by_body.csv"
            if cand.exists():
                paths.append(cand)
        elif p.name == "residuals_by_body.csv":
            paths.append(p)
        elif p.is_file() and p.suffix.lower() == ".csv":
            paths.append(p)
    paths = sorted(set(paths))
    if not paths:
        raise FileNotFoundError(f"Nenhum residuals_by_body.csv encontrado com: {pattern}")
    return paths


def find_snapshots(pattern: str) -> Dict[float, Path]:
    out: Dict[float, Path] = {}
    for hit in glob.glob(pattern):
        p = Path(hit)
        if not p.is_file():
            continue
        dt = parse_dt_from_path(p)
        if dt is None:
            continue
        out[dt] = p
    if not out:
        raise FileNotFoundError(f"Nenhum snapshot com dt detectável encontrado com: {pattern}")
    return dict(sorted(out.items()))


def find_state_container(payload: Dict[str, Any]) -> Tuple[Dict[str, Any], str]:
    candidates = ["bodies", "body_states", "states", "ephemerides"]
    for key in candidates:
        obj = payload.get(key)
        if isinstance(obj, dict) and obj and all(isinstance(k, str) for k in obj.keys()):
            return obj, key

    for top_key, top_val in payload.items():
        if isinstance(top_val, dict):
            for key in candidates:
                obj = top_val.get(key)
                if isinstance(obj, dict) and obj:
                    return obj, f"{top_key}.{key}"

    raise ValueError(
        "Não consegui encontrar container de estados no snapshot. "
        "Esperava payload['bodies'], ['body_states'], ['states'] ou ['ephemerides']."
    )


def get_nested_container_ref(payload: Dict[str, Any], path: str) -> Dict[str, Any]:
    cur: Any = payload
    for part in path.split("."):
        cur = cur[part]
    if not isinstance(cur, dict):
        raise ValueError(f"Container não é dict em path {path}")
    return cur


def score_metrics(metrics: Dict[str, float], score_mode: str) -> float:
    max_km = metrics.get("max", float("nan"))
    rms_km = metrics.get("rms", float("nan"))
    final_km = metrics.get("final", float("nan"))
    if score_mode == "max":
        return max_km
    if score_mode == "rms":
        return rms_km
    if score_mode == "final":
        return final_km
    if score_mode == "combo":
        vals = [v for v in (max_km, rms_km, final_km) if math.isfinite(v)]
        fallback = max(vals) if vals else float("inf")
        max_km = max_km if math.isfinite(max_km) else fallback
        rms_km = rms_km if math.isfinite(rms_km) else fallback
        final_km = final_km if math.isfinite(final_km) else fallback
        return 0.60 * max_km + 0.30 * rms_km + 0.10 * final_km
    raise ValueError(f"score desconhecido: {score_mode}")


def load_json(path: Path) -> Dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError(f"Snapshot JSON não é objeto/dict: {path}")
    return obj


def fmt_float(x: Any) -> str:
    try:
        f = float(x)
        if math.isfinite(f):
            return f"{f:.9g}"
        return ""
    except Exception:
        return ""


def main() -> None:
    ap = argparse.ArgumentParser(description="Seleciona melhor snapshot_dt por corpo e monta snapshot composto.")
    ap.add_argument("--snapshot-glob", required=True, help='Ex: "data/jnsq_gate0/snapshot_dt*.json"')
    ap.add_argument("--residual-dir-glob", required=True, help='Ex: "data/jnsq_gate0/level_a_5d_v3_snapshot_dt*"')
    ap.add_argument("--score", choices=["max", "rms", "final", "combo"], default="max")
    ap.add_argument("--output-snapshot", required=True)
    ap.add_argument("--output-csv", required=True)
    ap.add_argument("--body", action="append", default=[], help="Limita a corpos específicos. Pode repetir.")
    ap.add_argument("--exclude-body", action="append", default=[], help="Corpos a nunca substituir. Ex: --exclude-body Sun")
    ap.add_argument("--base-dt", type=float, default=None, help="dt usado como base estrutural. Default: menor dt comum.")
    ap.add_argument("--keep-base-for-missing", action="store_true", help="Mantém base se corpo não tiver residual.")
    ap.add_argument("--write-report-json", default=None)
    args = ap.parse_args()

    snapshot_by_dt = find_snapshots(args.snapshot_glob)
    residual_files = find_residual_files(args.residual_dir_glob)

    residuals_by_dt: Dict[float, Dict[str, Dict[str, float]]] = {}
    for rf in residual_files:
        dt = parse_dt_from_path(rf)
        if dt is None:
            print(f"[WARN] ignorando residual sem dt detectável: {rf}")
            continue
        residuals_by_dt[dt] = read_residuals_csv(rf)

    if not residuals_by_dt:
        raise FileNotFoundError("Nenhum residual com dt detectável foi carregado.")

    missing_snapshots = sorted(set(residuals_by_dt) - set(snapshot_by_dt))
    if missing_snapshots:
        print(f"[WARN] Há residuals sem snapshot correspondente: {missing_snapshots}")

    common_dts = sorted(set(snapshot_by_dt) & set(residuals_by_dt))
    if not common_dts:
        raise ValueError("Nenhum dt comum entre snapshots e residuals.")

    base_dt = args.base_dt if args.base_dt is not None else min(common_dts)
    if base_dt not in snapshot_by_dt:
        raise ValueError(f"--base-dt {base_dt} não existe nos snapshots disponíveis: {sorted(snapshot_by_dt)}")

    snapshots_payload: Dict[float, Dict[str, Any]] = {}
    containers: Dict[float, Dict[str, Any]] = {}
    for dt in common_dts:
        payload = load_json(snapshot_by_dt[dt])
        container, _ = find_state_container(payload)
        snapshots_payload[dt] = payload
        containers[dt] = container

    base_payload = copy.deepcopy(load_json(snapshot_by_dt[base_dt]))
    _, base_container_path = find_state_container(base_payload)
    base_container = get_nested_container_ref(base_payload, base_container_path)

    bodies = set(base_container.keys())
    for dt in common_dts:
        bodies.update(residuals_by_dt[dt].keys())
    bodies_sorted = sorted(bodies)

    if args.body:
        wanted = set(args.body)
        bodies_sorted = [b for b in bodies_sorted if b in wanted]

    excluded = set(args.exclude_body)
    selected_rows: List[Dict[str, Any]] = []
    selection_report: Dict[str, Any] = {
        "tool": "select_best_snapshot_by_body.py",
        "score_mode": args.score,
        "base_dt": base_dt,
        "snapshot_glob": args.snapshot_glob,
        "residual_dir_glob": args.residual_dir_glob,
        "common_dts": common_dts,
        "selected": {},
        "warnings": [],
    }

    replaced = kept_base = missing = 0

    for body in bodies_sorted:
        if body in excluded:
            if body in base_container:
                kept_base += 1
                selected_rows.append({
                    "body": body, "selected_dt": base_dt, "score_mode": args.score,
                    "score_value": "", "max_km": "", "rms_km": "", "final_km": "",
                    "snapshot_path": str(snapshot_by_dt[base_dt]), "status": "excluded_keep_base",
                })
            continue

        candidates: List[Tuple[float, float, Dict[str, float]]] = []
        for dt in common_dts:
            metrics = residuals_by_dt[dt].get(body)
            if not metrics:
                continue
            score_value = score_metrics(metrics, args.score)
            if math.isfinite(score_value):
                candidates.append((score_value, dt, metrics))

        if not candidates:
            if args.keep_base_for_missing and body in base_container:
                kept_base += 1
                selected_rows.append({
                    "body": body, "selected_dt": base_dt, "score_mode": args.score,
                    "score_value": "", "max_km": "", "rms_km": "", "final_km": "",
                    "snapshot_path": str(snapshot_by_dt[base_dt]), "status": "missing_residual_keep_base",
                })
                selection_report["warnings"].append(f"{body}: sem residual; mantido do base.")
                continue
            missing += 1
            selection_report["warnings"].append(f"{body}: sem residual; não selecionado.")
            continue

        candidates.sort(key=lambda x: (x[0], x[1]))
        best_score, best_dt, best_metrics = candidates[0]
        source_container = containers[best_dt]
        if body not in source_container:
            msg = f"{body}: selecionado dt={best_dt}, mas corpo ausente no snapshot {snapshot_by_dt[best_dt]}"
            selection_report["warnings"].append(msg)
            if args.keep_base_for_missing and body in base_container:
                kept_base += 1
                continue
            missing += 1
            continue

        base_container[body] = copy.deepcopy(source_container[body])
        replaced += 1

        row = {
            "body": body,
            "selected_dt": best_dt,
            "score_mode": args.score,
            "score_value": best_score,
            "max_km": best_metrics.get("max", float("nan")),
            "rms_km": best_metrics.get("rms", float("nan")),
            "final_km": best_metrics.get("final", float("nan")),
            "snapshot_path": str(snapshot_by_dt[best_dt]),
            "status": "selected",
        }
        selected_rows.append(row)
        selection_report["selected"][body] = row

    base_payload["composite_metadata"] = {
        "tool": "select_best_snapshot_by_body.py",
        "score_mode": args.score,
        "base_dt": base_dt,
        "base_snapshot": str(snapshot_by_dt[base_dt]),
        "common_dts": common_dts,
        "replaced_bodies": replaced,
        "kept_base_bodies": kept_base,
        "missing_bodies": missing,
        "body_selection_csv": str(args.output_csv),
        "note": "Cada corpo recebeu o estado do snapshot_dt que minimizou o score no Level A; validar contra holdout antes de produção.",
    }

    out_snapshot = Path(args.output_snapshot)
    out_snapshot.parent.mkdir(parents=True, exist_ok=True)
    with out_snapshot.open("w", encoding="utf-8") as f:
        json.dump(base_payload, f, indent=2, ensure_ascii=False)

    out_csv = Path(args.output_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    selected_rows.sort(key=lambda r: str(r["body"]))
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["body", "selected_dt", "score_mode", "score_value", "max_km", "rms_km", "final_km", "snapshot_path", "status"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in selected_rows:
            w.writerow(row)

    if args.write_report_json:
        rp = Path(args.write_report_json)
        rp.parent.mkdir(parents=True, exist_ok=True)
        with rp.open("w", encoding="utf-8") as f:
            json.dump(selection_report, f, indent=2, ensure_ascii=False)

    print("\n=== SELECT BEST SNAPSHOT BY BODY ===")
    print(f"score:           {args.score}")
    print(f"common dts:      {common_dts}")
    print(f"base dt:         {base_dt}")
    print(f"replaced:        {replaced}")
    print(f"kept base:       {kept_base}")
    print(f"missing:         {missing}")
    print(f"output snapshot: {out_snapshot}")
    print(f"output csv:      {out_csv}")

    selected_only = [r for r in selected_rows if r["status"] == "selected"]
    selected_only.sort(key=lambda r: float(r["score_value"]), reverse=True)
    print("\nPiores scores selecionados:")
    for row in selected_only[:20]:
        print(
            f'{row["body"]:<12} dt={fmt_float(row["selected_dt"]):>8} '
            f'score={fmt_float(row["score_value"]):>12} km '
            f'max={fmt_float(row["max_km"]):>12} rms={fmt_float(row["rms_km"]):>12} final={fmt_float(row["final_km"]):>12}'
        )

    if selection_report["warnings"]:
        print("\nWARNINGS:")
        for w in selection_report["warnings"][:30]:
            print(" -", w)


if __name__ == "__main__":
    main()
