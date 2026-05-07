#!/usr/bin/env python3
"""
pipeline_integrity_test.py

Auditoria integrada do pipeline KSP/Principia -> REBOUND -> cache Nível A -> SPK.

Objetivo:
  Testar o que já existe no repositório sem depender do KSP aberto por padrão.
  O foco é separar fatos de suspeitas:
    - integridade dos arquivos;
    - consistência do snapshot verdadeiro;
    - consistência do Nível A REBOUND direto, sem SPK;
    - drift de energia;
    - qualidade dos resíduos por corpo;
    - compatibilidade de mu entre snapshot/kRPC, Principia stock e configs Kopernicus;
    - patches orbitais que afetam os corpos problemáticos;
    - riscos metodológicos antes de ajustar massa, frame, epoch ou integrador.

Uso típico, dentro da raiz do projeto:

  python pipeline_integrity_test.py --root . --report-dir data/test_reports

Modo com catálogo vivo via kRPC, opcional:

  python pipeline_integrity_test.py --root . --live-krpc --live-output data/live_body_catalog.json

Este script usa apenas biblioteca padrão. Não executa propagação REBOUND pesada por padrão.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import statistics as stats
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

G0_KSP = 9.80665
G_SI = 6.67430e-11

# ----------------------------- infra -----------------------------

@dataclass
class Finding:
    level: str  # PASS, WARN, FAIL, INFO
    component: str
    message: str
    evidence: str = ""
    recommendation: str = ""

    def as_markdown(self) -> str:
        parts = [f"- **{self.level}** `{self.component}` — {self.message}"]
        if self.evidence:
            parts.append(f"  - Evidência: {self.evidence}")
        if self.recommendation:
            parts.append(f"  - Recomendação: {self.recommendation}")
        return "\n".join(parts)

@dataclass
class AuditContext:
    root: Path
    report_dir: Path
    findings: List[Finding] = field(default_factory=list)
    facts: Dict[str, Any] = field(default_factory=dict)

    def add(self, level: str, component: str, message: str, evidence: str = "", recommendation: str = "") -> None:
        self.findings.append(Finding(level, component, message, evidence, recommendation))

    def count(self, level: str) -> int:
        return sum(1 for f in self.findings if f.level == level)


def sha256_file(path: Path, chunk: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def find_file(root: Path, rel: str, fallback_names: Iterable[str] = ()) -> Optional[Path]:
    candidates = [root / rel]
    for n in fallback_names:
        candidates.append(root / n)
    for p in candidates:
        if p.exists():
            return p
    # small recursive fallback for local messy trees
    names = [Path(rel).name] + list(fallback_names)
    for name in names:
        hits = list(root.rglob(name))
        if hits:
            # prefer shortest path
            hits.sort(key=lambda x: len(str(x)))
            return hits[0]
    return None


def safe_float(x: Any, default: float = math.nan) -> float:
    try:
        return float(x)
    except Exception:
        return default


def norm3(x: float, y: float, z: float) -> float:
    return math.sqrt(x * x + y * y + z * z)


def finite(v: float) -> bool:
    return math.isfinite(v)


def load_csv_rows(path: Path, max_rows: Optional[int] = None) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if max_rows is not None and i >= max_rows:
                break
            rows.append(row)
    return rows

# ----------------------------- parsers de configs -----------------------------

MU_RE = re.compile(
    r'name\s*:\s*"(?P<name>[^"]+)".*?gravitational_parameter\s*:\s*"(?P<mu>[-+0-9.eE]+)\s*m\^3/s\^2"',
    re.DOTALL,
)

BODY_NAME_RE = re.compile(r'\bname\s*=\s*([^\s#/]+)')
RADIUS_RE = re.compile(r'\bradius\s*=\s*([-+0-9.eE]+)')
GEE_RE = re.compile(r'\bgeeASL\s*=\s*([-+0-9.eE]+)')
GM_RE = re.compile(r'\bgrav(?:itational)?(?:Parameter|parameter)\s*=\s*([-+0-9.eE]+)', re.IGNORECASE)
PATCH_BODY_RE = re.compile(r'@Body\[([^\]]+)\]')
ORBIT_FIELD_RE = re.compile(
    r'[%@]?\b(referenceBody|semiMajorAxis|inclination|eccentricity|longitudeOfAscendingNode|argumentOfPeriapsis|meanAnomalyAtEpochD?|epoch)\s*=\s*([^\n/]+)'
)


def parse_principia_stock_mu(text: str) -> Dict[str, float]:
    return {m.group("name"): float(m.group("mu")) for m in MU_RE.finditer(text)}


def split_kopernicus_body_blocks(text: str) -> List[str]:
    """Heurístico: encontra blocos Body {...} com contagem de chaves."""
    blocks: List[str] = []
    for m in re.finditer(r'\bBody\s*\{', text):
        start = m.start()
        i = m.end() - 1
        depth = 0
        while i < len(text):
            if text[i] == '{':
                depth += 1
            elif text[i] == '}':
                depth -= 1
                if depth == 0:
                    blocks.append(text[start : i + 1])
                    break
            i += 1
    return blocks


def parse_kopernicus_bodies(files: List[Path]) -> Dict[str, Dict[str, Any]]:
    bodies: Dict[str, Dict[str, Any]] = {}
    for path in files:
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        for block in split_kopernicus_body_blocks(text):
            nm = BODY_NAME_RE.search(block)
            if not nm:
                continue
            name = nm.group(1).strip()
            d = bodies.setdefault(name, {"sources": []})
            d["sources"].append(str(path))
            r = RADIUS_RE.search(block)
            g = GEE_RE.search(block)
            gm = GM_RE.search(block)
            if r:
                d["radius_m"] = safe_float(r.group(1))
            if g:
                d["geeASL"] = safe_float(g.group(1))
            if gm:
                d["mu_cfg_explicit_m3_s2"] = safe_float(gm.group(1))
            if "radius_m" in d and "geeASL" in d:
                d["mu_from_radius_gee_m3_s2"] = d["geeASL"] * G0_KSP * d["radius_m"] ** 2
    return bodies


def parse_patch_orbit_modifications(files: List[Path]) -> Dict[str, List[Dict[str, Any]]]:
    mods: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for path in files:
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        for m in PATCH_BODY_RE.finditer(text):
            name = m.group(1).strip()
            # slice from this @Body until next @Body or end-ish
            start = m.start()
            next_m = PATCH_BODY_RE.search(text, m.end())
            end = next_m.start() if next_m else min(len(text), start + 4000)
            chunk = text[start:end]
            fields = {}
            for f in ORBIT_FIELD_RE.finditer(chunk):
                fields[f.group(1)] = f.group(2).strip()
            if fields:
                mods[name].append({"source": str(path), "fields": fields})
    return mods

# ----------------------------- tests -----------------------------

SCRIPT_NAMES = [
    "principia_ephemeris_acquirer.py",
    "principia_true_snapshot.py",
    "principia_true_snapshot_v2.py",
    "rebound_level_a_cache.py",
    "rebound_ephemeris_to_spk_type3.py",
    "rebound_ephemeris_to_spk_type3_v2.py",
    "spk_snapshot_sanity_check.py",
    "spk_snapshot_sanity_check_v2.py",
    "spk_rebound_dense_audit.py",
    "ephemeris_validator.py",
    "ksp_spk_residual_analyzer.py",
]


def audit_file_inventory(ctx: AuditContext, args: argparse.Namespace) -> Dict[str, Optional[Path]]:
    root = ctx.root
    paths: Dict[str, Optional[Path]] = {
        "snapshot": find_file(root, args.snapshot, ["true_snapshot_v2.json"]),
        "level_a_manifest": find_file(root, args.level_a_manifest, ["manifest.json"]),
        "residuals_by_body": find_file(root, args.residuals_by_body, ["residuals_by_body.csv"]),
        "residual_samples": find_file(root, args.residual_samples, ["residual_samples.csv"]),
        "rebound_states": find_file(root, args.rebound_states, ["rebound_states.csv"]),
        "energy": find_file(root, args.energy, ["energy.csv"]),
        "ksp_states": find_file(root, args.ksp_states, ["states.csv"]),
        "principia_gravity": find_file(root, args.principia_gravity, ["kerbol_gravity_model.proto.txt"]),
        "modulemanager_cache": find_file(root, args.modulemanager_cache, ["ModuleManager.ConfigCache"]),
    }
    for key, p in paths.items():
        if p and p.exists():
            ctx.add("PASS", "inventory", f"Arquivo encontrado: {key}", str(p))
        elif key in {"snapshot", "level_a_manifest", "residuals_by_body"}:
            ctx.add("FAIL", "inventory", f"Arquivo essencial ausente: {key}", args.__dict__.get(key, ""))
        else:
            ctx.add("WARN", "inventory", f"Arquivo opcional ausente: {key}", recommendation="O teste continua, mas parte da auditoria será pulada.")

    for script in SCRIPT_NAMES:
        p = find_file(root, script, [script])
        if p:
            ctx.add("PASS", "scripts", f"Script encontrado: {script}", str(p))
        else:
            ctx.add("WARN", "scripts", f"Script ausente: {script}")

    return paths


def audit_snapshot(ctx: AuditContext, snapshot_path: Optional[Path]) -> Dict[str, Any]:
    if not snapshot_path:
        return {}
    try:
        snap = read_json(snapshot_path)
    except Exception as exc:
        ctx.add("FAIL", "snapshot", "Não foi possível ler o snapshot JSON", repr(exc))
        return {}

    schema = snap.get("schema")
    if schema == "principia_true_snapshot.v2":
        ctx.add("PASS", "snapshot", "Snapshot v2 detectado", f"schema={schema}")
    else:
        ctx.add("WARN", "snapshot", "Schema do snapshot não é v2 ou está ausente", f"schema={schema!r}")

    catalog = snap.get("body_catalog", {}).get("bodies", {})
    eph = snap.get("ephemerides", {})
    ctx.facts["snapshot_body_count"] = len(catalog)

    if catalog and eph and set(catalog) == set(eph):
        ctx.add("PASS", "snapshot", "Catálogo e ephemerides têm a mesma lista de corpos", f"n={len(catalog)}")
    else:
        missing_in_eph = sorted(set(catalog) - set(eph))[:10]
        missing_in_cat = sorted(set(eph) - set(catalog))[:10]
        ctx.add("FAIL", "snapshot", "Divergência entre catálogo e ephemerides", f"missing_in_eph={missing_in_eph}; missing_in_cat={missing_in_cat}")

    bad_state = []
    for name, item in eph.items():
        states = item.get("states", []) if isinstance(item, dict) else []
        if len(states) != 1 or len(states[0]) != 7:
            bad_state.append(name)
            continue
        vals = [safe_float(x) for x in states[0]]
        if not all(finite(v) for v in vals):
            bad_state.append(name)
    if not bad_state:
        ctx.add("PASS", "snapshot", "Todos os corpos têm exatamente um estado inicial finito [t,x,y,z,vx,vy,vz]", f"n={len(eph)}")
    else:
        ctx.add("FAIL", "snapshot", "Estados iniciais inválidos no snapshot", ", ".join(bad_state[:20]))

    bad_mu = []
    for name, d in catalog.items():
        mu = safe_float(d.get("mu_m3_s2"))
        mass = safe_float(d.get("mass_kg"))
        if not finite(mu) or mu <= 0 or not finite(mass) or mass <= 0:
            bad_mu.append(name)
    if bad_mu:
        ctx.add("FAIL", "snapshot", "Corpos sem mu/massa válida", ", ".join(bad_mu[:20]))
    else:
        ctx.add("PASS", "snapshot", "Todos os corpos têm mu e massa positivos no catálogo", f"n={len(catalog)}")

    diag = snap.get("sampling_diagnostics", {})
    if diag:
        baseline = safe_float(diag.get("baseline_seconds"))
        passes = diag.get("passes", {})
        pass_durations = [safe_float(v.get("pass_duration_s")) for v in passes.values() if isinstance(v, dict)]
        max_pass = max([p for p in pass_durations if finite(p)] or [math.nan])
        ctx.facts["snapshot_baseline_seconds"] = baseline
        ctx.facts["snapshot_max_rpc_pass_seconds"] = max_pass
        if finite(baseline) and baseline > 0 and finite(max_pass):
            ratio = max_pass / baseline
            if ratio < 0.005:
                ctx.add("PASS", "snapshot", "Skew de passagem RPC pequeno contra baseline da diferença central", f"max_pass={max_pass:.6g}s; baseline={baseline:.6g}s; ratio={ratio:.3e}")
            elif ratio < 0.02:
                ctx.add("WARN", "snapshot", "Skew de passagem RPC moderado", f"max_pass={max_pass:.6g}s; baseline={baseline:.6g}s; ratio={ratio:.3e}", "Para luas rápidas, considere snapshot multi-amostra por corpo.")
            else:
                ctx.add("FAIL", "snapshot", "Skew de passagem RPC alto", f"max_pass={max_pass:.6g}s; baseline={baseline:.6g}s; ratio={ratio:.3e}", "A coleta sequencial pode contaminar velocidades.")

        vdiag = diag.get("body_velocity_diagnostic_m_s", {})
        if vdiag:
            deltas = []
            for name, d in vdiag.items():
                if isinstance(d, dict) and "norm_delta_m_s" in d:
                    deltas.append((safe_float(d["norm_delta_m_s"]), name))
            deltas = [(v, n) for v, n in deltas if finite(v)]
            if deltas:
                deltas.sort(reverse=True)
                worst = deltas[:5]
                ctx.add("INFO", "snapshot", "Diagnóstico kRPC velocity vs diferença central disponível", "; ".join(f"{n}={v:.6g} m/s" for v, n in worst), "Use apenas como diagnóstico; velocity() pode não representar a mesma física/epoch que a diferença central.")

    return snap


def audit_level_a_manifest(ctx: AuditContext, manifest_path: Optional[Path], snapshot: Dict[str, Any]) -> Dict[str, Any]:
    if not manifest_path:
        return {}
    try:
        man = read_json(manifest_path)
    except Exception as exc:
        ctx.add("FAIL", "level_a", "Não foi possível ler manifest.json", repr(exc))
        return {}

    if man.get("schema") == "rebound_level_a_cache.v1":
        ctx.add("PASS", "level_a", "Manifest do Nível A detectado", man.get("purpose", ""))
    else:
        ctx.add("WARN", "level_a", "Schema inesperado no manifest", str(man.get("schema")))

    purpose = str(man.get("purpose", "")).lower()
    if "no spk" in purpose and "no chebyshev" in purpose:
        ctx.add("PASS", "level_a", "Medição Nível A não passa por SPK/Chebyshev", man.get("purpose", ""))
    else:
        ctx.add("WARN", "level_a", "Propósito não declara claramente ausência de SPK/Chebyshev", man.get("purpose", ""))

    integrator = man.get("integrator")
    eps = man.get("ias15_epsilon")
    if integrator == "ias15":
        ctx.add("PASS", "level_a", "Integrador IAS15 registrado", f"ias15_epsilon={eps}")
    else:
        ctx.add("WARN", "level_a", "Integrador diferente de IAS15", f"integrator={integrator}")

    energy = man.get("energy", {})
    drift = safe_float(energy.get("relative_drift"))
    ctx.facts["energy_relative_drift_manifest"] = drift
    if finite(drift):
        ad = abs(drift)
        if ad < 1e-11:
            ctx.add("PASS", "energy", "Drift de energia relativo excelente", f"relative_drift={drift:.3e}")
        elif ad < 1e-8:
            ctx.add("WARN", "energy", "Drift de energia relativo aceitável, mas não excelente", f"relative_drift={drift:.3e}")
        else:
            ctx.add("FAIL", "energy", "Drift de energia relativo alto", f"relative_drift={drift:.3e}", "Antes de calibrar física, reduza tolerância/valide integrador.")

    missing = man.get("missing_bodies_in_rebound", [])
    if not missing:
        ctx.add("PASS", "level_a", "Nenhum corpo faltante no REBOUND segundo manifest", "missing_bodies_in_rebound=[]")
    else:
        ctx.add("FAIL", "level_a", "Há corpos faltando no REBOUND", ", ".join(missing))

    n_bodies = int(man.get("n_bodies", 0) or 0)
    ordered = man.get("ordered_names", []) or []
    if n_bodies == len(ordered) and n_bodies > 0:
        ctx.add("PASS", "level_a", "n_bodies coincide com ordered_names", f"n={n_bodies}")
    else:
        ctx.add("FAIL", "level_a", "n_bodies não coincide com ordered_names", f"n_bodies={n_bodies}; len(ordered_names)={len(ordered)}")

    if snapshot:
        snap_start = safe_float(snapshot.get("start_ut_seconds"))
        man_start = safe_float(man.get("start_et_s"))
        if finite(snap_start) and finite(man_start):
            if abs(snap_start - man_start) < 1e-6:
                ctx.add("PASS", "time", "Epoch do snapshot e start_et_s do Nível A coincidem", f"{snap_start:.12g}s")
            else:
                ctx.add("WARN", "time", "Epoch do snapshot difere do start_et_s do Nível A", f"snapshot={snap_start}; level_a={man_start}")

    epochs = man.get("epochs", {})
    first_et = safe_float(epochs.get("first_et_s"))
    start_et = safe_float(man.get("start_et_s"))
    if finite(first_et) and finite(start_et):
        if first_et < start_et:
            ctx.add("WARN", "time", "CSV de validação começa antes do snapshot inicial", f"first_et={first_et:.9f}; start_et={start_et:.9f}; delta={first_et-start_et:.6f}s", "Recorte states.csv para et >= start_et ou gere snapshot antes da primeira amostra.")
        else:
            ctx.add("PASS", "time", "CSV de validação começa no/depois do snapshot", f"first_et={first_et:.9f}; start_et={start_et:.9f}")

    return man


def audit_energy_csv(ctx: AuditContext, energy_path: Optional[Path], manifest: Dict[str, Any]) -> None:
    if not energy_path:
        return
    try:
        rows = load_csv_rows(energy_path)
    except Exception as exc:
        ctx.add("FAIL", "energy", "Não foi possível ler energy.csv", repr(exc))
        return
    if len(rows) < 2:
        ctx.add("WARN", "energy", "energy.csv tem poucas linhas", f"rows={len(rows)}")
        return
    e0 = safe_float(rows[0].get("energy"))
    e1 = safe_float(rows[-1].get("energy"))
    if finite(e0) and finite(e1) and e0 != 0:
        drift = (e1 - e0) / abs(e0)
        ctx.facts["energy_relative_drift_csv"] = drift
        ctx.add("PASS" if abs(drift) < 1e-11 else "WARN", "energy", "Drift recalculado a partir de energy.csv", f"relative_drift={drift:.3e}; samples={len(rows)}")
    else:
        ctx.add("FAIL", "energy", "Valores inválidos em energy.csv")


def audit_rebound_states(ctx: AuditContext, rebound_states_path: Optional[Path], manifest: Dict[str, Any]) -> None:
    if not rebound_states_path:
        return
    try:
        n_rows = 0
        bodies = set()
        epochs = set()
        bad = 0
        min_t = math.inf
        max_t = -math.inf
        with rebound_states_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            required = {"et_seconds", "t_rebound_seconds", "body", "x_m", "y_m", "z_m", "vx_m_s", "vy_m_s", "vz_m_s"}
            if set(reader.fieldnames or []) >= required:
                ctx.add("PASS", "rebound_states", "Cabeçalho de rebound_states.csv contém colunas necessárias")
            else:
                ctx.add("FAIL", "rebound_states", "Cabeçalho de rebound_states.csv incompleto", str(reader.fieldnames))
            for row in reader:
                n_rows += 1
                bodies.add(row.get("body", ""))
                et = safe_float(row.get("et_seconds"))
                if finite(et):
                    epochs.add(et)
                    min_t = min(min_t, et)
                    max_t = max(max_t, et)
                vals = [safe_float(row.get(k)) for k in ("x_m", "y_m", "z_m", "vx_m_s", "vy_m_s", "vz_m_s")]
                if not all(finite(v) for v in vals):
                    bad += 1
        expected = len(bodies) * len(epochs)
        if n_rows == expected:
            ctx.add("PASS", "rebound_states", "Grade body×epoch completa em rebound_states.csv", f"rows={n_rows}; bodies={len(bodies)}; epochs={len(epochs)}")
        else:
            ctx.add("WARN", "rebound_states", "Grade body×epoch incompleta ou duplicada", f"rows={n_rows}; bodies={len(bodies)}; epochs={len(epochs)}; expected={expected}")
        if bad == 0:
            ctx.add("PASS", "rebound_states", "Todos os estados REBOUND são finitos", f"rows={n_rows}")
        else:
            ctx.add("FAIL", "rebound_states", "Estados REBOUND não finitos", f"bad_rows={bad}")
        ctx.facts["rebound_states_rows"] = n_rows
        ctx.facts["rebound_states_bodies"] = len(bodies)
        ctx.facts["rebound_states_epochs"] = len(epochs)
    except Exception as exc:
        ctx.add("FAIL", "rebound_states", "Erro lendo rebound_states.csv", repr(exc))


def classify_body(max_km: float, rms_km: float) -> str:
    if max_km <= 1.0:
        return "excellent"
    if max_km <= 10.0:
        return "good"
    if max_km <= 100.0:
        return "usable"
    if max_km <= 1000.0:
        return "warning"
    return "fail"


def audit_residuals_by_body(ctx: AuditContext, residuals_path: Optional[Path]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    if not residuals_path:
        return out
    try:
        rows = load_csv_rows(residuals_path)
    except Exception as exc:
        ctx.add("FAIL", "residuals", "Não foi possível ler residuals_by_body.csv", repr(exc))
        return out
    if not rows:
        ctx.add("FAIL", "residuals", "residuals_by_body.csv vazio")
        return out

    for r in rows:
        body = r.get("body", "")
        if not body:
            continue
        out[body] = {
            "samples": safe_float(r.get("samples")),
            "max_pos_err_km": safe_float(r.get("max_pos_err_km")),
            "rms_pos_err_km": safe_float(r.get("rms_pos_err_km")),
            "final_pos_err_km": safe_float(r.get("final_pos_err_km")),
            "max_vel_err_m_s": safe_float(r.get("max_vel_err_m_s")),
            "median_apparent_epoch_offset_s": safe_float(r.get("median_apparent_epoch_offset_s")),
        }

    # Alguns corpos centrais podem ter velocidade zero e, portanto, offset temporal aparente indefinido.
    # Isso não invalida max/RMS/final; tratamos dt indefinido como aviso apenas fora do corpo central/erro zero.
    bad_vals = []
    dt_undefined = []
    for b, d in out.items():
        core_keys = ["samples", "max_pos_err_km", "rms_pos_err_km", "final_pos_err_km", "max_vel_err_m_s"]
        if not all(finite(d[k]) for k in core_keys):
            bad_vals.append(b)
        if not finite(d["median_apparent_epoch_offset_s"]):
            if d.get("max_pos_err_km", math.inf) == 0 and d.get("max_vel_err_m_s", math.inf) == 0:
                dt_undefined.append(b)
            else:
                bad_vals.append(b)
    if bad_vals:
        ctx.add("FAIL", "residuals", "Valores essenciais não finitos nos resíduos por corpo", ", ".join(sorted(set(bad_vals))[:20]))
    else:
        ctx.add("PASS", "residuals", "Todos os resíduos essenciais por corpo são finitos", f"bodies={len(out)}")
    if dt_undefined:
        ctx.add("INFO", "residuals", "Offset temporal aparente indefinido em corpo(s) de erro zero/velocidade zero", ", ".join(dt_undefined[:20]))

    buckets = defaultdict(list)
    for body, d in out.items():
        buckets[classify_body(d["max_pos_err_km"], d["rms_pos_err_km"])].append(body)
    ctx.facts["residual_buckets"] = {k: sorted(v) for k, v in buckets.items()}

    top = sorted(out.items(), key=lambda kv: kv[1]["max_pos_err_km"], reverse=True)[:12]
    clean = sorted(out.items(), key=lambda kv: kv[1]["max_pos_err_km"])[:12]
    ctx.add("INFO", "residuals", "Piores corpos por erro máximo", "; ".join(f"{b}={d['max_pos_err_km']:.3g} km" for b, d in top))
    ctx.add("INFO", "residuals", "Melhores corpos por erro máximo", "; ".join(f"{b}={d['max_pos_err_km']:.3g} km" for b, d in clean))

    # global epoch offset test among good bodies
    good_offsets = [d["median_apparent_epoch_offset_s"] for b, d in out.items() if d["max_pos_err_km"] <= 100 and finite(d["median_apparent_epoch_offset_s"])]
    if len(good_offsets) >= 5:
        med = stats.median(good_offsets)
        spread = stats.median([abs(x - med) for x in good_offsets])
        ctx.facts["good_body_epoch_offset_median_s"] = med
        ctx.facts["good_body_epoch_offset_mad_s"] = spread
        if abs(med) < 1.0 and spread < 1.0:
            ctx.add("PASS", "time", "Corpos bons não indicam offset temporal global relevante", f"median={med:.6g}s; MAD={spread:.6g}s; n={len(good_offsets)}")
        else:
            ctx.add("WARN", "time", "Corpos bons sugerem possível offset temporal fino", f"median={med:.6g}s; MAD={spread:.6g}s; n={len(good_offsets)}", "Ajuste et_offset_seconds apenas após recortar o CSV para et>=snapshot.")

    fail_count = len(buckets.get("fail", []))
    if fail_count:
        ctx.add("WARN", "residuals", "Há corpos com erro >1000 km no Nível A", f"count={fail_count}; bodies={', '.join(sorted(buckets['fail'])[:30])}", "Não use esses corpos para calibrar μ global; investigue por família dinâmica.")
    else:
        ctx.add("PASS", "residuals", "Nenhum corpo com erro >1000 km no Nível A")

    return out


def audit_residual_samples_consistency(ctx: AuditContext, residual_samples_path: Optional[Path], residuals: Dict[str, Dict[str, float]]) -> None:
    if not residual_samples_path or not residuals:
        return
    try:
        per_body_pos: Dict[str, List[float]] = defaultdict(list)
        per_body_vel: Dict[str, List[float]] = defaultdict(list)
        n = 0
        with residual_samples_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                body = row.get("body", "")
                pe = safe_float(row.get("pos_err_m"))
                ve = safe_float(row.get("vel_err_m_s"))
                if body and finite(pe):
                    per_body_pos[body].append(pe / 1000.0)
                if body and finite(ve):
                    per_body_vel[body].append(ve)
                n += 1
        mismatches = []
        for body, vals in per_body_pos.items():
            if body not in residuals or not vals:
                continue
            max_km = max(vals)
            rms_km = math.sqrt(sum(v * v for v in vals) / len(vals))
            ref_max = residuals[body]["max_pos_err_km"]
            ref_rms = residuals[body]["rms_pos_err_km"]
            tol = max(1e-6, 1e-6 * max(1.0, abs(ref_max)))
            if abs(max_km - ref_max) > tol or abs(rms_km - ref_rms) > tol:
                mismatches.append((body, max_km, ref_max, rms_km, ref_rms))
        if mismatches:
            msg = "; ".join(f"{b}:max {m:.6g}!={rm:.6g}, rms {r:.6g}!={rr:.6g}" for b, m, rm, r, rr in mismatches[:8])
            ctx.add("FAIL", "residual_samples", "Agregados de residual_samples não batem com residuals_by_body", msg)
        else:
            ctx.add("PASS", "residual_samples", "residual_samples.csv reproduz max/RMS de residuals_by_body", f"rows={n}; bodies={len(per_body_pos)}")
    except Exception as exc:
        ctx.add("FAIL", "residual_samples", "Erro auditando residual_samples.csv", repr(exc))


def audit_configs_and_mu(ctx: AuditContext, paths: Dict[str, Optional[Path]], snapshot: Dict[str, Any], residuals: Dict[str, Dict[str, float]]) -> None:
    root = ctx.root
    # Gather cfg/proto-ish files from dados/ and root.
    cfg_files: List[Path] = []
    for folder in [root / "dados", root]:
        if folder.exists():
            for pat in ("*.txt", "*.cfg"):
                cfg_files.extend(folder.glob(pat))
    # unique
    cfg_files = sorted(set(cfg_files), key=lambda p: str(p))
    ctx.add("INFO", "configs", "Arquivos de config/proto encontrados para auditoria", f"count={len(cfg_files)}")

    if paths.get("modulemanager_cache"):
        ctx.add("PASS", "configs", "ModuleManager.ConfigCache encontrado", str(paths["modulemanager_cache"]), "Use esse arquivo como fonte canônica de configs pós-patch.")
    else:
        ctx.add("WARN", "configs", "ModuleManager.ConfigCache não encontrado", recommendation="Sem ConfigCache, configs e patches são auditoria parcial; não são verdade final do jogo carregado.")

    stock_mu: Dict[str, float] = {}
    if paths.get("principia_gravity"):
        try:
            stock_mu = parse_principia_stock_mu(paths["principia_gravity"].read_text(encoding="utf-8", errors="ignore"))
            ctx.add("PASS", "configs", "μ stock do Principia extraído", f"bodies={len(stock_mu)}")
        except Exception as exc:
            ctx.add("WARN", "configs", "Falha ao parsear kerbol_gravity_model.proto", repr(exc))

    cfg_bodies = parse_kopernicus_bodies(cfg_files)
    if cfg_bodies:
        ctx.add("PASS", "configs", "Corpos Kopernicus parseados de configs base", f"bodies={len(cfg_bodies)}")
    else:
        ctx.add("WARN", "configs", "Nenhum corpo Kopernicus parseado")

    patch_mods = parse_patch_orbit_modifications(cfg_files)
    if patch_mods:
        ctx.add("PASS", "configs", "Patches orbitais detectados", f"bodies_modified={len(patch_mods)}")
    else:
        ctx.add("WARN", "configs", "Nenhum patch orbital detectado")

    snap_catalog = snapshot.get("body_catalog", {}).get("bodies", {}) if snapshot else {}
    if snap_catalog and stock_mu:
        diffs = []
        for body, mu_ref in stock_mu.items():
            if body in snap_catalog:
                mu_snap = safe_float(snap_catalog[body].get("mu_m3_s2"))
                if finite(mu_snap) and mu_ref > 0:
                    rel = (mu_snap - mu_ref) / mu_ref
                    diffs.append((abs(rel), rel, body, mu_snap, mu_ref))
        diffs.sort(reverse=True)
        if diffs:
            worst = diffs[:8]
            max_abs = worst[0][0]
            if max_abs < 1e-10:
                ctx.add("PASS", "mu", "μ stock do snapshot coincide com Principia gravity_model", "; ".join(f"{b}:rel={rel:.3e}" for _, rel, b, _, _ in worst))
            elif max_abs < 1e-6:
                ctx.add("WARN", "mu", "μ stock do snapshot difere levemente do Principia gravity_model", "; ".join(f"{b}:rel={rel:.3e}" for _, rel, b, _, _ in worst))
            else:
                ctx.add("FAIL", "mu", "μ stock do snapshot difere do Principia gravity_model", "; ".join(f"{b}:rel={rel:.3e}" for _, rel, b, _, _ in worst), "Verifique versões KSP/Principia e se kRPC está reportando μ pós-patch.")

    if snap_catalog and cfg_bodies:
        cfg_mu_diffs = []
        missing_cfg = []
        for body, sd in snap_catalog.items():
            if body in stock_mu:
                continue  # stock covered above
            if body not in cfg_bodies:
                missing_cfg.append(body)
                continue
            cd = cfg_bodies[body]
            mu_cfg = cd.get("mu_cfg_explicit_m3_s2") or cd.get("mu_from_radius_gee_m3_s2")
            mu_snap = safe_float(sd.get("mu_m3_s2"))
            if mu_cfg and finite(mu_snap) and mu_cfg > 0:
                rel = (mu_snap - mu_cfg) / mu_cfg
                cfg_mu_diffs.append((abs(rel), rel, body, mu_snap, mu_cfg))
        cfg_mu_diffs.sort(reverse=True)
        if cfg_mu_diffs:
            worst = cfg_mu_diffs[:12]
            ctx.add("INFO", "mu", "Comparação μ snapshot vs μ derivado de configs para corpos não-stock", "; ".join(f"{b}:rel={rel:.3e}" for _, rel, b, _, _ in worst), "Diferenças aqui são diagnóstico; ConfigCache/kRPC são fontes melhores que geeASL*radius².")
        if missing_cfg:
            ctx.add("WARN", "configs", "Corpos do snapshot sem definição base parseada em configs", ", ".join(sorted(missing_cfg)[:30]))

    if residuals and patch_mods:
        bad = [b for b, d in residuals.items() if d["max_pos_err_km"] > 1000]
        impacted = [b for b in bad if b in patch_mods]
        not_impacted = [b for b in bad if b not in patch_mods]
        if impacted:
            detail = []
            for b in sorted(impacted)[:20]:
                fields = sorted({k for mod in patch_mods[b] for k in mod["fields"].keys()})
                detail.append(f"{b}({','.join(fields)})")
            ctx.add("INFO", "configs", "Corpos ruins no Nível A que têm patch orbital detectado", "; ".join(detail), "Isso não prova causalidade, mas prioriza auditoria por família/pós-ModuleManager.")
        if not_impacted:
            ctx.add("INFO", "configs", "Corpos ruins sem patch orbital detectado nos arquivos carregados", ", ".join(sorted(not_impacted)[:20]), "Procure no ModuleManager.ConfigCache ou em gravity models adicionais.")


def audit_live_krpc(ctx: AuditContext, output: Path) -> None:
    try:
        import krpc  # type: ignore
    except Exception as exc:
        ctx.add("FAIL", "live_krpc", "krpc não está disponível no ambiente Python", repr(exc))
        return
    try:
        conn = krpc.connect(name="Pipeline_Integrity_Test")
        sc = conn.space_center
        bodies = sc.bodies
        data = {
            "schema": "live_krpc_body_catalog.v1",
            "ut_s": float(sc.ut),
            "bodies": {},
        }
        for name, b in bodies.items():
            item = {
                "name": name,
                "gravitational_parameter_m3_s2": float(b.gravitational_parameter),
            }
            for attr, out_name in [
                ("equatorial_radius", "equatorial_radius_m"),
                ("sphere_of_influence", "sphere_of_influence_m"),
                ("rotational_period", "rotational_period_s"),
            ]:
                try:
                    item[out_name] = float(getattr(b, attr))
                except Exception:
                    pass
            try:
                item["has_atmosphere"] = bool(b.has_atmosphere)
                item["atmosphere_depth_m"] = float(b.atmosphere_depth)
            except Exception:
                pass
            data["bodies"][name] = item
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        ctx.add("PASS", "live_krpc", "Catálogo vivo extraído via kRPC", f"bodies={len(data['bodies'])}; output={output}")
    except Exception as exc:
        ctx.add("FAIL", "live_krpc", "Falha na extração viva via kRPC", repr(exc))
    finally:
        try:
            conn.close()  # type: ignore[name-defined]
        except Exception:
            pass

# ----------------------------- reports -----------------------------


def write_reports(ctx: AuditContext, paths: Dict[str, Optional[Path]]) -> Tuple[Path, Path]:
    ctx.report_dir.mkdir(parents=True, exist_ok=True)
    report_md = ctx.report_dir / "pipeline_integrity_report.md"
    report_json = ctx.report_dir / "pipeline_integrity_report.json"

    levels = ["FAIL", "WARN", "PASS", "INFO"]
    summary = {lv: ctx.count(lv) for lv in levels}

    md: List[str] = []
    md.append("# Relatório de Integridade do Pipeline KSP/Principia → REBOUND\n")
    md.append("## Sumário\n")
    md.append("| Nível | Contagem |\n|---|---:|")
    for lv in levels:
        md.append(f"| {lv} | {summary[lv]} |")
    md.append("")

    md.append("## Fatos extraídos\n")
    if ctx.facts:
        for k, v in sorted(ctx.facts.items()):
            md.append(f"- `{k}`: `{v}`")
    else:
        md.append("- Nenhum fato agregado.")
    md.append("")

    md.append("## Arquivos usados\n")
    for key, p in paths.items():
        md.append(f"- `{key}`: `{p if p else 'AUSENTE'}`")
    md.append("")

    for lv in levels:
        md.append(f"## {lv}\n")
        items = [f for f in ctx.findings if f.level == lv]
        if not items:
            md.append("Nenhum.\n")
        else:
            for f in items:
                md.append(f.as_markdown())
            md.append("")

    # final operational conclusion
    md.append("## Conclusão operacional\n")
    if summary["FAIL"] > 0:
        md.append("Há falhas que impedem chamar todos os resíduos de fatos físicos finais. Corrija os itens FAIL antes de calibrar μ/frame/epoch globalmente.\n")
    elif summary["WARN"] > 0:
        md.append("Não há falhas bloqueantes, mas há alertas metodológicos. Use apenas os corpos classificados como bons para calibração global e investigue os corpos ruins por família dinâmica.\n")
    else:
        md.append("O pacote auditado está consistente para avançar para calibração controlada por corpo/família.\n")

    report_md.write_text("\n".join(md), encoding="utf-8")

    payload = {
        "summary": summary,
        "facts": ctx.facts,
        "paths": {k: str(v) if v else None for k, v in paths.items()},
        "findings": [f.__dict__ for f in ctx.findings],
    }
    report_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return report_md, report_json


def print_console_summary(ctx: AuditContext, report_md: Path, report_json: Path) -> None:
    print("\n=== Pipeline Integrity Test ===")
    for lv in ["FAIL", "WARN", "PASS", "INFO"]:
        print(f"{lv:>4}: {ctx.count(lv)}")
    print(f"\nRelatórios:")
    print(f"  Markdown: {report_md}")
    print(f"  JSON    : {report_json}")

    print("\nPrincipais alertas/falhas:")
    shown = 0
    for f in ctx.findings:
        if f.level in {"FAIL", "WARN"}:
            print(f"[{f.level}] {f.component}: {f.message}")
            if f.evidence:
                print(f"       {f.evidence}")
            shown += 1
            if shown >= 12:
                break
    if shown == 0:
        print("  Nenhum FAIL/WARN.")

# ----------------------------- main -----------------------------


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Auditoria integrada do pipeline KSP/Principia -> REBOUND.")
    ap.add_argument("--root", type=Path, default=Path("."), help="Raiz do projeto.")
    ap.add_argument("--report-dir", type=Path, default=Path("data/test_reports"), help="Diretório de saída dos relatórios.")

    ap.add_argument("--snapshot", default="data/true_snapshot_v2.json")
    ap.add_argument("--level-a-manifest", default="data/level_a_rebound_vs_ksp_1y/manifest.json")
    ap.add_argument("--residuals-by-body", default="data/level_a_rebound_vs_ksp_1y/residuals_by_body.csv")
    ap.add_argument("--residual-samples", default="data/level_a_rebound_vs_ksp_1y/residual_samples.csv")
    ap.add_argument("--rebound-states", default="data/level_a_rebound_vs_ksp_1y/rebound_states.csv")
    ap.add_argument("--energy", default="data/level_a_rebound_vs_ksp_1y/energy.csv")
    ap.add_argument("--ksp-states", default="data/opm_mpe_360d/states.csv")
    ap.add_argument("--principia-gravity", default="dados/kerbol_gravity_model.proto.txt")
    ap.add_argument("--modulemanager-cache", default="GameData/ModuleManager.ConfigCache")

    ap.add_argument("--live-krpc", action="store_true", help="Opcional: conectar ao KSP/kRPC e extrair catálogo vivo.")
    ap.add_argument("--live-output", type=Path, default=Path("data/live_body_catalog.json"))
    return ap


def main(argv: Optional[List[str]] = None) -> int:
    args = build_argparser().parse_args(argv)
    root = args.root.resolve()
    report_dir = (root / args.report_dir) if not args.report_dir.is_absolute() else args.report_dir
    ctx = AuditContext(root=root, report_dir=report_dir)

    paths = audit_file_inventory(ctx, args)
    snapshot = audit_snapshot(ctx, paths.get("snapshot"))
    manifest = audit_level_a_manifest(ctx, paths.get("level_a_manifest"), snapshot)
    audit_energy_csv(ctx, paths.get("energy"), manifest)
    audit_rebound_states(ctx, paths.get("rebound_states"), manifest)
    residuals = audit_residuals_by_body(ctx, paths.get("residuals_by_body"))
    audit_residual_samples_consistency(ctx, paths.get("residual_samples"), residuals)
    audit_configs_and_mu(ctx, paths, snapshot, residuals)

    if args.live_krpc:
        live_out = args.live_output if args.live_output.is_absolute() else root / args.live_output
        audit_live_krpc(ctx, live_out)

    report_md, report_json = write_reports(ctx, paths)
    print_console_summary(ctx, report_md, report_json)
    return 2 if ctx.count("FAIL") > 0 else (1 if ctx.count("WARN") > 0 else 0)


if __name__ == "__main__":
    raise SystemExit(main())
