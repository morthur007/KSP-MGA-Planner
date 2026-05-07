#!/usr/bin/env python3
"""
MGA Pipeline Orchestrator V0.2

Purpose
-------
Turn the current collection of research/prototype scripts into a mission-level
workflow driven by a single JSON specification.

This orchestrator deliberately remains conservative:
  * it shells out to the validated stage scripts instead of reimplementing them;
  * it can print a plan without executing it;
  * it can execute stage ranges;
  * it keeps all outputs under a mission work directory;
  * it supports the current multi-flyby Galileo-like path.

Scope V0.1
----------
Global leg library -> beam search -> connected PyGMO refinement -> robust
selection -> patched arc correction -> flyby closure -> B-plane packet ->
local target/validation -> multi-flyby stitch -> patch correction -> 6D
selection -> final B6D packet.

The next orchestrator version should add automatic sequence mining from the beam
CSV, rocket-equation ranking, and REBOUND/Principia validation branches.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed


SCRIPT_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Cmd:
    stage: str
    name: str
    argv: List[str]
    cwd: Optional[Path] = None

    def shell(self) -> str:
        return " ".join(shlex.quote(x) for x in self.argv)


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def sanitize_body(name: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in name).strip("_")


def sanitize_seq(seq: str | Sequence[str]) -> str:
    if isinstance(seq, str):
        bodies = [x.strip() for x in seq.replace("->", ",").split(",") if x.strip()]
    else:
        bodies = list(seq)
    return "_".join(sanitize_body(x) for x in bodies)


def parse_sequence(seq: str) -> List[str]:
    return [x.strip() for x in seq.replace("->", ",").split(",") if x.strip()]


def script(name: str) -> str:
    # Prefer local working-tree script; fall back to /mnt/data script path.
    local = Path(name)
    if local.exists():
        return str(local)
    here = SCRIPT_DIR / name
    return str(here)


def add_flag(argv: List[str], flag: str, value: Any = None) -> None:
    if isinstance(value, bool):
        if value:
            argv.append(flag)
    elif value is None:
        argv.append(flag)
    elif isinstance(value, (list, tuple)):
        argv.append(flag)
        argv.extend(str(x) for x in value)
    else:
        argv.extend([flag, str(value)])


def common_spice_args(cfg: Dict[str, Any]) -> List[str]:
    p = cfg["paths"]
    s = cfg["spice"]
    return [
        "--bsp", p["bsp"],
        "--tpc", p["tpc"],
        "--central-body", s.get("central_body", "Sun"),
        "--mu-central-km3-s2", str(s["mu_central_km3_s2"]),
    ]


def body_catalog(cfg: Dict[str, Any]) -> str:
    return cfg["paths"]["body_catalog"]


def grav_bodies(cfg: Dict[str, Any]) -> List[str]:
    return list(cfg["bodies"].get("gravitating_bodies", cfg["bodies"].get("search_bodies", [])))


def leg_origins(cfg: Dict[str, Any]) -> List[str]:
    b = cfg["bodies"]
    if b.get("leg_origins"):
        return list(b["leg_origins"])
    return list(b.get("route_bodies", b.get("search_bodies", [])))


def leg_targets(cfg: Dict[str, Any]) -> List[str]:
    b = cfg["bodies"]
    if b.get("leg_targets"):
        return list(b["leg_targets"])
    return list(b.get("route_bodies", b.get("search_bodies", [])))


def allowed_flyby_bodies(cfg: Dict[str, Any]) -> List[str]:
    return list(cfg["bodies"].get("allowed_flyby_bodies", []))


def disallowed_flyby_bodies(cfg: Dict[str, Any]) -> List[str]:
    return list(cfg["bodies"].get("disallowed_flyby_bodies", []))


def work_dir(cfg: Dict[str, Any]) -> Path:
    return Path(cfg["paths"]["work_dir"])


def build_commands(cfg: Dict[str, Any], stages: Optional[set[str]] = None) -> List[Cmd]:
    w = work_dir(cfg)
    legs_dir = w / "legs"
    connected_dir = w / "connected"
    local_dir = w / "local"
    final_dir = w / "final"

    p = cfg["paths"]
    spice_cfg = cfg["spice"]
    bodies_cfg = cfg["bodies"]
    search_cfg = cfg["search"]
    post_cfg = search_cfg["postprocess"]
    beam_cfg = search_cfg["beam"]
    refine_cfg = cfg["refinement"]
    select_cfg = cfg["selection"]
    integ_cfg = cfg["integration"]
    vehicle_cfg = cfg.get("vehicle", {})

    cmds: List[Cmd] = []

    def want(stage: str) -> bool:
        return stages is None or stage in stages

    # 0. Body catalog extraction is optional because it needs KSP/kRPC alive.
    if want("body_catalog"):
        out = p["body_catalog"]
        argv = ["python", script("mga_extract_body_catalog_krpc_v0_1.py"),
                "--policy", p["policy"],
                "--output-json", out,
                "--report-csv", str(w / "body_catalog.krpc.csv")]
        cmds.append(Cmd("body_catalog", "extract_body_catalog_krpc", argv))

    # 1. Leg library: every origin to every other search body.
    if want("legs"):
        for origin in leg_origins(cfg):
            targets = [b for b in leg_targets(cfg) if b != origin]
            prefix = legs_dir / f"{sanitize_body(origin)}"
            scout_csv = f"{prefix}_lambert_scout.csv"
            scout_json = f"{prefix}_lambert_scout.summary.json"
            seed_jsonl = f"{prefix}_lambert_leg_seeds.jsonl"
            clustered_csv = f"{prefix}_lambert_scout.clustered.csv"
            post_json = f"{prefix}_lambert_scout.postprocess.summary.json"

            argv = ["python", script("spice_lambert_scout.py"),
                    "--bsp", p["bsp"], "--tpc", p["tpc"],
                    "--metadata", p["metadata"], "--policy", p["policy"],
                    "--origin", origin, "--targets", *targets,
                    "--central-body", spice_cfg.get("central_body", "Sun"),
                    "--depart-start-days", str(search_cfg["depart_start_days"]),
                    "--depart-stop-days", str(search_cfg["depart_stop_days"]),
                    "--depart-step-days", str(search_cfg["depart_step_days"]),
                    "--tof-min-days", str(search_cfg["tof_min_days"]),
                    "--tof-max-days", str(search_cfg["tof_max_days"]),
                    "--tof-step-days", str(search_cfg["tof_step_days"]),
                    "--max-c3", str(vehicle_cfg.get("max_departure_c3_km2_s2", 500)),
                    "--top-n-per-target", str(search_cfg.get("top_n_per_target_raw", 250)),
                    "--output-csv", scout_csv,
                    "--output-json", scout_json]
            cmds.append(Cmd("legs", f"scout_{origin}", argv))

            argv = ["python", script("mga_candidate_postprocess.py"),
                    "--input-csv", scout_csv,
                    "--input-summary", scout_json,
                    "--output-csv", clustered_csv,
                    "--output-json", post_json,
                    "--output-jsonl", seed_jsonl,
                    "--depart-bin-days", str(post_cfg["depart_bin_days"]),
                    "--tof-bin-days", str(post_cfg["tof_bin_days"]),
                    "--arrival-bin-days", str(post_cfg.get("arrival_bin_days", 80)),
                    "--top-n-per-cluster", str(post_cfg.get("top_n_per_cluster", 1)),
                    "--top-n-per-target", str(post_cfg.get("top_n_per_target", 80))]
            cmds.append(Cmd("legs", f"postprocess_{origin}", argv))

    # 2. Beam search over all leg seeds.
    routes_jsonl = str(w / "beam_routes.jsonl")
    if want("beam"):
        argv = ["python", script("mga_beam_search_v0_4.py"),
                "--input-jsonl", str(legs_dir / "*_lambert_leg_seeds.jsonl"),
                "--body-catalog", body_catalog(cfg),
                "--start-body", bodies_cfg["start_body"],
                "--final-targets", *bodies_cfg["final_targets"],
                "--allowed-flyby-bodies", *allowed_flyby_bodies(cfg),
                "--disallowed-flyby-bodies", *disallowed_flyby_bodies(cfg),
                "--require-final-target",
                "--min-depth", str(beam_cfg.get("min_depth", 2)),
                "--max-depth", str(beam_cfg.get("max_depth", 4)),
                "--min-layover-days", str(beam_cfg.get("min_layover_days", 0)),
                "--max-layover-days", str(beam_cfg.get("max_layover_days", 800)),
                "--max-vinf-mag-jump", str(beam_cfg.get("max_vinf_mag_jump", 4.0)),
                "--max-turn-angle-deg", str(beam_cfg.get("max_turn_angle_deg", 170)),
                "--flyby-vinf-mode", beam_cfg.get("flyby_vinf_mode", "conservative"),
                "--min-rp-margin-km", str(beam_cfg.get("min_rp_margin_km", 0)),
                "--beam-width", str(beam_cfg.get("beam_width", 500)),
                "--branch-factor-per-node", str(beam_cfg.get("branch_factor_per_node", 40)),
                "--arrival-bin-days", str(beam_cfg.get("arrival_bin_days", 80)),
                "--max-per-bucket", str(beam_cfg.get("max_per_bucket", 8)),
                "--output-top-n", str(beam_cfg.get("output_top_n", 0)),
                "--repeatable-bodies", *bodies_cfg.get("repeatable_bodies", []),
                "--max-visits-per-repeatable-body", str(bodies_cfg.get("max_visits_per_repeatable_body", 2)),
                *(["--allow-return-to-start"] if bodies_cfg.get("allow_return_to_start", bool(bodies_cfg.get("repeatable_bodies"))) else []),
                "--output-csv", str(w / "beam_routes.csv"),
                "--output-jsonl", routes_jsonl,
                "--output-json", str(w / "beam_routes.summary.json")]
        cmds.append(Cmd("beam", "beam_search", argv))

    # 3+. Per-sequence pipeline. Current V0.1 handles connected/multi-flyby sequences.
    for seq in cfg.get("candidate_sequences", []):
        seq_bodies = parse_sequence(seq)
        seq_key = ",".join(seq_bodies)
        seq_slug = sanitize_seq(seq_bodies)
        flyby_count = max(0, len(seq_bodies) - 2)
        if flyby_count < 1:
            continue
        required_flybys = bodies_cfg.get("required_flyby_bodies_by_sequence", {}).get(seq_key, seq_bodies[1:-1])
        seq_dir = connected_dir / seq_slug
        connected_jsonl = str(seq_dir / f"connected_{seq_slug}.jsonl")
        selected_jsonl = str(seq_dir / f"selected_{seq_slug}.jsonl")
        corrected_jsonl = str(seq_dir / f"arc_corrected_{seq_slug}.jsonl")
        closure_jsonl = str(seq_dir / f"flyby_closure_{seq_slug}.jsonl")
        bplane_packet_json = str(seq_dir / f"bplane_packet_{seq_slug}.json")
        bplane_spec_jsonl = str(seq_dir / f"bplane_target_spec_{seq_slug}.jsonl")
        local_target_jsonl = str(local_dir / seq_slug / f"local_target_{seq_slug}.jsonl")
        local_val_jsonl = str(local_dir / seq_slug / f"local_validate_{seq_slug}.jsonl")
        stitched_jsonl = str(local_dir / seq_slug / f"stitched_multiflyby_{seq_slug}.jsonl")
        mf_corrected_jsonl = str(final_dir / seq_slug / f"multiflyby_corrected_{seq_slug}.jsonl")
        diag_jsonl = str(final_dir / seq_slug / f"sixd_diag_{seq_slug}.jsonl")
        viable_jsonl = str(final_dir / seq_slug / f"sixd_viable_{seq_slug}.jsonl")
        b6d_jsonl = str(final_dir / seq_slug / f"b6d_packet_{seq_slug}.jsonl")

        if want("refine"):
            argv = ["python", script("mga_pygmo_refine_connected_flyby_v0_1.py"),
                    "--bsp", p["bsp"], "--tpc", p["tpc"], "--metadata", p["metadata"],
                    "--body-catalog", body_catalog(cfg),
                    "--routes-jsonl", routes_jsonl,
                    "--central-body", spice_cfg.get("central_body", "Sun"),
                    "--mu-central-km3-s2", str(spice_cfg["mu_central_km3_s2"]),
                    "--sequence", seq_key,
                    "--max-routes", str(refine_cfg.get("max_routes_per_sequence", 40)),
                    "--workers", str(refine_cfg.get("pygmo_workers", integ_cfg.get("workers", 1))),
                    "--generations", str(refine_cfg.get("pygmo_generations", 320)),
                    "--population", str(refine_cfg.get("pygmo_population", 128)),
                    "--runs", str(refine_cfg.get("pygmo_runs", 8)),
                    "--depart-window-days", str(refine_cfg.get("depart_window_days", 360)),
                    "--tof-window-days", str(refine_cfg.get("tof_window_days", 500)),
                    "--max-vinf-mismatch-m-s", str(refine_cfg.get("max_vinf_mismatch_m_s", 25)),
                    "--vinf-mismatch-soft-m-s", str(refine_cfg.get("vinf_mismatch_soft_m_s", 5)),
                    "--min-rp-margin-km", str(refine_cfg.get("min_rp_margin_km", 300)),
                    "--rp-soft-margin-km", str(refine_cfg.get("rp_soft_margin_km", 1000)),
                    "--output-csv", str(seq_dir / f"connected_{seq_slug}.csv"),
                    "--output-jsonl", connected_jsonl,
                    "--output-json", str(seq_dir / f"connected_{seq_slug}.summary.json")]
            cmds.append(Cmd("refine", f"refine_{seq_slug}", argv))

        if want("select"):
            sel = select_cfg["connected"]
            argv = ["python", script("mga_select_connected_flyby_routes_v0_1.py"),
                    "--input-jsonl", connected_jsonl,
                    "--require-sequence", seq_key,
                    "--require-flyby-bodies", *required_flybys,
                    "--min-rp-margin-km", str(sel.get("min_rp_margin_km", 300)),
                    "--min-per-flyby-rp-margin-km", str(sel.get("min_per_flyby_rp_margin_km", 100)),
                    "--max-vinf-mismatch-m-s", str(sel.get("max_vinf_mismatch_m_s", 25)),
                    "--max-per-flyby-vinf-mismatch-m-s", str(sel.get("max_per_flyby_vinf_mismatch_m_s", 25)),
                    "--max-turn-angle-deg", str(sel.get("max_turn_angle_deg", 90)),
                    "--max-tof-days", str(sel.get("max_tof_days", 1800)),
                    "--depart-bin-days", str(sel.get("depart_bin_days", 30)),
                    "--tof-bin-days", str(sel.get("tof_bin_days", 60)),
                    "--top-n-per-bin", str(sel.get("top_n_per_bin", 1)),
                    "--top-n", str(sel.get("top_n", 20)),
                    "--output-csv", str(seq_dir / f"selected_{seq_slug}.csv"),
                    "--output-jsonl", selected_jsonl,
                    "--output-json", str(seq_dir / f"selected_{seq_slug}.summary.json"),
                    "--output-best-json", str(seq_dir / f"selected_{seq_slug}.best.json")]
            cmds.append(Cmd("select", f"select_{seq_slug}", argv))

        if want("arc_correct"):
            argv = ["python", script("mga_spice_arc_departure_corrector_v0_1.py"),
                    "--bsp", p["bsp"], "--tpc", p["tpc"],
                    "--body-catalog", body_catalog(cfg),
                    "--input-jsonl", selected_jsonl,
                    "--central-body", spice_cfg.get("central_body", "Sun"),
                    "--mu-central-km3-s2", str(spice_cfg["mu_central_km3_s2"]),
                    "--gravitating-bodies", *grav_bodies(cfg),
                    "--dynamics-mode", "patched_heliocentric",
                    "--workers", str(integ_cfg.get("workers", 1)),
                    "--integrator", integ_cfg.get("integrator", "DOP853"),
                    "--rtol", str(integ_cfg.get("rtol", 1e-10)),
                    "--max-step-days", str(integ_cfg.get("max_step_days", 1)),
                    "--max-correction-m-s", str(vehicle_cfg.get("max_known_correction_m_s", 50)),
                    "--target-miss-km", "10",
                    "--max-nfev", "30",
                    "--output-csv", str(seq_dir / f"arc_corrected_{seq_slug}.csv"),
                    "--output-jsonl", corrected_jsonl,
                    "--output-json", str(seq_dir / f"arc_corrected_{seq_slug}.summary.json")]
            cmds.append(Cmd("arc_correct", f"arc_correct_{seq_slug}", argv))

        if want("flyby_audit"):
            argv = ["python", script("mga_flyby_closure_audit_v0_2.py"),
                    "--bsp", p["bsp"], "--tpc", p["tpc"],
                    "--body-catalog", body_catalog(cfg),
                    "--input-jsonl", corrected_jsonl,
                    "--central-body", spice_cfg.get("central_body", "Sun"),
                    "--mu-central-km3-s2", str(spice_cfg["mu_central_km3_s2"]),
                    "--gravitating-bodies", *grav_bodies(cfg),
                    "--dynamics-mode", "patched_heliocentric",
                    "--workers", str(integ_cfg.get("workers", 1)),
                    "--integrator", integ_cfg.get("integrator", "DOP853"),
                    "--rtol", str(integ_cfg.get("rtol", 1e-10)),
                    "--max-step-days", str(integ_cfg.get("max_step_days", 1)),
                    "--max-vinf-mismatch-m-s", str(select_cfg["bplane"].get("max_vinf_mismatch_m_s", 25)),
                    "--min-rp-margin-km", str(select_cfg["bplane"].get("min_rp_margin_km", 800)),
                    "--max-abs-layover-days", "0.1",
                    "--max-layover-to-soi-ratio", "0.1",
                    "--output-csv", str(seq_dir / f"flyby_closure_{seq_slug}.csv"),
                    "--output-jsonl", closure_jsonl,
                    "--output-json", str(seq_dir / f"flyby_closure_{seq_slug}.summary.json")]
            cmds.append(Cmd("flyby_audit", f"flyby_audit_{seq_slug}", argv))

        if want("bplane"):
            bp = select_cfg["bplane"]
            argv = ["python", script("mga_make_bplane_packet_v0_1.py"),
                    "--bsp", p["bsp"], "--tpc", p["tpc"],
                    "--body-catalog", body_catalog(cfg),
                    "--input-jsonl", closure_jsonl,
                    "--central-body", spice_cfg.get("central_body", "Sun"),
                    "--mu-central-km3-s2", str(spice_cfg["mu_central_km3_s2"]),
                    "--gravitating-bodies", *grav_bodies(cfg),
                    "--dynamics-mode", "patched_heliocentric",
                    "--workers", str(integ_cfg.get("workers", 1)),
                    "--integrator", integ_cfg.get("integrator", "DOP853"),
                    "--rtol", str(integ_cfg.get("rtol", 1e-10)),
                    "--max-step-days", str(integ_cfg.get("max_step_days", 1)),
                    "--max-vinf-mismatch-m-s", str(bp.get("max_vinf_mismatch_m_s", 25)),
                    "--min-rp-margin-km", str(bp.get("min_rp_margin_km", 800)),
                    "--top-n", str(bp.get("top_n", 10)),
                    "--output-csv", str(seq_dir / f"bplane_packet_{seq_slug}.csv"),
                    "--output-jsonl", str(seq_dir / f"bplane_packet_{seq_slug}.jsonl"),
                    "--output-json", str(seq_dir / f"bplane_packet_{seq_slug}.summary.json"),
                    "--output-packet-json", bplane_packet_json]
            cmds.append(Cmd("bplane", f"bplane_packet_{seq_slug}", argv))

            argv = ["python", script("mga_bplane_target_spec_v0_2.py"),
                    "--input-packet", bplane_packet_json,
                    "--body-catalog", body_catalog(cfg),
                    "--central-body", spice_cfg.get("central_body", "Sun"),
                    "--frame", spice_cfg.get("frame", "J2000"),
                    "--top-n", str(bp.get("top_n", 10)),
                    "--min-rp-margin-km", str(bp.get("min_rp_margin_km", 800)),
                    "--max-vinf-mismatch-m-s", str(bp.get("max_vinf_mismatch_m_s", 25)),
                    "--max-correction-m-s", str(bp.get("max_correction_m_s", 75)),
                    "--max-miss-after-km", str(bp.get("max_miss_after_km", 10)),
                    "--rp-soft-margin-km", str(bp.get("rp_soft_margin_km", 1500)),
                    "--vinf-soft-m-s", str(bp.get("vinf_soft_m_s", 5)),
                    "--output-csv", str(seq_dir / f"bplane_target_spec_{seq_slug}.csv"),
                    "--output-jsonl", bplane_spec_jsonl,
                    "--output-json", str(seq_dir / f"bplane_target_spec_{seq_slug}.summary.json"),
                    "--output-best-json", str(seq_dir / f"bplane_target_spec_{seq_slug}.best.json")]
            cmds.append(Cmd("bplane", f"bplane_spec_{seq_slug}", argv))

        if want("local"):
            argv = ["python", script("mga_local_flyby_target_builder_v0_1.py"),
                    "--input-spec", bplane_spec_jsonl,
                    "--body-catalog", body_catalog(cfg),
                    "--central-body", spice_cfg.get("central_body", "Sun"),
                    "--frame", spice_cfg.get("frame", "J2000"),
                    "--top-n", str(select_cfg["bplane"].get("top_n", 10)),
                    "--output-csv", str(local_dir / seq_slug / f"local_target_{seq_slug}.csv"),
                    "--output-jsonl", local_target_jsonl,
                    "--output-json", str(local_dir / seq_slug / f"local_target_{seq_slug}.summary.json"),
                    "--output-best-json", str(local_dir / seq_slug / f"local_target_{seq_slug}.best.json")]
            cmds.append(Cmd("local", f"local_target_{seq_slug}", argv))

            argv = ["python", script("mga_local_flyby_validate_v0_1.py"),
                    "--input-target", local_target_jsonl,
                    "--body-catalog", body_catalog(cfg),
                    "--top-n", str(select_cfg["bplane"].get("top_n", 10) * max(1, flyby_count)),
                    "--workers", str(integ_cfg.get("workers", 1)),
                    "--integrator", integ_cfg.get("integrator", "DOP853"),
                    "--rtol", str(integ_cfg.get("local_rtol", 1e-11)),
                    "--atol", str(integ_cfg.get("atol", 1e-13)),
                    "--max-step-hours", str(integ_cfg.get("local_max_step_hours", 0.25)),
                    "--endpoint-position-threshold-km", "1e-3",
                    "--endpoint-velocity-threshold-m-s", "1e-3",
                    "--periapsis-radius-threshold-km", "1e-3",
                    "--periapsis-time-threshold-s", "1e-2",
                    "--output-csv", str(local_dir / seq_slug / f"local_validate_{seq_slug}.csv"),
                    "--output-jsonl", local_val_jsonl,
                    "--output-json", str(local_dir / seq_slug / f"local_validate_{seq_slug}.summary.json"),
                    "--output-best-json", str(local_dir / seq_slug / f"local_validate_{seq_slug}.best.json")]
            cmds.append(Cmd("local", f"local_validate_{seq_slug}", argv))

        if want("stitch_correct"):
            argv = ["python", script("mga_stitch_multiflyby_packet_v0_1.py"),
                    "--bsp", p["bsp"], "--tpc", p["tpc"],
                    "--input-validation", local_val_jsonl,
                    "--central-body", spice_cfg.get("central_body", "Sun"),
                    "--frame", spice_cfg.get("frame", "J2000"),
                    "--min-flybys", str(max(1, flyby_count)),
                    "--top-n-groups", str(select_cfg["bplane"].get("top_n", 10)),
                    "--workers", str(integ_cfg.get("workers", 1)),
                    "--output-csv", str(local_dir / seq_slug / f"stitched_multiflyby_{seq_slug}.csv"),
                    "--output-jsonl", stitched_jsonl,
                    "--output-json", str(local_dir / seq_slug / f"stitched_multiflyby_{seq_slug}.summary.json"),
                    "--output-best-json", str(local_dir / seq_slug / f"stitched_multiflyby_{seq_slug}.best.json")]
            cmds.append(Cmd("stitch_correct", f"stitch_{seq_slug}", argv))

            argv = ["python", script("mga_multiflyby_patch_corrector_v0_2.py"),
                    "--bsp", p["bsp"], "--tpc", p["tpc"],
                    "--body-catalog", body_catalog(cfg),
                    "--stitched-jsonl", stitched_jsonl,
                    "--central-body", spice_cfg.get("central_body", "Sun"),
                    "--mu-central-km3-s2", str(spice_cfg["mu_central_km3_s2"]),
                    "--frame", spice_cfg.get("frame", "J2000"),
                    "--gravitating-bodies", *grav_bodies(cfg),
                    "--workers", str(integ_cfg.get("workers", 1)),
                    "--integrator", integ_cfg.get("integrator", "DOP853"),
                    "--rtol", str(integ_cfg.get("rtol", 1e-10)),
                    "--max-step-days", str(integ_cfg.get("max_step_days", 1)),
                    "--max-segment-correction-m-s", str(vehicle_cfg.get("max_segment_correction_m_s", 75)),
                    "--target-position-miss-km", "10",
                    "--target-velocity-miss-m-s", "250",
                    "--velocity-pass-mode", "none",
                    "--max-nfev", "30",
                    "--embed-source",
                    "--output-csv", str(final_dir / seq_slug / f"multiflyby_corrected_{seq_slug}.csv"),
                    "--output-jsonl", mf_corrected_jsonl,
                    "--output-json", str(final_dir / seq_slug / f"multiflyby_corrected_{seq_slug}.summary.json"),
                    "--output-best-json", str(final_dir / seq_slug / f"multiflyby_corrected_{seq_slug}.best.json")]
            cmds.append(Cmd("stitch_correct", f"patch_correct_{seq_slug}", argv))

        if want("sixd"):
            six = select_cfg["six_d"]
            argv = ["python", script("mga_multiflyby_6d_patch_diagnostics_v0_3.py"),
                    "--input-jsonl", mf_corrected_jsonl,
                    "--bsp", p["bsp"], "--tpc", p["tpc"],
                    "--body-catalog", body_catalog(cfg),
                    "--central-body", spice_cfg.get("central_body", "Sun"),
                    "--mu-central-km3-s2", str(spice_cfg["mu_central_km3_s2"]),
                    "--frame", spice_cfg.get("frame", "J2000"),
                    "--gravitating-bodies", *grav_bodies(cfg),
                    "--top-n", str(select_cfg["bplane"].get("top_n", 10)),
                    "--integrator", integ_cfg.get("integrator", "DOP853"),
                    "--rtol", str(integ_cfg.get("rtol", 1e-10)),
                    "--max-step-days", str(integ_cfg.get("max_step_days", 1)),
                    "--output-csv", str(final_dir / seq_slug / f"sixd_diag_{seq_slug}.csv"),
                    "--output-jsonl", diag_jsonl,
                    "--output-json", str(final_dir / seq_slug / f"sixd_diag_{seq_slug}.summary.json")]
            cmds.append(Cmd("sixd", f"sixd_diag_{seq_slug}", argv))

            argv = ["python", script("mga_select_6d_viable_multiflyby_v0_1.py"),
                    "--diagnostics-jsonl", diag_jsonl,
                    "--corrected-jsonl", mf_corrected_jsonl,
                    "--top-n", str(select_cfg["bplane"].get("top_n", 10)),
                    "--max-intermediate-velocity-m-s", str(six.get("max_intermediate_velocity_m_s", 100)),
                    "--max-position-miss-km", str(six.get("max_position_miss_km", 10)),
                    "--max-patch-dv-m-s", str(six.get("max_patch_dv_m_s", 150)),
                    "--min-rp-margin-km", str(six.get("min_rp_margin_km", 800)),
                    "--output-csv", str(final_dir / seq_slug / f"sixd_viable_{seq_slug}.csv"),
                    "--output-jsonl", viable_jsonl,
                    "--output-json", str(final_dir / seq_slug / f"sixd_viable_{seq_slug}.summary.json"),
                    "--output-best-json", str(final_dir / seq_slug / f"sixd_viable_{seq_slug}.best.json")]
            cmds.append(Cmd("sixd", f"sixd_select_{seq_slug}", argv))

        if want("export"):
            argv = ["python", script("mga_export_b6d_route_packet_v0_1.py"),
                    "--selected-jsonl", viable_jsonl,
                    "--corrected-jsonl", mf_corrected_jsonl,
                    "--diagnostics-jsonl", diag_jsonl,
                    "--top-n", "1",
                    "--max-patch-dv-m-s", str(select_cfg["six_d"].get("max_patch_dv_m_s", 150)),
                    "--max-position-miss-km", str(select_cfg["six_d"].get("max_position_miss_km", 10)),
                    "--max-intermediate-velocity-m-s", str(select_cfg["six_d"].get("max_intermediate_velocity_m_s", 100)),
                    "--min-rp-margin-km", str(select_cfg["six_d"].get("min_rp_margin_km", 800)),
                    "--embed-source",
                    "--output-csv", str(final_dir / seq_slug / f"b6d_packet_{seq_slug}.csv"),
                    "--output-jsonl", b6d_jsonl,
                    "--output-json", str(final_dir / seq_slug / f"b6d_packet_{seq_slug}.summary.json"),
                    "--output-best-json", str(final_dir / seq_slug / f"b6d_packet_{seq_slug}.best.json"),
                    "--output-md", str(final_dir / seq_slug / f"b6d_packet_{seq_slug}.md")]
            cmds.append(Cmd("export", f"export_b6d_{seq_slug}", argv))

    return cmds


def stage_order() -> List[str]:
    return [
        "body_catalog",
        "legs",
        "beam",
        "refine",
        "select",
        "arc_correct",
        "flyby_audit",
        "bplane",
        "local",
        "stitch_correct",
        "sixd",
        "export",
    ]


def parse_stage_filter(text: Optional[str]) -> Optional[set[str]]:
    if not text:
        return None
    allowed = stage_order()
    out: set[str] = set()
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if ".." in part:
            a, b = [x.strip() for x in part.split("..", 1)]
            if a not in allowed or b not in allowed:
                raise SystemExit(f"Unknown stage range: {part}")
            ia, ib = allowed.index(a), allowed.index(b)
            if ia > ib:
                ia, ib = ib, ia
            out.update(allowed[ia:ib + 1])
        else:
            if part not in allowed:
                raise SystemExit(f"Unknown stage: {part}. Allowed: {', '.join(allowed)}")
            out.add(part)
    return out


def prepare_dirs(cfg: Dict[str, Any]) -> None:
    w = work_dir(cfg)
    for rel in ["legs", "connected", "local", "final"]:
        ensure_dir(w / rel)
    for seq in cfg.get("candidate_sequences", []):
        slug = sanitize_seq(seq)
        ensure_dir(w / "connected" / slug)
        ensure_dir(w / "local" / slug)
        ensure_dir(w / "final" / slug)


def run_cmd(cmd: Cmd, dry_run: bool = False, continue_on_error: bool = False) -> int:
    print("\n" + "=" * 88)
    print(f"[{cmd.stage}] {cmd.name}")
    print(cmd.shell())
    if dry_run:
        return 0
    proc = subprocess.run(cmd.argv, cwd=str(cmd.cwd) if cmd.cwd else None)
    if proc.returncode != 0 and not continue_on_error:
        raise SystemExit(proc.returncode)
    return proc.returncode


def _leg_group_key(cmd: Cmd) -> Optional[str]:
    if cmd.stage != "legs":
        return None
    if cmd.name.startswith("scout_"):
        return cmd.name[len("scout_"):]
    if cmd.name.startswith("postprocess_"):
        return cmd.name[len("postprocess_"):]
    return None


def run_cmd_sequence(cmds: Sequence[Cmd], continue_on_error: bool = False) -> int:
    rc = 0
    for cmd in cmds:
        rc = run_cmd(cmd, dry_run=False, continue_on_error=continue_on_error)
        if rc != 0 and not continue_on_error:
            return rc
    return rc


def execute_commands(cmds: Sequence[Cmd], cfg: Dict[str, Any], continue_on_error: bool = False) -> None:
    exec_cfg = cfg.get("execution", {})
    leg_workers = int(exec_cfg.get("parallel_legs_workers", 1) or 1)
    if leg_workers <= 1:
        for cmd in cmds:
            run_cmd(cmd, dry_run=False, continue_on_error=continue_on_error)
        return

    i = 0
    while i < len(cmds):
        cmd = cmds[i]
        if cmd.stage != "legs":
            run_cmd(cmd, dry_run=False, continue_on_error=continue_on_error)
            i += 1
            continue

        # Collect the contiguous leg block and split into per-origin pipelines:
        # scout_origin -> postprocess_origin. Different origins are independent.
        j = i
        block: List[Cmd] = []
        while j < len(cmds) and cmds[j].stage == "legs":
            block.append(cmds[j])
            j += 1
        groups: Dict[str, List[Cmd]] = {}
        order: List[str] = []
        for c in block:
            key = _leg_group_key(c) or c.name
            if key not in groups:
                groups[key] = []
                order.append(key)
            groups[key].append(c)

        print("\n" + "=" * 88)
        print(f"[legs] parallel leg-library generation: {len(order)} origins, workers={leg_workers}")
        with ThreadPoolExecutor(max_workers=leg_workers) as ex:
            futs = {ex.submit(run_cmd_sequence, groups[k], continue_on_error): k for k in order}
            for fut in as_completed(futs):
                k = futs[fut]
                try:
                    rc = fut.result()
                except BaseException as e:
                    if continue_on_error:
                        print(f"[ERROR] leg group {k} failed: {e}")
                    else:
                        raise
                else:
                    if rc != 0 and not continue_on_error:
                        raise SystemExit(rc)
                    print(f"[OK] leg group completed: {k}")
        i = j


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Mission-level orchestrator for the KSP/Principia MGA pipeline.")
    ap.add_argument("--spec", required=True, help="Mission spec JSON.")
    ap.add_argument("--stages", default=None, help="Comma-separated stages or ranges, e.g. legs,beam or refine..export")
    ap.add_argument("--plan", action="store_true", help="Print commands only; do not execute.")
    ap.add_argument("--run", action="store_true", help="Execute commands.")
    ap.add_argument("--continue-on-error", action="store_true", help="Continue running after a command fails.")
    ap.add_argument("--write-plan", default=None, help="Optional path to write a shell plan.")
    args = ap.parse_args(argv)

    if not args.plan and not args.run:
        args.plan = True

    spec_path = Path(args.spec)
    cfg = load_json(spec_path)
    prepare_dirs(cfg)
    stages = parse_stage_filter(args.stages)
    cmds = build_commands(cfg, stages=stages)

    if args.write_plan:
        plan_path = Path(args.write_plan)
        ensure_dir(plan_path.parent)
        with plan_path.open("w", encoding="utf-8") as f:
            f.write("#!/usr/bin/env bash\nset -euo pipefail\n\n")
            f.write(f"# Generated from {spec_path}\n\n")
            for cmd in cmds:
                f.write(f"echo {shlex.quote('[' + cmd.stage + '] ' + cmd.name)}\n")
                f.write(cmd.shell() + "\n\n")
        plan_path.chmod(0o755)
        print(f"[OK] wrote plan: {plan_path}")

    if args.plan:
        print("=" * 88)
        print("MGA PIPELINE ORCHESTRATOR V0.2 — PLAN")
        print(f"Spec:   {spec_path}")
        print(f"Stages: {', '.join(sorted(stages)) if stages else 'all'}")
        print(f"Commands: {len(cmds)}")
        print("=" * 88)
        for cmd in cmds:
            print(f"\n# [{cmd.stage}] {cmd.name}")
            print(cmd.shell())

    if args.run:
        execute_commands(cmds, cfg, continue_on_error=args.continue_on_error)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
