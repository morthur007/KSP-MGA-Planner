#!/usr/bin/env python3
"""
MGA Route Database Builder V0.2

Builds a fast SQLite route atlas from validated route packet JSON/JSONL files.
V0.2 fixes the most common V0.1 atlas issue: accidentally indexing a leg/local
TOF instead of the full route TOF. It prioritizes explicit route-level metrics,
then sums segment TOFs, then falls back to max-min absolute event times.

The parser remains schema-tolerant and stores the original packet JSON.
"""
from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import math
import re
import sqlite3
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

SCHEMA_VERSION = "mga_route_db_v0_2"
DAY_S = 86400.0


def fnum(x: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if x is None or x == "":
            return default
        v = float(x)
        return v if math.isfinite(v) else default
    except Exception:
        return default


def sanitize(obj: Any) -> Any:
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {str(k): sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize(v) for v in obj]
    return obj


def read_json_or_jsonl(path: Path) -> List[Dict[str, Any]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if path.suffix.lower() == ".jsonl":
        rows: List[Dict[str, Any]] = []
        for line_no, line in enumerate(text.splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    rows.append(obj)
            except Exception as e:
                print(f"[WARN] could not parse {path}:{line_no}: {e}")
        return rows
    obj = json.loads(text)
    if isinstance(obj, list):
        return [x for x in obj if isinstance(x, dict)]
    if isinstance(obj, dict):
        for key in ("routes", "packets", "records", "items", "selected_routes"):
            if isinstance(obj.get(key), list):
                return [x for x in obj[key] if isinstance(x, dict)]
        return [obj]
    return []


def expand_inputs(patterns: List[str]) -> List[Path]:
    paths: List[Path] = []
    for p in patterns:
        hits = [Path(x) for x in glob.glob(p)]
        if hits:
            paths.extend(hits)
        else:
            pp = Path(p)
            if pp.exists():
                paths.append(pp)
            else:
                print(f"[WARN] input pattern matched nothing: {p}")
    out: List[Path] = []
    seen = set()
    for p in sorted(paths):
        rp = str(p.resolve())
        if rp not in seen:
            out.append(p)
            seen.add(rp)
    return out


def normalize_sequence(seq: Any) -> str:
    if isinstance(seq, str):
        if "->" in seq:
            return " -> ".join(x.strip() for x in seq.split("->") if x.strip())
        if "," in seq:
            return " -> ".join(x.strip() for x in seq.split(",") if x.strip())
        return seq.strip()
    if isinstance(seq, list):
        return " -> ".join(str(x) for x in seq)
    return ""


def split_sequence(seq: str) -> List[str]:
    return [x.strip() for x in seq.replace(",", "->").split("->") if x.strip()]


def find_first(obj: Any, keys: Iterable[str], max_depth: int = 8) -> Any:
    keys = set(keys)

    def rec(x: Any, depth: int) -> Any:
        if depth > max_depth:
            return None
        if isinstance(x, dict):
            for k in keys:
                if k in x and x[k] not in (None, ""):
                    return x[k]
            for v in x.values():
                got = rec(v, depth + 1)
                if got is not None:
                    return got
        elif isinstance(x, list):
            for v in x:
                got = rec(v, depth + 1)
                if got is not None:
                    return got
        return None

    return rec(obj, 0)


def get_path(obj: Any, path: str) -> Any:
    cur = obj
    for part in path.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return None
    return cur


def find_direct_metric(packet: Dict[str, Any], keys: Iterable[str]) -> Optional[float]:
    # Search only likely route-level containers first. This prevents grabbing the
    # first leg/local `tof_days` buried in an embedded segment.
    containers = [
        packet,
        packet.get("metrics"),
        packet.get("summary"),
        packet.get("route_metrics"),
        packet.get("route_summary"),
        packet.get("selected_summary"),
        get_path(packet, "source.metrics"),
        get_path(packet, "source.summary"),
        get_path(packet, "source_packet.metrics"),
        get_path(packet, "source_packet.summary"),
        get_path(packet, "source_corrected.metrics"),
        get_path(packet, "source_corrected.summary"),
        get_path(packet, "source_route.metrics"),
        get_path(packet, "source_route.summary"),
        packet.get("route"),
        packet.get("source_route"),
    ]
    for c in containers:
        if isinstance(c, dict):
            for k in keys:
                v = fnum(c.get(k))
                if v is not None:
                    return v
    return None


def walk(obj: Any, path: str = "", max_depth: int = 12) -> Iterable[Tuple[str, Any]]:
    if max_depth < 0:
        return
    yield path, obj
    if isinstance(obj, dict):
        for k, v in obj.items():
            np = f"{path}.{k}" if path else str(k)
            yield from walk(v, np, max_depth - 1)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            np = f"{path}[{i}]"
            yield from walk(v, np, max_depth - 1)


def extract_segment_tof_days(packet: Dict[str, Any]) -> Optional[float]:
    segment_list_names = (
        "segments", "segment_corrections", "corrected_segments", "patched_segments",
        "legs", "arcs", "trajectory_segments", "events"
    )
    candidates: List[float] = []

    for path, obj in walk(packet, max_depth=10):
        if not isinstance(obj, list):
            continue
        last_name = re.sub(r"\[\d+\]", "", path.split(".")[-1]).lower()
        if last_name not in segment_list_names:
            continue
        vals: List[float] = []
        for item in obj:
            if not isinstance(item, dict):
                continue
            for k in ("tof_days", "duration_days", "time_of_flight_days", "dt_days"):
                v = fnum(item.get(k))
                if v is not None and 0 <= v < 20000:
                    vals.append(v)
                    break
            else:
                for k in ("tof_s", "duration_s", "time_of_flight_s", "dt_s"):
                    v = fnum(item.get(k))
                    if v is not None and 0 <= v < 2e9:
                        vals.append(v / DAY_S)
                        break
        if len(vals) >= 2:
            candidates.append(sum(vals))

    if candidates:
        # Prefer the largest plausible sum. It is usually route total, while
        # smaller candidates can be nested local SOI segments.
        return max(candidates)
    return None


def extract_tof_from_absolute_times(packet: Dict[str, Any]) -> Tuple[Optional[float], Optional[float], Optional[float], str]:
    times: List[float] = []
    for path, val in walk(packet, max_depth=12):
        if not isinstance(val, (int, float)) or not math.isfinite(float(val)):
            continue
        key = path.split(".")[-1].lower()
        # Avoid durations and small counters. Look for ET/UT/epoch fields.
        if any(tok in key for tok in ("et", "epoch", "ut", "jd")) and not any(tok in key for tok in ("tof", "dt", "duration")):
            v = float(val)
            if abs(v) < 1e12:
                times.append(v)

    if len(times) >= 2:
        mn, mx = min(times), max(times)
        dt = mx - mn
        if 0 <= dt < 2e9:
            return dt / DAY_S, mn, mx, "absolute_time_span"
    return None, None, None, "missing"


def extract_route_tof(packet: Dict[str, Any]) -> Tuple[Optional[float], Optional[float], Optional[float], str]:
    direct_days = find_direct_metric(packet, [
        "route_tof_days", "mission_tof_days", "total_route_tof_days", "total_tof_days",
        "total_time_of_flight_days", "time_of_flight_days", "tof_total_days"
    ])
    if direct_days is not None:
        return direct_days, None, None, "route_level_days"

    direct_s = find_direct_metric(packet, [
        "route_tof_s", "mission_tof_s", "total_route_tof_s", "total_tof_s",
        "total_time_of_flight_s", "time_of_flight_s", "tof_total_s"
    ])
    if direct_s is not None:
        return direct_s / DAY_S, None, None, "route_level_seconds"

    summed = extract_segment_tof_days(packet)
    if summed is not None:
        return summed, None, None, "sum_segment_tofs"

    return extract_tof_from_absolute_times(packet)


def extract_depart_arrival(packet: Dict[str, Any]) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    depart_et = find_direct_metric(packet, ["depart_et", "departure_et", "depart_time_et", "t0_et", "launch_et"])
    arrival_et = find_direct_metric(packet, ["arrival_et", "final_et", "arrival_time_et", "tf_et", "target_et"])
    depart_day = find_direct_metric(packet, ["depart_day", "depart_days", "depart_days_from_coverage_start", "launch_day"])

    # Fall back recursively for these; if a wrong value is found it is less damaging than TOF.
    if depart_et is None:
        depart_et = fnum(find_first(packet, ["depart_et", "departure_et", "depart_time_et", "t0_et", "launch_et"], max_depth=8))
    if arrival_et is None:
        arrival_et = fnum(find_first(packet, ["arrival_et", "final_et", "arrival_time_et", "tf_et", "target_et"], max_depth=8))
    if depart_day is None:
        depart_day = fnum(find_first(packet, ["depart_day", "depart_days", "depart_days_from_coverage_start", "launch_day"], max_depth=8))
    return depart_et, arrival_et, depart_day


def stable_route_id(packet: Dict[str, Any], source_path: str, index: int) -> str:
    for key in ("route_id", "packet_id", "id", "correction_id", "selection_id"):
        v = find_first(packet, [key], max_depth=4)
        if isinstance(v, str) and v:
            return v
    seq = normalize_sequence(find_first(packet, ["sequence", "bodies", "body_sequence"], max_depth=5))
    payload = json.dumps(sanitize(packet), sort_keys=True, ensure_ascii=False, separators=(",", ":"))[:5000]
    h = hashlib.sha1(f"{source_path}:{index}:{seq}:{payload}".encode("utf-8")).hexdigest()[:16]
    return f"route_{h}"


def extract_flybys(packet: Dict[str, Any], seq: str) -> Tuple[int, Optional[float], Optional[float], str]:
    flyby_dicts: List[Dict[str, Any]] = []
    for key in ("flybys", "flyby_targets", "local_flybys"):
        v = find_first(packet, [key], max_depth=5)
        if isinstance(v, list) and v:
            flyby_dicts = [x for x in v if isinstance(x, dict)]
            break

    min_rp_margin = None
    min_altitude = None
    bodies: List[str] = []

    for f in flyby_dicts:
        b = f.get("body") or f.get("flyby_body") or f.get("target_body")
        if b:
            bodies.append(str(b))
        for k in ("rp_margin_km", "min_rp_margin_km", "periapsis_margin_km"):
            val = fnum(f.get(k))
            if val is not None:
                min_rp_margin = val if min_rp_margin is None else min(min_rp_margin, val)
        for k in ("altitude_km", "periapsis_altitude_km", "alt_km"):
            val = fnum(f.get(k))
            if val is not None:
                min_altitude = val if min_altitude is None else min(min_altitude, val)

    if min_rp_margin is None:
        min_rp_margin = fnum(find_first(packet, ["min_rp_margin_km", "rp_margin_km", "minimum_rp_margin_km"], max_depth=8))
    if min_altitude is None:
        min_altitude = fnum(find_first(packet, ["min_altitude_km", "altitude_km", "periapsis_altitude_km"], max_depth=8))

    if not bodies:
        parts = split_sequence(seq)
        bodies = parts[1:-1] if len(parts) >= 3 else []

    return len(bodies), min_rp_margin, min_altitude, ",".join(bodies)


def extract_route(packet: Dict[str, Any], source_path: Path, index: int) -> Dict[str, Any]:
    seq = normalize_sequence(find_first(packet, ["sequence", "bodies", "body_sequence", "route_sequence"], max_depth=8))
    parts = split_sequence(seq)
    route_id = stable_route_id(packet, str(source_path), index)
    n_flybys, min_rp_margin, min_altitude, flyby_bodies = extract_flybys(packet, seq)

    cls = find_first(packet, ["class", "route_class", "validation_class"], max_depth=6)
    pass_flag = find_first(packet, ["pass", "valid", "ready", "pass_manifest", "pass_validation"], max_depth=4)
    if isinstance(pass_flag, str):
        pass_flag = pass_flag.lower() in ("true", "yes", "1", "pass", "ok")
    elif pass_flag is None:
        pass_flag = True if cls in ("A", "B", "B6D", "C6D") else None

    patch_dv = fnum(find_first(packet, [
        "patch_dv_m_s", "total_patch_dv_m_s", "total_segment_correction_m_s",
        "total_correction_m_s", "known_correction_m_s", "correction_m_s"
    ], max_depth=8))
    pos_miss = fnum(find_first(packet, [
        "max_position_miss_km", "max_pos_miss_km", "max_miss_after_km",
        "position_miss_km", "max_endpoint_miss_km"
    ], max_depth=8))
    v_int = fnum(find_first(packet, [
        "max_intermediate_velocity_m_s", "intermediate_velocity_m_s",
        "max_intermediate_velocity_mismatch_m_s", "v_int_m_s"
    ], max_depth=8))
    v_final = fnum(find_first(packet, [
        "final_vinf_m_s", "arrival_vinf_m_s", "v_final_m_s", "max_final_velocity_m_s"
    ], max_depth=8))

    c3 = fnum(find_first(packet, ["c3_km2_s2", "launch_c3_km2_s2", "C3"], max_depth=8))
    vinf_dep = fnum(find_first(packet, ["vinf_depart_km_s", "launch_vinf_km_s", "departure_vinf_km_s"], max_depth=8))
    vinf_arr = fnum(find_first(packet, ["vinf_arrive_km_s", "arrival_vinf_km_s"], max_depth=8))

    tof_days, span_depart, span_arrival, tof_source = extract_route_tof(packet)
    depart_et, arrival_et, depart_day = extract_depart_arrival(packet)
    if depart_et is None:
        depart_et = span_depart
    if arrival_et is None:
        arrival_et = span_arrival

    score = fnum(find_first(packet, ["score", "robust_score", "objective", "obj"], max_depth=4))
    if score is None:
        score = (patch_dv or 0.0) + 0.05 * (v_int or 0.0)
        if min_rp_margin is not None and min_rp_margin < 1500:
            score += (1500 - min_rp_margin) / 150.0

    risk_flags: List[str] = []
    if cls in ("C", "C6D", "D"):
        risk_flags.append(f"class_{cls}")
    if v_int is not None and v_int > 100:
        risk_flags.append("high_intermediate_velocity_mismatch")
    if min_rp_margin is not None and min_rp_margin < 1000:
        risk_flags.append("low_flyby_margin")
    if patch_dv is not None and patch_dv > 75:
        risk_flags.append("high_patch_dv")
    if tof_source not in ("route_level_days", "route_level_seconds"):
        risk_flags.append(f"tof_{tof_source}")

    return {
        "route_id": route_id,
        "sequence": seq,
        "origin": parts[0] if parts else None,
        "target": parts[-1] if parts else None,
        "depth": max(0, len(parts) - 1),
        "flyby_count": n_flybys,
        "flyby_bodies": flyby_bodies,
        "class": str(cls) if cls is not None else None,
        "pass_flag": None if pass_flag is None else int(bool(pass_flag)),
        "score": score,
        "tof_days": tof_days,
        "tof_source": tof_source,
        "depart_et": depart_et,
        "arrival_et": arrival_et,
        "depart_day": depart_day,
        "c3_km2_s2": c3,
        "vinf_depart_km_s": vinf_dep,
        "vinf_arrive_km_s": vinf_arr,
        "patch_dv_m_s": patch_dv,
        "position_miss_km": pos_miss,
        "intermediate_velocity_m_s": v_int,
        "final_vinf_m_s": v_final,
        "min_rp_margin_km": min_rp_margin,
        "min_altitude_km": min_altitude,
        "risk_flags": ";".join(risk_flags),
        "source_file": str(source_path),
        "source_index": index,
        "packet_json": json.dumps(sanitize(packet), ensure_ascii=False, separators=(",", ":")),
    }


def init_db(conn: sqlite3.Connection, replace: bool = False) -> None:
    cur = conn.cursor()
    if replace:
        cur.execute("DROP TABLE IF EXISTS routes")
        cur.execute("DROP TABLE IF EXISTS metadata")
    cur.execute("""
    CREATE TABLE IF NOT EXISTS routes (
        route_id TEXT PRIMARY KEY,
        sequence TEXT,
        origin TEXT,
        target TEXT,
        depth INTEGER,
        flyby_count INTEGER,
        flyby_bodies TEXT,
        class TEXT,
        pass_flag INTEGER,
        score REAL,
        tof_days REAL,
        tof_source TEXT,
        depart_et REAL,
        arrival_et REAL,
        depart_day REAL,
        c3_km2_s2 REAL,
        vinf_depart_km_s REAL,
        vinf_arrive_km_s REAL,
        patch_dv_m_s REAL,
        position_miss_km REAL,
        intermediate_velocity_m_s REAL,
        final_vinf_m_s REAL,
        min_rp_margin_km REAL,
        min_altitude_km REAL,
        risk_flags TEXT,
        source_file TEXT,
        source_index INTEGER,
        packet_json TEXT
    )
    """)
    cur.execute("""CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT)""")
    for idx in [
        "target", "sequence", "class", "score", "tof_days", "c3_km2_s2",
        "patch_dv_m_s", "min_rp_margin_km", "depart_day", "tof_source"
    ]:
        cur.execute(f"CREATE INDEX IF NOT EXISTS idx_routes_{idx} ON routes({idx})")
    conn.commit()


def upsert_routes(conn: sqlite3.Connection, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    cols = list(rows[0].keys())
    placeholders = ",".join("?" for _ in cols)
    update = ",".join(f"{c}=excluded.{c}" for c in cols if c != "route_id")
    sql = f"""
    INSERT INTO routes ({','.join(cols)}) VALUES ({placeholders})
    ON CONFLICT(route_id) DO UPDATE SET {update}
    """
    conn.executemany(sql, [[r.get(c) for c in cols] for r in rows])
    conn.commit()


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    cols = [c for c in rows[0].keys() if c != "packet_json"]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c) for c in cols})


def main() -> int:
    ap = argparse.ArgumentParser(description="Build a SQLite route atlas from MGA route packets.")
    ap.add_argument("--input", nargs="+", required=True, help="JSON/JSONL route packet files or glob patterns")
    ap.add_argument("--output-db", required=True)
    ap.add_argument("--output-csv")
    ap.add_argument("--output-json")
    ap.add_argument("--replace", action="store_true", help="Drop existing routes table first")
    ap.add_argument("--mission-name", default="unknown")
    ap.add_argument("--system-id", default="opm_mpe_principia_spice_v0_1")
    args = ap.parse_args()

    paths = expand_inputs(args.input)
    rows: List[Dict[str, Any]] = []
    for path in paths:
        packets = read_json_or_jsonl(path)
        for i, packet in enumerate(packets, start=1):
            try:
                rows.append(extract_route(packet, path, i))
            except Exception as e:
                print(f"[WARN] failed to extract {path}:{i}: {e}")

    rows.sort(key=lambda r: (r.get("score") is None, r.get("score") or 1e99, r.get("route_id") or ""))

    db_path = Path(args.output_db)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    init_db(conn, replace=args.replace)
    upsert_routes(conn, rows)
    conn.execute("INSERT OR REPLACE INTO metadata(key,value) VALUES (?,?)", ("schema_version", SCHEMA_VERSION))
    conn.execute("INSERT OR REPLACE INTO metadata(key,value) VALUES (?,?)", ("mission_name", args.mission_name))
    conn.execute("INSERT OR REPLACE INTO metadata(key,value) VALUES (?,?)", ("system_id", args.system_id))
    conn.commit()

    if args.output_csv:
        write_csv(Path(args.output_csv), rows)

    summary = {
        "schema_version": SCHEMA_VERSION,
        "mission_name": args.mission_name,
        "system_id": args.system_id,
        "input_files": [str(p) for p in paths],
        "routes_loaded": len(rows),
        "by_target": {},
        "by_class": {},
        "by_tof_source": {},
        "top_routes": [{k: v for k, v in r.items() if k != "packet_json"} for r in rows[:10]],
    }
    for r in rows:
        summary["by_target"][r.get("target") or "unknown"] = summary["by_target"].get(r.get("target") or "unknown", 0) + 1
        summary["by_class"][r.get("class") or "unknown"] = summary["by_class"].get(r.get("class") or "unknown", 0) + 1
        summary["by_tof_source"][r.get("tof_source") or "unknown"] = summary["by_tof_source"].get(r.get("tof_source") or "unknown", 0) + 1

    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_json).write_text(json.dumps(sanitize(summary), indent=2, ensure_ascii=False), encoding="utf-8")

    print("=" * 80)
    print("MGA ROUTE DATABASE BUILDER V0.2")
    print("=" * 80)
    print(f"Input files:   {len(paths)}")
    print(f"Routes loaded: {len(rows)}")
    print(f"DB:            {db_path}")
    if args.output_csv:
        print(f"CSV:           {args.output_csv}")
    if args.output_json:
        print(f"Summary:       {args.output_json}")
    print("By class:", summary["by_class"])
    print("By target:", summary["by_target"])
    print("By TOF source:", summary["by_tof_source"])
    print("Top routes:")
    for i, r in enumerate(rows[:10], start=1):
        print(
            f" {i}. {r['sequence']} | class={r['class']} | score={r['score']} | "
            f"TOF={r['tof_days']} d ({r['tof_source']}) | dv={r['patch_dv_m_s']} m/s | "
            f"rpM={r['min_rp_margin_km']} km"
        )
    print("=" * 80)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
