#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

from google.protobuf.message import Message
from serialization import ksp_plugin_pb2


def scalar_value(m: Any) -> float | None:
    """Extrai magnitude de Quantity/Scalar-like protobuf messages."""
    if m is None:
        return None

    if hasattr(m, "magnitude"):
        try:
            return float(m.magnitude)
        except Exception:
            pass

    if isinstance(m, (int, float)):
        return float(m)

    if not isinstance(m, Message):
        return None

    for field, value in m.ListFields():
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, Message):
            x = scalar_value(value)
            if x is not None:
                return x

    return None


def has_xyz(m: Message) -> bool:
    fields = m.DESCRIPTOR.fields_by_name
    return "x" in fields and "y" in fields and "z" in fields


def extract_xyz_here(m: Message) -> list[float] | None:
    if not has_xyz(m):
        return None

    out = []
    for axis in ("x", "y", "z"):
        x = scalar_value(getattr(m, axis))
        if x is None or not math.isfinite(x):
            return None
        out.append(x)

    return out


def collect_vectors(m: Message, path: str = "") -> list[tuple[str, list[float]]]:
    """Coleta qualquer submensagem com campos x/y/z."""
    out: list[tuple[str, list[float]]] = []

    here = extract_xyz_here(m)
    if here is not None:
        out.append((path, here))

    for field, value in m.ListFields():
        name = field.name
        p = f"{path}.{name}" if path else name

        if getattr(field, "is_repeated", False):
            for i, item in enumerate(value):
                if isinstance(item, Message):
                    out.extend(collect_vectors(item, f"{p}[{i}]"))
        else:
            if isinstance(value, Message):
                out.extend(collect_vectors(value, p))

    return out


def get_message_if_present(m: Message, name: str) -> Message | None:
    if name not in m.DESCRIPTOR.fields_by_name:
        return None
    try:
        if not m.HasField(name):
            return None
    except ValueError:
        pass
    value = getattr(m, name)
    return value if isinstance(value, Message) else None


def dof_to_rv(dof: Message) -> tuple[list[float] | None, list[float] | None, list[str]]:
    """
    DegreesOfFreedom no Principia normalmente tem t1/t2.
    t1 tende a ser posição; t2 tende a ser velocidade.
    Faz fallback por busca de vetores.
    """
    debug_paths: list[str] = []

    t1 = get_message_if_present(dof, "t1")
    t2 = get_message_if_present(dof, "t2")

    if t1 is not None and t2 is not None:
        v1 = collect_vectors(t1, "t1")
        v2 = collect_vectors(t2, "t2")
        if v1 and v2:
            debug_paths.extend([v1[0][0], v2[0][0]])
            return v1[0][1], v2[0][1], debug_paths

    vectors = collect_vectors(dof, "degrees_of_freedom")
    debug_paths.extend([p for p, _ in vectors[:4]])

    if len(vectors) >= 2:
        return vectors[0][1], vectors[1][1], debug_paths

    return None, None, debug_paths


def point_time_s(point: Message) -> float | None:
    instant = get_message_if_present(point, "instant")
    if instant is None:
        return None
    return scalar_value(instant)


def latest_point_from_trajectory(traj: Message) -> Message | None:
    """
    Primeiro tenta trajectory.checkpoint[-1].last_point.
    """
    if "checkpoint" not in traj.DESCRIPTOR.fields_by_name:
        return None

    checkpoints = getattr(traj, "checkpoint")
    if len(checkpoints) == 0:
        return None

    cp = checkpoints[-1]

    lp = get_message_if_present(cp, "last_point")
    if lp is not None:
        return lp

    # Fallback: busca qualquer campo last_point por reflexão.
    for field, value in cp.ListFields():
        if field.name == "last_point" and isinstance(value, Message):
            return value

    return None


def sub3(a: list[float], b: list[float]) -> list[float]:
    return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pb", default="data/jnsq_gate0/principia_plugin_uncompressed.pb")
    ap.add_argument("--central-body", default="Sun")
    ap.add_argument("--output-json", default="data/jnsq_gate0/snapshot_principia_native.json")
    ap.add_argument("--output-csv", default="data/jnsq_gate0/snapshot_principia_native_report.csv")
    args = ap.parse_args()

    pb = Path(args.pb)
    msg = ksp_plugin_pb2.Plugin()
    msg.ParseFromString(pb.read_bytes())

    current_time_s = scalar_value(msg.current_time)
    game_epoch_s = scalar_value(msg.game_epoch)
    epoch_ut_s = None
    if current_time_s is not None and game_epoch_s is not None:
        epoch_ut_s = current_time_s - game_epoch_s

    eph = msg.ephemeris

    bodies = list(eph.body)
    trajectories = list(eph.trajectory)

    print(f"[INFO] bodies={len(bodies)} trajectories={len(trajectories)}")
    print(f"[INFO] current_time_s={current_time_s}")
    print(f"[INFO] game_epoch_s={game_epoch_s}")
    print(f"[INFO] epoch_ut_s={epoch_ut_s}")

    if len(bodies) != len(trajectories):
        print("[WARN] body/trajectory têm tamanhos diferentes; vou usar min(len).")

    rows = []
    raw_states: dict[str, dict[str, Any]] = {}

    n = min(len(bodies), len(trajectories))

    for i in range(n):
        body = bodies[i]
        traj = trajectories[i]

        name = getattr(body, "name", f"body_{i}")
        mu = scalar_value(body.gravitational_parameter)

        point = latest_point_from_trajectory(traj)
        if point is None:
            rows.append({
                "index": i,
                "body": name,
                "status": "no_latest_point",
            })
            continue

        dof = get_message_if_present(point, "degrees_of_freedom")
        if dof is None:
            rows.append({
                "index": i,
                "body": name,
                "status": "no_degrees_of_freedom",
            })
            continue

        r, v, paths = dof_to_rv(dof)
        t = point_time_s(point)

        if r is None or v is None:
            rows.append({
                "index": i,
                "body": name,
                "time_s": t,
                "status": "no_rv",
                "paths": ";".join(paths),
            })
            continue

        raw_states[name] = {
            "index": i,
            "mu": mu,
            "r_barycentric": r,
            "v_barycentric": v,
            "dof_time_s": t,
            "debug_paths": paths,
        }

        rows.append({
            "index": i,
            "body": name,
            "time_s": t,
            "mu": mu,
            "x": r[0],
            "y": r[1],
            "z": r[2],
            "vx": v[0],
            "vy": v[1],
            "vz": v[2],
            "paths": ";".join(paths),
            "status": "ok",
        })

    if args.central_body not in raw_states:
        print(f"[FIX] {args.central_body} (Corpo Central) não possui vetores móveis. Ancorando na origem [0,0,0].")
        # Caçamos o 'mu' (massa) do Sol direto da lista de corpos
        sun_mu = 0.0
        sun_idx = 0
        for idx, b in enumerate(bodies):
            if getattr(b, "name", "") == args.central_body:
                sun_mu = scalar_value(b.gravitational_parameter)
                sun_idx = idx
                break
        
        # Injetamos o Sol no dicionário forçando a posição [0, 0, 0]
        raw_states[args.central_body] = {
            "index": sun_idx,
            "mu": sun_mu,
            "r_barycentric": [0.0, 0.0, 0.0],
            "v_barycentric": [0.0, 0.0, 0.0],
            "dof_time_s": epoch_ut_s,
            "debug_paths": ["forced_origin"]
        }

    c = raw_states[args.central_body]
    cr = c["r_barycentric"]
    cv = c["v_barycentric"]

    snapshot_bodies: dict[str, Any] = {}

    for name, st in raw_states.items():
        r = sub3(st["r_barycentric"], cr)
        v = sub3(st["v_barycentric"], cv)

        if name == args.central_body:
            r = [0.0, 0.0, 0.0]
            v = [0.0, 0.0, 0.0]

        snapshot_bodies[name] = {
            "r": r,
            "v": v,
            "mu": st["mu"],
            "gravitational_parameter": st["mu"],
            "principia_body_index": st["index"],
            "principia_dof_time_s": st["dof_time_s"],
        }

    snapshot = {
        "schema": "principia_native_snapshot.v0_from_plugin_pb",
        "source_pb": str(pb),
        "reference_body": args.central_body,
        "frame_note": "Principia barycentric states shifted to central body; no axis swap applied.",
        "epoch_ut_s": epoch_ut_s,
        "principia_current_time_s": current_time_s,
        "principia_game_epoch_s": game_epoch_s,
        "bodies": snapshot_bodies,
    }

    out_json = Path(args.output_json)
    out_csv = Path(args.output_csv)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    out_json.write_text(json.dumps(snapshot, indent=2, ensure_ascii=False), encoding="utf-8")

    with out_csv.open("w", newline="") as f:
        fieldnames = [
            "index", "body", "time_s", "mu",
            "x", "y", "z", "vx", "vy", "vz",
            "paths", "status",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    ok = sum(1 for r in rows if r.get("status") == "ok")
    print(f"[OK] estados extraídos: {ok}/{n}")
    print(f"[OK] snapshot: {out_json}")
    print(f"[OK] relatório: {out_csv}")

    times = sorted({
        round(float(st["principia_dof_time_s"]), 9)
        for st in snapshot_bodies.values()
        if st["principia_dof_time_s"] is not None
    })
    print(f"[INFO] tempos DOF únicos: {len(times)}")
    print("[INFO] primeiros tempos:", times[:5])
    print("[INFO] últimos tempos:", times[-5:])

    if current_time_s is not None and times:
        max_dt = max(abs(t - current_time_s) for t in times)
        print(f"[CHECK] max |dof_time - current_time| = {max_dt:.9e} s")
        if max_dt > 1e-6:
            print("[WARN] Os DOFs extraídos não estão exatamente em current_time_s.")
            print("[WARN] Talvez precisemos avaliar a timeline zfp via C++ Ephemeris, não só usar last_point.")


if __name__ == "__main__":
    main()
