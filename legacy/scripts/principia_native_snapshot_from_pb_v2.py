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


NUMERIC_TYPES = {
    1,   # double
    2,   # float
    3,   # int64
    4,   # uint64
    5,   # int32
    13,  # uint32
    15,  # sint32
    16,  # sint64
}


def field_is_repeated(field: Any) -> bool:
    return bool(getattr(field, "is_repeated", False))


def get_field(m: Message, name: str) -> tuple[Any | None, Any | None]:
    field = m.DESCRIPTOR.fields_by_name.get(name)
    if field is None:
        return None, None
    return field, getattr(m, name)


def scalar_value(x: Any) -> float | None:
    if x is None:
        return None

    if isinstance(x, (int, float)):
        return float(x)

    if not isinstance(x, Message):
        return None

    if "magnitude" in x.DESCRIPTOR.fields_by_name:
        try:
            return float(getattr(x, "magnitude"))
        except Exception:
            pass

    for field, value in x.ListFields():
        if field.type in NUMERIC_TYPES:
            return float(value)
        if isinstance(value, Message):
            y = scalar_value(value)
            if y is not None:
                return y

    return None


def message_nonempty(m: Any) -> bool:
    return isinstance(m, Message) and len(m.ListFields()) > 0


def as_message_list(parent: Message, field_name: str) -> list[Message]:
    field, value = get_field(parent, field_name)
    if field is None:
        return []

    if field_is_repeated(field):
        return [v for v in value if isinstance(v, Message)]

    if isinstance(value, Message) and message_nonempty(value):
        return [value]

    return []


def get_message(parent: Message, field_name: str) -> Message | None:
    field, value = get_field(parent, field_name)
    if field is None:
        return None

    if field_is_repeated(field):
        return None

    if isinstance(value, Message) and message_nonempty(value):
        return value

    # Mesmo vazio, às vezes a submensagem existe conceitualmente.
    if isinstance(value, Message):
        return value

    return None


def find_messages_named(m: Message, wanted: str, path: str = "") -> list[tuple[str, Message]]:
    out: list[tuple[str, Message]] = []

    for field, value in m.ListFields():
        name = field.name
        base = f"{path}.{name}" if path else name

        if field_is_repeated(field):
            for i, item in enumerate(value):
                if isinstance(item, Message):
                    p = f"{base}[{i}]"
                    if name == wanted or item.DESCRIPTOR.name.lower() == wanted.lower():
                        out.append((p, item))
                    out.extend(find_messages_named(item, wanted, p))
        else:
            if isinstance(value, Message):
                if name == wanted or value.DESCRIPTOR.name.lower() == wanted.lower():
                    out.append((base, value))
                out.extend(find_messages_named(value, wanted, base))

    return out


def collect_xyz_vectors(m: Message, path: str = "") -> list[tuple[str, list[float]]]:
    out: list[tuple[str, list[float]]] = []

    fields = m.DESCRIPTOR.fields_by_name
    if all(axis in fields for axis in ("x", "y", "z")):
        vals = []
        ok = True
        for axis in ("x", "y", "z"):
            val = scalar_value(getattr(m, axis))
            if val is None or not math.isfinite(val):
                ok = False
                break
            vals.append(val)
        if ok:
            out.append((path, vals))

    for field, value in m.ListFields():
        name = field.name
        base = f"{path}.{name}" if path else name

        if field_is_repeated(field):
            for i, item in enumerate(value):
                if isinstance(item, Message):
                    out.extend(collect_xyz_vectors(item, f"{base}[{i}]"))
        else:
            if isinstance(value, Message):
                out.extend(collect_xyz_vectors(value, base))

    return out


def latest_point_from_trajectory(traj: Message) -> tuple[Message | None, str]:
    checkpoints = as_message_list(traj, "checkpoint")

    # Caso normal: trajectory.checkpoint { last_point { ... } }
    if checkpoints:
        cp = checkpoints[-1]
        lp = get_message(cp, "last_point")
        if lp is not None and message_nonempty(lp):
            return lp, "checkpoint[-1].last_point"

        hits = find_messages_named(cp, "last_point", "checkpoint[-1]")
        if hits:
            return hits[-1][1], hits[-1][0]

    # Fallback: procurar last_point em toda a trajectory.
    hits = find_messages_named(traj, "last_point", "trajectory")
    if hits:
        return hits[-1][1], hits[-1][0]

    return None, "not_found"


def extract_rv_from_point(point: Message) -> tuple[list[float] | None, list[float] | None, str]:
    dof = get_message(point, "degrees_of_freedom")
    dof_path = "point.degrees_of_freedom"

    if dof is None or not message_nonempty(dof):
        hits = find_messages_named(point, "degrees_of_freedom", "point")
        if not hits:
            return None, None, "no_degrees_of_freedom"
        dof_path, dof = hits[-1]

    # Caminho esperado: degrees_of_freedom.t1 e t2.
    t1 = get_message(dof, "t1")
    t2 = get_message(dof, "t2")

    if t1 is not None and t2 is not None:
        v1 = collect_xyz_vectors(t1, f"{dof_path}.t1")
        v2 = collect_xyz_vectors(t2, f"{dof_path}.t2")
        if v1 and v2:
            return v1[0][1], v2[0][1], f"{v1[0][0]} | {v2[0][0]}"

    # Fallback: pega os dois primeiros vetores xyz encontrados no DOF.
    vecs = collect_xyz_vectors(dof, dof_path)
    if len(vecs) >= 2:
        return vecs[0][1], vecs[1][1], f"{vecs[0][0]} | {vecs[1][0]}"

    return None, None, f"no_xyz_vectors; dof_fields={[f.name for f, _ in dof.ListFields()]}"


def point_time_s(point: Message) -> float | None:
    instant = get_message(point, "instant")
    if instant is not None:
        return scalar_value(instant)
    hits = find_messages_named(point, "instant", "point")
    if hits:
        return scalar_value(hits[-1][1])
    return None


def sub3(a: list[float], b: list[float]) -> list[float]:
    return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pb", default="data/jnsq_gate0/principia_plugin_uncompressed.pb")
    ap.add_argument("--central-body", default="Sun")
    ap.add_argument("--output-json", default="data/jnsq_gate0/snapshot_principia_native_v2.json")
    ap.add_argument("--output-csv", default="data/jnsq_gate0/snapshot_principia_native_v2_report.csv")
    args = ap.parse_args()

    msg = ksp_plugin_pb2.Plugin()
    msg.ParseFromString(Path(args.pb).read_bytes())

    current_time_s = scalar_value(msg.current_time)
    game_epoch_s = scalar_value(msg.game_epoch)
    epoch_ut_s = (
        current_time_s - game_epoch_s
        if current_time_s is not None and game_epoch_s is not None
        else None
    )

    eph = msg.ephemeris
    bodies = list(eph.body)
    trajectories = list(eph.trajectory)
    n = min(len(bodies), len(trajectories))

    print(f"[INFO] bodies={len(bodies)} trajectories={len(trajectories)}")
    print(f"[INFO] current_time_s={current_time_s}")
    print(f"[INFO] game_epoch_s={game_epoch_s}")
    print(f"[INFO] epoch_ut_s={epoch_ut_s}")

    rows = []
    raw_states: dict[str, dict[str, Any]] = {}

    for i in range(n):
        body = bodies[i]
        traj = trajectories[i]

        name = getattr(body, "name", f"body_{i}")
        mu = scalar_value(body.gravitational_parameter)

        point, point_path = latest_point_from_trajectory(traj)
        if point is None:
            rows.append({
                "index": i,
                "body": name,
                "status": "no_latest_point",
                "point_path": point_path,
            })
            continue

        r, v, rv_path = extract_rv_from_point(point)
        t = point_time_s(point)

        if r is None or v is None:
            rows.append({
                "index": i,
                "body": name,
                "time_s": t,
                "mu": mu,
                "status": "no_rv",
                "point_path": point_path,
                "rv_path": rv_path,
            })
            continue

        raw_states[name] = {
            "index": i,
            "mu": mu,
            "r_barycentric": r,
            "v_barycentric": v,
            "dof_time_s": t,
            "point_path": point_path,
            "rv_path": rv_path,
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
            "status": "ok",
            "point_path": point_path,
            "rv_path": rv_path,
        })

    # Em Principia, Sun às vezes não tem trajectory móvel útil no checkpoint
    # ou é usado como origem. Se não saiu, ancoramos na origem.
    if args.central_body not in raw_states:
        print(f"[FIX] {args.central_body} ausente dos estados extraídos; ancorando origem.")
        raw_states[args.central_body] = {
            "index": None,
            "mu": None,
            "r_barycentric": [0.0, 0.0, 0.0],
            "v_barycentric": [0.0, 0.0, 0.0],
            "dof_time_s": current_time_s,
            "point_path": "synthetic_origin",
            "rv_path": "synthetic_origin",
        }

    # Se o body central existe em eph.body, recupera μ dele.
    for i, body in enumerate(bodies):
        if getattr(body, "name", "") == args.central_body:
            raw_states[args.central_body]["mu"] = scalar_value(body.gravitational_parameter)
            raw_states[args.central_body]["index"] = i
            break

    cr = raw_states[args.central_body]["r_barycentric"]
    cv = raw_states[args.central_body]["v_barycentric"]

    snapshot_bodies = {}
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
            "principia_point_path": st["point_path"],
            "principia_rv_path": st["rv_path"],
        }

    snapshot = {
        "schema": "principia_native_snapshot.v2_from_plugin_pb",
        "source_pb": args.pb,
        "reference_body": args.central_body,
        "frame_note": "Principia barycentric states shifted to central body; no axis swap applied.",
        "epoch_ut_s": epoch_ut_s,
        "principia_current_time_s": current_time_s,
        "principia_game_epoch_s": game_epoch_s,
        "bodies": snapshot_bodies,
    }

    Path(args.output_json).write_text(json.dumps(snapshot, indent=2, ensure_ascii=False), encoding="utf-8")

    fieldnames = [
        "index", "body", "time_s", "mu",
        "x", "y", "z", "vx", "vy", "vz",
        "status", "point_path", "rv_path",
    ]
    with Path(args.output_csv).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    ok = sum(1 for r in rows if r.get("status") == "ok")
    print(f"[OK] estados extraídos: {ok}/{n}")
    print(f"[OK] snapshot: {args.output_json}")
    print(f"[OK] relatório: {args.output_csv}")

    times = sorted({
        round(float(b["principia_dof_time_s"]), 9)
        for b in snapshot_bodies.values()
        if b["principia_dof_time_s"] is not None
    })
    print(f"[INFO] tempos DOF únicos: {len(times)}")
    print("[INFO] primeiros tempos:", times[:5])
    print("[INFO] últimos tempos:", times[-5:])

    if current_time_s is not None and times:
        max_dt = max(abs(t - current_time_s) for t in times)
        print(f"[CHECK] max |dof_time - current_time| = {max_dt:.9e} s")


if __name__ == "__main__":
    main()