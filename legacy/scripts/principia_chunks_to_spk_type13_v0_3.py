#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
import math
import struct
from pathlib import Path

import numpy as np
import spiceypy as spice


MAGIC = b"PNCKV03\0"
DEFAULT_BASE_ID = -991000


def read_exact(f, n: int) -> bytes:
    b = f.read(n)
    if len(b) != n:
        raise EOFError(f"esperava {n} bytes, recebi {len(b)}")
    return b


def read_u32(f) -> int:
    return struct.unpack("<I", read_exact(f, 4))[0]


def read_i32(f) -> int:
    return struct.unpack("<i", read_exact(f, 4))[0]


def read_u64(f) -> int:
    return struct.unpack("<Q", read_exact(f, 8))[0]


def read_f64(f) -> float:
    return struct.unpack("<d", read_exact(f, 8))[0]


def read_string(f) -> str:
    n = read_u32(f)
    return read_exact(f, n).decode("utf-8")


def sanitize_segment_id(name: str, chunk_index: int) -> str:
    return f"JNSQ_V03_{name}_{chunk_index:06d}"[:40]


def load_chunk_header(path: Path):
    with path.open("rb") as f:
        magic = read_exact(f, 8)
        if magic != MAGIC:
            raise ValueError(f"{path}: magic inválido {magic!r}")

        version = read_u32(f)
        body_count = read_u32(f)
        sample_count = read_u64(f)
        central_index = read_i32(f)
        chunk_index = read_i32(f)
        chunk_start_offset_s = read_f64(f)
        chunk_end_offset_s = read_f64(f)
        step_s = read_f64(f)
        first_et = read_f64(f)
        last_et = read_f64(f)

        bodies = []
        for _ in range(body_count):
            body_index = read_i32(f)
            mu = read_f64(f)
            name = read_string(f)
            bodies.append({"index": body_index, "name": name, "mu_m3_s2": mu})

        epoch_offset = f.tell()

    return {
        "path": str(path),
        "version": version,
        "body_count": body_count,
        "sample_count": sample_count,
        "central_index": central_index,
        "chunk_index": chunk_index,
        "chunk_start_offset_s": chunk_start_offset_s,
        "chunk_end_offset_s": chunk_end_offset_s,
        "step_s": step_s,
        "first_et": first_et,
        "last_et": last_et,
        "bodies": bodies,
        "epoch_offset": epoch_offset,
    }


def read_chunk_arrays(path: Path, header: dict):
    body_count = header["body_count"]
    sample_count = header["sample_count"]
    epoch_offset = header["epoch_offset"]

    with path.open("rb") as f:
        f.seek(epoch_offset)

        epochs = np.fromfile(f, dtype="<f8", count=sample_count)
        expected = body_count * sample_count * 6
        states = np.fromfile(f, dtype="<f8", count=expected)

    if len(epochs) != sample_count:
        raise ValueError(f"{path}: epochs truncados")

    if len(states) != expected:
        raise ValueError(f"{path}: states truncados")

    states = states.reshape((body_count, sample_count, 6))
    return epochs, states


def build_ids(bodies: list[str], central_body: str, base_id: int):
    ids = {central_body: base_id}
    n = 1
    for body in sorted(bodies):
        if body == central_body:
            continue
        ids[body] = base_id - n
        n += 1
    return ids


def write_tpc(path: Path, ids: dict[str, int], mus: dict[str, float]):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        f.write("\\begindata\n\n")

        for name, naif_id in sorted(ids.items(), key=lambda kv: kv[1], reverse=True):
            f.write(f"NAIF_BODY_NAME += ( '{name.upper()}' )\n")
            f.write(f"NAIF_BODY_CODE += ( {naif_id} )\n")

        f.write("\n")

        for name, naif_id in sorted(ids.items(), key=lambda kv: kv[1], reverse=True):
            mu = mus.get(name)
            if mu is not None and math.isfinite(mu):
                mu_km3_s2 = mu / 1.0e9
                f.write(f"BODY{naif_id}_GM = ( {mu_km3_s2:.17e} )\n")

        f.write("\n\\begintext\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks-dir", required=True, type=Path)
    ap.add_argument("--output-bsp", required=True, type=Path)
    ap.add_argument("--output-tpc", required=True, type=Path)
    ap.add_argument("--output-metadata", required=True, type=Path)
    ap.add_argument("--central-body", default="Sun")
    ap.add_argument("--frame", default="J2000")
    ap.add_argument("--base-id", type=int, default=DEFAULT_BASE_ID)
    ap.add_argument("--degree", type=int, default=9)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    if args.degree % 2 == 0:
        raise ValueError("SPK Type 13 exige grau ímpar.")

    chunk_paths = sorted(Path(p) for p in glob.glob(str(args.chunks_dir / "chunk_*.bin")))
    if not chunk_paths:
        raise FileNotFoundError(f"nenhum chunk_*.bin em {args.chunks_dir}")

    headers = [load_chunk_header(p) for p in chunk_paths]
    first_header = headers[0]

    body_names = [b["name"] for b in first_header["bodies"]]
    if args.central_body not in body_names:
        raise ValueError(f"central body ausente: {args.central_body}")

    ids = build_ids(body_names, args.central_body, args.base_id)

    mus = {b["name"]: b["mu_m3_s2"] for b in first_header["bodies"]}

    central_index = body_names.index(args.central_body)
    center_id = ids[args.central_body]

    args.output_bsp.parent.mkdir(parents=True, exist_ok=True)
    args.output_tpc.parent.mkdir(parents=True, exist_ok=True)
    args.output_metadata.parent.mkdir(parents=True, exist_ok=True)

    if args.output_bsp.exists():
        if args.overwrite:
            args.output_bsp.unlink()
        else:
            raise FileExistsError(args.output_bsp)

    handle = spice.spkopn(str(args.output_bsp), "JNSQ PRINCIPIA NATIVE V0.3", 1024)

    written_segments = []

    try:
        for header, chunk_path in zip(headers, chunk_paths):
            print(
                f"[CHUNK] {chunk_path.name} "
                f"idx={header['chunk_index']} "
                f"n={header['sample_count']} "
                f"{header['first_et']:.6f}..{header['last_et']:.6f}"
            )

            epochs, states_m = read_chunk_arrays(chunk_path, header)

            if len(epochs) <= args.degree:
                raise ValueError(f"{chunk_path}: amostras insuficientes para degree={args.degree}")

            # m/m/s -> km/km/s
            states_spice = states_m.copy()
            states_spice[:, :, 0:3] /= 1000.0
            states_spice[:, :, 3:6] /= 1000.0

            for body_index, body in enumerate(header["bodies"]):
                name = body["name"]

                if body_index == central_index or name == args.central_body:
                    continue

                body_id = ids[name]
                body_states = np.ascontiguousarray(states_spice[body_index, :, :])

                segid = sanitize_segment_id(name, header["chunk_index"])

                spice.spkw13(
                    handle,
                    body_id,
                    center_id,
                    args.frame,
                    float(epochs[0]),
                    float(epochs[-1]),
                    segid,
                    args.degree,
                    len(epochs),
                    body_states,
                    epochs,
                )

                written_segments.append(
                    {
                        "body": name,
                        "body_id": body_id,
                        "center_id": center_id,
                        "chunk_index": header["chunk_index"],
                        "start_et": float(epochs[0]),
                        "end_et": float(epochs[-1]),
                        "samples": int(len(epochs)),
                    }
                )

    finally:
        spice.spkcls(handle)

    write_tpc(args.output_tpc, ids, mus)

    metadata = {
        "schema": "spice_v0_3_principia_native.type13.chunked",
        "source_chunks_dir": str(args.chunks_dir),
        "output_bsp": str(args.output_bsp),
        "output_tpc": str(args.output_tpc),
        "central_body": args.central_body,
        "central_body_id": center_id,
        "frame": args.frame,
        "spk_type": 13,
        "degree": args.degree,
        "chunk_count": len(headers),
        "body_count": len(body_names),
        "bodies": {
            name: {
                "naif_id": ids[name],
                "mu_m3_s2": mus.get(name),
            }
            for name in body_names
        },
        "first_et_seconds": min(h["first_et"] for h in headers),
        "last_et_seconds": max(h["last_et"] for h in headers),
        "segments": written_segments,
    }

    args.output_metadata.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"[OK] BSP: {args.output_bsp}")
    print(f"[OK] TPC: {args.output_tpc}")
    print(f"[OK] metadata: {args.output_metadata}")
    print(f"[OK] segments: {len(written_segments)}")


if __name__ == "__main__":
    main()