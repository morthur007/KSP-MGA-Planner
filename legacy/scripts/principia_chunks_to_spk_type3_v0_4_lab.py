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
from numpy.polynomial.chebyshev import chebfit


MAGIC = b"PNCKV03\0"
DEFAULT_BASE_ID = -992000


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

    with path.open("rb") as f:
        f.seek(header["epoch_offset"])
        epochs = np.fromfile(f, dtype="<f8", count=sample_count)
        states = np.fromfile(f, dtype="<f8", count=body_count * sample_count * 6)

    if len(epochs) != sample_count:
        raise ValueError(f"{path}: epochs truncados")

    if len(states) != body_count * sample_count * 6:
        raise ValueError(f"{path}: states truncados")

    return epochs, states.reshape((body_count, sample_count, 6))


def load_all_chunks(chunks_dir: Path):
    chunk_paths = sorted(Path(p) for p in glob.glob(str(chunks_dir / "chunk_*.bin")))
    if not chunk_paths:
        raise FileNotFoundError(f"nenhum chunk_*.bin em {chunks_dir}")

    headers = [load_chunk_header(p) for p in chunk_paths]
    first = headers[0]

    all_epochs = []
    all_states = []

    for path, header in zip(chunk_paths, headers):
        epochs, states = read_chunk_arrays(path, header)

        # Remove epoch duplicado em fronteiras de chunk.
        if all_epochs and np.isclose(epochs[0], all_epochs[-1][-1], rtol=0, atol=1e-9):
            epochs = epochs[1:]
            states = states[:, 1:, :]

        all_epochs.append(epochs)
        all_states.append(states)

    epochs = np.concatenate(all_epochs)
    states = np.concatenate(all_states, axis=1)

    return first["bodies"], first["central_index"], epochs, states, headers


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
                f.write(f"BODY{naif_id}_GM = ( {mu / 1e9:.17e} )\n")

        f.write("\n\\begintext\n")


def sanitize_segment_id(name: str) -> str:
    return f"JNSQ_T3_{name}"[:40]


def make_cheby_cdata(
    epochs: np.ndarray,
    states_m: np.ndarray,
    record_span_s: float,
    degree: int,
    coverage_last_et: float,
):
    first_et = float(epochs[0])
    last_et_available = float(epochs[-1])

    # Precisamos de records suficientes para que coverage_last_et possa cair
    # dentro de um record existente. Se coverage_last_et está exatamente numa
    # fronteira, isso exige o próximo record.
    n_records = int(math.floor((coverage_last_et - first_et) / record_span_s)) + 1

    required_available_last = first_et + n_records * record_span_s
    if last_et_available + 1e-9 < required_available_last:
        raise ValueError(
            f"dados insuficientes para Type 3 padding: "
            f"available_last={last_et_available}, "
            f"required={required_available_last}. "
            f"Gere dados por pelo menos mais um record_span."
        )

    segment_last_et = first_et + n_records * record_span_s
    last_et = segment_last_et

    # SPICE usa km e km/s.
    states = states_m.copy()
    states[:, 0:3] /= 1000.0
    states[:, 3:6] /= 1000.0

    record_size = 6 * (degree + 1)
    cdata = np.zeros((n_records, record_size), dtype=np.float64)

    for rec in range(n_records):
        a = first_et + rec * record_span_s
        b = a + record_span_s

        # Inclui endpoints.
        mask = (epochs >= a - 1e-9) & (epochs <= b + 1e-9)
        t = epochs[mask]
        y = states[mask, :]

        if len(t) < degree + 1:
            raise ValueError(
                f"record {rec}: amostras insuficientes: {len(t)} "
                f"para degree={degree}"
            )

        tau = 2.0 * (t - a) / record_span_s - 1.0

        pos = 0
        for comp in range(6):
            coeff = chebfit(tau, y[:, comp], degree)
            cdata[rec, pos : pos + degree + 1] = coeff
            pos += degree + 1

    return first_et, last_et, n_records, cdata.reshape(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks-dir", required=True, type=Path)
    ap.add_argument("--output-bsp", required=True, type=Path)
    ap.add_argument(
                        "--coverage-duration-seconds",
                        type=float,
                        default=None,
                        help="Duração real da cobertura SPK. Pode ser menor que os dados disponíveis para permitir padding.",
                    )
    ap.add_argument("--output-tpc", required=True, type=Path)
    ap.add_argument("--output-metadata", required=True, type=Path)
    ap.add_argument("--central-body", default="Sun")
    ap.add_argument("--frame", default="J2000")
    ap.add_argument("--base-id", type=int, default=DEFAULT_BASE_ID)
    ap.add_argument("--degree", type=int, default=15)
    ap.add_argument("--record-span-seconds", type=float, default=86400.0)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    if args.degree < 1 or args.degree > 27:
        raise ValueError("SPK Type 3: degree deve estar entre 1 e 27.")

    bodies_meta, central_index, epochs, states_m, headers = load_all_chunks(args.chunks_dir)
    coverage_first_et = float(epochs[0])
    if args.coverage_duration_seconds is None:
        coverage_last_et = float(epochs[-1])
    else:
        coverage_last_et = coverage_first_et + args.coverage_duration_seconds

    body_names = [b["name"] for b in bodies_meta]
    mus = {b["name"]: b["mu_m3_s2"] for b in bodies_meta}

    if args.central_body not in body_names:
        raise ValueError(f"central body ausente: {args.central_body}")

    ids = build_ids(body_names, args.central_body, args.base_id)
    center_id = ids[args.central_body]

    args.output_bsp.parent.mkdir(parents=True, exist_ok=True)
    args.output_tpc.parent.mkdir(parents=True, exist_ok=True)
    args.output_metadata.parent.mkdir(parents=True, exist_ok=True)

    if args.output_bsp.exists():
        if args.overwrite:
            args.output_bsp.unlink()
        else:
            raise FileExistsError(args.output_bsp)

    print(f"[INFO] bodies={len(body_names)} samples={len(epochs)}")
    print(f"[INFO] coverage={epochs[0]:.6f}..{epochs[-1]:.6f}")
    print(f"[INFO] record_span_s={args.record_span_seconds} degree={args.degree}")

    handle = spice.spkopn(str(args.output_bsp), "JNSQ PRINCIPIA NATIVE TYPE3 LAB", 1024)

    segments = []

    try:
        for body_index, name in enumerate(body_names):
            if name == args.central_body or body_index == central_index:
                print(f"[SKIP] {name:<12} corpo central")
                continue

            first_et, last_et, n_records, cdata = make_cheby_cdata(
                epochs=epochs,
                states_m=states_m[body_index, :, :],
                record_span_s=args.record_span_seconds,
                degree=args.degree,
                coverage_last_et=coverage_last_et,
            )

            body_id = ids[name]
            segid = sanitize_segment_id(name)

            spice.spkw03(
                handle,
                body_id,
                center_id,
                args.frame,
                first_et,
                last_et,
                segid,
                args.record_span_seconds,
                n_records,
                args.degree,
                cdata,
                first_et,
            )

            print(
                f"[OK] {name:<12} id={body_id} records={n_records} "
                f"{first_et:.6f}..{last_et:.6f}"
            )

            segments.append(
                {
                    "body": name,
                    "body_id": body_id,
                    "center_id": center_id,
                    "first_et": first_et,
                    "last_et": last_et,
                    "records": n_records,
                    "record_span_seconds": args.record_span_seconds,
                    "degree": args.degree,
                }
            )

    finally:
        if segments:
            spice.spkcls(handle)
        else:
            try:
                spice.spkcls(handle)
            except spice.utils.exceptions.SpiceNOSEGMENTSFOUND:
                pass
            if args.output_bsp.exists():
                args.output_bsp.unlink()

    write_tpc(args.output_tpc, ids, mus)

    metadata = {
        "schema": "spice_v0_4_lab_principia_native.type3",
        "source_chunks_dir": str(args.chunks_dir),
        "output_bsp": str(args.output_bsp),
        "output_tpc": str(args.output_tpc),
        "central_body": args.central_body,
        "central_body_id": center_id,
        "frame": args.frame,
        "spk_type": 3,
        "coverage_last_et_seconds": coverage_last_et,
        "intended_coverage_last_et_seconds": coverage_last_et,
        "segment_last_et_seconds": max(s["last_et"] for s in segments),
        "data_last_et_seconds": float(epochs[-1]),
        "degree": args.degree,
        "record_span_seconds": args.record_span_seconds,
        "first_et_seconds": float(epochs[0]),
        "last_et_seconds": float(epochs[-1]),
        "body_count": len(body_names),
        "bodies": {
            name: {
                "naif_id": ids[name],
                "mu_m3_s2": mus.get(name),
            }
            for name in body_names
        },
        "segments": segments,
    }

    args.output_metadata.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"[OK] BSP: {args.output_bsp}")
    print(f"[OK] TPC: {args.output_tpc}")
    print(f"[OK] metadata: {args.output_metadata}")
    print(f"[OK] segments: {len(segments)}")


if __name__ == "__main__":
    main()