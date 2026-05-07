#!/usr/bin/env python3
from __future__ import annotations

import base64
import csv
import inspect
import json
import re
import subprocess
from pathlib import Path
from typing import Callable

SFS = Path("/home/matheus/.steam/steam/steamapps/common/Kerbal Space Program/saves/JNSQ/principia_anchor.sfs")
OUT = Path("data/jnsq_gate0/principia_decode_trials")
OUT.mkdir(parents=True, exist_ok=True)


def pad_b64(s: str) -> str:
    s = re.sub(r"\s+", "", s)
    return s + "=" * ((4 - len(s) % 4) % 4)


def decode_urlsafe(s: str) -> bytes:
    return base64.urlsafe_b64decode(pad_b64(s))


def extract_chunks() -> list[str]:
    text = SFS.read_text(encoding="utf-8", errors="replace").splitlines()
    chunks = []
    in_principia = False
    depth = 0

    for raw in text:
        line = raw.strip()

        if line == "SCENARIO":
            in_candidate = True
            continue

        if "name = PrincipiaPluginAdapter" in line:
            in_principia = True
            continue

        if in_principia:
            if line.startswith("serialized_plugin") and "=" in line:
                chunks.append(line.split("=", 1)[1].strip())
            elif chunks and line == "}":
                break

    if not chunks:
        raise SystemExit("Nenhum serialized_plugin encontrado.")

    return chunks


def try_protoc(data: bytes, out_txt: Path) -> bool:
    try:
        p = subprocess.run(
            ["protoc", "--decode_raw"],
            input=data,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
    except FileNotFoundError:
        print("[WARN] protoc não instalado.")
        return False

    if p.returncode == 0 and len(p.stdout) > 20:
        out_txt.write_bytes(p.stdout)
        return True
    return False


def try_squash(data_path: Path, out_path: Path) -> bool:
    try:
        p = subprocess.run(
            ["squash", "-d", "-c", "gipfeli", str(data_path), str(out_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
    except FileNotFoundError:
        return False

    return p.returncode == 0 and out_path.exists() and out_path.stat().st_size > 0


def get_pygipfeli_functions() -> list[Callable]:
    try:
        import pygipfeli  # type: ignore
    except Exception as e:
        print(f"[WARN] pygipfeli indisponível: {e}")
        return []

    funcs = []
    for name in dir(pygipfeli):
        obj = getattr(pygipfeli, name)
        if callable(obj) and any(tok in name.lower() for tok in ["decompress", "uncompress", "decode"]):
            funcs.append(obj)

    print("[INFO] pygipfeli callables:", [getattr(f, "__name__", str(f)) for f in funcs])
    return funcs


def try_pygipfeli(data: bytes, funcs: list[Callable]) -> bytes | None:
    for fn in funcs:
        name = getattr(fn, "__name__", str(fn))
        try:
            out = fn(data)
            if isinstance(out, str):
                out = out.encode()
            if isinstance(out, bytearray):
                out = bytes(out)
            if isinstance(out, bytes) and len(out) > 0:
                print(f"[OK] pygipfeli.{name} retornou {len(out)} bytes")
                return out
        except Exception:
            pass
    return None


def main() -> None:
    chunks = extract_chunks()

    variants: dict[str, bytes] = {}

    # Variante A: Principia splitou uma string Base64 única em múltiplas linhas.
    variants["joined_urlsafe"] = decode_urlsafe("".join(chunks))

    # Variante B: cada linha é um Base64 independente.
    try:
        variants["per_line_urlsafe"] = b"".join(decode_urlsafe(c) for c in chunks)
    except Exception as e:
        print("[WARN] per_line_urlsafe falhou:", e)

    funcs = get_pygipfeli_functions()

    rows = []

    for variant_name, blob in variants.items():
        raw_path = OUT / f"{variant_name}.bin"
        raw_path.write_bytes(blob)
        print(f"\n=== {variant_name}: {len(blob)} bytes ===")
        print("first16:", blob[:16].hex())

        # Primeiro: testar protoc direto.
        if try_protoc(blob, OUT / f"{variant_name}.decode_raw.txt"):
            print(f"[HIT] protobuf direto: {variant_name}")

        # Agora: offsets.
        for off in range(0, 129):
            sliced = blob[off:]
            slice_path = OUT / f"{variant_name}_off{off:03d}.bin"
            slice_path.write_bytes(sliced)

            hit = False
            method = ""

            # pygipfeli
            if funcs:
                dec = try_pygipfeli(sliced, funcs)
                if dec:
                    pb = OUT / f"{variant_name}_off{off:03d}.pygipfeli.pb"
                    pb.write_bytes(dec)
                    if try_protoc(dec, OUT / f"{variant_name}_off{off:03d}.pygipfeli.decode_raw.txt"):
                        hit = True
                        method = "pygipfeli+protoc"
                    else:
                        method = "pygipfeli_only"

            # squash
            if not hit:
                pb = OUT / f"{variant_name}_off{off:03d}.squash.pb"
                if try_squash(slice_path, pb):
                    dec = pb.read_bytes()
                    if try_protoc(dec, OUT / f"{variant_name}_off{off:03d}.squash.decode_raw.txt"):
                        hit = True
                        method = "squash+protoc"
                    else:
                        method = "squash_only"

            if method:
                print(f"[CANDIDATE] {variant_name} off={off} method={method}")
                rows.append({
                    "variant": variant_name,
                    "offset": off,
                    "method": method,
                    "input_size": len(sliced),
                })

    with (OUT / "trial_summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["variant", "offset", "method", "input_size"])
        w.writeheader()
        w.writerows(rows)

    print("\n[OK] relatório:", OUT / "trial_summary.csv")
    if not rows:
        print("[RESULTADO] Nenhum offset abriu com as ferramentas disponíveis.")


if __name__ == "__main__":
    main()