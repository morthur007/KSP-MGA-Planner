#!/usr/bin/env python3
from __future__ import annotations

import re
from pathlib import Path

SFS_PATH = Path("/home/matheus/.steam/steam/steamapps/common/Kerbal Space Program/saves/JNSQ/principia_anchor.sfs")

KEYWORDS = [
    "principia",
    "serialized",
    "plugin",
    "proto",
    "hex",
    "scenario",
    "external",
    "persistent",
]


def main() -> None:
    if not SFS_PATH.exists():
        raise SystemExit(f"Arquivo não encontrado: {SFS_PATH}")

    lines = SFS_PATH.read_text(encoding="utf-8", errors="replace").splitlines()

    print(f"Arquivo: {SFS_PATH}")
    print(f"Linhas: {len(lines)}")
    print()

    print("=== Ocorrências por keyword ===")
    found_any = False
    for i, line in enumerate(lines, start=1):
        low = line.lower()
        if any(k in low for k in KEYWORDS):
            found_any = True
            print(f"{i:8d}: {line[:300]}")

    if not found_any:
        print("Nenhuma keyword encontrada.")
        print("Isso sugere que este .sfs não contém nenhum bloco textual óbvio do Principia.")
        return

    print()
    print("=== Contexto em torno de linhas com 'principia' ===")
    principia_lines = [i for i, line in enumerate(lines) if "principia" in line.lower()]
    for idx in principia_lines[:20]:
        print()
        print(f"--- contexto linha {idx + 1} ---")
        lo = max(0, idx - 10)
        hi = min(len(lines), idx + 30)
        for j in range(lo, hi):
            mark = ">>" if j == idx else "  "
            print(f"{mark} {j+1:8d}: {lines[j][:500]}")

    print()
    print("=== Maiores valores key = value do arquivo ===")
    candidates = []
    for i, line in enumerate(lines, start=1):
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if len(value) > 80:
            candidates.append((len(value), i, key, value[:120]))

    candidates.sort(reverse=True)
    for length, i, key, preview in candidates[:50]:
        kind = "hex" if re.fullmatch(r"[0-9a-fA-F]+", preview.replace(" ", "")) else "text/base64/other"
        print(f"{i:8d}: len={length:8d} key={key!r:<40} kind_guess={kind:<16} preview={preview}")


if __name__ == "__main__":
    main()