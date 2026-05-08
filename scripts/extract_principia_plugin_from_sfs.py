#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path

BASE64_RE = re.compile(r"^[A-Za-z0-9+/=]+$")

def extract_principia_blocks(text: str) -> list[str]:
    lines = text.splitlines()
    blocks = []

    i = 0
    while i < len(lines):
        if lines[i].strip() == "SCENARIO":
            start = i
            depth = 0
            block = []
            while i < len(lines):
                line = lines[i]
                block.append(line)
                depth += line.count("{")
                depth -= line.count("}")
                i += 1
                if depth <= 0 and "{" in "\n".join(block):
                    break

            joined = "\n".join(block)
            if re.search(r"Principia|principia", joined):
                blocks.append(joined)
        else:
            i += 1

    return blocks

def collect_base64_lines(block: str) -> list[str]:
    out = []

    for raw in block.splitlines():
        line = raw.strip()

        # Formato key = value
        if "=" in line:
            _, value = line.split("=", 1)
            value = value.strip().strip('"')
        else:
            value = line.strip().strip('"')

        # Evita capturar nomes pequenos tipo "True", "False", etc.
        if len(value) >= 64 and BASE64_RE.match(value):
            out.append(value)

    return out

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("sfs", type=Path)
    ap.add_argument("output_b64", type=Path)
    args = ap.parse_args()

    text = args.sfs.read_text(errors="ignore")
    blocks = extract_principia_blocks(text)

    if not blocks:
        raise SystemExit("[FAIL] no Principia SCENARIO block found")

    candidates = []
    for b in blocks:
        lines = collect_base64_lines(b)
        if lines:
            candidates.append(lines)

    if not candidates:
        raise SystemExit("[FAIL] Principia block found, but no base64 payload lines found")

    # Escolhe o maior payload.
    lines = max(candidates, key=lambda xs: sum(len(x) for x in xs))

    args.output_b64.parent.mkdir(parents=True, exist_ok=True)
    args.output_b64.write_text("\n".join(lines) + "\n")

    print(f"[OK] Principia blocks found: {len(blocks)}")
    print(f"[OK] base64 lines written : {len(lines)}")
    print(f"[OK] total chars          : {sum(len(x) for x in lines)}")
    print(f"[OK] output               : {args.output_b64}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
