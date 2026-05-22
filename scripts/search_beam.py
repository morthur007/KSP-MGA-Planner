#!/usr/bin/env python3
from __future__ import annotations

import sys

from ksp_mga.lambert.beam_search import main


def normalize_sequence_argv(argv: list[str]) -> list[str]:
    """Make --sequence robust to quoted or hyphenated input.

    The real parser in ksp_mga.lambert.beam_search declares:

        --sequence SEQUENCE [SEQUENCE ...]

    so the canonical call is:

        --sequence Kerbin Duna Jool

    If a caller passes a single token such as "Kerbin Duna Jool" or
    "KERBIN-DUNA-JOOL", argparse sees a one-body sequence, hence zero legs, and
    later rejects TOF vectors with the message:

        tof-min/max/step precisam ter 0 valores

    This wrapper expands those single-token forms before the real parser runs.
    """
    out: list[str] = []
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg == "--sequence" and i + 1 < len(argv):
            out.append(arg)
            i += 1
            seq_tokens: list[str] = []
            while i < len(argv) and not argv[i].startswith("--"):
                token = argv[i].strip()
                if token:
                    token = token.replace("-", " ")
                    seq_tokens.extend(x for x in token.split() if x)
                i += 1
            out.extend(seq_tokens)
            continue
        if arg.startswith("--sequence="):
            key, value = arg.split("=", 1)
            out.append(key)
            value = value.strip().replace("-", " ")
            out.extend(x for x in value.split() if x)
            i += 1
            continue
        out.append(arg)
        i += 1
    return out


if __name__ == "__main__":
    sys.argv = normalize_sequence_argv(sys.argv)
    raise SystemExit(main())
