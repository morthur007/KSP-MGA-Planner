#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events-jsonl", type=Path, required=True)
    ap.add_argument("--ksp-root", type=Path, required=True)
    ap.add_argument("--timeout-s", type=float, default=20.0)
    ap.add_argument("--pause-s", type=float, default=1.0)
    ap.add_argument("--limit", type=int, default=999)
    args = ap.parse_args()

    event_path = args.ksp_root / "GameData/MGAPlanner/mission_event.json"
    result_path = args.ksp_root / "GameData/MGAPlanner/mission_event_result.json"

    events = []
    with args.events_jsonl.open() as f:
        for line in f:
            line = line.strip()
            if line:
                events.append(json.loads(line))

    for i, ev in enumerate(events[:args.limit], 1):
        rid = ev["request_id"]
        print(f"\n=== PUSH {i}/{min(len(events), args.limit)} {rid} ===")
        event_path.write_text(json.dumps(ev, indent=2) + "\n")

        t0 = time.time()
        last = None
        while time.time() - t0 < args.timeout_s:
            if result_path.exists():
                try:
                    res = json.loads(result_path.read_text())
                    if res.get("request_id") == rid:
                        last = res
                        break
                except Exception:
                    pass
            time.sleep(0.25)

        if last is None:
            print("[FAIL] timeout waiting for mission_event_result.json")
            break

        print(json.dumps(last, indent=2))
        if not last.get("success"):
            print("[FAIL] event failed; stopping")
            break

        time.sleep(args.pause_s)

if __name__ == "__main__":
    main()
