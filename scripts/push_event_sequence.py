#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path


def write_in_place(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(obj, indent=2) + "\n"
    with open(path, "w") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.utime(path, None)


def wait_result(path: Path, request_id: str, timeout_s: float) -> dict | None:
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if path.exists():
            try:
                r = json.loads(path.read_text())
                if r.get("request_id") == request_id:
                    return r
            except Exception:
                pass
        time.sleep(0.25)
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ksp-root", type=Path, required=True)
    ap.add_argument("--event", type=Path, action="append", required=True)
    ap.add_argument("--timeout-s", type=float, default=180.0)
    ap.add_argument("--pause-s", type=float, default=2.0)
    ap.add_argument("--keep-enabled", action="store_true")
    args = ap.parse_args()

    event_path = args.ksp_root / "GameData/MGAPlanner/mission_event.json"
    result_path = args.ksp_root / "GameData/MGAPlanner/mission_event_result.json"

    for i, p in enumerate(args.event, 1):
        ev = json.loads(p.read_text())
        rid = ev["request_id"]

        ev["enabled"] = True

        if result_path.exists():
            result_path.unlink()

        print(f"\n=== PUSH {i}/{len(args.event)} {rid} ===")
        print("initial_time   :", ev.get("initial_time"))
        print("plan_final_time:", ev.get("plan_final_time"))
        print("mode           :", ev.get("mode"))
        print("dv             :", ev.get("delta_v_levela_m_s") or ev.get("delta_v_navigation_m_s"))

        write_in_place(event_path, ev)

        result = wait_result(result_path, rid, args.timeout_s)
        if result is None:
            print("[FAIL] timeout waiting for mission_event_result.json")
            return 2

        print(json.dumps(result, indent=2))

        if not result.get("success"):
            print("[FAIL] event failed; stopping sequence")
            return 3

        if not args.keep_enabled:
            ev["enabled"] = False
            ev["disarmed_after_result"] = True
            write_in_place(event_path, ev)
            print("[OK] disarmed event after success")

        time.sleep(args.pause_s)

    print("\n[OK] sequence complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
