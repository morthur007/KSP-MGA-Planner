#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

try:
    import krpc
except ImportError:
    krpc = None


def load_event(path: Path) -> dict:
    text = path.read_text().strip()
    if not text:
        raise SystemExit(f"[FAIL] empty event file: {path}")

    # Accept plain JSON or first line of JSONL.
    if "\n" in text and not text.lstrip().startswith("{"):
        text = text.splitlines()[0]

    return json.loads(text.splitlines()[0] if path.suffix == ".jsonl" else text)


def write_event_atomic(path: Path, event: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(event, indent=2) + "\n")
    shutil.move(str(tmp), str(path))


def wait_result(result_path: Path, request_id: str, timeout_s: float) -> dict | None:
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if result_path.exists():
            try:
                result = json.loads(result_path.read_text())
                if result.get("request_id") == request_id:
                    return result
            except Exception:
                pass
        time.sleep(0.25)
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ksp-root", type=Path, required=True)
    ap.add_argument("--event", type=Path, required=True,
                    help="mission_event.json or one-event JSONL")
    ap.add_argument("--burn-ut", type=float, required=True)
    ap.add_argument("--lead-days", type=float, default=1.0)
    ap.add_argument("--lead-seconds", type=float, default=None)
    ap.add_argument("--chunk-days", type=float, default=30.0)
    ap.add_argument("--max-rails-rate", type=float, default=100000.0)
    ap.add_argument("--max-physics-rate", type=float, default=2.0)
    ap.add_argument("--result-timeout-s", type=float, default=60.0)
    ap.add_argument("--push", action="store_true",
                    help="Write mission_event.json after warp")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if krpc is None:
        raise SystemExit("[FAIL] python package krpc is not installed/importable")

    lead_s = args.lead_seconds
    if lead_s is None:
        lead_s = args.lead_days * 86400.0

    stop_ut = args.burn_ut - lead_s
    if stop_ut <= 0:
        raise SystemExit(f"[FAIL] invalid stop_ut={stop_ut}")

    event = load_event(args.event)

    # Force event initial_time to real burn_ut unless user intentionally left it different.
    event["initial_time"] = args.burn_ut
    event["enabled"] = True

    # Make sure the plan reaches past the burn, but do NOT force a long plan.
    # For Principia, long final_time can hit max step count. Respect the event's
    # plan_final_time when it is already after the burn.
    event.setdefault("ensure_flight_plan", True)
    event.setdefault("extend_existing_flight_plan", True)

    current_plan_final = float(event.get("plan_final_time", 0.0) or 0.0)
    if current_plan_final <= args.burn_ut:
        event["plan_final_time"] = args.burn_ut + 3600.0
    else:
        event["plan_final_time"] = current_plan_final

    request_id = event.get("request_id")
    if not request_id:
        raise SystemExit("[FAIL] event must contain request_id")

    print("=== MGA WARP + PUSH ROUTE EVENT ===")
    print(f"burn_ut      : {args.burn_ut:.6f}")
    print(f"lead_s       : {lead_s:.1f}")
    print(f"stop_ut      : {stop_ut:.6f}")
    print(f"chunk_days   : {args.chunk_days}")
    print(f"request_id   : {request_id}")
    print(f"event mode   : {event.get('mode')}")
    print(f"event path   : {args.event}")

    conn = krpc.connect(name="MGA warp and push route event")
    sc = conn.space_center

    current_ut = sc.ut
    print(f"current_ut   : {current_ut:.6f}")

    # --- INÍCIO DA INJEÇÃO DE MASSA ---
    try:
        live_mass_tonnes = sc.active_vessel.mass / 1000.0
        event["mass_tonnes"] = live_mass_tonnes
        print(f"live_mass    : {live_mass_tonnes:.3f} t")
    except Exception as e:
        print(f"[WARN] could not fetch live mass from kRPC: {e}")
    # --- FIM DA INJEÇÃO DE MASSA ---

    if current_ut >= stop_ut:
        print("[OK] already at/after stop_ut; no warp needed")
    elif args.dry_run:
        print("[DRY] would warp to stop_ut")
    else:
        chunk_s = max(60.0, args.chunk_days * 86400.0)

        while True:
            current_ut = sc.ut
            remaining = stop_ut - current_ut
            if remaining <= 2.0:
                break

            next_ut = min(stop_ut, current_ut + chunk_s)
            print(
                f"[WARP] ut={current_ut:.3f} -> {next_ut:.3f} "
                f"remaining_days={remaining/86400.0:.3f}"
            )
            sc.warp_to(next_ut, args.max_rails_rate, args.max_physics_rate)
            time.sleep(0.5)

        print(f"[OK] warp complete; current_ut={sc.ut:.6f}")

    event_path = args.ksp_root / "GameData/MGAPlanner/mission_event.json"
    result_path = args.ksp_root / "GameData/MGAPlanner/mission_event_result.json"

    if not args.push:
        print("[OK] not pushing event; use --push to write mission_event.json")
        print(json.dumps(event, indent=2))
        return 0

    if args.dry_run:
        print("[DRY] would write:")
        print(json.dumps(event, indent=2))
        return 0

    if result_path.exists():
        # Avoid confusing old result with new request.
        try:
            old = json.loads(result_path.read_text())
            if old.get("request_id") == request_id:
                backup = result_path.with_suffix(".json.prev")
                shutil.copy2(result_path, backup)
                print(f"[INFO] backed up previous same-request result to {backup}")
        except Exception:
            pass

    write_event_atomic(event_path, event)
    print(f"[OK] wrote event: {event_path}")

    result = wait_result(result_path, request_id, args.result_timeout_s)
    if result is None:
        print("[FAIL] timeout waiting for mission_event_result.json")
        return 2

    print("=== RESULT ===")
    print(json.dumps(result, indent=2))

    # One-shot safety: once the bridge has answered this exact request,
    # disarm mission_event.json so scene reloads / daemon restarts do not
    # insert the same manoeuvre again.
    try:
        event["enabled"] = False
        event["disarmed_after_result"] = True
        write_event_atomic(event_path, event)
        print(f"[OK] disarmed event after result: {event_path}")
    except Exception as e:
        print(f"[WARN] could not disarm event after result: {e}")

    if not result.get("success"):
        return 3

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
