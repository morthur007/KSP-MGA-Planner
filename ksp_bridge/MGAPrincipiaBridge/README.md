# MGA Principia Bridge — hot-reload daemon

This package contains `MGAPrincipiaBridge.cs`, a KSP C# addon that talks to the live Principia `principia.ksp_plugin_adapter` via reflection.

It watches:

```text
<KSP>/GameData/MGAPlanner/mission_event.json
```

When `enabled=true` and `request_id` changes, it creates/extends a Principia FlightPlan if requested, inserts/replaces a manoeuvre, reads it back, converts navigation/raw/LevelA, and writes:

```text
<KSP>/GameData/MGAPlanner/mission_event_result.json
```

## Build

```bash
export KSP="/home/matheus/.local/share/Steam/steamapps/common/Kerbal Space Program"
./scripts/build_bridge.sh
./scripts/install_bridge.sh
```

After installing a new DLL, restart KSP once. After that, you can keep KSP open and only change `mission_event.json`.

## Important runtime notes

- `request_id` is the trigger. Reusing the same `request_id` will not re-run.
- `require_status_ok=true` prevents false success when Principia returns `Status.error != 0`.
- `cleanup_on_error=true` attempts to remove a manoeuvre inserted before a failed propagation/replace status.
- `ensure_flight_plan=true` calls `FlightPlanCreate` when no plan exists.
- `extend_existing_flight_plan=true` calls `FlightPlanSetDesiredFinalTime` when the requested final time is beyond the current plan.
- For a real route burn far in the future, warp close to the burn epoch first. Inserting a manoeuvre far outside the current FlightPlan will usually fail with `Does not fit` or max-step errors.

## Suggested first test

1. Load the vessel in flight.
2. Make sure `mission_event.json` is disabled or absent.
3. Copy `examples/create_plan_nav_smoke.json` to `GameData/MGAPlanner/mission_event.json`.
4. Edit:
   - `vessel_guid`
   - `initial_time`
   - `plan_final_time`
   - optionally `mass_tonnes`
5. Wait 1–2 seconds.
6. Inspect `mission_event_result.json` and `MGAPrincipiaBridge_probe.log`.

Success should have:

```json
"success": true,
"status": "ok",
"insert_error": 0,
"navigation_error_m_s": 0
```

## Example command to inspect logs

```bash
KSP="/home/matheus/.local/share/Steam/steamapps/common/Kerbal Space Program"
cat "$KSP/GameData/MGAPlanner/mission_event_result.json"
grep -nE "request_id|CREATE_FLIGHT_PLAN|SET_DESIRED|BEFORE|INSERT|REPLACE|AFTER|ROUNDTRIP|SUCCESS|FAIL|WROTE_RESULT" \
  "$KSP/MGAPrincipiaBridge_probe.log" | tail -160
```


## v0.4 frame safety notes

Principia crashes deliberately if `NavigationFrameParameters.extension` is 0.
This package refuses `frame_extension <= 0` for `burn_template=json`. Known navigation-frame extension values from `serialization/physics.pb.h` are:

- `6000` = `BodyCentredNonRotatingReferenceFrame`
- `6001` = `BarycentricRotatingReferenceFrame`
- `6002` = `BodyCentredBodyDirectionReferenceFrame`
- `6003` = `BodySurfaceReferenceFrame`

For a first from-scratch smoke test around the active vessel, use `examples/create_plan_nav_smoke_auto_body.json`; it uses `frame_extension=6000` and `frame_centre_from_active_body=true`. The bridge logs the inferred KSP body index as `FRAME_AUTO_CENTRE`.

If a flight plan already has at least one manual/Principia manoeuvre, `examples/insert_with_frame_from_clone.json` is safer because it clones only the frame from manoeuvre 0 and uses JSON for the rest of the burn.
