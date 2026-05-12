# KSP MGA Planner — Offline-first N-body trajectory planning for Kerbal Space Program + Principia

> **Status:** experimental research/tooling project. The code is useful, but it is not yet a one-click mission planner.
>
> **Goal:** plan and validate interplanetary Multiple Gravity Assist (MGA) trajectories for Kerbal Space Program, especially KSP installs using **Principia** and large/rescaled systems such as JNSQ/RSS-like setups.

This repository is an attempt to bridge three worlds that usually do not talk to each other cleanly:

1. **Global trajectory search** using patched-conic/Lambert-style methods.
2. **Offline N-body validation/refinement** using Principia-derived ephemerides and native propagation tools.
3. **In-game execution** using kRPC and a small KSP/Principia bridge DLL that can create Principia FlightPlan manoeuvres directly in-game.

The project is currently focused on proving a full workflow:

```text
search candidate MGA route
→ validate/refine legs in N-body
→ audit flybys / powered flybys
→ export manoeuvre events
→ inject Principia FlightPlan burns in KSP
→ visually and numerically validate in-game
```

A current example route under investigation is a JNSQ-style:

```text
Kerbin → Eve → Kerbin → Jool
```

The route search and N-body refinement pieces are working well enough to identify candidate trajectories. The active development area is the **field-engineering layer**: turning offline vectors into robust, game-executable Principia manoeuvres.

---

## Important repository note: large binaries and SPICE assets are not included

Some generated/native files are intentionally **not committed** because they are too large for normal GitHub hosting or are machine/build specific.

Examples of files you may need to build or generate locally:

```text
bin/x64/principia_impulsive_particle_server
bin/x64/principia_particle_validator
bin/x64/principia_impulsive_particle_validator
bin/x64/principia_flightplan_probe
bin/x64/principia_flightplan_copywrite_smoke
```

Similarly, SPICE/CSPICE toolkits and generated kernels may not be committed:

```text
data/kernels/**/*.bsp
data/kernels/**/*.tpc
data/kernels/**/*.tls
third_party/cspice/ or deps/cspice/
```

If you clone this repo and something says “missing binary” or “missing kernel”, that is expected. You need to build/generate the native tools and/or provide the SPICE Toolkit locally.

For collaboration, Git LFS may be useful, but at the moment the repo is designed to remain source-first.

---

## What this project can do today

### Working / mostly working

- Read/export Principia serialized plugin data from KSP save files.
- Probe Principia FlightPlans and manoeuvres from native tools.
- Convert between the project’s `raw` Principia frame and the LevelA/SPICE-like frame used by the Python pipeline.
- Start a Principia-backed native impulse propagation server.
- Propagate particles with one or multiple impulsive burns using native Principia code.
- Run offline N-body validation/refinement of candidate legs.
- Build a KSP C# addon/DLL that watches `GameData/MGAPlanner/mission_event.json` and inserts burns into the active vessel’s Principia FlightPlan.
- Insert live Principia FlightPlan manoeuvres from JSON events.
- Create FlightPlans from scratch when none exist.
- Read back inserted manoeuvres and verify vector roundtrips.

### In progress / fragile

- Robust departure targeting from an actual parking orbit.
- Multi-impulse optimisation from a live in-game vessel state to an Eve/Jool encounter.
- Safe long-preview FlightPlan handling in Principia without hitting max-step or memory limits.
- Automatic cleanup/removal/reset of existing FlightPlan manoeuvres.
- Polished command-line UX.

### Known hard problems

- Principia is N-body. A patched-conic solution can miss by millions of km if injected directly without N-body refinement.
- A visually plausible prograde departure burn is not enough; the burn must target the correct outgoing `v∞` vector and downstream encounter.
- Long FlightPlan previews from low orbit can be expensive or crash-prone. Insert burns in staged windows and extend previews gradually.
- KSP/Unity/Principia can accumulate memory pressure after many reloads; restarting the game between major tests is sometimes necessary.

---

## Architecture overview

```text
Python search / optimisation
  ├─ global candidate search
  ├─ leg optimisation
  ├─ flyby auditing
  ├─ powered flyby bridge
  └─ event export

Native Principia tools
  ├─ read serialized plugin state
  ├─ sample ephemerides
  ├─ validate particles
  ├─ impulse propagation server
  └─ FlightPlan probe/copywrite experiments

KSP live bridge
  ├─ C# KSPAddon
  ├─ watches GameData/MGAPlanner/mission_event.json
  ├─ creates/extends Principia FlightPlans
  ├─ inserts/replaces manoeuvres
  └─ writes mission_event_result.json
```

---

## Coordinate frames used in this project

One of the most important lessons from this project is that frame mistakes look exactly like “bad astrodynamics”.

The project commonly uses:

- **Principia raw frame**: native Principia/KSP plugin coordinates.
- **LevelA/SPICE-like frame**: frame used by generated kernels and many Python-side tools.
- **Navigation frame**: Principia FlightPlan manoeuvre frame, usually tangent/normal/binormal-like.

The currently validated mapping is:

```python
# raw -> LevelA
levela = [-raw_y, raw_z, raw_x]

# LevelA -> raw
raw = [levela_z, -levela_x, levela_y]
```

When debugging manoeuvres, always verify vector norm and roundtrip error. A burn with the correct magnitude but wrong frame can send the craft into a completely different heliocentric orbit.

---

## Setup overview

### System dependencies

Typical Linux setup:

```bash
sudo dnf install git make clang gcc gcc-c++ python3 python3-pip mono-devel
# or on Debian/Ubuntu:
sudo apt install git make clang g++ python3 python3-pip mono-devel
```

Python dependencies vary as the project evolves, but commonly include:

```bash
python -m pip install numpy scipy pandas spiceypy krpc
```

Optional but useful:

```bash
python -m pip install matplotlib tqdm
```

---

## Building native Principia tools

The native tools are built against a local checkout/build environment that can compile Principia-derived C++ code. The exact build can vary depending on your local Principia tree and dependencies.

Common targets used in this project:

```bash
make bin/x64/principia_particle_validator -j$(nproc)
make bin/x64/principia_impulsive_particle_validator -j$(nproc)
make bin/x64/principia_impulsive_particle_server -j$(nproc)
make bin/x64/principia_flightplan_probe -j$(nproc)
make bin/x64/principia_flightplan_copywrite_smoke -j$(nproc)
```

The most important one for trajectory refinement is:

```text
bin/x64/principia_impulsive_particle_server
```

It loads the serialized Principia plugin once and then accepts propagation commands over stdin/stdout.

### Server smoke test

```bash
bin/x64/principia_impulsive_particle_server data/principia/live_probe/principia_serialized_plugin_rocket.b64
```

Expected startup for the newer protocol:

```text
READY    principia_impulsive_particle_server_v0_2
```

Then type:

```text
PING
QUIT
```

Expected:

```text
PONG
BYE
```

The older server may not support `PING` and may only accept `PROP` commands.

---

## Extracting Principia serialized plugin data from a save

Principia stores a serialized plugin payload inside the `.sfs` save.

Example:

```bash
SAVE="$HOME/.steam/steam/steamapps/common/Kerbal Space Program/saves/JNSQ/with_rocket.sfs"
OUT="data/principia/live_probe/principia_rocket_lines.b64"

mkdir -p data/principia/live_probe

grep "serialized_plugin =" "$SAVE" \
  | sed 's/^[[:space:]]*serialized_plugin = //' \
  > "$OUT"
```

Keep the line structure intact. Some tools tolerate whitespace; others are stricter. Avoid manually editing the base64 unless you know exactly what you are doing.

Smoke test with an ephemeris sampler or probe tool if available:

```bash
bin/x64/principia_flightplan_probe \
  data/principia/live_probe/principia_rocket_lines.b64 \
  <VESSEL_GUID>
```

---

## KSP live bridge: creating Principia manoeuvres from JSON

The in-game bridge is a C# KSP addon that reads:

```text
<KSP>/GameData/MGAPlanner/mission_event.json
```

and writes:

```text
<KSP>/GameData/MGAPlanner/mission_event_result.json
```

A typical event:

```json
{
  "enabled": true,
  "request_id": "rank12_departure_attempt0",
  "mode": "insert_levela",
  "vessel_guid": "60735c81-7e29-4c06-9551-9e5283e37586",

  "ensure_flight_plan": true,
  "extend_existing_flight_plan": true,
  "plan_final_time": 749952394.0,
  "mass_tonnes": 2.6,

  "insert_index": 0,
  "burn_template": "json",
  "thrust_kN": 2686.87701225281,
  "specific_impulse_s_g0": 1000.0,
  "is_inertially_fixed": false,

  "frame_extension": 6000,
  "frame_centre_from_active_body": true,
  "frame_centre_index": -1,
  "frame_primary_index": -1,
  "frame_secondary_index": -1,

  "initial_time": 749951794.8778343,
  "delta_v_levela_m_s": [1220.7744451629458, -153.73776730710946, -61.75727305830709],

  "require_status_ok": true,
  "cleanup_on_error": true,
  "tolerance_time_s": 0.01,
  "tolerance_dv_m_s": 0.000001,

  "one_shot": true,
  "disable_after_success": true
}
```

### Important safety rules

- Do not generate a new random `request_id` every second. Use stable request IDs like `rank12_departure_attempt0`, then increment to `attempt1` only when intentionally retrying.
- Disarm `mission_event.json` after success:

```json
{
  "enabled": false,
  "request_id": "idle",
  "mode": "noop"
}
```

- For low orbit, do not create a FlightPlan that runs for many hours before the first burn. Warp close to the first burn first.
- If `INSERT_STATUS.error != 0`, treat the result as failed even if the manoeuvre count changed. Some failure modes can leave partial state.

---

## Recommended live workflow for a two-impulse departure

A robust in-game test should be staged:

```text
1. Rewind/load a save before the first burn.
2. Warp to about 5 minutes before burn 0.
3. Insert burn 0 with a short FlightPlan preview, e.g. burn0 + 10 minutes.
4. Extend/insert burn 1 only after burn 0 exists.
5. Preview gradually: +3 days, +15 days, +60 days, then encounter epoch.
```

Do **not** start by asking Principia to preview from low Kerbin orbit all the way to Eve. That can hit max step count or memory limits.

---

## Current research notes / known pitfalls

### Patched conics are not enough

A PyKEP/Lambert search can identify promising windows, but direct injection into Principia can miss by millions of km. This is expected: Principia is N-body and the execution state in KSP is not the same as an abstract patched-conic departure state.

### Corrections must be re-optimised with departure

A deep-space correction vector is not universal. If the departure burn changes, the correction time and vector usually need to be recalculated. The current direction is to optimise:

```text
burn0 departure + burn1 correction + final Eve/Jool target
```

as a single N-body multiple-shooting problem.

### Raw free-vector optimisation can produce nonsense

If an optimiser is allowed to choose arbitrary raw Δv vectors, it may find mathematically good but physically bad solutions, such as:

- a first burn dominated by inclination/normal component;
- a huge “correction” still inside Kerbin SOI;
- a non-escape first burn followed by an enormous second burn.

A useful departure optimiser should constrain or penalise:

- non-escape after burn 0;
- excessive normal/radial burn fraction;
- second burn too close to Kerbin;
- second burn magnitude that is too large for a correction;
- long low-orbit propagation before departure.

---

## Useful commands

### Check current KSP UT through kRPC

```bash
python - <<'PY'
import krpc
conn = krpc.connect(name="check ut")
sc = conn.space_center
print("UT:", sc.ut)
PY
```

### Capture live vessel state

```bash
python scripts/capture_live_state_raw.py \
  --bsp data/kernels/jnsq/v0_3/jnsq_principia_native_v0_3_1000y_1h_deg9.bsp \
  --tpc data/kernels/jnsq/v0_3/jnsq_principia_native_v0_3_1000y_1h_deg9.ids.tpc \
  --output-json data/runs/game_export/rank12_real/live_state_raw.json
```

### Run a multi-impulse optimisation

```bash
python scripts/optimize_propn_to_target.py \
  --plugin-b64 data/principia/live_probe/principia_serialized_plugin_rocket.b64 \
  --server bin/x64/principia_impulsive_particle_server \
  --live-state-json data/runs/game_export/rank12_real/live_state_raw.json \
  --leg-optimizations data/runs/finalists/rank12_kekj/leg_optimizations.csv \
  --leg 1 \
  --output-dir data/runs/game_export/rank12_real/propn_opt_v0 \
  --max-nfev 120
```

### Push one or more events to KSP

```bash
python scripts/push_event_sequence.py \
  --ksp-root "$KSP" \
  --event data/runs/game_export/rank12_real/propn_opt_v0/event1.json \
  --event data/runs/game_export/rank12_real/propn_opt_v0/event2.json \
  --timeout-s 240
```

---

## Contributing / asking for help

This project would benefit from feedback from:

- KSP Principia users who understand FlightPlans and reference frames.
- Astrodynamics people familiar with multiple shooting and DSM targeting.
- C++ developers comfortable with Principia internals.
- KSP modders comfortable with C# addon lifecycle and Unity/Mono quirks.

Especially useful contributions:

- safer FlightPlan reset/remove modes in the bridge DLL;
- better departure targeting constraints;
- robust multi-impulse optimisation examples;
- documentation for frame transforms;
- memory/step-count mitigation strategies for long Principia previews.

---

## Disclaimer

This is an experimental research project and may crash KSP, corrupt temporary saves, or produce nonsensical trajectories. Work on copied saves. Do not test on your only career save.

Principia, KSP, NAIF SPICE, PyKEP, kRPC, and other referenced projects belong to their respective authors/maintainers. This repository is not affiliated with Squad, Private Division, the Principia authors, NAIF/JPL, or NASA.