# Principia impulse server v0.2 package

This package keeps the legacy `PROP` protocol and adds:

- `PROP2`: exactly two impulses in one propagation request.
- `PROPN`: arbitrary number of impulses in one propagation request.

The goal is to optimize a complete live departure/correction chain against a final Eve target state, instead of optimizing a first burn to an intermediate match point.

## Install

From the KSP-MGA-Planner repository root:

```bash
cp native_cpp/ksp_plugin_test/principia_impulsive_particle_server.cpp \
   native_cpp/ksp_plugin_test/principia_impulsive_particle_server.cpp.bak.$(date +%s)

cp /path/to/package/native_cpp/ksp_plugin_test/principia_impulsive_particle_server.cpp \
   native_cpp/ksp_plugin_test/principia_impulsive_particle_server.cpp

cp /path/to/package/ksp_mga/native/impulse_server_client_v0_2.py \
   ksp_mga/native/impulse_server_client_v0_2.py

cp /path/to/package/scripts/optimize_propn_to_target.py \
   scripts/optimize_propn_to_target.py
chmod +x scripts/optimize_propn_to_target.py
```

Then rebuild the existing target:

```bash
make bin/x64/principia_impulsive_particle_server -j$(nproc)
```

## Smoke test server

```bash
bin/x64/principia_impulsive_particle_server data/principia/live_probe/principia_serialized_plugin_rocket.b64
# stdin:
PING
QUIT
```

Expected first line:

```text
READY\tprincipia_impulsive_particle_server_v0_2
```

## Protocol

Legacy, unchanged:

```text
PROP id t0 burn_t t1 x y z vx vy vz dvx dvy dvz
```

Two impulses:

```text
PROP2 id t0 tb0 tb1 t1 x y z vx vy vz dv0x dv0y dv0z dv1x dv1y dv1z
```

N impulses:

```text
PROPN id t0 t1 n x y z vx vy vz [burn_t dvx dvy dvz] * n
```

Response:

```text
OKN id t0 t1 n [burn_t burn_r[3] v_before[3] v_after[3]] * n final_r[3] final_v[3]
```

## Optimizer example

Use a live-state JSON captured at the current vessel epoch, then optimize two impulses:

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

Override impulse times explicitly when needed:

```bash
python scripts/optimize_propn_to_target.py ... \
  --impulse-time-s 749952081.851686 \
  --impulse-time-s 749972385.851686
```

Seed a specific impulse in raw m/s:

```bash
python scripts/optimize_propn_to_target.py ... \
  --seed-dv-raw 0:-500,1900,-400 \
  --seed-dv-raw 1:-113.1286,188.2892,-41.5635
```

The optimizer writes:

- `result.json`
- `mission_events.jsonl`

Each event uses stable request IDs (`attempt0`) to avoid accidental infinite insertion loops.
