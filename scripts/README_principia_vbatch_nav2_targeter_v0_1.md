# principia_vbatch_nav2_targeter_v0_1.py

Python targeter using the Principia `VBATCH_NAV2` protocol.

It refines the first navigation impulse in TNB by computing finite-difference
Jacobians in batch on the real Principia backend. It can target:

- `ca-radius-current-direction`: flyby-safe radius along the current CA side.
- `rel-r`: explicit raw relative position vector at CA.
- `bplane`: approximate B-plane coordinates.
- `minimize-ca`: diagnostic only, targets impact/zero distance.

This is intended after applying the C++ server patches that provide `VBATCH_NAV2`.

## Install

```bash
cp /mnt/data/principia_vbatch_nav2_targeter_v0_1.py scripts/
chmod +x scripts/principia_vbatch_nav2_targeter_v0_1.py
python -m py_compile scripts/principia_vbatch_nav2_targeter_v0_1.py
```

## Example: target a 200,000 km flyby radius

```bash
mkdir -p data/runs/game_export/current_orbit/principia_vbatch_nav2_target_row15_200kkm01

python scripts/principia_vbatch_nav2_targeter_v0_1.py \
  --server /home/matheus/Principia/bin/x64/principia_impulsive_particle_server \
  --plugin-b64 data/principia/live_probe/principia_serialized_plugin_rocket.b64 \
  --rank-json data/runs/game_export/current_orbit/snap_candidate_hunt01/candidate_departure_executability_rank.json \
  --row-index0 15 \
  --dep-body KERBIN \
  --arr-body EVE \
  --nav-body KERBIN \
  --flip-binormal \
  --scan-center-offset-days 0 \
  --scan-half-width-days 30 \
  --samples 121 \
  --target-mode ca-radius-current-direction \
  --target-ca-km 200000 \
  --radial-velocity-scale-s 1000 \
  --iterations 8 \
  --fd-step-m-s 1 \
  --step-max-m-s 80 \
  --lm 1e-4 \
  --line-search 1,0.5,0.25,0.125,0.0625 \
  --residual-scale-km 100000 \
  --edge-penalty 1000 \
  --quiet-stderr \
  --output-dir data/runs/game_export/current_orbit/principia_vbatch_nav2_target_row15_200kkm01
```

## Output

- `principia_vbatch_nav2_targeter_result.json`
- `rank_row15_principia_vbatch_nav2_targeted.json`

## Notes

This script refines only the first impulse. DSM support requires a server-side
`VBATCH_NAV2` evaluator that actually applies multiple impulses, not just parses
them.
