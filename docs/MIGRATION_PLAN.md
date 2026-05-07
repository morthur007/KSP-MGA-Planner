# Migration plan

## Phase 0 — freeze

- Commit current repo.
- Copy this scaffold into the repo root.
- Run `pip install -e .` or set `PYTHONPATH`.

## Phase 1 — make candidate seeds complete

Patch `spice_lambert_beam_search_v0_2.py` so every output candidate includes:

```text
legN_vdep_x_km_s,legN_vdep_y_km_s,legN_vdep_z_km_s
legN_varr_x_km_s,legN_varr_y_km_s,legN_varr_z_km_s
legN_dep,legN_arr
```

Then verify:

```bash
python - <<'PY'
from ksp_mga.core.schemas import read_candidate_seed
c = read_candidate_seed('data/mga_smoke/family_search_kj_v0_1/merged_candidates.csv', 1)
print(c.sequence)
for leg in c.legs:
    print(leg.leg, leg.dep_body, leg.arr_body, leg.path, leg.vdep_km_s)
PY
```

## Phase 2 — move daemon client

Replace local daemon client classes with:

```python
from ksp_mga.native.impulse_server_client import PrincipiaImpulseServer, ImpulseRequest
```

## Phase 3 — move leg optimizer

Move the working code from `native_optimize_candidate_legs_v0_1.py` into:

```python
ksp_mga.native.leg_optimizer.optimize_candidate_legs()
```

Rules:

- consume `CandidateSeed`;
- do not solve Lambert again;
- write `LegOptimizationResult` with `final_*`, `target_*`, `start_*` fields.

## Phase 4 — move flyby audit

Move `native_corrected_flyby_audit_v0_1.py` into:

```python
ksp_mga.native.flyby_audit.audit_corrected_flybys()
```

Rules:

- consume only `leg_optimization.csv` + SPICE body states;
- do not infer missing final states;
- output `FlybyAuditResult` JSON.

## Phase 5 — move orchestrator

Move `nbody_pipeline_orchestrator_v0_1.py` into:

```python
ksp_mga.pipeline.triage.run_triage()
```

Rules:

- no Python subprocess calls;
- only C++ impulse server may be subprocess;
- centralize logging per candidate.
