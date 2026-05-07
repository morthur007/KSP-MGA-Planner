# Canonical schemas

## candidate_seed.csv

One row per seed candidate.

Mandatory:

```text
candidate_id
rank
sequence or sequence_bodies
raw_sum_km_s
departure_vinf_km_s
arrival_vinf_km_s
powered_flyby_dv_km_s
turn_excess_deg
min_turn_margin_deg
tof_total_days
```

For each event:

```text
event0_BODY_et_s or event0_et_s
```

For each leg N:

```text
legN_dep
legN_arr
legN_tof_days
legN_path
legN_vdep_x_km_s
legN_vdep_y_km_s
legN_vdep_z_km_s
legN_varr_x_km_s
legN_varr_y_km_s
legN_varr_z_km_s
```

## leg_optimization.csv

One row per leg. Must contain final propagated state.

Important fields:

```text
candidate_id,leg,dep_body,arr_body,t_dep_s,t_arr_s,t_start_s,t_end_s
initial_v_*_m_s,dv_*_m_s,optimized_v_*_m_s
start_r_*_m,start_v_*_m_s
target_r_*_m,target_v_*_m_s
final_r_*_m,final_v_*_m_s
final_miss_km,final_relv_m_s,solver_success,solver_message
```

## flyby_audit.json

```json
{
  "candidate_id": "rank1",
  "sequence": ["KERBIN", "EVE", "KERBIN", "JOOL"],
  "status": "PASS|CHECK|FAIL",
  "total_leg_correction_m_s": 0.0,
  "max_vinf_mismatch_km_s": 0.0,
  "min_turn_margin_deg": 0.0,
  "flybys": []
}
```
