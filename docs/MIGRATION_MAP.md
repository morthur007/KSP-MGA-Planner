# Migration map from old prototype to KSP-MGA-Planner

## Search layer

spice_lambert_beam_search_v0_2.py
→ ksp_mga/lambert/beam_search.py

spice_flyby_family_search_v0_1.py
→ ksp_mga/lambert/family_search.py

pykep_gateway_v0_1.py
→ ksp_mga/lambert/pykep_gateway.py

## Native / Principia layer

native_optimize_candidate_legs_v0_1.py
→ ksp_mga/native/leg_optimizer.py

native_corrected_flyby_audit_v0_1.py
→ ksp_mga/native/flyby_audit.py

native_powered_flyby_bridge_v0_1.py
→ ksp_mga/native/powered_bridge.py

PrincipiaImpulseServer / daemon client code
→ ksp_mga/native/impulse_server_client.py

## Pipeline

nbody_pipeline_orchestrator_v0_1.py
→ ksp_mga/pipeline/triage.py

## C++ tools

principia_particle_validator.cpp
principia_impulsive_particle_validator.cpp
principia_impulsive_particle_server.cpp
sample_principia_ephemeris.cpp
dump_current_snapshot.cpp
→ native_cpp/ksp_plugin_test/
