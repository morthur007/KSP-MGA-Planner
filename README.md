# ksp_mga refactor base

Base limpa para organizar o pipeline MGA do KSP/Principia.

Objetivo imediato:

1. Congelar contratos de dados.
2. Separar biblioteca de scripts CLI.
3. Centralizar frames, unidades, schemas e cliente do daemon C++.
4. Evitar scripts Python chamando outros scripts Python via subprocess.

## Estrutura

```text
ksp_mga/
  core/
    config.py
    schemas.py
    transforms.py
    units.py
  ephem/
    spice.py
  lambert/
    pykep_gateway.py
  native/
    impulse_server_client.py
    leg_optimizer.py
    flyby_audit.py
    powered_bridge.py
  pipeline/
    triage.py
scripts/
  run_family_search.py
  run_leg_optimizer.py
  run_flyby_audit.py
  run_triage.py
configs/
  jnsq_kerbin_to_jool.example.yaml
```

## Contratos principais

### Candidate seed

Gerado pelo family/beam search. Deve conter o suficiente para não resolver Lambert de novo no refinador.

Campos essenciais:

```text
candidate_id, rank, sequence, sequence_bodies, n_legs,
event0_body,event0_et_s,...,
leg1_dep,leg1_arr,leg1_tof_days,leg1_path,
leg1_vdep_x_km_s,leg1_vdep_y_km_s,leg1_vdep_z_km_s,
leg1_varr_x_km_s,leg1_varr_y_km_s,leg1_varr_z_km_s,
raw_sum_km_s,departure_vinf_km_s,arrival_vinf_km_s,
powered_flyby_dv_km_s,turn_excess_deg,min_turn_margin_deg,tof_total_days
```

### Leg optimization

Gerado pelo refinador N-body. Deve conter estado inicial, alvo e estado final real.

### Flyby audit

JSON com `PASS|CHECK|FAIL`, métricas de v-infinity e margem de curva.

## Migração sugerida

1. Copie esta pasta para a raiz do repo.
2. Faça commit.
3. Mova o cliente do daemon antigo para `ksp_mga/native/impulse_server_client.py` ou use o cliente aqui.
4. Ajuste o beam search para gravar velocidades por perna no `candidate_seed.csv`.
5. Faça o leg optimizer consumir `CandidateSeed` em vez de reler/reconstruir Lambert.
6. Só depois reative orchestrator, powered bridge e UI.

## Instalação local

```bash
pip install -e .
```

ou direto pelo `PYTHONPATH`:

```bash
export PYTHONPATH=$PWD:$PYTHONPATH
```
