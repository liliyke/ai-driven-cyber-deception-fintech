# src/

The `addf` package — the ADDF reference implementation. See the
[top-level README](../README.md) for the architecture diagram, the module map, and usage.

Quick start:

```bash
pip install -e .
python -m addf.simulate
```

Module overview: `config` (tunables) · `events` (event/IOC model) · `finbank` (test-bed) ·
`telemetry` (stream + IOC export) · `profiler` (recurrent threat profiler) · `decoys` ·
`attacker` (simulators) · `env` (the agent's MDP) · `ppo` (agent) · `orchestrator` (closed loop) ·
`metrics` · `simulate` (CLI).
