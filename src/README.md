# src/

Reference implementation of the Adaptive Deception Defense Framework (ADDF) described in the
paper. Placeholder — implementation materials are organized here as they are prepared for
release.

Planned components, mirroring the paper's architecture:

- `profiler/` — recurrent-neural threat profiler that classifies live attacker behavior
- `agent/` — Proximal-Policy-Optimization (PPO) deception agent and policy training
- `decoys/` — shadow login API, database cloning, synthetic-ledger injection, honeytokens
- `telemetry/` — IOC capture and threat-intelligence export
- `finbank/` — the vulnerable Flask + MySQL test-bed used for evaluation

Test-bed and evaluation artifacts that aren't code live under [`../data/`](../data).
