# AI-Driven Cyber Deception in FinTech: An Adaptive Defense Strategy

Companion repository for the peer-reviewed paper presented at the **6th International Conference
on AI Research (ICAIR 2025)**.

> **Isaac Ojeh, Xavier Palmer, Lucas Potter** (BiosView Labs)
> *Proceedings of the 6th International Conference on AI Research (ICAIR 2025)*, Vol. 6 No. 1.
> Academic Conferences International. Published 2025-12-04.
> DOI: [10.34190/icair.5.1.4365](https://doi.org/10.34190/icair.5.1.4365)

## Abstract

FinTech platforms clear high-value transactions in milliseconds, making them lucrative targets
for adversaries who increasingly weaponize artificial intelligence. Once an attacker bypasses the
perimeter — via credential stuffing, supply-chain malware, or deep-fake social engineering —
traditional defenses often alert too late to prevent loss. We present an **Adaptive Deception
Defense Framework (ADDF)** that intertwines AI-orchestrated honeypots, honeytokens and decoy
micro-services within everyday banking and payment workflows. A recurrent-neural threat profiler
classifies live attacker behavior; a Proximal-Policy-Optimization agent then selects actions such
as spawning a shadow login API, cloning a database or injecting synthetic ledgers, thereby
misdirecting intruders while harvesting telemetry. In a controlled "FinBank" test-bed featuring a
vulnerable Flask-and-MySQL stack, ADDF shortened mean time-to-detect from 3 min 42 s to 29 s,
increased attacker dwell-time inside decoys to 12 min 18 s, and prevented all real data
exfiltration across ten attack trials. False-positive alerts remained below 1% per run, and added
resource use averaged 14% CPU/RAM on mid-range servers. The framework also produced high-fidelity
indicators of compromise — password lists, malware binaries and lateral-movement scripts — that
would have been unavailable under baseline controls. These findings indicate that AI-driven cyber
deception can transform FinTech security from passive monitoring into proactive engagement,
mitigating breach impact while supplying rich threat intelligence. The paper details system
architecture, reinforcement-learning policy training, empirical evaluation and operational
implications — showing how defenders can regain initiative in the AI-to-AI cyber arms race without
disrupting legitimate customers or breaching regulatory duties.

**Keywords:** AI-driven, deception, ADDF, honeypots, machine learning, adaptive, proactive
defense, FinTech.

## The framework in brief

The **Adaptive Deception Defense Framework (ADDF)** is a closed-loop deception system that
orchestrates established AI components into real-time FinTech operations:

- **Deception layer** — honeypots, honeytokens and decoy micro-services woven into live banking
  and payment workflows, rather than bolted on at the perimeter.
- **Threat profiler** — a recurrent neural network classifies attacker behavior as it unfolds.
- **Decision agent** — a Proximal-Policy-Optimization (PPO) reinforcement-learning agent selects
  the deception response: spawn a shadow login API, clone a database, inject synthetic ledgers.
- **Outcome** — intruders are misdirected into fake assets while the system harvests
  high-fidelity telemetry and indicators of compromise.

## Results

Evaluated on the **FinBank** test-bed (a deliberately vulnerable Flask + MySQL stack), across ten
attack trials:

| Metric | Baseline | ADDF |
|---|---|---|
| Mean time-to-detect | 3 min 42 s | **29 s** |
| Attacker dwell-time inside decoys | — | **12 min 18 s** |
| Real data exfiltration | — | **0 / 10 trials** |
| False-positive alerts | — | **< 1% per run** |
| Added resource overhead | — | **~14% CPU/RAM** (mid-range server) |

ADDF also generated threat intelligence — captured password lists, malware binaries and
lateral-movement scripts — unavailable under baseline monitoring.

## Repository structure

```
.
├── src/    # ADDF implementation — reference (placeholder; see src/README.md)
├── data/   # test-bed configuration and evaluation artifacts (placeholder; see data/README.md)
├── CITATION.cff
└── references.bib
```

The paper is the primary artifact. Implementation and evaluation materials are organized under
`src/` and `data/` as they are prepared for release.

## Citation

If you reference this work, please cite the paper (see [`CITATION.cff`](CITATION.cff) and
[`references.bib`](references.bib)):

```bibtex
@inproceedings{ojeh2025addf,
  title     = {AI Driven Cyber Deception in {FinTech}: An Adaptive Defense Strategy},
  author    = {Ojeh, Isaac and Palmer, Xavier and Potter, Lucas},
  booktitle = {Proceedings of the 6th International Conference on AI Research (ICAIR 2025)},
  volume    = {6},
  number    = {1},
  year      = {2025},
  month     = dec,
  publisher = {Academic Conferences International},
  doi       = {10.34190/icair.5.1.4365}
}
```

## Authors

- **Isaac Ojeh**
- **Xavier Palmer** — BiosView Labs
- **Lucas Potter** — BiosView Labs

## Licence

This repository is licensed under **CC-BY-4.0** (see [`LICENSE`](LICENSE)) — reuse with
attribution. If substantial implementation code is later added under `src/`, it may carry a
separate permissive (MIT) licence noted there.
