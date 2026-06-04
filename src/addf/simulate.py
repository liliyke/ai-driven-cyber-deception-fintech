"""End-to-end ADDF simulation: train, then evaluate baseline vs ADDF.

    python -m addf.simulate            # full run, prints the comparison table
    python -m addf.simulate --quiet    # metrics only
    addf-simulate --trials 20          # via the installed console script

Reproduces the *shape* of the paper's evaluation (time-to-detect, attacker dwell,
exfiltration prevented, false-positive rate, resource overhead) on a synthetic
FinBank test-bed. The absolute figures come from this reference simulation, not
the authors' original experimental environment.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from addf.config import ADDFConfig
from addf.env import DeceptionEnv
from addf.metrics import aggregate, render_table
from addf.orchestrator import Orchestrator
from addf.ppo import PPOAgent
from addf.profiler import get_profiler
from addf.telemetry import TelemetryCollector
from addf.training import build_profiler_dataset


def run(config: ADDFConfig | None = None, *, verbose: bool = True,
        write: bool = True, profiler_backend: str = "auto") -> dict:
    c = config or ADDFConfig()
    log = print if verbose else (lambda *a, **k: None)

    log(f"ADDF reference simulation (seed={c.seed})\n")

    log("[1/3] training recurrent threat profiler ...")
    profiler = get_profiler(c, np.random.default_rng(c.seed + 1), backend=profiler_backend)
    X, y = build_profiler_dataset(c, np.random.default_rng(c.seed + 2))
    profiler.fit(X, y)
    log(f"      trained on {len(y)} labelled windows "
        f"({int(y.sum())} hostile / {int((1 - y).sum())} benign)\n")

    log("[2/3] training PPO deception agent ...")
    env = DeceptionEnv(c, np.random.default_rng(c.seed + 3), profiler)
    agent = PPOAgent(c, np.random.default_rng(c.seed + 4))
    agent.train(env, verbose=verbose)
    log("")

    log("[3/3] evaluating baseline vs ADDF over attack + legitimate trials ...\n")
    orch = Orchestrator(c, profiler, agent)
    collector = TelemetryCollector(c.window)
    results = []
    for i in range(c.n_trials):
        results.append(orch.run_trial(np.random.default_rng(c.seed + 100 + i), True, False, "Baseline"))
    for i in range(c.n_trials):
        results.append(orch.run_trial(np.random.default_rng(c.seed + 200 + i), True, True, "ADDF", collector))
    for i in range(c.n_trials):
        results.append(orch.run_trial(np.random.default_rng(c.seed + 300 + i), False, False, "Baseline"))
    for i in range(c.n_trials):
        results.append(orch.run_trial(np.random.default_rng(c.seed + 400 + i), False, True, "ADDF"))

    agg = aggregate(results, c)
    ioc = collector.ioc_report()

    if verbose:
        print(render_table(agg))
        print(f"\nThreat intelligence: ADDF harvested {len(ioc['indicators'])} unique IOCs "
              f"from {ioc['decoy_interactions']} decoy interactions.")
        print("\nFigures are outputs of this reference simulation, not the paper's original test-bed.")

    if write:
        os.makedirs(os.path.dirname(c.results_path) or ".", exist_ok=True)
        with open(c.results_path, "w", encoding="utf-8") as fh:
            json.dump({"seed": c.seed, "n_trials": c.n_trials,
                       "metrics": agg, "ioc_report": ioc}, fh, indent=2)
        log(f"\nWrote {c.results_path}")

    return agg


def main() -> None:
    p = argparse.ArgumentParser(description="ADDF reference simulation")
    p.add_argument("--trials", type=int, help="attack/legit trials per condition")
    p.add_argument("--seed", type=int, help="random seed")
    p.add_argument("--profiler", default="auto", choices=["auto", "reservoir", "torch"],
                   help="threat-profiler backend")
    p.add_argument("--quiet", action="store_true", help="metrics only")
    p.add_argument("--no-write", action="store_true", help="don't write data/results.json")
    args = p.parse_args()

    cfg = ADDFConfig()
    if args.trials:
        cfg.n_trials = args.trials
    if args.seed is not None:
        cfg.seed = args.seed
    run(cfg, verbose=not args.quiet, write=not args.no_write, profiler_backend=args.profiler)


if __name__ == "__main__":
    main()
