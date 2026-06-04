"""Central configuration for the ADDF simulation.

Every tunable lives here so experiments are reproducible from a single object.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ADDFConfig:
    # Reproducibility
    seed: int = 7

    # FinBank testbed
    n_accounts: int = 200
    starting_balance: float = 5_000.0

    # Telemetry / profiler
    window: int = 12          # events per observation window fed to the profiler
    reservoir_size: int = 64  # Echo-State reservoir hidden units (recurrent profiler)
    spectral_radius: float = 0.9
    reservoir_leak: float = 0.3
    threat_threshold: float = 0.5  # profiler score above which we treat traffic as hostile

    # PPO deception agent
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    policy_lr: float = 0.02
    value_lr: float = 0.05
    train_iterations: int = 90
    rollouts_per_iter: int = 24
    ppo_epochs: int = 4

    # Reward shaping (defender's objective)
    reward_detect: float = 1.0        # correctly engaging a real attacker
    reward_dwell: float = 0.05        # per-step bonus while an attacker is held in a decoy
    reward_exfil_blocked: float = 3.0 # preventing real data exfiltration
    penalty_exfil: float = 5.0        # a real exfiltration getting through
    penalty_false_positive: float = 1.5  # deceiving a legitimate user
    decoy_cost: float = 0.04          # per-step resource cost of an active decoy

    # Evaluation
    n_trials: int = 10                # attack trials per condition (matches the paper)
    max_steps: int = 30               # steps per trial
    legit_traffic_ratio: float = 0.5  # share of sessions that are benign users
    seconds_per_step: float = 30.0    # wall-clock mapping for reporting MTTD/dwell in seconds
    cpu_per_active_decoy_pct: float = 15.0  # resource-overhead proxy when a decoy is engaged

    # Output
    results_path: str = "data/results.json"

    extra: dict = field(default_factory=dict)
