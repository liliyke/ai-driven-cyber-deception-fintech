"""Build the supervised dataset for the threat profiler.

Runs a mix of attacker and legitimate sessions under random deception actions
(so the profiler sees on-decoy and off-decoy behaviour), and labels each rolling
observation window by whether the session was hostile. The trained profiler then
feeds the PPO agent — the profile-then-decide pipeline from the paper.
"""
from __future__ import annotations

import numpy as np

from addf.attacker import AttackerSimulator, LegitUser
from addf.decoys import get_decoy
from addf.env import N_ACTIONS
from addf.events import DefenseAction
from addf.profiler import window_to_sequence


def build_profiler_dataset(config, rng: np.random.Generator, n_sessions: int = 180):
    sequences: list[np.ndarray] = []
    labels: list[float] = []
    for _ in range(n_sessions):
        is_attacker = rng.random() > config.legit_traffic_ratio
        sim_rng = np.random.default_rng(int(rng.integers(0, 2**31)))
        sim = AttackerSimulator(sim_rng) if is_attacker else LegitUser(sim_rng)
        events: list = []
        redirected_last = False
        n_steps = int(rng.integers(3, 10))
        for t in range(n_steps):
            action = int(rng.integers(0, N_ACTIONS))
            decoy = get_decoy(DefenseAction(action))
            redirected = is_attacker and rng.random() < decoy.redirect_prob
            ev = sim.intended_event(t, redirected_last)
            ev.on_decoy = redirected
            events.append(ev)
            sequences.append(window_to_sequence(events, config.window))
            labels.append(1.0 if is_attacker else 0.0)
            redirected_last = redirected
    return np.asarray(sequences), np.asarray(labels)
