"""DeceptionEnv — the decision problem the PPO agent solves.

One episode is one session (attacker or legitimate user). Each step the agent
observes the threat profiler's score plus light context and picks a DefenseAction.
The reward encodes the defender's objective: engage and hold real attackers, block
exfiltration, harvest intel, and avoid disrupting legitimate users or burning
resources.

This abstract environment is fast (no DB seeding) so PPO can train over many
rollouts; the full FinBank test-bed is exercised by the evaluation orchestrator.
"""
from __future__ import annotations

import numpy as np

from addf.attacker import AttackerSimulator, LegitUser
from addf.decoys import get_decoy
from addf.events import DefenseAction

OBS_DIM = 6
N_ACTIONS = len(DefenseAction)


def make_obs(config, threat, t, failed, bytes_out, redirected_last, decoy_active) -> np.ndarray:
    """The agent's observation vector. Shared by the training env and the
    evaluation orchestrator so the trained policy sees identical inputs."""
    return np.array([
        threat,
        t / config.max_steps,
        np.tanh(failed / 5.0),
        np.tanh(bytes_out / 5_000.0),
        1.0 if redirected_last else 0.0,
        1.0 if decoy_active else 0.0,
    ], dtype=np.float64)

# Actions that disrupt a session if aimed at a legitimate user (false positives).
_DISRUPTIVE = {
    DefenseAction.SHADOW_LOGIN_API,
    DefenseAction.CLONE_DATABASE,
    DefenseAction.SYNTHETIC_LEDGER,
    DefenseAction.ISOLATE,
}


class DeceptionEnv:
    def __init__(self, config, rng: np.random.Generator, profiler) -> None:
        self.config = config
        self.rng = rng
        self.profiler = profiler
        self._reset_state()

    def _reset_state(self) -> None:
        self.is_attacker = self.rng.random() > self.config.legit_traffic_ratio
        sim_rng = np.random.default_rng(int(self.rng.integers(0, 2**31)))
        self.sim = AttackerSimulator(sim_rng) if self.is_attacker else LegitUser(sim_rng)
        # Legitimate sessions end after a short, bounded horizon so attacker and
        # legit experience are balanced (otherwise long benign episodes swamp the
        # policy toward never acting).
        self.horizon = self.config.max_steps if self.is_attacker else int(self.rng.integers(5, 10))
        self.events: list = []
        self.t = 0
        self._redirected_last = False
        self._engaged = False
        self.done = False

    def reset(self) -> np.ndarray:
        self._reset_state()
        return self._observe(threat=0.0, failed=0, bytes_out=0, decoy_active=False)

    def _observe(self, threat, failed, bytes_out, decoy_active) -> np.ndarray:
        return make_obs(self.config, threat, self.t, failed, bytes_out,
                        self._redirected_last, decoy_active)

    def step(self, action: int):
        if self.done:
            raise RuntimeError("step() called on a finished episode; call reset().")
        decoy = get_decoy(DefenseAction(action))
        c = self.config

        # Resolve whether the attacker is misdirected onto the decoy this step.
        redirected = self.is_attacker and self.rng.random() < decoy.redirect_prob

        event = self.sim.intended_event(self.t, self._redirected_last)
        event.on_decoy = redirected
        self.events.append(event)

        reward = -decoy.cost * c.decoy_cost  # resource cost of an active decoy
        engaged_decoy = DefenseAction(action) != DefenseAction.NO_OP

        if self.is_attacker:
            if engaged_decoy and not self._engaged:
                reward += c.reward_detect   # one-time value of detecting/engaging a real attacker
                self._engaged = True
            if redirected:
                reward += c.reward_dwell    # holding them inside the decoy, harvesting intel

            if self.sim.attempting_exfiltration():
                if redirected and decoy.blocks_real_exfil:
                    reward += c.reward_exfil_blocked
                else:
                    reward -= c.penalty_exfil   # real records left the bank
                self.done = True
        else:
            if DefenseAction(action) in _DISRUPTIVE:
                reward -= c.penalty_false_positive  # disrupted a legitimate user

        self._redirected_last = redirected
        self.t += 1
        if self.t >= self.horizon:
            self.done = True

        threat = self.profiler.predict_proba(self.events)
        obs = self._observe(threat, event.failed_logins, event.bytes_out,
                            decoy_active=DefenseAction(action) != DefenseAction.NO_OP)
        info = {"is_attacker": self.is_attacker, "redirected": redirected,
                "phase": event.phase.name}
        return obs, reward, self.done, info
