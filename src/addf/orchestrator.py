"""ADDF closed loop and the evaluation orchestrator.

Runs one session against the FinBank test-bed: observe telemetry -> profile the
threat -> the PPO agent picks a deception action -> resolve it against the bank
(misdirection, blocked or successful exfiltration, intel capture) -> repeat. This
is where the framework's behaviour is measured for the paper's metrics.

`use_agent=False` reproduces the baseline ("passive monitoring"): no decoys, and
detection only when the loud exfiltration signal finally appears.
"""
from __future__ import annotations

import numpy as np

from addf.attacker import AttackerSimulator, LegitUser
from addf.decoys import get_decoy
from addf.env import _DISRUPTIVE, make_obs
from addf.events import DefenseAction
from addf.finbank import FinBank
from addf.metrics import TrialResult
from addf.telemetry import TelemetryCollector

_EXFIL_BYTES = 4_000  # volume a passive monitor would finally alert on


class Orchestrator:
    def __init__(self, config, profiler, agent) -> None:
        self.config = config
        self.profiler = profiler
        self.agent = agent

    def run_trial(self, rng: np.random.Generator, is_attacker: bool, use_agent: bool,
                  condition: str, collector: TelemetryCollector | None = None) -> TrialResult:
        c = self.config
        sim_rng = np.random.default_rng(int(rng.integers(0, 2**31)))
        sim = AttackerSimulator(sim_rng) if is_attacker else LegitUser(sim_rng)
        bank = FinBank(c, np.random.default_rng(int(rng.integers(0, 2**31)))) if is_attacker else None

        events: list = []
        redirected_last = False
        decoy_active_prev = False
        detect_step: int | None = None
        dwell = 0
        active_steps = 0
        exfiltration_prevented = True
        false_positive = False
        steps = 0

        for t in range(c.max_steps):
            steps = t + 1
            threat = self.profiler.predict_proba(events) if events else 0.0
            last = events[-1] if events else None
            obs = make_obs(c, threat, t,
                           last.failed_logins if last else 0,
                           last.bytes_out if last else 0,
                           redirected_last, decoy_active_prev)
            action = (self.agent.act(obs, greedy=True)[0] if use_agent
                      else int(DefenseAction.NO_OP))
            decoy = get_decoy(DefenseAction(action))
            if DefenseAction(action) != DefenseAction.NO_OP:
                active_steps += 1

            redirected = is_attacker and rng.random() < decoy.redirect_prob
            ev = sim.intended_event(t, redirected_last)
            ev.on_decoy = redirected
            events.append(ev)
            if collector is not None:
                collector.record(ev)

            if is_attacker:
                # Engagement is high-confidence detection: only an attacker is ever
                # misdirected onto a decoy.
                if redirected and detect_step is None:
                    detect_step = t
                # Passive monitoring only catches the loud exfiltration signal.
                if detect_step is None and ev.bytes_out >= _EXFIL_BYTES:
                    detect_step = t
                if redirected:
                    dwell += 1
                if sim.attempting_exfiltration():
                    if redirected and decoy.blocks_real_exfil:
                        exfiltration_prevented = True
                    else:
                        exfiltration_prevented = False
                        if bank is not None:
                            bank.register_real_exfiltration(bank.real_record_count())
                    break
            elif DefenseAction(action) in _DISRUPTIVE:
                false_positive = True

            redirected_last = redirected
            decoy_active_prev = DefenseAction(action) != DefenseAction.NO_OP

        return TrialResult(
            condition=condition, is_attacker=is_attacker,
            detected=detect_step is not None, detect_step=detect_step,
            dwell_steps=dwell, exfiltration_prevented=exfiltration_prevented if is_attacker else True,
            false_positive=false_positive, active_decoy_steps=active_steps, steps=steps,
        )
