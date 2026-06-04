"""Benign attacker and legitimate-user simulators.

To train and evaluate a *defence* you need adversary behaviour to defend against.
This is a synthetic behaviour generator — kill-chain phase transitions and traffic
shapes — not an exploit toolkit. It emits the *intended* action each step; the
environment then applies the active decoy to decide what actually happens
(misdirection, blocked exfiltration, intel capture).

Modelled attacker path: recon -> credential stuffing -> access -> lateral movement
-> discovery -> exfiltration, mirroring the perimeter-bypass scenarios in the paper.
"""
from __future__ import annotations

import numpy as np

from addf.events import Actor, AttackPhase, Event

# The endpoint an actor touches in each attack phase.
_PHASE_ENDPOINT = {
    AttackPhase.RECON: "/scan",
    AttackPhase.CREDENTIAL_STUFFING: "/login",
    AttackPhase.ACCESS: "/accounts",
    AttackPhase.LATERAL_MOVEMENT: "/internal",
    AttackPhase.DISCOVERY: "/ledger",
    AttackPhase.EXFILTRATION: "/ledger",
}
_KILL_CHAIN = [
    AttackPhase.RECON,
    AttackPhase.CREDENTIAL_STUFFING,
    AttackPhase.ACCESS,
    AttackPhase.LATERAL_MOVEMENT,
    AttackPhase.DISCOVERY,
    AttackPhase.EXFILTRATION,
]


class AttackerSimulator:
    def __init__(self, rng: np.random.Generator, source_ip: str | None = None) -> None:
        self.rng = rng
        self.source_ip = source_ip or f"203.0.113.{int(rng.integers(2, 254))}"
        self._phase_idx = -1  # first intended_event advances to RECON
        self.dwell_steps = 0  # steps spent misdirected on decoys

    @property
    def phase(self) -> AttackPhase:
        return _KILL_CHAIN[max(self._phase_idx, 0)]

    def intended_event(self, t: int, redirected_last: bool) -> Event:
        """Produce the next intended interaction.

        If the previous step misdirected the attacker onto a decoy, they linger
        (dwell) instead of advancing down the kill chain.
        """
        if redirected_last:
            self.dwell_steps += 1
        elif self._phase_idx < len(_KILL_CHAIN) - 1:
            self._phase_idx += 1

        phase = self.phase
        endpoint = _PHASE_ENDPOINT[phase]
        failed = int(self.rng.integers(3, 9)) if phase == AttackPhase.CREDENTIAL_STUFFING else 0
        # A real exfiltration attempt moves a large volume of records out.
        bytes_out = int(self.rng.integers(4_000, 9_000)) if phase == AttackPhase.EXFILTRATION else int(self.rng.integers(0, 200))
        return Event(
            t=t, actor=Actor.ATTACKER, phase=phase, endpoint=endpoint,
            source_ip=self.source_ip, success=phase != AttackPhase.CREDENTIAL_STUFFING,
            bytes_out=bytes_out, failed_logins=failed,
        )

    def attempting_exfiltration(self) -> bool:
        return self.phase == AttackPhase.EXFILTRATION


class LegitUser:
    def __init__(self, rng: np.random.Generator, source_ip: str | None = None) -> None:
        self.rng = rng
        self.source_ip = source_ip or f"198.51.100.{int(rng.integers(2, 254))}"

    def intended_event(self, t: int, redirected_last: bool) -> Event:
        endpoint = str(self.rng.choice(["/login", "/accounts", "/transfer", "/ledger"]))
        return Event(
            t=t, actor=Actor.LEGIT, phase=AttackPhase.BENIGN, endpoint=endpoint,
            source_ip=self.source_ip, success=True,
            bytes_out=int(self.rng.integers(0, 120)), failed_logins=0,
        )

    def attempting_exfiltration(self) -> bool:
        return False
