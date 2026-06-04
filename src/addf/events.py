"""Event model shared across the framework.

Telemetry is a stream of `Event`s. Attacker behaviour moves through `AttackPhase`s;
the agent responds with a `DefenseAction`. MITRE ATT&CK technique IDs are attached
to attacker events so the telemetry doubles as threat intelligence.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum


class Actor(IntEnum):
    LEGIT = 0      # a legitimate FinBank customer
    ATTACKER = 1


class AttackPhase(IntEnum):
    """Coarse kill-chain phase. LEGIT traffic is tagged BENIGN."""
    BENIGN = 0
    RECON = 1
    CREDENTIAL_STUFFING = 2
    ACCESS = 3
    LATERAL_MOVEMENT = 4
    DISCOVERY = 5
    EXFILTRATION = 6


class DefenseAction(IntEnum):
    """Actions the PPO agent can take in response to an observation window."""
    NO_OP = 0
    HONEYTOKEN = 1          # plant a honeytoken credential / record
    SHADOW_LOGIN_API = 2    # redirect to a decoy authentication surface
    CLONE_DATABASE = 3      # serve a cloned, fake database
    SYNTHETIC_LEDGER = 4    # inject a synthetic transaction ledger
    ISOLATE = 5             # cut the session off from real assets


# MITRE ATT&CK technique IDs associated with each attack phase (for IOC export).
PHASE_TECHNIQUES: dict[AttackPhase, str] = {
    AttackPhase.RECON: "T1595",                 # Active Scanning
    AttackPhase.CREDENTIAL_STUFFING: "T1110.004",  # Credential Stuffing
    AttackPhase.ACCESS: "T1078",                # Valid Accounts
    AttackPhase.LATERAL_MOVEMENT: "T1021",      # Remote Services
    AttackPhase.DISCOVERY: "T1087",             # Account Discovery
    AttackPhase.EXFILTRATION: "T1041",          # Exfiltration Over C2 Channel
}


@dataclass
class Event:
    """A single observed interaction with FinBank."""
    t: int
    actor: Actor
    phase: AttackPhase
    endpoint: str
    source_ip: str
    success: bool = False
    on_decoy: bool = False           # interaction landed on a decoy asset
    bytes_out: int = 0               # data volume leaving the system (real exfil signal)
    failed_logins: int = 0
    technique: str | None = None
    note: str = ""

    def __post_init__(self) -> None:
        if self.technique is None:
            self.technique = PHASE_TECHNIQUES.get(self.phase)


@dataclass
class IOC:
    """An indicator of compromise harvested from a decoy interaction."""
    kind: str            # ip | credential | technique | tool
    value: str
    first_seen: int
    technique: str | None = None
    context: dict = field(default_factory=dict)
