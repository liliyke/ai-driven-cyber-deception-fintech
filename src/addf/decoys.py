"""Decoy registry.

The deception actions the PPO agent can deploy, each described by how it affects
an attacker session: how likely it is to misdirect the attacker onto fake assets
this step, whether it blocks real data exfiltration, whether it positively reveals
the session as hostile (a honeytoken or shadow-login hit is high-confidence), and
its per-step resource cost.

These properties are what the closed-loop environment uses to resolve outcomes.
"""
from __future__ import annotations

import secrets
from dataclasses import dataclass

import numpy as np

from addf.events import DefenseAction


@dataclass(frozen=True)
class Decoy:
    action: DefenseAction
    name: str
    redirect_prob: float     # chance the attacker is misdirected onto the decoy this step
    blocks_real_exfil: bool   # real records cannot leave while this is engaged
    reveals_attacker: bool    # interaction is high-confidence attacker attribution
    harvests_intel: bool      # produces IOCs (isolation does not)
    cost: float               # relative per-step resource cost


# Tuned so that engagement-style decoys (shadow login, DB clone, synthetic ledger)
# trade higher cost for stronger misdirection and intel, while ISOLATE is a blunt,
# intel-poor cut-off and NO_OP is free.
_REGISTRY: dict[DefenseAction, Decoy] = {
    DefenseAction.NO_OP: Decoy(
        DefenseAction.NO_OP, "no-op", 0.0, False, False, False, 0.0),
    DefenseAction.HONEYTOKEN: Decoy(
        DefenseAction.HONEYTOKEN, "honeytoken", 0.35, False, True, True, 0.4),
    DefenseAction.SHADOW_LOGIN_API: Decoy(
        DefenseAction.SHADOW_LOGIN_API, "shadow-login-api", 0.75, True, True, True, 0.8),
    DefenseAction.CLONE_DATABASE: Decoy(
        DefenseAction.CLONE_DATABASE, "clone-database", 0.85, True, False, True, 1.0),
    DefenseAction.SYNTHETIC_LEDGER: Decoy(
        DefenseAction.SYNTHETIC_LEDGER, "synthetic-ledger", 0.80, True, False, True, 0.7),
    DefenseAction.ISOLATE: Decoy(
        DefenseAction.ISOLATE, "isolate", 1.0, True, False, False, 0.9),
}


def get_decoy(action: DefenseAction) -> Decoy:
    return _REGISTRY[action]


def generate_honeytoken(rng: np.random.Generator) -> tuple[str, str]:
    """A tempting but fake credential. Any use of it is attacker-attributable."""
    idx = int(rng.integers(0, 9999))
    return f"admin_svc{idx:04d}", secrets.token_hex(6)
