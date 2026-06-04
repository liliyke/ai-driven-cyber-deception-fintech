"""Tests for the ADDF reference implementation.

Fast by design — they use small configs and assert structural correctness plus a
few deterministic properties (e.g. the no-decoy baseline never blocks exfiltration).
"""
from __future__ import annotations

import numpy as np
import pytest

from addf.config import ADDFConfig
from addf.decoys import generate_honeytoken, get_decoy
from addf.env import N_ACTIONS, OBS_DIM, DeceptionEnv
from addf.events import DefenseAction
from addf.finbank import FinBank
from addf.metrics import aggregate
from addf.orchestrator import Orchestrator
from addf.ppo import PPOAgent
from addf.profiler import get_profiler
from addf.training import build_profiler_dataset


@pytest.fixture
def tiny_config() -> ADDFConfig:
    c = ADDFConfig()
    c.train_iterations = 3
    c.rollouts_per_iter = 4
    c.max_steps = 12
    c.n_trials = 3
    c.n_accounts = 20
    return c


def _profiler(c):
    prof = get_profiler(c, np.random.default_rng(c.seed + 1), backend="reservoir")
    X, y = build_profiler_dataset(c, np.random.default_rng(c.seed + 2), n_sessions=120)
    prof.fit(X, y)
    return prof


# --- FinBank ---------------------------------------------------------------
def test_finbank_auth_and_honeytoken(tiny_config):
    bank = FinBank(tiny_config, np.random.default_rng(0))
    assert bank.real_record_count() == tiny_config.n_accounts * 5
    assert bank.authenticate("user0000", "wrong-password") is None

    bank.plant_honeytoken("admin_svc0001", "tripwire")
    s = bank.authenticate("admin_svc0001", "tripwire")
    assert s is not None and s.is_attacker_flagged is True


def test_finbank_uses_only_test_pans(tiny_config):
    bank = FinBank(tiny_config, np.random.default_rng(0))
    rows = bank.db.execute("SELECT pan_masked FROM accounts").fetchall()
    assert all(r["pan_masked"].startswith("411111") for r in rows)


# --- decoys ----------------------------------------------------------------
def test_decoy_registry_complete():
    for action in DefenseAction:
        d = get_decoy(action)
        assert 0.0 <= d.redirect_prob <= 1.0
    assert get_decoy(DefenseAction.NO_OP).cost == 0.0
    assert get_decoy(DefenseAction.CLONE_DATABASE).blocks_real_exfil is True


def test_generate_honeytoken_is_random():
    rng = np.random.default_rng(0)
    assert generate_honeytoken(rng) != generate_honeytoken(rng)


# --- profiler --------------------------------------------------------------
def test_profiler_separates_attacker_from_legit(tiny_config):
    from addf.attacker import AttackerSimulator, LegitUser

    prof = _profiler(tiny_config)

    def score(is_atk):
        rng = np.random.default_rng(1 if is_atk else 2)
        vals = []
        for _ in range(20):
            sim = (AttackerSimulator if is_atk else LegitUser)(np.random.default_rng(int(rng.integers(0, 2**31))))
            ev = [sim.intended_event(t, False) for t in range(4)]
            vals.append(prof.predict_proba(ev))
        return float(np.mean(vals))

    assert score(True) > score(False) + 0.2


# --- environment & agent ---------------------------------------------------
def test_env_step_contract(tiny_config):
    prof = _profiler(tiny_config)
    env = DeceptionEnv(tiny_config, np.random.default_rng(3), prof)
    obs = env.reset()
    assert obs.shape == (OBS_DIM,)
    steps = 0
    done = False
    while not done and steps < tiny_config.max_steps + 1:
        obs, reward, done, info = env.step(int(DefenseAction.NO_OP))
        assert obs.shape == (OBS_DIM,)
        assert isinstance(reward, float)
        steps += 1
    assert done


def test_ppo_trains(tiny_config):
    prof = _profiler(tiny_config)
    env = DeceptionEnv(tiny_config, np.random.default_rng(3), prof)
    agent = PPOAgent(tiny_config, np.random.default_rng(4))
    history = agent.train(env)
    assert len(history) == tiny_config.train_iterations
    a, logp, v = agent.act(env.reset(), greedy=True)
    assert 0 <= a < N_ACTIONS


# --- orchestrator & metrics ------------------------------------------------
def test_orchestrator_trial_fields(tiny_config):
    prof = _profiler(tiny_config)
    agent = PPOAgent(tiny_config, np.random.default_rng(4))
    orch = Orchestrator(tiny_config, prof, agent)
    r = orch.run_trial(np.random.default_rng(5), is_attacker=True, use_agent=False,
                       condition="Baseline")
    assert r.condition == "Baseline" and r.is_attacker is True
    assert r.steps >= 1


def test_baseline_never_blocks_exfiltration(tiny_config):
    """With no decoys the attacker always reaches exfiltration: a deterministic floor."""
    prof = _profiler(tiny_config)
    agent = PPOAgent(tiny_config, np.random.default_rng(4))
    orch = Orchestrator(tiny_config, prof, agent)
    results = [orch.run_trial(np.random.default_rng(10 + i), True, False, "Baseline")
               for i in range(5)]
    agg = aggregate(results, tiny_config)
    assert agg["Baseline"]["exfiltration_prevented_rate"] == 0.0
    assert agg["Baseline"]["detection_rate"] == 1.0
