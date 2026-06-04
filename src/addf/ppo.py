"""Proximal Policy Optimization — compact numpy implementation.

A linear softmax policy and linear value baseline trained with the clipped PPO
objective, GAE advantages, and an entropy bonus. It learns the deception policy
the paper assigns to its PPO agent, with no deep-learning dependency. When
stable-baselines3 is installed, addf._sb3 offers the same training over a
Gymnasium-wrapped environment.
"""
from __future__ import annotations

import numpy as np

from addf.env import N_ACTIONS, OBS_DIM


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - np.max(z)
    e = np.exp(z)
    return e / np.sum(e)


class PPOAgent:
    def __init__(self, config, rng: np.random.Generator,
                 obs_dim: int = OBS_DIM, n_actions: int = N_ACTIONS,
                 entropy_coef: float = 0.03) -> None:
        self.config = config
        self.rng = rng
        self.n_actions = n_actions
        scale = 0.1
        self.W = rng.normal(0, scale, size=(n_actions, obs_dim))
        self.b = np.zeros(n_actions)
        self.vw = np.zeros(obs_dim)
        self.vb = 0.0
        self.entropy_coef = entropy_coef

    # -- policy / value ------------------------------------------------------
    def _probs(self, obs: np.ndarray) -> np.ndarray:
        return _softmax(self.W @ obs + self.b)

    def value(self, obs: np.ndarray) -> float:
        return float(self.vw @ obs + self.vb)

    def act(self, obs: np.ndarray, greedy: bool = False):
        p = self._probs(obs)
        action = int(np.argmax(p)) if greedy else int(self.rng.choice(self.n_actions, p=p))
        return action, float(np.log(p[action] + 1e-12)), self.value(obs)

    # -- rollout collection --------------------------------------------------
    def _collect(self, env):
        obs_buf, act_buf, lp_buf, rew_buf, val_buf = [], [], [], [], []
        obs = env.reset()
        done = False
        while not done:
            a, lp, v = self.act(obs)
            nobs, r, done, _ = env.step(a)
            obs_buf.append(obs); act_buf.append(a); lp_buf.append(lp)
            rew_buf.append(r); val_buf.append(v)
            obs = nobs
        return (np.array(obs_buf), np.array(act_buf), np.array(lp_buf),
                np.array(rew_buf), np.array(val_buf))

    def _gae(self, rewards: np.ndarray, values: np.ndarray):
        c = self.config
        adv = np.zeros_like(rewards)
        last = 0.0
        for t in reversed(range(len(rewards))):
            next_v = values[t + 1] if t + 1 < len(values) else 0.0
            delta = rewards[t] + c.gamma * next_v - values[t]
            last = delta + c.gamma * c.gae_lambda * last
            adv[t] = last
        return adv, adv + values

    # -- training ------------------------------------------------------------
    def train(self, env, verbose: bool = False) -> list[float]:
        c = self.config
        base_entropy = self.entropy_coef
        history: list[float] = []
        for it in range(c.train_iterations):
            # Anneal exploration so the policy sharpens toward a clean threshold.
            self.entropy_coef = base_entropy * (1.0 - it / max(c.train_iterations - 1, 1))
            O, A, LP, ADV, RET = [], [], [], [], []
            ep_returns = []
            for _ in range(c.rollouts_per_iter):
                obs, acts, lps, rews, vals = self._collect(env)
                adv, ret = self._gae(rews, vals)
                O.append(obs); A.append(acts); LP.append(lps); ADV.append(adv); RET.append(ret)
                ep_returns.append(float(np.sum(rews)))
            O = np.concatenate(O); A = np.concatenate(A); LP = np.concatenate(LP)
            ADV = np.concatenate(ADV); RET = np.concatenate(RET)
            ADV = (ADV - ADV.mean()) / (ADV.std() + 1e-8)
            self._update(O, A, LP, ADV, RET)
            mean_ret = float(np.mean(ep_returns))
            history.append(mean_ret)
            if verbose and (it % 10 == 0 or it == c.train_iterations - 1):
                print(f"  ppo iter {it:3d}  mean_return={mean_ret:+.3f}")
        return history

    def _update(self, O, A, LP_old, ADV, RET) -> None:
        c = self.config
        n = len(A)
        for _ in range(c.ppo_epochs):
            gW = np.zeros_like(self.W); gb = np.zeros_like(self.b)
            gvw = np.zeros_like(self.vw); gvb = 0.0
            for i in range(n):
                obs = O[i]; a = int(A[i]); adv = ADV[i]
                p = self._probs(obs)
                logp = np.log(p[a] + 1e-12)
                ratio = np.exp(logp - LP_old[i])
                # clipped surrogate gradient mask
                if adv >= 0:
                    use = ratio <= (1 + c.clip_eps)
                else:
                    use = ratio >= (1 - c.clip_eps)
                onehot = np.zeros(self.n_actions); onehot[a] = 1.0
                dlogits = (onehot - p)
                if use:
                    pg = adv * ratio * dlogits
                else:
                    pg = np.zeros(self.n_actions)
                # entropy bonus gradient: dH/dz_i = -p_i (log p_i + H)
                H = -np.sum(p * np.log(p + 1e-12))
                ent = -p * (np.log(p + 1e-12) + H)
                glog = pg + self.entropy_coef * ent
                gW += np.outer(glog, obs); gb += glog
                # value regression (descent)
                v = self.value(obs)
                verr = v - RET[i]
                gvw += verr * obs; gvb += verr
            # ascend policy, descend value
            self.W += c.policy_lr * gW / n
            self.b += c.policy_lr * gb / n
            self.vw -= c.value_lr * gvw / n
            self.vb -= c.value_lr * gvb / n
