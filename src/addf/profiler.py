"""Recurrent threat profiler.

The paper uses a recurrent-neural network to "classify live attacker behaviour".
The default implementation here is an Echo-State Network (a reservoir of fixed
random recurrent weights with a trained linear read-out): a genuine recurrent
neural model that trains in closed form / by simple gradient descent on numpy
alone, so the whole pipeline runs without a deep-learning stack.

`get_profiler(..., backend="torch")` swaps in a trained GRU classifier when
PyTorch is installed — same `fit` / `predict_proba` interface.
"""
from __future__ import annotations

import numpy as np

from addf.events import Event

# Observable endpoint categories (the profiler never sees the ground-truth actor/phase).
ENDPOINT_CATEGORIES = ["login", "accounts", "transfer", "ledger", "internal", "scan"]
_ENDPOINT_INDEX = {name: i for i, name in enumerate(ENDPOINT_CATEGORIES)}
FEATURE_DIM = len(ENDPOINT_CATEGORIES) + 5  # + success, on_decoy, failed_logins, bytes_out, is_write


def _endpoint_category(endpoint: str) -> int:
    ep = endpoint.lower().strip("/")
    for name, idx in _ENDPOINT_INDEX.items():
        if name in ep:
            return idx
    return _ENDPOINT_INDEX["scan"]


def event_features(event: Event) -> np.ndarray:
    """Behavioural signal for one event — deliberately excludes actor/phase labels."""
    vec = np.zeros(FEATURE_DIM, dtype=np.float64)
    vec[_endpoint_category(event.endpoint)] = 1.0
    base = len(ENDPOINT_CATEGORIES)
    vec[base + 0] = 1.0 if event.success else 0.0
    vec[base + 1] = 1.0 if event.on_decoy else 0.0
    vec[base + 2] = np.tanh(event.failed_logins / 5.0)
    vec[base + 3] = np.tanh(event.bytes_out / 5_000.0)
    vec[base + 4] = 1.0 if event.endpoint.lower().strip("/") in ("transfer", "ledger") else 0.0
    return vec


def window_to_sequence(events: list[Event], window: int) -> np.ndarray:
    """Left-pad a window of events into a (window, FEATURE_DIM) array."""
    seq = np.zeros((window, FEATURE_DIM), dtype=np.float64)
    recent = events[-window:]
    for i, ev in enumerate(recent):
        seq[window - len(recent) + i] = event_features(ev)
    return seq


class ReservoirProfiler:
    """Echo-State Network: fixed recurrent reservoir + trained logistic read-out."""

    def __init__(self, config, rng: np.random.Generator) -> None:
        self.config = config
        self.rng = rng
        n, f = config.reservoir_size, FEATURE_DIM
        self.w_in = rng.uniform(-0.5, 0.5, size=(n, f))
        w = rng.normal(0.0, 1.0, size=(n, n))
        radius = np.max(np.abs(np.linalg.eigvals(w)))
        self.w = w * (config.spectral_radius / (radius + 1e-9))
        self.leak = config.reservoir_leak
        self.w_out = np.zeros(n)
        self.b_out = 0.0
        self._fitted = False

    def _final_state(self, seq: np.ndarray) -> np.ndarray:
        x = np.zeros(self.config.reservoir_size)
        for u in seq:
            pre = self.w_in @ u + self.w @ x
            x = (1.0 - self.leak) * x + self.leak * np.tanh(pre)
        return x

    def _readout_vector(self, seq: np.ndarray) -> np.ndarray:
        """Recurrent state concatenated with a max-pool of the window's features,
        giving the linear read-out direct sight of discriminative signals
        (credential-stuffing failures, recon/lateral endpoints, exfil volume)."""
        return np.concatenate([self._final_state(seq), seq.max(axis=0)])

    def _states(self, sequences: np.ndarray) -> np.ndarray:
        return np.vstack([self._readout_vector(s) for s in sequences])

    def fit(self, sequences: np.ndarray, labels: np.ndarray, epochs: int = 300,
            lr: float = 0.1, l2: float = 1e-3) -> "ReservoirProfiler":
        states = self._states(sequences)
        y = np.asarray(labels, dtype=np.float64)
        w, b = np.zeros(states.shape[1]), 0.0
        m = len(y)
        for _ in range(epochs):
            z = states @ w + b
            p = 1.0 / (1.0 + np.exp(-z))
            grad_w = states.T @ (p - y) / m + l2 * w
            grad_b = float(np.mean(p - y))
            w -= lr * grad_w
            b -= lr * grad_b
        self.w_out, self.b_out, self._fitted = w, b, True
        return self

    def predict_proba(self, events: list[Event]) -> float:
        seq = window_to_sequence(events, self.config.window)
        x = self._readout_vector(seq)
        z = float(x @ self.w_out + self.b_out)
        return 1.0 / (1.0 + np.exp(-z))


def get_profiler(config, rng: np.random.Generator, backend: str = "auto"):
    """Return a profiler. backend: 'reservoir' | 'torch' | 'auto'."""
    if backend in ("torch", "auto"):
        try:
            from addf._torch_profiler import GRUProfiler  # optional
            return GRUProfiler(config, rng)
        except Exception:
            if backend == "torch":
                raise
    return ReservoirProfiler(config, rng)
