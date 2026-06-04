"""Optional PyTorch GRU threat profiler.

This is the "recurrent-neural threat profiler" as a trained GRU, matching the
paper's description more literally than the default reservoir. It is only imported
when PyTorch is available (see addf.profiler.get_profiler); the rest of the
framework treats it identically to ReservoirProfiler.

    pip install -e .[torch]
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from addf.events import Event
from addf.profiler import FEATURE_DIM, window_to_sequence


class _GRUNet(nn.Module):
    def __init__(self, feature_dim: int, hidden: int) -> None:
        super().__init__()
        self.gru = nn.GRU(feature_dim, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, h = self.gru(x)
        return self.head(h[-1]).squeeze(-1)


class GRUProfiler:
    def __init__(self, config, rng: np.random.Generator) -> None:
        self.config = config
        torch.manual_seed(int(config.seed))
        self.net = _GRUNet(FEATURE_DIM, config.reservoir_size)
        self._fitted = False

    def fit(self, sequences: np.ndarray, labels: np.ndarray, epochs: int = 150,
            lr: float = 1e-2) -> "GRUProfiler":
        x = torch.tensor(np.asarray(sequences), dtype=torch.float32)
        y = torch.tensor(np.asarray(labels), dtype=torch.float32)
        opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        loss_fn = nn.BCEWithLogitsLoss()
        self.net.train()
        for _ in range(epochs):
            opt.zero_grad()
            loss = loss_fn(self.net(x), y)
            loss.backward()
            opt.step()
        self._fitted = True
        return self

    @torch.no_grad()
    def predict_proba(self, events: list[Event]) -> float:
        seq = window_to_sequence(events, self.config.window)
        x = torch.tensor(seq[None, :, :], dtype=torch.float32)
        self.net.eval()
        return float(torch.sigmoid(self.net(x)).item())
