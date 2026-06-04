"""Adaptive Deception Defense Framework (ADDF).

Reference implementation of the architecture described in:

    Ojeh, I., Palmer, X., & Potter, L. (2025).
    "AI Driven Cyber Deception in FinTech: An Adaptive Defense Strategy."
    Proceedings of the 6th International Conference on AI Research (ICAIR 2025).
    DOI: 10.34190/icair.5.1.4365

The closed loop is: FinBank testbed + telemetry -> recurrent threat profiler ->
PPO deception agent -> decoys -> capture, repeat. The default code path runs on
numpy alone; PyTorch / stable-baselines3 / Flask are optional upgrades.
"""

__version__ = "0.1.0"

from addf.config import ADDFConfig

__all__ = ["ADDFConfig", "__version__"]
