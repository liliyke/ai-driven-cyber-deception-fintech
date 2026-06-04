"""Trial results and aggregation into the paper's evaluation metrics."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TrialResult:
    condition: str
    is_attacker: bool
    detected: bool
    detect_step: int | None
    dwell_steps: int
    exfiltration_prevented: bool
    false_positive: bool
    active_decoy_steps: int
    steps: int


def aggregate(results: list[TrialResult], config) -> dict:
    """Reduce raw trial results to the headline metrics, per condition."""
    out: dict[str, dict] = {}
    conditions = sorted({r.condition for r in results})
    for cond in conditions:
        attacker = [r for r in results if r.condition == cond and r.is_attacker]
        legit = [r for r in results if r.condition == cond and not r.is_attacker]

        detected = [r for r in attacker if r.detected and r.detect_step is not None]
        mttd_steps = (sum(r.detect_step + 1 for r in detected) / len(detected)
                      if detected else None)
        dwell_steps = (sum(r.dwell_steps for r in attacker) / len(attacker)
                       if attacker else 0.0)
        exfil_prevented = (sum(r.exfiltration_prevented for r in attacker) / len(attacker)
                           if attacker else 0.0)
        fp_rate = (sum(r.false_positive for r in legit) / len(legit)) if legit else 0.0
        overhead = ((sum(r.active_decoy_steps / max(r.steps, 1) for r in attacker)
                     / len(attacker)) * config.cpu_per_active_decoy_pct) if attacker else 0.0

        out[cond] = {
            "attacker_trials": len(attacker),
            "legit_trials": len(legit),
            "detection_rate": (len(detected) / len(attacker)) if attacker else 0.0,
            "mean_time_to_detect_s": (mttd_steps * config.seconds_per_step
                                      if mttd_steps is not None else None),
            "mean_dwell_s": dwell_steps * config.seconds_per_step,
            "exfiltration_prevented_rate": exfil_prevented,
            "false_positive_rate": fp_rate,
            "overhead_pct": round(overhead, 1),
        }
    return out


def _fmt_seconds(s: float | None) -> str:
    if s is None:
        return "n/a"
    m, sec = divmod(int(round(s)), 60)
    return f"{m}m {sec:02d}s" if m else f"{sec}s"


def render_table(agg: dict) -> str:
    """A compact comparison table for the CLI."""
    cols = list(agg.keys())
    rows = [
        ("Detection rate", lambda d: f"{d['detection_rate']*100:.0f}%"),
        ("Mean time-to-detect", lambda d: _fmt_seconds(d["mean_time_to_detect_s"])),
        ("Mean attacker dwell (decoys)", lambda d: _fmt_seconds(d["mean_dwell_s"])),
        ("Exfiltration prevented", lambda d: f"{d['exfiltration_prevented_rate']*100:.0f}%"),
        ("False-positive rate", lambda d: f"{d['false_positive_rate']*100:.1f}%"),
        ("Resource overhead", lambda d: f"{d['overhead_pct']:.1f}%"),
    ]
    w0 = max(len(name) for name, _ in rows) + 2
    head = "Metric".ljust(w0) + "".join(c.ljust(16) for c in cols)
    lines = [head, "-" * len(head)]
    for name, fn in rows:
        lines.append(name.ljust(w0) + "".join(fn(agg[c]).ljust(16) for c in cols))
    return "\n".join(lines)
