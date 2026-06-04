"""Telemetry collection and threat-intelligence export.

The collector keeps the rolling event stream the profiler reads, and harvests
indicators of compromise from decoy interactions — the "high-fidelity IOCs that
would have been unavailable under baseline controls" the paper highlights.
"""
from __future__ import annotations

import json
from collections import deque

from addf.events import IOC, Actor, Event


class TelemetryCollector:
    def __init__(self, window: int) -> None:
        self.window = window
        self.events: list[Event] = []
        self._recent: deque[Event] = deque(maxlen=window)
        self.iocs: list[IOC] = []

    def record(self, event: Event) -> None:
        self.events.append(event)
        self._recent.append(event)
        # IOCs are only trustworthy when harvested from a decoy: a legitimate user
        # never touches one, so anything seen there is attacker-attributable.
        if event.on_decoy:
            self._harvest(event)

    def _harvest(self, event: Event) -> None:
        self.iocs.append(
            IOC(kind="ip", value=event.source_ip, first_seen=event.t,
                technique=event.technique,
                context={"phase": event.phase.name, "endpoint": event.endpoint})
        )
        if event.technique:
            self.iocs.append(
                IOC(kind="technique", value=event.technique, first_seen=event.t,
                    technique=event.technique, context={"phase": event.phase.name})
            )

    def window_events(self) -> list[Event]:
        return list(self._recent)

    # -- export --------------------------------------------------------------
    def ioc_report(self) -> dict:
        seen: set[tuple[str, str]] = set()
        unique: list[dict] = []
        for ioc in self.iocs:
            key = (ioc.kind, ioc.value)
            if key in seen:
                continue
            seen.add(key)
            unique.append({
                "kind": ioc.kind, "value": ioc.value,
                "first_seen": ioc.first_seen, "technique": ioc.technique,
                "context": ioc.context,
            })
        attacker_events = sum(1 for e in self.events if e.actor == Actor.ATTACKER)
        return {
            "total_events": len(self.events),
            "attacker_events": attacker_events,
            "decoy_interactions": sum(1 for e in self.events if e.on_decoy),
            "indicators": unique,
        }

    def write_ioc_report(self, path: str) -> None:  # pragma: no cover
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.ioc_report(), fh, indent=2)
