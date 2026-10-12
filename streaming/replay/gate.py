"""The replay's gate: the stream is held until the control plane reports a production model
(`GET /replay` -> `released`) and while it is paused."""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Callable, Iterable, Iterator
from typing import Any

import httpx

from data.simulator.generate import TelemetryRow

log = logging.getLogger("thor.replay.gate")

GATE_POLL_S = float(os.getenv("REPLAY_GATE_POLL_S", "1.0"))


class ReplayGate:
    """Holds the stream until `GET /replay` says released and not paused."""

    def __init__(self, base_url: str, http: httpx.Client | None = None, poll_s: float = GATE_POLL_S):
        self.base_url = base_url.rstrip("/")
        self._http = http or httpx.Client(timeout=3.0)
        self.poll_s = poll_s
        self._last: dict[str, Any] | None = None
        self._last_at: float | None = None
        self._announced: str | None = None

    def fetch(self) -> dict[str, Any] | None:
        try:
            r = self._http.get(self.base_url + "/replay")
            r.raise_for_status()
            return r.json()
        except (httpx.HTTPError, OSError, ValueError) as exc:
            log.warning("replay gate poll failed: %s", exc)
            return None

    def running(self, now: float | None = None) -> bool:
        """True when the stream may advance (cached for `poll_s`). Unreachable -> keep the
        last answer, defaulting to held before the first successful poll."""
        now = time.monotonic() if now is None else now
        if self._last_at is None or now - self._last_at >= self.poll_s:
            fetched = self.fetch()
            self._last_at = now
            if fetched is not None:
                self._last = fetched
        if self._last is None:
            return False
        released = bool(self._last.get("released"))
        paused = bool(self._last.get("paused"))
        state = "held: no production model yet" if not released else "paused" if paused else "streaming"
        if state != self._announced:
            log.info("replay %s", state)
            self._announced = state
        return released and not paused

    def wait(self, sleep: Callable[[float], None] = time.sleep) -> None:
        while not self.running():
            sleep(self.poll_s)

    def apply(self, rows: Iterable[TelemetryRow]) -> Iterator[TelemetryRow]:
        for row in rows:
            self.wait()
            yield row
