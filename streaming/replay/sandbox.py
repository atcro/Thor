"""Replay-side sandbox: apply the control plane's sandbox overrides to the recorded stream.

The 3-D twin's sandbox (apps/api/sandbox.py) only records what the presenter asked for. This
module polls `GET {CONTROL_PLANE_URL}/sandbox` and rewrites the recorded rows of the motors
that have an active override, so sandbox readings take the real path -- replay -> MQTT -> edge
-> control plane -- and are scored like plant data. Every rewritten row is tagged
`sandbox=True`; untouched rows keep `sandbox=False`.

What an override does to a row (observable sensors only; no ground truth is read or written):

* `rpm` / `load_pct` setpoints replace the recorded values. Load-dependent sensors follow the
  simulator's own healthy relations (`data/simulator/generate.py`): current is
  `rated_a * (0.10 + 0.90 * load / 100)`, motor and bearing temperature shift by
  `0.46 * dload` / `0.50 * dload` through the same first-order lags, and vibration RMS is
  scaled from the recorded operating state's healthy level to the target state's.
* `bearing_temp_c` replaces the bearing temperature (lagged like a real housing).
* `fault` adds the drive-end bearing signature (`bearing_fault_delta`) at severity
  `min(1, ((t - fault_since) / ramp_h) ** 2)` in plant time, held at 1.0 afterwards.

Also the replay's own `ReplayGate`: the stream is held until the control plane reports a
production model (`GET /replay` -> `released`) and while it is paused.
"""

from __future__ import annotations

import logging
import math
import os
import time
from collections.abc import Callable, Iterable, Iterator
from datetime import UTC, datetime
from typing import Any

import httpx
import numpy as np

from data.simulator.generate import (
    REGIME_LOAD,
    REGIME_RPM,
    REGIME_VIB,
    TelemetryRow,
    bearing_fault_delta,
    rated_current_a,
)

log = logging.getLogger("thor.replay.sandbox")

DEFAULT_POLL_S = float(os.getenv("REPLAY_SANDBOX_POLL_S", "1.0"))
GATE_POLL_S = float(os.getenv("REPLAY_GATE_POLL_S", "1.0"))
#: First-order lag of the housing temperatures (minutes), as in the simulator.
MOTOR_TAU_MIN = 35.0
BEARING_TAU_MIN = 25.0


def _utc(value: Any) -> datetime | None:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        dt = value
    else:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=UTC)


def nearest_state(rpm: float, load: float) -> int:
    """Index of the simulator state (idle / running / heavy load) nearest to (rpm, load)."""
    d = ((REGIME_RPM - rpm) / 900.0) ** 2 + ((REGIME_LOAD - load) / 75.0) ** 2
    return int(np.argmin(d))


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


class SandboxClient:
    """GET /sandbox, never raising into the stream."""

    def __init__(self, base_url: str, http: httpx.Client | None = None):
        self.base_url = base_url.rstrip("/")
        self._http = http or httpx.Client(timeout=3.0)

    def fetch(self) -> list[dict[str, Any]] | None:
        try:
            r = self._http.get(self.base_url + "/sandbox")
            r.raise_for_status()
            return list(r.json().get("overrides") or [])
        except (httpx.HTTPError, OSError, ValueError) as exc:
            log.warning("sandbox poll failed: %s", exc)
            return None


class SandboxApplier:
    """Rewrites rows under an active override (see module docstring).

    Inputs: rated kW per asset id (for the current relation), an optional SandboxClient
    (None = never polls; overrides can be loaded with `load()` in tests), the poll interval,
    an injectable clock, the sampling step in minutes and a seeded Generator for the fault
    signature's noise.
    """

    def __init__(
        self,
        rated_kw: dict[str, float],
        client: SandboxClient | None = None,
        poll_interval_s: float = DEFAULT_POLL_S,
        clock: Callable[[], float] = time.monotonic,
        step_min: int = 10,
        seed: int = 7,
    ) -> None:
        self.rated_kw = rated_kw
        self.client = client
        self.poll_interval_s = poll_interval_s
        self.clock = clock
        self.step_min = step_min
        self.rng = np.random.default_rng(seed)
        self.overrides: dict[str, dict[str, Any]] = {}
        self._last_poll: float | None = None
        # per-asset lagged temperature deltas (first-order, like the simulator's housings)
        self._temp_lag: dict[str, dict[str, float]] = {}
        self.n_rewritten = 0

    # -- overrides ------------------------------------------------------------------------

    def load(self, overrides: Iterable[dict[str, Any]]) -> None:
        new = {str(o["asset_id"]): dict(o) for o in overrides}
        gone = set(self.overrides) - set(new)
        for aid in gone:
            self._temp_lag.pop(aid, None)
        if set(new) != set(self.overrides):
            log.info("sandbox overrides on %s", sorted(new) or "none")
        self.overrides = new

    def _maybe_poll(self) -> None:
        if self.client is None:
            return
        now = self.clock()
        if self._last_poll is None or now - self._last_poll >= self.poll_interval_s:
            self._last_poll = now
            fetched = self.client.fetch()
            if fetched is not None:
                self.load(fetched)

    # -- physics --------------------------------------------------------------------------

    def _alpha(self, tau_min: float) -> float:
        return float(1.0 - math.exp(-self.step_min / tau_min))

    def _lag(self, asset_id: str, key: str, target: float, tau_min: float) -> float:
        lag = self._temp_lag.setdefault(asset_id, {})
        prev = lag.get(key, 0.0)
        a = self._alpha(tau_min)
        val = (1.0 - a) * prev + a * target
        lag[key] = val
        return val

    def transform(self, row: TelemetryRow) -> TelemetryRow:
        """One recorded row in; the row the plant would produce under its override out."""
        o = self.overrides.get(row.asset_id)
        if o is None:
            return row
        rpm, load = float(row.rpm), float(row.load_pct)
        vib, kurt, crest = float(row.vibration_rms), float(row.vibration_kurtosis), float(row.vibration_crest)
        bearing, motor, current = float(row.bearing_temp_c), float(row.motor_temp_c), float(row.current_a)
        rated_a = rated_current_a(float(self.rated_kw.get(row.asset_id, 75.0)))
        rec_state = nearest_state(rpm, load)

        # sliders: rpm / load setpoints and their load-dependent consequences
        if o.get("rpm") is not None:
            rpm = float(o["rpm"])
        if o.get("load_pct") is not None:
            new_load = float(o["load_pct"])
            dload = new_load - load
            load = new_load
            current = rated_a * (0.10 + 0.90 * load / 100.0) + float(self.rng.normal(0, 0.012 * rated_a))
            motor += self._lag(row.asset_id, "motor", 0.46 * dload, MOTOR_TAU_MIN)
            bearing += self._lag(row.asset_id, "bearing", 0.50 * dload, BEARING_TAU_MIN)
        if o.get("rpm") is not None or o.get("load_pct") is not None:
            tgt_state = nearest_state(rpm, load)
            if tgt_state != rec_state:
                vib *= float(REGIME_VIB[tgt_state] / REGIME_VIB[rec_state])
        if o.get("bearing_temp_c") is not None:
            bearing = bearing + self._lag(
                row.asset_id, "bearing_set", float(o["bearing_temp_c"]) - bearing, BEARING_TAU_MIN
            )

        # fault: the simulator's drive-end bearing signature at the ramped severity
        if o.get("fault"):
            since = _utc(o.get("fault_since")) or row.ts
            ramp_h = float(o.get("fault_ramp_h") or 2.5)
            hours = max(0.0, (row.ts - since).total_seconds() / 3600.0)
            severity = min(1.0, (hours / max(ramp_h, 1e-6)) ** 2)
            d = bearing_fault_delta(severity, load, rated_a, self.rng, amp=1.0)
            vib += d["vibration_rms"]
            kurt += d["vibration_kurtosis"]
            crest += d["vibration_crest"]
            current += d["current_a"]
            bearing += self._lag(row.asset_id, "fault", d["bearing_target_c"], BEARING_TAU_MIN)

        self.n_rewritten += 1
        return row.model_copy(
            update={
                "rpm": round(max(0.0, rpm), 2),
                "load_pct": round(min(100.0, max(0.0, load)), 3),
                "vibration_rms": round(max(0.05, vib), 4),
                "vibration_kurtosis": round(max(1.5, kurt), 4),
                "vibration_crest": round(max(1.4, crest), 4),
                "bearing_temp_c": round(bearing, 3),
                "motor_temp_c": round(motor, 3),
                "current_a": round(max(0.0, current), 3),
                "sandbox": True,
            }
        )

    def apply(self, rows: Iterable[TelemetryRow]) -> Iterator[TelemetryRow]:
        for row in rows:
            self._maybe_poll()
            yield self.transform(row)
