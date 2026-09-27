"""Simulated CMMS: executes approved work orders inside the telemetry replay.

Thor stops at the Human Approval Gate. An approved maintain_now / maintain_later Decision
Contract is a handoff to the plant's CMMS, and Thor never touches the motor. The demo has no
plant, so the replay plays that part:

1. It polls `GET {CONTROL_PLANE_URL}/work-orders` for approved contracts.
2. When plant time (the replayed `ts`) reaches the approved window start, the motor goes
   offline for the contract's planned downtime -- no telemetry rows are published, exactly as
   a de-energised motor with its sensors off would look -- and `in_progress` is reported.
3. Afterwards the motor streams at its own healthy baseline again: every sensor value is taken
   from that asset's pre-fault history in the same regime and hour of day (no invented numbers),
   ground-truth `health` is 1.0, `failure_within_h` is unknown, and `completed` is reported.

Rejected contracts and run_to_failure recommendations are not work orders, so nothing happens.
Reports go to `POST /work-orders/{contract_id}/events` (insert-only on the API side); a failed
report is logged and retried on the next transition, never fatal to the stream.

This module runs inside the replay container without `apps/`; it depends only on pandas,
httpx and the TelemetryRow model from data.simulator.generate.
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx
import pandas as pd

from data.simulator.generate import SENSOR_COLUMNS, TelemetryRow

log = logging.getLogger("thor.replay.work_orders")

DEFAULT_POLL_S = float(os.getenv("REPLAY_WORK_ORDER_POLL_S", "3"))
DEFAULT_SOURCE = "cmms-sim"
HEALTHY_MIN = 0.999
_RANK = {"scheduled": 0, "in_progress": 1, "completed": 2}


def _utc(value: Any) -> datetime | None:
    """ISO string / Timestamp / datetime -> tz-aware UTC datetime (None stays None)."""
    if value is None or value == "":
        return None
    if isinstance(value, pd.Timestamp):
        dt = value.to_pydatetime()
    elif isinstance(value, datetime):
        dt = value
    else:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=UTC)


@dataclass
class WorkOrder:
    """The replay's view of one approved contract (mirrors apps.api.schemas.WorkOrder)."""

    contract_id: str
    asset_id: str
    window_start: datetime | None
    planned_downtime_h: float
    status: str = "scheduled"
    started_ts: datetime | None = None
    completed_ts: datetime | None = None
    approved_at: datetime | None = None

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> WorkOrder:
        return cls(
            contract_id=str(d["contract_id"]),
            asset_id=str(d["asset_id"]),
            window_start=_utc(d.get("window_start")),
            planned_downtime_h=float(d.get("planned_downtime_h") or 3.0),
            status=str(d.get("status") or "scheduled"),
            started_ts=_utc(d.get("started_ts")),
            completed_ts=_utc(d.get("completed_ts")),
            approved_at=_utc(d.get("approved_at")),
        )

    @property
    def rank(self) -> int:
        return _RANK.get(self.status, 0)


class WorkOrderClient:
    """Thin HTTP client for the control plane's work-order routes. Never raises to the stream."""

    def __init__(self, base_url: str, timeout: float = 2.0, http: httpx.Client | None = None):
        self.base_url = base_url.rstrip("/")
        self._http = http or httpx.Client(timeout=timeout)

    def fetch(self) -> list[WorkOrder] | None:
        """All approved work orders, or None when the control plane is unreachable."""
        try:
            r = self._http.get(self.base_url + "/work-orders")
            r.raise_for_status()
            return [WorkOrder.from_json(d) for d in r.json()]
        except (httpx.HTTPError, OSError, ValueError) as exc:
            log.warning("work-order poll failed: %s", exc)
            return None

    def report(self, contract_id: str, status: str, plant_ts: datetime, note: str = "") -> bool:
        """POST one progress event. 409 (already recorded, e.g. after a restart) counts as done."""
        payload = {
            "status": status,
            "plant_ts": plant_ts.isoformat(),
            "source": DEFAULT_SOURCE,
            "note": note,
        }
        try:
            r = self._http.post(f"{self.base_url}/work-orders/{contract_id}/events", json=payload)
            if r.status_code == 409:
                return True
            r.raise_for_status()
            return True
        except (httpx.HTTPError, OSError) as exc:
            log.warning("work-order report %s/%s failed: %s", contract_id, status, exc)
            return False


class HealthyBaseline:
    """Donor pool of one asset's healthy rows, keyed by (regime, hour of day).

    `next_row` walks each pool sequentially so consecutive replayed rows come from consecutive
    healthy history (the slow in-regime wander stays realistic) and cycles when it runs out.
    """

    def __init__(self, asset_df: pd.DataFrame):
        df = asset_df
        if "health" in df.columns:
            health = pd.to_numeric(df["health"], errors="coerce")
            df = df[health.isna() | (health >= HEALTHY_MIN)]
        df = df.sort_values("ts", kind="mergesort")
        self._pools: dict[tuple[str, int], list[dict[str, float]]] = {}
        self._by_regime: dict[str, list[dict[str, float]]] = {}
        self._all: list[dict[str, float]] = []
        self._cursor: dict[Any, int] = {}
        for rec in df.to_dict("records"):
            ts = _utc(rec["ts"])
            regime = str(rec.get("regime") or "")
            values = {c: float(rec[c]) for c in SENSOR_COLUMNS}
            self._pools.setdefault((regime, ts.hour if ts else 0), []).append(values)
            self._by_regime.setdefault(regime, []).append(values)
            self._all.append(values)

    def __len__(self) -> int:
        return len(self._all)

    def next_row(self, regime: str | None, ts: datetime) -> dict[str, float] | None:
        """Sensor values of the next healthy donor row for this regime/hour (None if no history)."""
        regime = str(regime or "")
        candidates: list[tuple[Any, list[dict[str, float]] | None]] = [
            ((regime, ts.hour), self._pools.get((regime, ts.hour))),
            (regime, self._by_regime.get(regime)),
            ("*", self._all),
        ]
        for key, pool in candidates:
            if pool:
                i = self._cursor.get(key, 0)
                self._cursor[key] = i + 1
                return pool[i % len(pool)]
        return None


@dataclass
class RepairSimulator:
    """Applies approved work orders to the replayed row stream (see module docstring).

    Inputs: the full telemetry DataFrame (donor pool for healthy baselines), an optional
    WorkOrderClient (None = never polls; orders can be injected with `load()` in tests), the poll
    interval in wall-clock seconds and an injectable clock. Output: `apply()` wraps a row
    iterator and yields the rows the plant would actually produce.
    """

    df: pd.DataFrame
    client: WorkOrderClient | None = None
    poll_interval_s: float = DEFAULT_POLL_S
    clock: Callable[[], float] = time.monotonic
    events: list[tuple[str, str, datetime]] = field(default_factory=list)
    _orders: dict[str, WorkOrder] = field(default_factory=dict)
    _by_asset: dict[str, WorkOrder] = field(default_factory=dict)
    _baselines: dict[str, HealthyBaseline] = field(default_factory=dict)
    _last_poll: float | None = None

    # -- orders ---------------------------------------------------------------------------

    def load(self, orders: Iterable[WorkOrder]) -> None:
        """Merge orders from the control plane; local progress is never rolled back."""
        for wo in orders:
            cur = self._orders.get(wo.contract_id)
            if cur is None or wo.rank >= cur.rank:
                self._orders[wo.contract_id] = wo
        self._by_asset = {}
        floor = datetime.min.replace(tzinfo=UTC)
        for wo in sorted(self._orders.values(), key=lambda w: w.approved_at or floor, reverse=True):
            self._by_asset.setdefault(wo.asset_id, wo)

    def refresh(self) -> None:
        if self.client is None:
            return
        fetched = self.client.fetch()
        if fetched is not None:
            self.load(fetched)

    def _maybe_poll(self) -> None:
        if self.client is None:
            return
        now = self.clock()
        if self._last_poll is None or now - self._last_poll >= self.poll_interval_s:
            self._last_poll = now
            self.refresh()

    def _report(self, wo: WorkOrder, status: str, ts: datetime) -> None:
        self.events.append((wo.contract_id, status, ts))
        log.info(
            "work order %s (%s) -> %s at plant time %s", wo.contract_id, wo.asset_id, status, ts
        )
        if self.client is not None:
            self.client.report(wo.contract_id, status, ts)

    # -- rows -----------------------------------------------------------------------------

    def _baseline(self, asset_id: str) -> HealthyBaseline:
        b = self._baselines.get(asset_id)
        if b is None:
            b = HealthyBaseline(self.df[self.df["asset_id"] == asset_id])
            if len(b) == 0:
                log.warning(
                    "%s has no healthy history to sample; only ground truth is reset", asset_id
                )
            self._baselines[asset_id] = b
        return b

    def _healthy(self, row: TelemetryRow) -> TelemetryRow:
        donor = self._baseline(row.asset_id).next_row(row.regime, row.ts)
        update: dict[str, Any] = {"health": 1.0, "failure_within_h": None}
        if donor:
            update.update(donor)
        return row.model_copy(update=update)

    def transform(self, row: TelemetryRow) -> TelemetryRow | None:
        """One row in, the row the plant would produce out (None = motor offline)."""
        wo = self._by_asset.get(row.asset_id)
        if wo is None:
            return row
        ts = row.ts
        if wo.status == "scheduled":
            if wo.window_start is not None and ts < wo.window_start:
                return row
            wo.status, wo.started_ts = "in_progress", ts
            self._report(wo, "in_progress", ts)
        if wo.status == "in_progress":
            started = wo.started_ts or ts
            if ts < started + timedelta(hours=wo.planned_downtime_h):
                return None
            wo.status, wo.completed_ts = "completed", ts
            self._report(wo, "completed", ts)
        # completed: history before the outage replays unchanged (the replay loops), the
        # outage stays silent, everything after streams healthy.
        started = wo.started_ts or wo.completed_ts or ts
        if ts < started:
            return row
        if wo.completed_ts is not None and ts < wo.completed_ts:
            return None
        return self._healthy(row)

    def apply(self, rows: Iterable[TelemetryRow]) -> Iterator[TelemetryRow]:
        for row in rows:
            self._maybe_poll()
            out = self.transform(row)
            if out is not None:
                yield out
