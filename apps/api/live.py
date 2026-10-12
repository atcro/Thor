"""In-process live bus for the 3-D Digital Twin (Phase A, CLAUDE.md section 15).

Every telemetry message and every prediction that reaches the control plane is recorded here
with its wall-clock receipt time, so the twin can show *measured* rates, freshness and the
motor -> edge -> control plane path latency instead of inventing motion. Nothing here is
stored: it is a ring buffer in the API process, and it restarts empty with the process.

Writers (one call per row, after validation):

    record_ingest(asset_id, ts)                      MQTT subscriber and POST /ingest
    record_prediction(asset_id, ts, p, inference_ms) POST /predictions (the edge sink)

Readers (all thread-safe; the MQTT subscriber writes from its own thread):

    data_state(asset_id)       -> DataState   live / stale / none against STALE_AFTER_S
    asset_live_stats()         -> {asset_id: (last_received_at, msgs_last_60s)}
    snapshot_counters()        -> LiveCounters (rates over WINDOW_S, p50/p95 latencies)
    events_since(seq, n)       -> (events, next_seq, dropped) for the pulse stream

Path latency = prediction receipt time - telemetry receipt time for the same (asset_id, ts),
both measured on this one clock. Edge inference time is whatever the edge reported per
prediction (`inference_ms`). Rates use a WINDOW_S wall-clock window, latencies the last
LATENCY_WINDOW_S. Ground truth never passes through here: only ids, timestamps and the
calibrated probability the edge already published.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from collections import OrderedDict, deque
from datetime import UTC, datetime
from typing import Any

from apps.api.schemas import DataState, LiveCounters, TwinEvent

log = logging.getLogger("thor.live")

#: A motor whose last message is older than this (wall clock, seconds) is `stale`.
STALE_AFTER_S: float = float(os.environ.get("TWIN_STALE_AFTER_S", "30"))
#: Rate window (seconds of wall clock) for msgs_per_s / preds_per_s.
WINDOW_S: int = 10
#: Window for the latency percentiles and the per-asset message count.
LATENCY_WINDOW_S: int = 60
#: Ring-buffer size for events and for the ingest receipt table (path latency).
RING_MAXLEN: int = 20_000
RECEIPT_MAXLEN: int = 20_000
#: The edge /health probe and the MQTT flag are re-checked at most this often.
PROBE_TTL_S: float = 5.0


def as_utc(ts: Any) -> datetime:
    """Normalise a timestamp (aware/naive datetime, ISO string incl. 'Z', pandas Timestamp) to
    an aware UTC datetime so the edge's string ts and the subscriber's datetime key alike."""
    if isinstance(ts, str):
        text = ts.strip()
        if text.endswith("Z") or text.endswith("z"):
            text = text[:-1] + "+00:00"
        ts = datetime.fromisoformat(text)
    elif not isinstance(ts, datetime) and hasattr(ts, "to_pydatetime"):
        ts = ts.to_pydatetime()
    if not isinstance(ts, datetime):
        raise TypeError(f"unsupported timestamp {ts!r}")
    if ts.tzinfo is None:
        return ts.replace(tzinfo=UTC)
    return ts.astimezone(UTC)


def _percentile(values: list[float], q: float) -> float | None:
    """Nearest-rank percentile of `values` (q in 0..1); None for an empty list."""
    if not values:
        return None
    ordered = sorted(values)
    idx = int(round(q * (len(ordered) - 1)))
    return float(ordered[max(0, min(len(ordered) - 1, idx))])


def _prune(times: deque[float], horizon: float) -> None:
    """Drop entries older than `horizon` (monotonic seconds) from the left of `times`."""
    while times and times[0] < horizon:
        times.popleft()


class _AssetStats:
    __slots__ = ("last_received_at", "times")

    def __init__(self) -> None:
        self.last_received_at: float = 0.0
        self.times: deque[float] = deque()


class LiveBus:
    """Thread-safe recorder of ingest / prediction arrivals (see module docstring).

    Inputs: `maxlen` (event ring size) and `receipt_maxlen` (ingest receipts kept for path
    latency). All public methods take and return plain values or Pydantic models.
    """

    def __init__(self, maxlen: int = RING_MAXLEN, receipt_maxlen: int = RECEIPT_MAXLEN) -> None:
        self._lock = threading.Lock()
        self._maxlen = maxlen
        self._receipt_maxlen = receipt_maxlen
        self._probe: tuple[float, bool, bool] | None = None
        self._clear()

    def _clear(self) -> None:
        self._seq = 0
        self._events: deque[TwinEvent] = deque(maxlen=self._maxlen)
        self._ingest_times: deque[float] = deque()
        self._pred_times: deque[float] = deque()
        self._inference: deque[tuple[float, float]] = deque()
        self._latency: deque[tuple[float, float]] = deque()
        self._receipts: OrderedDict[tuple[str, datetime], float] = OrderedDict()
        self._assets: dict[str, _AssetStats] = {}
        self._messages_total = 0
        self._predictions_total = 0

    def reset(self) -> None:
        """Forget everything (tests). Output: none."""
        with self._lock:
            self._clear()
            self._probe = None

    # -- writers ------------------------------------------------------------------------------

    def record_ingest(self, asset_id: str, ts: Any, received_at: float | None = None) -> TwinEvent:
        """Record one telemetry message for `asset_id` with plant time `ts`.

        Inputs: asset id, the row's ts (any form `as_utc` accepts), optional wall-clock receipt
        time (epoch seconds; default now). Output: the TwinEvent appended to the ring.
        """
        now = time.time() if received_at is None else float(received_at)
        key_ts = as_utc(ts)
        with self._lock:
            self._seq += 1
            ev = TwinEvent(
                seq=self._seq,
                kind="ingest",
                asset_id=asset_id,
                ts=key_ts,
                received_at=datetime.fromtimestamp(now, UTC),
            )
            self._events.append(ev)
            self._messages_total += 1
            self._ingest_times.append(now)
            stats = self._assets.get(asset_id)
            if stats is None:
                stats = self._assets[asset_id] = _AssetStats()
            stats.last_received_at = max(stats.last_received_at, now)
            stats.times.append(now)
            _prune(stats.times, now - LATENCY_WINDOW_S)
            self._receipts[(asset_id, key_ts)] = now
            while len(self._receipts) > self._receipt_maxlen:
                self._receipts.popitem(last=False)
        return ev

    def record_prediction(
        self,
        asset_id: str,
        ts: Any,
        failure_probability: float | None = None,
        inference_ms: float | None = None,
        received_at: float | None = None,
        source: str = "edge",
    ) -> TwinEvent:
        """Record one prediction arrival for `asset_id` scored at plant time `ts`.

        Inputs: asset id, the scored row's ts, the published probability, the edge-reported
        inference time in ms (optional), an optional receipt time and the `source` ("edge" or
        "control_plane"). Output: the TwinEvent, with `path_latency_ms` filled when the
        prediction came from the edge and the matching ingest receipt is still known. A
        control-plane score never contributes a path-latency sample: it did not travel
        motor -> edge -> control plane.
        """
        now = time.time() if received_at is None else float(received_at)
        key_ts = as_utc(ts)
        src = "control_plane" if source == "control_plane" else "edge"
        with self._lock:
            self._seq += 1
            latency_ms: float | None = None
            ingest_at = self._receipts.get((asset_id, key_ts)) if src == "edge" else None
            if ingest_at is not None:
                latency_ms = max(0.0, (now - ingest_at) * 1000.0)
                self._latency.append((now, latency_ms))
            if inference_ms is not None:
                self._inference.append((now, float(inference_ms)))
            ev = TwinEvent(
                seq=self._seq,
                kind="prediction",
                asset_id=asset_id,
                ts=key_ts,
                received_at=datetime.fromtimestamp(now, UTC),
                source=src,
                failure_probability=failure_probability,
                inference_ms=inference_ms,
                path_latency_ms=round(latency_ms, 3) if latency_ms is not None else None,
            )
            self._events.append(ev)
            self._predictions_total += 1
            self._pred_times.append(now)
        return ev

    # -- readers ------------------------------------------------------------------------------

    def data_state(self, asset_id: str, now: float | None = None) -> DataState:
        """live / stale / none for one motor against STALE_AFTER_S (wall clock)."""
        now = time.time() if now is None else now
        with self._lock:
            stats = self._assets.get(asset_id)
            last = stats.last_received_at if stats is not None else None
        if last is None:
            return DataState.none
        return DataState.live if now - last <= STALE_AFTER_S else DataState.stale

    def asset_live_stats(self, now: float | None = None) -> dict[str, tuple[datetime, int]]:
        """{asset_id: (last_received_at, messages in the last LATENCY_WINDOW_S)} in one lock."""
        now = time.time() if now is None else now
        horizon = now - LATENCY_WINDOW_S
        out: dict[str, tuple[datetime, int]] = {}
        with self._lock:
            for aid, stats in self._assets.items():
                _prune(stats.times, horizon)
                # receipts are appended in arrival order; a count (not len) tolerates an
                # injected out-of-order receipt time (tests) without a sort
                n = sum(1 for t in stats.times if t >= horizon)
                out[aid] = (datetime.fromtimestamp(stats.last_received_at, UTC), n)
        return out

    def _probe_external(self, now: float) -> tuple[bool, bool]:
        """(mqtt_connected, edge_up), re-probed at most every PROBE_TTL_S. Never raises."""
        cached = self._probe
        if cached is not None and now - cached[0] < PROBE_TTL_S:
            return cached[1], cached[2]
        mqtt_ok = False
        edge_ok = False
        try:
            from apps.api.routes.telemetry import is_mqtt_connected

            mqtt_ok = bool(is_mqtt_connected())
        except Exception as exc:  # noqa: BLE001 - a counters panel must never fail the twin
            log.debug("mqtt probe failed: %s", exc)
        try:
            from apps.api.routes.fleet import _http_ok

            edge_url = os.environ.get("EDGE_URL", "http://localhost:8001")
            edge_ok = _http_ok(edge_url.rstrip("/") + "/health", timeout=0.5)
        except Exception as exc:  # noqa: BLE001
            log.debug("edge probe failed: %s", exc)
        self._probe = (now, mqtt_ok, edge_ok)
        return mqtt_ok, edge_ok

    def snapshot_counters(self, now: float | None = None) -> LiveCounters:
        """Measured counters for the twin (rates, totals, latency percentiles, probes)."""
        now = time.time() if now is None else now
        rate_horizon = now - WINDOW_S
        lat_horizon = now - LATENCY_WINDOW_S
        with self._lock:
            _prune(self._ingest_times, rate_horizon)
            _prune(self._pred_times, rate_horizon)
            while self._inference and self._inference[0][0] < lat_horizon:
                self._inference.popleft()
            while self._latency and self._latency[0][0] < lat_horizon:
                self._latency.popleft()
            n_msgs = len(self._ingest_times)
            n_preds = len(self._pred_times)
            inference = [v for _, v in self._inference]
            latency = [v for _, v in self._latency]
            messages_total = self._messages_total
            predictions_total = self._predictions_total
        mqtt_ok, edge_ok = self._probe_external(now)
        return LiveCounters(
            server_time=datetime.fromtimestamp(now, UTC),
            window_s=WINDOW_S,
            msgs_per_s=round(n_msgs / WINDOW_S, 3),
            preds_per_s=round(n_preds / WINDOW_S, 3),
            messages_total=messages_total,
            predictions_total=predictions_total,
            edge_inference_ms_p50=_percentile(inference, 0.5),
            edge_inference_ms_p95=_percentile(inference, 0.95),
            path_latency_ms_p50=_percentile(latency, 0.5),
            path_latency_ms_p95=_percentile(latency, 0.95),
            n_latency_samples=len(latency),
            stale_after_s=STALE_AFTER_S,
            mqtt_connected=mqtt_ok,
            edge_up=edge_ok,
        )

    def events_since(self, since: int, max_events: int = 500) -> tuple[list[TwinEvent], int, int]:
        """Events with seq > `since`, oldest first.

        Output: (events, next_seq, dropped). `next_seq` is the newest seq known (pass it back
        as `since`). `dropped` counts events that left the ring before being read. When more
        than `max_events` are pending they are aggregated per (asset_id, kind) into one event
        carrying the newest values and `count` = rows merged, so pulses stay smooth and no
        arrival is invented or silently lost.
        """
        with self._lock:
            newest = self._seq
            if since < 0 or since > newest:
                since = 0
            if not self._events:
                return [], newest, 0
            oldest = self._events[0].seq
            dropped = max(0, oldest - since - 1)
            start = max(0, since - oldest + 1)
            pending = list(self._events)[start:]
        if len(pending) <= max_events:
            return pending, newest, dropped
        groups: dict[tuple[str, str], tuple[TwinEvent, int]] = {}
        for ev in pending:
            key = (ev.asset_id, ev.kind)
            prev = groups.get(key)
            groups[key] = (ev, prev[1] + 1 if prev else 1)
        merged = [
            ev.model_copy(update={"count": n})
            for ev, n in sorted(groups.values(), key=lambda g: g[0].seq)
        ]
        return merged, newest, dropped

    def seq(self) -> int:
        """Newest seq handed out (0 when nothing was recorded)."""
        with self._lock:
            return self._seq


#: The process-wide bus. Routes and the MQTT subscriber write into this one instance.
bus = LiveBus()


def record_ingest(asset_id: str, ts: Any, received_at: float | None = None) -> TwinEvent:
    """Module-level shortcut for `bus.record_ingest` (never raises on a bad ts: logs it)."""
    try:
        return bus.record_ingest(asset_id, ts, received_at)
    except Exception as exc:  # noqa: BLE001 - the twin must never break ingestion
        log.debug("record_ingest(%s) skipped: %s", asset_id, exc)
        return TwinEvent(
            seq=0, kind="ingest", asset_id=asset_id, ts=datetime.now(UTC), received_at=datetime.now(UTC)
        )


def record_prediction(
    asset_id: str,
    ts: Any,
    failure_probability: float | None = None,
    inference_ms: float | None = None,
    received_at: float | None = None,
    source: str = "edge",
) -> TwinEvent:
    """Module-level shortcut for `bus.record_prediction` (never raises; logs a bad ts)."""
    try:
        return bus.record_prediction(
            asset_id, ts, failure_probability, inference_ms, received_at, source
        )
    except Exception as exc:  # noqa: BLE001
        log.debug("record_prediction(%s) skipped: %s", asset_id, exc)
        return TwinEvent(
            seq=0,
            kind="prediction",
            asset_id=asset_id,
            ts=datetime.now(UTC),
            received_at=datetime.now(UTC),
        )


def reset() -> None:
    """Forget every recorded event (tests)."""
    bus.reset()


__all__ = [
    "LATENCY_WINDOW_S",
    "RING_MAXLEN",
    "STALE_AFTER_S",
    "WINDOW_S",
    "LiveBus",
    "as_utc",
    "bus",
    "record_ingest",
    "record_prediction",
    "reset",
]
