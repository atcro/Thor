"""Telemetry ingest / query routes, edge prediction sink and the MQTT subscriber (Agent A).

Routes (mounted by apps/api/main.py):

    POST /ingest                               list[TelemetryRow] | TelemetryRow -> {"inserted": n}
    GET  /assets/{asset_id}/telemetry          ?hours=24&limit=2000 -> list[TelemetryRow]
    GET  /assets/{asset_id}/predictions        ?limit=500 -> list[{ts, failure_probability, model_version, source}]
    POST /predictions                          PredictionIn | list[PredictionIn] -> {"ok": true[, "inserted": n]}

`start_mqtt_subscriber(engine)` runs a background paho client that subscribes to `telemetry/#`
and inserts rows in batches of 50 or every 2 s. It never raises when the broker is down --
it logs and retries every 5 s. `stop_mqtt_subscriber()` flushes and shuts it down.

The engine used by the routes comes from the `get_db_engine` dependency so tests can override
it with an in-memory SQLite engine.
"""

from __future__ import annotations

import json
import logging
import threading
from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
from fastapi import APIRouter, Body, Depends, Query
from paho.mqtt import client as mqtt
from pydantic import BaseModel, Field, ValidationError
from sqlalchemy import select
from sqlalchemy.engine import Engine

from apps.api import db
from apps.api.schemas import TelemetryRow
from apps.api.settings import get_settings

log = logging.getLogger("thor.telemetry")

router = APIRouter(tags=["telemetry"])

MQTT_TELEMETRY_TOPIC = "telemetry/#"


def get_db_engine() -> Engine:
    """FastAPI dependency: the process-wide engine (override in tests)."""
    return db.get_engine()


# --------------------------------------------------------------------------------------
# Request / response shapes local to these routes
# --------------------------------------------------------------------------------------


class PredictionIn(BaseModel):
    """Prediction pushed by the edge container (or the control plane itself)."""

    asset_id: str
    ts: datetime
    model_version: str
    failure_probability: float = Field(ge=0.0, le=1.0)
    source: str = "edge"


class PredictionOut(BaseModel):
    ts: datetime
    failure_probability: float
    model_version: str
    source: str


def _as_utc(ts: datetime) -> datetime:
    """Naive timestamps are interpreted as UTC; aware ones are converted to UTC."""
    if ts.tzinfo is None:
        return ts.replace(tzinfo=UTC)
    return ts.astimezone(UTC)


def _normalize(row: TelemetryRow) -> TelemetryRow:
    return row.model_copy(update={"ts": _as_utc(row.ts)})


def df_to_rows(df: pd.DataFrame) -> list[TelemetryRow]:
    """Telemetry-shaped DataFrame -> TelemetryRow list (NaN ground truth -> None)."""
    out: list[TelemetryRow] = []
    for rec in df.to_dict("records"):
        for key in ("health", "failure_within_h", "regime"):
            v = rec.get(key)
            if v is None or (isinstance(v, float) and np.isnan(v)):
                rec[key] = None
        ts = rec["ts"]
        rec["ts"] = _as_utc(ts.to_pydatetime() if isinstance(ts, pd.Timestamp) else ts)
        out.append(TelemetryRow(**rec))
    return out


# --------------------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------------------


@router.post("/ingest")
def ingest(
    payload: list[TelemetryRow] | TelemetryRow = Body(...),
    engine: Engine = Depends(get_db_engine),
) -> dict[str, int]:
    """Insert one row or a batch of rows. Duplicates on (asset_id, ts) are ignored.

    Input: TelemetryRow or list[TelemetryRow]. Output: {"inserted": rows submitted}.
    """
    rows = payload if isinstance(payload, list) else [payload]
    n = db.insert_telemetry([_normalize(r) for r in rows], engine=engine)
    return {"inserted": n}


@router.get("/assets/{asset_id}/telemetry", response_model=list[TelemetryRow])
def get_asset_telemetry(
    asset_id: str,
    hours: float = Query(24.0, gt=0.0, le=24 * 365),
    limit: int = Query(2000, ge=1, le=100_000),
    engine: Engine = Depends(get_db_engine),
) -> list[TelemetryRow]:
    """Most recent `hours` of telemetry for one asset, oldest first, capped at `limit` rows.

    The window is anchored on the asset's latest stored timestamp (not wall-clock now) so
    replayed historical data is always visible. Output: [] for unknown assets.
    """
    latest = db.read_telemetry(asset_id=asset_id, limit=1, engine=engine)
    if latest.empty:
        return []
    end = latest["ts"].iloc[0].to_pydatetime()
    start = end - timedelta(hours=hours)
    df = db.read_telemetry(asset_id=asset_id, start=start, limit=limit, engine=engine)
    return df_to_rows(df)


@router.get("/assets/{asset_id}/predictions", response_model=list[PredictionOut])
def get_asset_predictions(
    asset_id: str,
    limit: int = Query(500, ge=1, le=10_000),
    engine: Engine = Depends(get_db_engine),
) -> list[PredictionOut]:
    """Most recent `limit` predictions for one asset, oldest first."""
    q = (
        select(db.predictions)
        .where(db.predictions.c.asset_id == asset_id)
        .order_by(db.predictions.c.ts.desc(), db.predictions.c.id.desc())
        .limit(limit)
    )
    with engine.connect() as conn:
        rows = conn.execute(q).mappings().all()
    out = [
        PredictionOut(
            ts=_as_utc(r["ts"]),
            failure_probability=float(r["failure_probability"]),
            model_version=str(r["model_version"]),
            source=str(r["source"]),
        )
        for r in rows
    ]
    out.reverse()
    return out


@router.post("/predictions")
def post_prediction(
    payload: list[PredictionIn] | PredictionIn = Body(...),
    engine: Engine = Depends(get_db_engine),
) -> dict[str, Any]:
    """Store one prediction row or a batch (edge inference sink).

    Input: PredictionIn or list[PredictionIn]. Output: {"ok": true} for a single row,
    {"ok": true, "inserted": n} for a batch. The edge drains its queue into batches so the
    control plane sees a couple of requests per second instead of one per telemetry message.
    """
    preds = payload if isinstance(payload, list) else [payload]
    n = db.insert_predictions(
        [
            {
                "asset_id": p.asset_id,
                "ts": _as_utc(p.ts),
                "model_version": p.model_version,
                "failure_probability": float(p.failure_probability),
                "source": p.source,
            }
            for p in preds
        ],
        engine=engine,
    )
    if isinstance(payload, list):
        return {"ok": True, "inserted": n}
    return {"ok": True}


# --------------------------------------------------------------------------------------
# MQTT subscriber
# --------------------------------------------------------------------------------------


class MqttSubscriber:
    """Background paho-mqtt (v2 callback API) subscriber that batches rows into `telemetry`.

    Inputs: SQLAlchemy engine, broker host/port, batch size (default 50), flush interval
    (default 2 s), reconnect delay (default 5 s). Connection failures are logged and retried;
    nothing here raises into the caller.
    """

    def __init__(
        self,
        engine: Engine,
        host: str,
        port: int,
        batch_size: int = 50,
        flush_interval: float = 2.0,
        retry_delay: float = 5.0,
    ) -> None:
        self.engine = engine
        self.host = host
        self.port = port
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        self.retry_delay = retry_delay
        self._buffer: list[TelemetryRow] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._client: mqtt.Client | None = None
        self._threads: list[threading.Thread] = []
        self.connected = False
        self.rows_inserted = 0
        self.messages_received = 0
        self.last_error: str | None = None
        self.last_message_ts: datetime | None = None

    # -- message handling -----------------------------------------------------------------

    def handle_payload(self, payload: bytes | str) -> bool:
        """Parse one MQTT payload into a TelemetryRow and buffer it (flushing at batch size).

        Output: True if the payload was accepted, False if it was malformed (logged).
        """
        try:
            data: Any = json.loads(payload)
            row = _normalize(TelemetryRow.model_validate(data))
        except (ValueError, ValidationError) as exc:
            self.last_error = f"bad payload: {exc}"
            log.warning("dropping malformed telemetry payload: %s", exc)
            return False
        self.messages_received += 1
        self.last_message_ts = row.ts
        with self._lock:
            self._buffer.append(row)
            full = len(self._buffer) >= self.batch_size
        if full:
            self.flush()
        return True

    def flush(self) -> int:
        """Insert every buffered row. Output: rows inserted (0 if buffer empty or DB error)."""
        with self._lock:
            batch, self._buffer = self._buffer, []
        if not batch:
            return 0
        try:
            n = db.insert_telemetry(batch, engine=self.engine)
        except Exception as exc:  # noqa: BLE001 - never let a DB hiccup kill the subscriber
            self.last_error = f"insert failed: {exc}"
            log.exception("telemetry insert failed (%d rows dropped)", len(batch))
            return 0
        self.rows_inserted += n
        return n

    # -- paho callbacks ---------------------------------------------------------------------

    def _on_connect(
        self, client: mqtt.Client, userdata: Any, flags: Any, reason_code: Any, properties: Any
    ) -> None:
        if getattr(reason_code, "is_failure", False):
            self.connected = False
            self.last_error = f"connect refused: {reason_code}"
            log.warning("MQTT connect refused: %s", reason_code)
            return
        self.connected = True
        client.subscribe(MQTT_TELEMETRY_TOPIC, qos=0)
        log.info("MQTT subscribed to %s on %s:%s", MQTT_TELEMETRY_TOPIC, self.host, self.port)

    def _on_disconnect(
        self, client: mqtt.Client, userdata: Any, flags: Any, reason_code: Any, properties: Any
    ) -> None:
        self.connected = False
        if not self._stop.is_set():
            log.warning("MQTT disconnected (%s); paho will reconnect", reason_code)

    def _on_message(self, client: mqtt.Client, userdata: Any, msg: mqtt.MQTTMessage) -> None:
        self.handle_payload(msg.payload)

    # -- threads ----------------------------------------------------------------------------

    def _run(self) -> None:
        while not self._stop.is_set():
            client = mqtt.Client(
                mqtt.CallbackAPIVersion.VERSION2, client_id=f"thor-api-{id(self) & 0xFFFF:x}"
            )
            client.on_connect = self._on_connect
            client.on_disconnect = self._on_disconnect
            client.on_message = self._on_message
            client.reconnect_delay_set(
                min_delay=int(self.retry_delay), max_delay=int(self.retry_delay)
            )
            self._client = client
            try:
                client.connect(self.host, self.port, keepalive=30)
                client.loop_forever(retry_first_connection=False)
            except Exception as exc:  # noqa: BLE001 - broker down is expected in local dev
                self.connected = False
                self.last_error = f"connect failed: {exc}"
                log.warning(
                    "MQTT broker %s:%s unavailable (%s); retrying in %.0fs",
                    self.host,
                    self.port,
                    exc,
                    self.retry_delay,
                )
            finally:
                self._client = None
            self._stop.wait(self.retry_delay)

    def _flusher(self) -> None:
        while not self._stop.wait(self.flush_interval):
            self.flush()
        self.flush()

    def start(self) -> None:
        """Start the connection + flush threads (daemon threads; idempotent)."""
        if self._threads:
            return
        for name, target in (("thor-mqtt-sub", self._run), ("thor-mqtt-flush", self._flusher)):
            t = threading.Thread(target=target, name=name, daemon=True)
            t.start()
            self._threads.append(t)

    def stop(self, timeout: float = 5.0) -> None:
        """Signal shutdown, disconnect, flush remaining rows and join the threads."""
        self._stop.set()
        client = self._client
        if client is not None:
            try:
                client.disconnect()
            except Exception:  # noqa: BLE001
                pass
        for t in self._threads:
            t.join(timeout=timeout)
        self._threads.clear()
        self.flush()
        self.connected = False

    def status(self) -> dict[str, Any]:
        return {
            "connected": self.connected,
            "broker": f"{self.host}:{self.port}",
            "rows_inserted": self.rows_inserted,
            "messages_received": self.messages_received,
            "last_message_ts": self.last_message_ts.isoformat() if self.last_message_ts else None,
            "last_error": self.last_error,
        }


_subscriber: MqttSubscriber | None = None
_subscriber_lock = threading.Lock()


def start_mqtt_subscriber(engine: Engine, host: str | None = None, port: int | None = None) -> None:
    """Start the process-wide background MQTT subscriber (idempotent, never raises).

    Inputs: engine used for inserts; broker host/port default to settings.mqtt_broker_url.
    """
    global _subscriber
    try:
        with _subscriber_lock:
            if _subscriber is not None:
                return
            settings = get_settings()
            sub = MqttSubscriber(engine, host or settings.mqtt_host, port or settings.mqtt_port)
            sub.start()
            _subscriber = sub
        log.info("MQTT subscriber started for %s:%s", sub.host, sub.port)
    except Exception as exc:  # noqa: BLE001 - startup must survive a missing broker
        log.warning("MQTT subscriber could not start: %s", exc)


def stop_mqtt_subscriber() -> None:
    """Stop the background subscriber if it is running (never raises)."""
    global _subscriber
    with _subscriber_lock:
        sub, _subscriber = _subscriber, None
    if sub is None:
        return
    try:
        sub.stop()
    except Exception as exc:  # noqa: BLE001
        log.warning("MQTT subscriber stop failed: %s", exc)


def mqtt_status() -> dict[str, Any]:
    """Connection status for the /system/health panel: {connected, broker, rows_inserted, ...}."""
    sub = _subscriber
    if sub is None:
        return {
            "connected": False,
            "broker": None,
            "rows_inserted": 0,
            "messages_received": 0,
            "last_message_ts": None,
            "last_error": "not started",
        }
    return sub.status()


def is_mqtt_connected() -> bool:
    """True when the background subscriber currently holds a broker connection."""
    return _subscriber is not None and _subscriber.connected
