"""Historical telemetry replay -- time-accelerated MQTT publisher with HTTP fallback.

Reads `data/simulator/out/telemetry.parquet` (generating it if missing), orders rows by time
across all assets and publishes each one as `TelemetryRow.model_dump_json()` to
`telemetry/{asset_id}`. Between rows it sleeps `(next_ts - ts) / speed` seconds, so
`REPLAY_SPEED=600` replays ten minutes of plant time per second. When the sequence ends it
starts again from the beginning (unless `--once`).

While streaming, the replay also acts as the plant's CMMS for approved work orders (see
`streaming/replay/work_orders.py`): at the approved window the motor goes offline for the
planned downtime and then streams its healthy baseline again. `--no-work-orders` (or
`REPLAY_WORK_ORDERS=0`) disables that.

If the broker is unreachable after 10 connection attempts, rows are POSTed in batches to
`{CONTROL_PLANE_URL}/ingest` instead. `--once --speed 0` bulk-loads everything via HTTP as
fast as possible (local dev + tests).

    python -m streaming.replay.replay [--speed 600] [--once] [--transport auto|mqtt|http]

This module runs inside the replay container without `apps/`; it depends only on pandas,
pyarrow, paho-mqtt, httpx and pydantic (through data.simulator.generate).
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from collections.abc import Iterable, Iterator
from datetime import datetime
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
from paho.mqtt import client as mqtt

from data.simulator.generate import TelemetryRow, generate_fleet, write_outputs
from streaming.replay.work_orders import RepairSimulator, WorkOrderClient

log = logging.getLogger("thor.replay")

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PARQUET = REPO_ROOT / "data" / "simulator" / "out" / "telemetry.parquet"
DEFAULT_TOPIC_FMT = "telemetry/{asset_id}"
DEFAULT_BATCH = 500


def parse_broker_url(url: str) -> tuple[str, int]:
    """'mqtt://host:1883' -> ('host', 1883). Missing port defaults to 1883."""
    tail = url.replace("mqtt://", "").replace("tcp://", "").strip("/")
    host, _, port = tail.partition(":")
    return host or "localhost", int(port) if port else 1883


def load_telemetry(path: Path = DEFAULT_PARQUET) -> pd.DataFrame:
    """Load the replay dataset, generating it with the simulator if the parquet is missing.

    Inputs: parquet path. Output: Telemetry-shaped DataFrame with tz-aware UTC `ts`.
    """
    path = Path(path)
    if not path.exists():
        log.warning("%s missing -- generating synthetic fleet", path)
        df, assets = generate_fleet()
        write_outputs(df, assets, path.parent)
    df = pd.read_parquet(path)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df


def iter_rows(df: pd.DataFrame) -> Iterator[TelemetryRow]:
    """Yield TelemetryRow models time-ordered across assets (ties broken by asset_id).

    Inputs: Telemetry-shaped DataFrame. Output: iterator of TelemetryRow; NaN ground-truth
    columns become None.
    """
    ordered = df.sort_values(["ts", "asset_id"], kind="mergesort")
    for rec in ordered.to_dict("records"):
        for key in ("health", "failure_within_h", "regime"):
            v = rec.get(key)
            if v is None or (isinstance(v, float) and np.isnan(v)):
                rec[key] = None
        yield TelemetryRow(**rec)


def _sleep_between(prev_ts: datetime | None, ts: datetime, speed: float) -> None:
    if prev_ts is None or speed <= 0:
        return
    delta = (ts - prev_ts).total_seconds() / speed
    if delta > 0:
        time.sleep(delta)


def connect_mqtt(
    host: str, port: int, max_retries: int = 10, retry_delay: float = 2.0
) -> mqtt.Client:
    """Connect a paho v2 client, retrying `max_retries` times; raise ConnectionError if it fails."""
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=f"thor-replay-{os.getpid()}")
    client.reconnect_delay_set(min_delay=1, max_delay=5)
    last_err: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            client.connect(host, port, keepalive=30)
            client.loop_start()
            log.info("connected to MQTT broker %s:%s", host, port)
            return client
        except OSError as exc:  # refused / DNS / timeout
            last_err = exc
            log.warning("MQTT connect attempt %d/%d failed: %s", attempt, max_retries, exc)
            time.sleep(retry_delay)
    raise ConnectionError(f"MQTT broker {host}:{port} unreachable: {last_err}")


def publish_mqtt(
    rows: Iterable[TelemetryRow],
    host: str,
    port: int,
    speed: float,
    topic_fmt: str = DEFAULT_TOPIC_FMT,
    client: mqtt.Client | None = None,
    max_retries: int = 10,
) -> int:
    """Publish rows to MQTT with time acceleration.

    Inputs: TelemetryRow iterable (time-ordered), broker host/port, speed factor (0 = no
    sleeping), topic format, optional pre-connected client. Sleeps (ts - prev_ts)/speed between
    rows. Output: number of rows published. Raises ConnectionError if the broker cannot be
    reached (caller falls back to HTTP).
    """
    own_client = client is None
    if client is None:
        client = connect_mqtt(host, port, max_retries=max_retries)
    n = 0
    prev_ts: datetime | None = None
    try:
        for row in rows:
            _sleep_between(prev_ts, row.ts, speed)
            client.publish(topic_fmt.format(asset_id=row.asset_id), row.model_dump_json(), qos=0)
            prev_ts = row.ts
            n += 1
            if n % 5000 == 0:
                log.info("published %d rows (last ts %s)", n, row.ts.isoformat())
    finally:
        if own_client:
            client.loop_stop()
            client.disconnect()
    return n


def publish_http(
    rows: Iterable[TelemetryRow],
    base_url: str,
    speed: float = 0.0,
    batch_size: int = DEFAULT_BATCH,
    max_retries: int = 10,
    retry_delay: float = 5.0,
    client: httpx.Client | None = None,
) -> int:
    """POST rows in batches to `{base_url}/ingest`.

    Inputs: rows, control-plane base URL, speed (0 = as fast as possible, else each batch is
    delayed by its time span / speed), batch size, retry policy. Output: rows sent.
    Raises ConnectionError when a batch cannot be delivered after `max_retries` attempts.
    """
    own = client is None
    http = client or httpx.Client(timeout=30.0)
    url = base_url.rstrip("/") + "/ingest"
    sent = 0
    batch: list[TelemetryRow] = []
    prev_ts: datetime | None = None

    def flush() -> None:
        nonlocal sent
        if not batch:
            return
        payload = [r.model_dump(mode="json") for r in batch]
        for attempt in range(1, max_retries + 1):
            try:
                resp = http.post(url, json=payload)
                resp.raise_for_status()
                sent += len(batch)
                batch.clear()
                return
            except (httpx.HTTPError, OSError) as exc:
                log.warning("POST %s attempt %d/%d failed: %s", url, attempt, max_retries, exc)
                time.sleep(retry_delay)
        raise ConnectionError(f"control plane {url} unreachable after {max_retries} attempts")

    try:
        for row in rows:
            if len(batch) >= batch_size:
                _sleep_between(prev_ts, row.ts, speed)
                prev_ts = row.ts
                flush()
            batch.append(row)
        flush()
    finally:
        if own:
            http.close()
    return sent


def main() -> None:
    """CLI entrypoint. Env: REPLAY_SPEED, MQTT_BROKER_URL, CONTROL_PLANE_URL."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)  # the work-order poll would log every 3 s
    host_default, port_default = parse_broker_url(
        os.getenv("MQTT_BROKER_URL", "mqtt://localhost:1883")
    )
    parser = argparse.ArgumentParser(description="Replay Thor telemetry over MQTT (or HTTP)")
    parser.add_argument("--speed", type=float, default=float(os.getenv("REPLAY_SPEED", "1800")))
    parser.add_argument("--once", action="store_true", help="play the sequence once, then exit")
    parser.add_argument("--transport", choices=["auto", "mqtt", "http"], default="auto")
    parser.add_argument("--parquet", type=Path, default=DEFAULT_PARQUET)
    parser.add_argument("--host", default=host_default)
    parser.add_argument("--port", type=int, default=port_default)
    parser.add_argument(
        "--control-plane", default=os.getenv("CONTROL_PLANE_URL", "http://localhost:8000")
    )
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH)
    parser.add_argument("--asset", default=None, help="replay only this asset id")
    parser.add_argument(
        "--no-work-orders",
        action="store_true",
        default=os.getenv("REPLAY_WORK_ORDERS", "1").strip().lower() in {"0", "false", "no"},
        help="do not execute approved work orders (motors never come back online)",
    )
    parser.add_argument(
        "--start-fraction",
        type=float,
        default=float(os.getenv("REPLAY_START_FRACTION", "0.80")),
        help="first pass starts this far into the timeline (the API seed loads the head)",
    )
    args = parser.parse_args()

    df = load_telemetry(args.parquet)
    if args.asset:
        df = df[df["asset_id"] == args.asset].copy()
    log.info(
        "loaded %d rows for %d assets (%s -> %s); speed=%s transport=%s",
        len(df),
        df["asset_id"].nunique(),
        df["ts"].min(),
        df["ts"].max(),
        args.speed,
        args.transport,
    )
    transport = args.transport
    if transport == "auto" and args.speed <= 0:
        transport = "http"
    repairs: RepairSimulator | None = None
    if not args.no_work_orders and args.speed > 0:
        repairs = RepairSimulator(df, WorkOrderClient(args.control_plane))
        log.info("simulated CMMS on: executing approved work orders from %s", args.control_plane)

    first_pass = True
    while True:
        play = df
        if first_pass and 0.0 < args.start_fraction < 1.0 and args.speed > 0:
            t0, t1 = df["ts"].min(), df["ts"].max()
            play = df.loc[df["ts"] > t0 + (t1 - t0) * args.start_fraction]
            log.info(
                "first pass starts at %s (%.0f%% of timeline)",
                play["ts"].min(),
                args.start_fraction * 100,
            )
        first_pass = False
        rows = repairs.apply(iter_rows(play)) if repairs else iter_rows(play)
        try:
            if transport == "http":
                n = publish_http(
                    rows, args.control_plane, speed=args.speed, batch_size=args.batch_size
                )
                log.info("bulk-loaded %d rows via HTTP", n)
            else:
                try:
                    n = publish_mqtt(rows, args.host, args.port, args.speed)
                    log.info("published %d rows via MQTT", n)
                except ConnectionError as exc:
                    if transport == "mqtt":
                        raise
                    log.warning("%s -- falling back to HTTP ingest", exc)
                    n = publish_http(
                        rows, args.control_plane, speed=args.speed, batch_size=args.batch_size
                    )
                    log.info("published %d rows via HTTP fallback", n)
        except ConnectionError as exc:
            log.error("%s -- retrying in 5s", exc)
            time.sleep(5)
            continue
        if args.once:
            break
        log.info("sequence finished -- looping from the start")


if __name__ == "__main__":
    main()
