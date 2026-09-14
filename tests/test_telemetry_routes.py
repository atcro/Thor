"""Tests for apps/api/routes/telemetry.py (Agent A). No broker, no network.

The engine is an in-memory SQLite database on a StaticPool: `db.get_engine("sqlite://")` would
give every TestClient worker thread its own empty database (SingletonThreadPool), so the
fixture creates the shared-connection engine directly and passes it through `engine=`.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, func, select
from sqlalchemy.engine import Engine
from sqlalchemy.pool import StaticPool

from apps.api import db
from apps.api.routes import telemetry as tel
from apps.api.schemas import TelemetryRow
from data.simulator.generate import df_to_rows, generate_fleet


@pytest.fixture
def engine() -> Engine:
    eng = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool, future=True
    )
    db.init_db(eng)
    return eng


@pytest.fixture
def client(engine: Engine) -> Iterator[TestClient]:
    app = FastAPI()
    app.include_router(tel.router)
    app.dependency_overrides[tel.get_db_engine] = lambda: engine
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def demo_rows() -> list[TelemetryRow]:
    df, _ = generate_fleet(days=2, seed=11)
    return df_to_rows(df[df["asset_id"] == "MTR-042"])


def _json(rows: list[TelemetryRow]) -> list[dict]:
    return [r.model_dump(mode="json") for r in rows]


def test_ingest_single_and_batch(
    client: TestClient, engine: Engine, demo_rows: list[TelemetryRow]
) -> None:
    r = client.post("/ingest", json=demo_rows[0].model_dump(mode="json"))
    assert r.status_code == 200 and r.json() == {"inserted": 1}
    r = client.post("/ingest", json=_json(demo_rows[1:51]))
    assert r.status_code == 200 and r.json() == {"inserted": 50}
    # duplicates are ignored, not errors
    r = client.post("/ingest", json=_json(demo_rows[:10]))
    assert r.status_code == 200
    with engine.connect() as conn:
        n = conn.execute(select(func.count()).select_from(db.telemetry)).scalar_one()
    assert n == 51


def test_ingest_rejects_bad_payload(client: TestClient) -> None:
    r = client.post("/ingest", json={"asset_id": "MTR-001", "ts": "2025-08-01T00:00:00Z"})
    assert r.status_code == 422


def test_get_telemetry_window_and_limit(client: TestClient, demo_rows: list[TelemetryRow]) -> None:
    client.post("/ingest", json=_json(demo_rows))  # 288 rows over 2 days
    r = client.get("/assets/MTR-042/telemetry", params={"hours": 24})
    assert r.status_code == 200
    body = r.json()
    assert 144 <= len(body) <= 145  # 24 h at 10-min steps, anchored on the latest row
    ts = [datetime.fromisoformat(b["ts"]) for b in body]
    assert ts == sorted(ts)
    assert ts[-1] == demo_rows[-1].ts
    assert ts[-1] - ts[0] <= timedelta(hours=24)
    row = TelemetryRow.model_validate(body[-1])
    assert row.asset_id == "MTR-042" and row.regime in {"R1", "R2", "R3"}
    r = client.get("/assets/MTR-042/telemetry", params={"hours": 24, "limit": 10})
    assert len(r.json()) == 10
    assert datetime.fromisoformat(r.json()[-1]["ts"]) == demo_rows[-1].ts
    r = client.get("/assets/MTR-042/telemetry", params={"hours": 1000, "limit": 100000})
    assert len(r.json()) == len(demo_rows)


def test_get_telemetry_unknown_asset_is_empty(client: TestClient) -> None:
    r = client.get("/assets/MTR-999/telemetry")
    assert r.status_code == 200 and r.json() == []


def test_predictions_roundtrip(client: TestClient) -> None:
    base = datetime(2025, 8, 20, 12, 0, tzinfo=UTC)
    for i in range(7):
        r = client.post(
            "/predictions",
            json={
                "asset_id": "MTR-042",
                "ts": (base + timedelta(minutes=10 * i)).isoformat(),
                "model_version": "v1",
                "failure_probability": 0.1 * i,
                "source": "edge",
            },
        )
        assert r.status_code == 200 and r.json() == {"ok": True}
    r = client.post(
        "/predictions",
        json={
            "asset_id": "MTR-042",
            "ts": base.isoformat(),
            "model_version": "v1",
            "failure_probability": 1.5,
        },
    )
    assert r.status_code == 422
    r = client.get("/assets/MTR-042/predictions", params={"limit": 5})
    assert r.status_code == 200
    body = r.json()
    assert len(body) == 5
    assert [b["failure_probability"] for b in body] == pytest.approx([0.2, 0.3, 0.4, 0.5, 0.6])
    assert body[-1]["model_version"] == "v1" and body[-1]["source"] == "edge"
    assert datetime.fromisoformat(body[-1]["ts"]) == base + timedelta(minutes=60)
    assert client.get("/assets/MTR-001/predictions").json() == []
    assert db.latest_predictions(engine=client.app.dependency_overrides[tel.get_db_engine]())[
        "MTR-042"
    ] == pytest.approx(0.6)


def test_mqtt_subscriber_batches_payloads(engine: Engine, demo_rows: list[TelemetryRow]) -> None:
    sub = tel.MqttSubscriber(engine, host="127.0.0.1", port=1, batch_size=50, flush_interval=0.1)
    for r in demo_rows[:49]:
        assert sub.handle_payload(r.model_dump_json())
    assert sub.rows_inserted == 0  # below batch size, nothing flushed yet
    assert sub.handle_payload(demo_rows[49].model_dump_json().encode())
    assert sub.rows_inserted == 50  # 50th message triggers the batch insert
    assert not sub.handle_payload(b"not json")
    assert not sub.handle_payload(json.dumps({"asset_id": "x"}))
    assert sub.handle_payload(demo_rows[50].model_dump_json())
    assert sub.flush() == 1
    assert sub.messages_received == 51
    assert len(db.read_telemetry(asset_id="MTR-042", engine=engine)) == 51
    status = sub.status()
    assert status["connected"] is False and status["rows_inserted"] == 51


def test_start_mqtt_subscriber_never_raises_without_broker(engine: Engine) -> None:
    tel.stop_mqtt_subscriber()  # clean slate
    tel.start_mqtt_subscriber(engine, host="127.0.0.1", port=1)  # nothing listens on port 1
    tel.start_mqtt_subscriber(engine, host="127.0.0.1", port=1)  # idempotent
    status = tel.mqtt_status()
    assert status["broker"] == "127.0.0.1:1"
    assert tel.is_mqtt_connected() is False
    tel.stop_mqtt_subscriber()
    tel.stop_mqtt_subscriber()  # idempotent
    assert tel.mqtt_status()["last_error"] == "not started"
