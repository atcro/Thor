"""The one status rule (apps/api/alarms.py) and the replay gate.

Pinned here (CLAUDE.md section 15, finals hero demo):

* on-delay: an alarm needs EARLY_WARNING_SUSTAINED_ROWS consecutive rows over the threshold;
* latch: once critical, a motor stays critical until a completed work order clears it;
* a warning clears only after WARNING_CLEAR_ROWS consecutive rows below the line;
* start-up produces no model and no predictions, and the replay is held until a production
  model exists (`GET /replay`).

Everything runs through the FastAPI app on a temp SQLite file; no broker, no LLM.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from apps.api import alarms, db, live, replay_control
from apps.api.alarms import ALARM_RAISE_ROWS, WARNING_CLEAR_ROWS, asset_status, evaluate
from apps.api.schemas import (
    EARLY_WARNING_SUSTAINED_ROWS,
    AlarmLevel,
    Approval,
    DataState,
    TwinStatus,
    WorkOrder,
)
from tests.test_orchestrator import NOW, configure_env, make_bundle, make_contract, seed_small_fleet


@pytest.fixture(autouse=True)
def _reset_state() -> Iterator[None]:
    live.reset()
    alarms.reset()
    replay_control.set_paused(False)
    yield
    live.reset()
    alarms.reset()
    replay_control.set_paused(False)


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    configure_env(tmp_path, monkeypatch)
    from apps.api.main import app

    with TestClient(app) as c:
        seed_small_fleet(db.get_engine())
        yield c


def _post(client: TestClient, asset_id: str, probs: list[float], start: datetime) -> list[datetime]:
    stamps = [start + timedelta(minutes=10 * i) for i in range(len(probs))]
    body = [
        {
            "asset_id": asset_id,
            "ts": ts.isoformat(),
            "model_version": "v9",
            "failure_probability": p,
        }
        for ts, p in zip(stamps, probs, strict=True)
    ]
    assert client.post("/predictions", json=body).status_code == 200
    return stamps


def _fleet(client: TestClient) -> dict[str, dict[str, Any]]:
    return {f["asset"]["asset_id"]: f for f in client.get("/fleet").json()}


# --------------------------------------------------------------------------------------
# State machine
# --------------------------------------------------------------------------------------


def test_on_delay_latch_and_warning_clear_rule() -> None:
    n = EARLY_WARNING_SUSTAINED_ROWS
    assert ALARM_RAISE_ROWS == n and WARNING_CLEAR_ROWS == 2 * n
    # one spike, two spikes: nothing. three sustained: warning.
    assert evaluate([0.9]) == []
    assert evaluate([0.35, 0.35, 0.1, 0.35, 0.35]) == []
    assert evaluate([0.1, 0.35, 0.35, 0.35]) == [(3, AlarmLevel.warning)]
    # straight to critical when three rows are over the classification threshold
    assert evaluate([0.6, 0.6, 0.6]) == [(2, AlarmLevel.critical)]
    # warning escalates to critical; critical never clears on its own
    series = [0.35, 0.35, 0.35, 0.6, 0.6, 0.6] + [0.0] * 50
    assert evaluate(series) == [(2, AlarmLevel.warning), (5, AlarmLevel.critical)]
    # a warning clears after WARNING_CLEAR_ROWS rows below the line, not before
    assert evaluate([0.35] * 3 + [0.1] * (2 * n - 1)) == [(2, AlarmLevel.warning)]
    assert evaluate([0.35] * 3 + [0.1] * (2 * n)) == [(2, AlarmLevel.warning), (2 + 2 * n, AlarmLevel.healthy)]
    # a borderline probability never chatters: 0.3 on/off alternating stays warning
    assert evaluate([0.3] * 3 + [0.29, 0.3] * 10) == [(2, AlarmLevel.warning)]


def test_asset_status_precedence() -> None:
    wo_live = WorkOrder(
        contract_id="dc", asset_id="MTR-001", recommendation="maintain_now", window_start=None,
        window_end=None, planned_downtime_h=3.0, approved_at=NOW, approver="e", status="in_progress",
    )
    assert asset_status(AlarmLevel.critical, 0.9, DataState.live, wo_live) == TwinStatus.offline
    assert asset_status(None, None, DataState.none, None) == TwinStatus.no_prediction
    assert asset_status(AlarmLevel.critical, None, DataState.live, None) == TwinStatus.no_prediction
    assert asset_status(AlarmLevel.critical, 0.9, DataState.stale, None) == TwinStatus.no_data
    assert asset_status(AlarmLevel.critical, 0.0, DataState.none, None) == TwinStatus.critical
    assert asset_status(AlarmLevel.warning, 0.1, DataState.live, None) == TwinStatus.warning
    assert asset_status(None, 0.95, DataState.live, None) == TwinStatus.healthy  # not sustained yet


def test_latch_clears_only_on_work_order_completion(client: TestClient) -> None:
    engine = db.get_engine()
    stamps = _post(client, "MTR-042", [0.7, 0.8, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], NOW)
    f = _fleet(client)["MTR-042"]
    assert f["status"] == "critical" and f["alarm"] == "critical" and f["failure_probability"] == 0.0
    assert f["alarm_since"] == stamps[0].isoformat().replace("+00:00", "Z")  # first row of the run
    # a rejected contract or a pending one changes nothing
    dc = make_contract(make_bundle("MTR-042"), run_id="r1", text="t", source="template")
    db.insert_decision_contract(dc, engine=engine)
    assert _fleet(client)["MTR-042"]["status"] == "critical"
    db.insert_approval(
        Approval(approval_id="ap1", contract_id=dc.contract_id, decision="approved",
                 approver="engineer@plant", decided_at=datetime.now(UTC)),
        engine=engine,
    )
    assert _fleet(client)["MTR-042"]["status"] == "critical"  # approved, plant has not acted
    ev = {"status": "in_progress", "plant_ts": (NOW + timedelta(hours=3)).isoformat()}
    assert client.post(f"/work-orders/{dc.contract_id}/events", json=ev).status_code == 200
    assert _fleet(client)["MTR-042"]["status"] == "offline"
    done = {"status": "completed", "plant_ts": (NOW + timedelta(hours=6)).isoformat()}
    assert client.post(f"/work-orders/{dc.contract_id}/events", json=done).status_code == 200
    f = _fleet(client)["MTR-042"]
    assert f["status"] == "healthy" and f["alarm"] is None
    # after the clear a new alarm needs fresh sustained evidence
    _post(client, "MTR-042", [0.9, 0.9], NOW + timedelta(hours=7))
    assert _fleet(client)["MTR-042"]["status"] == "healthy"
    _post(client, "MTR-042", [0.9], NOW + timedelta(hours=7, minutes=20))
    assert _fleet(client)["MTR-042"]["status"] == "critical"
    levels = [e["level"] for e in client.get("/alarms", params={"asset_id": "MTR-042"}).json()]
    assert levels == ["critical", "healthy", "critical"]


def test_alarm_walk_survives_a_restart(client: TestClient) -> None:
    """The cursors are in memory; after a restart the level comes from the stored events and
    the walk resumes from the rows after the last event."""
    _post(client, "MTR-042", [0.35, 0.35, 0.35], NOW)
    assert _fleet(client)["MTR-042"]["alarm"] == "warning"
    alarms.reset()  # simulate a process restart
    _post(client, "MTR-042", [0.6, 0.6], NOW + timedelta(minutes=30))
    assert _fleet(client)["MTR-042"]["alarm"] == "warning"
    _post(client, "MTR-042", [0.6], NOW + timedelta(minutes=50))
    assert _fleet(client)["MTR-042"]["alarm"] == "critical"


# --------------------------------------------------------------------------------------
# Start-up and the replay gate
# --------------------------------------------------------------------------------------


def test_startup_has_no_model_no_predictions_and_holds_the_replay(client: TestClient) -> None:
    assert client.get("/models").json() == []
    fleet = client.get("/fleet").json()
    assert all(f["health_score"] is None and f["status"] == "no_prediction" for f in fleet)
    assert all(a["status"] == "no_prediction" for a in client.get("/twin").json()["assets"])
    with db.get_engine().connect() as conn:
        assert conn.execute(db.select(db.func.count()).select_from(db.predictions)).scalar() == 0
    r = client.get("/replay").json()
    assert r["released"] is False and r["paused"] is False
    assert r["label"] == "Replay (recorded plant history)"
    assert r["production_model_version"] is None and r["plant_time"] is not None
    # pause / resume are the only controls
    assert client.post("/replay/pause").json()["paused"] is True
    assert client.get("/replay").json()["paused"] is True
    assert client.post("/replay/resume").json()["paused"] is False


def test_replay_is_released_only_once_a_production_model_exists(client: TestClient) -> None:
    from sqlalchemy import insert

    engine = db.get_engine()
    with engine.begin() as conn:
        conn.execute(
            insert(db.model_registry).values(
                name="thor-pdm", version="v1", stage="validated", family="lightgbm",
                ims_total=0.8, registered_at=datetime.now(UTC), payload={},
            )
        )
    assert client.get("/replay").json()["released"] is False  # validated is not production
    with engine.begin() as conn:
        conn.execute(
            insert(db.model_registry).values(
                name="thor-pdm", version="v2", stage="production", family="lightgbm",
                ims_total=0.8, registered_at=datetime.now(UTC), payload={},
            )
        )
    r = client.get("/replay").json()
    assert r["released"] is True and r["production_model_version"] == "v2"


def _row_dict(asset_id: str, ts: datetime) -> dict[str, Any]:
    return {
        "asset_id": asset_id, "ts": ts.isoformat(), "vibration_rms": 1.1, "vibration_kurtosis": 3.1,
        "vibration_crest": 4.0, "bearing_temp_c": 61.0, "motor_temp_c": 55.0, "current_a": 40.0,
        "rpm": 1480.0, "load_pct": 55.0,
    }


# --------------------------------------------------------------------------------------
# Every approved motor comes back green: a repaired motor's window restarts after the outage
# --------------------------------------------------------------------------------------


def test_window_restarts_after_an_outage_gap() -> None:
    import pandas as pd

    from apps.api.scoring import after_last_gap
    from edge.main import EdgeConfig, EdgeState

    ts = pd.date_range("2025-08-26 00:00", periods=20, freq="10min", tz="UTC")
    ts = ts[:10].append(ts[10:] + pd.Timedelta(hours=3))  # three hours off for the repair
    df = pd.DataFrame({"asset_id": "MTR-042", "ts": ts, "vibration_rms": range(20)})
    tail = after_last_gap(df)
    assert len(tail) == 10 and int(tail["vibration_rms"].iloc[0]) == 10
    assert len(after_last_gap(df.iloc[:10])) == 10  # no gap, nothing dropped

    state = EdgeState(EdgeConfig(mqtt_enabled=False, buffer_rows=64))
    base = {c: 1.0 for c in ("vibration_rms", "vibration_kurtosis", "vibration_crest", "bearing_temp_c",
                             "motor_temp_c", "current_a", "rpm", "load_pct")}
    for t in ts[:10]:
        state.process_row({"asset_id": "MTR-042", "ts": t.isoformat(), **base})
    assert len(state.buffers["MTR-042"]) == 10
    state.process_row({"asset_id": "MTR-042", "ts": ts[10].isoformat(), **base})
    assert len(state.buffers["MTR-042"]) == 1  # the pre-outage rows never meet the healthy return
