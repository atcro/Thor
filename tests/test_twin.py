"""Tests for the 3-D Digital Twin backend (Phase A): apps/api/live.py, apps/api/plant.py and
apps/api/routes/twin.py.

Everything is driven through the FastAPI app on a temp SQLite file (no broker, no network,
no LLM). The live bus is reset around every test. These tests also pin two rules from
CLAUDE.md section 15: status keys off the displayed (median) probability against the named
thresholds, and the simulated MES feed is HUD context only -- it is imported by the twin
routes alone and never reaches a run state or a Decision Contract.
"""

from __future__ import annotations

import re
import time
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from apps.api import alarms, db, live, plant
from apps.api.routes import twin as twin_routes
from apps.api.schemas import SENSOR_COLUMNS, Asset, DataState, DecisionState, ObservedTelemetryRow
from tests.test_api import _decided_contract
from tests.test_ground_truth_isolation import _assert_no_ground_truth, _keys
from tests.test_orchestrator import (
    NOW,
    configure_env,
    install_fake_toolbox,
    make_bundle,
    make_contract,
    seed_small_fleet,
    wait_for_stage,
)

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _reset_live_bus() -> Iterator[None]:
    live.reset()
    alarms.reset()
    yield
    live.reset()
    alarms.reset()


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    configure_env(tmp_path, monkeypatch)
    from apps.api.main import app

    with TestClient(app) as c:
        seed_small_fleet(db.get_engine())
        yield c


def _row(asset_id: str, ts: datetime, **over: float) -> dict[str, Any]:
    base: dict[str, Any] = {
        "asset_id": asset_id,
        "ts": ts.isoformat(),
        "vibration_rms": 1.1,
        "vibration_kurtosis": 3.1,
        "vibration_crest": 4.0,
        "bearing_temp_c": 61.0,
        "motor_temp_c": 55.0,
        "current_a": 40.0,
        "rpm": 1480.0,
        "load_pct": 55.0,
    }
    base.update(over)
    return base


def _twin_asset(client: TestClient, asset_id: str) -> dict[str, Any]:
    snap = client.get("/twin").json()
    return next(a for a in snap["assets"] if a["asset"]["asset_id"] == asset_id)


def _predict(engine: Any, asset_id: str, probs: list[float], start: datetime) -> list[datetime]:
    """Insert `probs` as one prediction per 10 minutes from `start` (oldest first)."""
    stamps = [start + timedelta(minutes=10 * i) for i in range(len(probs))]
    for ts, p in zip(stamps, probs, strict=True):
        db.insert_prediction(asset_id, ts, "v9", p, engine=engine)
    return stamps


# --------------------------------------------------------------------------------------
# Status, data state, decision state
# --------------------------------------------------------------------------------------


def _post_predictions(client: TestClient, asset_id: str, probs: list[float], start: datetime) -> list[datetime]:
    """Push `probs` through POST /predictions (the edge sink), one per 10 minutes, so the
    alarm walk runs exactly as it does for the edge container."""
    stamps = [start + timedelta(minutes=10 * i) for i in range(len(probs))]
    body = [
        {"asset_id": asset_id, "ts": ts.isoformat(), "model_version": "v9", "failure_probability": p}
        for ts, p in zip(stamps, probs, strict=True)
    ]
    assert client.post("/predictions", json=body).status_code == 200
    return stamps


def test_status_precedence_and_data_state(client: TestClient) -> None:
    """The ONE status rule (apps/api/alarms.py): offline > no_prediction > no_data > latched
    critical > warning > healthy. Sustained rows raise; critical stays until a work order."""
    engine = db.get_engine()
    a = _twin_asset(client, "MTR-042")
    # nothing received and nothing scored: "no prediction yet", not "no data"
    assert a["data_state"] == "none" and a["status"] == "no_prediction"
    assert a["decision_state"] == "none" and a["failure_probability"] is None
    assert a["last_received_at"] is None and a["msgs_last_60s"] == 0
    assert a["alarm"] is None and a["alarm_since"] is None

    # a message through POST /ingest makes the motor live; nothing scored it yet
    r = client.post("/ingest", json=_row("MTR-042", NOW + timedelta(minutes=10)))
    assert r.status_code == 200
    a = _twin_asset(client, "MTR-042")
    assert a["data_state"] == "live" and a["status"] == "no_prediction"
    assert a["msgs_last_60s"] == 1 and a["last_received_at"] is not None
    assert a["latest"]["vibration_rms"] == 1.1 and a["last_ts"].startswith("2026-09-01T12:10")

    # on-delay: two rows over the classification threshold are not an alarm; the third is,
    # and the displayed probability is the median of the last three rows (same as /fleet)
    stamps = _post_predictions(client, "MTR-042", [0.6, 0.7], NOW)
    a = _twin_asset(client, "MTR-042")
    assert a["status"] == "healthy" and a["alarm"] is None and a["failure_probability"] == pytest.approx(0.65)
    stamps += _post_predictions(client, "MTR-042", [0.65], NOW + timedelta(minutes=20))
    a = _twin_asset(client, "MTR-042")
    assert a["status"] == "critical" and a["failure_probability"] == 0.65
    assert a["alarm"] == "critical" and a["alarm_since"] == stamps[2].isoformat().replace("+00:00", "Z")
    assert "3 consecutive predictions >= 0.5" in a["alarm_reason"]
    assert a["n_prediction_rows"] == 3 and a["health_score"] == 35.0
    # latched: the probability falling back changes the number, never the status
    _post_predictions(client, "MTR-042", [0.4, 0.35, 0.45], NOW + timedelta(hours=1))
    a = _twin_asset(client, "MTR-042")
    assert a["status"] == "critical" and a["failure_probability"] == 0.4
    _post_predictions(client, "MTR-042", [0.1, 0.05, 0.2, 0.0, 0.0, 0.0, 0.0], NOW + timedelta(hours=2))
    a = _twin_asset(client, "MTR-042")
    assert a["status"] == "critical" and a["failure_probability"] == 0.0

    # a warning needs three sustained rows too, and six rows below the line to clear
    _post_predictions(client, "MTR-002", [0.35, 0.4], NOW)
    # scored but no message yet (the stream has not started): healthy, never "no data"
    assert _twin_asset(client, "MTR-002")["status"] == "healthy"
    client.post("/ingest", json=_row("MTR-002", NOW))
    assert _twin_asset(client, "MTR-002")["status"] == "healthy"
    _post_predictions(client, "MTR-002", [0.33], NOW + timedelta(minutes=20))
    a = _twin_asset(client, "MTR-002")
    assert a["status"] == "warning" and a["alarm"] == "warning"
    _post_predictions(client, "MTR-002", [0.1] * 5, NOW + timedelta(minutes=30))
    assert _twin_asset(client, "MTR-002")["status"] == "warning"
    _post_predictions(client, "MTR-002", [0.1], NOW + timedelta(minutes=80))
    a = _twin_asset(client, "MTR-002")
    assert a["status"] == "healthy" and a["alarm"] is None

    # driving the bus directly: a receipt older than STALE_AFTER_S is stale -> no_data
    live.record_ingest("MTR-001", NOW, received_at=time.time() - live.STALE_AFTER_S - 60)
    assert live.bus.data_state("MTR-001") == DataState.stale
    assert live.bus.data_state("MTR-003") == DataState.none
    _predict(engine, "MTR-001", [0.9, 0.9, 0.9], NOW)
    a = _twin_asset(client, "MTR-001")
    assert a["data_state"] == "stale" and a["status"] == "no_data"
    assert a["failure_probability"] == 0.9  # the number is still reported, the glyph is not

    # an approved work order the plant is executing takes the motor offline, whatever the
    # data, and its completion is what clears the latched critical
    contract_id = _decided_contract(engine, "MTR-042", "maintain_now")
    a = _twin_asset(client, "MTR-042")
    assert a["decision_state"] == "approved" and a["status"] == "critical"
    assert a["work_order"]["status"] == "scheduled"
    ev = {"status": "in_progress", "plant_ts": (NOW + timedelta(hours=3)).isoformat()}
    assert client.post(f"/work-orders/{contract_id}/events", json=ev).status_code == 200
    a = _twin_asset(client, "MTR-042")
    assert a["status"] == "offline" and a["decision_state"] == "work_order_in_progress"
    done = {"status": "completed", "plant_ts": (NOW + timedelta(hours=6)).isoformat()}
    assert client.post(f"/work-orders/{contract_id}/events", json=done).status_code == 200
    a = _twin_asset(client, "MTR-042")
    assert a["status"] == "healthy" and a["decision_state"] == "work_order_completed"
    assert a["alarm"] is None
    events = client.get("/alarms", params={"asset_id": "MTR-042"}).json()
    assert [e["level"] for e in events] == ["critical", "healthy"]
    assert events[-1]["reason"] == "work_order_completed"
    # /fleet applies the same rule and carries the same fields
    f = {x["asset"]["asset_id"]: x for x in client.get("/fleet").json()}
    assert f["MTR-042"]["status"] == "healthy" and f["MTR-001"]["status"] == "no_data"
    assert f["MTR-003"]["status"] == "no_prediction" and f["MTR-002"]["alarm"] is None


def test_decision_state_badges(client: TestClient) -> None:
    engine = db.get_engine()
    # pending contract
    dc = make_contract(make_bundle("MTR-002"), run_id="run_x", text="t", source="template")
    db.insert_decision_contract(dc, engine=engine)
    a = _twin_asset(client, "MTR-002")
    assert a["decision_state"] == "contract_pending" and a["open_contract_id"] == dc.contract_id
    # rejected contract
    _decided_contract(engine, "MTR-003", "maintain_now", decision="rejected")
    assert _twin_asset(client, "MTR-003")["decision_state"] == "rejected"
    # a run_to_failure approval is no work order: badge falls back to none
    _decided_contract(engine, "MTR-001", "run_to_failure")
    assert _twin_asset(client, "MTR-001")["decision_state"] == "none"
    # a completed work order followed by a NEW pending contract shows the pending contract
    wo_id = _decided_contract(engine, "MTR-042", "maintain_now")
    done = {"status": "completed", "plant_ts": NOW.isoformat()}
    assert client.post(f"/work-orders/{wo_id}/events", json=done).status_code == 200
    assert _twin_asset(client, "MTR-042")["decision_state"] == "work_order_completed"
    dc2 = make_contract(make_bundle("MTR-042"), run_id="run_y", text="t", source="template")
    dc2 = dc2.model_copy(update={"created_at": datetime.now(UTC) + timedelta(seconds=5)})
    db.insert_decision_contract(dc2, engine=engine)
    assert _twin_asset(client, "MTR-042")["decision_state"] == "contract_pending"


# --------------------------------------------------------------------------------------
# Plant layout grouping
# --------------------------------------------------------------------------------------


_PACKAGING_ROLES = ("infeed conveyor", "filler", "capper/labeler", "case packer")


def _layout_assets() -> list[Asset]:
    """24 motors in the finals layout (names only; the generator is untouched)."""
    out: list[Asset] = []
    # the seeded fleet's four motors (MTR-001/002/003/042) are re-homed by the upsert
    ids = iter([f"MTR-{i:03d}" for i in range(1, 24)] + ["MTR-042"])
    for line_no in range(1, 6):
        # deliberately insert in reverse machine order so the test proves role ordering
        for role in reversed(_PACKAGING_ROLES):
            aid = next(ids)
            out.append(
                Asset(
                    asset_id=aid, name=f"L{line_no} {role}", site="Ludvika",
                    line=f"Packaging Line {line_no}", role=role,
                )
            )
    for k in range(1, 5):
        aid = next(ids)
        out.append(
            Asset(
                asset_id=aid, name=f"K-{k}", site="Ludvika", line="Compressor Hall",
                role="air compressor",
            )
        )
    return out


def test_snapshot_groups_lines_in_plant_order(client: TestClient) -> None:
    engine = db.get_engine()
    assets = _layout_assets()
    db.upsert_assets(assets, engine=engine)
    snap = client.get("/twin").json()
    assert snap["site"] == "Ludvika"
    assert [s["name"] for s in snap["sites"]] == ["Packaging", "Molding", "Batching"]
    assert [s["available"] for s in snap["sites"]] == [True, False, False]
    assert "not built" in snap["sites"][1]["note"]
    lines = snap["lines"]
    assert [ln["line"] for ln in lines] == [f"Packaging Line {i}" for i in range(1, 6)] + [
        "Compressor Hall"
    ]
    assert [ln["index"] for ln in lines] == list(range(6))
    assert [ln["kind"] for ln in lines] == ["packaging"] * 5 + ["compressor"]
    by_id = {a.asset_id: a for a in assets}
    for ln in lines[:5]:
        assert len(ln["assets"]) == 4
        assert [by_id[a].role for a in ln["assets"]] == list(_PACKAGING_ROLES)
        assert ln["mes"] is not None and ln["mes"]["source"] == "simulated MES feed"
        assert ln["mes"]["line"] == ln["line"]
    assert lines[5]["mes"] is None
    assert lines[5]["assets"] == sorted(lines[5]["assets"])
    assert len(snap["assets"]) == 24
    assert snap["plant_time"] is not None and snap["thresholds"]["early_warning"] == 0.3
    assert snap["model"] is None  # empty registry never raises
    assert snap["live"]["window_s"] == 10 and snap["live"]["stale_after_s"] == live.STALE_AFTER_S


def test_snapshot_handles_generic_fleet(client: TestClient) -> None:
    """The unit-test fleet (site "Plant A", lines L1/L2, no roles) still groups cleanly."""
    snap = client.get("/twin").json()
    assert snap["site"] == "Plant A"
    assert [(ln["line"], ln["kind"], ln["index"]) for ln in snap["lines"]] == [
        ("L1", "packaging", 0),
        ("L2", "packaging", 1),
    ]
    assert snap["lines"][0]["assets"] == ["MTR-001", "MTR-003"]
    assert snap["lines"][1]["assets"] == ["MTR-002", "MTR-042"]
    assert [a["asset"]["asset_id"] for a in snap["assets"]] == [
        "MTR-001", "MTR-002", "MTR-003", "MTR-042",
    ]


# --------------------------------------------------------------------------------------
# HUD
# --------------------------------------------------------------------------------------


def test_hud_trend_and_plc_tags(client: TestClient) -> None:
    engine = db.get_engine()
    assert client.get("/twin/hud/MTR-999").status_code == 404
    hud = client.get("/twin/hud/MTR-042").json()
    assert hud["trend"] is None and hud["prediction_trend"] == []
    assert hud["drivers"] == [] and hud["contract"] is None and hud["model"] is None
    assert hud["line"]["line"] == "L2" and hud["mes"]["source"] == "simulated MES feed"

    _predict(engine, "MTR-042", [0.1, 0.1, 0.1, 0.3, 0.3, 0.3], NOW)
    hud = client.get("/twin/hud/MTR-042").json()
    assert hud["trend"] == "rising"
    trend = hud["prediction_trend"]
    assert [p["failure_probability"] for p in trend] == [0.1, 0.1, 0.1, 0.3, 0.3, 0.3]
    assert trend[0]["ts"] < trend[-1]["ts"]  # oldest first

    latest = hud["asset"]["latest"]
    tags = {t["tag"]: t for t in hud["plc_tags"]}
    assert set(tags) == {
        "Run_Bit", "Speed_Act", "Load_Pct", "Motor_Current", "Bearing_Temp", "Winding_Temp",
        "Vib_RMS", "Vib_Kurtosis", "Speed_SP",
    }
    for t in hud["plc_tags"]:
        if t["simulated"]:
            assert t["source"] is None and t["basis"].startswith("simulated:")
            continue
        assert t["source"] in SENSOR_COLUMNS
        if t["tag"] == "Run_Bit":
            assert t["value"] is (latest["rpm"] > 100) and t["basis"] == "rpm > 100"
        else:
            assert t["value"] == latest[t["source"]]
    assert tags["Speed_SP"]["value"] is None
    assert tags["Speed_SP"]["basis"] == "simulated: no production baseline"

    # drivers come from the newest Decision Contract, with their source
    dc = make_contract(make_bundle("MTR-042"), run_id="run_h", text="t", source="template")
    db.insert_decision_contract(dc, engine=engine)
    hud = client.get("/twin/hud/MTR-042").json()
    assert [d["feature"] for d in hud["drivers"]] == [f.feature for f in dc.top_features]
    assert hud["drivers_source"]["contract_id"] == dc.contract_id
    assert hud["drivers_source"]["run_id"] == "run_h" and hud["contract_status"] == "pending"
    assert hud["contract"]["contract_id"] == dc.contract_id


def test_trend_rule_and_plc_setpoint_units() -> None:
    assert twin_routes.trend_of([0.1] * 5) is None
    assert twin_routes.trend_of([0.5, 0.5, 0.5, 0.4, 0.46, 0.44]) == "falling"
    assert twin_routes.trend_of([0.5, 0.5, 0.5, 0.52, 0.49, 0.53]) == "flat"
    assert twin_routes.trend_of([0.1, 0.9, 0.1, 0.2, 0.2, 0.2]) == "rising"

    baseline = {"_regime_centroids": {"R2_rpm": 1480.0, "R2_load_pct": 55.0}}
    latest = ObservedTelemetryRow(
        asset_id="MTR-042", ts=NOW, vibration_rms=1.0, vibration_kurtosis=3.0,
        vibration_crest=4.0, bearing_temp_c=60.0, motor_temp_c=55.0, current_a=40.0,
        rpm=1470.0, load_pct=50.0,
    )
    tags = {t.tag: t for t in plant.plc_tags(latest, "R2", baseline)}
    assert tags["Speed_SP"].value == 1480.0 and tags["Speed_SP"].simulated
    assert tags["Speed_SP"].basis == (
        "simulated: centroid rpm of the inferred operating state R2 (production baseline)"
    )
    assert tags["Speed_Act"].value == 1470.0 and tags["Speed_Act"].source == "rpm"
    assert tags["Run_Bit"].value is True
    assert all(t.value is None for t in plant.plc_tags(None, None, None))
    assert plant.plc_tags(latest, None, baseline)[-1].value is None


# --------------------------------------------------------------------------------------
# Rewind
# --------------------------------------------------------------------------------------


_SERIES = [0.1, 0.35, 0.1, 0.1, 0.35, 0.4, 0.45, 0.6, 0.7]


def test_rewind_markers_follow_sustained_rule(client: TestClient) -> None:
    engine = db.get_engine()
    stamps = _predict(engine, "MTR-042", _SERIES, NOW)
    # a same-ts row from another writer: counted once, the last written one wins
    db.insert_prediction("MTR-042", stamps[0], "v9", 0.12, source="api", engine=engine)
    _predict(engine, "MTR-001", [0.1, 0.9, 0.1, 0.1], NOW)  # a single spike

    rw = client.get("/twin/rewind").json()
    assert rw["validated_lead_time_h"] is None and rw["validated_lead_time_basis"]
    assert rw["start"].startswith("2026-09-01T12:00") and rw["end"].startswith("2026-09-01T13:20")
    series = {s["asset_id"]: s for s in rw["assets"]}
    s42 = series["MTR-042"]
    assert s42["n_rows"] == len(_SERIES) and len(s42["points"]) == len(_SERIES)
    assert s42["points"][0][1] == 0.12  # the duplicate ts was collapsed to the last row
    kinds = {m["kind"]: m for m in s42["markers"]}
    assert kinds["early_warning"]["ts"] == stamps[4].isoformat().replace("+00:00", "Z")
    assert kinds["early_warning"]["label"] == "Early warning fired"
    # the critical marker follows the latch rule (three sustained rows), not a single row
    assert "critical_alarm" not in kinds and "predicted_failure" not in kinds
    assert s42["lead_time_h"] is None
    assert s42["lead_time_basis"] == "no work order yet; warning active for 0.7 h"
    assert s42["lead_time_label"] == "this run: first sustained warning -> motor taken offline"
    assert rw["validated_lead_time_label"] == "validated on held-out motors"
    _predict(engine, "MTR-042", [0.8, 0.9, 0.95], stamps[-1] + timedelta(minutes=10))
    s42 = next(x for x in client.get("/twin/rewind").json()["assets"] if x["asset_id"] == "MTR-042")
    kinds = {m["kind"]: m for m in s42["markers"]}
    # the third sustained row >= 0.5 is the 0.8; the run started at the 0.6 (stamps[7])
    assert kinds["critical_alarm"]["ts"] == stamps[7].isoformat().replace("+00:00", "Z")
    assert kinds["critical_alarm"]["label"] == "Critical alarm latched"

    s1 = series["MTR-001"]
    assert "early_warning" not in {m["kind"] for m in s1["markers"]}  # a spike never fires
    assert s1["lead_time_basis"] == "no early warning"

    rw = client.get("/twin/rewind", params={"max_points": 4, "asset_id": "MTR-042"}).json()
    assert [s["asset_id"] for s in rw["assets"]] == ["MTR-042"]
    pts = rw["assets"][0]["points"]
    assert len(pts) == 4 and pts[0][0] == s42["points"][0][0]
    assert rw["assets"][0]["n_rows"] == len(_SERIES) + 3  # the three critical rows added above


def test_rewind_lead_time_with_work_order(client: TestClient) -> None:
    engine = db.get_engine()
    stamps = _predict(engine, "MTR-042", _SERIES, NOW)
    wo_id = _decided_contract(engine, "MTR-042", "maintain_now")
    started = stamps[4] + timedelta(hours=5)
    ev = {"status": "in_progress", "plant_ts": started.isoformat()}
    assert client.post(f"/work-orders/{wo_id}/events", json=ev).status_code == 200
    s = client.get("/twin/rewind", params={"asset_id": "MTR-042"}).json()["assets"][0]
    assert s["lead_time_h"] == 5.0
    assert s["lead_time_basis"] == "first sustained warning -> plant took the motor offline"
    kinds = {m["kind"]: m for m in s["markers"]}
    assert kinds["work_order_started"]["ts"] == started.isoformat().replace("+00:00", "Z")
    assert "work_order_completed" not in kinds
    assert not ({"contract_created", "approved"} & set(kinds))  # wall clock never on the axis
    done = {"status": "completed", "plant_ts": (started + timedelta(hours=3)).isoformat()}
    assert client.post(f"/work-orders/{wo_id}/events", json=done).status_code == 200
    s = client.get("/twin/rewind", params={"asset_id": "MTR-042"}).json()["assets"][0]
    assert "work_order_completed" in {m["kind"] for m in s["markers"]}
    assert s["markers"] == sorted(s["markers"], key=lambda m: m["ts"])


def test_first_sustained_index() -> None:
    assert twin_routes.first_sustained_index([0.1, 0.9, 0.1], 0.3, 3) is None
    assert twin_routes.first_sustained_index([0.3, 0.3, 0.1, 0.3, 0.3, 0.3], 0.3, 3) == 3
    assert twin_routes.first_sustained_index([], 0.3, 3) is None
    assert twin_routes.downsample([], 10) == []


# --------------------------------------------------------------------------------------
# Live bus, events, WebSocket, path latency
# --------------------------------------------------------------------------------------


def test_events_aggregate_beyond_max_batch() -> None:
    bus = live.LiveBus(maxlen=10_000)
    for i in range(700):
        bus.record_ingest("MTR-001", NOW + timedelta(seconds=i))
    for i in range(500):
        bus.record_prediction("MTR-002", NOW + timedelta(seconds=i), 0.4, inference_ms=1.0)
    events, nxt, dropped = bus.events_since(0, max_events=500)
    assert nxt == 1200 and dropped == 0
    assert [(e.asset_id, e.kind, e.count) for e in events] == [
        ("MTR-001", "ingest", 700),
        ("MTR-002", "prediction", 500),
    ]
    assert events[1].seq == 1200 and events[1].failure_probability == 0.4
    assert bus.events_since(nxt)[0] == []
    events, nxt2, _ = bus.events_since(1195)
    assert nxt2 == 1200 and [e.seq for e in events] == [1196, 1197, 1198, 1199, 1200]
    assert all(e.count == 1 for e in events)

    small = live.LiveBus(maxlen=10)
    for i in range(15):
        small.record_ingest("MTR-003", NOW + timedelta(seconds=i))
    events, nxt, dropped = small.events_since(0)
    assert dropped == 5 and nxt == 15 and [e.seq for e in events] == list(range(6, 16))
    assert small.events_since(999)[1] == 15  # a stale cursor resyncs instead of failing

    counters = bus.snapshot_counters()
    assert counters.messages_total == 700 and counters.predictions_total == 500
    assert counters.msgs_per_s == 70.0 and counters.preds_per_s == 50.0
    assert counters.edge_inference_ms_p50 == 1.0 and counters.n_latency_samples == 0


def test_events_route_and_websocket_frame(client: TestClient) -> None:
    rows = [_row("MTR-042", NOW + timedelta(minutes=10 * i)) for i in range(3)]
    assert client.post("/ingest", json=rows).json() == {"inserted": 3}
    batch = client.get("/twin/events", params={"since": 0}).json()
    assert batch["since"] == 0 and batch["next"] == 3 and batch["dropped"] == 0
    assert [e["kind"] for e in batch["events"]] == ["ingest"] * 3
    assert [e["seq"] for e in batch["events"]] == [1, 2, 3]
    assert batch["events"][0]["asset_id"] == "MTR-042" and batch["events"][0]["count"] == 1
    assert batch["live"]["messages_total"] == 3 and batch["live"]["mqtt_connected"] is False
    again = client.get("/twin/events", params={"since": batch["next"]}).json()
    assert again["events"] == [] and again["next"] == 3

    with client.websocket_connect("/ws/twin") as ws:
        frame = ws.receive_json()
    assert set(frame) == {"since", "next", "events", "dropped", "live"}
    assert frame["since"] == 0 and frame["next"] == 3 and len(frame["events"]) == 3
    assert {"server_time", "msgs_per_s", "preds_per_s", "stale_after_s", "edge_up"} <= set(
        frame["live"]
    )


def test_edge_inference_ms_feeds_path_latency(client: TestClient) -> None:
    ts = NOW + timedelta(minutes=10)
    assert client.post("/ingest", json=_row("MTR-042", ts)).status_code == 200
    pred = {
        "asset_id": "MTR-042",
        "ts": ts.strftime("%Y-%m-%dT%H:%M:%SZ"),  # the edge sends a string ts
        "model_version": "v9",
        "failure_probability": 0.42,
        "source": "edge",
        "inference_ms": 4.2,
    }
    r = client.post("/predictions", json=[pred])
    assert r.status_code == 200 and r.json() == {"ok": True, "inserted": 1}
    live_counters = client.get("/twin").json()["live"]
    assert live_counters["edge_inference_ms_p50"] == 4.2
    assert live_counters["edge_inference_ms_p95"] == 4.2
    assert live_counters["n_latency_samples"] == 1
    assert live_counters["path_latency_ms_p50"] is not None
    assert 0.0 <= live_counters["path_latency_ms_p50"] < 60_000
    assert live_counters["predictions_total"] == 1 and live_counters["messages_total"] == 1
    events = client.get("/twin/events").json()["events"]
    p_ev = next(e for e in events if e["kind"] == "prediction")
    assert p_ev["inference_ms"] == 4.2 and p_ev["path_latency_ms"] is not None
    assert p_ev["failure_probability"] == 0.42
    # the DB insert ignored inference_ms and stored the row
    assert db.latest_predictions(engine=db.get_engine())["MTR-042"] == 0.42
    # a prediction for a ts the control plane never ingested has no path latency
    other = {**pred, "ts": (ts + timedelta(hours=1)).isoformat(), "inference_ms": None}
    assert client.post("/predictions", json=other).status_code == 200
    events = client.get("/twin/events").json()["events"]
    assert events[-1]["path_latency_ms"] is None and events[-1]["inference_ms"] is None


def test_twin_asset_live_fields_from_the_bus(client: TestClient) -> None:
    now = time.time()
    for i in range(5):
        live.record_ingest("MTR-003", NOW + timedelta(seconds=i), received_at=now - 10 + i)
    live.record_ingest("MTR-003", NOW - timedelta(hours=1), received_at=now - 120)  # too old
    a = _twin_asset(client, "MTR-003")
    assert a["msgs_last_60s"] == 5 and a["data_state"] == "live"
    assert datetime.fromisoformat(a["last_received_at"].replace("Z", "+00:00")).timestamp() == (
        pytest.approx(now - 6, abs=0.01)
    )


# --------------------------------------------------------------------------------------
# The MES rule: context only, imported by the twin routes alone
# --------------------------------------------------------------------------------------


_MES_PATTERNS = (
    re.compile(r"plant\.(?=\w)"),  # module attribute access, not the prose "the plant. "
    re.compile(r"import plant\b"),
    re.compile(r"apps\.api\.plant"),
    re.compile(r"simulated_mes_context"),
    re.compile(r"MesContext"),
)


def test_mes_feed_is_imported_by_twin_routes_only() -> None:
    files = [
        *sorted((REPO / "agents").rglob("*.py")),
        *sorted((REPO / "edge").glob("*.py")),
        REPO / "apps/api/orchestrator.py",
        REPO / "apps/api/scoring.py",
        REPO / "apps/api/routes/pipeline.py",
        REPO / "apps/api/routes/approvals.py",
    ]
    assert len(files) > 10
    for path in files:
        src = path.read_text(encoding="utf-8")
        for pat in _MES_PATTERNS:
            assert not pat.search(src), f"{path.relative_to(REPO)} touches the MES feed: {pat.pattern}"
    twin_src = (REPO / "apps/api/routes/twin.py").read_text(encoding="utf-8")
    assert "plant.simulated_mes_context" in twin_src
    import_pat = re.compile(r"(from apps\.api import [^\n]*\bplant\b)|(apps\.api\.plant\b)|(^\s*import plant\b)", re.M)
    importers = [
        p for p in (REPO / "apps" / "api").rglob("*.py")
        if import_pat.search(p.read_text(encoding="utf-8"))
    ]
    assert [p.name for p in importers] == ["twin.py"]


def test_analyze_run_and_contract_carry_no_mes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    configure_env(tmp_path, monkeypatch)
    engine = db.init_db()
    seed_small_fleet(engine)
    from apps.api import orchestrator

    install_fake_toolbox(monkeypatch, orchestrator)
    gs0 = orchestrator.start_run("MTR-042", 48.0, 2, engine=engine)
    gs = wait_for_stage(orchestrator, gs0.run_id, {"awaiting_approval", "failed"}, engine)
    assert gs.stage.value == "awaiting_approval", gs.error
    raw = db.load_run(gs0.run_id, engine=engine)
    assert raw is not None
    forbidden = {"sku", "quantity_produced", "mes", "container", "uom"}
    assert not (set(_keys(raw)) & forbidden)
    contract = db.get_decision_contract(gs.contract.contract_id, engine=engine)
    assert contract is not None
    assert not (set(_keys(db.dump_model(contract))) & forbidden)


def test_simulated_mes_context_is_deterministic() -> None:
    ts = datetime(2026, 9, 1, 9, 30, tzinfo=UTC)
    a = plant.simulated_mes_context("Packaging Line 3", ts)
    b = plant.simulated_mes_context("Packaging Line 3", ts)
    assert a is not None and a == b
    assert a.source == "simulated MES feed" and a.uom == "cases" and a.shift == "day"
    assert a.quantity_produced == round((1800 + 90 * 3) * 1.5)
    assert str(a.quantity_produced) in a.basis and "2070 cases/h" in a.basis
    assert plant.simulated_mes_context("Compressor Hall", ts) is None
    assert plant.simulated_mes_context("Packaging Line 3", ts.replace(hour=2)).shift == "night"
    assert plant.simulated_mes_context("Packaging Line 3", ts.replace(hour=20)).shift == "evening"
    # naive timestamps are read as UTC
    assert plant.simulated_mes_context("Packaging Line 3", ts.replace(tzinfo=None)) == a
    skus = {
        plant.simulated_mes_context("Packaging Line 1", ts + timedelta(hours=8 * k)).sku
        for k in range(8)
    }
    assert len(skus) == 8  # the catalogue rotates every shift
    assert plant.simulated_mes_context("L1", ts) is not None


def test_decision_state_unit_rules() -> None:
    assert twin_routes.decision_state(None, None) == DecisionState.none
    assert twin_routes.decision_state(None, ("c", NOW, None)) == DecisionState.contract_pending
    assert twin_routes.decision_state(None, ("c", NOW, "rejected")) == DecisionState.rejected
    assert twin_routes.decision_state(None, ("c", NOW, "approved")) == DecisionState.none


def test_twin_routes_carry_no_ground_truth(client: TestClient) -> None:
    engine = db.get_engine()
    _predict(engine, "MTR-042", _SERIES, NOW)
    for path in ("/twin", "/twin/hud/MTR-042", "/twin/rewind", "/twin/events"):
        body = client.get(path).json()
        _assert_no_ground_truth(body, f"GET {path}")
