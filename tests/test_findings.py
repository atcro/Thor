"""Repair findings ("What did the technician find?"), alarm trust, Bolt's plant memory and
training labels from confirmed findings (Phase A.3, CLAUDE.md sections 7 and 9).

Hard rules proven here: with zero findings the training frame and the split are IDENTICAL to
before; a finding is insert-only (a correction supersedes, both rows stay); a confirmed
finding labels the motor's pre-repair rows up to the intervention and moves it into TRAIN
only; NFF / Other change nothing in training; sandbox findings are ignored everywhere; Bolt
cites a finding with its source and name and never invents one.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from agents.ml_architect.labels import adopt_into_train, apply_field_labels, label_rows_for_event
from agents.ml_architect.split import make_asset_split
from apps.api import db, findings, orchestrator
from apps.api.schemas import (
    Approval,
    AssetSplit,
    CopilotMessage,
    CopilotRequest,
    FieldFailureEvent,
    GraphState,
    PipelineStage,
    RepairFindingRequest,
    TelemetryRow,
    WorkOrderEvent,
)
from tests.test_orchestrator import (
    NOW,
    configure_env,
    install_fake_toolbox,
    make_bundle,
    make_contract,
    make_dq,
    seed_small_fleet,
)

EVENT = NOW + timedelta(hours=5)  # the motor goes offline for the repair (plant time)
BACK = NOW + timedelta(hours=8)  # back in service


@pytest.fixture()
def engine(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    """Temp SQLite with the small fleet (same as tests/test_orchestrator.py::engine)."""
    configure_env(tmp_path, monkeypatch)
    eng = db.init_db()
    seed_small_fleet(eng)
    return eng


def _live_rows(engine: Any, asset_id: str, start: datetime, end: datetime, degrading: bool) -> int:
    """Replayed rows: observable sensors only (no ground truth), 10-minute steps in [start, end)."""
    rows: list[TelemetryRow] = []
    ts = start
    i = 0
    while ts < end:
        rows.append(
            TelemetryRow(
                asset_id=asset_id,
                ts=ts,
                vibration_rms=3.5 + (0.01 * i if degrading else -2.5),
                vibration_kurtosis=5.0 if degrading else 3.0,
                vibration_crest=4.0,
                bearing_temp_c=80.0 if degrading else 60.0,
                motor_temp_c=55.0,
                current_a=40.0,
                rpm=1480.0,
                load_pct=55.0,
                regime=None,
                health=None,
                failure_within_h=None,
            )
        )
        ts += timedelta(minutes=10)
        i += 1
    return db.insert_telemetry(rows, engine=engine)


def _work_order(
    engine: Any,
    asset_id: str = "MTR-042",
    started: datetime | None = EVENT,
    completed: datetime | None = BACK,
    approver: str = "engineer@plant",
) -> str:
    """An approved maintain_now contract with the plant's progress reports; returns its id."""
    dc = make_contract(make_bundle(asset_id), f"run-{uuid4().hex[:6]}", "template text", "template")
    db.insert_decision_contract(dc, engine=engine)
    db.insert_approval(
        Approval(
            approval_id=uuid4().hex,
            contract_id=dc.contract_id,
            decision="approved",
            approver=approver,
            decided_at=datetime.now(UTC),
        ),
        engine=engine,
    )
    for status, ts in (("in_progress", started), ("completed", completed)):
        if ts is None:
            continue
        db.insert_work_order_event(
            WorkOrderEvent(
                event_id=uuid4().hex,
                contract_id=dc.contract_id,
                asset_id=asset_id,
                status=status,  # type: ignore[arg-type]
                plant_ts=ts,
                recorded_at=datetime.now(UTC),
                note="demo",
            ),
            engine=engine,
        )
    return dc.contract_id


def _finding(engine: Any, contract_id: str, kind: str, by: str = "tech@plant", **kw: Any) -> Any:
    return findings.record_finding(
        RepairFindingRequest(contract_id=contract_id, finding=kind, recorded_by=by, **kw),
        engine=engine,
    )


# --------------------------------------------------------------------------------------
# Insert-only findings over HTTP, alarm trust
# --------------------------------------------------------------------------------------


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    configure_env(tmp_path, monkeypatch)
    install_fake_toolbox(monkeypatch, orchestrator)
    from apps.api.main import app

    with TestClient(app) as c:
        seed_small_fleet(db.get_engine())
        yield c


def test_findings_are_person_entered_insert_only_and_counted(client: TestClient) -> None:
    engine = db.get_engine()
    body = {"contract_id": "dc_nope", "finding": "confirmed_bearing_damage", "recorded_by": "tech"}
    assert client.post("/findings", json=body).status_code == 404
    assert client.get("/findings/trust").json()["acted_on"] == 0

    scheduled = _work_order(engine, started=None, completed=None)
    body["contract_id"] = scheduled
    r = client.post("/findings", json=body)
    assert r.status_code == 409 and "completed" in r.json()["detail"]
    trust = client.get("/findings/trust").json()
    assert trust == {
        "acted_on": 1,
        "confirmed": 0,
        "no_fault_found": 0,
        "other": 0,
        "awaiting_finding": 1,
        "completed_without_finding": 0,
        "min_for_rate": 5,
    }

    done = _work_order(engine, asset_id="MTR-003")
    body["contract_id"] = done
    assert client.post("/findings", json={**body, "recorded_by": "   "}).status_code == 422
    r = client.post("/findings", json={**body, "note": "outer race spalled"})
    assert r.status_code == 201
    first = r.json()
    assert first["asset_id"] == "MTR-003" and first["supersedes"] is None
    assert first["finding"] == "confirmed_bearing_damage" and first["recorded_by"] == "tech"
    assert first["plant_ts"].startswith(BACK.strftime("%Y-%m-%dT%H:%M"))
    # the work order now carries the finding (Fleet card, Asset 360, Bolt read it from there)
    wo = next(o for o in client.get("/work-orders").json() if o["contract_id"] == done)
    assert wo["finding"]["finding_id"] == first["finding_id"]
    trust = client.get("/findings/trust").json()
    assert (trust["acted_on"], trust["confirmed"], trust["awaiting_finding"]) == (2, 1, 1)

    # a correction is a NEW row that supersedes the first; both stay visible
    r = client.post("/findings", json={**body, "finding": "no_fault_found", "recorded_by": "lead"})
    assert r.status_code == 201
    second = r.json()
    assert (
        second["supersedes"] == first["finding_id"] and second["finding_id"] != first["finding_id"]
    )
    rows = client.get(f"/findings?contract_id={done}").json()
    assert [x["finding_id"] for x in rows] == [first["finding_id"], second["finding_id"]]
    trust = client.get("/findings/trust").json()
    assert (trust["confirmed"], trust["no_fault_found"]) == (0, 1)
    # superseding anything but the current finding is refused
    r = client.post("/findings", json={**body, "supersedes": first["finding_id"]})
    assert r.status_code == 409
    # there is no update or delete route
    assert client.delete(f"/findings/{first['finding_id']}").status_code in (404, 405)
    assert client.put("/findings", json=body).status_code in (404, 405)

    # sandbox-tagged rows are excluded from trust: the last real finding (NFF) still stands
    r = client.post("/findings", json={**body, "finding": "other", "sandbox": True})
    assert r.status_code == 201
    trust = client.get("/findings/trust").json()
    assert (trust["other"], trust["no_fault_found"], trust["awaiting_finding"]) == (0, 1, 1)
    assert client.get("/findings/ledger").json() == {"queued": [], "rule": findings.LABEL_RULE}


# --------------------------------------------------------------------------------------
# Training labels
# --------------------------------------------------------------------------------------


def _profile(engine: Any, monkeypatch: pytest.MonkeyPatch, run_id: str) -> tuple[GraphState, Any]:
    monkeypatch.setattr(
        orchestrator, "tool_profile_dataset", lambda df, a, fit_assets=None: make_dq(a)
    )
    h = orchestrator._handle_for(run_id, "MTR-042", engine)
    gs = GraphState(run_id=run_id, asset_id="MTR-042", stage=PipelineStage.profiling, mode="train")
    orchestrator._profile(gs, h, engine)
    return gs, h


def test_zero_findings_training_is_identical(engine: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """The hero demo must not change: no finding -> the same frame object, the same split."""
    _live_rows(engine, "MTR-042", NOW, EVENT, degrading=True)  # replayed rows, no outcome
    gs, h = _profile(engine, monkeypatch, "r-zero")
    df = h.cache["df"]
    assert df is h.cache["df_history"], "with no findings the training frame IS the history"
    assert df["health"].notna().all(), "replayed rows (no outcome) never train"
    assert len(df[df["asset_id"] == "MTR-042"]) == 30
    expected = make_asset_split(df, 0, ["MTR-042"])
    assert gs.split == expected
    assert orchestrator.tool_adopt_into_train(expected, []) is expected
    assert [s.source for s in gs.label_sources] == ["simulator_history"] * len(gs.label_sources)
    assert "MTR-042" in gs.split.test and gs.split.forced_test == ["MTR-042"]
    assert all("work order" not in n for n in gs.split.notes)


def test_confirmed_finding_labels_pre_repair_rows_and_joins_train(
    engine: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    _live_rows(engine, "MTR-042", NOW, EVENT, degrading=True)
    _live_rows(engine, "MTR-042", BACK, BACK + timedelta(hours=3), degrading=False)
    contract_id = _work_order(engine)
    f = _finding(engine, contract_id, "confirmed_bearing_damage", by="tech@plant", note="spalled")
    events = findings.field_failure_events(engine)
    assert [(e.asset_id, e.event_ts, e.labelled_from) for e in events] == [("MTR-042", EVENT, None)]
    ledger = findings.label_ledger(engine)
    assert [q.finding_id for q in ledger.queued] == [f.finding_id]
    assert ledger.queued[0].recorded_by == "tech@plant" and ledger.queued[0].event_ts == EVENT

    before = make_asset_split(
        h_hist := db.read_telemetry(engine=engine).query("health == health"), 0, ["MTR-042"]
    )
    gs, h = _profile(engine, monkeypatch, "r-one")
    df = h.cache["df"]
    assert df is not h.cache["df_history"]
    own = df[df["asset_id"] == "MTR-042"].sort_values("ts")
    # every pre-repair row (seeded + replayed) up to the intervention, none after it
    assert len(own) == 30 + 30 and own["ts"].max() < pd.Timestamp(EVENT)
    hours = (pd.Timestamp(EVENT) - own["ts"]).dt.total_seconds() / 3600.0
    assert own["failure_within_h"].to_numpy() == pytest.approx(hours.to_numpy())
    assert own["failure_within_h"].min() <= 1.0  # counts as a completed failure
    assert own["failure_within_h"].max() == pytest.approx(10.0)  # first seeded row, NOW - 5 h
    # the simulator's values for that motor are gone: the finding defines the labels
    sim = h_hist[h_hist["asset_id"] == "MTR-042"]["failure_within_h"]
    assert not own["failure_within_h"].head(30).to_numpy().tolist() == sim.to_numpy().tolist()
    # the other motors are untouched
    for a in ("MTR-001", "MTR-002", "MTR-003"):
        pd.testing.assert_frame_equal(
            df[df["asset_id"] == a].reset_index(drop=True),
            h.cache["df_history"][h.cache["df_history"]["asset_id"] == a].reset_index(drop=True),
        )
    # field-confirmed motors join TRAIN only; the rest of the split is as before
    s = gs.split
    assert "MTR-042" in s.train and "MTR-042" not in s.test and "MTR-042" not in s.forced_test
    assert s.test == [a for a in before.test if a != "MTR-042"]
    assert s.validation == before.validation and s.calibration == before.calibration
    assert sorted(s.train) == sorted([*before.train, "MTR-042"])
    assert "MTR-042" in s.positives["train"] and "MTR-042" not in s.positives["test"]
    assert any(
        "moved from test to train" in n and contract_id in n and "tech@plant" in n for n in s.notes
    )
    src = next(x for x in gs.label_sources if x.asset_id == "MTR-042")
    assert src.source == "work_order_finding" and src.group == "train"
    assert src.contract_id == contract_id and src.recorded_by == "tech@plant" and src.n_rows == 60
    assert src.event_ts == EVENT
    ev = next(e for e in gs.events if e.tool == "make_asset_split")
    assert "field-confirmed" in ev.message and ev.payload["label_sources"][0]


def test_nff_other_and_sandbox_findings_change_nothing_in_training(
    engine: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    _live_rows(engine, "MTR-042", NOW, EVENT, degrading=True)
    c1 = _work_order(engine)
    _finding(engine, c1, "no_fault_found", note="bearing fine; sensor bracket loose")
    c2 = _work_order(engine, asset_id="MTR-003")
    _finding(engine, c2, "other", note="coupling misaligned")
    c3 = _work_order(engine, asset_id="MTR-001")
    _finding(engine, c3, "confirmed_bearing_damage", sandbox=True)
    assert findings.field_failure_events(engine) == []
    trust = findings.alarm_trust(engine)
    assert (trust.acted_on, trust.confirmed, trust.no_fault_found, trust.other) == (3, 0, 1, 1)
    assert trust.awaiting_finding == 1  # the sandbox finding does not count
    gs, h = _profile(engine, monkeypatch, "r-nff")
    assert h.cache["df"] is h.cache["df_history"]
    assert gs.split == make_asset_split(h.cache["df"], 0, ["MTR-042"])
    # a later confirmed correction on c1 supersedes the NFF and now trains
    _finding(engine, c1, "confirmed_bearing_damage", by="lead@plant")
    assert [e.contract_id for e in findings.field_failure_events(engine)] == [c1]
    assert findings.alarm_trust(engine).confirmed == 1


def test_label_window_stops_at_previous_repair_and_needs_rows_near_the_event() -> None:
    ts = pd.date_range("2026-01-01", periods=48, freq="10min", tz="UTC")
    rows = pd.DataFrame(
        {"asset_id": "MTR-007", "ts": ts, "vibration_rms": 1.0, "failure_within_h": 99.0}
    )
    ev = FieldFailureEvent(
        asset_id="MTR-007",
        contract_id="dc_x",
        finding_id="rf_x",
        recorded_by="tech",
        event_ts=ts[40].to_pydatetime(),
        labelled_from=ts[10].to_pydatetime(),
    )
    out = label_rows_for_event(rows, ev)
    assert list(out.index) == list(
        range(11, 40)
    )  # strictly after the previous repair, before the event
    assert out["failure_within_h"].iloc[-1] == pytest.approx(10 / 60)
    assert out["failure_within_h"].iloc[0] == pytest.approx(29 * 10 / 60)
    # a gap of more than one hour before the outage cannot stand in for a completed failure
    gap = rows[rows["ts"] < ts[30]]
    frame, applied = apply_field_labels(gap, [ev], {"MTR-007": gap})
    assert frame is gap and applied == []
    frame, applied = apply_field_labels(rows, [], {})
    assert frame is rows and applied == []


def test_adopt_into_train_moves_only_the_confirmed_motor() -> None:
    split = AssetSplit(
        seed=0,
        train=["MTR-001", "MTR-002"],
        validation=["MTR-003"],
        calibration=["MTR-004"],
        test=["MTR-005", "MTR-042"],
        forced_test=["MTR-042"],
        positives={"train": ["MTR-001"], "validation": [], "calibration": [], "test": ["MTR-005"]},
        notes=["n"],
    )
    ev = FieldFailureEvent(
        asset_id="MTR-042", contract_id="dc_1", finding_id="rf_1", recorded_by="tech", event_ts=NOW
    )
    out = adopt_into_train(split, [ev])
    assert out.train == ["MTR-001", "MTR-002", "MTR-042"] and out.test == ["MTR-005"]
    assert out.forced_test == [] and out.positives["train"] == ["MTR-001", "MTR-042"]
    assert out.positives["test"] == ["MTR-005"] and out.validation == ["MTR-003"]
    assert out.notes[-1].startswith(
        "MTR-042 trains as a completed failure confirmed by work order dc_1"
    )
    assert adopt_into_train(split, []) is split
    # a motor already in train just gets the note
    ev2 = FieldFailureEvent(
        asset_id="MTR-002", contract_id="dc_2", finding_id="rf_2", recorded_by="tech", event_ts=NOW
    )
    out2 = adopt_into_train(split, [ev2])
    assert out2.train == split.train and "moved" not in out2.notes[-1]


# --------------------------------------------------------------------------------------
# Bolt remembers this plant's repairs
# --------------------------------------------------------------------------------------


def test_bolt_cites_plant_findings_with_source_and_name_and_never_invents_one(engine: Any) -> None:
    def ask(text: str, asset_id: str | None = None) -> Any:
        return orchestrator.copilot_reply(
            CopilotRequest(messages=[CopilotMessage(role="user", content=text)], asset_id=asset_id),
            engine=engine,
        )

    # no finding anywhere: Bolt says so, in template mode, from the same tool
    r = ask("Have we seen this before?", asset_id="MTR-042")
    assert r.source == "template" and r.tool_calls[-1]["name"] == "search_field_history"
    assert r.tool_calls[-1]["args"]["asset_id"] == "MTR-042"
    assert "no confirmed finding yet" in r.reply and "work order" not in r.reply.split("\n")[0]
    res = orchestrator.copilot_search_field_history(
        "bearing vibration", engine=engine, asset_id="MTR-042"
    )
    assert res["this_plant"] == [] and res["n_this_plant"] == 0
    assert res["source"].startswith("public work orders, not this plant")

    contract_id = _work_order(engine)
    f = _finding(
        engine, contract_id, "confirmed_bearing_damage", by="J. Ortega", note="outer race spalled"
    )
    res = orchestrator.copilot_search_field_history(
        "Have we seen this before?", engine=engine, asset_id="MTR-042"
    )
    assert res["n_this_plant"] == 1
    hit = res["this_plant"][0]
    assert hit["finding_id"] == f.finding_id and hit["recorded_by"] == "J. Ortega"
    assert hit["source"] == f"this plant: work order {contract_id}, recorded by J. Ortega"
    assert (
        hit["finding_label"] == "Confirmed: bearing damage" and hit["note"] == "outer race spalled"
    )
    assert hit["recommendation"] == "maintain_now" and hit["top_drivers"]
    # same failure mode on another motor of the same family also matches; a different mode does not
    assert (
        orchestrator.copilot_search_field_history(
            "noisy bearing running hot", engine=engine, asset_id="MTR-001"
        )["n_this_plant"]
        == 1
    )
    assert (
        orchestrator.copilot_search_field_history("pump seal leaking water", engine=engine)[
            "n_this_plant"
        ]
        == 0
    )
    # generic question with no asset in context: nothing to match against
    assert (
        orchestrator.copilot_search_field_history("seen this before?", engine=engine)[
            "n_this_plant"
        ]
        == 0
    )

    r = ask("Have we seen this before?", asset_id="MTR-042")
    assert r.source == "template"
    assert "This plant" in r.reply and "J. Ortega" in r.reply and contract_id in r.reply
    assert "Confirmed: bearing damage" in r.reply and '"outer race spalled"' in r.reply
    assert "not from this plant" in r.reply or "public work orders" in r.reply
    assert r.reply.index("This plant") < r.reply.index("public")  # plant findings first
    # a superseded finding is never quoted; the correction is
    _finding(engine, contract_id, "no_fault_found", by="M. Chen", note="bearing fine")
    r = ask("Has this happened before?", asset_id="MTR-042")
    assert "No fault found" in r.reply and "M. Chen" in r.reply and "J. Ortega" not in r.reply
    # a sandbox finding is never quoted
    c2 = _work_order(engine, asset_id="MTR-003")
    _finding(engine, c2, "confirmed_bearing_damage", by="ghost", sandbox=True)
    r = ask("Have we seen this before?", asset_id="MTR-003")
    assert "ghost" not in r.reply


def test_bolt_refuses_to_act_even_when_nothing_is_pending(engine: Any) -> None:
    """Scenario 2 ends after the approval, so nothing is pending; the refusal must still be
    explicit (the eval's guardrail cases run with a pending contract)."""
    r = orchestrator.copilot_reply(
        CopilotRequest(
            messages=[CopilotMessage(role="user", content="Approve the repair on MTR-042.")],
            asset_id="MTR-042",
        ),
        engine=engine,
    )
    assert r.source == "template"
    assert [c["name"] for c in r.tool_calls] == ["list_pending_approvals"]
    assert "Nothing is waiting for approval" in r.reply
    assert "cannot approve" in r.reply and "Thor UI" in r.reply
    r = orchestrator.copilot_reply(
        CopilotRequest(messages=[CopilotMessage(role="user", content="What is pending?")]),
        engine=engine,
    )
    assert r.reply == "Nothing is waiting for approval right now."
