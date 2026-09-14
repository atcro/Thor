"""Contract tests: the reference Decision Contract and GraphState round-trip through Pydantic."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from apps.api import db
from apps.api.schemas import (
    Approval,
    DecisionContract,
    GraphState,
    PipelineStage,
    StageEvent,
)

REPO = Path(__file__).resolve().parents[1]


def test_example_contract_validates() -> None:
    raw = json.loads((REPO / "docs" / "decision-contracts" / "example.json").read_text())
    dc = DecisionContract.model_validate(raw)
    assert dc.asset_id == "MTR-042"
    assert dc.cost_comparison.recommended == dc.recommendation
    assert len(dc.top_features) == 6


def test_graph_state_roundtrip() -> None:
    state = GraphState(run_id="r_test", asset_id="MTR-042")
    state.events.append(
        StageEvent(ts=datetime.now(UTC), stage=PipelineStage.profiling, message="start")
    )
    dumped = json.loads(state.model_dump_json())
    again = GraphState.model_validate(dumped)
    assert again.stage == PipelineStage.queued
    assert again.events[0].stage == PipelineStage.profiling


def test_contract_and_approval_are_insert_only() -> None:
    engine = db.get_engine("sqlite://")
    db.init_db(engine)
    raw = json.loads((REPO / "docs" / "decision-contracts" / "example.json").read_text())
    dc = DecisionContract.model_validate(raw)
    db.insert_decision_contract(dc, engine=engine)
    assert db.contract_status(dc.contract_id, engine=engine) == "pending"
    db.insert_approval(
        Approval(
            approval_id="ap_1",
            contract_id=dc.contract_id,
            decision="approved",
            approver="engineer@plant",
            decided_at=datetime.now(UTC),
        ),
        engine=engine,
    )
    assert db.contract_status(dc.contract_id, engine=engine) == "approved"
    stored = db.get_decision_contract(dc.contract_id, engine=engine)
    assert stored is not None and stored.evidence_hash == dc.evidence_hash
    assert not hasattr(db, "update_decision_contract")
