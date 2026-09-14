"""Helpers around the typed LangGraph state (`apps.api.schemas.GraphState`).

Pure functions only: no LLM, no database. The orchestrator uses these to create run ids,
append progress events and (de)serialize state for `db.save_run` / `db.load_run`.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from apps.api.schemas import GraphState, PipelineStage, StageEvent


def new_run_id() -> str:
    """Return a new 16-char hex run id (fits the `pipeline_runs.run_id` String(32) column)."""
    return uuid4().hex[:16]


def append_event(
    state: GraphState,
    stage: PipelineStage | str,
    message: str,
    tool: str | None = None,
    payload: dict[str, Any] | None = None,
) -> StageEvent:
    """Append a `StageEvent` (UTC now) to `state.events` in place and return it.

    Inputs: the mutable state, the stage the event belongs to, a human-readable message, an
    optional tool name and an optional JSON-safe payload. Output: the appended event.
    """
    event = StageEvent(
        ts=datetime.now(UTC),
        stage=PipelineStage(stage),
        message=message,
        tool=tool,
        payload=payload,
    )
    state.events.append(event)
    return event


def state_to_json(state: GraphState) -> dict[str, Any]:
    """Serialize a `GraphState` into a JSON-safe dict (datetimes -> ISO strings)."""
    return json.loads(state.model_dump_json())


def state_from_json(data: dict[str, Any]) -> GraphState:
    """Rebuild a `GraphState` from the dict produced by `state_to_json` / stored in the DB."""
    return GraphState.model_validate(data)


def run_summary(state: dict[str, Any] | GraphState) -> dict[str, Any]:
    """Compact run summary for list endpoints: ids, stage, error, contract id, event count."""
    gs = state if isinstance(state, GraphState) else state_from_json(state)
    last_ts = gs.events[-1].ts.isoformat() if gs.events else None
    return {
        "run_id": gs.run_id,
        "asset_id": gs.asset_id,
        "stage": gs.stage.value,
        "error": gs.error,
        "contract_id": gs.contract.contract_id if gs.contract else None,
        "model_version": gs.validation.model_version if gs.validation else None,
        "llm_calls": gs.llm_calls,
        "n_events": len(gs.events),
        "last_event_ts": last_ts,
    }
