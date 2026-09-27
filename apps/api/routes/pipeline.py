"""Pipeline routes (02 -> 03): start runs, poll/stream GraphState, what-if sandbox."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

from apps.api import db, orchestrator
from apps.api.graph_state import public_state, run_summary
from apps.api.schemas import (
    PipelineRunRequest,
    PipelineRunResponse,
    PipelineStage,
    WhatIfResult,
)

log = logging.getLogger("thor.routes.pipeline")
router = APIRouter(tags=["pipeline"])

WS_TERMINAL: frozenset[str] = frozenset(
    {
        PipelineStage.awaiting_approval.value,
        PipelineStage.approved.value,
        PipelineStage.rejected.value,
        PipelineStage.failed.value,
    }
)


class WhatIfRequest(BaseModel):
    run_id: str
    scenario: dict[str, float] = Field(default_factory=dict)


@router.post("/pipeline/run", response_model=PipelineRunResponse)
def start_pipeline(req: PipelineRunRequest) -> PipelineRunResponse:
    """Queue a full AutoML run for one asset; returns immediately with the run id."""
    engine = db.get_engine()
    if not any(a.asset_id == req.asset_id for a in db.list_assets(engine)):
        raise HTTPException(status_code=404, detail=f"unknown asset {req.asset_id}")
    gs = orchestrator.start_run(req.asset_id, req.horizon_h, req.n_trials, engine=engine)
    return PipelineRunResponse(run_id=gs.run_id, stage=gs.stage)


@router.get("/pipeline")
def list_pipeline_runs(asset_id: str | None = None) -> list[dict[str, Any]]:
    """Run summaries (newest first), optionally filtered by asset."""
    out = []
    for r in db.list_runs(asset_id=asset_id, engine=db.get_engine()):
        s = run_summary(r["state"])
        s["created_at"] = r["created_at"]
        s["updated_at"] = r["updated_at"]
        out.append(s)
    return out


@router.get("/pipeline/{run_id}")
def get_pipeline_run(run_id: str) -> dict[str, Any]:
    """The persisted GraphState of a run as JSON (bulk per-row arrays stripped, see public_state)."""
    raw = db.load_run(run_id, engine=db.get_engine())
    if raw is None:
        raise HTTPException(status_code=404, detail=f"unknown run {run_id}")
    return public_state(raw)


@router.post("/whatif", response_model=WhatIfResult)
def whatif(req: WhatIfRequest) -> WhatIfResult:
    """Re-score the run's champion under feature overrides (P2 reliability sandbox)."""
    try:
        return orchestrator.whatif(req.run_id, req.scenario, engine=db.get_engine())
    except KeyError as e:
        raise HTTPException(status_code=404, detail=f"unknown run {req.run_id}") from e
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"what-if unavailable: {e}") from e


@router.websocket("/ws/runs/{run_id}")
async def ws_run(websocket: WebSocket, run_id: str) -> None:
    """Push the run's GraphState JSON every second until a terminal stage (sent once more)."""
    await websocket.accept()
    engine = db.get_engine()
    try:
        while True:
            raw = await asyncio.to_thread(db.load_run, run_id, engine)
            if raw is None:
                await websocket.send_json({"error": f"unknown run {run_id}"})
                break
            await websocket.send_json(public_state(raw))
            if raw.get("stage") in WS_TERMINAL:
                break
            await asyncio.sleep(1.0)
    except WebSocketDisconnect:
        return
    except Exception as e:
        log.warning("ws /ws/runs/%s closed: %s", run_id, e)
    try:
        await websocket.close()
    except Exception:
        pass
