"""Pipeline routes (02 -> 03): start runs, poll/stream GraphState, what-if."""

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
        PipelineStage.trained.value,
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
    """Queue a run for one asset; returns immediately with the run id.

    mode=analyze (default, "Run analysis"): reuse the production model to explain, cost and
    draft a Decision Contract -- 409 when no model has been promoted yet. mode=train: an
    explicit AutoML run that ends at `trained`; promotion is then decided in ModelOps.
    """
    engine = db.get_engine()
    if not any(a.asset_id == req.asset_id for a in db.list_assets(engine)):
        raise HTTPException(status_code=404, detail=f"unknown asset {req.asset_id}")
    if req.mode == "analyze":
        try:
            bundle = orchestrator.tool_production_bundle(engine)
        except Exception as e:
            raise HTTPException(status_code=409, detail=str(e)) from e
        if bundle is None:
            raise HTTPException(
                status_code=409,
                detail="no production model: train one in AutoML Studio and approve its "
                "promotion in ModelOps first",
            )
    gs = orchestrator.start_run(
        req.asset_id, req.horizon_h, req.n_trials, engine=engine, mode=req.mode
    )
    return PipelineRunResponse(run_id=gs.run_id, stage=gs.stage, mode=gs.mode)


class ScoreRequest(BaseModel):
    asset_ids: list[str] | None = Field(default=None, description="None = every asset")


@router.post("/pipeline/score")
def score_fleet(req: ScoreRequest | None = None) -> dict[str, Any]:
    """Score the latest window of every (or the given) asset with the production model.

    Deterministic inference only -- no training, no decision. Writes `predictions` rows
    with source "api". 409 when there is no production model.
    """
    from apps.api.scoring import score_assets

    try:
        res = score_assets(req.asset_ids if req else None, engine=db.get_engine())
    except LookupError as e:
        raise HTTPException(status_code=409, detail=str(e)) from e
    return res.model_dump(mode="json")


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
    """Re-score the run's champion under feature overrides (P2 what-if)."""
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
