"""Copilot (Bolt) chat route -- thin HTTP wrapper over `orchestrator.copilot_reply`."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from apps.api import db, orchestrator
from apps.api.schemas import CopilotRequest, CopilotResponse

router = APIRouter(tags=["copilot"])


@router.post("/copilot/chat", response_model=CopilotResponse)
def copilot_chat(req: CopilotRequest) -> CopilotResponse:
    """Answer a chat turn; the reply only contains numbers returned by deterministic tools."""
    if not req.messages:
        raise HTTPException(status_code=422, detail="messages must not be empty")
    return orchestrator.copilot_reply(req, engine=db.get_engine())
