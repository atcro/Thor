"""Replay control routes (see apps/api/replay_control.py).

    GET  /replay          -> ReplayState (polled by the replay container and the topbar)
    POST /replay/pause    -> ReplayState (recovery tool: the stream holds its position)
    POST /replay/resume   -> ReplayState
"""

from __future__ import annotations

from fastapi import APIRouter

from apps.api import db, replay_control
from apps.api.schemas import ReplayState

router = APIRouter(tags=["replay"])


@router.get("/replay", response_model=ReplayState)
def get_replay() -> ReplayState:
    """Whether the recorded plant history is released (a production model exists), paused,
    and the current plant time."""
    return replay_control.state(db.get_engine())


@router.post("/replay/pause", response_model=ReplayState)
def pause_replay() -> ReplayState:
    """Hold the replay where it is (the presenter's recovery switch)."""
    replay_control.set_paused(True)
    return replay_control.state(db.get_engine())


@router.post("/replay/resume", response_model=ReplayState)
def resume_replay() -> ReplayState:
    """Let the replay continue from where it was paused."""
    replay_control.set_paused(False)
    return replay_control.state(db.get_engine())
