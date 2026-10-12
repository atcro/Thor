"""Replay gate: the recorded plant history streams only after the human sign-off.

The replay container polls `GET /replay` (apps/api/routes/replay.py). It streams when
`released` is True -- which is exactly when a production model exists, i.e. a named approver
promoted one in ModelOps -- and `paused` is False. Pause / resume is the presenter's recovery
switch; it lives in this process's memory, so an API restart resumes.

Nothing here starts a timer at start-up: a clean `docker compose up` seeds the history and then
holds the replay until a promotion is approved.
"""

from __future__ import annotations

import threading
from datetime import UTC, datetime

from sqlalchemy import func, select
from sqlalchemy.engine import Engine

from apps.api import db
from apps.api.schemas import ModelStage, ReplayState

_lock = threading.Lock()
_paused = False
_paused_at: datetime | None = None


def set_paused(paused: bool) -> None:
    """Pause (True) or resume (False) the replay. Output: none."""
    global _paused, _paused_at
    with _lock:
        _paused = bool(paused)
        _paused_at = datetime.now(UTC) if paused else None


def is_paused() -> bool:
    with _lock:
        return _paused


def production_version(engine: Engine | None = None) -> str | None:
    """Version of the production model, or None when nothing has been promoted."""
    engine = engine or db.get_engine()
    r = db.model_registry
    with engine.connect() as conn:
        row = conn.execute(
            select(r.c.version)
            .where(r.c.stage == ModelStage.production.value)
            .order_by(r.c.registered_at.desc(), r.c.id.desc())
        ).first()
    return str(row[0]) if row else None


def state(engine: Engine | None = None) -> ReplayState:
    """The ReplayState the replay container and the topbar read.

    Inputs: engine. Output: released iff a production model exists; paused from memory; plant
    time = newest stored telemetry ts; seeded_until = newest non-sandbox row at start (the
    same number while the replay is held).
    """
    engine = engine or db.get_engine()
    version = production_version(engine)
    with engine.connect() as conn:
        raw_ts = conn.execute(select(func.max(db.telemetry.c.ts))).scalar()
    plant_time = db._as_utc(raw_ts) if raw_ts is not None else None
    with _lock:
        paused, paused_at = _paused, _paused_at
    return ReplayState(
        released=version is not None,
        paused=paused,
        production_model_version=version,
        plant_time=plant_time,
        seeded_until=plant_time if version is None else None,
        paused_at=paused_at,
    )
