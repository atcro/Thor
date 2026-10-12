"""Alarm state machine: the ONE rule that turns predictions into a status (CLAUDE.md §15).

Every surface that shows a motor's status -- Fleet cards and sprites, the 3-D Digital Twin, the
X-Ray HUD, the rewind markers and Bolt -- reads the alarm level computed here from the stored
per-sample predictions. Nothing in the browser recomputes it.

Rules (constants below; thresholds from `apps.api.schemas`):

* **On-delay.** A motor enters `warning` only after `ALARM_RAISE_ROWS` consecutive predictions
  at or above `EARLY_WARNING_THRESHOLD` (the same sustained rule validation uses to measure
  lead time, so the alarm on stage is the alarm the model card quotes), and `critical` only
  after `ALARM_RAISE_ROWS` consecutive predictions at or above `CLASSIFICATION_THRESHOLD`.
  A single spike never changes a colour.
* **Latch.** Once `critical`, a motor stays `critical` until a completed work order clears it
  (`clear_alarm(reason="work_order_completed")`) -- or, for an alarm raised by sandbox readings
  alone, until the sandbox is reset. A model whose probability later drops back does not
  silently cancel an alarm an engineer has not acted on.
* **Off-delay for warnings.** A `warning` returns to `healthy` only after `WARNING_CLEAR_ROWS`
  consecutive predictions below `EARLY_WARNING_THRESHOLD`. Why 6: twice the on-delay (one
  plant hour of 10-minute samples), so an alarm needs more evidence to clear than it needed
  to raise and can never chatter between amber and green on a borderline probability.

Transitions are recorded as insert-only `alarm_events` rows (plant time of the triggering
prediction, the probability and model that triggered it, and whether that prediction came from
sandbox readings). The current level of a motor is its newest event. `update_alarms()` is
called after every prediction insert; it walks only the rows that arrived since the last call,
so the cost per batch is a handful of tiny queries.

No ground truth, no LLM: inputs are prediction rows only.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from uuid import uuid4

from sqlalchemy import func, select
from sqlalchemy.engine import Engine

from apps.api import db
from apps.api.schemas import (
    CLASSIFICATION_THRESHOLD,
    EARLY_WARNING_SUSTAINED_ROWS,
    EARLY_WARNING_THRESHOLD,
    AlarmEvent,
    AlarmLevel,
    DataState,
    Thresholds,
    TwinStatus,
    WorkOrder,
)

log = logging.getLogger("thor.alarms")

#: Consecutive predictions at or above a threshold that raise (or escalate) an alarm.
ALARM_RAISE_ROWS: int = EARLY_WARNING_SUSTAINED_ROWS
#: Consecutive predictions below the early-warning threshold that clear a WARNING. Twice the
#: on-delay: more evidence to clear than to raise, never chattering (see module docstring).
WARNING_CLEAR_ROWS: int = 2 * EARLY_WARNING_SUSTAINED_ROWS

_LOCK = threading.Lock()


@dataclass
class _Runs:
    """Trailing run lengths of one motor's prediction series (the state machine's memory)."""

    warn: int = 0  # consecutive rows >= early_warning
    crit: int = 0  # consecutive rows >= classification
    clear: int = 0  # consecutive rows < early_warning


@dataclass
class _Cursor:
    level: AlarmLevel = AlarmLevel.healthy
    last_id: int = 0
    runs: _Runs = field(default_factory=_Runs)


#: asset_id -> in-memory cursor (rebuilt from the database on first use after a restart).
_cursors: dict[str, _Cursor] = {}


def reset() -> None:
    """Forget the in-memory cursors (tests)."""
    with _LOCK:
        _cursors.clear()


def step(level: AlarmLevel, runs: _Runs, p: float, thresholds: Thresholds | None = None) -> AlarmLevel:
    """Advance the state machine by one prediction.

    Inputs: current level, trailing runs (mutated in place), the probability, thresholds.
    Output: the new level (equal to `level` when nothing changed). Pure apart from `runs`.
    """
    t = thresholds or Thresholds()
    runs.warn = runs.warn + 1 if p >= t.early_warning else 0
    runs.crit = runs.crit + 1 if p >= t.classification else 0
    runs.clear = runs.clear + 1 if p < t.early_warning else 0
    if level == AlarmLevel.critical:
        return level  # latched: only a work order (or a sandbox reset) clears it
    if runs.crit >= ALARM_RAISE_ROWS:
        return AlarmLevel.critical
    if level == AlarmLevel.healthy and runs.warn >= ALARM_RAISE_ROWS:
        return AlarmLevel.warning
    if level == AlarmLevel.warning and runs.clear >= WARNING_CLEAR_ROWS:
        return AlarmLevel.healthy
    return level


def evaluate(
    probabilities: list[float], thresholds: Thresholds | None = None
) -> list[tuple[int, AlarmLevel]]:
    """Run the state machine over a whole series from `healthy` (tests, rewind markers).

    Input: probabilities oldest first. Output: [(row index, new level)] for every transition.
    """
    level = AlarmLevel.healthy
    runs = _Runs()
    out: list[tuple[int, AlarmLevel]] = []
    for i, p in enumerate(probabilities):
        new = step(level, runs, float(p), thresholds)
        if new != level:
            out.append((i, new))
            level = new
    return out


# --------------------------------------------------------------------------------------
# Storage
# --------------------------------------------------------------------------------------


def _row_to_event(r: Any) -> AlarmEvent:
    d = dict(r)
    d.pop("id", None)
    for k in ("plant_ts", "recorded_at"):
        if isinstance(d.get(k), datetime) and d[k].tzinfo is None:
            d[k] = db._as_utc(d[k])
    return AlarmEvent(**d)


def latest_alarms(engine: Engine | None = None) -> dict[str, AlarmEvent]:
    """asset_id -> its newest alarm event (one window-function query). Motors without an
    event are healthy by definition and absent from the dict."""
    engine = engine or db.get_engine()
    a = db.alarm_events
    rn = (
        func.row_number()
        .over(partition_by=a.c.asset_id, order_by=(a.c.recorded_at.desc(), a.c.id.desc()))
        .label("rn")
    )
    ranked = select(a, rn).subquery("ranked")
    q = select(ranked).where(ranked.c.rn == 1)
    with engine.connect() as conn:
        rows = conn.execute(q).mappings().all()
    out: dict[str, AlarmEvent] = {}
    for r in rows:
        d = {k: v for k, v in dict(r).items() if k != "rn"}
        out[str(d["asset_id"])] = _row_to_event(d)
    return out


def list_alarm_events(asset_id: str | None = None, engine: Engine | None = None) -> list[AlarmEvent]:
    """Alarm events oldest first (optionally one asset)."""
    engine = engine or db.get_engine()
    a = db.alarm_events
    q = select(a).order_by(a.c.recorded_at, a.c.id)
    if asset_id:
        q = q.where(a.c.asset_id == asset_id)
    with engine.connect() as conn:
        rows = conn.execute(q).mappings().all()
    return [_row_to_event(r) for r in rows]


def _insert(ev: AlarmEvent, engine: Engine) -> None:
    with engine.begin() as conn:
        conn.execute(db.alarm_events.insert().values(**ev.model_dump()))


def _new_event(
    asset_id: str,
    level: AlarmLevel,
    reason: str,
    plant_ts: datetime,
    probability: float | None,
    model_version: str | None,
    sandbox: bool,
) -> AlarmEvent:
    return AlarmEvent(
        event_id=uuid4().hex,
        asset_id=asset_id,
        level=level,
        reason=reason,
        plant_ts=plant_ts,
        recorded_at=db.now_utc(),
        probability=probability,
        model_version=model_version,
        sandbox=sandbox,
    )


def _rows_after(engine: Engine, asset_id: str, last_id: int, limit: int | None = None) -> list[Any]:
    p = db.predictions
    q = (
        select(p.c.id, p.c.ts, p.c.failure_probability, p.c.model_version, p.c.sandbox)
        .where(p.c.asset_id == asset_id, p.c.id > last_id)
        .order_by(p.c.ts, p.c.id)
    )
    if limit:
        q = q.limit(limit)
    with engine.connect() as conn:
        return conn.execute(q).all()


#: After a restart a motor with no alarm event is walked from at most this many stored rows.
REBUILD_MAX_ROWS: int = 5_000


def _rebuild_cursor(engine: Engine, asset_id: str, newest: AlarmEvent | None) -> _Cursor:
    """After a restart (or on first sight of a motor): level from its newest event, cursor at
    the last prediction row that event could have seen (plant time <= the event's), so the
    rows since then are walked normally by the caller. A motor with no event and a long
    history is walked from its last REBUILD_MAX_ROWS rows only."""
    cur = _Cursor()
    p = db.predictions
    with engine.connect() as conn:
        if newest is not None:
            cur.level = newest.level
            last = conn.execute(
                select(func.max(p.c.id)).where(p.c.asset_id == asset_id, p.c.ts <= newest.plant_ts)
            ).scalar()
            cur.last_id = int(last or 0)
        else:
            n = conn.execute(select(func.count()).where(p.c.asset_id == asset_id)).scalar() or 0
            if int(n) > REBUILD_MAX_ROWS:
                skip = (
                    select(p.c.id)
                    .where(p.c.asset_id == asset_id)
                    .order_by(p.c.ts.desc(), p.c.id.desc())
                    .offset(REBUILD_MAX_ROWS)
                    .limit(1)
                )
                first_kept = conn.execute(skip).scalar()
                cur.last_id = int(first_kept or 0)
    return cur


def update_alarms(engine: Engine | None = None, asset_ids: list[str] | None = None) -> list[AlarmEvent]:
    """Walk the predictions that arrived since the last call and record every transition.

    Inputs: engine; asset ids to evaluate (None = every asset that has a prediction row).
    Output: the alarm events inserted (empty when nothing changed). Thread-safe; called from
    POST /predictions (the edge sink) and from control-plane scoring.
    """
    engine = engine or db.get_engine()
    inserted: list[AlarmEvent] = []
    with _LOCK:
        newest_all: dict[str, AlarmEvent] | None = None
        if asset_ids is None:
            with engine.connect() as conn:
                asset_ids = [
                    str(r[0]) for r in conn.execute(select(db.predictions.c.asset_id).distinct()).all()
                ]
        for asset_id in asset_ids:
            cur = _cursors.get(asset_id)
            if cur is None:
                if newest_all is None:
                    newest_all = latest_alarms(engine)
                cur = _rebuild_cursor(engine, asset_id, newest_all.get(asset_id))
                _cursors[asset_id] = cur
            rows = _rows_after(engine, asset_id, cur.last_id)
            for row_id, ts, prob, version, sandbox in rows:
                new = step(cur.level, cur.runs, float(prob))
                cur.last_id = max(cur.last_id, int(row_id))
                if new != cur.level:
                    reason = {
                        AlarmLevel.warning: f"{ALARM_RAISE_ROWS} consecutive predictions >= {EARLY_WARNING_THRESHOLD:g}",
                        AlarmLevel.critical: f"{ALARM_RAISE_ROWS} consecutive predictions >= {CLASSIFICATION_THRESHOLD:g} (latched)",
                        AlarmLevel.healthy: f"{WARNING_CLEAR_ROWS} consecutive predictions < {EARLY_WARNING_THRESHOLD:g}",
                    }[new]
                    ev = _new_event(
                        asset_id, new, reason, db._as_utc(ts), float(prob), str(version), bool(sandbox)
                    )
                    _insert(ev, engine)
                    inserted.append(ev)
                    cur.level = new
    for ev in inserted:
        log.info("alarm %s -> %s at plant time %s (%s)", ev.asset_id, ev.level, ev.plant_ts, ev.reason)
    return inserted


def clear_alarm(
    asset_id: str,
    reason: str,
    plant_ts: datetime,
    engine: Engine | None = None,
    sandbox: bool = False,
) -> AlarmEvent | None:
    """Record a `healthy` event for `asset_id` if it is currently alarmed.

    Inputs: asset id, reason ("work_order_completed" | "sandbox_reset"), plant time of the
    clearing event, the engine, whether the clear is a sandbox event. Output: the event
    inserted, or None when the motor was already healthy. Resets the trailing runs so the
    next alarm needs fresh sustained evidence.
    """
    engine = engine or db.get_engine()
    with _LOCK:
        newest = latest_alarms(engine).get(asset_id)
        if newest is None or newest.level == AlarmLevel.healthy:
            cur = _cursors.get(asset_id)
            if cur is not None:
                cur.level = AlarmLevel.healthy
            return None
        ev = _new_event(asset_id, AlarmLevel.healthy, reason, plant_ts, None, None, sandbox)
        _insert(ev, engine)
        cur = _cursors.get(asset_id)
        if cur is not None:
            cur.level = AlarmLevel.healthy
            cur.runs = _Runs()
    log.info("alarm %s cleared: %s", asset_id, reason)
    return ev


def sandbox_raised_assets(engine: Engine | None = None) -> list[str]:
    """Assets whose CURRENT alarm (warning or critical) was raised by a sandbox prediction."""
    return sorted(
        aid
        for aid, ev in latest_alarms(engine).items()
        if ev.level != AlarmLevel.healthy and ev.sandbox
    )


# --------------------------------------------------------------------------------------
# The one status rule
# --------------------------------------------------------------------------------------


def asset_status(
    alarm: AlarmLevel | None,
    p_fail: float | None,
    data_state: DataState,
    work_order: WorkOrder | None,
) -> TwinStatus:
    """Status glyph of one motor, used by /fleet, /twin, the HUD and Bolt alike.

    Precedence: offline (the plant has the motor out for a work order in progress) >
    no_prediction (no approved-model prediction at all) > no_data (it had predictions but its
    messages stopped; `DataState.stale`) > critical / warning (the latched alarm level) >
    healthy. `DataState.none` (nothing received since the control plane started, e.g. the
    replay is still held) does not hide a prediction: the stream has not started, it has not
    stopped. Inputs: alarm level (None = healthy), displayed probability, data state, newest
    work order. Output: TwinStatus.
    """
    if work_order is not None and work_order.status == "in_progress":
        return TwinStatus.offline
    if p_fail is None:
        return TwinStatus.no_prediction
    if data_state == DataState.stale:
        return TwinStatus.no_data
    if alarm == AlarmLevel.critical:
        return TwinStatus.critical
    if alarm == AlarmLevel.warning:
        return TwinStatus.warning
    return TwinStatus.healthy
