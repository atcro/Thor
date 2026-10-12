"""3-D Digital Twin sandbox: overrides the replay applies to its recorded stream.

The sandbox never injects readings into the control plane directly. It records what the
presenter asked for as insert-only `sandbox_events` rows; the replay container polls
`GET /sandbox` and applies the active overrides to the rows it publishes
(streaming/replay/sandbox.py), so sandbox readings travel the real path -- replay -> MQTT ->
edge -> control plane -- and are scored exactly like plant data. Every such row is tagged
`sandbox=True`; the tag follows it into predictions, alarms, contracts and work orders.

Events (kind): `set` (sliders: rpm / load / bearing temperature setpoints for one motor),
`fault` (drive-end bearing fault on one motor, the simulator's own signature ramping over
`fault_ramp_h` plant hours), `shake` (random rpm / load / temperature within normal operating
ranges across motors), `reset` (every override removed; sandbox-only alarms cleared; sandbox
predictions up to this point retired from display). The active override of a motor is derived
from its newest `set` / `fault` / `shake` event after the newest `reset` (a per-asset `clear`
event, written when a sandbox work order starts, removes that one motor's override).

No simulator ground truth enters here: an override carries observable setpoints only.
"""

from __future__ import annotations

import logging
import random
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from sqlalchemy import func, select
from sqlalchemy.engine import Engine

from apps.api import alarms, db
from apps.api.schemas import SandboxEvent, SandboxOverride, SandboxState

log = logging.getLogger("thor.sandbox")

#: Plant hours for a sandbox fault to reach full severity ((t / ramp) ** 2, then held). At
#: 900x one plant hour is 4 s: a 2.5 h ramp reaches full severity in 10 s and the alarm's
#: three sustained rows follow within ~15 s (measured in docs/finals/demo-walkthrough.md).
FAULT_RAMP_H_DEFAULT: float = 2.5

#: Normal operating ranges for "Shake up operations" (the simulator's three states:
#: rpm / load centroid and spread). Nothing outside what a healthy motor does.
OPERATING_STATES: tuple[tuple[float, float, float, float], ...] = (
    (600.0, 15.0, 10.0, 2.5),  # idle / startup: rpm, rpm sd, load, load sd
    (1480.0, 8.0, 55.0, 4.0),  # running
    (1500.0, 8.0, 85.0, 4.0),  # heavy load
)


def _now() -> datetime:
    return datetime.now(UTC)


def _plant_now(engine: Engine) -> datetime:
    with engine.connect() as conn:
        raw = conn.execute(select(func.max(db.telemetry.c.ts))).scalar()
    return db._as_utc(raw) if raw is not None else _now()


def _insert(ev: SandboxEvent, engine: Engine) -> SandboxEvent:
    with engine.begin() as conn:
        conn.execute(db.sandbox_events.insert().values(**ev.model_dump()))
    return ev


def _event(kind: str, asset_id: str | None, payload: dict[str, Any], note: str = "") -> SandboxEvent:
    return SandboxEvent(
        event_id=uuid4().hex,
        kind=kind,  # type: ignore[arg-type]
        asset_id=asset_id,
        payload=payload,
        recorded_at=_now(),
        note=note,
    )


def list_events(engine: Engine | None = None, limit: int = 200) -> list[SandboxEvent]:
    """Newest `limit` sandbox events, oldest first."""
    engine = engine or db.get_engine()
    e = db.sandbox_events
    q = select(e).order_by(e.c.recorded_at.desc(), e.c.id.desc()).limit(limit)
    with engine.connect() as conn:
        rows = conn.execute(q).mappings().all()
    out = []
    for r in rows:
        d = dict(r)
        d.pop("id", None)
        d["recorded_at"] = db._as_utc(d["recorded_at"])
        d["payload"] = dict(d.get("payload") or {})
        out.append(SandboxEvent(**d))
    out.reverse()
    return out


def last_reset(engine: Engine) -> SandboxEvent | None:
    for ev in reversed(list_events(engine, limit=1000)):
        if ev.kind == "reset":
            return ev
    return None


def active_overrides(engine: Engine | None = None) -> list[SandboxOverride]:
    """Derive the active override per motor from the events after the newest reset.

    Output: one SandboxOverride per motor with an active `set`, `fault` or `shake` effect
    (later events of the same motor merge onto earlier ones: a slider change keeps a running
    fault, a fault keeps the sliders); a `clear` event for a motor drops it.
    """
    engine = engine or db.get_engine()
    events = list_events(engine, limit=1000)
    cut = max((i for i, ev in enumerate(events) if ev.kind == "reset"), default=-1)
    state: dict[str, dict[str, Any]] = {}
    for ev in events[cut + 1 :]:
        if ev.kind == "clear" and ev.asset_id:  # type: ignore[comparison-overlap]
            state.pop(ev.asset_id, None)
            continue
        targets: list[tuple[str, dict[str, Any]]]
        if ev.kind == "shake":
            targets = [(aid, dict(p)) for aid, p in dict(ev.payload.get("assets") or {}).items()]
        elif ev.asset_id:
            targets = [(ev.asset_id, dict(ev.payload))]
        else:
            continue
        for aid, p in targets:
            cur = state.setdefault(aid, {"asset_id": aid, "set_at": ev.recorded_at})
            cur["set_at"] = ev.recorded_at
            for k in ("rpm", "load_pct", "bearing_temp_c"):
                if k in p:
                    cur[k] = p[k]
            if ev.kind == "fault":
                cur["fault"] = True
                cur["fault_since"] = p.get("fault_since")
                cur["fault_ramp_h"] = p.get("fault_ramp_h", FAULT_RAMP_H_DEFAULT)
            if ev.note:
                cur["note"] = ev.note
    out = []
    for cur in state.values():
        if cur.get("fault_since"):
            cur["fault_since"] = db._as_utc(datetime.fromisoformat(str(cur["fault_since"])))
        out.append(SandboxOverride(**cur))
    return sorted(out, key=lambda o: o.asset_id)


def state(engine: Engine | None = None) -> SandboxState:
    """GET /sandbox payload."""
    engine = engine or db.get_engine()
    overrides = active_overrides(engine)
    reset = last_reset(engine)
    return SandboxState(
        active=bool(overrides),
        overrides=overrides,
        retired_prediction_max_id=db.sandbox_retired_max_id(engine),
        last_reset_at=reset.recorded_at if reset else None,
        events=list_events(engine, limit=50),
    )


# --------------------------------------------------------------------------------------
# Actions (each one insert-only)
# --------------------------------------------------------------------------------------


def _known(asset_id: str, engine: Engine) -> str:
    aid = asset_id.strip().upper()
    if not any(a.asset_id == aid for a in db.list_assets(engine)):
        raise KeyError(aid)
    return aid


def set_sliders(
    asset_id: str,
    rpm: float | None,
    load_pct: float | None,
    bearing_temp_c: float | None,
    engine: Engine | None = None,
) -> SandboxEvent:
    """Record slider setpoints for one motor (None leaves that signal recorded)."""
    engine = engine or db.get_engine()
    aid = _known(asset_id, engine)
    payload: dict[str, Any] = {}
    if rpm is not None:
        payload["rpm"] = float(max(0.0, min(3600.0, rpm)))
    if load_pct is not None:
        payload["load_pct"] = float(max(0.0, min(100.0, load_pct)))
    if bearing_temp_c is not None:
        payload["bearing_temp_c"] = float(max(-20.0, min(200.0, bearing_temp_c)))
    return _insert(_event("set", aid, payload, "sliders"), engine)


def inject_fault(
    asset_id: str, ramp_h: float = FAULT_RAMP_H_DEFAULT, engine: Engine | None = None
) -> SandboxEvent:
    """Troubleshoot <motor>: a drive-end bearing fault ramping to full severity over `ramp_h`
    plant hours from the current plant time, with the simulator's own signature."""
    engine = engine or db.get_engine()
    aid = _known(asset_id, engine)
    payload = {
        "fault_since": _plant_now(engine).isoformat(),
        "fault_ramp_h": float(max(0.1, ramp_h)),
        "signature": "drive-end bearing (data/simulator/generate.py::bearing_fault_delta)",
    }
    return _insert(_event("fault", aid, payload, f"Troubleshoot {aid}"), engine)


def random_fault(engine: Engine | None = None, seed: int | None = None) -> SandboxEvent:
    """The same fault on a random motor that is neither offline nor already faulted."""
    engine = engine or db.get_engine()
    offline = {aid for aid, wo in db.work_order_by_asset(engine).items() if wo.status == "in_progress"}
    faulted = {o.asset_id for o in active_overrides(engine) if o.fault}
    candidates = sorted(a.asset_id for a in db.list_assets(engine) if a.asset_id not in offline | faulted)
    if not candidates:
        raise LookupError("every motor is offline or already faulted")
    rng = random.Random(seed)
    return inject_fault(rng.choice(candidates), engine=engine)


def shake(engine: Engine | None = None, seed: int | None = None) -> SandboxEvent:
    """Shake up operations: a random operating state (rpm, load) per motor within the normal
    ranges; temperature follows load through the simulator's healthy relation on the replay
    side. Motors offline for a work order are left alone."""
    engine = engine or db.get_engine()
    offline = {aid for aid, wo in db.work_order_by_asset(engine).items() if wo.status == "in_progress"}
    rng = random.Random(seed)
    assets: dict[str, dict[str, float]] = {}
    for a in db.list_assets(engine):
        if a.asset_id in offline:
            continue
        rpm_c, rpm_sd, load_c, load_sd = rng.choice(OPERATING_STATES)
        assets[a.asset_id] = {
            "rpm": round(max(0.0, rng.gauss(rpm_c, rpm_sd)), 1),
            "load_pct": round(min(100.0, max(0.0, rng.gauss(load_c, load_sd))), 1),
        }
    return _insert(_event("shake", None, {"assets": assets}, "Shake up operations"), engine)


def clear_asset(asset_id: str, note: str, engine: Engine | None = None) -> SandboxEvent:
    """Drop one motor's override (written when its sandbox work order starts)."""
    engine = engine or db.get_engine()
    return _insert(_event("clear", asset_id, {}, note), engine)


def reset(engine: Engine | None = None) -> tuple[SandboxEvent, list[str]]:
    """Reset sandbox: remove every override, retire the sandbox predictions made so far and
    clear every alarm that sandbox readings alone raised (recorded as sandbox alarm events).

    Output: (the reset event, asset ids whose alarm was cleared).
    """
    engine = engine or db.get_engine()
    with engine.connect() as conn:
        max_id = int(conn.execute(select(func.max(db.predictions.c.id))).scalar() or 0)
    ev = _insert(
        _event("reset", None, {"retired_prediction_max_id": max_id}, "Reset sandbox"), engine
    )
    cleared: list[str] = []
    plant_ts = _plant_now(engine)
    for aid in alarms.sandbox_raised_assets(engine):
        if alarms.clear_alarm(aid, "sandbox_reset", plant_ts, engine=engine, sandbox=True):
            cleared.append(aid)
    log.info("sandbox reset: overrides removed, %d sandbox alarms cleared", len(cleared))
    return ev, cleared
