"""Sandbox routes (3-D Digital Twin). See apps/api/sandbox.py.

    GET  /sandbox                 -> SandboxState (polled by the replay and the twin)
    POST /sandbox/set             -> {asset_id, rpm?, load_pct?, bearing_temp_c?}
    POST /sandbox/fault           -> {asset_id, ramp_h?}      "Troubleshoot <motor>"
    POST /sandbox/random-fault    -> {}                        "Random fault"
    POST /sandbox/shake           -> {}                        "Shake up operations"
    POST /sandbox/reset           -> {}                        "Reset sandbox"

Every POST writes one insert-only sandbox event and returns the new SandboxState. Readings
are produced by the replay, scored by the edge and the control plane like plant data, and
tagged sandbox all the way through.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from apps.api import db, sandbox
from apps.api.schemas import SandboxState

router = APIRouter(tags=["sandbox"])


class SlidersRequest(BaseModel):
    asset_id: str
    rpm: float | None = Field(default=None, ge=0.0, le=3600.0)
    load_pct: float | None = Field(default=None, ge=0.0, le=100.0)
    bearing_temp_c: float | None = Field(default=None, ge=-20.0, le=200.0)


class FaultRequest(BaseModel):
    asset_id: str
    ramp_h: float = Field(default=sandbox.FAULT_RAMP_H_DEFAULT, gt=0.0, le=48.0)


@router.get("/sandbox", response_model=SandboxState)
def get_sandbox() -> SandboxState:
    """Active overrides, the newest reset marker and the recent sandbox events."""
    return sandbox.state(db.get_engine())


@router.post("/sandbox/set", response_model=SandboxState)
def post_sliders(req: SlidersRequest) -> SandboxState:
    """Apply rpm / load / bearing-temperature setpoints to one motor's recorded stream."""
    engine = db.get_engine()
    try:
        sandbox.set_sliders(req.asset_id, req.rpm, req.load_pct, req.bearing_temp_c, engine)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=f"unknown asset {e.args[0]}") from e
    return sandbox.state(engine)


@router.post("/sandbox/fault", response_model=SandboxState)
def post_fault(req: FaultRequest) -> SandboxState:
    """Troubleshoot <motor>: inject the simulator's drive-end bearing fault, ramping fast."""
    engine = db.get_engine()
    try:
        sandbox.inject_fault(req.asset_id, req.ramp_h, engine)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=f"unknown asset {e.args[0]}") from e
    return sandbox.state(engine)


@router.post("/sandbox/random-fault", response_model=SandboxState)
def post_random_fault() -> SandboxState:
    """The same fault on a random motor that is not offline or already faulted."""
    engine = db.get_engine()
    try:
        sandbox.random_fault(engine)
    except LookupError as e:
        raise HTTPException(status_code=409, detail=str(e)) from e
    return sandbox.state(engine)


@router.post("/sandbox/shake", response_model=SandboxState)
def post_shake() -> SandboxState:
    """Shake up operations: random rpm / load / temperature within normal ranges."""
    engine = db.get_engine()
    sandbox.shake(engine)
    return sandbox.state(engine)


@router.post("/sandbox/reset")
def post_reset() -> dict[str, Any]:
    """Reset sandbox: remove every override and clear sandbox-only alarms."""
    engine = db.get_engine()
    ev, cleared = sandbox.reset(engine)
    out = sandbox.state(engine).model_dump(mode="json")
    out["reset_event"] = ev.model_dump(mode="json")
    out["alarms_cleared"] = cleared
    return out
