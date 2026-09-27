"""Work orders: the handoff from an approved Decision Contract to the plant (02 control plane).

Thor stops at the Human Approval Gate. An approved maintain_now / maintain_later contract is a
work order for the plant's CMMS, and the plant -- not Thor -- takes the motor offline, does the
repair and returns it to service. These routes only *report* that progress: `GET /work-orders`
derives the list from approvals (nothing new is stored), and `POST /work-orders/{id}/events`
records what the plant says happened as an insert-only event. In the demo the "plant" is the
replay's simulated CMMS (`streaming/replay/work_orders.py`).
"""

from __future__ import annotations

from uuid import uuid4

from fastapi import APIRouter, HTTPException

from apps.api import db
from apps.api.schemas import WorkOrder, WorkOrderEvent, WorkOrderEventRequest

router = APIRouter(tags=["work-orders"])


@router.get("/work-orders", response_model=list[WorkOrder])
def get_work_orders(asset_id: str | None = None, active: bool = False) -> list[WorkOrder]:
    """Approved maintenance work orders, newest first. `active=true` hides completed ones."""
    orders = db.list_work_orders(asset_id=asset_id, engine=db.get_engine())
    if active:
        orders = [o for o in orders if o.status != "completed"]
    return orders


@router.post("/work-orders/{contract_id}/events", response_model=WorkOrder)
def post_work_order_event(contract_id: str, req: WorkOrderEventRequest) -> WorkOrder:
    """Record the plant's progress report for one work order (insert-only).

    404 when the contract is not an approved work order (pending, rejected, run_to_failure or
    unknown); 409 when the report would move the status backwards or repeat a terminal one, so
    a replaying reporter can post idempotently.
    """
    engine = db.get_engine()
    current = next(
        (o for o in db.list_work_orders(engine=engine) if o.contract_id == contract_id), None
    )
    if current is None:
        raise HTTPException(
            status_code=404, detail=f"no approved work order for contract {contract_id}"
        )
    if current.status == "completed":
        raise HTTPException(status_code=409, detail=f"work order {contract_id} already completed")
    if req.status == "in_progress" and current.status != "scheduled":
        raise HTTPException(status_code=409, detail=f"work order {contract_id} already in progress")
    db.insert_work_order_event(
        WorkOrderEvent(
            event_id=uuid4().hex,
            contract_id=contract_id,
            asset_id=current.asset_id,
            status=req.status,
            plant_ts=req.plant_ts,
            recorded_at=db.now_utc(),
            source=req.source,
            note=req.note,
        ),
        engine=engine,
    )
    updated = next(o for o in db.list_work_orders(engine=engine) if o.contract_id == contract_id)
    return updated
