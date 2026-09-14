"""ModelOps routes (08 read side): registry, promotions, drift reports and drift checks."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from sqlalchemy import select, update
from sqlalchemy.engine import Engine

from apps.api import db
from apps.api.schemas import DriftReport, PromotionRequest, RegisteredModel

log = logging.getLogger("thor.routes.models")
router = APIRouter(tags=["models"])


class DriftCheckRequest(BaseModel):
    asset_id: str


def list_models_db(engine: Engine | None = None) -> list[RegisteredModel]:
    """Registry rows (newest first) read straight from `model_registry`."""
    engine = engine or db.get_engine()
    with engine.connect() as conn:
        rows = (
            conn.execute(select(db.model_registry).order_by(db.model_registry.c.registered_at.desc()))
            .mappings()
            .all()
        )
    out = []
    for r in rows:
        d = {k: v for k, v in dict(r).items() if k not in ("id", "payload")}
        out.append(RegisteredModel.model_validate(d))
    return out


def list_promotions_db(engine: Engine | None = None) -> list[PromotionRequest]:
    """Promotion requests (newest first); `status` comes from the column, not the payload."""
    engine = engine or db.get_engine()
    with engine.connect() as conn:
        rows = (
            conn.execute(
                select(db.promotion_requests).order_by(db.promotion_requests.c.created_at.desc())
            )
            .mappings()
            .all()
        )
    out = []
    for r in rows:
        payload = dict(r["payload"] or {})
        payload["status"] = r["status"]
        payload.setdefault("promotion_id", r["promotion_id"])
        try:
            out.append(PromotionRequest.model_validate(payload))
        except Exception as e:
            log.warning("skipping malformed promotion row %s: %s", r["promotion_id"], e)
    return out


def set_promotion_status(promotion_id: str, status: str, engine: Engine | None = None) -> None:
    """Fallback status update used when `lifecycle.reject_promotion` is unavailable."""
    engine = engine or db.get_engine()
    with engine.begin() as conn:
        conn.execute(
            update(db.promotion_requests)
            .where(db.promotion_requests.c.promotion_id == promotion_id)
            .values(status=status)
        )


def list_drift_db(asset_id: str | None = None, engine: Engine | None = None) -> list[DriftReport]:
    engine = engine or db.get_engine()
    q = select(db.drift_reports.c.payload).order_by(db.drift_reports.c.created_at.desc())
    if asset_id:
        q = q.where(db.drift_reports.c.asset_id == asset_id)
    with engine.connect() as conn:
        rows = conn.execute(q).all()
    return [DriftReport.model_validate(r[0]) for r in rows]


@router.get("/models", response_model=list[RegisteredModel])
def get_models() -> list[RegisteredModel]:
    """Model registry (candidate -> validated -> shadow -> production -> archived)."""
    return list_models_db(db.get_engine())


@router.get("/models/promotions", response_model=list[PromotionRequest])
def get_promotions() -> list[PromotionRequest]:
    """All promotion requests, newest first."""
    return list_promotions_db(db.get_engine())


@router.get("/models/drift", response_model=list[DriftReport])
def get_drift(asset_id: str | None = None) -> list[DriftReport]:
    """Stored drift reports, optionally filtered by asset."""
    return list_drift_db(asset_id, db.get_engine())


@router.post("/models/drift/check", response_model=DriftReport)
def check_drift(req: DriftCheckRequest) -> DriftReport:
    """Run PSI drift detection for one asset: first 70% of its feature rows vs the last 30%."""
    engine = db.get_engine()
    try:
        from agents.ml_architect.features import build_feature_pipeline
        from agents.mlops.drift import detect_drift
    except ImportError as e:
        raise HTTPException(status_code=503, detail=f"drift toolbox unavailable: {e}") from e
    df = db.read_telemetry(engine=engine)
    if df.empty or req.asset_id not in set(df["asset_id"]):
        raise HTTPException(status_code=404, detail=f"no telemetry for {req.asset_id}")
    try:
        features_df, spec = build_feature_pipeline(df, None)
        sub = features_df[features_df["asset_id"] == req.asset_id].sort_values("ts")
        if len(sub) < 20:
            raise HTTPException(status_code=409, detail="not enough feature rows for drift check")
        cut = int(len(sub) * 0.7)
        train, recent = sub.iloc[:cut].copy(), sub.iloc[cut:].copy()
        prod = next((m for m in list_models_db(engine) if m.stage == "production"), None)
        version = prod.version if prod else "none"
        return detect_drift(train, recent, spec.features, req.asset_id, version, engine=engine)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"drift check failed: {e}") from e


def registry_row_payload(name: str, version: str, engine: Engine | None = None) -> dict[str, Any]:
    """Raw `payload` JSON of one registry row (empty dict if none)."""
    engine = engine or db.get_engine()
    with engine.connect() as conn:
        row = conn.execute(
            select(db.model_registry.c.payload).where(
                db.model_registry.c.name == name, db.model_registry.c.version == version
            )
        ).first()
    return dict(row[0]) if row and row[0] else {}
