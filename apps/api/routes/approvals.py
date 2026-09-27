"""Human Approval Gate (09): present_evidence() and record_decision() over HTTP.

Decision contracts and approvals are insert-only. A second decision on the same contract or
promotion is refused with 409. Nothing downstream (finalize, promote, deploy) runs without an
approvals row.
"""

from __future__ import annotations

import logging
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, HTTPException
from sqlalchemy.engine import Engine

from apps.api import db, orchestrator
from apps.api.graph_state import state_from_json
from apps.api.routes.models import list_promotions_db, registry_row_payload, set_promotion_status
from apps.api.schemas import Approval, DecisionRequest, EvidenceBundle
from apps.api.settings import get_settings

log = logging.getLogger("thor.routes.approvals")
router = APIRouter(tags=["approvals"])


def _new_approval(
    req: DecisionRequest, contract_id: str | None = None, promotion_id: str | None = None
) -> Approval:
    return Approval(
        approval_id=uuid4().hex,
        contract_id=contract_id,
        promotion_id=promotion_id,
        decision=req.decision,
        approver=req.approver,
        note=req.note,
        decided_at=datetime.now(UTC),
    )


@router.get("/approvals/pending")
def pending_approvals() -> dict[str, Any]:
    """Decision contracts and promotion requests still waiting for a human decision."""
    engine = db.get_engine()
    contracts = db.list_pending_decision_contracts(engine=engine)
    promotions = [p for p in list_promotions_db(engine) if p.status == "pending"]
    return {"contracts": contracts, "promotions": promotions}


def present_evidence(contract_id: str, engine: Engine | None = None) -> dict[str, Any]:
    """09: the contract, the run's EvidenceBundle (if the run state still has it) and status."""
    engine = engine or db.get_engine()
    dc = db.get_decision_contract(contract_id, engine=engine)
    if dc is None:
        raise HTTPException(status_code=404, detail=f"unknown contract {contract_id}")
    evidence: EvidenceBundle | None = None
    raw = db.load_run(dc.run_id, engine=engine)
    if raw:
        try:
            evidence = state_from_json(raw).evidence
        except Exception as e:
            log.warning("could not parse run state for %s: %s", dc.run_id, e)
    return {
        "contract": dc,
        "evidence": evidence,
        "status": db.contract_status(contract_id, engine=engine),
        "approvals": db.approvals_for(contract_id=contract_id, engine=engine),
    }


@router.get("/approvals/{contract_id}")
def get_approval(contract_id: str) -> dict[str, Any]:
    """present_evidence(): everything an engineer needs to approve or reject."""
    return present_evidence(contract_id, db.get_engine())


def record_decision(contract_id: str, req: DecisionRequest, engine: Engine | None = None) -> Approval:
    """09: insert the immutable Approval row for a contract, then resume the run in the background.

    Raises 404 for unknown contracts and 409 when a decision already exists.
    """
    engine = engine or db.get_engine()
    dc = db.get_decision_contract(contract_id, engine=engine)
    if dc is None:
        raise HTTPException(status_code=404, detail=f"unknown contract {contract_id}")
    if db.approvals_for(contract_id=contract_id, engine=engine):
        raise HTTPException(status_code=409, detail=f"contract {contract_id} already decided")
    approval = _new_approval(req, contract_id=contract_id)
    db.insert_approval(approval, engine=engine)

    def _resume() -> None:
        try:
            orchestrator.resume(dc.run_id, approval, engine=engine)
        except Exception as e:
            log.warning("resume(%s) after decision failed: %s", dc.run_id, e)

    threading.Thread(target=_resume, name=f"thor-resume-{dc.run_id}", daemon=True).start()
    return approval


@router.post("/approvals/{contract_id}/decision", response_model=Approval)
def decide_contract(contract_id: str, req: DecisionRequest) -> Approval:
    """record_decision(): the only way a maintenance recommendation becomes an action."""
    return record_decision(contract_id, req, db.get_engine())


def _features_for_version(version: str, engine: Engine) -> list[str]:
    """FeatureSpec.features of the run that produced `version`, else the canonical list."""
    for r in db.list_runs(engine=engine):
        try:
            gs = state_from_json(r["state"])
        except Exception:
            continue
        if gs.validation and gs.validation.model_version == version and gs.candidates:
            return list(gs.candidates.features.features)
    try:
        from agents.ml_architect.features import FEATURE_NAMES

        return list(FEATURE_NAMES)
    except Exception:
        return []


def _window_rows_for_version(version: str, engine: Engine) -> int:
    for r in db.list_runs(engine=engine):
        try:
            gs = state_from_json(r["state"])
        except Exception:
            continue
        if gs.validation and gs.validation.model_version == version and gs.candidates:
            return int(gs.candidates.features.window_rows)
    return 12


@router.post("/promotions/{promotion_id}/decision")
def decide_promotion(promotion_id: str, req: DecisionRequest) -> dict[str, Any]:
    """Record a promotion decision; on approval call lifecycle.promote + deploy_edge (best effort).

    Response: the Approval fields plus `detail` describing promote/deploy outcomes or errors.
    """
    engine = db.get_engine()
    pr = next((p for p in list_promotions_db(engine) if p.promotion_id == promotion_id), None)
    if pr is None:
        raise HTTPException(status_code=404, detail=f"unknown promotion {promotion_id}")
    if db.approvals_for(promotion_id=promotion_id, engine=engine) or pr.status != "pending":
        raise HTTPException(status_code=409, detail=f"promotion {promotion_id} already decided")
    approval = _new_approval(req, promotion_id=promotion_id)
    db.insert_approval(approval, engine=engine)
    detail: dict[str, Any] = {}
    if req.decision == "approved":
        try:
            rm = orchestrator.tool_promote(promotion_id, approval, engine)
            detail["promoted"] = {"name": rm.name, "version": rm.version, "stage": rm.stage.value}
            payload = registry_row_payload(rm.name, rm.version, engine)
            artifact = payload.get("artifact_path")
            if not artifact:
                raise RuntimeError("registry payload has no artifact_path; edge deploy skipped")
            dep = orchestrator.tool_deploy_edge(
                rm,
                Path(str(artifact)),
                _features_for_version(rm.version, engine),
                Path(get_settings().models_dir),
                _window_rows_for_version(rm.version, engine),
            )
            detail["edge"] = {"onnx_path": dep.onnx_path, "deployed_at": dep.deployed_at}
        except Exception as e:
            log.warning("promotion %s: promote/deploy error: %s", promotion_id, e)
            detail["error"] = f"{type(e).__name__}: {e}"
    else:
        try:
            from agents.mlops.lifecycle import reject_promotion

            reject_promotion(promotion_id, engine=engine)
            detail["rejected"] = True
        except Exception:
            set_promotion_status(promotion_id, "rejected", engine)
            detail["rejected"] = True
    out = approval.model_dump(mode="json")
    out["detail"] = detail
    return out
