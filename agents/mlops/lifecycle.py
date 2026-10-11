"""Model lifecycle toolbox for 08 MLOps: registry, champion comparison, governed promotion,
and ONNX edge deployment.

Storage: `model_registry`, `promotion_requests` tables in `apps/api/db.py` (SQLAlchemy Core).
`promote()` models the stage lifecycle (candidate -> validated -> shadow -> production) as a
database enum, not live traffic-splitting. It refuses to move a model without a recorded
`Approval` -- that is the governance pitch (CLAUDE.md section 7), do not weaken it.

No LLM calls anywhere in this module.
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import subprocess
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sqlalchemy import insert, select, update
from sqlalchemy.engine import Engine

from apps.api import db
from apps.api.schemas import (
    Approval,
    ChampionComparison,
    EdgeDeployment,
    FeatureSpec,
    ModelRef,
    ModelStage,
    ParityReport,
    PromotionRequest,
    RegisteredModel,
    ValidationReport,
)
from apps.api.settings import get_settings

log = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "thor-bearing-classifier"

# Forward lifecycle order. `archived` is a terminal side-state, never a promotion target.
STAGE_ORDER: tuple[ModelStage, ...] = (
    ModelStage.candidate,
    ModelStage.validated,
    ModelStage.shadow,
    ModelStage.production,
)

# opset pinned so tree converters emit ai.onnx.ml v3 (what the edge onnxruntime understands).
ONNX_OPSET: dict[str, int] = {"": 17, "ai.onnx.ml": 3}


# --------------------------------------------------------------------------------------
# Internal helpers
# --------------------------------------------------------------------------------------


def _utc(dt: datetime | None) -> datetime:
    """SQLite drops tzinfo on the way back; re-attach UTC so models compare cleanly."""
    if dt is None:
        return db.now_utc()
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=UTC)


def _row_to_model(row: Any) -> RegisteredModel:
    r = dict(row)
    return RegisteredModel(
        name=r["name"],
        version=r["version"],
        mlflow_run_id=r.get("mlflow_run_id"),
        stage=ModelStage(r["stage"]),
        family=r["family"],
        ims_total=float(r["ims_total"]),
        git_sha=r.get("git_sha"),
        dataset_version=r.get("dataset_version"),
        registered_at=_utc(r.get("registered_at")),
        onnx_path=r.get("onnx_path"),
    )


def _git_sha() -> str | None:
    """Best-effort short git SHA of the working tree; None when git is unavailable."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except Exception:  # noqa: BLE001 - any failure means "unknown", never a crash
        return None
    sha = out.stdout.strip()
    return sha or None


def _dataset_version(validation: ValidationReport) -> str | None:
    """sha256 (16 hex chars) over the sorted set of asset ids seen in the backtest folds."""
    ids: set[str] = set()
    for fold in validation.backtest:
        ids.update(fold.train_assets)
        ids.update(fold.test_assets)
    if not ids:
        return None
    return hashlib.sha256(",".join(sorted(ids)).encode()).hexdigest()[:16]


def validation_failures(validation: ValidationReport) -> list[str]:
    """Human-readable reasons a ValidationReport did not pass (empty when it passed).

    Input: a 06 `ValidationReport`. Output: one line per failed condition, taken from the
    report's `gate` checklist when present, else derived from the leakage / recall / lead-time
    fields. Stored in the registry payload so ModelOps can show *why* a model is blocked.
    """
    if validation.passed:
        return []
    gate = getattr(validation, "gate", None) or []
    reasons = [f"{g.name}: {g.detail}" for g in gate if not g.passed]
    if reasons:
        return reasons
    if not validation.leakage.passed:
        reasons.append("leakage check failed: " + "; ".join(validation.leakage.notes or ["-"]))
    reasons.append(f"recall {validation.ims.recall:.2f}, lead time {validation.lead_time.median_h:.1f} h")
    return reasons


def _registry_payload(
    validation: ValidationReport, artifact_path: Path, feature_spec: FeatureSpec | None = None
) -> dict[str, Any]:
    ims = validation.ims
    spec = (
        {
            "features": list(feature_spec.features),
            "window_rows": int(feature_spec.window_rows),
            "baseline": feature_spec.baseline,
            "fit_assets": feature_spec.fit_assets,
        }
        if feature_spec is not None
        else None
    )
    return {
        "artifact_path": str(artifact_path),
        "feature_spec": spec,
        "asset_id": validation.asset_id,
        "champion_id": validation.champion_id,
        "validation_passed": validation.passed,
        "validation_failures": validation_failures(validation),
        "parity_passed": validation.parity.passed if validation.parity is not None else None,
        "parity": db.dump_model(validation.parity) if validation.parity is not None else None,
        # The full report travels with the row so an analyze run (and the model card) can cite
        # the lead time, calibration and gate of the model it reuses.
        "validation": db.dump_model(validation),
        "metrics": {
            "ims_total": ims.total,
            "recall": ims.recall,
            "precision": ims.precision,
            "lead_time_score": ims.lead_time_score,
            "calibration_score": ims.calibration_score,
            "latency_score": ims.latency_score,
            "lead_time_median_h": validation.lead_time.median_h,
            "brier_after": validation.calibration.brier_after,
        },
    }


def _mlflow_register(validation: ValidationReport, name: str) -> None:
    """Best-effort `mlflow.register_model`; a missing run or server only logs a warning."""
    if not validation.champion_mlflow_run_id:
        return
    try:
        import mlflow

        mlflow.set_tracking_uri(get_settings().mlflow_tracking_uri)
        mlflow.register_model(f"runs:/{validation.champion_mlflow_run_id}/model", name)
    except Exception as exc:  # noqa: BLE001 - MLflow is never load-bearing
        log.warning("mlflow.register_model skipped: %s", exc)


class PromotionBlocked(ValueError):
    """Raised when a model may not be promoted: it failed validation (or, once the parity
    check exists, edge-vs-centre parity). A ValueError so existing callers keep working."""


def _assert_promotable(row: Any) -> None:
    """Refuse promotion of a registry row whose validation did not pass.

    Input: a `model_registry` row mapping. Raises `PromotionBlocked` (a ValueError) naming the
    recorded failure reasons when `payload.validation_passed` is False or when the recorded
    edge parity check failed (`payload.parity_passed` is False). Rows registered before the
    flag existed (no `validation_passed` key) are treated as not validated and refused too:
    a model with no recorded validation must never reach production.
    """
    payload = dict(row.get("payload") or {})
    if payload.get("validation_passed") is not True:
        reasons = payload.get("validation_failures") or ["no validation result recorded"]
        raise PromotionBlocked(
            f"model {row['name']}:{row['version']} failed validation; promotion refused "
            f"({'; '.join(str(r) for r in reasons)})"
        )
    if payload.get("parity_passed") is False:
        parity = payload.get("parity") or {}
        raise PromotionBlocked(
            f"model {row['name']}:{row['version']} failed the edge parity check; promotion "
            f"refused ({'; '.join(str(n) for n in parity.get('notes', []))})"
        )


def _fetch_row(conn: Any, name: str, version: str) -> Any:
    return (
        conn.execute(
            select(db.model_registry).where(
                db.model_registry.c.name == name, db.model_registry.c.version == version
            )
        )
        .mappings()
        .first()
    )


# --------------------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------------------


def register_model(
    validation: ValidationReport,
    artifact_path: Path,
    name: str = DEFAULT_MODEL_NAME,
    engine: Engine | None = None,
    feature_spec: FeatureSpec | None = None,
) -> RegisteredModel:
    """Insert the validated champion into `model_registry` at stage=candidate.

    Inputs: the 06 `ValidationReport` (version = `model_version`, family/IMS from the
    champion), the joblib `artifact_path` of the (calibrated) champion, the registry
    `name`, and optionally the `FeatureSpec` the model was trained with (features, window,
    regime baseline) which is stored in the row payload so the edge export, the fleet's
    regime inference and "Run analysis" can reuse the production model without a run state.
    Output: `RegisteredModel`. Re-registering an existing (name, version) returns the stored
    row unchanged. Also calls `mlflow.register_model` best-effort.
    """
    engine = engine or db.get_engine()
    version = validation.model_version
    with engine.begin() as conn:
        existing = _fetch_row(conn, name, version)
        if existing is not None:
            log.info("model %s:%s already registered; returning existing row", name, version)
            return _row_to_model(existing)
        conn.execute(
            insert(db.model_registry).values(
                name=name,
                version=version,
                mlflow_run_id=validation.champion_mlflow_run_id,
                stage=ModelStage.candidate.value,
                family=validation.champion_family,
                ims_total=float(validation.ims.total),
                git_sha=_git_sha(),
                dataset_version=_dataset_version(validation),
                registered_at=db.now_utc(),
                onnx_path=None,
                payload=_registry_payload(validation, artifact_path, feature_spec),
            )
        )
        row = _fetch_row(conn, name, version)
    _mlflow_register(validation, name)
    return _row_to_model(row)


def get_registered_model(
    name: str, version: str, engine: Engine | None = None
) -> RegisteredModel | None:
    """Look up one registry row by (name, version). Output: model or None."""
    engine = engine or db.get_engine()
    with engine.connect() as conn:
        row = _fetch_row(conn, name, version)
    return _row_to_model(row) if row is not None else None


def get_artifact_path(name: str, version: str, engine: Engine | None = None) -> Path | None:
    """Return the joblib artifact path stored in the registry payload, or None."""
    engine = engine or db.get_engine()
    with engine.connect() as conn:
        row = _fetch_row(conn, name, version)
    if row is None or not row.get("payload"):
        return None
    p = row["payload"].get("artifact_path")
    return Path(p) if p else None


def get_production_model(
    name: str = DEFAULT_MODEL_NAME, engine: Engine | None = None
) -> RegisteredModel | None:
    """Current stage=production row for `name` (newest if several). Output: model or None."""
    engine = engine or db.get_engine()
    with engine.connect() as conn:
        row = (
            conn.execute(
                select(db.model_registry)
                .where(
                    db.model_registry.c.name == name,
                    db.model_registry.c.stage == ModelStage.production.value,
                )
                .order_by(db.model_registry.c.registered_at.desc(), db.model_registry.c.id.desc())
            )
            .mappings()
            .first()
        )
    return _row_to_model(row) if row is not None else None


def production_feature_spec(
    name: str = DEFAULT_MODEL_NAME, engine: Engine | None = None
) -> FeatureSpec | None:
    """FeatureSpec (features, window_rows, regime baseline) stored with the production row.

    Output: the spec, or None when there is no production model or it was registered
    without one. Deterministic read; this is what operational paths (fleet regime, Run
    analysis, edge export) use so they depend on the *approved* model only.
    """
    engine = engine or db.get_engine()
    prod = get_production_model(name, engine=engine)
    if prod is None:
        return None
    with engine.connect() as conn:
        row = _fetch_row(conn, prod.name, prod.version)
    spec = (row.get("payload") or {}).get("feature_spec") if row is not None else None
    if not spec:
        return None
    return FeatureSpec(
        features=list(spec.get("features") or []),
        window_rows=int(spec.get("window_rows") or 12),
        description=f"production model {prod.version}",
        baseline=spec.get("baseline"),
    )


def production_bundle(
    name: str = DEFAULT_MODEL_NAME, engine: Engine | None = None
) -> tuple[ModelRef, FeatureSpec, ValidationReport] | None:
    """Everything an analyze run needs from the production row, or None when there is none.

    Output: (ModelRef with the calibrated artifact path, the FeatureSpec it was trained with
    -- features, window, regime baseline -- and its ValidationReport). Raises ValueError when
    the production row predates the stored spec / report (retrain and re-promote it).
    """
    engine = engine or db.get_engine()
    prod = get_production_model(name, engine=engine)
    if prod is None:
        return None
    with engine.connect() as conn:
        row = _fetch_row(conn, prod.name, prod.version)
    payload = dict((row.get("payload") or {}) if row is not None else {})
    spec = payload.get("feature_spec")
    report = payload.get("validation")
    if not spec or not report:
        raise ValueError(
            f"production model {prod.version} has no stored feature spec / validation report; "
            "train and promote a model with the current pipeline"
        )
    ref = ModelRef(
        name=prod.name,
        version=prod.version,
        stage=prod.stage,
        artifact_path=payload.get("artifact_path"),
        registered_at=prod.registered_at,
        ims_total=prod.ims_total,
    )
    feature_spec = FeatureSpec(
        features=list(spec.get("features") or []),
        window_rows=int(spec.get("window_rows") or 12),
        description=f"production model {prod.version}",
        baseline=spec.get("baseline"),
        fit_assets=spec.get("fit_assets"),
    )
    return ref, feature_spec, ValidationReport.model_validate(report)


def list_models(engine: Engine | None = None) -> list[RegisteredModel]:
    """Every registry row, newest first. Output: list of `RegisteredModel`."""
    engine = engine or db.get_engine()
    with engine.connect() as conn:
        rows = (
            conn.execute(
                select(db.model_registry).order_by(
                    db.model_registry.c.registered_at.desc(), db.model_registry.c.id.desc()
                )
            )
            .mappings()
            .all()
        )
    return [_row_to_model(r) for r in rows]


# --------------------------------------------------------------------------------------
# Champion comparison + promotion requests
# --------------------------------------------------------------------------------------


def _metrics_for(name: str, version: str, engine: Engine) -> dict[str, float]:
    with engine.connect() as conn:
        row = _fetch_row(conn, name, version)
    if row is None or not row.get("payload"):
        return {}
    return {k: float(v) for k, v in row["payload"].get("metrics", {}).items()}


def compare_champion(
    challenger: RegisteredModel, engine: Engine | None = None
) -> ChampionComparison:
    """Compare `challenger` with the current production model of the same name.

    Inputs: a `RegisteredModel`. Output: `ChampionComparison` with `delta` = challenger minus
    champion on `ims_total` (plus any shared payload metrics); `recommend_promote` is True when
    there is no champion or the IMS delta is positive.
    """
    engine = engine or db.get_engine()
    champion = get_production_model(challenger.name, engine=engine)
    if champion is None:
        return ChampionComparison(
            challenger=challenger,
            champion=None,
            delta={"ims_total": round(challenger.ims_total, 6)},
            recommend_promote=True,
            rationale=(
                f"No production model named '{challenger.name}'; challenger "
                f"{challenger.version} ({challenger.family}, IMS {challenger.ims_total:.3f}) "
                "would be the first champion."
            ),
        )
    delta: dict[str, float] = {"ims_total": round(challenger.ims_total - champion.ims_total, 6)}
    ch_m = _metrics_for(challenger.name, challenger.version, engine)
    cp_m = _metrics_for(champion.name, champion.version, engine)
    for k in sorted(set(ch_m) & set(cp_m)):
        if k != "ims_total":
            delta[k] = round(ch_m[k] - cp_m[k], 6)
    better = delta["ims_total"] > 0
    verb = "beats" if better else "does not beat"
    return ChampionComparison(
        challenger=challenger,
        champion=champion,
        delta=delta,
        recommend_promote=better,
        rationale=(
            f"Challenger {challenger.version} ({challenger.family}, IMS "
            f"{challenger.ims_total:.3f}) {verb} champion {champion.version} "
            f"({champion.family}, IMS {champion.ims_total:.3f}) by {delta['ims_total']:+.3f}."
        ),
    )


def request_promotion(
    comparison: ChampionComparison,
    to_stage: ModelStage,
    engine: Engine | None = None,
) -> PromotionRequest:
    """Open a pending `promotion_requests` row for the challenger.

    Inputs: a `ChampionComparison` and the requested `to_stage` (must be strictly later in
    STAGE_ORDER than the challenger's current stage; `archived` is not allowed). Output:
    `PromotionRequest` with status=pending. Nothing moves until `promote()` sees an approval.
    Raises `PromotionBlocked` (ValueError) when the challenger's registry row records
    `validation_passed = False`: a model that failed validation cannot even be proposed.
    """
    engine = engine or db.get_engine()
    challenger = comparison.challenger
    from_stage = challenger.stage
    if to_stage not in STAGE_ORDER:
        raise ValueError(f"cannot request promotion to '{to_stage}'")
    with engine.connect() as conn:
        row = _fetch_row(conn, challenger.name, challenger.version)
    if row is None:
        raise ValueError(f"model {challenger.name}:{challenger.version} not in registry")
    _assert_promotable(row)
    if from_stage not in STAGE_ORDER or STAGE_ORDER.index(to_stage) <= STAGE_ORDER.index(
        from_stage
    ):
        raise ValueError(f"invalid promotion {from_stage} -> {to_stage}")
    req = PromotionRequest(
        promotion_id=f"pr_{challenger.version}_{to_stage.value}_{uuid.uuid4().hex[:8]}",
        model_name=challenger.name,
        version=challenger.version,
        from_stage=from_stage,
        to_stage=to_stage,
        comparison=comparison,
        status="pending",
        created_at=db.now_utc(),
    )
    with engine.begin() as conn:
        conn.execute(
            insert(db.promotion_requests).values(
                promotion_id=req.promotion_id,
                model_name=req.model_name,
                version=req.version,
                from_stage=req.from_stage.value,
                to_stage=req.to_stage.value,
                status=req.status,
                created_at=req.created_at,
                payload=db.dump_model(req),
            )
        )
    return req


def get_promotion(promotion_id: str, engine: Engine | None = None) -> PromotionRequest | None:
    """Load one promotion request (status reflects the row, not the stale payload)."""
    engine = engine or db.get_engine()
    with engine.connect() as conn:
        row = (
            conn.execute(
                select(db.promotion_requests).where(
                    db.promotion_requests.c.promotion_id == promotion_id
                )
            )
            .mappings()
            .first()
        )
    if row is None:
        return None
    req = PromotionRequest.model_validate(row["payload"])
    return req.model_copy(update={"status": row["status"]})


def list_promotions(
    status: str | None = None, engine: Engine | None = None
) -> list[PromotionRequest]:
    """All promotion requests, newest first; optional `status` filter."""
    engine = engine or db.get_engine()
    q = select(db.promotion_requests).order_by(db.promotion_requests.c.created_at.desc())
    if status:
        q = q.where(db.promotion_requests.c.status == status)
    with engine.connect() as conn:
        rows = conn.execute(q).mappings().all()
    return [
        PromotionRequest.model_validate(r["payload"]).model_copy(update={"status": r["status"]})
        for r in rows
    ]


def _check_approval(promotion_id: str, approval: Approval, expected: str) -> None:
    if approval.promotion_id != promotion_id:
        raise PermissionError(
            f"approval {approval.approval_id} is for promotion "
            f"'{approval.promotion_id}', not '{promotion_id}'"
        )
    if approval.decision != expected:
        raise PermissionError(
            f"promotion {promotion_id} requires decision='{expected}', got '{approval.decision}'"
        )


def _set_promotion_status(conn: Any, promotion_id: str, status: str) -> None:
    conn.execute(
        update(db.promotion_requests)
        .where(db.promotion_requests.c.promotion_id == promotion_id)
        .values(status=status)
    )


def promote(promotion_id: str, approval: Approval, engine: Engine | None = None) -> RegisteredModel:
    """Apply an APPROVED promotion request: move the model to the requested stage.

    Inputs: the `promotion_id` and the human `Approval` recorded by 09. Raises
    `PermissionError` unless `approval.decision == "approved"` and
    `approval.promotion_id == promotion_id`; raises `ValueError` if the request is missing,
    no longer pending, or the model is no longer at the request's `from_stage`.
    Stage order is enforced (candidate -> validated -> shadow -> production); a request that
    was explicitly approved for a later stage jumps straight to it. Promotion to production
    archives every other production row of the same name. Marks the request approved.
    Raises `PromotionBlocked` (ValueError) if the registry row records a failed validation.
    Output: the updated `RegisteredModel`.
    """
    _check_approval(promotion_id, approval, "approved")
    engine = engine or db.get_engine()
    req = get_promotion(promotion_id, engine=engine)
    if req is None:
        raise ValueError(f"unknown promotion request '{promotion_id}'")
    if req.status != "pending":
        raise ValueError(f"promotion {promotion_id} already {req.status}")
    if req.to_stage not in STAGE_ORDER or STAGE_ORDER.index(req.to_stage) <= STAGE_ORDER.index(
        req.from_stage
    ):
        raise ValueError(f"invalid stage transition {req.from_stage} -> {req.to_stage}")
    with engine.begin() as conn:
        row = _fetch_row(conn, req.model_name, req.version)
        if row is None:
            raise ValueError(f"model {req.model_name}:{req.version} not in registry")
        if row["stage"] != req.from_stage.value:
            raise ValueError(
                f"model {req.model_name}:{req.version} is at stage '{row['stage']}', "
                f"request expected '{req.from_stage.value}'"
            )
        # Re-checked at promotion time, not only at request time: the registry row is the
        # source of truth and a request opened before a later validation failure must not win.
        _assert_promotable(row)
        if req.to_stage == ModelStage.production:
            conn.execute(
                update(db.model_registry)
                .where(
                    db.model_registry.c.name == req.model_name,
                    db.model_registry.c.stage == ModelStage.production.value,
                    db.model_registry.c.version != req.version,
                )
                .values(stage=ModelStage.archived.value)
            )
        conn.execute(
            update(db.model_registry)
            .where(db.model_registry.c.id == row["id"])
            .values(stage=req.to_stage.value)
        )
        _set_promotion_status(conn, promotion_id, "approved")
        updated = _fetch_row(conn, req.model_name, req.version)
    log.info(
        "promoted %s:%s %s -> %s (approval %s by %s)",
        req.model_name,
        req.version,
        req.from_stage.value,
        req.to_stage.value,
        approval.approval_id,
        approval.approver,
    )
    return _row_to_model(updated)


def reject_promotion(
    promotion_id: str, approval: Approval, engine: Engine | None = None
) -> PromotionRequest:
    """Record a human rejection: mark the request rejected, leave the model where it is.

    Inputs: `promotion_id` and an `Approval` with decision="rejected" for that id (else
    `PermissionError`). Output: the updated `PromotionRequest`.
    """
    _check_approval(promotion_id, approval, "rejected")
    engine = engine or db.get_engine()
    req = get_promotion(promotion_id, engine=engine)
    if req is None:
        raise ValueError(f"unknown promotion request '{promotion_id}'")
    with engine.begin() as conn:
        _set_promotion_status(conn, promotion_id, "rejected")
    return req.model_copy(update={"status": "rejected"})


# --------------------------------------------------------------------------------------
# Edge deployment (ONNX)
# --------------------------------------------------------------------------------------


def _unwrap_estimator(model: Any) -> tuple[Any, dict[str, Any]]:
    """Return (base tree estimator, calibration transform).

    A `CalibratedClassifierCV` over a prefit (`FrozenEstimator`) tree model is split into the
    tree (exported to ONNX) and its calibrator (exported as an explicit transform in the
    sidecar, see `agents.validation.checks.calibration_transform`), so the edge reproduces the
    centre's calibrated probability. Raises ValueError for a multi-fold calibrator.
    """
    from agents.validation.checks import calibration_transform

    transform = calibration_transform(model)
    current = model
    for _ in range(8):
        calibrated = getattr(current, "calibrated_classifiers_", None)
        if calibrated:
            current = calibrated[0].estimator
            continue
        # sklearn >= 1.6 wraps a prefit estimator in FrozenEstimator inside the calibrator
        if type(current).__name__ == "FrozenEstimator" and hasattr(current, "estimator"):
            current = current.estimator
            continue
        base = getattr(current, "estimator", None)
        if (
            base is not None
            and hasattr(base, "predict_proba")
            and not hasattr(current, "estimators_")
        ):
            # generic wrapper exposing .estimator (e.g. the identity calibration shim)
            current = base
            continue
        break
    return current, transform


def _register_boost_converters() -> None:
    """Teach skl2onnx how to convert XGBClassifier / LGBMClassifier via onnxmltools."""
    from skl2onnx import update_registered_converter
    from skl2onnx.common.shape_calculator import calculate_linear_classifier_output_shapes

    opts = {"nocl": [True, False], "zipmap": [True, False, "columns"]}
    try:
        from onnxmltools.convert.xgboost.operator_converters.XGBoost import convert_xgboost
        from xgboost import XGBClassifier

        update_registered_converter(
            XGBClassifier,
            "XGBoostXGBClassifier",
            calculate_linear_classifier_output_shapes,
            convert_xgboost,
            options=opts,
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("xgboost ONNX converter not registered: %s", exc)
    try:
        from lightgbm import LGBMClassifier
        from onnxmltools.convert.lightgbm.operator_converters.LightGbm import convert_lightgbm

        update_registered_converter(
            LGBMClassifier,
            "LightGbmLGBMClassifier",
            calculate_linear_classifier_output_shapes,
            convert_lightgbm,
            options=opts,
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("lightgbm ONNX converter not registered: %s", exc)


def _family_of(estimator: Any) -> str:
    mod = type(estimator).__module__
    if mod.startswith("xgboost"):
        return "xgboost"
    if mod.startswith("lightgbm"):
        return "lightgbm"
    return "random_forest"


def to_onnx(estimator: Any, n_features: int) -> Any:
    """Convert a fitted tree classifier (sklearn RF / XGBClassifier / LGBMClassifier) to an
    ONNX ModelProto with a single float input `input[None, n_features]` and outputs
    `label`, `probabilities[None, 2]` (zipmap disabled)."""
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType

    family = _family_of(estimator)
    if family in ("xgboost", "lightgbm"):
        _register_boost_converters()
    if family == "xgboost":
        # onnxmltools' XGBoost converter only understands f%d feature names; models trained on
        # a DataFrame carry real column names, so strip them on a copy.
        estimator = copy.deepcopy(estimator)
        try:
            estimator.get_booster().feature_names = None
        except Exception as exc:  # noqa: BLE001
            log.warning("could not clear xgboost feature names: %s", exc)
    initial_types = [("input", FloatTensorType([None, n_features]))]
    return convert_sklearn(
        estimator,
        initial_types=initial_types,
        options={id(estimator): {"zipmap": False}},
        target_opset=ONNX_OPSET,
    )


def _verify_onnx(onnx_path: Path, n_features: int) -> np.ndarray:
    """Run onnxruntime on a zeros row; return the probability output (shape (1, 2))."""
    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0].name
    outs = sess.run(None, {inp: np.zeros((1, n_features), dtype=np.float32)})
    probs = next(
        (np.asarray(o) for o in outs if getattr(o, "ndim", 0) == 2 and o.shape[-1] == 2), None
    )
    if probs is None or probs.shape != (1, 2):
        raise RuntimeError(f"ONNX verification failed: no (1, 2) probability output in {outs!r}")
    if not np.isfinite(probs).all() or abs(float(probs.sum()) - 1.0) > 1e-3:
        raise RuntimeError(f"ONNX verification failed: probabilities {probs.tolist()}")
    return probs


def export_bundle(
    artifact_path: Path,
    features: list[str],
    out_dir: Path,
    stem: str,
    window_rows: int = 12,
    baseline: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> tuple[Path, Path, dict[str, Any]]:
    """Write `<stem>.onnx` + `<stem>.json` for the edge from a joblib champion.

    Inputs: the joblib artifact (tree model, optionally calibrated with a prefit
    `CalibratedClassifierCV`), the ordered feature list, the target directory, file stem,
    the feature window length, the regime baseline (training format) and extra sidecar keys.
    Output: (onnx_path, sidecar_path, sidecar dict). The sidecar carries the calibration as an
    explicit transform and the baseline, so `edge.features` + `edge.inference` reproduce the
    centre's calibrated probability. Verified with onnxruntime on a zeros row.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fitted = joblib.load(artifact_path)
    estimator, calibration = _unwrap_estimator(fitted)
    onx = to_onnx(estimator, len(features))
    onnx_path = out_dir / f"{stem}.onnx"
    sidecar_path = out_dir / f"{stem}.json"
    onnx_path.write_bytes(onx.SerializeToString())
    _verify_onnx(onnx_path, len(features))
    sidecar: dict[str, Any] = {
        "features": list(features),
        "window_rows": int(window_rows),
        "calibration": calibration,
        "family": _family_of(estimator),
    }
    if baseline is not None:
        sidecar["baseline"] = baseline
    if extra:
        sidecar.update(extra)
    sidecar_path.write_text(json.dumps(sidecar, indent=2, default=str), encoding="utf-8")
    return onnx_path, sidecar_path, sidecar


def check_edge_parity(
    artifact_path: Path,
    features: list[str],
    X: Any,
    window_rows: int = 12,
    baseline: dict[str, Any] | None = None,
    raw_windows: list[np.ndarray] | None = None,
    tolerance_probability: float = 1e-3,
    tolerance_features: float = 1e-6,
    workdir: Path | None = None,
) -> ParityReport:
    """Prove the edge agrees with the centre before anything is promoted.

    Inputs: the calibrated joblib champion, its feature order, real held-out feature vectors
    `X` (DataFrame or array, (n, len(features))), the window length, the regime baseline and
    optional raw sensor windows ((window_rows, 8) arrays in SENSOR_COLUMNS order) from
    held-out assets. The model is exported exactly as `deploy_edge` would (ONNX + sidecar),
    loaded through `edge.inference` and scored; the centre scores the same vectors with the
    joblib model. Raw windows go through `edge.features.compute_features_np` and
    `agents.ml_architect.features.compute_features_np` with the same baseline.
    Output: `ParityReport` with the max absolute probability difference (must be <=
    `tolerance_probability`, default 1e-3: float32 ONNX trees vs float64 sklearn) and the max
    absolute feature difference (<= `tolerance_features`). Any exception during export or
    scoring yields passed=False with the error in `notes` -- never a silent pass.
    """
    import tempfile

    notes: list[str] = []
    n_rows = int(len(X))
    try:
        tmp_ctx = tempfile.TemporaryDirectory(prefix="thor-parity-") if workdir is None else None
        out_dir = Path(tmp_ctx.name) if tmp_ctx is not None else Path(workdir)
        try:
            onnx_path, sidecar_path, sidecar = export_bundle(
                Path(artifact_path), features, out_dir, "parity", window_rows, baseline
            )
            from edge.inference import load_model, predict_proba

            edge_model = load_model(onnx_path, sidecar_path)
            Xa = np.asarray(X[list(features)] if hasattr(X, "columns") else X, dtype=float)
            p_edge = predict_proba(edge_model, Xa)
        finally:
            if tmp_ctx is not None:
                tmp_ctx.cleanup()
        fitted = joblib.load(artifact_path)
        proba = np.asarray(fitted.predict_proba(X[list(features)] if hasattr(X, "columns") else X))
        p_centre = proba[:, 1] if proba.ndim == 2 else proba.reshape(-1)
        max_p = float(np.max(np.abs(p_edge - p_centre))) if n_rows else 0.0
        calibration_method = str(sidecar.get("calibration", {}).get("method", "none"))
        notes.append(
            f"{n_rows} held-out vectors: edge vs centre max |dp| = {max_p:.2e} "
            f"(calibration {calibration_method})"
        )
        max_f: float | None = None
        n_windows = 0
        if raw_windows:
            from agents.ml_architect.features import compute_features_np as centre_features
            from edge.features import compute_features_np as edge_features

            diffs = []
            for buf in raw_windows:
                a = centre_features(np.asarray(buf, dtype=float), baseline or {})
                b = edge_features(np.asarray(buf, dtype=float), baseline or {})
                scale = np.maximum(1.0, np.abs(a))
                diffs.append(float(np.max(np.abs(a - b) / scale)))
            n_windows = len(diffs)
            max_f = max(diffs) if diffs else 0.0
            notes.append(f"{n_windows} raw windows: edge vs centre max relative |df| = {max_f:.2e}")
        passed = max_p <= tolerance_probability and (max_f is None or max_f <= tolerance_features)
        if not passed:
            notes.append("PARITY FAILED: the edge would not reproduce the centre's probability")
        return ParityReport(
            passed=passed,
            n_rows=n_rows,
            max_abs_diff_probability=max_p,
            tolerance_probability=tolerance_probability,
            n_windows=n_windows,
            max_abs_diff_features=max_f,
            tolerance_features=tolerance_features,
            calibration_method=calibration_method,
            notes=notes,
        )
    except Exception as exc:  # noqa: BLE001 - a broken export is a failed parity, never a pass
        log.warning("edge parity check failed: %s", exc)
        return ParityReport(
            passed=False,
            n_rows=n_rows,
            max_abs_diff_probability=float("inf"),
            tolerance_probability=tolerance_probability,
            notes=notes + [f"parity check error: {type(exc).__name__}: {exc}"],
        )


def deploy_edge(
    model: RegisteredModel,
    artifact_path: Path,
    features: list[str],
    models_dir: Path,
    window_rows: int = 12,
    baseline: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
    engine: Engine | None = None,
) -> EdgeDeployment:
    """Export the registered model to ONNX + JSON sidecar in `models_dir` for the edge container.

    Inputs: the `RegisteredModel`, its joblib `artifact_path` (RF / XGB / LGBM, optionally
    wrapped in a prefit `CalibratedClassifierCV` -- the tree goes to ONNX and the calibration
    to the sidecar as an explicit transform), the ordered `features` list, target
    `models_dir`, the feature `window_rows`, the regime `baseline` (training format) and
    `extra` keys merged into the sidecar. Writes `<name>_<version>.onnx` and `.json` =
    {model_name, version, features, window_rows, calibration, family, deployed_at,
    [baseline], ...extra}. Verifies with onnxruntime and stores `onnx_path` on the registry
    row. Output: `EdgeDeployment`.
    """
    engine = engine or db.get_engine()
    deployed_at = db.now_utc()
    stem = f"{model.name}_{model.version}"
    onnx_path, _sidecar_path, sidecar = export_bundle(
        Path(artifact_path),
        features,
        Path(models_dir),
        stem,
        window_rows,
        baseline,
        {
            "model_name": model.name,
            "version": model.version,
            "deployed_at": deployed_at.isoformat(),
            **(extra or {}),
        },
    )
    with engine.begin() as conn:
        conn.execute(
            update(db.model_registry)
            .where(
                db.model_registry.c.name == model.name, db.model_registry.c.version == model.version
            )
            .values(onnx_path=str(onnx_path))
        )
    log.info(
        "deployed %s to %s (%s, calibration=%s)",
        stem,
        onnx_path,
        sidecar["family"],
        sidecar["calibration"].get("method"),
    )
    return EdgeDeployment(
        model_name=model.name,
        version=model.version,
        onnx_path=str(onnx_path),
        deployed_at=deployed_at,
        input_features=list(features),
    )
