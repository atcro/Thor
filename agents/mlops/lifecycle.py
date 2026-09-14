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
    ModelStage,
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


def _registry_payload(validation: ValidationReport, artifact_path: Path) -> dict[str, Any]:
    ims = validation.ims
    return {
        "artifact_path": str(artifact_path),
        "asset_id": validation.asset_id,
        "champion_id": validation.champion_id,
        "validation_passed": validation.passed,
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
) -> RegisteredModel:
    """Insert the validated champion into `model_registry` at stage=candidate.

    Inputs: the 06 `ValidationReport` (version = `model_version`, family/IMS from the
    champion), the joblib `artifact_path` of the (calibrated) champion, and the registry
    `name`. Output: `RegisteredModel`. Re-registering an existing (name, version) returns the
    stored row unchanged. Also calls `mlflow.register_model` best-effort.
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
                payload=_registry_payload(validation, artifact_path),
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
    """
    engine = engine or db.get_engine()
    challenger = comparison.challenger
    from_stage = challenger.stage
    if to_stage not in STAGE_ORDER:
        raise ValueError(f"cannot request promotion to '{to_stage}'")
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


def _unwrap_estimator(model: Any) -> tuple[Any, str]:
    """Return (base tree estimator, calibration note). CalibratedClassifierCV is unwrapped
    to its first fold's base estimator; calibration is NOT exported to ONNX -> "none"."""
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
            # generic wrapper exposing .estimator (e.g. a custom calibration shim)
            current = base
            continue
        break
    return current, "none"


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
    wrapped in `CalibratedClassifierCV` -- unwrapped, calibration recorded as "none"), the
    ordered `features` list, target `models_dir`, the feature `window_rows`, an optional
    feature `baseline` dict and `extra` keys merged into the sidecar. Writes
    `<name>_<version>.onnx` and `.json` = {model_name, version, features, window_rows,
    calibration, family, deployed_at, [baseline], ...extra}. Verifies with onnxruntime and
    stores `onnx_path` on the registry row. Output: `EdgeDeployment`.
    """
    engine = engine or db.get_engine()
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    fitted = joblib.load(artifact_path)
    estimator, calibration = _unwrap_estimator(fitted)
    onx = to_onnx(estimator, len(features))
    stem = f"{model.name}_{model.version}"
    onnx_path = models_dir / f"{stem}.onnx"
    sidecar_path = models_dir / f"{stem}.json"
    onnx_path.write_bytes(onx.SerializeToString())
    _verify_onnx(onnx_path, len(features))
    deployed_at = db.now_utc()
    sidecar: dict[str, Any] = {
        "model_name": model.name,
        "version": model.version,
        "features": list(features),
        "window_rows": int(window_rows),
        "calibration": calibration,
        "family": _family_of(estimator),
        "deployed_at": deployed_at.isoformat(),
    }
    if baseline is not None:
        sidecar["baseline"] = baseline
    if extra:
        sidecar.update(extra)
    sidecar_path.write_text(json.dumps(sidecar, indent=2, default=str), encoding="utf-8")
    with engine.begin() as conn:
        conn.execute(
            update(db.model_registry)
            .where(
                db.model_registry.c.name == model.name, db.model_registry.c.version == model.version
            )
            .values(onnx_path=str(onnx_path))
        )
    log.info(
        "deployed %s to %s (%s, calibration=%s)", stem, onnx_path, sidecar["family"], calibration
    )
    return EdgeDeployment(
        model_name=model.name,
        version=model.version,
        onnx_path=str(onnx_path),
        deployed_at=deployed_at,
        input_features=list(features),
    )
