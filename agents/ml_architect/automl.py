"""05 ML Architect: task inference, Optuna AutoML per model family, Industrial Model Score.

Governance notes
----------------
* Every split is asset-grouped and time-ordered: the last 25 % of assets (sorted by id) are
  held out and never seen by the model. No random shuffles anywhere (CLAUDE.md section 7).
* The saved artifact is the model fitted on the *training* assets only, so the Validation
  Agent (06) measures lead time on motors the model has never seen.
* MLflow is best effort: an unreachable tracking server produces a warning, never a crash.
* Nothing here calls an LLM.
"""

from __future__ import annotations

import logging
import os
import time
import uuid
import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import optuna
import pandas as pd
from optuna.samplers import TPESampler
from sklearn.metrics import (
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from apps.api.schemas import (
    CandidateModel,
    CandidateSet,
    DataQualityContract,
    FeatureSpec,
    IndustrialModelScore,
    TaskSpec,
)
from apps.api.settings import get_settings

logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)

FAMILIES: tuple[str, ...] = ("random_forest", "xgboost", "lightgbm")
HOLDOUT_FRACTION = 0.25
ALARM_THRESHOLD = 0.5
LEAD_TIME_CAP_H = 72.0
BRIER_CAP = 0.25
LATENCY_CAP_MS = 50.0
DEFAULT_EXPERIMENT = "thor-automl"
METRIC_KEYS: tuple[str, ...] = ("recall", "precision", "f1", "auroc", "brier", "lead_time_h")


# --------------------------------------------------------------------------------------
# infer_task
# --------------------------------------------------------------------------------------


def infer_task(dq: DataQualityContract, df: pd.DataFrame, horizon_h: float = 48.0) -> TaskSpec:
    """Decide what to train from the Data Quality Contract and the raw telemetry.

    Input: ``dq`` from ``profile_dataset``; the telemetry DataFrame (needs
    ``failure_within_h`` ground truth to build a label); the label horizon in hours.
    Output: ``TaskSpec``. Thor always frames predictive maintenance as
    ``classification`` of ``failure_within_h <= horizon_h`` (target ``"failure_within_h"``):
    with a handful of failure events per fleet, a horizon classifier plus a measured warning
    lead time is far more robust than direct RUL regression (a quantile RUL interval is still
    estimated downstream by the Validation Agent). The rationale records the evidence:
    number of failing assets, positive rate at the horizon and the data quality verdict.
    """
    n_fail_assets = 0
    pos_rate = 0.0
    if "failure_within_h" in df.columns and len(df):
        fw = df["failure_within_h"].astype(float)
        has = fw.notna()
        if "asset_id" in df.columns:
            n_fail_assets = int(df.loc[has, "asset_id"].nunique())
        pos_rate = float((fw <= horizon_h).fillna(False).mean())
    rationale = (
        f"{n_fail_assets} asset(s) with a recorded failure; {pos_rate:.1%} of rows fall within "
        f"the {horizon_h:g} h horizon; data quality {dq.quality_score:.0f}/100 "
        f"({'trainable' if dq.trainable else 'NOT trainable'}); {len(dq.regimes.regimes)} operating "
        "regimes detected. Framing as horizon classification: sparse failure events make a "
        "calibrated probability with measured lead time more reliable than direct RUL regression."
    )
    if n_fail_assets == 0:
        rationale += " WARNING: no failure events in the window; training will fail."
    return TaskSpec(
        task="classification",
        target="failure_within_h",
        horizon_h=float(horizon_h),
        rationale=rationale,
    )


# --------------------------------------------------------------------------------------
# industrial_model_score
# --------------------------------------------------------------------------------------


def industrial_model_score(metrics: dict[str, float], latency_ms: float) -> IndustrialModelScore:
    """Champion-selection score weighted for industrial use, not raw accuracy.

    Input: ``metrics`` with ``recall``, ``precision``, ``lead_time_h``, ``brier`` (missing keys
    count as their worst value); ``latency_ms`` = inference cost per 1000 rows.
    Output: ``IndustrialModelScore`` with
    ``lead_time_score = min(1, lead_time_h / 72)``,
    ``calibration_score = 1 - min(1, brier / 0.25)``,
    ``latency_score = 1 - min(1, latency_ms / 50)`` and
    ``total = sum(weight_k * score_k)`` using the schema's default weights
    (recall .35, lead_time .25, precision .15, calibration .15, latency .10).
    """

    def _clip01(v: float) -> float:
        v = float(v) if v is not None and np.isfinite(v) else 0.0
        return float(min(1.0, max(0.0, v)))

    recall = _clip01(metrics.get("recall", 0.0))
    precision = _clip01(metrics.get("precision", 0.0))
    lead_time_h = max(0.0, float(metrics.get("lead_time_h", 0.0) or 0.0))
    brier = float(metrics.get("brier", BRIER_CAP))
    if not np.isfinite(brier):
        brier = BRIER_CAP
    lat = float(latency_ms) if latency_ms is not None and np.isfinite(latency_ms) else LATENCY_CAP_MS

    lead_time_score = min(1.0, lead_time_h / LEAD_TIME_CAP_H)
    calibration_score = 1.0 - min(1.0, max(0.0, brier) / BRIER_CAP)
    latency_score = 1.0 - min(1.0, max(0.0, lat) / LATENCY_CAP_MS)
    ims = IndustrialModelScore(
        recall=recall,
        precision=precision,
        lead_time_score=lead_time_score,
        calibration_score=calibration_score,
        latency_score=latency_score,
        total=0.0,
    )
    parts = {
        "recall": recall,
        "lead_time_score": lead_time_score,
        "precision": precision,
        "calibration_score": calibration_score,
        "latency_score": latency_score,
    }
    ims.total = round(sum(ims.weights[k] * parts[k] for k in ims.weights), 6)
    return ims


# --------------------------------------------------------------------------------------
# Splits + evaluation helpers
# --------------------------------------------------------------------------------------


def split_by_asset(
    features_df: pd.DataFrame, holdout_fraction: float = HOLDOUT_FRACTION
) -> tuple[list[str], list[str]]:
    """Asset-grouped split: the last ``holdout_fraction`` of assets (sorted by id) are held out.

    Input: feature frame with an ``asset_id`` column. Output: ``(train_assets, holdout_assets)``,
    disjoint, both sorted; at least one asset on each side when >= 2 assets exist.
    """
    assets = sorted(features_df["asset_id"].astype(str).unique().tolist())
    if len(assets) < 2:
        return assets, []
    n_hold = int(round(holdout_fraction * len(assets)))
    n_hold = min(max(1, n_hold), len(assets) - 1)
    return assets[:-n_hold], assets[-n_hold:]


def _lead_time_hours(meta: pd.DataFrame, prob: np.ndarray, threshold: float = ALARM_THRESHOLD) -> float:
    """Median warning lead time over failing held-out assets.

    For each asset with ``rul_h`` present: failure time = last row ``ts + rul_h`` hours; lead
    time = hours from the first row with ``prob >= threshold`` to that failure (0 if never).
    Returns 0.0 when no held-out asset fails.
    """
    m = meta.assign(prob=prob)
    leads: list[float] = []
    for _, g in m.groupby("asset_id", sort=True):
        g = g.sort_values("ts")
        rul = g["rul_h"].astype(float)
        if not rul.notna().any():
            continue
        last = g.iloc[-1]
        failure_ts = pd.Timestamp(last["ts"]) + pd.Timedelta(hours=float(rul.iloc[-1]))
        alarms = g[g["prob"] >= threshold]
        if alarms.empty:
            leads.append(0.0)
            continue
        first = pd.Timestamp(alarms.iloc[0]["ts"])
        leads.append(max(0.0, (failure_ts - first).total_seconds() / 3600.0))
    return float(np.median(leads)) if leads else 0.0


def _latency_ms_per_1000(model: Any, X: pd.DataFrame, repeats: int = 3) -> float:
    """Best-of-``repeats`` wall time of ``predict_proba`` on ``X``, scaled to ms per 1000 rows."""
    if len(X) == 0:
        return 0.0
    model.predict_proba(X.iloc[: min(len(X), 64)])  # warm-up
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        model.predict_proba(X)
        best = min(best, time.perf_counter() - t0)
    return float(best / len(X) * 1000.0 * 1000.0)


def _positive_proba(model: Any, X: pd.DataFrame) -> np.ndarray:
    p = np.asarray(model.predict_proba(X))
    if p.ndim == 2 and p.shape[1] >= 2:
        return p[:, 1].astype(float)
    return p.reshape(-1).astype(float)


def _evaluate(model: Any, X: pd.DataFrame, y: np.ndarray, meta: pd.DataFrame) -> dict[str, float]:
    """Holdout metrics: recall, precision, f1, auroc, brier, lead_time_h (all floats)."""
    if len(X) == 0:
        return {k: 0.0 for k in METRIC_KEYS} | {"auroc": 0.5, "brier": BRIER_CAP}
    prob = np.clip(_positive_proba(model, X), 0.0, 1.0)
    pred = (prob >= ALARM_THRESHOLD).astype(int)
    metrics = {
        "recall": float(recall_score(y, pred, zero_division=0)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "auroc": float(roc_auc_score(y, prob)) if len(np.unique(y)) == 2 else 0.5,
        "brier": float(brier_score_loss(y, prob)),
        "lead_time_h": _lead_time_hours(meta, prob),
    }
    return metrics


# --------------------------------------------------------------------------------------
# Estimators + search spaces
# --------------------------------------------------------------------------------------


def _suggest_params(trial: optuna.Trial, family: str) -> dict[str, Any]:
    """Small, family-specific Optuna search space."""
    if family == "random_forest":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 200, step=50),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", 0.5, None]),
        }
    if family == "xgboost":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300, step=50),
            "max_depth": trial.suggest_int("max_depth", 2, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.3, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        }
    if family == "lightgbm":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300, step=50),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "num_leaves": trial.suggest_int("num_leaves", 7, 63),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.3, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        }
    raise ValueError(f"unknown family {family!r}; expected one of {FAMILIES}")


def _make_estimator(family: str, params: dict[str, Any], scale_pos_weight: float, seed: int) -> Any:
    """Instantiate the estimator with imbalance handling and a fixed seed."""
    if family == "random_forest":
        from sklearn.ensemble import RandomForestClassifier

        return RandomForestClassifier(
            **params, class_weight="balanced_subsample", random_state=seed, n_jobs=1
        )
    if family == "xgboost":
        from xgboost import XGBClassifier

        return XGBClassifier(
            **params,
            scale_pos_weight=scale_pos_weight,
            tree_method="hist",
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=seed,
            n_jobs=1,
            verbosity=0,
        )
    if family == "lightgbm":
        from lightgbm import LGBMClassifier

        return LGBMClassifier(
            **params,
            scale_pos_weight=scale_pos_weight,
            subsample_freq=1,
            random_state=seed,
            n_jobs=1,
            verbose=-1,
        )
    raise ValueError(f"unknown family {family!r}; expected one of {FAMILIES}")


def _fixed_params(family: str, scale_pos_weight: float, seed: int) -> dict[str, Any]:
    if family == "random_forest":
        return {"class_weight": "balanced_subsample", "random_state": seed}
    return {"scale_pos_weight": round(scale_pos_weight, 4), "random_state": seed}


# --------------------------------------------------------------------------------------
# MLflow (best effort)
# --------------------------------------------------------------------------------------


def _tracking_uri() -> str:
    uri = get_settings().mlflow_tracking_uri
    if "://" in uri:
        return uri
    return Path(uri).resolve().as_uri()


def _log_to_mlflow(
    experiment: str,
    run_name: str,
    params: dict[str, Any],
    metrics: dict[str, float],
    artifact: Path | None,
    tags: dict[str, str],
) -> str | None:
    """Log one candidate to MLflow. Returns the run id, or ``None`` (with a warning) on failure."""
    os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "1")
    os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "5")
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    # MLflow >= 3.x refuses the local file store (the settings default "./mlruns") unless
    # explicitly allowed; the compose stack uses an http tracking server instead.
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    try:
        import mlflow

        mlflow.set_tracking_uri(_tracking_uri())
        mlflow.set_experiment(experiment)
        with mlflow.start_run(run_name=run_name) as run:
            mlflow.log_params({k: str(v) for k, v in params.items()})
            mlflow.log_metrics({k: float(v) for k, v in metrics.items() if np.isfinite(v)})
            mlflow.set_tags(tags)
            if artifact is not None and artifact.exists():
                mlflow.log_artifact(str(artifact))
            return str(run.info.run_id)
    except Exception as exc:  # noqa: BLE001 - any MLflow failure must degrade to a warning
        warnings.warn(f"MLflow logging skipped ({type(exc).__name__}: {exc})", stacklevel=2)
        logger.warning("MLflow logging skipped: %s", exc)
        return None


# --------------------------------------------------------------------------------------
# launch_trial / compare_models / train_candidates
# --------------------------------------------------------------------------------------


def _default_artifacts_dir() -> Path:
    return Path(get_settings().models_dir) / "candidates"


def launch_trial(
    features_df: pd.DataFrame,
    spec: FeatureSpec,
    task: TaskSpec,
    family: str,
    n_trials: int = 6,
    seed: int = 0,
    artifacts_dir: Path | None = None,
    experiment: str | None = None,
) -> CandidateModel:
    """Run an Optuna search for one model family and persist the best model.

    Input: ``features_df`` from ``build_feature_pipeline`` (``asset_id, ts, features, label,
    rul_h``), its ``spec``, the ``task`` (classification), ``family`` in
    ``("random_forest", "xgboost", "lightgbm")``, number of Optuna trials, RNG seed, directory
    for the joblib artifact (default ``<models_dir>/candidates``) and an MLflow experiment name.
    Split: the last 25 % of assets by id are held out (``split_by_asset``); rows stay in time
    order; no shuffling. Objective: ``industrial_model_score`` on the held-out assets
    (recall, precision, f1, auroc, brier, lead_time_h + latency in ms per 1000 rows), maximised
    with ``TPESampler(seed)``. Class imbalance: ``class_weight="balanced_subsample"`` (RF) or
    ``scale_pos_weight = n_neg / n_pos`` (XGB/LGBM).
    Output: ``CandidateModel`` whose artifact (fitted on training assets only) is saved to
    ``artifacts_dir/<candidate_id>.joblib`` and exposes ``predict_proba(X)`` for ``X`` with
    exactly ``spec.features`` columns. ``metrics`` holds the holdout metrics of the best trial;
    ``params`` the best hyperparameters plus the fixed ones. MLflow logging is best effort.
    Raises ``ValueError`` if the training assets contain a single class or the family is unknown.
    """
    if family not in FAMILIES:
        raise ValueError(f"unknown family {family!r}; expected one of {FAMILIES}")
    if task.task != "classification":
        raise ValueError("launch_trial only supports classification tasks")
    feats = list(spec.features)
    missing = [c for c in feats + ["asset_id", "ts", "label", "rul_h"] if c not in features_df.columns]
    if missing:
        raise ValueError(f"features_df is missing columns: {missing}")

    train_assets, hold_assets = split_by_asset(features_df)
    df = features_df.sort_values(["asset_id", "ts"]).reset_index(drop=True)
    tr = df[df["asset_id"].isin(train_assets)]
    ho = df[df["asset_id"].isin(hold_assets)]
    X_tr, y_tr = tr[feats], tr["label"].to_numpy(dtype=int)
    X_ho, y_ho = ho[feats], ho["label"].to_numpy(dtype=int)
    meta_ho = ho[["asset_id", "ts", "rul_h"]].reset_index(drop=True)
    if len(np.unique(y_tr)) < 2:
        raise ValueError(
            "training assets contain a single class; need at least one failing asset among "
            f"{train_assets}"
        )
    n_pos = int(y_tr.sum())
    scale_pos_weight = float(max(1.0, (len(y_tr) - n_pos) / max(1, n_pos)))

    def _fit_and_score(params: dict[str, Any]) -> tuple[Any, dict[str, float], float]:
        model = _make_estimator(family, params, scale_pos_weight, seed)
        model.fit(X_tr, y_tr)
        metrics = _evaluate(model, X_ho, y_ho, meta_ho)
        latency = _latency_ms_per_1000(model, X_ho)
        return model, metrics, latency

    def objective(trial: optuna.Trial) -> float:
        params = _suggest_params(trial, family)
        _, metrics, latency = _fit_and_score(params)
        ims = industrial_model_score(metrics, latency)
        for k, v in metrics.items():
            trial.set_user_attr(k, float(v))
        trial.set_user_attr("latency_ms", float(latency))
        return float(ims.total)

    study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=seed))
    study.optimize(objective, n_trials=max(1, int(n_trials)), show_progress_bar=False)
    best_params = dict(study.best_trial.params)

    model, metrics, latency = _fit_and_score(best_params)
    ims = industrial_model_score(metrics, latency)

    candidate_id = f"{family}-{uuid.uuid4().hex[:8]}"
    out_dir = Path(artifacts_dir) if artifacts_dir is not None else _default_artifacts_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = out_dir / f"{candidate_id}.joblib"
    joblib.dump(model, artifact_path)

    all_params = {**best_params, **_fixed_params(family, scale_pos_weight, seed)}
    run_id = _log_to_mlflow(
        experiment or DEFAULT_EXPERIMENT,
        candidate_id,
        {**all_params, "family": family, "n_trials": n_trials, "window_rows": spec.window_rows,
         "horizon_h": task.horizon_h, "train_assets": ",".join(train_assets),
         "holdout_assets": ",".join(hold_assets)},
        {**metrics, "latency_ms": latency, "ims_total": ims.total},
        artifact_path,
        {"family": family, "candidate_id": candidate_id},
    )
    return CandidateModel(
        candidate_id=candidate_id,
        family=family,  # type: ignore[arg-type]
        mlflow_run_id=run_id,
        params=all_params,
        metrics={k: float(v) for k, v in metrics.items()},
        ims=ims,
        artifact_path=str(artifact_path),
        inference_latency_ms=float(latency),
    )


def compare_models(candidates: list[CandidateModel]) -> list[str]:
    """Rank candidates by Industrial Model Score.

    Input: list of ``CandidateModel``. Output: ``candidate_id`` list best -> worst by
    ``ims.total`` (candidates without an IMS sort last; ties broken by lower latency).
    """
    def _key(c: CandidateModel) -> tuple[float, float]:
        total = c.ims.total if c.ims is not None else float("-inf")
        return (-total, c.inference_latency_ms)

    return [c.candidate_id for c in sorted(candidates, key=_key)]


def train_candidates(
    features_df: pd.DataFrame,
    spec: FeatureSpec,
    task: TaskSpec,
    n_trials: int = 6,
    seed: int = 0,
    artifacts_dir: Path | None = None,
    asset_id: str = "",
) -> CandidateSet:
    """Train one candidate per family and rank them.

    Input: as ``launch_trial`` plus the target ``asset_id`` (used for the MLflow experiment name
    ``thor-automl[-<asset_id>]`` and ``CandidateSet.asset_id``).
    Output: ``CandidateSet`` with three ``CandidateModel`` entries (random_forest, xgboost,
    lightgbm) and ``ranked`` = ``compare_models(candidates)``. A family whose search raises is
    skipped with a warning so one broken library never blocks the run; at least one candidate
    must succeed or ``RuntimeError`` is raised.
    """
    experiment = f"{DEFAULT_EXPERIMENT}-{asset_id}" if asset_id else DEFAULT_EXPERIMENT
    candidates: list[CandidateModel] = []
    errors: list[str] = []
    for family in FAMILIES:
        try:
            candidates.append(
                launch_trial(
                    features_df,
                    spec,
                    task,
                    family,
                    n_trials=n_trials,
                    seed=seed,
                    artifacts_dir=artifacts_dir,
                    experiment=experiment,
                )
            )
        except ValueError:
            raise
        except Exception as exc:  # noqa: BLE001 - isolate a single broken family
            logger.warning("family %s failed: %s", family, exc)
            errors.append(f"{family}: {exc}")
    if not candidates:
        raise RuntimeError("no candidate could be trained: " + "; ".join(errors))
    return CandidateSet(
        asset_id=asset_id,
        task=task,
        features=spec,
        candidates=candidates,
        mlflow_experiment=experiment,
        ranked=compare_models(candidates),
    )
