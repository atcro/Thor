"""06 Validation toolbox: leakage checks, asset-level backtests, calibration, lead time, RUL.

Every function here is deterministic local compute. Nothing calls an LLM and nothing reads
`ANTHROPIC_API_KEY`. The orchestrator calls `validate()`; the individual checks are exposed
so the AutoML Studio screen and the tests can call them one at a time.

Splitting policy (CLAUDE.md section 7): there is no random shuffle split anywhere in this
module. Every held-out set is a set of whole assets the model never saw, and rows inside a
fold are kept in time order.
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import (
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from apps.api.schemas import (
    BacktestFold,
    CalibrationReport,
    CandidateModel,
    CandidateSet,
    FeatureSpec,
    IndustrialModelScore,
    LeadTimeReport,
    LeakageReport,
    RULInterval,
    ValidationReport,
)

log = logging.getLogger(__name__)

FORBIDDEN_FEATURES: frozenset[str] = frozenset(
    {"health", "failure_within_h", "rul_h", "label", "ts", "asset_id", "regime"}
)
LEAKAGE_CORR_LIMIT: float = 0.98
SUSTAINED_SAMPLES: int = 3
ISOTONIC_MIN_ROWS: int = 200
MIN_RECALL_TO_PASS: float = 0.6
MIN_LEAD_TIME_H_TO_PASS: float = 12.0
HOLDOUT_ASSET_FRACTION: float = 0.25

IMS_WEIGHTS: dict[str, float] = {
    "recall": 0.35,
    "lead_time_score": 0.25,
    "precision": 0.15,
    "calibration_score": 0.15,
    "latency_score": 0.10,
}


# --------------------------------------------------------------------------------------
# Private helpers
# --------------------------------------------------------------------------------------


def _nan_to(value: float, default: float = 0.0) -> float:
    """Return `default` when `value` is NaN or infinite, else `value` as float."""
    v = float(value)
    return default if (math.isnan(v) or math.isinf(v)) else v


def _industrial_model_score(metrics: dict[str, float], latency_ms: float) -> IndustrialModelScore:
    """Local copy of the IMS formula so this module does not import agents.ml_architect.

    Inputs: `metrics` with keys recall, precision, lead_time_h, brier (missing keys count as
    the worst value) and `latency_ms`. Output: IndustrialModelScore with the schema-default
    weights. lead_time_score = min(1, lead_time_h / 72); calibration_score = 1 - min(1, brier /
    0.25); latency_score = 1 - min(1, latency_ms / 50).
    """
    recall = min(1.0, max(0.0, _nan_to(metrics.get("recall", 0.0))))
    precision = min(1.0, max(0.0, _nan_to(metrics.get("precision", 0.0))))
    lead_time_h = max(0.0, _nan_to(metrics.get("lead_time_h", 0.0)))
    brier = max(0.0, _nan_to(metrics.get("brier", 0.25), default=0.25))
    lead_time_score = min(1.0, lead_time_h / 72.0)
    calibration_score = 1.0 - min(1.0, brier / 0.25)
    latency_score = 1.0 - min(1.0, max(0.0, _nan_to(latency_ms)) / 50.0)
    parts = {
        "recall": recall,
        "precision": precision,
        "lead_time_score": lead_time_score,
        "calibration_score": calibration_score,
        "latency_score": latency_score,
    }
    total = sum(IMS_WEIGHTS[k] * parts[k] for k in IMS_WEIGHTS)
    return IndustrialModelScore(**parts, weights=dict(IMS_WEIGHTS), total=float(total))


def _sorted_assets(df: pd.DataFrame) -> list[str]:
    return sorted(str(a) for a in df["asset_id"].unique())


def _split_assets(assets: list[str], holdout_fraction: float = HOLDOUT_ASSET_FRACTION) -> tuple[list[str], list[str]]:
    """Deterministic asset-level split: the last `holdout_fraction` of sorted ids are held out."""
    if len(assets) < 2:
        return list(assets), []
    n_hold = max(1, int(round(len(assets) * holdout_fraction)))
    n_hold = min(n_hold, len(assets) - 1)
    return assets[:-n_hold], assets[-n_hold:]


def _positive_proba(model: Any, X: pd.DataFrame) -> np.ndarray:
    """Return P(label = 1) for each row; tolerates single-column predict_proba outputs."""
    proba = np.asarray(model.predict_proba(X))
    if proba.ndim == 1:
        return proba.astype(float)
    if proba.shape[1] == 1:
        return proba[:, 0].astype(float)
    return proba[:, 1].astype(float)


def _classification_metrics(y_true: np.ndarray, p: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    """recall, precision, f1, auroc, brier for one held-out set. auroc = NaN if one class only."""
    y_pred = (p >= threshold).astype(int)
    out: dict[str, float] = {
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "brier": float(brier_score_loss(y_true, p)),
        "n_test": float(len(y_true)),
        "n_pos_test": float(int(y_true.sum())),
    }
    if len(np.unique(y_true)) == 2:
        out["auroc"] = float(roc_auc_score(y_true, p))
    else:
        out["auroc"] = float("nan")
    return out


def _failure_ts(asset_df: pd.DataFrame) -> pd.Timestamp | None:
    """Failure instant for one asset: last row's ts + rul_h; None if the asset never fails."""
    rul = asset_df["rul_h"]
    mask = rul.notna()
    if not mask.any():
        return None
    last = asset_df.loc[mask].iloc[-1]
    return pd.Timestamp(last["ts"]) + pd.Timedelta(hours=float(last["rul_h"]))


def _lead_times_h(df: pd.DataFrame, spec: FeatureSpec, model: Any, threshold: float) -> list[float]:
    """Per failing asset: hours from the first sustained alarm (3 consecutive rows with
    P >= threshold) to failure. A missed event contributes 0.0."""
    leads: list[float] = []
    for _asset, g in df.sort_values(["asset_id", "ts"]).groupby("asset_id", sort=True):
        fail_ts = _failure_ts(g)
        if fail_ts is None:
            continue
        p = _positive_proba(model, g[spec.features])
        hot = p >= threshold
        lead = 0.0
        for i in range(len(hot) - SUSTAINED_SAMPLES + 1):
            if bool(hot[i : i + SUSTAINED_SAMPLES].all()):
                first_ts = pd.Timestamp(g["ts"].iloc[i])
                lead = max(0.0, (fail_ts - first_ts).total_seconds() / 3600.0)
                break
        leads.append(float(lead))
    return leads


def _reliability_bins(y_true: np.ndarray, p: np.ndarray, n_bins: int = 10) -> list[tuple[float, float, int]]:
    """(mean_predicted, fraction_positive, count) per non-empty equal-width probability bin."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, n_bins - 1)
    bins: list[tuple[float, float, int]] = []
    for b in range(n_bins):
        m = idx == b
        if m.any():
            bins.append((float(p[m].mean()), float(y_true[m].mean()), int(m.sum())))
    return bins


class _IdentityCalibrator:
    """Pass-through wrapper used when calibration is impossible (one class in y_cal)."""

    def __init__(self, estimator: Any) -> None:
        self.estimator = estimator

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.estimator.predict_proba(X))

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (_positive_proba(self.estimator, X) >= 0.5).astype(int)


def _model_factory_for(model: Any) -> Callable[[], Any]:
    """Return a factory that builds an unfitted copy of `model` (sklearn clone, deepcopy fallback)."""

    def factory() -> Any:
        try:
            return clone(model)
        except Exception:  # noqa: BLE001 - clone fails on non-sklearn wrappers
            import copy

            return copy.deepcopy(model)

    return factory


# --------------------------------------------------------------------------------------
# Public tool functions
# --------------------------------------------------------------------------------------


def detect_leakage(
    features_df: pd.DataFrame,
    spec: FeatureSpec,
    train_assets: list[str],
    test_assets: list[str],
) -> LeakageReport:
    """Check a proposed train/test asset split for leakage.

    Inputs: the feature frame (asset_id, ts, features, label, rul_h), the FeatureSpec, and the
    asset ids assigned to train and test. Outputs a LeakageReport with:
      - asset_overlap: any asset in both lists;
      - temporal_leakage: for an overlapping asset, test rows dated on or before that asset's
        last train row (with disjoint assets this cannot occur; concurrent time on different
        machines is a legitimate asset-level holdout);
      - feature_leakage: features with a forbidden name (ground truth or label columns) or
        |corr(feature, label)| > 0.98 on the train rows;
      - passed: none of the above.
    """
    notes: list[str] = []
    train_set, test_set = set(train_assets), set(test_assets)
    overlap = sorted(train_set & test_set)
    asset_overlap = len(overlap) > 0
    if asset_overlap:
        notes.append(f"assets present in both train and test: {overlap}")

    temporal = False
    for asset in overlap:
        rows = features_df.loc[features_df["asset_id"] == asset]
        if rows.empty:
            continue
        train_max = rows["ts"].max()
        if bool((rows["ts"] <= train_max).any()):
            temporal = True
            notes.append(f"{asset}: test rows are not strictly after its last train row")
            break

    feature_leakage: list[str] = []
    for f in spec.features:
        if f in FORBIDDEN_FEATURES:
            feature_leakage.append(f)
            notes.append(f"forbidden feature name: {f}")
    train_rows = features_df.loc[features_df["asset_id"].isin(train_set)]
    if train_rows.empty:
        train_rows = features_df
    if "label" in train_rows.columns and train_rows["label"].nunique() > 1:
        y = train_rows["label"].astype(float)
        for f in spec.features:
            if f in feature_leakage or f not in train_rows.columns:
                continue
            x = pd.to_numeric(train_rows[f], errors="coerce")
            if x.nunique(dropna=True) < 2:
                continue
            corr = _nan_to(x.corr(y))
            if abs(corr) > LEAKAGE_CORR_LIMIT:
                feature_leakage.append(f)
                notes.append(f"{f}: |corr with label| = {abs(corr):.3f} > {LEAKAGE_CORR_LIMIT}")
    missing = [f for f in spec.features if f not in features_df.columns]
    if missing:
        notes.append(f"features missing from frame: {missing}")

    passed = not asset_overlap and not temporal and not feature_leakage and not missing
    return LeakageReport(
        temporal_leakage=temporal,
        feature_leakage=feature_leakage,
        asset_overlap=asset_overlap,
        passed=passed,
        notes=notes,
    )


def run_backtest(
    features_df: pd.DataFrame,
    spec: FeatureSpec,
    model_factory: Callable[[], Any],
    n_folds: int = 3,
    threshold: float = 0.5,
) -> list[BacktestFold]:
    """Asset-level, time-ordered backtest.

    Inputs: feature frame, FeatureSpec, a zero-arg factory returning an unfitted estimator
    with fit/predict_proba, the number of folds, and the alarm threshold. Assets are sorted
    and dealt round-robin into `n_folds` held-out groups; fold i trains on every other asset's
    rows (time-ordered, all dated <= train_end) and scores the held-out assets, which the
    model has never seen. Output: one BacktestFold per fold with metrics recall, precision,
    f1, auroc (NaN if the held-out set has one class), brier, n_test, n_pos_test and
    lead_time_h (median warning lead time over failing held-out assets; NaN if none fail).
    With a single asset the fallback is a time-ordered 70/30 split of that asset.
    """
    df = features_df.sort_values(["asset_id", "ts"]).reset_index(drop=True)
    assets = _sorted_assets(df)
    folds: list[BacktestFold] = []

    if len(assets) < 2:
        cut = int(len(df) * 0.7)
        train, test = df.iloc[:cut], df.iloc[cut:]
        if train.empty or test.empty or train["label"].nunique() < 2:
            return folds
        model = model_factory()
        model.fit(train[spec.features], train["label"].astype(int))
        p = _positive_proba(model, test[spec.features])
        metrics = _classification_metrics(test["label"].to_numpy().astype(int), p, threshold)
        leads = _lead_times_h(test, spec, model, threshold)
        metrics["lead_time_h"] = float(np.median(leads)) if leads else float("nan")
        folds.append(
            BacktestFold(
                fold=0,
                train_assets=assets,
                test_assets=assets,
                train_end=pd.Timestamp(train["ts"].max()).to_pydatetime(),
                metrics=metrics,
            )
        )
        return folds

    k = max(2, min(n_folds, len(assets)))
    groups = [assets[i::k] for i in range(k)]
    for i, test_assets in enumerate(groups):
        train_assets = [a for a in assets if a not in test_assets]
        train = df.loc[df["asset_id"].isin(train_assets)]
        test = df.loc[df["asset_id"].isin(test_assets)]
        train_end = pd.Timestamp(train["ts"].max())
        train = train.loc[train["ts"] <= train_end]
        if train["label"].nunique() < 2:
            log.warning("backtest fold %d skipped: training rows contain one class only", i)
            continue
        model = model_factory()
        model.fit(train[spec.features], train["label"].astype(int))
        p = _positive_proba(model, test[spec.features])
        metrics = _classification_metrics(test["label"].to_numpy().astype(int), p, threshold)
        leads = _lead_times_h(test, spec, model, threshold)
        metrics["lead_time_h"] = float(np.median(leads)) if leads else float("nan")
        folds.append(
            BacktestFold(
                fold=i,
                train_assets=train_assets,
                test_assets=list(test_assets),
                train_end=train_end.to_pydatetime(),
                metrics=metrics,
            )
        )
    return folds


def calibrate_probabilities(
    model: Any, X_cal: pd.DataFrame, y_cal: pd.Series
) -> tuple[Any, CalibrationReport]:
    """Fit a probability calibrator on top of an already-fitted classifier.

    Inputs: fitted model with predict_proba, calibration features (exactly the FeatureSpec
    columns) and labels from assets the model was not trained on. Isotonic regression when
    there are >= 200 calibration rows, Platt sigmoid otherwise. Output: (calibrated wrapper
    exposing predict_proba, CalibrationReport with Brier score before/after and 10-bin
    reliability data). If y_cal has a single class calibration is impossible; the model is
    returned inside a pass-through wrapper with method="none".
    """
    y = np.asarray(y_cal).astype(int)
    p_before = _positive_proba(model, X_cal)
    brier_before = float(brier_score_loss(y, p_before)) if len(y) else float("nan")
    if len(np.unique(y)) < 2 or len(y) < 10:
        log.warning("calibration skipped: %d rows, %d classes", len(y), len(np.unique(y)))
        wrapper = _IdentityCalibrator(model)
        return wrapper, CalibrationReport(
            method="none",
            brier_before=_nan_to(brier_before, 0.25),
            brier_after=_nan_to(brier_before, 0.25),
            reliability_bins=_reliability_bins(y, p_before) if len(y) else [],
        )
    method = "isotonic" if len(y) >= ISOTONIC_MIN_ROWS else "sigmoid"
    calibrated = CalibratedClassifierCV(FrozenEstimator(model), method=method)
    calibrated.fit(X_cal, y)
    p_after = _positive_proba(calibrated, X_cal)
    return calibrated, CalibrationReport(
        method=method,
        brier_before=brier_before,
        brier_after=float(brier_score_loss(y, p_after)),
        reliability_bins=_reliability_bins(y, p_after),
    )


def compute_lead_time(
    features_df: pd.DataFrame, spec: FeatureSpec, model: Any, threshold: float = 0.5
) -> LeadTimeReport:
    """Warning lead time: how far ahead of failure the model raises a sustained alarm.

    Inputs: feature frame with rul_h, FeatureSpec, a fitted model, alarm threshold. For each
    asset that fails (rul_h not null), the alarm instant is the first of 3 consecutive rows
    with P(failure) >= threshold; lead time = failure instant (last ts + rul_h) minus alarm
    instant, in hours; a never-alarmed failing asset counts as 0 h. Output: LeadTimeReport
    with median / p10 / p90 over events, n_events and the threshold used. All zeros if no
    asset in the frame fails.
    """
    leads = _lead_times_h(features_df, spec, model, threshold)
    if not leads:
        return LeadTimeReport(median_h=0.0, p90_h=0.0, p10_h=0.0, n_events=0, threshold=threshold)
    arr = np.asarray(leads, dtype=float)
    return LeadTimeReport(
        median_h=float(np.median(arr)),
        p90_h=float(np.percentile(arr, 90)),
        p10_h=float(np.percentile(arr, 10)),
        n_events=int(len(arr)),
        threshold=threshold,
    )


def estimate_rul_interval(
    features_df: pd.DataFrame,
    spec: FeatureSpec,
    seed: int = 0,
    asset_id: str | None = None,
) -> RULInterval:
    """Remaining-useful-life interval from quantile gradient boosting.

    Inputs: feature frame (rows with rul_h not null are the training set), FeatureSpec, seed,
    and optionally the asset to score. Three GradientBoostingRegressor models with
    loss="quantile" at alpha 0.1 / 0.5 / 0.9 are fitted; when more than one asset fails and
    `asset_id` is given, that asset's rows are excluded from training so the interval is
    out-of-sample. The interval is predicted on the latest row of `asset_id` (or, if not
    given, the latest row of the failing asset with the lowest median RUL). Output:
    RULInterval(p10_h, p50_h, p90_h) sorted so p10 <= p50 <= p90. Raises ValueError when
    fewer than 10 rows carry a RUL label.
    """
    df = features_df.sort_values(["asset_id", "ts"])
    labelled = df.loc[df["rul_h"].notna()]
    if len(labelled) < 10:
        raise ValueError("estimate_rul_interval needs at least 10 rows with rul_h")
    train = labelled
    if asset_id is not None and labelled["asset_id"].nunique() > 1:
        others = labelled.loc[labelled["asset_id"] != asset_id]
        if len(others) >= 10:
            train = others

    models = {}
    for alpha in (0.1, 0.5, 0.9):
        m = GradientBoostingRegressor(
            loss="quantile", alpha=alpha, n_estimators=150, max_depth=3, random_state=seed
        )
        m.fit(train[spec.features], train["rul_h"].astype(float))
        models[alpha] = m

    if asset_id is not None and (df["asset_id"] == asset_id).any():
        target = df.loc[df["asset_id"] == asset_id].tail(1)
    else:
        latest = labelled.groupby("asset_id", sort=True).tail(1)
        p50 = models[0.5].predict(latest[spec.features])
        target = latest.iloc[[int(np.argmin(p50))]]
    q = sorted(max(0.0, float(models[a].predict(target[spec.features])[0])) for a in (0.1, 0.5, 0.9))
    return RULInterval(p10_h=q[0], p50_h=q[1], p90_h=q[2], method="quantile_gbm")


def _load_candidate(c: CandidateModel) -> Any | None:
    if not c.artifact_path:
        log.warning("candidate %s has no artifact_path; skipped", c.candidate_id)
        return None
    path = Path(c.artifact_path)
    if not path.exists():
        log.warning("candidate %s artifact missing at %s; skipped", c.candidate_id, path)
        return None
    try:
        return joblib.load(path)
    except Exception as e:  # noqa: BLE001 - a corrupt artifact must not kill validation
        log.warning("candidate %s artifact failed to load: %s", c.candidate_id, e)
        return None


def _mean_metric(folds: list[BacktestFold], key: str) -> float:
    vals = [f.metrics.get(key, float("nan")) for f in folds]
    vals = [v for v in vals if not math.isnan(v)]
    return float(np.mean(vals)) if vals else float("nan")


def validate(candidates: CandidateSet, features_df: pd.DataFrame, run_id: str) -> ValidationReport:
    """Run every validation check on a CandidateSet and pick the champion.

    Inputs: CandidateSet (candidates must have artifact_path set), the feature frame the
    candidates were trained from, and the run id. Steps: leakage check on the deterministic
    asset split (last 25 percent of sorted assets held out); asset-level backtest per
    candidate (fresh clones fitted per fold); IMS recomputed from mean backtest metrics and
    the candidate's inference latency; champion = highest IMS; champion calibrated on the
    held-out assets and saved as <candidate_id>_calibrated.joblib next to the original;
    lead time measured with the calibrated champion on the held-out assets (falls back to
    the whole frame if no held-out asset fails); RUL interval (None on failure).
    Output: ValidationReport. `passed` = leakage passed and champion recall >= 0.6 and
    median lead time >= 12 h. notes[0] is always "champion_artifact=<path>" because the
    schema has no field for the calibrated artifact path.
    """
    spec = candidates.features
    df = features_df.sort_values(["asset_id", "ts"]).reset_index(drop=True)
    assets = _sorted_assets(df)
    train_assets, test_assets = _split_assets(assets)
    notes: list[str] = []

    leakage = detect_leakage(df, spec, train_assets, test_assets)

    scored: list[tuple[CandidateModel, Any, list[BacktestFold], IndustrialModelScore]] = []
    for c in candidates.candidates:
        model = _load_candidate(c)
        if model is None:
            continue
        folds = run_backtest(df, spec, _model_factory_for(model), n_folds=3)
        metrics = {
            "recall": _mean_metric(folds, "recall"),
            "precision": _mean_metric(folds, "precision"),
            "lead_time_h": _mean_metric(folds, "lead_time_h"),
            "brier": _mean_metric(folds, "brier"),
        }
        ims = _industrial_model_score(metrics, c.inference_latency_ms)
        scored.append((c, model, folds, ims))
        log.info("candidate %s (%s): IMS %.3f", c.candidate_id, c.family, ims.total)
    if not scored:
        raise ValueError("validate: no candidate could be loaded from artifact_path")

    champion, model, folds, ims = max(scored, key=lambda t: t[3].total)

    cal_rows = df.loc[df["asset_id"].isin(test_assets)] if test_assets else df
    if cal_rows.empty:
        cal_rows = df
    calibrated, calibration = calibrate_probabilities(
        model, cal_rows[spec.features], cal_rows["label"].astype(int)
    )
    original = Path(champion.artifact_path or f"{champion.candidate_id}.joblib")
    cal_path = original.with_name(f"{champion.candidate_id}_calibrated.joblib")
    cal_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(calibrated, cal_path)
    notes.append(f"champion_artifact={cal_path}")

    lead_time = compute_lead_time(cal_rows, spec, calibrated)
    if lead_time.n_events == 0:
        lead_time = compute_lead_time(df, spec, calibrated)
        notes.append("lead time measured on the full frame: no held-out asset fails")
    else:
        notes.append(f"lead time measured on held-out assets {test_assets}")

    rul: RULInterval | None
    try:
        rul = estimate_rul_interval(df, spec, asset_id=candidates.asset_id)
    except Exception as e:  # noqa: BLE001 - RUL is optional evidence
        rul = None
        notes.append(f"rul interval unavailable: {e}")

    passed = (
        leakage.passed
        and ims.recall >= MIN_RECALL_TO_PASS
        and lead_time.median_h >= MIN_LEAD_TIME_H_TO_PASS
    )
    if not passed:
        notes.append(
            f"gate: leakage_ok={leakage.passed} recall={ims.recall:.2f} "
            f"lead_time_median_h={lead_time.median_h:.1f}"
        )
    notes.append(
        "ranking: "
        + ", ".join(f"{c.candidate_id}={s.total:.3f}" for c, _, _, s in sorted(scored, key=lambda t: -t[3].total))
    )

    return ValidationReport(
        asset_id=candidates.asset_id,
        champion_id=champion.candidate_id,
        champion_family=champion.family,
        champion_mlflow_run_id=champion.mlflow_run_id,
        model_version=f"v{int(time.time()) % 100000}",
        leakage=leakage,
        backtest=folds,
        calibration=calibration,
        lead_time=lead_time,
        rul=rul,
        ims=ims,
        passed=passed,
        notes=notes,
    )


def champion_artifact_path(report: ValidationReport) -> Path | None:
    """Read the calibrated champion path back out of ValidationReport.notes.

    Input: a ValidationReport produced by validate(). Output: Path from the
    "champion_artifact=<path>" note, or None if absent. Convenience for 07 and 08.
    """
    for note in report.notes:
        if note.startswith("champion_artifact="):
            return Path(note.split("=", 1)[1])
    return None


def train_end_of(folds: list[BacktestFold]) -> datetime | None:
    """Latest train_end across folds (None if no folds). Used by ModelOps for dataset stamps."""
    return max((f.train_end for f in folds), default=None)
