"""06 Validation toolbox: leakage checks, asset-level backtests, calibration, lead time, RUL.

Every function here is deterministic local compute. Nothing calls an LLM and nothing reads
`ANTHROPIC_API_KEY`. The orchestrator calls `validate()`; the individual checks are exposed
so the AutoML Studio screen and the tests can call them one at a time.

Splitting policy (CLAUDE.md section 7): there is no random shuffle split anywhere in this
module. Every held-out set is a set of whole assets the model never saw, and rows inside a
fold are kept in time order. `validate()` uses the run's recorded `AssetSplit`
(agents/ml_architect/split.py): champion selection on the validation assets, calibration on
the calibration assets, final metrics on the untouched test assets.
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
from sklearn.metrics import brier_score_loss

from agents.metrics import (
    IMS_WEIGHTS,
    classification_metrics,
    industrial_model_score,
    lead_times_h,
    positive_proba,
)
from agents.ml_architect.split import make_asset_split, rows_in
from apps.api.schemas import (
    CLASSIFICATION_THRESHOLD,
    EARLY_WARNING_SUSTAINED_ROWS,
    EARLY_WARNING_THRESHOLD,
    AssetSplit,
    BacktestFold,
    CalibrationReport,
    CandidateModel,
    CandidateSet,
    FeatureSpec,
    GateCheck,
    IndustrialModelScore,
    LeadTimeReport,
    LeakageReport,
    RULInterval,
    Thresholds,
    ValidationReport,
)

log = logging.getLogger(__name__)

FORBIDDEN_FEATURES: frozenset[str] = frozenset(
    {"health", "failure_within_h", "rul_h", "label", "ts", "asset_id", "regime"}
)
LEAKAGE_CORR_LIMIT: float = 0.98
# Thresholds come from apps.api.schemas (one meaning each; see agents/metrics.py):
#   CLASSIFICATION_THRESHOLD  -> recall / precision / F1
#   EARLY_WARNING_THRESHOLD   -> the alarm that defines warning lead time (sustained rows)
ISOTONIC_MIN_ROWS: int = 200
#: Isotonic regression needs several independent failure events to shape its step function;
#: with fewer failing motors than this among the calibration assets it degenerates into a
#: near-step (probability jumps from ~0 to ~1 within a sample), so Platt scaling (two
#: parameters, smooth) is used instead.
ISOTONIC_MIN_POSITIVE_ASSETS: int = 2
MIN_RECALL_TO_PASS: float = 0.6
MIN_LEAD_TIME_H_TO_PASS: float = 12.0
HOLDOUT_ASSET_FRACTION: float = 0.25

__all__ = [
    "CLASSIFICATION_THRESHOLD",
    "EARLY_WARNING_SUSTAINED_ROWS",
    "EARLY_WARNING_THRESHOLD",
    "IMS_WEIGHTS",
    "apply_calibration",
    "calibrate_probabilities",
    "calibration_transform",
    "champion_artifact_path",
    "compute_lead_time",
    "detect_leakage",
    "estimate_rul_interval",
    "run_backtest",
    "train_end_of",
    "validate",
]


# --------------------------------------------------------------------------------------
# Private helpers
# --------------------------------------------------------------------------------------


def _nan_to(value: float, default: float = 0.0) -> float:
    """Return `default` when `value` is NaN or infinite, else `value` as float."""
    v = float(value)
    return default if (math.isnan(v) or math.isinf(v)) else v


def _industrial_model_score(metrics: dict[str, float], latency_ms: float) -> IndustrialModelScore:
    """The shared IMS (agents.metrics.industrial_model_score); kept under this name for callers."""
    return industrial_model_score(metrics, latency_ms)


def _sorted_assets(df: pd.DataFrame) -> list[str]:
    return sorted(str(a) for a in df["asset_id"].unique())


def _split_assets(
    assets: list[str], holdout_fraction: float = HOLDOUT_ASSET_FRACTION
) -> tuple[list[str], list[str]]:
    """Deterministic asset-level split: the last `holdout_fraction` of sorted ids are held out."""
    if len(assets) < 2:
        return list(assets), []
    n_hold = max(1, int(round(len(assets) * holdout_fraction)))
    n_hold = min(n_hold, len(assets) - 1)
    return assets[:-n_hold], assets[-n_hold:]


def _positive_proba(model: Any, X: pd.DataFrame) -> np.ndarray:
    """Return P(label = 1) for each row (shared helper from agents.metrics)."""
    return positive_proba(model, X)


def _classification_metrics(
    y_true: np.ndarray, p: np.ndarray, threshold: float = CLASSIFICATION_THRESHOLD
) -> dict[str, float]:
    """recall, precision, f1, auroc, brier, n_test, n_pos_test at CLASSIFICATION_THRESHOLD."""
    out = classification_metrics(y_true, p, threshold)
    out.pop("threshold", None)
    return out


def _lead_times_h(
    df: pd.DataFrame, spec: FeatureSpec, model: Any, threshold: float = EARLY_WARNING_THRESHOLD
) -> list[float]:
    """Per failing asset: hours from the sustained early-warning alarm to failure (shared
    definition, agents.metrics.lead_times_h). A missed event contributes 0.0."""
    ordered = df.sort_values(["asset_id", "ts"])
    p = positive_proba(model, ordered[spec.features])
    return lead_times_h(ordered[["asset_id", "ts", "rul_h"]], p, threshold)


def _reliability_bins(
    y_true: np.ndarray, p: np.ndarray, n_bins: int = 10
) -> list[tuple[float, float, int]]:
    """(mean_predicted, fraction_positive, count) per non-empty equal-width probability bin."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, n_bins - 1)
    bins: list[tuple[float, float, int]] = []
    for b in range(n_bins):
        m = idx == b
        if m.any():
            bins.append((float(p[m].mean()), float(y_true[m].mean()), int(m.sum())))
    return bins


class _ProbabilityOnly:
    """Expose a fitted classifier to the calibrator through `predict_proba` ONLY.

    sklearn's calibrator prefers `decision_function` when the estimator has one (LightGBM and
    XGBoost log-odds). On a pure leaf LightGBM's raw score can reach +-1e6, which turns Platt
    scaling into a flat line (a ~ 0: every row gets the same probability) and cannot be
    reproduced at the edge, which only has the ONNX probability. Calibrating on the bounded
    probability instead keeps the centre and the edge on the same input for every family.
    `estimator` is the wrapped model (the ONNX export unwraps it).
    """

    def __init__(self, estimator: Any) -> None:
        self.estimator = estimator

    def __getattr__(self, name: str) -> Any:
        if name in ("estimator", "decision_function", "predict_log_proba"):
            raise AttributeError(name)
        return getattr(self.estimator, name)

    def __sklearn_is_fitted__(self) -> bool:
        return True

    def fit(self, X: Any, y: Any, **kw: Any) -> _ProbabilityOnly:  # never refitted (frozen)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.estimator.predict_proba(X))

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (_positive_proba(self.estimator, X) >= CLASSIFICATION_THRESHOLD).astype(int)


class _IdentityCalibrator:
    """Pass-through wrapper used when calibration is impossible (one class in y_cal)."""

    def __init__(self, estimator: Any) -> None:
        self.estimator = estimator

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.estimator.predict_proba(X))

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (_positive_proba(self.estimator, X) >= CLASSIFICATION_THRESHOLD).astype(int)


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
    threshold: float = CLASSIFICATION_THRESHOLD,
    early_warning_threshold: float = EARLY_WARNING_THRESHOLD,
) -> list[BacktestFold]:
    """Asset-level, time-ordered backtest.

    Inputs: feature frame, FeatureSpec, a zero-arg factory returning an unfitted estimator
    with fit/predict_proba, the number of folds, the classification threshold (recall /
    precision / F1) and the early-warning threshold (lead time). Assets are sorted and dealt
    round-robin into `n_folds` held-out groups; fold i trains on every other asset's rows
    (time-ordered, all dated <= train_end) and scores the held-out assets, which the model
    has never seen. Output: one BacktestFold per fold with metrics recall, precision, f1,
    auroc (NaN if the held-out set has one class), brier, n_test, n_pos_test and lead_time_h
    (median warning lead time over failing held-out assets; NaN if none fail).
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
        leads = _lead_times_h(test, spec, model, early_warning_threshold)
        metrics["lead_time_h"] = float(np.median(leads)) if leads else float("nan")
        folds.append(
            BacktestFold(
                fold=0,
                train_assets=assets,
                test_assets=assets,
                train_end=pd.Timestamp(train["ts"].max()).to_pydatetime(),
                metrics=_finite_metrics(metrics),
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
        leads = _lead_times_h(test, spec, model, early_warning_threshold)
        metrics["lead_time_h"] = float(np.median(leads)) if leads else float("nan")
        folds.append(
            BacktestFold(
                fold=i,
                train_assets=train_assets,
                test_assets=list(test_assets),
                train_end=train_end.to_pydatetime(),
                metrics=_finite_metrics(metrics),
            )
        )
    return folds


def calibrate_probabilities(
    model: Any,
    X_cal: pd.DataFrame,
    y_cal: pd.Series,
    n_positive_assets: int | None = None,
) -> tuple[Any, CalibrationReport]:
    """Fit a probability calibrator on top of an already-fitted classifier.

    Inputs: fitted model with predict_proba, calibration features (exactly the FeatureSpec
    columns) and labels from assets the model was not trained on, and optionally the number of
    distinct failing assets among those rows. Method: isotonic regression when there are
    >= ISOTONIC_MIN_ROWS rows AND (when known) >= ISOTONIC_MIN_POSITIVE_ASSETS failing
    assets; Platt sigmoid otherwise -- isotonic fitted on one failure event is a near-step
    function that makes the early-warning and classification thresholds fire together.
    Output: (calibrated wrapper exposing predict_proba, CalibrationReport with Brier score
    before/after and 10-bin reliability data). If y_cal has a single class calibration is
    impossible; the model is returned inside a pass-through wrapper with method="none".
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
            transform={"method": "none"},
        )
    enough_events = n_positive_assets is None or n_positive_assets >= ISOTONIC_MIN_POSITIVE_ASSETS
    method = "isotonic" if len(y) >= ISOTONIC_MIN_ROWS and enough_events else "sigmoid"
    if len(y) >= ISOTONIC_MIN_ROWS and not enough_events:
        log.info(
            "calibration: %d failing asset(s) among calibration rows; using sigmoid, not isotonic",
            n_positive_assets,
        )
    calibrated = CalibratedClassifierCV(FrozenEstimator(_ProbabilityOnly(model)), method=method)
    calibrated.fit(X_cal, y)
    p_after = _positive_proba(calibrated, X_cal)
    return calibrated, CalibrationReport(
        method=method,
        brier_before=brier_before,
        brier_after=float(brier_score_loss(y, p_after)),
        reliability_bins=_reliability_bins(y, p_after),
        transform=calibration_transform(calibrated),
    )


def calibration_transform(model: Any) -> dict[str, Any]:
    """Export a fitted calibrator as an explicit, serialisable transform.

    Input: the (possibly) calibrated champion. Output: {"method": "sigmoid", "input", "a",
    "b"} for Platt scaling (p_cal = 1 / (1 + exp(a * x + b))), {"method": "isotonic", "input",
    "x": [...], "y": [...]} for isotonic regression (linear interpolation of x on the fitted
    thresholds, x clipped to [x_min, x_max] first), or {"method": "none"} when the model is not a
    single-estimator CalibratedClassifierCV (an uncalibrated tree model, or the identity
    wrapper). `input` says what x is: "predict_proba" (the raw P(class 1)) or
    "decision_function" (its log-odds -- what sklearn feeds the calibrator for LightGBM and
    XGBoost). Raises ValueError for a cross-validated calibrator with several folds: that
    averages several base models and cannot be exported as one ONNX graph.
    """
    calibrated = getattr(model, "calibrated_classifiers_", None)
    if not calibrated:
        return {"method": "none"}
    if len(calibrated) != 1:
        raise ValueError(
            f"calibration with {len(calibrated)} folds cannot be exported; "
            "calibrate a prefit (FrozenEstimator) model instead"
        )
    cal = calibrated[0].calibrators[0]
    # sklearn fits the calibrator on decision_function when the base estimator has one
    # (LightGBM / XGBoost log-odds), else on predict_proba[:, 1]; the edge only has the ONNX
    # probability, so it must know which one to reconstruct (see `_calibrator_input`).
    frozen = calibrated[0].estimator
    # Since 2026-10-12 every family is calibrated on predict_proba (see _ProbabilityOnly);
    # older artifacts may still say decision_function and the edge honours either.
    inp = "decision_function" if hasattr(frozen, "decision_function") else "predict_proba"
    if hasattr(cal, "a_") and hasattr(cal, "b_"):
        return {"method": "sigmoid", "input": inp, "a": float(cal.a_), "b": float(cal.b_)}
    if hasattr(cal, "X_thresholds_") and hasattr(cal, "y_thresholds_"):
        x = np.asarray(cal.X_thresholds_, dtype=float)
        y = np.asarray(cal.y_thresholds_, dtype=float)
        return {
            "method": "isotonic",
            "input": inp,
            "x": [float(v) for v in x],
            "y": [float(v) for v in y],
            "x_min": float(getattr(cal, "X_min_", x[0])),
            "x_max": float(getattr(cal, "X_max_", x[-1])),
        }
    raise ValueError(f"unsupported calibrator {type(cal).__name__}")


def _calibrator_input(p: np.ndarray, transform: dict[str, Any]) -> np.ndarray:
    """What the calibrator was fitted on: the raw P(class 1), or -- when the base estimator
    exposes `decision_function` (LightGBM, XGBoost log-odds) -- its logit, which sklearn's
    CalibratedClassifierCV prefers. Probabilities are clipped away from 0 / 1 first."""
    if transform.get("input") == "decision_function":
        q = np.clip(p, 1e-12, 1.0 - 1e-12)
        return np.log(q / (1.0 - q))
    return p


def apply_calibration(p_raw: np.ndarray, transform: dict[str, Any] | None) -> np.ndarray:
    """Apply a `calibration_transform` dict to raw P(class 1) values (reference implementation;
    `edge.inference.apply_calibration` is its numpy-only twin)."""
    p = np.asarray(p_raw, dtype=float)
    method = (transform or {}).get("method", "none")
    if method == "none" or not transform:
        return p
    x_in = _calibrator_input(p, transform)
    if method == "sigmoid":
        return 1.0 / (1.0 + np.exp(float(transform["a"]) * x_in + float(transform["b"])))
    if method == "isotonic":
        x = np.asarray(transform["x"], dtype=float)
        y = np.asarray(transform["y"], dtype=float)
        if len(x) == 1:
            return np.full(p.shape, float(y[0]))
        lo = float(transform.get("x_min", x[0]))
        hi = float(transform.get("x_max", x[-1]))
        return np.interp(np.clip(x_in, lo, hi), x, y)
    return p


def compute_lead_time(
    features_df: pd.DataFrame,
    spec: FeatureSpec,
    model: Any,
    threshold: float = EARLY_WARNING_THRESHOLD,
) -> LeadTimeReport:
    """Warning lead time: how far ahead of failure the model raises a sustained alarm.

    Inputs: feature frame with rul_h, FeatureSpec, a fitted model, the early-warning
    threshold (default EARLY_WARNING_THRESHOLD = 0.3, never the classification threshold).
    For each asset that fails (rul_h not null), the alarm instant is the first of
    EARLY_WARNING_SUSTAINED_ROWS consecutive rows with P(failure) >= threshold; lead time =
    failure instant (last ts + rul_h) minus alarm instant, in hours; a never-alarmed failing
    asset counts as 0 h. Output: LeadTimeReport with median / p10 / p90 over events, n_events
    and the threshold used. All zeros if no asset in the frame fails.
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
    q = sorted(
        max(0.0, float(models[a].predict(target[spec.features])[0])) for a in (0.1, 0.5, 0.9)
    )
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


def _finite_metrics(metrics: dict[str, float]) -> dict[str, float]:
    """Drop NaN/inf entries so BacktestFold.metrics survives a JSON round trip.

    `_mean_metric` treats a missing key as NaN, so averages are unchanged.
    """
    return {k: float(v) for k, v in metrics.items() if math.isfinite(float(v))}


def _mean_metric(folds: list[BacktestFold], key: str) -> float:
    vals = [f.metrics.get(key, float("nan")) for f in folds]
    vals = [v for v in vals if not math.isnan(v)]
    return float(np.mean(vals)) if vals else float("nan")


def _fallback_split(df: pd.DataFrame, candidates: CandidateSet) -> AssetSplit:
    """Split for a CandidateSet that recorded none: train = the spec's fit_assets (or the
    id-order 75 %), the rest dealt into tune / test by seed so calibration and test stay
    disjoint. Noted on the report as a fallback."""
    assets = _sorted_assets(df)
    train = list(candidates.features.fit_assets or _split_assets(assets)[0])
    rest = [a for a in assets if a not in set(train)]
    sub = make_asset_split(df.loc[df["asset_id"].isin(rest)], seed=0) if rest else None
    return AssetSplit(
        seed=0,
        train=sorted(train),
        validation=sorted((sub.validation if sub else []) + (sub.train if sub else [])),
        calibration=sorted((sub.calibration if sub else []) + (sub.train if sub else [])),
        test=sorted(sub.test if sub else []),
        positives=sub.positives if sub else {},
        notes=["fallback split: the CandidateSet recorded no AssetSplit"]
        + (sub.notes if sub else []),
    )


def _lead_time_report(leads: list[float]) -> LeadTimeReport:
    if not leads:
        return LeadTimeReport(
            median_h=0.0, p90_h=0.0, p10_h=0.0, n_events=0, threshold=EARLY_WARNING_THRESHOLD
        )
    arr = np.asarray(leads, dtype=float)
    return LeadTimeReport(
        median_h=float(np.median(arr)),
        p90_h=float(np.percentile(arr, 90)),
        p10_h=float(np.percentile(arr, 10)),
        n_events=int(len(arr)),
        threshold=EARLY_WARNING_THRESHOLD,
    )


def validate(candidates: CandidateSet, features_df: pd.DataFrame, run_id: str) -> ValidationReport:
    """Run every validation check on a CandidateSet and pick the champion -- leak-free.

    Inputs: CandidateSet (candidates must have artifact_path set; ``split`` records the
    train / validation / calibration / test assets fixed before training), the feature frame
    the candidates were built from (``label_known`` marks right-censored rows, skipped here),
    and the run id. Steps:
      1. leakage check: every non-test asset vs the test assets (disjoint, no forbidden or
         label-correlated features);
      2. champion = the candidate with the highest Industrial Model Score on the VALIDATION
         assets, scored with its fitted artifact (metrics at CLASSIFICATION_THRESHOLD, lead
         time at EARLY_WARNING_THRESHOLD, see agents/metrics.py) -- test assets untouched;
      3. the champion is calibrated on the CALIBRATION assets (isotonic when >= 200 rows and
         >= 2 failing assets, else Platt sigmoid; "none" with one class) and saved as
         <candidate_id>_calibrated.joblib;
      4. final metrics come from the calibrated champion on the TEST assets only: one
         ``BacktestFold`` (fold 0, train_assets = split.train, test_assets = split.test),
         ``test_metrics``, ``lead_time`` (median over failing test assets) and the reported
         ``ims``;
      5. RUL interval (None on failure);
      6. the gate checklist: leakage passed, test recall >= 0.6, median test lead time >= 12 h,
         at least one completed failure among test assets. ``passed`` is their conjunction.
    Output: ValidationReport with ``split``, ``gate``, ``test_metrics`` and ``thresholds``.
    notes[0] is always "champion_artifact=<path>" (the calibrated champion). Raises
    ValueError when no candidate artifact can be loaded.
    """
    spec = candidates.features
    df = features_df.sort_values(["asset_id", "ts"]).reset_index(drop=True)
    split = candidates.split or _fallback_split(df, candidates)
    notes: list[str] = []

    leakage = detect_leakage(df, spec, split.not_test, split.test)

    val_rows = rows_in(df, split.validation)
    cal_rows = rows_in(df, split.calibration)
    test_rows = rows_in(df, split.test)
    if val_rows.empty:
        val_rows = rows_in(df, split.train)
        notes.append("validation group empty: candidates ranked on training rows")

    scored: list[tuple[CandidateModel, Any, dict[str, float], IndustrialModelScore]] = []
    for c in candidates.candidates:
        model = _load_candidate(c)
        if model is None:
            continue
        p = positive_proba(model, val_rows[spec.features])
        metrics = classification_metrics(val_rows["label"].to_numpy().astype(int), p)
        leads = lead_times_h(val_rows[["asset_id", "ts", "rul_h"]], p)
        metrics["lead_time_h"] = float(np.median(leads)) if leads else float("nan")
        ims = _industrial_model_score(metrics, c.inference_latency_ms)
        scored.append((c, model, metrics, ims))
        log.info("candidate %s (%s): validation IMS %.3f", c.candidate_id, c.family, ims.total)
    if not scored:
        raise ValueError("validate: no candidate could be loaded from artifact_path")
    champion, model, _val_metrics, _val_ims = max(scored, key=lambda t: t[3].total)

    if cal_rows.empty:
        cal_rows = val_rows
        notes.append("calibration group empty: calibrated on validation rows")
    n_pos_assets = int(cal_rows.loc[cal_rows["label"] == 1, "asset_id"].nunique())
    calibrated, calibration = calibrate_probabilities(
        model, cal_rows[spec.features], cal_rows["label"].astype(int), n_positive_assets=n_pos_assets
    )
    original = Path(champion.artifact_path or f"{champion.candidate_id}.joblib")
    cal_path = original.with_name(f"{champion.candidate_id}_calibrated.joblib")
    cal_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(calibrated, cal_path)
    notes.append(f"champion_artifact={cal_path}")
    notes.append(
        f"calibration ({calibration.method}) fitted on {len(cal_rows)} rows of "
        f"{sorted(set(cal_rows['asset_id']))} with {n_pos_assets} failing asset(s)"
    )

    # Final, untouched test assets.
    if test_rows.empty:
        test_metrics: dict[str, float] = {}
        leads = []
        notes.append("no test rows: final metrics unavailable")
        train_end = pd.Timestamp(df["ts"].max())
    else:
        p_test = positive_proba(calibrated, test_rows[spec.features])
        test_metrics = classification_metrics(test_rows["label"].to_numpy().astype(int), p_test)
        leads = lead_times_h(test_rows[["asset_id", "ts", "rul_h"]], p_test)
        test_metrics["lead_time_h"] = float(np.median(leads)) if leads else float("nan")
        test_metrics["n_assets"] = float(test_rows["asset_id"].nunique())
        test_metrics["n_failure_events"] = float(len(leads))
        train_rows = rows_in(df, split.train, known_only=False)
        train_end = pd.Timestamp((train_rows if not train_rows.empty else df)["ts"].max())
    lead_time = _lead_time_report(leads)
    fold = BacktestFold(
        fold=0,
        train_assets=list(split.train),
        test_assets=list(split.test),
        train_end=train_end.to_pydatetime(),
        metrics=_finite_metrics(test_metrics),
    )
    ims = _industrial_model_score(
        {
            "recall": test_metrics.get("recall", 0.0),
            "precision": test_metrics.get("precision", 0.0),
            "lead_time_h": lead_time.median_h,
            "brier": test_metrics.get("brier", float("nan")),
        },
        champion.inference_latency_ms,
    )
    notes.append(f"final metrics on test assets {split.test} (never used before this step)")

    rul: RULInterval | None
    try:
        rul = estimate_rul_interval(df, spec, asset_id=candidates.asset_id)
    except Exception as e:  # noqa: BLE001 - RUL is optional evidence
        rul = None
        notes.append(f"rul interval unavailable: {e}")

    gate = [
        GateCheck(
            name="leakage",
            passed=leakage.passed,
            detail="no asset overlap, no forbidden or label-correlated feature"
            if leakage.passed
            else "; ".join(leakage.notes),
        ),
        GateCheck(
            name="test_failure_events",
            passed=lead_time.n_events >= 1,
            value=float(lead_time.n_events),
            threshold=1.0,
            detail=f"{lead_time.n_events} completed failure(s) among test assets {split.test}",
        ),
        GateCheck(
            name="recall",
            passed=ims.recall >= MIN_RECALL_TO_PASS,
            value=float(ims.recall),
            threshold=MIN_RECALL_TO_PASS,
            detail=f"test recall at P>={CLASSIFICATION_THRESHOLD}",
        ),
        GateCheck(
            name="lead_time",
            passed=lead_time.median_h >= MIN_LEAD_TIME_H_TO_PASS,
            value=float(lead_time.median_h),
            threshold=MIN_LEAD_TIME_H_TO_PASS,
            detail=(
                f"median warning lead time at P>={EARLY_WARNING_THRESHOLD} sustained "
                f"{EARLY_WARNING_SUSTAINED_ROWS} rows, over test failures"
            ),
        ),
    ]
    passed = all(g.passed for g in gate)
    if not passed:
        notes.append(
            "gate: " + "; ".join(f"{g.name}={'ok' if g.passed else 'FAIL'}" for g in gate)
        )
    notes.append(
        "ranking (validation IMS): "
        + ", ".join(
            f"{c.candidate_id}={s.total:.3f}"
            for c, _, _, s in sorted(scored, key=lambda t: -t[3].total)
        )
    )

    return ValidationReport(
        asset_id=candidates.asset_id,
        champion_id=champion.candidate_id,
        champion_family=champion.family,
        champion_mlflow_run_id=champion.mlflow_run_id,
        model_version=f"v{int(time.time()) % 100000}",
        leakage=leakage,
        backtest=[fold],
        calibration=calibration,
        lead_time=lead_time,
        rul=rul,
        ims=ims,
        passed=passed,
        notes=notes,
        thresholds=Thresholds(),
        split=split,
        gate=gate,
        test_metrics=_finite_metrics(test_metrics),
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
