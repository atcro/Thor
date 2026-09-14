"""07 Reliability: SHAP explanation and what-if re-scoring for the champion model.

Deterministic local compute only. The numbers produced here (failure probability, SHAP
attributions) are what the orchestrator's explanation-drafting LLM call is allowed to
describe; it never produces them itself (CLAUDE.md section 7).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from apps.api.schemas import Explanation, FeatureSpec, ShapFeature, WhatIfResult

log = logging.getLogger(__name__)

# Short plain-language names for the standard feature set (used by rag.py and contract.py).
FEATURE_PHRASES: dict[str, str] = {
    "vib_rms_mean": "vibration RMS",
    "vib_rms_slope": "rising vibration RMS trend",
    "vib_kurt_mean": "vibration kurtosis",
    "vib_crest_mean": "vibration crest factor",
    "bearing_temp_mean": "bearing temperature",
    "bearing_temp_slope": "rising bearing temperature trend",
    "temp_delta": "bearing minus motor temperature delta",
    "current_mean": "motor current",
    "rpm_mean": "motor speed",
    "load_mean": "motor load",
    "vib_rms_z": "vibration RMS above regime baseline",
    "bearing_temp_z": "bearing temperature above regime baseline",
    "vib_rms_resid": "vibration RMS residual versus regime expectation",
}

_MAX_BACKGROUND_ROWS = 50


def _is_tree_model(obj: Any) -> bool:
    """True for fitted sklearn tree ensembles / trees and xgboost / lightgbm classifiers."""
    return any(hasattr(obj, attr) for attr in ("estimators_", "tree_", "get_booster", "booster_"))


def unwrap_tree_model(model: Any) -> Any:
    """Return the underlying tree estimator inside calibration / frozen wrappers.

    Input: any fitted classifier. Output: the innermost estimator reachable through
    `calibrated_classifiers_[0].estimator` (sklearn CalibratedClassifierCV), `.estimator`
    (FrozenEstimator and similar), or `.base_estimator_`; the model itself if none apply.
    """
    seen = 0
    current = model
    while seen < 8:
        seen += 1
        # 1. known wrappers first (FrozenEstimator forwards attribute lookups, so it would
        #    otherwise pass the tree check below)
        calibrated = getattr(current, "calibrated_classifiers_", None)
        if calibrated:
            inner = getattr(calibrated[0], "estimator", None)
            if inner is not None and inner is not current:
                current = inner
                continue
        if type(current).__name__ == "FrozenEstimator" and hasattr(current, "estimator"):
            current = current.estimator
            continue
        # 2. stop at a fitted tree model (RandomForest exposes an unfitted `.estimator`
        #    template, so the tree check must precede the generic attribute walk)
        if _is_tree_model(current):
            return current
        # 3. generic wrappers
        for attr in ("base_estimator_", "estimator_", "estimator"):
            inner = getattr(current, attr, None)
            if inner is not None and inner is not current and hasattr(inner, "predict_proba"):
                current = inner
                break
        else:
            return current
    return current


def _latest_row(X_latest: pd.DataFrame, spec: FeatureSpec) -> pd.DataFrame:
    """One-row frame with exactly the FeatureSpec columns for the most recent sample."""
    if X_latest.empty:
        raise ValueError("explain: X_latest is empty")
    df = X_latest
    if "ts" in df.columns:
        df = df.sort_values("ts")
    missing = [f for f in spec.features if f not in df.columns]
    if missing:
        raise ValueError(f"explain: features missing from X_latest: {missing}")
    return df[spec.features].tail(1).astype(float).reset_index(drop=True)


def _background(X_latest: pd.DataFrame, spec: FeatureSpec) -> pd.DataFrame:
    df = X_latest
    if "ts" in df.columns:
        df = df.sort_values("ts")
    return df[spec.features].tail(_MAX_BACKGROUND_ROWS).astype(float).reset_index(drop=True)


def _positive_proba(model: Any, X: pd.DataFrame) -> np.ndarray:
    proba = np.asarray(model.predict_proba(X))
    if proba.ndim == 1:
        return proba.astype(float)
    return proba[:, -1].astype(float)


def _positive_class_values(raw: Any, n_features: int) -> np.ndarray:
    """Normalise the shape variants shap returns into a 1-D array of length n_features.

    Handles: list of per-class arrays, (n, f, classes) 3-D arrays, (n, f) 2-D arrays and
    1-D arrays. For multi-class outputs the positive (last) class is used.
    """
    if isinstance(raw, list):
        arr = np.asarray(raw[-1])
    else:
        arr = np.asarray(raw)
    if arr.ndim == 3:
        # (n_rows, n_features, n_classes) -> first row, positive class
        arr = arr[0, :, -1]
    elif arr.ndim == 2:
        if arr.shape[1] == n_features:
            arr = arr[0]  # (n_rows, n_features) -> first (only) row
        else:
            arr = arr[:, -1]  # (n_features, n_classes) -> positive class column
    arr = np.asarray(arr, dtype=float).reshape(-1)
    if arr.shape[0] != n_features:
        raise ValueError(f"unexpected shap shape {np.shape(raw)} for {n_features} features")
    return arr


def _positive_base_value(expected: Any) -> float:
    arr = np.asarray(expected, dtype=float).reshape(-1)
    return float(arr[-1]) if arr.size else 0.0


def _additive(values: np.ndarray, base: float, target: float | None, tol: float = 0.05) -> bool:
    """SHAP additivity check: base + sum(values) must reproduce the prediction (probability
    space) within `tol`. Catches explainer/output-space mismatches that yield absurd values."""
    if not np.all(np.isfinite(values)) or not np.isfinite(base):
        return False
    if target is None:
        return bool(np.max(np.abs(values)) < 50.0)
    return abs(float(base + values.sum()) - float(target)) <= tol


def _tree_shap(
    tree_model: Any,
    x: pd.DataFrame,
    background: pd.DataFrame,
    probability: float | None = None,
) -> tuple[np.ndarray, float]:
    """TreeExplainer attributions for one row. Tries probability space first (needs a
    background sample) and falls back to the model's native output space. Every result is
    checked for additivity against `probability`; a non-additive probability-space result
    (seen with LightGBM) is discarded and a ValueError is raised if no path is consistent, so
    the caller can fall back to the model-agnostic explainer."""
    import shap

    n = len(x.columns)
    if len(background) >= 5:
        try:
            explainer = shap.TreeExplainer(
                tree_model,
                data=background,
                model_output="probability",
                feature_perturbation="interventional",
            )
            values = _positive_class_values(explainer.shap_values(x), n)
            base = _positive_base_value(explainer.expected_value)
            if _additive(values, base, probability):
                return values, base
            log.debug("probability-space TreeExplainer not additive; trying native output")
        except Exception as e:  # noqa: BLE001 - e.g. xgboost categorical splits
            log.debug("probability-space TreeExplainer unavailable: %s", e)
    explainer = shap.TreeExplainer(tree_model)
    values = _positive_class_values(explainer.shap_values(x), n)
    base = _positive_base_value(explainer.expected_value)
    if _additive(values, base, None):  # native margin: only sanity-bound the magnitudes
        return values, base
    raise ValueError("TreeExplainer produced non-additive / unbounded attributions")


def _model_agnostic_shap(
    model: Any, x: pd.DataFrame, background: pd.DataFrame
) -> tuple[np.ndarray, float]:
    """Fallback: shap.Explainer over predict_proba with a small background sample."""
    import shap

    columns = list(x.columns)

    def f(arr: np.ndarray) -> np.ndarray:
        return _positive_proba(model, pd.DataFrame(np.asarray(arr), columns=columns))

    bg = background if len(background) >= 2 else pd.concat([x, x], ignore_index=True)
    explainer = shap.Explainer(f, bg)
    out = explainer(x)
    values = np.asarray(out.values, dtype=float).reshape(-1)[: len(columns)]
    base = np.asarray(out.base_values, dtype=float).reshape(-1)
    return values, float(base[0]) if base.size else 0.0


def explain(
    model: Any,
    X_latest: pd.DataFrame,
    spec: FeatureSpec,
    asset_id: str,
    model_version: str,
    top_k: int = 6,
    background: pd.DataFrame | None = None,
) -> Explanation:
    """SHAP explanation of the champion's prediction for the latest sample.

    Inputs: fitted (preferably calibrated) model with predict_proba, a frame of recent
    feature rows for one asset (latest row = last by ts, or last row if no ts column; the
    earlier rows serve as SHAP background), FeatureSpec, asset id, model version, top_k.
    Output: Explanation with failure_probability = calibrated predict_proba on the latest
    row, top_features sorted by |shap| descending (direction = sign), and base_value =
    the explainer's expected value (probability space when available, else the tree
    model's native margin). Attribution path: shap.TreeExplainer on the unwrapped tree
    model, falling back to shap.Explainer over predict_proba, and finally to an empty
    feature list with a warning so the pipeline never dies on explainability.
    """
    x = _latest_row(X_latest, spec)
    if background is not None and len(background):
        # Caller-supplied background (e.g. healthy fleet rows + the asset's recent rows):
        # gives the explainer real variation instead of 50 near-identical failing rows.
        background = background[spec.features].astype(float).reset_index(drop=True)
    else:
        background = _background(X_latest, spec)
    probability = float(np.clip(_positive_proba(model, x)[0], 0.0, 1.0))
    tree = unwrap_tree_model(model)

    values: np.ndarray | None = None
    base_value = probability
    try:
        values, base_value = _tree_shap(tree, x, background, probability)
    except Exception as e:  # noqa: BLE001 - fall through to model-agnostic explainer
        log.warning("TreeExplainer failed for %s (%s); using model-agnostic explainer", asset_id, e)
        try:
            # Explain the tree model's own probability: an isotonic calibration layer is a
            # step function that flattens perturbations to zero attribution.
            values, base_value = _model_agnostic_shap(tree, x, background)
        except Exception as e2:  # noqa: BLE001 - last resort: no attributions
            log.error("SHAP unavailable for %s: %s", asset_id, e2)
            values = None

    top: list[ShapFeature] = []
    if values is not None:
        order = np.argsort(-np.abs(values))[: max(0, top_k)]
        for i in order:
            sv = float(values[i])
            top.append(
                ShapFeature(
                    feature=spec.features[i],
                    shap_value=sv,
                    feature_value=float(x.iloc[0, i]),
                    direction="raises_risk" if sv > 0 else "lowers_risk",
                )
            )

    return Explanation(
        asset_id=asset_id,
        failure_probability=probability,
        top_features=top,
        base_value=float(base_value),
        model_version=model_version,
    )


def run_whatif(
    model: Any,
    X_latest: pd.DataFrame,
    spec: FeatureSpec,
    asset_id: str,
    scenario: dict[str, float],
) -> WhatIfResult:
    """Re-score the latest sample with some feature values overridden.

    Inputs: fitted model, recent feature rows (latest = last by ts), FeatureSpec, asset id
    and a scenario mapping feature name -> new value (e.g. {"load_mean": 40}). Every key
    must be in FeatureSpec.features, otherwise ValueError. Output: WhatIfResult with the
    baseline probability, the scenario probability and delta = scenario - baseline.
    """
    unknown = sorted(k for k in scenario if k not in spec.features)
    if unknown:
        raise ValueError(f"run_whatif: unknown features {unknown}; allowed: {spec.features}")
    x = _latest_row(X_latest, spec)
    baseline = float(np.clip(_positive_proba(model, x)[0], 0.0, 1.0))
    x2 = x.copy()
    for k, v in scenario.items():
        x2.loc[0, k] = float(v)
    scenario_p = float(np.clip(_positive_proba(model, x2)[0], 0.0, 1.0))
    return WhatIfResult(
        asset_id=asset_id,
        scenario={k: float(v) for k, v in scenario.items()},
        baseline_probability=baseline,
        scenario_probability=scenario_p,
        delta=scenario_p - baseline,
    )
