"""Shared metric definitions for 05 (AutoML search) and 06 (Validation).

One meaning per threshold, defined once in `apps.api.schemas` and used here by both stages so
the number Optuna optimises is the same number the validation gate checks:

* `CLASSIFICATION_THRESHOLD` (0.5): P(failure within horizon) >= 0.5 counts as a predicted
  failure. Recall, precision and F1 use it. Nothing else does.
* `EARLY_WARNING_THRESHOLD` (0.3) with `EARLY_WARNING_SUSTAINED_ROWS` (3): the alarm. Warning
  lead time is measured from the first of three consecutive rows at or above 0.3 to the
  failure instant; the maintenance window must end before P(failure) exceeds 0.3. Nothing else
  uses 0.3.

Everything here is deterministic numpy/sklearn; no LLM, no database.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from apps.api.schemas import (
    CLASSIFICATION_THRESHOLD,
    EARLY_WARNING_SUSTAINED_ROWS,
    EARLY_WARNING_THRESHOLD,
    IndustrialModelScore,
)

LEAD_TIME_CAP_H = 72.0
BRIER_CAP = 0.25
LATENCY_CAP_MS = 50.0
IMS_WEIGHTS: dict[str, float] = {
    "recall": 0.35,
    "lead_time_score": 0.25,
    "precision": 0.15,
    "calibration_score": 0.15,
    "latency_score": 0.10,
}


def _finite(value: Any, default: float) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return default
    return default if (math.isnan(v) or math.isinf(v)) else v


def positive_proba(model: Any, X: pd.DataFrame) -> np.ndarray:
    """P(label = 1) per row from any classifier exposing predict_proba (1-D or (n, k))."""
    p = np.asarray(model.predict_proba(X))
    if p.ndim == 2 and p.shape[1] >= 2:
        return np.clip(p[:, 1].astype(float), 0.0, 1.0)
    return np.clip(p.reshape(-1).astype(float), 0.0, 1.0)


def classification_metrics(
    y_true: np.ndarray, p: np.ndarray, threshold: float = CLASSIFICATION_THRESHOLD
) -> dict[str, float]:
    """recall, precision, f1 at `threshold` (default CLASSIFICATION_THRESHOLD), plus auroc
    (NaN when one class only), brier, n_test, n_pos_test.

    Input: integer labels and calibrated-or-raw probabilities of the same length.
    """
    y = np.asarray(y_true).astype(int)
    p = np.asarray(p, dtype=float)
    y_pred = (p >= threshold).astype(int)
    out: dict[str, float] = {
        "recall": float(recall_score(y, y_pred, zero_division=0)),
        "precision": float(precision_score(y, y_pred, zero_division=0)),
        "f1": float(f1_score(y, y_pred, zero_division=0)),
        "brier": float(brier_score_loss(y, p)) if len(y) else float("nan"),
        "n_test": float(len(y)),
        "n_pos_test": float(int(y.sum())) if len(y) else 0.0,
        "threshold": float(threshold),
    }
    out["auroc"] = float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else float("nan")
    return out


def failure_ts(asset_df: pd.DataFrame) -> pd.Timestamp | None:
    """Failure instant of one asset: ts + rul_h at the last row that still carries a rul_h
    (after a failure the motor is repaired and rul_h is NaN again). None if it never fails."""
    rul = pd.to_numeric(asset_df["rul_h"], errors="coerce")
    mask = rul.notna().to_numpy()
    if not mask.any():
        return None
    last_pos = int(np.flatnonzero(mask)[-1])
    return pd.Timestamp(asset_df["ts"].iloc[last_pos]) + pd.Timedelta(
        hours=float(rul.iloc[last_pos])
    )


def alarm_index(
    prob: np.ndarray,
    threshold: float = EARLY_WARNING_THRESHOLD,
    sustained_rows: int = EARLY_WARNING_SUSTAINED_ROWS,
) -> int | None:
    """Row index of the first alarm: the first of `sustained_rows` consecutive rows with
    prob >= threshold. None if the alarm never fires. This is THE lead-time definition."""
    hot = np.asarray(prob, dtype=float) >= threshold
    n = int(sustained_rows)
    for i in range(len(hot) - n + 1):
        if bool(hot[i : i + n].all()):
            return i
    return None


def lead_times_h(
    meta: pd.DataFrame,
    prob: np.ndarray,
    threshold: float = EARLY_WARNING_THRESHOLD,
    sustained_rows: int = EARLY_WARNING_SUSTAINED_ROWS,
) -> list[float]:
    """Warning lead time per failing asset, in hours, in sorted asset order.

    Input: `meta` with columns asset_id, ts, rul_h (feature-frame rows) and `prob` aligned to
    it. For each asset that fails (rul_h present): hours from the alarm (`alarm_index` on the
    rows up to the failure) to the failure instant; a never-alarmed failure contributes 0.0.
    Assets that never fail are skipped. Used identically by the Optuna objective (05) and the
    validation gate (06).
    """
    m = meta[["asset_id", "ts", "rul_h"]].copy()
    m["prob"] = np.asarray(prob, dtype=float)
    leads: list[float] = []
    for _asset, g in m.groupby("asset_id", sort=True):
        g = g.sort_values("ts")
        fail_ts = failure_ts(g)
        if fail_ts is None:
            continue
        rul = pd.to_numeric(g["rul_h"], errors="coerce")
        last_pos = int(np.flatnonzero(rul.notna().to_numpy())[-1])
        before = g.iloc[: last_pos + 1]
        idx = alarm_index(before["prob"].to_numpy(), threshold, sustained_rows)
        if idx is None:
            leads.append(0.0)
            continue
        first_ts = pd.Timestamp(before["ts"].iloc[idx])
        leads.append(max(0.0, (fail_ts - first_ts).total_seconds() / 3600.0))
    return leads


def median_lead_time_h(meta: pd.DataFrame, prob: np.ndarray) -> float:
    """Median of `lead_times_h`; NaN when no asset in `meta` fails."""
    leads = lead_times_h(meta, prob)
    return float(np.median(leads)) if leads else float("nan")


def industrial_model_score(metrics: dict[str, float], latency_ms: float) -> IndustrialModelScore:
    """Champion-selection score weighted for industrial use, not raw accuracy.

    Input: `metrics` with `recall`, `precision` (at CLASSIFICATION_THRESHOLD), `lead_time_h`
    (median warning lead time at EARLY_WARNING_THRESHOLD) and `brier`; missing or non-finite
    values count as their worst case. `latency_ms` = inference cost per 1000 rows.
    Output: `IndustrialModelScore` with lead_time_score = min(1, lead_time_h / 72),
    calibration_score = 1 - min(1, brier / 0.25), latency_score = 1 - min(1, latency_ms / 50)
    and total = 0.35 recall + 0.25 lead_time + 0.15 precision + 0.15 calibration + 0.10 latency.
    """
    recall = min(1.0, max(0.0, _finite(metrics.get("recall", 0.0), 0.0)))
    precision = min(1.0, max(0.0, _finite(metrics.get("precision", 0.0), 0.0)))
    lead_time_h = max(0.0, _finite(metrics.get("lead_time_h", 0.0), 0.0))
    brier = max(0.0, _finite(metrics.get("brier", BRIER_CAP), BRIER_CAP))
    lat = max(0.0, _finite(latency_ms, LATENCY_CAP_MS))
    parts = {
        "recall": recall,
        "precision": precision,
        "lead_time_score": min(1.0, lead_time_h / LEAD_TIME_CAP_H),
        "calibration_score": 1.0 - min(1.0, brier / BRIER_CAP),
        "latency_score": 1.0 - min(1.0, lat / LATENCY_CAP_MS),
    }
    total = round(sum(IMS_WEIGHTS[k] * parts[k] for k in IMS_WEIGHTS), 6)
    return IndustrialModelScore(**parts, weights=dict(IMS_WEIGHTS), total=float(total))
