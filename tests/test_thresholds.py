"""One meaning per threshold (defect 6).

CLASSIFICATION_THRESHOLD (0.5) decides what a predicted failure is: recall / precision / F1.
EARLY_WARNING_THRESHOLD (0.3) with EARLY_WARNING_SUSTAINED_ROWS (3) is the alarm: it defines
warning lead time and the maintenance-window limit. The AutoML search (05) and the Validation
gate (06) must use the same definitions, and the validation report must say which it used.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from agents import metrics
from agents.ml_architect import automl
from agents.reliability import cost
from agents.validation import checks
from apps.api import schemas


def test_thresholds_have_one_source_and_distinct_meanings() -> None:
    assert schemas.CLASSIFICATION_THRESHOLD == 0.5
    assert schemas.EARLY_WARNING_THRESHOLD == 0.3
    assert schemas.EARLY_WARNING_SUSTAINED_ROWS == 3
    # every stage reads the same constants (no private copies)
    assert automl.CLASSIFICATION_THRESHOLD is schemas.CLASSIFICATION_THRESHOLD
    assert automl.EARLY_WARNING_THRESHOLD is schemas.EARLY_WARNING_THRESHOLD
    assert checks.CLASSIFICATION_THRESHOLD is schemas.CLASSIFICATION_THRESHOLD
    assert checks.EARLY_WARNING_THRESHOLD is schemas.EARLY_WARNING_THRESHOLD
    assert cost.WINDOW_P_FAIL_LIMIT == schemas.EARLY_WARNING_THRESHOLD
    assert not hasattr(automl, "ALARM_THRESHOLD") and not hasattr(checks, "ALARM_THRESHOLD")
    # the IMS formula exists exactly once
    assert automl.industrial_model_score is metrics.industrial_model_score
    assert checks.industrial_model_score is metrics.industrial_model_score
    t = schemas.Thresholds()
    assert (t.classification, t.early_warning, t.sustained_rows) == (0.5, 0.3, 3)


def test_classification_metrics_use_the_classification_threshold() -> None:
    y = np.array([1, 1, 0, 0])
    p = np.array([0.45, 0.6, 0.35, 0.1])  # two rows sit between 0.3 and 0.5
    m = metrics.classification_metrics(y, p)
    assert m["threshold"] == 0.5
    assert m["recall"] == 0.5 and m["precision"] == 1.0  # 0.45 is NOT a predicted failure
    low = metrics.classification_metrics(y, p, threshold=0.3)
    assert low["recall"] == 1.0 and low["precision"] == 2 / 3


def _meta(n: int, step_h: float = 1.0) -> pd.DataFrame:
    ts = pd.Timestamp("2025-09-01", tz="UTC") + pd.to_timedelta(np.arange(n) * step_h, unit="h")
    rul = (n - 1 - np.arange(n)) * step_h  # fails right after the last row
    return pd.DataFrame({"asset_id": "MTR-003", "ts": ts, "rul_h": rul})


def test_lead_time_is_sustained_early_warning_crossing() -> None:
    meta = _meta(10)
    # a two-row blip at 0.3 does not alarm; the first of three sustained rows (index 5) does
    prob = np.array([0.1, 0.35, 0.35, 0.1, 0.1, 0.4, 0.45, 0.5, 0.9, 0.95])
    assert metrics.alarm_index(prob) == 5
    assert metrics.lead_times_h(meta, prob) == [4.0]  # rows 5..9 = 4 h before failure
    # 0.4 >= 0.3 counts for the alarm even though it is below the classification threshold
    assert prob[5] < schemas.CLASSIFICATION_THRESHOLD
    # never sustained -> missed event contributes 0.0, not NaN
    assert metrics.lead_times_h(meta, np.full(10, 0.1)) == [0.0]
    # a healthy asset (no rul) is skipped
    healthy = meta.assign(rul_h=np.nan)
    assert metrics.lead_times_h(healthy, prob) == []


class _FixedModel:
    def __init__(self, p: np.ndarray) -> None:
        self.p = p

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        p = self.p[: len(X)]
        return np.column_stack([1 - p, p])


def test_automl_and_validation_agree_on_lead_time() -> None:
    """The number Optuna optimises is the number the gate checks."""
    meta = _meta(10)
    prob = np.array([0.1, 0.35, 0.35, 0.1, 0.1, 0.4, 0.45, 0.5, 0.9, 0.95])
    frame = meta.assign(f=np.arange(10, dtype=float), label=(meta["rul_h"] <= 4).astype(int))
    spec = schemas.FeatureSpec(features=["f"], window_rows=2, description="t")
    model = _FixedModel(prob)
    automl_metrics = automl._evaluate(model, frame[["f"]], frame["label"].to_numpy(), meta)
    report = checks.compute_lead_time(frame, spec, model)
    assert automl_metrics["lead_time_h"] == report.median_h == 4.0
    assert report.threshold == schemas.EARLY_WARNING_THRESHOLD
    assert report.sustained_rows == schemas.EARLY_WARNING_SUSTAINED_ROWS
    # classification side: recall at 0.5 counts rows 7,8,9 of the five positives (rul <= 4)
    assert automl_metrics["recall"] == 3 / 5


def test_validation_report_states_its_thresholds() -> None:
    v = schemas.ValidationReport(
        asset_id="MTR-042",
        champion_id="c",
        champion_family="lightgbm",
        model_version="v1",
        leakage=schemas.LeakageReport(temporal_leakage=False, asset_overlap=False, passed=True),
        backtest=[],
        calibration=schemas.CalibrationReport(method="none", brier_before=0.1, brier_after=0.1),
        lead_time=schemas.LeadTimeReport(
            median_h=20.0, p90_h=30.0, p10_h=10.0, n_events=1, threshold=0.3
        ),
        ims=metrics.industrial_model_score({"recall": 1.0, "precision": 1.0}, 1.0),
        passed=True,
    )
    assert v.thresholds.classification == 0.5 and v.thresholds.early_warning == 0.3
    assert v.lead_time.threshold == v.thresholds.early_warning


def test_maintenance_window_limit_is_the_early_warning_threshold() -> None:
    from datetime import UTC, datetime

    from apps.api.settings import Settings

    now = datetime(2026, 9, 1, 0, 0, tzinfo=UTC)
    cmp = cost.calculate_failure_cost(0.05, Settings(), now=now, lead_time_median_h=48.0)
    lt = schemas.LeadTimeReport(median_h=48.0, p90_h=60.0, p10_h=30.0, n_events=2, threshold=0.3)
    w = cost.find_maintenance_window(cmp, lt, now=now)
    assert f"exceeds {schemas.EARLY_WARNING_THRESHOLD:.2f}" in w.reason
    assert w.p_failure_before_window <= schemas.EARLY_WARNING_THRESHOLD
