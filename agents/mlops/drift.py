"""Hand-rolled Population Stability Index drift detector (08 MLOps).

Deterministic numpy only -- no Evidently, no LLM. `detect_drift` persists a `drift_reports`
row and returns the Pydantic `DriftReport` so the orchestrator can re-trigger 05 on drift.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sqlalchemy import insert
from sqlalchemy.engine import Engine

from apps.api import db
from apps.api.schemas import DriftReport

_EPS = 1e-4


def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index between two 1-D samples.

    Inputs: `expected` (training/baseline values), `actual` (recent values), `bins` (quantile
    bins derived from `expected`). Output: PSI >= 0; ~0 = same distribution, > 0.2 = drifted.
    NaNs are dropped; degenerate (constant) baselines return 0.0.
    """
    exp = np.asarray(expected, dtype=float)
    act = np.asarray(actual, dtype=float)
    exp, act = exp[~np.isnan(exp)], act[~np.isnan(act)]
    if exp.size == 0 or act.size == 0:
        return 0.0
    edges = np.unique(np.quantile(exp, np.linspace(0.0, 1.0, bins + 1)))
    if edges.size < 2:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    e_frac = np.histogram(exp, bins=edges)[0] / exp.size + _EPS
    a_frac = np.histogram(act, bins=edges)[0] / act.size + _EPS
    return float(np.sum((a_frac - e_frac) * np.log(a_frac / e_frac)))


def detect_drift(
    train_features: pd.DataFrame,
    recent_features: pd.DataFrame,
    feature_names: list[str],
    asset_id: str,
    model_version: str,
    threshold: float = 0.2,
    engine: Engine | None = None,
) -> DriftReport:
    """Compute per-feature PSI of `recent_features` against `train_features`.

    Inputs: two feature frames sharing `feature_names` columns (an optional `ts` column in
    `recent_features` sets the report window), the asset/model the check is for, and the PSI
    `threshold`. Output: `DriftReport` (also inserted as a `drift_reports` row).
    """
    engine = engine or db.get_engine()
    scores = {
        f: psi(train_features[f].to_numpy(), recent_features[f].to_numpy()) for f in feature_names
    }
    drifted = sorted(f for f, v in scores.items() if v > threshold)
    now = db.now_utc()
    ts = pd.to_datetime(recent_features["ts"], utc=True) if "ts" in recent_features else None
    report = DriftReport(
        asset_id=asset_id,
        model_version=model_version,
        psi=scores,
        drifted_features=drifted,
        threshold=threshold,
        drift_detected=bool(drifted),
        window_start=ts.min().to_pydatetime() if ts is not None and len(ts) else now,
        window_end=ts.max().to_pydatetime() if ts is not None and len(ts) else now,
    )
    with engine.begin() as conn:
        conn.execute(
            insert(db.drift_reports).values(
                asset_id=asset_id,
                model_version=model_version,
                created_at=now,
                drift_detected=report.drift_detected,
                payload=db.dump_model(report),
            )
        )
    return report
