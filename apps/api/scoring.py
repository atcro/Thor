"""Score assets with the approved production model (deterministic inference, no LLM).

Used by "Run analysis" (one asset, before explaining it), the demo-prep script and
`POST /pipeline/score` (whole fleet) so the Dashboard shows an approved-model health for every
motor without waiting for the edge container. Each score goes into `predictions` with
`source="api"`; the edge writes the same table with `source="edge"`.

Only observable telemetry is read: simulator ground truth and the simulator's regime label are
dropped before features are built, and the production model's own regime baseline is reused
verbatim (nothing is refitted).
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from pydantic import BaseModel
from sqlalchemy.engine import Engine

from apps.api import db
from apps.api.schemas import SIMULATOR_ONLY_COLUMNS, FeatureSpec, ModelRef

log = logging.getLogger("thor.scoring")

_MODEL_CACHE: dict[tuple[str, float], Any] = {}


class ScoredAsset(BaseModel):
    asset_id: str
    ts: datetime
    failure_probability: float
    model_version: str


class ScoreResult(BaseModel):
    model_version: str
    scored: list[ScoredAsset]
    skipped: dict[str, str]


def load_production(engine: Engine | None = None) -> tuple[ModelRef, FeatureSpec, Any] | None:
    """(ModelRef, FeatureSpec, fitted calibrated model) for the production row, or None.

    The joblib artifact is cached per (path, mtime) so fleet-wide scoring loads it once.
    """
    from agents.mlops.lifecycle import production_bundle

    bundle = production_bundle(engine=engine)
    if bundle is None:
        return None
    ref, spec, _report = bundle
    if not ref.artifact_path:
        return None
    path = Path(ref.artifact_path)
    key = (str(path), path.stat().st_mtime if path.exists() else 0.0)
    model = _MODEL_CACHE.get(key)
    if model is None:
        model = joblib.load(path)
        _MODEL_CACHE.clear()
        _MODEL_CACHE[key] = model
    return ref, spec, model


def observed_only(df: pd.DataFrame) -> pd.DataFrame:
    """Drop simulator-only columns (ground truth + regime label) from a telemetry frame."""
    return df.drop(columns=[c for c in SIMULATOR_ONLY_COLUMNS if c in df.columns])


def latest_feature_row(
    df_asset: pd.DataFrame, spec: FeatureSpec, horizon_h: float = 48.0
) -> tuple[pd.DataFrame, datetime] | None:
    """Feature vector of the asset's latest window with the production baseline (no refit).

    Input: one asset's telemetry (at least `spec.window_rows` rows, observed columns), the
    production FeatureSpec. Output: (one-row frame with `spec.features`, plant timestamp of the
    window's last row), or None when the asset has fewer rows than the window.
    """
    from agents.ml_architect.features import build_feature_pipeline

    if len(df_asset) < int(spec.window_rows):
        return None
    fdf, _ = build_feature_pipeline(
        observed_only(df_asset),
        None,
        horizon_h=horizon_h,
        window_rows=int(spec.window_rows),
        baseline=spec.baseline,
    )
    if fdf.empty:
        return None
    last = fdf.sort_values("ts").tail(1)
    ts = pd.Timestamp(last["ts"].iloc[0])
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return last[list(spec.features)].reset_index(drop=True).astype(float), ts.to_pydatetime()


def score_assets(
    asset_ids: list[str] | None = None,
    engine: Engine | None = None,
    source: str = "api",
    horizon_h: float = 48.0,
) -> ScoreResult:
    """Score the latest window of each asset with the production model and store predictions.

    Input: asset ids (None = every registered asset), the engine, the prediction `source`
    tag. Output: `ScoreResult` with one `ScoredAsset` per scored asset and `skipped` reasons
    (too few rows). Raises LookupError when there is no production model: scoring never
    falls back to an unapproved model.
    """
    engine = engine or db.get_engine()
    loaded = load_production(engine)
    if loaded is None:
        raise LookupError("no production model: train one and approve its promotion first")
    ref, spec, model = loaded
    ids = asset_ids or [a.asset_id for a in db.list_assets(engine)]
    scored: list[ScoredAsset] = []
    skipped: dict[str, str] = {}
    rows: list[dict[str, Any]] = []
    for asset_id in ids:
        tail = db.read_telemetry(asset_id=asset_id, limit=int(spec.window_rows), engine=engine)
        latest = latest_feature_row(tail, spec, horizon_h)
        if latest is None:
            skipped[asset_id] = f"fewer than {spec.window_rows} telemetry rows"
            continue
        x, ts = latest
        proba = model.predict_proba(x)
        p = float(proba[0, 1]) if getattr(proba, "ndim", 1) == 2 else float(proba[0])
        p = max(0.0, min(1.0, p))
        scored.append(
            ScoredAsset(asset_id=asset_id, ts=ts, failure_probability=p, model_version=ref.version)
        )
        rows.append(
            {
                "asset_id": asset_id,
                "ts": ts,
                "model_version": ref.version,
                "failure_probability": p,
                "source": source,
            }
        )
    if rows:
        db.insert_predictions(rows, engine=engine)
    log.info("scored %d assets with %s (%d skipped)", len(scored), ref.version, len(skipped))
    return ScoreResult(model_version=ref.version, scored=scored, skipped=skipped)
