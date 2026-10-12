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
from apps.api.schemas import DISPLAY_MEDIAN_ROWS, SIMULATOR_ONLY_COLUMNS, FeatureSpec, ModelRef

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


def latest_feature_rows(
    df_asset: pd.DataFrame, spec: FeatureSpec, horizon_h: float = 48.0, n: int = DISPLAY_MEDIAN_ROWS
) -> list[tuple[pd.DataFrame, datetime]] | None:
    """Feature vectors of the asset's last `n` windows with the production baseline (no refit).

    Input: one asset's telemetry (at least `spec.window_rows` rows, observed columns), the
    production FeatureSpec. Output: up to `n` (one-row frame with `spec.features`, plant
    timestamp of the window's last row) pairs, oldest first, or None when the asset has fewer
    rows than one window.
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
    out: list[tuple[pd.DataFrame, datetime]] = []
    for _, row in fdf.sort_values("ts").tail(n).iterrows():
        ts = pd.Timestamp(row["ts"])
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        x = pd.DataFrame([row[list(spec.features)].astype(float).to_dict()])
        out.append((x, ts.to_pydatetime()))
    return out


def window_is_sandbox(df_asset: pd.DataFrame, ts: datetime, window_rows: int) -> bool:
    """True when any of the `window_rows` raw rows ending at `ts` is a sandbox reading."""
    if "sandbox" not in df_asset.columns or df_asset.empty:
        return False
    sub = df_asset[pd.to_datetime(df_asset["ts"], utc=True) <= pd.Timestamp(ts)].tail(window_rows)
    return bool(sub["sandbox"].fillna(False).astype(bool).any())


def latest_feature_row(
    df_asset: pd.DataFrame, spec: FeatureSpec, horizon_h: float = 48.0
) -> tuple[pd.DataFrame, datetime] | None:
    """The asset's latest window only (see `latest_feature_rows`)."""
    rows = latest_feature_rows(df_asset, spec, horizon_h, n=1)
    return rows[-1] if rows else None


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
    n_windows = int(DISPLAY_MEDIAN_ROWS)
    for asset_id in ids:
        # enough raw rows for the last n windows: window_rows + (n - 1)
        tail = db.read_telemetry(
            asset_id=asset_id, limit=int(spec.window_rows) + n_windows - 1, engine=engine
        )
        windows = latest_feature_rows(tail, spec, horizon_h, n=n_windows)
        if not windows:
            skipped[asset_id] = f"fewer than {spec.window_rows} telemetry rows"
            continue
        for x, ts in windows:  # one prediction row per window; the fleet view takes the median
            proba = model.predict_proba(x)
            p = float(proba[0, 1]) if getattr(proba, "ndim", 1) == 2 else float(proba[0])
            p = max(0.0, min(1.0, p))
            rows.append(
                {
                    "asset_id": asset_id,
                    "ts": ts,
                    "model_version": ref.version,
                    "failure_probability": p,
                    "source": source,
                    "sandbox": window_is_sandbox(tail, ts, int(spec.window_rows)),
                }
            )
        x, ts = windows[-1]
        scored.append(
            ScoredAsset(
                asset_id=asset_id,
                ts=ts,
                failure_probability=rows[-1]["failure_probability"],
                model_version=ref.version,
            )
        )
    if rows:
        db.insert_predictions(rows, engine=engine)
        # Control-plane scores are prediction arrivals too (the twin's pulses and counters);
        # no inference_ms: the centre does not time itself, only the edge reports that.
        from apps.api.live import record_prediction

        for r in rows:
            record_prediction(
                r["asset_id"], r["ts"], float(r["failure_probability"]), source="control_plane"
            )
        from apps.api import alarms

        try:
            alarms.update_alarms(engine, sorted({r["asset_id"] for r in rows}))
        except Exception as exc:  # noqa: BLE001 - scoring succeeded; alarms catch up next time
            log.warning("alarm update after scoring failed: %s", exc)
    log.info("scored %d assets with %s (%d skipped)", len(scored), ref.version, len(skipped))
    return ScoreResult(model_version=ref.version, scored=scored, skipped=skipped)
