"""Fleet + Asset 360 + system health routes (02 control plane).

GET /fleet, GET /assets/{asset_id}, GET /system/health. Everything here is deterministic
read-only aggregation over the database; `compute_fleet()` is reused by the copilot tools.

Ground-truth isolation (CLAUDE.md section 15): nothing here reads the simulator's `health`,
`failure_within_h` or `regime` columns. Health comes from approved-model predictions only and
the regime is inferred from rpm/load with the production model's centroids.
"""

from __future__ import annotations

import logging
import os
import socket
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import APIRouter, HTTPException
from sqlalchemy import func, select
from sqlalchemy.engine import Engine

from apps.api import db
from apps.api.graph_state import run_summary
from apps.api.schemas import FeatureSpec, FleetAsset, ModelStage, ObservedTelemetryRow
from apps.api.settings import get_settings

log = logging.getLogger("thor.routes.fleet")
router = APIRouter(tags=["fleet"])


def _clean_row(rec: dict[str, Any]) -> dict[str, Any]:
    """NaN -> None so the row validates as TelemetryRow."""
    out: dict[str, Any] = {}
    for k, v in rec.items():
        if isinstance(v, float) and pd.isna(v):
            out[k] = None
        elif isinstance(v, pd.Timestamp):
            out[k] = v.to_pydatetime()
        else:
            out[k] = v
    return out


def _registry_stage(engine: Engine | None) -> ModelStage | None:
    """Production stage if any registry row is in production, else the latest row's stage."""
    engine = engine or db.get_engine()
    with engine.connect() as conn:
        prod = conn.execute(
            select(db.model_registry.c.stage).where(db.model_registry.c.stage == "production")
        ).first()
        if prod:
            return ModelStage.production
        latest = conn.execute(
            select(db.model_registry.c.stage).order_by(db.model_registry.c.registered_at.desc())
        ).first()
    return ModelStage(latest[0]) if latest else None


def _health_score(p_fail: float | None) -> float | None:
    """100 * (1 - p) when an approved-model prediction exists, else None ("no prediction yet").

    There is deliberately no fallback: simulator ground truth (`health`) and ad-hoc vibration
    heuristics are not health. A motor nobody has scored shows no score.
    """
    if p_fail is None:
        return None
    return float(max(0.0, min(100.0, 100.0 * (1.0 - float(p_fail)))))


def production_spec(engine: Engine | None) -> FeatureSpec | None:
    """FeatureSpec of the production model (None when there is none); never raises."""
    try:
        from agents.mlops.lifecycle import production_feature_spec

        return production_feature_spec(engine=engine)
    except Exception as e:  # noqa: BLE001 - toolbox absent or registry empty
        log.debug("production feature spec unavailable: %s", e)
        return None


def infer_regimes(
    rows: list[dict[str, Any]], spec: FeatureSpec | None
) -> list[str | None]:
    """Inferred regime per telemetry row from its rpm/load and the production baseline.

    Uses the same nearest-centroid rule as the feature pipeline and the edge
    (`agents.ml_architect.features.assign_regime_ids`). Output: one regime id or None per
    row; all None when there is no production baseline. The simulator's `regime` column is
    never consulted.
    """
    if not rows or spec is None or not spec.baseline:
        return [None] * len(rows)
    try:
        from agents.ml_architect.features import assign_regime_ids

        rpm = [float(r.get("rpm")) if r.get("rpm") is not None else float("nan") for r in rows]
        load = [
            float(r.get("load_pct")) if r.get("load_pct") is not None else float("nan")
            for r in rows
        ]
        out = assign_regime_ids(rpm, load, spec.baseline)
        return [o if (o is not None and not pd.isna(rpm[i])) else None for i, o in enumerate(out)]
    except Exception as e:  # noqa: BLE001 - a display nicety must never fail the fleet call
        log.warning("regime inference failed: %s", e)
        return [None] * len(rows)


def _ts_of(value: Any) -> datetime | None:
    if isinstance(value, pd.Timestamp):
        return value.to_pydatetime()
    if isinstance(value, datetime):
        return value
    return None


def compute_fleet(engine: Engine | None = None) -> list[FleetAsset]:
    """Build the fleet table: one FleetAsset per registered asset.

    Health and failure probability come only from the latest approved-model prediction; the
    regime is inferred from the latest rpm/load with the production model's centroids. Sorted
    by health ascending (worst first); assets with no prediction yet come last.
    """
    engine = engine or db.get_engine()
    assets = db.list_assets(engine)
    latest = db.latest_telemetry(engine)
    preds = db.latest_prediction_rows(engine)
    open_contracts = db.open_contract_by_asset(engine)
    work_orders = db.work_order_by_asset(engine)
    stage = _registry_stage(engine)
    spec = production_spec(engine)
    latest_by_asset: dict[str, dict[str, Any]] = {}
    if not latest.empty:
        latest_by_asset = {r["asset_id"]: r for r in latest.to_dict("records")}
    ordered = [latest_by_asset.get(a.asset_id) for a in assets]
    regimes = infer_regimes([r for r in ordered if r is not None], spec)
    regime_by_asset: dict[str, str | None] = {}
    i = 0
    for a, row in zip(assets, ordered, strict=True):
        if row is not None:
            regime_by_asset[a.asset_id] = regimes[i]
            i += 1
    out: list[FleetAsset] = []
    for a in assets:
        row = latest_by_asset.get(a.asset_id)
        p = preds.get(a.asset_id)
        p_fail = float(p["failure_probability"]) if p else None
        health = _health_score(p_fail)
        out.append(
            FleetAsset(
                asset=a,
                health_score=round(health, 1) if health is not None else None,
                failure_probability=p_fail,
                prediction_ts=_ts_of(p["ts"]) if p else None,
                model_version=p["model_version"] if p else None,
                regime=regime_by_asset.get(a.asset_id),
                last_ts=_ts_of(row["ts"]) if row else None,
                open_contract_id=open_contracts.get(a.asset_id),
                stage=stage,
                work_order=work_orders.get(a.asset_id),
            )
        )
    out.sort(key=lambda f: (f.health_score is None, f.health_score or 0.0, f.asset.asset_id))
    return out


@router.get("/fleet", response_model=list[FleetAsset])
def get_fleet() -> list[FleetAsset]:
    """Fleet overview sorted by risk (lowest health score first; unscored assets last)."""
    return compute_fleet(db.get_engine())


@router.get("/assets/{asset_id}")
def get_asset(asset_id: str) -> dict[str, Any]:
    """Asset 360: asset, latest observed sample, latest prediction, contracts, work orders, runs.

    `latest` is an `ObservedTelemetryRow` (no simulator fields); `regime` is inferred from it
    with the production baseline; `prediction` / `prediction_ts` / `model_version` come from
    the newest approved-model prediction (all None when nothing has scored this asset yet).
    """
    engine = db.get_engine()
    asset = next((a for a in db.list_assets(engine) if a.asset_id == asset_id), None)
    if asset is None:
        raise HTTPException(status_code=404, detail=f"unknown asset {asset_id}")
    latest_df = db.read_telemetry(asset_id=asset_id, limit=1, engine=engine)
    latest: ObservedTelemetryRow | None = None
    regime: str | None = None
    if not latest_df.empty:
        rec = _clean_row(latest_df.iloc[-1].to_dict())
        latest = ObservedTelemetryRow.from_row(rec)
        regime = infer_regimes([rec], production_spec(engine))[0]
    pred = db.latest_prediction_rows(engine).get(asset_id)
    contracts = db.list_decision_contracts(asset_id=asset_id, engine=engine)
    work_orders = db.list_work_orders(asset_id=asset_id, engine=engine)
    runs = []
    for r in db.list_runs(asset_id=asset_id, engine=engine):
        s = run_summary(r["state"])
        s["created_at"] = r["created_at"]
        s["updated_at"] = r["updated_at"]
        runs.append(s)
    p_fail = float(pred["failure_probability"]) if pred else None
    return {
        "asset": asset,
        "latest": latest,
        "regime": regime,
        "prediction": p_fail,
        "prediction_ts": _ts_of(pred["ts"]) if pred else None,
        "model_version": pred["model_version"] if pred else None,
        "health_score": _health_score(p_fail),
        "contracts": contracts,
        "work_orders": work_orders,
        "runs": runs,
    }


def _tcp_open(host: str, port: int, timeout: float = 0.5) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _http_ok(url: str, timeout: float = 1.0) -> bool:
    try:
        import httpx

        return httpx.get(url, timeout=timeout).status_code < 500
    except Exception:
        return False


@router.get("/system/health")
def system_health() -> dict[str, Any]:
    """Health panel: db, mqtt broker, mlflow, edge reachability and telemetry freshness."""
    settings = get_settings()
    engine = db.get_engine()
    db_ok = True
    n_rows = 0
    last_ts: datetime | None = None
    try:
        with engine.connect() as conn:
            n_rows = int(conn.execute(select(func.count()).select_from(db.telemetry)).scalar() or 0)
            raw_ts = conn.execute(select(func.max(db.telemetry.c.ts))).scalar()
        if raw_ts is not None:
            last_ts = pd.Timestamp(raw_ts).to_pydatetime()
    except Exception:
        db_ok = False
    uri = settings.mlflow_tracking_uri
    mlflow_ok = _http_ok(uri.rstrip("/") + "/health") if uri.startswith("http") else Path(uri).exists()
    edge_url = os.environ.get("EDGE_URL", "http://localhost:8001")
    from apps.api.orchestrator import llm_status  # lazy: orchestrator imports this module

    status = llm_status()
    return {
        "db": "ok" if db_ok else "error",
        "mqtt": _tcp_open(settings.mqtt_host, settings.mqtt_port),
        "mlflow": mlflow_ok,
        "edge": _http_ok(edge_url.rstrip("/") + "/health"),
        "llm": {
            "mode": status["mode"],
            "provider": status["provider"],
            "model": status["model"],
            "warning": status["warning"],
        },
        "n_telemetry_rows": n_rows,
        "last_ingest_ts": last_ts,
    }
