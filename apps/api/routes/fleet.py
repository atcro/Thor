"""Fleet + Asset 360 + system health routes (02 control plane).

GET /fleet, GET /assets/{asset_id}, GET /system/health. Everything here is deterministic
read-only aggregation over the database; `compute_fleet()` is reused by the copilot tools.
"""

from __future__ import annotations

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
from apps.api.schemas import FleetAsset, ModelStage, TelemetryRow
from apps.api.settings import get_settings

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


def _health_score(
    p_fail: float | None, health: float | None, vib: float | None, vib_mean: float, vib_std: float
) -> float:
    """100*(1-p) if a prediction exists; else 100*health; else 100 - clipped vibration z-score."""
    if p_fail is not None:
        return float(max(0.0, min(100.0, 100.0 * (1.0 - p_fail))))
    if health is not None and not pd.isna(health):
        return float(max(0.0, min(100.0, 100.0 * health)))
    if vib is None or pd.isna(vib) or vib_std <= 0:
        return 100.0
    z = (vib - vib_mean) / vib_std
    return float(100.0 - max(0.0, min(4.0, z)) * 12.5)


def compute_fleet(engine: Engine | None = None) -> list[FleetAsset]:
    """Build the fleet table: one FleetAsset per registered asset, sorted by health ascending."""
    engine = engine or db.get_engine()
    assets = db.list_assets(engine)
    latest = db.latest_telemetry(engine)
    preds = db.latest_predictions(engine)
    open_contracts: dict[str, str] = {}
    for c in db.list_decision_contracts(engine=engine):
        if c.asset_id in open_contracts:
            continue
        if db.contract_status(c.contract_id, engine=engine) == "pending":
            open_contracts[c.asset_id] = c.contract_id
    stage = _registry_stage(engine)
    vib_mean = vib_std = 0.0
    latest_by_asset: dict[str, dict[str, Any]] = {}
    if not latest.empty:
        vib_mean = float(latest["vibration_rms"].mean())
        vib_std = float(latest["vibration_rms"].std(ddof=0) or 0.0)
        latest_by_asset = {r["asset_id"]: r for r in latest.to_dict("records")}
    out: list[FleetAsset] = []
    for a in assets:
        row = latest_by_asset.get(a.asset_id)
        p = preds.get(a.asset_id)
        health = row.get("health") if row else None
        vib = row.get("vibration_rms") if row else None
        regime = row.get("regime") if row else None
        if isinstance(regime, float) and pd.isna(regime):
            regime = None
        last_ts = row["ts"].to_pydatetime() if row and isinstance(row["ts"], pd.Timestamp) else None
        out.append(
            FleetAsset(
                asset=a,
                health_score=round(_health_score(p, health, vib, vib_mean, vib_std), 1),
                failure_probability=float(p) if p is not None else None,
                regime=regime,
                last_ts=last_ts,
                open_contract_id=open_contracts.get(a.asset_id),
                stage=stage,
            )
        )
    out.sort(key=lambda f: f.health_score)
    return out


@router.get("/fleet", response_model=list[FleetAsset])
def get_fleet() -> list[FleetAsset]:
    """Fleet overview sorted by risk (lowest health score first)."""
    return compute_fleet(db.get_engine())


@router.get("/assets/{asset_id}")
def get_asset(asset_id: str) -> dict[str, Any]:
    """Asset 360: asset, latest telemetry row, latest prediction, contracts and run summaries."""
    engine = db.get_engine()
    asset = next((a for a in db.list_assets(engine) if a.asset_id == asset_id), None)
    if asset is None:
        raise HTTPException(status_code=404, detail=f"unknown asset {asset_id}")
    latest_df = db.read_telemetry(asset_id=asset_id, limit=1, engine=engine)
    latest: TelemetryRow | None = None
    if not latest_df.empty:
        latest = TelemetryRow.model_validate(_clean_row(latest_df.iloc[-1].to_dict()))
    prediction = db.latest_predictions(engine).get(asset_id)
    contracts = db.list_decision_contracts(asset_id=asset_id, engine=engine)
    runs = []
    for r in db.list_runs(asset_id=asset_id, engine=engine):
        s = run_summary(r["state"])
        s["created_at"] = r["created_at"]
        s["updated_at"] = r["updated_at"]
        runs.append(s)
    return {
        "asset": asset,
        "latest": latest,
        "prediction": float(prediction) if prediction is not None else None,
        "contracts": contracts,
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
    return {
        "db": "ok" if db_ok else "error",
        "mqtt": _tcp_open(settings.mqtt_host, settings.mqtt_port),
        "mlflow": mlflow_ok,
        "edge": _http_ok(edge_url.rstrip("/") + "/health"),
        "n_telemetry_rows": n_rows,
        "last_ingest_ts": last_ts,
    }
