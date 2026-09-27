"""Database layer -- SQLAlchemy Core, one engine, dialect-agnostic tables.

Runs on SQLite (default, for local dev and tests) and PostgreSQL + TimescaleDB (compose).
On Postgres, `telemetry` is converted to a hypertable at init. All JSON columns hold serialized
Pydantic models from `apps/api/schemas.py`.

`decision_contracts`, `approvals` and `work_order_events` are insert-only. There is deliberately
no update helper for them -- see CLAUDE.md section 7.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable
from datetime import UTC, datetime
from functools import lru_cache
from typing import Any

import pandas as pd
from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    create_engine,
    func,
    insert,
    select,
    text,
)
from sqlalchemy.engine import Engine

from apps.api.schemas import (
    SENSOR_COLUMNS,
    Approval,
    Asset,
    DecisionContract,
    TelemetryRow,
    WorkOrder,
    WorkOrderEvent,
)
from apps.api.settings import get_settings

log = logging.getLogger("thor.db")

metadata = MetaData()

assets = Table(
    "assets",
    metadata,
    Column("asset_id", String(32), primary_key=True),
    Column("name", String(128), nullable=False),
    Column("site", String(64), nullable=False),
    Column("line", String(64), nullable=False),
    Column("asset_type", String(64), nullable=False, default="induction_motor"),
    Column("rated_kw", Float, nullable=False, default=75.0),
    Column("criticality", String(16), nullable=False, default="medium"),
)

telemetry = Table(
    "telemetry",
    metadata,
    Column("asset_id", String(32), primary_key=True),
    Column("ts", DateTime(timezone=True), primary_key=True),
    *[Column(c, Float, nullable=False) for c in SENSOR_COLUMNS],
    Column("regime", String(16)),
    Column("health", Float),
    Column("failure_within_h", Float),
)
Index("ix_telemetry_ts", telemetry.c.ts)

predictions = Table(
    "predictions",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("asset_id", String(32), nullable=False, index=True),
    Column("ts", DateTime(timezone=True), nullable=False),
    Column("model_version", String(64), nullable=False),
    Column("failure_probability", Float, nullable=False),
    Column("source", String(16), nullable=False, default="edge"),
)
# One row per (asset, ts, model, source). The replay loops forever and the edge re-scores every
# row on each pass, so inserts are ON CONFLICT DO NOTHING against this key -- otherwise the table
# grows without bound (7M+ rows after a day of demo loops). It also backs the fleet view's
# "latest prediction per asset" probe.
PREDICTIONS_KEY = ("asset_id", "ts", "model_version", "source")
Index("ux_predictions_key", *[predictions.c[c] for c in PREDICTIONS_KEY], unique=True)

pipeline_runs = Table(
    "pipeline_runs",
    metadata,
    Column("run_id", String(32), primary_key=True),
    Column("asset_id", String(32), nullable=False, index=True),
    Column("stage", String(32), nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("updated_at", DateTime(timezone=True), nullable=False),
    Column("state", JSON, nullable=False),
)

data_quality_contracts = Table(
    "data_quality_contracts",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("run_id", String(32), nullable=False, index=True),
    Column("asset_id", String(32), nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("quality_score", Float, nullable=False),
    Column("payload", JSON, nullable=False),
)

validation_reports = Table(
    "validation_reports",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("run_id", String(32), nullable=False, index=True),
    Column("asset_id", String(32), nullable=False),
    Column("model_version", String(64), nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("passed", Boolean, nullable=False),
    Column("payload", JSON, nullable=False),
)

decision_contracts = Table(
    "decision_contracts",
    metadata,
    Column("contract_id", String(64), primary_key=True),
    Column("run_id", String(32), nullable=False, index=True),
    Column("asset_id", String(32), nullable=False, index=True),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("model_version", String(64), nullable=False),
    Column("recommendation", String(32), nullable=False),
    Column("failure_probability", Float, nullable=False),
    Column("expected_cost", Float, nullable=False),
    Column("evidence_hash", String(64), nullable=False),
    Column("payload", JSON, nullable=False),
)

approvals = Table(
    "approvals",
    metadata,
    Column("approval_id", String(64), primary_key=True),
    Column("contract_id", String(64), ForeignKey("decision_contracts.contract_id"), index=True),
    Column("promotion_id", String(64), index=True),
    Column("decision", String(16), nullable=False),
    Column("approver", String(128), nullable=False),
    Column("note", Text, nullable=False, default=""),
    Column("decided_at", DateTime(timezone=True), nullable=False),
)

work_order_events = Table(
    "work_order_events",
    metadata,
    Column("event_id", String(64), primary_key=True),
    Column(
        "contract_id",
        String(64),
        ForeignKey("decision_contracts.contract_id"),
        nullable=False,
        index=True,
    ),
    Column("asset_id", String(32), nullable=False, index=True),
    Column("status", String(16), nullable=False),
    Column("plant_ts", DateTime(timezone=True), nullable=False),
    Column("recorded_at", DateTime(timezone=True), nullable=False),
    Column("source", String(32), nullable=False, default="cmms-sim"),
    Column("note", Text, nullable=False, default=""),
)

model_registry = Table(
    "model_registry",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("name", String(128), nullable=False),
    Column("version", String(64), nullable=False),
    Column("mlflow_run_id", String(64)),
    Column("stage", String(16), nullable=False),
    Column("family", String(32), nullable=False),
    Column("ims_total", Float, nullable=False),
    Column("git_sha", String(64)),
    Column("dataset_version", String(64)),
    Column("registered_at", DateTime(timezone=True), nullable=False),
    Column("onnx_path", String(256)),
    Column("payload", JSON),
)
Index(
    "ix_model_registry_name_version", model_registry.c.name, model_registry.c.version, unique=True
)

promotion_requests = Table(
    "promotion_requests",
    metadata,
    Column("promotion_id", String(64), primary_key=True),
    Column("model_name", String(128), nullable=False),
    Column("version", String(64), nullable=False),
    Column("from_stage", String(16), nullable=False),
    Column("to_stage", String(16), nullable=False),
    Column("status", String(16), nullable=False, default="pending"),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("payload", JSON, nullable=False),
)

drift_reports = Table(
    "drift_reports",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("asset_id", String(32), nullable=False, index=True),
    Column("model_version", String(64), nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("drift_detected", Boolean, nullable=False),
    Column("payload", JSON, nullable=False),
)


# --------------------------------------------------------------------------------------
# Engine / init
# --------------------------------------------------------------------------------------


def _make_engine(url: str) -> Engine:
    if url.startswith("sqlite"):
        return create_engine(url, connect_args={"check_same_thread": False}, future=True)
    return create_engine(url, pool_pre_ping=True, future=True)


@lru_cache(maxsize=4)
def get_engine(url: str | None = None) -> Engine:
    """Return the process-wide engine. Pass `url` only in tests."""
    return _make_engine(url or get_settings().database_url)


def is_postgres(engine: Engine) -> bool:
    return engine.dialect.name == "postgresql"


HYPERTABLE_SQL = (
    "SELECT create_hypertable('telemetry', 'ts', if_not_exists => TRUE, migrate_data => TRUE)"
)


def init_db(engine: Engine | None = None) -> Engine:
    """Create every table (idempotent). On Postgres, enable Timescale + hypertable."""
    engine = engine or get_engine()
    metadata.create_all(engine)
    _ensure_predictions_key(engine)
    if is_postgres(engine):
        with engine.begin() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS timescaledb"))
            conn.execute(text(HYPERTABLE_SQL))
    return engine


_PREDICTIONS_KEY_SQL = (
    "CREATE UNIQUE INDEX IF NOT EXISTS ux_predictions_key "
    "ON predictions (asset_id, ts, model_version, source)"
)
_PREDICTIONS_DEDUPE_SQL = (
    "DELETE FROM predictions WHERE id NOT IN "
    "(SELECT MAX(id) FROM predictions GROUP BY asset_id, ts, model_version, source)"
)


def _ensure_predictions_key(engine: Engine) -> None:
    """Create ux_predictions_key on databases that predate it, deduplicating first if needed.

    create_all skips tables that already exist, so a volume populated by an older build has
    no key and (because the replay loops) millions of duplicate prediction rows. Keeping the
    newest row per key is lossless -- duplicates are identical edge outputs for the same
    telemetry row -- and takes seconds once, after which inserts simply ignore repeats.
    """
    try:
        with engine.begin() as conn:
            conn.execute(text(_PREDICTIONS_KEY_SQL))
        return
    except Exception as exc:  # noqa: BLE001 - duplicates present: dedupe, then retry
        log.warning("predictions key not yet enforceable (%s); deduplicating", exc)
    with engine.begin() as conn:
        before = int(conn.execute(select(func.count()).select_from(predictions)).scalar() or 0)
        conn.execute(text(_PREDICTIONS_DEDUPE_SQL))
        after = int(conn.execute(select(func.count()).select_from(predictions)).scalar() or 0)
        conn.execute(text(_PREDICTIONS_KEY_SQL))
    log.info("deduplicated predictions: %d -> %d rows; ux_predictions_key created", before, after)


def now_utc() -> datetime:
    return datetime.now(UTC)


def dump_model(model: Any) -> dict[str, Any]:
    """Pydantic -> JSON-safe dict (datetimes become ISO strings)."""
    return json.loads(model.model_dump_json())


# --------------------------------------------------------------------------------------
# Assets
# --------------------------------------------------------------------------------------


def upsert_assets(rows: Iterable[Asset], engine: Engine | None = None) -> int:
    engine = engine or get_engine()
    n = 0
    with engine.begin() as conn:
        for a in rows:
            existing = conn.execute(
                select(assets.c.asset_id).where(assets.c.asset_id == a.asset_id)
            ).first()
            if existing is None:
                conn.execute(insert(assets).values(**a.model_dump()))
                n += 1
    return n


def list_assets(engine: Engine | None = None) -> list[Asset]:
    engine = engine or get_engine()
    with engine.connect() as conn:
        rows = conn.execute(select(assets).order_by(assets.c.asset_id)).mappings().all()
    return [Asset(**dict(r)) for r in rows]


# --------------------------------------------------------------------------------------
# Telemetry
# --------------------------------------------------------------------------------------


def insert_telemetry(rows: Iterable[TelemetryRow], engine: Engine | None = None) -> int:
    """Bulk insert; duplicates on (asset_id, ts) are skipped."""
    engine = engine or get_engine()
    payload = [r.model_dump() for r in rows]
    if not payload:
        return 0
    if is_postgres(engine):
        from sqlalchemy.dialects.postgresql import insert as pg_insert

        stmt = pg_insert(telemetry).on_conflict_do_nothing(index_elements=["asset_id", "ts"])
    else:
        stmt = insert(telemetry).prefix_with("OR IGNORE")
    with engine.begin() as conn:
        conn.execute(stmt, payload)
    return len(payload)


def read_telemetry(
    asset_id: str | None = None,
    start: datetime | None = None,
    end: datetime | None = None,
    limit: int | None = None,
    engine: Engine | None = None,
) -> pd.DataFrame:
    """Return telemetry as a DataFrame sorted by (asset_id, ts). Empty frame if none.

    `limit` returns the most recent N rows (per query, not per asset).
    """
    engine = engine or get_engine()
    q = select(telemetry)
    if asset_id:
        q = q.where(telemetry.c.asset_id == asset_id)
    if start:
        q = q.where(telemetry.c.ts >= start)
    if end:
        q = q.where(telemetry.c.ts <= end)
    q = q.order_by(telemetry.c.ts.desc() if limit else telemetry.c.ts)
    if limit:
        q = q.limit(limit)
    with engine.connect() as conn:
        df = pd.read_sql(q, conn)
    if df.empty:
        return pd.DataFrame(columns=[c.name for c in telemetry.columns])
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df.sort_values(["asset_id", "ts"]).reset_index(drop=True)


def _latest_rows_query(table: Table) -> Any:
    """SELECT the newest row of `table` for every registered asset.

    Drives from `assets` with a correlated max(ts) per asset, which Postgres and SQLite both
    answer with one backward index probe on (asset_id, ts) -- independent of table size, so
    the fleet poll costs the same at 100k rows and at 10M.
    """
    a = assets.alias("a")
    t = table.alias("t")
    newest_ts = select(func.max(t.c.ts)).where(t.c.asset_id == a.c.asset_id).scalar_subquery()
    newest = select(a.c.asset_id, newest_ts.label("ts")).subquery("newest")
    return select(table).join(
        newest, (table.c.asset_id == newest.c.asset_id) & (table.c.ts == newest.c.ts)
    )


def latest_telemetry(engine: Engine | None = None) -> pd.DataFrame:
    """One most-recent telemetry row per registered asset (see `_latest_rows_query`)."""
    engine = engine or get_engine()
    q = _latest_rows_query(telemetry)
    with engine.connect() as conn:
        df = pd.read_sql(q, conn)
    if df.empty:
        return df
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df.sort_values("ts").reset_index(drop=True)


def insert_predictions(rows: Iterable[dict[str, Any]], engine: Engine | None = None) -> int:
    """Bulk insert prediction rows in one transaction.

    Input: dicts with asset_id, ts, model_version, failure_probability and optional source.
    Output: number of rows inserted. The edge posts these in batches of a few hundred.
    """
    engine = engine or get_engine()
    payload = [{"source": "edge", **r} for r in rows]
    if not payload:
        return 0
    if is_postgres(engine):
        from sqlalchemy.dialects.postgresql import insert as pg_insert

        stmt = pg_insert(predictions).on_conflict_do_nothing(index_elements=list(PREDICTIONS_KEY))
    else:
        stmt = insert(predictions).prefix_with("OR IGNORE")
    with engine.begin() as conn:
        conn.execute(stmt, payload)
    return len(payload)


def insert_prediction(
    asset_id: str,
    ts: datetime,
    model_version: str,
    failure_probability: float,
    source: str = "edge",
    engine: Engine | None = None,
) -> None:
    insert_predictions(
        [
            {
                "asset_id": asset_id,
                "ts": ts,
                "model_version": model_version,
                "failure_probability": failure_probability,
                "source": source,
            }
        ],
        engine=engine,
    )


def latest_predictions(engine: Engine | None = None) -> dict[str, float]:
    """asset_id -> most recent failure_probability (ties on ts broken by the newest row id).

    One index probe per registered asset (see `_latest_rows_query`); the predictions table
    grows by one row per telemetry message, so it must never be read whole.
    """
    engine = engine or get_engine()
    q = _latest_rows_query(predictions).order_by(predictions.c.asset_id, predictions.c.id)
    with engine.connect() as conn:
        rows = conn.execute(q).mappings().all()
    out: dict[str, float] = {}
    for r in rows:  # ordered by id, so the last write per asset wins
        out[r["asset_id"]] = float(r["failure_probability"])
    return out


# --------------------------------------------------------------------------------------
# Pipeline runs
# --------------------------------------------------------------------------------------


def save_run(
    run_id: str,
    asset_id: str,
    stage: str,
    state: dict[str, Any],
    engine: Engine | None = None,
) -> None:
    """Upsert a pipeline run's serialized GraphState. Runs are mutable; contracts are not."""
    engine = engine or get_engine()
    now = now_utc()
    with engine.begin() as conn:
        exists = conn.execute(
            select(pipeline_runs.c.run_id).where(pipeline_runs.c.run_id == run_id)
        ).first()
        if exists:
            conn.execute(
                pipeline_runs.update()
                .where(pipeline_runs.c.run_id == run_id)
                .values(stage=stage, updated_at=now, state=state)
            )
        else:
            conn.execute(
                insert(pipeline_runs).values(
                    run_id=run_id,
                    asset_id=asset_id,
                    stage=stage,
                    created_at=now,
                    updated_at=now,
                    state=state,
                )
            )


def load_run(run_id: str, engine: Engine | None = None) -> dict[str, Any] | None:
    engine = engine or get_engine()
    with engine.connect() as conn:
        row = conn.execute(
            select(pipeline_runs.c.state).where(pipeline_runs.c.run_id == run_id)
        ).first()
    return row[0] if row else None


def list_runs(asset_id: str | None = None, engine: Engine | None = None) -> list[dict[str, Any]]:
    engine = engine or get_engine()
    q = select(pipeline_runs).order_by(pipeline_runs.c.created_at.desc())
    if asset_id:
        q = q.where(pipeline_runs.c.asset_id == asset_id)
    with engine.connect() as conn:
        rows = conn.execute(q).mappings().all()
    return [dict(r) for r in rows]


# --------------------------------------------------------------------------------------
# Decision contracts + approvals (insert-only)
# --------------------------------------------------------------------------------------


def insert_decision_contract(dc: DecisionContract, engine: Engine | None = None) -> None:
    engine = engine or get_engine()
    with engine.begin() as conn:
        conn.execute(
            insert(decision_contracts).values(
                contract_id=dc.contract_id,
                run_id=dc.run_id,
                asset_id=dc.asset_id,
                created_at=dc.created_at,
                model_version=dc.model_version,
                recommendation=dc.recommendation,
                failure_probability=dc.failure_probability,
                expected_cost=dc.expected_cost,
                evidence_hash=dc.evidence_hash,
                payload=dump_model(dc),
            )
        )


def get_decision_contract(
    contract_id: str, engine: Engine | None = None
) -> DecisionContract | None:
    engine = engine or get_engine()
    with engine.connect() as conn:
        row = conn.execute(
            select(decision_contracts.c.payload).where(
                decision_contracts.c.contract_id == contract_id
            )
        ).first()
    return DecisionContract.model_validate(row[0]) if row else None


def list_decision_contracts(
    asset_id: str | None = None, engine: Engine | None = None
) -> list[DecisionContract]:
    engine = engine or get_engine()
    q = select(decision_contracts.c.payload).order_by(decision_contracts.c.created_at.desc())
    if asset_id:
        q = q.where(decision_contracts.c.asset_id == asset_id)
    with engine.connect() as conn:
        rows = conn.execute(q).all()
    return [DecisionContract.model_validate(r[0]) for r in rows]


def insert_approval(a: Approval, engine: Engine | None = None) -> None:
    engine = engine or get_engine()
    with engine.begin() as conn:
        conn.execute(insert(approvals).values(**a.model_dump()))


def approvals_for(
    contract_id: str | None = None,
    promotion_id: str | None = None,
    engine: Engine | None = None,
) -> list[Approval]:
    engine = engine or get_engine()
    q = select(approvals).order_by(approvals.c.decided_at)
    if contract_id:
        q = q.where(approvals.c.contract_id == contract_id)
    if promotion_id:
        q = q.where(approvals.c.promotion_id == promotion_id)
    with engine.connect() as conn:
        rows = conn.execute(q).mappings().all()
    return [Approval(**dict(r)) for r in rows]


def decided_contract_ids(engine: Engine | None = None) -> set[str]:
    """Every contract_id that has at least one approvals row (i.e. is no longer pending)."""
    engine = engine or get_engine()
    with engine.connect() as conn:
        rows = conn.execute(
            select(approvals.c.contract_id).where(approvals.c.contract_id.is_not(None)).distinct()
        ).all()
    return {r[0] for r in rows}


def list_pending_decision_contracts(engine: Engine | None = None) -> list[DecisionContract]:
    """Contracts with no recorded decision yet, newest first -- two queries, not one per row."""
    engine = engine or get_engine()
    decided = decided_contract_ids(engine)
    return [c for c in list_decision_contracts(engine=engine) if c.contract_id not in decided]


def open_contract_by_asset(engine: Engine | None = None) -> dict[str, str]:
    """asset_id -> newest pending contract_id, reading only ids (no JSON payloads)."""
    engine = engine or get_engine()
    decided = decided_contract_ids(engine)
    q = select(decision_contracts.c.asset_id, decision_contracts.c.contract_id).order_by(
        decision_contracts.c.created_at.desc()
    )
    with engine.connect() as conn:
        rows = conn.execute(q).all()
    out: dict[str, str] = {}
    for asset_id, contract_id in rows:
        if asset_id not in out and contract_id not in decided:
            out[asset_id] = contract_id
    return out


# --------------------------------------------------------------------------------------
# Work orders (derived from approvals; progress events are insert-only)
# --------------------------------------------------------------------------------------


def insert_work_order_event(ev: WorkOrderEvent, engine: Engine | None = None) -> None:
    engine = engine or get_engine()
    with engine.begin() as conn:
        conn.execute(insert(work_order_events).values(**ev.model_dump()))


def list_work_order_events(
    contract_id: str | None = None, engine: Engine | None = None
) -> list[WorkOrderEvent]:
    """Progress events oldest first (optionally for one contract)."""
    engine = engine or get_engine()
    q = select(work_order_events).order_by(
        work_order_events.c.plant_ts, work_order_events.c.recorded_at
    )
    if contract_id:
        q = q.where(work_order_events.c.contract_id == contract_id)
    with engine.connect() as conn:
        rows = conn.execute(q).mappings().all()
    return [WorkOrderEvent(**dict(r)) for r in rows]


def _as_utc(dt: datetime) -> datetime:
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=UTC)


def list_work_orders(asset_id: str | None = None, engine: Engine | None = None) -> list[WorkOrder]:
    """Approved maintain_now / maintain_later contracts with the plant's reported status.

    Inputs: optional asset filter. Output: WorkOrder list, newest approval first. Rejected
    contracts and run_to_failure recommendations are not work orders -- nothing is handed to
    the plant for them. Derived on every call; nothing here is stored beyond the approvals and
    the events.
    """
    engine = engine or get_engine()
    q = (
        select(
            approvals.c.contract_id,
            approvals.c.approver,
            approvals.c.decided_at,
            decision_contracts.c.payload,
        )
        .select_from(
            approvals.join(
                decision_contracts, approvals.c.contract_id == decision_contracts.c.contract_id
            )
        )
        .where(approvals.c.decision == "approved")
        .order_by(approvals.c.decided_at.desc())
    )
    if asset_id:
        q = q.where(decision_contracts.c.asset_id == asset_id)
    with engine.connect() as conn:
        rows = conn.execute(q).all()
    if not rows:
        return []
    events_by_contract: dict[str, list[WorkOrderEvent]] = {}
    for ev in list_work_order_events(engine=engine):
        events_by_contract.setdefault(ev.contract_id, []).append(ev)
    out: list[WorkOrder] = []
    for contract_id, approver, decided_at, payload in rows:
        dc = DecisionContract.model_validate(payload)
        if dc.recommendation == "run_to_failure":
            continue
        started = completed = None
        for ev in events_by_contract.get(contract_id, []):
            plant_ts = _as_utc(ev.plant_ts)  # SQLite hands back naive datetimes
            if ev.status == "in_progress" and started is None:
                started = plant_ts
            elif ev.status == "completed":
                completed = plant_ts
                if started is None:
                    started = plant_ts
        status = "completed" if completed else "in_progress" if started else "scheduled"
        out.append(
            WorkOrder(
                contract_id=contract_id,
                asset_id=dc.asset_id,
                recommendation=dc.recommendation,  # type: ignore[arg-type]
                window_start=dc.window_start,
                window_end=dc.window_end,
                planned_downtime_h=float(
                    dc.cost_comparison.assumptions.get("downtime_planned_h", 3.0)
                ),
                approved_at=_as_utc(decided_at),
                approver=approver,
                status=status,  # type: ignore[arg-type]
                started_ts=started,
                completed_ts=completed,
            )
        )
    return out


def work_order_by_asset(engine: Engine | None = None) -> dict[str, WorkOrder]:
    """asset_id -> its most recently approved work order (any status)."""
    out: dict[str, WorkOrder] = {}
    for wo in list_work_orders(engine=engine):
        out.setdefault(wo.asset_id, wo)
    return out


def contract_status(contract_id: str, engine: Engine | None = None) -> str:
    """Derived, never stored: 'pending' until an approvals row exists."""
    decisions = approvals_for(contract_id=contract_id, engine=engine)
    if not decisions:
        return "pending"
    return decisions[-1].decision
