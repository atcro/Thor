"""3-D Digital Twin + X-Ray routes (Phase A, CLAUDE.md section 15).

    GET /twin                         -> TwinSnapshot   (polled every 2 s by the browser)
    GET /twin/hud/{asset_id}          -> TwinHud        (X-Ray HUD panel, 404 unknown asset)
    GET /twin/rewind?max_points=&asset_id=  -> TwinRewind (stored predictions only)
    GET /twin/events?since=0          -> TwinEventBatch (pulse source, polling form)
    WS  /ws/twin                      -> TwinEventBatch frames every 250 ms when there are
                                         new events, a heartbeat at least every 2 s

Read-only and deterministic: every number comes from the same stores as /fleet (assets,
telemetry, predictions, contracts, approvals, work-order events, the model registry) plus
the in-process live bus (`apps/api/live.py`) for measured arrival rates and latencies. No
LLM call, no per-asset query loops on the polled path, and -- as everywhere else -- the
simulator's `health`, `failure_within_h` and `regime` columns are never read: the regime is
inferred from rpm/load with the production baseline (`routes/fleet.py::infer_regimes`).

Status glyph: offline (plant has the motor out for an approved work order) > no_data (its
messages stopped) > critical / warning / healthy from the DISPLAYED (median) failure
probability against `Thresholds` > no_prediction. Decision badge: work order in progress /
completed / approved (scheduled) > contract pending > rejected > none.

`apps/api/plant.py` (PLC tags and the simulated MES feed) is imported here and nowhere else:
MES context is HUD context only and never reaches a model, score or decision.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from datetime import UTC, datetime, timedelta
from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Query, WebSocket, WebSocketDisconnect
from sqlalchemy import func, select
from sqlalchemy.engine import Engine

from apps.api import db, live, plant
from apps.api.routes.fleet import (
    FleetInputs,
    _clean_row,
    _health_score,
    _ts_of,
    load_fleet_inputs,
)
from apps.api.schemas import (
    Asset,
    DataState,
    DecisionState,
    ModelProvenance,
    ModelStage,
    ObservedTelemetryRow,
    PredictionPoint,
    RewindMarker,
    RewindSeries,
    SiteTab,
    Thresholds,
    TwinAsset,
    TwinEventBatch,
    TwinHud,
    TwinLine,
    TwinRewind,
    TwinSnapshot,
    TwinStatus,
    WorkOrder,
)

log = logging.getLogger("thor.routes.twin")
router = APIRouter(tags=["twin"])

#: Machine order along a packaging line (infeed first, case packer last).
PACKAGING_ROLE_ORDER: tuple[str, ...] = (
    "infeed conveyor",
    "filler",
    "capper/labeler",
    "case packer",
)
SITE_TABS: tuple[SiteTab, ...] = (
    SiteTab(name="Packaging", available=True),
    SiteTab(name="Molding", available=False, note="site tab only -- not built"),
    SiteTab(name="Batching", available=False, note="site tab only -- not built"),
)
#: Per-sample prediction rows behind the HUD trend and the rewind scrubber.
HUD_TREND_ROWS: int = 36
REWIND_ROWS_PER_ASSET: int = 6_000
#: |median(last 3) - median(previous 3)| at or above this is a trend, below it is flat.
TREND_DELTA: float = 0.05
WS_TICK_S: float = 0.25
WS_HEARTBEAT_S: float = 2.0
EVENTS_MAX_PER_BATCH: int = 500


# --------------------------------------------------------------------------------------
# Plant layout (asset metadata only)
# --------------------------------------------------------------------------------------


def _trailing_number(name: str) -> int | None:
    digits = ""
    for ch in reversed(name.strip()):
        if ch.isdigit():
            digits = ch + digits
        else:
            break
    return int(digits) if digits else None


def _line_sort_key(line: str) -> tuple[int, int, str]:
    """Packaging lines first by their number, then by name; the Compressor Hall last."""
    n = _trailing_number(line)
    return (1 if plant.is_compressor_line(line) else 0, n if n is not None else 10**6, line)


def _asset_order_key(a: Asset) -> tuple[int, str]:
    role = a.role.strip().lower()
    idx = PACKAGING_ROLE_ORDER.index(role) if role in PACKAGING_ROLE_ORDER else len(
        PACKAGING_ROLE_ORDER
    )
    return (idx, a.asset_id)


def group_lines(assets: list[Asset]) -> list[TwinLine]:
    """Group assets into TwinLines from their `line` / `role` metadata only.

    Input: registered assets. Output: lines in plant order (Packaging Line 1..5, then the
    Compressor Hall; unknown names alphabetically before any compressor line), `index`
    sequential in that order, `kind` "compressor" iff the name contains "Compressor", and
    the assets inside a line in machine order (infeed conveyor -> filler -> capper/labeler ->
    case packer, unknown roles after them by asset id). No MES context here.
    """
    by_line: dict[str, list[Asset]] = {}
    for a in assets:
        by_line.setdefault(a.line, []).append(a)
    out: list[TwinLine] = []
    for idx, name in enumerate(sorted(by_line, key=_line_sort_key)):
        members = sorted(by_line[name], key=_asset_order_key)
        out.append(
            TwinLine(
                line=name,
                kind="compressor" if plant.is_compressor_line(name) else "packaging",
                index=idx,
                assets=[a.asset_id for a in members],
            )
        )
    return out


def lines_with_mes(assets: list[Asset], plant_time: datetime | None) -> list[TwinLine]:
    """`group_lines` plus the simulated MES feed per packaging line at `plant_time`."""
    lines = group_lines(assets)
    if plant_time is None:
        return lines
    return [
        ln.model_copy(update={"mes": plant.simulated_mes_context(ln.line, plant_time)})
        for ln in lines
    ]


# --------------------------------------------------------------------------------------
# Status and governance badge
# --------------------------------------------------------------------------------------


def twin_status(
    p_fail: float | None,
    data_state: DataState,
    work_order: WorkOrder | None,
    thresholds: Thresholds,
) -> TwinStatus:
    """Status glyph with the precedence offline > no_data > critical > warning > healthy >
    no_prediction. Inputs: displayed (median) failure probability, wall-clock data state,
    newest work order and the thresholds. Output: TwinStatus."""
    if work_order is not None and work_order.status == "in_progress":
        return TwinStatus.offline
    if data_state != DataState.live:
        return TwinStatus.no_data
    if p_fail is None:
        return TwinStatus.no_prediction
    if p_fail >= thresholds.classification:
        return TwinStatus.critical
    if p_fail >= thresholds.early_warning:
        return TwinStatus.warning
    return TwinStatus.healthy


def _newest_contract_states(engine: Engine) -> dict[str, tuple[str, datetime, str | None]]:
    """asset_id -> (contract_id, created_at, last decision or None) of its newest contract.

    Two column-only queries for the whole fleet (no JSON payloads, nothing per asset).
    """
    a = db.approvals
    with engine.connect() as conn:
        decisions = conn.execute(
            select(a.c.contract_id, a.c.decision)
            .where(a.c.contract_id.is_not(None))
            .order_by(a.c.decided_at)
        ).all()
        contracts = conn.execute(
            select(
                db.decision_contracts.c.asset_id,
                db.decision_contracts.c.contract_id,
                db.decision_contracts.c.created_at,
            ).order_by(db.decision_contracts.c.created_at.desc())
        ).all()
    decision_by_contract: dict[str, str] = {}
    for contract_id, decision in decisions:
        decision_by_contract[str(contract_id)] = str(decision)  # last one wins
    out: dict[str, tuple[str, datetime, str | None]] = {}
    for asset_id, contract_id, created_at in contracts:
        if asset_id in out:
            continue
        created = created_at if created_at.tzinfo else created_at.replace(tzinfo=UTC)
        out[asset_id] = (str(contract_id), created, decision_by_contract.get(str(contract_id)))
    return out


def decision_state(
    work_order: WorkOrder | None,
    newest_contract: tuple[str, datetime, str | None] | None,
) -> DecisionState:
    """Governance badge for one motor.

    Inputs: its newest approved work order (any status) and its newest contract as
    (id, created_at, decision | None). Output: work_order_in_progress / work_order_completed /
    approved (scheduled) from the work order; else contract_pending when the newest contract
    is undecided; else rejected when it was rejected; else none. A pending contract created
    after the work order's approval wins over that older work order.
    """
    pending = newest_contract is not None and newest_contract[2] is None
    if pending and (work_order is None or newest_contract[1] > work_order.approved_at):
        return DecisionState.contract_pending
    if work_order is not None:
        if work_order.status == "in_progress":
            return DecisionState.work_order_in_progress
        if work_order.status == "completed":
            return DecisionState.work_order_completed
        return DecisionState.approved
    if pending:
        return DecisionState.contract_pending
    if newest_contract is not None and newest_contract[2] == "rejected":
        return DecisionState.rejected
    return DecisionState.none


# --------------------------------------------------------------------------------------
# Snapshot
# --------------------------------------------------------------------------------------


def compute_twin_assets(
    engine: Engine, now: float | None = None
) -> tuple[FleetInputs, list[TwinAsset]]:
    """One TwinAsset per registered asset, from `load_fleet_inputs` + the live bus.

    Inputs: engine and an optional wall-clock `now` (epoch seconds; tests). Output: the
    FleetInputs used (so callers can reuse the spec / latest rows) and the assets in
    asset-id order. Fixed query count regardless of fleet size.
    """
    now = time.time() if now is None else now
    inputs = load_fleet_inputs(engine)
    contract_states = _newest_contract_states(engine)
    live_stats = live.bus.asset_live_stats(now)
    thresholds = Thresholds()
    out: list[TwinAsset] = []
    for a in inputs.assets:
        row = inputs.latest_by_asset.get(a.asset_id)
        p = inputs.preds.get(a.asset_id)
        p_fail = float(p["failure_probability"]) if p else None
        health = _health_score(p_fail)
        wo = inputs.work_orders.get(a.asset_id)
        stats = live_stats.get(a.asset_id)
        if stats is None:
            state = DataState.none
        else:
            state = (
                DataState.live
                if now - stats[0].timestamp() <= live.STALE_AFTER_S
                else DataState.stale
            )
        latest: ObservedTelemetryRow | None = None
        if row is not None:
            latest = ObservedTelemetryRow.from_row(_clean_row(row))
        out.append(
            TwinAsset(
                asset=a,
                status=twin_status(p_fail, state, wo, thresholds),
                data_state=state,
                decision_state=decision_state(wo, contract_states.get(a.asset_id)),
                failure_probability=p_fail,
                health_score=round(health, 1) if health is not None else None,
                prediction_ts=_ts_of(p["ts"]) if p else None,
                model_version=p["model_version"] if p else None,
                n_prediction_rows=int(p["n_rows"]) if p and p.get("n_rows") is not None else None,
                regime=inputs.regime_by_asset.get(a.asset_id),
                latest=latest,
                last_ts=latest.ts if latest is not None else None,
                last_received_at=stats[0] if stats else None,
                msgs_last_60s=stats[1] if stats else 0,
                open_contract_id=inputs.open_contracts.get(a.asset_id),
                work_order=wo,
            )
        )
    return inputs, out


def model_provenance(engine: Engine) -> ModelProvenance | None:
    """The production registry row joined with its approved promotion request and approval.

    Output: ModelProvenance, or None when no model is in production (never raises). Parity,
    validation, lead-time and calibration facts come from the registry row's payload as
    `register_model()` wrote it; each is None when the payload lacks it.
    """
    try:
        r = db.model_registry
        with engine.connect() as conn:
            row = (
                conn.execute(
                    select(r)
                    .where(r.c.stage == ModelStage.production.value)
                    .order_by(r.c.registered_at.desc(), r.c.id.desc())
                )
                .mappings()
                .first()
            )
            if row is None:
                return None
            pr = db.promotion_requests
            promo = conn.execute(
                select(pr.c.promotion_id)
                .where(
                    pr.c.model_name == row["name"],
                    pr.c.version == row["version"],
                    pr.c.status == "approved",
                )
                .order_by(pr.c.created_at.desc())
            ).first()
        promotion_id = str(promo[0]) if promo else None
        approver: str | None = None
        approved_at: datetime | None = None
        if promotion_id:
            decided = [
                ap
                for ap in db.approvals_for(promotion_id=promotion_id, engine=engine)
                if ap.decision == "approved"
            ]
            if decided:
                approver = decided[-1].approver
                approved_at = decided[-1].decided_at
                if approved_at.tzinfo is None:
                    approved_at = approved_at.replace(tzinfo=UTC)
        payload: dict[str, Any] = dict(row["payload"] or {})
        metrics = payload.get("metrics") or {}
        parity = payload.get("parity") or {}
        validation = payload.get("validation") or {}
        calibration = validation.get("calibration") or {}
        registered_at = row["registered_at"]
        if registered_at.tzinfo is None:
            registered_at = registered_at.replace(tzinfo=UTC)
        return ModelProvenance(
            name=str(row["name"]),
            version=str(row["version"]),
            family=str(row["family"]),
            stage=ModelStage(row["stage"]),
            ims_total=float(row["ims_total"]),
            registered_at=registered_at,
            approver=approver,
            approved_at=approved_at,
            promotion_id=promotion_id,
            validation_passed=payload.get("validation_passed"),
            parity_passed=payload.get("parity_passed"),
            parity_max_abs_diff=parity.get("max_abs_diff_probability"),
            lead_time_median_h=metrics.get("lead_time_median_h"),
            calibration_method=calibration.get("method"),
            onnx_path=row.get("onnx_path"),
        )
    except Exception as exc:  # noqa: BLE001 - provenance is a label, never a failure
        log.warning("model provenance unavailable: %s", exc)
        return None


def build_snapshot(engine: Engine, now: float | None = None) -> TwinSnapshot:
    """Assemble the TwinSnapshot (see module docstring). Inputs: engine, optional `now`."""
    inputs, assets = compute_twin_assets(engine, now)
    plant_time = max((a.last_ts for a in assets if a.last_ts is not None), default=None)
    return TwinSnapshot(
        site=inputs.assets[0].site if inputs.assets else "",
        sites=list(SITE_TABS),
        plant_time=plant_time,
        thresholds=Thresholds(),
        lines=lines_with_mes(inputs.assets, plant_time),
        assets=assets,
        model=model_provenance(engine),
        live=live.bus.snapshot_counters(now),
    )


@router.get("/twin", response_model=TwinSnapshot)
def get_twin() -> TwinSnapshot:
    """Everything the 3-D view needs in one call (cheap: a fixed handful of queries)."""
    return build_snapshot(db.get_engine())


# --------------------------------------------------------------------------------------
# HUD
# --------------------------------------------------------------------------------------


def _prediction_history(engine: Engine, asset_id: str, limit: int) -> list[PredictionPoint]:
    """Newest `limit` per-sample prediction rows for one asset, returned oldest first."""
    p = db.predictions
    q = (
        select(p.c.ts, p.c.failure_probability, p.c.model_version, p.c.source)
        .where(p.c.asset_id == asset_id)
        .order_by(p.c.ts.desc(), p.c.id.desc())
        .limit(limit)
    )
    with engine.connect() as conn:
        rows = conn.execute(q).all()
    out = [
        PredictionPoint(
            ts=live.as_utc(ts),
            failure_probability=float(prob),
            model_version=str(version),
            source=str(source),
        )
        for ts, prob, version, source in rows
    ]
    out.reverse()
    return out


def _median(values: list[float]) -> float:
    s = sorted(values)
    mid = len(s) // 2
    return float(s[mid] if len(s) % 2 else (s[mid - 1] + s[mid]) / 2.0)


def trend_of(probabilities: list[float]) -> Literal["rising", "falling", "flat"] | None:
    """Trend of a per-sample series (oldest first): median of the last 3 vs the median of the
    3 before. rising when >= +TREND_DELTA, falling when <= -TREND_DELTA, else flat; None with
    fewer than 6 rows."""
    if len(probabilities) < 6:
        return None
    delta = _median(probabilities[-3:]) - _median(probabilities[-6:-3])
    if delta >= TREND_DELTA:
        return "rising"
    if delta <= -TREND_DELTA:
        return "falling"
    return "flat"


@router.get("/twin/hud/{asset_id}", response_model=TwinHud)
def get_hud(asset_id: str) -> TwinHud:
    """X-Ray HUD for one motor: its TwinAsset and line (with MES), the last HUD_TREND_ROWS
    per-sample predictions with a trend, PLC tags, the newest Decision Contract's SHAP drivers
    with their source, the contract status and the production model. 404 unknown asset."""
    engine = db.get_engine()
    inputs, assets = compute_twin_assets(engine)
    twin_asset = next((a for a in assets if a.asset.asset_id == asset_id), None)
    if twin_asset is None:
        raise HTTPException(status_code=404, detail=f"unknown asset {asset_id}")
    plant_time = max((a.last_ts for a in assets if a.last_ts is not None), default=None)
    line = next(
        (ln for ln in lines_with_mes(inputs.assets, plant_time) if asset_id in ln.assets), None
    )
    history = _prediction_history(engine, asset_id, HUD_TREND_ROWS)
    baseline = inputs.spec.baseline if inputs.spec is not None else None
    contracts = db.list_decision_contracts(asset_id=asset_id, engine=engine)
    contract = contracts[0] if contracts else None
    drivers_source = None
    status: str | None = None
    if contract is not None:
        from apps.api.schemas import DriverSource

        drivers_source = DriverSource(
            contract_id=contract.contract_id,
            run_id=contract.run_id,
            created_at=contract.created_at,
            model_version=contract.model_version,
        )
        status = db.contract_status(contract.contract_id, engine=engine)
    return TwinHud(
        asset=twin_asset,
        line=line,
        thresholds=Thresholds(),
        prediction_trend=history,
        trend=trend_of([pt.failure_probability for pt in history]),
        plc_tags=plant.plc_tags(twin_asset.latest, twin_asset.regime, baseline),
        drivers=list(contract.top_features) if contract is not None else [],
        drivers_source=drivers_source,
        contract=contract,
        contract_status=status,
        model=model_provenance(engine),
        mes=line.mes if line is not None else None,
    )


# --------------------------------------------------------------------------------------
# Rewind
# --------------------------------------------------------------------------------------


def _rewind_rows(
    engine: Engine, asset_id: str | None, per_asset: int
) -> dict[str, list[tuple[datetime, float]]]:
    """Newest `per_asset` prediction rows per asset (one window-function query), oldest
    first, with same-ts rows collapsed to the last written one."""
    p = db.predictions
    rn = (
        func.row_number()
        .over(partition_by=p.c.asset_id, order_by=(p.c.ts.desc(), p.c.id.desc()))
        .label("rn")
    )
    ranked = select(p.c.asset_id, p.c.ts, p.c.id, p.c.failure_probability, rn)
    if asset_id:
        ranked = ranked.where(p.c.asset_id == asset_id)
    ranked = ranked.subquery("ranked")
    q = (
        select(ranked.c.asset_id, ranked.c.ts, ranked.c.failure_probability)
        .where(ranked.c.rn <= per_asset)
        .order_by(ranked.c.asset_id, ranked.c.ts, ranked.c.id)
    )
    with engine.connect() as conn:
        rows = conn.execute(q).all()
    out: dict[str, list[tuple[datetime, float]]] = {}
    for aid, ts, prob in rows:
        series = out.setdefault(str(aid), [])
        t = live.as_utc(ts)
        if series and series[-1][0] == t:
            series[-1] = (t, float(prob))  # same ts: the last written row wins
        else:
            series.append((t, float(prob)))
    return out


def downsample(points: list[tuple[datetime, float]], max_points: int) -> list[tuple[datetime, float]]:
    """Uniformly thin `points` to at most `max_points`, always keeping the first and last."""
    n = len(points)
    if n <= max_points:
        return points
    if max_points <= 1:
        return [points[-1]]
    idx = sorted({round(i * (n - 1) / (max_points - 1)) for i in range(max_points)})
    return [points[i] for i in idx]


def first_sustained_index(
    probabilities: list[float], threshold: float, sustained_rows: int
) -> int | None:
    """Index of the first row that starts `sustained_rows` consecutive rows >= threshold
    (the early-warning alarm's own rule); None if it never happens. A single spike never fires."""
    run = 0
    for i, p in enumerate(probabilities):
        run = run + 1 if p >= threshold else 0
        if run >= sustained_rows:
            return i - sustained_rows + 1
    return None


def rewind_series(
    asset_id: str,
    points: list[tuple[datetime, float]],
    work_orders: list[WorkOrder],
    thresholds: Thresholds,
    max_points: int,
) -> RewindSeries:
    """Markers and lead time for one asset's prediction series (oldest first).

    Markers sit on the plant-time axis only: `early_warning` (first of
    `thresholds.sustained_rows` consecutive rows >= early_warning), `predicted_failure` (first
    row >= classification) and the plant's work-order events (`started_ts` / `completed_ts`).
    Contract and approval wall-clock times are never placed on this axis. `lead_time_h` is
    work_order_started - early_warning in plant hours when both exist.
    """
    probs = [p for _, p in points]
    markers: list[RewindMarker] = []
    ew_ts: datetime | None = None
    i_ew = first_sustained_index(probs, thresholds.early_warning, thresholds.sustained_rows)
    if i_ew is not None:
        ew_ts = points[i_ew][0]
        markers.append(
            RewindMarker(
                kind="early_warning",
                ts=ew_ts,
                label="Early warning fired",
                detail=(
                    f"{thresholds.sustained_rows} consecutive predictions >= "
                    f"{thresholds.early_warning:g}"
                ),
            )
        )
    i_pf = next((i for i, p in enumerate(probs) if p >= thresholds.classification), None)
    if i_pf is not None:
        markers.append(
            RewindMarker(
                kind="predicted_failure",
                ts=points[i_pf][0],
                label="Predicted failure",
                detail=f"first prediction >= {thresholds.classification:g}",
            )
        )
    started: list[datetime] = []
    for wo in work_orders:
        if wo.started_ts is not None:
            started.append(wo.started_ts)
            markers.append(
                RewindMarker(
                    kind="work_order_started",
                    ts=wo.started_ts,
                    label="Plant took the motor offline",
                    detail=f"work order {wo.contract_id} in progress (plant report)",
                )
            )
        if wo.completed_ts is not None:
            markers.append(
                RewindMarker(
                    kind="work_order_completed",
                    ts=wo.completed_ts,
                    label="Motor back in service",
                    detail=f"work order {wo.contract_id} completed (plant report)",
                )
            )
    markers.sort(key=lambda m: m.ts)
    lead: float | None = None
    if ew_ts is None:
        basis = "no early warning"
    else:
        after = sorted(t for t in started if t >= ew_ts)
        if after:
            lead = round((after[0] - ew_ts) / timedelta(hours=1), 2)
            basis = "early warning -> plant took the motor offline"
        else:
            active_h = (points[-1][0] - ew_ts) / timedelta(hours=1) if points else 0.0
            basis = f"no work order yet; warning active for {active_h:.1f} h"
    return RewindSeries(
        asset_id=asset_id,
        points=downsample(points, max_points),
        n_rows=len(points),
        markers=markers,
        lead_time_h=lead,
        lead_time_basis=basis,
    )


@router.get("/twin/rewind", response_model=TwinRewind)
def get_rewind(
    max_points: int = Query(400, ge=2, le=10_000),
    asset_id: str | None = Query(None),
) -> TwinRewind:
    """Time scrubber over STORED predictions (newest REWIND_ROWS_PER_ASSET rows per asset,
    downsampled to <= max_points) with plant-time markers; optionally one asset."""
    engine = db.get_engine()
    thresholds = Thresholds()
    rows = _rewind_rows(engine, asset_id, REWIND_ROWS_PER_ASSET)
    wos_by_asset: dict[str, list[WorkOrder]] = {}
    for wo in db.list_work_orders(asset_id=asset_id, engine=engine):
        wos_by_asset.setdefault(wo.asset_id, []).append(wo)
    series = [
        rewind_series(aid, pts, wos_by_asset.get(aid, []), thresholds, max_points)
        for aid, pts in sorted(rows.items())
        if pts
    ]
    all_ts = [t for pts in rows.values() for t, _ in pts]
    model = model_provenance(engine)
    return TwinRewind(
        start=min(all_ts) if all_ts else None,
        end=max(all_ts) if all_ts else None,
        thresholds=thresholds,
        assets=series,
        validated_lead_time_h=model.lead_time_median_h if model else None,
        validated_lead_time_basis=(
            "validation report, held-out test motors" if model else "no production model"
        ),
    )


# --------------------------------------------------------------------------------------
# Events (pulse source)
# --------------------------------------------------------------------------------------


def event_batch(since: int) -> TwinEventBatch:
    """Events after `since` (aggregated per asset/kind beyond EVENTS_MAX_PER_BATCH) + counters."""
    events, nxt, dropped = live.bus.events_since(since, EVENTS_MAX_PER_BATCH)
    return TwinEventBatch(
        since=since, next=nxt, events=events, dropped=dropped, live=live.bus.snapshot_counters()
    )


@router.get("/twin/events", response_model=TwinEventBatch)
def get_events(since: int = Query(0, ge=0)) -> TwinEventBatch:
    """Polling form of the pulse stream: pass the returned `next` back as `since`."""
    return event_batch(since)


@router.websocket("/ws/twin")
async def ws_twin(websocket: WebSocket) -> None:
    """Push a TwinEventBatch frame every WS_TICK_S when new events arrived and at least every
    WS_HEARTBEAT_S regardless (the heartbeat carries `live`). The first frame is sent at once.
    A client text frame `{"since": N}` resyncs the cursor; a disconnect ends the stream."""
    await websocket.accept()
    state: dict[str, Any] = {"since": 0, "closed": False}

    async def _reader() -> None:
        try:
            while True:
                msg = await websocket.receive()
                if msg.get("type") == "websocket.disconnect":
                    break
                text = msg.get("text")
                if text:
                    try:
                        state["since"] = max(0, int(json.loads(text).get("since", 0)))
                    except (ValueError, TypeError, AttributeError):
                        log.debug("ignoring ws frame %r", text[:80])
        except Exception as e:  # noqa: BLE001 - any receive failure means the client is gone
            log.debug("ws /ws/twin reader stopped: %s", e)
        state["closed"] = True

    reader = asyncio.create_task(_reader())
    last_sent = float("-inf")
    try:
        while not state["closed"]:
            batch = await asyncio.to_thread(event_batch, int(state["since"]))
            now = time.monotonic()
            if batch.events or now - last_sent >= WS_HEARTBEAT_S:
                await websocket.send_text(batch.model_dump_json())
                state["since"] = batch.next
                last_sent = now
            await asyncio.sleep(WS_TICK_S)
    except WebSocketDisconnect:
        pass
    except Exception as e:  # noqa: BLE001
        log.debug("ws /ws/twin closed: %s", e)
    finally:
        reader.cancel()
    try:
        await websocket.close()
    except Exception:  # noqa: BLE001
        pass
