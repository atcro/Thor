"""Shared Pydantic contracts for every Thor toolbox and the LangGraph state.

Every deterministic tool function takes and returns one of these models. This is the single
source of truth for what flows between agents 04 → 05 → 06 → 07/08 → 09. If you change a
field here, update CLAUDE.md §6 in the same commit.

Nothing in this module calls an LLM or touches the database.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

# --------------------------------------------------------------------------------------
# Shared primitives
# --------------------------------------------------------------------------------------

SENSOR_COLUMNS: tuple[str, ...] = (
    "vibration_rms",
    "vibration_kurtosis",
    "vibration_crest",
    "bearing_temp_c",
    "motor_temp_c",
    "current_a",
    "rpm",
    "load_pct",
)


class TelemetryRow(BaseModel):
    """One sensor sample for one asset — the unit that flows over MQTT and into TimescaleDB."""

    asset_id: str
    ts: datetime
    vibration_rms: float
    vibration_kurtosis: float
    vibration_crest: float
    bearing_temp_c: float
    motor_temp_c: float
    current_a: float
    rpm: float
    load_pct: float
    regime: str | None = None
    health: float | None = Field(
        default=None, ge=0.0, le=1.0, description="1 = healthy; simulator ground truth"
    )
    failure_within_h: float | None = Field(
        default=None, description="Simulator ground truth: hours to failure, null if none"
    )


class Asset(BaseModel):
    asset_id: str
    name: str
    site: str
    line: str
    asset_type: str = "induction_motor"
    rated_kw: float = 75.0
    criticality: Literal["low", "medium", "high"] = "medium"


# --------------------------------------------------------------------------------------
# 04 — Data Reliability
# --------------------------------------------------------------------------------------


class MissingnessReport(BaseModel):
    missing_fraction: dict[str, float] = Field(
        default_factory=dict, description="column → fraction missing"
    )
    gap_windows: list[tuple[datetime, datetime]] = Field(default_factory=list)
    flatlined_sensors: list[str] = Field(default_factory=list)


class Regime(BaseModel):
    regime_id: str
    label: str
    n_rows: int
    rpm_mean: float
    load_mean: float
    share: float = Field(ge=0.0, le=1.0)


class RegimeReport(BaseModel):
    regimes: list[Regime]
    row_regime: list[str] = Field(
        default_factory=list, description="per-row regime_id, same order as input"
    )
    method: str = "kmeans"
    silhouette: float | None = None


class SchemaIssue(BaseModel):
    column: str
    issue: str
    severity: Literal["info", "warning", "error"]


class DataQualityContract(BaseModel):
    """Output of 04. Consumed by 05. Score < 60 blocks training."""

    asset_id: str
    window_start: datetime
    window_end: datetime
    n_rows: int
    n_assets: int
    sampling_rate_hz: float
    columns: list[str]
    quality_score: float = Field(ge=0.0, le=100.0)
    missingness: MissingnessReport
    regimes: RegimeReport
    schema_issues: list[SchemaIssue] = Field(default_factory=list)
    trainable: bool
    notes: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------------------
# 05 — ML Architect
# --------------------------------------------------------------------------------------


class TaskSpec(BaseModel):
    task: Literal["classification", "rul_regression"]
    target: str
    horizon_h: float = Field(default=48.0, description="Label = failure within this many hours")
    rationale: str


class FeatureSpec(BaseModel):
    features: list[str]
    window_rows: int
    regime_normalized: bool = True
    description: str
    baseline: dict[str, Any] | None = Field(
        default=None,
        description="Per-sensor regime baseline (mean/std + centroids) shipped to the edge sidecar",
    )


class IndustrialModelScore(BaseModel):
    """Weighted champion-selection score. Weights sum to 1; every input is 0–1."""

    recall: float
    precision: float
    lead_time_score: float
    calibration_score: float
    latency_score: float
    weights: dict[str, float] = Field(
        default_factory=lambda: {
            "recall": 0.35,
            "lead_time_score": 0.25,
            "precision": 0.15,
            "calibration_score": 0.15,
            "latency_score": 0.10,
        }
    )
    total: float


class CandidateModel(BaseModel):
    candidate_id: str
    family: Literal["random_forest", "xgboost", "lightgbm"]
    mlflow_run_id: str | None = None
    params: dict[str, Any]
    metrics: dict[str, float]
    ims: IndustrialModelScore | None = None
    artifact_path: str | None = None
    inference_latency_ms: float


class CandidateSet(BaseModel):
    """Output of 05. Consumed by 06."""

    asset_id: str
    task: TaskSpec
    features: FeatureSpec
    candidates: list[CandidateModel]
    mlflow_experiment: str
    ranked: list[str] = Field(default_factory=list, description="candidate_ids best → worst by IMS")


# --------------------------------------------------------------------------------------
# 06 — Validation
# --------------------------------------------------------------------------------------


class LeakageReport(BaseModel):
    temporal_leakage: bool
    feature_leakage: list[str] = Field(default_factory=list)
    asset_overlap: bool
    passed: bool
    notes: list[str] = Field(default_factory=list)


class BacktestFold(BaseModel):
    fold: int
    train_assets: list[str]
    test_assets: list[str]
    train_end: datetime
    metrics: dict[str, float]


class CalibrationReport(BaseModel):
    method: Literal["isotonic", "sigmoid", "none"]
    brier_before: float
    brier_after: float
    reliability_bins: list[tuple[float, float, int]] = Field(
        default_factory=list, description="(mean_predicted, fraction_positive, count)"
    )


class LeadTimeReport(BaseModel):
    median_h: float
    p90_h: float
    p10_h: float
    n_events: int
    threshold: float


class RULInterval(BaseModel):
    p10_h: float
    p50_h: float
    p90_h: float
    method: str = "quantile_gbm"


class ValidationReport(BaseModel):
    """Output of 06. Consumed by 07 and 08."""

    asset_id: str
    champion_id: str
    champion_family: str
    champion_mlflow_run_id: str | None = None
    model_version: str
    leakage: LeakageReport
    backtest: list[BacktestFold]
    calibration: CalibrationReport
    lead_time: LeadTimeReport
    rul: RULInterval | None = None
    ims: IndustrialModelScore
    passed: bool
    notes: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------------------
# 07 — Reliability
# --------------------------------------------------------------------------------------


class ShapFeature(BaseModel):
    feature: str
    shap_value: float
    feature_value: float
    direction: Literal["raises_risk", "lowers_risk"]


class Explanation(BaseModel):
    asset_id: str
    failure_probability: float = Field(ge=0.0, le=1.0)
    top_features: list[ShapFeature]
    base_value: float
    model_version: str


class ManualPassage(BaseModel):
    source: str
    section: str
    page: int | None = None
    text: str
    score: float


class FieldCase(BaseModel):
    """One real unplanned work order from the FMUCD field-history slice (Bolt precedent, not
    manual guidance -- never cited in a Decision Contract explanation)."""

    case_id: str
    source: str = "FMUCD"
    university: int
    component: str
    description: str
    start_date: str | None = None
    labor_hours: float | None = None
    total_cost: float | None = None
    score: float


class CostOption(BaseModel):
    option: Literal["maintain_now", "maintain_later", "run_to_failure"]
    when: datetime | None
    expected_cost: float
    p_failure_before: float
    downtime_h: float
    breakdown: dict[str, float]


class CostComparison(BaseModel):
    asset_id: str
    options: list[CostOption]
    recommended: Literal["maintain_now", "maintain_later", "run_to_failure"]
    assumptions: dict[str, float] = Field(
        description="plant-cost assumptions used — configurable, not fact"
    )


class MaintenanceWindow(BaseModel):
    start: datetime
    end: datetime
    reason: str
    p_failure_before_window: float


class WhatIfResult(BaseModel):
    asset_id: str
    scenario: dict[str, float]
    baseline_probability: float
    scenario_probability: float
    delta: float


class EvidenceBundle(BaseModel):
    """Everything 07 hands to the orchestrator for the single explanation-drafting LLM call."""

    explanation: Explanation
    manual_context: list[ManualPassage]
    cost: CostComparison
    window: MaintenanceWindow
    validation: ValidationReport
    data_quality_score: float


class DecisionContractStatus(StrEnum):
    pending = "pending"
    approved = "approved"
    rejected = "rejected"
    expired = "expired"


class DecisionContract(BaseModel):
    """Immutable audit record. Insert-only. Approval is a separate linked row (Approval)."""

    contract_id: str
    created_at: datetime
    asset_id: str
    run_id: str
    model_version: str
    champion_mlflow_run_id: str | None = None
    failure_probability: float
    recommendation: Literal["maintain_now", "maintain_later", "run_to_failure"]
    window_start: datetime | None
    window_end: datetime | None
    expected_cost: float
    cost_comparison: CostComparison
    top_features: list[ShapFeature]
    manual_context: list[ManualPassage]
    lead_time_h: float
    calibration_brier: float
    data_quality_score: float
    explanation_text: str
    explanation_source: Literal["llm", "template"]
    evidence_hash: str


class Approval(BaseModel):
    approval_id: str
    contract_id: str | None = None
    promotion_id: str | None = None
    decision: Literal["approved", "rejected"]
    approver: str
    note: str = ""
    decided_at: datetime


WorkOrderStatus = Literal["scheduled", "in_progress", "completed"]


class WorkOrderEvent(BaseModel):
    """One insert-only progress report from the plant side.

    Thor hands an approved Decision Contract to the plant's CMMS and never executes the work
    itself; the plant (in the demo: the replay's simulated CMMS) reports back when the motor is
    taken offline and when it is back in service. `plant_ts` is plant time (the replayed
    telemetry clock), `recorded_at` is wall clock.
    """

    event_id: str
    contract_id: str
    asset_id: str
    status: Literal["in_progress", "completed"]
    plant_ts: datetime
    recorded_at: datetime
    source: str = "cmms-sim"
    note: str = ""


class WorkOrderEventRequest(BaseModel):
    status: Literal["in_progress", "completed"]
    plant_ts: datetime
    source: str = "cmms-sim"
    note: str = ""


class WorkOrder(BaseModel):
    """Derived view (never stored): an approved maintain_now / maintain_later contract plus the
    plant's reported progress. `status` is `scheduled` until the plant reports otherwise."""

    contract_id: str
    asset_id: str
    recommendation: Literal["maintain_now", "maintain_later"]
    window_start: datetime | None
    window_end: datetime | None
    planned_downtime_h: float
    approved_at: datetime
    approver: str
    status: WorkOrderStatus
    started_ts: datetime | None = None
    completed_ts: datetime | None = None


# --------------------------------------------------------------------------------------
# 08 — MLOps
# --------------------------------------------------------------------------------------


class ModelStage(StrEnum):
    candidate = "candidate"
    validated = "validated"
    shadow = "shadow"
    production = "production"
    archived = "archived"


class RegisteredModel(BaseModel):
    name: str
    version: str
    mlflow_run_id: str | None
    stage: ModelStage
    family: str
    ims_total: float
    git_sha: str | None = None
    dataset_version: str | None = None
    registered_at: datetime
    onnx_path: str | None = None


class ChampionComparison(BaseModel):
    challenger: RegisteredModel
    champion: RegisteredModel | None
    delta: dict[str, float]
    recommend_promote: bool
    rationale: str


class PromotionRequest(BaseModel):
    promotion_id: str
    model_name: str
    version: str
    from_stage: ModelStage
    to_stage: ModelStage
    comparison: ChampionComparison
    status: Literal["pending", "approved", "rejected"] = "pending"
    created_at: datetime


class DriftReport(BaseModel):
    asset_id: str
    model_version: str
    psi: dict[str, float]
    drifted_features: list[str]
    threshold: float
    drift_detected: bool
    window_start: datetime
    window_end: datetime


class EdgeDeployment(BaseModel):
    model_name: str
    version: str
    onnx_path: str
    deployed_at: datetime
    input_features: list[str]


# --------------------------------------------------------------------------------------
# 03 — Orchestrator state
# --------------------------------------------------------------------------------------


class PipelineStage(StrEnum):
    queued = "queued"
    profiling = "profiling"
    training = "training"
    validating = "validating"
    explaining = "explaining"
    awaiting_approval = "awaiting_approval"
    approved = "approved"
    rejected = "rejected"
    failed = "failed"


class StageEvent(BaseModel):
    ts: datetime
    stage: PipelineStage
    message: str
    tool: str | None = None
    payload: dict[str, Any] | None = None


class GraphState(BaseModel):
    """Typed LangGraph state. Every node reads/writes this and nothing else."""

    run_id: str
    asset_id: str
    stage: PipelineStage = PipelineStage.queued
    events: list[StageEvent] = Field(default_factory=list)
    data_quality: DataQualityContract | None = None
    candidates: CandidateSet | None = None
    validation: ValidationReport | None = None
    evidence: EvidenceBundle | None = None
    contract: DecisionContract | None = None
    approval: Approval | None = None
    error: str | None = None
    llm_calls: int = 0


# --------------------------------------------------------------------------------------
# API request/response shapes (02)
# --------------------------------------------------------------------------------------


class FleetAsset(BaseModel):
    asset: Asset
    health_score: float
    failure_probability: float | None
    regime: str | None
    last_ts: datetime | None
    open_contract_id: str | None
    stage: ModelStage | None
    #: Newest approved work order for this asset (plant-reported status), if any.
    work_order: WorkOrder | None = None


class PipelineRunRequest(BaseModel):
    asset_id: str
    horizon_h: float = 48.0
    n_trials: int = Field(default=6, ge=1, le=100, description="Optuna trials per family")


class PipelineRunResponse(BaseModel):
    run_id: str
    stage: PipelineStage


class DecisionRequest(BaseModel):
    decision: Literal["approved", "rejected"]
    approver: str
    note: str = ""


COPILOT_MAX_MESSAGE_CHARS = 2000
# Bolt's own replies (max_tokens=1024, roughly 4-5k chars) come back as history on the next
# turn. They are clipped, never rejected: the server produced them, so a 422 would lock the
# chat after one long answer.
COPILOT_MAX_HISTORY_CHARS = 8000


class CopilotMessage(BaseModel):
    role: Literal["user", "assistant", "tool"]
    content: str

    @model_validator(mode="after")
    def _cap_content(self) -> CopilotMessage:
        """User input over COPILOT_MAX_MESSAGE_CHARS fails validation (422), so a pasted log
        file never becomes a large prompt. Assistant/tool history over
        COPILOT_MAX_HISTORY_CHARS is truncated in place."""
        if self.role == "user":
            if len(self.content) > COPILOT_MAX_MESSAGE_CHARS:
                raise ValueError(
                    f"String should have at most {COPILOT_MAX_MESSAGE_CHARS} characters"
                )
        elif len(self.content) > COPILOT_MAX_HISTORY_CHARS:
            self.content = self.content[: COPILOT_MAX_HISTORY_CHARS - 15] + "\n...[truncated]"
        return self


class CopilotRequest(BaseModel):
    messages: list[CopilotMessage]
    asset_id: str | None = None


class CopilotResponse(BaseModel):
    reply: str
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    source: Literal["llm", "template"]
    # LLM mode only: {"input_tokens", "output_tokens", "llm_calls"} summed over the turn.
    usage: dict[str, int] | None = None
