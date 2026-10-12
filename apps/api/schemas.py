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

# --------------------------------------------------------------------------------------
# Thresholds: one meaning each, defined once, used everywhere (agents/metrics.py, cost.py,
# the validation report and the Asset 360 chart). Never mix them.
# --------------------------------------------------------------------------------------

#: P(failure within the horizon) >= this counts as a *predicted failure*. Used for recall,
#: precision and F1 in the Optuna objective, the validation gate and the AI4I benchmark.
CLASSIFICATION_THRESHOLD: float = 0.5
#: The *early-warning alarm*: lead time is measured from the first of
#: EARLY_WARNING_SUSTAINED_ROWS consecutive rows at or above this, and a maintenance window
#: must end before P(failure) exceeds it. A calibrated 30 % chance within the horizon is
#: actionable because the unplanned cost far exceeds the planned cost.
EARLY_WARNING_THRESHOLD: float = 0.3
#: Consecutive rows (10-min samples) required for an alarm, so a single noisy row never counts.
EARLY_WARNING_SUSTAINED_ROWS: int = 3
#: The same rule for what is DISPLAYED and ANALYSED: an asset's current failure probability is
#: the median of its last EARLY_WARNING_SUSTAINED_ROWS predictions (fleet, Asset 360, Bolt) and
#: "Run analysis" explains the median window of its last EARLY_WARNING_SUSTAINED_ROWS windows,
#: so one noisy sample never drives a colour or a Decision Contract.
DISPLAY_MEDIAN_ROWS: int = EARLY_WARNING_SUSTAINED_ROWS


class Thresholds(BaseModel):
    """The named thresholds a run used, reported in every ValidationReport and by the API."""

    classification: float = Field(
        default=CLASSIFICATION_THRESHOLD,
        description="P >= this is a predicted failure (recall / precision / F1)",
    )
    early_warning: float = Field(
        default=EARLY_WARNING_THRESHOLD,
        description="alarm: lead time and the maintenance-window limit",
    )
    sustained_rows: int = Field(
        default=EARLY_WARNING_SUSTAINED_ROWS,
        description="consecutive rows >= early_warning that make an alarm",
    )


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


#: Simulator ground truth. Allowed ONLY as training labels and in offline evaluation. It is
#: stripped at the replay's publish boundary and never appears in an operational response
#: (fleet, asset, telemetry, copilot tools, Decision Contracts, edge) -- see
#: `ObservedTelemetryRow` and tests/test_ground_truth_isolation.py.
GROUND_TRUTH_COLUMNS: tuple[str, ...] = ("health", "failure_within_h")
#: Columns the simulator writes that the plant could not observe directly. `regime` is a
#: simulator label; operational responses carry an *inferred* regime (rpm/load vs the
#: production model's regime centroids) instead.
SIMULATOR_ONLY_COLUMNS: tuple[str, ...] = ("regime", *GROUND_TRUTH_COLUMNS)


class TelemetryRow(BaseModel):
    """One sensor sample for one asset — the unit that flows into TimescaleDB.

    The optional `regime` / `health` / `failure_within_h` fields exist for the historical
    import (seed) that carries training labels. Live rows and every operational response use
    `ObservedTelemetryRow`, which has no such fields.
    """

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


class ObservedTelemetryRow(BaseModel):
    """A telemetry sample as the plant can observe it: id, time and the eight sensors.

    This is the only telemetry shape that leaves the control plane (GET /assets/{id},
    GET /assets/{id}/telemetry, Bolt's get_asset). It cannot carry simulator ground truth.
    """

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

    @classmethod
    def from_row(cls, row: TelemetryRow | dict[str, Any]) -> ObservedTelemetryRow:
        """Project a stored TelemetryRow (or dict) onto the observable fields only."""
        data = row if isinstance(row, dict) else row.model_dump()
        return cls(**{k: data[k] for k in cls.model_fields})


class Asset(BaseModel):
    asset_id: str
    name: str
    site: str
    line: str
    asset_type: str = "induction_motor"
    rated_kw: float = 75.0
    criticality: Literal["low", "medium", "high"] = "medium"
    #: The machine this motor drives on its line (plant layout, data/simulator/layout.py):
    #: "infeed conveyor" | "filler" | "capper/labeler" | "case packer" | "air compressor".
    role: str = ""


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
    #: Assets whose rows the clustering was fitted on (None = every row; a leak-free run
    #: passes the training assets only and the other rows are merely assigned).
    fit_assets: list[str] | None = None


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
    #: Assets the baseline / scales were fitted on (the training group); None = all rows.
    fit_assets: list[str] | None = None


class AssetSplit(BaseModel):
    """The seeded asset-level split a run used, fixed before any preprocessing was fitted.

    `train`: model, regimes and baselines are fitted here only. `validation`: the Optuna
    objective. `calibration`: probability calibration. `test`: final metrics, untouched until
    then. `validation` and `calibration` are the same assets when the fleet has too few
    completed failures for four groups (see `notes`).
    """

    seed: int
    train: list[str]
    validation: list[str]
    calibration: list[str]
    test: list[str]
    #: Assets forced into `test` (the asset under analysis), so its score is out-of-sample.
    forced_test: list[str] = Field(default_factory=list)
    #: group -> assets in it that carry a completed failure (positive labels).
    positives: dict[str, list[str]] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)

    @property
    def not_test(self) -> list[str]:
        return sorted(set(self.train) | set(self.validation) | set(self.calibration))


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
    #: The split the candidates were trained / selected with (validation uses the same one).
    split: AssetSplit | None = None


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
    #: The calibration as an explicit, serialisable transform of the base model's P(class 1):
    #: {"method": "sigmoid", "a", "b"} -> 1 / (1 + exp(a * p + b)); {"method": "isotonic",
    #: "x": [...], "y": [...]} -> linear interpolation clipped to [x[0], x[-1]];
    #: {"method": "none"}. Shipped in the edge sidecar and applied there.
    transform: dict[str, Any] | None = None


class ParityReport(BaseModel):
    """Edge-vs-centre agreement, checked before a model can be promoted (08).

    The central path is the calibrated joblib champion; the edge path is the exported ONNX
    model + sidecar (baseline, calibration transform) run through `edge.inference` and
    `edge.features`, exactly as the edge container would.
    """

    passed: bool
    #: Calibrated probability agreement on real held-out feature vectors.
    n_rows: int
    max_abs_diff_probability: float
    tolerance_probability: float
    #: Feature agreement on raw windows (edge.features vs agents.ml_architect.features).
    n_windows: int = 0
    max_abs_diff_features: float | None = None
    tolerance_features: float = 1e-6
    calibration_method: str = "none"
    notes: list[str] = Field(default_factory=list)


class GateCheck(BaseModel):
    """One line of the validation gate checklist (shown in AutoML Studio)."""

    name: str
    passed: bool
    value: float | None = None
    threshold: float | None = None
    detail: str = ""


class LeadTimeReport(BaseModel):
    median_h: float
    p90_h: float
    p10_h: float
    n_events: int
    #: The early-warning threshold the alarm used (== Thresholds.early_warning).
    threshold: float
    sustained_rows: int = EARLY_WARNING_SUSTAINED_ROWS


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
    #: The thresholds every metric above was computed with (one meaning each).
    thresholds: Thresholds = Field(default_factory=Thresholds)
    #: Which motors were train / validation / calibration / test (None for legacy reports).
    split: AssetSplit | None = None
    #: Pass/fail checklist with reasons; `passed` is their conjunction.
    gate: list[GateCheck] = Field(default_factory=list)
    #: Final metrics of the calibrated champion on the untouched test assets.
    test_metrics: dict[str, float] = Field(default_factory=dict)
    #: Edge-vs-centre parity on held-out vectors; failing parity blocks promotion.
    parity: ParityReport | None = None


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


class ModelRef(BaseModel):
    """Which approved model a run (or a prediction) used: the registry row, by value."""

    name: str
    version: str
    stage: ModelStage
    artifact_path: str | None = None
    registered_at: datetime | None = None
    ims_total: float | None = None


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
    #: Terminal stage of a `train` run: champion registered (promotion is a ModelOps decision).
    trained = "trained"
    explaining = "explaining"
    awaiting_approval = "awaiting_approval"
    approved = "approved"
    rejected = "rejected"
    failed = "failed"


#: `analyze` (the default for "Run analysis"): score, explain, cost and draft a Decision
#: Contract for one asset with the current PRODUCTION model -- seconds, nothing is trained.
#: `train`: profile -> split -> AutoML -> validate -> register; ends at `trained`. Promotion is
#: then a separate ModelOps decision. Training never runs as a side effect of an analysis.
RunMode = Literal["analyze", "train"]


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
    mode: RunMode = "train"
    #: analyze runs: the production model that was reused (never a freshly trained one).
    model_ref: ModelRef | None = None
    events: list[StageEvent] = Field(default_factory=list)
    data_quality: DataQualityContract | None = None
    #: Fixed before profiling; every later stage fits only on `split.train`.
    split: AssetSplit | None = None
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
    """One fleet-table row. Everything here is observable telemetry or an approved-model
    prediction; simulator ground truth never reaches it."""

    asset: Asset
    #: 100 * (1 - failure_probability) from the latest prediction; None = "no prediction yet".
    health_score: float | None
    failure_probability: float | None
    #: When the prediction was made (plant time of the scored window) and by which model.
    prediction_ts: datetime | None = None
    model_version: str | None = None
    #: Operating regime inferred from the latest rpm/load against the production model's
    #: regime centroids; None when no production model exists.
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
    mode: RunMode = Field(
        default="analyze",
        description="analyze = reuse the production model (seconds); train = AutoML run",
    )


class PipelineRunResponse(BaseModel):
    run_id: str
    stage: PipelineStage
    mode: RunMode = "train"


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


# --------------------------------------------------------------------------------------
# 3-D Digital Twin + X-Ray (Phase A, CLAUDE.md section 15). Read-only views over the same
# stores as /fleet: nothing here is computed by an LLM or in the browser. Status keys off the
# displayed (median) failure probability against `Thresholds`; the simulator's ground truth
# never appears (tests/test_ground_truth_isolation.py covers these routes too).
# --------------------------------------------------------------------------------------


class TwinStatus(StrEnum):
    """Status glyph of one motor. Precedence: offline > no_data > critical > warning >
    healthy > no_prediction. Encoded as green circle / amber triangle / red stop-octagon."""

    healthy = "healthy"  # p < early_warning
    warning = "warning"  # early_warning <= p < classification
    critical = "critical"  # p >= classification
    no_prediction = "no_prediction"  # no approved-model prediction yet
    offline = "offline"  # the plant has it offline for an approved work order
    no_data = "no_data"  # its messages stopped (see LiveCounters.stale_after_s)


class DataState(StrEnum):
    """Wall-clock freshness of a motor's telemetry at the control plane."""

    live = "live"  # a message arrived within stale_after_s
    stale = "stale"  # messages arrived earlier, none within stale_after_s
    none = "none"  # nothing received since the control plane started


class DecisionState(StrEnum):
    """Governance badge: where this motor stands at the Human Approval Gate."""

    none = "none"
    contract_pending = "contract_pending"
    approved = "approved"  # approved, the plant has not started the work order yet
    work_order_in_progress = "work_order_in_progress"
    work_order_completed = "work_order_completed"
    rejected = "rejected"


class TwinAsset(BaseModel):
    """One motor in the twin: FleetAsset facts plus live sensor values and the glyph state."""

    asset: Asset
    status: TwinStatus
    data_state: DataState
    decision_state: DecisionState
    #: Median of the last DISPLAY_MEDIAN_ROWS predictions (same as /fleet); None = none yet.
    failure_probability: float | None = None
    health_score: float | None = None
    prediction_ts: datetime | None = None
    model_version: str | None = None
    n_prediction_rows: int | None = None
    #: Inferred from the latest rpm/load with the production baseline; None without a model.
    regime: str | None = None
    #: Latest observed sample (eight sensors). Machine motion follows `latest.rpm`.
    latest: ObservedTelemetryRow | None = None
    last_ts: datetime | None = None
    #: Wall clock at which the control plane last received a message for this motor.
    last_received_at: datetime | None = None
    #: Messages received for this motor in the last 60 s of wall clock (measured).
    msgs_last_60s: int = 0
    open_contract_id: str | None = None
    work_order: WorkOrder | None = None


class MesContext(BaseModel):
    """Line-level MES context. A deterministic *simulated MES feed* (apps/api/plant.py):
    context for the HUD only. It feeds no model, score, recommendation or decision
    (tests/test_twin.py asserts that)."""

    line: str
    sku: str
    container: str
    quantity_produced: int
    uom: str
    shift: str
    plant_ts: datetime
    source: Literal["simulated MES feed"] = "simulated MES feed"
    basis: str


class TwinLine(BaseModel):
    line: str
    kind: Literal["packaging", "compressor"]
    index: int
    #: asset ids in machine order (infeed conveyor -> filler -> capper/labeler -> case packer).
    assets: list[str]
    mes: MesContext | None = None


class SiteTab(BaseModel):
    name: str
    available: bool
    note: str = ""


class ModelProvenance(BaseModel):
    """The production model as the registry and the promotion approval record it."""

    name: str
    version: str
    family: str
    stage: ModelStage
    ims_total: float
    registered_at: datetime
    approver: str | None = None
    approved_at: datetime | None = None
    promotion_id: str | None = None
    validation_passed: bool | None = None
    parity_passed: bool | None = None
    parity_max_abs_diff: float | None = None
    lead_time_median_h: float | None = None
    calibration_method: str | None = None
    onnx_path: str | None = None


class LiveCounters(BaseModel):
    """Measured at the control plane (apps/api/live.py): message and prediction arrival
    rates over the last `window_s` of wall clock, edge inference time as reported by the
    edge per prediction, and the motor -> edge -> control plane path latency (prediction
    receipt minus telemetry receipt for the same (asset, ts), one clock)."""

    server_time: datetime
    window_s: int
    msgs_per_s: float
    preds_per_s: float
    messages_total: int
    predictions_total: int
    edge_inference_ms_p50: float | None = None
    edge_inference_ms_p95: float | None = None
    path_latency_ms_p50: float | None = None
    path_latency_ms_p95: float | None = None
    n_latency_samples: int = 0
    stale_after_s: float
    mqtt_connected: bool
    edge_up: bool


class TwinSnapshot(BaseModel):
    """GET /twin: everything the 3-D view needs in one call."""

    site: str
    sites: list[SiteTab]
    plant_time: datetime | None
    thresholds: Thresholds
    lines: list[TwinLine]
    assets: list[TwinAsset]
    model: ModelProvenance | None
    live: LiveCounters


class PlcTag(BaseModel):
    """One PLC tag of the X-Ray integration panel. `source` names the telemetry field it is
    mapped from (one-to-one); `simulated` tags have no telemetry source and `basis` says how
    they were derived."""

    tag: str
    value: float | int | bool | str | None
    unit: str = ""
    source: str | None = None
    simulated: bool = False
    basis: str = ""


class PredictionPoint(BaseModel):
    ts: datetime
    failure_probability: float
    model_version: str
    source: str


class DriverSource(BaseModel):
    contract_id: str
    run_id: str
    created_at: datetime
    model_version: str


class TwinHud(BaseModel):
    """GET /twin/hud/{asset_id}: the X-Ray HUD panel for one motor."""

    asset: TwinAsset
    line: TwinLine | None
    thresholds: Thresholds
    #: Newest-last stored predictions (per-sample rows, not the median).
    prediction_trend: list[PredictionPoint]
    trend: Literal["rising", "falling", "flat"] | None
    plc_tags: list[PlcTag]
    #: Top SHAP drivers from the newest Decision Contract (empty when no analysis exists).
    drivers: list[ShapFeature]
    drivers_source: DriverSource | None = None
    contract: DecisionContract | None = None
    contract_status: str | None = None
    model: ModelProvenance | None
    mes: MesContext | None = None


class RewindMarker(BaseModel):
    kind: Literal[
        "early_warning",
        "predicted_failure",
        "contract_created",
        "approved",
        "rejected",
        "work_order_started",
        "work_order_completed",
    ]
    ts: datetime
    label: str
    detail: str = ""


class RewindSeries(BaseModel):
    asset_id: str
    #: [ts, failure_probability] pairs, oldest first, downsampled to <= max_points.
    points: list[tuple[datetime, float]]
    n_rows: int
    markers: list[RewindMarker]
    #: Plant hours from the first sustained early warning to the plant acting (work order
    #: started, else the approval); None when either end is missing.
    lead_time_h: float | None = None
    lead_time_basis: str = ""


class TwinRewind(BaseModel):
    """GET /twin/rewind: stored predictions only, for the time scrubber."""

    start: datetime | None
    end: datetime | None
    thresholds: Thresholds
    assets: list[RewindSeries]
    validated_lead_time_h: float | None = None
    validated_lead_time_basis: str = ""


class TwinEvent(BaseModel):
    """One ingest or prediction arrival at the control plane (the pulse source). Events for
    the same motor inside one batch interval may be aggregated: `count` > 1."""

    seq: int
    kind: Literal["ingest", "prediction"]
    asset_id: str
    ts: datetime
    received_at: datetime
    count: int = 1
    failure_probability: float | None = None
    inference_ms: float | None = None
    path_latency_ms: float | None = None


class TwinEventBatch(BaseModel):
    """GET /twin/events?since=N and every /ws/twin frame."""

    since: int
    next: int
    events: list[TwinEvent]
    dropped: int = 0
    live: LiveCounters
