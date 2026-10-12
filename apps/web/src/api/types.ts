/**
 * TypeScript mirrors of apps/api/schemas.py (Pydantic contracts).
 * Datetimes arrive as ISO-8601 strings. Keep field names identical to the Python side.
 */

// ---------------------------------------------------------------------------
// Shared primitives
// ---------------------------------------------------------------------------

export type ISODate = string;

/**
 * A telemetry sample as the plant can observe it: id, time and the eight sensors. This is the
 * only telemetry shape the API returns; simulator ground truth (health, failure_within_h) and
 * the simulator's regime label never reach the UI.
 */
export interface ObservedTelemetryRow {
  asset_id: string;
  ts: ISODate;
  vibration_rms: number;
  vibration_kurtosis: number;
  vibration_crest: number;
  bearing_temp_c: number;
  motor_temp_c: number;
  current_a: number;
  rpm: number;
  load_pct: number;
}

/** Rows accepted by POST /ingest (same observable shape). */
export type TelemetryRow = ObservedTelemetryRow;

export type Criticality = "low" | "medium" | "high";

export interface Asset {
  asset_id: string;
  name: string;
  site: string;
  line: string;
  asset_type: string;
  rated_kw: number;
  criticality: Criticality;
  /** Machine this motor drives (plant layout): infeed conveyor | filler | capper/labeler | case packer | air compressor. */
  role?: string;
}

// ---------------------------------------------------------------------------
// 04 — Data Reliability
// ---------------------------------------------------------------------------

export interface MissingnessReport {
  missing_fraction: Record<string, number>;
  gap_windows: [ISODate, ISODate][];
  flatlined_sensors: string[];
}

export interface Regime {
  regime_id: string;
  label: string;
  n_rows: number;
  rpm_mean: number;
  load_mean: number;
  share: number;
}

export interface RegimeReport {
  regimes: Regime[];
  /** Per-row labels are stripped by the API (see public_state); only the count is sent. */
  row_regime?: string[];
  row_regime_count?: number;
  method: string;
  silhouette?: number | null;
}

export type IssueSeverity = "info" | "warning" | "error";

export interface SchemaIssue {
  column: string;
  issue: string;
  severity: IssueSeverity;
}

export interface DataQualityContract {
  asset_id: string;
  window_start: ISODate;
  window_end: ISODate;
  n_rows: number;
  n_assets: number;
  sampling_rate_hz: number;
  columns: string[];
  quality_score: number;
  missingness: MissingnessReport;
  regimes: RegimeReport;
  schema_issues: SchemaIssue[];
  trainable: boolean;
  notes: string[];
}

// ---------------------------------------------------------------------------
// 05 — ML Architect
// ---------------------------------------------------------------------------

export type TaskKind = "classification" | "rul_regression";

export interface TaskSpec {
  task: TaskKind;
  target: string;
  horizon_h: number;
  rationale: string;
}

export interface FeatureSpec {
  features: string[];
  window_rows: number;
  regime_normalized: boolean;
  description: string;
  /** Assets the baseline was fitted on (the training group). */
  fit_assets?: string[] | null;
}

/** The seeded asset-level split a run used (apps/api/schemas.py AssetSplit). */
export interface AssetSplit {
  seed: number;
  train: string[];
  validation: string[];
  calibration: string[];
  test: string[];
  forced_test: string[];
  positives: Record<string, string[]>;
  notes: string[];
}

export interface GateCheck {
  name: string;
  passed: boolean;
  value?: number | null;
  threshold?: number | null;
  detail: string;
}

export interface ParityReport {
  passed: boolean;
  n_rows: number;
  max_abs_diff_probability: number;
  tolerance_probability: number;
  n_windows: number;
  max_abs_diff_features?: number | null;
  tolerance_features: number;
  calibration_method: string;
  notes: string[];
}

export interface IndustrialModelScore {
  recall: number;
  precision: number;
  lead_time_score: number;
  calibration_score: number;
  latency_score: number;
  weights: Record<string, number>;
  total: number;
}

export type ModelFamily = "random_forest" | "xgboost" | "lightgbm";

export interface CandidateModel {
  candidate_id: string;
  family: ModelFamily;
  mlflow_run_id?: string | null;
  params: Record<string, unknown>;
  metrics: Record<string, number>;
  ims?: IndustrialModelScore | null;
  artifact_path?: string | null;
  inference_latency_ms: number;
}

export interface CandidateSet {
  asset_id: string;
  task: TaskSpec;
  features: FeatureSpec;
  candidates: CandidateModel[];
  mlflow_experiment: string;
  ranked: string[];
  split?: AssetSplit | null;
}

// ---------------------------------------------------------------------------
// 06 — Validation
// ---------------------------------------------------------------------------

export interface LeakageReport {
  temporal_leakage: boolean;
  feature_leakage: string[];
  asset_overlap: boolean;
  passed: boolean;
  notes: string[];
}

export interface BacktestFold {
  fold: number;
  train_assets: string[];
  test_assets: string[];
  train_end: ISODate;
  metrics: Record<string, number>;
}

export type CalibrationMethod = "isotonic" | "sigmoid" | "none";

export interface CalibrationReport {
  method: CalibrationMethod;
  brier_before: number;
  brier_after: number;
  /** (mean_predicted, fraction_positive, count) */
  reliability_bins: [number, number, number][];
}

export interface LeadTimeReport {
  median_h: number;
  p90_h: number;
  p10_h: number;
  n_events: number;
  /** The early-warning threshold the alarm used. */
  threshold: number;
  sustained_rows?: number;
}

/** One meaning per threshold (apps/api/schemas.py Thresholds). */
export interface Thresholds {
  /** P >= this is a predicted failure (recall / precision / F1). */
  classification: number;
  /** The alarm: lead time and the maintenance-window limit. */
  early_warning: number;
  sustained_rows: number;
}

export interface RULInterval {
  p10_h: number;
  p50_h: number;
  p90_h: number;
  method: string;
}

export interface ValidationReport {
  asset_id: string;
  champion_id: string;
  champion_family: string;
  champion_mlflow_run_id?: string | null;
  model_version: string;
  leakage: LeakageReport;
  backtest: BacktestFold[];
  calibration: CalibrationReport;
  lead_time: LeadTimeReport;
  rul?: RULInterval | null;
  ims: IndustrialModelScore;
  passed: boolean;
  notes: string[];
  thresholds?: Thresholds;
  split?: AssetSplit | null;
  gate?: GateCheck[];
  test_metrics?: Record<string, number>;
  parity?: ParityReport | null;
}

// ---------------------------------------------------------------------------
// 07 — Reliability
// ---------------------------------------------------------------------------

export type ShapDirection = "raises_risk" | "lowers_risk";

export interface ShapFeature {
  feature: string;
  shap_value: number;
  feature_value: number;
  direction: ShapDirection;
}

export interface Explanation {
  asset_id: string;
  failure_probability: number;
  top_features: ShapFeature[];
  base_value: number;
  model_version: string;
}

export interface ManualPassage {
  source: string;
  section: string;
  page?: number | null;
  text: string;
  score: number;
}

export type MaintenanceOption = "maintain_now" | "maintain_later" | "run_to_failure";

export interface CostOption {
  option: MaintenanceOption;
  when: ISODate | null;
  expected_cost: number;
  p_failure_before: number;
  downtime_h: number;
  breakdown: Record<string, number>;
}

export interface CostComparison {
  asset_id: string;
  options: CostOption[];
  recommended: MaintenanceOption;
  assumptions: Record<string, number>;
}

export interface MaintenanceWindow {
  start: ISODate;
  end: ISODate;
  reason: string;
  p_failure_before_window: number;
}

export interface WhatIfResult {
  asset_id: string;
  scenario: Record<string, number>;
  baseline_probability: number;
  scenario_probability: number;
  delta: number;
}

export interface EvidenceBundle {
  explanation: Explanation;
  manual_context: ManualPassage[];
  cost: CostComparison;
  window: MaintenanceWindow;
  validation: ValidationReport;
  data_quality_score: number;
}

export type DecisionContractStatus = "pending" | "approved" | "rejected" | "expired";

export type ExplanationSource = "llm" | "template";

export interface DecisionContract {
  contract_id: string;
  created_at: ISODate;
  asset_id: string;
  run_id: string;
  model_version: string;
  champion_mlflow_run_id?: string | null;
  failure_probability: number;
  recommendation: MaintenanceOption;
  window_start: ISODate | null;
  window_end: ISODate | null;
  expected_cost: number;
  cost_comparison: CostComparison;
  top_features: ShapFeature[];
  manual_context: ManualPassage[];
  lead_time_h: number;
  calibration_brier: number;
  data_quality_score: number;
  explanation_text: string;
  explanation_source: ExplanationSource;
  evidence_hash: string;
}

export type Decision = "approved" | "rejected";

export interface Approval {
  approval_id: string;
  contract_id?: string | null;
  promotion_id?: string | null;
  decision: Decision;
  approver: string;
  note: string;
  decided_at: ISODate;
}

// ---------------------------------------------------------------------------
// 08 — MLOps
// ---------------------------------------------------------------------------

export type ModelStage = "candidate" | "validated" | "shadow" | "production" | "archived";

export interface RegisteredModel {
  name: string;
  version: string;
  mlflow_run_id: string | null;
  stage: ModelStage;
  family: string;
  ims_total: number;
  git_sha?: string | null;
  dataset_version?: string | null;
  registered_at: ISODate;
  onnx_path?: string | null;
}

export interface ChampionComparison {
  challenger: RegisteredModel;
  champion: RegisteredModel | null;
  delta: Record<string, number>;
  recommend_promote: boolean;
  rationale: string;
}

export type PromotionStatus = "pending" | "approved" | "rejected";

export interface PromotionRequest {
  promotion_id: string;
  model_name: string;
  version: string;
  from_stage: ModelStage;
  to_stage: ModelStage;
  comparison: ChampionComparison;
  status: PromotionStatus;
  created_at: ISODate;
}

export interface DriftReport {
  asset_id: string;
  model_version: string;
  psi: Record<string, number>;
  drifted_features: string[];
  threshold: number;
  drift_detected: boolean;
  window_start: ISODate;
  window_end: ISODate;
}

export interface EdgeDeployment {
  model_name: string;
  version: string;
  onnx_path: string;
  deployed_at: ISODate;
  input_features: string[];
}

// ---------------------------------------------------------------------------
// 03 — Orchestrator state
// ---------------------------------------------------------------------------

export type PipelineStage =
  | "queued"
  | "profiling"
  | "training"
  | "validating"
  | "trained"
  | "explaining"
  | "awaiting_approval"
  | "approved"
  | "rejected"
  | "failed";

export const TERMINAL_STAGES: readonly PipelineStage[] = ["trained", "approved", "rejected", "failed"];

/** analyze = reuse the production model (seconds); train = explicit AutoML run ending at "trained". */
export type RunMode = "analyze" | "train";

export interface ModelRef {
  name: string;
  version: string;
  stage: ModelStage;
  artifact_path?: string | null;
  registered_at?: ISODate | null;
  ims_total?: number | null;
}

export interface StageEvent {
  ts: ISODate;
  stage: PipelineStage;
  message: string;
  tool?: string | null;
  payload?: Record<string, unknown> | null;
}

export interface GraphState {
  run_id: string;
  asset_id: string;
  stage: PipelineStage;
  mode?: RunMode;
  /** analyze runs: the production model that was reused. */
  model_ref?: ModelRef | null;
  split?: AssetSplit | null;
  /** Train runs: every completed failure this run learns from and its label source. */
  label_sources?: LabelSource[];
  events: StageEvent[];
  data_quality?: DataQualityContract | null;
  candidates?: CandidateSet | null;
  validation?: ValidationReport | null;
  evidence?: EvidenceBundle | null;
  contract?: DecisionContract | null;
  approval?: Approval | null;
  error?: string | null;
  llm_calls: number;
}

// ---------------------------------------------------------------------------
// API request/response shapes (02)
// ---------------------------------------------------------------------------

export type WorkOrderStatus = "scheduled" | "in_progress" | "completed";

/**
 * An approved maintain_* Decision Contract handed to the plant, with the plant-reported
 * progress. Derived by the API from approvals + insert-only work-order events; Thor never
 * executes the work itself.
 */
export interface WorkOrder {
  contract_id: string;
  asset_id: string;
  recommendation: "maintain_now" | "maintain_later";
  window_start: ISODate | null;
  window_end: ISODate | null;
  planned_downtime_h: number;
  approved_at: ISODate;
  approver: string;
  status: WorkOrderStatus;
  started_ts: ISODate | null;
  completed_ts: ISODate | null;
  /** The effective (non-superseded) repair finding, once a person has recorded one. */
  finding?: RepairFinding | null;
}

// ---------------------------------------------------------------------------
// Repair findings: "What did the technician find?" (routes/findings.py). Insert-only; a
// correction is a new row whose `supersedes` names the one it replaces.
// ---------------------------------------------------------------------------

export type FindingKind = "confirmed_bearing_damage" | "no_fault_found" | "other";

export const FINDING_LABEL: Record<FindingKind, string> = {
  confirmed_bearing_damage: "Confirmed: bearing damage",
  no_fault_found: "No fault found",
  other: "Other",
};

export interface RepairFinding {
  finding_id: string;
  contract_id: string;
  asset_id: string;
  finding: FindingKind;
  note: string;
  recorded_by: string;
  recorded_at: ISODate;
  /** Plant time the work order completed. */
  plant_ts: ISODate;
  sandbox: boolean;
  supersedes: string | null;
}

export interface RepairFindingRequest {
  contract_id: string;
  finding: FindingKind;
  note?: string;
  recorded_by: string;
  supersedes?: string | null;
  sandbox?: boolean;
}

/** Deterministic counts over approved maintenance contracts and their findings. */
export interface AlarmTrust {
  acted_on: number;
  confirmed: number;
  no_fault_found: number;
  other: number;
  awaiting_finding: number;
  completed_without_finding: number;
  /** Below this many acted-on alarms the UI shows counts only, never a percentage. */
  min_for_rate: number;
}

/** Where one completed failure a training run learns from comes from. */
export interface LabelSource {
  asset_id: string;
  source: "simulator_history" | "work_order_finding";
  group: string;
  contract_id?: string | null;
  finding_id?: string | null;
  recorded_by?: string | null;
  event_ts?: ISODate | null;
  labelled_from?: ISODate | null;
  n_rows?: number | null;
}

export interface LabelLedgerEntry {
  finding_id: string;
  asset_id: string;
  contract_id: string;
  recorded_by: string;
  recorded_at: ISODate;
  event_ts: ISODate | null;
  note: string;
}

export interface LabelLedger {
  queued: LabelLedgerEntry[];
  rule: string;
}

/** GET /llm/key: whether the provider key is held in memory. Never carries the key. */
export interface LlmKeyState {
  key: "up" | "down";
  provider: "anthropic" | "openai" | "none";
}

/**
 * Alarm level of one motor (apps/api/alarms.py): healthy (within healthy baseline) / warning /
 * critical. Critical is LATCHED: it stays until a completed work order
 * clears it, whatever the probability does afterwards.
 */
export type AlarmLevel = "healthy" | "warning" | "critical";

/** GET /alarms?asset_id= — one insert-only alarm transition. `plant_ts` is plant time. */
export interface AlarmEvent {
  event_id: string;
  asset_id: string;
  level: AlarmLevel;
  reason: string;
  plant_ts: ISODate;
  recorded_at: ISODate;
  probability: number | null;
  model_version: string | null;
}

export interface FleetAsset {
  asset: Asset;
  /** 100 * (1 - failure_probability) from the latest approved-model prediction; null = no prediction yet. */
  health_score: number | null;
  failure_probability: number | null;
  /** Plant time of the scored window and the model that scored it (null without a prediction). */
  prediction_ts?: ISODate | null;
  model_version?: string | null;
  /** Operating state inferred from rpm/load against the production model's centroids; null without a production model. */
  regime: string | null;
  last_ts: ISODate | null;
  open_contract_id: string | null;
  stage: ModelStage | null;
  /** Newest approved work order for this asset (plant-reported status), if any. */
  work_order: WorkOrder | null;
  /** The ONE status rule (offline > no_prediction > no_data > critical > warning > healthy); null = not computed. */
  status: TwinStatus | null;
  /** Latched alarm level behind `status`, when it was raised (plant time) and the rule that raised it. */
  alarm: AlarmLevel | null;
  alarm_since: ISODate | null;
  alarm_reason: string | null;
}

export interface PipelineRunRequest {
  asset_id: string;
  horizon_h?: number;
  n_trials?: number;
  mode?: RunMode;
}

export interface PipelineRunResponse {
  run_id: string;
  stage: PipelineStage;
  mode?: RunMode;
}

export interface DecisionRequest {
  decision: Decision;
  approver: string;
  note?: string;
}

export type CopilotRole = "user" | "assistant" | "tool";

export interface CopilotMessage {
  role: CopilotRole;
  content: string;
}

export interface CopilotRequest {
  messages: CopilotMessage[];
  asset_id?: string | null;
}

/** tool_calls is `list[dict[str, Any]]` server-side; the common keys are typed loosely. */
export interface CopilotToolCall {
  name?: string;
  tool?: string;
  args?: Record<string, unknown>;
  input?: Record<string, unknown>;
  result?: unknown;
  [key: string]: unknown;
}

export interface CopilotResponse {
  reply: string;
  tool_calls: CopilotToolCall[];
  source: ExplanationSource;
}

// ---------------------------------------------------------------------------
// Route-level shapes that are not Pydantic models in schemas.py (from INTERFACES.md)
// ---------------------------------------------------------------------------

/** GET /system/health */
export interface SystemHealth {
  db: string | boolean;
  mqtt: boolean;
  mlflow: boolean;
  edge: boolean;
  /** "llm" when a usable provider key is configured on the API, else "template". */
  llm?: {
    mode: "llm" | "template";
    provider?: "anthropic" | "openai" | "none";
    model: string;
    /** The API Key switch (GET /llm/key). */
    key?: "up" | "down";
    warning?: string | null;
  };
  n_telemetry_rows: number;
  last_ingest_ts: ISODate | null;
  thresholds?: Thresholds;
}

/** GET /assets/:id/predictions */
export interface PredictionRow {
  ts: ISODate;
  failure_probability: number;
  model_version: string;
  source: string;
}

// ---------------------------------------------------------------------------
// Replay control (GET /replay, POST /replay/pause|resume)
// ---------------------------------------------------------------------------

/**
 * `released` is a consequence of the human sign-off: true exactly when a production model
 * exists (the replay container holds until then). `paused` is the presenter's recovery switch.
 */
export interface ReplayState {
  label: "Replay (recorded plant history)";
  released: boolean;
  paused: boolean;
  production_model_version: string | null;
  plant_time: ISODate | null;
  /** Rows seeded at start-up (the recorded history the replay continues from). */
  seeded_until: ISODate | null;
  paused_at: ISODate | null;
}

/** Run summary rows returned by GET /pipeline?asset_id= and inside GET /assets/:id. */
export interface RunSummary {
  run_id: string;
  asset_id: string;
  stage: PipelineStage;
  mode?: RunMode;
  model_version_used?: string | null;
  created_at?: ISODate | null;
  updated_at?: ISODate | null;
  contract_id?: string | null;
  error?: string | null;
  model_version?: string | null;
  llm_calls?: number;
  n_events?: number;
  last_event_ts?: ISODate | null;
}

/** GET /assets/:id */
export interface AssetDetail {
  asset: Asset;
  latest: ObservedTelemetryRow | null;
  /** Inferred from the latest rpm/load with the production baseline; null without a production model. */
  regime: string | null;
  prediction: number | null;
  prediction_ts: ISODate | null;
  model_version: string | null;
  /** null = no prediction yet (never derived from anything but a prediction). */
  health_score: number | null;
  thresholds?: Thresholds;
  contracts: DecisionContract[];
  work_orders: WorkOrder[];
  runs: RunSummary[];
  /** Same status rule and latched alarm as /fleet. */
  status: TwinStatus | null;
  alarm: AlarmLevel | null;
  alarm_since: ISODate | null;
  alarm_reason: string | null;
}

/** GET /approvals/pending */
export interface PendingApprovals {
  contracts: DecisionContract[];
  promotions: PromotionRequest[];
}

/** GET /approvals/:contract_id  (present_evidence) */
export interface ContractEvidence {
  contract: DecisionContract;
  evidence: Partial<EvidenceBundle> & Record<string, unknown>;
  status: DecisionContractStatus | string;
}

export interface WhatIfRequest {
  run_id: string;
  scenario: Record<string, number>;
}

export interface DriftCheckRequest {
  asset_id: string;
}

// ---------------------------------------------------------------------------
// 3-D Digital Twin + X-Ray (apps/api/schemas.py "Twin" section, routes/twin.py).
// Every value here is produced by deterministic Python; the browser only renders it.
// ---------------------------------------------------------------------------

/**
 * Status glyph (apps/api/alarms.py::asset_status). Precedence: offline > no_prediction >
 * no_data > critical > warning > healthy. Green means "within healthy baseline"; critical is
 * latched until a completed work order clears it.
 */
export type TwinStatus = "healthy" | "warning" | "critical" | "no_prediction" | "offline" | "no_data";

/** Wall-clock freshness of a motor's telemetry at the control plane. */
export type DataState = "live" | "stale" | "none";

export type DecisionState =
  | "none"
  | "contract_pending"
  | "approved"
  | "work_order_in_progress"
  | "work_order_completed"
  | "rejected";

export interface TwinAsset {
  asset: Asset;
  status: TwinStatus;
  data_state: DataState;
  decision_state: DecisionState;
  /** Median of the last three predictions (same as /fleet); null = no prediction yet. */
  failure_probability: number | null;
  health_score: number | null;
  prediction_ts: ISODate | null;
  model_version: string | null;
  n_prediction_rows: number | null;
  /** Inferred from rpm/load with the production baseline; null without a production model. */
  regime: string | null;
  /** Latest observed sample; machine motion follows `latest.rpm`. */
  latest: ObservedTelemetryRow | null;
  last_ts: ISODate | null;
  /** Wall clock when the control plane last received a message for this motor. */
  last_received_at: ISODate | null;
  /** Messages received for this motor in the last 60 s (measured). */
  msgs_last_60s: number;
  open_contract_id: string | null;
  work_order: WorkOrder | null;
  /** Latched alarm behind `status`: level (null = healthy), raised at (plant time), rule. */
  alarm: AlarmLevel | null;
  alarm_since: ISODate | null;
  alarm_reason: string | null;
}

/** Simulated MES feed: context only, feeds no model, score or decision. */
export interface MesContext {
  line: string;
  sku: string;
  container: string;
  quantity_produced: number;
  uom: string;
  shift: string;
  plant_ts: ISODate;
  source: "simulated MES feed";
  basis: string;
}

export interface TwinLine {
  line: string;
  kind: "packaging" | "compressor";
  index: number;
  /** asset ids in machine order (infeed conveyor → filler → capper/labeler → case packer). */
  assets: string[];
  mes: MesContext | null;
}

export interface SiteTab {
  name: string;
  available: boolean;
  note: string;
}

export interface ModelProvenance {
  name: string;
  version: string;
  family: string;
  stage: ModelStage;
  ims_total: number;
  registered_at: ISODate;
  approver: string | null;
  approved_at: ISODate | null;
  promotion_id: string | null;
  validation_passed: boolean | null;
  parity_passed: boolean | null;
  parity_max_abs_diff: number | null;
  lead_time_median_h: number | null;
  calibration_method: string | null;
  onnx_path: string | null;
}

/** Measured at the control plane (apps/api/live.py). */
export interface LiveCounters {
  server_time: ISODate;
  window_s: number;
  msgs_per_s: number;
  preds_per_s: number;
  messages_total: number;
  predictions_total: number;
  edge_inference_ms_p50: number | null;
  edge_inference_ms_p95: number | null;
  path_latency_ms_p50: number | null;
  path_latency_ms_p95: number | null;
  n_latency_samples: number;
  stale_after_s: number;
  mqtt_connected: boolean;
  edge_up: boolean;
}

/** GET /twin */
export interface TwinSnapshot {
  site: string;
  sites: SiteTab[];
  plant_time: ISODate | null;
  thresholds: Thresholds;
  lines: TwinLine[];
  assets: TwinAsset[];
  model: ModelProvenance | null;
  live: LiveCounters;
}

export interface PlcTag {
  tag: string;
  value: number | boolean | string | null;
  unit: string;
  /** Telemetry field this tag is mapped from (one-to-one); null when simulated. */
  source: string | null;
  simulated: boolean;
  basis: string;
}

export interface PredictionPoint {
  ts: ISODate;
  failure_probability: number;
  model_version: string;
  source: string;
}

export interface DriverSource {
  contract_id: string;
  run_id: string;
  created_at: ISODate;
  model_version: string;
}

/** GET /twin/hud/:asset_id */
export interface TwinHud {
  asset: TwinAsset;
  line: TwinLine | null;
  thresholds: Thresholds;
  /** Newest-last per-sample predictions. */
  prediction_trend: PredictionPoint[];
  trend: "rising" | "falling" | "flat" | null;
  plc_tags: PlcTag[];
  drivers: ShapFeature[];
  drivers_source: DriverSource | null;
  contract: DecisionContract | null;
  contract_status: string | null;
  model: ModelProvenance | null;
  mes: MesContext | null;
}

export type RewindMarkerKind =
  | "early_warning"
  | "critical_alarm"
  | "contract_created"
  | "approved"
  | "rejected"
  | "work_order_started"
  | "work_order_completed";

export interface RewindMarker {
  kind: RewindMarkerKind;
  ts: ISODate;
  label: string;
  detail: string;
}

export interface RewindSeries {
  asset_id: string;
  /** [ts, failure_probability] oldest first, downsampled. */
  points: [ISODate, number][];
  n_rows: number;
  markers: RewindMarker[];
  /** Plant hours from the first sustained early warning to the plant taking the motor offline. */
  lead_time_h: number | null;
  lead_time_basis: string;
  /** "this run: first sustained warning -> motor taken offline" */
  lead_time_label: string;
}

/** GET /twin/rewind — stored predictions only. */
export interface TwinRewind {
  start: ISODate | null;
  end: ISODate | null;
  thresholds: Thresholds;
  assets: RewindSeries[];
  validated_lead_time_h: number | null;
  validated_lead_time_basis: string;
  /** "validated on held-out motors" */
  validated_lead_time_label: string;
}

export interface TwinEvent {
  seq: number;
  kind: "ingest" | "prediction";
  asset_id: string;
  ts: ISODate;
  received_at: ISODate;
  count: number;
  /** Predictions only: scored at the edge (edge path, inference_ms) or centrally. */
  source: "edge" | "control_plane" | null;
  failure_probability: number | null;
  inference_ms: number | null;
  path_latency_ms: number | null;
}

/** GET /twin/events?since=N and every /ws/twin frame. */
export interface TwinEventBatch {
  since: number;
  next: number;
  events: TwinEvent[];
  dropped: number;
  live: LiveCounters;
}
