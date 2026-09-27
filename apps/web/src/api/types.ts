/**
 * TypeScript mirrors of apps/api/schemas.py (Pydantic contracts).
 * Datetimes arrive as ISO-8601 strings. Keep field names identical to the Python side.
 */

// ---------------------------------------------------------------------------
// Shared primitives
// ---------------------------------------------------------------------------

export type ISODate = string;

export interface TelemetryRow {
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
  regime?: string | null;
  health?: number | null;
  failure_within_h?: number | null;
}

export type Criticality = "low" | "medium" | "high";

export interface Asset {
  asset_id: string;
  name: string;
  site: string;
  line: string;
  asset_type: string;
  rated_kw: number;
  criticality: Criticality;
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
  threshold: number;
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
  | "explaining"
  | "awaiting_approval"
  | "approved"
  | "rejected"
  | "failed";

export const TERMINAL_STAGES: readonly PipelineStage[] = ["approved", "rejected", "failed"];

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

export interface FleetAsset {
  asset: Asset;
  health_score: number;
  failure_probability: number | null;
  regime: string | null;
  last_ts: ISODate | null;
  open_contract_id: string | null;
  stage: ModelStage | null;
}

export interface PipelineRunRequest {
  asset_id: string;
  horizon_h?: number;
  n_trials?: number;
}

export interface PipelineRunResponse {
  run_id: string;
  stage: PipelineStage;
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
    warning?: string | null;
  };
  n_telemetry_rows: number;
  last_ingest_ts: ISODate | null;
}

/** GET /assets/:id/predictions */
export interface PredictionRow {
  ts: ISODate;
  failure_probability: number;
  model_version: string;
  source: string;
}

/** Run summary rows returned by GET /pipeline?asset_id= and inside GET /assets/:id. */
export interface RunSummary {
  run_id: string;
  asset_id: string;
  stage: PipelineStage;
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
  latest: TelemetryRow | null;
  prediction: number | null;
  contracts: DecisionContract[];
  runs: RunSummary[];
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
