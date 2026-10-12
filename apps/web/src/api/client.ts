/**
 * Typed HTTP client for the Thor control plane (apps/api).
 * Every endpoint in docs/INTERFACES.md has one helper here; screens import from this file
 * and colocate their own polling/state logic.
 */
import type {
  AlarmEvent,
  AlarmTrust,
  Approval,
  AssetDetail,
  ContractEvidence,
  CopilotRequest,
  CopilotResponse,
  DecisionRequest,
  DriftCheckRequest,
  DriftReport,
  FleetAsset,
  GraphState,
  LabelLedger,
  LlmKeyState,
  PendingApprovals,
  PipelineRunRequest,
  PipelineRunResponse,
  PredictionRow,
  PromotionRequest,
  RegisteredModel,
  RepairFinding,
  RepairFindingRequest,
  ReplayState,
  RunSummary,
  SystemHealth,
  TelemetryRow,
  TwinEventBatch,
  TwinHud,
  TwinRewind,
  TwinSnapshot,
  WhatIfRequest,
  WhatIfResult,
} from "./types";

export const API_BASE_URL: string = (
  import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000"
).replace(/\/+$/, "");

/** Derive the WebSocket base from the HTTP base (http -> ws, https -> wss). */
export const WS_BASE_URL: string = API_BASE_URL.replace(/^http/i, "ws");

export class ApiError extends Error {
  readonly status: number;
  readonly body: string;

  constructor(status: number, body: string, path: string) {
    super(`API ${status} on ${path}: ${body.slice(0, 200)}`);
    this.name = "ApiError";
    this.status = status;
    this.body = body;
  }
}

export async function apiFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const url = path.startsWith("http") ? path : `${API_BASE_URL}${path}`;
  const headers: Record<string, string> = { Accept: "application/json" };
  if (init?.body !== undefined) headers["Content-Type"] = "application/json";
  const res = await fetch(url, {
    ...init,
    headers: { ...headers, ...((init?.headers as Record<string, string> | undefined) ?? {}) },
  });
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new ApiError(res.status, text, path);
  }
  if (res.status === 204) return undefined as T;
  return (await res.json()) as T;
}

function post<T>(path: string, body: unknown): Promise<T> {
  return apiFetch<T>(path, { method: "POST", body: JSON.stringify(body) });
}

function qs(params: Record<string, string | number | undefined | null>): string {
  const entries = Object.entries(params).filter(
    ([, v]) => v !== undefined && v !== null && v !== "",
  );
  if (entries.length === 0) return "";
  return (
    "?" +
    entries
      .map(([k, v]) => `${encodeURIComponent(k)}=${encodeURIComponent(String(v))}`)
      .join("&")
  );
}

// --- routes/fleet.py --------------------------------------------------------

export const getFleet = () => apiFetch<FleetAsset[]>("/fleet");
export const getAsset = (assetId: string) =>
  apiFetch<AssetDetail>(`/assets/${encodeURIComponent(assetId)}`);
export const getSystemHealth = () => apiFetch<SystemHealth>("/system/health");
/** Insert-only alarm transitions (oldest first), optionally for one asset. */
export const getAlarms = (assetId?: string) =>
  apiFetch<AlarmEvent[]>(`/alarms${qs({ asset_id: assetId })}`);

// --- routes/replay.py -------------------------------------------------------

export const getReplay = () => apiFetch<ReplayState>("/replay");
export const pauseReplay = () => post<ReplayState>("/replay/pause", {});
export const resumeReplay = () => post<ReplayState>("/replay/resume", {});

// --- routes/telemetry.py ----------------------------------------------------

export const getTelemetry = (assetId: string, hours = 24, limit = 2000) =>
  apiFetch<TelemetryRow[]>(
    `/assets/${encodeURIComponent(assetId)}/telemetry${qs({ hours, limit })}`,
  );
export const getPredictions = (assetId: string, limit = 500) =>
  apiFetch<PredictionRow[]>(`/assets/${encodeURIComponent(assetId)}/predictions${qs({ limit })}`);
export const postIngest = (rows: TelemetryRow[]) => post<{ inserted: number }>("/ingest", rows);

// --- routes/pipeline.py -----------------------------------------------------

export const runPipeline = (req: PipelineRunRequest) =>
  post<PipelineRunResponse>("/pipeline/run", req);
export const getRun = (runId: string) =>
  apiFetch<GraphState>(`/pipeline/${encodeURIComponent(runId)}`);
export const listRuns = (assetId?: string) =>
  apiFetch<RunSummary[]>(`/pipeline${qs({ asset_id: assetId })}`);
export const runWhatIf = (req: WhatIfRequest) => post<WhatIfResult>("/whatif", req);
export const runWsUrl = (runId: string) => `${WS_BASE_URL}/ws/runs/${encodeURIComponent(runId)}`;

// --- routes/approvals.py ----------------------------------------------------

export const getPendingApprovals = () => apiFetch<PendingApprovals>("/approvals/pending");
export const getContractEvidence = (contractId: string) =>
  apiFetch<ContractEvidence>(`/approvals/${encodeURIComponent(contractId)}`);
export const decideContract = (contractId: string, req: DecisionRequest) =>
  post<Approval>(`/approvals/${encodeURIComponent(contractId)}/decision`, req);
export const decidePromotion = (promotionId: string, req: DecisionRequest) =>
  post<Approval>(`/promotions/${encodeURIComponent(promotionId)}/decision`, req);

// --- routes/models.py -------------------------------------------------------

export const listModels = () => apiFetch<RegisteredModel[]>("/models");
export const listPromotions = () => apiFetch<PromotionRequest[]>("/models/promotions");
export const listDrift = (assetId?: string) =>
  apiFetch<DriftReport[]>(`/models/drift${qs({ asset_id: assetId })}`);
export const checkDrift = (req: DriftCheckRequest) => post<DriftReport>("/models/drift/check", req);

// --- routes/llm.py (API Key Up / Down) ----------------------------------------

/** The switch state only: {key: "up" | "down", provider}. The key itself never leaves the API. */
export const getKeyState = () => apiFetch<LlmKeyState>("/llm/key");
export const keyUp = () => post<LlmKeyState>("/llm/key/up", {});
export const keyDown = () => post<LlmKeyState>("/llm/key/down", {});

// --- routes/findings.py ("What did the technician find?") ----------------------

export const listFindings = (assetId?: string, contractId?: string) =>
  apiFetch<RepairFinding[]>(`/findings${qs({ asset_id: assetId, contract_id: contractId })}`);
export const postFinding = (req: RepairFindingRequest) => post<RepairFinding>("/findings", req);
export const getAlarmTrust = () => apiFetch<AlarmTrust>("/findings/trust");
export const getLabelLedger = () => apiFetch<LabelLedger>("/findings/ledger");

// --- routes/copilot.py ------------------------------------------------------

export const copilotChat = (req: CopilotRequest) => post<CopilotResponse>("/copilot/chat", req);

// --- routes/twin.py (3-D Digital Twin + X-Ray) --------------------------------

export const getTwin = () => apiFetch<TwinSnapshot>("/twin");
export const getTwinHud = (assetId: string) =>
  apiFetch<TwinHud>(`/twin/hud/${encodeURIComponent(assetId)}`);
export const getTwinRewind = (maxPoints = 400) =>
  apiFetch<TwinRewind>(`/twin/rewind${qs({ max_points: maxPoints })}`);
/** Short-poll fallback for the live event stream (same frame shape as /ws/twin). */
export const getTwinEvents = (since: number) =>
  apiFetch<TwinEventBatch>(`/twin/events${qs({ since })}`);
export const twinWsUrl = () => `${WS_BASE_URL}/ws/twin`;
