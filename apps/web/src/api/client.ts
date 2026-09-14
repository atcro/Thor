/**
 * Typed HTTP client for the Thor control plane (apps/api).
 * Every endpoint in docs/INTERFACES.md has one helper here; screens import from this file
 * and colocate their own polling/state logic.
 */
import type {
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
  PendingApprovals,
  PipelineRunRequest,
  PipelineRunResponse,
  PredictionRow,
  PromotionRequest,
  RegisteredModel,
  RunSummary,
  SystemHealth,
  TelemetryRow,
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

// --- routes/copilot.py ------------------------------------------------------

export const copilotChat = (req: CopilotRequest) => post<CopilotResponse>("/copilot/chat", req);
