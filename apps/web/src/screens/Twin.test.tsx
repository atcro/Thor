import { render, screen } from "@testing-library/react";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { TwinEventBatch, TwinSnapshot } from "../api/types";

// The scene needs WebGL; the smoke test checks the page around it and that the scene
// receives the API snapshot (the stub prints the line names it was given).
vi.mock("../twin/scene/PlantScene", () => ({
  default: ({ snapshot }: { snapshot: TwinSnapshot | null }) => (
    <div data-testid="plant-scene">{snapshot?.lines.map((l) => l.line).join(" | ") ?? "no snapshot"}</div>
  ),
}));

const webgl = vi.hoisted(() => ({ supported: true }));
vi.mock("../twin/webgl", () => ({ supportsWebGL: () => webgl.supported }));

import { Twin } from "./Twin";

const live = {
  server_time: "2026-10-11T18:00:00Z",
  window_s: 10,
  msgs_per_s: 24.5,
  preds_per_s: 23.9,
  messages_total: 1200,
  predictions_total: 1180,
  edge_inference_ms_p50: 1.8,
  edge_inference_ms_p95: 3.2,
  path_latency_ms_p50: 640,
  path_latency_ms_p95: 910,
  n_latency_samples: 400,
  stale_after_s: 30,
  mqtt_connected: true,
  edge_up: true,
};

const latest = {
  asset_id: "MTR-042",
  ts: "2025-08-27T02:00:00Z",
  vibration_rms: 6.1,
  vibration_kurtosis: 5.2,
  vibration_crest: 4.1,
  bearing_temp_c: 81.2,
  motor_temp_c: 52.0,
  current_a: 160.3,
  rpm: 1478,
  load_pct: 56,
};

const snapshot: TwinSnapshot = {
  site: "Ludvika",
  sites: [
    { name: "Packaging", available: true, note: "" },
    { name: "Molding", available: false, note: "site tab only — not built" },
    { name: "Batching", available: false, note: "site tab only — not built" },
  ],
  plant_time: "2025-08-27T02:00:00Z",
  thresholds: { classification: 0.5, early_warning: 0.3, sustained_rows: 3 },
  lines: [
    { line: "Packaging Line 3", kind: "packaging", index: 3, assets: ["MTR-042"], mes: null },
    { line: "Compressor Hall", kind: "compressor", index: 6, assets: ["MTR-012"], mes: null },
  ],
  assets: [
    {
      asset: {
        asset_id: "MTR-042",
        name: "Line 3 filler drive motor",
        site: "Ludvika",
        line: "Packaging Line 3",
        asset_type: "induction_motor",
        rated_kw: 160,
        criticality: "high",
        role: "filler",
      },
      status: "critical",
      data_state: "live",
      decision_state: "contract_pending",
      failure_probability: 0.64,
      health_score: 36,
      prediction_ts: "2025-08-27T02:00:00Z",
      model_version: "v64634",
      n_prediction_rows: 3,
      regime: "R2",
      latest,
      last_ts: "2025-08-27T02:00:00Z",
      last_received_at: "2026-10-11T18:00:00Z",
      msgs_last_60s: 15,
      open_contract_id: "dc_MTR-042_1",
      work_order: null,
      alarm: "critical",
      alarm_since: "2025-08-27T00:00:00Z",
      alarm_reason: "sustained p >= 0.5 over 3 rows",
    },
    {
      asset: {
        asset_id: "MTR-012",
        name: "Air compressor K-1 drive motor",
        site: "Ludvika",
        line: "Compressor Hall",
        asset_type: "induction_motor",
        rated_kw: 110,
        criticality: "medium",
        role: "air compressor",
      },
      status: "healthy",
      data_state: "live",
      decision_state: "none",
      failure_probability: 0.01,
      health_score: 99,
      prediction_ts: "2025-08-27T02:00:00Z",
      model_version: "v64634",
      n_prediction_rows: 3,
      regime: "R3",
      latest: { ...latest, asset_id: "MTR-012" },
      last_ts: "2025-08-27T02:00:00Z",
      last_received_at: "2026-10-11T18:00:00Z",
      msgs_last_60s: 15,
      open_contract_id: null,
      work_order: null,
      alarm: null,
      alarm_since: null,
      alarm_reason: null,
    },
  ],
  model: {
    name: "thor-pdm",
    version: "v64634",
    family: "lightgbm",
    stage: "production",
    ims_total: 0.844,
    registered_at: "2026-10-11T17:30:00Z",
    approver: "demo-prep@thor",
    approved_at: "2026-10-11T17:31:00Z",
    promotion_id: "pr_1",
    validation_passed: true,
    parity_passed: true,
    parity_max_abs_diff: 0.00003,
    lead_time_median_h: 41.5,
    calibration_method: "sigmoid",
    onnx_path: "/models/thor-pdm-v64634.onnx",
  },
  live,
};

const batch: TwinEventBatch = { since: 0, next: 0, events: [], dropped: 0, live };

function jsonResponse(body: unknown): Response {
  return new Response(JSON.stringify(body), { status: 200, headers: { "Content-Type": "application/json" } });
}

function renderTwin() {
  return render(
    <MemoryRouter initialEntries={["/twin"]}>
      <Routes>
        <Route path="/twin" element={<Twin />} />
        <Route path="/" element={<div>Fleet fallback view</div>} />
      </Routes>
    </MemoryRouter>,
  );
}

describe("3-D Digital Twin route", () => {
  beforeEach(() => {
    webgl.supported = true;
    // No socket in jsdom: the hook must fall back to polling /twin/events.
    vi.stubGlobal("WebSocket", undefined);
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (globalThis as any).fetch = vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url.includes("/twin/events")) return jsonResponse(batch);
      if (url.endsWith("/twin")) return jsonResponse(snapshot);
      return new Response("not found", { status: 404 });
    });
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
  });

  it("renders the twin page from GET /twin: site tabs, scene lines, provenance and counters", async () => {
    renderTwin();
    expect(screen.getByRole("heading", { name: "3-D Digital Twin" })).toBeInTheDocument();
    // greyed-out site tabs
    expect(screen.getByRole("tab", { name: "Packaging" })).toHaveAttribute("aria-selected", "true");
    expect(screen.getByRole("tab", { name: "Molding" })).toBeDisabled();
    expect(screen.getByRole("tab", { name: "Batching" })).toBeDisabled();
    // the scene received the API snapshot
    expect(await screen.findByText("Packaging Line 3 | Compressor Hall")).toBeInTheDocument();
    // fly-to buttons per line
    expect(screen.getByRole("button", { name: "Line 3" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Compressors" })).toBeInTheDocument();
    // model provenance from the registry
    const prov = screen.getByTestId("twin-provenance");
    expect(prov).toHaveTextContent("thor-pdm v64634");
    expect(prov).toHaveTextContent("demo-prep@thor");
    expect(prov).toHaveTextContent("passed");
    // measured counters (from /twin/events or the snapshot's live block)
    const counters = await screen.findByTestId("twin-counters");
    expect(counters).toHaveTextContent("24.5 /s");
    expect(counters).toHaveTextContent("1.8 / 3.2 ms");
    expect(counters).toHaveTextContent("640 / 910 ms");
    expect(counters).toHaveTextContent("up / up");
    // legend states the encoding in words: green means "within healthy baseline"
    const legend = screen.getByTestId("twin-legend");
    expect(legend).toHaveTextContent("critical");
    expect(legend).toHaveTextContent("healthy · within healthy baseline");
    expect(legend).toHaveTextContent("ring: state running");
    // a live-state twin: no sandbox and no performance-mode controls
    expect(screen.queryByRole("button", { name: "Sandbox" })).toBeNull();
    expect(screen.queryByRole("button", { name: "Performance mode" })).toBeNull();
    // no visible "regime" anywhere on the twin page
    expect(document.body.textContent).not.toMatch(/regime/i);
  });

  it("falls back to the Fleet (/) when WebGL is unavailable", () => {
    webgl.supported = false;
    renderTwin();
    expect(screen.getByText("Fleet fallback view")).toBeInTheDocument();
    expect(screen.queryByTestId("plant-scene")).toBeNull();
  });
});
