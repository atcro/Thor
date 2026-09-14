import { render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { FleetAsset, SystemHealth } from "../api/types";
import { Fleet } from "./Fleet";

const fleet: FleetAsset[] = [
  {
    asset: {
      asset_id: "MTR-042",
      name: "Cooling pump drive",
      site: "Plant A",
      line: "Line 3",
      asset_type: "induction_motor",
      rated_kw: 75,
      criticality: "high",
    },
    health_score: 41,
    failure_probability: 0.62,
    regime: "R2",
    last_ts: "2026-09-13T10:00:00Z",
    open_contract_id: "dc_MTR-042_20260913100000",
    stage: null,
  },
  {
    asset: {
      asset_id: "MTR-001",
      name: "Conveyor drive",
      site: "Plant A",
      line: "Line 1",
      asset_type: "induction_motor",
      rated_kw: 55,
      criticality: "medium",
    },
    health_score: 97,
    failure_probability: 0.02,
    regime: "R2",
    last_ts: "2026-09-13T10:00:00Z",
    open_contract_id: null,
    stage: "production",
  },
];

const health: SystemHealth = {
  db: "ok",
  mqtt: true,
  mlflow: false,
  edge: true,
  n_telemetry_rows: 103680,
  last_ingest_ts: "2026-09-13T10:00:00Z",
};

function jsonResponse(body: unknown): Response {
  return new Response(JSON.stringify(body), {
    status: 200,
    headers: { "Content-Type": "application/json" },
  });
}

describe("Fleet screen", () => {
  beforeEach(() => {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (globalThis as any).fetch = vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url.includes("/system/health")) return jsonResponse(health);
      if (url.includes("/fleet")) return jsonResponse(fleet);
      return new Response("not found", { status: 404 });
    });
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("renders asset ids from GET /fleet sorted by risk", async () => {
    render(
      <MemoryRouter>
        <Fleet />
      </MemoryRouter>,
    );
    expect(await screen.findByText("MTR-042")).toBeInTheDocument();
    expect(await screen.findByText("MTR-001")).toBeInTheDocument();
    // risk-first ordering: MTR-042 (p=0.62) card precedes MTR-001 (p=0.02)
    const ids = screen.getAllByText(/^MTR-\d{3}$/).map((el) => el.textContent);
    expect(ids.indexOf("MTR-042")).toBeLessThan(ids.indexOf("MTR-001"));
    // system panel rendered from GET /system/health
    expect(await screen.findByText("103,680")).toBeInTheDocument();
  });
});
