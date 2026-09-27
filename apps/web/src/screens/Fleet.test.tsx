import { fireEvent, render, screen } from "@testing-library/react";
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

  it("draws one motor sprite per asset, tiered by failure probability", async () => {
    render(
      <MemoryRouter>
        <Fleet />
      </MemoryRouter>,
    );
    const hot = await screen.findByRole("img", { name: /^MTR-042: high risk, nominal, p\(fail\) 62.0%, contract open$/ });
    expect(hot).toHaveAttribute("data-tier", "high");
    expect(hot).toHaveClass("iso", "tier-high", "vibrating");
    expect(hot.querySelector(".smoke")).not.toBeNull();
    expect(hot.querySelector(".beacon")).not.toBeNull();

    const calm = screen.getByRole("img", { name: /^MTR-001: low risk, nominal, p\(fail\) 2.0%$/ });
    expect(calm).toHaveAttribute("data-tier", "low");
    expect(calm).not.toHaveClass("vibrating");
    expect(calm.querySelector(".smoke")).toBeNull();
    expect(calm.querySelector(".beacon")).toBeNull();

    // tier is also written as text so colour is never the only cue
    expect(screen.getByText("high risk")).toBeInTheDocument();
    expect(screen.getByText("low risk")).toBeInTheDocument();
  });

  it("switches every sprite to pixel style from the toggle and remembers it", async () => {
    render(
      <MemoryRouter>
        <Fleet />
      </MemoryRouter>,
    );
    await screen.findByText("MTR-042");
    const pixel = screen.getByRole("button", { name: "Pixel" });
    expect(pixel).toHaveAttribute("aria-pressed", "false");
    fireEvent.click(pixel);
    expect(pixel).toHaveAttribute("aria-pressed", "true");
    for (const img of screen.getAllByRole("img")) {
      expect(img).toHaveClass("pixel");
      expect(img).not.toHaveClass("iso");
    }
    expect(localStorage.getItem("thor.fleet.spriteStyle")).toBe("pixel");
    localStorage.removeItem("thor.fleet.spriteStyle");
  });
});
