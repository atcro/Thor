import { render, screen, within } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { hudFixture } from "./testFixtures";
import { XRayHud } from "./XRayHud";


function jsonResponse(body: unknown): Response {
  return new Response(JSON.stringify(body), { status: 200, headers: { "Content-Type": "application/json" } });
}

describe("X-Ray HUD", () => {
  beforeEach(() => {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (globalThis as any).fetch = vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url.includes("/twin/hud/MTR-042")) return jsonResponse(hudFixture);
      return new Response("not found", { status: 404 });
    });
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("renders every module from GET /twin/hud/MTR-042", async () => {
    const onClose = vi.fn();
    render(
      <MemoryRouter>
        <XRayHud assetId="MTR-042" onClose={onClose} />
      </MemoryRouter>,
    );
    // failure probability + trend
    expect(await screen.findByTestId("hud-pfail")).toHaveTextContent("64.0%");
    expect(screen.getByTestId("hud-probability")).toHaveTextContent("rising");
    expect(screen.getByTestId("hud-probability")).toHaveTextContent("median of last 3 samples");
    // the latched alarm from the API, in the status line and the header
    expect(screen.getByTestId("hud-status-line")).toHaveTextContent("Status: critical · latched since 08-27 00:00 · sustained p >= 0.5 over 3 rows");
    // operating state, as a plain name (the module is titled "State"; the test id is historical)
    expect(screen.getByTestId("hud-regime")).toHaveTextContent("State");
    expect(screen.getByTestId("hud-regime")).toHaveTextContent("running");
    expect(screen.getByText("R2 · running")).toHaveAttribute("aria-current", "true");
    // live sensors
    expect(screen.getByTestId("hud-sensors")).toHaveTextContent("1,478");
    expect(screen.getByTestId("hud-sensors")).toHaveTextContent("live");
    // drivers with their source
    expect(screen.getByTestId("hud-drivers")).toHaveTextContent("vibration rms ratio");
    expect(screen.getByTestId("hud-drivers")).toHaveTextContent("dc_MTR-042_20250827020000");
    // decision state
    expect(screen.getByTestId("hud-decision")).toHaveTextContent("Decision: contract pending");
    // PLC tags: mapped row carries its source on hover, simulated row is marked
    expect(screen.getByTestId("plc-Speed_Act")).toHaveAttribute("title", "Speed_Act ← rpm");
    const sp = screen.getByTestId("plc-Speed_SP");
    expect(sp).not.toHaveAttribute("title");
    const chip = within(sp).getByText("simulated");
    expect(chip).toHaveAttribute("title", "nominal speed of the asset's rated class (no setpoint in telemetry)");
    // MES context is labelled as simulated and context-only
    expect(screen.getByTestId("hud-mes")).toHaveTextContent("SKU-330ML-CAN");
    expect(screen.getByText("simulated MES feed · context only, feeds no model or decision")).toBeInTheDocument();
    // provenance
    expect(screen.getByTestId("hud-provenance")).toHaveTextContent("demo-prep@thor");
    expect(screen.getByTestId("hud-provenance")).toHaveTextContent("✓ passed · max |Δp| 0.0000");
    // header + Asset 360 link
    expect(screen.getByTestId("hud-status")).toHaveTextContent("critical");
    expect(screen.getByTestId("hud-status")).toHaveTextContent("latched since 08-27 00:00");
    // the visible word "regime" never appears in the HUD
    expect(document.body.textContent).not.toMatch(/regime/i);
    expect(screen.getByRole("link", { name: "Open in Asset 360" })).toHaveAttribute("href", "/assets/MTR-042");
    // the fetch hit the HUD endpoint
    const calls = (fetch as unknown as ReturnType<typeof vi.fn>).mock.calls.map((c) => String(c[0]));
    expect(calls.some((u) => u.endsWith("/twin/hud/MTR-042"))).toBe(true);
  });

  it("marks a sandbox scenario: header chip, sandbox samples on the sparkline, decision chip", async () => {
    const sandboxed = {
      ...hudFixture,
      asset: { ...hudFixture.asset, sandbox: true, alarm_sandbox: true },
      prediction_trend: hudFixture.prediction_trend.map((pt, i) => ({ ...pt, sandbox: i > 0 })),
      contract: { ...hudFixture.contract, sandbox_scenario: true },
    };
    // The HUD only needs the contract's id, recommendation, window and probability here.
    sandboxed.contract = {
      contract_id: "dc_MTR-042_sandbox",
      created_at: "2025-08-27T02:05:00Z",
      asset_id: "MTR-042",
      run_id: "run_1",
      model_version: "v64634",
      failure_probability: 0.64,
      recommendation: "maintain_now",
      window_start: "2025-08-27T04:00:00Z",
      window_end: "2025-08-27T08:00:00Z",
      expected_cost: 1000,
      cost_comparison: { asset_id: "MTR-042", options: [], recommended: "maintain_now", assumptions: {} },
      top_features: [],
      manual_context: [],
      lead_time_h: 10,
      calibration_brier: 0.05,
      data_quality_score: 90,
      explanation_text: "",
      explanation_source: "template",
      evidence_hash: "abc",
      sandbox_scenario: true,
    };
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (globalThis as any).fetch = vi.fn(async () => jsonResponse(sandboxed));
    render(
      <MemoryRouter>
        <XRayHud assetId="MTR-042" onClose={() => undefined} />
      </MemoryRouter>,
    );
    await screen.findByTestId("hud-pfail");
    expect(within(screen.getByTestId("hud-status")).getByText("sandbox scenario")).toBeInTheDocument();
    expect(screen.getByTestId("hud-probability")).toHaveTextContent("2 from sandbox readings");
    expect(screen.getByTestId("hud-probability").querySelectorAll("circle[data-sandbox]")).toHaveLength(2);
    expect(within(screen.getByTestId("hud-decision")).getByText("sandbox scenario")).toBeInTheDocument();
  });

  it("hides the MES module when the line has no MES context and closes on Escape", async () => {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (globalThis as any).fetch = vi.fn(async () => jsonResponse({ ...hudFixture, mes: null, drivers: [], drivers_source: null }));
    const onClose = vi.fn();
    render(
      <MemoryRouter>
        <XRayHud assetId="MTR-042" onClose={onClose} />
      </MemoryRouter>,
    );
    await screen.findByTestId("hud-pfail");
    expect(screen.queryByTestId("hud-mes")).toBeNull();
    expect(screen.getByTestId("hud-drivers")).toHaveTextContent("no analysis yet — Run analysis in Asset 360");
    window.dispatchEvent(new KeyboardEvent("keydown", { key: "Escape" }));
    expect(onClose).toHaveBeenCalled();
  });
});
