import { fireEvent, render, screen, within } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { FleetAsset, SystemHealth, TwinHud, TwinSnapshot } from "../api/types";
import { hudFixture } from "../twin/testFixtures";
import { XRayProvider } from "../twin/useXRay";
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
    work_order: null,
    status: "critical",
    alarm: "critical",
    alarm_since: "2026-09-13T08:30:00Z",
    alarm_reason: "sustained p >= 0.5 over 3 rows",
    sandbox: false,
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
    work_order: null,
    status: "healthy",
    alarm: null,
    alarm_since: null,
    alarm_reason: null,
    sandbox: true,
  },
  {
    asset: {
      asset_id: "MTR-021",
      name: "Boiler feed pump",
      site: "Plant A",
      line: "Line 2",
      asset_type: "induction_motor",
      rated_kw: 110,
      criticality: "high",
    },
    health_score: null,
    failure_probability: null,
    regime: "R2",
    last_ts: "2026-09-13T09:40:00Z",
    open_contract_id: null,
    stage: "production",
    work_order: {
      contract_id: "dc_MTR-021_20260913090000",
      asset_id: "MTR-021",
      recommendation: "maintain_now",
      window_start: "2025-08-27T02:00:00Z",
      window_end: "2025-08-27T05:00:00Z",
      planned_downtime_h: 3,
      approved_at: "2026-09-13T09:05:00Z",
      approver: "engineer@plant",
      status: "in_progress",
      started_ts: "2025-08-27T02:00:00Z",
      completed_ts: null,
      sandbox_scenario: false,
    },
    status: "offline",
    alarm: null,
    alarm_since: null,
    alarm_reason: null,
    sandbox: false,
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

/** GET /twin as the X-Ray tooltip needs it: status, regime and decision state per asset. */
const twinSnapshot: TwinSnapshot = {
  site: "Plant A",
  sites: [{ name: "Packaging", available: true, note: "" }],
  plant_time: "2026-09-13T10:00:00Z",
  thresholds: { classification: 0.5, early_warning: 0.3, sustained_rows: 3 },
  lines: [{ line: "Line 3", kind: "packaging", index: 3, assets: ["MTR-042"], mes: null }],
  assets: [
    {
      ...hudFixture.asset,
      asset: fleet[0].asset,
      failure_probability: 0.62,
      health_score: 41,
      regime: "R2",
      status: "critical",
      decision_state: "contract_pending",
      alarm: "critical",
      alarm_since: "2026-09-13T08:30:00Z",
      alarm_reason: "sustained p >= 0.5 over 3 rows",
    },
  ],
  model: hudFixture.model,
  live: {
    server_time: "2026-09-13T10:00:00Z",
    window_s: 10,
    msgs_per_s: 2,
    preds_per_s: 2,
    messages_total: 10,
    predictions_total: 10,
    edge_inference_ms_p50: null,
    edge_inference_ms_p95: null,
    path_latency_ms_p50: null,
    path_latency_ms_p95: null,
    n_latency_samples: 0,
    stale_after_s: 30,
    mqtt_connected: true,
    edge_up: true,
  },
};

const hud: TwinHud = { ...hudFixture, asset: twinSnapshot.assets[0] };

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
      if (url.includes("/twin/hud/MTR-042")) return jsonResponse(hud);
      if (url.endsWith("/twin")) return jsonResponse(twinSnapshot);
      return new Response("not found", { status: 404 });
    });
  });

  afterEach(() => {
    vi.restoreAllMocks();
    localStorage.removeItem("thor.xray");
  });

  it("renders asset ids from GET /fleet sorted by risk", async () => {
    render(
      <MemoryRouter>
        <XRayProvider>
          <Fleet />
        </XRayProvider>
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

  it("draws one motor sprite per asset, tiered by the API status (critical is latched)", async () => {
    render(
      <MemoryRouter>
        <XRayProvider>
          <Fleet />
        </XRayProvider>
      </MemoryRouter>,
    );
    const hot = await screen.findByRole("img", { name: /^MTR-042: critical, running, p\(fail\) 62.0%, contract open$/ });
    expect(hot).toHaveAttribute("data-tier", "high");
    expect(hot).toHaveClass("iso", "tier-high", "vibrating");
    expect(hot.querySelector(".smoke")).not.toBeNull();
    expect(hot.querySelector(".beacon")).not.toBeNull();

    const calm = screen.getByRole("img", { name: /^MTR-001: within healthy baseline, running, p\(fail\) 2.0%$/ });
    expect(calm).toHaveAttribute("data-tier", "low");
    expect(calm).not.toHaveClass("vibrating");
    expect(calm.querySelector(".smoke")).toBeNull();
    expect(calm.querySelector(".beacon")).toBeNull();

    // tier is also written as text so colour is never the only cue
    expect(screen.getByText("critical")).toBeInTheDocument();
    expect(screen.getByText("within healthy baseline")).toBeInTheDocument();
    // the latched alarm and the sandbox flag are API fields shown as pills
    const latched = screen.getByText("latched since 09-13 08:30");
    expect(latched).toHaveAttribute("title", "sustained p >= 0.5 over 3 rows");
    expect(screen.getByText("sandbox scenario")).toBeInTheDocument();
    expect(screen.getAllByText("State: running")).toHaveLength(3);
  });

  it("switches every sprite to pixel style from the toggle and remembers it", async () => {
    render(
      <MemoryRouter>
        <XRayProvider>
          <Fleet />
        </XRayProvider>
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

  it("shows a motor under an approved work order as offline, not as failing", async () => {
    render(
      <MemoryRouter>
        <XRayProvider>
          <Fleet />
        </XRayProvider>
      </MemoryRouter>,
    );
    const offline = await screen.findByRole("img", { name: /^MTR-021: no prediction, offline for maintenance/ });
    expect(offline).toHaveClass("stopped");
    expect(offline).not.toHaveClass("vibrating");
    expect(screen.getByText("offline · in maintenance")).toBeInTheDocument();
    expect(screen.getByText(/1 work order ·/)).toBeInTheDocument();
  });

  it("X-Ray: toggling on polls GET /twin, hovering a card shows the three facts, clicking opens the HUD drawer", async () => {
    render(
      <MemoryRouter>
        <XRayProvider>
          <Fleet />
        </XRayProvider>
      </MemoryRouter>,
    );
    await screen.findByText("MTR-042");
    const urls = () => (fetch as unknown as ReturnType<typeof vi.fn>).mock.calls.map((c) => String(c[0]));
    expect(urls().some((u) => u.endsWith("/twin"))).toBe(false);

    const toggle = screen.getByRole("button", { name: "X-Ray" });
    expect(toggle).toHaveAttribute("aria-pressed", "false");
    fireEvent.click(toggle);
    expect(toggle).toHaveAttribute("aria-pressed", "true");
    expect(localStorage.getItem("thor.xray")).toBe("1");

    // the card is now a button (no navigation) and the tooltip appears on hover
    const card = await screen.findByRole("button", { name: "MTR-042: open X-Ray HUD" });
    const host = screen.getByTestId("xray-host-MTR-042");
    fireEvent.mouseEnter(host);
    const tip = await screen.findByTestId("xray-tip-MTR-042");
    expect(urls().some((u) => u.endsWith("/twin"))).toBe(true);
    expect(tip).toHaveTextContent("critical");
    expect(tip).toHaveTextContent("62.0%");
    expect(tip).toHaveTextContent("running");
    expect(tip).toHaveTextContent("latched since 09-13 08:30");
    expect(tip).toHaveTextContent("contract pending");
    expect(screen.getByRole("tooltip")).toBeInTheDocument();

    // click opens the HUD drawer, fed by GET /twin/hud/MTR-042
    fireEvent.click(card);
    const drawer = await screen.findByTestId("xray-drawer");
    expect(within(drawer).getByRole("dialog", { name: "X-Ray HUD MTR-042" })).toBeInTheDocument();
    expect(await within(drawer).findByTestId("hud-pfail")).toHaveTextContent("62.0%");
    expect(within(drawer).getByRole("link", { name: "Open in Asset 360" })).toHaveAttribute("href", "/assets/MTR-042");

    // close returns to the cards; the X-Ray setting stays on
    fireEvent.click(within(drawer).getByRole("button", { name: "Close HUD" }));
    expect(screen.queryByTestId("xray-drawer")).toBeNull();
    expect(toggle).toHaveAttribute("aria-pressed", "true");
  });
});
